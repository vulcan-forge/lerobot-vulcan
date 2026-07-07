from __future__ import annotations

import argparse
import json
import math
import socket
import time
from pathlib import Path

import numpy as np

from ldlidar_defaults import DEFAULT_LIDAR_FORWARD_ANGLE_DEG
from ldlidar_direct_snapshot_client import DirectLidarFeed, _save_snapshot, _scan_to_local_points
from ldlidar_direct_snapshot_stitch import (
    HTML_TEMPLATE,
    COLORS,
    Pose2D,
    _generate_svg,
    _load_snapshots,
    _pose_dict,
    _search_pose,
    _transform_points,
)
from ldlidar_two_pose_snapshot import (
    _build_rotation_signature,
    _estimate_signature_rotation_deg,
    _normalize_angle_deg,
    _rotation_progress_deg,
    _turned_to_raw_deg,
)
from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient
from lerobot.utils.import_utils import require_package


def _clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    if path.exists():
        path.rmdir()


def _ensure_clean_directory(path: Path) -> None:
    _clear_directory(path)
    path.mkdir(parents=True, exist_ok=True)


def _wait_for_initial_frame(feed: DirectLidarFeed, timeout_s: float) -> tuple[int, object]:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        frame_id, frame = feed.latest()
        if frame is not None:
            return frame_id, frame
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for initial LiDAR frame after {timeout_s:.1f}s.")


def _capture_snapshot(
    *,
    feed: DirectLidarFeed,
    output_dir: Path,
    request_index: int,
    after_frame_id: int,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    invert_lateral_axis: bool,
    max_distance_m: float,
    min_range_m: float,
    min_confidence: int,
    fresh_frame_timeout_s: float,
    fresh_frame_advances: int,
) -> tuple[int, object, np.ndarray]:
    armed_wall_ts = time.time()
    print(
        "[capture] waiting for next fresh revolution "
        f"(after frame_id={after_frame_id}, min advances={fresh_frame_advances})"
    )
    frame_id, frame = feed.wait_for_frame_after(
        after_frame_id=after_frame_id,
        timeout_s=float(fresh_frame_timeout_s),
        min_frame_advances=int(fresh_frame_advances),
        armed_wall_ts=armed_wall_ts,
    )
    if frame is None:
        raise TimeoutError("Timed out waiting for a fresh LiDAR revolution to capture.")

    local_points_xy = _scan_to_local_points(
        points=frame.points,
        forward_angle_deg=float(forward_angle_deg),
        valid_angle_half_width_deg=float(valid_angle_half_width_deg),
        invert_lateral_axis=bool(invert_lateral_axis),
        max_distance_m=float(max_distance_m),
        min_confidence=int(min_confidence),
        min_range_m=float(min_range_m),
    )
    capture_config = {
        "forward_angle_deg": float(forward_angle_deg),
        "valid_angle_half_width_deg": float(valid_angle_half_width_deg),
        "invert_lateral_axis": bool(invert_lateral_axis),
        "max_distance_m": float(max_distance_m),
        "min_range_m": float(min_range_m),
        "min_confidence": int(min_confidence),
        "capture_mode": "auto_snapshot_stitch",
    }
    _save_snapshot(
        output_dir=output_dir,
        request_index=int(request_index),
        frame_id=int(frame_id),
        frame=frame,
        local_points_xy=local_points_xy,
        capture_config=capture_config,
    )
    return int(frame_id), frame, local_points_xy


def _send_stop(robot: SourcceyClient) -> None:
    action = {
        "x.vel": 0.0,
        "y.vel": 0.0,
        "theta.vel": 0.0,
        "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
        "untorque_left": True,
        "untorque_right": True,
    }
    for _ in range(3):
        robot.send_action(action)
        time.sleep(0.04)


def _execute_turn_burst(
    *,
    robot: SourcceyClient,
    direction_sign: float,
    turn_speed: float,
    turn_burst_s: float,
    turn_settle_s: float,
) -> None:
    burst_deadline = time.monotonic() + float(turn_burst_s)
    while time.monotonic() < burst_deadline:
        robot.send_action(
            {
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": float(direction_sign) * float(turn_speed),
                "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
                "untorque_left": True,
                "untorque_right": True,
            }
        )
        time.sleep(0.05)
    _send_stop(robot)
    time.sleep(float(turn_settle_s))


def _turn_to_progress(
    *,
    robot: SourcceyClient,
    feed: DirectLidarFeed,
    reference_frame: object,
    target_turn_deg: float,
    direction_sign: float,
    turn_speed: float,
    turn_burst_s: float,
    turn_settle_s: float,
    rotation_signature_min_distance_m: float,
    max_distance_m: float,
    min_confidence: int,
    rotation_signature_bin_deg: float,
    stop_tolerance_deg: float,
    min_overlap_ratio: float,
    max_bursts: int,
    allow_partial_progress: bool = False,
    progress_window_half_deg: float = 45.0,
    progress_window_lead_deg: float = 15.0,
    stall_bursts_before_boost: int = 3,
    stall_speed_boost: float = 1.3,
    max_speed_scale: float = 2.5,
    min_delta_overlap_ratio: float = 0.40,
) -> dict[str, float]:
    reference_signature = _build_rotation_signature(
        [reference_frame],
        min_distance_m=float(rotation_signature_min_distance_m),
        max_distance_m=float(max_distance_m),
        min_confidence=int(min_confidence),
        bin_size_deg=float(rotation_signature_bin_deg),
    )
    if not np.isfinite(reference_signature).any():
        raise RuntimeError("Reference turn signature is empty; cannot track automatic sweep turn.")

    # Cumulative progress is tracked monotonically inside a search window anchored
    # to the previous estimate. An unwindowed 360deg match is multi-modal in
    # square/symmetric rooms (the range profile repeats every ~90deg), which
    # previously let a single wrong mode (e.g. 234deg during a 90deg turn) poison
    # the whole turn measurement.
    cumulative_progress_deg = 0.0
    cumulative_overlap_ratio = 0.0
    cumulative_score = float("inf")
    speed_scale = 1.0
    bursts_since_progress = 0
    last_frame_id = -1
    confirmations = 0
    consecutive_frame_timeouts = 0
    frame_wait_timeout_s = max(2.0, float(turn_burst_s) + float(turn_settle_s) + 1.5)

    for burst_index in range(1, int(max_bursts) + 1):
        _execute_turn_burst(
            robot=robot,
            direction_sign=float(direction_sign),
            turn_speed=float(turn_speed) * float(speed_scale),
            turn_burst_s=float(turn_burst_s),
            turn_settle_s=float(turn_settle_s),
        )

        frame_id, latest_frame = feed.wait_for_frame_after(
            after_frame_id=int(last_frame_id),
            timeout_s=float(frame_wait_timeout_s),
            min_frame_advances=1,
        )
        if latest_frame is None or int(frame_id) == int(last_frame_id):
            consecutive_frame_timeouts += 1
            print(
                f"[turn] burst={burst_index:02d} waiting for fresh LiDAR revolution timed out "
                f"(timeout={frame_wait_timeout_s:.2f}s, consecutive_timeouts={consecutive_frame_timeouts})"
            )
            if consecutive_frame_timeouts >= 3:
                raise RuntimeError(
                    "LiDAR feed stalled while tracking turn progress. "
                    f"No fresh revolutions arrived for {consecutive_frame_timeouts} consecutive bursts."
                )
            continue

        consecutive_frame_timeouts = 0
        last_frame_id = int(frame_id)
        current_signature = _build_rotation_signature(
            [latest_frame],
            min_distance_m=float(rotation_signature_min_distance_m),
            max_distance_m=float(max_distance_m),
            min_confidence=int(min_confidence),
            bin_size_deg=float(rotation_signature_bin_deg),
        )
        window_center_raw_deg = _turned_to_raw_deg(
            cumulative_progress_deg + float(progress_window_lead_deg),
            direction_sign=float(direction_sign),
        )
        raw_rotation_deg, score, overlap_ratio = _estimate_signature_rotation_deg(
            reference_signature,
            current_signature,
            bin_size_deg=float(rotation_signature_bin_deg),
            center_deg=float(window_center_raw_deg),
            search_half_window_deg=float(progress_window_half_deg),
        )
        if not math.isfinite(score):
            bursts_since_progress += 1
            print(
                f"[turn] burst={burst_index:02d} no valid match inside window "
                f"(cumulative={cumulative_progress_deg:6.1f}deg target={float(target_turn_deg):5.1f}deg)"
            )
        else:
            measured_progress_deg = _rotation_progress_deg(raw_rotation_deg, direction_sign=float(direction_sign))
            progress_delta_deg = _normalize_angle_deg(measured_progress_deg - cumulative_progress_deg)
            delta_status = "hold"
            if progress_delta_deg > 0.5 and float(overlap_ratio) >= float(min_delta_overlap_ratio):
                cumulative_progress_deg += float(progress_delta_deg)
                cumulative_overlap_ratio = float(overlap_ratio)
                cumulative_score = float(score)
                bursts_since_progress = 0
                delta_status = "accepted"
            elif progress_delta_deg > 0.5:
                # A big shift whose signature overlap collapsed is a wrong
                # room-symmetry mode (legit matches at <=90deg keep overlap
                # around 0.5+ with a ~180deg FOV). Ignore it.
                bursts_since_progress += 1
                delta_status = "low_overlap_rejected"
            else:
                # Small negative deltas are matcher noise; the robot only turns
                # one way, so hold the previous monotone estimate.
                bursts_since_progress += 1
            print(
                f"[turn] burst={burst_index:02d} progress={cumulative_progress_deg:6.1f}deg "
                f"(delta={progress_delta_deg:+5.1f}deg {delta_status}) target={float(target_turn_deg):5.1f}deg "
                f"overlap={overlap_ratio:0.2f} score={score:0.4f}"
            )

        if bursts_since_progress >= max(1, int(stall_bursts_before_boost)) and speed_scale < float(max_speed_scale):
            speed_scale = min(float(max_speed_scale), speed_scale * float(stall_speed_boost))
            bursts_since_progress = 0
            print(
                f"[turn] burst={burst_index:02d} no rotation progress; boosting turn speed "
                f"(scale={speed_scale:.2f})"
            )

        if cumulative_progress_deg >= (float(target_turn_deg) - float(stop_tolerance_deg)):
            if cumulative_overlap_ratio >= float(min_overlap_ratio):
                confirmations += 1
            if confirmations >= 2 or cumulative_progress_deg >= (float(target_turn_deg) + 2.0 * float(stop_tolerance_deg)):
                _send_stop(robot)
                return {
                    "progress_deg": float(cumulative_progress_deg),
                    "overlap_ratio": float(cumulative_overlap_ratio),
                    "score": float(cumulative_score),
                    "frame_id": float(last_frame_id),
                    "completed": 1.0,
                }
        else:
            confirmations = 0

    _send_stop(robot)
    if allow_partial_progress and last_frame_id >= 0 and cumulative_progress_deg > 0.0:
        print(
            "[turn] warning: automatic turn did not fully confirm target; "
            f"using tracked progress={cumulative_progress_deg:.1f}deg "
            f"overlap={cumulative_overlap_ratio:0.2f} score={cumulative_score:0.4f}"
        )
        return {
            "progress_deg": float(cumulative_progress_deg),
            "overlap_ratio": float(cumulative_overlap_ratio),
            "score": float(cumulative_score),
            "frame_id": float(last_frame_id),
            "completed": 0.0,
        }

    raise RuntimeError(
        "Automatic sweep turn failed to reach its target. "
        f"Tracked progress={cumulative_progress_deg:.1f}deg overlap={cumulative_overlap_ratio:0.2f} score={cumulative_score:0.4f}"
    )


def _stitch_snapshot_directory(
    *,
    snapshot_dir: Path,
    output_dir: Path,
    resolution_m: float,
    search_xy_m: float,
    expected_turn_deg: float,
    turn_sign: str,
) -> dict[str, object]:
    snapshots = _load_snapshots(snapshot_dir)
    poses: list[Pose2D] = [Pose2D(0.0, 0.0, 0.0)]
    transformed_sets: list[np.ndarray] = [_transform_points(snapshots[0].points_xy, poses[0])]
    solve_log: list[dict[str, object]] = [
        {
            "snapshot": snapshots[0].name,
            "pose": _pose_dict(poses[0]),
            "score": None,
            "host_revolution_index": snapshots[0].metadata.get("host_revolution_index"),
            "host_point_digest": snapshots[0].metadata.get("host_point_digest"),
        }
    ]

    signed_turn = float(expected_turn_deg) * (1.0 if turn_sign == "ccw" else -1.0)

    for idx in range(1, len(snapshots)):
        initial_pose = Pose2D(
            x=poses[-1].x,
            y=poses[-1].y,
            theta_deg=poses[-1].theta_deg + signed_turn,
        )
        global_points_xy = np.concatenate(transformed_sets, axis=0)
        solved_pose, score_meta = _search_pose(
            snapshot_points_xy=snapshots[idx].points_xy,
            global_points_xy=global_points_xy,
            initial_pose=initial_pose,
            resolution_m=float(resolution_m),
            search_xy_m=float(search_xy_m),
            coarse_angle_step_deg=4.0,
            fine_angle_step_deg=0.5,
            expected_turn_deg=float(expected_turn_deg),
        )
        poses.append(solved_pose)
        transformed = _transform_points(snapshots[idx].points_xy, solved_pose)
        transformed_sets.append(transformed)
        solve_log.append(
            {
                "snapshot": snapshots[idx].name,
                "pose": _pose_dict(solved_pose),
                "score": score_meta["score"],
                "host_revolution_index": snapshots[idx].metadata.get("host_revolution_index"),
                "host_point_digest": snapshots[idx].metadata.get("host_point_digest"),
            }
        )

    svg = _generate_svg(transformed_sets, poses)
    report = {
        "schema": "sourccey.direct_snapshot_stitch.v1",
        "snapshot_dir": str(snapshot_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "parameters": {
            "resolution_m": float(resolution_m),
            "search_xy_m": float(search_xy_m),
            "expected_turn_deg": float(expected_turn_deg),
            "turn_sign": turn_sign,
        },
        "solve_log": solve_log,
    }
    html = HTML_TEMPLATE.format(svg=svg, report=json.dumps(report, indent=2))
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "latest_stitched_overlay.html"
    report_path = output_dir / "latest_stitched_overlay.json"
    html_path.write_text(html, encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "snapshots": snapshots,
        "poses": poses,
        "transformed_sets": transformed_sets,
        "solve_log": solve_log,
        "html_path": html_path,
        "report_path": report_path,
    }


def _init_rerun(
    *,
    session_name: str,
    mode: str,
    grpc_port: int,
    web_port: int,
):
    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    rr.init(session_name, spawn=(mode == "local"))
    viewer_url = None
    if mode == "web":
        server_uri = rr.serve_grpc(grpc_port=int(grpc_port))
        rr.serve_web_viewer(open_browser=True, web_port=int(web_port), connect_to=server_uri)
        viewer_url = f"http://127.0.0.1:{int(web_port)}"
        print(f"[rerun] web viewer ready at {viewer_url}")
        print(f"[rerun] live recording source: {server_uri}")

        alt_url = None
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(("8.8.8.8", 80))
            local_ip = probe.getsockname()[0]
            probe.close()
            if local_ip and local_ip != "127.0.0.1":
                alt_url = f"http://{local_ip}:{int(web_port)}"
        except Exception:
            alt_url = None

        if alt_url:
            print(f"[rerun] alternate viewer URL: {alt_url}")
        print("[rerun] the viewer must be opened by rerun itself so it attaches to the live recording")
    else:
        print("[rerun] spawned local viewer (if available in PATH)")
    return rr, viewer_url


def _color_rgb(index: int) -> tuple[int, int, int]:
    color_hex = COLORS[index % len(COLORS)].lstrip("#")
    return tuple(int(color_hex[i : i + 2], 16) for i in (0, 2, 4))


def _pose_arc_points(pose: Pose2D, *, radius_m: float, half_width_deg: float, steps: int = 32) -> np.ndarray:
    theta_center = math.radians(float(pose.theta_deg))
    arc_angles = np.linspace(-float(half_width_deg), float(half_width_deg), max(int(steps), 3))
    points = []
    for delta_deg in arc_angles:
        theta = theta_center + math.radians(float(delta_deg))
        points.append([pose.x + radius_m * math.cos(theta), pose.y + radius_m * math.sin(theta), 0.0])
    return np.asarray(points, dtype=np.float32)


def _log_rerun_state(
    rr,
    *,
    capture_index: int,
    transformed_sets: list[np.ndarray],
    poses: list[Pose2D],
    solve_log: list[dict[str, object]],
    cone_half_width_deg: float,
) -> None:
    rr.set_time("capture_index", sequence=int(capture_index))
    combined_points = []
    for idx, points_xy in enumerate(transformed_sets):
        color = _color_rgb(idx)
        if len(points_xy):
            points_xyz = np.column_stack(
                [points_xy[:, 0], points_xy[:, 1], np.zeros((len(points_xy),), dtype=np.float32)]
            )
            combined_points.append(points_xyz)
            rr.log(
                f"world/snapshots/{idx + 1:02d}",
                rr.Points3D(points_xyz, colors=np.tile(np.asarray(color), (len(points_xyz), 1)), radii=0.015),
            )
        pose = poses[idx]
        origin = np.asarray([[pose.x, pose.y, 0.0]], dtype=np.float32)
        vector = np.asarray(
            [[0.30 * math.cos(math.radians(pose.theta_deg)), 0.30 * math.sin(math.radians(pose.theta_deg)), 0.0]],
            dtype=np.float32,
        )
        rr.log(f"world/poses/{idx + 1:02d}/origin", rr.Points3D(origin, colors=[color], radii=0.05))
        rr.log(f"world/poses/{idx + 1:02d}/heading", rr.Arrows3D(origins=origin, vectors=vector, colors=[color]))
        arc = _pose_arc_points(pose, radius_m=0.45, half_width_deg=float(cone_half_width_deg))
        rr.log(f"world/poses/{idx + 1:02d}/cone", rr.LineStrips3D([arc], colors=[color], radii=0.004))

    if combined_points:
        combined = np.vstack(combined_points)
        rr.log("world/stitched_map", rr.Points3D(combined, colors=[200, 220, 255], radii=0.01))

    rr.log("world/status/capture_count", rr.Scalars(float(capture_index)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatic stop-capture-stitch LiDAR sweep with live Rerun visualization."
    )
    parser.add_argument("--remote-ip", default="192.168.1.237", help="Sourccey host IP.")
    parser.add_argument("--robot-id", default="sourccey")
    parser.add_argument("--lidar-host", default="192.168.1.237", help="Pi LiDAR stream host.")
    parser.add_argument("--lidar-port", type=int, default=8765)
    parser.add_argument("--output-dir", default="artifacts/auto_snapshot_stitch")
    parser.add_argument("--capture-count", type=int, default=4, help="Total snapshots including the initial front view.")
    parser.add_argument("--turn-direction", choices=("ccw", "cw"), default="ccw")
    parser.add_argument("--turn-deg", type=float, default=90.0)
    parser.add_argument("--turn-speed", type=float, default=0.82)
    parser.add_argument("--turn-burst-s", type=float, default=0.14)
    parser.add_argument("--turn-settle-s", type=float, default=0.18)
    parser.add_argument("--capture-settle-s", type=float, default=0.90)
    parser.add_argument("--max-turn-bursts", type=int, default=32)
    parser.add_argument("--stop-tolerance-deg", type=float, default=12.0)
    parser.add_argument("--min-turn-overlap-ratio", type=float, default=0.18)
    parser.add_argument("--fresh-frame-timeout-s", type=float, default=3.0)
    parser.add_argument("--fresh-frame-advances", type=int, default=1)
    parser.add_argument("--forward-angle-deg", type=float, default=DEFAULT_LIDAR_FORWARD_ANGLE_DEG)
    parser.add_argument("--valid-angle-half-width-deg", type=float, default=180.0)
    parser.add_argument("--invert-lateral-axis", action="store_true", default=True)
    parser.add_argument("--no-invert-lateral-axis", dest="invert_lateral_axis", action="store_false")
    parser.add_argument("--max-distance-m", type=float, default=8.0)
    parser.add_argument("--min-range-m", type=float, default=0.03)
    parser.add_argument("--min-confidence", type=int, default=0)
    parser.add_argument("--rotation-signature-min-distance-m", type=float, default=0.10)
    parser.add_argument("--rotation-signature-bin-deg", type=float, default=3.0)
    parser.add_argument("--stitch-resolution-m", type=float, default=0.03)
    parser.add_argument("--stitch-search-xy-m", type=float, default=0.45)
    parser.add_argument("--rerun-mode", choices=("web", "local"), default="web")
    parser.add_argument("--rerun-grpc-port", type=int, default=9877)
    parser.add_argument("--rerun-web-port", type=int, default=9878)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    snapshot_dir = output_dir / "snapshots"
    stitch_dir = output_dir / "offline_stitch"
    _ensure_clean_directory(output_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stitch_dir.mkdir(parents=True, exist_ok=True)

    rr, viewer_url = _init_rerun(
        session_name="ldlidar_auto_snapshot_stitch",
        mode=args.rerun_mode,
        grpc_port=int(args.rerun_grpc_port),
        web_port=int(args.rerun_web_port),
    )

    feed = DirectLidarFeed(args.lidar_host, int(args.lidar_port))
    feed.start()

    robot = SourcceyClient(SourcceyClientConfig(id=args.robot_id, remote_ip=args.remote_ip))
    robot.connect()
    _send_stop(robot)

    direction_sign = 1.0 if args.turn_direction == "ccw" else -1.0

    try:
        last_frame_id, first_frame = _wait_for_initial_frame(feed, timeout_s=5.0)
        print(
            "[run] LiDAR feed ready "
            f"(frame_id={last_frame_id}, host_rev={first_frame.revolution_index}, viewer={viewer_url or 'local'})"
        )

        request_index = 1
        frame_id, captured_frame, _ = _capture_snapshot(
            feed=feed,
            output_dir=snapshot_dir,
            request_index=request_index,
            after_frame_id=last_frame_id,
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            invert_lateral_axis=bool(args.invert_lateral_axis),
            max_distance_m=float(args.max_distance_m),
            min_range_m=float(args.min_range_m),
            min_confidence=int(args.min_confidence),
            fresh_frame_timeout_s=float(args.fresh_frame_timeout_s),
            fresh_frame_advances=int(args.fresh_frame_advances),
        )
        print("[run] snapshot 1 captured; stitching current map")

        stitch_state = _stitch_snapshot_directory(
            snapshot_dir=snapshot_dir,
            output_dir=stitch_dir,
            resolution_m=float(args.stitch_resolution_m),
            search_xy_m=float(args.stitch_search_xy_m),
            expected_turn_deg=float(args.turn_deg),
            turn_sign=str(args.turn_direction),
        )
        _log_rerun_state(
            rr,
            capture_index=request_index,
            transformed_sets=stitch_state["transformed_sets"],
            poses=stitch_state["poses"],
            solve_log=stitch_state["solve_log"],
            cone_half_width_deg=float(args.valid_angle_half_width_deg),
        )

        last_captured_frame = captured_frame
        last_frame_id = int(frame_id)

        for snapshot_index in range(2, int(args.capture_count) + 1):
            print(
                f"[run] rotating for snapshot {snapshot_index}/{int(args.capture_count)} "
                f"({args.turn_deg:.1f}deg {args.turn_direction})"
            )
            turn_meta = _turn_to_progress(
                robot=robot,
                feed=feed,
                reference_frame=last_captured_frame,
                target_turn_deg=float(args.turn_deg),
                direction_sign=float(direction_sign),
                turn_speed=float(args.turn_speed),
                turn_burst_s=float(args.turn_burst_s),
                turn_settle_s=float(args.turn_settle_s),
                rotation_signature_min_distance_m=float(args.rotation_signature_min_distance_m),
                max_distance_m=float(args.max_distance_m),
                min_confidence=int(args.min_confidence),
                rotation_signature_bin_deg=float(args.rotation_signature_bin_deg),
                stop_tolerance_deg=float(args.stop_tolerance_deg),
                min_overlap_ratio=float(args.min_turn_overlap_ratio),
                max_bursts=int(args.max_turn_bursts),
            )
            print(
                "[run] turn complete "
                f"(progress={turn_meta['progress_deg']:.1f}deg overlap={turn_meta['overlap_ratio']:.2f} score={turn_meta['score']:.4f})"
            )
            print(f"[run] settling for {float(args.capture_settle_s):.2f}s before capture")
            time.sleep(float(args.capture_settle_s))

            request_index = snapshot_index
            frame_id, captured_frame, _ = _capture_snapshot(
                feed=feed,
                output_dir=snapshot_dir,
                request_index=request_index,
                after_frame_id=last_frame_id,
                forward_angle_deg=float(args.forward_angle_deg),
                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                invert_lateral_axis=bool(args.invert_lateral_axis),
                max_distance_m=float(args.max_distance_m),
                min_range_m=float(args.min_range_m),
                min_confidence=int(args.min_confidence),
                fresh_frame_timeout_s=float(args.fresh_frame_timeout_s),
                fresh_frame_advances=int(args.fresh_frame_advances),
            )
            print(f"[run] snapshot {snapshot_index} captured; rebuilding stitched map from saved snapshots")
            stitch_state = _stitch_snapshot_directory(
                snapshot_dir=snapshot_dir,
                output_dir=stitch_dir,
                resolution_m=float(args.stitch_resolution_m),
                search_xy_m=float(args.stitch_search_xy_m),
                expected_turn_deg=float(args.turn_deg),
                turn_sign=str(args.turn_direction),
            )
            _log_rerun_state(
                rr,
                capture_index=request_index,
                transformed_sets=stitch_state["transformed_sets"],
                poses=stitch_state["poses"],
                solve_log=stitch_state["solve_log"],
                cone_half_width_deg=float(args.valid_angle_half_width_deg),
            )
            print(
                "[run] stitched preview updated "
                f"({stitch_state['html_path']})"
            )
            last_captured_frame = captured_frame
            last_frame_id = int(frame_id)

        print(f"[run] complete. snapshots={request_index} viewer={viewer_url or 'local rerun'}")
        print(f"[run] stitched html: {stitch_state['html_path']}")
        print(f"[run] stitched report: {stitch_state['report_path']}")
        return 0
    finally:
        try:
            _send_stop(robot)
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        feed.stop()


if __name__ == "__main__":
    raise SystemExit(main())
