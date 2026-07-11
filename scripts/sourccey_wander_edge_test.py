"""Live edge-detection stop test: drive forward, put a table edge in the way,
watch it stop.

No mapping, no trials, no planning. Start the script and the robot creeps
forward. The Rerun view shows both eye cameras plus a big color status banner:

    GREEN  "CLEAR - DRIVING"                    gate is happy, robot moves
    RED    "EDGE DETECTED - STOPPING ROBOT"     gate is forcibly holding it
    ORANGE "CAMERA STALE - STOPPING ROBOT"      frames stopped, failing safe

When the hazard clears (move the table, or the robot is repositioned) the
robot resumes driving on its own, so you can test edge after edge in one run.

  --motion forward   (default) drive straight forward, stop on any detected
                     elevated edge in range.
  --motion turn      rotate in place instead (direction flips every
                     --sweep-s). Rotation only freezes in the NEAR tier
                     (<= --near-freeze-distance-m), matching the wander
                     policy — park the robot with a table edge close by and
                     confirm it refuses to turn into it.

The optional lidar stop box (--lidar-host, recommended for forward runs)
acts as a backstop in case the camera gate misses. Hard timeout via
--max-run-s. Ctrl+C stops everything. Arms are never commanded.

Example:
  uv run --with rerun-sdk python scripts/sourccey_wander_edge_test.py \
    --remote-ip 192.168.1.237 --lidar-host 192.168.1.237
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from ldlidar_auto_snapshot_stitch import _init_rerun, _send_stop
from ldlidar_defaults import (
    DEFAULT_LIDAR_FORWARD_ANGLE_DEG,
    DEFAULT_LIDAR_STOP_BOX_DISTANCE_M,
    DEFAULT_LIDAR_STOP_BOX_HALF_WIDTH_M,
    DEFAULT_LIDAR_STOP_BOX_MIN_DISTANCE_M,
    DEFAULT_LIDAR_STOP_BOX_THICKNESS_M,
)
from ldlidar_direct_snapshot_client import DirectLidarFeed, _scan_to_local_points
from ldlidar_wander_snapshot_stitch import StopZoneConfig, _blocked_points_for_frame
from sourccey_elevated_safety import (
    ElevatedHazardMonitor,
    ElevatedSafetyConfig,
    SlamCameraSubscriber,
    endpoint_from_remote_ip,
    format_hazard_log,
)

GREEN = (40, 200, 60)
RED = (40, 40, 230)
ORANGE = (30, 140, 250)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--remote-ip", type=str, default="192.168.1.237")
    parser.add_argument("--robot-id", type=str, default="sourccey")
    parser.add_argument("--slam-input-endpoint", type=str, default="")
    parser.add_argument(
        "--lidar-host",
        type=str,
        default="",
        help="Enable the lidar stop-box backstop (recommended for forward runs).",
    )
    parser.add_argument("--lidar-port", type=int, default=8765)
    parser.add_argument("--motion", choices=("forward", "turn"), default="forward")
    parser.add_argument("--max-run-s", type=float, default=120.0, help="Hard session timeout.")
    parser.add_argument("--drive-speed", type=float, default=0.78)
    parser.add_argument("--min-effective-speed", type=float, default=0.75)
    parser.add_argument("--turn-speed", type=float, default=0.75)
    parser.add_argument("--min-effective-turn-speed", type=float, default=0.72)
    parser.add_argument("--sweep-s", type=float, default=4.0, help="Turn-mode direction flip period.")
    parser.add_argument("--rerun-mode", choices=("web", "local", "off"), default="web")
    parser.add_argument("--rerun-grpc-port", type=int, default=9877)
    parser.add_argument("--rerun-web-port", type=int, default=9878)
    # Gate tunables for boundary calibration.
    parser.add_argument(
        "--forward-block-distance-m",
        type=float,
        default=0.35,
        help="Stop when the estimated edge distance is at or below this. The robot "
        "travels a few more cm during the trip-frames confirmation.",
    )
    parser.add_argument("--near-freeze-distance-m", type=float, default=0.35)
    parser.add_argument("--trip-frames", type=int, default=3)
    parser.add_argument("--clear-frames", type=int, default=5)
    parser.add_argument("--roi-min-y-ratio", type=float, default=0.46)
    parser.add_argument("--roi-max-y-ratio", type=float, default=0.95)
    parser.add_argument("--min-line-score", type=float, default=860.0)
    # Distance-model anchors: edge image-row maps linearly from far (top of
    # ROI) to near (bottom of ROI). Calibrate with a tape measure: park at a
    # known distance, read cy/dist off the overlay, adjust these.
    parser.add_argument("--near-distance-m", type=float, default=0.20)
    parser.add_argument("--far-distance-m", type=float, default=1.55)
    return parser


def _status_for(state, motion: str, lidar_blocked: bool) -> tuple[str, tuple[int, int, int], bool]:
    """Returns (banner text, BGR color, motion_allowed)."""
    if state.frames_stale:
        return "CAMERA STALE - STOPPING ROBOT", ORANGE, False
    if lidar_blocked:
        return "LIDAR STOP BOX - STOPPING ROBOT", ORANGE, False
    if getattr(state, "blind_zone", False):
        # The edge was being tracked and then vanished at close range: the
        # robot is nearer than the detector can see. Latched until the robot
        # backs away ~0.5m or the edge is re-seen at a safe distance.
        return "EDGE LOST TOO CLOSE - HOLDING (back the robot away)", RED, False
    if getattr(state, "ground_active", False):
        dist = (
            "-"
            if state.ground_distance_m is None
            else f"{state.ground_distance_m:.2f}m"
        )
        return f"GROUND OBSTACLE {dist} {state.ground_side.upper()} - STOPPING ROBOT", RED, False
    if motion == "turn":
        if state.active and state.near_freeze:
            dist = "-" if state.est_distance_m is None else f"{state.est_distance_m:.2f}m"
            return f"EDGE {dist} {state.side.upper()} - ROTATION FROZEN", RED, False
        if state.active:
            dist = "-" if state.est_distance_m is None else f"{state.est_distance_m:.2f}m"
            return f"edge seen {dist} (far) - still turning", GREEN, True
        return "CLEAR - TURNING", GREEN, True
    if state.active:
        dist = "-" if state.est_distance_m is None else f"{state.est_distance_m:.2f}m"
        return f"EDGE DETECTED {dist} {state.side.upper()} - STOPPING ROBOT", RED, False
    if getattr(state, "hold", False):
        tag = "PERMANENT - reverse to release" if state.permanent_hold else "back away or wait"
        return f"HOLDING AFTER STOP ({state.hold_reason}) - {tag}", ORANGE, False
    if state.classification == "elevated_unresolved":
        # Seeing a possible edge; keeps approaching so parallax can verify
        # whether it is real or a floor-object illusion.
        return "INVESTIGATING EDGE - verifying by approach", GREEN, True
    if state.classification == "floor_parallax":
        return "FLOOR OBJECT (illusion ruled out) - continuing", GREEN, True
    return "CLEAR - DRIVING", GREEN, True


def _banner_image(text: str, color: tuple[int, int, int]) -> np.ndarray:
    """Big solid-color status card: unmistakable green/red flip in Rerun."""
    canvas = np.zeros((120, 640, 3), dtype=np.uint8)
    canvas[:, :] = color
    cv2.putText(canvas, text, (14, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(canvas, text, (14, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _bannered_frame(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    canvas = frame.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), color, -1)
    cv2.putText(canvas, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _log_view(rr, monitor, config, text: str, color) -> None:
    if rr is None:
        return
    for cam in (config.left_key, config.right_key, config.bottom_key):
        frame = monitor.annotated(cam)
        if frame is not None:
            rr.log(f"cameras/{cam}", rr.Image(_bannered_frame(frame, text, color)[:, :, ::-1]))
    rr.log("safety/status", rr.Image(_banner_image(text, color)[:, :, ::-1]))


def make_lidar_ranges_fn(feed: DirectLidarFeed):
    """Newest scan as (bearings_deg, distances_m) in the robot local frame,
    decoded exactly like the wander mapper decodes it."""

    def lidar_ranges():
        _, frame = feed.latest()
        if frame is None:
            return None
        points_xy = _scan_to_local_points(
            points=frame.points,
            forward_angle_deg=float(DEFAULT_LIDAR_FORWARD_ANGLE_DEG),
            valid_angle_half_width_deg=180.0,
            invert_lateral_axis=True,
            max_distance_m=8.0,
            min_confidence=0,
            min_range_m=0.30,
        )
        if len(points_xy) == 0:
            return None
        bearings = np.degrees(np.arctan2(points_xy[:, 1], points_xy[:, 0]))
        distances = np.hypot(points_xy[:, 0], points_xy[:, 1])
        return bearings, distances

    return lidar_ranges


def _send_motion(robot, *, x_vel: float = 0.0, theta_vel: float = 0.0) -> None:
    robot.send_action(
        {
            "x.vel": float(x_vel),
            "y.vel": 0.0,
            "theta.vel": float(theta_vel),
            "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
            "untorque_left": True,
            "untorque_right": True,
        }
    )


class LidarBackstop:
    """Lidar stop-box tripwire with the same startup baseline calibration the
    wander script uses (the robot's own shell inside the box is subtracted)."""

    def __init__(self, host: str, port: int) -> None:
        self.feed = DirectLidarFeed(host, int(port))
        self.feed.start()
        self.zone_cfg = StopZoneConfig(
            forward_angle_deg=float(DEFAULT_LIDAR_FORWARD_ANGLE_DEG),
            min_distance_m=float(DEFAULT_LIDAR_STOP_BOX_MIN_DISTANCE_M),
            tripwire_distance_m=float(DEFAULT_LIDAR_STOP_BOX_DISTANCE_M),
            tripwire_half_width_m=float(DEFAULT_LIDAR_STOP_BOX_HALF_WIDTH_M),
            tripwire_thickness_m=float(DEFAULT_LIDAR_STOP_BOX_THICKNESS_M),
            min_points_to_trigger=6,
        )

    def calibrate_baseline(self, samples: int = 10) -> None:
        counts: list[int] = []
        deadline = time.monotonic() + 6.0
        last_frame_id = -1
        while len(counts) < int(samples) and time.monotonic() < deadline:
            frame_id, frame = self.feed.latest()
            if frame is None or int(frame_id) == last_frame_id:
                time.sleep(0.05)
                continue
            last_frame_id = int(frame_id)
            counts.append(_blocked_points_for_frame(frame, self.zone_cfg))
        if counts:
            counts.sort()
            self.zone_cfg.baseline_points = int(counts[len(counts) // 2])
        print(
            f"[edge-test] lidar stop box baseline={self.zone_cfg.baseline_points} "
            f"(trigger at {self.zone_cfg.blocked_trigger_count()} points)"
        )

    _blocked_latched_until: float = 0.0

    def blocked(self) -> bool:
        """Instant trip, slow clear: once the box trips, report blocked for at
        least 1s — a flickering box must not let the robot ratchet forward."""
        _, frame = self.feed.latest()
        now = time.monotonic()
        if frame is not None and _blocked_points_for_frame(
            frame, self.zone_cfg
        ) >= self.zone_cfg.blocked_trigger_count():
            self._blocked_latched_until = now + 1.0
            return True
        return now < self._blocked_latched_until

    def stop(self) -> None:
        self.feed.stop()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    endpoint = str(args.slam_input_endpoint).strip() or endpoint_from_remote_ip(args.remote_ip)
    config = ElevatedSafetyConfig(
        slam_input_endpoint=endpoint,
        forward_block_distance_m=float(args.forward_block_distance_m),
        near_freeze_distance_m=float(args.near_freeze_distance_m),
        trip_frames=int(args.trip_frames),
        clear_frames=int(args.clear_frames),
        roi_min_y_ratio=float(args.roi_min_y_ratio),
        roi_max_y_ratio=float(args.roi_max_y_ratio),
        min_line_score=float(args.min_line_score),
        near_distance_m=float(args.near_distance_m),
        far_distance_m=float(args.far_distance_m),
    )

    rr = None
    if args.rerun_mode != "off":
        rr, viewer_url = _init_rerun(
            session_name="sourccey_wander_edge_test",
            mode=str(args.rerun_mode),
            grpc_port=int(args.rerun_grpc_port),
            web_port=int(args.rerun_web_port),
        )
        if viewer_url:
            print(f"[edge-test] rerun viewer: {viewer_url}")

    print(f"[edge-test] subscribing to {endpoint}")
    subscriber = SlamCameraSubscriber(
        endpoint=endpoint,
        camera_keys=(config.left_key, config.right_key, config.bottom_key),
    )
    subscriber.start()
    if not subscriber.wait_for_frames(
        timeout_s=6.0, required=(config.left_key, config.right_key)
    ):
        print(
            "[edge-test] ERROR: no eye camera frames from the slam_input stream — refusing "
            f"to move (endpoint={endpoint}, packets={subscriber.packets_received})"
        )
        subscriber.stop()
        return 2
    if subscriber.latest(config.bottom_key)[0] is None:
        print(
            "[edge-test] WARNING: bottom camera not in the stream — low floor objects "
            "cannot be ruled out or ground-gated. Enable it on the Pi host with "
            "--bottom_camera_enabled=true (see slam_three_camera_front_priority_mode)."
        )

    backstop = None
    if str(args.lidar_host).strip():
        backstop = LidarBackstop(str(args.lidar_host).strip(), int(args.lidar_port))
        backstop.calibrate_baseline()
    elif args.motion == "forward":
        print(
            "[edge-test] WARNING: no --lidar-host given; lidar cannot referee eye candidates "
            "and the camera gate is the ONLY thing stopping the robot"
        )

    lidar_ranges_fn = make_lidar_ranges_fn(backstop.feed) if backstop is not None else None
    monitor = ElevatedHazardMonitor(config, subscriber, lidar_ranges_fn=lidar_ranges_fn)
    monitor.start()
    time.sleep(0.6)  # let the hysteresis window fill before moving

    from lerobot.control.sourccey.sourccey.survey_rotation_protocol import (
        _apply_min_effective_magnitude,
    )
    from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
    from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient

    drive_speed = _apply_min_effective_magnitude(
        float(args.drive_speed), minimum_abs=float(args.min_effective_speed)
    )
    turn_speed = _apply_min_effective_magnitude(
        abs(float(args.turn_speed)), minimum_abs=float(args.min_effective_turn_speed)
    )

    robot = SourcceyClient(SourcceyClientConfig(id=args.robot_id, remote_ip=args.remote_ip))
    robot.connect()
    _send_stop(robot)
    print(
        f"[edge-test] {args.motion} test running for up to {float(args.max_run_s):.0f}s — "
        "put a table edge in the robot's way and watch the banner. Ctrl+C to stop."
    )

    stops = 0
    was_moving = False
    last_text = None
    last_view_s = 0.0
    last_verdict_s = 0.0
    turn_direction = 1.0
    next_flip = time.monotonic() + float(args.sweep_s)
    deadline = time.monotonic() + float(args.max_run_s)
    try:
        while time.monotonic() < deadline:
            state = monitor.state()
            lidar_blocked = backstop.blocked() if backstop is not None else False
            if lidar_blocked:
                # The lidar box participates in the post-stop hold: no gate
                # may ratchet the robot forward through flickers.
                monitor.report_external_stop("lidar_stop_box")
            text, color, allowed = _status_for(state, str(args.motion), lidar_blocked)

            now = time.monotonic()
            measured_edges = monitor.drain_confirmed_edges()
            if measured_edges and now - last_verdict_s >= 1.0:
                latest_edge = measured_edges[-1]
                edge_width = (
                    (latest_edge.p2_robot_xy[0] - latest_edge.p1_robot_xy[0]) ** 2
                    + (latest_edge.p2_robot_xy[1] - latest_edge.p1_robot_xy[1]) ** 2
                ) ** 0.5
                print(
                    f"[verdict] REAL EDGE measured: {latest_edge.nearest_m:.2f}m away, "
                    f"{edge_width:.2f}m wide (visible span), {latest_edge.height_m:.2f}m tall "
                    f"({latest_edge.eye})"
                )
                last_verdict_s = now
            if text != last_text:
                print(f"[edge-test] {text}  |  {format_hazard_log(state)}")
                last_text = text
                _log_view(rr, monitor, config, text, color)
                last_view_s = now
            elif now - last_view_s >= 0.35:
                _log_view(rr, monitor, config, text, color)
                last_view_s = now

            if not allowed:
                if was_moving:
                    stops += 1
                    print(f"[edge-test] >>> STOP #{stops}: motion forcibly halted <<<")
                _send_stop(robot)
                was_moving = False
                time.sleep(0.1)
                continue

            if str(args.motion) == "turn":
                if now >= next_flip:
                    turn_direction = -turn_direction
                    next_flip = now + float(args.sweep_s)
                    print(f"[edge-test] sweep direction -> {'ccw' if turn_direction > 0 else 'cw'}")
                _send_motion(robot, theta_vel=turn_direction * turn_speed)
            else:
                _send_motion(robot, x_vel=drive_speed)
            was_moving = True
            time.sleep(0.07)
    except KeyboardInterrupt:
        print("[edge-test] interrupted")
    finally:
        # BaseException: a second Ctrl+C lands here as KeyboardInterrupt
        # (not an Exception) and must not abort the stop/cleanup sequence.
        try:
            _send_stop(robot)
        except BaseException:
            pass
        try:
            robot.disconnect()
        except BaseException:
            pass
        try:
            monitor.stop()
            subscriber.stop()
            if backstop is not None:
                backstop.stop()
        except BaseException:
            pass

    print(f"\n[edge-test] session over: the gate forcibly stopped motion {stops} time(s)")
    if str(args.motion) == "forward" and stops == 0:
        print(
            "[edge-test] the gate NEVER stopped the robot — if a table edge was in the way, "
            "recheck the ROI band / thresholds in watch mode before trusting --elevated-safety"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
