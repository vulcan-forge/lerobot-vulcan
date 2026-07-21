"""Sourccey SPIN-MAP — the clean-slate foundation (2026-07-20, user rebuild).

ONE job, done right, with nothing else in the way:

    Spin the robot a single continuous 360 degrees (no stop-start bursts),
    gated by the IMU + LiDAR, accumulate every LiDAR return taken DURING that
    spin into one world-frame point cloud, and render the finished map live in
    the rerun viewer. That is the whole program.

Deliberately NOT here (the point is a minimal, trustworthy base):
    - no cameras, no depth model, no semantic/edge detection,
    - no exploration, frontier planning, doorway logic, or recovery,
    - no snapshot-by-snapshot scan-match stitching.

How it stays honest:
    - The IMU yaw (published continuous/unwrapped) is the rotation clock: the
      spin ends when the gyro says a full ``--spin-degrees`` has turned. The
      calibration proved the gyro tracks relative rotation to ~0.5deg.
    - Each LiDAR revolution's points are placed in the world by the heading the
      IMU reports at that instant (the robot pivots in place, so the only motion
      is rotation, plus the small fixed lidar lever-arm which is accounted for).
    - The self-mask (arms/shell) is calibrated once at startup so the robot's own
      body is filtered out of the map, same as the main system.

Run from the repo root:
    uv run --with rerun-sdk python scripts/sourccey_spin_map.py --remote-ip 192.168.1.237
"""
from __future__ import annotations

import argparse
import math
import socket
import sys
import time

import numpy as np

from ldlidar_auto_snapshot_stitch import _init_rerun, _send_stop
from ldlidar_direct_snapshot_stitch import Pose2D, _search_pose, _transform_points
from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient
from sourccey_wander.imu_heading import ImuYawClient
from sourccey_wander.lidar_feed import SelfMaskedLidarFeed


def _abort(reason: str) -> "None":
    print(f"\n[spin] ABORT: {reason}")
    sys.exit(1)


def _pick_free_ports(grpc_port: int, web_port: int) -> tuple[int, int]:
    """Step past ports a crashed run may have leaked, so a rerun always opens.

    Bind on 0.0.0.0 (the address rerun's own server binds) — a 127.0.0.1 probe
    falsely passes while a stale server holds 0.0.0.0:<port>, which is exactly how
    the gRPC server then crashes with 'address already in use'."""
    def _free(port: int) -> bool:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("0.0.0.0", int(port)))
            return True
        except OSError:
            return False
        finally:
            probe.close()

    for bump in range(0, 20, 2):
        candidate_grpc, candidate_web = int(grpc_port) + bump, int(web_port) + bump
        if _free(candidate_grpc) and _free(candidate_web):
            if bump:
                print(f"[rerun] ports {grpc_port}/{web_port} in use (stale server from a previous "
                      f"run?); using {candidate_grpc}/{candidate_web} instead")
            return candidate_grpc, candidate_web
    print(f"[rerun] WARNING: no free port pair near {grpc_port}/{web_port}; trying the defaults.")
    return int(grpc_port), int(web_port)


def _spin_command(robot: SourcceyClient, theta_vel: float) -> dict:
    """One continuous-rotation action (pivot in place, no translation)."""
    return {
        "x.vel": 0.0,
        "y.vel": 0.0,
        "theta.vel": float(theta_vel),
        "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
        "untorque_left": True,
        "untorque_right": True,
    }


def _scan_local(frame, args, heading_change_deg: float = 0.0, reverse_sweep: bool = False) -> np.ndarray:
    """One LiDAR revolution reduced to local (forward, lateral) points.

    When ``heading_change_deg`` is non-zero (the amount the robot rotated DURING
    this revolution), each point is MOTION-COMPENSATED: rotated back to the scan's
    END heading so the whole revolution reads as if captured in one instant. That
    un-bends the smear a moving capture has, giving the scan-matcher a clean scan —
    so we get crisp stitching WITHOUT ever stopping the spin. With 0 it is the
    plain extraction (identical to the main system's, e.g. for a settled scan).
    """
    pts = frame.points
    n = len(pts)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    angle = np.fromiter((float(p[0]) for p in pts), dtype=np.float64, count=n)
    dist = np.fromiter((float(p[1]) for p in pts), dtype=np.float64, count=n)
    conf = np.fromiter((float(p[2]) for p in pts), dtype=np.float64, count=n)
    idx = np.arange(n, dtype=np.float64)

    delta = ((angle - float(args.forward_angle_deg) + 180.0) % 360.0) - 180.0
    keep = (
        (conf >= float(args.min_confidence))
        & (dist >= float(args.min_range_m))
        & (dist <= float(args.max_distance_m))
        & (np.abs(delta) <= float(args.valid_angle_half_width_deg))
    )
    if not np.any(keep):
        return np.zeros((0, 2), dtype=np.float32)
    delta = delta[keep]
    dist = dist[keep]
    idx = idx[keep]

    if heading_change_deg and n > 1:
        # Point at sweep-fraction f (0=first sample, 1=last) was seen when the robot
        # had turned heading_change*f of the way; rotating CCW makes a fixed point's
        # local bearing DECREASE, so its bearing in the END frame is
        # delta - heading_change*(1 - f).
        frac = idx / float(n - 1)
        if reverse_sweep:
            frac = 1.0 - frac
        delta = delta - float(heading_change_deg) * (1.0 - frac)

    theta = np.radians(delta)
    fwd = dist * np.cos(theta)
    lat = dist * np.sin(theta)
    if bool(args.invert_lateral_axis):
        lat = -lat
    return np.column_stack([fwd, lat]).astype(np.float32)


def _occupancy_grid_points(pts_xy: np.ndarray, res_m: float, min_hits: int) -> tuple[np.ndarray, np.ndarray]:
    """Vote every point into a grid and keep only well-supported cells — the way
    real SLAM maps stay crisp. A true wall is seen by nearly every scan, so its
    cell racks up a huge count; the misalignment scatter around it only gets a few
    votes and is dropped. Returns (cell_centres_xy, hit_counts) for the kept cells."""
    if len(pts_xy) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    cells = np.round(np.asarray(pts_xy, dtype=np.float64) / float(res_m)).astype(np.int64)
    uniq, counts = np.unique(cells, axis=0, return_counts=True)
    keep = counts >= int(min_hits)
    centres = (uniq[keep].astype(np.float32) * float(res_m))
    return centres, counts[keep]


def _log_occupancy(rr, pts_xy: np.ndarray, counts: np.ndarray, heading_deg: float) -> None:
    """Render the occupancy grid: kept cells, brighter where more scans agree."""
    if len(pts_xy):
        xyz = np.column_stack(
            [pts_xy[:, 0], pts_xy[:, 1], np.zeros(len(pts_xy), dtype=np.float32)]
        ).astype(np.float32)
        # Brightness by vote strength (well-supported walls glow).
        strength = np.clip(counts / max(1.0, float(np.percentile(counts, 90))), 0.25, 1.0)
        colors = np.column_stack([
            (90 * strength).astype(np.uint8),
            (190 * strength).astype(np.uint8),
            (255 * strength).astype(np.uint8),
        ])
        rr.log("world/lidar_map", rr.Points3D(xyz, colors=colors, radii=0.02))
    rr.log("world/robot", rr.Points3D([[0.0, 0.0, 0.0]], colors=[[255, 130, 60]], radii=0.06))
    rr.log("spin/progress", rr.TextLog(f"spun {heading_deg:+.0f} deg"))


def _log_map(rr, transformed_sets: list[np.ndarray], poses: list[Pose2D], heading_deg: float) -> None:
    """Render the stitched world map (all placed snapshots) + the robot in rerun."""
    non_empty = [pts for pts in transformed_sets if len(pts)]
    if non_empty:
        allpts = np.concatenate(non_empty, axis=0)
        xyz = np.column_stack(
            [allpts[:, 0], allpts[:, 1], np.zeros(len(allpts), dtype=np.float32)]
        ).astype(np.float32)
        rr.log(
            "world/lidar_map",
            rr.Points3D(xyz, colors=[[120, 200, 255]] * len(xyz), radii=0.015),
        )
    rx, ry = (float(poses[-1].x), float(poses[-1].y)) if poses else (0.0, 0.0)
    rr.log("world/robot", rr.Points3D([[rx, ry, 0.0]], colors=[[255, 130, 60]], radii=0.06))
    rr.log("spin/progress", rr.TextLog(f"spun {heading_deg:+.0f} deg"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-ip", required=True, help="Robot host IP.")
    parser.add_argument("--robot-id", default="sourccey")
    parser.add_argument("--lidar-host", default=None, help="LiDAR stream host (defaults to --remote-ip).")
    parser.add_argument("--lidar-port", type=int, default=8765)
    parser.add_argument("--imu-host", default=None, help="IMU yaw host (defaults to --remote-ip).")
    parser.add_argument("--imu-yaw-port", type=int, default=8770)
    parser.add_argument("--imu-yaw-sign", type=float, default=1.0)
    # Spin.
    parser.add_argument("--spin-degrees", type=float, default=360.0,
                        help="How far to spin, measured by the IMU (one continuous turn).")
    parser.add_argument("--spin-speed", type=float, default=0.7,
                        help="Rotation rate command (rad/s). The IMU, not this value, decides when "
                             "the spin is complete, so it only needs to be smooth. Lower it if the "
                             "map smears; raise it to spin faster.")
    parser.add_argument("--max-spin-seconds", type=float, default=60.0,
                        help="Safety cap: stop spinning after this long even if the IMU never reaches "
                             "the target (a stalled base / dead gyro).")
    parser.add_argument("--snapshots", type=int, default=120,
                        help="Maximum scans to align (evenly strided from every revolution recorded "
                             "during the spin). Consecutive scans overlap almost entirely, so each "
                             "aligns tightly against the growing map — more scans = smoother tracking "
                             "of the base's wander; fewer = faster post-processing.")
    parser.add_argument("--settle-ms", type=float, default=0.0,
                        help="At each snapshot, briefly STOP and settle this long before grabbing the "
                             "scan (0 = never stop, keep spinning). Motion de-skew makes stopping "
                             "unnecessary; raise this only if you want the base to pause anyway.")
    parser.add_argument("--deskew", choices=["on", "off"], default="off",
                        help="Motion-compensate each moving scan before matching. Off by default (the "
                             "occupancy grid handles the smear; de-skew needs the sweep direction right).")
    # Occupancy-grid rendering (how real SLAM maps stay crisp: vote into cells, keep
    # only well-supported ones — the wall every scan agrees on survives, scatter drops).
    parser.add_argument("--map-mode", choices=["grid", "points"], default="grid",
                        help="'grid' = occupancy grid (crisp, votes out misalignment); 'points' = raw "
                             "point overlay.")
    parser.add_argument("--grid-res-m", type=float, default=0.03,
                        help="Occupancy-grid cell size for the RENDERED map (bigger = more solid walls).")
    parser.add_argument("--grid-min-hits", type=int, default=0,
                        help="Keep a cell only if this many scans voted for it (0 = auto ~15%% of scans). "
                             "Higher = crisper but sparser; lower = fuller but fuzzier.")
    parser.add_argument("--deskew-reverse", action="store_true", default=False,
                        help="Flip the sweep order for de-skew. If de-skew makes matching WORSE (lower "
                             "scores), the feed streams revolutions newest-first — set this.")
    # LiDAR point extraction (same conventions as the rest of the system).
    parser.add_argument("--forward-angle-deg", type=float, default=90.0)
    parser.add_argument("--valid-angle-half-width-deg", type=float, default=180.0)
    parser.add_argument("--invert-lateral-axis", action="store_true", default=True)
    parser.add_argument("--max-distance-m", type=float, default=8.0)
    parser.add_argument("--min-range-m", type=float, default=0.03)
    parser.add_argument("--min-confidence", type=int, default=0)
    parser.add_argument("--lidar-offset-forward-m", type=float, default=0.229,
                        help="Forward mount offset of the LiDAR from the pivot centre (lever-arm) — the "
                             "IMU-seeded initial guess for each scan-match.")
    # Sequential scan-to-map alignment (Hector-SLAM style). Consecutive revolutions
    # are ~0.1s apart, so the seed (previous solved pose + gyro delta) is accurate
    # to ~1-2deg and a few cm — the search windows stay TIGHT, which is what makes
    # the match unambiguous (no 90/180deg aliases) and lets it track the base's
    # translational wander during the spin instead of assuming a perfect pivot.
    parser.add_argument("--stitch-resolution-m", type=float, default=0.03,
                        help="Score-grid resolution for scan-matching.")
    parser.add_argument("--search-xy-m", type=float, default=0.22,
                        help="Translation search radius around the seeded guess (tracks the base's "
                             "wander between consecutive scans; keep tight).")
    parser.add_argument("--theta-window-deg", type=float, default=10.0,
                        help="Heading search window around the gyro-seeded guess (the gyro delta "
                             "between consecutive scans is ~0.5deg-accurate; keep tight).")
    parser.add_argument("--min-match-score", type=float, default=6.0,
                        help="Below this the solve is rejected and the scan is placed by dead-reckoning "
                             "(previous solved pose + gyro delta) instead of a bad alignment.")
    # Camera panorama (DISPLAY ONLY — a live view alongside the map; it feeds no
    # decision. Fail-soft: if the camera stream or panorama calibration is absent
    # the spin still maps, just without the panorama panel).
    parser.add_argument("--panorama", choices=["on", "off"], default="on")
    parser.add_argument("--slam-input-endpoint", default=None,
                        help="Camera stream endpoint (default tcp://<remote-ip>:5560).")
    parser.add_argument("--panorama-hz", type=float, default=5.0,
                        help="How often to refresh the panorama panel in rerun.")
    # Rerun.
    parser.add_argument("--rerun-mode", choices=["web", "local"], default="web")
    parser.add_argument("--rerun-grpc-port", type=int, default=9877)
    parser.add_argument("--rerun-web-port", type=int, default=9878)
    args = parser.parse_args()

    lidar_host = args.lidar_host or args.remote_ip
    imu_host = args.imu_host or args.remote_ip

    # ---- LiDAR feed (mandatory) ----
    feed = SelfMaskedLidarFeed(lidar_host, int(args.lidar_port))
    feed.start()
    print(f"[spin] LiDAR feed connecting to {lidar_host}:{args.lidar_port} ...")
    first_id, first_frame = feed.wait_for_frame_after(after_frame_id=-1, timeout_s=8.0, min_frame_advances=1)
    if first_frame is None:
        _abort("No LiDAR frames within 8s (feed not running?).")
    last_frame_id = int(first_id)

    # ---- Self-mask (filter the arms/shell out of the map) ----
    mask_frames = []
    mask_id = last_frame_id
    for _ in range(24):
        mid, mframe = feed.wait_for_frame_after(after_frame_id=mask_id, timeout_s=1.5, min_frame_advances=1)
        if mframe is None:
            break
        mask_id = int(mid)
        if len(mframe.points) < 150:
            continue
        mask_frames.append(mframe)
        if len(mask_frames) >= 8:
            break
    if len(mask_frames) >= 4:
        _, masked_deg = feed.calibrate_self_mask(mask_frames)
        last_frame_id = mask_id
        print(f"[spin] self-mask armed: {masked_deg:.0f}deg of bearings filtered (robot arms/shell).")
    else:
        print("[spin] WARNING: could not gather enough frames to arm the self-mask; mapping unmasked.")

    # ---- IMU (mandatory — it is the rotation clock) ----
    imu = ImuYawClient(f"tcp://{imu_host}:{int(args.imu_yaw_port)}", sign=float(args.imu_yaw_sign))
    imu.start()
    yaw0 = None
    imu_deadline = time.time() + 5.0
    while time.time() < imu_deadline:
        yaw0 = imu.deg_fresh(wait_up_to_s=0.5, max_age_s=0.5)
        if yaw0 is not None:
            break
    if yaw0 is None:
        _abort("No IMU yaw within 5s — the spin is IMU-gated, so a live gyro is required.")
    print(f"[spin] IMU live; start heading = {yaw0:+.1f}deg (gyro is the 360 clock).")

    # ---- Robot ----
    robot = SourcceyClient(SourcceyClientConfig(id=args.robot_id, remote_ip=args.remote_ip))
    robot.connect()
    _send_stop(robot)
    print(f"[spin] robot connected ({args.remote_ip}).")

    # ---- Rerun viewer ----
    grpc_port, web_port = _pick_free_ports(int(args.rerun_grpc_port), int(args.rerun_web_port))
    rr, _viewer_url = _init_rerun(
        session_name="sourccey_spin_map", mode=args.rerun_mode,
        grpc_port=grpc_port, web_port=web_port,
    )

    # ---- Optional camera panorama (display only, fail-soft) ----
    cam_sub = None
    eye_mosaic = None
    cam_left_key = cam_right_key = None
    if str(args.panorama) == "on":
        try:
            from sourccey_elevated_safety import (
                SlamCameraSubscriber,
                endpoint_from_remote_ip,
            )
            from sourccey_eye_panorama import load_perception_mosaic

            cam_left_key, cam_right_key = "front_left", "front_right"
            cam_endpoint = str(args.slam_input_endpoint or "").strip() or endpoint_from_remote_ip(
                args.remote_ip
            )
            cam_sub = SlamCameraSubscriber(
                endpoint=cam_endpoint, camera_keys=(cam_left_key, cam_right_key)
            )
            cam_sub.start()
            if not cam_sub.wait_for_frames(timeout_s=5.0, required=(cam_left_key, cam_right_key)):
                raise RuntimeError("no camera frames within 5s")
            eye_mosaic = load_perception_mosaic()
            print(f"[spin] panorama view ON (fused {eye_mosaic.virt.width}x{eye_mosaic.virt.height}, "
                  f"hfov={eye_mosaic.model.hfov_deg:.0f}deg) — rerun entity cameras/panorama")
        except Exception as exc:  # noqa: BLE001
            print(f"[spin] panorama view unavailable ({type(exc).__name__}: {exc}); "
                  "mapping continues without it.")
            if cam_sub is not None:
                try:
                    cam_sub.stop()
                except Exception:
                    pass
            cam_sub = None
            eye_mosaic = None

    def _log_panorama() -> None:
        """Compose + log the fused panorama to rerun (best-effort; never raises)."""
        if cam_sub is None or eye_mosaic is None:
            return
        try:
            left, _al = cam_sub.latest(cam_left_key)
            right, _ar = cam_sub.latest(cam_right_key)
            if left is None or right is None:
                return
            pano = eye_mosaic.compose(left, right)
            rr.log("cameras/panorama", rr.Image(pano[:, :, ::-1]))  # BGR -> RGB
        except Exception:
            pass

    lever_m = float(args.lidar_offset_forward_m)
    pano_interval_s = 1.0 / max(0.5, float(args.panorama_hz))
    last_pano_log = 0.0
    # Collect EVERY revolution as (clean local scan, IMU heading). After the spin,
    # SEQUENTIAL SCAN-TO-MAP alignment (Hector-SLAM style) places each scan by
    # matching it against the growing map — seeded by the previous SOLVED pose plus
    # the gyro delta, with tight windows. That tracks the base's translational
    # WANDER during the spin (a mecanum base does not pivot perfectly in place),
    # which is the error the raw IMU placement cannot see. The first scan anchors
    # the frame; the occupancy grid then votes out residual scatter.
    revolutions: list[tuple[np.ndarray, float]] = []
    reference_pts: list[np.ndarray] = []    # world points (IMU-placed) for the live view only
    total_points = 0
    prev_frame_heading = None
    deskew_on = str(args.deskew) == "on"
    last_ref_log = 0.0

    print(f"[spin] SPINNING one continuous {float(args.spin_degrees):.0f}deg turn — recording every "
          "revolution (sequential scan-to-map alignment + occupancy grid after) ...")
    spin_start = time.monotonic()
    completed = False
    try:
        while True:
            # Keep the base turning (one continuous command stream, never stops).
            robot.send_action(_spin_command(robot, float(args.spin_speed)))

            # Refresh the live panorama panel on its own cadence.
            now = time.monotonic()
            if now - last_pano_log >= pano_interval_s:
                last_pano_log = now
                _log_panorama()

            # Record EVERY revolution (clean local scan + IMU heading). Placed by
            # the raw IMU heading only for the live view; the accurate placement
            # (loop-closure-corrected) is computed after the spin.
            frame_id, frame = feed.latest()
            yaw_now = imu.deg()
            if frame is not None and int(frame_id) != last_frame_id and yaw_now is not None:
                last_frame_id = int(frame_id)
                heading_deg = float(yaw_now) - float(yaw0)   # relative to spin start
                heading_change = (
                    heading_deg - prev_frame_heading if prev_frame_heading is not None else 0.0
                )
                prev_frame_heading = heading_deg
                deskew_change = heading_change if deskew_on else 0.0
                local_xy = _scan_local(
                    frame, args, heading_change_deg=deskew_change,
                    reverse_sweep=bool(args.deskew_reverse),
                )
                if len(local_xy) >= 12:
                    revolutions.append((local_xy, heading_deg))
                    total_points += len(local_xy)
                    hr = math.radians(heading_deg)
                    imu_pose = Pose2D(
                        x=lever_m * math.cos(hr), y=lever_m * math.sin(hr), theta_deg=heading_deg
                    )
                    reference_pts.append(_transform_points(local_xy, imu_pose))
                    # Live rough view (throttled) so progress is visible as it spins.
                    now2 = time.monotonic()
                    if now2 - last_ref_log >= 0.3:
                        last_ref_log = now2
                        _log_map(rr, reference_pts, [], heading_deg)

            # Done when the gyro has turned the full target.
            if yaw_now is not None and abs(float(yaw_now) - float(yaw0)) >= float(args.spin_degrees):
                completed = True
                break
            if time.monotonic() - spin_start > float(args.max_spin_seconds):
                print("[spin] WARNING: spin time cap reached before the IMU hit the target "
                      "(base stalled or gyro drift?) — stopping with what was mapped.")
                break
            time.sleep(0.03)
    finally:
        _send_stop(robot)

    final_deg = None
    _y = imu.deg()
    if _y is not None:
        final_deg = float(_y) - float(yaw0)
    heading_for_log = final_deg if final_deg is not None else 0.0

    # ---- SEQUENTIAL SCAN-TO-MAP ALIGNMENT (Hector-SLAM style) -----------------
    # Pick up to N scans, evenly strided in TIME order, and place each one by
    # scan-matching it against the map built from all the scans before it.
    #
    # The seed for each match is the previous SOLVED pose advanced by the gyro
    # delta, with the lidar's lever-arm pivot modelled around the previous solved
    # robot centre. Consecutive scans are ~0.1-0.3s apart, so that seed is right to
    # ~1-2deg and a few cm — which lets the search windows stay TIGHT. Tight
    # windows are the whole trick: the match cannot snap to the room's 90/180deg
    # look-alikes, it is fast, and crucially it SOLVES the base's translational
    # wander during the spin (a mecanum base never pivots perfectly in place; the
    # earlier IMU-only placement assumed it did, which is where the smear came
    # from — the loop-closure run measured only -2.5deg of heading drift, so the
    # error was never heading).
    #
    # The first scan anchors the frame, so the map cannot fragment (that anchor is
    # what the round-robin refinement lacked). Loop closure is inherent: by the
    # time the spin returns to the start heading, the reference map already
    # contains the start-of-spin walls, so the final scans re-attach to them.
    n_rev = len(revolutions)
    stride = max(1, n_rev // max(1, int(args.snapshots)))
    picked = revolutions[::stride]

    aligned: list[np.ndarray] = []
    prev_pose: Pose2D | None = None
    prev_heading: float | None = None
    n_matched = 0
    n_deadreck = 0
    score_sum = 0.0
    align_t0 = time.monotonic()
    print(f"\n[spin] aligning {len(picked)}/{n_rev} scans sequentially against the growing map ...")
    for k, (local_xy, h) in enumerate(picked):
        if prev_pose is None:
            hr = math.radians(h)
            pose = Pose2D(x=lever_m * math.cos(hr), y=lever_m * math.sin(hr), theta_deg=h)
        else:
            # Dead-reckoned seed: previous solved robot centre, heading advanced by
            # the gyro delta, lidar riding the lever-arm circle about that centre.
            theta_seed = float(prev_pose.theta_deg) + (h - prev_heading)
            pr = math.radians(float(prev_pose.theta_deg))
            centre_x = float(prev_pose.x) - lever_m * math.cos(pr)
            centre_y = float(prev_pose.y) - lever_m * math.sin(pr)
            sr = math.radians(theta_seed)
            seed = Pose2D(
                x=centre_x + lever_m * math.cos(sr),
                y=centre_y + lever_m * math.sin(sr),
                theta_deg=theta_seed,
            )
            ref = np.concatenate(aligned, axis=0)
            if len(ref) > 9000:
                ref = ref[:: (len(ref) // 9000) + 1]
            solved, meta = _search_pose(
                snapshot_points_xy=local_xy,
                global_points_xy=ref,
                initial_pose=seed,
                resolution_m=float(args.stitch_resolution_m),
                search_xy_m=float(args.search_xy_m),
                coarse_angle_step_deg=2.0,
                fine_angle_step_deg=0.5,
                theta_window_deg=float(args.theta_window_deg),
                # Keep even the weak-score fallback sweep NEAR the seed — a wide
                # whole-map sweep is how a scan snaps to a rotational alias.
                whole_map_theta_center_deg=float(seed.theta_deg),
                whole_map_theta_window_deg=float(args.theta_window_deg) + 4.0,
                max_translation_from_initial_m=float(args.search_xy_m) + 0.08,
                prior_pose=seed,
                prior_translation_weight=0.08,
                prior_theta_weight=0.05,
            )
            score = float(meta.get("score") or 0.0)
            dtheta = ((float(solved.theta_deg) - theta_seed + 180.0) % 360.0) - 180.0
            jump = math.hypot(float(solved.x) - seed.x, float(solved.y) - seed.y)
            if score >= float(args.min_match_score) and abs(dtheta) <= 12.0 and jump <= 0.35:
                pose = solved
                n_matched += 1
                score_sum += score
            else:
                # Weak/implausible solve: trust the dead-reckon for this one scan
                # (gyro delta is ~0.5deg-accurate over 0.1s) and keep tracking.
                pose = seed
                n_deadreck += 1
        aligned.append(_transform_points(local_xy, pose))
        prev_pose = pose
        prev_heading = h
        if (k + 1) % 20 == 0:
            print(f"[spin]   {k + 1}/{len(picked)} placed "
                  f"({n_matched} matched, {n_deadreck} dead-reckoned)")
    mean_score = (score_sum / n_matched) if n_matched else 0.0
    print(f"[spin] alignment done in {time.monotonic() - align_t0:.1f}s: "
          f"{n_matched} matched (mean score {mean_score:.1f}), {n_deadreck} dead-reckoned.")

    all_pts = np.concatenate(aligned, axis=0) if aligned else np.zeros((0, 2), dtype=np.float32)

    # ---- Render: occupancy grid (crisp) or raw aligned points ----
    n_map = 0
    num_scans = len(aligned)
    if len(all_pts) and str(args.map_mode) == "grid":
        min_hits = (
            int(args.grid_min_hits) if int(args.grid_min_hits) > 0 else max(2, int(0.15 * num_scans))
        )
        centres, counts = _occupancy_grid_points(all_pts, float(args.grid_res_m), min_hits)
        n_map = len(centres)
        print(f"[spin] occupancy grid: kept {n_map} cells (>= {min_hits} of {num_scans} scans agreed).")
        _log_occupancy(rr, centres, counts, heading_for_log)
    elif len(all_pts):
        n_map = len(all_pts)
        _log_map(rr, [all_pts], [], heading_for_log)

    print("\n=========== SPIN-MAP COMPLETE ===========")
    print(f"  {'full turn' if completed else 'PARTIAL turn'}: "
          f"IMU measured {final_deg:+.0f}deg" if final_deg is not None else "  IMU reading unavailable")
    print(f"  {num_scans} scans aligned scan-to-map ({n_matched} matched, mean score {mean_score:.1f}, "
          f"{n_deadreck} dead-reckoned) -> {str(args.map_mode)} map "
          f"({n_map} {'cells' if args.map_mode == 'grid' else 'points'}).")
    print("  Map = world/lidar_map. Leave running to keep the viewer up; Ctrl+C to exit.")
    print("=========================================")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        _send_stop(robot)
        imu.stop()
        feed.stop()
        if cam_sub is not None:
            try:
                cam_sub.stop()
            except Exception:
                pass
        try:
            robot.disconnect()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
