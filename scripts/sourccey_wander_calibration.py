"""Sourccey motion calibration: learn how far/much the robot ACTUALLY moves vs.
what it was commanded, using the LiDAR as ground truth, and cross-check the IMU.

WHY THIS EXISTS
    When the robot is briefly blind (crossing a doorway into an unmapped room),
    it has to dead-reckon: "I was here, I commanded to move forward ~0.6m and
    turn ~90deg, so I should now be there." Open-loop that is wrong — wheels slip,
    a commanded 0.6m might really be 0.4m. This routine measures the mismatch once
    so the dead-reckoning fallback can correct for it.

WHAT IT MEASURES  (LiDAR = absolute truth)
    * TRANSLATION: command a straight move, measure the real distance with the
      LiDAR (scan-match before vs. after). The IMU is a yaw gyro and CANNOT sense
      linear distance, so it takes no part in the translation factor — but its
      rotation reading during a straight move is logged as a drift sanity check
      (it should read ~0).
    * ROTATION: command an in-place turn, measure the real rotation with the LiDAR,
      AND record what the IMU's gyro reported. This yields two factors: a wheel
      factor (commanded->actual) and an IMU factor (imu_reported->actual). The IMU
      is NEVER trusted as truth; the point is to learn whether, after scaling, it
      is consistent enough to serve as a cross-check.

FAIL-HARD (per operator directive: no silent fallbacks)
    If the LiDAR feed or the IMU drops out at any point, the run ABORTS with a
    clear message and a non-zero exit code. Both sensors are mandatory here.

SETUP
    Place the robot with clear space ahead and walls/features in LiDAR view (the
    scan-match needs stable geometry to measure against). Run from the repo root:

      uv run python scripts/sourccey_wander_calibration.py --remote-ip 192.168.1.237

    Writes artifacts/sourccey_motion_calibration.json (commit code, not the file).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from ldlidar_auto_snapshot_stitch import _execute_turn_burst, _send_stop
from ldlidar_direct_snapshot_client import _scan_to_local_points
from ldlidar_direct_snapshot_stitch import Pose2D
from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient
from sourccey_wander.imu_heading import ImuYawClient
from sourccey_wander.lidar_feed import SelfMaskedLidarFeed
from sourccey_wander.localization import _estimate_pose_against_stitched_map
from sourccey_wander.wander_types import _normalize_angle_deg, _turn_lever_arm_local_delta


def _abort(reason: str) -> "None":
    """Print a loud failure and exit non-zero. Used for every mandatory-sensor
    dropout so the operator sees exactly why calibration stopped."""
    print(f"\n[calib] ABORT: {reason}")
    sys.exit(1)


def _capture_points(feed, args, after_frame_id: int):
    """Grab the next fresh LiDAR revolution and reduce it to local (x, y) points.

    Returns (frame_id, points_xy). ABORTS if the feed delivers no fresh revolution
    within the timeout — that is the "LiDAR cut out" case the operator asked to
    treat as fatal, not something to paper over.
    """
    frame_id, frame = feed.wait_for_frame_after(
        after_frame_id=int(after_frame_id),
        timeout_s=float(args.fresh_frame_timeout_s),
        min_frame_advances=1,
        armed_wall_ts=time.time(),
    )
    if frame is None:
        _abort("LiDAR feed delivered no fresh revolution (sensor dropped out).")
    points = _scan_to_local_points(
        points=frame.points,
        forward_angle_deg=float(args.forward_angle_deg),
        valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
        invert_lateral_axis=bool(args.invert_lateral_axis),
        max_distance_m=float(args.max_distance_m),
        min_confidence=int(args.min_confidence),
        min_range_m=float(args.min_range_m),
    )
    if len(points) < 40:
        _abort(
            f"Only {len(points)} usable LiDAR points — too sparse to measure motion. "
            "Reposition the robot where the LiDAR sees stable walls/features."
        )
    return int(frame_id), points


def _read_imu_or_abort(imu: ImuYawClient) -> float:
    """Read a fresh IMU yaw, or ABORT. The IMU is mandatory during calibration
    (its whole point here is to be measured), so a dropout is fatal by directive."""
    yaw = imu.deg_fresh(wait_up_to_s=0.6, max_age_s=0.4)
    if yaw is None:
        _abort("IMU yaw sample went stale/absent (sensor dropped out).")
    return float(yaw)


def _measure_motion(before_points, after_points, guess: Pose2D, *, resolution_m, search_xy_m, theta_window_deg):
    """LiDAR ground-truth of how the robot moved between two scans.

    Treats the BEFORE scan as a tiny fixed map (at the identity pose) and solves
    where the AFTER scan sits relative to it, seeded by the commanded guess. The
    solved pose IS the real motion: x/y = how far the sensor translated, theta =
    how far it rotated. Returns (solved_pose, score) or None if the match failed.
    """
    result = _estimate_pose_against_stitched_map(
        points_xy=after_points,
        transformed_sets=[before_points],
        initial_pose=guess,
        resolution_m=float(resolution_m),
        search_xy_m=float(search_xy_m),
        theta_window_deg=float(theta_window_deg),
        max_translation_from_initial_m=float(search_xy_m) + 0.20,
        prior_translation_weight=0.02,   # only a whisper of bias toward the guess;
        prior_theta_weight=0.02,         # the LiDAR match should decide, not the guess
    )
    if result is None:
        return None
    solved_pose, meta = result
    return solved_pose, float(meta.get("score") or -1e9)


def _drive_straight(robot, args, *, signed_speed: float, nominal_distance_m: float) -> float:
    """Open-loop straight move. Commands a forward/back velocity for the time that
    WOULD cover nominal_distance_m at that speed, then stops. Returns the nominal
    (commanded) distance — the real distance is measured afterward by LiDAR."""
    speed = abs(float(signed_speed))
    if speed < 1e-6:
        return 0.0
    duration_s = float(nominal_distance_m) / speed
    x_vel = float(signed_speed)
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        robot.send_action(
            {
                "x.vel": x_vel,
                "y.vel": 0.0,
                "theta.vel": 0.0,
                "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
                "untorque_left": True,
                "untorque_right": True,
            }
        )
        time.sleep(0.04)
    _send_stop(robot)
    time.sleep(0.6)  # let the base fully settle before the "after" scan
    return float(nominal_distance_m)


def _rotate(robot, args, *, signed_angle_deg: float) -> float:
    """Open-loop in-place turn via fixed turn-bursts. Returns the nominal
    (commanded) rotation in degrees; the real rotation is measured by LiDAR."""
    turn_speed = float(args.turn_speed)
    turn_burst_s = float(args.turn_burst_s)
    deg_per_burst = max(1.0, math.degrees(turn_speed * turn_burst_s))
    n_bursts = max(1, int(round(abs(float(signed_angle_deg)) / deg_per_burst)))
    sign = 1.0 if float(signed_angle_deg) >= 0.0 else -1.0
    for _ in range(n_bursts):
        _execute_turn_burst(
            robot=robot,
            direction_sign=sign,
            turn_speed=turn_speed,
            turn_burst_s=turn_burst_s,
            turn_settle_s=0.08,
        )
    _send_stop(robot)
    time.sleep(0.6)
    return float(sign * n_bursts * deg_per_burst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-ip", required=True, help="Robot host IP.")
    parser.add_argument("--robot-id", default="sourccey")
    parser.add_argument("--lidar-host", default=None, help="LiDAR stream host (defaults to --remote-ip).")
    parser.add_argument("--lidar-port", type=int, default=8765)
    parser.add_argument("--imu-host", default=None, help="IMU yaw host (defaults to --remote-ip).")
    parser.add_argument("--imu-yaw-port", type=int, default=8770)
    parser.add_argument("--imu-yaw-sign", type=float, default=1.0)
    parser.add_argument("--forward-angle-deg", type=float, default=90.0)
    parser.add_argument("--valid-angle-half-width-deg", type=float, default=180.0)
    parser.add_argument("--invert-lateral-axis", action="store_true", default=True)
    parser.add_argument("--max-distance-m", type=float, default=8.0)
    parser.add_argument("--min-range-m", type=float, default=0.03)
    parser.add_argument("--min-confidence", type=int, default=0)
    parser.add_argument("--move-speed", type=float, default=0.85)
    parser.add_argument("--turn-speed", type=float, default=0.82)
    parser.add_argument("--turn-burst-s", type=float, default=0.24)
    parser.add_argument("--stitch-resolution-m", type=float, default=0.03)
    parser.add_argument("--lidar-offset-forward-m", type=float, default=0.229)
    parser.add_argument("--fresh-frame-timeout-s", type=float, default=3.0)
    parser.add_argument("--min-match-score", type=float, default=7.0,
                        help="Reject a measurement whose LiDAR scan-match scores below this (weak/ambiguous view).")
    parser.add_argument("--forward-nominal-m", type=float, default=0.6)
    parser.add_argument("--turn-nominal-deg", type=float, default=90.0)
    parser.add_argument("--reps", type=int, default=2, help="Repetitions of each move.")
    parser.add_argument("--output", default="artifacts/sourccey_motion_calibration.json")
    args = parser.parse_args()

    lidar_host = args.lidar_host or args.remote_ip
    imu_host = args.imu_host or args.remote_ip

    # ---- Connect the LiDAR feed (mandatory) ----
    feed = SelfMaskedLidarFeed(lidar_host, int(args.lidar_port))
    feed.start()
    print(f"[calib] LiDAR feed connecting to {lidar_host}:{args.lidar_port} ...")
    first_id, first_frame = feed.wait_for_frame_after(after_frame_id=-1, timeout_s=8.0, min_frame_advances=1)
    if first_frame is None:
        _abort("No LiDAR frames within 8s at startup (feed not running?).")
    last_frame_id = int(first_id)

    # ---- Calibrate the arm self-mask (same as the wander loop) ----
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
        masked_bins, masked_deg = feed.calibrate_self_mask(mask_frames)
        last_frame_id = mask_id
        print(f"[calib] self-mask armed: {masked_deg:.0f}deg of bearings filtered (robot arms).")
    else:
        print("[calib] WARNING: could not gather enough frames to arm the self-mask; continuing unmasked.")

    # ---- Connect the robot ----
    robot = SourcceyClient(SourcceyClientConfig(id=args.robot_id, remote_ip=args.remote_ip))
    robot.connect()
    _send_stop(robot)
    print(f"[calib] robot connected ({args.remote_ip}).")

    # ---- Connect the IMU (mandatory) ----
    imu = ImuYawClient(f"tcp://{imu_host}:{int(args.imu_yaw_port)}", sign=float(args.imu_yaw_sign))
    imu.start()
    print(f"[calib] waiting for IMU yaw feed ({imu_host}:{args.imu_yaw_port}) ...")
    imu_deadline = time.time() + 8.0
    while time.time() < imu_deadline and imu.deg() is None:
        time.sleep(0.1)
    if imu.deg() is None:
        _abort("No IMU yaw samples within 8s (host not publishing? IMU unplugged?).")
    print("[calib] IMU yaw feed live.")

    # ---- Build the move plan ----
    moves = []
    for _ in range(max(1, int(args.reps))):
        moves.append(("forward", +float(args.move_speed), float(args.forward_nominal_m)))
        moves.append(("back", -float(args.move_speed), float(args.forward_nominal_m)))
    for _ in range(max(1, int(args.reps))):
        moves.append(("turn_ccw", 0.0, +float(args.turn_nominal_deg)))
        moves.append(("turn_cw", 0.0, -float(args.turn_nominal_deg)))

    samples = []
    try:
        for move_index, (kind, signed_speed, nominal) in enumerate(moves):
            print(f"\n[calib] move {move_index + 1}/{len(moves)}: {kind} (nominal={nominal:+.2f})")
            # --- BEFORE: scan + IMU ---
            last_frame_id, before_pts = _capture_points(feed, args, last_frame_id)
            imu_before = _read_imu_or_abort(imu)

            # --- COMMAND the open-loop move ---
            if kind in ("forward", "back"):
                nominal_cmd = _drive_straight(robot, args, signed_speed=signed_speed, nominal_distance_m=nominal)
                guess = Pose2D(x=float(nominal_cmd), y=0.0, theta_deg=0.0)   # +x local = forward
                search_xy_m = abs(float(nominal_cmd)) + 0.45
                theta_window_deg = 20.0
            else:
                nominal_cmd = _rotate(robot, args, signed_angle_deg=nominal)
                lever_dx, lever_dy = _turn_lever_arm_local_delta(
                    nominal_cmd, lidar_offset_forward_m=float(args.lidar_offset_forward_m)
                )
                guess = Pose2D(x=float(lever_dx), y=float(lever_dy), theta_deg=float(nominal_cmd))
                search_xy_m = abs(float(args.lidar_offset_forward_m)) * 2.0 + 0.30
                theta_window_deg = abs(float(nominal_cmd)) * 0.5 + 45.0

            # --- AFTER: scan + IMU ---
            last_frame_id, after_pts = _capture_points(feed, args, last_frame_id)
            imu_after = _read_imu_or_abort(imu)
            imu_delta = float(imu_after) - float(imu_before)   # RAW gyro change (never wrapped)

            # --- MEASURE the real motion with the LiDAR ---
            measured = _measure_motion(
                before_pts, after_pts, guess,
                resolution_m=float(args.stitch_resolution_m),
                search_xy_m=search_xy_m, theta_window_deg=theta_window_deg,
            )
            if measured is None:
                print("[calib]   SKIP: LiDAR could not solve the motion (too little overlap).")
                continue
            solved, score = measured
            actual_dist_m = math.hypot(float(solved.x), float(solved.y))
            actual_dtheta_deg = _normalize_angle_deg(float(solved.theta_deg))
            reliable = score >= float(args.min_match_score)
            if kind in ("turn_ccw", "turn_cw") and abs(abs(actual_dtheta_deg) - abs(nominal_cmd)) > 40.0:
                # The LiDAR most likely locked onto a room-symmetric MIRROR mode
                # (reading ~0 or ~180deg instead of the real ~90deg turn) rather
                # than measuring a genuine 40deg+ of wheel slip. Exclude it so a
                # bad lock can't poison the rotation factors.
                reliable = False

            record = {
                "kind": kind,
                "nominal_distance_m": float(nominal_cmd) if kind in ("forward", "back") else 0.0,
                "nominal_rotation_deg": 0.0 if kind in ("forward", "back") else float(nominal_cmd),
                "lidar_actual_distance_m": float(actual_dist_m),
                "lidar_actual_rotation_deg": float(actual_dtheta_deg),
                "imu_delta_deg": float(imu_delta),
                "match_score": float(score),
                "reliable": bool(reliable),
            }
            samples.append(record)
            flag = "" if reliable else "  (LOW SCORE — excluded from factors)"
            if kind in ("forward", "back"):
                print(f"[calib]   commanded {nominal_cmd:+.2f}m -> LiDAR measured {actual_dist_m:.3f}m; "
                      f"IMU rotation during move {imu_delta:+.1f}deg (should be ~0); score={score:.1f}{flag}")
            else:
                print(f"[calib]   commanded {nominal_cmd:+.1f}deg -> LiDAR measured {actual_dtheta_deg:+.1f}deg; "
                      f"IMU reported {imu_delta:+.1f}deg; score={score:.1f}{flag}")
    finally:
        _send_stop(robot)

    # ---- Aggregate the factors from the RELIABLE samples ----
    def _mean(vals):
        vals = list(vals)
        return sum(vals) / len(vals) if vals else None

    def _stdev(vals, mean):
        vals = list(vals)
        if len(vals) < 2 or mean is None:
            return None
        return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))

    trans = [s for s in samples if s["kind"] in ("forward", "back") and s["reliable"] and s["nominal_distance_m"]]
    rots = [s for s in samples if s["kind"] in ("turn_ccw", "turn_cw") and s["reliable"] and s["nominal_rotation_deg"]]

    translation_scale = _mean(s["lidar_actual_distance_m"] / abs(s["nominal_distance_m"]) for s in trans)
    rotation_scale_wheel = _mean(
        abs(s["lidar_actual_rotation_deg"]) / abs(s["nominal_rotation_deg"]) for s in rots
    )
    # IMU factor: multiply the IMU's reported rotation by this to get truth. Only
    # meaningful where the IMU actually reported a rotation.
    imu_rot_samples = [s for s in rots if abs(s["imu_delta_deg"]) > 1.0]
    imu_ratios = [abs(s["lidar_actual_rotation_deg"]) / abs(s["imu_delta_deg"]) for s in imu_rot_samples]
    rotation_scale_imu = _mean(imu_ratios)
    rotation_scale_imu_stdev = _stdev(imu_ratios, rotation_scale_imu)
    # How much the IMU drifted during straight moves (should be ~0):
    imu_straight_drift = [abs(s["imu_delta_deg"]) for s in trans]
    imu_drift_mean = _mean(imu_straight_drift)

    # Verdict on whether the IMU rotation is consistent enough to be a cross-check:
    imu_usable = (
        rotation_scale_imu is not None
        and rotation_scale_imu_stdev is not None
        and rotation_scale_imu > 0.2
        and (rotation_scale_imu_stdev / rotation_scale_imu) < 0.15  # <15% spread = consistent
    )

    calibration = {
        "schema": "sourccey.motion_calibration.v1",
        "translation_scale": translation_scale,          # multiply commanded distance by this to get real
        "rotation_scale_wheel": rotation_scale_wheel,     # multiply commanded turn by this to get real
        "rotation_scale_imu": rotation_scale_imu,         # multiply IMU-reported turn by this to get real
        "rotation_scale_imu_spread": rotation_scale_imu_stdev,
        "imu_straight_drift_deg_mean": imu_drift_mean,
        "imu_rotation_usable_as_check": bool(imu_usable),
        "n_translation_samples": len(trans),
        "n_rotation_samples": len(rots),
        "samples": samples,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    # ---- Human summary ----
    print("\n=========== CALIBRATION SUMMARY ===========")
    if translation_scale is not None:
        print(f"  Translation: a commanded 1.00m actually goes ~{translation_scale:.2f}m "
              f"({len(trans)} samples).")
    else:
        print("  Translation: NO reliable samples (reposition where the LiDAR sees features and rerun).")
    if rotation_scale_wheel is not None:
        print(f"  Rotation (wheels): a commanded 90deg actually turns ~{rotation_scale_wheel * 90.0:.0f}deg "
              f"({len(rots)} samples).")
    else:
        print("  Rotation (wheels): NO reliable samples.")
    if rotation_scale_imu is not None:
        verdict = "USABLE as a cross-check" if imu_usable else "TOO INCONSISTENT to trust (spread too high)"
        print(f"  Rotation (IMU): IMU-reported x {rotation_scale_imu:.2f} = truth, "
              f"spread +/-{(rotation_scale_imu_stdev or 0.0):.2f} -> {verdict}.")
    else:
        print("  Rotation (IMU): the IMU reported no usable rotation (dead/again?).")
    if imu_drift_mean is not None:
        print(f"  IMU drift during straight moves: ~{imu_drift_mean:.1f}deg (ideally ~0).")
    print(f"  Written: {out_path}")
    print("===========================================")

    imu.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
