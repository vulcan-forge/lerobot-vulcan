"""Sourccey motion calibration: learn how far/much the robot ACTUALLY moves vs.
what it was commanded. The LiDAR is the SOLE source of truth. The IMU is NEVER
trusted — it is logged only, as an aside.

WHY THIS EXISTS
    When the robot is briefly blind (crossing a doorway into an unmapped room),
    it has to dead-reckon: "I was here, I commanded to move forward ~0.6m and
    turn ~60deg, so I should now be there." Open-loop that is wrong — wheels slip,
    a commanded 0.6m might really be 0.15m. This measures the mismatch once so the
    dead-reckoning fallback can correct for it.

HOW IT STAYS HONEST  (LiDAR = the only truth)
    1. MAP THE AREA FIRST: rotate in place a full circle and stitch the scans into
       one reference map. A rich accumulated map gives a strong, unambiguous match.
    2. Measure every move by LOCALIZING the scan in that map (before AND after),
       with a NARROW search window centered on the expected pose — so the room's
       ~90/180deg rotational lookalikes fall outside the search and the LiDAR can't
       mirror-lock. The real motion is how the map-pose changed.
    3. TRANSLATION factor = LiDAR-measured distance / commanded distance.
       ROTATION factor    = LiDAR-measured turn / commanded turn.
    The IMU is recorded next to each move for reference only; it decides nothing.

FAIL-HARD  (per operator directive: no silent fallbacks)
    If the LiDAR feed drops out, the run ABORTS with a non-zero exit code. The
    IMU is NOT required — it is observational, so its absence is not fatal.

SETUP
    Place the robot where the LiDAR sees walls/furniture all around (the mapping
    step and the localization need structure). Run from the repo root:

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
from ldlidar_direct_snapshot_stitch import Pose2D, _transform_points
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


def _read_imu_observe(imu: ImuYawClient) -> float | None:
    """Read a fresh IMU yaw for LOGGING ONLY. The IMU is NOT trusted and NOT
    ground truth (operator directive: use the LiDAR). A dropout is not fatal —
    we just record None for that observation and carry on."""
    return imu.deg_fresh(wait_up_to_s=0.4, max_age_s=0.4)


def _localize(points_xy, map_sets, guess: Pose2D, *, resolution_m, search_xy_m, theta_window_deg):
    """Solve where a scan sits in the stitched reference MAP, seeded by a guess.

    The map (``map_sets``, a list of point clouds in the map frame) is the LiDAR's
    ground truth. The search window is kept NARROW and centered on the expected
    pose so the room's ~90/180deg rotational lookalikes fall OUTSIDE the search —
    this is what stops the mirror-locking that a wide before-vs-after single-frame
    match suffered. Returns (solved_pose, score) or None.
    """
    result = _estimate_pose_against_stitched_map(
        points_xy=points_xy,
        transformed_sets=map_sets,
        initial_pose=guess,
        resolution_m=float(resolution_m),
        search_xy_m=float(search_xy_m),
        theta_window_deg=float(theta_window_deg),
        max_translation_from_initial_m=float(search_xy_m) + 0.20,
        prior_translation_weight=0.02,
        prior_theta_weight=0.02,
    )
    if result is None:
        return None
    solved_pose, meta = result
    return solved_pose, float(meta.get("score") or -1e9)


def _advance_in_place_turn(pose: Pose2D, *, dtheta_deg: float, lidar_offset_forward_m: float) -> Pose2D:
    """Predict the pose after an in-place turn of dtheta_deg: the heading changes,
    and the LiDAR (mounted forward of the pivot) swings along an arc. Used only to
    SEED the narrow search window, never as a measurement."""
    lever_dx, lever_dy = _turn_lever_arm_local_delta(dtheta_deg, lidar_offset_forward_m=float(lidar_offset_forward_m))
    t = math.radians(float(pose.theta_deg))
    world_dx = math.cos(t) * lever_dx - math.sin(t) * lever_dy
    world_dy = math.sin(t) * lever_dx + math.cos(t) * lever_dy
    return Pose2D(x=float(pose.x) + world_dx, y=float(pose.y) + world_dy, theta_deg=float(pose.theta_deg) + float(dtheta_deg))


def _advance_forward(pose: Pose2D, *, distance_m: float) -> Pose2D:
    """Predict the pose after a straight move of distance_m along the current
    heading. Seeds the search window only."""
    t = math.radians(float(pose.theta_deg))
    return Pose2D(x=float(pose.x) + math.cos(t) * float(distance_m), y=float(pose.y) + math.sin(t) * float(distance_m),
                  theta_deg=float(pose.theta_deg))


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


def _build_reference_map(feed, robot, args, last_frame_id: int):
    """MAP THE AREA FIRST (operator directive). Rotate in place a full circle,
    capturing a scan at each step and stitching them into one reference map with
    the LiDAR alone. A rich accumulated map (many more points, from a full
    revolution) gives a far stronger, less-ambiguous rotation match than a single
    before-vs-after scan pair — this is what makes the LiDAR trustworthy for
    rotation without ever leaning on the IMU.

    Each step is solved against the map-so-far with a NARROW window centered on the
    expected heading, so the stitch itself cannot mirror-lock. Returns
    (map_sets, end_pose, last_frame_id).
    """
    step_deg = float(args.map_step_deg)
    n_steps = max(4, int(round(360.0 / max(step_deg, 10.0))))
    print(f"[calib] MAPPING the area first: {n_steps} steps of ~{step_deg:.0f}deg (LiDAR reference map) ...")

    last_frame_id, first_pts = _capture_points(feed, args, last_frame_id)
    map_sets = [first_pts]                       # first scan defines the map frame (identity pose)
    current = Pose2D(x=0.0, y=0.0, theta_deg=0.0)
    added = 1
    for step_index in range(n_steps):
        _rotate(robot, args, signed_angle_deg=step_deg)
        last_frame_id, pts = _capture_points(feed, args, last_frame_id)
        guess = _advance_in_place_turn(current, dtheta_deg=step_deg,
                                       lidar_offset_forward_m=float(args.lidar_offset_forward_m))
        loc = _localize(
            pts, map_sets, guess,
            resolution_m=float(args.stitch_resolution_m),
            search_xy_m=abs(float(args.lidar_offset_forward_m)) * 2.0 + 0.35,
            theta_window_deg=abs(step_deg) + 25.0,   # narrow: excludes the 90/180 aliases
        )
        if loc is None or loc[1] < float(args.min_match_score):
            score_str = "none" if loc is None else f"{loc[1]:.1f}"
            print(f"[calib]   map step {step_index + 1}/{n_steps}: weak match (score={score_str}); "
                  "keeping dead-reckoned heading, not adding this scan")
            current = guess                      # couldn't localize; keep the guess, don't pollute the map
            continue
        current = loc[0]
        map_sets.append(_transform_points(pts, current))   # add this scan to the map at its solved pose
        added += 1
    print(f"[calib] reference map built: {added}/{n_steps + 1} scans stitched.")
    if added < 3:
        _abort("Could not stitch a usable reference map (LiDAR sees too little structure here). "
               "Move the robot somewhere with walls/furniture in view and rerun.")
    return map_sets, current, last_frame_id


def _cone_clearance_m(frame, args, center_deg: float, cone_half_deg: float = 18.0) -> float:
    """Nearest LiDAR return within +-cone_half_deg of ``center_deg`` (an absolute
    scan bearing), or max_distance if that cone is empty. This is the number that
    STOPS an open-loop drive before it burrows — call it with the FORWARD bearing
    for a forward drive, or forward+180 for a reverse."""
    best = float(args.max_distance_m)
    for angle_deg, distance_m, confidence in frame.points:
        if int(confidence) < int(args.min_confidence):
            continue
        d = float(distance_m)
        if not (float(args.min_range_m) <= d <= float(args.max_distance_m)):
            continue
        if abs(_normalize_angle_deg(float(angle_deg) - float(center_deg))) <= float(cone_half_deg):
            best = min(best, d)
    return best


def _deepest_open_bearing_deg(frame, args):
    """Bearing (relative to forward, in the TURN-command sign convention) of the
    DEEPEST clear direction, plus its range. Bins the full surroundings, records
    each bin's NEAREST return, smooths, and returns the bin whose nearest return is
    FARTHEST — i.e. the direction with the most room to drive. Empty bins count as
    0 (not open) so a self-masked/no-data bearing is never mistaken for a corridor.
    Returns (delta_deg, range_m) or (None, 0.0) if the scan is empty. Unlike the
    wander loop's WIDEST-gap selector, this prefers DEPTH — the calibration needs a
    long clear run, not a wide shallow one."""
    half = 170.0
    bin_deg = 6.0
    n = max(9, int((half * 2.0) / bin_deg) + 1)
    edges = [-half + i * (2.0 * half) / (n - 1) for i in range(n)]
    nearest = [0.0] * n
    hits = [0] * n
    for angle_deg, distance_m, confidence in frame.points:
        if int(confidence) < int(args.min_confidence):
            continue
        d = float(distance_m)
        if not (float(args.min_range_m) <= d <= float(args.max_distance_m)):
            continue
        delta = _normalize_angle_deg(float(angle_deg) - float(args.forward_angle_deg))
        if abs(delta) > half:
            continue
        idx = int(round((delta + half) / bin_deg))
        idx = max(0, min(n - 1, idx))
        if hits[idx] == 0 or d < nearest[idx]:
            nearest[idx] = d
        hits[idx] += 1
    if not any(hits):
        return None, 0.0
    # Light smoothing so one noisy return does not decide the heading.
    smoothed = list(nearest)
    for i in range(n):
        lo = nearest[i - 1] if i > 0 else nearest[i]
        hi = nearest[i + 1] if i < n - 1 else nearest[i]
        smoothed[i] = 0.2 * lo + 0.6 * nearest[i] + 0.2 * hi
    best = max(range(n), key=lambda i: smoothed[i])
    return float(edges[best]), float(smoothed[best])


def _drive_gated(
    robot, feed, args, *, signed_speed: float, nominal_distance_m: float, cone_center_deg: float,
    stop_clearance_m: float = 0.55,
) -> float:
    """Open-loop drive (forward OR reverse) that STOPS the instant the LiDAR sees an
    obstacle within ``stop_clearance_m`` in the travel direction — so neither a
    drift drive nor its return can ever burrow into a wall. ``cone_center_deg`` is
    the scan bearing to watch (forward for a forward drive, forward+180 for a
    reverse). Returns the nominal commanded distance; the REAL distance is measured
    afterward by the map localization."""
    speed = abs(float(signed_speed))
    if speed < 1e-6:
        return 0.0
    duration_s = float(nominal_distance_m) / speed
    deadline = time.monotonic() + duration_s
    stopped_early = False
    while time.monotonic() < deadline:
        _, frame = feed.latest()
        if frame is not None and _cone_clearance_m(frame, args, float(cone_center_deg)) < float(stop_clearance_m):
            stopped_early = True
            break
        robot.send_action(
            {
                "x.vel": float(signed_speed),
                "y.vel": 0.0,
                "theta.vel": 0.0,
                "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
                "untorque_left": True,
                "untorque_right": True,
            }
        )
        time.sleep(0.04)
    _send_stop(robot)
    time.sleep(0.6)
    if stopped_early:
        direction = "ahead" if float(signed_speed) >= 0.0 else "behind"
        print(f"[calib]   (drive stopped early — LiDAR saw a wall {direction}; measuring what it covered)")
    return float(nominal_distance_m)


def _face_open_corridor(feed, robot, args, map_sets, current_pose, last_frame_id):
    """Rotate to face the LONGEST OPEN corridor before any forward driving, so the
    forward/drift drives are not shoved into a wall the robot happens to face
    (operator 2026-07-20 — a run measured only 0.44m of a 2m drive because it was
    nose-to-wall). Reuses the SAME live-scan gap selector the wander loop drives
    by, so the bearing sign already matches the turn command. The open-loop turn
    slips to ~0.6x the command, so re-localize against the reference map afterward
    to keep the tracked pose honest. Returns (current_pose, last_frame_id)."""
    frame_id, frame = feed.wait_for_frame_after(
        after_frame_id=int(last_frame_id),
        timeout_s=float(args.fresh_frame_timeout_s),
        min_frame_advances=1,
        armed_wall_ts=time.time(),
    )
    if frame is None:
        _abort("LiDAR feed delivered no fresh revolution while choosing the open corridor.")
    last_frame_id = int(frame_id)
    delta, clear_m = _deepest_open_bearing_deg(frame, args)
    if delta is None:
        print("[calib] no returns in the scan; keeping the current heading for the forward drives.")
        return current_pose, last_frame_id
    if clear_m < 1.2:
        print(
            f"[calib] WARNING: the deepest clear direction only opens ~{clear_m:.1f}m "
            f"(delta={delta:+.0f}deg) — this spot is too tight for a full drift measurement. "
            "The gated drive will stop short (no burrowing); move the robot to a more open "
            "spot for a clean drift number."
        )
    if abs(delta) < 15.0:
        print(
            f"[calib] already facing the deepest direction (delta={delta:+.0f}deg, "
            f"{clear_m:.1f}m clear); no pre-turn needed."
        )
        return current_pose, last_frame_id
    print(
        f"[calib] facing the deepest open direction before driving: turning {delta:+.0f}deg toward "
        f"{clear_m:.1f}m of clear space (so the forward drives have room)."
    )
    nominal_turn = _rotate(robot, args, signed_angle_deg=delta)
    # Re-anchor: seed at the ~0.6x-slip estimate, with a window that brackets the
    # slip range but stays clear of the room's 90deg rotational alias.
    guess = _advance_in_place_turn(
        current_pose, dtheta_deg=nominal_turn * 0.61, lidar_offset_forward_m=float(args.lidar_offset_forward_m)
    )
    last_frame_id, pts = _capture_points(feed, args, last_frame_id)
    loc = _localize(
        pts, map_sets, guess,
        resolution_m=float(args.stitch_resolution_m),
        search_xy_m=abs(float(args.lidar_offset_forward_m)) * 2.0 + 0.40, theta_window_deg=35.0,
    )
    if loc is not None and loc[1] >= float(args.min_match_score):
        current_pose = loc[0]
    else:
        current_pose = guess
        print("[calib]   (could not re-localize precisely after the corridor turn; using dead-reckon seed).")
    return current_pose, last_frame_id


def _measure_forward_drift(feed, robot, args, map_sets, current_pose, last_frame_id, imu):
    """Measure how a mecanum base curves when told to drive STRAIGHT (field
    2026-07-20, user). The magnitude calibration above uses short bursts where the
    curve is invisible; here we drive a LONG open-loop straight line (theta.vel=0)
    and decompose the LiDAR-measured displacement in the START-heading frame:

        forward_m  = how far it actually advanced along its start heading
        lateral_m  = how far it slid sideways (the drift we care about)
        yaw_deg    = how much its heading rotated (the other drift)

    Reported PER METRE of real forward travel, so the numbers feed a straight-line
    correction regardless of the speed/scale. The IMU yaw delta is recorded beside
    the LiDAR yaw so the operator can SEE the gyro agreement over a long drive (the
    turn samples already showed ~0.5deg agreement; this confirms it holds over
    distance). LiDAR stays the sole source of truth; the IMU is logged only.
    """
    drift_samples = []
    reps = max(1, int(args.drift_reps))
    print(
        f"\n[calib] MEASURING forward drift: {reps} long straight drives "
        f"(~{float(args.drift_nominal_m):.1f}m nominal each, open-loop / no steering)."
    )
    for rep in range(reps):
        last_frame_id, before_pts = _capture_points(feed, args, last_frame_id)
        imu_before = _read_imu_observe(imu)
        before_loc = _localize(
            before_pts, map_sets, current_pose,
            resolution_m=float(args.stitch_resolution_m), search_xy_m=0.45, theta_window_deg=25.0,
        )
        if before_loc is None or before_loc[1] < float(args.min_match_score):
            print(f"[calib]   drift {rep + 1}/{reps}: SKIP (could not localize BEFORE).")
            continue
        pose_before = before_loc[0]

        nominal_cmd = _drive_gated(
            robot, feed, args, signed_speed=float(args.move_speed),
            nominal_distance_m=float(args.drift_nominal_m),
            cone_center_deg=float(args.forward_angle_deg),
        )
        # Wide-ish search: the drive may curve tens of degrees and slide sideways,
        # but stays well inside the room's 90deg alias, so a 55deg window is safe.
        guess_after = _advance_forward(pose_before, distance_m=float(nominal_cmd))
        last_frame_id, after_pts = _capture_points(feed, args, last_frame_id)
        imu_after = _read_imu_observe(imu)
        after_loc = _localize(
            after_pts, map_sets, guess_after,
            resolution_m=float(args.stitch_resolution_m),
            search_xy_m=abs(float(nominal_cmd)) + 0.7, theta_window_deg=55.0,
        )
        if after_loc is None or after_loc[1] < float(args.min_match_score):
            print(f"[calib]   drift {rep + 1}/{reps}: SKIP (could not localize AFTER).")
            continue
        pose_after, after_score = after_loc

        th0 = math.radians(float(pose_before.theta_deg))
        dx = float(pose_after.x) - float(pose_before.x)
        dy = float(pose_after.y) - float(pose_before.y)
        forward_m = dx * math.cos(th0) + dy * math.sin(th0)
        lateral_m = -dx * math.sin(th0) + dy * math.cos(th0)
        yaw_deg = _normalize_angle_deg(float(pose_after.theta_deg) - float(pose_before.theta_deg))
        current_pose = pose_after
        imu_delta = (
            float(imu_after) - float(imu_before) if (imu_before is not None and imu_after is not None) else None
        )
        if abs(forward_m) < 0.15:
            print(
                f"[calib]   drift {rep + 1}/{reps}: only {forward_m:.2f}m forward — too short to measure "
                "drift reliably; skipping (increase --drift-nominal-m or check for an obstacle)."
            )
            continue
        drift_samples.append(
            {
                "forward_m": float(forward_m),
                "lateral_m": float(lateral_m),
                "yaw_deg": float(yaw_deg),
                "yaw_per_m_deg": float(yaw_deg / forward_m),
                "lateral_per_m": float(lateral_m / forward_m),
                "imu_yaw_delta_deg": (None if imu_delta is None else float(imu_delta)),
                "match_score": float(after_score),
            }
        )
        imu_str = "n/a" if imu_delta is None else f"{imu_delta:+.1f}deg"
        print(
            f"[calib]   drift {rep + 1}/{reps}: forward {forward_m:.2f}m, lateral {lateral_m:+.2f}m, "
            f"yaw {yaw_deg:+.1f}deg  ->  {yaw_deg / forward_m:+.1f}deg/m, {lateral_m / forward_m:+.2f}m/m  "
            f"(LiDAR yaw vs IMU obs {imu_str})"
        )
        # Reverse back toward the start so the next rep does not MARCH the robot
        # forward across the room (into a wall, or out of the reference map where
        # it can no longer localize). Open-loop reverse of the same commanded
        # distance returns it roughly to where this rep began; the next rep
        # re-localizes exactly before measuring, so residual error is harmless.
        if rep < reps - 1:
            _drive_gated(
                robot, feed, args, signed_speed=-float(args.move_speed),
                nominal_distance_m=float(nominal_cmd),
                cone_center_deg=float(args.forward_angle_deg) + 180.0,
            )
    return drift_samples, current_pose, last_frame_id


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
                        help="Reject any measurement whose LiDAR match against the reference map scores "
                             "below this (weak/ambiguous view).")
    parser.add_argument("--map-step-deg", type=float, default=45.0,
                        help="Step size for the pre-calibration mapping rotation (LiDAR reference map).")
    parser.add_argument("--forward-nominal-m", type=float, default=0.6)
    parser.add_argument("--turn-nominal-deg", type=float, default=60.0)
    parser.add_argument("--reps", type=int, default=3, help="Repetitions of each move.")
    parser.add_argument(
        "--drift-nominal-m",
        type=float,
        default=2.0,
        help="Nominal distance for the LONG open-loop straight drive that measures mecanum drift. Must "
        "be long enough that the curve accumulates (short bursts hide it). Real distance is measured by "
        "lidar; drift is reported PER METRE so the exact scale does not matter.",
    )
    parser.add_argument(
        "--drift-reps",
        type=int,
        default=3,
        help="Repetitions of the long straight drift-measurement drive.",
    )
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

    # ---- Connect the IMU (OBSERVATIONAL ONLY — never trusted, never ground truth) ----
    # Operator directive: use the LiDAR, do NOT trust the IMU. So the IMU is not
    # mandatory here and its dropout is not fatal; we only LOG its reading for the
    # record. The LiDAR is the sole source of truth.
    imu = ImuYawClient(f"tcp://{imu_host}:{int(args.imu_yaw_port)}", sign=float(args.imu_yaw_sign))
    imu.start()
    imu_deadline = time.time() + 4.0
    while time.time() < imu_deadline and imu.deg() is None:
        time.sleep(0.1)
    print("[calib] IMU yaw feed live (logged for reference only)." if imu.deg() is not None
          else "[calib] IMU absent/silent — continuing LiDAR-only (IMU is not required).")

    # ---- MAP THE AREA FIRST, then calibrate against that LiDAR map ----
    map_sets, current_pose, last_frame_id = _build_reference_map(feed, robot, args, last_frame_id)

    # ---- Build the move plan ----
    # FORWARD MOTION FIRST, TURNS LAST (operator directive 2026-07-20: never turn
    # toward a wall and THEN drive forward into it). The straight moves — including
    # the long drift drives — all run while the robot still faces the clear
    # direction the operator set it up in; the in-place turns come afterward.
    forward_moves = []
    for _ in range(max(1, int(args.reps))):
        forward_moves.append(("forward", +float(args.move_speed), float(args.forward_nominal_m)))
        forward_moves.append(("back", -float(args.move_speed), float(args.forward_nominal_m)))
    turn_moves = []
    for _ in range(max(1, int(args.reps))):
        turn_moves.append(("turn_ccw", 0.0, +float(args.turn_nominal_deg)))
        turn_moves.append(("turn_cw", 0.0, -float(args.turn_nominal_deg)))
    total_moves = len(forward_moves) + len(turn_moves)

    samples = []
    drift_samples = []

    def _run_moves(move_list, index_offset: int) -> None:
        """Execute one phase of the move plan, appending a sample per move. Mutates
        the outer tracked pose / frame cursor (nonlocal) so phases chain."""
        nonlocal current_pose, last_frame_id
        for i, (kind, signed_speed, nominal) in enumerate(move_list):
            move_index = index_offset + i
            print(f"\n[calib] move {move_index + 1}/{total_moves}: {kind} (nominal={nominal:+.2f})")

            # --- BEFORE: localize the current scan in the reference MAP (LiDAR truth) ---
            last_frame_id, before_pts = _capture_points(feed, args, last_frame_id)
            imu_before = _read_imu_observe(imu)   # logged only
            before_loc = _localize(
                before_pts, map_sets, current_pose,
                resolution_m=float(args.stitch_resolution_m), search_xy_m=0.45, theta_window_deg=25.0,
            )
            if before_loc is None or before_loc[1] < float(args.min_match_score):
                bscore = "none" if before_loc is None else f"{before_loc[1]:.1f}"
                print(f"[calib]   SKIP: could not localize BEFORE the move against the map (score={bscore}).")
                continue
            pose_before = before_loc[0]

            # --- COMMAND the open-loop move, and PREDICT where it lands (to seed the search) ---
            if kind in ("forward", "back"):
                nominal_cmd = _drive_straight(robot, args, signed_speed=signed_speed, nominal_distance_m=nominal)
                signed_dist = nominal_cmd if kind == "forward" else -nominal_cmd
                guess_after = _advance_forward(pose_before, distance_m=signed_dist)
                search_xy_m = abs(float(nominal_cmd)) + 0.45
                theta_window_deg = 20.0        # a straight move barely rotates -> tiny theta window
            else:
                nominal_cmd = _rotate(robot, args, signed_angle_deg=nominal)
                guess_after = _advance_in_place_turn(
                    pose_before, dtheta_deg=nominal_cmd, lidar_offset_forward_m=float(args.lidar_offset_forward_m)
                )
                search_xy_m = abs(float(args.lidar_offset_forward_m)) * 2.0 + 0.30
                theta_window_deg = 30.0        # NARROW around the expected turn -> excludes 90/180 aliases

            # --- AFTER: localize the new scan in the SAME map, seeded at the prediction ---
            last_frame_id, after_pts = _capture_points(feed, args, last_frame_id)
            imu_after = _read_imu_observe(imu)   # logged only
            after_loc = _localize(
                after_pts, map_sets, guess_after,
                resolution_m=float(args.stitch_resolution_m), search_xy_m=search_xy_m, theta_window_deg=theta_window_deg,
            )
            if after_loc is None or after_loc[1] < float(args.min_match_score):
                ascore = "none" if after_loc is None else f"{after_loc[1]:.1f}"
                print(f"[calib]   SKIP: could not localize AFTER the move against the map (score={ascore}).")
                continue
            pose_after, after_score = after_loc

            # --- The real motion = how the map-pose changed. LiDAR ONLY. ---
            actual_dist_m = math.hypot(float(pose_after.x) - float(pose_before.x),
                                       float(pose_after.y) - float(pose_before.y))
            actual_dtheta_deg = _normalize_angle_deg(float(pose_after.theta_deg) - float(pose_before.theta_deg))
            current_pose = pose_after           # carry the tracked pose forward for the next move
            # IMU is LOGGED ONLY — it never gates or corrects anything:
            imu_delta = (
                float(imu_after) - float(imu_before) if (imu_before is not None and imu_after is not None) else None
            )

            record = {
                "kind": kind,
                "nominal_distance_m": float(nominal_cmd) if kind in ("forward", "back") else 0.0,
                "nominal_rotation_deg": 0.0 if kind in ("forward", "back") else float(nominal_cmd),
                "lidar_actual_distance_m": float(actual_dist_m),
                "lidar_actual_rotation_deg": float(actual_dtheta_deg),
                "imu_delta_deg": (None if imu_delta is None else float(imu_delta)),
                "match_score": float(after_score),
                "reliable": True,               # both localizations already passed the score gate
            }
            samples.append(record)
            imu_str = "n/a" if imu_delta is None else f"{imu_delta:+.1f}deg"
            if kind in ("forward", "back"):
                print(f"[calib]   commanded {nominal_cmd:+.2f}m -> LiDAR measured {actual_dist_m:.3f}m "
                      f"(map score {after_score:.1f})  [imu obs only: {imu_str} rot, expect ~0]")
            else:
                print(f"[calib]   commanded {nominal_cmd:+.1f}deg -> LiDAR measured {actual_dtheta_deg:+.1f}deg "
                      f"(map score {after_score:.1f})  [imu obs only: {imu_str}]")

    try:
        # 0) Point down the longest open corridor so the forward drives are not
        #    shoved into a wall (operator 2026-07-20). Uses the map already built.
        current_pose, last_frame_id = _face_open_corridor(
            feed, robot, args, map_sets, current_pose, last_frame_id
        )
        # 1) All FORWARD motion first (short magnitude bursts, then the long drift
        #    drives) — the robot stays pointed the clear way the operator set it up.
        _run_moves(forward_moves, 0)
        drift_samples, current_pose, last_frame_id = _measure_forward_drift(
            feed, robot, args, map_sets, current_pose, last_frame_id, imu
        )
        # 2) Only THEN the in-place turns.
        _run_moves(turn_moves, len(forward_moves))
    finally:
        _send_stop(robot)

    # ---- Aggregate the factors from the RELIABLE samples ----
    def _mean(vals):
        vals = list(vals)
        return sum(vals) / len(vals) if vals else None

    trans = [s for s in samples if s["kind"] in ("forward", "back") and s["reliable"] and s["nominal_distance_m"]]
    rots = [s for s in samples if s["kind"] in ("turn_ccw", "turn_cw") and s["reliable"] and s["nominal_rotation_deg"]]

    # THE CALIBRATION FACTORS — LiDAR ONLY. Every reliable sample already passed the
    # map-localization score gate at both ends, so the LiDAR-measured motion is the
    # ground truth. The IMU takes no part in these numbers.
    translation_scale = _mean(s["lidar_actual_distance_m"] / abs(s["nominal_distance_m"]) for s in trans)
    rotation_scale_wheel = _mean(abs(s["lidar_actual_rotation_deg"]) / abs(s["nominal_rotation_deg"]) for s in rots)

    # IMU OBSERVATIONS ONLY — recorded for reference, NOT trusted, NOT used to
    # calibrate anything. How far the (untrusted) gyro landed from the LiDAR truth
    # on turns, and its drift on straight moves.
    imu_turn_gaps = [
        abs(s["lidar_actual_rotation_deg"] - s["imu_delta_deg"]) for s in rots if s["imu_delta_deg"] is not None
    ]
    imu_turn_gap_mean = _mean(imu_turn_gaps)
    imu_drift_mean = _mean(abs(s["imu_delta_deg"]) for s in trans if s["imu_delta_deg"] is not None)

    # LONG-DRIVE DRIFT (LiDAR-measured): the systematic curve of a "straight" drive,
    # reported per metre of real forward travel. The wander loop's IMU yaw-hold is a
    # CLOSED loop and does not require these numbers, but they quantify the problem,
    # confirm the gyro tracks yaw over distance, and provide a feed-forward term.
    drift_yaw_per_m_mean = _mean(s["yaw_per_m_deg"] for s in drift_samples)
    drift_lateral_per_m_mean = _mean(s["lateral_per_m"] for s in drift_samples)
    drift_imu_gaps = [
        abs(s["yaw_deg"] - s["imu_yaw_delta_deg"]) for s in drift_samples if s["imu_yaw_delta_deg"] is not None
    ]
    drift_imu_yaw_gap_mean = _mean(drift_imu_gaps)

    calibration = {
        "schema": "sourccey.motion_calibration.v4",
        "source_of_truth": "lidar_map",
        "imu_trusted": False,
        "translation_scale": translation_scale,        # multiply a commanded distance by this to get the real one
        "rotation_scale_wheel": rotation_scale_wheel,   # multiply a commanded turn by this to get the real one
        "n_translation_samples": len(trans),
        "n_rotation_samples": len(rots),
        # forward-drive drift (LiDAR-measured, per metre of real forward travel):
        "forward_yaw_drift_deg_per_m": drift_yaw_per_m_mean,
        "forward_lateral_drift_m_per_m": drift_lateral_per_m_mean,
        "n_drift_samples": len(drift_samples),
        # observations only — the IMU is not part of the calibration:
        "imu_observed_turn_gap_deg_mean": imu_turn_gap_mean,
        "imu_observed_straight_drift_deg_mean": imu_drift_mean,
        "imu_observed_longdrive_yaw_gap_deg_mean": drift_imu_yaw_gap_mean,
        "samples": samples,
        "drift_samples": drift_samples,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    # ---- Human summary (LiDAR is the truth; IMU shown as an aside only) ----
    print("\n=========== CALIBRATION SUMMARY (LiDAR-measured) ===========")
    if translation_scale is not None:
        print(f"  Translation: a commanded {args.forward_nominal_m:.2f}m short burst actually goes "
              f"~{translation_scale * args.forward_nominal_m:.2f}m (scale {translation_scale:.2f}, "
              f"{len(trans)} samples; short-burst dynamics).")
    else:
        print("  Translation: NO reliable samples — the LiDAR could not localize the moves against the "
              "map (reposition where it sees more structure and rerun).")
    if rotation_scale_wheel is not None:
        print(f"  Rotation (wheels): a commanded {args.turn_nominal_deg:.0f}deg actually turns "
              f"~{rotation_scale_wheel * args.turn_nominal_deg:.0f}deg (scale {rotation_scale_wheel:.2f}, "
              f"{len(rots)} samples).")
    else:
        print("  Rotation: NO reliable samples — the LiDAR could not localize the turns against the map "
              "(too little structure, or the map was too weak; try a more feature-rich spot).")
    if drift_yaw_per_m_mean is not None or drift_lateral_per_m_mean is not None:
        yaw_s = "n/a" if drift_yaw_per_m_mean is None else f"{drift_yaw_per_m_mean:+.1f}deg/m of yaw"
        lat_s = "n/a" if drift_lateral_per_m_mean is None else f"{drift_lateral_per_m_mean:+.2f}m/m sideways"
        print(f"  Forward drift (straight-drive curve, LiDAR): {yaw_s}, {lat_s} "
              f"({len(drift_samples)} long drives). The wander loop's IMU yaw-hold corrects this live.")
    else:
        print("  Forward drift: NO reliable long-drive samples (drive was too short or un-localizable; "
              "increase --drift-nominal-m or use a more feature-rich spot).")
    if imu_turn_gap_mean is not None or imu_drift_mean is not None:
        gap_str = "n/a" if imu_turn_gap_mean is None else f"~{imu_turn_gap_mean:.1f}deg off the LiDAR on turns"
        drift_str = "n/a" if imu_drift_mean is None else f"~{imu_drift_mean:.1f}deg drift on straight moves"
        longgap_str = (
            "" if drift_imu_yaw_gap_mean is None
            else f"; ~{drift_imu_yaw_gap_mean:.1f}deg off the LiDAR yaw over the LONG drives"
        )
        print(f"  IMU (observed): {gap_str}; {drift_str}{longgap_str}. "
              "(Relative yaw is now USED for the drive heading-hold — see the wander loop.)")
    print(f"  Written: {out_path}")
    print("============================================================")

    imu.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
