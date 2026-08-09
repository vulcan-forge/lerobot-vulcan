"""Low-level MOTION primitives — the only code that actually commands the base.

``_drive_forward_burst`` runs one short timed forward push (stopping instantly on
a hazard); ``_run_drive_sequence`` chains bursts with host-feedback tracking;
``_turn_with_arc_tracking`` rotates in place while tracking the turn against the
map arc; ``_solve_drive_step_pose`` re-solves the pose after a burst;
``_drive_with_tracking`` is the workhorse that drives toward a waypoint burst by
burst, re-localizing after each and honoring the stop box, the side-guard, and
the exit ratchet; ``_reverse_escape`` backs out of a pocket with a frame-to-frame
bump check; ``_map_clearance_behind_m`` measures how far it can safely reverse.
Every planner decision upstream funnels through here to become real wheel motion.
"""
from __future__ import annotations

import math
import time
import traceback

import numpy as np

from ..lidar.auto_snapshot_stitch import _execute_turn_burst, _send_stop
from ..lidar.direct_snapshot_client import DirectLidarFeed, _scan_to_local_points
from ..lidar.direct_snapshot_stitch import (
    Pose2D,
    _build_score_grids,
    _score_candidate,
    _solve_arc_pose_on_grids,
    _transform_points,
)
from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient
from .frontier import _build_world_occupancy, _dilate_bool_grid
from .stop_zone import StopZoneConfig, _blocked_points_for_frame
from .wander_types import (
    DriveBurstMeta,
    DriveSequenceMeta,
    _apply_min_effective_magnitude,
    _normalize_angle_deg,
    _poll_remote_base_state,
)

def _record_turn_exception(tb_text: str) -> None:
    """Append a turn-loop traceback to a durable file so an intermittent
    hardware exception is diagnosable even when the terminal buffer truncates
    it (field 2026-07-20). Best-effort: never raises."""
    import os

    path = os.path.join("artifacts", "wander_snapshot_stitch", "turn_crash_traceback.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(tb_text)
        handle.write("\n" + "=" * 70 + "\n")


def _drive_forward_burst(
    *,
    robot: SourcceyClient,
    feed: DirectLidarFeed,
    zone_cfg: StopZoneConfig,
    forward_speed: float,
    burst_s: float,
    steer_theta_vel: float,
    min_effective_move_speed: float,
    hazard_monitor=None,
    imu_yaw_fn=None,
    yaw_ref_deg: float | None = None,
    yaw_hold_gain: float = 0.0,
    yaw_hold_max: float = 0.0,
) -> DriveBurstMeta:
    """Drive forward for ONE short timed burst, stopping instantly on a hazard.

    Sends the forward (+ optional steer) velocity command in a tight loop for
    ``burst_s`` seconds, clamping speed up to the motor dead-band minimum, while
    watching the elevated-hazard monitor every tick — if it flags a table edge
    the burst aborts immediately. Records host-reported velocity/distance
    telemetry and returns it as a ``DriveBurstMeta`` (including whether a hazard
    stopped it). This is the smallest unit of forward motion; everything else
    stacks bursts on top of it.
    """
    start = time.monotonic()
    iterations = 0
    host_feedback_updates = 0
    host_forward_distance_estimate_m = 0.0
    host_x_vel_peak = 0.0
    host_theta_vel_peak = 0.0
    host_x_vel_last = 0.0
    host_theta_vel_last = 0.0
    stopped_by_hazard = False
    hazard_reason = ""
    squeeze_notified = False
    effective_forward_speed = _apply_min_effective_magnitude(
        float(forward_speed),
        minimum_abs=float(min_effective_move_speed),
    )
    while (time.monotonic() - start) < float(burst_s):
        frame_id, frame = feed.latest()
        if frame is not None:
            blocked_points = _blocked_points_for_frame(frame, zone_cfg)
            if blocked_points >= zone_cfg.blocked_trigger_count():
                print(
                    "[drive] stop box became occupied during forward burst "
                    f"(frame_id={frame_id}, blocked_points={blocked_points}, baseline={zone_cfg.baseline_points})"
                )
                break
        commanded_forward_speed = effective_forward_speed
        if hazard_monitor is not None:
            hazard_ok, hazard_reason_now = hazard_monitor.forward_allowed()
            if not hazard_ok:
                stopped_by_hazard = True
                hazard_reason = str(hazard_reason_now)
                print(
                    "[safety] elevated hazard denied forward motion during burst "
                    f"({hazard_reason}); halting"
                )
                break
            if str(hazard_reason_now) == "squeeze_creep":
                # Passable gap beside furniture: advance at the minimum
                # effective speed so the approach keeps refining the edge
                # and the narrow-corridor gate can veto in time.
                commanded_forward_speed = min(
                    effective_forward_speed, float(min_effective_move_speed)
                )
                if not squeeze_notified:
                    squeeze_notified = True
                    measured_width = getattr(
                        hazard_monitor.state(), "clear_width_m", None
                    )
                    width_note = (
                        f"measured gap {measured_width:.2f}m wide"
                        if measured_width is not None
                        else "gap width unmeasured"
                    )
                    print(
                        "[safety] passable gap beside furniture: creeping through "
                        f"at minimum speed (squeeze; {width_note})"
                    )
        # IMU YAW-HOLD (field 2026-07-20, user): mecanum bases curve when driving
        # "straight" — a pure-forward command produced up to ~40deg of unwanted yaw
        # and 25cm of lateral drift over a single leg, walking the robot into
        # furniture. The calibration proved the gyro tracks RELATIVE rotation to
        # ~0.5deg, so hold the leg's start heading in real time: each tick, command
        # a corrective theta.vel that counters however far the yaw has drifted from
        # the reference captured when the leg began. Uses only the CHANGE in yaw
        # (never an absolute frame), and falls back to the caller's lidar-derived
        # steer when the IMU is absent/stale/sign-untrustworthy (yaw_fn returns
        # None). The lidar remains the source of truth for POSITION between bursts.
        theta_cmd = float(steer_theta_vel)
        if (
            imu_yaw_fn is not None
            and yaw_ref_deg is not None
            and float(yaw_hold_gain) > 0.0
        ):
            yaw_now = imu_yaw_fn()
            if yaw_now is not None:
                drift_deg = float(yaw_now) - float(yaw_ref_deg)
                correction = -float(yaw_hold_gain) * drift_deg
                theta_cmd = max(-float(yaw_hold_max), min(float(yaw_hold_max), correction))
        robot.send_action(
            {
                "x.vel": float(commanded_forward_speed),
                "y.vel": 0.0,
                "theta.vel": float(theta_cmd),
                "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
                "untorque_left": True,
                "untorque_right": True,
            }
        )
        iterations += 1
        remote_base_state, is_fresh_state = _poll_remote_base_state(robot)
        host_x_vel_last = float(remote_base_state.get("x.vel", 0.0))
        host_theta_vel_last = float(remote_base_state.get("theta.vel", 0.0))
        host_x_vel_peak = max(host_x_vel_peak, abs(host_x_vel_last))
        host_theta_vel_peak = max(host_theta_vel_peak, abs(host_theta_vel_last))
        if is_fresh_state:
            host_feedback_updates += 1
            host_forward_distance_estimate_m += float(host_x_vel_last) * 0.05
        time.sleep(0.05)
    _send_stop(robot)
    return DriveBurstMeta(
        elapsed_s=float(time.monotonic() - start),
        iterations=int(iterations),
        host_feedback_updates=int(host_feedback_updates),
        host_forward_distance_estimate_m=float(host_forward_distance_estimate_m),
        host_x_vel_peak=float(host_x_vel_peak),
        host_theta_vel_peak=float(host_theta_vel_peak),
        host_x_vel_last=float(host_x_vel_last),
        host_theta_vel_last=float(host_theta_vel_last),
        stopped_by_hazard=bool(stopped_by_hazard),
        hazard_reason=str(hazard_reason),
    )


def _run_drive_sequence(
    *,
    robot: SourcceyClient,
    feed: DirectLidarFeed,
    zone_cfg: StopZoneConfig,
    forward_speed: float,
    burst_s: float,
    burst_count: int,
    inter_burst_pause_s: float,
    steer_theta_vel: float,
    min_effective_move_speed: float,
) -> DriveSequenceMeta:
    """Chain several forward bursts, checking the stop box between each.

    Runs up to ``burst_count`` ``_drive_forward_burst`` calls with a short pause
    between them, and after each burst reads a fresh frame and counts stop-box
    hits; if the box is tripped it halts the sequence early. Accumulates the
    host-feedback telemetry across all bursts into one ``DriveSequenceMeta``.
    This is the simpler open-loop driver (no per-burst re-localization) — the
    map-tracked workhorse is ``_drive_with_tracking``.
    """
    total_elapsed_s = 0.0
    total_iterations = 0
    bursts_completed = 0
    stopped_by_block = False
    last_blocked_points = 0
    host_feedback_updates = 0
    host_forward_distance_estimate_m = 0.0
    host_x_vel_peak = 0.0
    host_theta_vel_peak = 0.0
    host_x_vel_last = 0.0
    host_theta_vel_last = 0.0
    effective_forward_speed = _apply_min_effective_magnitude(
        float(forward_speed),
        minimum_abs=float(min_effective_move_speed),
    )

    for burst_index in range(max(1, int(burst_count))):
        burst_meta = _drive_forward_burst(
            robot=robot,
            feed=feed,
            zone_cfg=zone_cfg,
            forward_speed=float(forward_speed),
            burst_s=float(burst_s),
            steer_theta_vel=float(steer_theta_vel),
            min_effective_move_speed=float(min_effective_move_speed),
        )
        total_elapsed_s += float(burst_meta.elapsed_s)
        total_iterations += int(burst_meta.iterations)
        bursts_completed += 1
        host_feedback_updates += int(burst_meta.host_feedback_updates)
        host_forward_distance_estimate_m += float(burst_meta.host_forward_distance_estimate_m)
        host_x_vel_peak = max(host_x_vel_peak, float(burst_meta.host_x_vel_peak))
        host_theta_vel_peak = max(host_theta_vel_peak, float(burst_meta.host_theta_vel_peak))
        host_x_vel_last = float(burst_meta.host_x_vel_last)
        host_theta_vel_last = float(burst_meta.host_theta_vel_last)

        frame_id, frame = feed.latest()
        if frame is not None:
            last_blocked_points = _blocked_points_for_frame(frame, zone_cfg)
            print(
                "[drive] burst checkpoint "
                f"(index={burst_index + 1}/{int(burst_count)}, frame_id={frame_id}, blocked_points={last_blocked_points}, "
                f"cmd_x={effective_forward_speed:.3f}, cmd_theta={float(steer_theta_vel):.3f}, "
                f"host_dx_est={burst_meta.host_forward_distance_estimate_m:.3f}, "
                f"host_updates={burst_meta.host_feedback_updates}, host_x_peak={burst_meta.host_x_vel_peak:.3f}, "
                f"host_theta_peak={burst_meta.host_theta_vel_peak:.3f}, host_x_last={burst_meta.host_x_vel_last:.3f}, "
                f"host_theta_last={burst_meta.host_theta_vel_last:.3f})"
            )
            if last_blocked_points >= zone_cfg.blocked_trigger_count():
                stopped_by_block = True
                break
        if burst_index < int(burst_count) - 1 and float(inter_burst_pause_s) > 0.0:
            time.sleep(float(inter_burst_pause_s))

    return DriveSequenceMeta(
        elapsed_s=float(total_elapsed_s),
        iterations=int(total_iterations),
        bursts_completed=int(bursts_completed),
        stopped_by_block=bool(stopped_by_block),
        blocked_points=int(last_blocked_points),
        commanded_forward_speed=float(effective_forward_speed),
        steer_theta_vel=float(steer_theta_vel),
        host_feedback_updates=int(host_feedback_updates),
        host_forward_distance_estimate_m=float(host_forward_distance_estimate_m),
        host_x_vel_peak=float(host_x_vel_peak),
        host_theta_vel_peak=float(host_theta_vel_peak),
        host_x_vel_last=float(host_x_vel_last),
        host_theta_vel_last=float(host_theta_vel_last),
    )


def _turn_with_arc_tracking(
    *,
    robot: SourcceyClient,
    feed: DirectLidarFeed,
    transformed_sets: list[np.ndarray],
    start_pose: Pose2D,
    lidar_offset_forward_m: float,
    resolution_m: float,
    target_turn_deg: float,
    direction_sign: float,
    turn_speed: float,
    turn_burst_s: float,
    turn_settle_s: float,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    invert_lateral_axis: bool,
    max_distance_m: float,
    min_range_m: float,
    min_confidence: int,
    stop_tolerance_deg: float,
    max_bursts: int,
    min_track_score: float = 5.0,
    track_theta_half_window_deg: float = 32.0,
    max_burst_delta_deg: float = 50.0,
    stall_bursts_before_boost: int = 3,
    stall_speed_boost: float = 1.3,
    max_speed_scale: float = 2.0,
    max_consecutive_held: int = 3,
    hazard_monitor=None,
) -> dict[str, float]:
    """Rotate in place while tracking the pose CONTINUOUSLY with the lidar.

    After every burst (robot momentarily at rest) the fresh revolution is
    arc-solved against the stitched map inside a small theta window around the
    last known heading. Between bursts the robot turns at most a few tens of
    degrees, so the solve is unambiguous — the ~90deg symmetry modes of a
    square room are simply outside the window. The turn stops when the
    lidar-tracked rotation reaches the target, so the amount turned is
    measured geometry, not commanded wheel motion."""
    non_empty_sets = [points for points in transformed_sets if len(points)]
    global_points_xy = np.concatenate(non_empty_sets, axis=0) if non_empty_sets else np.zeros((0, 2), np.float32)
    grids = _build_score_grids(global_points_xy, resolution_m=float(resolution_m), padding_m=0.9)
    r = float(lidar_offset_forward_m)
    start_theta_rad = math.radians(float(start_pose.theta_deg))
    arc_center_xy = (
        float(start_pose.x) - r * math.cos(start_theta_rad),
        float(start_pose.y) - r * math.sin(start_theta_rad),
    )

    current_theta_deg = float(start_pose.theta_deg)
    turned_deg = 0.0
    missed_updates = 0
    consecutive_held = 0
    lock_lost = False
    consecutive_timeouts = 0
    stall_count = 0
    speed_scale = 1.0
    last_frame_id = -1
    frame_wait_timeout_s = max(2.0, float(turn_burst_s) + float(turn_settle_s) + 1.5)

    hazard_frozen = False
    for burst_index in range(1, int(max_bursts) + 1):
        if hazard_monitor is not None:
            turn_ok, turn_deny_reason = hazard_monitor.turn_allowed()
            if not turn_ok:
                # Near-tier elevated hazard: rotation is frozen by policy
                # (only backing away is allowed). End the turn with the
                # tracked progress so the motion hint stays truthful.
                hazard_frozen = True
                print(
                    f"[safety] {turn_deny_reason}: rotation frozen near an elevated "
                    f"hazard; ending turn at tracked={abs(turned_deg):.1f}deg"
                )
                break
        try:
            _execute_turn_burst(
                robot=robot,
                direction_sign=float(direction_sign),
                turn_speed=float(turn_speed) * float(speed_scale),
                turn_burst_s=float(turn_burst_s),
                turn_settle_s=float(turn_settle_s),
            )
            frame_id, frame = feed.wait_for_frame_after(
                after_frame_id=int(last_frame_id),
                timeout_s=float(frame_wait_timeout_s),
                min_frame_advances=1,
            )
        except Exception as exc:
            # An intermittent robot-comms or feed exception mid-turn must NOT
            # crash the whole run (field 2026-07-20: a hardware exception inside
            # the burst's send_action loop hard-crashed after ~2000 frames).
            # Stop the base, end the turn as lock-lost with the tracked progress
            # so the caller relocalizes/captures, and record the traceback for
            # diagnosis instead of dying.
            lock_lost = True
            missed_updates += 1
            print(
                f"[turn] burst={burst_index:02d} EXCEPTION during turn I/O "
                f"({type(exc).__name__}: {exc}); stopping the base and ending the "
                f"turn at tracked={abs(turned_deg):.1f}deg instead of crashing"
            )
            try:
                _record_turn_exception(traceback.format_exc())
            except Exception:
                pass
            try:
                _send_stop(robot)
            except Exception:
                pass
            break
        if frame is None or int(frame_id) == int(last_frame_id):
            consecutive_timeouts += 1
            print(f"[turn] burst={burst_index:02d} timed out waiting for a fresh revolution")
            if consecutive_timeouts >= 3:
                lock_lost = True
                missed_updates += consecutive_timeouts
                print(
                    f"[turn] burst={burst_index:02d} lidar feed stalled during turn; "
                    f"ending turn with best tracked progress={abs(turned_deg):.1f}deg "
                    "and continuing to capture/relocalize instead of crashing"
                )
                break
            continue
        consecutive_timeouts = 0
        last_frame_id = int(frame_id)

        points_xy = _scan_to_local_points(
            points=frame.points,
            forward_angle_deg=float(forward_angle_deg),
            valid_angle_half_width_deg=float(valid_angle_half_width_deg),
            invert_lateral_axis=bool(invert_lateral_axis),
            max_distance_m=float(max_distance_m),
            min_confidence=int(min_confidence),
            min_range_m=float(min_range_m),
        )
        if len(points_xy) < 12:
            missed_updates += 1
            consecutive_held += 1
            print(f"[turn] burst={burst_index:02d} too few valid points to track; holding")
            if consecutive_held >= max(1, int(max_consecutive_held)):
                lock_lost = True
                print(
                    f"[turn] burst={burst_index:02d} tracking lock lost; "
                    "stopping the turn instead of rotating blind"
                )
                break
        else:
            # Window slightly leads in the commanded direction; per-burst
            # rotation cannot reach the next symmetry mode from here.
            expected_theta_deg = current_theta_deg + float(direction_sign) * 8.0
            solved_pose, solved_score = _solve_arc_pose_on_grids(
                snapshot_points_xy=points_xy,
                grids=grids,
                arc_center_xy=arc_center_xy,
                lidar_offset_forward_m=r,
                expected_theta_deg=float(expected_theta_deg),
                theta_half_window_deg=float(track_theta_half_window_deg),
                theta_step_deg=1.5,
                center_slack_m=0.08,
                slack_step_m=0.04,
                theta_prior_weight_per_deg=0.01,
                refine=False,
            )
            delta_deg = _normalize_angle_deg(float(solved_pose.theta_deg) - current_theta_deg)
            delta_along_deg = float(delta_deg) * float(direction_sign)
            if (
                float(solved_score) >= float(min_track_score)
                and -8.0 <= delta_along_deg <= float(max_burst_delta_deg)
            ):
                current_theta_deg = float(solved_pose.theta_deg)
                turned_deg += float(delta_deg)
                consecutive_held = 0
                if abs(delta_deg) >= 1.5:
                    stall_count = 0
                    speed_scale = 1.0
                else:
                    # Locked but not moving: genuine wheel stall, safe to boost.
                    stall_count += 1
                print(
                    f"[turn] burst={burst_index:02d} tracked={abs(turned_deg):6.1f}deg "
                    f"(delta={delta_along_deg:+5.1f}deg) target={float(target_turn_deg):5.1f}deg "
                    f"score={float(solved_score):0.3f}"
                )
            else:
                # Low-confidence solve: the robot may still be rotating but we
                # cannot see how far. NEVER boost here, and stop the turn after
                # a few held bursts — continuing means rotating blind, which is
                # how the map got corrupted before.
                missed_updates += 1
                consecutive_held += 1
                print(
                    f"[turn] burst={burst_index:02d} tracked={abs(turned_deg):6.1f}deg "
                    f"(solve held: delta={delta_along_deg:+5.1f}deg score={float(solved_score):0.3f}) "
                    f"target={float(target_turn_deg):5.1f}deg"
                )
                if consecutive_held >= max(1, int(max_consecutive_held)):
                    lock_lost = True
                    print(
                        f"[turn] burst={burst_index:02d} tracking lock lost; "
                        "stopping the turn instead of rotating blind"
                    )
                    break

        if stall_count >= max(1, int(stall_bursts_before_boost)) and speed_scale < float(max_speed_scale):
            speed_scale = min(float(max_speed_scale), speed_scale * float(stall_speed_boost))
            stall_count = 0
            print(f"[turn] burst={burst_index:02d} no rotation progress; boosting turn speed (scale={speed_scale:.2f})")

        if abs(turned_deg) >= (float(target_turn_deg) - float(stop_tolerance_deg)):
            break

    _send_stop(robot)
    if hazard_monitor is not None and abs(turned_deg) > 0.5:
        # Tracked rotation releases reverse-only holds once the nose points
        # well away from the stopped-at hazard (wedged-pocket escape).
        report_rotation = getattr(hazard_monitor, "report_rotation", None)
        if callable(report_rotation):
            report_rotation(float(turned_deg))
    reached = abs(turned_deg) >= (float(target_turn_deg) - float(stop_tolerance_deg))
    if not reached:
        print(
            "[turn] warning: arc-tracked turn ended before reaching its target "
            f"(tracked={abs(turned_deg):.1f}deg of {float(target_turn_deg):.1f}deg, "
            f"missed_updates={missed_updates}, lock_lost={lock_lost})"
        )
    return {
        "turned_deg": float(turned_deg),
        "final_theta_deg": float(current_theta_deg),
        "missed_updates": float(missed_updates),
        "lock_lost": 1.0 if lock_lost else 0.0,
        "completed": 1.0 if reached else 0.0,
        "hazard_frozen": 1.0 if hazard_frozen else 0.0,
    }


def _solve_drive_step_pose(
    *,
    points_xy: np.ndarray,
    grids: dict[str, object],
    current_pose: Pose2D,
    max_step_m: float = 0.60,
    min_step_m: float = -0.05,
    lateral_slack_m: float = 0.15,
    theta_half_window_deg: float = 12.0,
) -> tuple[Pose2D, float]:
    """Solve one drive-burst pose update: the robot moved at most one burst
    forward from `current_pose`, so search a short strip ahead (with a little
    lateral slack and a small heading window). Small windows keep the solve
    unambiguous, the same principle as the arc-tracked turn."""
    heading_rad = math.radians(float(current_pose.theta_deg))
    c = math.cos(heading_rad)
    s = math.sin(heading_rad)
    best_pose = current_pose
    best_score = -1e9
    for dtheta_deg in np.arange(-theta_half_window_deg, theta_half_window_deg + 1e-6, 2.0, dtype=np.float32):
        for forward_m in np.arange(min_step_m, max_step_m + 1e-6, 0.05, dtype=np.float32):
            for lateral_m in np.arange(-lateral_slack_m, lateral_slack_m + 1e-6, 0.05, dtype=np.float32):
                pose = Pose2D(
                    x=float(current_pose.x) + c * float(forward_m) - s * float(lateral_m),
                    y=float(current_pose.y) + s * float(forward_m) + c * float(lateral_m),
                    theta_deg=float(current_pose.theta_deg) + float(dtheta_deg),
                )
                score = _score_candidate(
                    _transform_points(points_xy, pose),
                    use_nearest_penalty=False,
                    **grids,
                )
                if score > best_score:
                    best_score = float(score)
                    best_pose = pose
    return best_pose, float(best_score)


def _drive_with_tracking(
    *,
    robot: SourcceyClient,
    feed: DirectLidarFeed,
    zone_cfg: StopZoneConfig,
    transformed_sets: list[np.ndarray],
    start_pose: Pose2D,
    resolution_m: float,
    forward_speed: float,
    min_effective_move_speed: float,
    burst_s: float,
    burst_count: int,
    inter_burst_pause_s: float,
    steer_theta_vel: float,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    invert_lateral_axis: bool,
    max_distance_m: float,
    min_range_m: float,
    min_confidence: int,
    min_track_score: float = 5.0,
    max_consecutive_held: int = 2,
    hazard_monitor=None,
    map_side_guard=None,
    forward_guard=None,
    blind_forward_scale: float = 0.0,
    pose_trusted: bool = True,
    frame_recorder=None,
    imu_yaw_fn=None,
    yaw_hold_gain: float = 0.0,
    yaw_hold_max: float = 0.0,
) -> tuple[Pose2D, dict[str, object]]:
    """Drive forward burst-by-burst while tracking the pose with the lidar
    after every burst. The drive stops when blocked, when the plan completes,
    or when tracking loses lock — the robot never travels more than one burst
    beyond its last confirmed pose, so there is no big-jump solve afterwards.

    map_side_guard: optional callable(pose) -> str | None. A virtual side
    bumper from the ELEVATED MAP: tabletop edges BESIDE the robot are
    invisible to every live sensor (panorama covers ±52° forward, the lidar
    passes under tabletops, the bottom camera watches the floor), but the
    map remembers where they are. A non-None reason halts the drive like a
    hazard stop."""
    if not bool(pose_trusted):
        # HARD SAFETY RULE (field 2026-07-20: robot drove into a table and fell
        # over during a pose-lost recovery thrash). A robot that does not know
        # where it is must NOT drive FORWARD — a forward command on a lost pose
        # is a blind lunge, and the map-based guards below cannot be trusted
        # either. Recovery must be rotation-in-place or a mapped reverse only
        # (both handled by other code paths). Refuse the drive outright.
        print(
            "[drive] REFUSED: pose is LOST — no forward driving while unlocalized "
            "(rotate/reverse recovery only); the robot will not lunge blind"
        )
        return start_pose, {
            "bursts_completed": 0,
            "elapsed_s": 0.0,
            "stopped_by_block": True,
            "blocked_points": 0,
            "lock_lost": True,
            "stopped_by_hazard": False,
            "hazard_reason": "pose_lost_no_forward",
            "missed_updates": 0,
            "blind_forward_m": 0.0,
        }

    non_empty_sets = [points for points in transformed_sets if len(points)]
    global_points_xy = np.concatenate(non_empty_sets, axis=0) if non_empty_sets else np.zeros((0, 2), np.float32)
    grids = _build_score_grids(global_points_xy, resolution_m=float(resolution_m), padding_m=1.4)

    current_pose = start_pose
    bursts_completed = 0
    stopped_by_block = False
    lock_lost = False
    stopped_by_hazard = False
    hazard_reason = ""
    consecutive_held = 0
    missed_updates = 0
    last_blocked_points = 0
    last_frame_id = -1
    elapsed_total_s = 0.0
    # Calibrated dead-reckon of forward motion the lidar could NOT track. Each
    # burst physically drives ~this far; when the solve is held (blind), the
    # tracked pose misses it, so with a calibration scale we estimate it instead
    # of counting zero. Consumed by the exit-run door-crossing odometry so a blind
    # creep THROUGH the doorway still registers as progress.
    blind_forward_m = 0.0
    nominal_burst_forward_m = float(forward_speed) * float(burst_s) * float(blind_forward_scale)
    frame_wait_timeout_s = max(2.0, float(burst_s) + 1.5)
    # IMU yaw-hold reference: the heading the leg starts on, captured ONCE so every
    # burst holds the SAME straight line (re-reading it per burst would let an
    # accumulating drift redefine "straight" and bake the curve in). None when the
    # IMU is unusable — the bursts then fall back to the lidar-derived steer.
    yaw_ref_deg = imu_yaw_fn() if imu_yaw_fn is not None and float(yaw_hold_gain) > 0.0 else None

    for burst_index in range(max(1, int(burst_count))):
        if frame_recorder is not None:
            # Capture what the eyes see DURING the approach — this is when the
            # robot reaches a table edge (the main loop is blocked in here, so
            # without this the critical frames are never recorded).
            try:
                frame_recorder()
            except Exception:
                pass
        if map_side_guard is not None:
            guard_reason = map_side_guard(current_pose)
            if guard_reason:
                stopped_by_hazard = True
                hazard_reason = str(guard_reason)
                print(f"[drive] {guard_reason}; halting before burst {burst_index + 1:02d}")
                break
        # Independent EXIT-RUN hard-stop: a robot-frame ratchet that halts any
        # burst that has carried the robot back toward the room center past its
        # committed outward-max. Checked on the LIVE tracked pose every burst, so
        # it catches a retreat no matter which planner/recovery path issued the
        # drive — the last line of defense against "got to the door then left".
        if forward_guard is not None:
            forward_reason = forward_guard(current_pose)
            if forward_reason:
                stopped_by_block = True
                print(f"[drive] {forward_reason}; halting before burst {burst_index + 1:02d}")
                break
        burst_meta = _drive_forward_burst(
            robot=robot,
            feed=feed,
            zone_cfg=zone_cfg,
            forward_speed=float(forward_speed),
            burst_s=float(burst_s),
            steer_theta_vel=float(steer_theta_vel),
            min_effective_move_speed=float(min_effective_move_speed),
            hazard_monitor=hazard_monitor,
            imu_yaw_fn=imu_yaw_fn,
            yaw_ref_deg=yaw_ref_deg,
            yaw_hold_gain=float(yaw_hold_gain),
            yaw_hold_max=float(yaw_hold_max),
        )
        bursts_completed += 1
        elapsed_total_s += float(burst_meta.elapsed_s)
        if burst_meta.stopped_by_hazard:
            stopped_by_hazard = True
            hazard_reason = str(burst_meta.hazard_reason)

        frame_id, frame = feed.wait_for_frame_after(
            after_frame_id=int(last_frame_id),
            timeout_s=float(frame_wait_timeout_s),
            min_frame_advances=1,
        )
        if frame is None:
            missed_updates += 1
            consecutive_held += 1
            if not burst_meta.stopped_by_hazard and blind_forward_scale > 0.0:
                blind_forward_m += nominal_burst_forward_m   # drove but couldn't see it -> estimate
            print(f"[drive] burst={burst_index + 1:02d} no fresh revolution to track against")
        else:
            last_frame_id = int(frame_id)
            points_xy = _scan_to_local_points(
                points=frame.points,
                forward_angle_deg=float(forward_angle_deg),
                valid_angle_half_width_deg=float(valid_angle_half_width_deg),
                invert_lateral_axis=bool(invert_lateral_axis),
                max_distance_m=float(max_distance_m),
                min_confidence=int(min_confidence),
                min_range_m=float(min_range_m),
            )
            solved_pose, solved_score = _solve_drive_step_pose(
                points_xy=points_xy,
                grids=grids,
                current_pose=current_pose,
            )
            step_m = math.hypot(solved_pose.x - current_pose.x, solved_pose.y - current_pose.y)
            if solved_score >= float(min_track_score):
                current_pose = solved_pose
                consecutive_held = 0
                print(
                    f"[drive] burst={burst_index + 1:02d} tracked pose=({current_pose.x:.3f}, "
                    f"{current_pose.y:.3f}, {current_pose.theta_deg:.1f}deg) step={step_m:.3f}m "
                    f"score={solved_score:.3f}"
                )
            else:
                missed_updates += 1
                consecutive_held += 1
                if not burst_meta.stopped_by_hazard and blind_forward_scale > 0.0:
                    blind_forward_m += nominal_burst_forward_m   # drove but couldn't solve it -> estimate
                print(
                    f"[drive] burst={burst_index + 1:02d} solve held "
                    f"(score={solved_score:.3f}); not updating pose"
                )
            last_blocked_points = _blocked_points_for_frame(frame, zone_cfg)
            if last_blocked_points >= zone_cfg.blocked_trigger_count():
                stopped_by_block = True
                print(
                    f"[drive] burst={burst_index + 1:02d} stop box occupied "
                    f"(blocked_points={last_blocked_points}, baseline={zone_cfg.baseline_points})"
                )
                break

        if stopped_by_hazard:
            # Pose above is already tracked for the partial burst; stop the
            # drive here — the elevated hazard is invisible to the lidar, so
            # continuing would ram it.
            break
        if consecutive_held >= max(1, int(max_consecutive_held)):
            lock_lost = True
            print(
                f"[drive] burst={burst_index + 1:02d} tracking lock lost; "
                "stopping the drive instead of moving blind"
            )
            break
        if burst_index < int(burst_count) - 1 and float(inter_burst_pause_s) > 0.0:
            time.sleep(float(inter_burst_pause_s))

    _send_stop(robot)
    return current_pose, {
        "bursts_completed": int(bursts_completed),
        "elapsed_s": float(elapsed_total_s),
        "stopped_by_block": bool(stopped_by_block),
        "blocked_points": int(last_blocked_points),
        "lock_lost": bool(lock_lost),
        "stopped_by_hazard": bool(stopped_by_hazard),
        "hazard_reason": str(hazard_reason),
        "missed_updates": int(missed_updates),
        "blind_forward_m": float(blind_forward_m),
    }


def _map_clearance_behind_m(
    *,
    transformed_sets: list[np.ndarray],
    pose: Pose2D,
    resolution_m: float,
    robot_radius_m: float,
    lidar_offset_forward_m: float,
    max_check_m: float = 0.80,
) -> float:
    """How far straight backwards the MAP says the robot can move. The lidar
    cannot see behind the robot, but the space behind was traversed to get
    here, so the stitched map covers it. Marches from the ROBOT CENTER (the
    sensor sits lidar_offset forward of it) and ignores everything inside the
    robot's own footprint — any map point there is stitching noise, not an
    obstacle the robot could collide with."""
    occupancy = _build_world_occupancy(
        transformed_sets,
        resolution_m=max(0.03, float(resolution_m)),
        padding_m=0.5,
    )
    if occupancy is None:
        return 0.0
    occupied_grid, grid_origin_xy, _world_points = occupancy
    inflated_grid = _dilate_bool_grid(
        occupied_grid,
        radius_cells=max(1, int(round(float(robot_radius_m) / max(0.03, float(resolution_m))))),
    )
    heading_rad = math.radians(float(pose.theta_deg))
    back_dir = np.asarray([-math.cos(heading_rad), -math.sin(heading_rad)], dtype=np.float32)
    center_xy = np.asarray(
        [
            float(pose.x) - float(lidar_offset_forward_m) * math.cos(heading_rad),
            float(pose.y) - float(lidar_offset_forward_m) * math.sin(heading_rad),
        ],
        dtype=np.float32,
    )
    step_m = 0.05
    footprint_m = float(robot_radius_m)
    clearance_m = 0.0
    distance_m = step_m
    while distance_m <= footprint_m + float(max_check_m):
        sample_xy = center_xy + back_dir * float(distance_m)
        cell_xy = np.round((sample_xy - grid_origin_xy) / max(0.03, float(resolution_m))).astype(np.int32)
        cell_x = int(cell_xy[0])
        cell_y = int(cell_xy[1])
        if (
            cell_x < 0
            or cell_x >= int(inflated_grid.shape[1])
            or cell_y < 0
            or cell_y >= int(inflated_grid.shape[0])
        ):
            break
        if float(distance_m) > footprint_m:
            if bool(inflated_grid[cell_y, cell_x]):
                break
            clearance_m = float(distance_m) - footprint_m
        distance_m += step_m
    return clearance_m


def _reverse_escape(
    *,
    robot: SourcceyClient,
    feed: DirectLidarFeed,
    transformed_sets: list[np.ndarray],
    start_pose: Pose2D,
    resolution_m: float,
    robot_radius_m: float,
    lidar_offset_forward_m: float,
    reverse_speed: float,
    burst_s: float,
    bursts: int,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    invert_lateral_axis: bool,
    max_distance_m: float,
    min_range_m: float,
    min_confidence: int,
    min_track_score: float = 5.0,
    allow_blind: bool = False,
) -> tuple[Pose2D, dict[str, object]] | None:
    """Back straight out of a pocket the robot is wedged in. Rotating in place
    cannot free a robot whose stop box is occupied at every heading; reversing
    along the path it came in on can. Clearance is checked against the map
    (the lidar cannot see behind), and the resulting pose is re-solved from
    the lidar afterwards."""
    clearance_m = _map_clearance_behind_m(
        transformed_sets=transformed_sets,
        pose=start_pose,
        resolution_m=float(resolution_m),
        robot_radius_m=float(robot_radius_m),
        lidar_offset_forward_m=float(lidar_offset_forward_m),
    )
    # LIVE rear check: the map clearance is cast from the TRACKED pose, and
    # reverse escapes fire mostly during relocalization crises when that
    # pose is a stale fallback (field 2026-07-13: pose error >1m after a
    # doorway transit — "0.76m clear behind" was measured somewhere else and
    # the robot backed into the open door leaf). The live scan is robot-
    # relative, immune to pose error; ~270deg of it is usable, so much of
    # the rear quarter IS visible. Ranges below 0.65m are the robot's own
    # body (rear corner sits ~0.58m from the forward-offset lidar) and are
    # skipped; a fully occluded rear yields no points and falls back to the
    # map check unchanged.
    live_rear_m: float | None = None
    _live_id, live_frame = feed.latest()
    if live_frame is not None:
        all_points = _scan_to_local_points(
            points=live_frame.points,
            forward_angle_deg=float(forward_angle_deg),
            valid_angle_half_width_deg=180.0,
            invert_lateral_axis=bool(invert_lateral_axis),
            max_distance_m=3.0,
            min_confidence=int(min_confidence),
            min_range_m=float(lidar_offset_forward_m) + float(robot_radius_m) + 0.14,
        )
        if len(all_points):
            rear_bumper_x = -(float(lidar_offset_forward_m) + float(robot_radius_m))
            behind = (all_points[:, 0] < rear_bumper_x) & (
                np.abs(all_points[:, 1]) <= float(robot_radius_m) + 0.08
            )
            if np.any(behind):
                live_rear_m = float(np.min(-all_points[behind, 0]) + rear_bumper_x)
    # rear_visible: the live scan actually returned points in the rear
    # corridor, so we CAN see what is behind (not occluded). live_rear_m is
    # None either when the rear is genuinely open (no points) or when an
    # obstacle sits closer than the scan's near cutoff (~0.14m behind the
    # bumper) — those two are disambiguated by the map clearance below.
    rear_visible = live_rear_m is not None
    if live_rear_m is not None and live_rear_m < clearance_m:
        print(
            f"[wander] live lidar shows only {live_rear_m:.2f}m clear behind "
            f"(map claimed {clearance_m:.2f}m); trusting the live scan"
        )
        clearance_m = live_rear_m
    if clearance_m >= 0.30:
        target_reverse_m = min(0.45, clearance_m - 0.10)
        print(
            f"[wander] reversing out of pocket (map shows {clearance_m:.2f}m clear behind, "
            f"target={target_reverse_m:.2f}m)"
        )
    elif rear_visible:
        # The live lidar SEES the rear and it is tight — a real wall, not a
        # blind spot. Never drive into a wall we can see (field 2026-07-18:
        # the old blind reverse backed the robot into the wall behind it
        # repeatedly). Take only the genuinely clear margin; if there is
        # none, refuse and let the recovery ladder ROTATE instead.
        safe_reverse_m = clearance_m - 0.12
        if safe_reverse_m >= 0.08:
            target_reverse_m = safe_reverse_m
            reverse_speed = min(abs(float(reverse_speed)), 0.8)
            bursts = 1
            print(
                f"[wander] reversing the safe margin only ({target_reverse_m:.2f}m); live "
                f"lidar sees a wall {clearance_m:.2f}m behind"
            )
        else:
            print(
                f"[wander] cannot reverse-escape: live lidar sees a wall {clearance_m:.2f}m "
                "behind — refusing to back into it (recovery will rotate instead)"
            )
            return None
    elif allow_blind and clearance_m >= 0.15:
        # Rear OCCLUDED (no live points — too close to the scan cutoff or
        # genuinely empty) AND the map still credits some room: the robot is
        # wedged despite that, most likely pose drift. A short slow blind
        # nudge is the last resort; the stop box protects the front. NOT
        # taken when the map itself reports a wall right there (< 0.15m):
        # a confident map wall is a real wall, don't gamble into it.
        target_reverse_m = 0.18
        reverse_speed = min(abs(float(reverse_speed)), 0.8)
        bursts = 1
        print(
            f"[wander] WARNING: map shows {clearance_m:.2f}m clear behind but robot is "
            f"wedged and the rear is occluded; attempting short blind reverse "
            f"({target_reverse_m:.2f}m, slow)"
        )
    else:
        blocked_by = "map+live wall" if clearance_m < 0.15 else "rear occluded"
        print(
            f"[wander] cannot reverse-escape: only {clearance_m:.2f}m clear behind "
            f"({blocked_by}) — recovery will rotate instead"
        )
        return None
    # Pre-reverse reference scan for BUMP DETECTION: after the reverse, the
    # post-scan is matched against THIS scan directly (frame-to-frame, no map
    # involved) to measure how far the robot ACTUALLY moved. Immune to pose
    # loss — which is exactly when reverses fire and exactly when the old
    # dead-reckoning invented 0.8x-target motion that never happened (field
    # 2026-07-18: robot backed into a desk and kept commanding reverses at
    # the same fictional pose, grinding against it, because nothing checked
    # whether the scene had changed).
    pre_points_xy = None
    _pre_id, pre_frame = feed.latest()
    if pre_frame is not None:
        pre_points_xy = _scan_to_local_points(
            points=pre_frame.points,
            forward_angle_deg=float(forward_angle_deg),
            valid_angle_half_width_deg=float(valid_angle_half_width_deg),
            invert_lateral_axis=bool(invert_lateral_axis),
            max_distance_m=float(max_distance_m),
            min_confidence=int(min_confidence),
            min_range_m=float(min_range_m),
        )
        if len(pre_points_xy) < 30:
            pre_points_xy = None

    for _burst_index in range(max(1, int(bursts))):
        burst_deadline = time.monotonic() + float(burst_s)
        while time.monotonic() < burst_deadline:
            robot.send_action(
                {
                    "x.vel": -abs(float(reverse_speed)),
                    "y.vel": 0.0,
                    "theta.vel": 0.0,
                    "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
                    "untorque_left": True,
                    "untorque_right": True,
                }
            )
            time.sleep(0.05)
        _send_stop(robot)
        time.sleep(0.15)

    latest_frame_id, _ = feed.latest()
    frame_id, frame = feed.wait_for_frame_after(
        after_frame_id=int(latest_frame_id),
        timeout_s=3.0,
        min_frame_advances=1,
    )
    solved_pose = start_pose
    solved_score = -1e9
    if frame is not None:
        points_xy = _scan_to_local_points(
            points=frame.points,
            forward_angle_deg=float(forward_angle_deg),
            valid_angle_half_width_deg=float(valid_angle_half_width_deg),
            invert_lateral_axis=bool(invert_lateral_axis),
            max_distance_m=float(max_distance_m),
            min_confidence=int(min_confidence),
            min_range_m=float(min_range_m),
        )
        if len(points_xy) >= 12:
            non_empty_sets = [points for points in transformed_sets if len(points)]
            grids = _build_score_grids(
                np.concatenate(non_empty_sets, axis=0), resolution_m=float(resolution_m), padding_m=1.2
            )
            solved_pose, solved_score = _solve_drive_step_pose(
                points_xy=points_xy,
                grids=grids,
                current_pose=start_pose,
                max_step_m=0.10,
                min_step_m=-(target_reverse_m + 0.25),
                lateral_slack_m=0.12,
                theta_half_window_deg=10.0,
            )
    # BUMP DETECTION + honest motion measurement: match the post-reverse scan
    # against the pre-reverse scan. The solved translation IS the robot's real
    # motion between the two stationary moments, regardless of map/pose state.
    f2f_moved_m: float | None = None
    f2f_pose = None
    f2f_score = -1e9
    if pre_points_xy is not None and frame is not None and len(points_xy) >= 30:
        f2f_grids = _build_score_grids(
            pre_points_xy, resolution_m=float(resolution_m), padding_m=1.2
        )
        f2f_pose, f2f_score = _solve_drive_step_pose(
            points_xy=points_xy,
            grids=f2f_grids,
            current_pose=Pose2D(0.0, 0.0, 0.0),
            max_step_m=0.10,
            min_step_m=-(target_reverse_m + 0.25),
            lateral_slack_m=0.12,
            theta_half_window_deg=10.0,
        )
        if float(f2f_score) >= 6.0:
            f2f_moved_m = -float(f2f_pose.x)
    bumped = (
        f2f_moved_m is not None
        and target_reverse_m >= 0.10
        and f2f_moved_m < 0.35 * target_reverse_m
    )
    locked = float(solved_score) >= float(min_track_score)
    if bumped:
        # The wheels turned but the scene barely moved: the body is against
        # something the lidar cannot see behind it. The honest pose update is
        # the MEASURED (near-zero) motion — never the commanded distance.
        heading_rad = math.radians(float(start_pose.theta_deg))
        actual_m = max(0.0, float(f2f_moved_m))
        solved_pose = Pose2D(
            x=float(start_pose.x) - math.cos(heading_rad) * actual_m,
            y=float(start_pose.y) - math.sin(heading_rad) * actual_m,
            theta_deg=float(start_pose.theta_deg),
        )
        locked = True
        print(
            f"[wander] BUMP detected: commanded {target_reverse_m:.2f}m reverse but the "
            f"scene shows only {actual_m:.2f}m of real motion (frame-to-frame score "
            f"{float(f2f_score):.1f}) — the robot is pressed against an unseen obstacle "
            "behind it; reverse aborted and pose held at the measured position"
        )
    elif not locked:
        if f2f_moved_m is not None:
            # Map solve failed (typical while lost) but the frame-to-frame
            # match measured the true motion — use it instead of inventing
            # 0.8x the commanded distance.
            heading_rad = math.radians(float(start_pose.theta_deg))
            solved_pose = Pose2D(
                x=float(start_pose.x) - math.cos(heading_rad) * float(f2f_moved_m),
                y=float(start_pose.y) - math.sin(heading_rad) * float(f2f_moved_m),
                theta_deg=float(start_pose.theta_deg),
            )
            locked = True
            solved_score = float(f2f_score)
        else:
            # Dead-reckon the reverse; the wide capture search will refine it.
            heading_rad = math.radians(float(start_pose.theta_deg))
            solved_pose = Pose2D(
                x=float(start_pose.x) - math.cos(heading_rad) * target_reverse_m * 0.8,
                y=float(start_pose.y) - math.sin(heading_rad) * target_reverse_m * 0.8,
                theta_deg=float(start_pose.theta_deg),
            )
    print(
        "[wander] reverse escape "
        f"({'tracked' if locked else 'dead-reckoned'} pose=({solved_pose.x:.3f}, {solved_pose.y:.3f}, "
        f"{solved_pose.theta_deg:.1f}deg), score={float(solved_score):.3f}"
        f"{', BUMPED' if bumped else ''})"
    )
    return solved_pose, {
        "locked": bool(locked),
        "score": float(solved_score),
        "bumped": bool(bumped),
    }

