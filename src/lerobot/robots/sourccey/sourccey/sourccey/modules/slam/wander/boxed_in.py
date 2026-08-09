"""Recovery behavior: escape when the robot has boxed itself into a dead end.

``_escape_boxed_in`` runs a deliberate in-place rotation survey — spin a full
circle capturing the open distance in each direction, then commit to driving
toward the most open bearing found — used when the normal frontier planner has
nothing reachable and the robot needs to physically relocate to un-stick itself.
Its nested helpers read fresh frames, measure forward clearance, and issue the
turn/drive bursts.
"""
from __future__ import annotations

import math
import time


from ..lidar.auto_snapshot_stitch import _execute_turn_burst, _send_stop
from ..lidar.direct_snapshot_client import DirectLidarFeed, _scan_to_local_points
from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient
from .frontier import _select_frontier_choice
from .wander_types import _normalize_angle_deg

def _escape_boxed_in(
    *,
    robot: SourcceyClient,
    feed: DirectLidarFeed,
    args,
    after_frame_id: int,
    min_usable: int,
) -> int:
    """The founding scan is BOXED IN — a full revolution arrives but almost
    every return is within min_range of the lidar. Rotating in place cannot
    fix this (a 360deg scan is heading-invariant: the same obstacles stay at
    the same distances, just relabeled). The only escape is to TRANSLATE
    toward open space. Face the most-open direction, drive a short gentle
    burst toward it (gated on a LIVE clearance check, independent of the
    saturated stop box), and rescan — repeat until a healthy revolution
    arrives. Returns the frame_id of the last processed revolution. Raises
    only if the robot is genuinely surrounded (no direction with enough
    clearance) after the attempt budget — a physical situation the operator
    must fix. User directive 2026-07-18: 'if it finds itself boxed in, just
    turn around and keep scanning.'"""
    ESCAPE_CLEAR_M = 0.45  # an opening must be at least this deep to drive into
    FWD_BLOCKED_M = 0.35   # nearest return straight ahead below this = blocked
    ESCAPE_BURST_S = 0.22  # ~0.16m at the min effective speed
    turn_speed = float(args.turn_speed)
    turn_burst_s = float(args.turn_burst_s)
    deg_per_burst = max(4.0, math.degrees(turn_speed * turn_burst_s))
    last_id = int(after_frame_id)

    def _fresh_frame():
        """Block for the next LiDAR revolution newer than the last one seen,
        advancing the ``last_id`` cursor. Returns the frame, or None on timeout."""
        nonlocal last_id
        fid, fr = feed.wait_for_frame_after(
            after_frame_id=last_id,
            timeout_s=float(args.fresh_frame_timeout_s),
            min_frame_advances=1,
            armed_wall_ts=time.time(),
        )
        if fr is not None:
            last_id = int(fid)
        return fr

    def _usable(fr) -> int:
        """How many points in this frame survive the normal filtering (beyond the
        near-cutoff, within range/confidence) — the "is this scan rich enough to
        map" count."""
        return len(
            _scan_to_local_points(
                points=fr.points,
                forward_angle_deg=float(args.forward_angle_deg),
                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                invert_lateral_axis=bool(args.invert_lateral_axis),
                max_distance_m=float(args.max_distance_m),
                min_confidence=int(args.min_confidence),
                min_range_m=float(args.min_range_m),
            )
        )

    def _forward_clear_m(fr) -> float:
        """Distance to the nearest obstacle within a narrow (+-20deg) cone straight
        ahead — how much room there is to drive forward right now."""
        # Nearest return within a narrow (+-20deg) cone straight ahead.
        best = float(args.max_distance_m)
        for angle_deg, distance_m, confidence in fr.points:
            if int(confidence) < int(args.min_confidence):
                continue
            d = float(distance_m)
            if d <= 0.05:
                continue
            if abs(_normalize_angle_deg(float(angle_deg) - float(args.forward_angle_deg))) <= 20.0:
                best = min(best, d)
        return best

    def _best_open(fr):
        """The most open direction in this frame (reusing the frontier picker with
        the escape's minimum-clearance bar) — where to aim to get unstuck."""
        return _select_frontier_choice(
            fr,
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            max_distance_m=float(args.max_distance_m),
            min_range_m=float(args.min_range_m),
            min_confidence=int(args.min_confidence),
            frontier_min_distance_m=ESCAPE_CLEAR_M,
            frontier_bin_deg=float(args.frontier_bin_deg),
        )

    def _turn(delta_deg: float) -> None:
        """Rotate in place by roughly ``delta_deg`` (skipping tiny turns under
        8deg) by issuing the right number of fixed turn bursts in the correct
        direction. Open-loop — no map tracking, since the point is just to reface."""
        if abs(delta_deg) < 8.0:
            return
        sign = 1.0 if delta_deg >= 0.0 else -1.0
        for _ in range(min(10, max(1, int(round(abs(delta_deg) / deg_per_burst))))):
            _execute_turn_burst(
                robot=robot, direction_sign=sign,
                turn_speed=turn_speed, turn_burst_s=turn_burst_s, turn_settle_s=0.08,
            )

    def _drive_forward() -> None:
        """Nudge straight forward one short, slow burst (~0.16m) toward the
        committed opening, then stop and settle. Deliberately gentle because the
        saturated stop box can't be trusted while boxed in."""
        deadline = time.monotonic() + ESCAPE_BURST_S
        while time.monotonic() < deadline:
            robot.send_action(
                {
                    "x.vel": max(float(args.min_effective_move_speed), 0.75),
                    "y.vel": 0.0,
                    "theta.vel": 0.0,
                    "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
                    "untorque_left": True,
                    "untorque_right": True,
                }
            )
            time.sleep(0.05)
        _send_stop(robot)
        time.sleep(0.25)

    def _full_rotation_survey() -> float:
        """Spin a deliberate full circle, checking each direction, and return the
        best open clearance seen. If a genuinely healthy view appears partway
        round it bails early. This is the "look everywhere before giving up"
        survey the user asked for — a visible, complete sweep rather than trusting
        one scan."""
        # Physically complete a ~360deg rotation (per user: never conclude
        # boxed without a full look-around), returning the best open clearance
        # seen anywhere. The lidar already sees 360deg, so this is confirmation
        # + a deliberate, visible survey — not information the single scan
        # lacked.
        print("[wander] BOXED IN — no opening ahead; doing a FULL rotation to survey all directions")
        best = 0.0
        bursts_per_step = 2
        steps = max(6, int(round(360.0 / max(deg_per_burst * bursts_per_step, 24.0))))
        for _ in range(steps):
            for _ in range(bursts_per_step):
                _execute_turn_burst(
                    robot=robot, direction_sign=1.0,
                    turn_speed=turn_speed, turn_burst_s=turn_burst_s, turn_settle_s=0.05,
                )
            fr = _fresh_frame()
            if fr is None:
                continue
            if _usable(fr) >= int(min_usable):
                return float(args.max_distance_m)  # a healthy view appeared mid-survey
            c = _best_open(fr)
            if c is not None:
                best = max(best, float(c.mean_distance_m))
        print(f"[wander] full-rotation survey: best clearance found = {best:.2f}m")
        return best

    # COMMIT-AND-DRIVE: pick an opening, face it ONCE, then drive FORWARD
    # through it in repeated bursts (re-picking a new lateral bearing every
    # cycle made it oscillate between opposite openings and never translate —
    # field 2026-07-18: drove -84deg then +84deg then gave up). Re-survey only
    # when the committed direction becomes blocked ahead.
    for direction_attempt in range(6):
        fr = _fresh_frame()
        if fr is None:
            print(f"[wander] boxed-in escape: no fresh revolution (direction {direction_attempt + 1}/6)")
            continue
        if _usable(fr) >= int(min_usable):
            print(f"[wander] escaped the box (usable={_usable(fr)}); founding the map here")
            _send_stop(robot)
            return last_id
        choice = _best_open(fr)
        if choice is None or float(choice.mean_distance_m) < ESCAPE_CLEAR_M:
            best = _full_rotation_survey()
            if best >= float(args.max_distance_m):
                _send_stop(robot)
                return last_id  # healthy view appeared during the survey
            if best < ESCAPE_CLEAR_M:
                _send_stop(robot)
                raise RuntimeError(
                    "BOXED IN with no escape route: after a FULL rotation, every direction "
                    f"still has an obstacle within {ESCAPE_CLEAR_M:.2f}m of the lidar. The robot "
                    "is genuinely surrounded (draped material all around, or wedged in a tight "
                    "nook). Physically reposition it with clear space around the lidar — this is "
                    "NOT a host/feed problem."
                )
            continue  # an opening exists somewhere; re-pick and drive next loop
        # Face the opening, then commit to driving FORWARD through it.
        print(
            "[wander] BOXED IN — committing to the open direction "
            f"(bearing {float(choice.delta_deg):+.0f}deg, {float(choice.mean_distance_m):.2f}m clear) "
            "and driving through it"
        )
        _turn(float(choice.delta_deg))
        for _ in range(6):
            fr = _fresh_frame()
            if fr is None:
                break
            if _usable(fr) >= int(min_usable):
                print(f"[wander] escaped the box (usable={_usable(fr)}); founding the map here")
                _send_stop(robot)
                return last_id
            if _forward_clear_m(fr) < FWD_BLOCKED_M:
                break  # committed direction is now blocked — re-survey next loop
            _drive_forward()
    _send_stop(robot)
    raise RuntimeError(
        "Could not escape the boxed-in area after driving through 6 openings — the robot is "
        "likely surrounded or wedged. Physically reposition it with clear space around the lidar."
    )


