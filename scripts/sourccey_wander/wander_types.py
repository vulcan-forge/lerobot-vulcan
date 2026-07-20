"""Shared dataclasses and pure geometry helpers used across the wander subsystem.

Nothing here has side effects or talks to hardware — these are the small value
types (a chosen frontier ``FrontierChoice``, an ``ExploreTarget``, the
``DriveSequenceMeta``/``DriveBurstMeta`` result records) and the trigonometry
utilities (angle normalization, turn lever-arm offsets, composing two motion
hints into one, remote base-state polling, heading-bin indexing) that every
other module leans on. Kept near the bottom of the import graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ldlidar_direct_snapshot_stitch import MotionHint
from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient

@dataclass(slots=True)
class FrontierChoice:
    delta_deg: float
    abs_angle_deg: float
    mean_distance_m: float
    width_deg: float
    score: float
    source: str = "live"


@dataclass(slots=True)
class ExploreTarget:
    world_x_m: float
    world_y_m: float
    source: str
    seeded_capture_index: int
    coarse_turns_used: int = 0


@dataclass(slots=True)
class DriveSequenceMeta:
    elapsed_s: float
    iterations: int
    bursts_completed: int
    stopped_by_block: bool
    blocked_points: int
    commanded_forward_speed: float
    steer_theta_vel: float
    host_feedback_updates: int
    host_forward_distance_estimate_m: float
    host_x_vel_peak: float
    host_theta_vel_peak: float
    host_x_vel_last: float
    host_theta_vel_last: float


@dataclass(slots=True)
class DriveBurstMeta:
    elapsed_s: float
    iterations: int
    host_feedback_updates: int
    host_forward_distance_estimate_m: float
    host_x_vel_peak: float
    host_theta_vel_peak: float
    host_x_vel_last: float
    host_theta_vel_last: float
    stopped_by_hazard: bool = False
    hazard_reason: str = ""


def _normalize_angle_deg(angle_deg: float) -> float:
    """Wrap any angle into the (-180, 180] range.

    Add 180, take it modulo 360 (so it lands in [0, 360)), then subtract 180
    back off. The net effect: 190deg becomes -170deg, -350deg becomes 10deg, so
    two headings can be compared/subtracted without a 359-vs-1 degree cliff.
    Used everywhere a heading difference is taken.
    """
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _turn_lever_arm_local_delta(
    dtheta_deg: float,
    *,
    lidar_offset_forward_m: float,
) -> tuple[float, float]:
    """Expected LiDAR translation (in the pre-turn sensor frame) for an in-place
    robot turn of `dtheta_deg`, given the sensor is mounted `lidar_offset_forward_m`
    forward of the robot's rotation center. The sensor traces an arc around the
    center, so pure robot rotation still translates the sensor by
    2*r*sin(|dtheta|/2) — ~0.32 m for a 90deg turn at r=0.23 m."""
    r = float(lidar_offset_forward_m)
    dtheta_rad = math.radians(float(dtheta_deg))
    return r * (math.cos(dtheta_rad) - 1.0), r * math.sin(dtheta_rad)


def _compose_motion_hints(first: MotionHint, second: MotionHint) -> MotionHint:
    """SE(2)-compose two consecutive expected motions into one hint (used when a
    capture is discarded and its motion must carry into the next capture)."""
    theta_rad = math.radians(float(first.expected_dtheta_deg))
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    dx = float(first.expected_dx_local_m) + c * float(second.expected_dx_local_m) - s * float(second.expected_dy_local_m)
    dy = float(first.expected_dy_local_m) + s * float(second.expected_dx_local_m) + c * float(second.expected_dy_local_m)
    # Two chained turns are still a pure in-place turn (same rotation center);
    # any other mix loses that guarantee and must use the generic solver.
    composed_kind = str(second.kind) if str(first.kind) == str(second.kind) else "mixed"
    return MotionHint(
        kind=composed_kind,
        expected_dx_local_m=float(dx),
        expected_dy_local_m=float(dy),
        expected_dtheta_deg=float(first.expected_dtheta_deg) + float(second.expected_dtheta_deg),
        search_xy_m=max(float(first.search_xy_m), float(second.search_xy_m)) + 0.10,
        search_theta_window_deg=max(float(first.search_theta_window_deg), float(second.search_theta_window_deg)) + 10.0,
        label=f"{first.label}+{second.label}",
    )


def _apply_min_effective_magnitude(value: float, *, minimum_abs: float) -> float:
    """Bump a command up to a minimum magnitude WITHOUT changing its sign or
    zeroing it.

    The base motors have a dead band: a velocity command below some threshold
    produces no motion at all. This takes a desired value and, if it is nonzero
    but smaller than ``minimum_abs``, snaps it out to +/-minimum_abs (keeping the
    direction). An exact zero stays zero (a real stop), and a value already big
    enough passes through untouched.
    """
    minimum_abs = max(float(minimum_abs), 0.0)
    value = float(value)
    if abs(value) <= 1e-9 or minimum_abs <= 1e-9:
        return value
    if abs(value) < minimum_abs:
        return float(minimum_abs if value > 0.0 else -minimum_abs)
    return value


def _poll_remote_base_state(robot: SourcceyClient) -> tuple[dict[str, float], bool]:
    """Read the base's reported velocities (x, y, theta) from the robot host.

    Asks the client for its latest data frame. On success it returns the three
    velocity fields plus ``True`` (fresh). If the fetch throws (comms hiccup) it
    falls back to the last cached remote state and returns ``False`` (stale), so
    a caller can tell "the base really reports ~0 velocity" apart from "we just
    couldn't reach the base." Every field is coerced to float and defaulted to 0.
    """
    try:
        _frames, remote_state, is_fresh = robot._get_data()
    except Exception:
        remote_state = getattr(robot, "last_remote_state", {}) or {}
        return {
            "x.vel": float(remote_state.get("x.vel", 0.0) or 0.0),
            "y.vel": float(remote_state.get("y.vel", 0.0) or 0.0),
            "theta.vel": float(remote_state.get("theta.vel", 0.0) or 0.0),
        }, False

    remote_state = remote_state or {}
    return {
        "x.vel": float(remote_state.get("x.vel", 0.0) or 0.0),
        "y.vel": float(remote_state.get("y.vel", 0.0) or 0.0),
        "theta.vel": float(remote_state.get("theta.vel", 0.0) or 0.0),
    }, bool(is_fresh)


def _heading_bin_index(theta_deg: float, *, bin_count: int) -> int:
    """Which angular "slice" of a full circle a heading falls into.

    Splits 360deg into ``bin_count`` equal wedges and returns the wedge index
    (0-based) for ``theta_deg``. The heading is wrapped into [0, 360) first, so
    negative or multi-turn angles still land in a valid bin. Used by the
    bootstrap rotation to track which directions have already been scanned.
    """
    if int(bin_count) <= 0:
        return 0
    wrapped = float(theta_deg) % 360.0
    bin_width_deg = 360.0 / float(bin_count)
    return int(math.floor(wrapped / bin_width_deg)) % int(bin_count)


