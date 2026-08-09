"""Learn, save, load, and evaluate Sourccey's polar LiDAR collision envelope."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

DEFAULT_COLLISION_BOX_PATH = Path(__file__).with_name("sourccey_collision_box.json")


def calibrate_collision_box(
    scans_forward_xy: list[np.ndarray],
    *,
    bin_size_deg: float = 4.0,
    min_scans_per_bin: int = 8,
    max_boundary_m: float = 1.5,
    noise_tolerance_m: float = 0.02,
) -> dict:
    """Build a polar collision envelope from stationary, boxed-in LiDAR scans."""
    bin_size = float(bin_size_deg)
    if not 1.0 <= bin_size <= 30.0:
        raise ValueError("bin_size_deg must be between 1 and 30 degrees")
    n_bins = int(math.ceil(360.0 / bin_size))
    observations: list[list[float]] = [[] for _ in range(n_bins)]

    for scan in scans_forward_xy:
        points = np.asarray(scan, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (2,) or not len(points):
            continue
        ranges = np.hypot(points[:, 0], points[:, 1])
        angles = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        valid = np.isfinite(ranges) & (ranges >= 0.03) & (ranges <= float(max_boundary_m))
        if not np.any(valid):
            continue
        indices = np.floor((angles[valid] + 180.0) / bin_size).astype(np.int64) % n_bins
        valid_ranges = ranges[valid]
        for idx in np.unique(indices):
            observations[int(idx)].append(float(np.min(valid_ranges[indices == idx])))

    learned: list[float | None] = []
    counts: list[int] = []
    for values in observations:
        counts.append(len(values))
        learned.append(
            round(float(np.median(values)), 4)
            if len(values) >= int(min_scans_per_bin)
            else None
        )
    valid_bins = sum(value is not None for value in learned)
    if valid_bins < max(12, int(0.35 * n_bins)):
        raise ValueError(
            f"collision calibration covered only {valid_bins}/{n_bins} angular bins; "
            "ensure the surrounding box is visible and collect more scans"
        )
    return {
        "version": 1,
        "frame": "physical_forward_xy",
        "bin_size_deg": bin_size,
        "ranges_m": learned,
        "samples_per_bin": counts,
        "noise_tolerance_m": float(noise_tolerance_m),
        # Preserve the learned/displayed line as the actual stop boundary.
        # The equal margin offsets measurement tolerance instead of silently
        # allowing the robot to penetrate inside its calibrated envelope.
        "safety_margin_m": float(noise_tolerance_m),
        "min_violation_bins": 2,
        "side_min_violation_bins": 1,
        "min_violation_points": 3,
        "side_min_violation_points": 2,
        "confirmation_frames": 2,
        "calibration_scans": len(scans_forward_xy),
        "created_unix_s": time.time(),
    }


def save_collision_box(profile: dict, path: str | Path = DEFAULT_COLLISION_BOX_PATH) -> Path:
    profile_path = Path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return profile_path


def load_collision_box(path: str | Path = DEFAULT_COLLISION_BOX_PATH) -> dict | None:
    profile_path = Path(path)
    if not profile_path.exists():
        return None
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or data.get("frame") != "physical_forward_xy":
        raise ValueError(f"unsupported collision calibration in {profile_path}")
    return data


def _completed_rounded_box_ranges(
    *,
    count: int,
    bin_size_deg: float,
    width_m: float,
    length_m: float,
    corner_radius_m: float,
    front_m: float | None = None,
    rear_m: float | None = None,
    front_width_m: float | None = None,
) -> np.ndarray:
    """Radial boundary of a rounded box with optional 45-degree front shoulders."""
    if front_m is None or rear_m is None:
        front = rear = 0.5 * max(0.001, float(length_m))
    else:
        front = max(0.001, float(front_m))
        rear = max(0.001, float(rear_m))
    half_x = 0.5 * (front + rear)
    centre_x = 0.5 * (front - rear)
    half_y = 0.5 * max(0.001, float(width_m))
    front_half_y = min(
        half_y,
        0.5 * max(0.001, float(
            width_m if front_width_m is None else front_width_m
        )),
    )
    # The front face transitions to the full side width at 45 degrees. When
    # both widths match this degenerates exactly to the old rounded rectangle.
    shoulder_depth = max(0.0, half_y - front_half_y)
    shoulder_start_x = front - shoulder_depth
    radius = min(
        max(0.0, float(corner_radius_m)),
        half_x,
        half_y,
    )
    angles = np.radians(
        -180.0 + (np.arange(int(count)) + 0.5) * float(bin_size_deg)
    )
    directions = np.column_stack([np.cos(angles), np.sin(angles)])

    def inside(distance: np.ndarray) -> np.ndarray:
        points = directions * distance[:, None]
        qx = np.abs(points[:, 0] - centre_x) - (half_x - radius)
        qy = np.abs(points[:, 1]) - (half_y - radius)
        outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
        signed = outside + np.minimum(np.maximum(qx, qy), 0.0) - radius
        rounded_body = signed <= 0.0
        abs_y = np.abs(points[:, 1])
        front_shoulder = (
            (points[:, 0] <= shoulder_start_x)
            | (abs_y <= front_half_y + np.maximum(0.0, front - points[:, 0]))
        )
        return rounded_body & front_shoulder

    low = np.zeros(int(count), dtype=np.float64)
    high = np.full(
        int(count),
        max(math.hypot(front, half_y), math.hypot(rear, half_y))
        + max(0.01, radius),
        dtype=np.float64,
    )
    for _ in range(40):
        middle = 0.5 * (low + high)
        within = inside(middle)
        low[within] = middle[within]
        high[~within] = middle[~within]
    return low


def effective_ranges(profile: dict) -> np.ndarray:
    """Return the envelope ranges after applying UI width/length adjustments."""
    raw = np.array([
        np.nan if value is None else float(value) for value in profile.get("ranges_m", [])
    ])
    if not len(raw):
        return raw
    bin_size = float(profile["bin_size_deg"])
    if bool(profile.get("complete_box", False)):
        base_width, base_length = collision_box_dimensions({
            **profile,
            "complete_box": False,
        })
        width = float(
            profile.get("completed_width_m", profile.get("width_m", base_width))
        )
        length = float(
            profile.get(
                "completed_length_m",
                profile.get("length_m", max(base_length, width)),
            )
        )
        front = float(profile.get("completed_front_m", 0.5 * length))
        rear = float(profile.get("completed_rear_m", 0.5 * length))
        default_radius = 0.20 * min(width, length)
        return _completed_rounded_box_ranges(
            count=len(raw),
            bin_size_deg=bin_size,
            width_m=width,
            length_m=length,
            corner_radius_m=float(profile.get("corner_radius_m", default_radius)),
            front_m=front,
            rear_m=rear,
            front_width_m=float(profile.get("completed_front_width_m", width)),
        )
    angles = np.radians(-180.0 + (np.arange(len(raw)) + 0.5) * bin_size)
    x = raw * np.cos(angles)
    y = raw * np.sin(angles)
    valid = np.isfinite(raw)
    base_length = float(np.nanmax(x[valid]) - np.nanmin(x[valid]))
    base_width = float(np.nanmax(y[valid]) - np.nanmin(y[valid]))
    target_length = float(profile.get("length_m", base_length))
    target_width = float(profile.get("width_m", base_width))
    if base_length > 1e-6:
        x *= target_length / base_length
    if base_width > 1e-6:
        y *= target_width / base_width
    return np.hypot(x, y)


def collision_box_dimensions(profile: dict) -> tuple[float, float]:
    """Return effective (width, length) in metres."""
    raw_profile = dict(profile)
    raw_profile.pop("width_m", None)
    raw_profile.pop("length_m", None)
    ranges = effective_ranges(raw_profile)
    angles = np.radians(
        -180.0 + (np.arange(len(ranges)) + 0.5) * float(profile["bin_size_deg"])
    )
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    valid = np.isfinite(ranges)
    base_width = float(np.nanmax(y[valid]) - np.nanmin(y[valid]))
    base_length = float(np.nanmax(x[valid]) - np.nanmin(x[valid]))
    return (
        float(profile.get("width_m", base_width)),
        float(profile.get("length_m", base_length)),
    )


def physical_body_self_return_mask(
    points_forward_xy: np.ndarray,
    *,
    lidar_offset_forward_m: float,
    physical_body_radius_m: float,
    self_mask_inset_m: float,
) -> np.ndarray:
    """Identify LiDAR returns securely inside the robot's physical footprint.

    Sourccey's mecanum chassis has a square footprint with rounded corners.  A
    circular mask leaves the front/side shoulder regions exposed and mistakes
    fixed chassis returns there for nearby walls.  This inset rounded-square
    footprint is the standard geometric self-filter: points near or outside
    the measured chassis boundary remain collision-sensitive.
    """
    points = np.asarray(points_forward_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,):
        return np.zeros(len(points) if points.ndim else 0, dtype=bool)
    half_extent = max(
        0.0,
        float(physical_body_radius_m) - max(0.0, float(self_mask_inset_m)),
    )
    if half_extent <= 0.0 or not len(points):
        return np.zeros(len(points), dtype=bool)

    # Keep the self-filter conservative at the physical corners.  The 6 cm
    # rounding matches a square mobile base without treating the full planning
    # envelope as robot material.
    corner_radius = min(0.06, 0.25 * half_extent)
    centred_x = points[:, 0] + float(lidar_offset_forward_m)
    centred_y = points[:, 1]
    qx = np.abs(centred_x) - (half_extent - corner_radius)
    qy = np.abs(centred_y) - (half_extent - corner_radius)
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    signed_distance = (
        outside + np.minimum(np.maximum(qx, qy), 0.0) - corner_radius
    )
    return signed_distance <= 0.0


def collision_box_violation(
    points_forward_xy: np.ndarray,
    profile: dict | None,
    *,
    lidar_offset_forward_m: float = 0.0,
    physical_body_radius_m: float = 0.0,
    self_mask_inset_m: float = 0.0,
) -> tuple[np.ndarray, str, float, float, float] | None:
    """Return violating point mask and details when a scan enters the envelope.

    The LiDAR sits forward of the robot centre and can see chassis structure
    beneath/behind its scan plane. Returns safely inside the measured physical
    body cannot be environmental obstacles, so exclude only an inset core of
    that body. The larger planning/collision margin is intentionally *not*
    excluded: a wall beside the shoulder must still stop the robot.
    """
    if not profile:
        return None
    points = np.asarray(points_forward_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or not len(points):
        return None
    learned = profile.get("ranges_m")
    bin_size = float(profile.get("bin_size_deg") or 0.0)
    if not isinstance(learned, list) or not learned or bin_size <= 0.0:
        return None
    thresholds = effective_ranges(profile)
    ranges = np.hypot(points[:, 0], points[:, 1])
    angles = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    indices = np.floor((angles + 180.0) / bin_size).astype(np.int64) % len(thresholds)
    noise_tolerance = float(profile.get("noise_tolerance_m", 0.02))
    # Legacy profiles have no safety_margin_m. Default it to the tolerance so
    # their displayed/calibrated envelope is the real trip line, rather than a
    # hidden boundary 2cm inside it. An explicit larger margin stops earlier.
    safety_margin = float(profile.get("safety_margin_m", noise_tolerance))
    limits = thresholds[indices] + safety_margin - noise_tolerance
    violating = np.isfinite(limits) & np.isfinite(ranges) & (ranges >= 0.03) & (ranges < limits)
    self_returns = physical_body_self_return_mask(
        points,
        lidar_offset_forward_m=lidar_offset_forward_m,
        physical_body_radius_m=physical_body_radius_m,
        self_mask_inset_m=self_mask_inset_m,
    )
    violating &= ~self_returns
    if not np.any(violating):
        return None

    side = violating & (np.abs(angles) >= 45.0) & (np.abs(angles) <= 135.0)
    side_bins = np.unique(indices[side]).size
    all_bins = np.unique(indices[violating]).size
    side_points = int(np.count_nonzero(side))
    all_points = int(np.count_nonzero(violating))
    if (
        (
            side_bins < int(profile.get("side_min_violation_bins", 1))
            or side_points < int(profile.get("side_min_violation_points", 2))
        )
        and (
            all_bins < int(profile.get("min_violation_bins", 2))
            or all_points < int(profile.get("min_violation_points", 3))
        )
    ):
        return None

    accepted = side if side_bins >= int(profile.get("side_min_violation_bins", 1)) else violating
    candidates = np.flatnonzero(accepted)
    pick = int(candidates[np.argmax(limits[candidates] - ranges[candidates])])
    angle = float(angles[pick])
    if 45.0 <= angle <= 135.0:
        sector = "left side"
    elif -135.0 <= angle <= -45.0:
        sector = "right side"
    elif abs(angle) > 135.0:
        sector = "rear"
    else:
        sector = "front"
    return accepted, sector, angle, float(ranges[pick]), float(limits[pick])


def collision_box_rotation_violation(
    points_forward_xy: np.ndarray,
    profile: dict | None,
    rotation_deg: float,
    *,
    lidar_offset_forward_m: float = 0.0,
    physical_body_radius_m: float = 0.0,
    self_mask_inset_m: float = 0.0,
    sweep_step_deg: float = 3.0,
) -> tuple[tuple[np.ndarray, str, float, float, float], float] | None:
    """Predict whether a requested in-place rotation sweeps into the envelope.

    Points are stationary in the world. For a robot turn ``delta``, rotate each
    point by ``-delta`` about the robot centre, then express it from the
    forward-offset LiDAR again. If the robot is already inside its safety
    margin, permit only the direction whose first step strictly reduces the
    deepest penetration. This is the rotate-in-place equivalent of a local
    planner's footprint trajectory check.
    """
    points = np.asarray(points_forward_xy, dtype=np.float64)
    requested = float(rotation_deg)
    if (
        not profile
        or points.ndim != 2
        or points.shape[1:] != (2,)
        or not len(points)
        or abs(requested) < 0.25
    ):
        return None
    original_count = len(points)
    retained_indices = np.arange(original_count, dtype=np.int64)
    if float(physical_body_radius_m) > 0.0:
        # Self returns rotate with the robot, not with the stationary world.
        # Remove the inset physical-body core before projecting world points
        # through future robot headings.
        self_returns = physical_body_self_return_mask(
            points,
            lidar_offset_forward_m=lidar_offset_forward_m,
            physical_body_radius_m=physical_body_radius_m,
            self_mask_inset_m=self_mask_inset_m,
        )
        keep = ~self_returns
        retained_indices = retained_indices[keep]
        points = points[keep]
        if not len(points):
            return None

    # Self returns have now been removed exactly once in the current robot
    # frame. Do not apply the self mask again to projected future frames: an
    # external point swept into the physical core is a collision, not a new
    # self return. Violation masks are remapped to the caller's original point
    # array before leaving this function.
    geometry = {
        "lidar_offset_forward_m": float(lidar_offset_forward_m),
        "physical_body_radius_m": 0.0,
        "self_mask_inset_m": 0.0,
    }

    def remap_hit(
        hit: tuple[np.ndarray, str, float, float, float],
    ) -> tuple[np.ndarray, str, float, float, float]:
        full_mask = np.zeros(original_count, dtype=bool)
        full_mask[retained_indices] = np.asarray(hit[0], dtype=bool)
        return full_mask, hit[1], hit[2], hit[3], hit[4]

    def points_after_turn(delta_deg: float) -> np.ndarray:
        theta = math.radians(-float(delta_deg))
        c, s = math.cos(theta), math.sin(theta)
        centred = points.copy()
        centred[:, 0] += float(lidar_offset_forward_m)
        turned = np.column_stack([
            centred[:, 0] * c - centred[:, 1] * s,
            centred[:, 0] * s + centred[:, 1] * c,
        ])
        turned[:, 0] -= float(lidar_offset_forward_m)
        return turned

    current = collision_box_violation(points, profile, **geometry)
    step = max(0.5, min(abs(requested), abs(float(sweep_step_deg))))
    signed_step = math.copysign(step, requested)
    if current is not None:
        after_step = collision_box_violation(
            points_after_turn(signed_step),
            profile,
            **geometry,
        )
        current_depth = float(current[4] - current[3])
        next_depth = (
            float(after_step[4] - after_step[3])
            if after_step is not None
            else -math.inf
        )
        # Allow only an escape rotation that measurably opens clearance. The
        # controller repeats this test on every fresh pose/scan while turning.
        if next_depth < current_depth - 0.001:
            return None
        return remap_hit(current), 0.0

    sample_count = max(1, int(math.ceil(abs(requested) / step)))
    for delta in np.linspace(signed_step, requested, sample_count):
        hit = collision_box_violation(
            points_after_turn(float(delta)),
            profile,
            **geometry,
        )
        if hit is not None:
            return remap_hit(hit), float(delta)
    return None
