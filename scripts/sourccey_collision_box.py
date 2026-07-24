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


def effective_ranges(profile: dict) -> np.ndarray:
    """Return the envelope ranges after applying UI width/length adjustments."""
    raw = np.array([
        np.nan if value is None else float(value) for value in profile.get("ranges_m", [])
    ])
    if not len(raw):
        return raw
    bin_size = float(profile["bin_size_deg"])
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


def collision_box_violation(
    points_forward_xy: np.ndarray,
    profile: dict | None,
) -> tuple[np.ndarray, str, float, float, float] | None:
    """Return violating point mask and details when a scan enters the envelope."""
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
    return violating, sector, angle, float(ranges[pick]), float(limits[pick])
