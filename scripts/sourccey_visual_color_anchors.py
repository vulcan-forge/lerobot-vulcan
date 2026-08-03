"""Small, explainable color landmarks for LiDAR localization tie-breaking.

The camera is not used as a depth sensor here.  A calibrated upright fused
panorama is reduced to coarse HSV sector signatures and attached to accepted
LiDAR map poses.  Signatures only rank already-plausible LiDAR hypotheses;
they never create or override a pose by themselves.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


SECTOR_COUNT = 8
SIGNATURE_VERSION = 1


def color_signature(
    image_bgr: np.ndarray,
    *,
    sectors: int = SECTOR_COUNT,
    vertical_range: tuple[float, float] = (0.18, 0.82),
) -> list[float]:
    """Return a coarse, lighting-tolerant HSV signature for an image.

    ``vertical_range`` lets callers choose the part of a camera that carries
    useful landmarks.  The fused eye panorama keeps its middle band, while
    the downward-facing bottom camera uses its upper half: that region looks
    farther ahead and is substantially less dominated by uniform carpet.
    """
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] < 3 or image.shape[0] < 4:
        return []
    # Ignore sky/ceiling and the lowest floor strip; those regions are poor
    # directional landmarks and are especially sensitive to exposure changes.
    h, w = image.shape[:2]
    top, bottom = vertical_range
    top = float(np.clip(top, 0.0, 1.0))
    bottom = float(np.clip(bottom, top + 1.0 / h, 1.0))
    crop = image[int(top * h) : max(int((top + 1.0 / h) * h), int(bottom * h)), :, :3]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    result: list[float] = []
    for sector in range(max(1, int(sectors))):
        lo = int(round(sector * w / sectors))
        hi = int(round((sector + 1) * w / sectors))
        values = hsv[:, lo:hi].reshape((-1, 3))
        if not len(values):
            result.extend((0.0, 0.0, 0.0, 0.0))
            continue
        # Median HSV is robust to small moving objects and compression noise.
        median = np.median(values, axis=0)
        hue = float(median[0]) / 180.0
        sat = float(median[1]) / 255.0
        val = float(median[2]) / 255.0
        # A low-saturation sector has no useful hue; store that confidence so
        # matching can naturally down-weight hue in plain white/gray rooms.
        result.extend((hue, sat, val, min(1.0, sat * 2.0)))
    return result


def bottom_upper_color_signature(image_bgr: np.ndarray, *, sectors: int = SECTOR_COUNT) -> list[float]:
    """Return the upper-half signature of the downward-facing bottom camera.

    The upper image half is the long-range portion of this camera's view.  It
    is intentionally kept separate from the fused-eye signature so the map
    can compare two independent appearance cues without pretending either
    camera provides depth.
    """

    return color_signature(image_bgr, sectors=sectors, vertical_range=(0.0, 0.55))


def _sector_rows(signature: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(signature, dtype=np.float32).reshape((-1, 4))
    return values if len(values) else np.empty((0, 4), dtype=np.float32)


def color_similarity(
    current: list[float] | np.ndarray,
    reference: list[float] | np.ndarray,
    *,
    heading_delta_deg: float = 0.0,
    hfov_deg: float = 104.5,
) -> float:
    """Compare signatures in [0, 1], allowing a small heading offset."""
    a = _sector_rows(current)
    b = _sector_rows(reference)
    if len(a) == 0 or a.shape != b.shape:
        return 0.0
    sectors = len(a)
    shift = int(round(float(heading_delta_deg) / max(1.0, float(hfov_deg) / sectors)))
    if abs(shift) > max(2, sectors // 2):
        return 0.0
    b = np.roll(b, shift, axis=0)
    hue_delta = np.abs(a[:, 0] - b[:, 0])
    hue_delta = np.minimum(hue_delta, 1.0 - hue_delta)
    sat_delta = np.abs(a[:, 1] - b[:, 1])
    val_delta = np.abs(a[:, 2] - b[:, 2])
    hue_weight = np.minimum(a[:, 3], b[:, 3])
    error = hue_delta * hue_weight + sat_delta * 0.45 + val_delta * 0.25
    return float(np.clip(1.0 - np.mean(error) * 2.0, 0.0, 1.0))


def best_landmark_score(
    current: list[float] | np.ndarray,
    landmarks: list[dict[str, object]],
    *,
    x: float,
    y: float,
    theta_deg: float,
    max_distance_m: float = 0.75,
    signature_key: str = "signature",
) -> float:
    """Find the strongest nearby stored color landmark for a LiDAR pose."""
    best = 0.0
    for landmark in landmarks:
        try:
            pose = np.asarray(landmark["pose"], dtype=np.float64).reshape(3)
            distance = math.hypot(float(pose[0]) - x, float(pose[1]) - y)
            if distance > float(max_distance_m):
                continue
            score = color_similarity(
                current,
                landmark[signature_key],
                heading_delta_deg=float(theta_deg) - float(pose[2]),
            )
            # Nearby keyframes are more trustworthy than a far appearance
            # coincidence; retain a gentle spatial weighting only.
            score *= max(0.0, 1.0 - distance / max(1e-6, float(max_distance_m)))
            best = max(best, float(score))
        except (KeyError, TypeError, ValueError):
            continue
    return float(best)
