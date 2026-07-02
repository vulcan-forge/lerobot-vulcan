from __future__ import annotations

import argparse
import json
import math
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk


def _normalize_angle_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _normalize_angle_360_deg(angle_deg: float) -> float:
    return float(angle_deg) % 360.0


@dataclass
class CalibrationConfig:
    capture_scans: int
    angle_bin_deg: float
    self_hit_max_distance_m: float
    min_confidence: int
    min_hit_ratio: float
    mask_grow_deg: float
    sector_margin_deg: float
    max_sector_deg: float
    default_min_distance_m: float
    min_distance_margin_m: float
    tripwire_distance_m: float
    max_tripwire_half_width_m: float
    nominal_forward_angle_deg: float | None


@dataclass
class CalibrationResult:
    schema: str
    captured_scans: int
    angle_bin_deg: float
    self_hit_max_distance_m: float
    min_hit_ratio: float
    recommended_forward_angle_deg: float
    recommended_half_angle_deg: float
    recommended_min_distance_m: float
    recommended_tripwire_half_width_m: float
    recommended_sector_start_deg: float
    recommended_sector_end_deg: float
    self_hit_intervals_deg: list[list[float]]
    self_hit_bins_deg: list[float]
    self_hit_median_distance_m: list[float | None]
    self_hit_ratio: list[float]


class _TkViewer:
    def __init__(self, title: str) -> None:
        self.root = tk.Tk()
        self.root.title(title)
        self.label = tk.Label(self.root)
        self.label.pack()
        self._photo: ImageTk.PhotoImage | None = None
        self._closed = False
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self) -> None:
        self._closed = True
        self.root.destroy()

    @property
    def closed(self) -> bool:
        return self._closed

    def show_bgr(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(image=image)
        self.label.configure(image=self._photo)
        self.root.update_idletasks()
        self.root.update()


def _n_bins(cfg: CalibrationConfig) -> int:
    return max(int(round(360.0 / max(cfg.angle_bin_deg, 0.25))), 4)


def _angle_to_idx(angle_deg: float, cfg: CalibrationConfig) -> int:
    return int(_normalize_angle_360_deg(angle_deg) / cfg.angle_bin_deg) % _n_bins(cfg)


def _idx_center_angle_deg(idx: float, cfg: CalibrationConfig) -> float:
    return _normalize_angle_360_deg((float(idx) + 0.5) * cfg.angle_bin_deg)


def _expand_mask(mask: np.ndarray, grow_bins: int) -> np.ndarray:
    if grow_bins <= 0 or mask.size == 0 or not bool(np.any(mask)):
        return mask.copy()
    expanded = mask.copy()
    true_indices = np.flatnonzero(mask)
    for idx in true_indices:
        for offset in range(-grow_bins, grow_bins + 1):
            expanded[(idx + offset) % len(mask)] = True
    return expanded


def _mask_intervals(mask: np.ndarray, cfg: CalibrationConfig) -> list[list[float]]:
    if mask.size == 0 or not bool(np.any(mask)):
        return []
    indices = np.flatnonzero(mask)
    intervals: list[list[int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for idx in map(int, indices[1:]):
        if idx == prev + 1:
            prev = idx
            continue
        intervals.append([start, prev])
        start = idx
        prev = idx
    intervals.append([start, prev])
    if len(intervals) > 1 and intervals[0][0] == 0 and intervals[-1][1] == len(mask) - 1:
        intervals[0][0] = intervals[-1][0]
        intervals.pop()
    result: list[list[float]] = []
    for start_idx, end_idx in intervals:
        start_deg = _idx_center_angle_deg(start_idx - 0.5, cfg)
        end_deg = _idx_center_angle_deg(end_idx + 0.5, cfg)
        result.append([round(start_deg, 2), round(end_deg, 2)])
    return result


def _mask_intervals_idx(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0 or not bool(np.any(mask)):
        return []
    indices = np.flatnonzero(mask)
    intervals: list[list[int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for idx in map(int, indices[1:]):
        if idx == prev + 1:
            prev = idx
            continue
        intervals.append([start, prev])
        start = idx
        prev = idx
    intervals.append([start, prev])
    if len(intervals) > 1 and intervals[0][0] == 0 and intervals[-1][1] == len(mask) - 1:
        intervals[0][0] = intervals[-1][0]
        intervals.pop()
    return [(int(start_idx), int(end_idx)) for start_idx, end_idx in intervals]


def _largest_clear_interval(mask: np.ndarray) -> tuple[int, int]:
    n = len(mask)
    if n == 0:
        return 0, 0
    if not bool(np.any(mask)):
        return 0, n
    best_start = 0
    best_len = 0
    current_start = None
    current_len = 0
    doubled = np.concatenate((mask, mask))
    for idx, is_blocked in enumerate(doubled):
        if not bool(is_blocked):
            if current_start is None:
                current_start = idx
                current_len = 1
            else:
                current_len += 1
            if current_len > n:
                current_start += 1
                current_len = n
            if current_len > best_len:
                best_start = int(current_start)
                best_len = int(current_len)
        else:
            current_start = None
            current_len = 0
    return best_start % n, min(best_len, n)


def _interval_center_deg(start_idx: int, end_idx: int, cfg: CalibrationConfig) -> float:
    length = (int(end_idx) - int(start_idx) + 1) % _n_bins(cfg)
    if length <= 0:
        length = int(end_idx) - int(start_idx) + 1
    midpoint_idx = float(start_idx) + (float(length) / 2.0)
    return _idx_center_angle_deg(midpoint_idx, cfg)


def _interval_span_deg(start_idx: int, end_idx: int, cfg: CalibrationConfig) -> float:
    if end_idx >= start_idx:
        bins = end_idx - start_idx + 1
    else:
        bins = (_n_bins(cfg) - start_idx) + end_idx + 1
    return bins * float(cfg.angle_bin_deg)


def _select_forward_interval(mask: np.ndarray, cfg: CalibrationConfig) -> tuple[int, int] | None:
    intervals = _mask_intervals_idx(mask)
    if not intervals:
        return None
    nominal = cfg.nominal_forward_angle_deg
    best_interval = intervals[0]
    best_score = -math.inf
    for start_idx, end_idx in intervals:
        span_deg = _interval_span_deg(start_idx, end_idx, cfg)
        score = span_deg
        if nominal is not None:
            center_deg = _interval_center_deg(start_idx, end_idx, cfg)
            score -= abs(_normalize_angle_deg(center_deg - float(nominal))) * 2.0
        if score > best_score:
            best_score = score
            best_interval = (start_idx, end_idx)
    return best_interval


def _sector_mask(center_deg: float, half_angle_deg: float, cfg: CalibrationConfig) -> np.ndarray:
    angles = np.asarray([_idx_center_angle_deg(i, cfg) for i in range(_n_bins(cfg))], dtype=np.float32)
    delta = np.asarray([_normalize_angle_deg(float(a) - float(center_deg)) for a in angles], dtype=np.float32)
    return np.abs(delta) <= float(half_angle_deg)


def _draw_sector_overlay(
    canvas: np.ndarray,
    *,
    size: int,
    max_distance_m: float,
    forward_angle_deg: float,
    half_angle_deg: float,
    min_distance_m: float,
    color: tuple[int, int, int],
) -> None:
    center = size // 2
    radius_px = center - 24

    def to_xy(angle_deg: float, distance_m: float) -> tuple[int, int]:
        norm = min(max(distance_m, 0.0), max_distance_m) / max(max_distance_m, 1e-6)
        px_radius = int(norm * radius_px)
        theta = math.radians(angle_deg - 90.0)
        x = int(center + px_radius * math.cos(theta))
        y = int(center + px_radius * math.sin(theta))
        return x, y

    inner_left = to_xy(forward_angle_deg - half_angle_deg, min_distance_m)
    inner_right = to_xy(forward_angle_deg + half_angle_deg, min_distance_m)
    outer_left = to_xy(forward_angle_deg - half_angle_deg, max_distance_m)
    outer_right = to_xy(forward_angle_deg + half_angle_deg, max_distance_m)
    cv2.line(canvas, inner_left, outer_left, color, 2, cv2.LINE_AA)
    cv2.line(canvas, inner_right, outer_right, color, 2, cv2.LINE_AA)
    cv2.ellipse(
        canvas,
        (center, center),
        (radius_px, radius_px),
        0.0,
        forward_angle_deg - half_angle_deg - 90.0,
        forward_angle_deg + half_angle_deg - 90.0,
        color,
        2,
        cv2.LINE_AA,
    )
    if min_distance_m > 0.0:
        inner_px = int(radius_px * min(min_distance_m, max_distance_m) / max(max_distance_m, 1e-6))
        cv2.ellipse(
            canvas,
            (center, center),
            (inner_px, inner_px),
            0.0,
            forward_angle_deg - half_angle_deg - 90.0,
            forward_angle_deg + half_angle_deg - 90.0,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_self_hit_arcs(
    canvas: np.ndarray,
    *,
    size: int,
    max_distance_m: float,
    intervals_deg: list[list[float]],
    self_hit_max_distance_m: float,
) -> None:
    if not intervals_deg:
        return
    center = size // 2
    radius_px = int((center - 24) * min(self_hit_max_distance_m, max_distance_m) / max(max_distance_m, 1e-6))
    radius_px = max(radius_px, 12)
    for start_deg, end_deg in intervals_deg:
        start = float(start_deg) - 90.0
        end = float(end_deg) - 90.0
        if end < start:
            end += 360.0
        cv2.ellipse(
            canvas,
            (center, center),
            (radius_px, radius_px),
            0.0,
            start,
            end,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )


def _draw_points(
    points: list[list[float]],
    *,
    size: int,
    max_distance_m: float,
    rpm: float,
    progress_scans: int,
    total_scans: int,
    result: CalibrationResult | None,
    cfg: CalibrationConfig,
) -> np.ndarray:
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    radius_px = center - 24

    for ring_fraction in (0.25, 0.5, 0.75, 1.0):
        cv2.circle(canvas, (center, center), int(radius_px * ring_fraction), (40, 40, 40), 1)
    cv2.line(canvas, (center, 16), (center, size - 16), (30, 30, 30), 1)
    cv2.line(canvas, (16, center), (size - 16, center), (30, 30, 30), 1)
    cv2.circle(canvas, (center, center), 4, (0, 180, 255), -1)

    self_hit_angle_set = set(result.self_hit_bins_deg) if result is not None else set()
    for angle_deg, distance_m, confidence in points:
        clipped_distance = min(max(float(distance_m), 0.0), max_distance_m)
        norm = clipped_distance / max(max_distance_m, 1e-6)
        px_radius = int(norm * radius_px)
        theta = math.radians(float(angle_deg) - 90.0)
        x = int(center + px_radius * math.cos(theta))
        y = int(center + px_radius * math.sin(theta))
        conf = max(0, min(int(confidence), 255))
        color = (0, conf, 255 - min(conf, 180))
        bin_angle = round(_idx_center_angle_deg(_angle_to_idx(float(angle_deg), cfg), cfg), 2)
        if (
            result is not None
            and clipped_distance <= float(cfg.self_hit_max_distance_m)
            and bin_angle in self_hit_angle_set
        ):
            color = (0, 0, 255)
        cv2.circle(canvas, (x, y), 2, color, -1)

    if result is not None:
        _draw_self_hit_arcs(
            canvas,
            size=size,
            max_distance_m=max_distance_m,
            intervals_deg=result.self_hit_intervals_deg,
            self_hit_max_distance_m=result.self_hit_max_distance_m,
        )
        _draw_sector_overlay(
            canvas,
            size=size,
            max_distance_m=max_distance_m,
            forward_angle_deg=result.recommended_forward_angle_deg,
            half_angle_deg=result.recommended_half_angle_deg,
            min_distance_m=result.recommended_min_distance_m,
            color=(0, 220, 80),
        )

    top_line = (
        f"Calibrated  scans={progress_scans}/{total_scans}  rpm={rpm:.1f}"
        if result is not None
        else f"Capturing baseline  scans={progress_scans}/{total_scans}  rpm={rpm:.1f}"
    )
    cv2.putText(
        canvas,
        top_line,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    if result is not None:
        cv2.putText(
            canvas,
            (
                f"forward={result.recommended_forward_angle_deg:.1f}deg  "
                f"half={result.recommended_half_angle_deg:.1f}deg  "
                f"min_distance={result.recommended_min_distance_m:.2f}m"
            ),
            (16, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 220, 80),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            (
                f"tripwire_half_width@{cfg.tripwire_distance_m:.2f}m="
                f"{result.recommended_tripwire_half_width_m:.2f}m"
            ),
            (16, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            canvas,
            (
                f"Place robot in open space. Near returns <= {cfg.self_hit_max_distance_m:.2f}m "
                "will be treated as robot self-occlusion."
            ),
            (16, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "Red near-field arc = robot self-hits. Green wedge = recommended forward sector after near-field ignore distance.",
        (16, size - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _compute_result(scans: list[list[list[float]]], cfg: CalibrationConfig) -> CalibrationResult:
    n_bins = _n_bins(cfg)
    hit_counts = np.zeros((n_bins,), dtype=np.int32)
    distance_samples: list[list[float]] = [[] for _ in range(n_bins)]

    for points in scans:
        per_scan_nearest: dict[int, float] = {}
        for angle_deg, distance_m, confidence in points:
            if int(confidence) < int(cfg.min_confidence):
                continue
            distance = float(distance_m)
            if not math.isfinite(distance) or distance <= 0.0 or distance > float(cfg.self_hit_max_distance_m):
                continue
            idx = _angle_to_idx(float(angle_deg), cfg)
            nearest = per_scan_nearest.get(idx)
            if nearest is None or distance < nearest:
                per_scan_nearest[idx] = distance
        for idx, nearest in per_scan_nearest.items():
            hit_counts[idx] += 1
            distance_samples[idx].append(float(nearest))

    captured_scans = max(len(scans), 1)
    hit_ratio = hit_counts.astype(np.float32) / float(captured_scans)
    median_distance = np.full((n_bins,), np.nan, dtype=np.float32)
    for idx, samples in enumerate(distance_samples):
        if samples:
            median_distance[idx] = float(median(samples))

    raw_mask = (hit_ratio >= float(cfg.min_hit_ratio)) & np.isfinite(median_distance)
    if not bool(np.any(raw_mask)) and bool(np.any(hit_counts > 0)):
        adaptive_ratio = max(0.05, float(np.nanmax(hit_ratio)) * 0.6)
        raw_mask = (hit_ratio >= adaptive_ratio) & np.isfinite(median_distance)
    grow_bins = max(int(round(float(cfg.mask_grow_deg) / max(float(cfg.angle_bin_deg), 0.25))), 0)
    expanded_mask = _expand_mask(raw_mask, grow_bins)

    raw_indices = np.flatnonzero(raw_mask)
    if raw_indices.size == 0:
        recommended_forward_angle_deg = float(cfg.nominal_forward_angle_deg or 180.0)
        recommended_half_angle_deg = 35.0
    else:
        cartesian_points: list[tuple[float, float, float]] = []
        for idx in map(int, raw_indices):
            distance = float(median_distance[idx])
            angle_deg = _idx_center_angle_deg(idx, cfg)
            theta = math.radians(angle_deg - 90.0)
            x = distance * math.cos(theta)
            y = distance * math.sin(theta)
            cartesian_points.append((x, y, angle_deg))

        centroid_x = sum(x for x, _y, _angle in cartesian_points) / float(len(cartesian_points))
        centroid_y = sum(y for _x, y, _angle in cartesian_points) / float(len(cartesian_points))
        if abs(centroid_x) < 1e-6 and abs(centroid_y) < 1e-6:
            recommended_forward_angle_deg = float(cfg.nominal_forward_angle_deg or 180.0)
        else:
            recommended_forward_angle_deg = _normalize_angle_360_deg(
                math.degrees(math.atan2(centroid_y, centroid_x)) + 90.0
            )

        raw_deltas = np.asarray(
            [
                abs(_normalize_angle_deg(float(angle_deg) - float(recommended_forward_angle_deg)))
                for _x, _y, angle_deg in cartesian_points
            ],
            dtype=np.float32,
        )
        if raw_deltas.size == 0:
            visible_half_angle_deg = 25.0
        else:
            visible_half_angle_deg = float(np.percentile(raw_deltas, 92.0))
        recommended_half_angle_deg = min(
            float(cfg.max_sector_deg) / 2.0,
            max(12.0, visible_half_angle_deg + float(cfg.sector_margin_deg)),
        )

    sector_mask = _sector_mask(recommended_forward_angle_deg, recommended_half_angle_deg, cfg)

    self_in_sector = raw_mask & sector_mask & np.isfinite(median_distance)
    if bool(np.any(self_in_sector)):
        recommended_min_distance_m = max(
            float(cfg.default_min_distance_m),
            float(np.nanmax(median_distance[self_in_sector])) + float(cfg.min_distance_margin_m),
        )
    else:
        recommended_min_distance_m = float(cfg.default_min_distance_m)

    width_half_angle_deg = min(float(recommended_half_angle_deg), 80.0)
    recommended_tripwire_half_width_m = max(
        0.12,
        float(cfg.tripwire_distance_m) * math.tan(math.radians(width_half_angle_deg)),
    )
    recommended_tripwire_half_width_m = min(
        recommended_tripwire_half_width_m,
        float(cfg.max_tripwire_half_width_m),
    )

    recommended_sector_start_deg = _normalize_angle_360_deg(
        float(recommended_forward_angle_deg) - float(recommended_half_angle_deg)
    )
    recommended_sector_end_deg = _normalize_angle_360_deg(
        float(recommended_forward_angle_deg) + float(recommended_half_angle_deg)
    )

    self_hit_bins_deg = [
        round(_idx_center_angle_deg(idx, cfg), 2)
        for idx in np.flatnonzero(expanded_mask)
    ]
    self_hit_median_distance_m = [
        None if not math.isfinite(float(x)) else round(float(x), 4) for x in median_distance.tolist()
    ]
    self_hit_ratio = [round(float(x), 4) for x in hit_ratio.tolist()]

    return CalibrationResult(
        schema="sourccey.lidar_collision_calibration.v1",
        captured_scans=captured_scans,
        angle_bin_deg=float(cfg.angle_bin_deg),
        self_hit_max_distance_m=float(cfg.self_hit_max_distance_m),
        min_hit_ratio=float(cfg.min_hit_ratio),
        recommended_forward_angle_deg=round(float(recommended_forward_angle_deg), 3),
        recommended_half_angle_deg=round(float(recommended_half_angle_deg), 3),
        recommended_min_distance_m=round(float(recommended_min_distance_m), 3),
        recommended_tripwire_half_width_m=round(float(recommended_tripwire_half_width_m), 3),
        recommended_sector_start_deg=round(float(recommended_sector_start_deg), 3),
        recommended_sector_end_deg=round(float(recommended_sector_end_deg), 3),
        self_hit_intervals_deg=_mask_intervals(expanded_mask, cfg),
        self_hit_bins_deg=self_hit_bins_deg,
        self_hit_median_distance_m=self_hit_median_distance_m,
        self_hit_ratio=self_hit_ratio,
    )


def _print_summary(result: CalibrationResult, output_path: Path, host: str, port: int) -> None:
    print()
    print("LiDAR collision calibration complete")
    print(f"Saved: {output_path}")
    print(f"Recommended forward angle: {result.recommended_forward_angle_deg:.2f} deg")
    print(f"Recommended visible half-angle: {result.recommended_half_angle_deg:.2f} deg")
    print(f"Recommended min-distance: {result.recommended_min_distance_m:.2f} m")
    print(
        f"Recommended tripwire half-width @ 0.55m-ish: {result.recommended_tripwire_half_width_m:.2f} m"
    )
    print(
        f"Recommended sector: {result.recommended_sector_start_deg:.2f} deg -> "
        f"{result.recommended_sector_end_deg:.2f} deg"
    )
    print()
    print("Suggested wedge check command:")
    print(
        "uv run python scripts/ldlidar_stop_zone_client.py "
        f"--host {host} --port {port} --mode wedge "
        f"--forward-angle-deg {result.recommended_forward_angle_deg:.2f} "
        f"--half-angle-deg {result.recommended_half_angle_deg:.2f} "
        f"--min-distance-m {result.recommended_min_distance_m:.2f}"
    )
    print()
    print("Suggested tripwire starting values:")
    print(
        "  "
        f"--forward-angle-deg {result.recommended_forward_angle_deg:.2f} "
        f"--min-distance-m {result.recommended_min_distance_m:.2f} "
        f"--tripwire-half-width-m {result.recommended_tripwire_half_width_m:.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the LiDAR collision-detection sector by capturing a baseline "
            "scan in open space and learning which near returns belong to the robot itself."
        )
    )
    parser.add_argument("--host", required=True, help="Host IP running ldlidar_stream_host.py.")
    parser.add_argument("--port", type=int, default=8765, help="TCP port exposed by the host.")
    parser.add_argument("--capture-scans", type=int, default=120, help="Number of scans to capture before calibrating.")
    parser.add_argument("--angle-bin-deg", type=float, default=2.0, help="Angular bin size in degrees.")
    parser.add_argument(
        "--self-hit-max-distance-m",
        type=float,
        default=1.5,
        help="Near returns inside this distance are candidates for robot self-occlusion.",
    )
    parser.add_argument("--min-confidence", type=int, default=0, help="Minimum confidence to use a point.")
    parser.add_argument(
        "--min-hit-ratio",
        type=float,
        default=0.18,
        help="Fraction of captured scans that must show a near return before an angle is treated as robot self-hit.",
    )
    parser.add_argument("--mask-grow-deg", type=float, default=4.0, help="Extra angular padding around learned self-hit intervals.")
    parser.add_argument("--sector-margin-deg", type=float, default=6.0, help="Extra angular padding added around the learned forward opening.")
    parser.add_argument("--max-sector-deg", type=float, default=180.0, help="Maximum recommended sector width.")
    parser.add_argument("--default-min-distance-m", type=float, default=0.12, help="Fallback minimum ignore distance.")
    parser.add_argument("--min-distance-margin-m", type=float, default=0.05, help="Extra margin added beyond learned self-hit depth.")
    parser.add_argument(
        "--nominal-forward-angle-deg",
        type=float,
        default=None,
        help="Optional prior guess for robot forward direction in raw LiDAR angle coordinates.",
    )
    parser.add_argument(
        "--tripwire-distance-m",
        type=float,
        default=0.55,
        help="Reference distance used when estimating a matching tripwire half-width.",
    )
    parser.add_argument(
        "--max-tripwire-half-width-m",
        type=float,
        default=0.9,
        help="Clamp the suggested tripwire half-width to a sane value.",
    )
    parser.add_argument("--max-distance-m", type=float, default=8.0, help="Viewer distance scale.")
    parser.add_argument("--canvas-size", type=int, default=900, help="Viewer size in pixels.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lidar_calibration/latest_collision_calibration.json"),
        help="Where to save the calibration JSON.",
    )
    args = parser.parse_args()

    cfg = CalibrationConfig(
        capture_scans=max(int(args.capture_scans), 8),
        angle_bin_deg=max(float(args.angle_bin_deg), 0.25),
        self_hit_max_distance_m=max(float(args.self_hit_max_distance_m), 0.05),
        min_confidence=max(int(args.min_confidence), 0),
        min_hit_ratio=min(max(float(args.min_hit_ratio), 0.05), 1.0),
        mask_grow_deg=max(float(args.mask_grow_deg), 0.0),
        sector_margin_deg=max(float(args.sector_margin_deg), 0.0),
        max_sector_deg=min(max(float(args.max_sector_deg), 20.0), 330.0),
        default_min_distance_m=max(float(args.default_min_distance_m), 0.0),
        min_distance_margin_m=max(float(args.min_distance_margin_m), 0.0),
        tripwire_distance_m=max(float(args.tripwire_distance_m), 0.05),
        max_tripwire_half_width_m=max(float(args.max_tripwire_half_width_m), 0.12),
        nominal_forward_angle_deg=(
            None if args.nominal_forward_angle_deg is None else float(args.nominal_forward_angle_deg)
        ),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    sock = socket.create_connection((args.host, int(args.port)), timeout=10.0)
    file_obj = sock.makefile("r", encoding="utf-8")
    print(f"Connected to tcp://{args.host}:{args.port}")
    print(
        "Make sure the robot has open space around the LiDAR. "
        f"Capturing {cfg.capture_scans} baseline scans..."
    )
    viewer = _TkViewer("ldlidar_collision_calibration")

    captured_scans: list[list[list[float]]] = []
    result: CalibrationResult | None = None
    saved_result = False

    try:
        for line in file_obj:
            payload = json.loads(line)
            points = payload.get("points", [])
            rpm = float(payload.get("rpm", 0.0))
            if result is None and len(captured_scans) < cfg.capture_scans:
                captured_scans.append(points)
                if len(captured_scans) >= cfg.capture_scans:
                    result = _compute_result(captured_scans, cfg)
                    args.output.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
                    _print_summary(result, args.output, args.host, int(args.port))
                    saved_result = True

            frame = _draw_points(
                points,
                size=int(args.canvas_size),
                max_distance_m=float(args.max_distance_m),
                rpm=rpm,
                progress_scans=min(len(captured_scans), cfg.capture_scans),
                total_scans=cfg.capture_scans,
                result=result,
                cfg=cfg,
            )
            viewer.show_bgr(frame)
            if viewer.closed:
                break
    finally:
        if not saved_result and captured_scans:
            result = _compute_result(captured_scans, cfg)
            args.output.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
            _print_summary(result, args.output, args.host, int(args.port))
        try:
            file_obj.close()
        except OSError:
            pass
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
