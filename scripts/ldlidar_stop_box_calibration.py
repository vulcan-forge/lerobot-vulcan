from __future__ import annotations

import argparse
import json
import math
import socket
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

from ldlidar_defaults import (
    DEFAULT_LIDAR_FORWARD_ANGLE_DEG,
    DEFAULT_LIDAR_STOP_BOX_DISTANCE_M,
    DEFAULT_LIDAR_STOP_BOX_HALF_WIDTH_M,
    DEFAULT_LIDAR_STOP_BOX_MIN_DISTANCE_M,
    DEFAULT_LIDAR_STOP_BOX_THICKNESS_M,
)


def _normalize_angle_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _normalize_angle_360_deg(angle_deg: float) -> float:
    return float(angle_deg) % 360.0


@dataclass
class CalibrationConfig:
    capture_scans: int
    min_confidence: int
    max_box_distance_m: float
    nominal_forward_angle_deg: float | None
    forward_percentile: float
    lateral_percentile: float
    depth_padding_m: float
    width_padding_m: float
    min_tripwire_distance_m: float
    min_tripwire_half_width_m: float
    tripwire_thickness_m: float
    min_distance_m: float


@dataclass
class CalibrationResult:
    schema: str
    captured_scans: int
    recommended_forward_angle_deg: float
    recommended_min_distance_m: float
    recommended_tripwire_distance_m: float
    recommended_tripwire_half_width_m: float
    recommended_tripwire_thickness_m: float
    forward_samples_m: list[float]
    lateral_samples_m: list[float]
    front_face_points: int
    side_face_points: int


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


def _infer_forward_angle(points: list[list[float]], cfg: CalibrationConfig) -> float:
    if cfg.nominal_forward_angle_deg is not None:
        return _normalize_angle_360_deg(cfg.nominal_forward_angle_deg)
    near_points: list[tuple[float, float]] = []
    for angle_deg, distance_m, confidence in points:
        if int(confidence) < cfg.min_confidence:
            continue
        distance = float(distance_m)
        if not math.isfinite(distance) or distance <= 0.0 or distance > float(cfg.max_box_distance_m):
            continue
        theta = math.radians(float(angle_deg) - 90.0)
        x = distance * math.cos(theta)
        y = distance * math.sin(theta)
        near_points.append((x, y))
    if not near_points:
        return 270.0
    centroid_x = sum(x for x, _y in near_points) / float(len(near_points))
    centroid_y = sum(y for _x, y in near_points) / float(len(near_points))
    return _normalize_angle_360_deg(math.degrees(math.atan2(centroid_y, centroid_x)) + 90.0)


def _to_local(angle_deg: float, distance_m: float, forward_angle_deg: float) -> tuple[float, float]:
    delta = _normalize_angle_deg(float(angle_deg) - float(forward_angle_deg))
    theta = math.radians(delta)
    forward_m = float(distance_m) * math.cos(theta)
    lateral_m = float(distance_m) * math.sin(theta)
    return forward_m, lateral_m


def _robust_percentile(values: list[float], pct: float, fallback: float) -> float:
    if not values:
        return float(fallback)
    return float(np.percentile(np.asarray(values, dtype=np.float32), pct))


def _compute_result(scans: list[list[list[float]]], cfg: CalibrationConfig) -> CalibrationResult:
    aggregate_points = [point for scan in scans for point in scan]
    forward_angle_deg = _infer_forward_angle(aggregate_points, cfg)

    local_points: list[tuple[float, float]] = []
    for angle_deg, distance_m, confidence in aggregate_points:
        if int(confidence) < cfg.min_confidence:
            continue
        distance = float(distance_m)
        if not math.isfinite(distance) or distance <= 0.0 or distance > float(cfg.max_box_distance_m):
            continue
        forward_m, lateral_m = _to_local(float(angle_deg), distance, forward_angle_deg)
        if forward_m < 0.0:
            continue
        local_points.append((forward_m, lateral_m))

    if not local_points:
        raise RuntimeError("No usable boxed-in LiDAR points were captured. Move the box closer or raise --max-box-distance-m.")

    forward_all = [forward_m for forward_m, _lateral_m in local_points]
    lateral_all = [abs(lateral_m) for _forward_m, lateral_m in local_points]

    front_depth_raw = _robust_percentile(forward_all, cfg.forward_percentile, fallback=0.30)
    front_depth_m = max(float(cfg.min_tripwire_distance_m), front_depth_raw - float(cfg.depth_padding_m))

    side_candidates = [
        abs(lateral_m)
        for forward_m, lateral_m in local_points
        if forward_m >= max(0.03, front_depth_raw * 0.30)
    ]
    half_width_raw = _robust_percentile(side_candidates, cfg.lateral_percentile, fallback=max(lateral_all) if lateral_all else 0.25)
    half_width_m = max(float(cfg.min_tripwire_half_width_m), half_width_raw - float(cfg.width_padding_m))

    front_face_points = sum(1 for forward_m, lateral_m in local_points if forward_m >= front_depth_raw * 0.85 and abs(lateral_m) <= half_width_raw * 0.65)
    side_face_points = sum(1 for forward_m, lateral_m in local_points if forward_m >= front_depth_raw * 0.20 and abs(lateral_m) >= half_width_raw * 0.60)

    return CalibrationResult(
        schema="sourccey.lidar_stop_box_calibration.v1",
        captured_scans=len(scans),
        recommended_forward_angle_deg=round(float(forward_angle_deg), 3),
        recommended_min_distance_m=round(float(cfg.min_distance_m), 3),
        recommended_tripwire_distance_m=round(float(front_depth_m), 3),
        recommended_tripwire_half_width_m=round(float(half_width_m), 3),
        recommended_tripwire_thickness_m=round(float(cfg.tripwire_thickness_m), 3),
        forward_samples_m=[round(float(x), 4) for x in forward_all[:2000]],
        lateral_samples_m=[round(float(x), 4) for x in lateral_all[:2000]],
        front_face_points=int(front_face_points),
        side_face_points=int(side_face_points),
    )


def _draw_stop_box_overlay(
    canvas: np.ndarray,
    *,
    size: int,
    max_distance_m: float,
    forward_angle_deg: float,
    tripwire_distance_m: float,
    tripwire_half_width_m: float,
    tripwire_thickness_m: float,
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

    def local_to_xy(forward_m: float, lateral_m: float) -> tuple[int, int]:
        angle_deg = float(forward_angle_deg) + math.degrees(math.atan2(lateral_m, forward_m))
        distance_m = math.hypot(forward_m, lateral_m)
        return to_xy(angle_deg, distance_m)

    near_edge_m = 0.0
    far_edge_m = float(tripwire_distance_m) + float(tripwire_thickness_m) / 2.0
    box_points = np.array(
        [
            local_to_xy(near_edge_m, -tripwire_half_width_m),
            local_to_xy(far_edge_m, -tripwire_half_width_m),
            local_to_xy(far_edge_m, tripwire_half_width_m),
            local_to_xy(near_edge_m, tripwire_half_width_m),
        ],
        dtype=np.int32,
    )
    overlay = canvas.copy()
    cv2.fillConvexPoly(overlay, box_points, color)
    cv2.addWeighted(overlay, 0.16, canvas, 0.84, 0.0, dst=canvas)

    near_left = local_to_xy(near_edge_m, -tripwire_half_width_m)
    near_right = local_to_xy(near_edge_m, tripwire_half_width_m)
    far_left = local_to_xy(far_edge_m, -tripwire_half_width_m)
    far_right = local_to_xy(far_edge_m, tripwire_half_width_m)
    center_left = local_to_xy(tripwire_distance_m, -tripwire_half_width_m)
    center_right = local_to_xy(tripwire_distance_m, tripwire_half_width_m)
    cv2.line(canvas, near_left, near_right, color, 1, cv2.LINE_AA)
    cv2.line(canvas, far_left, far_right, color, 1, cv2.LINE_AA)
    cv2.line(canvas, center_left, center_right, color, 3, cv2.LINE_AA)
    cv2.line(canvas, near_left, far_left, color, 1, cv2.LINE_AA)
    cv2.line(canvas, near_right, far_right, color, 1, cv2.LINE_AA)


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

    for angle_deg, distance_m, confidence in points:
        clipped_distance = min(max(float(distance_m), 0.0), max_distance_m)
        norm = clipped_distance / max(max_distance_m, 1e-6)
        px_radius = int(norm * radius_px)
        theta = math.radians(float(angle_deg) - 90.0)
        x = int(center + px_radius * math.cos(theta))
        y = int(center + px_radius * math.sin(theta))
        conf = max(0, min(int(confidence), 255))
        color = (0, conf, 255 - min(conf, 180))
        cv2.circle(canvas, (x, y), 2, color, -1)

    if result is not None:
        _draw_stop_box_overlay(
            canvas,
            size=size,
            max_distance_m=max_distance_m,
            forward_angle_deg=result.recommended_forward_angle_deg,
            tripwire_distance_m=result.recommended_tripwire_distance_m,
            tripwire_half_width_m=result.recommended_tripwire_half_width_m,
            tripwire_thickness_m=result.recommended_tripwire_thickness_m,
            color=(0, 220, 80),
        )

    title = (
        f"Calibrated  scans={progress_scans}/{total_scans}  rpm={rpm:.1f}"
        if result is not None
        else f"Capturing stop box  scans={progress_scans}/{total_scans}  rpm={rpm:.1f}"
    )
    cv2.putText(canvas, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

    if result is None:
        cv2.putText(
            canvas,
            "Physically box the LiDAR in at the desired stop boundary, then hold still.",
            (16, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            canvas,
            (
                f"forward={result.recommended_forward_angle_deg:.1f}deg  "
                f"depth={result.recommended_tripwire_distance_m:.2f}m  "
                f"width={result.recommended_tripwire_half_width_m * 2.0:.2f}m"
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
                f"min_distance={result.recommended_min_distance_m:.2f}m  "
                f"thickness={result.recommended_tripwire_thickness_m:.2f}m"
            ),
            (16, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "Green rectangle = learned forward no-go stop box.",
        (16, size - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _print_summary(result: CalibrationResult, output_path: Path, host: str, port: int) -> None:
    print()
    print("LiDAR stop-box calibration complete")
    print(f"Saved: {output_path}")
    print(f"Recommended forward angle: {result.recommended_forward_angle_deg:.2f} deg")
    print(f"Recommended tripwire distance: {result.recommended_tripwire_distance_m:.2f} m")
    print(f"Recommended tripwire half-width: {result.recommended_tripwire_half_width_m:.2f} m")
    print(f"Recommended tripwire thickness: {result.recommended_tripwire_thickness_m:.2f} m")
    print()
    print("Suggested stop-zone viewer command:")
    print(
        "uv run python scripts/ldlidar_stop_zone_client.py "
        f"--host {host} --port {port} --mode tripwire "
        f"--forward-angle-deg {result.recommended_forward_angle_deg:.2f} "
        f"--min-distance-m {result.recommended_min_distance_m:.2f} "
        f"--tripwire-distance-m {result.recommended_tripwire_distance_m:.2f} "
        f"--tripwire-half-width-m {result.recommended_tripwire_half_width_m:.2f} "
        f"--tripwire-thickness-m {result.recommended_tripwire_thickness_m:.2f}"
    )
    print()
    print("Suggested safe teleop command:")
    print(
        "uv run python scripts/ldlidar_safe_base_teleop.py "
        f"--forward-angle-deg {result.recommended_forward_angle_deg:.2f} "
        f"--min-distance-m {result.recommended_min_distance_m:.2f} "
        f"--tripwire-distance-m {result.recommended_tripwire_distance_m:.2f} "
        f"--tripwire-half-width-m {result.recommended_tripwire_half_width_m:.2f} "
        f"--tripwire-thickness-m {result.recommended_tripwire_thickness_m:.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Learn a forward no-go stop box by physically boxing the LiDAR in at the "
            "desired collision boundary."
        )
    )
    parser.add_argument("--host", required=True, help="Host IP running ldlidar_stream_host.py.")
    parser.add_argument("--port", type=int, default=8765, help="TCP port exposed by the host.")
    parser.add_argument("--capture-scans", type=int, default=120, help="Number of scans to capture before calibrating.")
    parser.add_argument("--min-confidence", type=int, default=0, help="Minimum confidence to use a point.")
    parser.add_argument("--max-box-distance-m", type=float, default=1.8, help="Ignore points farther away than this while learning the stop box.")
    parser.add_argument("--forward-angle-deg", type=float, default=DEFAULT_LIDAR_FORWARD_ANGLE_DEG, help="Known forward direction in raw LiDAR angle coordinates.")
    parser.add_argument("--forward-percentile", type=float, default=92.0, help="Percentile used to estimate the front face depth.")
    parser.add_argument("--lateral-percentile", type=float, default=92.0, help="Percentile used to estimate side-face half-width.")
    parser.add_argument("--depth-padding-m", type=float, default=0.03, help="Pull the learned front face slightly inward for a conservative stop boundary.")
    parser.add_argument("--width-padding-m", type=float, default=0.02, help="Pull the learned side faces slightly inward for a conservative stop boundary.")
    parser.add_argument("--min-tripwire-distance-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_DISTANCE_M, help="Minimum allowed learned stop depth.")
    parser.add_argument("--min-tripwire-half-width-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_HALF_WIDTH_M, help="Minimum allowed learned half-width.")
    parser.add_argument("--tripwire-thickness-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_THICKNESS_M, help="Thickness of the resulting stop box.")
    parser.add_argument("--min-distance-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_MIN_DISTANCE_M, help="Min-distance flag to emit with the learned stop box.")
    parser.add_argument("--max-distance-m", type=float, default=8.0, help="Viewer distance scale.")
    parser.add_argument("--canvas-size", type=int, default=900, help="Viewer size in pixels.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lidar_calibration/latest_stop_box_calibration.json"),
        help="Where to save the calibration JSON.",
    )
    args = parser.parse_args()

    cfg = CalibrationConfig(
        capture_scans=max(int(args.capture_scans), 8),
        min_confidence=max(int(args.min_confidence), 0),
        max_box_distance_m=max(float(args.max_box_distance_m), 0.10),
        nominal_forward_angle_deg=(None if args.forward_angle_deg is None else float(args.forward_angle_deg)),
        forward_percentile=min(max(float(args.forward_percentile), 50.0), 100.0),
        lateral_percentile=min(max(float(args.lateral_percentile), 50.0), 100.0),
        depth_padding_m=max(float(args.depth_padding_m), 0.0),
        width_padding_m=max(float(args.width_padding_m), 0.0),
        min_tripwire_distance_m=max(float(args.min_tripwire_distance_m), 0.0),
        min_tripwire_half_width_m=max(float(args.min_tripwire_half_width_m), 0.02),
        tripwire_thickness_m=max(float(args.tripwire_thickness_m), 0.02),
        min_distance_m=max(float(args.min_distance_m), 0.0),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    sock = socket.create_connection((args.host, int(args.port)), timeout=10.0)
    file_obj = sock.makefile("r", encoding="utf-8")
    print(f"Connected to tcp://{args.host}:{args.port}")
    print(
        "Box the LiDAR in from the front and both sides at the exact boundary where the robot must stop. "
        f"Capturing {cfg.capture_scans} scans..."
    )
    viewer = _TkViewer("ldlidar_stop_box_calibration")

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
