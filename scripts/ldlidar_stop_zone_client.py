from __future__ import annotations

import argparse
import json
import math
import socket
import tkinter as tk
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageTk

from ldlidar_defaults import (
    DEFAULT_LIDAR_FORWARD_ANGLE_DEG,
    DEFAULT_LIDAR_STOP_BOX_DISTANCE_M,
    DEFAULT_LIDAR_STOP_BOX_HALF_WIDTH_M,
    DEFAULT_LIDAR_STOP_BOX_MIN_DISTANCE_M,
    DEFAULT_LIDAR_STOP_BOX_THICKNESS_M,
)


@dataclass
class StopZoneConfig:
    forward_angle_deg: float
    half_angle_deg: float
    min_distance_m: float
    max_distance_m: float
    min_points_to_trigger: int
    mode: str
    tripwire_distance_m: float
    tripwire_half_width_m: float
    tripwire_thickness_m: float


def _normalize_angle_deg(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def _point_in_stop_zone(
    angle_deg: float,
    distance_m: float,
    cfg: StopZoneConfig,
) -> bool:
    delta = _normalize_angle_deg(angle_deg - cfg.forward_angle_deg)
    theta = math.radians(delta)
    forward_m = distance_m * math.cos(theta)
    lateral_m = distance_m * math.sin(theta)

    if cfg.mode == "tripwire":
        near_edge_m = 0.0
        far_edge_m = max(cfg.min_distance_m, cfg.tripwire_distance_m + cfg.tripwire_thickness_m / 2.0)
        return (
            forward_m >= near_edge_m
            and forward_m <= far_edge_m
            and abs(lateral_m) <= cfg.tripwire_half_width_m
        )

    if distance_m < cfg.min_distance_m or distance_m > cfg.max_distance_m:
        return False
    return abs(delta) <= cfg.half_angle_deg


def _draw_stop_zone_overlay(
    canvas: np.ndarray,
    *,
    size: int,
    max_distance_m: float,
    cfg: StopZoneConfig,
    blocked: bool,
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

    color = (0, 0, 255) if blocked else (255, 200, 0)
    if cfg.mode == "tripwire":
        def local_to_xy(forward_m: float, lateral_m: float) -> tuple[int, int]:
            angle_deg = cfg.forward_angle_deg + math.degrees(math.atan2(lateral_m, forward_m))
            distance_m = math.hypot(forward_m, lateral_m)
            return to_xy(angle_deg, distance_m)

        near_edge_m = 0.0
        far_edge_m = max(cfg.min_distance_m, cfg.tripwire_distance_m + cfg.tripwire_thickness_m / 2.0)

        box_points = np.array(
            [
                local_to_xy(near_edge_m, -cfg.tripwire_half_width_m),
                local_to_xy(far_edge_m, -cfg.tripwire_half_width_m),
                local_to_xy(far_edge_m, cfg.tripwire_half_width_m),
                local_to_xy(near_edge_m, cfg.tripwire_half_width_m),
            ],
            dtype=np.int32,
        )
        overlay = canvas.copy()
        cv2.fillConvexPoly(overlay, box_points, color)
        cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0.0, dst=canvas)

        near_left = local_to_xy(near_edge_m, -cfg.tripwire_half_width_m)
        near_right = local_to_xy(near_edge_m, cfg.tripwire_half_width_m)
        far_left = local_to_xy(far_edge_m, -cfg.tripwire_half_width_m)
        far_right = local_to_xy(far_edge_m, cfg.tripwire_half_width_m)
        center_left = local_to_xy(cfg.tripwire_distance_m, -cfg.tripwire_half_width_m)
        center_right = local_to_xy(cfg.tripwire_distance_m, cfg.tripwire_half_width_m)

        cv2.line(canvas, near_left, near_right, color, 1, cv2.LINE_AA)
        cv2.line(canvas, far_left, far_right, color, 1, cv2.LINE_AA)
        cv2.line(canvas, center_left, center_right, color, 3, cv2.LINE_AA)
        cv2.line(canvas, near_left, far_left, color, 1, cv2.LINE_AA)
        cv2.line(canvas, near_right, far_right, color, 1, cv2.LINE_AA)
        return

    start_inner = to_xy(cfg.forward_angle_deg - cfg.half_angle_deg, cfg.min_distance_m)
    end_inner = to_xy(cfg.forward_angle_deg + cfg.half_angle_deg, cfg.min_distance_m)
    start_outer = to_xy(cfg.forward_angle_deg - cfg.half_angle_deg, cfg.max_distance_m)
    end_outer = to_xy(cfg.forward_angle_deg + cfg.half_angle_deg, cfg.max_distance_m)

    cv2.line(canvas, start_inner, start_outer, color, 2, cv2.LINE_AA)
    cv2.line(canvas, end_inner, end_outer, color, 2, cv2.LINE_AA)
    cv2.ellipse(
        canvas,
        (center, center),
        (int(radius_px * cfg.max_distance_m / max_distance_m), int(radius_px * cfg.max_distance_m / max_distance_m)),
        0.0,
        cfg.forward_angle_deg - cfg.half_angle_deg - 90.0,
        cfg.forward_angle_deg + cfg.half_angle_deg - 90.0,
        color,
        2,
        cv2.LINE_AA,
    )
    if cfg.min_distance_m > 0.0:
        cv2.ellipse(
            canvas,
            (center, center),
            (int(radius_px * cfg.min_distance_m / max_distance_m), int(radius_px * cfg.min_distance_m / max_distance_m)),
            0.0,
            cfg.forward_angle_deg - cfg.half_angle_deg - 90.0,
            cfg.forward_angle_deg + cfg.half_angle_deg - 90.0,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_points(
    points: list[list[float]],
    *,
    size: int,
    max_distance_m: float,
    rpm: float,
    zone_cfg: StopZoneConfig,
    blocked: bool,
    blocked_points: int,
) -> np.ndarray:
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    radius_px = center - 24

    for ring_fraction in (0.25, 0.5, 0.75, 1.0):
        cv2.circle(canvas, (center, center), int(radius_px * ring_fraction), (40, 40, 40), 1)
    cv2.line(canvas, (center, 16), (center, size - 16), (30, 30, 30), 1)
    cv2.line(canvas, (16, center), (size - 16, center), (30, 30, 30), 1)
    cv2.circle(canvas, (center, center), 4, (0, 180, 255), -1)

    _draw_stop_zone_overlay(
        canvas,
        size=size,
        max_distance_m=max_distance_m,
        cfg=zone_cfg,
        blocked=blocked,
    )

    for angle_deg, distance_m, confidence in points:
        clipped_distance = min(max(float(distance_m), 0.0), max_distance_m)
        norm = clipped_distance / max(max_distance_m, 1e-6)
        px_radius = int(norm * radius_px)
        theta = math.radians(float(angle_deg) - 90.0)
        x = int(center + px_radius * math.cos(theta))
        y = int(center + px_radius * math.sin(theta))
        conf = max(0, min(int(confidence), 255))
        color = (0, conf, 255 - min(conf, 180))
        if _point_in_stop_zone(float(angle_deg), float(distance_m), zone_cfg):
            color = (0, 0, 255) if blocked else (0, 200, 255)
        cv2.circle(canvas, (x, y), 2, color, -1)

    status_text = "STOP" if blocked else "CLEAR"
    status_color = (0, 0, 255) if blocked else (0, 220, 80)
    cv2.putText(
        canvas,
        f"{status_text}  zone_points={blocked_points}/{zone_cfg.min_points_to_trigger}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"rpm={rpm:.1f} total_points={len(points)} forward={zone_cfg.forward_angle_deg:.0f}deg",
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (
            f"tripwire_box={zone_cfg.tripwire_distance_m:.2f}m width={zone_cfg.tripwire_half_width_m * 2.0:.2f}m"
            if zone_cfg.mode == "tripwire"
            else f"zone={zone_cfg.min_distance_m:.2f}m..{zone_cfg.max_distance_m:.2f}m half_angle={zone_cfg.half_angle_deg:.0f}deg"
        ),
        (16, size - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return canvas


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Remote viewer with a front stop-zone detector.")
    parser.add_argument("--host", required=True, help="Host IP running ldlidar_stream_host.py.")
    parser.add_argument("--port", type=int, default=8765, help="TCP port exposed by the host.")
    parser.add_argument("--max-distance-m", type=float, default=8.0, help="Viewer distance scale.")
    parser.add_argument("--canvas-size", type=int, default=900, help="Viewer size in pixels.")
    parser.add_argument("--mode", choices=("wedge", "tripwire"), default="tripwire", help="Detection shape.")
    parser.add_argument("--forward-angle-deg", type=float, default=DEFAULT_LIDAR_FORWARD_ANGLE_DEG, help="Center direction for the stop zone.")
    parser.add_argument("--half-angle-deg", type=float, default=35.0, help="Half-width of the stop zone wedge.")
    parser.add_argument("--min-distance-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_MIN_DISTANCE_M, help="Ignore points closer than this.")
    parser.add_argument("--stop-distance-m", type=float, default=2.50, help="Trigger zone far edge.")
    parser.add_argument("--tripwire-distance-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_DISTANCE_M, help="Distance of the near stop line from the robot.")
    parser.add_argument("--tripwire-half-width-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_HALF_WIDTH_M, help="Half-width of the stop line corridor.")
    parser.add_argument("--tripwire-thickness-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_THICKNESS_M, help="Thickness of the stop line band.")
    parser.add_argument("--min-points", type=int, default=8, help="Points required in the zone before STOP.")
    args = parser.parse_args()

    zone_cfg = StopZoneConfig(
        forward_angle_deg=float(args.forward_angle_deg),
        half_angle_deg=float(args.half_angle_deg),
        min_distance_m=float(args.min_distance_m),
        max_distance_m=float(args.stop_distance_m),
        min_points_to_trigger=max(int(args.min_points), 1),
        mode=str(args.mode),
        tripwire_distance_m=float(args.tripwire_distance_m),
        tripwire_half_width_m=float(args.tripwire_half_width_m),
        tripwire_thickness_m=float(args.tripwire_thickness_m),
    )

    sock = socket.create_connection((args.host, args.port), timeout=10.0)
    file_obj = sock.makefile("r", encoding="utf-8")
    print(f"Connected to tcp://{args.host}:{args.port}")
    viewer = _TkViewer("ldlidar_stop_zone_view")

    try:
        for line in file_obj:
            payload = json.loads(line)
            points = payload.get("points", [])
            rpm = float(payload.get("rpm", 0.0))
            blocked_points = sum(
                1
                for angle_deg, distance_m, _confidence in points
                if _point_in_stop_zone(float(angle_deg), float(distance_m), zone_cfg)
            )
            blocked = blocked_points >= zone_cfg.min_points_to_trigger
            frame = _draw_points(
                points,
                size=args.canvas_size,
                max_distance_m=args.max_distance_m,
                rpm=rpm,
                zone_cfg=zone_cfg,
                blocked=blocked,
                blocked_points=blocked_points,
            )
            viewer.show_bgr(frame)
            if viewer.closed:
                break
    finally:
        try:
            file_obj.close()
        except OSError:
            pass
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
