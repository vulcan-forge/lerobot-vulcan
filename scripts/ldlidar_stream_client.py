from __future__ import annotations

import argparse
import json
import math
import socket

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk


def _draw_points(points: list[list[float]], *, size: int, max_distance_m: float, rpm: float) -> np.ndarray:
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

    cv2.putText(
        canvas,
        f"LDLidar remote view  points={len(points)}  rpm={rpm:.1f}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"max_distance={max_distance_m:.1f}m",
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
    parser = argparse.ArgumentParser(description="Remote viewer for the quick LD LiDAR host streamer.")
    parser.add_argument("--host", required=True, help="Host IP running ldlidar_stream_host.py.")
    parser.add_argument("--port", type=int, default=8765, help="TCP port exposed by the host.")
    parser.add_argument("--max-distance-m", type=float, default=8.0, help="Viewer distance scale.")
    parser.add_argument("--canvas-size", type=int, default=900, help="Viewer size in pixels.")
    args = parser.parse_args()

    sock = socket.create_connection((args.host, args.port), timeout=10.0)
    file_obj = sock.makefile("r", encoding="utf-8")
    print(f"Connected to tcp://{args.host}:{args.port}")
    viewer = _TkViewer("ldlidar_remote_view")

    try:
        for line in file_obj:
            payload = json.loads(line)
            points = payload.get("points", [])
            rpm = float(payload.get("rpm", 0.0))
            frame = _draw_points(
                points,
                size=args.canvas_size,
                max_distance_m=args.max_distance_m,
                rpm=rpm,
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
