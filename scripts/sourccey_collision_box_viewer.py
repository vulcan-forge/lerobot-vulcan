"""Live LiDAR collision-box viewer and width/length editor."""

from __future__ import annotations

import argparse
import copy
import math
import tkinter as tk
from tkinter import ttk

import numpy as np
from ldlidar_direct_snapshot_client import DirectLidarFeed
from sourccey_collision_box import (
    DEFAULT_COLLISION_BOX_PATH,
    collision_box_dimensions,
    collision_box_violation,
    effective_ranges,
    load_collision_box,
    save_collision_box,
)
from sourccey_spin_map import _scan_local

_METRES_PER_INCH = 0.0254


def _forward_offset_deg(args) -> float:
    delta = ((270.0 - float(args.forward_angle_deg) + 180.0) % 360.0) - 180.0
    lateral = math.sin(math.radians(delta))
    if bool(args.invert_lateral_axis):
        lateral = -lateral
    return math.degrees(math.atan2(lateral, math.cos(math.radians(delta))))


def _to_forward_frame(points: np.ndarray, offset_deg: float) -> np.ndarray:
    c = math.cos(math.radians(-offset_deg))
    s = math.sin(math.radians(-offset_deg))
    return np.column_stack([
        points[:, 0] * c - points[:, 1] * s,
        points[:, 0] * s + points[:, 1] * c,
    ])


class CollisionBoxViewer:
    def __init__(self, root: tk.Tk, feed: DirectLidarFeed, profile: dict, path: str, args) -> None:
        self.root = root
        self.feed = feed
        self.profile = profile
        self.startup_profile = copy.deepcopy(profile)
        self.path = path
        self.args = args
        self.offset_deg = _forward_offset_deg(args)
        self.canvas_size = 720
        self.scale_px_m = 250.0

        root.title("Sourccey LiDAR Collision Box")
        root.protocol("WM_DELETE_WINDOW", self.close)
        frame = ttk.Frame(root, padding=10)
        frame.grid(sticky="nsew")
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            frame, width=self.canvas_size, height=self.canvas_size,
            background="#101418", highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, rowspan=11, sticky="nsew", padx=(0, 14))

        width_m, length_m = collision_box_dimensions(profile)
        self.width_var = tk.DoubleVar(value=width_m / _METRES_PER_INCH)
        self.length_var = tk.DoubleVar(value=length_m / _METRES_PER_INCH)
        ttk.Label(frame, text="Collision envelope", font=("Segoe UI", 14, "bold")).grid(
            row=0, column=1, sticky="nw"
        )
        ttk.Label(frame, text="Width (inches)").grid(row=1, column=1, sticky="sw")
        self.width_entry = ttk.Entry(frame, textvariable=self.width_var, width=12)
        self.width_entry.grid(row=2, column=1, sticky="ew", pady=(2, 4))
        self.width_entry.bind("<Return>", self._dimensions_changed)
        self.width_entry.bind("<FocusOut>", self._dimensions_changed)
        ttk.Scale(
            frame, from_=8.0, to=60.0, variable=self.width_var,
            command=self._dimensions_changed, length=260,
        ).grid(row=3, column=1, sticky="ew")
        self.width_label = ttk.Label(frame)
        self.width_label.grid(row=4, column=1, sticky="nw")
        ttk.Label(frame, text="Length (inches)").grid(row=5, column=1, sticky="sw")
        self.length_entry = ttk.Entry(frame, textvariable=self.length_var, width=12)
        self.length_entry.grid(row=6, column=1, sticky="ew", pady=(2, 4))
        self.length_entry.bind("<Return>", self._dimensions_changed)
        self.length_entry.bind("<FocusOut>", self._dimensions_changed)
        ttk.Scale(
            frame, from_=8.0, to=72.0, variable=self.length_var,
            command=self._dimensions_changed, length=260,
        ).grid(row=7, column=1, sticky="ew")
        self.length_label = ttk.Label(frame)
        self.length_label.grid(row=8, column=1, sticky="nw")
        buttons = ttk.Frame(frame)
        buttons.grid(row=9, column=1, sticky="new", pady=(16, 4))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        ttk.Button(buttons, text="Reset to default", command=self.reset).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(buttons, text="Save collision box", command=self.save).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )
        self.status = ttk.Label(frame, text="Waiting for LiDAR…", wraplength=260)
        self.status.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self._dimensions_changed()
        self.root.after(50, self.refresh)

    def _dimensions_changed(self, _value=None) -> None:
        try:
            width_in = float(self.width_var.get())
            length_in = float(self.length_var.get())
        except (tk.TclError, ValueError):
            self.status.configure(text="Width and length must be valid numbers.")
            return
        if width_in <= 0.0 or length_in <= 0.0:
            self.status.configure(text="Width and length must be greater than zero.")
            return
        self.profile["width_m"] = width_in * _METRES_PER_INCH
        self.profile["length_m"] = length_in * _METRES_PER_INCH
        self.width_label.configure(
            text=f"{width_in:.2f} in  /  {self.profile['width_m']:.3f} m"
        )
        self.length_label.configure(
            text=f"{length_in:.2f} in  /  {self.profile['length_m']:.3f} m"
        )

    def _xy(self, point: np.ndarray) -> tuple[float, float]:
        centre = self.canvas_size / 2.0
        # Physical forward is up on screen; physical left is screen-left.
        return centre - float(point[1]) * self.scale_px_m, centre - float(point[0]) * self.scale_px_m

    def refresh(self) -> None:
        self.canvas.delete("all")
        centre = self.canvas_size / 2.0
        for radius_m in (0.25, 0.5, 1.0):
            r = radius_m * self.scale_px_m
            self.canvas.create_oval(centre-r, centre-r, centre+r, centre+r, outline="#26333d")
        self.canvas.create_line(centre, centre, centre, centre-45, fill="#ffffff", width=3, arrow="last")

        _frame_id, frame = self.feed.latest()
        points = np.empty((0, 2), dtype=np.float64)
        hit = None
        if frame is not None:
            local = _scan_local(frame, self.args)
            if len(local):
                points = _to_forward_frame(local, self.offset_deg)
                hit = collision_box_violation(points, self.profile)

        violating = np.zeros(len(points), dtype=bool) if hit is None else hit[0]
        for idx, point in enumerate(points):
            x, y = self._xy(point)
            color = "#ff3b30" if violating[idx] else "#7ee787"
            self.canvas.create_oval(x-1.5, y-1.5, x+1.5, y+1.5, fill=color, outline="")

        ranges = effective_ranges(self.profile)
        bin_size = float(self.profile["bin_size_deg"])
        angles = np.radians(-180.0 + (np.arange(len(ranges)) + 0.5) * bin_size)
        boundary = np.column_stack([ranges * np.cos(angles), ranges * np.sin(angles)])
        valid = np.isfinite(ranges)
        hit_bins: set[int] = set()
        if hit is not None:
            hit_angles = np.degrees(np.arctan2(points[violating, 1], points[violating, 0]))
            hit_bins = set(
                (np.floor((hit_angles + 180.0) / bin_size).astype(int) % len(ranges)).tolist()
            )
        for idx in range(len(boundary)):
            nxt = (idx + 1) % len(boundary)
            if not (valid[idx] and valid[nxt]):
                continue
            color = "#ff453a" if idx in hit_bins or nxt in hit_bins else "#00d7ff"
            self.canvas.create_line(*self._xy(boundary[idx]), *self._xy(boundary[nxt]), fill=color, width=3)

        if hit is None:
            self.status.configure(text="CLEAR — no LiDAR return is inside the collision envelope")
        else:
            _mask, sector, angle, distance, limit = hit
            self.status.configure(
                text=f"COLLISION: {sector}, {angle:+.0f}°, return {distance:.2f} m, boundary {limit:.2f} m"
            )
        self.root.after(50, self.refresh)

    def save(self) -> None:
        output = save_collision_box(self.profile, self.path)
        self.status.configure(text=f"Saved collision envelope to {output}")

    def reset(self) -> None:
        self.profile = copy.deepcopy(self.startup_profile)
        width_m, length_m = collision_box_dimensions(self.profile)
        self.width_var.set(width_m / _METRES_PER_INCH)
        self.length_var.set(length_m / _METRES_PER_INCH)
        self._dimensions_changed()
        self.status.configure(text="Reset to the collision box loaded when this window opened")

    def close(self) -> None:
        self.feed.stop()
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-ip", required=True)
    parser.add_argument("--lidar-host", default=None)
    parser.add_argument("--lidar-port", type=int, default=8765)
    parser.add_argument("--collision-box-file", default=str(DEFAULT_COLLISION_BOX_PATH))
    parser.add_argument("--forward-angle-deg", type=float, default=90.0)
    parser.add_argument("--valid-angle-half-width-deg", type=float, default=180.0)
    parser.add_argument("--invert-lateral-axis", action="store_true", default=True)
    parser.add_argument("--max-distance-m", type=float, default=8.0)
    parser.add_argument("--min-range-m", type=float, default=0.03)
    parser.add_argument("--min-confidence", type=int, default=0)
    args = parser.parse_args()

    profile = load_collision_box(args.collision_box_file)
    if profile is None:
        parser.error(f"collision calibration not found: {args.collision_box_file}")
    feed = DirectLidarFeed(args.lidar_host or args.remote_ip, int(args.lidar_port))
    feed.start()
    root = tk.Tk()
    try:
        CollisionBoxViewer(root, feed, profile, args.collision_box_file, args)
        root.mainloop()
    except BaseException:
        feed.stop()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
