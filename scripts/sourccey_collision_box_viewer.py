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


def _raw_scan_forward_360(frame, args) -> np.ndarray:
    """Viewer-only full revolution, with no angle, range, or self masking."""
    points = frame.points
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    angle = np.asarray([float(point[0]) for point in points], dtype=np.float64)
    distance = np.asarray([float(point[1]) for point in points], dtype=np.float64)
    confidence = np.asarray([float(point[2]) for point in points], dtype=np.float64)
    valid = (
        np.isfinite(angle)
        & np.isfinite(distance)
        & (distance > 0.0)
        & (confidence >= float(args.min_confidence))
    )
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64)
    delta = np.radians(
        ((angle[valid] - float(args.forward_angle_deg) + 180.0) % 360.0) - 180.0
    )
    forward = distance[valid] * np.cos(delta)
    lateral = distance[valid] * np.sin(delta)
    if bool(args.invert_lateral_axis):
        lateral = -lateral
    return _to_forward_frame(
        np.column_stack([forward, lateral]),
        _forward_offset_deg(args),
    )


def _zero_return_directions_forward_360(frame, args) -> np.ndarray:
    """Unit vectors for rays where the sensor supplied no valid range."""
    points = frame.points
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    angle = np.asarray([float(point[0]) for point in points], dtype=np.float64)
    distance = np.asarray([float(point[1]) for point in points], dtype=np.float64)
    missing = np.isfinite(angle) & (~np.isfinite(distance) | (distance <= 0.0))
    if not np.any(missing):
        return np.empty((0, 2), dtype=np.float64)
    delta = np.radians(
        ((angle[missing] - float(args.forward_angle_deg) + 180.0) % 360.0) - 180.0
    )
    lateral = np.sin(delta)
    if bool(args.invert_lateral_axis):
        lateral = -lateral
    return _to_forward_frame(
        np.column_stack([np.cos(delta), lateral]),
        _forward_offset_deg(args),
    )


class CollisionBoxViewer:
    def __init__(self, root: tk.Tk, feed: DirectLidarFeed, profile: dict, path: str, args) -> None:
        self.root = root
        self.feed = feed
        self.profile = profile
        self.startup_profile = copy.deepcopy(profile)
        self.path = path
        self.args = args
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
        self.canvas.grid(row=0, column=0, rowspan=17, sticky="nsew", padx=(0, 14))

        self.complete_var = tk.BooleanVar(value=bool(profile.get("complete_box", False)))
        self.width_var = tk.DoubleVar()
        self.length_var = tk.DoubleVar()
        self.corner_var = tk.DoubleVar()
        self._load_controls_for_mode()
        ttk.Label(frame, text="Collision envelope", font=("Segoe UI", 14, "bold")).grid(
            row=0, column=1, sticky="nw"
        )
        ttk.Checkbutton(
            frame,
            text="Complete 360° rounded box",
            variable=self.complete_var,
            command=self._completion_changed,
        ).grid(row=1, column=1, sticky="w", pady=(6, 4))
        ttk.Label(frame, text="Width (inches)").grid(row=2, column=1, sticky="sw")
        self.width_entry = ttk.Entry(frame, textvariable=self.width_var, width=12)
        self.width_entry.grid(row=3, column=1, sticky="ew", pady=(2, 4))
        self.width_entry.bind("<Return>", self._dimensions_changed)
        self.width_entry.bind("<FocusOut>", self._dimensions_changed)
        ttk.Scale(
            frame, from_=8.0, to=60.0, variable=self.width_var,
            command=self._dimensions_changed, length=260,
        ).grid(row=4, column=1, sticky="ew")
        self.width_label = ttk.Label(frame)
        self.width_label.grid(row=5, column=1, sticky="nw")
        ttk.Label(frame, text="Length (inches)").grid(row=6, column=1, sticky="sw")
        self.length_entry = ttk.Entry(frame, textvariable=self.length_var, width=12)
        self.length_entry.grid(row=7, column=1, sticky="ew", pady=(2, 4))
        self.length_entry.bind("<Return>", self._dimensions_changed)
        self.length_entry.bind("<FocusOut>", self._dimensions_changed)
        ttk.Scale(
            frame, from_=8.0, to=72.0, variable=self.length_var,
            command=self._dimensions_changed, length=260,
        ).grid(row=8, column=1, sticky="ew")
        self.length_label = ttk.Label(frame)
        self.length_label.grid(row=9, column=1, sticky="nw")
        ttk.Label(frame, text="Side/rear corner radius (inches)").grid(
            row=10, column=1, sticky="sw", pady=(6, 0)
        )
        self.corner_entry = ttk.Entry(frame, textvariable=self.corner_var, width=12)
        self.corner_entry.grid(row=11, column=1, sticky="ew", pady=(2, 4))
        self.corner_entry.bind("<Return>", self._dimensions_changed)
        self.corner_entry.bind("<FocusOut>", self._dimensions_changed)
        ttk.Scale(
            frame, from_=0.0, to=18.0, variable=self.corner_var,
            command=self._dimensions_changed, length=260,
        ).grid(row=12, column=1, sticky="ew")
        self.corner_label = ttk.Label(frame)
        self.corner_label.grid(row=13, column=1, sticky="nw")
        buttons = ttk.Frame(frame)
        buttons.grid(row=14, column=1, sticky="new", pady=(16, 4))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        ttk.Button(buttons, text="Reset to default", command=self.reset).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(buttons, text="Save collision box", command=self.save).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )
        self.status = ttk.Label(frame, text="Waiting for LiDAR…", wraplength=260)
        self.status.grid(row=15, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self._dimensions_changed()
        self.root.after(50, self.refresh)

    def _completion_changed(self) -> None:
        completing = bool(self.complete_var.get())
        self.profile["complete_box"] = completing
        self._load_controls_for_mode()
        self._dimensions_changed()
        mode = "completed 360° box" if completing else "learned front/side envelope"
        self.status.configure(text=f"Displaying {mode}; save to make it active in exploration.")

    def _load_controls_for_mode(self) -> None:
        learned_width_m, learned_length_m = collision_box_dimensions(
            {**self.profile, "complete_box": False}
        )
        if bool(self.profile.get("complete_box", False)):
            width_m = float(self.profile.get("completed_width_m", learned_width_m))
            length_m = float(
                self.profile.get(
                    "completed_length_m",
                    max(learned_width_m, learned_length_m),
                )
            )
        else:
            width_m, length_m = learned_width_m, learned_length_m
        default_corner_m = 0.20 * min(width_m, length_m)
        self.width_var.set(width_m / _METRES_PER_INCH)
        self.length_var.set(length_m / _METRES_PER_INCH)
        self.corner_var.set(
            float(self.profile.get("corner_radius_m", default_corner_m))
            / _METRES_PER_INCH
        )

    def _dimensions_changed(self, _value=None) -> None:
        try:
            width_in = float(self.width_var.get())
            length_in = float(self.length_var.get())
            corner_in = float(self.corner_var.get())
        except (tk.TclError, ValueError):
            self.status.configure(text="Width, length, and corner radius must be valid numbers.")
            return
        if width_in <= 0.0 or length_in <= 0.0 or corner_in < 0.0:
            self.status.configure(
                text="Width/length must be positive and corner radius cannot be negative."
            )
            return
        max_corner_in = 0.5 * min(width_in, length_in)
        corner_in = min(corner_in, max_corner_in)
        self.corner_var.set(corner_in)
        width_m = width_in * _METRES_PER_INCH
        length_m = length_in * _METRES_PER_INCH
        completing = bool(self.complete_var.get())
        self.profile["complete_box"] = completing
        if completing:
            self.profile["completed_width_m"] = width_m
            self.profile["completed_length_m"] = length_m
        else:
            self.profile["width_m"] = width_m
            self.profile["length_m"] = length_m
        self.profile["corner_radius_m"] = corner_in * _METRES_PER_INCH
        self.width_label.configure(
            text=f"{width_in:.2f} in  /  {width_m:.3f} m"
        )
        self.length_label.configure(
            text=f"{length_in:.2f} in  /  {length_m:.3f} m"
        )
        self.corner_label.configure(
            text=f"{corner_in:.2f} in  /  {self.profile['corner_radius_m']:.3f} m"
        )

    def _xy(self, point: np.ndarray) -> tuple[float, float]:
        centre = self.canvas_size / 2.0
        # Physical forward is up on screen; physical left is screen-left.
        return centre - float(point[1]) * self.scale_px_m, centre - float(point[0]) * self.scale_px_m

    def refresh(self) -> None:
        _frame_id, frame = self.feed.latest()
        points = np.empty((0, 2), dtype=np.float64)
        zero_directions = np.empty((0, 2), dtype=np.float64)
        hit = None
        if frame is not None:
            points = _raw_scan_forward_360(frame, self.args)
            zero_directions = _zero_return_directions_forward_360(frame, self.args)
            if len(points):
                hit = collision_box_violation(points, self.profile)

        ranges = effective_ranges(self.profile)
        finite_ranges = ranges[np.isfinite(ranges)]
        scan_radius_m = float(np.max(np.linalg.norm(points, axis=1))) if len(points) else 0.0
        box_radius_m = float(np.max(finite_ranges)) if len(finite_ranges) else 0.0
        display_radius_m = max(0.75, scan_radius_m, box_radius_m * 1.15)
        self.scale_px_m = (self.canvas_size / 2.0 - 24.0) / display_radius_m
        rear_returns = int(np.count_nonzero(points[:, 0] < 0.0)) if len(points) else 0

        self.canvas.delete("all")
        centre = self.canvas_size / 2.0
        for radius_m in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            if radius_m > display_radius_m:
                continue
            r = radius_m * self.scale_px_m
            self.canvas.create_oval(centre-r, centre-r, centre+r, centre+r, outline="#26333d")
        self.canvas.create_text(
            10,
            10,
            anchor="nw",
            fill="#a8b3bd",
            text=(
                f"RAW 360° VIEW  |  {len(points)} returns  |  "
                f"rear {rear_returns}  |  no-range {len(zero_directions)}  |  "
                f"radius {display_radius_m:.2f} m"
            ),
        )
        self.canvas.create_line(centre, centre, centre, centre-45, fill="#ffffff", width=3, arrow="last")

        violating = np.zeros(len(points), dtype=bool) if hit is None else hit[0]
        for idx, point in enumerate(points):
            x, y = self._xy(point)
            color = "#ff3b30" if violating[idx] else "#7ee787"
            self.canvas.create_oval(x-1.5, y-1.5, x+1.5, y+1.5, fill=color, outline="")
        # The LD19 reports a zero distance when a surface is too close or a ray
        # otherwise has no usable range. Draw those bearings on a small orange
        # diagnostic ring. They are deliberately excluded from collision logic:
        # an invalid range does not tell us the obstacle's actual distance.
        for direction in zero_directions:
            x, y = self._xy(direction * 0.07)
            self.canvas.create_oval(x-2.0, y-2.0, x+2.0, y+2.0, fill="#ff9f0a", outline="")

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
        self.complete_var.set(bool(self.profile.get("complete_box", False)))
        self._load_controls_for_mode()
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
