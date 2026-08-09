from __future__ import annotations

import argparse
import base64
import json
import math
import socket
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import zmq

try:
    from PIL import ImageTk
    import tkinter as tk
except Exception:
    ImageTk = None
    tk = None

from .defaults import DEFAULT_LIDAR_FORWARD_ANGLE_DEG
from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient


def _normalize_angle_deg(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def _wrap_angle_rad(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _scan_to_local_points(
    points: list[list[float]],
    *,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    invert_lateral_axis: bool,
    max_distance_m: float,
    min_distance_m: float,
    min_confidence: int,
) -> np.ndarray:
    out: list[tuple[float, float]] = []
    for raw_point in points:
        if len(raw_point) < 3:
            continue
        angle_deg, distance_m, confidence = float(raw_point[0]), float(raw_point[1]), int(raw_point[2])
        if confidence < int(min_confidence):
            continue
        if not math.isfinite(distance_m) or distance_m < float(min_distance_m) or distance_m > float(max_distance_m):
            continue
        delta_deg = _normalize_angle_deg(angle_deg - float(forward_angle_deg))
        if abs(delta_deg) > float(valid_angle_half_width_deg):
            continue
        theta_rad = math.radians(delta_deg)
        forward_m = distance_m * math.cos(theta_rad)
        lateral_m = distance_m * math.sin(theta_rad)
        if invert_lateral_axis:
            lateral_m = -lateral_m
        out.append((forward_m, lateral_m))
    if not out:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def _voxel_downsample(points_xy: np.ndarray, voxel_m: float) -> np.ndarray:
    if points_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    voxel = max(float(voxel_m), 1e-3)
    quantized = np.round(points_xy / voxel).astype(np.int32, copy=False)
    _, keep_indices = np.unique(quantized, axis=0, return_index=True)
    return points_xy[np.sort(keep_indices)]


def _build_rotation_signature(
    frames: list["_ScanFrame"],
    *,
    min_distance_m: float,
    max_distance_m: float,
    min_confidence: int,
    bin_size_deg: float,
) -> np.ndarray:
    bin_size_deg = max(float(bin_size_deg), 0.5)
    num_bins = max(int(round(360.0 / bin_size_deg)), 36)
    bins: list[list[float]] = [[] for _ in range(num_bins)]
    for frame in frames:
        for raw_point in frame.points:
            if len(raw_point) < 3:
                continue
            angle_deg, distance_m, confidence = float(raw_point[0]), float(raw_point[1]), int(raw_point[2])
            if confidence < int(min_confidence):
                continue
            if not math.isfinite(distance_m):
                continue
            if distance_m < float(min_distance_m) or distance_m > float(max_distance_m):
                continue
            angle_360 = angle_deg % 360.0
            bin_index = int(round(angle_360 / bin_size_deg)) % num_bins
            bins[bin_index].append(distance_m)
    signature = np.full((num_bins,), np.nan, dtype=np.float32)
    for idx, distances in enumerate(bins):
        if distances:
            signature[idx] = float(np.median(np.asarray(distances, dtype=np.float32)))
    return signature


def _estimate_signature_rotation_deg(
    reference_signature: np.ndarray,
    candidate_signature: np.ndarray,
    *,
    bin_size_deg: float,
    center_deg: float | None = None,
    search_half_window_deg: float | None = None,
) -> tuple[float, float, float]:
    if reference_signature.size == 0 or candidate_signature.size == 0:
        return 0.0, float("inf"), 0.0
    if reference_signature.shape != candidate_signature.shape:
        raise ValueError("Rotation signatures must have the same shape.")

    num_bins = int(reference_signature.shape[0])
    ref_valid = np.isfinite(reference_signature)
    cand_valid = np.isfinite(candidate_signature)
    valid_ref_count = int(np.count_nonzero(ref_valid))
    if valid_ref_count <= 0:
        return 0.0, float("inf"), 0.0

    best_shift = 0
    best_score = float("inf")
    best_overlap_ratio = 0.0
    allowed_min_deg = None if center_deg is None or search_half_window_deg is None else float(center_deg) - float(search_half_window_deg)
    allowed_max_deg = None if center_deg is None or search_half_window_deg is None else float(center_deg) + float(search_half_window_deg)

    for shift in range(num_bins):
        shift_deg = shift * float(bin_size_deg)
        if allowed_min_deg is not None and allowed_max_deg is not None:
            shifted_center_error_deg = abs(_normalize_angle_deg(shift_deg - float(center_deg)))
            if shifted_center_error_deg > float(search_half_window_deg):
                continue
        shifted = np.roll(candidate_signature, shift)
        shifted_valid = np.roll(cand_valid, shift)
        valid = ref_valid & shifted_valid
        overlap_count = int(np.count_nonzero(valid))
        if overlap_count < max(18, num_bins // 10):
            continue
        diffs = np.abs(reference_signature[valid] - shifted[valid])
        overlap_ratio = overlap_count / max(valid_ref_count, 1)
        score = float(np.mean(diffs)) + (0.25 * max(0.0, 0.45 - overlap_ratio))
        if score < best_score:
            best_score = score
            best_shift = shift
            best_overlap_ratio = overlap_ratio
    return float(best_shift * float(bin_size_deg)), float(best_score), float(best_overlap_ratio)


def _downsample_points(points_xy: np.ndarray, max_points: int) -> np.ndarray:
    if points_xy.size == 0 or len(points_xy) <= max_points:
        return points_xy
    indices = np.linspace(0, len(points_xy) - 1, max_points, dtype=np.int32)
    return points_xy[indices]


def _transform_snapshot_to_world(
    local_points_xy: np.ndarray,
    *,
    base_x_m: float,
    base_y_m: float,
    base_yaw_rad: float,
    lidar_mount_x_m: float,
    lidar_mount_y_m: float,
) -> np.ndarray:
    if local_points_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    cos_yaw = math.cos(float(base_yaw_rad))
    sin_yaw = math.sin(float(base_yaw_rad))
    sensor_x = float(base_x_m) + (float(lidar_mount_x_m) * cos_yaw) - (float(lidar_mount_y_m) * sin_yaw)
    sensor_y = float(base_y_m) + (float(lidar_mount_x_m) * sin_yaw) + (float(lidar_mount_y_m) * cos_yaw)
    rotation = np.asarray(((cos_yaw, -sin_yaw), (sin_yaw, cos_yaw)), dtype=np.float32)
    world = local_points_xy @ rotation.T
    world[:, 0] += sensor_x
    world[:, 1] += sensor_y
    return world


def _draw_arrow(
    canvas: np.ndarray,
    *,
    origin_px: tuple[int, int],
    yaw_rad: float,
    length_px: int,
    color: tuple[int, int, int],
) -> None:
    dx = int(round(math.cos(float(yaw_rad)) * length_px))
    dy = int(round(math.sin(float(yaw_rad)) * length_px))
    end_point = (origin_px[0] + dx, origin_px[1] - dy)
    cv2.arrowedLine(canvas, origin_px, end_point, color, 2, cv2.LINE_AA, tipLength=0.25)


def _render_overlay(
    first_world_xy: np.ndarray,
    second_world_xy: np.ndarray,
    *,
    second_yaw_rad: float,
    image_size_px: int,
) -> np.ndarray:
    all_points = []
    if first_world_xy.size != 0:
        all_points.append(first_world_xy)
    if second_world_xy.size != 0:
        all_points.append(second_world_xy)
    if not all_points:
        return np.zeros((image_size_px, image_size_px, 3), dtype=np.uint8)

    world_xy = np.vstack(all_points).astype(np.float32, copy=False)
    min_x = float(np.min(world_xy[:, 0])) - 0.4
    max_x = float(np.max(world_xy[:, 0])) + 0.4
    min_y = float(np.min(world_xy[:, 1])) - 0.4
    max_y = float(np.max(world_xy[:, 1])) + 0.4

    span_x = max(max_x - min_x, 0.5)
    span_y = max(max_y - min_y, 0.5)
    scale = min((image_size_px - 60) / span_x, (image_size_px - 60) / span_y)
    center_x = image_size_px // 2
    center_y = image_size_px // 2
    world_center_x = (min_x + max_x) * 0.5
    world_center_y = (min_y + max_y) * 0.5

    def to_px(x_m: float, y_m: float) -> tuple[int, int]:
        x_px = int(round(center_x + (x_m - world_center_x) * scale))
        y_px = int(round(center_y - (y_m - world_center_y) * scale))
        return x_px, y_px

    canvas = np.zeros((image_size_px, image_size_px, 3), dtype=np.uint8)

    grid_step_px = max(int(round(scale * 0.25)), 24)
    for x in range(center_x % grid_step_px, image_size_px, grid_step_px):
        cv2.line(canvas, (x, 0), (x, image_size_px), (32, 32, 32), 1, cv2.LINE_AA)
    for y in range(center_y % grid_step_px, image_size_px, grid_step_px):
        cv2.line(canvas, (0, y), (image_size_px, y), (32, 32, 32), 1, cv2.LINE_AA)

    for point in first_world_xy:
        cv2.circle(canvas, to_px(float(point[0]), float(point[1])), 2, (0, 220, 255), -1)
    for point in second_world_xy:
        cv2.circle(canvas, to_px(float(point[0]), float(point[1])), 2, (255, 180, 0), -1)

    robot_origin_px = to_px(0.0, 0.0)
    cv2.circle(canvas, robot_origin_px, 9, (255, 255, 255), 2, cv2.LINE_AA)
    _draw_arrow(canvas, origin_px=robot_origin_px, yaw_rad=0.0, length_px=42, color=(0, 220, 255))
    _draw_arrow(canvas, origin_px=robot_origin_px, yaw_rad=second_yaw_rad, length_px=42, color=(255, 180, 0))

    cv2.putText(
        canvas,
        "cyan = snapshot A    orange = snapshot B",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _render_snapshot_view(
    local_points_xy: np.ndarray,
    *,
    title: str,
    image_size_px: int,
    max_distance_m: float,
    forward_angle_deg: float,
) -> np.ndarray:
    canvas = np.zeros((image_size_px, image_size_px, 3), dtype=np.uint8)
    center = image_size_px // 2
    radius_px = center - 24

    for ring_fraction in (0.25, 0.5, 0.75, 1.0):
        cv2.circle(canvas, (center, center), int(radius_px * ring_fraction), (40, 40, 40), 1)
    cv2.line(canvas, (center, 16), (center, image_size_px - 16), (30, 30, 30), 1)
    cv2.line(canvas, (16, center), (image_size_px - 16, center), (30, 30, 30), 1)
    cv2.circle(canvas, (center, center), 4, (0, 180, 255), -1)

    def to_xy(forward_m: float, lateral_m: float) -> tuple[int, int]:
        distance_m = math.hypot(float(forward_m), float(lateral_m))
        norm = min(max(distance_m, 0.0), max_distance_m) / max(max_distance_m, 1e-6)
        px_radius = int(norm * radius_px)
        angle_deg = float(forward_angle_deg) + math.degrees(math.atan2(float(lateral_m), float(forward_m)))
        theta = math.radians(angle_deg - 90.0)
        x = int(center + px_radius * math.cos(theta))
        y = int(center + px_radius * math.sin(theta))
        return x, y

    for point in local_points_xy:
        x, y = to_xy(float(point[0]), float(point[1]))
        cv2.circle(canvas, (x, y), 2, (0, 220, 255), -1)

    cv2.putText(
        canvas,
        title,
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _scan_match_score(reference_world_xy: np.ndarray, candidate_world_xy: np.ndarray) -> float:
    if reference_world_xy.size == 0 or candidate_world_xy.size == 0:
        return float("inf")
    ref = _downsample_points(reference_world_xy.astype(np.float32, copy=False), 240)
    cand = _downsample_points(candidate_world_xy.astype(np.float32, copy=False), 240)
    deltas = cand[:, None, :] - ref[None, :, :]
    distances_sq = np.sum(deltas * deltas, axis=2)
    cand_to_ref = np.min(distances_sq, axis=1)
    ref_to_cand = np.min(distances_sq, axis=0)
    score = float(
        0.5
        * (
            np.mean(np.clip(cand_to_ref, 0.0, 0.25))
            + np.mean(np.clip(ref_to_cand, 0.0, 0.25))
        )
    )
    return score


def _local_dissimilarity_score(reference_local_xy: np.ndarray, candidate_local_xy: np.ndarray) -> float:
    if reference_local_xy.size == 0 or candidate_local_xy.size == 0:
        return 0.0
    return _scan_match_score(reference_local_xy, candidate_local_xy)


def _estimate_alignment_yaw_rad(
    *,
    reference_local_xy: np.ndarray,
    current_local_xy: np.ndarray,
    lidar_mount_x_m: float,
    lidar_mount_y_m: float,
    center_yaw_rad: float,
    search_half_window_deg: float,
    search_step_deg: float,
) -> tuple[float, float]:
    if reference_local_xy.size == 0 or current_local_xy.size == 0:
        return float(center_yaw_rad), float("inf")

    reference_world_xy = _transform_snapshot_to_world(
        reference_local_xy,
        base_x_m=0.0,
        base_y_m=0.0,
        base_yaw_rad=0.0,
        lidar_mount_x_m=lidar_mount_x_m,
        lidar_mount_y_m=lidar_mount_y_m,
    )

    # First do a full 360-degree coarse search so we don't get trapped near a bad
    # commanded-turn estimate. Then refine locally around the best coarse yaw.
    global_step_deg = max(float(search_step_deg) * 3.0, 6.0)
    best_yaw = 0.0
    best_score = float("inf")
    coarse_count = max(int(round(360.0 / global_step_deg)), 24)
    for offset_deg in np.linspace(-180.0, 180.0, coarse_count + 1):
        candidate_yaw = _wrap_angle_rad(math.radians(float(offset_deg)))
        candidate_world_xy = _transform_snapshot_to_world(
            current_local_xy,
            base_x_m=0.0,
            base_y_m=0.0,
            base_yaw_rad=candidate_yaw,
            lidar_mount_x_m=lidar_mount_x_m,
            lidar_mount_y_m=lidar_mount_y_m,
        )
        score = _scan_match_score(reference_world_xy, candidate_world_xy)
        if score < best_score:
            best_score = score
            best_yaw = candidate_yaw

    refine_half_window_deg = max(float(search_half_window_deg), global_step_deg * 1.5, 12.0)
    step_count = max(
        int(round((2.0 * float(refine_half_window_deg)) / max(float(search_step_deg), 0.5))),
        1,
    )
    for offset_deg in np.linspace(-float(refine_half_window_deg), float(refine_half_window_deg), step_count + 1):
        candidate_yaw = _wrap_angle_rad(float(best_yaw) + math.radians(float(offset_deg)))
        candidate_world_xy = _transform_snapshot_to_world(
            current_local_xy,
            base_x_m=0.0,
            base_y_m=0.0,
            base_yaw_rad=candidate_yaw,
            lidar_mount_x_m=lidar_mount_x_m,
            lidar_mount_y_m=lidar_mount_y_m,
        )
        score = _scan_match_score(reference_world_xy, candidate_world_xy)
        center_penalty = abs(_wrap_angle_rad(candidate_yaw - float(center_yaw_rad))) * 1e-4
        adjusted_score = score + center_penalty
        if adjusted_score < (best_score + 1e-12):
            best_score = score
            best_yaw = candidate_yaw
    return best_yaw, best_score


def _yaw_progress_rad(lock_yaw_rad: float, *, direction_sign: float) -> float:
    progress_rad = float(lock_yaw_rad) if float(direction_sign) >= 0.0 else -float(lock_yaw_rad)
    progress_rad = math.fmod(progress_rad, 2.0 * math.pi)
    if progress_rad < 0.0:
        progress_rad += 2.0 * math.pi
    return progress_rad


def _rotation_progress_deg(raw_rotation_deg: float, *, direction_sign: float) -> float:
    # `raw_rotation_deg` is the shift that aligns the CURRENT scan back onto the
    # REFERENCE scan, so it runs OPPOSITE to the robot's motion: a small left
    # (CCW) turn shows up as a raw shift near 360 (e.g. 354 == -6). Convert it
    # into "how far have we turned in the commanded direction", which then climbs
    # cleanly 0 -> 180 as the robot rotates, for either turn direction.
    raw = float(raw_rotation_deg) % 360.0
    if float(direction_sign) >= 0.0:  # left / CCW
        return (360.0 - raw) % 360.0
    return raw  # right / CW


def _turned_to_raw_deg(turned_deg: float, *, direction_sign: float) -> float:
    """Inverse of `_rotation_progress_deg`: degrees turned -> matcher raw shift."""
    turned = float(turned_deg) % 360.0
    if float(direction_sign) >= 0.0:  # left / CCW
        return (360.0 - turned) % 360.0
    return turned  # right / CW


@dataclass(slots=True)
class _ScanFrame:
    ts: float
    rpm: float
    points: list[list[float]]


class _LidarFeed:
    def __init__(self, host: str, port: int, *, history_size: int = 64) -> None:
        self.host = host
        self.port = int(port)
        self._history: deque[_ScanFrame] = deque(maxlen=max(history_size, 8))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = False
        self.last_error: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="two_pose_lidar_feed", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def capture_snapshot(self, *, scan_count: int, timeout_s: float) -> list[_ScanFrame]:
        deadline = time.time() + max(float(timeout_s), 0.5)
        while time.time() < deadline:
            with self._lock:
                frames = list(self._history)[-max(int(scan_count), 1) :]
            if len(frames) >= max(int(scan_count), 1):
                return frames
            time.sleep(0.05)
        raise TimeoutError("Timed out waiting for enough LiDAR scans.")

    def latest_frames(self, *, scan_count: int = 1) -> list[_ScanFrame]:
        with self._lock:
            return list(self._history)[-max(int(scan_count), 1) :]

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with socket.create_connection((self.host, self.port), timeout=5.0) as sock:
                    file_obj = sock.makefile("r", encoding="utf-8")
                    with self._lock:
                        self.connected = True
                        self.last_error = None
                    for line in file_obj:
                        if self._stop.is_set():
                            break
                        payload = json.loads(line)
                        frame = _ScanFrame(
                            ts=float(payload.get("ts", time.time())),
                            rpm=float(payload.get("rpm", 0.0)),
                            points=list(payload.get("points", [])),
                        )
                        with self._lock:
                            self._history.append(frame)
                            self.connected = True
                            self.last_error = None
            except Exception as exc:
                with self._lock:
                    self.connected = False
                    self.last_error = str(exc)
                time.sleep(0.5)


class _TkPreview:
    def __init__(self, *, title: str) -> None:
        if tk is None or ImageTk is None:
            raise RuntimeError("Tk preview is unavailable in this environment.")
        self.root = tk.Tk()
        self.root.title(title)
        self.image_label = tk.Label(self.root)
        self.image_label.pack()
        self.status_label = tk.Label(self.root, anchor="w", justify="left", font=("Consolas", 11))
        self.status_label.pack(fill="x", padx=8, pady=8)
        self._photo: ImageTk.PhotoImage | None = None
        self._closed = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def show(self, frame_bgr: np.ndarray, *, status_lines: list[str] | None = None) -> None:
        if self._closed:
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(image=image)
        self.image_label.configure(image=self._photo)
        self.status_label.configure(text="\n".join(status_lines or []))
        self.root.update_idletasks()
        self.root.update()

    def wait_until_closed(self) -> None:
        if self._closed:
            return
        try:
            self.root.mainloop()
        except tk.TclError:
            self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.root.quit()
            self.root.destroy()
        except tk.TclError:
            pass


class _ImuTurnTracker:
    def __init__(self, endpoint: str, *, invert_yaw_sign: bool) -> None:
        self.endpoint = endpoint
        self.invert_yaw_sign = bool(invert_yaw_sign)
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._tracking = False
        self._prev_capture_ns: int | None = None
        self._accum_abs_turn_rad = 0.0
        self._accum_signed_turn_rad = 0.0
        self._packets_with_imu = 0
        self.last_error: str | None = None

    def start(self) -> None:
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.connect(self.endpoint)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._thread = threading.Thread(target=self._run, name="two_pose_imu_tracker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._socket is not None:
            self._socket.close(0)
        if self._context is not None:
            self._context.term()

    def begin(self) -> None:
        with self._lock:
            self._tracking = True
            self._prev_capture_ns = None
            self._accum_abs_turn_rad = 0.0
            self._accum_signed_turn_rad = 0.0

    def end(self) -> tuple[float, float, int]:
        with self._lock:
            self._tracking = False
            return self._accum_abs_turn_rad, self._accum_signed_turn_rad, self._packets_with_imu

    def progress(self) -> tuple[float, float, int]:
        with self._lock:
            return self._accum_abs_turn_rad, self._accum_signed_turn_rad, self._packets_with_imu

    def _run(self) -> None:
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                if not self._socket.poll(100):
                    continue
                payload = self._socket.recv()
                message = json.loads(payload.decode("utf-8"))
                imu_samples = message.get("imu_samples", [])
                if not isinstance(imu_samples, list) or not imu_samples:
                    continue
                with self._lock:
                    self._packets_with_imu += 1
                    for sample in imu_samples:
                        if not isinstance(sample, dict):
                            continue
                        capture_ns = int(sample.get("capture_monotonic_ns", 0))
                        gz_rad_s = float(sample.get("gz", 0.0))
                        if self.invert_yaw_sign:
                            gz_rad_s = -gz_rad_s
                        if not self._tracking:
                            self._prev_capture_ns = capture_ns
                            continue
                        if self._prev_capture_ns is None or capture_ns <= self._prev_capture_ns:
                            self._prev_capture_ns = capture_ns
                            continue
                        dt_s = (capture_ns - self._prev_capture_ns) / 1_000_000_000.0
                        self._prev_capture_ns = capture_ns
                        self._accum_signed_turn_rad += gz_rad_s * dt_s
                        self._accum_abs_turn_rad += abs(gz_rad_s) * dt_s
                    self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)
                time.sleep(0.05)


class _ObservedTurnTracker:
    def __init__(self, *, angular_scale_rad_per_unit: float = 1.0) -> None:
        self.angular_scale_rad_per_unit = float(angular_scale_rad_per_unit)
        self._tracking = False
        self._direction_sign = 1.0
        self._prev_monotonic_s: float | None = None
        self._accum_abs_turn_rad = 0.0
        self._accum_signed_turn_rad = 0.0
        self._samples = 0

    def begin(self, *, direction_sign: float) -> None:
        self._tracking = True
        self._direction_sign = 1.0 if float(direction_sign) >= 0.0 else -1.0
        self._prev_monotonic_s = None
        self._accum_abs_turn_rad = 0.0
        self._accum_signed_turn_rad = 0.0
        self._samples = 0

    def update(self, observation: dict[str, object] | None, *, monotonic_s: float) -> None:
        if not self._tracking:
            self._prev_monotonic_s = float(monotonic_s)
            return

        if self._prev_monotonic_s is None:
            self._prev_monotonic_s = float(monotonic_s)
            return

        dt_s = max(0.0, min(float(monotonic_s) - float(self._prev_monotonic_s), 0.25))
        self._prev_monotonic_s = float(monotonic_s)
        if dt_s <= 0.0 or not isinstance(observation, dict):
            return

        try:
            theta_vel_unit = float(observation.get("theta.vel", 0.0))
        except Exception:
            return

        turn_rate_rad_s = abs(theta_vel_unit) * float(self.angular_scale_rad_per_unit)
        self._accum_abs_turn_rad += turn_rate_rad_s * dt_s
        self._accum_signed_turn_rad += self._direction_sign * turn_rate_rad_s * dt_s
        self._samples += 1

    def progress(self) -> tuple[float, float, int]:
        return self._accum_abs_turn_rad, self._accum_signed_turn_rad, self._samples

    def end(self) -> tuple[float, float, int]:
        self._tracking = False
        return self.progress()


def _build_snapshot_points(
    frames: list[_ScanFrame],
    *,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    invert_lateral_axis: bool,
    max_distance_m: float,
    min_distance_m: float,
    min_confidence: int,
    voxel_m: float,
) -> np.ndarray:
    scans = [
        _scan_to_local_points(
            frame.points,
            forward_angle_deg=forward_angle_deg,
            valid_angle_half_width_deg=valid_angle_half_width_deg,
            invert_lateral_axis=invert_lateral_axis,
            max_distance_m=max_distance_m,
            min_distance_m=min_distance_m,
            min_confidence=min_confidence,
        )
        for frame in frames
    ]
    merged = [scan for scan in scans if scan.size != 0]
    if not merged:
        return np.zeros((0, 2), dtype=np.float32)
    return _voxel_downsample(np.vstack(merged).astype(np.float32, copy=False), voxel_m=voxel_m)


def _save_outputs(
    *,
    output_dir: Path,
    overlay: np.ndarray,
    metadata: dict[str, object],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "latest_two_pose_overlay.png"
    json_path = output_dir / "latest_two_pose_overlay.json"
    html_path = output_dir / "latest_two_pose_overlay.html"
    cv2.imwrite(str(png_path), overlay)
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    png_bytes = png_path.read_bytes()
    png_b64 = base64.b64encode(png_bytes).decode("ascii")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LDLiDAR Two-Pose Overlay</title>
  <style>
    body {{
      margin: 0;
      background: #0b0f14;
      color: #e7eef7;
      font-family: system-ui, sans-serif;
      padding: 20px;
    }}
    .wrap {{
      max-width: 1100px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 12px 0;
      font-size: 28px;
    }}
    p {{
      margin: 6px 0 16px 0;
      color: #b7c3d1;
    }}
    img {{
      width: 100%;
      max-width: 900px;
      height: auto;
      border: 1px solid #223041;
      background: #000;
      display: block;
    }}
    pre {{
      margin-top: 18px;
      padding: 14px;
      background: #101720;
      color: #dbe7f3;
      border: 1px solid #223041;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>LDLiDAR Two-Pose Overlay</h1>
    <p>cyan = snapshot A, orange = snapshot B</p>
    <img alt="Two pose overlay" src="data:image/png;base64,{png_b64}">
    <pre>{json.dumps(metadata, indent=2)}</pre>
  </div>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return png_path, json_path, html_path


def _wsl_to_windows_path(path: Path) -> str:
    text = str(path.resolve())
    if text.startswith("/mnt/") and len(text) > 6:
        drive_letter = text[5]
        remainder = text[6:].replace("/", "\\")
        return f"{drive_letter.upper()}:{remainder}"
    return text


def _open_image_with_system_viewer(path: Path) -> bool:
    resolved = path.resolve()
    try:
        if os.name == "nt":
            os.startfile(str(resolved))  # type: ignore[attr-defined]
            return True
    except Exception:
        pass

    windows_path = _wsl_to_windows_path(resolved)
    open_attempts = [
        ["cmd.exe", "/c", "start", "", windows_path],
        ["powershell.exe", "-NoProfile", "-Command", "Start-Process", windows_path],
        ["xdg-open", str(resolved)],
    ]
    for command in open_attempts:
        try:
            subprocess.Popen(command)
            return True
        except Exception:
            continue
    return False


def _open_browser_preview(path: Path) -> bool:
    resolved = path.resolve()
    windows_path = _wsl_to_windows_path(resolved)
    windows_uri = "file:///" + windows_path.replace("\\", "/")
    open_attempts = [
        ["cmd.exe", "/c", "start", "", windows_uri],
        ["cmd.exe", "/c", "start", "", windows_path],
        ["powershell.exe", "-NoProfile", "-Command", "Start-Process", windows_uri],
        ["wslview", str(resolved)],
        ["xdg-open", str(resolved)],
    ]
    for command in open_attempts:
        try:
            subprocess.Popen(command)
            return True
        except Exception:
            continue
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture two stationary LiDAR snapshots separated by a controlled ~180 degree turn."
    )
    parser.add_argument("--remote-ip", type=str, default="192.168.1.237", help="Sourccey host IP.")
    parser.add_argument("--robot-id", type=str, default="sourccey", help="Robot id for SourcceyClient.")
    parser.add_argument("--lidar-host", type=str, default="192.168.1.237", help="LDLiDAR TCP host.")
    parser.add_argument("--lidar-port", type=int, default=8765, help="LDLiDAR TCP port.")
    parser.add_argument(
        "--slam-input-endpoint",
        type=str,
        default="tcp://192.168.1.237:5560",
        help="Sourccey slam input PUB endpoint carrying IMU samples.",
    )
    parser.add_argument("--turn-direction", choices=("left", "right"), default="left")
    parser.add_argument("--target-turn-deg", type=float, default=180.0)
    parser.add_argument("--turn-speed", type=float, default=0.82, help="Base theta.vel command magnitude.")
    parser.add_argument("--max-turn-duration-s", type=float, default=30.0)
    parser.add_argument("--turn-burst-s", type=float, default=0.20)
    parser.add_argument("--turn-settle-s", type=float, default=0.18)
    parser.add_argument("--settle-before-s", type=float, default=1.0)
    parser.add_argument("--settle-after-s", type=float, default=1.2)
    parser.add_argument("--snapshot-scans", type=int, default=3)
    parser.add_argument("--snapshot-timeout-s", type=float, default=6.0)
    parser.add_argument("--forward-angle-deg", type=float, default=DEFAULT_LIDAR_FORWARD_ANGLE_DEG)
    parser.add_argument("--valid-angle-half-width-deg", type=float, default=90.0)
    parser.set_defaults(invert_lateral_axis=True)
    parser.add_argument("--invert-lateral-axis", dest="invert_lateral_axis", action="store_true")
    parser.add_argument("--no-invert-lateral-axis", dest="invert_lateral_axis", action="store_false")
    parser.add_argument("--max-distance-m", type=float, default=8.0)
    parser.add_argument("--min-distance-m", type=float, default=0.22)
    parser.add_argument("--min-confidence", type=int, default=5)
    parser.add_argument("--lidar-mount-x-m", type=float, default=0.2286)
    parser.add_argument("--lidar-mount-y-m", type=float, default=0.0)
    parser.add_argument("--voxel-m", type=float, default=0.03)
    parser.add_argument("--image-size", type=int, default=900)
    parser.add_argument(
        "--rotation-signature-min-distance-m",
        type=float,
        default=0.35,
        help="Ignore near-field scan returns when estimating turn amount from LiDAR scan rotation.",
    )
    parser.add_argument(
        "--rotation-signature-bin-deg",
        type=float,
        default=2.0,
        help="Angular bin size for LiDAR scan-rotation matching.",
    )
    parser.add_argument("--lock-min-progress-deg", type=float, default=120.0)
    parser.add_argument(
        "--lock-signature-search-half-window-deg",
        type=float,
        default=70.0,
        help="LiDAR rotation search half-window once the turn is far enough along to expect the opposite-facing view.",
    )
    parser.add_argument("--lock-stop-margin-deg", type=float, default=55.0)
    parser.add_argument("--lock-score-drop-threshold", type=float, default=0.006)
    parser.add_argument("--lock-score-drop-count", type=int, default=3)
    parser.add_argument("--lock-search-half-window-deg", type=float, default=40.0)
    parser.add_argument("--lock-search-step-deg", type=float, default=2.0)
    parser.add_argument("--lock-stop-tolerance-deg", type=float, default=5.0)
    parser.add_argument(
        "--lock-confirm-hits",
        type=int,
        default=2,
        help="Consecutive on-target LiDAR measurements (taken without moving) required before stopping the turn.",
    )
    parser.add_argument(
        "--min-turn-speed",
        type=float,
        default=0.16,
        help="Lower bound on theta.vel magnitude so small correction bursts still overcome base stiction.",
    )
    parser.add_argument("--lock-good-score", type=float, default=0.035)
    parser.add_argument("--lock-refine-bursts", type=int, default=8)
    parser.add_argument("--lock-refine-burst-s", type=float, default=0.12)
    parser.add_argument("--lock-refine-settle-s", type=float, default=0.18)
    parser.add_argument(
        "--observation-angular-scale-rad-per-unit",
        type=float,
        default=1.0,
        help="Fallback turn-rate scale for live robot observation theta.vel when IMU packets are unavailable.",
    )
    parser.set_defaults(invert_yaw_sign=True)
    parser.add_argument("--invert-yaw-sign", dest="invert_yaw_sign", action="store_true")
    parser.add_argument("--no-invert-yaw-sign", dest="invert_yaw_sign", action="store_false")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "two_pose_debug")
    parser.add_argument("--headless", action="store_true", help="Save files only; do not open a window.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    lidar = _LidarFeed(args.lidar_host, args.lidar_port)
    imu = _ImuTurnTracker(args.slam_input_endpoint, invert_yaw_sign=bool(args.invert_yaw_sign))
    observed_turn = _ObservedTurnTracker(
        angular_scale_rad_per_unit=float(args.observation_angular_scale_rad_per_unit)
    )
    robot = SourcceyClient(SourcceyClientConfig(id=args.robot_id, remote_ip=args.remote_ip))
    preview = None
    if not bool(args.headless):
        try:
            preview = _TkPreview(title="LDLiDAR Two-Pose Snapshot Test")
        except Exception:
            preview = None

    lidar.start()
    imu.start()
    robot.connect()
    robot.untorque_left_active = True
    robot.untorque_right_active = True

    try:
        observation = robot.get_observation()
        z_hold = float(observation.get("z.pos", robot._z_pos_cmd))

        print(f"Settling before snapshot A for {args.settle_before_s:.1f}s...")
        time.sleep(max(float(args.settle_before_s), 0.0))

        first_frames = lidar.capture_snapshot(scan_count=int(args.snapshot_scans), timeout_s=float(args.snapshot_timeout_s))
        first_local_xy = _build_snapshot_points(
            first_frames,
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            invert_lateral_axis=bool(args.invert_lateral_axis),
            max_distance_m=float(args.max_distance_m),
            min_distance_m=float(args.min_distance_m),
            min_confidence=int(args.min_confidence),
            voxel_m=float(args.voxel_m),
        )
        first_lock_local_xy = _build_snapshot_points(
            first_frames,
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=180.0,
            invert_lateral_axis=bool(args.invert_lateral_axis),
            max_distance_m=float(args.max_distance_m),
            min_distance_m=float(args.min_distance_m),
            min_confidence=int(args.min_confidence),
            voxel_m=float(args.voxel_m),
        )
        first_rotation_signature = _build_rotation_signature(
            first_frames,
            min_distance_m=max(float(args.rotation_signature_min_distance_m), float(args.min_distance_m)),
            max_distance_m=float(args.max_distance_m),
            min_confidence=int(args.min_confidence),
            bin_size_deg=float(args.rotation_signature_bin_deg),
        )
        print(f"Snapshot A captured with {len(first_local_xy)} points from {len(first_frames)} scans.")
        if preview is not None:
            preview.show(
                _render_snapshot_view(
                    first_local_xy,
                    title="Snapshot A",
                    image_size_px=int(args.image_size),
                    max_distance_m=float(args.max_distance_m),
                    forward_angle_deg=float(args.forward_angle_deg),
                ),
                status_lines=[
                    f"Snapshot A points: {len(first_local_xy)}",
                    "Turning toward opposite-facing view...",
                ],
            )

        imu.begin()
        direction_sign = 1.0 if str(args.turn_direction).lower() == "left" else -1.0
        observed_turn.begin(direction_sign=direction_sign)
        target_turn_rad = math.radians(max(float(args.target_turn_deg), 1.0))
        turn_deadline = time.monotonic() + max(float(args.max_turn_duration_s), 1.0)
        print(
            f"Turning {args.turn_direction} toward {args.target_turn_deg:.1f} deg "
            "using LiDAR-guided turn locking..."
        )
        lock_yaw_rad = 0.0
        lock_score = 0.0
        lock_overlap_ratio = 0.0
        lock_rotation_deg = 0.0
        best_lock_yaw_rad = 0.0
        best_lock_overlap_ratio = 0.0
        best_live_frames: list[_ScanFrame] = []
        best_target_error_rad = float("inf")
        target_lock_stable_hits = 0
        commanded_turn_rad = 0.0
        current_target_error_rad = float("inf")
        current_lock_overlap_ratio = 0.0
        current_progress_deg = 0.0

        def _sample_current_lock() -> tuple[
            list[_ScanFrame], np.ndarray, float, float, float, float, float, float, np.ndarray
        ]:
            current_live_frames = lidar.latest_frames(scan_count=1)
            current_live_lock_local_xy = np.zeros((0, 2), dtype=np.float32)
            empty_signature = np.zeros((0,), dtype=np.float32)
            if not current_live_frames:
                return (
                    current_live_frames,
                    current_live_lock_local_xy,
                    0.0,
                    float("inf"),
                    0.0,
                    float("inf"),
                    0.0,
                    float("nan"),
                    empty_signature,
                )
            current_live_lock_local_xy = _build_snapshot_points(
                current_live_frames,
                forward_angle_deg=float(args.forward_angle_deg),
                valid_angle_half_width_deg=180.0,
                invert_lateral_axis=bool(args.invert_lateral_axis),
                max_distance_m=float(args.max_distance_m),
                min_distance_m=float(args.min_distance_m),
                min_confidence=int(args.min_confidence),
                voxel_m=float(args.voxel_m),
            )
            current_live_rotation_signature = _build_rotation_signature(
                current_live_frames,
                min_distance_m=max(float(args.rotation_signature_min_distance_m), float(args.min_distance_m)),
                max_distance_m=float(args.max_distance_m),
                min_confidence=int(args.min_confidence),
                bin_size_deg=float(args.rotation_signature_bin_deg),
            )
            # Full 360 search: robust and simple. The robot's true heading is
            # whatever best matches the start scan, regardless of how far we think
            # we have commanded.
            current_signature_rotation_deg, current_lock_score, current_lock_overlap_ratio = _estimate_signature_rotation_deg(
                first_rotation_signature,
                current_live_rotation_signature,
                bin_size_deg=float(args.rotation_signature_bin_deg),
            )
            current_progress_deg = _rotation_progress_deg(
                current_signature_rotation_deg,
                direction_sign=direction_sign,
            )
            current_target_error_signed_deg = float(args.target_turn_deg) - current_progress_deg
            current_target_error_rad = abs(math.radians(current_target_error_signed_deg))
            return (
                current_live_frames,
                current_live_lock_local_xy,
                float(current_progress_deg),
                float(current_lock_score),
                float(current_lock_overlap_ratio),
                float(current_target_error_rad),
                float(current_target_error_signed_deg),
                float(current_signature_rotation_deg),
                current_live_rotation_signature,
            )

        confirm_hits_needed = max(int(args.lock_confirm_hits), 1)
        min_turn_speed = max(float(args.min_turn_speed), 1e-3)
        current_target_error_signed_deg = float(args.target_turn_deg)
        stop_index = 0

        # Incremental ("keyframe") rotation odometry. Matching every live scan
        # directly against snapshot A collapses past ~130 deg of turn because the
        # two views stop overlapping. Instead we accumulate rotation against a
        # keyframe scan that is refreshed whenever overlap fades, so every match
        # stays high-overlap and we can track a full 180 deg turn reliably.
        keyframe_signature = first_rotation_signature
        keyframe_base_deg = 0.0
        accum_turn_deg = 0.0
        incremental_search_half_window_deg = 55.0
        incremental_min_overlap = 0.30
        keyframe_refresh_overlap = 0.55

        # Stall watchdog: if the LiDAR-measured rotation stops advancing while we
        # are still commanding turns, the base is stuck (or the feed is stale). We
        # escalate the turn speed to break free, and abort with a clear message
        # rather than spinning uselessly until the deadline.
        accum_at_last_burst_deg = 0.0
        no_progress_bursts = 0
        stall_speed_boost = 1.0
        last_scan_ts = float("nan")
        stale_scan_count = 0
        turn_aborted_reason: str | None = None

        while time.monotonic() < turn_deadline:
            # --- 1. LOOK before you leap. Measure heading while the base is
            # settled. We keep an absolute match against snapshot A for the debug
            # log, but the AUTHORITATIVE progress comes from incremental keyframe
            # odometry below.
            measured = False
            raw_rotation_deg = float("nan")
            abs_turned_deg = float("nan")
            abs_overlap = 0.0
            rel_turned_deg = float("nan")
            live_frames = lidar.latest_frames(scan_count=1)
            live_lock_local_xy = np.zeros((0, 2), dtype=np.float32)
            current_scan_ts = float(live_frames[-1].ts) if live_frames else float("nan")
            scan_is_fresh = bool(live_frames) and not (
                math.isfinite(last_scan_ts) and current_scan_ts == last_scan_ts
            )
            if scan_is_fresh:
                last_scan_ts = current_scan_ts
                stale_scan_count = 0
            elif live_frames:
                stale_scan_count += 1
            if live_frames:
                (
                    live_frames,
                    live_lock_local_xy,
                    abs_turned_deg,
                    abs_score,
                    abs_overlap,
                    _abs_target_error_rad,
                    _abs_target_error_signed_deg,
                    raw_rotation_deg,
                    current_signature,
                ) = _sample_current_lock()

                # Incremental match against the current keyframe, searched only in a
                # small window around where we expect to be. This prevents the
                # spurious global-minimum jumps and stays locked far from snapshot A.
                predicted_rel_deg = max(accum_turn_deg - keyframe_base_deg, 0.0)
                center_raw_deg = _turned_to_raw_deg(predicted_rel_deg, direction_sign=direction_sign)
                rel_raw_deg, rel_score, rel_overlap = _estimate_signature_rotation_deg(
                    keyframe_signature,
                    current_signature,
                    bin_size_deg=float(args.rotation_signature_bin_deg),
                    center_deg=center_raw_deg,
                    search_half_window_deg=incremental_search_half_window_deg,
                )
                if rel_overlap >= incremental_min_overlap and math.isfinite(rel_score):
                    rel_turned_deg = _rotation_progress_deg(rel_raw_deg, direction_sign=direction_sign)
                    # The incremental step is small (bounded by the search window), so
                    # anything near 360 is really a tiny step the OTHER way that wrapped
                    # (e.g. a 2 deg shift reading as 358). Fold it back to (-180, 180]
                    # so noise near zero can't explode into a bogus ~360 deg jump.
                    if rel_turned_deg > 180.0:
                        rel_turned_deg -= 360.0
                    accum_turn_deg = keyframe_base_deg + rel_turned_deg
                    measured = True
                    current_progress_deg = float(accum_turn_deg)
                    current_target_error_signed_deg = float(args.target_turn_deg) - float(accum_turn_deg)
                    current_target_error_rad = abs(math.radians(current_target_error_signed_deg))
                    current_lock_overlap_ratio = float(rel_overlap)
                    lock_rotation_deg = float(accum_turn_deg)
                    lock_score = float(rel_score)
                    lock_overlap_ratio = float(rel_overlap)
                    lock_yaw_rad = direction_sign * math.radians(accum_turn_deg)
                    if current_target_error_rad < best_target_error_rad:
                        best_target_error_rad = float(current_target_error_rad)
                        best_lock_yaw_rad = float(lock_yaw_rad)
                        best_lock_overlap_ratio = float(rel_overlap)
                        best_live_frames = list(live_frames)
                    # Refresh the keyframe once overlap starts to fade so the next
                    # match is again between near-identical views.
                    if rel_overlap < keyframe_refresh_overlap and current_signature.size > 0:
                        keyframe_signature = current_signature
                        keyframe_base_deg = float(accum_turn_deg)

            # --- DEBUG: persistent per-stop log. ACCUM_turned is the authoritative
            # incremental estimate of how far we have turned; absA_* is the (now
            # only diagnostic) direct match against snapshot A, which is expected to
            # lose overlap and break down past ~130 deg.
            stop_index += 1
            print(
                f"[stop {stop_index:02d}] "
                f"turn_dir={'left' if direction_sign > 0.0 else 'right'} "
                f"ACCUM_turned={current_progress_deg:7.1f}deg "
                f"target_err={math.degrees(current_target_error_rad) if current_target_error_rad != float('inf') else float('nan'):6.1f}deg "
                f"rel_overlap={lock_overlap_ratio:0.2f} rel_score={lock_score:0.4f} "
                f"| absA_turned={abs_turned_deg:7.1f}deg absA_overlap={abs_overlap:0.2f} "
                f"keyframe_base={keyframe_base_deg:6.1f}deg "
                f"cmd_accum={math.degrees(commanded_turn_rad):6.1f}deg "
                f"scan={'fresh' if scan_is_fresh else 'STALE'} "
                f"no_prog={no_progress_bursts} boost={stall_speed_boost:0.2f} "
                f"measured={'yes' if measured else 'NO '}",
                flush=True,
            )

            imu_abs_turn_rad, imu_signed_turn_rad, imu_packets = imu.progress()
            obs_abs_turn_rad, obs_signed_turn_rad, obs_samples = observed_turn.progress()
            use_imu = imu_packets > 0 and imu_abs_turn_rad > 1e-6
            abs_turn_rad = imu_abs_turn_rad if use_imu else obs_abs_turn_rad
            signed_turn_rad = imu_signed_turn_rad if use_imu else obs_signed_turn_rad
            tracker_label = "imu" if use_imu else "obs"

            measured_error_deg = (
                abs(math.degrees(current_target_error_rad))
                if current_target_error_rad != float("inf")
                else float("inf")
            )
            # Monotonic approach: we only ever turn in `direction_sign`, so the
            # measured rotation progress climbs from 0 toward the target. Treat the
            # target as reached the moment progress enters the tolerance band (or
            # beyond). We deliberately never reverse-correct: tiny velocity bursts
            # on a stiction-y base cannot land inside the band reliably, and reverse
            # nudging around a noisy 180-degree estimate just produces the
            # back-and-forth limit cycle you saw.
            reached_target = (
                measured
                and current_lock_overlap_ratio >= 0.22
                and current_progress_deg
                >= (float(args.target_turn_deg) - float(args.lock_stop_tolerance_deg))
            )

            if preview is not None and live_frames:
                preview.show(
                    _render_snapshot_view(
                        live_lock_local_xy,
                        title="Live Turn View",
                        image_size_px=int(args.image_size),
                        max_distance_m=float(args.max_distance_m),
                        forward_angle_deg=float(args.forward_angle_deg),
                    ),
                    status_lines=[
                        f"Commanded progress: {math.degrees(commanded_turn_rad):.1f} deg",
                        f"Estimated LiDAR yaw: {math.degrees(lock_yaw_rad):.1f} deg",
                        f"Estimated LiDAR progress: {current_progress_deg:.1f} deg",
                        f"Current target error: {measured_error_deg:.1f} deg",
                        f"On-target confirmations: {target_lock_stable_hits}/{confirm_hits_needed}",
                        f"LiDAR rotation score: {lock_score:.4f}",
                        f"Signature overlap: {lock_overlap_ratio:.2f}",
                        f"Turn direction: {'left' if direction_sign > 0.0 else 'right'}",
                        f"Tracker: {tracker_label}  IMU packets: {imu_packets}  Obs samples: {obs_samples}",
                    ],
                )

            # --- 2. Stop the instant the LiDAR confirms we are on target. The
            # confirmation re-measures WITHOUT moving, so verifying the stop can
            # never itself cause an overshoot.
            if reached_target:
                target_lock_stable_hits += 1
                if target_lock_stable_hits >= confirm_hits_needed:
                    break
                # Re-confirm WITHOUT moving. A stationary re-measure can never
                # cause overshoot, so this only guards against a single noisy read.
                time.sleep(max(float(args.lock_refine_settle_s), 0.12))
                continue
            target_lock_stable_hits = 0

            # --- 3. Size the next burst from the measured remaining error so we
            # approach the target instead of charging past it.
            # Remaining angle to the target, measured from the LiDAR. Always >= 0
            # because we approach from below and stop on the first crossing, so an
            # overshoot would have already broken out of the loop above.
            if measured:
                remaining_error_deg = max(
                    float(args.target_turn_deg) - current_progress_deg, 0.0
                )
            else:
                remaining_error_deg = float(args.target_turn_deg)

            # Stall watchdog. If we have a fresh measurement that did NOT advance
            # since the last burst, the base is physically stuck (this is what made
            # it stop ~40 deg short before). Escalate speed to break free; if that
            # keeps failing, abort instead of spinning to the deadline.
            if measured and scan_is_fresh and commanded_turn_rad > 0.0:
                if (accum_turn_deg - accum_at_last_burst_deg) < 1.5:
                    no_progress_bursts += 1
                else:
                    no_progress_bursts = 0
                    stall_speed_boost = 1.0
            if not scan_is_fresh and stale_scan_count >= 12:
                turn_aborted_reason = (
                    f"LiDAR feed went stale (no new scans for {stale_scan_count} checks) "
                    "during the turn; aborting so we don't spin blind."
                )
                break
            if no_progress_bursts >= 3:
                stall_speed_boost = min(stall_speed_boost + 0.25, 2.0)
            if no_progress_bursts >= 10:
                turn_aborted_reason = (
                    f"Base stopped rotating at ~{accum_turn_deg:.0f} deg "
                    f"(no LiDAR progress over {no_progress_bursts} bursts even at boosted speed); "
                    "aborting. Try a higher --turn-speed."
                )
                break

            # Keep the turn speed STRONG the whole way so the base never stalls; get
            # fine resolution near the target by shortening the burst DURATION, not
            # by dropping speed (dropping speed below stiction is what stalled it).
            turn_speed_cmd = min(float(args.turn_speed) * stall_speed_boost, 1.0)
            turn_settle_s = max(float(args.turn_settle_s), 0.05)
            if remaining_error_deg <= 25.0:
                turn_settle_s = max(float(args.lock_refine_settle_s), 0.10)

            if remaining_error_deg > 40.0:
                turn_burst_s = max(float(args.turn_burst_s), 0.10)
            elif remaining_error_deg > 15.0:
                turn_burst_s = 0.12
            else:
                turn_burst_s = 0.08

            accum_at_last_burst_deg = accum_turn_deg
            burst_deadline = time.monotonic() + turn_burst_s
            while time.monotonic() < burst_deadline:
                action = {
                    "x.vel": 0.0,
                    "y.vel": 0.0,
                    "theta.vel": direction_sign * turn_speed_cmd,
                    "z.pos": z_hold,
                    "untorque_left": True,
                    "untorque_right": True,
                }
                robot.send_action(action)
                try:
                    turn_observation = robot.get_observation()
                except Exception:
                    turn_observation = {}
                observed_turn.update(turn_observation, monotonic_s=time.monotonic())
                time.sleep(0.05)
            commanded_turn_rad += abs(turn_speed_cmd) * turn_burst_s

            robot.send_action(
                {
                    "x.vel": 0.0,
                    "y.vel": 0.0,
                    "theta.vel": 0.0,
                    "z.pos": z_hold,
                    "untorque_left": True,
                    "untorque_right": True,
                }
            )
            time.sleep(turn_settle_s)
        print()
        if turn_aborted_reason is not None:
            print(f"WARNING: {turn_aborted_reason}")

        for _ in range(3):
            robot.send_action(
                {
                    "x.vel": 0.0,
                    "y.vel": 0.0,
                    "theta.vel": 0.0,
                    "z.pos": z_hold,
                    "untorque_left": True,
                    "untorque_right": True,
                }
            )
            time.sleep(0.04)

        imu_abs_turn_rad, imu_signed_turn_rad, imu_packets = imu.end()
        obs_abs_turn_rad, obs_signed_turn_rad, obs_samples = observed_turn.end()
        use_imu = imu_packets > 0 and imu_abs_turn_rad > 1e-6
        abs_turn_rad = imu_abs_turn_rad if use_imu else obs_abs_turn_rad
        signed_turn_rad = imu_signed_turn_rad if use_imu else obs_signed_turn_rad
        tracker_label = "imu" if use_imu else "obs"
        final_source = "lock"
        if (
            math.degrees(current_target_error_rad) <= max(float(args.lock_stop_tolerance_deg) * 1.5, 8.0)
            and current_lock_overlap_ratio >= 0.20
        ):
            signed_turn_rad = float(lock_yaw_rad)
            abs_turn_rad = abs(float(lock_yaw_rad))
            final_source = "lock_refined_current"
        elif (
            best_target_error_rad < float("inf")
            and math.degrees(best_target_error_rad) <= max(float(args.lock_stop_tolerance_deg) * 2.0, 12.0)
            and best_lock_overlap_ratio >= 0.18
        ):
            signed_turn_rad = float(best_lock_yaw_rad)
            abs_turn_rad = abs(float(best_lock_yaw_rad))
            lock_yaw_rad = float(best_lock_yaw_rad)
            if best_live_frames:
                best_lock_local_xy = _build_snapshot_points(
                    best_live_frames,
                    forward_angle_deg=float(args.forward_angle_deg),
                    valid_angle_half_width_deg=180.0,
                    invert_lateral_axis=bool(args.invert_lateral_axis),
                    max_distance_m=float(args.max_distance_m),
                    min_distance_m=float(args.min_distance_m),
                    min_confidence=int(args.min_confidence),
                    voxel_m=float(args.voxel_m),
                )
                lock_score = _local_dissimilarity_score(first_lock_local_xy, best_lock_local_xy)
            if best_live_frames:
                live_frames = list(best_live_frames)
        else:
            final_source = "current_unlocked"
            signed_turn_rad = float(lock_yaw_rad)
            abs_turn_rad = abs(float(lock_yaw_rad))

        print(
            f"Turn complete: abs={math.degrees(abs_turn_rad):.1f}deg "
            f"signed={math.degrees(signed_turn_rad):.1f}deg "
            f"lock={math.degrees(lock_yaw_rad):.1f}deg "
            f"score={lock_score:0.4f} "
            f"tracker={tracker_label} imu_packets={imu_packets} obs_samples={obs_samples} "
            f"source={final_source}"
        )

        print(f"Settling before snapshot B for {args.settle_after_s:.1f}s...")
        time.sleep(max(float(args.settle_after_s), 0.0))

        second_frames = lidar.capture_snapshot(scan_count=int(args.snapshot_scans), timeout_s=float(args.snapshot_timeout_s))
        second_local_xy = _build_snapshot_points(
            second_frames,
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            invert_lateral_axis=bool(args.invert_lateral_axis),
            max_distance_m=float(args.max_distance_m),
            min_distance_m=float(args.min_distance_m),
            min_confidence=int(args.min_confidence),
            voxel_m=float(args.voxel_m),
        )
        print(f"Snapshot B captured with {len(second_local_xy)} points from {len(second_frames)} scans.")

        # --- DEBUG: independent heading estimate taken directly from snapshot B,
        # compared against the snapshot A reference. This is the ground-truth-ish
        # answer to "which way is the robot actually facing now?", computed fresh
        # from the final stationary scans (not carried over from the turn loop).
        second_rotation_signature = _build_rotation_signature(
            second_frames,
            min_distance_m=max(float(args.rotation_signature_min_distance_m), float(args.min_distance_m)),
            max_distance_m=float(args.max_distance_m),
            min_confidence=int(args.min_confidence),
            bin_size_deg=float(args.rotation_signature_bin_deg),
        )
        b_raw_deg, b_score, b_overlap = _estimate_signature_rotation_deg(
            first_rotation_signature,
            second_rotation_signature,
            bin_size_deg=float(args.rotation_signature_bin_deg),
        )
        b_turned_deg = _rotation_progress_deg(b_raw_deg, direction_sign=direction_sign)
        b_target_err_deg = abs(float(args.target_turn_deg) - b_turned_deg)
        print(
            "[snapshot B heading] "
            f"ACCUM_turned(trusted)={accum_turn_deg:7.1f}deg "
            f"| absA_match: raw={b_raw_deg:7.1f}deg turned={b_turned_deg:7.1f}deg "
            f"target_err={b_target_err_deg:6.1f}deg overlap={b_overlap:0.2f} score={b_score:0.4f} "
            f"(absA unreliable near 180deg; trust ACCUM)"
        )

        # The heading arrow must use the SAME yaw the point cloud is rotated by,
        # otherwise it points the mirror-image direction (the bug you saw). The
        # cloud is placed with base_yaw_rad=signed_turn_rad and reads correctly, so
        # the arrow uses signed_turn_rad too.
        render_signed_turn_rad = float(signed_turn_rad)
        print(
            "[render mapping] "
            f"final_source={final_source} "
            f"signed_turn={math.degrees(signed_turn_rad):7.1f}deg "
            f"abs_turn={math.degrees(abs_turn_rad):7.1f}deg "
            f"lock_yaw={math.degrees(lock_yaw_rad):7.1f}deg "
            f"arrow_yaw(applied to B)={math.degrees(render_signed_turn_rad):7.1f}deg "
            f"invert_lateral_axis={bool(args.invert_lateral_axis)}"
        )

        first_world_xy = _transform_snapshot_to_world(
            first_local_xy,
            base_x_m=0.0,
            base_y_m=0.0,
            base_yaw_rad=0.0,
            lidar_mount_x_m=float(args.lidar_mount_x_m),
            lidar_mount_y_m=float(args.lidar_mount_y_m),
        )
        second_world_xy = _transform_snapshot_to_world(
            second_local_xy,
            base_x_m=0.0,
            base_y_m=0.0,
            base_yaw_rad=float(signed_turn_rad),
            lidar_mount_x_m=float(args.lidar_mount_x_m),
            lidar_mount_y_m=float(args.lidar_mount_y_m),
        )
        overlay = _render_overlay(
            first_world_xy,
            second_world_xy,
            second_yaw_rad=render_signed_turn_rad,
            image_size_px=int(args.image_size),
        )

        metadata = {
            "schema": "sourccey.two_pose_overlay.v1",
            "ts": time.time(),
            "remote_ip": args.remote_ip,
            "lidar_host": args.lidar_host,
            "target_turn_deg": float(args.target_turn_deg),
            "turn_tracker": tracker_label,
            "measured_signed_turn_deg": math.degrees(float(signed_turn_rad)),
            "measured_abs_turn_deg": math.degrees(float(abs_turn_rad)),
            "lock_signed_turn_deg": math.degrees(float(lock_yaw_rad)),
            "render_signed_turn_deg": math.degrees(float(render_signed_turn_rad)),
            "lock_score": float(lock_score),
            "lock_overlap_ratio": float(lock_overlap_ratio),
            "best_target_error_deg": math.degrees(float(best_target_error_rad))
            if best_target_error_rad < float("inf")
            else None,
            "final_turn_source": final_source,
            "imu_packets": int(imu_packets),
            "observation_turn_samples": int(obs_samples),
            "observation_abs_turn_deg": math.degrees(float(obs_abs_turn_rad)),
            "observation_signed_turn_deg": math.degrees(float(obs_signed_turn_rad)),
            "snapshot_scans": int(args.snapshot_scans),
            "snapshot_a_points": int(len(first_world_xy)),
            "snapshot_b_points": int(len(second_world_xy)),
            "lidar_mount_x_m": float(args.lidar_mount_x_m),
            "lidar_mount_y_m": float(args.lidar_mount_y_m),
            "forward_angle_deg": float(args.forward_angle_deg),
        }
        png_path, json_path, html_path = _save_outputs(output_dir=args.output_dir, overlay=overlay, metadata=metadata)
        print(f"Saved overlay to {png_path}")
        print(f"Saved metadata to {json_path}")
        print(f"Saved browser preview to {html_path}")

        if not bool(args.headless):
            if preview is not None:
                preview.show(
                    overlay,
                    status_lines=[
                        f"Turn complete: {math.degrees(signed_turn_rad):.1f} deg",
                        f"Difference score: {lock_score:.4f}",
                        f"Saved overlay: {png_path}",
                        "Close this window when you are done inspecting the result.",
                    ],
                )
                preview.wait_until_closed()
            else:
                opened = _open_browser_preview(html_path)
                if opened:
                    print(f"Preview unavailable; opened browser preview: {html_path}")
                else:
                    print(f"Preview unavailable; could not auto-open. Open this in your browser: {html_path}")
        return 0
    finally:
        if preview is not None:
            preview.close()
        try:
            robot.send_action(
                {
                    "x.vel": 0.0,
                    "y.vel": 0.0,
                    "theta.vel": 0.0,
                    "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
                    "untorque_left": True,
                    "untorque_right": True,
                }
            )
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        imu.stop()
        lidar.stop()


if __name__ == "__main__":
    raise SystemExit(main())
