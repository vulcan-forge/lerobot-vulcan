from __future__ import annotations

import argparse
import json
import math
import socket
import threading
import time
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
from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient


@dataclass
class StopZoneConfig:
    forward_angle_deg: float
    min_distance_m: float
    tripwire_distance_m: float
    tripwire_half_width_m: float
    tripwire_thickness_m: float
    min_points_to_trigger: int


def _normalize_angle_deg(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def _point_in_stop_zone(angle_deg: float, distance_m: float, cfg: StopZoneConfig) -> bool:
    delta = _normalize_angle_deg(angle_deg - cfg.forward_angle_deg)
    theta = math.radians(delta)
    forward_m = distance_m * math.cos(theta)
    lateral_m = distance_m * math.sin(theta)
    near_edge_m = 0.0
    far_edge_m = max(cfg.min_distance_m, cfg.tripwire_distance_m + cfg.tripwire_thickness_m / 2.0)
    return (
        forward_m >= near_edge_m
        and forward_m <= far_edge_m
        and abs(lateral_m) <= cfg.tripwire_half_width_m
    )


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

    def local_to_xy(forward_m: float, lateral_m: float) -> tuple[int, int]:
        angle_deg = cfg.forward_angle_deg + math.degrees(math.atan2(lateral_m, forward_m))
        distance_m = math.hypot(forward_m, lateral_m)
        return to_xy(angle_deg, distance_m)

    near_edge_m = 0.0
    far_edge_m = max(cfg.min_distance_m, cfg.tripwire_distance_m + cfg.tripwire_thickness_m / 2.0)

    color = (0, 0, 255) if blocked else (255, 200, 0)
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


def _draw_lidar_view(
    points: list[list[float]],
    *,
    size: int,
    max_distance_m: float,
    rpm: float,
    zone_cfg: StopZoneConfig,
    blocked: bool,
    blocked_points: int,
    speed_level: int,
    forward_blocked: bool,
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
    cv2.putText(canvas, f"{status_text} zone_points={blocked_points}/{zone_cfg.min_points_to_trigger}", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, status_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"rpm={rpm:.1f} total_points={len(points)} speed={speed_level}/3", (16, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(canvas, "W/S forward-back  A/D strafe  Z/X rotate  Q/E z  R/F speed", (16, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Forward is clamped to 0 while the stop box is occupied.", (16, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, (180, 180, 180), 1, cv2.LINE_AA)
    if forward_blocked:
        cv2.putText(canvas, "Forward command blocked by LiDAR safety zone.", (16, 136),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"box_depth={zone_cfg.tripwire_distance_m:.3f}m width={zone_cfg.tripwire_half_width_m * 2.0:.2f}m",
        (16, size - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return canvas


class _LidarFeed:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.latest_points: list[list[float]] = []
        self.latest_rpm: float = 0.0
        self.connected = False
        self.last_error: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="ldlidar_feed", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def snapshot(self) -> tuple[list[list[float]], float, bool, str | None]:
        with self._lock:
            return list(self.latest_points), self.latest_rpm, self.connected, self.last_error

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
                        points = payload.get("points", [])
                        rpm = float(payload.get("rpm", 0.0))
                        with self._lock:
                            self.latest_points = points
                            self.latest_rpm = rpm
                            self.connected = True
                            self.last_error = None
            except Exception as exc:
                with self._lock:
                    self.connected = False
                    self.last_error = str(exc)
                time.sleep(0.5)


class _TeleopApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.zone_cfg = StopZoneConfig(
            forward_angle_deg=float(args.forward_angle_deg),
            min_distance_m=float(args.min_distance_m),
            tripwire_distance_m=float(args.tripwire_distance_m),
            tripwire_half_width_m=float(args.tripwire_half_width_m),
            tripwire_thickness_m=float(args.tripwire_thickness_m),
            min_points_to_trigger=max(int(args.min_points), 1),
        )
        self.lidar = _LidarFeed(args.lidar_host, args.lidar_port)
        self.robot = SourcceyClient(SourcceyClientConfig(id=args.robot_id, remote_ip=args.remote_ip))
        self.pressed_keys: set[str] = set()
        self.forward_blocked = False
        self._photo: ImageTk.PhotoImage | None = None
        self.root = tk.Tk()
        self.root.title(args.window_title)
        self.label = tk.Label(self.root)
        self.label.pack()
        self.status_label = tk.Label(self.root, anchor="w", justify="left", font=("Consolas", 11))
        self.status_label.pack(fill="x", padx=8, pady=6)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.focus_force()
        self._closed = False

    def run(self) -> int:
        self.lidar.start()
        self.robot.connect()
        self.robot.untorque_left_active = True
        self.robot.untorque_right_active = True
        self.root.after(0, self._tick)
        try:
            self.root.mainloop()
        finally:
            self._shutdown()
        return 0

    def _on_key_press(self, event: tk.Event) -> None:
        key = str(event.keysym).lower()
        key_map = {
            "escape": "esc",
            "space": "space",
        }
        key = key_map.get(key, key)
        if key in {"w", "a", "s", "d", "z", "x", "q", "e"}:
            self.pressed_keys.add(key)
        elif key in {"r", "f"}:
            try:
                self.robot.on_key_down(key)
            except Exception:
                pass
        elif key in {"esc", "space"}:
            self._close()

    def _on_key_release(self, event: tk.Event) -> None:
        key = str(event.keysym).lower()
        if key in self.pressed_keys:
            self.pressed_keys.discard(key)

    def _tick(self) -> None:
        if self._closed:
            return

        points, rpm, lidar_connected, lidar_error = self.lidar.snapshot()
        blocked_points = sum(
            1
            for angle_deg, distance_m, _confidence in points
            if _point_in_stop_zone(float(angle_deg), float(distance_m), self.zone_cfg)
        )
        blocked = blocked_points >= self.zone_cfg.min_points_to_trigger

        try:
            observation = self.robot.get_observation()
        except Exception:
            observation = {}

        z_obs = float(observation.get("z.pos", self.robot._z_pos_cmd))
        action = self.robot._from_keyboard_to_base_action(np.array(sorted(self.pressed_keys), dtype=object), z_obs_pos=z_obs)
        action["untorque_left"] = True
        action["untorque_right"] = True

        self.forward_blocked = blocked and float(action.get("x.vel", 0.0)) > 0.0
        if self.forward_blocked:
            action["x.vel"] = 0.0

        try:
            self.robot.send_action(action)
        except Exception:
            pass

        frame = _draw_lidar_view(
            points,
            size=self.args.canvas_size,
            max_distance_m=self.args.max_distance_m,
            rpm=rpm,
            zone_cfg=self.zone_cfg,
            blocked=blocked,
            blocked_points=blocked_points,
            speed_level=int(self.robot.speed_index) + 1,
            forward_blocked=self.forward_blocked,
        )
        self._show_bgr(frame)

        status_lines = [
            f"robot={self.args.remote_ip}  lidar={self.args.lidar_host}:{self.args.lidar_port}",
            f"lidar_connected={lidar_connected}  last_error={lidar_error or 'none'}",
            f"pressed_keys={''.join(sorted(self.pressed_keys)) or '-'}  blocked={blocked}  forward_blocked={self.forward_blocked}",
        ]
        self.status_label.configure(text="\n".join(status_lines))
        self.root.after(max(int(self.args.tick_ms), 15), self._tick)

    def _show_bgr(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(image=image)
        self.label.configure(image=self._photo)

    def _close(self) -> None:
        self._closed = True
        self.root.quit()
        self.root.destroy()

    def _shutdown(self) -> None:
        try:
            self.robot.send_action(
                {
                    "x.vel": 0.0,
                    "y.vel": 0.0,
                    "theta.vel": 0.0,
                    "z.pos": float(self.robot._z_pos_cmd),
                    "untorque_left": True,
                    "untorque_right": True,
                }
            )
        except Exception:
            pass
        try:
            self.robot.disconnect()
        except Exception:
            pass
        self.lidar.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyboard base teleop with live LiDAR forward-stop safety.")
    parser.add_argument("--remote-ip", type=str, default="192.168.1.211", help="IP address of the Sourccey host.")
    parser.add_argument("--robot-id", type=str, default="sourccey", help="Robot id for the Sourccey client.")
    parser.add_argument("--lidar-host", type=str, default="192.168.1.237", help="Host running ldlidar_stream_host.py.")
    parser.add_argument("--lidar-port", type=int, default=8765, help="TCP port exposed by the LiDAR host streamer.")
    parser.add_argument("--window-title", type=str, default="Sourccey LiDAR Safe Base Teleop", help="Tk window title.")
    parser.add_argument("--canvas-size", type=int, default=900, help="LiDAR viewer size in pixels.")
    parser.add_argument("--max-distance-m", type=float, default=8.0, help="LiDAR viewer distance scale.")
    parser.add_argument("--tick-ms", type=int, default=33, help="UI/control loop interval in milliseconds.")
    parser.add_argument("--forward-angle-deg", type=float, default=DEFAULT_LIDAR_FORWARD_ANGLE_DEG, help="Center direction for the stop box.")
    parser.add_argument("--min-distance-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_MIN_DISTANCE_M, help="Ignore points closer than this.")
    parser.add_argument("--tripwire-distance-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_DISTANCE_M, help="Stop box center depth from the robot.")
    parser.add_argument("--tripwire-half-width-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_HALF_WIDTH_M, help="Half-width of the stop box.")
    parser.add_argument("--tripwire-thickness-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_THICKNESS_M, help="Visual thickness of the stop boundary.")
    parser.add_argument("--min-points", type=int, default=6, help="Points required inside the stop box before blocking forward motion.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _TeleopApp(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
