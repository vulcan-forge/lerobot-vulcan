from __future__ import annotations

import argparse
import json
import logging
import math
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import zmq


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_angle_rad(angle_rad: float) -> float:
    while angle_rad <= -math.pi:
        angle_rad += 2.0 * math.pi
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    return angle_rad


def _normalize_angle_deg(angle_deg: float) -> float:
    return math.degrees(_normalize_angle_rad(math.radians(angle_deg)))


@dataclass(slots=True)
class Pose2D:
    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(slots=True)
class MotionDelta:
    dx_local_m: float
    dy_local_m: float
    dyaw_rad: float
    dt_s: float
    vx_m_s: float
    vy_m_s: float
    wz_rad_s: float


@dataclass(slots=True)
class ScanFrame:
    ts_wall_s: float
    rpm: float
    points_local_m: np.ndarray
    valid_points: int


class MotionAccumulator:
    def __init__(self, endpoint: str | None, *, use_imu_heading: bool) -> None:
        self.endpoint = None if endpoint is None or not str(endpoint).strip() else str(endpoint).strip()
        self.use_imu_heading = bool(use_imu_heading)
        self._lock = threading.Lock()
        self._pending_dx_local_m = 0.0
        self._pending_dy_local_m = 0.0
        self._pending_dyaw_rad = 0.0
        self._pending_dt_s = 0.0
        self._latest_vx_m_s = 0.0
        self._latest_vy_m_s = 0.0
        self._latest_wz_rad_s = 0.0
        self._last_packet_ns: int | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self.endpoint is None:
            logging.info("Column-carving mapper running without slam_input motion prior.")
            return
        self._thread = threading.Thread(
            target=self._run,
            name="column_carving_motion_accumulator",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def consume(self) -> MotionDelta:
        with self._lock:
            delta = MotionDelta(
                dx_local_m=self._pending_dx_local_m,
                dy_local_m=self._pending_dy_local_m,
                dyaw_rad=self._pending_dyaw_rad,
                dt_s=self._pending_dt_s,
                vx_m_s=self._latest_vx_m_s,
                vy_m_s=self._latest_vy_m_s,
                wz_rad_s=self._latest_wz_rad_s,
            )
            self._pending_dx_local_m = 0.0
            self._pending_dy_local_m = 0.0
            self._pending_dyaw_rad = 0.0
            self._pending_dt_s = 0.0
            return delta

    def _run(self) -> None:
        assert self.endpoint is not None
        context = zmq.Context.instance()
        sock = context.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.RCVTIMEO, 500)
        sock.connect(self.endpoint)
        logging.info("Motion accumulator connected to %s", self.endpoint)
        try:
            while not self._stop.is_set():
                try:
                    payload = sock.recv()
                except zmq.Again:
                    continue
                except Exception as exc:
                    if not self._stop.is_set():
                        logging.warning("Motion accumulator recv error: %s", exc)
                    continue
                try:
                    packet = json.loads(payload.decode("utf-8"))
                except Exception:
                    continue
                if packet.get("schema") != "slam_input.v1":
                    continue
                self._ingest(packet)
        finally:
            sock.close(0)

    def _ingest(self, packet: dict[str, Any]) -> None:
        host_ns = int(packet.get("host_monotonic_ns", 0))
        if host_ns <= 0:
            return
        if self._last_packet_ns is None:
            self._last_packet_ns = host_ns
            return
        dt_s = (host_ns - self._last_packet_ns) / 1_000_000_000.0
        self._last_packet_ns = host_ns
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            return
        dt_s = min(dt_s, 0.25)

        base_velocity = packet.get("base_velocity", {})
        vx_m_s = _safe_float(base_velocity.get("x.vel"), 0.0)
        vy_m_s = _safe_float(base_velocity.get("y.vel"), 0.0)
        wz_rad_s = _safe_float(base_velocity.get("theta.vel"), 0.0)

        latest_gz = self._latest_imu_gz(packet.get("imu_samples"))
        if self.use_imu_heading and latest_gz is not None and math.isfinite(latest_gz):
            wz_rad_s = latest_gz

        with self._lock:
            self._pending_dx_local_m += vx_m_s * dt_s
            self._pending_dy_local_m += vy_m_s * dt_s
            self._pending_dyaw_rad += wz_rad_s * dt_s
            self._pending_dt_s += dt_s
            self._latest_vx_m_s = vx_m_s
            self._latest_vy_m_s = vy_m_s
            self._latest_wz_rad_s = wz_rad_s

    @staticmethod
    def _latest_imu_gz(raw_samples: Any) -> float | None:
        if not isinstance(raw_samples, list):
            return None
        best_ns = -1
        best_gz: float | None = None
        for sample in raw_samples:
            if not isinstance(sample, dict):
                continue
            ts_ns = int(sample.get("capture_monotonic_ns", 0))
            if ts_ns <= best_ns:
                continue
            try:
                gz = float(sample.get("gz"))
            except Exception:
                continue
            best_ns = ts_ns
            best_gz = gz
        return best_gz


class LidarScanReader:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        forward_angle_deg: float,
        min_range_m: float,
        max_range_m: float,
        min_confidence: int,
        max_points: int,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.forward_angle_deg = float(forward_angle_deg)
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)
        self.min_confidence = int(min_confidence)
        self.max_points = max(int(max_points), 1)

    def scans(self):
        while True:
            try:
                with socket.create_connection((self.host, self.port), timeout=5.0) as sock:
                    sock.settimeout(2.0)
                    with sock.makefile("r", encoding="utf-8") as file_obj:
                        logging.info("LiDAR scan reader connected to tcp://%s:%d", self.host, self.port)
                        for line in file_obj:
                            payload = json.loads(line)
                            frame = self._frame_from_payload(payload)
                            if frame is None:
                                continue
                            yield frame
            except Exception as exc:
                logging.warning("LiDAR scan reader reconnecting after error: %s", exc)
                time.sleep(0.5)

    def _frame_from_payload(self, payload: dict[str, Any]) -> ScanFrame | None:
        points = payload.get("points", [])
        if not isinstance(points, list):
            return None
        local_points: list[tuple[float, float]] = []
        for raw_point in points:
            if not isinstance(raw_point, list | tuple) or len(raw_point) < 3:
                continue
            angle_deg = _safe_float(raw_point[0], 0.0)
            distance_m = _safe_float(raw_point[1], 0.0)
            confidence = int(_safe_float(raw_point[2], 0.0))
            if confidence < self.min_confidence:
                continue
            if not math.isfinite(distance_m) or distance_m < self.min_range_m or distance_m > self.max_range_m:
                continue
            theta = math.radians(angle_deg - self.forward_angle_deg)
            x_local = distance_m * math.cos(theta)
            y_local = distance_m * math.sin(theta)
            local_points.append((x_local, y_local))

        if not local_points:
            return None

        points_array = np.asarray(local_points, dtype=np.float32)
        if len(points_array) > self.max_points:
            stride = max(len(points_array) // self.max_points, 1)
            points_array = points_array[::stride][: self.max_points]

        return ScanFrame(
            ts_wall_s=_safe_float(payload.get("ts"), time.time()),
            rpm=_safe_float(payload.get("rpm"), 0.0),
            points_local_m=points_array,
            valid_points=len(local_points),
        )


class ColumnCarvingOccupancyMap:
    def __init__(
        self,
        *,
        resolution_m: float,
        map_size_m: float,
        robot_clear_radius_m: float,
        occupied_dilation_px: int,
        max_range_m: float,
    ) -> None:
        self.resolution_m = float(resolution_m)
        self.map_size_m = float(map_size_m)
        self.size_px = int(round(self.map_size_m / self.resolution_m))
        self.size_px = max(self.size_px, 64)
        self.center_col = self.size_px // 2
        self.center_row = self.size_px // 2
        self.state = np.full((self.size_px, self.size_px), -1, dtype=np.int16)
        self.robot_clear_radius_m = max(float(robot_clear_radius_m), 0.0)
        self.occupied_dilation_px = max(int(occupied_dilation_px), 0)
        self.max_range_m = float(max_range_m)

    def world_to_grid(self, x_m: float, y_m: float) -> tuple[int, int]:
        col = int(round(x_m / self.resolution_m + self.center_col))
        row = int(round(self.center_row - y_m / self.resolution_m))
        return row, col

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size_px and 0 <= col < self.size_px

    def transform_points(self, points_local_m: np.ndarray, pose: Pose2D) -> np.ndarray:
        c = math.cos(pose.yaw_rad)
        s = math.sin(pose.yaw_rad)
        x_world = pose.x_m + c * points_local_m[:, 0] - s * points_local_m[:, 1]
        y_world = pose.y_m + s * points_local_m[:, 0] + c * points_local_m[:, 1]
        return np.column_stack((x_world, y_world))

    def integrate_scan(self, *, points_local_m: np.ndarray, pose: Pose2D) -> bool:
        origin_row, origin_col = self.world_to_grid(pose.x_m, pose.y_m)
        if not self.in_bounds(origin_row, origin_col):
            return False

        if len(points_local_m) == 0:
            return False

        points_world = self.transform_points(points_local_m, pose)
        endpoint_cells: list[tuple[int, int]] = []

        for x_world, y_world in points_world:
            row, col = self.world_to_grid(float(x_world), float(y_world))
            if not self.in_bounds(row, col):
                continue
            ray = _bresenham_line(origin_row, origin_col, row, col)
            if len(ray) <= 1:
                endpoint_cells.append((row, col))
                continue
            free_rows = ray[:-1, 0]
            free_cols = ray[:-1, 1]
            self.state[free_rows, free_cols] = 0
            endpoint_cells.append((row, col))

        if not endpoint_cells:
            return False

        if self.robot_clear_radius_m > 0.0:
            self._clear_robot_disc(origin_row=origin_row, origin_col=origin_col)

        occ_mask = np.zeros_like(self.state, dtype=np.uint8)
        for row, col in endpoint_cells:
            occ_mask[row, col] = 1

        if self.occupied_dilation_px > 0:
            kernel_size = self.occupied_dilation_px * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            occ_mask = cv2.dilate(occ_mask, kernel, iterations=1)

        self.state[occ_mask > 0] = 100
        if self.robot_clear_radius_m > 0.0:
            self._clear_robot_disc(origin_row=origin_row, origin_col=origin_col)
        return True

    def _clear_robot_disc(self, *, origin_row: int, origin_col: int) -> None:
        radius_px = int(math.ceil(self.robot_clear_radius_m / self.resolution_m))
        if radius_px <= 0:
            return
        row_start = max(origin_row - radius_px, 0)
        row_end = min(origin_row + radius_px + 1, self.size_px)
        col_start = max(origin_col - radius_px, 0)
        col_end = min(origin_col + radius_px + 1, self.size_px)
        rows, cols = np.ogrid[row_start:row_end, col_start:col_end]
        mask = (rows - origin_row) ** 2 + (cols - origin_col) ** 2 <= radius_px**2
        patch = self.state[row_start:row_end, col_start:col_end]
        patch[mask] = 0

    def render_image(self) -> np.ndarray:
        image = np.full((self.size_px, self.size_px), 205, dtype=np.uint8)
        image[self.state == 0] = 255
        image[self.state == 100] = 0
        return image

    def render_debug_image(self, *, pose: Pose2D, points_local_m: np.ndarray) -> np.ndarray:
        image = self.render_image()
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        points_world = self.transform_points(points_local_m, pose)
        for x_world, y_world in points_world:
            row, col = self.world_to_grid(float(x_world), float(y_world))
            if self.in_bounds(row, col):
                rgb[row, col] = (0, 200, 255)

        origin_row, origin_col = self.world_to_grid(pose.x_m, pose.y_m)
        if self.in_bounds(origin_row, origin_col):
            cv2.circle(rgb, (origin_col, origin_row), 4, (0, 0, 255), thickness=-1)
            arrow_len_px = max(int(round(0.35 / self.resolution_m)), 6)
            tip_col = int(round(origin_col + math.cos(pose.yaw_rad) * arrow_len_px))
            tip_row = int(round(origin_row - math.sin(pose.yaw_rad) * arrow_len_px))
            cv2.arrowedLine(
                rgb,
                (origin_col, origin_row),
                (tip_col, tip_row),
                (0, 0, 255),
                thickness=2,
                tipLength=0.25,
            )
        return rgb

    def counts(self) -> dict[str, int]:
        occupied = int(np.count_nonzero(self.state == 100))
        free = int(np.count_nonzero(self.state == 0))
        unknown = int(self.state.size - occupied - free)
        return {"occupied": occupied, "free": free, "unknown": unknown}


def _bresenham_line(row0: int, col0: int, row1: int, col1: int) -> np.ndarray:
    points: list[tuple[int, int]] = []
    dcol = abs(col1 - col0)
    drow = -abs(row1 - row0)
    step_col = 1 if col0 < col1 else -1
    step_row = 1 if row0 < row1 else -1
    error = dcol + drow
    row, col = row0, col0
    while True:
        points.append((row, col))
        if row == row1 and col == col1:
            break
        e2 = 2 * error
        if e2 >= drow:
            error += drow
            col += step_col
        if e2 <= dcol:
            error += dcol
            row += step_row
    return np.asarray(points, dtype=np.int32)


class MapArtifactWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        occupancy_map: ColumnCarvingOccupancyMap,
        pose: Pose2D,
        scan_index: int,
        frame: ScanFrame,
        integrated: bool,
        motion_delta: MotionDelta,
    ) -> None:
        gray_image = occupancy_map.render_image()
        debug_image = occupancy_map.render_debug_image(pose=pose, points_local_m=frame.points_local_m)
        png_path = self.output_dir / "latest_map.png"
        debug_png_path = self.output_dir / "latest_map_debug.png"
        Image.fromarray(gray_image, mode="L").save(png_path)
        Image.fromarray(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB), mode="RGB").save(debug_png_path)

        metadata = {
            "schema": "sourccey.column_carving_map.v1",
            "scan_index": int(scan_index),
            "ts": float(frame.ts_wall_s),
            "rpm": float(frame.rpm),
            "valid_points": int(frame.valid_points),
            "resolution_m": float(occupancy_map.resolution_m),
            "map_size_m": float(occupancy_map.map_size_m),
            "size_px": int(occupancy_map.size_px),
            "pose": {
                "x_m": float(pose.x_m),
                "y_m": float(pose.y_m),
                "yaw_deg": float(_normalize_angle_deg(math.degrees(pose.yaw_rad))),
            },
            "integrated": bool(integrated),
            "motion": {
                "dt_s": float(motion_delta.dt_s),
                "vx_m_s": float(motion_delta.vx_m_s),
                "vy_m_s": float(motion_delta.vy_m_s),
                "wz_rad_s": float(motion_delta.wz_rad_s),
                "dx_local_m": float(motion_delta.dx_local_m),
                "dy_local_m": float(motion_delta.dy_local_m),
                "dyaw_deg": float(math.degrees(motion_delta.dyaw_rad)),
            },
            "counts": occupancy_map.counts(),
            "png_path": str(png_path),
            "debug_png_path": str(debug_png_path),
        }
        json_path = self.output_dir / "latest_map.json"
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _apply_motion_prior(
    pose: Pose2D,
    motion: MotionDelta,
    *,
    translation_scale: float,
    heading_scale: float,
) -> Pose2D:
    dx_local_m = motion.dx_local_m * float(translation_scale)
    dy_local_m = motion.dy_local_m * float(translation_scale)
    dyaw_rad = motion.dyaw_rad * float(heading_scale)
    c = math.cos(pose.yaw_rad)
    s = math.sin(pose.yaw_rad)
    dx_world_m = c * dx_local_m - s * dy_local_m
    dy_world_m = s * dx_local_m + c * dy_local_m
    return Pose2D(
        x_m=pose.x_m + dx_world_m,
        y_m=pose.y_m + dy_world_m,
        yaw_rad=_normalize_angle_rad(pose.yaw_rad + dyaw_rad),
    )


def _should_integrate_scan(
    *,
    frame: ScanFrame,
    motion: MotionDelta,
    min_points: int,
    max_linear_speed_m_s: float,
    max_angular_speed_rad_s: float,
) -> bool:
    if frame.valid_points < max(int(min_points), 1):
        return False
    if abs(float(motion.vx_m_s)) > float(max_linear_speed_m_s):
        return False
    if abs(float(motion.vy_m_s)) > float(max_linear_speed_m_s):
        return False
    if abs(float(motion.wz_rad_s)) > float(max_angular_speed_rad_s):
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dimos-style column-carving 2D LiDAR mapper for Sourccey."
    )
    parser.add_argument("--lidar-host", default="127.0.0.1", help="Host running scripts/ldlidar_stream_host.py.")
    parser.add_argument("--lidar-port", type=int, default=8765, help="TCP port exposed by the LiDAR host streamer.")
    parser.add_argument(
        "--slam-input-endpoint",
        default="tcp://127.0.0.1:5560",
        help="Optional ZMQ slam_input.v1 endpoint for motion/IMU prior. Set to '' to disable.",
    )
    parser.add_argument("--use-imu-heading", action="store_true", help="Prefer IMU gz over theta.vel for yaw integration.")
    parser.add_argument("--forward-angle-deg", type=float, default=180.0, help="Raw LiDAR angle that points forward on the robot.")
    parser.add_argument("--resolution-m", type=float, default=0.05, help="Occupancy grid resolution.")
    parser.add_argument("--map-size-m", type=float, default=12.0, help="Square map width/height in meters.")
    parser.add_argument("--min-range-m", type=float, default=0.05, help="Minimum usable LiDAR range.")
    parser.add_argument("--max-range-m", type=float, default=8.0, help="Maximum usable LiDAR range.")
    parser.add_argument("--min-confidence", type=int, default=5, help="Minimum LiDAR confidence to keep a point.")
    parser.add_argument("--max-points", type=int, default=480, help="Maximum scan points retained after downsampling.")
    parser.add_argument("--translation-scale", type=float, default=1.0, help="Scale applied to x/y velocity prior.")
    parser.add_argument("--heading-scale", type=float, default=1.0, help="Scale applied to yaw prior.")
    parser.add_argument("--min-points-for-integration", type=int, default=120, help="Minimum valid scan points before a frame enters the map.")
    parser.add_argument("--max-linear-speed-m-s", type=float, default=0.12, help="Only integrate scans below this |vx| and |vy|.")
    parser.add_argument("--max-angular-speed-rad-s", type=float, default=0.35, help="Only integrate scans below this |wz|.")
    parser.add_argument("--robot-clear-radius-m", type=float, default=0.18, help="Always keep this disc around the robot free.")
    parser.add_argument("--occupied-dilation-px", type=int, default=1, help="Endpoint dilation in pixels to make walls more continuous.")
    parser.add_argument("--save-every-scans", type=int, default=3, help="Write latest_map every N scans.")
    parser.add_argument("--log-every-scans", type=int, default=3, help="Print status every N scans.")
    parser.add_argument("--output-dir", default="artifacts/column_carving_maps", help="Directory for latest_map outputs.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    motion = MotionAccumulator(args.slam_input_endpoint, use_imu_heading=bool(args.use_imu_heading))
    motion.start()

    scan_reader = LidarScanReader(
        host=str(args.lidar_host),
        port=int(args.lidar_port),
        forward_angle_deg=float(args.forward_angle_deg),
        min_range_m=float(args.min_range_m),
        max_range_m=float(args.max_range_m),
        min_confidence=int(args.min_confidence),
        max_points=int(args.max_points),
    )

    occupancy_map = ColumnCarvingOccupancyMap(
        resolution_m=float(args.resolution_m),
        map_size_m=float(args.map_size_m),
        robot_clear_radius_m=float(args.robot_clear_radius_m),
        occupied_dilation_px=int(args.occupied_dilation_px),
        max_range_m=float(args.max_range_m),
    )
    writer = MapArtifactWriter(Path(args.output_dir))

    pose = Pose2D(0.0, 0.0, 0.0)
    scan_index = 0
    last_frame = ScanFrame(
        ts_wall_s=time.time(),
        rpm=0.0,
        points_local_m=np.zeros((0, 2), dtype=np.float32),
        valid_points=0,
    )
    last_motion = MotionDelta(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    try:
        for frame in scan_reader.scans():
            last_frame = frame
            motion_delta = motion.consume()
            last_motion = motion_delta
            pose = _apply_motion_prior(
                pose,
                motion_delta,
                translation_scale=float(args.translation_scale),
                heading_scale=float(args.heading_scale),
            )
            scan_index += 1

            integrated = _should_integrate_scan(
                frame=frame,
                motion=motion_delta,
                min_points=int(args.min_points_for_integration),
                max_linear_speed_m_s=float(args.max_linear_speed_m_s),
                max_angular_speed_rad_s=float(args.max_angular_speed_rad_s),
            )
            if integrated:
                integrated = occupancy_map.integrate_scan(points_local_m=frame.points_local_m, pose=pose)

            if scan_index % max(int(args.log_every_scans), 1) == 0:
                counts = occupancy_map.counts()
                logging.info(
                    "scan=%d points=%d rpm=%.1f pose=(%.2fm, %.2fm, %.1fdeg) integrated=%s vx=%.3f vy=%.3f wz=%.3f occ=%d free=%d unk=%d",
                    scan_index,
                    frame.valid_points,
                    frame.rpm,
                    pose.x_m,
                    pose.y_m,
                    _normalize_angle_deg(math.degrees(pose.yaw_rad)),
                    integrated,
                    motion_delta.vx_m_s,
                    motion_delta.vy_m_s,
                    motion_delta.wz_rad_s,
                    counts["occupied"],
                    counts["free"],
                    counts["unknown"],
                )

            if scan_index % max(int(args.save_every_scans), 1) == 0:
                writer.write(
                    occupancy_map=occupancy_map,
                    pose=pose,
                    scan_index=scan_index,
                    frame=frame,
                    integrated=integrated,
                    motion_delta=motion_delta,
                )
    except KeyboardInterrupt:
        logging.info("Stopping column-carving mapper...")
    finally:
        try:
            writer.write(
                occupancy_map=occupancy_map,
                pose=pose,
                scan_index=scan_index,
                frame=last_frame,
                integrated=False,
                motion_delta=last_motion,
            )
        except Exception:
            pass
        motion.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
