from __future__ import annotations

import argparse
import json
import math
import socket
import threading
import time
from dataclasses import dataclass
from hashlib import sha1

import cv2
import numpy as np
import serial


POINTS_PER_PACKET = 12
PACKET_LEN = 47
HEADER_BYTE = 0x54
VER_LEN_BYTE = 0x2C


@dataclass
class ScanPoint:
    angle_deg: float
    distance_m: float
    confidence: int


def _read_exact(ser: serial.Serial, length: int) -> bytes:
    payload = bytearray()
    while len(payload) < length:
        chunk = ser.read(length - len(payload))
        if not chunk:
            raise TimeoutError("Timed out waiting for LiDAR bytes.")
        payload.extend(chunk)
    return bytes(payload)


def _read_packet(ser: serial.Serial) -> bytes:
    while True:
        first = ser.read(1)
        if not first:
            raise TimeoutError("Timed out waiting for LiDAR header.")
        if first[0] != HEADER_BYTE:
            continue
        second = ser.read(1)
        if not second:
            raise TimeoutError("Timed out waiting for LiDAR length byte.")
        if second[0] != VER_LEN_BYTE:
            continue
        remainder = _read_exact(ser, PACKET_LEN - 2)
        return first + second + remainder


def _u16(lo: int, hi: int) -> int:
    return lo | (hi << 8)


def _parse_packet(packet: bytes) -> tuple[float, float, list[ScanPoint]]:
    speed_deg_s = float(_u16(packet[2], packet[3]))
    start_angle_deg = _u16(packet[4], packet[5]) / 100.0
    end_angle_deg = _u16(packet[42], packet[43]) / 100.0

    span_deg = end_angle_deg - start_angle_deg
    if span_deg < 0:
        span_deg += 360.0

    points: list[ScanPoint] = []
    for idx in range(POINTS_PER_PACKET):
        offset = 6 + idx * 3
        distance_mm = _u16(packet[offset], packet[offset + 1])
        confidence = int(packet[offset + 2])
        if distance_mm <= 0:
            continue

        if POINTS_PER_PACKET == 1:
            angle_deg = start_angle_deg
        else:
            angle_deg = (start_angle_deg + span_deg * idx / (POINTS_PER_PACKET - 1)) % 360.0
        points.append(
            ScanPoint(
                angle_deg=angle_deg,
                distance_m=distance_mm / 1000.0,
                confidence=confidence,
            )
        )
    return speed_deg_s, start_angle_deg, points


def _draw_scan(points: list[ScanPoint], *, size: int, max_distance_m: float, rpm: float) -> np.ndarray:
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    radius_px = center - 24

    for ring_fraction in (0.25, 0.5, 0.75, 1.0):
        cv2.circle(canvas, (center, center), int(radius_px * ring_fraction), (40, 40, 40), 1)
    cv2.line(canvas, (center, 16), (center, size - 16), (30, 30, 30), 1)
    cv2.line(canvas, (16, center), (size - 16, center), (30, 30, 30), 1)
    cv2.circle(canvas, (center, center), 4, (0, 180, 255), -1)

    for point in points:
        clipped_distance = min(max(point.distance_m, 0.0), max_distance_m)
        norm = clipped_distance / max(max_distance_m, 1e-6)
        px_radius = int(norm * radius_px)
        theta = math.radians(point.angle_deg - 90.0)
        x = int(center + px_radius * math.cos(theta))
        y = int(center + px_radius * math.sin(theta))
        confidence = max(0, min(point.confidence, 255))
        color = (0, confidence, 255 - min(confidence, 180))
        cv2.circle(canvas, (x, y), 2, color, -1)

    cv2.putText(
        canvas,
        f"LDLidar live view  points={len(points)}  rpm={rpm:.1f}",
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


class ScanServer:
    def __init__(self, bind_host: str, bind_port: int) -> None:
        self.bind_host = bind_host
        self.bind_port = bind_port
        self._clients: set[socket.socket] = set()
        self._lock = threading.Lock()
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.bind_host, self.bind_port))
        server.listen()
        self._server = server
        self._thread = threading.Thread(target=self._accept_loop, name="ldlidar_tcp_accept", daemon=True)
        self._thread.start()

    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                client, addr = self._server.accept()
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._lock:
                    self._clients.add(client)
                print(f"viewer connected from {addr[0]}:{addr[1]}")
            except OSError:
                return

    def broadcast_scan(
        self,
        *,
        rpm: float,
        points: list[ScanPoint],
        revolution_index: int,
        revolution_started_ts: float,
        revolution_completed_ts: float,
    ) -> None:
        emitted_ts = time.time()
        rounded_points = [[round(p.angle_deg, 3), round(p.distance_m, 4), int(p.confidence)] for p in points]
        digest = sha1(json.dumps(rounded_points, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
        message = {
            "ts": emitted_ts,
            "host_emitted_ts": emitted_ts,
            "rpm": rpm,
            "revolution_index": int(revolution_index),
            "revolution_started_ts": float(revolution_started_ts),
            "revolution_completed_ts": float(revolution_completed_ts),
            "point_digest": digest,
            "points": rounded_points,
        }
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        print(
            "[host] emitted revolution "
            f"idx={revolution_index} points={len(points)} rpm={rpm:.2f} digest={digest}"
        )
        dead_clients: list[socket.socket] = []
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.sendall(payload)
            except OSError:
                dead_clients.append(client)
        if dead_clients:
            with self._lock:
                for client in dead_clients:
                    self._clients.discard(client)
                    try:
                        client.close()
                    except OSError:
                        pass

    def close(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for client in clients:
            try:
                client.close()
            except OSError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick LD06/LD19-style LiDAR host streamer.")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port for the LiDAR.")
    parser.add_argument("--baud", type=int, default=230400, help="Serial baud rate.")
    parser.add_argument("--bind-host", default="0.0.0.0", help="TCP bind host.")
    parser.add_argument("--bind-port", type=int, default=8765, help="TCP port for remote viewers.")
    parser.add_argument("--max-distance-m", type=float, default=8.0, help="Viewer distance scale.")
    parser.add_argument("--canvas-size", type=int, default=900, help="Local viewer size in pixels.")
    parser.add_argument("--headless", action="store_true", help="Disable local OpenCV window.")
    args = parser.parse_args()

    server = ScanServer(bind_host=args.bind_host, bind_port=args.bind_port)
    server.start()
    print(
        f"Opening LiDAR on {args.port} @ {args.baud} baud. "
        f"Streaming scans on tcp://{args.bind_host}:{args.bind_port}"
    )

    ser = serial.Serial(args.port, args.baud, timeout=1.0)
    revolution_points: list[ScanPoint] = []
    previous_angle: float | None = None
    latest_rpm = 0.0
    revolution_index = 0
    revolution_started_ts = time.time()

    try:
        while True:
            packet = _read_packet(ser)
            packet_wall_ts = time.time()
            speed_deg_s, start_angle_deg, packet_points = _parse_packet(packet)
            latest_rpm = speed_deg_s / 360.0 * 60.0

            if previous_angle is not None and start_angle_deg + 2.0 < previous_angle and revolution_points:
                revolution_index += 1
                server.broadcast_scan(
                    rpm=latest_rpm,
                    points=revolution_points,
                    revolution_index=revolution_index,
                    revolution_started_ts=revolution_started_ts,
                    revolution_completed_ts=packet_wall_ts,
                )
                if not args.headless:
                    frame = _draw_scan(
                        revolution_points,
                        size=args.canvas_size,
                        max_distance_m=args.max_distance_m,
                        rpm=latest_rpm,
                    )
                    cv2.imshow("ldlidar_host_view", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                revolution_points = []
                revolution_started_ts = packet_wall_ts

            revolution_points.extend(packet_points)
            previous_angle = start_angle_deg
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        server.close()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
