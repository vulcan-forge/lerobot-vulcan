from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class ScanFrame:
    ts: float
    rpm: float
    points: list[tuple[float, float, int]]
    raw_line: str
    received_wall_ts: float
    revolution_index: int | None = None
    host_emitted_ts: float | None = None
    revolution_started_ts: float | None = None
    revolution_completed_ts: float | None = None
    point_digest: str | None = None


def _normalize_angle_deg(angle_deg: float) -> float:
    value = math.fmod(angle_deg + 180.0, 360.0)
    if value < 0.0:
        value += 360.0
    return value - 180.0


def _scan_to_local_points(
    *,
    points: list[tuple[float, float, int]],
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    invert_lateral_axis: bool,
    max_distance_m: float,
    min_confidence: int,
    min_range_m: float,
) -> np.ndarray:
    xy_points: list[tuple[float, float]] = []
    for angle_deg, distance_m, confidence in points:
        distance = float(distance_m)
        conf = int(confidence)
        if conf < int(min_confidence):
            continue
        if not math.isfinite(distance) or distance <= 0.0:
            continue
        if distance < float(min_range_m) or distance > float(max_distance_m):
            continue
        delta_deg = _normalize_angle_deg(float(angle_deg) - float(forward_angle_deg))
        if abs(delta_deg) > float(valid_angle_half_width_deg):
            continue
        theta = math.radians(delta_deg)
        forward_m = distance * math.cos(theta)
        lateral_m = distance * math.sin(theta)
        if invert_lateral_axis:
            lateral_m = -lateral_m
        xy_points.append((forward_m, lateral_m))
    if not xy_points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(xy_points, dtype=np.float32)


def _point_signature(points_xy: np.ndarray) -> dict[str, object]:
    if points_xy.size == 0:
        return {
            "digest": "empty",
            "point_count": 0,
            "min_xy": [0.0, 0.0],
            "max_xy": [0.0, 0.0],
            "span_xy": [0.0, 0.0],
        }
    rounded = np.round(points_xy.astype(np.float32, copy=False), 3).astype(np.float32, copy=False)
    min_xy = np.min(rounded, axis=0)
    max_xy = np.max(rounded, axis=0)
    digest = hashlib.sha1(rounded.tobytes()).hexdigest()[:16]
    return {
        "digest": digest,
        "point_count": int(len(rounded)),
        "min_xy": [round(float(min_xy[0]), 4), round(float(min_xy[1]), 4)],
        "max_xy": [round(float(max_xy[0]), 4), round(float(max_xy[1]), 4)],
        "span_xy": [round(float(max_xy[0] - min_xy[0]), 4), round(float(max_xy[1] - min_xy[1]), 4)],
    }


class DirectLidarFeed:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_frame: ScanFrame | None = None
        self._latest_frame_id = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ldlidar_direct_feed", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def latest(self) -> tuple[int, ScanFrame | None]:
        with self._lock:
            return self._latest_frame_id, self._latest_frame

    def wait_for_frame_after(
        self,
        *,
        after_frame_id: int,
        timeout_s: float,
        min_frame_advances: int = 1,
        armed_wall_ts: float | None = None,
    ) -> tuple[int, ScanFrame | None]:
        deadline = time.monotonic() + float(timeout_s)
        target_frame_id = int(after_frame_id) + max(1, int(min_frame_advances))
        while time.monotonic() < deadline and not self._stop.is_set():
            frame_id, frame = self.latest()
            if frame is not None and frame_id >= target_frame_id:
                if armed_wall_ts is not None and frame.received_wall_ts < armed_wall_ts:
                    time.sleep(0.01)
                    continue
                return frame_id, frame
            time.sleep(0.01)
        # Never return a cached frame as though it satisfied the requested
        # advancement. Reusing one revolution corrupts multi-scan consensus and
        # makes a healthy-but-delayed consumer look like independent evidence.
        frame_id, _frame = self.latest()
        return frame_id, None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with socket.create_connection((self.host, self.port), timeout=5.0) as sock:
                    file_obj = sock.makefile("r", encoding="utf-8")
                    print(f"[feed] connected to tcp://{self.host}:{self.port}")
                    for line in file_obj:
                        if self._stop.is_set():
                            return
                        payload = json.loads(line)
                        raw_points = payload.get("points", [])
                        points = [
                            (float(angle_deg), float(distance_m), int(confidence))
                            for angle_deg, distance_m, confidence in raw_points
                        ]
                        frame = ScanFrame(
                            ts=float(payload.get("ts", time.time())),
                            rpm=float(payload.get("rpm", 0.0)),
                            points=points,
                            raw_line=line.rstrip("\n"),
                            received_wall_ts=time.time(),
                            revolution_index=(
                                int(payload["revolution_index"])
                                if payload.get("revolution_index") is not None
                                else None
                            ),
                            host_emitted_ts=(
                                float(payload["host_emitted_ts"])
                                if payload.get("host_emitted_ts") is not None
                                else None
                            ),
                            revolution_started_ts=(
                                float(payload["revolution_started_ts"])
                                if payload.get("revolution_started_ts") is not None
                                else None
                            ),
                            revolution_completed_ts=(
                                float(payload["revolution_completed_ts"])
                                if payload.get("revolution_completed_ts") is not None
                                else None
                            ),
                            point_digest=str(payload["point_digest"]) if payload.get("point_digest") else None,
                        )
                        with self._lock:
                            self._latest_frame = frame
                            self._latest_frame_id += 1
            except Exception as exc:
                print(f"[feed] reconnecting after error: {exc}")
                time.sleep(0.5)


def _save_snapshot(
    *,
    output_dir: Path,
    request_index: int,
    frame_id: int,
    frame: ScanFrame,
    local_points_xy: np.ndarray,
    capture_config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"snapshot_{request_index:03d}"
    json_path = output_dir / f"{stem}.json"
    raw_json_path = output_dir / f"{stem}_raw.json"
    npy_path = output_dir / f"{stem}_local.npy"

    np.save(npy_path, local_points_xy.astype(np.float32, copy=False))
    raw_json_path.write_text(frame.raw_line + "\n", encoding="utf-8")

    signature = _point_signature(local_points_xy)
    metadata = {
        "schema": "sourccey.direct_snapshot.v1",
        "request_index": int(request_index),
        "frame_id": int(frame_id),
        "scan_ts": float(frame.ts),
        "rpm": float(frame.rpm),
        "received_wall_ts": float(frame.received_wall_ts),
        "host_revolution_index": frame.revolution_index,
        "host_emitted_ts": frame.host_emitted_ts,
        "host_revolution_started_ts": frame.revolution_started_ts,
        "host_revolution_completed_ts": frame.revolution_completed_ts,
        "host_point_digest": frame.point_digest,
        "raw_point_count": int(len(frame.points)),
        "local_point_count": int(len(local_points_xy)),
        "local_signature": signature,
        "capture_config": capture_config,
        "local_npy_path": str(npy_path),
        "raw_json_path": str(raw_json_path),
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        "[snapshot] saved "
        f"#{request_index} frame_id={frame_id} host_rev={frame.revolution_index} "
        f"raw_points={len(frame.points)} local_points={len(local_points_xy)} "
        f"digest={signature['digest']} host_digest={frame.point_digest}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct raw LiDAR snapshot saver.")
    parser.add_argument("--host", default="192.168.1.237")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--output-dir",
        default="artifacts/direct_lidar_snapshots",
        help="Where to save raw snapshot artifacts.",
    )
    parser.add_argument("--forward-angle-deg", type=float, default=270.0)
    parser.add_argument("--valid-angle-half-width-deg", type=float, default=180.0)
    parser.add_argument("--invert-lateral-axis", action="store_true", default=True)
    parser.add_argument("--no-invert-lateral-axis", dest="invert_lateral_axis", action="store_false")
    parser.add_argument("--max-distance-m", type=float, default=8.0)
    parser.add_argument("--min-range-m", type=float, default=0.03)
    parser.add_argument("--min-confidence", type=int, default=0)
    parser.add_argument(
        "--fresh-frame-timeout-s",
        type=float,
        default=3.0,
        help="How long to wait for a new LiDAR revolution after pressing Enter.",
    )
    parser.add_argument(
        "--fresh-frame-advances",
        type=int,
        default=1,
        help="How many new frame ids must arrive after pressing Enter before saving.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                for nested in child.rglob("*"):
                    if nested.is_file():
                        nested.unlink()
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_dir():
                        nested.rmdir()
                child.rmdir()
    output_dir.mkdir(parents=True, exist_ok=True)

    feed = DirectLidarFeed(args.host, int(args.port))
    feed.start()

    print("[snapshot] direct raw client started")
    print("[snapshot] press Enter to arm capture on the NEXT fresh frame, or type q then Enter to quit")
    print(f"[snapshot] output_dir={output_dir.resolve()}")

    request_index = 0
    last_saved_frame_id = -1
    capture_config = {
        "forward_angle_deg": float(args.forward_angle_deg),
        "valid_angle_half_width_deg": float(args.valid_angle_half_width_deg),
        "invert_lateral_axis": bool(args.invert_lateral_axis),
        "max_distance_m": float(args.max_distance_m),
        "min_range_m": float(args.min_range_m),
        "min_confidence": int(args.min_confidence),
    }

    try:
        while True:
            user_input = input("> ").strip().lower()
            if user_input in {"q", "quit", "exit"}:
                break
            before_frame_id, before_frame = feed.latest()
            if before_frame is None:
                print("[snapshot] no frame received yet")
                continue
            armed_wall_ts = time.time()
            print(
                "[snapshot] capture armed: "
                f"waiting for frame_id > {before_frame_id} "
                f"(min advances={int(args.fresh_frame_advances)})"
            )
            frame_id, frame = feed.wait_for_frame_after(
                after_frame_id=before_frame_id,
                timeout_s=float(args.fresh_frame_timeout_s),
                min_frame_advances=int(args.fresh_frame_advances),
                armed_wall_ts=armed_wall_ts,
            )
            if frame is None:
                print("[snapshot] no frame available after waiting")
                continue
            if frame_id <= before_frame_id:
                print(
                    "[snapshot] warning: timed out waiting for a fresh frame; "
                    f"still using frame_id={frame_id}"
                )
            if frame_id == last_saved_frame_id:
                print(
                    "[snapshot] warning: about to save the same frame id as last time "
                    f"(frame_id={frame_id})"
                )
            request_index += 1
            local_points_xy = _scan_to_local_points(
                points=frame.points,
                forward_angle_deg=float(args.forward_angle_deg),
                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                invert_lateral_axis=bool(args.invert_lateral_axis),
                max_distance_m=float(args.max_distance_m),
                min_confidence=int(args.min_confidence),
                min_range_m=float(args.min_range_m),
            )
            _save_snapshot(
                output_dir=output_dir,
                request_index=request_index,
                frame_id=frame_id,
                frame=frame,
                local_points_xy=local_points_xy,
                capture_config=capture_config,
            )
            last_saved_frame_id = frame_id
    finally:
        feed.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
