from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np
import zmq

from lerobot.sensors.imu.types import IMUSample


class SlamInputPublisher:
    """Builds and publishes `slam_input.v1` packets from Sourccey observations."""

    def __init__(
        self,
        *,
        source_prefix: str = "sourccey_client",
        source_id: str,
        stereo_left_key: str,
        stereo_right_key: str,
        jpeg_quality: int,
        eye_only_mode: bool,
        publish_fps: float,
        resize_width: int | None,
        resize_height: int | None,
        extra_camera_keys: Iterable[str] = (),
        warn_log_interval_s: float = 5.0,
    ) -> None:
        self._source_prefix = source_prefix
        self._source_id = source_id
        self._stereo_left_key = stereo_left_key
        self._stereo_right_key = stereo_right_key
        self._jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self._eye_only_mode = bool(eye_only_mode)
        self._publish_interval_s = 0.0 if publish_fps <= 0 else 1.0 / float(publish_fps)
        self._resize_width = None if resize_width is None else int(resize_width)
        self._resize_height = None if resize_height is None else int(resize_height)
        self._extra_camera_keys = tuple(
            key.strip() for key in extra_camera_keys if isinstance(key, str) and key.strip()
        )
        self._warn_log_interval_s = warn_log_interval_s

        self._camera_frame_ids: dict[str, int] = {}
        self._warn_last_ts: dict[str, float] = {}
        self._warn_suppressed: dict[str, int] = {}
        self._last_publish_ts: float = 0.0
        self._first_packet_announced = False

    def publish(
        self,
        *,
        socket: zmq.Socket | None,
        observation: dict[str, Any],
        frames: dict[str, np.ndarray],
        imu_samples: Iterable[IMUSample] | None = None,
    ) -> None:
        now = time.monotonic()
        if self._publish_interval_s > 0 and (now - self._last_publish_ts) < self._publish_interval_s:
            return

        payload = self.build_packet(
            observation=observation,
            frames=frames,
            imu_samples=imu_samples,
        )
        if payload is None or socket is None:
            return
        try:
            socket.send(payload, flags=zmq.NOBLOCK)
            self._last_publish_ts = now
            if not self._first_packet_announced:
                self._first_packet_announced = True
                print(
                    f"[SLAM] First stereo packet published: {self._stereo_left_key}, {self._stereo_right_key}"
                )
        except zmq.Again:
            logging.debug("Dropping SLAM input packet, no subscriber connected.")
        except Exception as e:
            self.log_warning_throttled(
                "slam_send_failed",
                f"Failed to publish SLAM input packet: {e}",
            )

    def build_packet(
        self,
        *,
        observation: dict[str, Any],
        frames: dict[str, np.ndarray],
        imu_samples: Iterable[IMUSample] | None = None,
    ) -> bytes | None:
        left_key = self._stereo_left_key
        right_key = self._stereo_right_key
        required_keys = (left_key, right_key)

        missing_keys = [key for key in required_keys if key not in frames or frames[key] is None]
        if missing_keys:
            self.log_warning_throttled(
                "slam_missing_stereo",
                (
                    "Skipping SLAM input publish: missing required stereo frames "
                    f"{missing_keys} (configured left={left_key}, right={right_key})."
                ),
            )
            return None

        cameras_payload: dict[str, dict[str, Any]] = {}
        now_ns = time.monotonic_ns()
        camera_keys = (
            tuple(dict.fromkeys(required_keys + self._extra_camera_keys))
            if self._eye_only_mode
            else tuple(frames.keys())
        )
        for cam_name in self._extra_camera_keys:
            if not isinstance(frames.get(cam_name), np.ndarray):
                self.log_warning_throttled(
                    f"slam_missing_extra:{cam_name}",
                    (
                        f"SLAM auxiliary camera '{cam_name}' is configured but "
                        "is not delivering image data; clients that require it "
                        "will reject the stream."
                    ),
                )
        for cam_name in camera_keys:
            frame = frames.get(cam_name)
            if not isinstance(frame, np.ndarray):
                continue

            encode_frame = frame
            if self._resize_width is not None and self._resize_height is not None:
                encode_frame = cv2.resize(
                    frame,
                    (self._resize_width, self._resize_height),
                    interpolation=cv2.INTER_AREA,
                )

            ok, encoded_jpg = cv2.imencode(
                ".jpg",
                encode_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self._jpeg_quality)],
            )
            if not ok or encoded_jpg is None:
                self.log_warning_throttled(
                    f"slam_jpeg_encode_failed:{cam_name}",
                    f"Skipping SLAM camera frame '{cam_name}': JPEG encoding failed.",
                )
                if cam_name in required_keys:
                    return None
                continue

            frame_id = self._camera_frame_ids.get(cam_name, 0) + 1
            self._camera_frame_ids[cam_name] = frame_id

            cameras_payload[cam_name] = {
                "frame_id": frame_id,
                "capture_monotonic_ns": now_ns,
                "jpeg_b64": base64.b64encode(encoded_jpg.tobytes()).decode("ascii"),
            }

        if left_key not in cameras_payload or right_key not in cameras_payload:
            self.log_warning_throttled(
                "slam_missing_required_encoded",
                (
                    "Skipping SLAM input publish: required stereo cameras were not encoded "
                    f"(left={left_key in cameras_payload}, right={right_key in cameras_payload})."
                ),
            )
            return None

        packet = {
            "schema": "slam_input.v1",
            "source": f"{self._source_prefix}:{self._source_id}",
            "host_monotonic_ns": now_ns,
            "base_velocity": {
                "x.vel": float(observation.get("x.vel", 0.0)),
                "y.vel": float(observation.get("y.vel", 0.0)),
                "theta.vel": float(observation.get("theta.vel", 0.0)),
            },
            "stereo_left": left_key,
            "stereo_right": right_key,
            "imu_samples": _serialize_imu_samples(imu_samples),
            "cameras": cameras_payload,
        }
        return json.dumps(packet, separators=(",", ":")).encode("utf-8")

    def log_warning_throttled(self, key: str, message: str) -> None:
        now = time.monotonic()
        last_ts = self._warn_last_ts.get(key, 0.0)
        elapsed = now - last_ts
        if elapsed >= self._warn_log_interval_s:
            suppressed = self._warn_suppressed.get(key, 0)
            if suppressed > 0:
                logging.warning("%s (suppressed %d similar warnings)", message, suppressed)
            else:
                logging.warning("%s", message)
            self._warn_last_ts[key] = now
            self._warn_suppressed[key] = 0
        else:
            self._warn_suppressed[key] = self._warn_suppressed.get(key, 0) + 1


def _serialize_imu_samples(imu_samples: Iterable[IMUSample] | None) -> list[dict[str, float | int | None]]:
    if imu_samples is None:
        return []
    payload: list[dict[str, float | int | None]] = []
    for sample in imu_samples:
        if not isinstance(sample, IMUSample) or not sample.valid:
            continue
        payload.append(
            {
                "capture_monotonic_ns": int(sample.timestamp_ns),
                "ax": float(sample.accel_m_s2[0]),
                "ay": float(sample.accel_m_s2[1]),
                "az": float(sample.accel_m_s2[2]),
                "gx": float(sample.gyro_rad_s[0]),
                "gy": float(sample.gyro_rad_s[1]),
                "gz": float(sample.gyro_rad_s[2]),
                "mx": float(sample.mag_uT[0]),
                "my": float(sample.mag_uT[1]),
                "mz": float(sample.mag_uT[2]),
                "temperature_c": (None if sample.temperature_c is None else float(sample.temperature_c)),
            }
        )
    return payload
