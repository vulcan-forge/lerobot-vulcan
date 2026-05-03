#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dedicated transport codecs and sockets for SLAM input/output streams."""

from __future__ import annotations

import base64
import json
from dataclasses import asdict
from time import monotonic_ns
from typing import Any

import cv2
import numpy as np
import zmq

from .types import CameraFrame, SlamHealth, SlamInput, SlamOutput, SlamPose, SlamStatus

SLAM_INPUT_SCHEMA = "slam_input.v1"
SLAM_OUTPUT_SCHEMA = "slam_output.v1"

DEFAULT_SLAM_INPUT_ENDPOINT = "tcp://127.0.0.1:5560"
DEFAULT_SLAM_OUTPUT_ENDPOINT = "tcp://127.0.0.1:5561"


def _encode_jpeg(image: np.ndarray, quality: int = 80) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode SLAM frame")
    return encoded.tobytes()


def _decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode SLAM JPEG frame")
    return frame


def build_slam_input_packet(
    *,
    source: str,
    host_monotonic_ns: int,
    base_velocity: dict[str, float],
    frame_ids: dict[str, int],
    capture_monotonic_ns: dict[str, int],
    camera_frames: dict[str, np.ndarray],
    stereo_left: str = "front_left",
    stereo_right: str = "front_right",
    jpeg_quality: int = 80,
) -> bytes:
    """Serialize host-side SLAM input data as JSON bytes."""
    cameras: dict[str, dict[str, Any]] = {}
    for name in (stereo_left, stereo_right):
        frame = camera_frames.get(name)
        if frame is None:
            continue
        payload = _encode_jpeg(frame, quality=jpeg_quality)
        cameras[name] = {
            "frame_id": int(frame_ids.get(name, 0)),
            "capture_monotonic_ns": int(capture_monotonic_ns.get(name, host_monotonic_ns)),
            "jpeg_b64": base64.b64encode(payload).decode("ascii"),
        }
    packet = {
        "schema": SLAM_INPUT_SCHEMA,
        "source": source,
        "host_monotonic_ns": int(host_monotonic_ns),
        "base_velocity": {
            "x.vel": float(base_velocity.get("x.vel", 0.0)),
            "y.vel": float(base_velocity.get("y.vel", 0.0)),
            "theta.vel": float(base_velocity.get("theta.vel", 0.0)),
        },
        "stereo_left": stereo_left,
        "stereo_right": stereo_right,
        "cameras": cameras,
    }
    return json.dumps(packet, separators=(",", ":")).encode("utf-8")


def parse_slam_input_packet(payload: bytes) -> SlamInput:
    """Deserialize transport packet into typed SLAM input."""
    data = json.loads(payload.decode("utf-8"))
    if data.get("schema") != SLAM_INPUT_SCHEMA:
        raise ValueError(f"Unsupported SLAM input schema: {data.get('schema')}")

    stereo_left = data["stereo_left"]
    stereo_right = data["stereo_right"]
    cameras = data["cameras"]

    left_payload = base64.b64decode(cameras[stereo_left]["jpeg_b64"])
    right_payload = base64.b64decode(cameras[stereo_right]["jpeg_b64"])
    left_frame = CameraFrame(
        name=stereo_left,
        frame_id=int(cameras[stereo_left]["frame_id"]),
        capture_monotonic_ns=int(cameras[stereo_left]["capture_monotonic_ns"]),
        image=_decode_jpeg(left_payload),
    )
    right_frame = CameraFrame(
        name=stereo_right,
        frame_id=int(cameras[stereo_right]["frame_id"]),
        capture_monotonic_ns=int(cameras[stereo_right]["capture_monotonic_ns"]),
        image=_decode_jpeg(right_payload),
    )
    return SlamInput(
        source=str(data.get("source", "unknown")),
        host_monotonic_ns=int(data["host_monotonic_ns"]),
        base_velocity={
            "x.vel": float(data["base_velocity"].get("x.vel", 0.0)),
            "y.vel": float(data["base_velocity"].get("y.vel", 0.0)),
            "theta.vel": float(data["base_velocity"].get("theta.vel", 0.0)),
        },
        left=left_frame,
        right=right_frame,
    )


def serialize_slam_output(output: SlamOutput) -> bytes:
    data = {
        "schema": SLAM_OUTPUT_SCHEMA,
        "timestamp_ns": int(output.timestamp_ns),
        "pose": asdict(output.pose),
        "covariance": [float(x) for x in output.covariance],
        "health": {
            "status": output.health.status.value,
            "detail": output.health.detail,
            "latency_ms": float(output.health.latency_ms),
            "dropped_frames": int(output.health.dropped_frames),
        },
        "backend": output.backend,
        "map_artifact_path": output.map_artifact_path,
    }
    return json.dumps(data, separators=(",", ":")).encode("utf-8")


def parse_slam_output(payload: bytes) -> SlamOutput:
    data = json.loads(payload.decode("utf-8"))
    if data.get("schema") != SLAM_OUTPUT_SCHEMA:
        raise ValueError(f"Unsupported SLAM output schema: {data.get('schema')}")
    pose_data = data.get("pose", {})
    health_data = data.get("health", {})
    return SlamOutput(
        timestamp_ns=int(data.get("timestamp_ns", monotonic_ns())),
        pose=SlamPose(
            x=float(pose_data.get("x", 0.0)),
            y=float(pose_data.get("y", 0.0)),
            z=float(pose_data.get("z", 0.0)),
            qx=float(pose_data.get("qx", 0.0)),
            qy=float(pose_data.get("qy", 0.0)),
            qz=float(pose_data.get("qz", 0.0)),
            qw=float(pose_data.get("qw", 1.0)),
        ),
        covariance=[float(x) for x in data.get("covariance", [0.0] * 36)],
        health=SlamHealth(
            status=SlamStatus(health_data.get("status", SlamStatus.NO_DATA.value)),
            detail=str(health_data.get("detail", "")),
            latency_ms=float(health_data.get("latency_ms", 0.0)),
            dropped_frames=int(health_data.get("dropped_frames", 0)),
        ),
        backend=str(data.get("backend", "unknown")),
        map_artifact_path=data.get("map_artifact_path"),
    )


class RolloutSlamSubscriber:
    """Non-blocking rollout-side SLAM output subscriber."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self._context = zmq.Context.instance()
        self._socket: zmq.Socket | None = None
        self._latest: SlamOutput | None = None

    def start(self) -> None:
        if self._socket is not None:
            return
        sock = self._context.socket(zmq.SUB)
        sock.connect(self.endpoint)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket = sock

    def poll_latest(self) -> SlamOutput | None:
        if self._socket is None:
            return self._latest
        while True:
            try:
                payload = self._socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            self._latest = parse_slam_output(payload)
        return self._latest

    def stop(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

