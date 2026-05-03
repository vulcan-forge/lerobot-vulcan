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

"""Rollout-side SLAM bridge with strict boundaries and proxy wrappers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.slam.orchestrator import SlamOrchestrator
from lerobot.slam.orbslam3_adapter import ORBSLAM3Adapter
from lerobot.slam.runtime import InProcessSlamRuntime
from lerobot.slam.types import SlamOutput, SlamStatus
from lerobot.utils.import_utils import _zmq_available

logger = logging.getLogger(__name__)


@dataclass
class RolloutSlamConfig:
    """Optional rollout-side SLAM config (in-process by default)."""

    enabled: bool = False
    backend: str = "orbslam3"
    mode: str = "live_localization"
    source_mode: str = "client_observation"
    remote_endpoint: str = "tcp://127.0.0.1:5561"
    stereo_left_key: str = "front_left"
    stereo_right_key: str = "front_right"
    target_hz: float = 15.0
    map_save_enabled: bool = False
    healthy_timeout_s: float = 0.75
    stale_timeout_s: float = 2.0

    def __post_init__(self):
        if self.mode not in ("live_localization",):
            raise ValueError(f"Unsupported slam.mode '{self.mode}'. Expected 'live_localization'.")
        if self.source_mode not in ("client_observation", "remote_endpoint"):
            raise ValueError(
                f"Unsupported slam.source_mode '{self.source_mode}'. "
                "Expected one of: client_observation, remote_endpoint."
            )
        if self.target_hz <= 0:
            raise ValueError("--slam.target_hz must be > 0")
        if self.healthy_timeout_s <= 0:
            raise ValueError("--slam.healthy_timeout_s must be > 0")
        if self.stale_timeout_s <= 0:
            raise ValueError("--slam.stale_timeout_s must be > 0")
        if self.stale_timeout_s < self.healthy_timeout_s:
            raise ValueError("--slam.stale_timeout_s must be >= --slam.healthy_timeout_s")


class RolloutSlamSession:
    """Owns rollout SLAM runtime/subscriber lifecycle and telemetry formatting."""

    def __init__(
        self,
        runtime: InProcessSlamRuntime | None = None,
        subscriber: Any | None = None,
    ) -> None:
        self.runtime = runtime
        self.subscriber = subscriber
        self._last_status: str | None = None
        self._last_log_ts: float = 0.0

    def observe(self, observation: dict[str, Any]) -> None:
        if self.runtime is not None:
            self.runtime.submit_observation(observation)
        self._log_health()

    def get_latest(self) -> SlamOutput | None:
        if self.runtime is not None:
            return self.runtime.get_latest()
        if self.subscriber is not None:
            return self.subscriber.poll_latest()
        return None

    def stop(self) -> None:
        if self.runtime is not None:
            self.runtime.stop()
        if self.subscriber is not None:
            self.subscriber.stop()

    def _log_health(self) -> None:
        output = self.get_latest()
        if output is None:
            return
        status = output.health.status.value
        now = time.perf_counter()
        should_log = status != self._last_status or (now - self._last_log_ts) >= 5.0
        if should_log:
            logger.info(
                "SLAM health: %s | backend=%s | pose=(%.3f, %.3f, %.3f) | detail=%s",
                status,
                output.backend,
                output.pose.x,
                output.pose.y,
                output.pose.z,
                output.health.detail,
            )
            self._last_status = status
            self._last_log_ts = now

    def frame_fields(self) -> dict[str, np.ndarray]:
        output = self.get_latest()
        if output is None:
            return {}
        status_to_code = {
            SlamStatus.HEALTHY.value: 0,
            SlamStatus.DEGRADED.value: 1,
            SlamStatus.STALE.value: 2,
            SlamStatus.NO_DATA.value: 3,
        }
        status_code = status_to_code.get(output.health.status.value, 3)
        return {
            "slam.pose.xyz": np.array([output.pose.x, output.pose.y, output.pose.z], dtype=np.float32),
            "slam.pose.quat": np.array(
                [output.pose.qx, output.pose.qy, output.pose.qz, output.pose.qw], dtype=np.float32
            ),
            "slam.health.status": np.array([status_code], dtype=np.int64),
            "slam.health.latency_ms": np.array([output.health.latency_ms], dtype=np.float32),
        }


class SlamAwareRobotProxy:
    """Robot proxy that feeds observations to SLAM without touching strategy code."""

    def __init__(self, robot: Any, slam_session: RolloutSlamSession) -> None:
        self._robot = robot
        self._slam_session = slam_session

    def get_observation(self) -> dict[str, Any]:
        obs = self._robot.get_observation()
        self._slam_session.observe(obs)
        return obs

    def send_action(self, action: dict[str, Any]) -> Any:
        return self._robot.send_action(action)

    def disconnect(self) -> None:
        try:
            self._slam_session.stop()
        finally:
            self._robot.disconnect()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._robot, name)


class SlamAwareDatasetProxy:
    """Dataset proxy that appends SLAM telemetry fields on every frame write."""

    def __init__(self, dataset: Any, slam_session: RolloutSlamSession) -> None:
        self._dataset = dataset
        self._slam_session = slam_session

    def add_frame(self, frame: dict[str, Any]) -> None:
        merged = dict(frame)
        merged.update(self._slam_session.frame_fields())
        self._dataset.add_frame(merged)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)


def add_rollout_slam_dataset_features(dataset_features: dict[str, Any]) -> None:
    """Mutates dataset feature schema in-place with SLAM telemetry channels."""
    dataset_features["slam.pose.xyz"] = {"dtype": "float32", "shape": (3,), "names": None}
    dataset_features["slam.pose.quat"] = {"dtype": "float32", "shape": (4,), "names": None}
    dataset_features["slam.health.status"] = {"dtype": "int64", "shape": (1,), "names": None}
    dataset_features["slam.health.latency_ms"] = {"dtype": "float32", "shape": (1,), "names": None}


def build_rollout_slam_session(cfg: RolloutSlamConfig) -> RolloutSlamSession:
    """Build and start a rollout SLAM session according to config."""
    if cfg.source_mode == "client_observation":
        logger.info(
            "Starting in-process SLAM runtime (backend=%s, target_hz=%.1f, stereo=(%s,%s))",
            cfg.backend,
            cfg.target_hz,
            cfg.stereo_left_key,
            cfg.stereo_right_key,
        )
        runtime = InProcessSlamRuntime(
            orchestrator=SlamOrchestrator(
                adapter=ORBSLAM3Adapter(),
                target_hz=cfg.target_hz,
                healthy_timeout_s=cfg.healthy_timeout_s,
                stale_timeout_s=cfg.stale_timeout_s,
                map_save_enabled=cfg.map_save_enabled,
            ),
            stereo_left_key=cfg.stereo_left_key,
            stereo_right_key=cfg.stereo_right_key,
            source="rollout_client",
        )
        runtime.start()
        return RolloutSlamSession(runtime=runtime)

    if not _zmq_available:
        raise ImportError("SLAM rollout telemetry requires pyzmq. Install with `lerobot[pyzmq-dep]`.")
    from lerobot.slam.transport import RolloutSlamSubscriber

    logger.info("Starting SLAM telemetry subscriber (endpoint=%s)", cfg.remote_endpoint)
    subscriber = RolloutSlamSubscriber(cfg.remote_endpoint)
    subscriber.start()
    return RolloutSlamSession(subscriber=subscriber)
