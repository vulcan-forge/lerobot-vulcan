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

"""Typed SLAM interfaces shared across host, service, and rollout consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SlamStatus(str, Enum):
    """Operational health state of the SLAM runtime."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    STALE = "stale"
    NO_DATA = "no_data"


@dataclass
class CameraFrame:
    """One camera frame plus timing metadata in monotonic clock domain."""

    name: str
    frame_id: int
    capture_monotonic_ns: int
    image: Any


@dataclass
class SlamInput:
    """Input payload expected by SLAM adapters."""

    source: str
    host_monotonic_ns: int
    base_velocity: dict[str, float]
    left: CameraFrame
    right: CameraFrame


@dataclass
class SlamPose:
    """Robot pose in SLAM/world frame."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0


@dataclass
class SlamHealth:
    """Health and telemetry metadata."""

    status: SlamStatus
    detail: str = ""
    latency_ms: float = 0.0
    dropped_frames: int = 0


@dataclass
class SlamOutput:
    """Adapter output contract consumed by rollout observability hooks."""

    timestamp_ns: int
    pose: SlamPose
    covariance: list[float] = field(default_factory=lambda: [0.0] * 36)
    health: SlamHealth = field(default_factory=lambda: SlamHealth(status=SlamStatus.NO_DATA))
    backend: str = "unknown"
    map_artifact_path: str | None = None

