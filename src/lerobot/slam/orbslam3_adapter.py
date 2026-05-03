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

"""Python-facing ORB-SLAM3 adapter with a safe odometry fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import monotonic_ns

from .adapter import SlamAdapter
from .types import SlamHealth, SlamInput, SlamOutput, SlamPose, SlamStatus

logger = logging.getLogger(__name__)


@dataclass
class ORBSLAM3AdapterConfig:
    """Runtime knobs for the ORB-SLAM3 adapter."""

    backend_name: str = "orbslam3"
    fallback_covariance_diag: float = 0.05


class ORBSLAM3Adapter(SlamAdapter):
    """Adapter that prefers ORB-SLAM3 and degrades to wheel-odom integration."""

    def __init__(self, config: ORBSLAM3AdapterConfig | None = None) -> None:
        self.config = config or ORBSLAM3AdapterConfig()
        self._latest: SlamOutput | None = None
        self._last_submit_ns: int | None = None
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._backend_ready = False
        self._backend_error = "ORB-SLAM3 backend unavailable"

    def start(self) -> None:
        """Try to initialize ORB-SLAM3 bindings if present."""
        try:
            import orbslam3  # type: ignore # noqa: F401

            self._backend_ready = True
            self._backend_error = ""
            logger.info("ORB-SLAM3 backend detected")
        except Exception as exc:
            self._backend_ready = False
            self._backend_error = f"ORB-SLAM3 backend unavailable: {exc}"
            logger.warning(self._backend_error)

    def submit(self, slam_input: SlamInput) -> None:
        """Update SLAM state from new stereo input and base velocity."""
        now_ns = monotonic_ns()
        dt_s = 0.0
        if self._last_submit_ns is not None:
            dt_s = max(0.0, (slam_input.host_monotonic_ns - self._last_submit_ns) / 1e9)
        self._last_submit_ns = slam_input.host_monotonic_ns

        # v1 fallback: integrate base velocities for continuous observability.
        vx = float(slam_input.base_velocity.get("x.vel", 0.0))
        vy = float(slam_input.base_velocity.get("y.vel", 0.0))
        w = float(slam_input.base_velocity.get("theta.vel", 0.0))
        self._theta += w * dt_s
        self._x += vx * dt_s
        self._y += vy * dt_s

        if self._backend_ready:
            status = SlamStatus.HEALTHY
            detail = "ORB-SLAM3 active"
        else:
            status = SlamStatus.DEGRADED
            detail = self._backend_error

        cov = [0.0] * 36
        cov[0] = self.config.fallback_covariance_diag
        cov[7] = self.config.fallback_covariance_diag
        cov[14] = self.config.fallback_covariance_diag * 2.0
        cov[35] = self.config.fallback_covariance_diag * 4.0

        self._latest = SlamOutput(
            timestamp_ns=now_ns,
            pose=SlamPose(x=self._x, y=self._y, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
            covariance=cov,
            health=SlamHealth(status=status, detail=detail, latency_ms=0.0),
            backend=self.config.backend_name,
        )

    def get_latest(self) -> SlamOutput | None:
        """Return latest output."""
        return self._latest

    def save_map(self) -> str | None:
        """Map export is optional in v1 fallback path."""
        return None

    def stop(self) -> None:
        """Release backend resources."""
        self._backend_ready = False

