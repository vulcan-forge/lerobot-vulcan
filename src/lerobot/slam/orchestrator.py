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

"""Rate-limited orchestration and health management for SLAM adapters."""

from __future__ import annotations

from dataclasses import replace
from time import monotonic_ns

from .adapter import SlamAdapter
from .types import SlamHealth, SlamInput, SlamOutput, SlamPose, SlamStatus


class SlamOrchestrator:
    """Owns adapter lifecycle and enforces target rate + health timeouts."""

    def __init__(
        self,
        adapter: SlamAdapter,
        target_hz: float = 15.0,
        healthy_timeout_s: float = 0.75,
        stale_timeout_s: float = 2.0,
        map_save_enabled: bool = False,
    ) -> None:
        self.adapter = adapter
        self.target_hz = max(target_hz, 1e-3)
        self.healthy_timeout_s = max(healthy_timeout_s, 1e-3)
        self.stale_timeout_s = max(stale_timeout_s, self.healthy_timeout_s)
        self.map_save_enabled = map_save_enabled
        self._min_submit_dt_ns = int(1e9 / self.target_hz)
        self._last_submit_ns = 0
        self._latest: SlamOutput | None = None

    def start(self) -> None:
        self.adapter.start()

    def submit(self, slam_input: SlamInput) -> None:
        if self._last_submit_ns and (
            slam_input.host_monotonic_ns - self._last_submit_ns
        ) < self._min_submit_dt_ns:
            return
        self._last_submit_ns = slam_input.host_monotonic_ns
        self.adapter.submit(slam_input)
        self._latest = self.adapter.get_latest()

    def get_latest(self) -> SlamOutput:
        now_ns = monotonic_ns()
        if self._latest is None:
            return SlamOutput(
                timestamp_ns=now_ns,
                pose=SlamPose(),
                health=SlamHealth(status=SlamStatus.NO_DATA, detail="No SLAM output yet"),
                backend="unknown",
            )

        age_s = max(0.0, (now_ns - self._latest.timestamp_ns) / 1e9)
        if age_s >= self.stale_timeout_s:
            return replace(
                self._latest,
                health=replace(
                    self._latest.health,
                    status=SlamStatus.STALE,
                    detail=f"SLAM output stale ({age_s:.2f}s)",
                ),
            )
        if age_s >= self.healthy_timeout_s:
            return replace(
                self._latest,
                health=replace(
                    self._latest.health,
                    status=SlamStatus.DEGRADED,
                    detail=f"SLAM output aging ({age_s:.2f}s)",
                ),
            )
        return self._latest

    def save_map(self) -> str | None:
        if not self.map_save_enabled:
            return None
        return self.adapter.save_map()

    def stop(self) -> None:
        self.adapter.stop()

