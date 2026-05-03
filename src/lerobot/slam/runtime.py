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

"""In-process SLAM runtime fed directly from client-side robot observations."""

from __future__ import annotations

from time import monotonic_ns
from typing import Any

from .orchestrator import SlamOrchestrator
from .types import CameraFrame, SlamInput, SlamOutput


class InProcessSlamRuntime:
    """Runs adapter orchestration in-process using rollout/client observations."""

    def __init__(
        self,
        orchestrator: SlamOrchestrator,
        stereo_left_key: str = "front_left",
        stereo_right_key: str = "front_right",
        source: str = "rollout_client",
    ) -> None:
        self.orchestrator = orchestrator
        self.stereo_left_key = stereo_left_key
        self.stereo_right_key = stereo_right_key
        self.source = source
        self._frame_ids: dict[str, int] = {
            self.stereo_left_key: 0,
            self.stereo_right_key: 0,
        }

    def start(self) -> None:
        self.orchestrator.start()

    def submit_observation(self, observation: dict[str, Any]) -> None:
        left = observation.get(self.stereo_left_key)
        right = observation.get(self.stereo_right_key)
        if left is None or right is None:
            return

        now_ns = monotonic_ns()
        self._frame_ids[self.stereo_left_key] += 1
        self._frame_ids[self.stereo_right_key] += 1

        slam_input = SlamInput(
            source=self.source,
            host_monotonic_ns=now_ns,
            base_velocity={
                "x.vel": float(observation.get("x.vel", 0.0)),
                "y.vel": float(observation.get("y.vel", 0.0)),
                "theta.vel": float(observation.get("theta.vel", 0.0)),
            },
            left=CameraFrame(
                name=self.stereo_left_key,
                frame_id=self._frame_ids[self.stereo_left_key],
                capture_monotonic_ns=now_ns,
                image=left,
            ),
            right=CameraFrame(
                name=self.stereo_right_key,
                frame_id=self._frame_ids[self.stereo_right_key],
                capture_monotonic_ns=now_ns,
                image=right,
            ),
        )
        self.orchestrator.submit(slam_input)

    def get_latest(self) -> SlamOutput:
        return self.orchestrator.get_latest()

    def stop(self) -> None:
        self.orchestrator.stop()

