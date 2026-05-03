#!/usr/bin/env python

from __future__ import annotations

import numpy as np

from lerobot.slam.adapter import SlamAdapter
from lerobot.slam.orchestrator import SlamOrchestrator
from lerobot.slam.runtime import InProcessSlamRuntime
from lerobot.slam.types import SlamHealth, SlamOutput, SlamPose, SlamStatus


class MockAdapter(SlamAdapter):
    def __init__(self) -> None:
        self.latest: SlamOutput | None = None
        self.started = False
        self.stopped = False
        self.submit_count = 0

    def start(self) -> None:
        self.started = True

    def submit(self, slam_input) -> None:
        self.submit_count += 1
        self.latest = SlamOutput(
            timestamp_ns=slam_input.host_monotonic_ns,
            pose=SlamPose(x=0.5, y=1.5, z=0.0),
            health=SlamHealth(status=SlamStatus.HEALTHY, detail="mock"),
            backend="mock",
        )

    def get_latest(self):
        return self.latest

    def save_map(self):
        return None

    def stop(self) -> None:
        self.stopped = True


def test_inprocess_runtime_uses_observation_stream() -> None:
    adapter = MockAdapter()
    runtime = InProcessSlamRuntime(
        orchestrator=SlamOrchestrator(adapter=adapter, target_hz=1000.0),
        stereo_left_key="front_left",
        stereo_right_key="front_right",
    )
    runtime.start()
    assert adapter.started

    obs = {
        "front_left": np.zeros((8, 8, 3), dtype=np.uint8),
        "front_right": np.zeros((8, 8, 3), dtype=np.uint8),
        "x.vel": 0.1,
        "y.vel": 0.0,
        "theta.vel": 0.0,
    }
    runtime.submit_observation(obs)
    out = runtime.get_latest()
    assert out.health.status == SlamStatus.HEALTHY
    assert out.pose.x == 0.5
    assert adapter.submit_count == 1

    runtime.stop()
    assert adapter.stopped

