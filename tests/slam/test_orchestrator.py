#!/usr/bin/env python

from __future__ import annotations

from dataclasses import replace
from time import monotonic_ns
import time

from lerobot.slam.adapter import SlamAdapter
from lerobot.slam.orchestrator import SlamOrchestrator
from lerobot.slam.types import CameraFrame, SlamHealth, SlamInput, SlamOutput, SlamPose, SlamStatus


class MockAdapter(SlamAdapter):
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.submit_count = 0
        self.latest: SlamOutput | None = None

    def start(self) -> None:
        self.started = True

    def submit(self, slam_input: SlamInput) -> None:
        self.submit_count += 1
        self.latest = SlamOutput(
            timestamp_ns=slam_input.host_monotonic_ns,
            pose=SlamPose(x=1.0, y=2.0, z=0.0),
            health=SlamHealth(status=SlamStatus.HEALTHY, detail="mock"),
            backend="mock",
        )

    def get_latest(self) -> SlamOutput | None:
        return self.latest

    def save_map(self) -> str | None:
        return None

    def stop(self) -> None:
        self.stopped = True


def _slam_input(ts_ns: int) -> SlamInput:
    dummy = CameraFrame(name="front_left", frame_id=1, capture_monotonic_ns=ts_ns, image=None)
    return SlamInput(
        source="test",
        host_monotonic_ns=ts_ns,
        base_velocity={"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0},
        left=dummy,
        right=replace(dummy, name="front_right", frame_id=2),
    )


def test_orchestrator_rate_limit_and_lifecycle() -> None:
    adapter = MockAdapter()
    orchestrator = SlamOrchestrator(adapter=adapter, target_hz=10.0)
    orchestrator.start()
    assert adapter.started

    t0 = monotonic_ns()
    orchestrator.submit(_slam_input(t0))
    orchestrator.submit(_slam_input(t0 + 1_000_000))  # 1ms later, should be throttled at 10Hz
    orchestrator.submit(_slam_input(t0 + 150_000_000))  # 150ms later
    assert adapter.submit_count == 2

    latest = orchestrator.get_latest()
    assert latest.health.status == SlamStatus.HEALTHY
    assert latest.pose.x == 1.0

    orchestrator.stop()
    assert adapter.stopped


def test_orchestrator_health_transitions() -> None:
    adapter = MockAdapter()
    orchestrator = SlamOrchestrator(
        adapter=adapter,
        target_hz=15.0,
        healthy_timeout_s=0.01,
        stale_timeout_s=0.02,
    )
    orchestrator.start()
    now = monotonic_ns()
    orchestrator.submit(_slam_input(now))

    # Wait beyond stale timeout and verify stale transition.
    time.sleep(0.03)
    stale = orchestrator.get_latest()
    assert stale.health.status == SlamStatus.STALE
