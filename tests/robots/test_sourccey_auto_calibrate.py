#!/usr/bin/env python

import pytest

import lerobot.robots.sourccey.sourccey.sourccey.sourccey as sourccey_module
from lerobot.robots.sourccey.sourccey.sourccey.sourccey import Sourccey
from lerobot.teleoperators.sourccey.sourccey.bi_sourccey_leader.bi_sourccey_leader import (
    BiSourcceyLeader,
)


class _DummyArm:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[dict] = []

    def auto_calibrate(self, **kwargs) -> None:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error


class _DummyZCalibrator:
    def __init__(self):
        self.calls = 0

    def auto_calibrate(self) -> None:
        self.calls += 1


class _DummyZActuator:
    def __init__(self):
        self.calibrator = _DummyZCalibrator()


def test_sourccey_auto_calibrate_raises_when_arm_thread_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sourccey_module.time, "sleep", lambda _seconds: None)

    robot = Sourccey.__new__(Sourccey)
    robot.left_arm = _DummyArm(RuntimeError("left arm failed"))
    robot.right_arm = _DummyArm()
    robot.z_actuator = _DummyZActuator()

    with pytest.raises(RuntimeError, match="left arm"):
        robot.auto_calibrate(full_reset=True)

    assert robot.z_actuator.calibrator.calls == 1
    assert robot.right_arm.calls == [{"reverse": True, "full_reset": True}]


def test_bi_sourccey_leader_auto_calibrate_raises_when_arm_thread_fails() -> None:
    teleop = BiSourcceyLeader.__new__(BiSourcceyLeader)
    teleop.left_arm = _DummyArm()
    teleop.right_arm = _DummyArm(RuntimeError("right leader failed"))

    with pytest.raises(RuntimeError, match="right"):
        teleop.auto_calibrate()

    assert teleop.left_arm.calls == [{"reverse": False}]
