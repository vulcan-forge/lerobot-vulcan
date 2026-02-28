#!/usr/bin/env python

from types import SimpleNamespace

from lerobot.robots.sourccey.sourccey.sourccey_follower.config_sourccey_follower import SourcceyFollowerConfig
from lerobot.robots.sourccey.sourccey.sourccey_follower.sourccey_follower_safety import SourcceyFollowerSafety


class _FakeBus:
    def __init__(self, currents: dict[str, float]):
        self._currents = currents
        self.disabled_motors: list[str] = []
        self.enabled_motors: list[str] = []

    def sync_read(self, register: str) -> dict[str, float]:
        assert register == "Present_Current"
        return self._currents

    def disable_torque(self, motors: str | list[str] | None = None) -> None:
        if isinstance(motors, list):
            self.disabled_motors.extend(motors)
        elif motors is not None:
            self.disabled_motors.append(motors)

    def enable_torque(self, motors: str | list[str] | None = None) -> None:
        if isinstance(motors, list):
            self.enabled_motors.extend(motors)
        elif motors is not None:
            self.enabled_motors.append(motors)


def _make_robot(config: SourcceyFollowerConfig, currents: dict[str, float]) -> SimpleNamespace:
    return SimpleNamespace(config=config, bus=_FakeBus(currents))


def test_current_safety_rests_motor_before_hard_overcurrent():
    config = SourcceyFollowerConfig(
        port="/dev/null",
        current_rest_safety_threshold_ratio=0.5,
        current_rest_backoff=1.0,
        current_safety_backoff=3.0,
    )
    safety = SourcceyFollowerSafety(
        _make_robot(
            config,
            {
                "shoulder_lift": 1000.0,
            },
        )
    )
    goal_pos = {"shoulder_lift": 10.0}
    present_pos = {"shoulder_lift": 8.0}
    safety.remember_goal(goal_pos)

    assert safety.apply_current_safety(goal_pos, present_pos) == {"shoulder_lift": 7.0}


def test_current_safety_uses_hard_backoff_once_over_limit():
    config = SourcceyFollowerConfig(
        port="/dev/null",
        current_rest_safety_threshold_ratio=0.5,
        current_rest_backoff=1.0,
        current_safety_backoff=3.0,
        disable_torque_on_hard_overcurrent=False,
    )
    safety = SourcceyFollowerSafety(
        _make_robot(
            config,
            {
                "shoulder_lift": 2200.0,
            },
        )
    )
    goal_pos = {"shoulder_lift": 10.0}
    present_pos = {"shoulder_lift": 8.0}
    safety.remember_goal(goal_pos)

    assert safety.apply_current_safety(goal_pos, present_pos) == {"shoulder_lift": 5.0}


def test_current_safety_latches_safe_position_until_current_settles():
    config = SourcceyFollowerConfig(
        port="/dev/null",
        current_rest_safety_threshold_ratio=0.5,
        current_safe_release_threshold_ratio=0.25,
        current_safe_hold_cycles=2,
        current_rest_backoff=1.0,
    )
    robot = _make_robot(
        config,
        {
            "shoulder_lift": 1000.0,
        },
    )
    safety = SourcceyFollowerSafety(robot)
    goal_pos = {"shoulder_lift": 10.0}
    present_pos = {"shoulder_lift": 8.0}
    safety.remember_goal(goal_pos)

    assert safety.apply_current_safety(goal_pos, present_pos) == {"shoulder_lift": 7.0}

    robot.bus._currents = {"shoulder_lift": 100.0}
    latched_present_pos = {"shoulder_lift": 7.0}
    safety.remember_goal(goal_pos)
    assert safety.apply_current_safety(goal_pos, latched_present_pos) == {"shoulder_lift": 7.0}

    safety.remember_goal(goal_pos)
    assert safety.apply_current_safety(goal_pos, latched_present_pos) == {"shoulder_lift": 10.0}


def test_current_safety_disables_torque_on_hard_overcurrent_and_reenables_after_settle():
    config = SourcceyFollowerConfig(
        port="/dev/null",
        current_safe_release_threshold_ratio=0.25,
        current_safe_hold_cycles=2,
        disable_torque_on_hard_overcurrent=True,
    )
    robot = _make_robot(
        config,
        {
            "shoulder_lift": 2200.0,
        },
    )
    safety = SourcceyFollowerSafety(robot)
    goal_pos = {"shoulder_lift": 10.0}
    present_pos = {"shoulder_lift": 8.0}
    safety.remember_goal(goal_pos)

    assert safety.apply_current_safety(goal_pos, present_pos) == {"shoulder_lift": 8.0}
    assert robot.bus.disabled_motors == ["shoulder_lift"]

    robot.bus._currents = {"shoulder_lift": 100.0}
    safety.remember_goal(goal_pos)
    assert safety.apply_current_safety(goal_pos, present_pos) == {"shoulder_lift": 8.0}
    assert robot.bus.enabled_motors == []

    safety.remember_goal(goal_pos)
    assert safety.apply_current_safety(goal_pos, present_pos) == {"shoulder_lift": 8.0}
    assert robot.bus.enabled_motors == ["shoulder_lift"]

    safety.remember_goal(goal_pos)
    assert safety.apply_current_safety(goal_pos, present_pos) == {"shoulder_lift": 8.0}

    safety.remember_goal(goal_pos)
    assert safety.apply_current_safety(goal_pos, present_pos) == {"shoulder_lift": 10.0}
