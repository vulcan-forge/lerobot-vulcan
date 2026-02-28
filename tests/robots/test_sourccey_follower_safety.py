#!/usr/bin/env python

from types import SimpleNamespace

from lerobot.robots.sourccey.sourccey.sourccey_follower.config_sourccey_follower import SourcceyFollowerConfig
from lerobot.robots.sourccey.sourccey.sourccey_follower.sourccey_follower_safety import SourcceyFollowerSafety


class _FakeBus:
    def __init__(self, currents: dict[str, float]):
        self._currents = currents

    def sync_read(self, register: str) -> dict[str, float]:
        assert register == "Present_Current"
        return self._currents


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
