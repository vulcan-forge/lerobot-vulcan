#!/usr/bin/env python

import time
from types import SimpleNamespace

from lerobot.robots.sourccey.sourccey.sourccey_follower.sourccey_follower_safety import SourcceyFollowerSafety


def _make_safety() -> SourcceyFollowerSafety:
    robot = SimpleNamespace(config=SimpleNamespace(orientation="left"))
    return SourcceyFollowerSafety(robot)


def test_large_distance_uses_step_safety() -> None:
    safety = _make_safety()
    safety._action_stream_start_time = time.monotonic() - safety.STEP_SAFETY_STARTUP_WINDOW_S - 1.0

    assert safety.should_use_step_safety({"shoulder_lift": 121.0}, {"shoulder_lift": 0.0})


def test_small_distance_does_not_use_step_safety_after_startup() -> None:
    safety = _make_safety()
    safety._action_stream_start_time = time.monotonic() - safety.STEP_SAFETY_STARTUP_WINDOW_S - 1.0

    assert not safety.should_use_step_safety({"shoulder_lift": 119.0}, {"shoulder_lift": 0.0})


def test_step_safety_limits_motion_in_both_directions() -> None:
    safety = _make_safety()

    positive = safety.apply_step_safety({"shoulder_lift": 100.0}, {"shoulder_lift": 10.0})
    negative = safety.apply_step_safety({"shoulder_lift": -100.0}, {"shoulder_lift": 10.0})

    assert positive["shoulder_lift"] == 15.0
    assert negative["shoulder_lift"] == 5.0


def test_step_safety_preserves_nearby_target() -> None:
    safety = _make_safety()

    safe_goal = safety.apply_step_safety({"shoulder_lift": 12.0}, {"shoulder_lift": 10.0})

    assert safe_goal["shoulder_lift"] == 12.0
