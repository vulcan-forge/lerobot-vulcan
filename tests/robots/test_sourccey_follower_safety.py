#!/usr/bin/env python

from types import SimpleNamespace

from lerobot.robots.sourccey.sourccey.sourccey_follower.sourccey_follower_safety import SourcceyFollowerSafety


def _make_safety() -> SourcceyFollowerSafety:
    robot = SimpleNamespace(
        bus=SimpleNamespace(motors={"elbow_flex": object()}),
        config=SimpleNamespace(
            orientation="left",
            gripper_current_safety_threshold=None,
            max_current_safety_threshold=2500,
        ),
    )
    return SourcceyFollowerSafety(robot)


def test_apply_overcurrent_hold_blocks_only_deeper_motion() -> None:
    safety = _make_safety()
    safety.remember_goal({"elbow_flex": 110.0}, {"elbow_flex": 100.0})

    safe_goal = safety.apply_overcurrent_hold(
        {"elbow_flex": 111.0},
        {"elbow_flex": 101.0},
        {"elbow_flex": 200.0},
    )

    assert safe_goal["elbow_flex"] == 101.0


def test_apply_overcurrent_hold_allows_backing_away() -> None:
    safety = _make_safety()
    safety.remember_goal({"elbow_flex": 110.0}, {"elbow_flex": 100.0})

    safe_goal = safety.apply_overcurrent_hold(
        {"elbow_flex": 95.0},
        {"elbow_flex": 101.0},
        {"elbow_flex": 200.0},
    )

    assert safe_goal["elbow_flex"] == 95.0


def test_apply_overcurrent_hold_does_not_freeze_unknown_reverse_direction() -> None:
    safety = _make_safety()

    safe_goal = safety.apply_overcurrent_hold(
        {"elbow_flex": 95.0},
        {"elbow_flex": 101.0},
        {"elbow_flex": 200.0},
    )

    assert safe_goal["elbow_flex"] == 95.0
