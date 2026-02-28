#!/usr/bin/env python

from lerobot.robots.sourccey.sourccey.sourccey_follower.config_sourccey_follower import SourcceyFollowerConfig
from lerobot.robots.sourccey.sourccey.sourccey_follower.sourccey_follower_pathing import SourcceyFollowerPathing


def _make_joint_state(value: float) -> dict[str, float]:
    return {
        "shoulder_pan": value,
        "shoulder_lift": value,
        "elbow_flex": value,
        "wrist_flex": value,
        "wrist_roll": value,
        "gripper": value,
    }


def test_recovery_pathing_is_disabled_by_default():
    config = SourcceyFollowerConfig(port="/dev/null", enable_recovery_pathing=False)
    planner = SourcceyFollowerPathing(config)
    present_pos = _make_joint_state(6.0)
    goal_pos = _make_joint_state(12.0)

    assert planner.apply_recovery_pathing(goal_pos, present_pos) == goal_pos
    assert planner.apply_recovery_pathing(goal_pos, present_pos) == goal_pos
    assert planner.apply_recovery_pathing(goal_pos, present_pos) == goal_pos


def test_recovery_pathing_inserts_backoff_then_default_core_tuck_when_stalled():
    config = SourcceyFollowerConfig(
        port="/dev/null",
        enable_recovery_pathing=True,
        recovery_stall_window=2,
        recovery_min_progress=0.25,
        recovery_min_remaining_error=2.0,
        recovery_stage_hold_cycles=1,
        recovery_joint_backoff=3.0,
        recovery_posture_step=2.0,
    )
    planner = SourcceyFollowerPathing(config)
    present_pos = {
        "shoulder_pan": 6.0,
        "shoulder_lift": 6.0,
        "elbow_flex": 6.0,
        "wrist_flex": 6.0,
        "wrist_roll": 1.0,
        "gripper": 20.0,
    }
    goal_pos = {
        "shoulder_pan": 12.0,
        "shoulder_lift": 12.0,
        "elbow_flex": 10.0,
        "wrist_flex": 9.0,
        "wrist_roll": 1.0,
        "gripper": 20.0,
    }

    assert planner.apply_recovery_pathing(goal_pos, present_pos) == goal_pos
    assert planner.apply_recovery_pathing(goal_pos, present_pos) == goal_pos

    assert planner.apply_recovery_pathing(goal_pos, present_pos) == {
        "shoulder_pan": 3.0,
        "shoulder_lift": 3.0,
        "elbow_flex": 3.0,
        "wrist_flex": 3.0,
        "wrist_roll": 1.0,
        "gripper": 20.0,
    }
    assert planner.apply_recovery_pathing(goal_pos, present_pos) == {
        "shoulder_pan": 3.8570084666039577,
        "shoulder_lift": 1.0,
        "elbow_flex": 1.0,
        "wrist_flex": 3.0,
        "wrist_roll": 1.0,
        "gripper": 20.0,
    }

    assert planner.apply_recovery_pathing(goal_pos, present_pos) == goal_pos


def test_recovery_pathing_rotates_to_default_full_tuck_on_repeated_stalls():
    config = SourcceyFollowerConfig(
        port="/dev/null",
        enable_recovery_pathing=True,
        recovery_stall_window=2,
        recovery_min_progress=0.25,
        recovery_min_remaining_error=2.0,
        recovery_stage_hold_cycles=1,
        recovery_joint_backoff=3.0,
        recovery_posture_step=2.0,
    )
    planner = SourcceyFollowerPathing(config)
    present_pos = {
        "shoulder_pan": 6.0,
        "shoulder_lift": 6.0,
        "elbow_flex": 6.0,
        "wrist_flex": 6.0,
        "wrist_roll": 1.0,
        "gripper": 20.0,
    }
    goal_pos = {
        "shoulder_pan": 12.0,
        "shoulder_lift": 12.0,
        "elbow_flex": 10.0,
        "wrist_flex": 9.0,
        "wrist_roll": 1.0,
        "gripper": 20.0,
    }

    assert planner.apply_recovery_pathing(goal_pos, present_pos) == goal_pos
    assert planner.apply_recovery_pathing(goal_pos, present_pos) == goal_pos
    assert planner.apply_recovery_pathing(goal_pos, present_pos) == {
        "shoulder_pan": 3.0,
        "shoulder_lift": 3.0,
        "elbow_flex": 3.0,
        "wrist_flex": 3.0,
        "wrist_roll": 1.0,
        "gripper": 20.0,
    }
    assert planner.apply_recovery_pathing(goal_pos, present_pos) == {
        "shoulder_pan": 3.8570084666039577,
        "shoulder_lift": 1.0,
        "elbow_flex": 1.0,
        "wrist_flex": 3.0,
        "wrist_roll": 1.0,
        "gripper": 20.0,
    }

    assert planner.apply_recovery_pathing(goal_pos, present_pos) == goal_pos
    assert planner.apply_recovery_pathing(goal_pos, present_pos) == {
        "shoulder_pan": 3.0,
        "shoulder_lift": 3.0,
        "elbow_flex": 3.0,
        "wrist_flex": 3.0,
        "wrist_roll": 1.0,
        "gripper": 20.0,
    }
    assert planner.apply_recovery_pathing(goal_pos, present_pos) == {
        "shoulder_pan": 3.8570084666039577,
        "shoulder_lift": 1.0,
        "elbow_flex": 1.0,
        "wrist_flex": 2.021993614757008,
        "wrist_roll": 1.0,
        "gripper": 20.0,
    }


def test_recovery_pathing_builds_many_recovery_strategies():
    config = SourcceyFollowerConfig(port="/dev/null", enable_recovery_pathing=True)
    planner = SourcceyFollowerPathing(config)

    assert len(planner._recovery_strategies) >= 20


def test_recovery_pathing_pauses_stage_consumption_while_motor_is_hot():
    config = SourcceyFollowerConfig(
        port="/dev/null",
        enable_recovery_pathing=True,
        recovery_stall_window=2,
        recovery_min_progress=0.25,
        recovery_min_remaining_error=2.0,
        recovery_stage_hold_cycles=1,
        recovery_joint_backoff=3.0,
        recovery_posture_step=2.0,
    )
    planner = SourcceyFollowerPathing(config)
    present_pos = {
        "shoulder_pan": 6.0,
        "shoulder_lift": 6.0,
        "elbow_flex": 6.0,
        "wrist_flex": 6.0,
        "wrist_roll": 1.0,
        "gripper": 20.0,
    }
    goal_pos = {
        "shoulder_pan": 12.0,
        "shoulder_lift": 12.0,
        "elbow_flex": 10.0,
        "wrist_flex": 9.0,
        "wrist_roll": 1.0,
        "gripper": 20.0,
    }

    assert planner.apply_recovery_pathing(goal_pos, present_pos) == goal_pos
    assert planner.apply_recovery_pathing(goal_pos, present_pos) == goal_pos

    paused_backoff = planner.apply_recovery_pathing(goal_pos, present_pos, pause_motors={"shoulder_lift"})
    assert paused_backoff == {
        "shoulder_pan": 3.0,
        "shoulder_lift": 3.0,
        "elbow_flex": 3.0,
        "wrist_flex": 3.0,
        "wrist_roll": 1.0,
        "gripper": 20.0,
    }

    assert planner.apply_recovery_pathing(goal_pos, present_pos, pause_motors={"shoulder_lift"}) == paused_backoff
    assert planner.apply_recovery_pathing(goal_pos, present_pos) == paused_backoff
    assert planner.apply_recovery_pathing(goal_pos, present_pos) == {
        "shoulder_pan": 3.8570084666039577,
        "shoulder_lift": 1.0,
        "elbow_flex": 1.0,
        "wrist_flex": 3.0,
        "wrist_roll": 1.0,
        "gripper": 20.0,
    }
