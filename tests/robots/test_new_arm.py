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

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.common.so_arm import make_new_bot_follower_joint_configs
from lerobot.robots.bi_new_arm import (
    BiNewArm,
    BiNewArmConfig,
    LeftBiNewArmFollowerArmConfig,
    RightBiNewArmFollowerArmConfig,
)
from lerobot.robots.new_arm import NewArm, NewArmConfig
from lerobot.teleoperators.bi_new_arm_leader import (
    BiNewArmLeader,
    BiNewArmLeaderConfig,
    LeftBiNewArmLeaderArmConfig,
    RightBiNewArmLeaderArmConfig,
)
from lerobot.teleoperators.new_arm_leader import NewArmLeader, NewArmLeaderConfig


EXPECTED_MOTORS = [
    "shoulder_roll",
    "shoulder_pitch",
    "elbow_roll",
    "elbow_pitch",
    "wrist_pitch",
    "wrist_roll",
    "gripper",
]


def _make_bus_mock() -> MagicMock:
    bus = MagicMock(name="FeetechBusMock")
    bus.is_connected = False

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect

    @contextmanager
    def _dummy_cm():
        yield

    bus.torque_disabled.side_effect = _dummy_cm
    return bus


def _bus_side_effect(bus_mock):
    def _side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        motors_order: list[str] = list(bus_mock.motors)
        bus_mock.sync_read.return_value = {motor: idx for idx, motor in enumerate(motors_order, 1)}
        bus_mock._normalize.side_effect = lambda ids_values: dict(ids_values)
        bus_mock.sync_write.return_value = None
        bus_mock.write.return_value = None
        bus_mock.disable_torque.return_value = None
        bus_mock.is_calibrated = True
        return bus_mock

    return _side_effect


@pytest.fixture
def new_arm():
    bus_mock = _make_bus_mock()
    with (
        patch("lerobot.robots.new_arm.new_arm.FeetechMotorsBus", side_effect=_bus_side_effect(bus_mock)),
        patch.object(NewArm, "configure", lambda self: None),
    ):
        robot = NewArm(NewArmConfig(port="/dev/null"))
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_new_arm_motor_layout(new_arm):
    assert list(new_arm.bus.motors) == EXPECTED_MOTORS
    assert set(new_arm.action_features) == {f"{motor}.pos" for motor in EXPECTED_MOTORS}


def test_new_arm_get_observation(new_arm):
    new_arm.connect()
    obs = new_arm.get_observation()

    assert set(obs) == {f"{motor}.pos" for motor in EXPECTED_MOTORS}
    assert obs == {
        "shoulder_roll.pos": -0.5,
        "shoulder_pitch.pos": -1.0,
        "elbow_roll.pos": -1.5,
        "elbow_pitch.pos": 2.0,
        "wrist_pitch.pos": 5,
        "wrist_roll.pos": -6,
        "gripper.pos": 7,
    }


def test_new_arm_send_action(new_arm):
    new_arm.connect()
    action = {f"{motor}.pos": i * 10 for i, motor in enumerate(EXPECTED_MOTORS, 1)}

    returned = new_arm.send_action(action)

    assert returned == action
    goal_pos = {
        "shoulder_roll": -20,
        "shoulder_pitch": -40,
        "elbow_roll": -60,
        "elbow_pitch": 80,
        "wrist_pitch": 50,
        "wrist_roll": -60,
        "gripper": 70,
    }
    new_arm.bus.sync_write.assert_called_once_with("Goal_Position", goal_pos)


def test_new_arm_leader_reads_matching_action_keys():
    bus_mock = _make_bus_mock()
    with (
        patch(
            "lerobot.teleoperators.new_arm_leader.new_arm_leader.FeetechMotorsBus",
            side_effect=_bus_side_effect(bus_mock),
        ),
        patch.object(NewArmLeader, "configure", lambda self: None),
    ):
        leader = NewArmLeader(NewArmLeaderConfig(port="/dev/null"))
        leader.connect()
        action = leader.get_action()
        leader.disconnect()

    assert set(action) == {f"{motor}.pos" for motor in EXPECTED_MOTORS}


def test_new_arm_follower_calibration_motor_models():
    motors = make_new_bot_follower_joint_configs()

    assert {name: cfg.model for name, cfg in motors.items()} == {
        "shoulder_roll": "sts3032",
        "shoulder_pitch": "sts3032",
        "elbow_roll": "sts3250",
        "elbow_pitch": "sts3250",
        "wrist_pitch": "sts3215",
        "wrist_roll": "sts3215",
        "gripper": "sts3215",
    }
    assert motors["gripper"].id == 7
    assert motors["gripper"].is_gripper


def test_bi_new_arm_default_motor_ids():
    left_follower = LeftBiNewArmFollowerArmConfig(port="/dev/left-follower")
    right_follower = RightBiNewArmFollowerArmConfig(port="/dev/right-follower")
    left_leader = LeftBiNewArmLeaderArmConfig(port="/dev/left-leader")
    right_leader = RightBiNewArmLeaderArmConfig(port="/dev/right-leader")

    assert [cfg.id for cfg in right_follower.motors.values()] == list(range(1, 8))
    assert [cfg.id for cfg in left_follower.motors.values()] == list(range(8, 15))
    assert [cfg.id for cfg in right_leader.motors.values()] == list(range(1, 8))
    assert [cfg.id for cfg in left_leader.motors.values()] == list(range(8, 15))


def _fresh_bus_side_effect():
    def _side_effect(*_args, **kwargs):
        bus_mock = _make_bus_mock()
        return _bus_side_effect(bus_mock)(*_args, **kwargs)

    return _side_effect


def test_bi_new_arm_uses_left_right_action_keys():
    with (
        patch("lerobot.robots.new_arm.new_arm.FeetechMotorsBus", side_effect=_fresh_bus_side_effect()),
        patch.object(NewArm, "configure", lambda self: None),
    ):
        robot = BiNewArm(
            BiNewArmConfig(
                left_arm_config=LeftBiNewArmFollowerArmConfig(port="/dev/left-follower"),
                right_arm_config=RightBiNewArmFollowerArmConfig(port="/dev/right-follower"),
            )
        )
        robot.connect()
        action = {key: index for index, key in enumerate(robot.action_features, 1)}

        returned = robot.send_action(action)
        obs = robot.get_observation()
        robot.disconnect()

    assert set(robot.action_features) == {
        *(f"left_{motor}.pos" for motor in EXPECTED_MOTORS),
        *(f"right_{motor}.pos" for motor in EXPECTED_MOTORS),
    }
    assert returned == action
    assert set(obs) == set(robot.observation_features)


def test_bi_new_arm_leader_reads_left_right_action_keys():
    with (
        patch(
            "lerobot.teleoperators.new_arm_leader.new_arm_leader.FeetechMotorsBus",
            side_effect=_fresh_bus_side_effect(),
        ),
        patch.object(NewArmLeader, "configure", lambda self: None),
    ):
        leader = BiNewArmLeader(
            BiNewArmLeaderConfig(
                left_arm_config=LeftBiNewArmLeaderArmConfig(port="/dev/left-leader"),
                right_arm_config=RightBiNewArmLeaderArmConfig(port="/dev/right-leader"),
            )
        )
        leader.connect()
        action = leader.get_action()
        leader.disconnect()

    assert set(action) == {
        *(f"left_{motor}.pos" for motor in EXPECTED_MOTORS),
        *(f"right_{motor}.pos" for motor in EXPECTED_MOTORS),
    }
