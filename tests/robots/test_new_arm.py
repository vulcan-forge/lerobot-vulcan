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

from lerobot.robots.new_arm import NewArm, NewArmConfig
from lerobot.scripts.lerobot_calibrate_new_arm_follower import make_new_arm_follower_joint_configs
from lerobot.teleoperators.new_arm_leader import NewArmLeader, NewArmLeaderConfig


EXPECTED_MOTORS = [
    "shoulder_twist",
    "shoulder_lift",
    "elbow_twist",
    "elbow_lift",
    "wrist_twist",
    "wrist_lift",
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
    for idx, motor in enumerate(EXPECTED_MOTORS, 1):
        assert obs[f"{motor}.pos"] == idx


def test_new_arm_send_action(new_arm):
    new_arm.connect()
    action = {f"{motor}.pos": i * 10 for i, motor in enumerate(EXPECTED_MOTORS, 1)}

    returned = new_arm.send_action(action)

    assert returned == action
    goal_pos = {motor: (i + 1) * 10 for i, motor in enumerate(EXPECTED_MOTORS)}
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
    motors = make_new_arm_follower_joint_configs()

    assert {name: cfg.model for name, cfg in motors.items()} == {
        "roll_1": "sts3032",
        "pitch_1": "sts3032",
        "roll_2": "sts3250",
        "pitch_2": "sts3250",
        "roll_3": "sts3215",
        "pitch_3": "sts3215",
        "gripper": "sts3215",
    }
    assert motors["gripper"].id == 7
    assert motors["gripper"].is_gripper
