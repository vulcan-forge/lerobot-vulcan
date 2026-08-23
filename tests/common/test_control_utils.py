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

from unittest.mock import MagicMock, call

import pytest

from lerobot.common.control_utils import move_robot


@pytest.fixture
def robot() -> MagicMock:
    mock = MagicMock()
    mock.is_connected = True
    mock.action_features = {"shoulder.pos": float, "elbow.pos": float}
    mock.get_observation.return_value = {"shoulder.pos": 0.0, "elbow.pos": 10.0}
    mock.send_action.side_effect = lambda action: action
    return mock


def test_move_robot_interpolates_to_target(robot, monkeypatch):
    sleep = MagicMock()
    monkeypatch.setattr("lerobot.common.control_utils.time.sleep", sleep)

    sent = move_robot(
        robot,
        {"shoulder.pos": 10.0, "elbow.pos": -10.0},
        duration_s=1.0,
        fps=2,
    )

    assert [call.args[0] for call in robot.send_action.call_args_list] == [
        {"shoulder.pos": 5.0, "elbow.pos": 0.0},
        {"shoulder.pos": 10.0, "elbow.pos": -10.0},
    ]
    assert sent == {"shoulder.pos": 10.0, "elbow.pos": -10.0}
    assert sleep.call_args_list == [call(0.5), call(0.5)]


def test_move_robot_only_commands_targeted_joints(robot, monkeypatch):
    monkeypatch.setattr("lerobot.common.control_utils.time.sleep", MagicMock())

    move_robot(robot, {"elbow.pos": 20.0}, duration_s=1.0, fps=1)

    robot.send_action.assert_called_once_with({"elbow.pos": 20.0})


@pytest.mark.parametrize(
    ("target", "duration_s", "fps", "message"),
    [
        ({"shoulder.pos": 1.0}, 0.0, 30, "duration_s"),
        ({"shoulder.pos": 1.0}, 1.0, 0, "fps"),
        ({}, 1.0, 30, "target"),
        ({"unknown.pos": 1.0}, 1.0, 30, "Unknown robot action features"),
    ],
)
def test_move_robot_rejects_invalid_requests(robot, target, duration_s, fps, message):
    with pytest.raises(ValueError, match=message):
        move_robot(robot, target, duration_s=duration_s, fps=fps)


def test_move_robot_requires_connection(robot):
    robot.is_connected = False

    with pytest.raises(RuntimeError, match="must be connected"):
        move_robot(robot, {"shoulder.pos": 1.0})


def test_move_robot_requires_matching_observation(robot):
    robot.get_observation.return_value = {"elbow.pos": 10.0}

    with pytest.raises(ValueError, match="Cannot determine current values"):
        move_robot(robot, {"shoulder.pos": 1.0})
