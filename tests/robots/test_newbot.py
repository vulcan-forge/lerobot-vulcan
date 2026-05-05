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

from unittest.mock import MagicMock, patch

from lerobot.robots.new_arm import NewBot, NewBotConfig


def test_newbot_default_joint_layout():
    bus_mock = MagicMock(name="FeetechBusMock")
    bus_mock.is_calibrated = True

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        return bus_mock

    with patch("lerobot.robots.new_arm.new_arm.FeetechMotorsBus", side_effect=_bus_side_effect):
        robot = NewBot(NewBotConfig(port="/dev/null"))

    assert list(robot.bus.motors) == [
        "pitch_1",
        "roll_1",
        "pitch_2",
        "roll_2",
        "pitch_3",
        "roll_3",
        "gripper",
    ]
