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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.common.so_arm import SOJointConfig, make_new_bot_follower_joint_configs

from ..config import RobotConfig


@dataclass
class NewBotConfig:
    """Base configuration for the 7-DoF NewBot follower."""

    port: str
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    use_degrees: bool = True
    motors: dict[str, SOJointConfig] = field(default_factory=make_new_bot_follower_joint_configs)


@RobotConfig.register_subclass("newbot")
@RobotConfig.register_subclass("new_bot")
@RobotConfig.register_subclass("new_arm")
@dataclass
class NewBotRobotConfig(RobotConfig, NewBotConfig):
    pass


NewBotConfig = NewBotRobotConfig
NewArmConfig = NewBotRobotConfig
