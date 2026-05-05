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

from lerobot.common.so_arm import SOJointConfig, make_new_bot_leader_joint_configs
from ..config import TeleoperatorConfig


@dataclass
class NewBotLeaderConfig:
    """Base configuration for the 7-DoF NewBot leader."""

    port: str
    use_degrees: bool = True
    motors: dict[str, SOJointConfig] = field(default_factory=make_new_bot_leader_joint_configs)


@TeleoperatorConfig.register_subclass("newbot_leader")
@TeleoperatorConfig.register_subclass("new_bot_leader")
@TeleoperatorConfig.register_subclass("new_arm_leader")
@dataclass
class NewBotLeaderTeleopConfig(TeleoperatorConfig, NewBotLeaderConfig):
    pass


NewBotLeaderConfig = NewBotLeaderTeleopConfig
NewArmLeaderConfig = NewBotLeaderTeleopConfig
