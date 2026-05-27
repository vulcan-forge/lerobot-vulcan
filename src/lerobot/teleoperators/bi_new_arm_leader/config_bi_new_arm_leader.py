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


def make_left_new_arm_leader_joint_configs() -> dict[str, SOJointConfig]:
    return make_new_bot_leader_joint_configs(start_id=8)


def make_right_new_arm_leader_joint_configs() -> dict[str, SOJointConfig]:
    return make_new_bot_leader_joint_configs(start_id=1)


@dataclass
class BiNewArmLeaderArmConfig:
    """Configuration for one NewArm leader in a bimanual pair."""

    port: str
    use_degrees: bool = True
    motors: dict[str, SOJointConfig] = field(default_factory=make_right_new_arm_leader_joint_configs)


@dataclass
class LeftBiNewArmLeaderArmConfig(BiNewArmLeaderArmConfig):
    motors: dict[str, SOJointConfig] = field(default_factory=make_left_new_arm_leader_joint_configs)


@dataclass
class RightBiNewArmLeaderArmConfig(BiNewArmLeaderArmConfig):
    motors: dict[str, SOJointConfig] = field(default_factory=make_right_new_arm_leader_joint_configs)


@TeleoperatorConfig.register_subclass("bi_new_arm_leader")
@dataclass
class BiNewArmLeaderConfig(TeleoperatorConfig):
    """Configuration class for bimanual NewArm leader teleoperators."""

    left_arm_config: LeftBiNewArmLeaderArmConfig
    right_arm_config: RightBiNewArmLeaderArmConfig
