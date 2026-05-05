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

from lerobot.scripts.lerobot_calibrate_new_arm_leader import make_new_arm_leader_joint_configs


def test_new_arm_leader_calibration_motor_models():
    motors = make_new_arm_leader_joint_configs()

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
