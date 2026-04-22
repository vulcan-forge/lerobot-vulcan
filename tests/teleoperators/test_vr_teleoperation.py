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

import numpy as np

from lerobot.teleoperators.vr_teleoperation.assets import resolve_sourccey_teleop_assets
from lerobot.teleoperators.vr_teleoperation.models import BaseMotionCommand, VRTeleopSample
from lerobot.teleoperators.vr_teleoperation.observation import ControlledArmObservationSelector
from lerobot.teleoperators.vr_teleoperation.postprocess import (
    FixedRateJointLimit,
    JointPostprocessConfig,
    JointPostprocessor,
)


def test_controlled_arm_observation_selector_prefers_right_arm_keys():
    selector = ControlledArmObservationSelector(
        arm_side="right",
        joint_names=("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"),
        observation_uses_degrees=True,
        denormalize_observation=lambda values: list(values),
    )

    observation = {
        "left_shoulder_pan.pos": -10.0,
        "left_shoulder_lift.pos": -20.0,
        "left_elbow_flex.pos": -30.0,
        "left_wrist_flex.pos": -40.0,
        "left_wrist_roll.pos": -50.0,
        "left_gripper.pos": -60.0,
        "right_shoulder_pan.pos": 10.0,
        "right_shoulder_lift.pos": 20.0,
        "right_elbow_flex.pos": 30.0,
        "right_wrist_flex.pos": 40.0,
        "right_wrist_roll.pos": 50.0,
        "right_gripper.pos": 60.0,
    }

    controlled = selector.extract(observation)

    assert controlled is not None
    assert controlled.arm_side == "right"
    assert controlled.joint_keys == (
        "right_shoulder_pan.pos",
        "right_shoulder_lift.pos",
        "right_elbow_flex.pos",
        "right_wrist_flex.pos",
        "right_wrist_roll.pos",
        "right_gripper.pos",
    )
    assert controlled.joint_positions_deg == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]


def test_base_motion_command_marks_nonzero_payload_as_active():
    base = BaseMotionCommand.from_payload({"x.vel": 0.25, "y.vel": 0.0, "theta.vel": 0.0})

    assert base.active is True
    assert base.x == 0.25
    assert base.y == 0.0
    assert base.theta == 0.0


def test_vr_teleop_sample_parses_base_and_pose_payload():
    sample = VRTeleopSample.from_payload(
        {
            "position": [1, 2, 3],
            "rotation": [1, 0, 0, 0],
            "gripper_value": 42,
            "switch": True,
            "precision": True,
            "reset_mapping": False,
            "is_resetting": True,
            "base": {"x.vel": 0.1, "theta.vel": -0.2},
        }
    )

    assert sample is not None
    assert np.allclose(sample.position, [1.0, 2.0, 3.0])
    assert np.allclose(sample.rotation_wxyz, [1.0, 0.0, 0.0, 0.0])
    assert sample.gripper_value == 42.0
    assert sample.teleop_active is True
    assert sample.precision_mode is True
    assert sample.is_resetting is True
    assert sample.base.active is True
    assert sample.base.theta == -0.2


def test_joint_postprocessor_applies_fixed_rate_limit_after_baseline():
    postprocessor = JointPostprocessor(
        JointPostprocessConfig(
            lowpass_alpha=1.0,
            delta_scale={},
            wrist_roll_bias_enabled=False,
            elbow_soft_stop_enabled=False,
            elbow_back_block_enabled=False,
            fixed_rate_enabled=True,
            fixed_rate_limits={0: FixedRateJointLimit(step_deg=2.0, deadband_deg=0.0)},
        )
    )

    baseline = postprocessor.apply(np.deg2rad(np.array([0.0])), elbow_soft_stop=None)
    limited = postprocessor.apply(np.deg2rad(np.array([10.0])), elbow_soft_stop=None)

    assert np.allclose(baseline, np.deg2rad([0.0]))
    assert np.allclose(limited, np.deg2rad([2.0]))


def test_resolve_sourccey_teleop_assets_finds_repo_defaults():
    paths = resolve_sourccey_teleop_assets(
        urdf_path=None,
        mesh_path=None,
        calibration_path=None,
        arm_side="left",
    )

    assert paths.urdf_path is not None
    assert paths.calibration_path is not None
    assert str(paths.urdf_path).replace("\\", "/").endswith(
        "src/lerobot/robots/sourccey/sourccey/sourccey/model/Arm.urdf"
    )
    assert str(paths.calibration_path).replace("\\", "/").endswith(
        "src/lerobot/robots/sourccey/sourccey/sourccey/defaults/left_arm_default_calibration.json"
    )
