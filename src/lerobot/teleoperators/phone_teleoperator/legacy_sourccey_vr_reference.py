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

"""Reference-only archive of the legacy Sourccey VR post-IK shaping path.

This module is intentionally not imported by the live teleoperator path. It exists
so we can compare the pre-refactor behavior while validating the new modular VR
teleoperation architecture.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def legacy_finalize_action(
    teleop: Any,
    solution_rad: np.ndarray,
    *,
    switch_state: bool,
    gripper_value: float,
    base: Any = None,
) -> dict[str, Any]:
    """Apply the legacy monolithic post-IK shaping path for comparison only."""
    solution_rad = np.asarray(solution_rad, dtype=float)
    prev_q = getattr(teleop, "_legacy_prev_q", None)

    if getattr(teleop, "tune", None) and teleop.tune.get("bypass_all_mods", False):
        setattr(teleop, "_legacy_prev_q", solution_rad.copy())
        solution_final = np.rad2deg(solution_rad)

        prev_start_teleop = bool(getattr(teleop, "start_teleop", False))
        teleop.start_teleop = switch_state

        if teleop.arm_side == "right" and not prev_start_teleop and teleop.start_teleop:
            teleop.current_t_R = teleop._original_right_position.copy()
            teleop.current_t_R[1] = -teleop.current_t_R[1]
            teleop.current_q_R = teleop._original_right_quat.copy()
            teleop.current_q_R[2] = -teleop.current_q_R[2]
            teleop.current_q_R[3] = -teleop.current_q_R[3]

        action_ctrl = teleop._format_action_dict(
            list(solution_final),
            gripper_percent=float(gripper_value),
        )
        if teleop.arm_side == "right":
            action_ctrl = {key.replace("left_", "right_"): value for key, value in action_ctrl.items()}
        return teleop._merge_base_with_action(action_ctrl, base=base)

    if prev_q is None:
        prev_q = solution_rad.copy()

    alpha = float(teleop.tune["lowpass_alpha"])
    solution_rad = alpha * solution_rad + (1.0 - alpha) * prev_q

    elbow_idx = int(teleop.tune["elbow_soft_stop"]["index"])
    soft_stop = teleop._elbow_soft_stop
    if soft_stop is not None:
        if solution_rad[elbow_idx] < soft_stop:
            small_step = np.deg2rad(float(teleop.tune["elbow_soft_stop"]["below_small_step_deg"]))
            margin = np.deg2rad(float(teleop.tune["elbow_soft_stop"]["below_margin_deg"]))
            allowed = max(solution_rad[elbow_idx], prev_q[elbow_idx] - small_step)
            solution_rad[elbow_idx] = max(allowed, soft_stop - margin)
        else:
            max_down_per_call = np.deg2rad(float(teleop.tune["elbow_soft_stop"]["above_max_down_deg"]))
            solution_rad[elbow_idx] = max(solution_rad[elbow_idx], prev_q[elbow_idx] - max_down_per_call)
    else:
        max_down_per_call = np.deg2rad(25.0)
        solution_rad[elbow_idx] = max(solution_rad[elbow_idx], prev_q[elbow_idx] - max_down_per_call)

    wrist_roll_idx = 4
    solution_rad[wrist_roll_idx] = 0.95 * solution_rad[wrist_roll_idx] + 0.05 * 0.0

    for joint_idx, scale in teleop.tune["delta_scale"].items():
        joint_idx = int(joint_idx)
        scale = float(scale)
        if 0 <= joint_idx < len(solution_rad) and scale != 1.0:
            delta = solution_rad[joint_idx] - prev_q[joint_idx]
            solution_rad[joint_idx] = prev_q[joint_idx] + scale * delta

    if teleop.tune.get("fixed_rate", {}).get("enabled", False):
        for joint_idx, cfg in teleop.tune["fixed_rate"].get("joints", {}).items():
            joint_idx = int(joint_idx)
            if 0 <= joint_idx < len(solution_rad):
                step_rad = np.deg2rad(float(cfg.get("step_deg", 2.0)))
                deadband_rad = np.deg2rad(float(cfg.get("deadband_deg", 0.0)))
                target = float(solution_rad[joint_idx])
                previous = float(prev_q[joint_idx])
                error = target - previous
                if abs(error) <= deadband_rad:
                    solution_rad[joint_idx] = previous
                else:
                    direction = 1.0 if error > 0 else -1.0
                    solution_rad[joint_idx] = previous + direction * min(step_rad, abs(error))

    legacy_elbow_back_limit = getattr(teleop, "_legacy_elbow_back_limit", None)
    if teleop.tune["elbow_back_block"]["enabled"]:
        elbow_idx = int(teleop.tune["elbow_back_block"]["index"])
        if legacy_elbow_back_limit is None:
            legacy_elbow_back_limit = float(solution_rad[elbow_idx])
            setattr(teleop, "_legacy_elbow_back_limit", legacy_elbow_back_limit)
        back_margin = np.deg2rad(float(teleop.tune["elbow_back_block"]["tolerance_deg"]))
        direction = str(teleop.tune["elbow_back_block"]["direction"]).lower()
        if direction == "decrease":
            solution_rad[elbow_idx] = max(solution_rad[elbow_idx], legacy_elbow_back_limit - back_margin)
        else:
            solution_rad[elbow_idx] = min(solution_rad[elbow_idx], legacy_elbow_back_limit + back_margin)

    setattr(teleop, "_legacy_prev_q", solution_rad.copy())

    if teleop.config.enable_visualization and teleop.urdf_vis:
        teleop.urdf_vis.update_cfg(solution_rad)

    solution_final = np.rad2deg(solution_rad)

    if teleop.arm_side == "right":
        for idx in range(min(5, len(solution_final))):
            solution_final[idx] = -solution_final[idx]
    elif len(solution_final) > 4:
        solution_final[4] = -solution_final[4]

    if getattr(teleop.config, "joint_offsets_deg", None):
        offsets = teleop.config.joint_offsets_deg or {}
        name_by_index = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        for idx, name in enumerate(name_by_index):
            if idx < len(solution_final) and name in offsets and name != "gripper":
                solution_final[idx] += float(offsets[name])

    gripper_range = teleop.config.gripper_max_pos - teleop.config.gripper_min_pos
    gripper_position = teleop.config.gripper_min_pos + (float(gripper_value) / 100.0) * gripper_range
    if len(solution_final) > 0:
        solution_final[-1] = gripper_position

    prev_start_teleop = bool(getattr(teleop, "start_teleop", False))
    teleop.start_teleop = switch_state
    if teleop.arm_side == "right" and not prev_start_teleop and teleop.start_teleop:
        teleop.current_t_R = teleop._original_right_position.copy()
        teleop.current_t_R[1] = -teleop.current_t_R[1]
        teleop.current_q_R = teleop._original_right_quat.copy()
        teleop.current_q_R[2] = -teleop.current_q_R[2]
        teleop.current_q_R[3] = -teleop.current_q_R[3]

    action_ctrl = teleop._format_action_dict(
        list(solution_final),
        gripper_percent=float(gripper_value),
    )
    if teleop.arm_side == "right":
        action_ctrl = {key.replace("left_", "right_"): value for key, value in action_ctrl.items()}
    return teleop._merge_base_with_action(action_ctrl, base=base)
