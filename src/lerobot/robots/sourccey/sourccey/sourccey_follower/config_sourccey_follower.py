# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 Vulcan Robotics, Inc. All rights reserved.
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

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.motors.motors_bus import Motor
from lerobot.robots.config import RobotConfig


def sourccey_motor_models() -> dict[str, str]:
    return {
        "shoulder_pan": "sts3215",
        "shoulder_lift": "sts3250",
        "elbow_flex": "sts3250",
        "wrist_flex": "sts3215",
        "wrist_roll": "sts3215",
        "gripper": "sts3215",
    }

def sourccey_cameras_config() -> dict[str, CameraConfig]:
    return {
        "wrist": OpenCVCameraConfig(
            index_or_path="/dev/video0", fps=30, width=320, height=240
        ),
    }


def sourccey_max_relative_target() -> dict[str, float]:
    """Conservative per-cycle step limits so streamed teleop behaves like a soft approach."""
    return {
        "shoulder_pan": 0.75,
        "shoulder_lift": 0.5,
        "elbow_flex": 0.5,
        "wrist_flex": 0.75,
        "wrist_roll": 0.75,
        "gripper": 1.5,
    }

@RobotConfig.register_subclass("sourccey_follower")
@dataclass
class SourcceyFollowerConfig(RobotConfig):
    # -------------------------------------------------------------------------
    # Connection
    # -------------------------------------------------------------------------
    port: str
    orientation: str = "left"
    motor_models: dict[str, str] = field(default_factory=sourccey_motor_models)
    disable_torque_on_disconnect: bool = True

    # -------------------------------------------------------------------------
    # Motion Safety
    # -------------------------------------------------------------------------
    # `max_relative_target` limits the per-command positional delta.
    # Sourccey defaults to small per-joint steps so live teleop ramps toward the streamed target
    # instead of attempting a large immediate move.
    max_relative_target: float | dict[str, float] | None = field(default_factory=sourccey_max_relative_target)

    # -------------------------------------------------------------------------
    # Recovery Pathing
    # -------------------------------------------------------------------------
    # `enable_recovery_pathing` inserts temporary intermediate poses only after repeated stalled progress.
    # It is disabled by default so normal direct behavior is unchanged unless explicitly enabled.
    enable_recovery_pathing: bool = True

    # `recovery_stall_window` is the number of consecutive stalled action cycles required before recovery starts.
    recovery_stall_window: int = 5

    # `recovery_min_progress` is the minimum per-cycle joint movement that still counts as progress.
    recovery_min_progress: float = 0.5

    # `recovery_min_remaining_error` is the minimum remaining joint error that still counts as stuck.
    recovery_min_remaining_error: float = 4.0

    # `recovery_stage_hold_cycles` is how many action cycles each temporary recovery waypoint is held.
    recovery_stage_hold_cycles: int = 4

    # `recovery_joint_backoff` is how far a stalled joint retreats opposite the blocked direction.
    recovery_joint_backoff: float = 4.0

    # `recovery_posture_step` is how far recovery nudges posture joints toward the neutral recovery pose.
    recovery_posture_step: float = 8.0

    # `recovery_neutral_pose_value` is the neutral joint value used during the posture-tuck stage.
    recovery_neutral_pose_value: float = 0.0

    # -------------------------------------------------------------------------
    # Current Safety
    # -------------------------------------------------------------------------
    # `max_current_safety_threshold` is the fallback runtime current threshold for joints without a tuned override.
    max_current_safety_threshold: int = 2500

    # `gripper_current_safety_threshold` optionally overrides the gripper's runtime current threshold.
    # If left as `None`, the safety module uses a tuned default for the gripper.
    gripper_current_safety_threshold: int | None = None

    # `current_rest_safety_threshold_ratio` triggers an early unload step before the hard current limit.
    # A value of `0.7` means "start trying to rest the joint at 70% of its current safety limit."
    # Set this to `None` to disable the early-rest behavior.
    current_rest_safety_threshold_ratio: float | None = 0.7

    # `current_safe_release_threshold_ratio` is the lower current level a protected joint must fall below
    # before the safety latch is allowed to release it back to incoming streamed commands.
    current_safe_release_threshold_ratio: float = 0.35

    # `current_safe_hold_cycles` is how many consecutive low-current cycles a protected joint must remain
    # stable before it is released back to the normal command stream.
    current_safe_hold_cycles: int = 3

    # `disable_torque_on_hard_overcurrent` turns off torque on a joint after a hard overcurrent event
    # instead of continuing to hold it under load. This is the most aggressive software protection mode.
    disable_torque_on_hard_overcurrent: bool = True

    # `current_rest_backoff` is the small retreat used when a joint is drawing elevated current but
    # has not yet reached the hard overcurrent threshold.
    current_rest_backoff: float = 0.75

    # `gripper_current_rest_backoff` is a smaller early-rest retreat for the gripper.
    gripper_current_rest_backoff: float = 0.25

    # `current_safety_backoff` is how far to retreat a joint after an overcurrent event while it is still
    # being commanded deeper into the obstruction.
    current_safety_backoff: float = 2.0

    # `gripper_current_safety_backoff` is a smaller retreat so the gripper can keep a light hold on objects.
    gripper_current_safety_backoff: float = 0.5

    # -------------------------------------------------------------------------
    # Calibration Safety
    # -------------------------------------------------------------------------
    # `max_current_calibration_threshold` is the maximum current threshold for calibration purposes.
    max_current_calibration_threshold: int = 75

    # -------------------------------------------------------------------------
    # Sensors
    # -------------------------------------------------------------------------
    cameras: dict[str, CameraConfig] = field(default_factory=sourccey_cameras_config)

    # -------------------------------------------------------------------------
    # Compatibility
    # -------------------------------------------------------------------------
    # Set to `True` for backward compatibility with previous policies/dataset.
    use_degrees: bool = False
