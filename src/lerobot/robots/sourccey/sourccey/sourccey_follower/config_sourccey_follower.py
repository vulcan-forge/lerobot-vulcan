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

@RobotConfig.register_subclass("sourccey_follower")
@dataclass
class SourcceyFollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str
    orientation: str = "left"

    # The models of the motors to use for the follower arms.
    motor_models: dict[str, str] = field(default_factory=sourccey_motor_models)

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # `max_current_safety_threshold` is the maximum current threshold for safety purposes.
    max_current_safety_threshold: int = 2500

    # `max_current_calibration_threshold` is the maximum current threshold for calibration purposes.
    max_current_calibration_threshold: int = 75

    # Gripper force control settings
    # When enabled, the gripper will stop closing when it detects resistance (object contact)
    # instead of just going to the commanded position.
    gripper_force_control_enabled: bool = True
    # Current threshold (mA) to detect object contact - when exceeded, gripper stops closing
    gripper_contact_current_threshold: int = 220
    # Current threshold (mA) for grip force - gripper will try to maintain this grip strength
    gripper_grip_current_threshold: int = 500
    # Position deadband - minimum position change to consider gripper "closing" (in normalized units)
    gripper_closing_deadband: float = 1.0
    # Position threshold considered "fully closed enough" for thin-object squeeze support.
    gripper_fully_closed_threshold: float = 5.0
    # Enable an additional squeeze mode when gripper is near fully closed.
    gripper_post_close_squeeze_enabled: bool = True
    # Normal gripper safety/torque limits used during free motion.
    gripper_nominal_max_torque_limit: int = 700
    gripper_nominal_protection_current: int = 550
    # Slightly stronger limits used only while squeezing near closed position.
    gripper_squeeze_max_torque_limit: int = 1000
    gripper_squeeze_protection_current: int = 800
    # Optional: command a small raw-position over-close past normalized "0"
    # while keeping current bounded, to improve thin-fabric grip.
    gripper_overclose_enabled: bool = True
    gripper_overclose_steps: int = 90
    gripper_overclose_max_current_threshold: int = 700

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=sourccey_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False
