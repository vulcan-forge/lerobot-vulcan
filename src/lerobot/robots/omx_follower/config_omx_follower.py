# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from ..config import RobotConfig
from ..so_follower.config_so_follower import SO100FollowerConfig


@RobotConfig.register_subclass("omx_follower")
@dataclass
class OmxFollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

    reverse: bool = False


@RobotConfig.register_subclass("so100_follower_end_effector")
@dataclass
class SO100FollowerEndEffectorConfig(SO100FollowerConfig):
    """Configure an SO-100 follower controlled through its end-effector pose.

    The inherited `id` and `calibration_dir` select the arm's saved motor calibration.
    Calibrate the arm before using Cartesian control.

    Args:
        port (`str`):
            Serial device for the follower's motor bus, such as `/dev/ttyACM0` or `COM3`.
        disable_torque_on_disconnect (`bool`, *optional*, defaults to `True`):
            Release the motors when disconnecting the arm.
        max_relative_target (`float | dict[str, float] | None`, *optional*):
            Limit each commanded joint displacement, globally or by motor name.
        cameras (`dict`, *optional*):
            Camera configurations indexed by observation name.
        use_degrees (`bool`, *optional*, defaults to `True`):
            Express joint positions in degrees rather than normalized motor units.
        position_p_coefficient (`int`, *optional*, defaults to 16):
            Proportional position-control gain written to Feetech motors on connection.
        position_i_coefficient (`int`, *optional*, defaults to 0):
            Integral position-control gain written to Feetech motors on connection.
        position_d_coefficient (`int`, *optional*, defaults to 32):
            Derivative position-control gain written to Feetech motors on connection.
        num_read_retries (`int`, *optional*, defaults to 2):
            Extra attempts after a failed motor read.
        urdf_path (`str | None`, *optional*):
            URDF model used for forward and inverse kinematics.
        target_frame_name (`str`, *optional*, defaults to `"gripper_frame_link"`):
            URDF frame whose Cartesian pose is controlled.
        end_effector_bounds (`dict`, *optional*):
            Minimum and maximum Cartesian coordinates in meters.
        max_gripper_pos (`float`, *optional*, defaults to 50):
            Upper bound on the gripper position command.
        end_effector_step_sizes (`dict`, *optional*):
            Cartesian increments in meters for the x, y and z axes.
        id (`str | None`, *optional*):
            Arm identifier used to select its saved calibration file.
        calibration_dir (`pathlib.Path | None`, *optional*):
            Directory containing motor calibration files; defaults to LeRobot's calibration cache.
    """

    # Path to URDF file for kinematics
    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo:
    # https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    urdf_path: str | None = None

    # End-effector frame name in the URDF
    target_frame_name: str = "gripper_frame_link"

    # Default bounds for the end-effector position (in meters)
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "min": [-1.0, -1.0, -1.0],  # min x, y, z
            "max": [1.0, 1.0, 1.0],  # max x, y, z
        }
    )

    max_gripper_pos: float = 50

    end_effector_step_sizes: dict[str, float] = field(
        default_factory=lambda: {
            "x": 0.02,
            "y": 0.02,
            "z": 0.02,
        }
    )
