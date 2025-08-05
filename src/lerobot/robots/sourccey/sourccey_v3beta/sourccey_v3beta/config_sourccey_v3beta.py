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

from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.motors.dc_motors_controller import DCMotor, MotorNormMode

from lerobot.robots.config import RobotConfig
from lerobot.constants import HF_LEROBOT_CONFIGURATION


def sourccey_v3beta_cameras_config() -> dict[str, CameraConfig]:
    config = {
        "front_left": OpenCVCameraConfig(
            index_or_path="/dev/cameraFrontLeft", fps=30, width=640, height=360
        ),
        "front_right": OpenCVCameraConfig(
            index_or_path="/dev/cameraFrontRight", fps=30, width=640, height=360
        ),
        "wrist_left": OpenCVCameraConfig(
            index_or_path="/dev/cameraWristLeft", fps=30, width=640, height=360
        ),
        "wrist_right": OpenCVCameraConfig(
            index_or_path="/dev/cameraWristRight", fps=30, width=640, height=360
        ),
    }
    return config

def sourccey_v3beta_dc_motors_config() -> dict[str, DCMotor]:
    return {
        "motors": {
            "front_left": DCMotor(id=1, model="mecanum_wheel", norm_mode=MotorNormMode.RANGE_M100_100),
            "front_right": DCMotor(id=2, model="mecanum_wheel", norm_mode=MotorNormMode.RANGE_M100_100),
            "rear_left": DCMotor(id=3, model="mecanum_wheel", norm_mode=MotorNormMode.RANGE_M100_100),
            "rear_right": DCMotor(id=4, model="mecanum_wheel", norm_mode=MotorNormMode.RANGE_M100_100),
            "actuator": DCMotor(id=5, model="linear_actuator", norm_mode=MotorNormMode.RANGE_M100_100),
        },
        "pwm_pins": [12, 13, 14, 15, 18],
        "direction_pins": [2, 3, 4, 5, 6],
        "enable_pins": [7, 8, 9, 10, 11],
        "pwm_frequency": 25000,
        "invert_direction": False,
        "invert_enable": False,
        "invert_brake": False,
    }

@RobotConfig.register_subclass("sourccey_v3beta")
@dataclass
class SourcceyV3BetaConfig(RobotConfig):
    left_arm_port: str = "/dev/robotLeftArm"
    right_arm_port: str = "/dev/robotRightArm"

    # Optional
    left_arm_disable_torque_on_disconnect: bool = True
    left_arm_max_relative_target: int | None = None
    left_arm_use_degrees: bool = False
    right_arm_disable_torque_on_disconnect: bool = True
    right_arm_max_relative_target: int | None = None
    right_arm_use_degrees: bool = False

    cameras: dict[str, CameraConfig] = field(default_factory=sourccey_v3beta_cameras_config)
    dc_motors: dict = field(default_factory=sourccey_v3beta_dc_motors_config)

@dataclass
class SourcceyV3BetaHostConfig:
    # Network Configuration
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    # Duration of the application
    connection_time_s: int = 86400

    # Watchdog: stop the robot if no command is received for over 1 hour.
    watchdog_timeout_ms: int = 3600000

    # If robot jitters decrease the frequency and monitor cpu load with `top` in cmd
    max_loop_freq_hz: int = 30


@RobotConfig.register_subclass("sourccey_v3beta_client")
@dataclass
class SourcceyV3BetaClientConfig(RobotConfig):
    # Network Configuration
    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # quit teleop
            "quit": "q",
        }
    )

    cameras: dict[str, CameraConfig] = field(default_factory=sourccey_v3beta_cameras_config)

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5
