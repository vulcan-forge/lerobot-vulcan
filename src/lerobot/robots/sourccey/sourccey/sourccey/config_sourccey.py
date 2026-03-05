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
from lerobot.motors.dc_motors_controller import DCMotor, MotorNormMode

from lerobot.robots.config import RobotConfig


def sourccey_cameras_config() -> dict[str, CameraConfig]:
    config = {
         "front_left": OpenCVCameraConfig(
             index_or_path="/dev/cameraFrontLeft", fps=30, width=320, height=240
         ),
         "front_right": OpenCVCameraConfig(
             index_or_path="/dev/cameraFrontRight", fps=30, width=320, height=240
         ),
         "wrist_left": OpenCVCameraConfig(
             index_or_path="/dev/cameraWristLeft", fps=30, width=320, height=240
         ),
         "wrist_right": OpenCVCameraConfig(
             index_or_path="/dev/cameraWristRight", fps=30, width=320, height=240
         ),
    }
    return config

def sourccey_motor_models() -> dict[str, str]:
    return {
        "shoulder_pan": "sts3215",
        "shoulder_lift": "sts3250",
        "elbow_flex": "sts3250",
        "wrist_flex": "sts3215",
        "wrist_roll": "sts3215",
        "gripper": "sts3215",
    }

def sourccey_dc_motors() -> dict[str, DCMotor]:
    return {
        "front_left": DCMotor(id=1, model="mecanum_wheel", norm_mode=MotorNormMode.PWM_DUTY_CYCLE),
        "front_right": DCMotor(id=2, model="mecanum_wheel", norm_mode=MotorNormMode.PWM_DUTY_CYCLE),
        "rear_left": DCMotor(id=3, model="mecanum_wheel", norm_mode=MotorNormMode.PWM_DUTY_CYCLE),
        "rear_right": DCMotor(id=4, model="mecanum_wheel", norm_mode=MotorNormMode.PWM_DUTY_CYCLE),
        "linear_actuator": DCMotor(id=5, model="linear_actuator", norm_mode=MotorNormMode.PWM_DUTY_CYCLE),
    }

def sourccey_dc_motors_config() -> dict:
    return {
        "in1_pins": [17,23,24,26,5], # Physical pins: [11, 16, 18, 37, 29]
        "in2_pins": [27,22,25,16,6], # Physical pins: [13, 15, 22, 36, 31]
        "pwm_frequency": 10000,  # 5 kHz - balance between performance and noise reduction
    }

@RobotConfig.register_subclass("sourccey")
@dataclass
class SourcceyConfig(RobotConfig):
    left_arm_port: str = "/dev/robotLeftArm"
    right_arm_port: str = "/dev/robotRightArm"

    left_arm_motor_models: dict[str, str] = field(default_factory=sourccey_motor_models)
    right_arm_motor_models: dict[str, str] = field(default_factory=sourccey_motor_models)

    cameras: dict[str, CameraConfig] = field(default_factory=sourccey_cameras_config)

    dc_motors_config: dict = field(default_factory=sourccey_dc_motors_config)
    dc_motors: dict = field(default_factory=sourccey_dc_motors)

    # Optional
    left_arm_disable_torque_on_disconnect: bool = True
    left_arm_max_relative_target: int | None = None
    left_arm_use_degrees: bool = False
    right_arm_disable_torque_on_disconnect: bool = True
    right_arm_max_relative_target: int | None = None
    right_arm_use_degrees: bool = False


@dataclass
class SourcceyHostConfig:
    # Network Configuration
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    # Text + audio (used by voice pipeline)
    port_zmq_text_in: int = 5557  # receive text from client
    port_zmq_text_out: int = 5558  # send text/events to client
    port_zmq_audio: int = 5559  # publish PCM16 audio stream

    # Duration of the application
    connection_time_s: int = 86400

    # Watchdog: if command stream stalls, immediately stop base and release arm torque.
    watchdog_timeout_ms: int = 500

    # If robot jitters decrease the frequency and monitor cpu load with `top` in cmd
    max_loop_freq_hz: int = 30


@RobotConfig.register_subclass("sourccey_client")
@dataclass
class SourcceyClientConfig(RobotConfig):
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
            "up": "e",
            "down": "q",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # Host control (toggle per-arm untorque)
            "untorque_left": "n",
            "untorque_right": "m",
            # quit teleop
            "quit": "space",
        }
    )

    cameras: dict[str, CameraConfig] = field(default_factory=sourccey_cameras_config)

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5

    reverse: bool = False

    # Debug capture: first N seconds after arm motion starts.
    debug_capture_enabled: bool = False
    debug_capture_duration_s: float = 5.0
    debug_capture_motion_threshold: float = 1.0
    debug_capture_path: str | None = None

    # Startup alignment: prevent large first policy command jumps after connect/reset.
    # This is not a hard stop; it smoothly blends from observed pose to policy target.
    startup_alignment_enabled: bool = True
    # Re-arm startup alignment if there has been no arm command for this long.
    startup_alignment_rearm_gap_s: float = 1.0
    # Engage alignment only when first-frame arm mismatch exceeds this amount (normalized units).
    startup_alignment_engage_delta: float = 35.0
    # Per-command max arm offset from current observation while alignment is active.
    startup_alignment_max_delta: float = 20.0
    # Maximum alignment phase duration before handing over to raw policy targets.
    startup_alignment_max_duration_s: float = 2.5
    # Alignment ends early once max arm mismatch drops below this amount.
    startup_alignment_complete_delta: float = 10.0
