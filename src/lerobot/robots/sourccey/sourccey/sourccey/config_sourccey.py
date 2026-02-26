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
import os

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.motors.dc_motors_controller import DCMotor, MotorNormMode

from lerobot.robots.config import RobotConfig


DEFAULT_CAMERA_PROFILE = "default"
SINGLE_MAX_RES_CAMERA_PROFILE = "locate_single_max_res"

_LOCATE_CAMERA_PROFILES = {
    "locate",
    "locate_mode",
    "single_max_res",
    "single_max_res_camera",
    "locate_single_max_res",
}
_DEFAULT_CAMERA_PROFILES = {
    "",
    "default",
    "model",
    "model_mode",
    "activate_model",
    "activation",
    "all_cameras",
    "four_cameras",
    "four_cam",
    "4cam",
}


def normalize_sourccey_camera_profile(profile: str | None) -> str:
    raw = str(profile or "").strip().lower()
    if raw in _LOCATE_CAMERA_PROFILES:
        return SINGLE_MAX_RES_CAMERA_PROFILE
    if raw in _DEFAULT_CAMERA_PROFILES:
        return DEFAULT_CAMERA_PROFILE
    return raw


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return max(int(minimum), int(value))


def _env_str(name: str, default: str) -> str:
    raw = str(os.getenv(name, "")).strip()
    return raw if raw else str(default)


def _coerce_positive_int(value: int | float | str | None, default: int, *, minimum: int = 1) -> int:
    if value is None:
        return int(default)
    try:
        return max(int(minimum), int(value))
    except (TypeError, ValueError):
        return int(default)


def _default_cameras_config() -> dict[str, CameraConfig]:
    return {
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


def _default_device_for_camera_name(camera_name: str) -> str:
    mapping = {
        "front_left": "/dev/cameraFrontLeft",
        "front_right": "/dev/cameraFrontRight",
        "wrist_left": "/dev/cameraWristLeft",
        "wrist_right": "/dev/cameraWristRight",
    }
    return mapping.get(camera_name, "/dev/cameraFrontRight")


def _single_max_res_cameras_config(
    *,
    camera_name: str | None = None,
    device: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
) -> dict[str, CameraConfig]:
    """Single-camera locate profile tuned for better visual target quality."""
    default_camera_name = _env_str("SOURCCEY_LOCATE_SINGLE_CAMERA_NAME", "front_right")
    selected_camera_name = str(camera_name).strip() if camera_name is not None else default_camera_name
    if not selected_camera_name:
        selected_camera_name = "front_right"

    default_device = _env_str("SOURCCEY_LOCATE_SINGLE_CAMERA_DEVICE", _default_device_for_camera_name(selected_camera_name))
    default_width = _env_int("SOURCCEY_LOCATE_SINGLE_CAMERA_WIDTH", 1280)
    default_height = _env_int("SOURCCEY_LOCATE_SINGLE_CAMERA_HEIGHT", 720)
    default_fps = _env_int("SOURCCEY_LOCATE_SINGLE_CAMERA_FPS", 30)

    device_value = str(device).strip() if device is not None else default_device
    if not device_value:
        device_value = default_device

    return {
        selected_camera_name: OpenCVCameraConfig(
            index_or_path=device_value,
            fps=_coerce_positive_int(fps, default_fps),
            width=_coerce_positive_int(width, default_width),
            height=_coerce_positive_int(height, default_height),
        ),
    }


def resolve_sourccey_cameras_config(
    profile: str | None = None,
    *,
    camera_name: str | None = None,
    device: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
) -> dict[str, CameraConfig]:
    resolved_profile = normalize_sourccey_camera_profile(
        os.getenv("SOURCCEY_CAMERA_PROFILE", "") if profile is None else profile
    )
    if resolved_profile == SINGLE_MAX_RES_CAMERA_PROFILE:
        return _single_max_res_cameras_config(
            camera_name=camera_name,
            device=device,
            width=width,
            height=height,
            fps=fps,
        )
    return _default_cameras_config()


def sourccey_cameras_config() -> dict[str, CameraConfig]:
    return resolve_sourccey_cameras_config()

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

    # Watchdog: stop the robot if no command is received for over 1 hour.
    watchdog_timeout_ms: int = 3600000

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
            # Camera profile switching
            "camera_locate_mode": "1",
            "camera_model_mode": "2",
            # quit teleop
            "quit": "space",
        }
    )

    cameras: dict[str, CameraConfig] = field(default_factory=sourccey_cameras_config)

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5

    reverse: bool = False
