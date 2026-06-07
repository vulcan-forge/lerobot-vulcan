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
from .modules.slam import SlamInputConfig


def sourccey_cameras_config(
    *,
    front_fps: int = 30,
    front_width: int = 320,
    front_height: int = 240,
    front_fourcc: str | None = None,
    wrist_fps: int = 30,
    wrist_width: int = 320,
    wrist_height: int = 240,
    wrist_fourcc: str | None = None,
    include_wrist: bool = True,
) -> dict[str, CameraConfig]:
    config = {
        "front_left": OpenCVCameraConfig(
            index_or_path="/dev/cameraFrontLeft",
            fps=front_fps,
            width=front_width,
            height=front_height,
            fourcc=front_fourcc,
            auto_reconnect=True,
            max_consecutive_read_failures=2,
            fast_reconnect_interval_s=0.05,
            fast_reconnect_window_s=2.0,
            reconnect_interval_s=0.5,
        ),
        "front_right": OpenCVCameraConfig(
            index_or_path="/dev/cameraFrontRight",
            fps=front_fps,
            width=front_width,
            height=front_height,
            fourcc=front_fourcc,
            auto_reconnect=True,
            max_consecutive_read_failures=2,
            fast_reconnect_interval_s=0.05,
            fast_reconnect_window_s=2.0,
            reconnect_interval_s=0.5,
        ),
    }
    if include_wrist:
        config["wrist_left"] = OpenCVCameraConfig(
            index_or_path="/dev/cameraWristLeft",
            fps=wrist_fps,
            width=wrist_width,
            height=wrist_height,
            fourcc=wrist_fourcc,
            auto_reconnect=True,
            max_consecutive_read_failures=2,
            fast_reconnect_interval_s=0.05,
            fast_reconnect_window_s=2.0,
            reconnect_interval_s=0.5,
        )
        config["wrist_right"] = OpenCVCameraConfig(
            index_or_path="/dev/cameraWristRight",
            fps=wrist_fps,
            width=wrist_width,
            height=wrist_height,
            fourcc=wrist_fourcc,
            auto_reconnect=True,
            max_consecutive_read_failures=2,
            fast_reconnect_interval_s=0.05,
            fast_reconnect_window_s=2.0,
            reconnect_interval_s=0.5,
        )
    return config


def sourccey_slam_eye_only_cameras_config(
    *,
    front_fps: int = 60,
    front_width: int = 320,
    front_height: int = 240,
    front_fourcc: str | None = None,
    include_wrist: bool = False,
) -> dict[str, CameraConfig]:
    return sourccey_cameras_config(
        front_fps=front_fps,
        front_width=front_width,
        front_height=front_height,
        front_fourcc=front_fourcc,
        wrist_fps=front_fps,
        wrist_width=front_width,
        wrist_height=front_height,
        wrist_fourcc=front_fourcc,
        include_wrist=include_wrist,
    )


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
    watchdog_timeout_ms: int = 60000

    # If robot jitters decrease the frequency and monitor cpu load with `top` in cmd
    max_loop_freq_hz: int = 30
    arm_connect_on_startup: bool = False
    arm_calibrate_on_connect: bool = False
    arm_relax_on_startup: bool = True
    slam_eye_only_mode: bool = False
    slam_eye_camera_fps: int = 30
    slam_eye_loop_freq_hz: int = 30
    slam_eye_width: int = 320
    slam_eye_height: int = 240
    slam_eye_fourcc: str | None = "MJPG"
    # 0=disabled, 1=50Hz, 2=60Hz. US indoor lighting usually wants 2.
    slam_eye_power_line_frequency: int = 2
    # 1=manual, 3=aperture priority for these UVC cameras.
    slam_eye_auto_exposure: int | None = 3
    slam_eye_exposure_dynamic_framerate: bool = True
    slam_eye_exposure_time_absolute: int | None = None
    slam_eye_gain: int | None = None
    slam_eye_sharpness: int | None = None
    slam_eye_backlight_compensation: int | None = None
    # Keep camera fallback warnings visible in SLAM mode so black-frame issues aren't silent.
    slam_eye_log_camera_warnings: bool = True
    # Direct SLAM sidecar publishing from the robot host.
    slam_input_enabled: bool = False
    slam_input_endpoint: str = "tcp://*:5560"
    slam_stereo_left_key: str = "front_left"
    slam_stereo_right_key: str = "front_right"
    slam_jpeg_quality: int = 80
    slam_publish_eye_only_mode: bool = True
    slam_publish_fps: float = 0.0
    slam_resize_width: int | None = None
    slam_resize_height: int | None = None
    # Secondary wrist-camera stereo stream for near-field obstacle detection.
    slam_obstacle_input_enabled: bool = False
    slam_obstacle_input_endpoint: str = "tcp://*:5562"
    slam_obstacle_stereo_left_key: str = "wrist_left"
    slam_obstacle_stereo_right_key: str = "wrist_right"
    slam_obstacle_jpeg_quality: int = 80
    slam_obstacle_publish_eye_only_mode: bool = True
    slam_obstacle_publish_fps: float = 15.0
    slam_obstacle_resize_width: int | None = 320
    slam_obstacle_resize_height: int | None = 240

    # IMU periodic logging on host (disabled by default to avoid loop spam)
    imu_print_enabled: bool = False
    imu_print_interval_s: float = 10.0
    imu_bus_num: int = 1
    imu_lsm6dsox_address: int = 0x6A
    imu_lis3mdl_address: int = 0x1C


@RobotConfig.register_subclass("sourccey_client")
@dataclass
class SourcceyClientConfig(RobotConfig):
    # Network Configuration
    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    # SLAM sidecar input stream (sourccey-slam expects slam_input.v1).
    # Canonical config lives under this nested field.
    slam: SlamInputConfig = field(default_factory=SlamInputConfig)
    # Backward-compatibility aliases for existing commands/docs.
    # If provided, these values override the nested slam config in __post_init__.
    slam_input_enabled: bool | None = None
    slam_input_endpoint: str | None = None
    slam_stereo_left_key: str | None = None
    slam_stereo_right_key: str | None = None
    slam_jpeg_quality: int | None = None
    slam_eye_only_mode: bool | None = None
    slam_publish_fps: float | None = None
    slam_resize_width: int | None = None
    slam_resize_height: int | None = None

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            "up": "q",
            "down": "e",
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
    # Toggle periodic timeout logs when no observation packet arrives.
    log_no_data_timeouts: bool = True
    # Minimum interval between timeout log lines (seconds) when logging is enabled.
    no_data_log_interval_s: float = 5.0
    connect_timeout_s: int = 5

    def __post_init__(self) -> None:
        super().__post_init__()

        # Migrate flat legacy flags into nested SLAM config when explicitly provided.
        if self.slam_input_enabled is not None:
            self.slam.input_enabled = self.slam_input_enabled
        if self.slam_input_endpoint is not None:
            self.slam.input_endpoint = self.slam_input_endpoint
        if self.slam_stereo_left_key is not None:
            self.slam.stereo_left_key = self.slam_stereo_left_key
        if self.slam_stereo_right_key is not None:
            self.slam.stereo_right_key = self.slam_stereo_right_key
        if self.slam_jpeg_quality is not None:
            self.slam.jpeg_quality = self.slam_jpeg_quality
        if self.slam_eye_only_mode is not None:
            self.slam.eye_only_mode = self.slam_eye_only_mode
        if self.slam_publish_fps is not None:
            self.slam.publish_fps = self.slam_publish_fps
        if self.slam_resize_width is not None:
            self.slam.resize_width = self.slam_resize_width
        if self.slam_resize_height is not None:
            self.slam.resize_height = self.slam_resize_height
