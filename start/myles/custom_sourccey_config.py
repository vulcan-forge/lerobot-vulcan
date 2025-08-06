#!/usr/bin/env python3

from dataclasses import dataclass, field

from lerobot.common.cameras.configs import CameraConfig
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.robots.config import RobotConfig

def custom_sourccey_cameras_config() -> dict[str, CameraConfig]:
    """Custom camera configuration using the working camera devices"""
    return {
        "front_left": OpenCVCameraConfig(
            index_or_path="/dev/video2", fps=30, width=320, height=240
        ),
        "front_right": OpenCVCameraConfig(
            index_or_path="/dev/video4", fps=30, width=320, height=240
        ),
        "wrist_left": OpenCVCameraConfig(
            index_or_path="/dev/video6", fps=30, width=320, height=240
        ),
        # Note: You only have 3 working cameras, so we'll use video6 for wrist_right too
        # or you can comment this out if you don't need 4 cameras
        "wrist_right": OpenCVCameraConfig(
            index_or_path="/dev/video6", fps=30, width=320, height=240
        ),
    }

@RobotConfig.register_subclass("custom_sourccey_v2beta")
@dataclass
class CustomSourcceyV2BetaConfig(RobotConfig):
    left_arm_port: str = "/dev/ttyACM0"
    right_arm_port: str = "/dev/ttyACM1"

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    max_relative_target: int | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=custom_sourccey_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

@dataclass
class CustomSourcceyV2BetaHostConfig:
    # Network Configuration
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    # Duration of the application
    connection_timeout_s: int = 86400

    # Watchdog: stop the robot if no command is received for over 1 hour.
    watchdog_timeout_ms: int = 3600000

    # If robot jitters decrease the frequency and monitor cpu load with `top` in cmd
    max_loop_freq_hz: int = 30 