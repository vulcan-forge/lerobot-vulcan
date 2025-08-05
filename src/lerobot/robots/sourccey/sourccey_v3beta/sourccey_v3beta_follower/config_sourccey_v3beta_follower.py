from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.config import RobotConfig


def sourccey_v3beta_cameras_config() -> dict[str, CameraConfig]:
    return {
        "wrist": OpenCVCameraConfig(
            index_or_path="/dev/video0", fps=30, width=640, height=480
        ),
    }

@RobotConfig.register_subclass("sourccey_v3beta_follower")
@dataclass
class SourcceyV3BetaFollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str
    orientation: str = "left"

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # `max_current_safety_threshold` is the maximum current threshold for safety purposes.
    max_current_safety_threshold: int = 500

    # `min_action_threshold` is the minimum action threshold for motors during ai evaluation
    # to avoid sticktion in the gearbox preventing the arm from moving.
    min_action_threshold: float = 0.5

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=sourccey_v3beta_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False
