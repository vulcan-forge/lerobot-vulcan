import json

import numpy as np

from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import (
    SourcceyHostConfig,
    sourccey_slam_eye_only_cameras_config,
)
from lerobot.robots.sourccey.sourccey.sourccey.sourccey_host import (
    _build_host_slam_input_publisher,
    _build_host_slam_obstacle_publisher,
    _build_slam_eye_v4l2_controls,
    _handle_command_watchdog_timeout,
)
from lerobot.sensors.imu.types import IMUSample


def test_slam_eye_only_cameras_config_uses_requested_mode() -> None:
    cameras = sourccey_slam_eye_only_cameras_config(
        front_fps=30,
        front_width=640,
        front_height=480,
        front_fourcc="MJPG",
    )

    assert set(cameras) == {"front_left", "front_right"}
    for camera in cameras.values():
        assert camera.fps == 30
        assert camera.width == 640
        assert camera.height == 480
        assert camera.fourcc == "MJPG"


def test_slam_eye_only_cameras_config_can_include_wrist_cameras() -> None:
    cameras = sourccey_slam_eye_only_cameras_config(
        front_fps=30,
        front_width=320,
        front_height=240,
        front_fourcc="MJPG",
        include_wrist=True,
    )

    assert set(cameras) == {"front_left", "front_right", "wrist_left", "wrist_right"}


def test_build_slam_eye_v4l2_controls_includes_experiment_knobs() -> None:
    config = SourcceyHostConfig(
        slam_eye_power_line_frequency=2,
        slam_eye_auto_exposure=3,
        slam_eye_exposure_dynamic_framerate=False,
        slam_eye_exposure_time_absolute=180,
        slam_eye_gain=12,
        slam_eye_sharpness=4,
        slam_eye_backlight_compensation=32,
    )

    controls = _build_slam_eye_v4l2_controls(config)

    assert controls == {
        "power_line_frequency": 2,
        "auto_exposure": 3,
        "exposure_dynamic_framerate": 0,
        "exposure_time_absolute": 180,
        "gain": 12,
        "sharpness": 4,
        "backlight_compensation": 32,
    }


def test_build_slam_eye_v4l2_controls_omits_optional_controls_when_unset() -> None:
    config = SourcceyHostConfig(
        slam_eye_auto_exposure=None,
        slam_eye_exposure_time_absolute=None,
        slam_eye_gain=None,
        slam_eye_sharpness=None,
        slam_eye_backlight_compensation=None,
    )

    controls = _build_slam_eye_v4l2_controls(config)

    assert controls == {
        "power_line_frequency": 2,
        "exposure_dynamic_framerate": 0,
    }


def test_build_host_slam_input_publisher_uses_host_source_and_knobs() -> None:
    config = SourcceyHostConfig(
        slam_input_enabled=True,
        slam_stereo_left_key="front_left",
        slam_stereo_right_key="front_right",
        slam_jpeg_quality=62,
        slam_publish_eye_only_mode=True,
        slam_publish_fps=5.0,
        slam_resize_width=16,
        slam_resize_height=12,
    )

    publisher = _build_host_slam_input_publisher(config)

    assert publisher is not None
    payload = publisher.build_packet(
        observation={"x.vel": 0.1, "y.vel": 0.0, "theta.vel": -0.2},
        frames={
            "front_left": np.full((24, 24, 3), 32, dtype=np.uint8),
            "front_right": np.full((24, 24, 3), 64, dtype=np.uint8),
            "wrist_left": np.full((24, 24, 3), 96, dtype=np.uint8),
        },
    )

    assert payload is not None
    packet = json.loads(payload.decode("utf-8"))
    assert packet["source"] == "sourccey_host:sourccey"
    assert packet["stereo_left"] == "front_left"
    assert packet["stereo_right"] == "front_right"
    assert set(packet["cameras"]) == {"front_left", "front_right"}


def test_build_host_slam_obstacle_publisher_uses_wrist_pair() -> None:
    config = SourcceyHostConfig(
        slam_obstacle_input_enabled=True,
        slam_obstacle_stereo_left_key="wrist_left",
        slam_obstacle_stereo_right_key="wrist_right",
        slam_obstacle_jpeg_quality=70,
        slam_obstacle_publish_eye_only_mode=True,
        slam_obstacle_publish_fps=10.0,
        slam_obstacle_resize_width=20,
        slam_obstacle_resize_height=16,
    )

    publisher = _build_host_slam_obstacle_publisher(config)

    assert publisher is not None
    payload = publisher.build_packet(
        observation={"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0},
        frames={
            "front_left": np.full((24, 24, 3), 32, dtype=np.uint8),
            "wrist_left": np.full((24, 24, 3), 96, dtype=np.uint8),
            "wrist_right": np.full((24, 24, 3), 128, dtype=np.uint8),
        },
    )

    assert payload is not None
    packet = json.loads(payload.decode("utf-8"))
    assert packet["source"] == "sourccey_host_obstacle:sourccey"
    assert packet["stereo_left"] == "wrist_left"
    assert packet["stereo_right"] == "wrist_right"
    assert set(packet["cameras"]) == {"wrist_left", "wrist_right"}


def test_build_host_slam_input_publisher_serializes_imu_samples() -> None:
    config = SourcceyHostConfig(
        slam_input_enabled=True,
        slam_imu_enabled=True,
    )

    publisher = _build_host_slam_input_publisher(config)

    assert publisher is not None
    payload = publisher.build_packet(
        observation={"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0},
        frames={
            "front_left": np.full((24, 24, 3), 32, dtype=np.uint8),
            "front_right": np.full((24, 24, 3), 64, dtype=np.uint8),
        },
        imu_samples=[
            IMUSample(
                timestamp_ns=555,
                accel_m_s2=(1.0, 0.0, -1.0),
                gyro_rad_s=(0.1, 0.2, 0.3),
                mag_uT=(4.0, 5.0, 6.0),
                temperature_c=26.0,
                valid=True,
            )
        ],
    )

    assert payload is not None
    packet = json.loads(payload.decode("utf-8"))
    assert packet["imu_samples"] == [
        {
            "capture_monotonic_ns": 555,
            "ax": 1.0,
            "ay": 0.0,
            "az": -1.0,
            "gx": 0.1,
            "gy": 0.2,
            "gz": 0.3,
            "mx": 4.0,
            "my": 5.0,
            "mz": 6.0,
            "temperature_c": 26.0,
        }
    ]


def test_handle_command_watchdog_timeout_stops_motion() -> None:
    class _FakeRobot:
        def __init__(self) -> None:
            self.stop_calls = 0
            self.update_calls = 0

        def watchdog_stop_motion(self) -> None:
            self.stop_calls += 1

        def update(self) -> None:
            self.update_calls += 1

    robot = _FakeRobot()

    _handle_command_watchdog_timeout(robot, 250)

    assert robot.stop_calls == 1
    assert robot.update_calls == 1
