from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import (
    SourcceyHostConfig,
    sourccey_slam_eye_only_cameras_config,
)
from lerobot.robots.sourccey.sourccey.sourccey.sourccey_host import _build_slam_eye_v4l2_controls


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
