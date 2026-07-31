from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from sourccey_bottom_camera import (  # noqa: E402
    BottomGroundSafetyGate,
    FloorObstacle,
    bottom_obstacle_explains_floor_point,
    default_bottom_camera_model,
    estimate_ground_translation_from_tracks,
    wait_for_live_bottom_camera,
)


class _FrameSequence:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.index = 0

    def latest_sample(self, _cam_name: str):
        index = min(self.index, len(self.frames) - 1)
        frame = self.frames[index]
        self.index += 1
        return frame, 0.0, min(self.index, len(self.frames))


def _textured_frame(shift: int = 0) -> np.ndarray:
    yy, xx = np.indices((120, 160))
    gray = (((xx + shift) // 8 + yy // 8) % 2 * 180 + 35).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def test_bottom_stream_health_accepts_fresh_changing_textured_frames() -> None:
    source = _FrameSequence([_textured_frame(i) for i in range(3)])
    health = wait_for_live_bottom_camera(source, timeout_s=0.3)

    assert health.ready
    assert health.fresh_frames >= 3
    assert health.usable_frames >= 3


def test_bottom_stream_health_rejects_repeated_placeholder() -> None:
    source = _FrameSequence([np.zeros((120, 160, 3), dtype=np.uint8)] * 3)
    health = wait_for_live_bottom_camera(source, timeout_s=0.2)

    assert not health.ready
    assert "usable" in health.reason or "contrast" in health.reason


def _project_floor_points(points_robot_xy: np.ndarray) -> np.ndarray:
    model = default_bottom_camera_model()
    points = np.asarray(points_robot_xy, dtype=np.float64)
    camera = points - np.asarray([model.forward_offset_m, model.lateral_offset_m], dtype=np.float64)
    distance = np.hypot(camera[:, 0], camera[:, 1])
    bearing = np.degrees(np.arctan2(camera[:, 1], camera[:, 0]))
    x_ratio = 0.5 + (model.yaw_deg - bearing / model.bearing_sign) / model.hfov_deg
    depression = np.degrees(np.arctan2(model.height_m, distance))
    y_ratio = 0.5 + (depression - model.pitch_down_deg) / model.vfov_deg
    return np.column_stack((x_ratio * 319.0, y_ratio * 239.0))


def test_ground_translation_uses_imu_yaw_and_camera_lever_arm() -> None:
    model = default_bottom_camera_model()
    current_points = np.asarray(
        [
            [forward, lateral]
            for forward in np.linspace(0.36, 0.82, 7)
            for lateral in np.linspace(-0.20, 0.20, 5)
        ],
        dtype=np.float64,
    )
    yaw_delta_deg = 7.0
    angle = math.radians(yaw_delta_deg)
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    expected = np.asarray([0.055, -0.014], dtype=np.float64)
    previous_points = expected + current_points @ rotation.T

    result = estimate_ground_translation_from_tracks(
        _project_floor_points(previous_points),
        _project_floor_points(current_points),
        (240, 320),
        model,
        yaw_delta_deg,
    )

    assert result is not None
    assert result.support >= 30
    np.testing.assert_allclose([result.forward_m, result.left_m], expected, atol=0.003)


def test_ground_translation_rejects_sparse_tracks() -> None:
    pixels = np.asarray([[100.0, 180.0], [120.0, 190.0]])
    result = estimate_ground_translation_from_tracks(
        pixels,
        pixels,
        (240, 320),
        default_bottom_camera_model(),
        0.0,
        minimum_support=10,
    )
    assert result is None


def _bottom_safety_settings() -> SimpleNamespace:
    return SimpleNamespace(
        ground_gate_enabled=True,
        ground_stop_distance_m=0.55,
        ground_corridor_half_width_m=0.35,
        ground_trip_frames=2,
        ground_clear_frames=2,
        bottom_bearing_tolerance_deg=9.0,
        bottom_distance_tolerance_m=0.40,
    )


def test_bottom_ground_gate_owns_trip_and_clear_hysteresis() -> None:
    settings = _bottom_safety_settings()
    gate = BottomGroundSafetyGate(settings)
    obstacle = FloorObstacle(
        bearing_deg=15.0,
        distance_m=0.30,
        x_ratio=0.5,
        y_ratio=0.8,
        height_m=0.16,
        crosses_horizon=True,
    )

    assert not gate.update([obstacle], camera_available=True).active
    tripped = gate.update([obstacle], camera_available=True)
    assert tripped.active
    assert tripped.side == "left"
    assert tripped.distance_m == 0.30

    assert gate.update([], camera_available=True).active
    assert not gate.update([], camera_available=True).active
    assert not gate.update([obstacle], camera_available=False).active


def test_bottom_obstacle_can_referee_an_eye_floor_projection() -> None:
    settings = _bottom_safety_settings()
    obstacle = FloorObstacle(
        bearing_deg=-12.0,
        distance_m=0.70,
        x_ratio=0.7,
        y_ratio=0.7,
    )

    assert bottom_obstacle_explains_floor_point(
        [obstacle],
        bearing_deg=-15.0,
        distance_m=0.82,
        settings=settings,
    )
    assert not bottom_obstacle_explains_floor_point(
        [obstacle],
        bearing_deg=25.0,
        distance_m=0.82,
        settings=settings,
    )
