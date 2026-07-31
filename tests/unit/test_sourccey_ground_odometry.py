from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from sourccey_camera_geometry import default_bottom  # noqa: E402
from sourccey_ground_odometry import estimate_ground_translation_from_tracks  # noqa: E402


def _project_floor_points(points_robot_xy: np.ndarray) -> np.ndarray:
    model = default_bottom()
    points = np.asarray(points_robot_xy, dtype=np.float64)
    camera = points - np.asarray(
        [model.forward_offset_m, model.lateral_offset_m], dtype=np.float64
    )
    distance = np.hypot(camera[:, 0], camera[:, 1])
    bearing = np.degrees(np.arctan2(camera[:, 1], camera[:, 0]))
    x_ratio = 0.5 + (model.yaw_deg - bearing / model.bearing_sign) / model.hfov_deg
    depression = np.degrees(np.arctan2(model.height_m, distance))
    y_ratio = 0.5 + (depression - model.pitch_down_deg) / model.vfov_deg
    return np.column_stack((x_ratio * 319.0, y_ratio * 239.0))


def test_ground_translation_uses_imu_yaw_and_camera_lever_arm() -> None:
    model = default_bottom()
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
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
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
    np.testing.assert_allclose(
        [result.forward_m, result.left_m], expected, atol=0.003
    )


def test_ground_translation_rejects_sparse_tracks() -> None:
    pixels = np.asarray([[100.0, 180.0], [120.0, 190.0]])
    result = estimate_ground_translation_from_tracks(
        pixels,
        pixels,
        (240, 320),
        default_bottom(),
        0.0,
        minimum_support=10,
    )
    assert result is None
