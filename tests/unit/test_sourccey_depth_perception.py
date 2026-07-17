"""Regression tests for Sourccey's metric-depth safety geometry."""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import sourccey_depth_perception as depth_perception
from sourccey_camera_geometry import default_eye_right
from sourccey_elevated_safety import (
    ElevatedHazardMonitor,
    ElevatedSafetyConfig,
    HazardState,
    gate_forward_allowed,
)


def test_self_body_pixels_cannot_trigger_depth_gate(monkeypatch) -> None:
    """A retained -49 degree chassis return must not defeat squeeze mode.

    The 2026-07-12 field stop had a clear narrow lane but a 0.11 m
    near-freeze return from the right eye's view of the robot itself.  Keep
    metric scaling fixed so this test isolates the robot-frame mask.
    """
    monkeypatch.setattr(depth_perception, "_floor_scale", lambda *_: 1.0)
    result = depth_perception.analyze_depth(
        default_eye_right(),
        np.full((240, 320), 0.07, dtype=np.float32),
        eye="front_right",
        frame_monotonic=1.0,
    )

    assert result.nearest_gate_m is None
    assert result.nearest_gate_narrow_m is None
    assert result.nearest_any_m is None
    assert result.elevated_ratio == 0.0


def test_outer_near_eye_artifact_cannot_trigger_depth_gate(monkeypatch) -> None:
    """A retained 49-degree, 0.4m lens-rim return is outside forward travel."""
    shape = (24, 24)
    fields = {
        "fwd_unit": np.ones(shape, dtype=np.float32),
        "up_unit": np.full(shape, 0.4, dtype=np.float32),
        "x_unit": np.zeros(shape, dtype=np.float32),
        "bearing_deg": np.full(shape, 49.0, dtype=np.float32),
    }
    monkeypatch.setattr(depth_perception, "_pixel_fields", lambda *_: fields)
    monkeypatch.setattr(depth_perception, "_floor_scale", lambda *_: 1.0)
    model = replace(default_eye_right(), yaw_deg=0.0, forward_offset_m=0.0)

    result = depth_perception.analyze_depth(
        model,
        np.full(shape, 0.4, dtype=np.float32),
        eye="front_right",
        frame_monotonic=1.0,
    )

    assert result.nearest_gate_m is None
    assert result.nearest_any_m is None


def test_depth_map_publishes_table_boundary_not_entire_top(monkeypatch) -> None:
    """A full tabletop view must become a thin 2-D obstacle boundary."""
    model = default_eye_right()
    fields = depth_perception._pixel_fields(model, (240, 320))
    depth = np.full((240, 320), 8.0, dtype=np.float32)
    below_horizon = fields["up_unit"] < -0.02
    depth[below_horizon] = (0.70 - model.height_m) / fields["up_unit"][below_horizon]
    monkeypatch.setattr(depth_perception, "_floor_scale", lambda *_: 1.0)

    result = depth_perception.analyze_depth(
        model,
        depth,
        eye="front_right",
        frame_monotonic=1.0,
        proposal_line_xy=((60, 180), (260, 180)),
    )

    assert result.nearest_gate_m is not None
    assert 8 <= len(result.region_points_xy) <= 40


def test_floor_connected_door_face_is_delegated_to_lidar() -> None:
    """A door next to a table must not become part of the table's red edge."""
    elevated = np.zeros((12, 12), dtype=bool)
    elevated[2:7, 3:9] = True
    height = np.zeros((12, 12), dtype=np.float32)
    height[2:7, 3:9] = 0.65
    # Same-range vertical face continues from the elevated band down toward
    # the floor in every column, which is what a closed door/wall looks like.
    height[7:11, 3:9] = 0.15
    forward = np.full((12, 12), 0.90, dtype=np.float32)

    ground_connected = depth_perception._ground_connected_components(
        elevated, height, forward
    )

    assert np.array_equal(ground_connected, elevated)


def test_suspended_tabletop_is_not_delegated_to_lidar() -> None:
    elevated = np.zeros((12, 12), dtype=bool)
    elevated[2:7, 3:9] = True
    height = np.zeros((12, 12), dtype=np.float32)
    height[2:7, 3:9] = 0.65
    # Visible floor below the top is farther away, as it is beneath a table.
    height[7:11, 3:9] = 0.0
    forward = np.full((12, 12), 0.90, dtype=np.float32)
    forward[7:11, 3:9] = 1.35

    ground_connected = depth_perception._ground_connected_components(
        elevated, height, forward
    )

    assert not np.any(ground_connected)


def test_depth_requires_a_hough_edge_proposal(monkeypatch) -> None:
    """A broad depth region alone must never produce an eye hazard."""
    model = default_eye_right()
    fields = depth_perception._pixel_fields(model, (240, 320))
    depth = np.full((240, 320), 8.0, dtype=np.float32)
    below_horizon = fields["up_unit"] < -0.02
    depth[below_horizon] = (0.70 - model.height_m) / fields["up_unit"][below_horizon]
    monkeypatch.setattr(depth_perception, "_floor_scale", lambda *_: 1.0)

    result = depth_perception.analyze_depth(
        model, depth, eye="front_right", frame_monotonic=1.0
    )

    assert result.proposal_detected is False
    assert result.nearest_gate_m is None
    assert result.region_points_xy == ()


def test_clear_narrow_corridor_can_creep_past_close_side_edge() -> None:
    """A side edge may freeze turns without blocking a straight squeeze path."""
    allowed, reason = gate_forward_allowed(
        HazardState(enabled=True, active=True, near_freeze=True, squeeze=True)
    )

    assert (allowed, reason) == (True, "squeeze_creep")


def test_depth_gate_applies_six_inch_approach_allowance() -> None:
    """The camera-only gate is corrected; the source measurement is not."""
    monitor = ElevatedHazardMonitor(ElevatedSafetyConfig(), frame_source=object())
    raw_distance_m = 0.55
    result = SimpleNamespace(
        frame_monotonic=time.monotonic(),
        nearest_gate_m=raw_distance_m,
        nearest_any_m=raw_distance_m,
        bearing_deg=0.0,
        edge_height_m=None,
        floor_points_xy=(),
        region_points_xy=(),
        segment_p1=None,
        segment_p2=None,
    )

    _, gate_distance_m, nearest_distance_m, _, _ = monitor._candidates_from_depth(
        {"front_right": result}
    )["front_right"]

    expected = raw_distance_m + 0.1524
    assert gate_distance_m == expected
    assert nearest_distance_m == expected


def test_outboard_chassis_return_at_49deg_cannot_trigger_gate(monkeypatch) -> None:
    """Field 2026-07-13 (second run): every stop's nearest reading sat at
    bearing +-49deg with n=0.20..0.38m — the chassis, seen just inside the
    trimmed FOV, escaping the metric self zone whenever the frame's scale
    wobbled high (x1.08..x1.77 observed). The OUTER_NEAR band must reject
    near returns there regardless of the frame's scale anchor."""
    shape = (24, 24)
    fields = {
        "fwd_unit": np.ones(shape, dtype=np.float32),
        "up_unit": np.full(shape, -0.55, dtype=np.float32),
        "x_unit": np.zeros(shape, dtype=np.float32),
        "bearing_deg": np.full(shape, 49.0, dtype=np.float32),
    }
    monkeypatch.setattr(depth_perception, "_pixel_fields", lambda *_: fields)
    monkeypatch.setattr(depth_perception, "_floor_scale", lambda *_: 1.0)
    model = replace(default_eye_right(), yaw_deg=0.0, forward_offset_m=0.0)

    # 0.42m at 49 deg: outside SELF_BODY_FORWARD_M (a wobbled-scale chassis
    # return), inside the OUTER_NEAR band.
    result = depth_perception.analyze_depth(
        model,
        np.full(shape, 0.42, dtype=np.float32),
        eye="front_right",
        frame_monotonic=1.0,
        proposal_line_xy=((2, 12), (22, 12)),
    )

    assert result.nearest_gate_m is None
    assert result.nearest_any_m is None


def test_side_sighting_footprint_publishes_for_the_map() -> None:
    """A table beside the path (no corridor gate at all) must still publish
    its boundary footprint so the planner learns it (field 2026-07-13: the
    starting room's table was never mapped because the exit path never
    pointed at it inside gate range)."""
    monitor = ElevatedHazardMonitor(ElevatedSafetyConfig(), frame_source=object())
    result = SimpleNamespace(
        frame_monotonic=time.monotonic(),
        nearest_gate_m=None,  # nothing in the forward corridor
        nearest_any_m=1.10,  # table off to the side
        bearing_deg=48.0,
        edge_height_m=0.72,
        floor_points_xy=(),
        region_points_xy=((1.10, 0.80), (1.16, 0.86)),
        segment_p1=None,
        segment_p2=None,
    )

    monitor._candidates_from_depth({"front_left": result})
    edges = monitor.drain_confirmed_edges()

    assert len(edges) == 1
    assert edges[0].points_robot_xy


def _seam_test_fields(shape, blob_cols, up_blob=-0.3, up_bg=-0.9):
    h, w = shape
    up = np.full(shape, up_bg, dtype=np.float32)
    up[6:18, blob_cols[0] : blob_cols[1]] = up_blob
    return {
        "fwd_unit": np.ones(shape, dtype=np.float32),
        "up_unit": up,
        "x_unit": np.zeros(shape, dtype=np.float32),
        "bearing_deg": np.zeros(shape, dtype=np.float32),
    }


def test_seam_only_blob_is_forgiven(monkeypatch) -> None:
    """An elevated blob living entirely inside the mosaic's seam band is a
    stitch artifact dead ahead — it must not gate (the user's 'edges get a
    bit chopped up in the middle' forgiveness)."""
    shape = (24, 48)
    monkeypatch.setattr(
        depth_perception, "_pixel_fields", lambda *_: _seam_test_fields(shape, (21, 27))
    )
    monkeypatch.setattr(depth_perception, "_floor_scale", lambda *_: 1.0)
    model = replace(default_eye_right(), yaw_deg=0.0, forward_offset_m=0.0)

    result = depth_perception.analyze_depth(
        model,
        np.ones(shape, dtype=np.float32),
        eye="panorama",
        frame_monotonic=1.0,
        proposal_line_xy=((0, 12), (47, 12)),
        seam_cols=(20, 28),
    )

    assert result.nearest_gate_m is None
    assert result.nearest_any_m is None


def test_blob_crossing_seam_still_gates(monkeypatch) -> None:
    """A REAL edge spanning the seam keeps gating via its out-of-band pixels."""
    shape = (24, 48)
    monkeypatch.setattr(
        depth_perception, "_pixel_fields", lambda *_: _seam_test_fields(shape, (10, 27))
    )
    monkeypatch.setattr(depth_perception, "_floor_scale", lambda *_: 1.0)
    model = replace(default_eye_right(), yaw_deg=0.0, forward_offset_m=0.0)

    result = depth_perception.analyze_depth(
        model,
        np.ones(shape, dtype=np.float32),
        eye="panorama",
        frame_monotonic=1.0,
        proposal_line_xy=((0, 12), (47, 12)),
        seam_cols=(20, 28),
    )

    assert result.nearest_gate_m is not None


def test_depth_worker_panorama_keys() -> None:
    """With a mosaic attached the worker serves the single 'panorama' key."""
    mosaic = SimpleNamespace(model=default_eye_right(), seam_cols=(0, 1), coverage=lambda: None)
    worker = depth_perception.DepthWorker(object(), {"a": default_eye_right()}, mosaic=mosaic)
    assert worker.eye_keys == ("panorama",)
    worker_per_eye = depth_perception.DepthWorker(
        object(), {"front_left": default_eye_right(), "front_right": default_eye_right()}
    )
    assert worker_per_eye.eye_keys == ("front_left", "front_right")


def _fields_with_lateral(shape, up_map, x_map, fwd=0.7):
    return {
        "fwd_unit": np.full(shape, fwd, dtype=np.float32),
        "up_unit": up_map,
        "x_unit": x_map,
        "bearing_deg": np.zeros(shape, dtype=np.float32),
    }


def test_fused_mode_gates_without_a_hough_proposal(monkeypatch) -> None:
    """require_proposal=False: the metric 3D owns candidacy — a large real
    elevated region gates with NO detected line (the 'no hough' dark-cabinet
    blindness is gone in fused mode)."""
    shape = (24, 48)
    up = np.full(shape, -0.9, dtype=np.float32)
    up[4:20, 8:40] = -0.3  # broad elevated region, no proposal anywhere
    monkeypatch.setattr(
        depth_perception,
        "_pixel_fields",
        lambda *_: _fields_with_lateral(shape, up, np.zeros(shape, dtype=np.float32)),
    )
    monkeypatch.setattr(depth_perception, "_floor_scale", lambda *_: 1.0)
    model = replace(default_eye_right(), yaw_deg=0.0, forward_offset_m=0.0)

    result = depth_perception.analyze_depth(
        model, np.ones(shape, dtype=np.float32), eye="panorama",
        frame_monotonic=1.0, require_proposal=False,
    )
    assert result.nearest_gate_m is not None
    assert len(result.region_points_xy) > 0  # edge dimensions map without a line


def test_fused_mode_speckle_needs_size_or_line(monkeypatch) -> None:
    """Without the line gate, tiny depth-noise blobs must not gate."""
    shape = (24, 48)
    up = np.full(shape, -0.9, dtype=np.float32)
    up[10:12, 20:23] = -0.3  # 6px speckle
    monkeypatch.setattr(
        depth_perception,
        "_pixel_fields",
        lambda *_: _fields_with_lateral(shape, up, np.zeros(shape, dtype=np.float32)),
    )
    monkeypatch.setattr(depth_perception, "_floor_scale", lambda *_: 1.0)
    model = replace(default_eye_right(), yaw_deg=0.0, forward_offset_m=0.0)

    result = depth_perception.analyze_depth(
        model, np.ones(shape, dtype=np.float32), eye="panorama",
        frame_monotonic=1.0, require_proposal=False,
    )
    assert result.nearest_any_m is None


def test_measured_gap_bounds_and_width(monkeypatch) -> None:
    """Two posts at lateral ~±0.5m within the decision zone must yield gap
    bounds ~∓0.5 and a ~1.0m measured clear width — real dimensions, not a
    binary corridor guess."""
    shape = (24, 48)
    up = np.full(shape, -0.9, dtype=np.float32)
    x = np.zeros(shape, dtype=np.float32)
    up[4:20, 4:12] = -0.3   # left-image post
    x[:, 4:12] = 0.5        # robot_lateral = -x_unit*d = -0.5
    up[4:20, 36:44] = -0.3  # right-image post
    x[:, 36:44] = -0.5      # robot_lateral = +0.5
    monkeypatch.setattr(
        depth_perception, "_pixel_fields", lambda *_: _fields_with_lateral(shape, up, x)
    )
    monkeypatch.setattr(depth_perception, "_floor_scale", lambda *_: 1.0)
    model = replace(default_eye_right(), yaw_deg=0.0, forward_offset_m=0.0)

    result = depth_perception.analyze_depth(
        model, np.ones(shape, dtype=np.float32), eye="panorama",
        frame_monotonic=1.0, require_proposal=False,
    )
    assert result.clear_gap_left_m is not None and abs(result.clear_gap_left_m + 0.5) < 0.05
    assert result.clear_gap_right_m is not None and abs(result.clear_gap_right_m - 0.5) < 0.05
    assert result.clear_width_m is not None and abs(result.clear_width_m - 1.0) < 0.1


def test_fused_mode_maps_multiple_edges_per_frame(monkeypatch) -> None:
    """Per-column boundary: TWO objects at different depths both contribute
    their leading boundary in one frame (legacy mapped only the nearest
    component)."""
    shape = (24, 48)
    up = np.full(shape, -0.9, dtype=np.float32)
    x = np.zeros(shape, dtype=np.float32)
    fwd = np.full(shape, 0.7, dtype=np.float32)
    up[4:20, 4:12] = -0.3
    x[:, 4:12] = 0.5
    up[4:20, 36:44] = -0.3
    x[:, 36:44] = -0.5
    fwd[:, 36:44] = 1.4  # second object twice as far
    fields = _fields_with_lateral(shape, up, x)
    fields["fwd_unit"] = fwd
    monkeypatch.setattr(depth_perception, "_pixel_fields", lambda *_: fields)
    monkeypatch.setattr(depth_perception, "_floor_scale", lambda *_: 1.0)
    model = replace(default_eye_right(), yaw_deg=0.0, forward_offset_m=0.0)

    result = depth_perception.analyze_depth(
        model, np.ones(shape, dtype=np.float32), eye="panorama",
        frame_monotonic=1.0, require_proposal=False,
    )
    lats = [pl for _, pl in result.region_points_xy]
    assert lats and min(lats) < -0.3 and max(lats) > 0.3  # both objects mapped


def test_gap_clears_body_helper() -> None:
    from sourccey_elevated_safety import gap_clears_body

    assert gap_clears_body(None, None, 0.31)  # nothing measured = open
    assert gap_clears_body(-0.5, 0.5, 0.31)  # 1.0m gap centered: fits
    assert not gap_clears_body(-0.5, 0.2, 0.31)  # right wall too close
    assert not gap_clears_body(-0.1, 0.5, 0.31)  # left wall too close
    assert gap_clears_body(None, 0.35, 0.31)  # open left, right clears
