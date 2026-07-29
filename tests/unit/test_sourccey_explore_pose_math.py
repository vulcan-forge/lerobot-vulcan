"""Regression tests for the explorer's LiDAR/IMU pose-frame math."""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ldlidar_direct_snapshot_stitch as stitch  # noqa: E402
import sourccey_explore as explore  # noqa: E402
from ldlidar_direct_snapshot_client import DirectLidarFeed, ScanFrame  # noqa: E402
from ldlidar_direct_snapshot_stitch import Pose2D, _search_pose, _transform_points  # noqa: E402
from sourccey_collision_box import (  # noqa: E402
    calibrate_collision_box,
    collision_box_dimensions,
    collision_box_rotation_violation,
    collision_box_violation,
    effective_ranges,
    physical_body_self_return_mask,
)
from sourccey_wander.imu_heading import ImuYawClient  # noqa: E402

from lerobot.robots.sourccey.sourccey.protobuf.sourccey_protobuf import (  # noqa: E402
    SourcceyProtobuf,
)
from lerobot.robots.sourccey.sourccey.sourccey.sourccey import Sourccey  # noqa: E402


def test_zero_exploration_budgets_are_unlimited() -> None:
    assert not explore._budget_exhausted(10_000, 0)
    assert not explore._budget_exhausted(10_000, -1)
    assert not explore._budget_exhausted(19, 20)
    assert explore._budget_exhausted(20, 20)


def test_transient_planner_obstacle_does_not_mutate_slam_grid() -> None:
    grid = explore.OccupancyGrid(0.10)
    before = grid.L.copy()
    before_version = grid.version
    point = np.array([[0.05, 0.05]], dtype=np.float32)

    analysis = explore.analyze_grid(
        grid,
        robot_radius_m=0.30,
        min_frontier_span_m=0.40,
        min_frontier_cells=6,
        extra_occupied_xy=point,
    )

    cell = analysis.to_cell(point[0])
    assert analysis.occupied[cell]
    assert not grid.occupied()[cell]
    assert grid.version == before_version
    assert np.array_equal(grid.L, before)


def test_indefinite_patrol_selects_reachable_underobserved_free_space() -> None:
    traversable = np.ones((9, 9), dtype=bool)
    analysis = explore.Analysis(
        origin_xy=np.array([0.0, 0.0]),
        res_m=0.20,
        occupied=np.zeros_like(traversable),
        free=traversable.copy(),
        traversable=traversable,
        cost=np.ones(traversable.shape, dtype=np.float64),
    )
    robot = analysis.to_world((4, 4))
    patrol = explore._pick_patrol_route(analysis, robot, [robot.copy()])
    assert patrol is not None
    goal, route = patrol
    assert float(np.hypot(*(goal - robot))) >= 0.75
    assert route


def test_indefinite_patrol_excludes_recently_failed_goal() -> None:
    traversable = np.ones((13, 13), dtype=bool)
    analysis = explore.Analysis(
        origin_xy=np.array([0.0, 0.0]),
        res_m=0.20,
        occupied=np.zeros_like(traversable),
        free=traversable.copy(),
        traversable=traversable,
        cost=np.ones(traversable.shape, dtype=np.float64),
    )
    robot = analysis.to_world((6, 6))
    first = explore._pick_patrol_route(analysis, robot, [robot.copy()])
    assert first is not None

    second = explore._pick_patrol_route(
        analysis,
        robot,
        [robot.copy()],
        blocked_xy=[first[0]],
    )

    assert second is not None
    assert float(np.hypot(*(second[0] - first[0]))) >= 0.45


def test_indefinite_patrol_requires_a_real_translation() -> None:
    traversable = np.ones((3, 3), dtype=bool)
    analysis = explore.Analysis(
        origin_xy=np.array([0.0, 0.0]),
        res_m=0.10,
        occupied=np.zeros_like(traversable),
        free=traversable.copy(),
        traversable=traversable,
        cost=np.ones(traversable.shape, dtype=np.float64),
    )
    robot = analysis.to_world((1, 1))

    assert explore._pick_patrol_route(analysis, robot, []) is None


def test_navigation_progress_checker_rejects_stationary_discarded_observation() -> None:
    start = np.array([1.0, -2.0])

    assert not explore._navigation_action_made_progress(start, start, 0, 0)
    assert explore._navigation_action_made_progress(
        start,
        start + np.array([0.13, 0.0]),
        0,
        0,
    )
    assert explore._navigation_action_made_progress(start, start, 3, 20)
    assert not explore._navigation_action_made_progress(start, start, 3, 0)


def test_lidar_wait_never_returns_a_cached_frame_as_fresh() -> None:
    feed = DirectLidarFeed("127.0.0.1", 8765)
    cached = ScanFrame(
        ts=1.0,
        rpm=300.0,
        points=[],
        raw_line="{}",
        received_wall_ts=time.time(),
    )
    with feed._lock:
        feed._latest_frame_id = 7
        feed._latest_frame = cached

    frame_id, frame = feed.wait_for_frame_after(
        after_frame_id=7,
        timeout_s=0.02,
        min_frame_advances=1,
    )
    assert frame_id == 7
    assert frame is None

    frame_id, frame = feed.wait_for_frame_after(
        after_frame_id=6,
        timeout_s=0.02,
        min_frame_advances=1,
    )
    assert frame_id == 7
    assert frame is cached


def test_anchor_motion_rejects_stationary_duplicate_capture() -> None:
    assert not explore._anchor_motion_is_complete(-3.9, 0.0, 360.0)
    assert not explore._anchor_motion_is_complete(350.0, 0.20, 360.0)
    assert explore._anchor_motion_is_complete(360.0, 0.99, 360.0)


def test_anchor_safety_interruptions_do_not_consume_motor_budget() -> None:
    assert not explore._anchor_failure_consumes_motion_budget("safety_blocked")
    assert explore._anchor_failure_consumes_motion_budget("motion_stall")
    assert explore._anchor_failure_consumes_motion_budget("timeout")


def test_bidirectional_consensus_outranks_inconclusive_stationary_refinement() -> None:
    assert explore._bidirectional_anchor_is_sufficient(True, 49, 1.0, 19.3, 0.75, 25.0)
    # A blind angular wedge becomes a frontier. It does not invalidate geometry
    # independently reproduced by the reverse trajectory.
    assert explore._bidirectional_anchor_is_sufficient(True, 49, 1.0, 30.2, 0.75, 25.0)
    assert not explore._bidirectional_anchor_is_sufficient(False, 49, 1.0, 19.3, 0.75, 25.0)
    assert not explore._bidirectional_anchor_is_sufficient(True, 8, 1.0, 30.2, 0.75, 25.0)
    assert not explore._bidirectional_anchor_is_sufficient(True, 49, 0.50, 30.2, 0.75, 25.0)


def test_bidirectional_consensus_accepts_field_resolution_residuals() -> None:
    # The logged 40/40 agreement at 4cm/nominally 2deg is strong evidence, not
    # LiDAR dropout. Numeric and grid quantization must not flip this boundary.
    assert explore._bidirectional_consensus_is_acceptable(
        1.0, 0.040, 2.00001, 0.75, 0.10, 3.0, linear_quantization_slack_m=0.025
    )
    assert not explore._bidirectional_consensus_is_acceptable(
        0.50, 0.040, 2.0, 0.75, 0.10, 3.0, linear_quantization_slack_m=0.025
    )
    assert not explore._bidirectional_consensus_is_acceptable(
        1.0, 0.20, 2.0, 0.75, 0.10, 3.0, linear_quantization_slack_m=0.025
    )


def test_keyframe_transaction_requires_map_support_and_bounded_pose_innovation() -> None:
    assert explore._keyframe_batch_is_committable(6, 15.0, 12.0, 0.35, 0.08, 0.04, 0.25, 1.0, 8.0)
    assert not explore._keyframe_batch_is_committable(6, 8.2, 12.0, 0.35, 0.08, 0.04, 0.25, 1.0, 8.0)
    assert not explore._keyframe_batch_is_committable(6, 15.0, 12.0, 0.03, 0.08, 0.04, 0.25, 1.0, 8.0)
    assert not explore._keyframe_batch_is_committable(6, 15.0, 12.0, 0.35, 0.08, 0.40, 0.25, 1.0, 8.0)
    assert not explore._keyframe_batch_is_committable(6, 15.0, 12.0, 0.35, 0.08, 0.04, 0.25, 12.0, 8.0)


def test_large_lidar_yaw_correction_requires_high_confidence_submap_overlap() -> None:
    assert explore._keyframe_heading_correction_limit_deg(
        5.0, 8, 8, 15.4, 13.0, 1.0
    ) == 6.0
    assert explore._keyframe_heading_correction_limit_deg(
        5.0, 8, 8, 11.7, 13.0, 0.60
    ) == 5.0
    assert explore._keyframe_heading_correction_limit_deg(
        5.0, 8, 8, 15.4, 13.0, 0.30
    ) == 5.0
    assert explore._keyframe_heading_correction_limit_deg(
        5.0, 5, 8, 15.4, 13.0, 1.0
    ) == 5.0


def test_supported_stationary_batch_never_commits_below_score_twelve() -> None:
    assert explore._keyframe_match_commit_threshold(13.0, 8, 8, 0.95) == 12.0
    assert explore._keyframe_match_commit_threshold(13.0, 5, 8, 0.95) == 13.0
    assert explore._keyframe_match_commit_threshold(13.0, 8, 8, 0.40) == 13.0


def test_stationary_map_commit_keeps_only_strongest_representative_keyframes() -> None:
    assert explore._representative_keyframe_indices(
        [11.0, 15.0, 13.0, 14.0, 12.0],
        [0, 1, 2, 3, 4],
        max_keyframes=2,
    ) == [1, 3]
    assert explore._representative_keyframe_indices(
        [11.0, 15.0],
        [-1, 7, 0],
        max_keyframes=2,
    ) == [0]


def test_pose_only_recovery_is_wider_than_irreversible_map_commit() -> None:
    # Field case: eight stationary scans coherently requested a 38cm/15deg
    # correction at score 9.3. Recover localization, but never paint that batch.
    assert explore._stationary_pose_is_recoverable(8, 9.3, 6.0, 0.45, 0.08, 0.38, 0.60, 15.0, 20.0)
    assert not explore._keyframe_batch_is_committable(8, 9.3, 12.0, 0.45, 0.08, 0.38, 0.25, 15.0, 8.0)
    assert not explore._stationary_pose_is_recoverable(8, 9.3, 6.0, 0.45, 0.08, 0.75, 0.60, 15.0, 20.0)


def test_anchor_candidate_requires_accepted_scans_and_complete_heading_coverage() -> None:
    assert explore._anchor_pass_is_eligible(66, 1.0, 21.3, 0.75, 25.0)
    assert not explore._anchor_pass_is_eligible(35, 0.919, 82.5, 0.75, 25.0)
    assert not explore._anchor_pass_is_eligible(66, 0.70, 10.0, 0.75, 25.0)
    assert not explore._anchor_pass_is_eligible(0, 1.0, 0.0, 0.75, 25.0)


def test_collision_interrupted_anchor_falls_back_instead_of_aborting_mission() -> None:
    assert explore._anchor_interruption_uses_stationary_fallback("safety_blocked")
    assert not explore._anchor_interruption_uses_stationary_fallback("motion_stall")
    assert not explore._anchor_failure_consumes_motion_budget("safety_blocked")


def test_stationary_anchor_consensus_controls_navigation_release() -> None:
    assert explore._stationary_anchor_is_accepted(3, 3, 0.04, 0.06, 2.0, 3.0)
    assert not explore._stationary_anchor_is_accepted(3, 4, 0.04, 0.06, 2.0, 3.0)
    assert not explore._stationary_anchor_is_accepted(2, 3, 0.04, 0.06, 2.0, 3.0)
    assert not explore._stationary_anchor_is_accepted(5, 3, 0.07, 0.06, 2.0, 3.0)
    assert not explore._stationary_anchor_is_accepted(5, 3, 0.04, 0.06, 4.0, 3.0)
    assert not explore._stationary_anchor_is_accepted(
        5, 3, float("inf"), 0.06, 2.0, 3.0
    )


def test_anchor_endpoint_support_rejects_a_locally_matching_wall_subset() -> None:
    support = np.zeros((10, 10), dtype=bool)
    support[2, 2] = True
    support[2, 3] = True
    points = np.array([
        [2.1, 2.1],
        [3.1, 2.1],
        [5.1, 5.1],
        [7.1, 7.1],
    ])

    assert explore._grid_endpoint_support_ratio(
        points,
        support,
        np.zeros(2),
        1.0,
    ) == pytest.approx(0.5)


def test_exceptional_reciprocal_overlap_can_validate_one_clean_full_scan() -> None:
    assert explore._high_confidence_anchor_validation_index(
        [15.5, 9.0, 14.0],
        [0.996, 0.99, 0.70],
        12.0,
        0.85,
    ) == 0
    assert explore._high_confidence_anchor_validation_index(
        [15.5, 14.0],
        [0.80, 0.70],
        12.0,
        0.85,
    ) is None


def test_anchor_stiction_escalation_is_bounded() -> None:
    assert explore._next_anchor_spin_speed(0.80, 1.0) == pytest.approx(0.90)
    assert explore._next_anchor_spin_speed(0.95, 1.0) == pytest.approx(1.0)
    assert explore._next_anchor_spin_speed(1.0, 1.0) == pytest.approx(1.0)


def test_base_stream_holds_stowed_arm_torque_without_joint_bus_writes() -> None:
    stowed = explore._latched_arm_torque_state(stowed=True)
    assert stowed == {}
    assert not any(key.endswith(".pos") for key in stowed)
    assert explore._latched_arm_torque_state(stowed=False) == {
        "untorque_left": True,
        "untorque_right": True,
    }


def test_base_only_protobuf_packet_has_no_phantom_arm_targets() -> None:
    converter = SourcceyProtobuf()
    message = converter.action_to_protobuf({
        "x.vel": 0.0,
        "y.vel": 0.0,
        "theta.vel": 0.9,
    })

    assert not message.HasField("left_arm_target_joints")
    assert not message.HasField("right_arm_target_joints")
    decoded = converter.protobuf_to_action(message)
    assert not any(key.startswith(("left_", "right_")) for key in decoded)
    assert decoded["theta.vel"] == pytest.approx(0.9)


def test_arm_bus_failure_does_not_suppress_base_wheel_command() -> None:
    class FailingArm:
        def send_action(self, _action):
            raise RuntimeError("simulated follower bus fault")

    class RecordingBase:
        def __init__(self) -> None:
            self.commands = []

        def set_velocities(self, command) -> None:
            self.commands.append(command)

    robot = Sourccey.__new__(Sourccey)
    robot.apply_untorque_flags = lambda action: action
    robot._arms_connected = True
    robot.left_arm = FailingArm()
    robot.right_arm = FailingArm()
    robot.dc_motors_controller = RecordingBase()
    robot.z_actuator = SimpleNamespace(use_z_actuator=False)

    sent = robot.send_action({
        "left_shoulder_pan.pos": 25.0,
        "x.vel": 0.0,
        "y.vel": 0.0,
        "theta.vel": 0.9,
    })

    assert sent["theta.vel"] == pytest.approx(0.9)
    assert robot.dc_motors_controller.commands == [{
        "front_left": 0.9,
        "front_right": 0.9,
        "rear_left": 0.9,
        "rear_right": 0.9,
    }]


def test_arm_torque_bus_failure_cannot_suppress_base_wheel_command() -> None:
    class RecordingBase:
        def __init__(self) -> None:
            self.commands = []

        def set_velocities(self, command) -> None:
            self.commands.append(command)

    robot = Sourccey.__new__(Sourccey)
    robot.apply_untorque_flags = lambda _action: (_ for _ in ()).throw(
        RuntimeError("simulated blocked torque transaction")
    )
    robot.dc_motors_controller = RecordingBase()

    sent = robot.send_action({"x.vel": -0.5, "y.vel": 0.0, "theta.vel": 0.0})

    assert sent == {}
    assert robot.dc_motors_controller.commands == [{
        "front_left": -0.5,
        "front_right": 0.5,
        "rear_left": -0.5,
        "rear_right": 0.5,
    }]


def test_startup_translation_rollout_only_accepts_clearance_increasing_escape() -> None:
    profile = {
        "version": 1,
        "frame": "physical_forward_xy",
        "bin_size_deg": 4.0,
        "ranges_m": [0.50] * 90,
        "noise_tolerance_m": 0.0,
        "safety_margin_m": 0.0,
        "min_violation_bins": 2,
        "min_violation_points": 2,
        "side_min_violation_bins": 1,
        "side_min_violation_points": 2,
    }
    front = np.array([[0.38, -0.01], [0.38, 0.01], [0.39, 0.03]])
    geometry = {
        "lidar_offset_forward_m": 0.0,
        "physical_body_radius_m": 0.0,
        "self_mask_inset_m": 0.0,
    }

    safe, improvement = explore._translation_escape_is_safe(
        front, profile, np.array([-0.12, 0.0]), **geometry
    )
    assert safe
    assert improvement > 0.05
    worsening, _ = explore._translation_escape_is_safe(
        front, profile, np.array([0.12, 0.0]), **geometry
    )
    assert not worsening
    parallel, parallel_gain = explore._translation_escape_is_safe(
        front, profile, np.array([0.0, 0.12]), **geometry
    )
    assert parallel
    assert parallel_gain >= 0.0


def test_corner_initialization_scores_full_pivot_clearance_and_safe_escape() -> None:
    profile = {
        "version": 1,
        "frame": "physical_forward_xy",
        "bin_size_deg": 4.0,
        "ranges_m": [0.40] * 90,
        "noise_tolerance_m": 0.0,
        "safety_margin_m": 0.0,
        "min_violation_bins": 1,
        "min_violation_points": 2,
        "side_min_violation_bins": 1,
        "side_min_violation_points": 2,
    }
    geometry = {
        "lidar_offset_forward_m": 0.0,
        "physical_body_radius_m": 0.0,
        "self_mask_inset_m": 0.0,
    }
    corner_wall = np.array([[0.30, -0.02], [0.30, 0.00], [0.30, 0.02]])
    open_wall = np.array([[0.80, -0.02], [0.80, 0.00], [0.80, 0.02]])

    assert explore._full_pivot_clearance_margin_m(
        corner_wall, profile, **geometry
    ) < 0.0
    assert explore._full_pivot_clearance_margin_m(
        open_wall, profile, **geometry
    ) > 0.0
    assert explore._translation_trajectory_is_safe(
        corner_wall,
        profile,
        np.array([-0.20, 0.0]),
        **geometry,
    )
    assert not explore._translation_trajectory_is_safe(
        corner_wall,
        profile,
        np.array([0.20, 0.0]),
        **geometry,
    )


def test_startup_escape_diagonal_clears_each_axis_stiction_floor() -> None:
    diagonal = explore._startup_escape_velocity(np.array([0.08, -0.08]), 0.80)
    lateral = explore._startup_escape_velocity(np.array([0.0, 0.12]), 0.80)
    shallow_diagonal = explore._startup_escape_velocity(
        np.array([math.cos(math.radians(15.0)), math.sin(math.radians(15.0))]),
        0.80,
    )

    assert diagonal == pytest.approx(np.array([0.80, -0.80]))
    assert lateral == pytest.approx(np.array([0.0, 0.80]))
    assert math.degrees(math.atan2(*shallow_diagonal[::-1])) == pytest.approx(15.0)
    assert explore._startup_escape_velocity(
        np.array([-0.12, 0.0]), 1.5
    ) == pytest.approx(np.array([-1.0, 0.0]))


def test_open_area_planner_rotates_scan_into_proposed_body_heading() -> None:
    points = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    turned_left = explore._points_in_rotated_body_frame(points, 90.0)
    turned_right = explore._points_in_rotated_body_frame(points, -90.0)

    assert turned_left == pytest.approx(
        np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0]]), abs=1e-12
    )
    assert turned_right == pytest.approx(
        np.array([[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]), abs=1e-12
    )
    offset_turn = explore._points_in_rotated_body_frame(
        np.array([[1.0, 0.0]]),
        90.0,
        lidar_offset_forward_m=0.25,
    )
    assert offset_turn == pytest.approx(np.array([[-0.25, -1.25]]), abs=1e-12)


def test_live_lidar_gap_selector_prefers_wide_forward_corridor() -> None:
    longitudinal = np.linspace(0.4, 2.0, 20)
    corridor_walls = np.vstack((
        np.column_stack((longitudinal, np.full_like(longitudinal, 0.50))),
        np.column_stack((longitudinal, np.full_like(longitudinal, -0.50))),
    ))

    selected = explore._widest_visible_corridor_bearing_deg(
        corridor_walls,
        lidar_offset_forward_m=0.0,
        corridor_half_width_m=0.30,
    )

    assert selected is not None
    bearing, visible_range = selected
    assert bearing == pytest.approx(0.0)
    assert visible_range == pytest.approx(2.0)


def test_startup_lidar_speck_filter_removes_only_isolated_return() -> None:
    wall = np.column_stack((
        np.full(8, 1.0),
        np.linspace(-0.20, 0.20, 8),
    ))
    isolated = np.array([[0.35, 0.42]])

    filtered = explore._filter_isolated_lidar_specks(
        np.vstack((wall, isolated))
    )

    assert filtered == pytest.approx(wall)


def test_overlap_turn_steps_are_signed_bounded_and_do_not_overshoot() -> None:
    assert explore._bounded_heading_step(75.0, 30.0) == 30.0
    assert explore._bounded_heading_step(-75.0, 30.0) == -30.0
    assert explore._bounded_heading_step(12.0, 30.0) == 12.0


def test_heading_behind_robot_can_be_reached_by_more_than_three_control_steps() -> None:
    remaining = 175.9
    steps: list[float] = []
    while abs(remaining) >= 8.0:
        step = explore._bounded_heading_step(remaining, 25.0)
        steps.append(step)
        remaining -= step

    assert len(steps) == 7
    assert steps == pytest.approx([25.0] * 7)
    assert remaining == pytest.approx(0.9)


def test_startup_escape_route_is_exactly_one_bounded_forward_segment() -> None:
    route = [
        np.array([0.60, 0.80]),
        np.array([1.20, 1.10]),
        np.array([1.80, 1.30]),
    ]

    escape = explore._single_startup_escape_waypoint(
        route,
        maximum_distance_m=0.45,
    )

    assert len(escape) == 1
    assert float(np.hypot(*escape[0])) == pytest.approx(0.45)
    assert escape[0] == pytest.approx(np.array([0.27, 0.36]))


def test_short_startup_escape_is_not_extended_past_first_route_segment() -> None:
    escape = explore._single_startup_escape_waypoint(
        [np.array([0.12, -0.05]), np.array([0.50, -0.20])],
        maximum_distance_m=0.45,
    )

    assert len(escape) == 1
    assert escape[0] == pytest.approx(np.array([0.12, -0.05]))


def test_zero_motion_localization_probe_switches_to_parallax() -> None:
    assert explore._active_localization_rotation_stalled(0.0)
    assert explore._active_localization_rotation_stalled(1.9)
    assert not explore._active_localization_rotation_stalled(2.0)


def test_bounded_turn_status_stops_wrong_direction_and_overshoot() -> None:
    assert explore._bounded_turn_status(15.0, 5.0) == "turning"
    assert explore._bounded_turn_status(15.0, 12.0) == "complete"
    assert explore._bounded_turn_status(15.0, -3.0) == "wrong_direction"
    assert explore._bounded_turn_status(15.0, 20.0) == "overshoot"
    assert explore._bounded_turn_status(-15.0, -12.0) == "complete"
    assert explore._bounded_turn_status(-15.0, 3.0) == "wrong_direction"


def test_navigation_pivot_progress_timeout_is_bounded() -> None:
    assert not explore._pivot_progress_stalled(12.49, 10.0, timeout_s=2.5)
    assert explore._pivot_progress_stalled(12.50, 10.0, timeout_s=2.5)


def test_doorway_keyframe_requires_translation_or_heading_diversity() -> None:
    prior = np.array([1.0, 2.0])

    assert not explore._keyframe_view_is_diverse(
        np.array([1.05, 2.0]), 8.0, prior, 0.0
    )
    assert explore._keyframe_view_is_diverse(
        np.array([1.13, 2.0]), 8.0, prior, 0.0
    )
    assert explore._keyframe_view_is_diverse(
        np.array([1.05, 2.0]), 13.0, prior, 0.0
    )
    assert explore._keyframe_view_is_diverse(
        prior, 0.0, prior, None
    )


def test_anchor_heading_coverage_handles_wraparound() -> None:
    headings = [350.0, 0.0, 10.0, 90.0, 180.0, 270.0]

    assert explore._max_heading_gap_deg(headings) == 90.0
    assert explore._max_heading_gap_deg([0.0]) == 360.0


def test_stationary_lidar_revolutions_do_not_fake_occluded_body_coverage() -> None:
    headings = [0.0, 0.1, -0.1, 0.0, 0.1]

    assert explore._anchor_heading_gap_deg(headings) > 359.0
    assert explore._anchor_heading_gap_deg(
        headings, stationary_full_revolutions=True
    ) == pytest.approx(179.8)
    fan_headings = [0.0, 45.0, 90.0, 135.0, 180.0]
    fan_gap = explore._anchor_heading_gap_deg(
        fan_headings, stationary_full_revolutions=True
    )
    assert fan_gap == 0.0
    assert not explore._anchor_pass_is_eligible(
        len(headings), 1.0, 179.8, 0.75, 25.0
    )
    assert explore._anchor_pass_is_eligible(
        len(fan_headings), 1.0, fan_gap, 0.75, 25.0
    )


def test_occluded_anchor_validation_requires_scan_to_map_not_whole_map_reproduction() -> None:
    assert explore._anchor_validation_support_ratio(0.91, 0.42) == 0.91
    assert explore._anchor_validation_support_ratio(
        0.91,
        0.42,
        body_occluded_sensor=False,
    ) == 0.42


def test_anchor_consensus_uses_robust_p90_but_keeps_worst_case_diagnostic() -> None:
    p90, worst = explore._p90_and_max([0.01] * 19 + [0.20])

    assert p90 == pytest.approx(0.01)
    assert worst == pytest.approx(0.20)


def test_reflected_scan_frame_places_forward_lidar_on_negative_local_x() -> None:
    centre = np.array([1.0, 2.0])
    pose = explore._lidar_pose_from_robot_centre(
        centre, theta_deg=0.0, lever_m=0.229, forward_offset_deg=180.0
    )

    assert pose.x == pytest.approx(1.0 - 0.229)
    assert pose.y == pytest.approx(2.0)
    assert explore._robot_centre_from_lidar_pose(pose, 0.229, 180.0) == pytest.approx(centre)


def test_gyro_turn_moves_lidar_around_fixed_robot_centre() -> None:
    centre = np.array([0.4, -0.7])
    start = explore._lidar_pose_from_robot_centre(centre, 10.0, 0.229, 180.0)

    turned = explore._rotate_lidar_pose_about_robot_centre(start, 90.0, 0.229, 180.0)

    assert turned.theta_deg == pytest.approx(100.0)
    assert explore._robot_centre_from_lidar_pose(turned, 0.229, 180.0) == pytest.approx(centre)
    assert np.hypot(turned.x - centre[0], turned.y - centre[1]) == pytest.approx(0.229)


def test_body_translation_moves_robot_centre_without_changing_heading() -> None:
    centre = np.array([0.4, -0.7])
    start = explore._lidar_pose_from_robot_centre(centre, 10.0, 0.229, 180.0)

    moved = explore._translate_lidar_pose_in_body_frame(
        start,
        forward_m=0.12,
        left_m=0.05,
        lever_m=0.229,
        forward_offset_deg=180.0,
    )

    physical_heading = np.radians(190.0)
    expected = centre + np.array([
        0.12 * np.cos(physical_heading) - 0.05 * np.sin(physical_heading),
        0.12 * np.sin(physical_heading) + 0.05 * np.cos(physical_heading),
    ])
    assert moved.theta_deg == pytest.approx(start.theta_deg)
    assert explore._robot_centre_from_lidar_pose(
        moved, 0.229, 180.0
    ) == pytest.approx(expected)


def test_lidar_inertial_pose_uses_match_translation_but_never_wall_alias_heading() -> None:
    solved = explore._lidar_pose_from_robot_centre(
        np.array([0.42, -0.31]),
        27.0,
        0.229,
        180.0,
    )

    fused = explore._pose_with_imu_heading(
        solved,
        imu_theta_deg=6.0,
        lever_m=0.229,
        forward_offset_deg=180.0,
    )

    assert fused.theta_deg == pytest.approx(6.0)
    assert explore._robot_centre_from_lidar_pose(
        fused, 0.229, 180.0
    ) == pytest.approx([0.42, -0.31])


def test_stationary_batch_fuses_heading_wobble_at_one_robot_centre() -> None:
    centre = np.array([0.6, -0.4])
    poses = [
        explore._lidar_pose_from_robot_centre(centre, theta, 0.229, 180.0)
        for theta in (358.0, 359.0, 361.0, 362.0)
    ]

    consensus, position_scatter, heading_scatter = explore._stationary_pose_consensus(
        poses,
        reference_theta_deg=359.0,
        lever_m=0.229,
        forward_offset_deg=180.0,
    )

    assert explore._robot_centre_from_lidar_pose(
        consensus, 0.229, 180.0
    ) == pytest.approx(centre)
    assert consensus.theta_deg == pytest.approx(360.0)
    assert position_scatter == pytest.approx(0.0)
    assert heading_scatter == pytest.approx(2.0)


def _stationary_pose_at(centre_xy, theta_deg: float) -> Pose2D:
    return explore._lidar_pose_from_robot_centre(
        np.asarray(centre_xy, dtype=np.float64),
        theta_deg,
        0.229,
        180.0,
    )


def test_stationary_robust_consensus_integrates_only_compact_pose_mode() -> None:
    poses = [
        _stationary_pose_at((x, y), theta)
        for x, y, theta in (
            (-0.01, 0.00, -1.0),
            (0.00, 0.01, 0.0),
            (0.01, 0.00, 1.0),
            (0.00, -0.01, 0.5),
            (0.015, 0.005, -0.5),
            (-0.005, -0.015, 0.0),
            (0.24, 0.00, 18.0),
            (-0.25, 0.00, -20.0),
        )
    ]

    consensus, inliers, position_scatter, heading_scatter = (
        explore._stationary_pose_inlier_consensus(
            poses,
            reference_theta_deg=0.0,
            lever_m=0.229,
            forward_offset_deg=180.0,
            max_position_residual_m=0.08,
            max_heading_residual_deg=8.0,
            min_inliers=4,
            reference_centre_xy=np.zeros(2),
        )
    )

    assert inliers == [0, 1, 2, 3, 4, 5]
    assert explore._robot_centre_from_lidar_pose(
        consensus, 0.229, 180.0
    ) == pytest.approx([0.0, 0.0], abs=0.01)
    assert position_scatter < 0.03
    assert heading_scatter < 2.0


def test_stationary_robust_consensus_rejects_equal_disjoint_aliases_without_prior() -> None:
    poses = [
        *[_stationary_pose_at((-0.12 + dy, dy), 0.0) for dy in (-0.01, 0.0, 0.01, 0.02)],
        *[_stationary_pose_at((0.12 + dy, dy), 0.0) for dy in (-0.01, 0.0, 0.01, 0.02)],
    ]

    _consensus, inliers, _position_scatter, _heading_scatter = (
        explore._stationary_pose_inlier_consensus(
            poses,
            reference_theta_deg=0.0,
            lever_m=0.229,
            forward_offset_deg=180.0,
            max_position_residual_m=0.08,
            max_heading_residual_deg=8.0,
            min_inliers=4,
        )
    )

    assert inliers == []


def test_stationary_motion_prior_selects_nearby_mode_from_repeated_wall_alias() -> None:
    poses = [
        *[_stationary_pose_at((-0.12 + dy, dy), 0.0) for dy in (-0.01, 0.0, 0.01, 0.02)],
        *[_stationary_pose_at((0.12 + dy, dy), 0.0) for dy in (-0.01, 0.0, 0.01, 0.02)],
    ]

    consensus, inliers, _position_scatter, _heading_scatter = (
        explore._stationary_pose_inlier_consensus(
            poses,
            reference_theta_deg=0.0,
            lever_m=0.229,
            forward_offset_deg=180.0,
            max_position_residual_m=0.08,
            max_heading_residual_deg=8.0,
            min_inliers=4,
            reference_centre_xy=np.array([0.12, 0.0]),
        )
    )

    assert inliers == [4, 5, 6, 7]
    centre = explore._robot_centre_from_lidar_pose(consensus, 0.229, 180.0)
    assert centre[0] > 0.10


def test_timestamped_imu_yaw_interpolates_unwrapped_heading() -> None:
    imu = ImuYawClient("tcp://unused")
    imu._history = deque([(100.0, 350.0), (100.1, 370.0)], maxlen=2048)

    assert imu.deg_at_wall_time(100.05) == pytest.approx(360.0)
    assert imu.deg_at_wall_time(99.0) is None
    assert imu.deg_at_wall_time(101.0) is None


def test_imu_status_distinguishes_no_sample_from_running_receiver() -> None:
    imu = ImuYawClient("tcp://unused")

    assert imu.sample_age_s() is None
    assert not imu.receiver_running()


def test_anchor_scan_deskew_places_points_in_revolution_end_frame() -> None:
    frame = SimpleNamespace(
        points=[
            (90.0, 1.0, 255),
            (90.0, 1.0, 255),
            (90.0, 1.0, 255),
        ]
    )
    args = SimpleNamespace(
        forward_angle_deg=90.0,
        min_confidence=1,
        min_range_m=0.01,
        max_distance_m=10.0,
        valid_angle_half_width_deg=180.0,
        invert_lateral_axis=False,
    )

    deskewed = explore._scan_local(frame, args, heading_change_deg=10.0)

    assert deskewed[0] == pytest.approx(
        [np.cos(np.radians(10.0)), -np.sin(np.radians(10.0))]
    )
    assert deskewed[-1] == pytest.approx([1.0, 0.0], abs=1e-6)


def _open_analysis() -> explore.Analysis:
    shape = (21, 21)
    return explore.Analysis(
        origin_xy=np.array([-0.525, -0.525]),
        res_m=0.05,
        occupied=np.zeros(shape, dtype=bool),
        free=np.ones(shape, dtype=bool),
        traversable=np.ones(shape, dtype=bool),
    )


def _frontier(
    x: float, y: float, span: float = 0.6, *, passable: bool = False
) -> explore.FrontierCluster:
    return explore.FrontierCluster(
        cells_ij=np.array([[0, 0]], dtype=np.int64),
        centroid_xy=np.array([x, y], dtype=np.float64),
        span_m=span,
        size=8,
        passable=passable,
    )


def test_active_frontier_tracks_the_nearest_shifted_cluster() -> None:
    intended = _frontier(-1.1, -0.7)
    tempting_old_area = _frontier(-1.0, 1.1, span=1.4)

    matched = explore._match_active_frontier(
        [tempting_old_area, intended], np.array([-1.2, -0.8])
    )

    assert matched is intended


def test_distinct_blocked_approaches_defer_without_counting_duplicates() -> None:
    history: dict[tuple[int, int], list[np.ndarray]] = {}
    key = (4, 2)

    count, exhausted = explore._record_distinct_blocked_approach(
        history, key, np.array([1.0, 0.5])
    )
    assert (count, exhausted) == (1, False)

    count, exhausted = explore._record_distinct_blocked_approach(
        history, key, np.array([1.08, 0.52])
    )
    assert (count, exhausted) == (1, False)

    count, exhausted = explore._record_distinct_blocked_approach(
        history, key, np.array([1.45, 0.5])
    )
    assert (count, exhausted) == (2, True)


def test_navigation_yaw_wait_uses_fresh_reacquisition() -> None:
    class RecoveringImu:
        def deg(self):
            return None

        def deg_fresh(self, *, wait_up_to_s, max_age_s):
            assert wait_up_to_s == pytest.approx(0.1)
            assert max_age_s == pytest.approx(0.75)
            return 123.5

    assert explore._wait_for_navigation_yaw(
        RecoveringImu(), wait_up_to_s=0.1
    ) == pytest.approx(123.5)


def test_pick_target_does_not_globally_switch_away_from_active_frontier() -> None:
    analysis = _open_analysis()
    active = _frontier(0.40, 0.0, span=0.4)
    globally_tempting = _frontier(-0.40, 0.0, span=2.0)
    analysis.clusters = [globally_tempting, active]

    picked = explore._pick_target(
        analysis,
        robot_centre_xy=np.zeros(2),
        visited_xy=[],
        pullback_m=0.0,
        visited_skip_m=0.2,
        preferred_frontier_xy=np.array([0.42, 0.0]),
    )

    assert picked is not None
    assert picked[0] is active


def test_pick_target_turn_cost_is_soft_and_never_makes_rear_unreachable() -> None:
    analysis = _open_analysis()
    rear = explore.FrontierCluster(
        cells_ij=np.array([[10, 2]], dtype=np.int64),
        centroid_xy=analysis.to_world((10, 2)),
        span_m=1.4,
        size=8,
        passable=False,
    )
    analysis.clusters = [rear]

    picked = explore._pick_target(
        analysis,
        robot_centre_xy=np.zeros(2),
        visited_xy=[],
        pullback_m=0.4,
        visited_skip_m=0.2,
        current_heading_deg=0.0,
        turn_cost_per_deg=1.0,
    )

    assert picked is not None
    assert picked[0] is rear


def test_pick_target_softly_prefers_less_turn_when_frontiers_are_available() -> None:
    analysis = _open_analysis()
    rear = explore.FrontierCluster(
        cells_ij=np.array([[10, 2]], dtype=np.int64),
        centroid_xy=analysis.to_world((10, 2)),
        span_m=1.4,
        size=8,
        passable=False,
    )
    forward = explore.FrontierCluster(
        cells_ij=np.array([[10, 18]], dtype=np.int64),
        centroid_xy=analysis.to_world((10, 18)),
        span_m=0.4,
        size=8,
        passable=False,
    )
    analysis.clusters = [rear, forward]

    picked = explore._pick_target(
        analysis,
        robot_centre_xy=np.zeros(2),
        visited_xy=[],
        pullback_m=0.4,
        visited_skip_m=0.2,
        current_heading_deg=0.0,
        turn_cost_per_deg=1.0,
    )

    assert picked is not None
    assert picked[0] is forward


def test_high_gain_doorway_observation_arms_directed_room_commitment() -> None:
    transition = explore._doorway_transition(
        anchor_xy=np.array([1.0, 0.0]),
        objective_start_xy=np.array([0.0, 0.0]),
        objective_end_xy=np.array([1.2, 0.0]),
        observe_xy=np.array([2.0, 0.0]),
        scans_added=8,
        newly_known_cells=400,
        min_gain_cells=300,
    )

    assert transition is not None
    anchor, outward = transition
    assert anchor == pytest.approx([1.0, 0.0])
    assert outward == pytest.approx([1.0, 0.0])


def test_weak_or_empty_doorway_observation_does_not_arm_commitment() -> None:
    common = {
        "anchor_xy": np.array([1.0, 0.0]),
        "objective_start_xy": np.array([0.0, 0.0]),
        "objective_end_xy": np.array([1.2, 0.0]),
        "observe_xy": np.array([2.0, 0.0]),
        "min_gain_cells": 300,
    }

    assert explore._doorway_transition(
        **common, scans_added=0, newly_known_cells=400
    ) is None
    assert explore._doorway_transition(
        **common, scans_added=8, newly_known_cells=299
    ) is None


def test_high_gain_doorway_photo_does_not_fake_an_unexecuted_crossing() -> None:
    assert explore._doorway_transition(
        anchor_xy=np.array([1.0, 0.0]),
        objective_start_xy=np.array([0.0, 0.0]),
        objective_end_xy=np.array([0.35, 0.0]),
        observe_xy=np.array([2.0, 0.0]),
        scans_added=2,
        newly_known_cells=800,
        min_gain_cells=300,
    ) is None


def test_executed_patrol_crossing_detects_passable_frontier_direction() -> None:
    traversable = np.ones((24, 24), dtype=bool)
    analysis = explore.Analysis(
        origin_xy=np.zeros(2),
        res_m=0.10,
        occupied=np.zeros_like(traversable),
        free=traversable.copy(),
        traversable=traversable,
        cost=np.ones(traversable.shape, dtype=np.float64),
    )
    cells = np.array([[row, 10] for row in range(7, 16)], dtype=np.int64)
    analysis.clusters = [
        explore.FrontierCluster(
            cells_ij=cells,
            centroid_xy=analysis.to_world((11, 10)),
            span_m=0.90,
            size=len(cells),
            passable=True,
        )
    ]

    crossing = explore._crossed_passable_frontier(
        analysis,
        [np.array([0.45, 1.15]), np.array([1.55, 1.15])],
    )

    assert crossing is not None
    anchor, outward = crossing
    assert anchor[0] == pytest.approx(1.05)
    assert outward[0] > 0.99


def test_path_that_does_not_pass_doorway_does_not_arm_transition() -> None:
    traversable = np.ones((24, 24), dtype=bool)
    analysis = explore.Analysis(
        origin_xy=np.zeros(2),
        res_m=0.10,
        occupied=np.zeros_like(traversable),
        free=traversable.copy(),
        traversable=traversable,
        cost=np.ones(traversable.shape, dtype=np.float64),
    )
    cells = np.array([[row, 10] for row in range(7, 16)], dtype=np.int64)
    analysis.clusters = [
        explore.FrontierCluster(
            cells_ij=cells,
            centroid_xy=analysis.to_world((11, 10)),
            span_m=0.90,
            size=len(cells),
            passable=True,
        )
    ]

    assert explore._crossed_passable_frontier(
        analysis,
        [np.array([0.45, 0.20]), np.array([1.55, 0.20])],
    ) is None


def test_global_relocalization_rejects_old_room_teleport_alias() -> None:
    prior = np.array([2.75, 1.94])
    old_room_alias = np.array([1.15, 0.87])

    assert not explore._relocalization_candidate_is_continuous(
        prior,
        old_room_alias,
        score=9.4,
        min_score=6.0,
        support_ratio=0.294,
        min_support_ratio=0.35,
        max_pose_innovation_m=0.60,
    )
    assert explore._relocalization_candidate_is_continuous(
        prior,
        prior + np.array([0.18, -0.05]),
        score=9.4,
        min_score=6.0,
        support_ratio=0.70,
        min_support_ratio=0.35,
        max_pose_innovation_m=0.60,
    )


def test_pick_target_aims_at_visible_frontier_cell_and_predicts_gain() -> None:
    analysis = _open_analysis()
    # Known free space ends at column 17; column 18 is the free frontier and
    # columns beyond it are unknown. The cluster centroid is deliberately not
    # used as the observation ray target.
    analysis.free[:, 19:] = False
    analysis.traversable[:, 19:] = False
    cells = np.array([[i, 18] for i in range(7, 14)], dtype=np.int64)
    cluster = explore.FrontierCluster(
        cells_ij=cells,
        centroid_xy=np.array([-0.40, 0.40]),
        span_m=0.4,
        size=len(cells),
    )
    analysis.clusters = [cluster]

    picked = explore._pick_target(
        analysis,
        robot_centre_xy=np.zeros(2),
        visited_xy=[],
        pullback_m=0.4,
        visited_skip_m=0.2,
    )

    assert picked is not None
    _cluster, _goal, _path, _close, observe_xy, expected_gain = picked
    assert analysis.to_cell(observe_xy) in [tuple(cell) for cell in cells]
    assert not np.allclose(observe_xy, cluster.centroid_xy)
    assert expected_gain > 0


def test_pick_target_prioritizes_reachable_narrow_look_before_passage() -> None:
    analysis = _open_analysis()
    analysis.free[:, 19:] = False
    analysis.traversable[:, 19:] = False
    narrow_cells = np.array([[i, 18] for i in range(5, 10)], dtype=np.int64)
    passage_cells = np.array([[i, 18] for i in range(12, 19)], dtype=np.int64)
    narrow = explore.FrontierCluster(
        cells_ij=narrow_cells,
        centroid_xy=analysis.to_world((7, 18)),
        span_m=0.45,
        size=len(narrow_cells),
        passable=False,
    )
    passage = explore.FrontierCluster(
        cells_ij=passage_cells,
        centroid_xy=analysis.to_world((15, 18)),
        span_m=1.2,
        size=len(passage_cells),
        passable=True,
    )
    analysis.clusters = [passage, narrow]

    picked = explore._pick_target(
        analysis,
        robot_centre_xy=np.zeros(2),
        visited_xy=[],
        pullback_m=0.4,
        visited_skip_m=0.55,
    )

    assert picked is not None
    assert picked[0] is narrow


def test_pick_target_observes_through_long_nontraversable_gap() -> None:
    """A frontier remains observable when the footprint cannot enter its throat.

    The former planner searched only 0.30-1.00 m from each frontier cell and
    only tested ten poses near the preferred standoff. A long narrow opening
    therefore produced ``0 candidates`` even though LiDAR had a clear view
    from the robot's reachable room. Professional frontier projection searches
    the complete safe band and permits a bounded long-range observation pose.
    """
    shape = (61, 61)
    free = np.zeros(shape, dtype=bool)
    traversable = np.zeros(shape, dtype=bool)

    # Reachable room on the left. The 1.25 m-long free throat is visible to
    # LiDAR but too narrow for the inflated robot configuration space.
    free[10:51, 5:21] = True
    traversable[10:51, 5:21] = True
    free[30, 21:46] = True
    analysis = explore.Analysis(
        origin_xy=np.zeros(2),
        res_m=0.05,
        occupied=np.zeros(shape, dtype=bool),
        free=free,
        traversable=traversable,
        cost=np.ones(shape, dtype=np.float32),
    )
    frontier_cell = (30, 45)
    cluster = explore.FrontierCluster(
        cells_ij=np.array([frontier_cell], dtype=np.int64),
        centroid_xy=analysis.to_world(frontier_cell),
        span_m=0.6,
        size=8,
        passable=False,
    )
    analysis.clusters = [cluster]

    picked = explore._pick_target(
        analysis,
        robot_centre_xy=analysis.to_world((30, 10)),
        visited_xy=[],
        pullback_m=0.70,
        visited_skip_m=0.55,
    )

    assert picked is not None
    assert picked[0] is cluster
    assert np.hypot(*(picked[1] - picked[4])) > 1.0
    assert explore._line_free(
        analysis.free,
        analysis.to_cell(picked[1]),
        frontier_cell,
    )


def test_pick_target_does_not_abandon_frontier_after_ten_occluded_poses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _open_analysis()
    cells = np.array([[i, 18] for i in range(7, 14)], dtype=np.int64)
    cluster = explore.FrontierCluster(
        cells_ij=cells,
        centroid_xy=analysis.to_world((10, 18)),
        span_m=0.5,
        size=len(cells),
    )
    analysis.clusters = [cluster]

    real_line_free = explore._line_free
    sightline_calls = 0

    def _first_ten_occluded(mask, start, goal):
        nonlocal sightline_calls
        sightline_calls += 1
        if sightline_calls <= 10:
            return False
        return real_line_free(mask, start, goal)

    monkeypatch.setattr(explore, "_line_free", _first_ten_occluded)
    picked = explore._pick_target(
        analysis,
        robot_centre_xy=np.zeros(2),
        visited_xy=[],
        pullback_m=0.70,
        visited_skip_m=0.55,
    )

    assert picked is not None
    assert picked[0] is cluster
    assert sightline_calls > 10


def test_successful_narrow_viewpoint_retires_but_passable_exit_advances() -> None:
    assert explore._viewpoint_completes_frontier(
        _frontier(0.0, 0.0, passable=False),
        scans_added=8,
    )
    assert not explore._viewpoint_completes_frontier(
        _frontier(0.0, 0.0, passable=True),
        scans_added=8,
    )
    assert not explore._viewpoint_completes_frontier(
        _frontier(0.0, 0.0, passable=False),
        scans_added=0,
    )


def test_completed_doorway_excludes_old_room_but_keeps_new_room_and_sides() -> None:
    transitions = [
        (
            np.array([0.0, 0.0]),
            np.array([0.0, -1.0]),  # new room is toward negative y
        )
    ]

    assert explore._behind_completed_transition(
        np.array([0.0, 0.6]), transitions, slack_m=0.25
    )
    assert not explore._behind_completed_transition(
        np.array([0.0, -0.6]), transitions, slack_m=0.25
    )
    assert not explore._behind_completed_transition(
        np.array([1.0, -0.05]), transitions, slack_m=0.25
    )


def test_sampled_viewpoint_is_penalized_but_not_made_unreachable() -> None:
    analysis = _open_analysis()
    analysis.free[:, 19:] = False
    analysis.traversable[:, 19:] = False
    cells = np.array([[i, 18] for i in range(7, 14)], dtype=np.int64)
    cluster = explore.FrontierCluster(
        cells_ij=cells,
        centroid_xy=analysis.to_world((10, 18)),
        span_m=0.5,
        size=len(cells),
    )
    analysis.clusters = [cluster]

    picked = explore._pick_target(
        analysis,
        robot_centre_xy=np.zeros(2),
        visited_xy=[],
        pullback_m=0.4,
        visited_skip_m=2.0,
        sampled_viewpoints_xy=[np.zeros(2)],
    )

    assert picked is not None


def test_sampled_viewpoint_penalty_changes_final_frontier_choice() -> None:
    analysis = _open_analysis()
    right = explore.FrontierCluster(
        cells_ij=np.array([[10, 18]], dtype=np.int64),
        centroid_xy=analysis.to_world((10, 18)),
        span_m=0.5,
        size=8,
    )
    left = explore.FrontierCluster(
        cells_ij=np.array([[10, 2]], dtype=np.int64),
        centroid_xy=analysis.to_world((10, 2)),
        span_m=0.5,
        size=8,
    )
    analysis.clusters = [right, left]

    first = explore._pick_target(
        analysis,
        robot_centre_xy=np.zeros(2),
        visited_xy=[],
        pullback_m=0.4,
        visited_skip_m=0.55,
    )
    assert first is not None

    second = explore._pick_target(
        analysis,
        robot_centre_xy=np.zeros(2),
        visited_xy=[],
        pullback_m=0.4,
        visited_skip_m=0.55,
        sampled_viewpoints_xy=[first[1]],
    )

    assert second is not None
    assert second[0] is not first[0]


def test_collision_blocked_goal_changes_approach_without_retiring_frontier() -> None:
    analysis = _open_analysis()
    cluster = explore.FrontierCluster(
        cells_ij=np.array([[10, 18]], dtype=np.int64),
        centroid_xy=analysis.to_world((10, 18)),
        span_m=0.5,
        size=8,
    )
    analysis.clusters = [cluster]

    first = explore._pick_target(
        analysis,
        robot_centre_xy=np.zeros(2),
        visited_xy=[],
        pullback_m=0.4,
        visited_skip_m=0.55,
    )
    assert first is not None

    second = explore._pick_target(
        analysis,
        robot_centre_xy=np.zeros(2),
        visited_xy=[],
        pullback_m=0.4,
        visited_skip_m=0.55,
        blocked_viewpoints_xy=[first[1]],
        blocked_viewpoint_skip_m=0.45,
    )

    assert second is not None
    assert second[0] is cluster
    assert np.hypot(*(second[1] - first[1])) >= 0.45


def test_metric_inflation_does_not_round_31cm_up_to_35cm() -> None:
    assert explore._inflation_radius_cells(0.31, 0.05) == 6
    assert explore._inflation_radius_cells(0.28, 0.05) == 5

    # A 65cm clear doorway fits a 56cm-wide base plus the configured margin.
    # Rounding 31cm up to seven cells sealed it; six centre-offset cells leave
    # the valid centreline represented in the planner.
    wall = np.zeros((31, 31), dtype=bool)
    wall[15, :9] = True
    wall[15, 22:] = True
    inflated = explore._inflate(wall, explore._inflation_radius_cells(0.31, 0.05))
    assert not inflated[15, 15]


def test_replan_exposes_real_to_snapped_start_escape_segment() -> None:
    analysis = _open_analysis()
    centre = np.zeros(2)
    ci, cj = analysis.to_cell(centre)
    analysis.traversable[ci, cj] = False

    waypoints = explore._plan_waypoints(analysis, centre, np.array([0.40, 0.0]))

    assert waypoints is not None
    first = waypoints[0]
    assert analysis.traversable[analysis.to_cell(first)]
    heading = np.degrees(np.arctan2(first[1] - centre[1], first[0] - centre[0]))
    state, _distance = explore._swept_footprint_state(analysis, centre, heading, 0.12)
    assert state == "clear"


def test_straight_segment_path_collapses_grid_staircase_but_keeps_corner() -> None:
    traversable = np.ones((20, 20), dtype=bool)
    staircase = [
        (0, 0), (0, 1), (1, 1), (1, 2), (2, 2),
        (2, 3), (3, 3), (3, 4), (4, 4), (4, 5), (5, 5),
    ]
    assert explore._straight_segment_path(traversable, staircase, 1.0) == [
        staircase[0], staircase[-1]
    ]

    corner = [(10, j) for j in range(2, 11)] + [(i, 10) for i in range(9, 1, -1)]
    segmented = explore._straight_segment_path(traversable, corner, 1.0)
    assert segmented[0] == corner[0]
    assert segmented[-1] == corner[-1]
    assert len(segmented) >= 3
    assert all(
        explore._line_free(traversable, start, end)
        for start, end in zip(segmented[:-1], segmented[1:], strict=True)
    )


def test_segment_completion_accepts_endpoint_or_passed_endpoint() -> None:
    start = np.array([0.0, 0.0])
    end = np.array([1.0, 0.0])

    assert explore._segment_complete(np.array([0.92, 0.02]), start, end, 0.10)
    assert explore._segment_complete(np.array([1.05, 0.20]), start, end, 0.10)
    assert not explore._segment_complete(np.array([0.70, 0.20]), start, end, 0.10)


def test_collision_calibration_learns_full_angle_box_envelope() -> None:
    angles = np.radians(np.arange(-178.0, 180.0, 4.0))
    c = np.cos(angles)
    s = np.sin(angles)
    # Ray intersection with a 1.0m-long x 0.70m-wide calibration rectangle.
    ranges = np.minimum(
        0.50 / np.maximum(np.abs(c), 1e-9),
        0.35 / np.maximum(np.abs(s), 1e-9),
    )
    base = np.column_stack([ranges * c, ranges * s])
    scans = [base * (1.0 + 0.002 * ((i % 3) - 1)) for i in range(12)]

    profile = calibrate_collision_box(
        scans, bin_size_deg=4.0, min_scans_per_bin=8, max_boundary_m=1.0
    )

    learned = profile["ranges_m"]
    assert sum(value is not None for value in learned) >= 88
    front_idx = int((0.0 + 180.0) // 4.0)
    left_idx = int((90.0 + 180.0) // 4.0)
    assert learned[front_idx] == pytest.approx(0.50, abs=0.03)
    assert learned[left_idx] == pytest.approx(0.35, abs=0.03)


def test_collision_box_prioritizes_single_bin_side_intrusion() -> None:
    profile = {
        "version": 1,
        "frame": "physical_forward_xy",
        "bin_size_deg": 4.0,
        "ranges_m": [0.50] * 90,
        "noise_tolerance_m": 0.02,
        "min_violation_bins": 2,
        "side_min_violation_bins": 1,
    }

    assert collision_box_violation(np.array([[0.55, 0.0]]), profile) is None
    side_hit = collision_box_violation(np.array([[0.0, 0.40], [0.005, 0.41]]), profile)
    assert side_hit is not None
    mask, sector, angle, hit_range, limit = side_hit
    assert mask.tolist() == [True, True]
    assert sector == "left side"
    assert angle == pytest.approx(90.0)
    assert hit_range == pytest.approx(0.40)
    # Legacy profiles without safety_margin_m use the visible calibrated
    # envelope as the stop line; tolerance no longer shrinks it by 2cm.
    assert limit == pytest.approx(0.50)

    # A lone front speck is rejected, but adjacent front bins form a stop.
    assert collision_box_violation(np.array([[0.40, 0.0]]), profile) is None
    front = np.radians(np.array([0.0, 2.0, 5.0]))
    front_points = np.column_stack([0.40 * np.cos(front), 0.40 * np.sin(front)])
    assert collision_box_violation(front_points, profile)[1] == "front"


def test_collision_box_excludes_only_returns_safely_inside_physical_chassis() -> None:
    profile = {
        "version": 1,
        "frame": "physical_forward_xy",
        "bin_size_deg": 4.0,
        "ranges_m": [0.50] * 90,
        "noise_tolerance_m": 0.0,
        "safety_margin_m": 0.0,
        "min_violation_bins": 2,
        "side_min_violation_bins": 1,
        "side_min_violation_points": 2,
    }
    geometry = {
        "lidar_offset_forward_m": 0.229,
        "physical_body_radius_m": 0.28,
        "self_mask_inset_m": 0.02,
    }

    # At the LiDAR's lateral plane these points are deep inside the measured
    # chassis circle about the robot centre and cannot be an external obstacle.
    internal = np.array([[0.0, -0.08], [0.002, -0.085]])
    assert collision_box_violation(internal, profile, **geometry) is None

    # A wall 26cm to the side is outside that physical core but still inside
    # the larger collision envelope, so shoulder protection remains active.
    external = np.array([[0.0, -0.26], [0.002, -0.265]])
    hit = collision_box_violation(external, profile, **geometry)
    assert hit is not None
    assert hit[1] == "right side"


def test_rotational_sweep_allows_clearance_increasing_turn_and_blocks_worsening_turn() -> None:
    profile = {
        "version": 1,
        "frame": "physical_forward_xy",
        "bin_size_deg": 4.0,
        "ranges_m": [0.25 if 22 <= idx <= 67 else None for idx in range(90)],
        "complete_box": True,
        "completed_width_m": 0.724,
        "completed_front_m": 0.230,
        "completed_rear_m": 0.583,
        "corner_radius_m": 0.343,
        "noise_tolerance_m": 0.0,
        "safety_margin_m": 0.0,
        "min_violation_bins": 2,
        "side_min_violation_bins": 1,
        "side_min_violation_points": 2,
    }
    angles = np.radians(np.array([-81.0, -80.0, -79.0, -78.0]))
    right_obstacle = np.column_stack([
        0.30 * np.cos(angles),
        0.30 * np.sin(angles),
    ])
    geometry = {
        "lidar_offset_forward_m": 0.229,
        "physical_body_radius_m": 0.28,
        "self_mask_inset_m": 0.02,
    }

    # Positive yaw moves this right-side surface out of the rounded footprint;
    # the opposite direction sweeps the body farther into it.
    assert collision_box_rotation_violation(
        right_obstacle, profile, +30.0, **geometry
    ) is None
    blocked = collision_box_rotation_violation(
        right_obstacle, profile, -30.0, **geometry
    )
    assert blocked is not None
    assert blocked[0][1] == "right side"
    assert blocked[1] == pytest.approx(0.0)

    left_angles = np.radians(np.array([78.0, 79.0, 80.0, 81.0]))
    left_obstacle = np.column_stack([
        0.29 * np.cos(left_angles),
        0.29 * np.sin(left_angles),
    ])
    assert collision_box_rotation_violation(
        left_obstacle, profile, -30.0, **geometry
    ) is None
    left_blocked = collision_box_rotation_violation(
        left_obstacle, profile, +30.0, **geometry
    )
    assert left_blocked is not None
    assert left_blocked[0][1] == "left side"


def test_rotational_collision_mask_stays_aligned_after_self_return_filtering() -> None:
    profile = {
        "version": 1,
        "frame": "physical_forward_xy",
        "bin_size_deg": 4.0,
        "ranges_m": [0.50] * 90,
        "noise_tolerance_m": 0.0,
        "safety_margin_m": 0.0,
        "min_violation_bins": 1,
        "min_violation_points": 2,
        "side_min_violation_bins": 1,
        "side_min_violation_points": 2,
    }
    internal_self_return = np.array([[0.0, -0.08]])
    external_wall = np.array([
        [0.00, -0.30],
        [0.01, -0.30],
        [0.02, -0.30],
    ])
    points = np.vstack([internal_self_return, external_wall])

    result = collision_box_rotation_violation(
        points,
        profile,
        -20.0,
        lidar_offset_forward_m=0.229,
        physical_body_radius_m=0.28,
        self_mask_inset_m=0.02,
    )

    assert result is not None
    mask = result[0][0]
    assert mask.shape == (len(points),)
    assert not mask[0]
    assert np.any(mask[1:])


def test_rounded_square_self_filter_removes_fixed_chassis_shoulder_return() -> None:
    # This is the persistent +84deg/0.18m return from the failed physical run.
    angle = np.radians(84.0)
    chassis_return = np.array([[0.18 * np.cos(angle), 0.18 * np.sin(angle)]])
    # A point just beyond the inset side of the measured chassis must remain an
    # environmental collision observation.
    external_side = np.array([[-0.229, 0.27]])
    points = np.vstack([chassis_return, external_side])

    self_mask = physical_body_self_return_mask(
        points,
        lidar_offset_forward_m=0.229,
        physical_body_radius_m=0.28,
        self_mask_inset_m=0.02,
    )

    assert self_mask.tolist() == [True, False]


def test_chassis_shoulder_returns_cannot_latch_collision_guard() -> None:
    profile = {
        "version": 1,
        "frame": "physical_forward_xy",
        "bin_size_deg": 4.0,
        "ranges_m": [0.40] * 90,
        "noise_tolerance_m": 0.0,
        "safety_margin_m": 0.0,
        "min_violation_bins": 2,
        "min_violation_points": 3,
        "side_min_violation_bins": 1,
        "side_min_violation_points": 2,
    }
    chassis_angles = np.radians([83.0, 85.0, 84.0])
    chassis = np.column_stack([
        0.18 * np.cos(chassis_angles),
        0.18 * np.sin(chassis_angles),
    ])
    geometry = {
        "lidar_offset_forward_m": 0.229,
        "physical_body_radius_m": 0.28,
        "self_mask_inset_m": 0.02,
    }

    assert collision_box_violation(chassis, profile, **geometry) is None

    external_side = np.array([[-0.229, 0.27], [-0.225, 0.272]])
    assert collision_box_violation(external_side, profile, **geometry) is not None


def test_collision_box_dimensions_expand_live_envelope() -> None:
    profile = {
        "bin_size_deg": 4.0,
        "ranges_m": [0.50] * 90,
    }
    base_width, base_length = collision_box_dimensions(profile)
    base_ranges = effective_ranges(profile)
    profile["width_m"] = base_width + 0.0254
    profile["length_m"] = base_length

    expanded = effective_ranges(profile)

    assert collision_box_dimensions(profile)[0] == pytest.approx(base_width + 0.0254)
    assert expanded[67] > base_ranges[67]  # approximately the left side
    assert expanded[45] == pytest.approx(base_ranges[45], rel=0.01)  # approximately front


def test_completed_collision_box_fills_rear_bins_and_detects_rear_intrusion() -> None:
    profile = {
        "version": 1,
        "frame": "physical_forward_xy",
        "bin_size_deg": 4.0,
        "ranges_m": [0.50 if 22 <= idx <= 67 else None for idx in range(90)],
        "complete_box": True,
        "width_m": 1.0,
        "length_m": 0.8,
        "corner_radius_m": 0.10,
        "noise_tolerance_m": 0.0,
        "safety_margin_m": 0.0,
        "min_violation_bins": 2,
        "min_violation_points": 3,
    }

    completed = effective_ranges(profile)

    assert np.all(np.isfinite(completed))
    assert completed[45] == pytest.approx(0.40, abs=0.02)  # front
    assert completed[0] == pytest.approx(0.40, abs=0.02)  # rear
    rear_angles = np.radians(np.array([176.0, 179.0, -176.0]))
    rear_points = np.column_stack([
        0.30 * np.cos(rear_angles),
        0.30 * np.sin(rear_angles),
    ])
    hit = collision_box_violation(rear_points, profile)
    assert hit is not None
    assert hit[1] == "rear"


def test_completed_collision_box_corner_radius_rounds_square_corners() -> None:
    profile = {
        "bin_size_deg": 4.0,
        "ranges_m": [0.50] * 90,
        "complete_box": True,
        "width_m": 1.0,
        "length_m": 1.0,
        "corner_radius_m": 0.0,
    }
    square = effective_ranges(profile)
    profile["corner_radius_m"] = 0.20
    rounded = effective_ranges(profile)

    diagonal_bin = 56  # approximately +46 degrees
    assert rounded[diagonal_bin] < square[diagonal_bin]
    assert rounded[45] == pytest.approx(square[45], abs=0.01)


def test_completed_collision_box_keeps_separate_dimensions_from_learned_box() -> None:
    profile = {
        "bin_size_deg": 4.0,
        "ranges_m": [0.50] * 90,
        "complete_box": True,
        "width_m": 0.50,
        "length_m": 0.40,
        "completed_width_m": 0.80,
        "completed_length_m": 1.00,
        "corner_radius_m": 0.0,
    }

    completed = effective_ranges(profile)
    assert completed[45] == pytest.approx(0.50, abs=0.02)  # completed front
    assert completed[67] == pytest.approx(0.40, abs=0.02)  # completed side

    profile["complete_box"] = False
    assert collision_box_dimensions(profile) == pytest.approx((0.50, 0.40))


def test_completed_collision_box_has_independent_front_and_rear_depths() -> None:
    profile = {
        "bin_size_deg": 4.0,
        "ranges_m": [0.50] * 90,
        "complete_box": True,
        "completed_width_m": 0.60,
        "completed_length_m": 0.80,
        "completed_front_m": 0.25,
        "completed_rear_m": 0.55,
        "corner_radius_m": 0.0,
    }

    completed = effective_ranges(profile)

    assert completed[45] == pytest.approx(0.25, abs=0.02)
    assert completed[0] == pytest.approx(0.55, abs=0.02)
    assert completed[67] == pytest.approx(0.30, abs=0.02)


def test_swept_footprint_reports_mapped_shoulder_clearance() -> None:
    analysis = _open_analysis()
    # A centre position 10cm ahead is inside the obstacle-inflation layer,
    # representing a desk corner that would catch the shoulder.
    i, j = analysis.to_cell(np.array([0.10, 0.0]))
    analysis.traversable[i, j] = False

    state, distance = explore._swept_footprint_state(analysis, np.zeros(2), 0.0, 0.20)

    assert state == "blocked"
    assert distance == pytest.approx(0.10, abs=0.03)


def test_swept_footprint_distinguishes_map_edge_from_obstacle() -> None:
    analysis = _open_analysis()
    i, j = analysis.to_cell(np.array([0.15, 0.0]))
    analysis.free[i, j] = False
    analysis.traversable[i, j] = False

    state, distance = explore._swept_footprint_state(analysis, np.zeros(2), 0.0, 0.20)

    assert state == "unknown"
    assert distance == pytest.approx(0.15, abs=0.03)


def test_swept_footprint_uses_nearby_validated_planner_start() -> None:
    analysis = _open_analysis()
    raw_centre = np.zeros(2)
    ci, cj = analysis.to_cell(raw_centre)
    analysis.free[ci, cj] = False
    analysis.traversable[ci, cj] = False
    planned_start = np.array([0.05, 0.0])

    without_reference, distance = explore._swept_footprint_state(
        analysis, raw_centre, 0.0, 0.20
    )
    with_reference, _ = explore._swept_footprint_state(
        analysis,
        raw_centre,
        0.0,
        0.20,
        planned_start_xy=planned_start,
    )

    assert without_reference == "unknown"
    assert distance == 0.0
    assert with_reference == "clear"


def test_swept_footprint_rejects_distant_planner_start() -> None:
    analysis = _open_analysis()
    raw_centre = np.zeros(2)
    ci, cj = analysis.to_cell(raw_centre)
    analysis.free[ci, cj] = False
    analysis.traversable[ci, cj] = False

    state, distance = explore._swept_footprint_state(
        analysis,
        raw_centre,
        0.0,
        0.20,
        planned_start_xy=np.array([0.40, 0.0]),
    )

    assert state == "unknown"
    assert distance == 0.0


def test_zero_distance_map_edge_is_pose_disagreement_not_arrival() -> None:
    assert explore._map_edge_is_pose_disagreement(0.0, 0.05)
    assert explore._map_edge_is_pose_disagreement(0.02, 0.05)
    assert not explore._map_edge_is_pose_disagreement(0.10, 0.05)


def test_rolling_local_submap_is_bounded_and_replaceable() -> None:
    submap = explore.RollingLocalSubmap(max_scans=3, max_points=500)
    local = np.column_stack([np.linspace(0.2, 1.0, 40), np.zeros(40)]).astype(
        np.float32
    )
    for index in range(5):
        submap.add(local, Pose2D(0.1 * index, 0.0, 0.0))

    assert len(submap.scans) == 3
    assert len(submap.reference()) == 120

    replacement = Pose2D(2.0, -1.0, 5.0)
    submap.reset(local, replacement)
    assert len(submap.scans) == 1
    assert submap.scans[0].pose == replacement


def test_local_endpoint_support_accepts_overlap_and_rejects_disjoint_scan() -> None:
    reference = np.column_stack([
        np.linspace(0.0, 1.0, 50),
        np.zeros(50),
    ])
    overlapping = reference + np.array([0.02, 0.01])
    disjoint = reference + np.array([2.0, 0.0])

    assert explore._endpoint_support_ratio(overlapping, reference, 0.05) > 0.90
    assert explore._endpoint_support_ratio(disjoint, reference, 0.05) == 0.0


def test_local_submap_match_recovers_relative_translation() -> None:
    wall_x = np.column_stack([
        np.linspace(0.2, 2.0, 80),
        np.full(80, 0.3),
    ])
    wall_y = np.column_stack([
        np.full(70, 0.2),
        np.linspace(0.3, 1.8, 70),
    ])
    local = np.vstack([wall_x, wall_y]).astype(np.float32)
    expected = Pose2D(0.18, -0.09, 4.0)
    reference = _transform_points(local, expected)
    args = SimpleNamespace(stitch_resolution_m=0.03, match_max_points=700)

    solved, score, support = explore._localize_against_points(
        local,
        reference,
        Pose2D(0.14, -0.05, 3.0),
        args,
        search_xy_m=0.15,
        theta_window_deg=4.0,
    )

    assert solved.x == pytest.approx(expected.x, abs=0.05)
    assert solved.y == pytest.approx(expected.y, abs=0.05)
    assert score > 10.0
    assert support > 0.90


def test_swept_footprint_allows_escape_from_inflation_band() -> None:
    analysis = _open_analysis()
    # Put the centre and cells to its left inside inflation; +x is the shortest
    # route back to traversable space.
    ci, cj = analysis.to_cell(np.zeros(2))
    analysis.traversable[ci, : cj + 1] = False

    escaping, _distance = explore._swept_footprint_state(
        analysis, np.zeros(2), 0.0, 0.25
    )
    worsening, worsening_distance = explore._swept_footprint_state(
        analysis, np.zeros(2), 180.0, 0.25
    )

    assert escaping == "clear"
    assert worsening == "blocked"
    assert worsening_distance == 0.0


def test_memory_bounded_matcher_keeps_theta_major_candidate_result() -> None:
    # An asymmetric L plus diagonal gives the matcher one unambiguous basin and
    # exercises both the coarse and refine theta-slab scoring paths.
    wall_x = np.column_stack([np.linspace(0.2, 2.0, 50), np.full(50, 0.3)])
    wall_y = np.column_stack([np.full(45, 0.2), np.linspace(0.3, 1.8, 45)])
    diagonal = np.column_stack([np.linspace(0.5, 1.4, 30), np.linspace(0.6, 1.1, 30)])
    local = np.vstack([wall_x, wall_y, diagonal]).astype(np.float32)
    expected = Pose2D(0.20, -0.10, 7.0)
    world = _transform_points(local, expected)

    solved, meta = _search_pose(
        snapshot_points_xy=local,
        global_points_xy=world,
        initial_pose=Pose2D(0.15, -0.05, 5.0),
        resolution_m=0.03,
        search_xy_m=0.15,
        coarse_angle_step_deg=2.0,
        fine_angle_step_deg=0.5,
        theta_window_deg=8.0,
        max_translation_from_initial_m=0.25,
    )

    assert solved.x == pytest.approx(expected.x, abs=0.04)
    assert solved.y == pytest.approx(expected.y, abs=0.04)
    assert solved.theta_deg == pytest.approx(expected.theta_deg, abs=1.0)
    assert float(meta["score"]) > 10.0


def test_cuda_candidate_scores_match_cpu_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(4)
    exact = rng.random((80, 90)) < 0.06
    dilated = stitch._dilate(exact, radius_cells=2)
    known = stitch._dilate(exact, radius_cells=8)
    origin = np.array([-1.2, -1.0], dtype=np.float32)
    occupied_ij = np.column_stack(np.nonzero(exact))
    global_xy = np.column_stack([
        occupied_ij[:, 1] * 0.03 + origin[0],
        occupied_ij[:, 0] * 0.03 + origin[1],
    ]).astype(np.float32)[::3]
    candidates = rng.uniform(-0.9, 1.0, size=(48, 90, 2)).astype(np.float32)
    kwargs = {
        "exact_grid": exact,
        "dilated_grid": dilated,
        "known_grid": known,
        "grid_origin_xy": origin,
        "resolution_m": 0.03,
        "global_sampled_xy": global_xy,
        "use_nearest_penalty": True,
    }
    try:
        stitch.configure_candidate_scoring_device("cpu")
        cpu_scores = stitch._score_candidates_batch(candidates, **kwargs)
        stitch.configure_candidate_scoring_device("cuda")
        cuda_scores = stitch._score_candidates_batch(candidates, **kwargs)
    finally:
        stitch.configure_candidate_scoring_device("cpu")

    assert cuda_scores == pytest.approx(cpu_scores, abs=2e-6)
    assert int(np.argmax(cuda_scores)) == int(np.argmax(cpu_scores))


def test_routine_localization_can_disable_expensive_whole_map_fallback() -> None:
    local = np.column_stack([np.linspace(0.1, 0.8, 24), np.zeros(24)]).astype(np.float32)
    unrelated_map = local + np.array([4.0, 4.0], dtype=np.float32)

    _solved, meta = _search_pose(
        snapshot_points_xy=local,
        global_points_xy=unrelated_map,
        initial_pose=Pose2D(0.0, 0.0, 0.0),
        resolution_m=0.03,
        search_xy_m=0.10,
        coarse_angle_step_deg=2.0,
        fine_angle_step_deg=0.5,
        theta_window_deg=4.0,
        max_translation_from_initial_m=0.20,
        allow_whole_map_search=False,
    )

    assert float(meta["local_score"]) < 8.0
    assert meta["whole_map_searched"] is False


def test_fast_controller_latches_safety_before_forward_command() -> None:
    class FakeRobot:
        _z_pos_cmd = 100.0

        def __init__(self) -> None:
            self.actions: list[dict[str, float]] = []

        def send_action(self, action) -> None:
            self.actions.append(dict(action))

    class FakeImu:
        def deg(self) -> float:
            return 0.0

    robot = FakeRobot()
    controller = explore.BaseController(robot, {}, FakeImu(), rate_hz=100.0)
    controller.set_safety_check(lambda: "mapped full-body clearance")
    controller.start()
    controller.drive_toward(0.8, 0.0)
    time.sleep(0.06)
    controller.shutdown()

    assert controller.safety_latched_reason() == "mapped full-body clearance"
    assert robot.actions
    assert all(float(action["x.vel"]) <= 0.0 for action in robot.actions)


def test_fast_controller_latches_rotational_safety_before_pivot_command() -> None:
    class FakeRobot:
        _z_pos_cmd = 100.0

        def __init__(self) -> None:
            self.actions: list[dict[str, float]] = []

        def send_action(self, action) -> None:
            self.actions.append(dict(action))

    class FakeImu:
        def deg(self) -> float:
            return 0.0

    robot = FakeRobot()
    controller = explore.BaseController(robot, {}, FakeImu(), rate_hz=100.0)
    controller.set_rotation_safety_check(
        lambda _remaining_deg: "rotational collision box left side"
    )
    controller.start()
    controller.rotate_to(90.0)
    time.sleep(0.06)
    controller.shutdown()

    assert controller.safety_latched_reason() == "rotational collision box left side"
    assert robot.actions
    assert all(float(action["theta.vel"]) == 0.0 for action in robot.actions)


def test_heading_hold_deadband_does_not_hunt_small_imu_error() -> None:
    class FakeRobot:
        _z_pos_cmd = 100.0

        def __init__(self) -> None:
            self.actions: list[dict[str, float]] = []

        def send_action(self, action) -> None:
            self.actions.append(dict(action))

    class FakeImu:
        def deg(self) -> float:
            return 1.0

    robot = FakeRobot()
    controller = explore.BaseController(
        robot, {}, FakeImu(), rate_hz=100.0, hold_deadband_deg=1.5
    )
    controller.start()
    controller.drive_toward(0.8, 0.0)
    time.sleep(0.06)
    controller.shutdown()

    moving = [action for action in robot.actions if float(action["x.vel"]) > 0.0]
    assert moving
    assert all(float(action["theta.vel"]) == 0.0 for action in moving)


def test_base_controller_streams_body_frame_strafe_recovery() -> None:
    class FakeRobot:
        _z_pos_cmd = 100.0

        def __init__(self) -> None:
            self.actions: list[dict[str, float]] = []

        def send_action(self, action) -> None:
            self.actions.append(dict(action))

    class FakeImu:
        def deg(self) -> float:
            return 12.0

    robot = FakeRobot()
    controller = explore.BaseController(robot, {}, FakeImu(), rate_hz=100.0)
    controller.start()
    controller.translate_body(-0.6, 0.8, 12.0)
    time.sleep(0.06)
    controller.shutdown()

    recovery = [
        action
        for action in robot.actions
        if float(action["x.vel"]) < 0.0 and float(action["y.vel"]) > 0.0
    ]
    assert recovery
    assert all(float(action["theta.vel"]) == 0.0 for action in recovery)


def test_base_controller_legacy_host_sentinel_matches_proven_spin_packet() -> None:
    class FakeRobot:
        _z_pos_cmd = 100.0

        def __init__(self) -> None:
            self.actions: list[dict[str, float | bool]] = []

        def send_action(self, action) -> None:
            self.actions.append(dict(action))

    class FakeImu:
        def deg(self) -> float:
            return 0.0

    robot = FakeRobot()
    controller = explore.BaseController(robot, {}, FakeImu(), rate_hz=100.0)
    controller.enable_legacy_host_untorque_sentinel()
    controller.start()
    controller.rotate_to(45.0)
    time.sleep(0.06)
    controller.shutdown()

    moving = [
        action for action in robot.actions if abs(float(action["theta.vel"])) > 0.0
    ]
    assert moving
    assert all(action["untorque_left"] is True for action in moving)
    assert all(action["untorque_right"] is True for action in moving)
