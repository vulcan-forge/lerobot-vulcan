"""Regression tests for the explorer's LiDAR/IMU pose-frame math."""

from __future__ import annotations

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
from ldlidar_direct_snapshot_stitch import Pose2D, _search_pose, _transform_points  # noqa: E402
from sourccey_collision_box import (  # noqa: E402
    calibrate_collision_box,
    collision_box_dimensions,
    collision_box_violation,
    effective_ranges,
)
from sourccey_wander.imu_heading import ImuYawClient  # noqa: E402


def test_zero_exploration_budgets_are_unlimited() -> None:
    assert not explore._budget_exhausted(10_000, 0)
    assert not explore._budget_exhausted(10_000, -1)
    assert not explore._budget_exhausted(19, 20)
    assert explore._budget_exhausted(20, 20)


def test_anchor_heading_coverage_handles_wraparound() -> None:
    headings = [350.0, 0.0, 10.0, 90.0, 180.0, 270.0]

    assert explore._max_heading_gap_deg(headings) == 90.0
    assert explore._max_heading_gap_deg([0.0]) == 360.0


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


def test_timestamped_imu_yaw_interpolates_unwrapped_heading() -> None:
    imu = ImuYawClient("tcp://unused")
    imu._history = deque([(100.0, 350.0), (100.1, 370.0)], maxlen=2048)

    assert imu.deg_at_wall_time(100.05) == pytest.approx(360.0)
    assert imu.deg_at_wall_time(99.0) is None
    assert imu.deg_at_wall_time(101.0) is None


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
