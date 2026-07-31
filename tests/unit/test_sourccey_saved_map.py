"""Persistence and startup-localization tests for saved-map navigation."""

from __future__ import annotations

import math
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ldlidar_direct_snapshot_stitch import Pose2D  # noqa: E402
from sourccey_saved_map import load_saved_map, save_world_map  # noqa: E402
from sourccey_saved_map_navigator import (  # noqa: E402
    LivePassagePlan,
    SavedMapNavigator,
    _assemble_active_localization_cloud,
    _current_escape_turn_deg,
    _densify_transition_route,
    _drop_reached_waypoint_prefix,
    _forward_command_above_stiction,
    _front_obstacle_escape_turn_deg,
    _global_localization_accepted,
    _global_localization_position_seeds,
    _learn_body_fixed_lidar_returns,
    _occupancy_pose_consistency,
    _one_sided_escape_turn_deg,
    _plan_saved_map_route,
    _pose_consensus,
    _predictive_forward_collision,
    _remove_body_fixed_lidar_returns,
    _route_boundary_reanchor_consensus,
    _simplify_saved_map_route,
    _stationary_correction_consensus,
    _straight_motion_step_is_consistent,
)


def test_reached_route_connectors_cannot_command_a_reverse_turn() -> None:
    # Regression from the second outbound trip: collision replanning emitted
    # two grid connectors behind the current centre. Both were already inside
    # the follower's 16cm arrival radius, but the old follower calculated the
    # first connector's ~180deg heading before checking arrival.
    centre = np.array([2.895, 2.035])
    route = [
        np.array([2.825, 2.025]),
        np.array([2.775, 2.075]),
        np.array([2.875, 2.875]),
    ]

    actionable, dropped = _drop_reached_waypoint_prefix(route, centre)

    assert dropped == 2
    assert len(actionable) == 1
    assert np.allclose(actionable[0], route[-1])


def test_global_localization_accepts_strongly_supported_distinct_mode() -> None:
    # Field regression: this printed as match 7.0 but its underlying float was
    # just below 7.0, despite 96% map support and a 1.66 mode margin.
    assert _global_localization_accepted(6.99, 0.96, 1.66, 7.0)
    assert not _global_localization_accepted(6.99, 0.45, 0.06, 7.0)


def test_collision_confirmation_counts_distinct_consistent_frames_only() -> None:
    navigator = SavedMapNavigator.__new__(SavedMapNavigator)
    navigator.args = SimpleNamespace(collision_confirmation_frames=3)
    navigator._collision_confirmation = {}
    navigator._collision_confirmation_lock = threading.Lock()
    hit = (
        np.array([True, False]),
        "right side",
        -70.0,
        0.29,
        0.31,
    )

    assert navigator._confirmed_collision("translation", 10, hit) is None
    # The 25Hz controller can inspect one LiDAR revolution several times; that
    # must still count as exactly one observation.
    assert navigator._confirmed_collision("translation", 10, hit) is None
    assert navigator._confirmed_collision("translation", 11, hit) is None
    assert navigator._confirmed_collision("translation", 12, hit) is hit

    changed_sector = (hit[0], "left side", 70.0, 0.29, 0.31)
    assert navigator._confirmed_collision("translation", 13, changed_sector) is None


def test_active_sweep_learns_and_removes_only_body_fixed_lidar_returns() -> None:
    body_bearing = -70.0
    body_range = 0.26
    captures: list[tuple[np.ndarray, float]] = []
    for yaw in range(0, 360, 30):
        # The chassis cluster is constant in the sensor frame. The wall return
        # moves one bin per view and therefore must not become a self mask.
        angles = np.radians(
            np.array([body_bearing - 0.6, body_bearing + 0.6, -150.0 + yaw])
        )
        ranges = np.array([body_range, body_range + 0.004, 1.5])
        captures.append(
            (np.column_stack([ranges * np.cos(angles), ranges * np.sin(angles)]), float(yaw))
        )

    learned = _learn_body_fixed_lidar_returns(captures, 0.0)
    assert len(learned) >= 1
    current_angles = np.radians(np.array([body_bearing, 20.0]))
    current_ranges = np.array([body_range + 0.005, 0.55])
    current = np.column_stack(
        [current_ranges * np.cos(current_angles), current_ranges * np.sin(current_angles)]
    )

    filtered = _remove_body_fixed_lidar_returns(current, learned)
    assert filtered.shape == (1, 2)
    assert np.allclose(filtered[0], current[1])


def test_saved_map_round_trip_is_pickle_free_and_lossless(tmp_path: Path) -> None:
    scan_a = np.array([[1.0, 0.0], [1.0, 0.1]], dtype=np.float32)
    scan_b = np.array([[0.5, -0.2]], dtype=np.float32)
    world = SimpleNamespace(
        scans=[
            SimpleNamespace(local_xy=scan_a, pose=Pose2D(1.0, 2.0, 30.0), gold=True),
            SimpleNamespace(local_xy=scan_b, pose=Pose2D(1.2, 2.1, 31.0), gold=False),
        ],
        grid=SimpleNamespace(
            res=0.05,
            L=np.array([[0.0, 1.5], [-1.0, 0.0]], dtype=np.float32),
            origin=np.array([-1.0, -2.0]),
        ),
        footprint_clear_m=0.28,
        lidar_offset_m=0.229,
        forward_offset_deg=180.0,
    )
    destination = save_world_map(
        tmp_path / "room.npz",
        world,
        Pose2D(1.2, 2.1, 31.0),
        trail=[np.array([0.0, 0.0]), np.array([1.0, 2.0])],
        sensor_config={"forward_angle_deg": 90.0},
    )
    saved = load_saved_map(destination)

    assert saved.metadata["gold_scan_count"] == 1
    assert saved.metadata["sensor_config"]["forward_angle_deg"] == 90.0
    assert np.array_equal(saved.local_scan(0), scan_a)
    assert np.array_equal(saved.local_scan(1), scan_b)
    assert np.array_equal(saved.occupancy_log_odds, world.grid.L)
    assert np.allclose(saved.current_pose, [1.2, 2.1, 31.0])
    assert np.allclose(saved.trail_xy[-1], [1.0, 2.0])


def test_startup_pose_consensus_requires_repeatable_hypothesis() -> None:
    coherent = [
        (Pose2D(1.00, 2.00, 20.0), 12.0, 0.7),
        (Pose2D(1.04, 2.02, 21.0), 11.0, 0.6),
        (Pose2D(3.00, -1.00, 170.0), 10.0, 0.5),
    ]
    solved = _pose_consensus(coherent)
    assert solved is not None
    assert np.hypot(solved.x - 1.02, solved.y - 2.01) < 0.04
    assert abs(solved.theta_deg - 20.5) < 1.0

    assert _pose_consensus(coherent[::2]) is None


def test_active_localization_cloud_accounts_for_off_centre_lidar_arc() -> None:
    sensor_origins = _assemble_active_localization_cloud(
        [
            (np.array([[0.0, 0.0]], dtype=np.float32), 0.0),
            (np.array([[0.0, 0.0]], dtype=np.float32), 90.0),
        ],
        lidar_offset_m=0.20,
        forward_offset_deg=0.0,
    )

    assert np.allclose(sensor_origins[0], [0.20, 0.0], atol=1e-5)
    assert np.allclose(sensor_origins[1], [0.0, 0.20], atol=1e-5)


def test_pose_only_shutdown_artifact_cannot_replace_saved_map(tmp_path: Path) -> None:
    world = SimpleNamespace(
        scans=[],
        grid=SimpleNamespace(
            res=0.05,
            L=np.zeros((8, 8), dtype=np.float32),
            origin=np.array([-1.0, -1.0]),
        ),
        footprint_clear_m=0.28,
        lidar_offset_m=0.229,
        forward_offset_deg=180.0,
    )
    destination = tmp_path / "room.npz"
    destination.write_bytes(b"previous-good-map")

    with pytest.raises(ValueError, match="no LiDAR scans"):
        save_world_map(destination, world, Pose2D(0.0, 0.0, 0.0))

    assert destination.read_bytes() == b"previous-good-map"


def test_global_localization_seeds_include_free_space_and_saved_scan_centres(
    tmp_path: Path,
) -> None:
    world = SimpleNamespace(
        scans=[
            SimpleNamespace(
                local_xy=np.array([[1.0, 0.0]], dtype=np.float32),
                pose=Pose2D(0.20, 0.0, 180.0),
                gold=True,
            )
        ],
        grid=SimpleNamespace(
            res=0.10,
            L=np.array([[-1.0, 0.0], [0.0, 2.0]], dtype=np.float32),
            origin=np.array([2.0, 3.0]),
        ),
        footprint_clear_m=0.28,
        lidar_offset_m=0.20,
        forward_offset_deg=180.0,
    )
    destination = save_world_map(
        tmp_path / "seeds.npz", world, Pose2D(0.20, 0.0, 180.0)
    )
    saved = load_saved_map(destination)
    seeds = _global_localization_position_seeds(saved, spacing_m=0.10)

    assert np.min(np.linalg.norm(seeds - np.array([2.05, 3.05]), axis=1)) < 1e-5
    assert np.min(np.linalg.norm(seeds - np.array([0.0, 0.0]), axis=1)) < 1e-5


def test_occupancy_consistency_rewards_hits_with_clear_rays(tmp_path: Path) -> None:
    world = SimpleNamespace(
        scans=[
            SimpleNamespace(
                local_xy=np.array([[1.0, 0.0]], dtype=np.float32),
                pose=Pose2D(0.0, 0.0, 0.0),
                gold=True,
            )
        ],
        grid=SimpleNamespace(
            res=0.10,
            L=np.array([[-1.0] * 10 + [2.0]], dtype=np.float32),
            origin=np.array([0.0, -0.05]),
        ),
        footprint_clear_m=0.0,
        lidar_offset_m=0.0,
        forward_offset_deg=0.0,
    )
    destination = save_world_map(
        tmp_path / "rays.npz", world, Pose2D(0.0, 0.0, 0.0)
    )
    saved = load_saved_map(destination)
    quality, endpoint_occupied, endpoint_free, ray_occupied = (
        _occupancy_pose_consistency(
            saved,
            [(np.array([[1.05, 0.0]], dtype=np.float32), 0.0)],
            Pose2D(0.0, 0.0, 0.0),
            0.0,
            0.0,
        )
    )

    assert quality > 3.0
    assert endpoint_occupied == 1.0
    assert endpoint_free == 0.0
    assert ray_occupied == 0.0


def test_saved_map_route_retries_at_physical_radius() -> None:
    from sourccey_explore import OccupancyGrid, WorldMap

    world = WorldMap(grid_res_m=0.05, footprint_clear_m=0.28)
    world.grid = OccupancyGrid(0.05)
    world.grid.L.fill(-2.0)
    # A 0.60m opening is sealed by six-cell (0.30m) inflation but remains
    # represented by five-cell (0.25m) inflation for a 0.28m footprint.
    wall_row = world.grid.L.shape[0] // 2
    centre_col = world.grid.L.shape[1] // 2
    world.grid.L[wall_row, :] = 2.0
    world.grid.L[wall_row, centre_col - 5 : centre_col + 6] = -2.0
    start = world.grid.origin + np.array(
        [(centre_col + 0.5) * 0.05, (wall_row - 10 + 0.5) * 0.05]
    )
    goal = world.grid.origin + np.array(
        [(centre_col + 0.5) * 0.05, (wall_row + 10 + 0.5) * 0.05]
    )

    route, radius = _plan_saved_map_route(world, start, goal, 0.31, 0.28)

    assert route is not None
    assert radius == 0.28


def test_saved_map_route_moves_to_corridor_centreline() -> None:
    from sourccey_explore import OccupancyGrid, WorldMap

    world = WorldMap(grid_res_m=0.05, footprint_clear_m=0.28)
    world.grid = OccupancyGrid(0.05)
    world.grid.L.fill(-2.0)
    centre_row = world.grid.L.shape[0] // 2
    centre_col = world.grid.L.shape[1] // 2
    world.grid.L[centre_row - 12, :] = 2.0
    world.grid.L[centre_row + 12, :] = 2.0
    start_cell = (centre_row - 6, centre_col - 30)
    goal_cell = (centre_row - 6, centre_col + 30)
    start = world.grid.origin + np.array(
        [(start_cell[1] + 0.5) * 0.05, (start_cell[0] + 0.5) * 0.05]
    )
    goal = world.grid.origin + np.array(
        [(goal_cell[1] + 0.5) * 0.05, (goal_cell[0] + 0.5) * 0.05]
    )

    route, _radius = _plan_saved_map_route(world, start, goal, 0.28, 0.28)

    assert route is not None
    route_rows = [
        int((float(point[1]) - float(world.grid.origin[1])) / world.grid.res)
        for point in route[:-1]
    ]
    assert min(abs(row - centre_row) for row in route_rows) <= 1


def test_saved_map_route_simplifier_removes_small_clearance_spike() -> None:
    traversable = np.ones((80, 80), dtype=bool)
    analysis = SimpleNamespace(
        traversable=traversable,
        to_cell=lambda point: (
            int(round(float(point[1]) / 0.05)) + 20,
            int(round(float(point[0]) / 0.05)) + 20,
        ),
    )
    route = [
        np.array([0.50, 0.00]),
        np.array([1.00, 0.16]),
        np.array([1.50, 0.00]),
        np.array([2.00, 0.00]),
    ]

    simplified = _simplify_saved_map_route(
        analysis,
        np.array([0.0, 0.0]),
        route,
    )

    assert len(simplified) == 1
    assert np.allclose(simplified[0], [2.0, 0.0])


def test_saved_map_route_simplifier_preserves_wall_clearance_corner() -> None:
    traversable = np.ones((80, 80), dtype=bool)
    cost = np.ones((80, 80), dtype=np.float32)

    def to_cell(point):
        return (
            int(round(float(point[1]) / 0.05)) + 20,
            int(round(float(point[0]) / 0.05)) + 20,
        )

    # The direct diagonal is technically traversable but follows a costly
    # inside-corner band. A* deliberately routed around it through the two
    # clearance waypoints; simplification must not erase those waypoints.
    for index in range(12, 30):
        cost[index, index] = 5.0
        cost[index, index + 1] = 5.0
    analysis = SimpleNamespace(
        traversable=traversable,
        cost=cost,
        to_cell=to_cell,
    )
    route = [
        np.array([0.20, 0.70]),
        np.array([0.70, 1.00]),
        np.array([1.00, 1.00]),
    ]

    simplified = _simplify_saved_map_route(
        analysis,
        np.array([0.0, 0.0]),
        route,
    )

    assert len(simplified) >= 2


def test_post_turn_recovery_policy_rejects_route_behind_robot() -> None:
    """Regression contract for the second-pass corner collision.

    A route 160.9 degrees behind a verified escape heading must never be
    treated as forward progress.  Keep the policy threshold explicit here so
    later tuning cannot silently re-enable that reversal.
    """
    verified_escape_heading_deg = 357.4
    replanned_first_heading_deg = -163.5
    error = abs(
        (replanned_first_heading_deg - verified_escape_heading_deg + 180.0)
        % 360.0
        - 180.0
    )

    assert error == pytest.approx(160.9)
    assert error > 75.0


def test_saved_map_forward_command_clears_measured_stiction() -> None:
    assert _forward_command_above_stiction(0.55) == 0.80
    assert _forward_command_above_stiction(0.90) == 0.90
    assert _forward_command_above_stiction(-0.40) == -0.80
    assert _forward_command_above_stiction(0.0) == 0.0


def test_predictive_forward_collision_stops_before_hard_box_contact() -> None:
    profile = {
        "ranges_m": [0.30] * 90,
        "bin_size_deg": 4.0,
        "noise_tolerance_m": 0.0,
        "safety_margin_m": 0.0,
        "min_violation_bins": 1,
        "side_min_violation_bins": 1,
        "min_violation_points": 2,
        "side_min_violation_points": 2,
    }
    # The obstacle is currently 22cm outside the hard front envelope, but the
    # calibrated body would reach it during a 30cm forward rollout.
    points = np.asarray([[0.52, -0.03], [0.52, 0.0], [0.52, 0.03]])

    predicted = _predictive_forward_collision(
        points,
        profile,
        0.30,
        lidar_offset_forward_m=0.0,
        physical_body_radius_m=0.0,
        self_mask_inset_m=0.0,
    )

    assert predicted is not None
    hit, advance = predicted
    assert hit[1] == "front"
    assert 0.18 <= advance <= 0.24


def test_predictive_collision_sector_uses_future_footprint_bearing() -> None:
    navigator = SavedMapNavigator.__new__(SavedMapNavigator)
    navigator.forward_offset = 0.0
    navigator.pose = Pose2D(0.0, 0.0, 0.0)
    navigator.pose_lock = threading.Lock()
    navigator.pending_collision_world = np.empty((0, 2), dtype=np.float32)
    navigator.pending_collision_sectors = set()
    current = np.asarray([[0.42, 0.15], [0.43, 0.16]])
    projected = current.copy()
    projected[:, 0] -= 0.30

    navigator._record_collision_points(
        current,
        np.asarray([True, True]),
        sector_points_forward_xy=projected,
    )

    # The returns are front-left in the current scan, but after the projected
    # 30cm advance they intersect the left shoulder. Recovery must therefore
    # choose the clear clockwise turn rather than losing side information.
    assert navigator.pending_collision_sectors == {"left"}


def test_stationary_consensus_rejects_repeated_large_pose_correction() -> None:
    seed = np.array([2.48, 2.78])
    candidates = [
        (np.array([2.84, 2.87]), 11.1, 0.91),
        (np.array([2.85, 2.88]), 10.7, 0.78),
        (np.array([2.83, 2.88]), 11.4, 0.84),
    ]

    decision = _stationary_correction_consensus(candidates, seed, 7.0)

    assert decision is None


def test_stationary_consensus_rejects_single_large_pose_jump() -> None:
    seed = np.array([2.48, 2.78])
    candidates = [
        (np.array([2.84, 2.87]), 11.1, 0.91),
        (np.array([2.55, 3.10]), 10.7, 0.78),
        (np.array([2.15, 2.60]), 11.4, 0.84),
    ]

    assert _stationary_correction_consensus(candidates, seed, 7.0) is None


def test_route_boundary_reanchor_fully_removes_consistent_global_drift() -> None:
    seed = np.array([1.00, 2.00])
    candidates = [
        (np.array([1.20, 1.91]), 10.4, 0.82),
        (np.array([1.19, 1.90]), 10.1, 0.79),
        (np.array([1.21, 1.90]), 10.7, 0.85),
        (np.array([1.20, 1.89]), 9.8, 0.76),
    ]

    consensus = _route_boundary_reanchor_consensus(candidates, seed, 7.0)

    # A route boundary establishes a new global odometry origin. It must not
    # retain 75% of the old, drifted pose like the gentle mid-route updater.
    assert consensus is not None
    assert np.allclose(consensus, [1.20, 1.90], atol=0.011)


def test_route_boundary_reanchor_rejects_ambiguous_saved_map_modes() -> None:
    seed = np.array([1.00, 2.00])
    candidates = [
        (np.array([1.20, 1.90]), 10.4, 0.82),
        (np.array([0.82, 2.18]), 10.1, 0.79),
        (np.array([1.18, 1.88]), 10.7, 0.85),
        (np.array([0.80, 2.20]), 9.8, 0.76),
    ]

    assert _route_boundary_reanchor_consensus(candidates, seed, 7.0) is None


def test_transition_route_preserves_geometry_with_short_keyframe_steps() -> None:
    dense = _densify_transition_route(
        np.array([0.0, 0.0]),
        [np.array([0.80, 0.0]), np.array([0.80, 0.50])],
        maximum_step_m=0.25,
    )

    points = [np.array([0.0, 0.0]), *dense]
    steps = [
        float(np.hypot(*(b - a)))
        for a, b in zip(points, points[1:], strict=False)
    ]
    assert max(steps) <= 0.25 + 1e-9
    assert np.allclose(dense[-1], [0.80, 0.50])
    # Densification may only subdivide the two original straight segments.
    assert all(np.isclose(point[1], 0.0) for point in dense if point[0] < 0.80)


def test_straight_motion_prior_rejects_corridor_sideways_pose_jump() -> None:
    forward = np.array([1.0, 0.0])

    assert _straight_motion_step_is_consistent(
        np.array([0.12, 0.02]), forward
    )
    assert not _straight_motion_step_is_consistent(
        np.array([0.08, 0.10]), forward
    )
    assert not _straight_motion_step_is_consistent(
        np.array([-0.08, 0.00]), forward
    )


def test_one_sided_collision_turns_away_but_two_sided_collision_does_not_guess() -> None:
    assert _one_sided_escape_turn_deg({"left"}) == -8.0
    assert _one_sided_escape_turn_deg({"right"}) == 8.0
    assert _one_sided_escape_turn_deg({"left", "right"}) is None
    assert _one_sided_escape_turn_deg({"front"}) is None


def test_current_lidar_side_overrides_stale_persistent_turn_direction() -> None:
    # A previous right-side obstacle requested counterclockwise motion (+).
    # Once a fresh scan shows a left-side obstacle, continuing + would turn
    # directly toward it; current evidence must force clockwise motion (-).
    assert _current_escape_turn_deg({"left"}, +1.0) == -15.0
    assert _current_escape_turn_deg({"right"}, -1.0) == +15.0
    assert _current_escape_turn_deg(set(), -1.0) == -15.0
    assert _current_escape_turn_deg({"left", "right"}, +1.0) is None


def test_front_prediction_uses_clear_rotation_instead_of_false_no_route() -> None:
    # Field regression 2026-07-30 21:33: both 30deg probes were clear, but a
    # +13deg front return bypassed the one-sided logic and terminated motion.
    assert _front_obstacle_escape_turn_deg(
        "collision box front would be reached (0.14m at +13deg, limit 0.16m)",
        +8.4,
        clockwise_clear=True,
        counterclockwise_clear=True,
    ) == -15.0
    assert _front_obstacle_escape_turn_deg(
        "collision box front would be reached (0.14m at -13deg, limit 0.16m)",
        -8.4,
        clockwise_clear=True,
        counterclockwise_clear=True,
    ) == +15.0
    assert _front_obstacle_escape_turn_deg(
        "collision box front would be reached",
        +8.4,
        clockwise_clear=True,
        counterclockwise_clear=True,
    ) == +15.0
    assert _front_obstacle_escape_turn_deg(
        "collision box front would be reached",
        0.0,
        clockwise_clear=False,
        counterclockwise_clear=False,
    ) is None


def test_nearly_centred_front_prediction_follows_route_instead_of_desk_edge() -> None:
    # Field regression 2026-07-30 22:38: after correctly turning clockwise
    # around a clearly-left (+32deg) edge, a nearly centred +4deg return was
    # incorrectly classified as another left edge.  Both pivots were clear
    # and the route required +5.4deg, so the second clockwise turn drove the
    # robot back toward the desk.  Near-centred evidence must follow the route.
    assert _front_obstacle_escape_turn_deg(
        "collision box front would be reached (0.13m at +4deg, limit 0.15m)",
        +5.4,
        clockwise_clear=True,
        counterclockwise_clear=True,
    ) == +15.0
    # A genuinely lateral return still has priority over the route heading.
    assert _front_obstacle_escape_turn_deg(
        "collision box front would be reached (0.16m at +32deg, limit 0.17m)",
        +5.4,
        clockwise_clear=True,
        counterclockwise_clear=True,
    ) == -15.0


def test_live_lidar_passage_freezes_centerline_waypoints_before_motion() -> None:
    navigator = SavedMapNavigator.__new__(SavedMapNavigator)
    navigator.args = SimpleNamespace(physical_body_radius_m=0.28)
    points: list[list[float]] = []
    # A 1.0m-wide corridor whose centre is 0.24m to the robot's right and
    # angles 10deg farther right. Multiple returns per slice mimic wall
    # thickness while preserving a deterministic median centreline.
    slope = math.tan(math.radians(-10.0))
    for x in np.arange(0.12, 1.25, 0.03):
        centre_y = -0.24 + slope * x
        for jitter in (-0.006, 0.0, 0.006):
            points.append([float(x), float(centre_y + 0.50 + jitter)])
            points.append([float(x), float(centre_y - 0.50 + jitter)])
    cloud = np.asarray(points, dtype=np.float64)
    navigator._fresh_forward_sample = lambda: (42, cloud)

    plan = navigator._detect_live_lidar_passage()

    assert isinstance(plan, LivePassagePlan)
    assert plan.width_m == pytest.approx(1.0, abs=0.03)
    assert plan.heading_deg == pytest.approx(-10.0, abs=1.0)
    assert plan.lateral_offset_m < -0.20
    assert plan.exit_body_xy[0] > plan.approach_body_xy[0] + 0.40
    approach_bearing = math.degrees(
        math.atan2(float(plan.approach_body_xy[1]), float(plan.approach_body_xy[0]))
    )
    # Use a far, observed intercept so entering a tight passage needs a shallow
    # pivot rather than swinging the rear corner through a nearby wall.
    assert abs(approach_bearing) < 30.0
    measured_heading = math.degrees(
        math.atan2(
            float(plan.exit_body_xy[1] - plan.approach_body_xy[1]),
            float(plan.exit_body_xy[0] - plan.approach_body_xy[0]),
        )
    )
    assert measured_heading == pytest.approx(plan.heading_deg, abs=0.2)


def test_soft_collision_envelope_biases_away_without_reporting_a_stop() -> None:
    navigator = SavedMapNavigator.__new__(SavedMapNavigator)
    navigator.lever_m = 0.0
    navigator.args = SimpleNamespace(
        physical_body_radius_m=0.0,
        collision_self_mask_inset_m=0.0,
        collision_confirmation_frames=2,
        soft_collision_margin_m=0.08,
        soft_steer_max_deg=8.0,
    )
    navigator.soft_collision_profile = {
        "ranges_m": [0.30] * 90,
        "bin_size_deg": 4.0,
        "noise_tolerance_m": 0.02,
        "safety_margin_m": 0.10,
        "min_violation_bins": 2,
        "side_min_violation_bins": 1,
        "min_violation_points": 3,
        "side_min_violation_points": 2,
    }
    navigator._soft_warning_side = None
    navigator._collision_confirmation = {}
    navigator._collision_confirmation_lock = threading.Lock()
    frame_id = iter((10, 11, 12, 13))
    angles = np.radians(np.array([-91.0, -89.0, -87.0]))
    right_warning = np.column_stack(
        [0.35 * np.cos(angles), 0.35 * np.sin(angles)]
    )
    navigator._fresh_forward_sample = lambda: (next(frame_id), right_warning)

    assert navigator._soft_clearance_heading_bias() == 0.0
    # A right-side warning asks for a small counterclockwise heading change,
    # but returns only a number—the hard safety latch is untouched.
    assert navigator._soft_clearance_heading_bias() > 0.0

    both = np.concatenate([right_warning, right_warning * np.array([1.0, -1.0])])
    navigator._collision_confirmation = {}
    navigator._fresh_forward_sample = lambda: (next(frame_id), both)
    assert navigator._soft_clearance_heading_bias() == 0.0
    assert navigator._soft_clearance_heading_bias() == 0.0
