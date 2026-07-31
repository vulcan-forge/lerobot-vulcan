"""Load a saved Sourccey SLAM map, localize, and click a goal to navigate.

Run from the repository root::

    uv run python scripts/sourccey_saved_map_navigator.py \
      --remote-ip 192.168.1.237

The application never enables a goal until a multi-scan LiDAR localization has
been accepted against the saved GOLD reference.  Mouse clicks are snapped to
known collision-inflated free space and followed as pivot/straight segments.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import re
import signal
import sys
import threading
import time
import tkinter as tk
import traceback
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from ldlidar_auto_snapshot_stitch import _send_stop
from ldlidar_direct_snapshot_client import DirectLidarFeed
from ldlidar_direct_snapshot_stitch import (
    _search_pose,
    configure_candidate_scoring_device,
)
from PIL import ImageGrab
from sourccey_collision_box import (
    DEFAULT_COLLISION_BOX_PATH,
    collision_box_rotation_violation,
    collision_box_violation,
    effective_ranges,
    load_collision_box,
)
from sourccey_explore import (
    BaseController,
    MapScan,
    Pose2D,
    RollingLocalSubmap,
    WorldMap,
    _endpoint_support_ratio,
    _filter_isolated_lidar_specks,
    _imu_yaw_for_scan,
    _inflate,
    _latched_arm_torque_state,
    _lidar_pose_from_robot_centre,
    _line_free,
    _localize,
    _localize_against_points,
    _plan_waypoints,
    _pose_with_imu_heading,
    _robot_centre_from_lidar_pose,
    _rotate_lidar_pose_about_robot_centre,
    _scan_local,
    _startup_escape_velocity,
    _to_forward_frame,
    _transform_points,
    _translation_escape_is_safe,
    _translation_trajectory_is_safe,
    analyze_grid,
)
from sourccey_saved_map import DEFAULT_SAVED_MAP_PATH, SavedMap, load_saved_map
from sourccey_wander.imu_heading import ImuYawClient

from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient


@dataclass(frozen=True)
class LivePassagePlan:
    """A corridor centreline measured in the robot's current body frame."""

    width_m: float
    lateral_offset_m: float
    heading_deg: float
    approach_body_xy: np.ndarray
    exit_body_xy: np.ndarray


@dataclass(frozen=True)
class CollisionDiagnostic:
    """Frozen robot-frame evidence for the most recent confirmed safety stop."""

    kind: str
    frame_id: int
    points_forward_xy: np.ndarray
    hit_mask: np.ndarray
    reason: str
    pose: Pose2D
    robot_centre_xy: np.ndarray
    imu_heading_deg: float | None
    route_heading_deg: float | None
    route_heading_error_deg: float | None
    requested_rotation_deg: float | None
    clockwise_result: str
    counterclockwise_result: str
    captured_at: float


def _stationary_correction_consensus(
    candidates: list[tuple[np.ndarray, float, float]],
    seed_centre: np.ndarray,
    minimum_score: float,
) -> tuple[np.ndarray, str] | None:
    """Select a safe saved-map correction from stationary scan matches.

    Small backend corrections retain the original gently blended behaviour.
    This routine is the local back-end correction used during navigation, not
    global relocalization.  It may gently remove accumulated drift, but it may
    never teleport the robot between repeated-looking doorway/corridor modes.
    Large innovations remain the responsibility of explicit global
    relocalization, where distinct hypotheses and a complete angular baseline
    are evaluated.
    """
    seed = np.asarray(seed_centre, dtype=np.float64)
    usable = [
        (np.asarray(centre, dtype=np.float64), float(score), float(support))
        for centre, score, support in candidates
        if float(score) >= float(minimum_score) and float(support) >= 0.15
    ]
    if not usable:
        return None

    nearest = min(usable, key=lambda item: float(np.hypot(*(item[0] - seed))))
    nearest_shift = float(np.hypot(*(nearest[0] - seed)))
    if nearest_shift <= 0.25:
        return 0.75 * seed + 0.25 * nearest[0], "blended"

    # Even three mutually consistent scans can agree on the same wrong mode
    # in a repetitive corridor. Field runs produced 30.4cm and 50.1cm jumps
    # here, after which an otherwise valid route was projected through the
    # inside corner. Never accept a large innovation in this local updater.
    return None


def _route_boundary_reanchor_consensus(
    candidates: list[tuple[np.ndarray, float, float]],
    seed_centre: np.ndarray,
    minimum_score: float,
) -> np.ndarray | None:
    """Return one fully global pose for a new navigation command.

    Unlike the gentle mid-route backend correction above, this runs only while
    the base is stopped between commands.  Several fresh scans must agree with
    the immutable saved map in both position and endpoint support.  Once they
    do, their global consensus becomes the new odometry origin instead of
    allowing the previous trip's rolling-submap drift to validate itself.
    """
    seed = np.asarray(seed_centre, dtype=np.float64)
    usable = [
        (np.asarray(centre, dtype=np.float64), float(score), float(support))
        for centre, score, support in candidates
        if float(score) >= float(minimum_score)
        and float(support) >= 0.35
        and float(np.hypot(*(np.asarray(centre, dtype=np.float64) - seed))) <= 0.35
    ]
    if len(usable) < 3:
        return None
    centres = np.asarray([item[0] for item in usable], dtype=np.float64)
    consensus = np.median(centres, axis=0)
    scatter = np.hypot(*(centres - consensus).T)
    if float(np.percentile(scatter, 90.0)) > 0.06:
        return None
    return consensus


def _densify_transition_route(
    start: np.ndarray,
    route: list[np.ndarray],
    maximum_step_m: float = 0.25,
) -> list[np.ndarray]:
    """Insert stationary-keyframe stops along a doorway/new-room route.

    The geometry of the committed route is unchanged.  Extra points merely
    prevent the follower from travelling so far that its 180-degree view loses
    all overlap with the doorframe or wall that anchored the preceding scan.
    """
    dense: list[np.ndarray] = []
    previous = np.asarray(start, dtype=np.float64)
    maximum_step = max(0.10, float(maximum_step_m))
    for endpoint_value in route:
        endpoint = np.asarray(endpoint_value, dtype=np.float64)
        delta = endpoint - previous
        distance = float(np.hypot(*delta))
        pieces = max(1, int(math.ceil(distance / maximum_step)))
        for piece in range(1, pieces + 1):
            candidate = previous + delta * (piece / pieces)
            if not dense or float(np.hypot(*(candidate - dense[-1]))) > 0.02:
                dense.append(candidate.copy())
        previous = endpoint
    return dense


def _restore_world_map(saved: SavedMap) -> WorldMap:
    metadata = saved.metadata
    world = WorldMap(
        grid_res_m=float(metadata["grid_resolution_m"]),
        footprint_clear_m=float(metadata["footprint_clear_m"]),
        lidar_offset_m=float(metadata["lidar_offset_m"]),
        forward_offset_deg=float(metadata["forward_offset_deg"]),
    )
    world.grid.L = saved.occupancy_log_odds.copy()
    world.grid.origin = saved.occupancy_origin_xy.copy()
    world.grid.version = 1
    for index, pose_values in enumerate(saved.scan_poses):
        local = saved.local_scan(index).copy()
        pose = Pose2D(*map(float, pose_values))
        world.scans.append(
            MapScan(
                local_xy=local,
                pose=pose,
                world_xy=_transform_points(local, pose),
                gold=bool(saved.scan_gold[index]),
            )
        )
    world._ref_cache = None
    return world


def _circular_mean_deg(values: list[float]) -> float:
    radians = np.radians(np.asarray(values, dtype=np.float64))
    return math.degrees(math.atan2(float(np.sin(radians).mean()), float(np.cos(radians).mean())))


def _pose_consensus(candidates: list[tuple[Pose2D, float, float]]) -> Pose2D | None:
    """Require two mutually consistent, supported saved-map hypotheses."""
    if len(candidates) < 2:
        return None
    best_group: list[tuple[Pose2D, float, float]] = []
    for anchor in candidates:
        group = []
        for candidate in candidates:
            position = math.hypot(
                candidate[0].x - anchor[0].x,
                candidate[0].y - anchor[0].y,
            )
            heading = abs((candidate[0].theta_deg - anchor[0].theta_deg + 180.0) % 360.0 - 180.0)
            if position <= 0.12 and heading <= 5.0:
                group.append(candidate)
        if len(group) > len(best_group):
            best_group = group
    if len(best_group) < 2:
        return None
    weights = np.asarray([max(1.0, item[1]) for item in best_group], dtype=np.float64)
    positions = np.asarray([[item[0].x, item[0].y] for item in best_group])
    position = np.average(positions, axis=0, weights=weights)
    theta = _circular_mean_deg([item[0].theta_deg for item in best_group])
    return Pose2D(float(position[0]), float(position[1]), float(theta))


def _global_localization_accepted(
    score: float,
    support: float,
    mode_margin: float,
    minimum_score: float,
) -> bool:
    """Joint confidence test for an active global-localization hypothesis.

    Matcher score is not a calibrated probability and falls as a 360-degree
    panorama includes more grazing/noisy endpoints. Strong endpoint support
    plus a clearly separated best mode is better evidence than whether an
    internal float landed one hundredth above or below a printed threshold.
    """
    ordinary = (
        float(score) >= float(minimum_score) - 0.25
        and float(support) >= 0.35
        and float(mode_margin) >= 0.35
    )
    strongly_disambiguated = (
        float(score) >= max(5.0, float(minimum_score) - 2.0)
        and float(support) >= 0.70
        and float(mode_margin) >= 0.75
    )
    return ordinary or strongly_disambiguated


def _assemble_active_localization_cloud(
    captures: list[tuple[np.ndarray, float]],
    lidar_offset_m: float,
    forward_offset_deg: float,
) -> np.ndarray:
    """Rigidly combine angular views in the first-view LiDAR frame.

    ``captures`` contains local scans and measured IMU yaw deltas from the
    first view. The off-centre LiDAR follows an arc during a chassis pivot, so
    rotating points around the sensor origin is incorrect; generate each
    sensor pose by rotating it around the robot centre instead.
    """
    if not captures:
        return np.empty((0, 2), dtype=np.float32)
    initial = _lidar_pose_from_robot_centre(
        np.zeros(2, dtype=np.float64),
        0.0,
        float(lidar_offset_m),
        float(forward_offset_deg),
    )
    transformed: list[np.ndarray] = []
    for local, yaw_delta in captures:
        points = np.asarray(local, dtype=np.float32)
        if not len(points):
            continue
        relative_pose = _rotate_lidar_pose_about_robot_centre(
            initial,
            float(yaw_delta),
            float(lidar_offset_m),
            float(forward_offset_deg),
        )
        transformed.append(_transform_points(points, relative_pose))
    if not transformed:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(transformed, axis=0).astype(np.float32, copy=False)


def _global_localization_position_seeds(saved: SavedMap, spacing_m: float = 0.15) -> np.ndarray:
    """Cover known-free space with coarse robot-position hypotheses.

    Endpoint-only Hough peaks are unreliable in rooms containing repeated
    parallel walls. A professional global localizer searches the traversable
    state space. Occupancy free cells cover places the robot has observed but
    may not have driven through; saved scan centres and the executed trail are
    included so previously occupied poses are represented exactly.
    """
    resolution = float(saved.metadata["grid_resolution_m"])
    origin = np.asarray(saved.occupancy_origin_xy, dtype=np.float64)
    free_ii, free_jj = np.nonzero(saved.occupancy_log_odds <= -0.8)
    free_xy = np.column_stack(
        [
            origin[0] + (free_jj.astype(np.float64) + 0.5) * resolution,
            origin[1] + (free_ii.astype(np.float64) + 0.5) * resolution,
        ]
    )

    lever_m = float(saved.metadata["lidar_offset_m"])
    forward_offset_deg = float(saved.metadata["forward_offset_deg"])
    scan_centres = np.asarray(
        [
            _robot_centre_from_lidar_pose(
                Pose2D(*map(float, pose)), lever_m, forward_offset_deg
            )
            for pose in saved.scan_poses
        ],
        dtype=np.float64,
    ).reshape((-1, 2))
    trail = np.asarray(saved.trail_xy, dtype=np.float64).reshape((-1, 2))
    sources = [points for points in (free_xy, scan_centres, trail) if len(points)]
    if not sources:
        return np.empty((0, 2), dtype=np.float32)
    candidates = np.concatenate(sources, axis=0)

    spacing = max(resolution, float(spacing_m))
    quantized = np.round(candidates / spacing).astype(np.int64)
    _keys, first = np.unique(quantized, axis=0, return_index=True)
    return candidates[np.sort(first)].astype(np.float32, copy=False)


def _occupancy_pose_consistency(
    saved: SavedMap,
    captures: list[tuple[np.ndarray, float]],
    robot_pose: Pose2D,
    lidar_offset_m: float,
    forward_offset_deg: float,
) -> tuple[float, float, float, float]:
    """Score endpoints and free beam paths against saved occupancy.

    Point-cloud correlation alone cannot distinguish two similar parallel
    walls. Occupancy-grid localization also checks the negative information in
    every LiDAR ray: its endpoint should be occupied and its interior should
    not cross occupied cells. Returns ``(quality, endpoint_occupied,
    endpoint_free, ray_occupied)``.
    """
    grid = np.asarray(saved.occupancy_log_odds)
    origin = np.asarray(saved.occupancy_origin_xy, dtype=np.float64)
    resolution = float(saved.metadata["grid_resolution_m"])
    initial_sensor = _lidar_pose_from_robot_centre(
        np.zeros(2, dtype=np.float64),
        0.0,
        float(lidar_offset_m),
        float(forward_offset_deg),
    )
    endpoint_values: list[np.ndarray] = []
    ray_values: list[np.ndarray] = []

    def values_at(points: np.ndarray) -> np.ndarray:
        jj = ((points[:, 0] - origin[0]) / resolution).astype(np.int64)
        ii = ((points[:, 1] - origin[1]) / resolution).astype(np.int64)
        inside = (
            (ii >= 0)
            & (ii < grid.shape[0])
            & (jj >= 0)
            & (jj < grid.shape[1])
        )
        return grid[ii[inside], jj[inside]]

    for local, yaw_delta in captures:
        points = np.asarray(local, dtype=np.float32)
        if not len(points):
            continue
        relative_sensor = _rotate_lidar_pose_about_robot_centre(
            initial_sensor,
            float(yaw_delta),
            float(lidar_offset_m),
            float(forward_offset_deg),
        )
        points_centre = _transform_points(points, relative_sensor)
        points_world = _transform_points(points_centre, robot_pose)
        sensor_world = _transform_points(
            np.asarray([[relative_sensor.x, relative_sensor.y]], dtype=np.float32),
            robot_pose,
        )[0]
        indices = np.linspace(
            0,
            len(points_world) - 1,
            min(80, len(points_world)),
            dtype=np.int64,
        )
        endpoints = points_world[indices]
        endpoint_values.append(values_at(endpoints))
        fractions = np.linspace(0.08, 0.90, 9, dtype=np.float32)
        ray_samples = sensor_world[None, None, :] + (
            endpoints[:, None, :] - sensor_world[None, None, :]
        ) * fractions[None, :, None]
        ray_values.append(values_at(ray_samples.reshape((-1, 2))))

    endpoints = (
        np.concatenate(endpoint_values) if endpoint_values else np.empty(0)
    )
    rays = np.concatenate(ray_values) if ray_values else np.empty(0)
    endpoint_occupied = float(np.mean(endpoints >= 1.2)) if len(endpoints) else 0.0
    endpoint_free = float(np.mean(endpoints <= -0.8)) if len(endpoints) else 1.0
    known_rays = rays[np.abs(rays) >= 0.5]
    ray_occupied = (
        float(np.mean(known_rays >= 1.2)) if len(known_rays) else 1.0
    )
    quality = (
        4.0 * endpoint_occupied
        - 3.0 * endpoint_free
        - 6.0 * ray_occupied
    )
    return quality, endpoint_occupied, endpoint_free, ray_occupied


def _plan_saved_map_route(
    world: WorldMap,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    conservative_radius_m: float,
    physical_radius_m: float,
    extra_occupied_xy: np.ndarray | None = None,
) -> tuple[list[np.ndarray] | None, float]:
    """Plan conservatively, then retry with the calibrated physical radius.

    A circular global costmap is deliberately more conservative than the live
    direction-aware collision envelope. If that extra margin alone disconnects
    two known-free regions, retry at the measured physical radius; execution
    remains protected by the calibrated collision box at control-loop rate.
    """
    radii = [float(conservative_radius_m)]
    physical = float(physical_radius_m)
    if 0.0 < physical < radii[0] - 1e-6:
        radii.append(physical)
    for radius in radii:
        analysis = analyze_grid(
            world.grid,
            robot_radius_m=radius,
            min_frontier_span_m=0.4,
            min_frontier_cells=6,
            extra_occupied_xy=extra_occupied_xy,
        )
        # Saved-map navigation knows the complete static geometry, so add a
        # Nav2-style inflation cost outside the hard footprint.  This is a
        # preference, not another wall: an earlier 35x/0.90m penalty routinely
        # sent A* on large clearance spikes even when a simple, safe corridor
        # route existed.
        blocked = ~analysis.traversable
        clearance_cost = np.ones_like(analysis.cost, dtype=np.float32)
        rings = max(1, int(round(0.50 / analysis.res_m)))
        grown = blocked.copy()
        for ring in range(rings):
            next_grown = _inflate(grown, 1)
            band = next_grown & ~grown & analysis.traversable
            normalized = float(rings - ring) / float(rings)
            # Front shoulders are the widest part of Sourccey and can catch
            # an isolated door/table edge even when the circular hard grid is
            # technically traversable. Keep this a cost (not a new wall), but
            # make the first few clearance rings expensive enough that A*
            # chooses the corridor centre whenever one exists.
            clearance_cost[band] += 8.0 * normalized * normalized
            grown = next_grown
        # Add, rather than replace, the clearance field so the original map
        # costs and the stronger saved-map wall penalty both influence A*.
        analysis.cost = np.asarray(analysis.cost, dtype=np.float32).copy()
        analysis.cost[analysis.traversable] += (
            clearance_cost[analysis.traversable] - 1.0
        )
        route = _plan_waypoints(analysis, start_xy, goal_xy)
        if route:
            return _simplify_saved_map_route(
                analysis,
                np.asarray(start_xy, dtype=np.float64),
                route,
            ), radius
    return None, radii[-1]


def _simplify_saved_map_route(
    analysis,
    start_xy: np.ndarray,
    route: list[np.ndarray],
    *,
    maximum_corner_deviation_m: float = 0.22,
) -> list[np.ndarray]:
    """Remove small A* clearance spikes without cutting a real corner.

    The hard, inflated traversability grid remains authoritative.  A group of
    waypoints is replaced by one straight segment only when the whole chord is
    collision-free and the discarded contour stays close to that chord.  The
    result is the requested pivot/straight/pivot route while a meaningful move
    to a corridor centreline is retained.
    """
    points = [
        np.asarray(start_xy, dtype=np.float64),
        *[np.asarray(point, dtype=np.float64) for point in route],
    ]
    if len(points) <= 2:
        return [point.copy() for point in points[1:]]

    cells = [analysis.to_cell(point) for point in points]

    def sampled_cost(first: int, last: int, *, chord: bool) -> float:
        """Return the mean inflated cost along a chord or original contour."""
        if not hasattr(analysis, "cost"):
            return 1.0
        pairs = (
            [(cells[first], cells[last])]
            if chord
            else list(zip(cells[first:last], cells[first + 1 : last + 1], strict=False))
        )
        values: list[float] = []
        for start_cell, end_cell in pairs:
            dr = int(end_cell[0]) - int(start_cell[0])
            dc = int(end_cell[1]) - int(start_cell[1])
            count = max(abs(dr), abs(dc), 1) + 1
            rows = np.rint(np.linspace(start_cell[0], end_cell[0], count)).astype(int)
            cols = np.rint(np.linspace(start_cell[1], end_cell[1], count)).astype(int)
            valid = (
                (rows >= 0)
                & (rows < analysis.cost.shape[0])
                & (cols >= 0)
                & (cols < analysis.cost.shape[1])
            )
            if np.any(valid):
                values.extend(
                    np.asarray(analysis.cost[rows[valid], cols[valid]], dtype=float).tolist()
                )
        # A mean hides a short, dangerous tangent past an inside corner among
        # many low-cost cells. The upper-quartile cost preserves the waypoint
        # that moves the shoulders clear of that corner without reacting to a
        # single noisy grid cell like a strict maximum would.
        return float(np.percentile(values, 75.0)) if values else math.inf

    def contour_deviation(first: int, last: int) -> float:
        a = points[first]
        b = points[last]
        ab = b - a
        length = float(np.hypot(*ab))
        if length < 1e-9:
            return 0.0
        middle = np.asarray(points[first + 1 : last], dtype=np.float64)
        if not len(middle):
            return 0.0
        relative = middle - a
        return float(
            np.max(np.abs(relative[:, 0] * ab[1] - relative[:, 1] * ab[0]) / length)
        )

    kept = [0]
    first = 0
    while first < len(points) - 1:
        farthest = first + 1
        for candidate in range(len(points) - 1, first, -1):
            chord_cost = sampled_cost(first, candidate, chord=True)
            contour_cost = sampled_cost(first, candidate, chord=False)
            if (
                _line_free(analysis.traversable, cells[first], cells[candidate])
                and contour_deviation(first, candidate)
                <= float(maximum_corner_deviation_m)
                # Collision-free is not the same as comfortably clear. The
                # previous simplifier erased A*'s corner-clearance waypoint
                # and replaced it with a 2.07m diagonal tangent to the inside
                # corner. Preserve the contour whenever its chord materially
                # increases the inflated wall cost.
                and chord_cost <= contour_cost + 0.20
            ):
                farthest = candidate
                break
        kept.append(farthest)
        first = farthest
    return [points[index].copy() for index in kept[1:]]


def _drop_reached_waypoint_prefix(
    route: list[np.ndarray],
    robot_centre_xy: np.ndarray,
    reached_radius_m: float = 0.16,
) -> tuple[list[np.ndarray], int]:
    """Remove route-prefix points already reached by the robot centre.

    A* and collision replans commonly emit one or two grid-connector points
    only a few centimetres from the current pose.  They describe the start of
    the polyline; they are not headings the chassis must face.  The follower
    previously calculated and executed their heading *before* its 16cm
    arrival check.  A connector just behind the robot could therefore command
    a meaningless 180-degree turn into the obstacle that triggered the
    replan.  Waypoint reachability is now an invariant at the route boundary.
    """
    centre = np.asarray(robot_centre_xy, dtype=np.float64).reshape(2)
    first_actionable = 0
    for point in route:
        candidate = np.asarray(point, dtype=np.float64).reshape(2)
        if float(np.hypot(*(candidate - centre))) > float(reached_radius_m):
            break
        first_actionable += 1
    return [
        np.asarray(point, dtype=np.float64).copy()
        for point in route[first_actionable:]
    ], first_actionable


def _forward_command_above_stiction(
    requested: float,
    minimum_moving_command: float = 0.80,
) -> float:
    """Apply the measured base deadband to a nonzero forward command."""
    value = float(requested)
    if abs(value) < 1e-9:
        return 0.0
    return math.copysign(max(abs(value), float(minimum_moving_command)), value)


def _straight_motion_step_is_consistent(
    step_xy: np.ndarray,
    forward_direction_xy: np.ndarray,
    *,
    maximum_forward_m: float = 0.20,
    maximum_reverse_m: float = 0.03,
    maximum_lateral_m: float = 0.04,
) -> bool:
    """Validate one LiDAR pose update against a straight-drive command.

    IMU heading hold plus a zero lateral command is a strong motion prior. A
    repeated wall may give scan matching a good score at many positions, but
    it cannot make the physical base jump sideways between two fresh scans.
    """
    step = np.asarray(step_xy, dtype=np.float64).reshape(2)
    forward = np.asarray(forward_direction_xy, dtype=np.float64).reshape(2)
    norm = float(np.hypot(*forward))
    if norm < 1e-9:
        return float(np.hypot(*step)) <= float(maximum_forward_m)
    forward /= norm
    lateral = np.asarray([-forward[1], forward[0]], dtype=np.float64)
    along_m = float(step @ forward)
    lateral_m = abs(float(step @ lateral))
    return (
        -float(maximum_reverse_m) <= along_m <= float(maximum_forward_m)
        and lateral_m <= float(maximum_lateral_m)
    )


def _predictive_forward_collision(
    points_forward_xy: np.ndarray,
    profile: dict,
    lookahead_m: float,
    *,
    lidar_offset_forward_m: float,
    physical_body_radius_m: float,
    self_mask_inset_m: float,
) -> tuple[tuple[np.ndarray, str, float, float, float], float] | None:
    """Return the first footprint collision along a forward rollout.

    LiDAR points are stationary obstacles in the current robot frame. Moving
    the robot forward by ``advance`` is equivalent to translating every point
    backward by that amount, after which the normal calibrated-envelope test
    can be reused without inventing a second footprint model.
    """
    points = np.asarray(points_forward_xy, dtype=np.float64).reshape((-1, 2))
    distance = max(0.0, float(lookahead_m))
    if not len(points) or distance <= 0.0:
        return None
    step_count = max(1, int(math.ceil(distance / 0.04)))
    for advance in np.linspace(distance / step_count, distance, step_count):
        projected = points.copy()
        projected[:, 0] -= float(advance)
        hit = collision_box_violation(
            projected,
            profile,
            lidar_offset_forward_m=float(lidar_offset_forward_m),
            physical_body_radius_m=float(physical_body_radius_m),
            self_mask_inset_m=float(self_mask_inset_m),
        )
        if hit is not None:
            return hit, float(advance)
    return None


def _one_sided_escape_turn_deg(sectors: set[str], step_deg: float = 8.0) -> float | None:
    """Turn away from a single side intrusion; never guess if both sides hit."""
    left = "left" in sectors
    right = "right" in sectors
    if left == right:
        return None
    return -abs(float(step_deg)) if left else abs(float(step_deg))


def _current_escape_turn_deg(
    sectors: set[str],
    persistent_turn_sign: float | None,
    step_deg: float = 15.0,
) -> float | None:
    """Select an escape sign without ever turning into current evidence.

    A persistent sign is useful only between scans where no side has yet been
    classified.  As soon as fresh LiDAR identifies one occupied side, that
    measurement is authoritative.  This prevents a previously-right obstacle
    from forcing another counterclockwise turn after the new scan shows the
    obstruction has moved to the left side of the chassis.
    """
    measured = _one_sided_escape_turn_deg(sectors, step_deg=step_deg)
    if measured is not None:
        return measured
    if sectors:
        return None
    if persistent_turn_sign is None or abs(float(persistent_turn_sign)) <= 1e-9:
        return None
    return math.copysign(abs(float(step_deg)), float(persistent_turn_sign))


def _front_obstacle_escape_turn_deg(
    reason: str,
    route_heading_error_deg: float,
    *,
    clockwise_clear: bool,
    counterclockwise_clear: bool,
    step_deg: float = 15.0,
) -> float | None:
    """Choose one bounded pivot for a front prediction.

    Front obstacles used to fall through the one-sided recovery because their
    sector set contains neither ``left`` nor ``right``.  That could report "no
    safe route" even when both rotation probes were clear.  Prefer turning
    away from a clearly lateral measured bearing.  A nearly centred return
    does *not* contain enough side information to choose a turn: in that case
    use the direction that reduces the current route-heading error.  Treating
    a few degrees of LiDAR bearing noise as a left/right obstacle previously
    made consecutive recoveries turn away from the route and into the same
    desk corner.
    """
    if not clockwise_clear and not counterclockwise_clear:
        return None

    bearing_match = re.search(r"at\s+([+-]?\d+(?:\.\d+)?)deg", str(reason))
    bearing_deg = float(bearing_match.group(1)) if bearing_match else 0.0
    preferred_sign = 0.0
    # Bearings inside this deadband are effectively straight ahead once beam
    # width, chassis/LiDAR offset, and scan noise are considered.  Let the
    # already-planned route break that tie rather than inventing a side.
    side_evidence_deadband_deg = 10.0
    if bearing_deg > side_evidence_deadband_deg:
        preferred_sign = -1.0
    elif bearing_deg < -side_evidence_deadband_deg:
        preferred_sign = 1.0
    elif abs(float(route_heading_error_deg)) > 1.0:
        preferred_sign = math.copysign(1.0, float(route_heading_error_deg))

    candidates: list[float] = []
    if clockwise_clear:
        candidates.append(-abs(float(step_deg)))
    if counterclockwise_clear:
        candidates.append(abs(float(step_deg)))
    if preferred_sign:
        preferred = math.copysign(abs(float(step_deg)), preferred_sign)
        if preferred in candidates:
            return preferred
    return min(
        candidates,
        key=lambda delta: abs(float(route_heading_error_deg) - float(delta)),
    )


def _learn_body_fixed_lidar_returns(
    captures: list[tuple[np.ndarray, float]],
    forward_offset_deg: float,
    *,
    bin_size_deg: float = 2.0,
    maximum_range_m: float = 0.36,
) -> np.ndarray:
    """Learn chassis returns that remain fixed in the LiDAR/body frame.

    During an in-place localization sweep, environmental geometry moves
    through sensor bearings while returns from the chassis remain at the same
    bearing and range.  Requiring agreement in most angular views prevents a
    nearby wall from becoming part of the self mask.  The range cap confines
    this model to the robot's immediate physical envelope.

    Returns are ``[bearing_deg, range_m]`` rows in physical-forward space.
    """
    view_count = len(captures)
    if view_count < 4:
        return np.empty((0, 2), dtype=np.float32)
    bin_size = max(0.5, float(bin_size_deg))
    bin_count = int(math.ceil(360.0 / bin_size))
    ranges_by_bin: list[list[float]] = [[] for _ in range(bin_count)]
    for local, _yaw_delta in captures:
        forward = _filter_isolated_lidar_specks(
            _to_forward_frame(np.asarray(local), float(forward_offset_deg))
        )
        if not len(forward):
            continue
        ranges = np.hypot(forward[:, 0], forward[:, 1])
        angles = np.degrees(np.arctan2(forward[:, 1], forward[:, 0]))
        indices = np.floor((angles + 180.0) / bin_size).astype(np.int64) % bin_count
        # One representative per view/bin prevents dense scans from counting
        # as temporal persistence. The nearest return owns collision safety.
        for index in np.unique(indices):
            values = ranges[indices == index]
            finite = values[np.isfinite(values)]
            if len(finite):
                ranges_by_bin[int(index)].append(float(np.min(finite)))

    required_views = max(4, int(math.ceil(0.65 * view_count)))
    templates: list[tuple[float, float]] = []
    for index, values in enumerate(ranges_by_bin):
        if len(values) < required_views:
            continue
        samples = np.asarray(values, dtype=np.float64)
        median = float(np.median(samples))
        spread = float(np.percentile(samples, 90) - np.percentile(samples, 10))
        if 0.03 <= median <= float(maximum_range_m) and spread <= 0.025:
            bearing = -180.0 + (float(index) + 0.5) * bin_size
            templates.append((bearing, median))
    return np.asarray(templates, dtype=np.float32).reshape((-1, 2))


def _remove_body_fixed_lidar_returns(
    points_forward_xy: np.ndarray,
    templates: np.ndarray,
    *,
    bin_size_deg: float = 2.0,
    range_tolerance_m: float = 0.035,
) -> np.ndarray:
    """Remove only returns matching the learned robot-fixed polar signature."""
    points = np.asarray(points_forward_xy)
    learned = np.asarray(templates, dtype=np.float64).reshape((-1, 2))
    if not len(points) or not len(learned):
        return points
    ranges = np.hypot(points[:, 0], points[:, 1])
    angles = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    delta_angle = np.abs(
        (angles[:, None] - learned[None, :, 0] + 180.0) % 360.0 - 180.0
    )
    delta_range = np.abs(ranges[:, None] - learned[None, :, 1])
    body_fixed = np.any(
        (delta_angle <= max(1.0, float(bin_size_deg)))
        & (delta_range <= float(range_tolerance_m)),
        axis=1,
    )
    return points[~body_fixed]


class SavedMapNavigator:
    def __init__(self, root: tk.Tk, args: Namespace) -> None:
        self.root = root
        self.args = args
        self.saved = load_saved_map(args.map)
        for name, value in self.saved.metadata.get("sensor_config", {}).items():
            setattr(self.args, name, value)
        for name, value in self.saved.metadata.get("navigation_config", {}).items():
            setattr(self.args, name, value)
        self.world = _restore_world_map(self.saved)
        self.forward_offset = float(self.saved.metadata["forward_offset_deg"])
        self.lever_m = float(self.saved.metadata["lidar_offset_m"])
        self.pose = Pose2D(*map(float, self.saved.current_pose))
        self.pose_lock = threading.Lock()
        self.localized = False
        self.localization_imu: float | None = None
        self.navigation_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.closing = False
        self.path: list[np.ndarray] = []
        self.goal: np.ndarray | None = None
        self.status_text = "Connecting to robot sensors..."
        self._map_canvas_size = (0, 0)
        self.localization_attempt = 0
        self.dynamic_obstacles_world = np.empty((0, 2), dtype=np.float32)
        self.pending_collision_world = np.empty((0, 2), dtype=np.float32)
        self.pending_collision_sectors: set[str] = set()
        # A newly calculated route is not proof of recovery. Preserve this
        # streak until the chassis actually completes a waypoint so the same
        # blocked first turn cannot recursively replan forever.
        self.collision_replan_streak = 0
        self._collision_confirmation: dict[str, dict[str, object]] = {}
        self._collision_confirmation_lock = threading.Lock()
        self._collision_diagnostic: CollisionDiagnostic | None = None
        self._collision_diagnostic_lock = threading.Lock()
        self._last_collision_diagnostic_key: tuple[str, int] | None = None
        self.body_fixed_lidar_returns = np.empty((0, 2), dtype=np.float32)
        # Navigation uses a conventional SLAM front/back end split.  This
        # rolling submap supplies continuous scan-to-scan LiDAR odometry when
        # the current view contains new doorway/room geometry that is not yet
        # represented in the immutable saved map.  The saved map remains the
        # lower-rate global drift correction; it is no longer a per-frame gate.
        # Retain enough overlapping half-view scans to bridge a doorway turn
        # without consulting the ambiguous whole saved map. This remains a
        # small rolling front-end submap, not permanent occupancy.
        # A 16-frame window covered only a couple of seconds of motion. At a
        # far goal in a sparsely saved room, turning around could therefore
        # discard every feature that anchored the arrival trajectory. Retain
        # several seconds of overlapping scans so the return command starts
        # from the same locally built geometry even when the immutable saved
        # map has little structure in that room.
        self.local_odometry = RollingLocalSubmap(max_scans=48, max_points=24000)
        self.global_tracking_cycle = 0
        self._strong_local_motion_relaxations = 0
        self._endpoint_recheck_count = 0
        self._last_passage_commit_centre: np.ndarray | None = None
        # A LiDAR-measured centre/align/straight passage route remains
        # authoritative through shoulder warnings. Recovery must refresh that
        # manoeuvre rather than immediately replacing it with generic turns.
        self._passage_route_active = False

        # Record exactly the robot-frame evidence panel the operator sees.
        # This is intentionally independent of the saved-map pose, making a
        # post-run video useful even when global localization is the failure
        # under investigation.
        self._diagnostic_video_writer: cv2.VideoWriter | None = None
        self._diagnostic_video_path: Path | None = None
        self._diagnostic_video_last_frame_at = 0.0
        self._diagnostic_video_failed = False
        self._run_started_monotonic = time.monotonic()

        self.feed = DirectLidarFeed(args.remote_ip, int(args.lidar_port))
        self.feed.start()
        self.imu = ImuYawClient(
            f"tcp://{args.remote_ip}:{int(args.imu_yaw_port)}",
            sign=float(args.imu_yaw_sign),
        )
        self.imu.start()
        self.robot = SourcceyClient(
            SourcceyClientConfig(id=args.robot_id, remote_ip=args.remote_ip)
        )
        self.robot.connect()
        _send_stop(self.robot)
        self.collision_profile = load_collision_box(args.collision_box_file)
        self.soft_collision_profile = dict(self.collision_profile or {})
        hard_margin = float(
            self.soft_collision_profile.get(
                "safety_margin_m",
                self.soft_collision_profile.get("noise_tolerance_m", 0.02),
            )
        )
        self.soft_collision_profile["safety_margin_m"] = (
            hard_margin + max(0.0, float(args.soft_collision_margin_m))
        )
        self._soft_warning_side: str | None = None
        self._soft_warning_bias = 0.0
        self._soft_warning_last_seen_at = -math.inf
        self.controller = BaseController(
            self.robot,
            _latched_arm_torque_state(stowed=False),
            self.imu,
            rate_hz=25.0,
            turn_speed=float(args.turn_speed),
        )
        self.controller.set_safety_check(self._translation_safety)
        self.controller.set_rotation_safety_check(self._rotation_safety)
        self.controller.start()

        root.title("Sourccey Saved Map Navigator")
        root.protocol("WM_DELETE_WINDOW", self.close)
        shell = ttk.Frame(root, padding=8)
        shell.grid(sticky="nsew")
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)
        shell.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(shell, width=900, height=760, background="#10170f")
        self.canvas.grid(row=0, column=0, columnspan=3, sticky="nsew")
        self.canvas.bind("<Button-1>", self._click_goal)
        self.collision_canvas = tk.Canvas(
            shell,
            width=360,
            height=520,
            background="#0b1114",
            highlightthickness=1,
            highlightbackground="#52616a",
        )
        self.collision_canvas.grid(
            row=0,
            column=3,
            sticky="ne",
            padx=(8, 0),
        )
        ttk.Button(shell, text="Relocalize", command=self.relocalize).grid(
            row=1, column=0, sticky="ew", padx=(0, 4), pady=(8, 0)
        )
        ttk.Button(shell, text="STOP", command=self.stop).grid(
            row=1, column=1, sticky="ew", padx=4, pady=(8, 0)
        )
        self.status = ttk.Label(shell, text=self.status_text)
        self.status.grid(row=1, column=2, sticky="ew", padx=(8, 0), pady=(8, 0))
        ttk.Label(shell, text="Robot-frame collision evidence").grid(
            row=1, column=3, sticky="ew", padx=(8, 0), pady=(8, 0)
        )
        self._compute_view()
        # Tk commonly reports a 1x1 canvas until the first real layout pass.
        # Drawing static geometry here put the entire saved map offscreen while
        # later refreshes drew only the dynamic robot marker. _refresh() owns
        # the first size-valid static draw and every subsequent resize.
        self.root.after(100, self._refresh)
        self.status_text = (
            "Map loaded—waiting for live LiDAR and IMU before automatic "
            "relocalization"
        )
        # Active localization moves the chassis, so start it only after both
        # sensor streams have produced a sample. Polling from Tk keeps startup
        # non-blocking and avoids a false unavailable result while a healthy
        # sensor connection is still warming up.
        self.root.after(250, self._start_initial_relocalization)

    def _record_collision_diagnostic_frame(self) -> None:
        """Append the visible live-LiDAR diagnostic canvas to an MP4 file."""
        if self._diagnostic_video_failed:
            return
        fps = max(1.0, float(self.args.diagnostic_video_fps))
        now = time.monotonic()
        if now - self._diagnostic_video_last_frame_at < 1.0 / fps:
            return
        canvas = self.collision_canvas
        width = int(canvas.winfo_width())
        height = int(canvas.winfo_height())
        if width < 20 or height < 20 or not canvas.winfo_ismapped():
            return
        try:
            left = int(canvas.winfo_rootx())
            top = int(canvas.winfo_rooty())
            image = ImageGrab.grab(
                bbox=(left, top, left + width, top + height),
                all_screens=True,
            )
            frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            if self._diagnostic_video_writer is None:
                destination = Path(self.args.diagnostic_video_dir)
                destination.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                path = destination / f"saved_map_lidar_diagnostic_{stamp}.mp4"
                writer = cv2.VideoWriter(
                    str(path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError("OpenCV could not open the MP4 video writer")
                self._diagnostic_video_writer = writer
                self._diagnostic_video_path = path.resolve()
                print(
                    "[saved-map] recording live LiDAR collision diagnostics to "
                    f"{self._diagnostic_video_path}"
                )
            self._diagnostic_video_writer.write(frame)
            self._diagnostic_video_last_frame_at = now
        except Exception as exc:
            self._diagnostic_video_failed = True
            print(
                "[saved-map] WARNING: collision diagnostic video recording "
                f"disabled after capture failure: {exc}"
            )

    def _close_collision_diagnostic_video(self) -> None:
        writer = self._diagnostic_video_writer
        self._diagnostic_video_writer = None
        if writer is not None:
            writer.release()
            print(
                "[saved-map] saved live LiDAR collision diagnostic video: "
                f"{self._diagnostic_video_path}"
            )

    def _centre(self, pose: Pose2D) -> np.ndarray:
        return _robot_centre_from_lidar_pose(
            pose, self.lever_m, self.forward_offset
        )

    def _start_initial_relocalization(self) -> None:
        """Automatically localize once the first live sensor samples exist."""
        if self.closing or self.localized:
            return
        if self.navigation_thread is not None and self.navigation_thread.is_alive():
            return
        _frame_id, frame = self.feed.latest()
        if frame is None or self.imu.deg() is None:
            self.status_text = (
                "Waiting for live LiDAR and IMU before automatic relocalization..."
            )
            self.status.configure(text=self.status_text)
            self.root.after(250, self._start_initial_relocalization)
            return
        print("[saved-map] live sensors ready; starting automatic relocalization.")
        self.relocalize()

    def _compute_view(self) -> None:
        known = self.world.grid.occupied() | self.world.grid.free()
        cells = np.column_stack(np.nonzero(known))
        if len(cells):
            lo = self.world.grid.origin + np.array(
                [cells[:, 1].min(), cells[:, 0].min()]
            ) * self.world.grid.res
            hi = self.world.grid.origin + np.array(
                [cells[:, 1].max() + 1, cells[:, 0].max() + 1]
            ) * self.world.grid.res
        else:
            lo, hi = np.array([-2.0, -2.0]), np.array([2.0, 2.0])
        margin = 0.5
        self.view_lo = lo - margin
        self.view_hi = hi + margin

    def _screen(self, xy: np.ndarray) -> tuple[float, float]:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        span = np.maximum(0.1, self.view_hi - self.view_lo)
        scale = min((width - 20) / span[0], (height - 20) / span[1])
        x = 10 + (float(xy[0]) - self.view_lo[0]) * scale
        y = height - 10 - (float(xy[1]) - self.view_lo[1]) * scale
        return x, y

    def _world_xy(self, x: float, y: float) -> np.ndarray:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        span = np.maximum(0.1, self.view_hi - self.view_lo)
        scale = min((width - 20) / span[0], (height - 20) / span[1])
        return np.array([
            self.view_lo[0] + (x - 10) / scale,
            self.view_lo[1] + (height - 10 - y) / scale,
        ])

    def _draw_map(self) -> None:
        self.canvas.delete("all")
        grid = self.world.grid
        occupied = grid.occupied()
        free = grid.free() & ~occupied
        known_cells = np.column_stack(np.nonzero(free))
        stride = max(1, len(known_cells) // 10000)
        for i, j in known_cells[::stride]:
            p0 = grid.origin + np.array([j, i]) * grid.res
            p1 = p0 + grid.res
            x0, y1 = self._screen(p0)
            x1, y0 = self._screen(p1)
            self.canvas.create_rectangle(x0, y0, x1, y1, fill="#203523", outline="")
        for i, j in np.column_stack(np.nonzero(occupied)):
            p0 = grid.origin + np.array([j, i]) * grid.res
            p1 = p0 + grid.res
            x0, y1 = self._screen(p0)
            x1, y0 = self._screen(p1)
            self.canvas.create_rectangle(x0, y0, x1, y1, fill="#d9d5c7", outline="")

    def _draw(self) -> None:
        self.canvas.delete("dynamic")
        if self.path:
            coords = [coordinate for point in self.path for coordinate in self._screen(point)]
            if len(coords) >= 4:
                self.canvas.create_line(*coords, fill="#4cff70", width=3, tags="dynamic")
        if self.goal is not None:
            gx, gy = self._screen(self.goal)
            self.canvas.create_oval(
                gx - 7,
                gy - 7,
                gx + 7,
                gy + 7,
                fill="#55ff72",
                outline="",
                tags="dynamic",
            )
        with self.pose_lock:
            pose = self.pose
        centre = self._centre(pose)
        cx, cy = self._screen(centre)
        colour = "#42d7ff" if self.localized else "#ffb52e"
        self.canvas.create_oval(
            cx - 9,
            cy - 9,
            cx + 9,
            cy + 9,
            fill=colour,
            outline="white",
            tags="dynamic",
        )
        heading = math.radians(pose.theta_deg + self.forward_offset)
        hx, hy = self._screen(centre + 0.35 * np.array([math.cos(heading), math.sin(heading)]))
        self.canvas.create_line(
            cx,
            cy,
            hx,
            hy,
            fill=colour,
            width=4,
            arrow=tk.LAST,
            tags="dynamic",
        )

    def _refresh(self) -> None:
        self.status.configure(text=self.status_text)
        canvas_size = (self.canvas.winfo_width(), self.canvas.winfo_height())
        if (
            canvas_size != self._map_canvas_size
            and canvas_size[0] > 20
            and canvas_size[1] > 20
        ):
            self._compute_view()
            self._draw_map()
            self._map_canvas_size = canvas_size
        self._draw()
        self._draw_collision_diagnostic()
        self._record_collision_diagnostic_frame()
        if not self.closing:
            self.root.after(150, self._refresh)

    def _fresh_forward_sample(self) -> tuple[int, np.ndarray] | None:
        frame_id, frame = self.feed.latest()
        if frame is None:
            return None
        frame_age = time.time() - float(
            getattr(frame, "received_wall_ts", 0.0) or 0.0
        )
        if not math.isfinite(frame_age) or frame_age > 0.5:
            return None
        local = _scan_local(frame, self.args)
        if not len(local):
            return None
        forward = _filter_isolated_lidar_specks(
            _to_forward_frame(local, self.forward_offset)
        )
        forward = _remove_body_fixed_lidar_returns(
            forward,
            self.body_fixed_lidar_returns,
        )
        return int(frame_id), forward

    def _fresh_forward_points(self) -> np.ndarray | None:
        sample = self._fresh_forward_sample()
        return None if sample is None else sample[1]

    def _confirmed_collision(
        self,
        kind: str,
        frame_id: int,
        hit: tuple[np.ndarray, str, float, float, float] | None,
    ) -> tuple[np.ndarray, str, float, float, float] | None:
        """Require a spatially consistent hit on distinct LiDAR frames."""
        required = max(2, int(self.args.collision_confirmation_frames))
        with self._collision_confirmation_lock:
            if hit is None:
                self._collision_confirmation.pop(kind, None)
                return None
            _mask, sector, angle, distance, _limit = hit
            previous = self._collision_confirmation.get(kind)
            if previous is not None and int(previous["frame_id"]) == int(frame_id):
                return hit if int(previous["count"]) >= required else None
            consistent = (
                previous is not None
                and str(previous["sector"]) == str(sector)
                and abs(float(previous["angle"]) - float(angle)) <= 12.0
                and abs(float(previous["distance"]) - float(distance)) <= 0.10
            )
            count = int(previous["count"]) + 1 if consistent else 1
            self._collision_confirmation[kind] = {
                "frame_id": int(frame_id),
                "sector": str(sector),
                "angle": float(angle),
                "distance": float(distance),
                "count": count,
            }
            return hit if count >= required else None

    def _translation_safety(self) -> str | None:
        sample = self._fresh_forward_sample()
        if sample is None:
            return "LiDAR safety unavailable"
        frame_id, points = sample
        hit = collision_box_violation(
            points,
            self.collision_profile,
            lidar_offset_forward_m=self.lever_m,
            physical_body_radius_m=float(self.args.physical_body_radius_m),
            self_mask_inset_m=float(self.args.collision_self_mask_inset_m),
        )
        hit = self._confirmed_collision("translation", frame_id, hit)
        predictive_advance = 0.0
        if hit is None:
            predicted = _predictive_forward_collision(
                points,
                self.collision_profile,
                float(self.args.forward_collision_lookahead_m),
                lidar_offset_forward_m=self.lever_m,
                physical_body_radius_m=float(self.args.physical_body_radius_m),
                self_mask_inset_m=float(self.args.collision_self_mask_inset_m),
            )
            predicted_hit = None if predicted is None else predicted[0]
            confirmed_prediction = self._confirmed_collision(
                "predictive_translation", frame_id, predicted_hit
            )
            if confirmed_prediction is not None and predicted is not None:
                hit = confirmed_prediction
                predictive_advance = float(predicted[1])
        if hit is not None:
            mask = np.asarray(hit[0], dtype=bool)
            sector_points = points
            if predictive_advance > 0.0:
                sector_points = np.asarray(points, dtype=np.float64).copy()
                sector_points[:, 0] -= predictive_advance
            self._record_collision_points(
                points,
                mask,
                sector_points_forward_xy=sector_points,
            )
            angles = np.degrees(np.arctan2(points[mask, 1], points[mask, 0]))
            bin_size = float(self.collision_profile.get("bin_size_deg", 4.0))
            evidence_bins = np.unique(
                np.floor((angles + 180.0) / bin_size).astype(np.int64)
            ).size
            if predictive_advance > 0.0:
                reason = (
                    f"predictive LiDAR path stop {predictive_advance:.2f}m "
                    f"ahead: collision box {hit[1]} would be reached "
                    f"({hit[3]:.2f}m at {hit[2]:+.0f}deg, limit "
                    f"{hit[4]:.2f}m; {int(np.count_nonzero(mask))} "
                    f"points/{evidence_bins} bins)"
                )
                diagnostic_kind = "predictive translation"
            else:
                reason = (
                    f"collision box {hit[1]}: {hit[3]:.2f}m at "
                    f"{hit[2]:+.0f}deg (limit {hit[4]:.2f}m; "
                    f"{int(np.count_nonzero(mask))} points/{evidence_bins} bins)"
                )
                diagnostic_kind = "translation"
            self._capture_collision_diagnostic(
                kind=diagnostic_kind,
                frame_id=frame_id,
                points=points,
                hit_mask=mask,
                reason=reason,
            )
            return reason
        return None

    def _rotation_safety(self, remaining_deg: float) -> str | None:
        sample = self._fresh_forward_sample()
        if sample is None:
            return "LiDAR safety unavailable"
        frame_id, points = sample
        rotation_hit = collision_box_rotation_violation(
            points,
            self.collision_profile,
            float(remaining_deg),
            lidar_offset_forward_m=self.lever_m,
            physical_body_radius_m=float(self.args.physical_body_radius_m),
            self_mask_inset_m=float(self.args.collision_self_mask_inset_m),
        )

        raw_hit = None if rotation_hit is None else rotation_hit[0]
        hit = self._confirmed_collision("rotation", frame_id, raw_hit)
        if hit is not None:
            mask = np.asarray(hit[0], dtype=bool)
            self._record_collision_points(points, mask)
            angles = np.degrees(np.arctan2(points[mask, 1], points[mask, 0]))
            bin_size = float(self.collision_profile.get("bin_size_deg", 4.0))
            evidence_bins = np.unique(
                np.floor((angles + 180.0) / bin_size).astype(np.int64)
            ).size
        if hit is None:
            return None
        reason = (
            f"rotational collision box {hit[1]}: {hit[3]:.2f}m at "
            f"{hit[2]:+.0f}deg (limit {hit[4]:.2f}m; "
            f"{int(np.count_nonzero(mask))} points/{evidence_bins} bins)"
        )
        self._capture_collision_diagnostic(
            kind="rotation",
            frame_id=frame_id,
            points=points,
            hit_mask=mask,
            reason=reason,
            requested_rotation_deg=float(remaining_deg),
        )
        return reason

    def _rotation_probe_description(
        self,
        points: np.ndarray,
        rotation_deg: float,
    ) -> str:
        result = collision_box_rotation_violation(
            points,
            self.collision_profile,
            float(rotation_deg),
            lidar_offset_forward_m=self.lever_m,
            physical_body_radius_m=float(self.args.physical_body_radius_m),
            self_mask_inset_m=float(self.args.collision_self_mask_inset_m),
        )
        if result is None:
            return "CLEAR"
        hit, sweep_deg = result
        return (
            f"BLOCKED {hit[1]} at sweep {sweep_deg:+.0f}deg "
            f"({hit[3]:.2f}/{hit[4]:.2f}m)"
        )

    def _capture_collision_diagnostic(
        self,
        *,
        kind: str,
        frame_id: int,
        points: np.ndarray,
        hit_mask: np.ndarray,
        reason: str,
        requested_rotation_deg: float | None = None,
    ) -> None:
        """Freeze and print the exact scan used by a confirmed safety stop."""
        points_copy = np.asarray(points, dtype=np.float64).copy()
        mask_copy = np.asarray(hit_mask, dtype=bool).copy()
        with self.pose_lock:
            pose = self.pose
        centre = self._centre(pose)
        route_heading: float | None = None
        for waypoint in list(self.path):
            delta = np.asarray(waypoint, dtype=np.float64) - centre
            if float(np.linalg.norm(delta)) > 0.08:
                route_heading = math.degrees(math.atan2(delta[1], delta[0]))
                break
        map_heading = float(pose.theta_deg + self.forward_offset)
        route_error = (
            None
            if route_heading is None
            else (route_heading - map_heading + 180.0) % 360.0 - 180.0
        )
        diagnostic = CollisionDiagnostic(
            kind=str(kind),
            frame_id=int(frame_id),
            points_forward_xy=points_copy,
            hit_mask=mask_copy,
            reason=str(reason),
            pose=pose,
            robot_centre_xy=np.asarray(centre, dtype=np.float64).copy(),
            imu_heading_deg=self.imu.deg(),
            route_heading_deg=route_heading,
            route_heading_error_deg=route_error,
            requested_rotation_deg=requested_rotation_deg,
            clockwise_result=self._rotation_probe_description(points_copy, -30.0),
            counterclockwise_result=self._rotation_probe_description(points_copy, +30.0),
            captured_at=time.time(),
        )
        with self._collision_diagnostic_lock:
            self._collision_diagnostic = diagnostic

        key = (str(kind), int(frame_id))
        if key == self._last_collision_diagnostic_key:
            return
        self._last_collision_diagnostic_key = key
        selected = points_copy[mask_copy]
        theta = math.radians(float(self.forward_offset))
        c, s = math.cos(theta), math.sin(theta)
        selected_local = np.column_stack(
            [
                selected[:, 0] * c - selected[:, 1] * s,
                selected[:, 0] * s + selected[:, 1] * c,
            ]
        )
        selected_world = _transform_points(selected_local, pose)
        print(
            f"[collision-debug] CONFIRMED {kind} stop on LiDAR frame {frame_id}: "
            f"{reason}"
        )
        print(
            "[collision-debug] estimated robot centre "
            f"({centre[0]:+.3f}, {centre[1]:+.3f})m, map heading "
            f"{map_heading:+.1f}deg, route heading "
            f"{route_heading if route_heading is not None else float('nan'):+.1f}deg "
            f"(error {route_error if route_error is not None else float('nan'):+.1f}deg), IMU "
            f"{diagnostic.imu_heading_deg if diagnostic.imu_heading_deg is not None else float('nan'):+.1f}deg; "
            f"CW -30deg: {diagnostic.clockwise_result}; "
            f"CCW +30deg: {diagnostic.counterclockwise_result}."
        )
        for index, (body, world) in enumerate(
            zip(selected[:24], selected_world[:24], strict=False), start=1
        ):
            print(
                f"[collision-debug] hit {index:02d}: robot-frame "
                f"forward={body[0]:+.3f}m left={body[1]:+.3f}m; "
                f"projected world x={world[0]:+.3f}m y={world[1]:+.3f}m"
            )
        if len(selected) > 24:
            print(
                f"[collision-debug] ... {len(selected) - 24} additional hit points "
                "are visible in the frozen diagnostic panel."
            )

    def _collision_envelope_points(self, profile: dict) -> np.ndarray:
        ranges = effective_ranges(profile)
        if not len(ranges):
            return np.empty((0, 2), dtype=np.float64)
        bin_size = float(profile.get("bin_size_deg", 4.0))
        angles = np.radians(-180.0 + (np.arange(len(ranges)) + 0.5) * bin_size)
        noise = float(profile.get("noise_tolerance_m", 0.02))
        margin = float(profile.get("safety_margin_m", noise))
        radii = ranges + margin - noise
        valid = np.isfinite(radii)
        return np.column_stack(
            [radii[valid] * np.cos(angles[valid]), radii[valid] * np.sin(angles[valid])]
        )

    def _draw_collision_diagnostic(self) -> None:
        """Draw current LiDAR plus the saved-map-independent last-stop evidence."""
        canvas = self.collision_canvas
        canvas.delete("all")
        width = max(20, canvas.winfo_width())
        height = max(20, canvas.winfo_height())
        with self._collision_diagnostic_lock:
            diagnostic = self._collision_diagnostic
        live_sample = self._fresh_forward_sample()
        live_frame_id = None
        live_points = np.empty((0, 2), dtype=np.float64)
        live_hit = None
        if live_sample is not None:
            live_frame_id, live_points = live_sample
            live_hit = collision_box_violation(
                live_points,
                self.collision_profile,
                lidar_offset_forward_m=self.lever_m,
                physical_body_radius_m=float(self.args.physical_body_radius_m),
                self_mask_inset_m=float(self.args.collision_self_mask_inset_m),
            )
        canvas.create_text(
            10,
            10,
            anchor="nw",
            fill="#d7e4ea",
            font=("TkDefaultFont", 10, "bold"),
            text=(
                "LIVE LiDAR + LAST STOP (robot frame) | "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"run +{time.monotonic() - self._run_started_monotonic:.1f}s"
            ),
        )
        live_status = "UNAVAILABLE"
        if live_frame_id is not None:
            live_status = "CLEAR" if live_hit is None else (
                f"COLLISION {live_hit[1]} {live_hit[3]:.2f}/{live_hit[4]:.2f}m"
            )
        plot_top = 118.0 if diagnostic is not None else 68.0
        plot_bottom = float(height - 14)
        plot_size = max(100.0, min(float(width - 20), plot_bottom - plot_top))
        cx = float(width) / 2.0
        cy = plot_top + plot_size / 2.0
        extent_m = 1.25
        scale = 0.5 * plot_size / extent_m

        def screen_from_lidar(points_xy: np.ndarray) -> np.ndarray:
            points_xy = np.asarray(points_xy, dtype=np.float64)
            # Centre the view on the robot centre. LiDAR is lever_m forward.
            centred_x = points_xy[:, 0] + self.lever_m
            centred_y = points_xy[:, 1]
            return np.column_stack([cx - centred_y * scale, cy - centred_x * scale])

        canvas.create_oval(
            cx - extent_m * scale,
            cy - extent_m * scale,
            cx + extent_m * scale,
            cy + extent_m * scale,
            outline="#263840",
        )
        hard = self._collision_envelope_points(self.collision_profile)
        soft = self._collision_envelope_points(self.soft_collision_profile)
        for envelope, colour, dash, width_px in (
            (soft, "#e8c94d", (4, 3), 1),
            (hard, "#22c8f2", None, 2),
        ):
            if len(envelope) >= 3:
                pixels = screen_from_lidar(envelope)
                coords = pixels.reshape(-1).tolist() + pixels[0].tolist()
                canvas.create_line(
                    *coords,
                    fill=colour,
                    width=width_px,
                    dash=dash,
                )

        body_half = float(self.args.physical_body_radius_m)
        canvas.create_rectangle(
            cx - body_half * scale,
            cy - body_half * scale,
            cx + body_half * scale,
            cy + body_half * scale,
            outline="#9ca7aa",
            width=2,
        )
        lidar_xy = screen_from_lidar(np.asarray([[0.0, 0.0]]))[0]
        canvas.create_oval(
            lidar_xy[0] - 3,
            lidar_xy[1] - 3,
            lidar_xy[0] + 3,
            lidar_xy[1] + 3,
            fill="#53d8ff",
            outline="",
        )
        canvas.create_line(cx, cy, cx, cy - 0.42 * scale, fill="white", width=2, arrow=tk.LAST)

        # The old trigger remains visible in purple/red, but it is never
        # presented as the current scan. This distinction matters after an
        # escape turn: the robot can be clear now even though the reason that
        # initiated recovery remains useful forensic evidence.
        if diagnostic is not None:
            frozen_points = diagnostic.points_forward_xy
            frozen_near = (
                np.hypot(frozen_points[:, 0] + self.lever_m, frozen_points[:, 1])
                <= extent_m
            )
            frozen_pixels = screen_from_lidar(frozen_points[frozen_near])
            frozen_hit_near = diagnostic.hit_mask[frozen_near]
            for pixel, is_hit in zip(
                frozen_pixels, frozen_hit_near, strict=False
            ):
                radius = 3 if bool(is_hit) else 1
                colour = "#ff3b30" if bool(is_hit) else "#6f638f"
                canvas.create_oval(
                    pixel[0] - radius,
                    pixel[1] - radius,
                    pixel[0] + radius,
                    pixel[1] + radius,
                    fill=colour,
                    outline="",
                )

        live_near = (
            np.hypot(live_points[:, 0] + self.lever_m, live_points[:, 1])
            <= extent_m
        )
        live_pixels = screen_from_lidar(live_points[live_near])
        live_hit_mask = np.zeros(len(live_points), dtype=bool)
        if live_hit is not None:
            live_hit_mask = np.asarray(live_hit[0], dtype=bool)
        live_hit_near = live_hit_mask[live_near]
        for pixel, is_hit in zip(live_pixels, live_hit_near, strict=False):
            radius = 3 if bool(is_hit) else 1
            colour = "#ffb000" if bool(is_hit) else "#55ef75"
            canvas.create_oval(
                pixel[0] - radius,
                pixel[1] - radius,
                pixel[0] + radius,
                pixel[1] + radius,
                fill=colour,
                outline="",
            )

        if diagnostic is None:
            detail_text = (
                f"LIVE frame {live_frame_id if live_frame_id is not None else 'n/a'}: "
                f"{live_status}\nLAST STOP: none yet"
            )
        else:
            age = max(0.0, time.time() - diagnostic.captured_at)
            imu_text = (
                "n/a"
                if diagnostic.imu_heading_deg is None
                else f"{diagnostic.imu_heading_deg:+.1f}deg"
            )
            route_text = (
                "n/a"
                if diagnostic.route_heading_deg is None
                else (
                    f"{diagnostic.route_heading_deg:+.1f}deg "
                    f"(error {diagnostic.route_heading_error_deg:+.1f}deg)"
                )
            )
            detail_text = (
                f"LIVE frame {live_frame_id if live_frame_id is not None else 'n/a'}: "
                f"{live_status}\n"
                f"LAST STOP frame {diagnostic.frame_id}: frozen {age:.1f}s ago | "
                f"{diagnostic.kind}\n"
                f"centre ({diagnostic.robot_centre_xy[0]:+.2f}, "
                f"{diagnostic.robot_centre_xy[1]:+.2f})m | "
                f"map heading {diagnostic.pose.theta_deg + self.forward_offset:+.1f}deg | "
                f"route {route_text} | IMU {imu_text}\n"
                f"CW: {diagnostic.clockwise_result}\n"
                f"CCW: {diagnostic.counterclockwise_result}"
            )
        canvas.create_text(
            10,
            30,
            anchor="nw",
            fill="#d7e4ea",
            width=width - 20,
            text=detail_text,
        )
        canvas.create_text(
            10,
            height - 8,
            anchor="sw",
            fill="#aebbc0",
            width=width - 20,
            text=(
                "green=LIVE  orange=live trigger  purple=frozen scan  "
                "red=frozen trigger\ncyan=hard box  yellow=soft warning  white=forward"
            ),
        )

    def _soft_clearance_heading_bias(self) -> float:
        """Return a non-latching steering hint from an outer warning envelope.

        The calibrated collision box remains the sole stop authority. This
        larger envelope only asks the heading controller to lean away from a
        one-sided return before a shoulder reaches the hard box. Equal evidence
        on both sides is a valid narrow corridor and produces no bias.
        """
        if self._edge_escape_anchor is not None:
            return 0.0
        sample = self._fresh_forward_sample()
        if sample is None or not self.soft_collision_profile:
            return 0.0
        frame_id, points = sample
        raw_hit = collision_box_violation(
            points,
            self.soft_collision_profile,
            lidar_offset_forward_m=self.lever_m,
            physical_body_radius_m=float(self.args.physical_body_radius_m),
            self_mask_inset_m=float(self.args.collision_self_mask_inset_m),
        )
        hit = self._confirmed_collision("soft", frame_id, raw_hit)
        if hit is None:
            # A newly observed warning (especially evidence on the opposite or
            # both sides) must cancel the old steering hint immediately even
            # while it is waiting for multi-frame confirmation. Hysteresis is
            # only appropriate when the *raw* current scan is clear.
            if raw_hit is not None:
                self._soft_warning_bias = 0.0
                return 0.0
            # A shoulder can pass beyond the LiDAR ray that first saw a sharp
            # edge while the physical corner is still alongside it. Preserve
            # the last gentle steer briefly instead of snapping immediately
            # back to the route heading and hooking that edge.
            if (
                self._soft_warning_side not in (None, "both", "front")
                and time.monotonic()
                - float(getattr(self, "_soft_warning_last_seen_at", -math.inf))
                < 0.65
            ):
                return float(getattr(self, "_soft_warning_bias", 0.0))
            if self._soft_warning_side is not None:
                print("[saved-map] soft clearance envelope is clear; restoring route heading.")
            self._soft_warning_side = None
            self._soft_warning_bias = 0.0
            return 0.0

        mask = np.asarray(hit[0], dtype=bool)
        selected = np.asarray(points, dtype=np.float64)[mask]
        # Forward travel should not steer in response to furniture already
        # behind the chassis. Retain the shoulder band and everything ahead.
        selected = selected[selected[:, 0] >= -0.10]
        if not len(selected):
            return 0.0
        angles = np.degrees(np.arctan2(selected[:, 1], selected[:, 0]))
        left_count = int(np.count_nonzero((angles >= 25.0) & (angles <= 145.0)))
        right_count = int(np.count_nonzero((angles <= -25.0) & (angles >= -145.0)))
        side: str | None
        sign = 0.0
        if left_count and right_count:
            side = "both"
        elif left_count:
            side = "left"
            sign = -1.0
        elif right_count:
            side = "right"
            sign = 1.0
        else:
            bearing = float(np.median(angles))
            if abs(bearing) < 4.0:
                side = "front"
            elif bearing > 0.0:
                side = "left-front"
                sign = -1.0
            else:
                side = "right-front"
                sign = 1.0

        if side != self._soft_warning_side:
            if sign:
                direction = "clockwise" if sign < 0.0 else "counterclockwise"
                print(
                    f"[saved-map] soft clearance warning on {side}; biasing "
                    f"heading {direction} without stopping."
                )
            elif side == "both":
                print(
                    "[saved-map] soft clearance warning on both sides; preserving "
                    "the measured corridor heading without stopping."
                )
            self._soft_warning_side = side
        if not sign:
            self._soft_warning_bias = 0.0
            self._soft_warning_last_seen_at = time.monotonic()
            return 0.0

        warning_margin = max(0.01, float(self.args.soft_collision_margin_m))
        penetration = max(0.0, float(hit[4]) - float(hit[3]))
        fraction = min(1.0, penetration / warning_margin)
        magnitude = min(
            float(self.args.soft_steer_max_deg),
            2.0 + fraction * float(self.args.soft_steer_max_deg),
        )
        self._soft_warning_bias = sign * magnitude
        self._soft_warning_last_seen_at = time.monotonic()
        return float(self._soft_warning_bias)

    def _record_collision_points(
        self,
        points_forward_xy: np.ndarray,
        mask: np.ndarray,
        *,
        sector_points_forward_xy: np.ndarray | None = None,
    ) -> None:
        """Freeze the exact live stop returns in world coordinates."""
        points = np.asarray(points_forward_xy, dtype=np.float64)
        selected = points[np.asarray(mask, dtype=bool)]
        if not len(selected):
            return
        theta = math.radians(float(self.forward_offset))
        c, s = math.cos(theta), math.sin(theta)
        local = np.column_stack(
            [
                selected[:, 0] * c - selected[:, 1] * s,
                selected[:, 0] * s + selected[:, 1] * c,
            ]
        ).astype(np.float32)
        with self.pose_lock:
            pose = self.pose
        self.pending_collision_world = _transform_points(local, pose).astype(
            np.float32, copy=False
        )
        sector_points = (
            points
            if sector_points_forward_xy is None
            else np.asarray(sector_points_forward_xy, dtype=np.float64).reshape((-1, 2))
        )
        sector_selected = sector_points[np.asarray(mask, dtype=bool)]
        angles = np.degrees(
            np.arctan2(sector_selected[:, 1], sector_selected[:, 0])
        )
        sectors: set[str] = set()
        # Sourccey's shoulders begin well before a mathematically exact 45deg
        # diagonal. Treat the front-corner bands as one-sided contacts so a
        # +43deg hit turns right and a -43deg hit turns left instead of being
        # misclassified as a front contact with no open-side preference.
        side_boundary_deg = 30.0
        rear_boundary_deg = 150.0
        if np.any(
            (angles >= side_boundary_deg) & (angles <= rear_boundary_deg)
        ):
            sectors.add("left")
        if np.any(
            (angles <= -side_boundary_deg) & (angles >= -rear_boundary_deg)
        ):
            sectors.add("right")
        if np.any(np.abs(angles) < side_boundary_deg):
            sectors.add("front")
        if np.any(np.abs(angles) > rear_boundary_deg):
            sectors.add("rear")
        self.pending_collision_sectors = sectors

    def _strafe_away_from_side(
        self,
        lateral_sign: float,
        distance_m: float = 0.12,
    ) -> bool:
        """Execute a LiDAR-rolled-out mecanum strafe away from one side.

        Positive body lateral is left. A right-side contact therefore requests
        ``+1`` and a left-side contact ``-1``. The robot holds its IMU heading,
        checks every fresh scan against the complete swept translation, and
        accepts motion only after LiDAR odometry verifies the requested lateral
        displacement. No open-loop sideways command survives a stale scan.
        """
        sign = math.copysign(1.0, float(lateral_sign))
        requested_delta = np.asarray(
            [0.0, sign * float(distance_m)], dtype=np.float64
        )
        sample = self._fresh_forward_sample()
        if sample is None:
            return False
        _frame_id, points = sample
        safe, improvement = _translation_escape_is_safe(
            points,
            self.collision_profile,
            requested_delta,
            lidar_offset_forward_m=self.lever_m,
            physical_body_radius_m=float(self.args.physical_body_radius_m),
            self_mask_inset_m=float(self.args.collision_self_mask_inset_m),
        )
        if not safe:
            # A preceding lateral step may already have cleared the original
            # envelope. In that state there is no penetration for the escape
            # predicate to improve, but another sideways step can still be a
            # completely clear swept trajectory toward the passage centre.
            safe = _translation_trajectory_is_safe(
                points,
                self.collision_profile,
                requested_delta,
                lidar_offset_forward_m=self.lever_m,
                physical_body_radius_m=float(self.args.physical_body_radius_m),
                self_mask_inset_m=float(self.args.collision_self_mask_inset_m),
            )
            improvement = 0.0
        if not safe:
            print(
                "[saved-map] live LiDAR rejected the direct lateral escape; "
                "using the collision-checked turning fallback."
            )
            return False

        def _contact_side_clearance(side_points: np.ndarray) -> float | None:
            cloud = np.asarray(side_points, dtype=np.float64).reshape((-1, 2))
            # ``sign`` points away from the contacted side, hence contacted
            # returns have lateral coordinate with the opposite sign. Limit
            # this measurement to the chassis-length band so distant room
            # geometry cannot masquerade as shoulder clearance.
            contact = cloud[
                (cloud[:, 1] * sign < 0.0)
                & (cloud[:, 0] >= -0.45)
                & (cloud[:, 0] <= 0.45)
            ]
            if len(contact) < 2:
                return None
            return float(np.percentile(np.abs(contact[:, 1]), 15.0))

        initial_clearance = _contact_side_clearance(points)

        initial_yaw = self.imu.deg()
        if initial_yaw is None:
            return False
        with self.pose_lock:
            start_pose = self.pose
            start_centre = self._centre(start_pose)
        heading_rad = math.radians(start_pose.theta_deg + self.forward_offset)
        lateral_axis = sign * np.asarray(
            [-math.sin(heading_rad), math.cos(heading_rad)], dtype=np.float64
        )
        forward_axis = np.asarray(
            [math.cos(heading_rad), math.sin(heading_rad)], dtype=np.float64
        )
        command_xy = _startup_escape_velocity(
            requested_delta,
            abs(float(self.args.drive_speed)),
        )
        after = self.feed.latest()[0]
        self.controller.clear_safety_latch()
        self.controller.translate_body(
            float(command_xy[0]), float(command_xy[1]), float(initial_yaw)
        )
        started_at = time.monotonic()
        command_escalated = False
        deadline = time.monotonic() + 1.8
        try:
            while time.monotonic() < deadline and not self.stop_event.is_set():
                frame_id, frame = self.feed.wait_for_frame_after(
                    after_frame_id=after,
                    timeout_s=0.55,
                    min_frame_advances=1,
                )
                if frame is None:
                    print(
                        "[saved-map] lateral escape stopped because no fresh "
                        "LiDAR frame arrived."
                    )
                    return False
                after = int(frame_id)
                fresh = self._fresh_forward_sample()
                if fresh is None:
                    return False
                _fresh_id, fresh_points = fresh
                fresh_clearance = _contact_side_clearance(fresh_points)
                clearance_gain = (
                    float(fresh_clearance) - float(initial_clearance)
                    if fresh_clearance is not None and initial_clearance is not None
                    else 0.0
                )
                with self.pose_lock:
                    current_pose = self.pose
                current_centre = self._centre(current_pose)
                measured = current_centre - start_centre
                current_progress = max(0.0, float(measured @ lateral_axis))
                remaining = max(0.02, float(distance_m) - current_progress)
                # The initial predicate proves that motion monotonically exits
                # an existing intrusion. Once the side is clear that predicate
                # intentionally returns False (there is no intrusion left to
                # escape), so continued motion must use the normal swept-path
                # validator instead of treating successful clearance as a
                # failure.
                safe = _translation_trajectory_is_safe(
                    fresh_points,
                    self.collision_profile,
                    np.asarray([0.0, sign * remaining], dtype=np.float64),
                    lidar_offset_forward_m=self.lever_m,
                    physical_body_radius_m=float(self.args.physical_body_radius_m),
                    self_mask_inset_m=float(self.args.collision_self_mask_inset_m),
                )
                if not safe:
                    print(
                        "[saved-map] lateral escape stopped because the fresh "
                        "LiDAR swept path became occupied."
                    )
                    return False

                # Lateral wheel stiction is independent of forward stiction.
                # If a healthy live scan shows virtually no response after the
                # initial command, raise only this bounded escape to full
                # effort; the same swept-LiDAR checks continue every frame.
                if (
                    not command_escalated
                    and time.monotonic() - started_at >= 0.55
                    and clearance_gain < 0.015
                ):
                    command_xy = _startup_escape_velocity(
                        requested_delta,
                        1.0,
                        stiction_floor=1.0,
                    )
                    self.controller.translate_body(
                        float(command_xy[0]),
                        float(command_xy[1]),
                        float(initial_yaw),
                    )
                    command_escalated = True
                    print(
                        "[saved-map] lateral motion watchdog measured no "
                        "clearance response; raising the bounded strafe command "
                        "to full effort."
                    )

                local = _scan_local(frame, self.args)
                if len(local) < 30:
                    continue
                yaw = _imu_yaw_for_scan(self.imu, frame)
                theta_seed = current_pose.theta_deg + (
                    (float(yaw) - float(initial_yaw)) if yaw is not None else 0.0
                )
                seed = _pose_with_imu_heading(
                    current_pose,
                    theta_seed,
                    self.lever_m,
                    self.forward_offset,
                )
                solved, score, support = _localize_against_points(
                    local,
                    self.local_odometry.reference(),
                    seed,
                    self.args,
                    0.35,
                    4.0,
                )
                solved = _pose_with_imu_heading(
                    solved, theta_seed, self.lever_m, self.forward_offset
                )
                step = self._centre(solved) - current_centre
                if float(np.hypot(*step)) > 0.16:
                    continue
                if (
                    score < max(4.0, float(self.args.localization_min_score) - 2.0)
                    or support < 0.18
                ):
                    continue
                displacement = self._centre(solved) - start_centre
                progress = float(displacement @ lateral_axis)
                cross_track = abs(float(displacement @ forward_axis))
                if progress < -0.02 or cross_track > 0.10:
                    continue
                with self.pose_lock:
                    self.pose = solved
                self.local_odometry.add(local, solved)
                if progress >= float(distance_m):
                    print(
                        f"[saved-map] LiDAR-verified lateral escape completed "
                        f"{progress:.2f}m with predicted clearance improvement "
                        f"{improvement * 100.0:.1f}cm."
                    )
                    return True
            print(
                "[saved-map] lateral escape did not produce enough verified "
                "sideways displacement; using the turning fallback."
            )
            return False
        finally:
            self.controller.halt()
            self.controller.clear_safety_latch()

    def _unlocalized_strafe_away_from_side(
        self,
        lateral_sign: float,
        blocked_rotation_deg: float,
        distance_m: float = 0.12,
    ) -> bool:
        """Create pivot clearance before a saved-map pose is available.

        Global localization cannot use ``self.pose`` to verify an escape: that
        pose is precisely what the active sweep is trying to discover.  This
        relocation is therefore verified entirely in the robot frame.  It
        checks the complete lateral swept volume on every fresh LiDAR frame
        and accepts the move only when the contacted-side clearance measurably
        increases (or the nearby contacted surface disappears).  The caller
        then discards every pre-move localization view and starts a new 360deg
        acquisition at the new physical position.
        """
        sign = math.copysign(1.0, float(lateral_sign))
        requested_delta = np.asarray(
            [0.0, sign * float(distance_m)], dtype=np.float64
        )
        sample = self._fresh_forward_sample()
        if sample is None:
            return False
        after, points = sample

        geometry = {
            "lidar_offset_forward_m": self.lever_m,
            "physical_body_radius_m": float(self.args.physical_body_radius_m),
            "self_mask_inset_m": float(self.args.collision_self_mask_inset_m),
        }

        # A pivot-only obstruction can leave the current (unrotated)
        # translation envelope completely clear.  In that case the normal
        # monotonic-penetration escape predicate correctly says "nothing to
        # escape", so fall through to the ordinary swept-translation test.
        escape_safe = _translation_escape_is_safe(
            points,
            self.collision_profile,
            requested_delta,
            **geometry,
        )[0]
        trajectory_safe = _translation_trajectory_is_safe(
            points,
            self.collision_profile,
            requested_delta,
            **geometry,
        )
        if not escape_safe and not trajectory_safe:
            print(
                "[saved-map] outward localization relocation rejected: the "
                "live LiDAR lateral swept volume is occupied."
            )
            return False

        def blocked_pivot_hit(cloud_xy: np.ndarray):
            result = collision_box_rotation_violation(
                cloud_xy,
                self.collision_profile,
                float(blocked_rotation_deg),
                **geometry,
            )
            return None if result is None else result[0]

        initial_hit = blocked_pivot_hit(points)
        initial_side = None if initial_hit is None else str(initial_hit[1])
        initial_distance = None if initial_hit is None else float(initial_hit[3])
        initial_yaw = self.imu.deg()
        if initial_yaw is None:
            return False

        command_xy = _startup_escape_velocity(
            requested_delta,
            abs(float(self.args.drive_speed)),
        )
        self.controller.clear_safety_latch()
        self.controller.translate_body(
            float(command_xy[0]), float(command_xy[1]), float(initial_yaw)
        )
        started_at = time.monotonic()
        best_gain = 0.0
        clear_frames = 0
        try:
            while (
                time.monotonic() - started_at < 1.35
                and not self.stop_event.is_set()
            ):
                frame_id, frame = self.feed.wait_for_frame_after(
                    after_frame_id=int(after),
                    timeout_s=0.45,
                    min_frame_advances=1,
                )
                if frame is None:
                    return False
                after = int(frame_id)
                fresh = self._fresh_forward_sample()
                if fresh is None:
                    return False
                _fresh_id, fresh_points = fresh
                elapsed = time.monotonic() - started_at
                pivot_hit = blocked_pivot_hit(fresh_points)
                if pivot_hit is None:
                    clear_frames += 1
                else:
                    clear_frames = 0
                    if (
                        initial_distance is not None
                        and (initial_side is None or str(pivot_hit[1]) == initial_side)
                    ):
                        best_gain = max(
                            best_gain,
                            float(pivot_hit[3]) - initial_distance,
                        )

                remaining_fraction = max(0.20, 1.0 - elapsed / 1.35)
                if not _translation_trajectory_is_safe(
                    fresh_points,
                    self.collision_profile,
                    requested_delta * remaining_fraction,
                    **geometry,
                ):
                    print(
                        "[saved-map] outward localization relocation stopped: "
                        "a fresh LiDAR frame closed the remaining lateral path."
                    )
                    return False
                if elapsed >= 0.45 and (best_gain >= 0.025 or clear_frames >= 2):
                    print(
                        "[saved-map] unlocalized lateral relocation verified "
                        "against the formerly blocked rotational sweep "
                        f"(clearance gain {best_gain * 100.0:.1f}cm; "
                        f"clear frames {clear_frames})."
                    )
                    return True
            if best_gain >= 0.020:
                print(
                    "[saved-map] unlocalized lateral relocation accepted at "
                    f"deadline with {best_gain * 100.0:.1f}cm of measured "
                    "rotational-sweep clearance gain."
                )
                return True
            print(
                "[saved-map] outward localization relocation command ended "
                "without enough rotational-sweep clearance change "
                f"({best_gain * 100.0:.1f}cm; clear frames {clear_frames})."
            )
            return False
        finally:
            self.controller.halt()
            self.controller.clear_safety_latch()

    def _stationary_lidar_observation(
        self,
        frame_count: int = 4,
    ) -> np.ndarray:
        """Stop, settle, and fuse several distinct robot-frame LiDAR scans."""
        self.controller.halt()
        # Let the host watchdog and drivetrain reach zero before interpreting
        # any scan as stationary geometry. This is deliberately blocking: no
        # recovery command may race the observation used to plan it.
        time.sleep(0.30)
        after = self.feed.latest()[0]
        captures: list[np.ndarray] = []
        for _ in range(max(3, int(frame_count))):
            frame_id, frame = self.feed.wait_for_frame_after(
                after_frame_id=after,
                timeout_s=0.8,
                min_frame_advances=1,
            )
            if frame is None:
                continue
            after = int(frame_id)
            local = _scan_local(frame, self.args)
            if not len(local):
                continue
            forward = _filter_isolated_lidar_specks(
                _to_forward_frame(local, self.forward_offset)
            )
            forward = _remove_body_fixed_lidar_returns(
                forward,
                self.body_fixed_lidar_returns,
            )
            if len(forward):
                captures.append(np.asarray(forward, dtype=np.float64))
        if not captures:
            return np.empty((0, 2), dtype=np.float64)
        fused = np.concatenate(captures, axis=0)
        # Collapse repeated stationary returns into 2cm cells. Persistent wall
        # geometry remains dense while duplicate samples cannot overweight one
        # particular LiDAR revolution in the local costmap.
        cells = np.round(fused / 0.02).astype(np.int64)
        _unique, indices = np.unique(cells, axis=0, return_index=True)
        fused = fused[np.sort(indices)]
        print(
            f"[saved-map] stationary collision observation fused "
            f"{len(captures)} distinct scan(s) into {len(fused)} local returns."
        )
        return fused

    def _refresh_local_obstacles_from_lidar(
        self,
        forward_points: np.ndarray | None = None,
    ) -> int:
        """Insert a stationary LiDAR observation for local replanning."""
        if forward_points is None:
            forward = self._stationary_lidar_observation()
        else:
            forward = np.asarray(forward_points, dtype=np.float64).reshape((-1, 2))
        if not len(forward):
            return 0
        ranges = np.hypot(forward[:, 0], forward[:, 1])
        nearby = forward[(ranges >= 0.20) & (ranges <= 2.0)]
        if not len(nearby):
            return 0
        if len(nearby) > 400:
            indices = np.linspace(0, len(nearby) - 1, 400).astype(np.int64)
            nearby = nearby[indices]
        theta = math.radians(float(self.forward_offset))
        c, s = math.cos(theta), math.sin(theta)
        local = np.column_stack(
            [
                nearby[:, 0] * c - nearby[:, 1] * s,
                nearby[:, 0] * s + nearby[:, 1] * c,
            ]
        ).astype(np.float32)
        with self.pose_lock:
            pose = self.pose
        world_points = _transform_points(local, pose).astype(np.float32, copy=False)
        if len(self.dynamic_obstacles_world):
            self.dynamic_obstacles_world = np.concatenate(
                [self.dynamic_obstacles_world, world_points], axis=0
            )[-800:]
        else:
            self.dynamic_obstacles_world = world_points[-800:].copy()
        return len(world_points)

    def _detect_live_lidar_passage(
        self,
        forward_points: np.ndarray | None = None,
    ) -> LivePassagePlan | None:
        """Measure a passage and freeze one centre-align-straight plan.

        The saved map chooses the destination; this local measurement chooses
        where the chassis belongs between the walls that physically exist now.
        Robust per-depth quantiles avoid letting one chair leg or isolated
        return define an entire corridor boundary.  Detection is deliberately
        side-effect free: no wheel command is issued until the whole local
        manoeuvre has been calculated.
        """
        if forward_points is None:
            sample = self._fresh_forward_sample()
            if sample is None:
                return None
            _frame_id, points = sample
        else:
            points = np.asarray(forward_points, dtype=np.float64).reshape((-1, 2))
        cloud = np.asarray(points, dtype=np.float64).reshape((-1, 2))
        centres: list[tuple[float, float]] = []
        widths: list[float] = []
        # Estimate both boundaries in successive forward slices. Requiring
        # paired support in several slices distinguishes a corridor from a
        # single nearby object that happens to touch one shoulder.
        for near_x in np.arange(0.10, 1.01, 0.15):
            band = cloud[
                (cloud[:, 0] >= near_x)
                & (cloud[:, 0] < near_x + 0.20)
                & (np.abs(cloud[:, 1]) <= 1.10)
            ]
            left = band[band[:, 1] > 0.0, 1]
            right = band[band[:, 1] < 0.0, 1]
            if len(left) < 2 or len(right) < 2:
                continue
            left_wall = float(np.percentile(left, 20.0))
            right_wall = float(np.percentile(right, 80.0))
            width = left_wall - right_wall
            if not 0.40 <= width <= 1.80:
                continue
            centres.append((float(near_x + 0.10), 0.5 * (left_wall + right_wall)))
            widths.append(width)
        if len(centres) < 3:
            print(
                "[saved-map] fresh LiDAR did not contain enough paired wall "
                "support for passage centering."
            )
            return None

        centre_samples = np.asarray(centres, dtype=np.float64)
        passage_width = float(np.median(widths))
        # Use the near half of the measured centerline for the immediate
        # lateral correction. Far geometry can widen into the next room and
        # should not pull the robot toward one jamb.
        near = centre_samples[centre_samples[:, 0] <= 0.65]
        if not len(near):
            near = centre_samples
        lateral_offset = float(np.median(near[:, 1]))
        heading_slope = float(
            np.polyfit(centre_samples[:, 0], centre_samples[:, 1], 1)[0]
        )
        passage_heading = math.degrees(math.atan(heading_slope))
        minimum_usable_width = max(
            0.44,
            2.0 * float(self.args.physical_body_radius_m) - 0.06,
        )
        print(
            f"[saved-map] live LiDAR passage: width {passage_width:.2f}m, "
            f"centre offset {lateral_offset:+.2f}m, heading "
            f"{passage_heading:+.1f}deg."
        )
        if passage_width < minimum_usable_width:
            print(
                "[saved-map] paired walls are narrower than the calibrated "
                f"usable passage width {minimum_usable_width:.2f}m."
            )
            return None

        # Do not chase the centreline sideways from a stopped contact pose.
        # Pick a visible point on the centreline, then a second point farther
        # along the same fitted line.  The ordinary pivot/straight follower
        # therefore executes exactly: turn toward centre, drive to centre,
        # turn parallel to the walls, drive straight through the throat.
        max_x = float(np.max(centre_samples[:, 0]))
        # The farther ahead the intercept is, the smaller the initial pivot
        # needed to reach the same lateral centreline. Tight corridors benefit
        # from this shallow approach because the rear corner does not sweep as
        # far toward the contacted wall. Keep it inside actually observed wall
        # support rather than projecting through unknown space.
        approach_x = max(0.45, min(0.75, max_x - 0.25))
        exit_x = min(1.35, max(approach_x + 0.45, max_x))
        intercept = float(np.median(centre_samples[:, 1] - heading_slope * centre_samples[:, 0]))
        approach_y = heading_slope * approach_x + intercept
        exit_y = heading_slope * exit_x + intercept
        return LivePassagePlan(
            width_m=passage_width,
            lateral_offset_m=lateral_offset,
            heading_deg=passage_heading,
            approach_body_xy=np.asarray([approach_x, approach_y], dtype=np.float64),
            exit_body_xy=np.asarray([exit_x, exit_y], dtype=np.float64),
        )

    def _route_through_live_lidar_passage(
        self,
        plan: LivePassagePlan,
    ) -> tuple[list[np.ndarray], float]:
        """Convert one frozen body-frame passage plan into world waypoints.

        The two local points are kept verbatim and are never replaced by a
        costmap corner.  Once beyond the observed throat, the saved-map planner
        supplies the remaining route to the original clicked destination.
        """
        with self.pose_lock:
            start = self._centre(self.pose)
            heading_deg = self.pose.theta_deg + self.forward_offset
        heading = math.radians(float(heading_deg))
        c, s = math.cos(heading), math.sin(heading)

        def body_to_world(point: np.ndarray) -> np.ndarray:
            x, y = map(float, np.asarray(point, dtype=np.float64))
            return start + np.asarray([x * c - y * s, x * s + y * c])

        approach = body_to_world(plan.approach_body_xy)
        exit_point = body_to_world(plan.exit_body_xy)
        route: list[np.ndarray] = [approach, exit_point]
        radius = float(self.args.physical_body_radius_m)
        if self.goal is not None:
            continuation, radius = _plan_saved_map_route(
                self.world,
                exit_point,
                np.asarray(self.goal),
                float(self.args.robot_radius_m),
                float(self.args.physical_body_radius_m),
                # The measured walls define this local manoeuvre. Reusing the
                # contact returns here can close the very passage just proven
                # open; live collision protection remains active throughout.
                extra_occupied_xy=np.empty((0, 2), dtype=np.float32),
            )
            for point in continuation or []:
                candidate = np.asarray(point, dtype=np.float64)
                if float(np.hypot(*(candidate - route[-1]))) > 0.08:
                    route.append(candidate)
        sparse_count = len(route)
        route = _densify_transition_route(start, route, maximum_step_m=0.25)
        print(
            "[saved-map] committed LiDAR passage manoeuvre: turn/drive "
            f"{float(np.hypot(*plan.approach_body_xy)):.2f}m to the measured "
            f"centre, align {plan.heading_deg:+.1f}deg with the walls, then "
            f"drive {float(np.hypot(*(plan.exit_body_xy - plan.approach_body_xy))):.2f}m "
            "straight through before resuming the saved-map route; "
            f"{len(route)} overlap-preserving keyframes replace "
            f"{sparse_count} long waypoint(s)."
        )
        return route, radius

    def _center_in_live_lidar_passage(self) -> bool:
        """Compatibility probe retained for diagnostics outside recovery."""
        return self._detect_live_lidar_passage() is not None

    def _try_one_sided_clearance_turn(
        self,
        reason: str,
        *,
        persistent_turn_sign: float | None = None,
    ) -> bool:
        """Execute one collision-checked pivot away from a one-sided hit."""
        # Direction comes from the exact points in the fresh confirmed scan,
        # not from human-readable reason text. In particular, a message that
        # says "beyond the front edge at +43deg" is a left-shoulder hit even
        # though the sentence contains the word "front."
        # Five-degree corrections are commonly swallowed by the controller's
        # heading tolerance and leave the same shoulder inside the envelope.
        # A 15-degree local-planner escape is still small, but produces useful
        # lateral clearance before the next straight segment.
        preferred_delta = _current_escape_turn_deg(
            self.pending_collision_sectors,
            persistent_turn_sign,
            step_deg=15.0,
        )
        if preferred_delta is None:
            return False

        # Use the largest safe turn away from the occupied side. A full 15deg
        # sweep can be constrained near a corner even though a useful 5-10deg
        # counter-steer is clear. Never interpret that as a reason to skip the
        # turn-away-first policy and immediately reverse.
        delta = None
        sign = math.copysign(1.0, float(preferred_delta))
        for magnitude in (15.0, 10.0, 5.0):
            candidate = sign * magnitude
            if self._rotation_safety(candidate) is None:
                delta = candidate
                break
        if delta is None:
            return False
        initial_yaw = self.imu.deg()
        if initial_yaw is None:
            return False
        target = float(initial_yaw) + float(delta)
        self.controller.clear_safety_latch()
        self.controller.rotate_to(target)
        deadline = time.monotonic() + 7.0
        reached = False
        while time.monotonic() < deadline and not self.stop_event.is_set():
            yaw = self.imu.deg()
            if yaw is not None and abs(target - float(yaw)) <= 3.0:
                reached = True
                break
            if self.controller.safety_latched_reason():
                break
            time.sleep(0.05)
        self.controller.halt()
        final_yaw = self.imu.deg()
        if not reached or final_yaw is None:
            return False
        actual_delta = float(final_yaw) - float(initial_yaw)
        with self.pose_lock:
            self.pose = _pose_with_imu_heading(
                self.pose,
                self.pose.theta_deg + actual_delta,
                self.lever_m,
                self.forward_offset,
            )
        direction = "clockwise" if delta < 0.0 else "counterclockwise"
        constrained_side = "left" if delta < 0.0 else "right"
        print(
            f"[saved-map] {reason}; only the {constrained_side} side "
            f"is constrained, so a {abs(actual_delta):.1f}deg {direction} "
            "clearance turn completed before replanning."
        )
        self.pending_collision_world = np.empty((0, 2), dtype=np.float32)
        self.pending_collision_sectors = set()
        self.controller.clear_safety_latch()
        return True

    def _try_front_clearance_turn(self, reason: str) -> bool:
        """Make one route-aware pivot when forward motion, but not rotation, is blocked."""
        with self.pose_lock:
            centre = self._centre(self.pose)
            map_heading = float(self.pose.theta_deg + self.forward_offset)
        route_error = 0.0
        with self._collision_diagnostic_lock:
            diagnostic = self._collision_diagnostic
        if (
            diagnostic is not None
            and diagnostic.route_heading_error_deg is not None
            and math.isfinite(float(diagnostic.route_heading_error_deg))
        ):
            route_error = float(diagnostic.route_heading_error_deg)
        if len(self.path) >= 2:
            target = np.asarray(self.path[1], dtype=np.float64)
            delta_xy = target - centre
            if (
                diagnostic is None
                or diagnostic.route_heading_error_deg is None
            ) and float(np.hypot(*delta_xy)) > 1e-6:
                route_heading = math.degrees(math.atan2(delta_xy[1], delta_xy[0]))
                route_error = (route_heading - map_heading + 180.0) % 360.0 - 180.0

        clockwise_clear = self._rotation_safety(-30.0) is None
        counterclockwise_clear = self._rotation_safety(30.0) is None
        preferred = _front_obstacle_escape_turn_deg(
            reason,
            route_error,
            clockwise_clear=clockwise_clear,
            counterclockwise_clear=counterclockwise_clear,
        )
        if preferred is None:
            return False

        # Use the largest safe bounded correction in the selected direction.
        sign = math.copysign(1.0, preferred)
        delta = next(
            (
                sign * magnitude
                for magnitude in (15.0, 10.0, 5.0)
                if self._rotation_safety(sign * magnitude) is None
            ),
            None,
        )
        initial_yaw = self.imu.deg()
        if delta is None or initial_yaw is None:
            return False
        target_yaw = float(initial_yaw) + float(delta)
        self.controller.clear_safety_latch()
        self.controller.rotate_to(target_yaw)
        deadline = time.monotonic() + 7.0
        reached = False
        while time.monotonic() < deadline and not self.stop_event.is_set():
            yaw = self.imu.deg()
            if yaw is not None and abs(target_yaw - float(yaw)) <= 3.0:
                reached = True
                break
            if self.controller.safety_latched_reason():
                break
            time.sleep(0.05)
        self.controller.halt()
        final_yaw = self.imu.deg()
        if not reached or final_yaw is None:
            return False
        actual_delta = float(final_yaw) - float(initial_yaw)
        with self.pose_lock:
            self.pose = _pose_with_imu_heading(
                self.pose,
                self.pose.theta_deg + actual_delta,
                self.lever_m,
                self.forward_offset,
            )
        direction = "clockwise" if actual_delta < 0.0 else "counterclockwise"
        print(
            "[saved-map] forward path is locally occupied but rotation is clear; "
            f"completed one {abs(actual_delta):.1f}deg {direction} avoidance turn "
            f"(route error was {route_error:+.1f}deg), then rescanning before replanning."
        )
        self.pending_collision_world = np.empty((0, 2), dtype=np.float32)
        self.pending_collision_sectors = set()
        self.controller.clear_safety_latch()
        return True

    def _advance_after_clearance_turn(self, distance_m: float = 0.10) -> bool:
        """Move clear in the newly opened heading before global replanning.

        Replanning immediately after a side-escape pivot can ask the chassis to
        rotate straight back into the same shoulder obstruction. Commit a
        short, ordinary collision-guarded forward step first, tracked by LiDAR
        odometry, so the next global route starts outside that local envelope.
        """
        initial_yaw = self.imu.deg()
        if initial_yaw is None:
            return False
        with self.pose_lock:
            start_pose = self.pose
            start_centre = self._centre(start_pose)
        heading_rad = math.radians(start_pose.theta_deg + self.forward_offset)
        forward_axis = np.asarray(
            [math.cos(heading_rad), math.sin(heading_rad)], dtype=np.float64
        )
        side_axis = np.asarray(
            [-forward_axis[1], forward_axis[0]], dtype=np.float64
        )
        after = self.feed.latest()[0]
        drive_command = _forward_command_above_stiction(float(self.args.drive_speed))
        self.controller.clear_safety_latch()
        self.controller.drive_toward(drive_command, float(initial_yaw))
        # This is an escape step, not an open-ended drive leg. Bound command
        # exposure even if scan matching becomes unavailable mid-manoeuvre.
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and not self.stop_event.is_set():
            reason = self.controller.safety_latched_reason()
            if reason:
                self.controller.halt()
                print(
                    f"[saved-map] one-sided clearance advance was blocked by {reason}."
                )
                return False
            frame_id, frame = self.feed.wait_for_frame_after(
                after_frame_id=after,
                timeout_s=0.8,
                min_frame_advances=1,
            )
            if frame is None:
                continue
            after = int(frame_id)
            local = _scan_local(frame, self.args)
            if len(local) < 30:
                continue
            yaw = _imu_yaw_for_scan(self.imu, frame)
            with self.pose_lock:
                current_pose = self.pose
            theta_seed = current_pose.theta_deg + (
                (float(yaw) - float(initial_yaw)) if yaw is not None else 0.0
            )
            seed = _pose_with_imu_heading(
                current_pose,
                theta_seed,
                self.lever_m,
                self.forward_offset,
            )
            solved, score, support = _localize_against_points(
                local,
                self.local_odometry.reference(),
                seed,
                self.args,
                0.45,
                4.0,
            )
            solved = _pose_with_imu_heading(
                solved,
                theta_seed,
                self.lever_m,
                self.forward_offset,
            )
            current_centre = self._centre(current_pose)
            solved_centre = self._centre(solved)
            scan_step = solved_centre - current_centre
            # A single LiDAR revolution cannot represent a 30-45cm chassis
            # translation at this robot's commanded speed. Such a solution is
            # the matcher sliding along a repeated wall, not physical motion.
            if float(np.hypot(*scan_step)) > 0.16:
                continue
            if (
                score < max(4.0, float(self.args.localization_min_score) - 2.0)
                or support < 0.18
            ):
                continue
            displacement = solved_centre - start_centre
            progress = float(np.dot(displacement, forward_axis))
            cross_track = abs(float(np.dot(displacement, side_axis)))
            if progress < -0.02 or cross_track > 0.10:
                continue
            with self.pose_lock:
                self.pose = solved
            self.local_odometry.add(local, solved)
            if progress >= float(distance_m):
                self.controller.halt()
                print(
                    f"[saved-map] one-sided clearance advance completed "
                    f"{progress:.2f}m before global replanning."
                )
                return True
        self.controller.halt()
        print(
            "[saved-map] one-sided clearance advance did not produce verified "
            "forward progress; falling back to obstacle-costmap replanning."
        )
        return False

    def _replan_after_collision(
        self,
        reason: str,
        backup_failures: int,
    ) -> tuple[list[np.ndarray] | None, int]:
        """Map a live hit, evaluate safe local escapes, and globally replan.

        A collision-free motion is not automatically a useful recovery. Each
        turn/advance rollout must also leave a saved-map route whose first
        segment agrees with the new chassis heading and whose remaining cost
        has not grown. This prevents several individually safe shoulder
        escapes from walking the robot away from its route and into the wall
        on the opposite side.
        """
        self.controller.halt()
        if self.goal is None:
            return None, backup_failures

        # Freeze a local observation before choosing a recovery motion. The
        # former turn/drive-before-planning fast path accumulated individually
        # safe pivots until the chassis was 155deg away from the route and
        # facing a wall. The progress-aware candidates below are allowed one
        # bounded manoeuvre only after its resulting route has been scored.
        recovery_collision_sectors = set(self.pending_collision_sectors)
        pending = np.asarray(
            self.pending_collision_world, dtype=np.float32
        ).reshape((-1, 2))
        immediate_turn = _one_sided_escape_turn_deg(
            recovery_collision_sectors,
            step_deg=15.0,
        )
        edge_escape_pending = False
        if self._edge_escape_anchor is not None:
            with self.pose_lock:
                edge_now = self._centre(self.pose)
            edge_progress = float(
                np.hypot(*(edge_now - self._edge_escape_anchor))
            )
            if edge_progress >= 0.16:
                self._edge_escape_anchor = None
            else:
                edge_escape_pending = True
                immediate_turn = None
                # Prevent every later one-sided fallback in this recovery
                # from issuing another turn around the same physical edge.
                recovery_collision_sectors = {"left", "right"}
                print(
                    "[saved-map] repeated edge trigger suppressed after only "
                    f"{edge_progress:.2f}m localized progress; holding for a "
                    "stationary replan instead of accumulating another turn."
                )

        # SIMPLE EDGE AVOIDANCE: a predictive hit on exactly one shoulder is
        # not a localization or passage-planning problem. Stop, pivot once
        # away from that side, verify the new forward corridor from fresh
        # LiDAR, and commit a short straight segment in that heading. Planning
        # from the end of that segment prevents the global follower from
        # immediately undoing the turn and steering back into the same edge.
        if (
            reason.startswith("predictive LiDAR path stop")
            and immediate_turn is not None
            and not edge_escape_pending
        ):
            occupied_side = "left" if immediate_turn < 0.0 else "right"
            direction = "clockwise" if immediate_turn < 0.0 else "counterclockwise"
            print(
                f"[saved-map] predictive edge on the {occupied_side}; turning "
                f"{direction} once, then continuing straight in the verified "
                "open heading."
            )
            if self._try_one_sided_clearance_turn(
                reason,
                persistent_turn_sign=math.copysign(1.0, float(immediate_turn)),
            ):
                time.sleep(0.12)
                live_reason = self._translation_safety()
                if live_reason is None:
                    with self.pose_lock:
                        edge_start = self._centre(self.pose)
                        edge_heading = math.radians(
                            self.pose.theta_deg + self.forward_offset
                        )
                    edge_exit = edge_start + 0.22 * np.asarray(
                        [math.cos(edge_heading), math.sin(edge_heading)],
                        dtype=np.float64,
                    )
                    continuation, edge_radius = _plan_saved_map_route(
                        self.world,
                        edge_exit,
                        np.asarray(self.goal),
                        float(self.args.robot_radius_m),
                        float(self.args.physical_body_radius_m),
                        extra_occupied_xy=self.dynamic_obstacles_world,
                    )
                    if not continuation:
                        continuation, edge_radius = _plan_saved_map_route(
                            self.world,
                            edge_exit,
                            np.asarray(self.goal),
                            float(self.args.robot_radius_m),
                            float(self.args.physical_body_radius_m),
                            extra_occupied_xy=np.empty((0, 2), dtype=np.float32),
                        )
                    if continuation:
                        edge_route = [
                            edge_exit,
                            *[
                                np.asarray(point, dtype=np.float64)
                                for point in continuation
                            ],
                        ]
                        self.pending_collision_world = np.empty(
                            (0, 2), dtype=np.float32
                        )
                        self.pending_collision_sectors = set()
                        self.path = [edge_start.copy(), *edge_route]
                        self._edge_escape_anchor = edge_start.copy()
                        self.status_text = (
                            "Edge avoided—continuing through the open side"
                        )
                        self.controller.clear_safety_latch()
                        print(
                            "[saved-map] edge turn exposed a clear forward "
                            "corridor; committed 0.22m straight before the "
                            f"remaining {len(continuation)} waypoint(s) at "
                            f"radius {edge_radius:.2f}m."
                        )
                        return edge_route, 0
                else:
                    print(
                        "[saved-map] the one edge-avoidance turn exposed a "
                        f"different live obstruction ({live_reason}); using "
                        "the stationary local planner instead of turning again."
                    )
        print(
            "[saved-map] collision prediction confirmed; holding position "
            "for stationary localization and local LiDAR replanning."
        )
        self._stationary_saved_map_correction()
        with self.pose_lock:
            corrected_centre = self._centre(self.pose)
        if float(np.hypot(*(np.asarray(self.goal) - corrected_centre))) <= 0.22:
            self.path = []
            self.goal = None
            self.status_text = "Goal reached—click another known free point"
            print(
                "[saved-map] stationary LiDAR consensus confirms the robot is "
                "already at the requested destination; collision recovery cancelled."
            )
            return None, 0
        stationary_points = self._stationary_lidar_observation()
        # Rotation helpers clear the live latch after a successful turn. Keep
        # the collision classification that caused this recovery so later
        # policy decisions cannot mistake a cleared latch for "there was no
        # one-sided obstacle" and recommit the same blocked route.
        if len(pending):
            if len(self.dynamic_obstacles_world):
                self.dynamic_obstacles_world = np.concatenate(
                    [self.dynamic_obstacles_world, pending], axis=0
                )[-300:]
            else:
                self.dynamic_obstacles_world = pending[-300:].copy()

        def _route_metrics(
            origin: np.ndarray,
            candidate_route: list[np.ndarray],
            body_heading_deg: float,
        ) -> tuple[float, float]:
            points = [np.asarray(origin, dtype=np.float64)]
            points.extend(np.asarray(point, dtype=np.float64) for point in candidate_route)
            route_cost = sum(
                float(np.hypot(*(b - a)))
                for a, b in zip(points, points[1:], strict=False)
            )
            first_vector = next(
                (point - points[0] for point in points[1:] if np.hypot(*(point - points[0])) > 0.05),
                np.zeros(2, dtype=np.float64),
            )
            if not np.any(first_vector):
                return route_cost, 0.0
            first_heading = math.degrees(math.atan2(first_vector[1], first_vector[0]))
            heading_error = abs(
                (first_heading - float(body_heading_deg) + 180.0) % 360.0 - 180.0
            )
            return route_cost, heading_error

        with self.pose_lock:
            recovery_start = self._centre(self.pose)
            recovery_heading = self.pose.theta_deg + self.forward_offset
        baseline_route, _ = _plan_saved_map_route(
            self.world,
            recovery_start,
            np.asarray(self.goal),
            float(self.args.robot_radius_m),
            float(self.args.physical_body_radius_m),
            extra_occupied_xy=self.dynamic_obstacles_world,
        )
        baseline_cost = math.inf
        if baseline_route:
            baseline_cost, _ = _route_metrics(
                recovery_start, baseline_route, recovery_heading
            )

        def _passage_commit_made_progress(centre_xy: np.ndarray) -> bool:
            """Forbid replaying one passage manoeuvre from the same pose."""
            centre_xy = np.asarray(centre_xy, dtype=np.float64)
            previous = self._last_passage_commit_centre
            if previous is not None:
                displacement = float(np.hypot(*(centre_xy - previous)))
                if displacement < 0.12:
                    print(
                        "[saved-map] passage replay suppressed: the previous "
                        f"fixed manoeuvre produced only {displacement:.2f}m of "
                        "localized progress. Selecting a different local "
                        "escape instead of issuing the same drive again."
                    )
                    return False
            self._last_passage_commit_centre = centre_xy.copy()
            return True

        # PASSAGE RECOVERY (primary path): get enough stand-off for one clean
        # stationary observation, measure both walls, and commit the complete
        # centre-align-straight manoeuvre before moving again.  Previously this
        # happened only after several blind side escapes; the run could measure
        # a valid 1m-wide corridor and then discard it in favour of more
        # collision-triggered guesses.
        if len(pending):
            # First inspect the pose where the contact was reported. A tight
            # corridor may have ample forward clearance while its rear swept
            # footprint may not have room to reverse. Passage detection never
            # depends on reverse motion.
            observed = self._refresh_local_obstacles_from_lidar(stationary_points)
            passage = self._detect_live_lidar_passage(stationary_points)
            if passage is not None:
                passage_route, passage_radius = self._route_through_live_lidar_passage(
                    passage
                )
                with self.pose_lock:
                    passage_start = self._centre(self.pose)
                if not _passage_commit_made_progress(passage_start):
                    passage = None
            if passage is not None:
                # These points were accumulated while rubbing along the two
                # jambs and can form a false solid plug when transformed with
                # a drifting contact pose. The immutable saved map still owns
                # global geometry; the fixed passage route and live envelope
                # own this local traversal.
                self.dynamic_obstacles_world = np.empty((0, 2), dtype=np.float32)
                self.pending_collision_world = np.empty((0, 2), dtype=np.float32)
                self.pending_collision_sectors = set()
                self.collision_replan_streak = 0
                self._passage_route_active = True
                self.path = [
                    passage_start.copy(),
                    *[np.asarray(point).copy() for point in passage_route],
                ]
                self.status_text = "LiDAR passage measured—centering and driving straight"
                print(
                    f"[saved-map] stationary passage scan added {observed} live "
                    f"return(s); executing {len(passage_route)} fixed waypoint(s) "
                    f"at planning radius {passage_radius:.2f}m instead of reactive "
                    "collision recovery."
                )
                self.controller.clear_safety_latch()
                return passage_route, backup_failures + 1
            print(
                "[saved-map] stationary LiDAR found no paired passage walls; "
                "using ordinary one-sided obstacle recovery for this obstacle."
            )
            self._passage_route_active = False

        # If a predictive stop was not classifiable as one-sided above, let
        # the stationary planner handle front/both-side geometry.  Never
        # recommit an unchanged route for a one-sided prediction: that was the
        # expensive stop/replan/stop loop removed by the fast local policy.
        if (
            reason.startswith("predictive LiDAR path stop")
            and len(stationary_points)
            and immediate_turn is None
        ):
            with self.pose_lock:
                stationary_start = self._centre(self.pose)
            stationary_route, stationary_radius = _plan_saved_map_route(
                self.world,
                stationary_start,
                np.asarray(self.goal),
                float(self.args.robot_radius_m),
                float(self.args.physical_body_radius_m),
                extra_occupied_xy=self.dynamic_obstacles_world,
            )
            if stationary_route:
                self.pending_collision_world = np.empty((0, 2), dtype=np.float32)
                self.pending_collision_sectors = set()
                self.path = [
                    stationary_start.copy(),
                    *[np.asarray(point).copy() for point in stationary_route],
                ]
                self.status_text = (
                    "Stationary LiDAR replan completeâ€”executing the new route"
                )
                print(
                    f"[saved-map] stationary predictive-stop replan committed "
                    f"{len(stationary_route)} waypoint(s) at radius "
                    f"{stationary_radius:.2f}m; no reactive clearance advance "
                    "was issued."
                )
                self.controller.clear_safety_latch()
                return stationary_route, backup_failures + 1
            print(
                "[saved-map] stationary predictive-stop costmap found no "
                "connected route; only now entering bounded local recovery."
            )
            predicted_escape = _one_sided_escape_turn_deg(
                recovery_collision_sectors,
                step_deg=15.0,
            )
            if predicted_escape is not None:
                escape_sign = math.copysign(1.0, float(predicted_escape))
                direction = "clockwise" if escape_sign < 0.0 else "counterclockwise"
                print(
                    f"[saved-map] predictive obstacle is one-sided; the "
                    f"{direction} sweep is clear, so turning away and making "
                    "one short clearance advance before replanning."
                )
                if self._try_one_sided_clearance_turn(
                    reason,
                    persistent_turn_sign=escape_sign,
                ):
                    # Do not immediately hand the new heading back to A*. In
                    # the field failure A* selected the same shortest first
                    # segment, so the follower undid this safe turn and drove
                    # back toward the very same shoulder four times. Translate
                    # far enough to leave that local envelope first; the move
                    # is continuously guarded by the ordinary hard collision
                    # box and tracked by rolling LiDAR odometry.
                    advanced = self._advance_after_clearance_turn(0.14)
                    if not advanced:
                        print(
                            "[saved-map] predictive clearance heading was safe "
                            "but did not yield a verified forward displacement; "
                            "the new stationary scan will decide the fallback."
                        )
                    post_turn_points = self._stationary_lidar_observation()
                    # The pre-turn projection can look like a solid plug once
                    # inflated. Rebuild the temporary layer in the new robot
                    # frame instead of carrying that obsolete plug through the
                    # throat.
                    self.dynamic_obstacles_world = np.empty((0, 2), dtype=np.float32)
                    self._refresh_local_obstacles_from_lidar(post_turn_points)
                    with self.pose_lock:
                        turned_start = self._centre(self.pose)
                        turned_body_heading = (
                            self.pose.theta_deg + self.forward_offset
                        )
                    turned_route, turned_radius = _plan_saved_map_route(
                        self.world,
                        turned_start,
                        np.asarray(self.goal),
                        float(self.args.robot_radius_m),
                        float(self.args.physical_body_radius_m),
                        extra_occupied_xy=self.dynamic_obstacles_world,
                    )
                    if not turned_route:
                        # The immutable map already describes the corridor;
                        # live predictive safety remains authoritative for any
                        # deviation. Do not let a conservative temporary layer
                        # veto a direction whose real rotation and forward
                        # rollout are both observed clear.
                        turned_route, turned_radius = _plan_saved_map_route(
                            self.world,
                            turned_start,
                            np.asarray(self.goal),
                            float(self.args.robot_radius_m),
                            float(self.args.physical_body_radius_m),
                            extra_occupied_xy=np.empty((0, 2), dtype=np.float32),
                        )
                    if turned_route and not advanced:
                        # The just-commanded forward probe was rejected by
                        # live LiDAR.  Accepting any global route from the same
                        # pose simply asks the follower to issue that rejected
                        # translation again (the repeated corner-ramming loop
                        # seen in the 18:14 field run).  A route is eligible
                        # only after this escape direction has produced real,
                        # localized displacement.
                        print(
                            "[saved-map] rejected post-turn route because the "
                            "live-LiDAR forward probe made no progress; the "
                            "same corner approach will not be recommitted."
                        )
                        turned_route = None
                    if turned_route:
                        turned_cost, turned_heading_error = _route_metrics(
                            turned_start,
                            turned_route,
                            turned_body_heading,
                        )
                        # A collision recovery turn establishes which local
                        # direction is physically clear.  Do not immediately
                        # accept an A* route whose first segment points back
                        # through the obstacle that caused the stop.  The
                        # 2026-07-30 field failure accepted a 160.9deg first
                        # segment here, undid the clockwise escape, and drove
                        # the chassis into the corner until both shoulders
                        # were genuinely constrained.
                        cost_ok = (
                            not math.isfinite(baseline_cost)
                            or turned_cost <= baseline_cost + 0.15
                        )
                        heading_ok = turned_heading_error <= 75.0
                        if not cost_ok or not heading_ok:
                            print(
                                "[saved-map] rejected post-turn route that "
                                "would undo the verified escape: remaining "
                                f"{turned_cost:.2f}m vs {baseline_cost:.2f}m, "
                                "first-segment heading error "
                                f"{turned_heading_error:.1f}deg. Rescanning "
                                "from the safe heading instead of turning "
                                "back into the corner."
                            )
                            turned_route = None
                    if turned_route:
                        self.pending_collision_world = np.empty(
                            (0, 2), dtype=np.float32
                        )
                        self.pending_collision_sectors = set()
                        self.path = [
                            turned_start.copy(),
                            *[
                                np.asarray(point).copy()
                                for point in turned_route
                            ],
                        ]
                        self.status_text = (
                            "Aligned away from the predicted edgeâ€”following "
                            "the rescanned route"
                        )
                        print(
                            f"[saved-map] one-sided predictive recovery turned "
                            f"{direction}, "
                            f"{'advanced clear, ' if advanced else ''}"
                            "rescanned while stopped, and "
                            f"committed {len(turned_route)} waypoint(s) at "
                            f"radius {turned_radius:.2f}m."
                        )
                        self.controller.clear_safety_latch()
                        return turned_route, backup_failures + 1

        # Local collision avoidance owns the first manoeuvre. If exactly one
        # side is occupied, turn away from it and test a short forward rollout.
        # Both operations are guarded by fresh LiDAR, and the rollout is only
        # committed when a progress-aware global route accepts its endpoint.
        directional_escape = False
        accepted_route: list[np.ndarray] | None = None
        accepted_radius = float(self.args.robot_radius_m)
        initial_escape = _one_sided_escape_turn_deg(
            recovery_collision_sectors,
            step_deg=15.0,
        )
        lateral_sign = None
        if recovery_collision_sectors == {"right"}:
            lateral_sign = 1.0
        elif recovery_collision_sectors == {"left"}:
            lateral_sign = -1.0
        lateral_recovery_owned = False
        if len(pending) and lateral_sign is not None:
            side_name = "right" if lateral_sign > 0.0 else "left"
            print(
                f"[saved-map] one-sided {side_name} contact; trying a direct "
                "LiDAR-guarded mecanum strafe away from it."
            )
            # The first 8-12cm step can clear the physical shoulder while the
            # inflated saved-map costmap is still disconnected. Continue in
            # the already verified open lateral direction instead of throwing
            # that successful recovery away and beginning a turning sequence.
            for strafe_attempt in range(1, 5):
                if not self._strafe_away_from_side(lateral_sign, 0.10):
                    break
                lateral_recovery_owned = True
                with self.pose_lock:
                    strafe_start = self._centre(self.pose)
                    strafe_heading = self.pose.theta_deg + self.forward_offset
                strafe_route, strafe_radius = _plan_saved_map_route(
                    self.world,
                    strafe_start,
                    np.asarray(self.goal),
                    float(self.args.robot_radius_m),
                    float(self.args.physical_body_radius_m),
                    extra_occupied_xy=self.dynamic_obstacles_world,
                )
                if strafe_route:
                    strafe_cost, heading_error = _route_metrics(
                        strafe_start, strafe_route, strafe_heading
                    )
                    cost_ok = (
                        not math.isfinite(baseline_cost)
                        or strafe_cost <= baseline_cost + 0.05
                    )
                    if cost_ok and heading_error <= 55.0:
                        directional_escape = True
                        accepted_route = strafe_route
                        accepted_radius = strafe_radius
                        print(
                            "[saved-map] progress-aware lateral escape accepted: "
                            f"remaining route {strafe_cost:.2f}m, first-segment "
                            f"heading error {heading_error:.1f}deg."
                        )
                        break
                    else:
                        print(
                            "[saved-map] lateral displacement was safe but its "
                            "route did not yet pass the progress check; "
                            f"continuing the same open-side strafe "
                            f"({strafe_attempt}/4)."
                        )
                else:
                    print(
                        "[saved-map] lateral displacement was safe but did not "
                        "yet leave a connected saved-map route; continuing the "
                        f"same open-side strafe ({strafe_attempt}/4)."
                    )
        if (
            len(pending)
            and initial_escape is not None
            and accepted_route is None
        ):
            if lateral_recovery_owned:
                print(
                    "[saved-map] lateral escape moved away from the contacted "
                    "side but did not reconnect a route; now executing the "
                    "promised turn-away fallback."
                )
            escape_sign = math.copysign(1.0, float(initial_escape))
            # Keep the original open-side decision. A failed forward probe can
            # see the opposite wall and used to reverse the next turn, causing
            # left/right dithering. Continue around the obstruction in safe
            # increments and probe forward after each increment.
            # One counter-steer/forward rollout is sufficient to test the
            # obvious open-side escape. If it fails, stop and rescan in place;
            # repeatedly executing rejected rollouts physically walks the
            # robot away from the saved route before they can be scored.
            for attempt in range(1, 2):
                if not self._try_one_sided_clearance_turn(
                    reason,
                    persistent_turn_sign=escape_sign,
                ):
                    print(
                        "[saved-map] LiDAR-local avoidance found no further "
                        "safe turn in the chosen open-side direction."
                    )
                    break
                if self._advance_after_clearance_turn(0.12):
                    with self.pose_lock:
                        rollout_start = self._centre(self.pose)
                        rollout_heading = self.pose.theta_deg + self.forward_offset
                    rollout_route, rollout_radius = _plan_saved_map_route(
                        self.world,
                        rollout_start,
                        np.asarray(self.goal),
                        float(self.args.robot_radius_m),
                        float(self.args.physical_body_radius_m),
                        extra_occupied_xy=self.dynamic_obstacles_world,
                    )
                    if rollout_route:
                        rollout_cost, heading_error = _route_metrics(
                            rollout_start, rollout_route, rollout_heading
                        )
                        # Twelve centimetres of local avoidance may be nearly
                        # tangent to the goal, but it must not lengthen the
                        # remaining route or require an immediate turn back
                        # into the obstacle it just escaped.
                        cost_ok = (
                            not math.isfinite(baseline_cost)
                            or rollout_cost <= baseline_cost + 0.05
                        )
                        heading_ok = heading_error <= 55.0
                        if cost_ok and heading_ok:
                            directional_escape = True
                            accepted_route = rollout_route
                            accepted_radius = rollout_radius
                            print(
                                "[saved-map] progress-aware LiDAR escape "
                                f"accepted: remaining route {rollout_cost:.2f}m, "
                                f"first-segment heading error {heading_error:.1f}deg."
                            )
                            break
                        print(
                            "[saved-map] safe forward probe rejected by route "
                            f"progress check (remaining {rollout_cost:.2f}m vs "
                            f"{baseline_cost:.2f}m, heading error "
                            f"{heading_error:.1f}deg); rescanning in place "
                            "instead of repeating the deviation."
                        )
                    else:
                        print(
                            "[saved-map] safe forward probe has no connected "
                            "saved-map route; rescanning in place."
                        )
                    continue
                print(
                    "[saved-map] forward probe remained occupied after "
                    f"open-side turn {attempt}/1; rescanning in place."
                )
        self.pending_collision_world = np.empty((0, 2), dtype=np.float32)
        self.pending_collision_sectors = set()
        with self.pose_lock:
            start = self._centre(self.pose)
        if accepted_route is not None:
            route, radius = accepted_route, accepted_radius
        else:
            route, radius = _plan_saved_map_route(
                self.world,
                start,
                np.asarray(self.goal),
                float(self.args.robot_radius_m),
                float(self.args.physical_body_radius_m),
                extra_occupied_xy=self.dynamic_obstacles_world,
            )
        if route and len(pending) and (
            directional_escape
            or (
                initial_escape is None
                and not lateral_recovery_owned
                and self.collision_replan_streak < 2
            )
        ):
            self.collision_replan_streak += 1
            self.path = [start.copy(), *[np.asarray(point).copy() for point in route]]
            self.status_text = (
                "Turned away from obstacle—following replanned route"
                if directional_escape
                else "Live obstacle mapped—following clearance route"
            )
            print(
                f"[saved-map] {reason}; committed {len(pending)} confirmed live "
                f"return(s) to the temporary costmap and planned a "
                f"{len(route)}-waypoint clearance route at radius {radius:.2f}m "
                + (
                    "after the counter-steer/advance escape."
                    if directional_escape
                    else f"(no-progress replan {self.collision_replan_streak}/2)."
                )
            )
            self.controller.clear_safety_latch()
            return route, 0

        if route and len(pending):
            print(
                "[saved-map] oscillation guard: two clearance routes were "
                "blocked before any waypoint progress; taking a new stationary "
                "LiDAR passage measurement without reversing."
            )

        # Reverse recovery is intentionally absent. Reversing repeatedly moved
        # the robot out of a correctly measured throat and caused the same
        # passage to be reacquired from alternating poses. After the permitted
        # strafe/turn manoeuvre, take one more stationary look and either commit
        # a new centreline or report that the hard box still blocks motion.
        observed = self._refresh_local_obstacles_from_lidar()
        passage = self._detect_live_lidar_passage()
        if passage is not None:
            passage_route, passage_radius = self._route_through_live_lidar_passage(
                passage
            )
            with self.pose_lock:
                passage_start = self._centre(self.pose)
            if not _passage_commit_made_progress(passage_start):
                passage = None
        if passage is not None:
            self.dynamic_obstacles_world = np.empty((0, 2), dtype=np.float32)
            self._passage_route_active = True
            self.path = [
                passage_start.copy(),
                *[np.asarray(point).copy() for point in passage_route],
            ]
            self.status_text = "LiDAR passage measured—centering and driving straight"
            print(
                f"[saved-map] post-turn/strafe scan added {observed} live return(s) "
                f"and committed {len(passage_route)} passage waypoint(s) at radius "
                f"{passage_radius:.2f}m; reverse recovery is disabled."
            )
            self.controller.clear_safety_latch()
            return passage_route, backup_failures

        # Recovery may already have turned the old obstacle out of the hard
        # envelope. Never terminate on the historical latch in that state.
        # Re-evaluate the current robot-frame scan and, when it is clear,
        # resume from the pose reached by the collision-checked manoeuvre. The
        # old red points remain available only as diagnostic evidence.
        current_reason = self._translation_safety()
        # A front prediction is not a geometric dead end when either pivot
        # direction is clear.  The old final loop below understands only a
        # single occupied side, so a front-only sector fell straight through
        # to the terminal "could not establish a safe route" message.  Make
        # at most two bounded, freshly rescanned avoidance turns here.  This
        # covers a visible jamb/corner without creating another endless-spin
        # recovery mode.
        for front_attempt in range(1, 3):
            live_sectors = set(self.pending_collision_sectors)
            front_only = (
                current_reason is not None
                and "front" in live_sectors
                and "left" not in live_sectors
                and "right" not in live_sectors
            )
            if not front_only:
                break
            if not self._try_front_clearance_turn(current_reason):
                break
            time.sleep(0.12)
            current_reason = self._translation_safety()
            if current_reason is None:
                print(
                    "[saved-map] bounded front-obstacle turn exposed clear "
                    "forward space; replanning from the fresh pose."
                )
                break
            print(
                f"[saved-map] front-obstacle avoidance turn {front_attempt}/2 "
                "still sees a forward obstruction; rescanning once more "
                "instead of declaring the clear pivot directions unusable."
            )
        # A one-sided live obstruction is never a terminal state. Re-read the
        # robot-frame scan after every small pivot and let the *current* side
        # choose the next direction. In particular, when a right-side escape
        # exposes a left-side edge, the next action must be clockwise; it must
        # not preserve the now-stale counterclockwise preference. This loop is
        # bounded only to prevent an actuator/sensor fault from commanding an
        # endless pivot. A normal geometric dead end presents both sides and
        # exits immediately without guessing.
        # One freshly observed primitive is permitted here. Repeating twelve
        # locally clear pivots was the direct cause of the wall-facing spiral.
        final_escape_turn: float | None = None
        for turn_attempt in range(1, 2):
            live_sectors = set(self.pending_collision_sectors)
            live_turn = _current_escape_turn_deg(
                live_sectors,
                None,
                step_deg=15.0,
            )
            if current_reason is None or live_turn is None:
                break
            occupied_side = "left" if live_turn < 0.0 else "right"
            direction = "clockwise" if live_turn < 0.0 else "counterclockwise"
            print(
                f"[saved-map] final one-sided escape {turn_attempt}/1: "
                f"fresh LiDAR shows {occupied_side} occupied; trying the "
                f"{direction} direction before any stop is permitted."
            )
            if not self._try_one_sided_clearance_turn(current_reason):
                print(
                    "[saved-map] the turn away from the currently occupied "
                    "side has no collision-free 5deg sweep."
                )
                break
            final_escape_turn = float(live_turn)
            time.sleep(0.12)
            current_reason = self._translation_safety()
            if current_reason is None:
                print(
                    "[saved-map] repeated live-LiDAR turn-away recovery "
                    "established clear forward space; replanning now."
                )
                break
        # A clear scan immediately after a pivot is not permission to undo the
        # pivot. The old code handed control straight back to A*, whose first
        # waypoint repeatedly turned left into the desk that the clockwise
        # escape had just avoided. Establish a small amount of verified
        # translation in the open heading before accepting any new route.
        escape_advanced = True
        if current_reason is None and final_escape_turn is not None:
            escape_advanced = self._advance_after_clearance_turn(0.10)
            if not escape_advanced:
                print(
                    "[saved-map] clear pivot did not produce verified forward "
                    "separation; refusing to turn back toward the previous "
                    "contact side."
                )

        if current_reason is None:
            with self.pose_lock:
                clear_start = self._centre(self.pose)
                clear_heading = self.pose.theta_deg + self.forward_offset
            clear_route, clear_radius = _plan_saved_map_route(
                self.world,
                clear_start,
                np.asarray(self.goal),
                float(self.args.robot_radius_m),
                float(self.args.physical_body_radius_m),
                extra_occupied_xy=self.dynamic_obstacles_world,
            )
            if not clear_route:
                # Collision points projected through a contact pose can close
                # a narrow throat after the robot has visibly cleared it. The
                # immutable map still supplies global geometry and live LiDAR
                # remains the hard stop authority, so retry without only the
                # temporary obstacle overlay.
                clear_route, clear_radius = _plan_saved_map_route(
                    self.world,
                    clear_start,
                    np.asarray(self.goal),
                    float(self.args.robot_radius_m),
                    float(self.args.physical_body_radius_m),
                    extra_occupied_xy=np.empty((0, 2), dtype=np.float32),
                )
            if clear_route:
                clear_cost, clear_heading_error = _route_metrics(
                    clear_start,
                    clear_route,
                    clear_heading,
                )
                clear_first_vector = next(
                    (
                        np.asarray(point, dtype=np.float64) - clear_start
                        for point in clear_route
                        if float(
                            np.hypot(
                                *(
                                    np.asarray(point, dtype=np.float64)
                                    - clear_start
                                )
                            )
                        )
                        > 0.05
                    ),
                    np.zeros(2, dtype=np.float64),
                )
                clear_signed_heading_error = 0.0
                if np.any(clear_first_vector):
                    clear_first_heading = math.degrees(
                        math.atan2(clear_first_vector[1], clear_first_vector[0])
                    )
                    clear_signed_heading_error = (
                        clear_first_heading - clear_heading + 180.0
                    ) % 360.0 - 180.0
                clear_cost_ok = (
                    not math.isfinite(baseline_cost)
                    or clear_cost <= baseline_cost + 0.15
                )
                clear_heading_ok = clear_heading_error <= (
                    35.0 if final_escape_turn is not None else 55.0
                )
                continues_escape = (
                    escape_advanced
                    or final_escape_turn is None
                    or abs(clear_signed_heading_error) <= 3.0
                    or math.copysign(1.0, clear_signed_heading_error)
                    == math.copysign(1.0, final_escape_turn)
                )
                required_sweep_clear = (
                    abs(clear_signed_heading_error) <= 3.0
                    or self._rotation_safety(clear_signed_heading_error) is None
                )
                if (
                    not clear_cost_ok
                    or not clear_heading_ok
                    or not continues_escape
                    or not required_sweep_clear
                ):
                    print(
                        "[saved-map] rejected clear-scan route that would undo "
                        "the collision escape: remaining "
                        f"{clear_cost:.2f}m vs {baseline_cost:.2f}m, "
                        "first-segment heading error "
                        f"{clear_signed_heading_error:+.1f}deg, "
                        f"same-direction={continues_escape}, "
                        f"sweep-clear={required_sweep_clear}."
                    )
                    clear_route = None
                elif not escape_advanced:
                    print(
                        "[saved-map] forward separation was not measurable, "
                        "but the new route continues in the verified escape "
                        f"direction ({clear_signed_heading_error:+.1f}deg) and "
                        "its complete rotation sweep is clear; accepting it."
                    )
            if clear_route:
                self.pending_collision_world = np.empty((0, 2), dtype=np.float32)
                self.pending_collision_sectors = set()
                self.collision_replan_streak = 0
                self.path = [
                    clear_start.copy(),
                    *[np.asarray(point).copy() for point in clear_route],
                ]
                self.status_text = "Current LiDAR clearâ€”resuming the route"
                print(
                    "[saved-map] the current robot-frame LiDAR is CLEAR; "
                    f"the old latched collision came from an earlier scan. "
                    f"Resuming a {len(clear_route)}-waypoint route at radius "
                    f"{clear_radius:.2f}m instead of stopping on stale evidence."
                )
                self.controller.clear_safety_latch()
                return clear_route, 0

        self.status_text = (
            "LiDAR rotate/strafe recovery could not establish a safe route: "
            f"{current_reason or reason}"
        )
        print(f"[saved-map] {self.status_text}.")
        return None, backup_failures

    def relocalize(self) -> None:
        if self.navigation_thread is not None and self.navigation_thread.is_alive():
            self.status_text = "Relocalization already in progress—collecting fresh LiDAR scans..."
            self.status.configure(text=self.status_text)
            return
        self.localization_attempt += 1
        self.stop_event.clear()
        self.localized = False
        self.status_text = (
            f"Relocalizing (attempt {self.localization_attempt})—"
            "collecting a complete 360deg set of stationary LiDAR views..."
        )
        self.status.configure(text=self.status_text)
        self.root.update_idletasks()
        self.navigation_thread = threading.Thread(target=self._localize_worker, daemon=True)
        self.navigation_thread.start()

    def _localize_worker(self, clearance_relocations: int = 0) -> None:
        self.controller.halt()
        self.localized = False
        attempt = self.localization_attempt
        self.status_text = f"Relocalizing (attempt {attempt})—acquiring angular LiDAR views..."
        reference = self.world.reference()
        initial_yaw = self.imu.deg()
        if initial_yaw is None:
            self.status_text = "IMU unavailable—active localization cannot measure scan angles"
            return

        captures: list[tuple[np.ndarray, float]] = []
        after = self.feed.latest()[0]

        def capture_view() -> bool:
            nonlocal after
            frame_id, frame = self.feed.wait_for_frame_after(
                after_frame_id=after,
                timeout_s=1.8,
                min_frame_advances=1,
            )
            if frame is None:
                return False
            after = int(frame_id)
            local = _scan_local(frame, self.args)
            yaw = _imu_yaw_for_scan(self.imu, frame)
            if len(local) < 30 or yaw is None:
                return False
            captures.append((local, float(yaw) - float(initial_yaw)))
            return True

        capture_view()
        # Choose one safe direction, then acquire a complete monotonic sweep. Unlike the
        # explorer, saved-map localization has no trusted pose yet; all motion
        # safety is therefore evaluated directly in the calibrated body frame.
        step_deg = 30.0
        direction = 0.0
        if self._rotation_safety(step_deg) is None:
            direction = 1.0
        elif self._rotation_safety(-step_deg) is None:
            direction = -1.0
        else:
            self.status_text = (
                "Active localization cannot begin: calibrated footprint blocks "
                "both 30deg pivot directions"
            )
            return

        for step in range(1, 13):
            target = float(initial_yaw) + direction * step_deg * step
            self.pending_collision_sectors = set()
            self.controller.clear_safety_latch()
            self.controller.rotate_to(target)
            deadline = time.monotonic() + 8.0
            reached = False
            while time.monotonic() < deadline and not self.stop_event.is_set():
                yaw = self.imu.deg()
                if yaw is not None and abs(target - float(yaw)) <= 3.0:
                    reached = True
                    break
                if self.controller.safety_latched_reason():
                    break
                time.sleep(0.05)
            self.controller.halt()
            if not reached:
                stop_reason = self.controller.safety_latched_reason()
                collision_sectors = set(self.pending_collision_sectors)
                # The real-time controller can latch between the safety
                # callback's reason update and the UI diagnostic snapshot.
                # Preserve the side encoded in that authoritative latch so a
                # scheduling race cannot suppress the outward relocation.
                reason_lower = str(stop_reason or "").lower()
                if not collision_sectors:
                    if "left side" in reason_lower:
                        collision_sectors = {"left"}
                    elif "right side" in reason_lower:
                        collision_sectors = {"right"}
                print(
                    "[saved-map] active localization fan stopped at "
                    f"{len(captures)} view(s): "
                    f"{stop_reason or 'turn timeout'}."
                )
                lateral_sign = (
                    -1.0
                    if collision_sectors == {"left"}
                    else 1.0
                    if collision_sectors == {"right"}
                    else None
                )
                if (
                    stop_reason
                    and lateral_sign is not None
                    and clearance_relocations < 3
                ):
                    contacted_side = next(iter(collision_sectors))
                    self.status_text = (
                        f"Localization pivot constrained on the {contacted_side}; "
                        "moving outward under live LiDAR protection..."
                    )
                    print(
                        f"[saved-map] localization pivot is constrained on the "
                        f"{contacted_side}; translating 0.12m away before "
                        "restarting the complete 360deg acquisition."
                    )
                    if self._unlocalized_strafe_away_from_side(
                        lateral_sign,
                        direction * step_deg,
                    ):
                        self.pending_collision_sectors = set()
                        print(
                            "[saved-map] outward relocation succeeded; "
                            "discarding the interrupted views and restarting "
                            "localization from the new position."
                        )
                        return self._localize_worker(clearance_relocations + 1)
                    print(
                        "[saved-map] outward relocation was not verified by "
                        "fresh LiDAR; retaining the normal safe-stop path."
                    )
                break
            time.sleep(0.18)
            capture_view()
            self.status_text = (
                f"Relocalizing (attempt {attempt})—captured {len(captures)} "
                f"view(s), angular baseline {abs(captures[-1][1]):.0f}deg"
            )

        sweep_yaw = self.imu.deg()
        completed_full_sweep = (
            sweep_yaw is not None
            and abs(float(sweep_yaw) - float(initial_yaw)) >= 350.0
        )
        if completed_full_sweep:
            # One complete revolution is already the original physical
            # heading. Do not command an unnecessary reverse revolution.
            returned_to_start = True
            return_yaw = sweep_yaw
        else:
            # An interrupted sweep returns over only the arc already traversed.
            self.controller.clear_safety_latch()
            self.controller.rotate_to(float(initial_yaw))
            deadline = time.monotonic() + 14.0
            returned_to_start = False
            while time.monotonic() < deadline and not self.stop_event.is_set():
                yaw = self.imu.deg()
                if yaw is not None and abs(float(initial_yaw) - float(yaw)) <= 3.0:
                    returned_to_start = True
                    break
                if self.controller.safety_latched_reason():
                    break
                time.sleep(0.05)
            self.controller.halt()
            return_yaw = self.imu.deg()
        if not returned_to_start or return_yaw is None:
            self.status_text = (
                "Active localization stopped safely but could not return to its "
                "starting heading; pose was not accepted"
            )
            return

        angular_baseline = (
            max(abs(delta) for _scan, delta in captures) if captures else 0.0
        )
        if len(captures) < 3 or angular_baseline < 55.0:
            self.status_text = (
                f"Relocalization attempt {attempt} stopped safely: only "
                f"{len(captures)} views/{angular_baseline:.0f}deg baseline; "
                "at least three views/55deg are required"
            )
            return

        panorama = _assemble_active_localization_cloud(
            captures,
            self.lever_m,
            self.forward_offset,
        )
        position_seeds = _global_localization_position_seeds(self.saved)
        self.status_text = (
            f"Relocalizing (attempt {attempt})—globally matching "
            f"{len(captures)} angular views..."
        )
        solved, metadata = _search_pose(
            snapshot_points_xy=panorama,
            global_points_xy=reference,
            initial_pose=self.pose,
            resolution_m=float(self.args.stitch_resolution_m),
            search_xy_m=0.50,
            coarse_angle_step_deg=8.0,
            fine_angle_step_deg=0.5,
            theta_window_deg=180.0,
            whole_map_theta_center_deg=float(self.pose.theta_deg),
            whole_map_theta_window_deg=180.0,
            max_translation_from_initial_m=None,
            prior_pose=None,
            allow_whole_map_search=True,
            force_whole_map_search=True,
            whole_map_seed_count=64,
            whole_map_refine_count=20,
            whole_map_position_seeds_xy=position_seeds,
        )
        modes = list(metadata.get("whole_map_modes") or [])
        if modes:
            for mode in modes:
                candidate = Pose2D(
                    float(mode["x"]),
                    float(mode["y"]),
                    float(mode["theta_deg"]),
                )
                quality, endpoint_occ, endpoint_free, ray_occ = (
                    _occupancy_pose_consistency(
                        self.saved,
                        captures,
                        candidate,
                        self.lever_m,
                        self.forward_offset,
                    )
                )
                mode["occupancy_quality"] = quality
                mode["endpoint_occupied"] = endpoint_occ
                mode["endpoint_free"] = endpoint_free
                mode["ray_occupied"] = ray_occ
                mode["combined_score"] = float(mode["score"]) + quality
            modes.sort(key=lambda mode: float(mode["combined_score"]), reverse=True)
            winner = modes[0]
            solved = Pose2D(
                float(winner["x"]),
                float(winner["y"]),
                float(winner["theta_deg"]),
            )
            score = float(winner["score"])
        else:
            score = float(metadata.get("score") or 0.0)
        combined_score = (
            float(modes[0]["combined_score"]) if modes else float(score)
        )
        runner_up_score = (
            float(modes[1]["combined_score"]) if len(modes) > 1 else -math.inf
        )
        winner_margin = combined_score - runner_up_score
        support = _endpoint_support_ratio(
            _transform_points(panorama, solved), reference, 0.08
        )
        print(
            f"[saved-map] active global localization: {len(captures)} views, "
            f"baseline {angular_baseline:.0f}deg, match {score:.1f}, "
            f"support {support:.1%}, distinct-mode margin "
            f"{winner_margin:.2f}, pose "
            f"({solved.x:+.2f}, {solved.y:+.2f}, {solved.theta_deg:+.1f}deg)."
        )
        for mode_index, mode in enumerate(modes[:5], start=1):
            print(
                f"[saved-map]   global mode {mode_index}: score "
                f"{float(mode['score']):.1f}, occupancy "
                f"{float(mode.get('occupancy_quality', 0.0)):+.2f}, combined "
                f"{float(mode.get('combined_score', mode['score'])):.1f} at "
                f"({float(mode['x']):+.2f}, "
                f"{float(mode['y']):+.2f}, {float(mode['theta_deg']):+.1f}deg)."
            )
        if not _global_localization_accepted(
            score,
            support,
            winner_margin,
            float(self.args.localization_min_score),
        ):
            self.status_text = (
                f"Relocalization attempt {attempt} rejected: match {score:.1f}, "
                f"support {support:.0%}, mode margin {winner_margin:.2f}; "
                "robot remains disabled"
            )
            return
        # The active panorama is expressed about the robot centre. Navigation
        # stores a LiDAR pose, so restore the calibrated lever arm only after
        # the global robot pose has passed every acceptance test.
        solved = _lidar_pose_from_robot_centre(
            np.asarray([solved.x, solved.y], dtype=np.float64),
            solved.theta_deg,
            self.lever_m,
            self.forward_offset,
        )
        solved = _rotate_lidar_pose_about_robot_centre(
            solved,
            float(return_yaw) - float(initial_yaw),
            self.lever_m,
            self.forward_offset,
        )
        with self.pose_lock:
            self.pose = solved
        # Seed continuous local odometry from a fresh scan at the accepted
        # returned heading.  Do not seed it from the saved map's final scans:
        # those may have been recorded in a completely different room.
        seed_after = self.feed.latest()[0]
        _frame_id, seed_frame = self.feed.wait_for_frame_after(
            after_frame_id=seed_after,
            timeout_s=1.5,
            min_frame_advances=1,
        )
        seed_local = _scan_local(seed_frame, self.args) if seed_frame is not None else captures[0][0]
        self.local_odometry.reset(seed_local, solved)
        self.global_tracking_cycle = 0
        self.localization_imu = float(return_yaw)
        self.localized = self.localization_imu is not None
        self.body_fixed_lidar_returns = _learn_body_fixed_lidar_returns(
            captures,
            self.forward_offset,
        )
        print(
            "[saved-map] learned body-fixed LiDAR self-filter: "
            f"{len(self.body_fixed_lidar_returns)} persistent angular bin(s); "
            "these chassis returns are excluded from collision safety only."
        )
        solved_centre = self._centre(solved)
        self.status_text = (
            f"Localized robot centre at ({solved_centre[0]:+.2f}, "
            f"{solved_centre[1]:+.2f}), "
            f"heading {solved.theta_deg:+.1f}deg; "
            f"{len(captures)} views/{angular_baseline:.0f}deg, match {score:.1f}, "
            f"support {support:.0%}—click a known free point"
            if self.localized else "IMU unavailable—motion disabled"
        )
        print(f"[saved-map] {self.status_text}")

    def _click_goal(self, event) -> None:
        if not self.localized:
            messagebox.showwarning("Not localized", "Relocalize before commanding motion.")
            return
        if self.navigation_thread is not None and self.navigation_thread.is_alive():
            messagebox.showinfo("Busy", "Stop the current navigation command first.")
            return
        requested = self._world_xy(event.x, event.y)
        # Route preparation includes sensor acquisition and, when necessary,
        # a complete global angular localization.  Keep that work off Tk's UI
        # thread and keep this same worker alive through route execution so a
        # second click cannot overlap the reset/planning boundary.
        self.stop_event.clear()
        self.navigation_thread = threading.Thread(
            target=self._prepare_navigation_command,
            args=(requested,),
            daemon=True,
        )
        self.navigation_thread.start()

    def _prepare_navigation_command(self, requested: np.ndarray) -> None:
        """Establish an observable global pose, plan, then execute one click."""
        # A navigation command starts a new odometry session. First try the
        # inexpensive stationary saved-map consensus. If one 180-degree LiDAR
        # view is ambiguous, do not retain the previous trip's accumulated
        # transform: acquire the same complete angular global localization
        # used at application startup before any route is planned.
        self.status_text = "Re-anchoring LiDAR pose before planning..."
        boundary = self._route_boundary_reanchor("new route")
        if boundary is None:
            self.status_text = "Could not acquire a fresh route-boundary LiDAR scan"
            return
        if boundary == "needs_global":
            self.localization_attempt += 1
            print(
                "[saved-map] route-boundary single-view geometry is ambiguous; "
                "performing a complete angular global localization before "
                "planning this command."
            )
            self._localize_worker()
            if not self.localized:
                print(
                    "[saved-map] route command remains disabled because the "
                    "complete global localization was not accepted."
                )
                return
            self._reset_route_transient_state()

        with self.pose_lock:
            start = self._centre(self.pose)
        route, planning_radius = _plan_saved_map_route(
            self.world,
            start,
            requested,
            float(self.args.robot_radius_m),
            float(self.args.physical_body_radius_m),
        )
        if not route:
            print(
                f"[saved-map] no route from ({start[0]:+.2f}, {start[1]:+.2f}) "
                f"to click ({requested[0]:+.2f}, {requested[1]:+.2f}) even at "
                f"the calibrated {planning_radius:.2f}m physical radius."
            )
            self.status_text = "Clicked point is not connected through known free space"
            return
        if planning_radius < float(self.args.robot_radius_m) - 1e-6:
            print(
                f"[saved-map] conservative {float(self.args.robot_radius_m):.2f}m "
                "costmap closed the route; using the calibrated "
                f"{planning_radius:.2f}m physical footprint with live collision-box protection."
            )
        print(
            f"[saved-map] route accepted from ({start[0]:+.2f}, {start[1]:+.2f}) "
            f"to ({route[-1][0]:+.2f}, {route[-1][1]:+.2f}); "
            f"{len(route)} waypoint(s), planning radius {planning_radius:.2f}m."
        )
        self.goal = np.asarray(route[-1]).copy()
        self.dynamic_obstacles_world = np.empty((0, 2), dtype=np.float32)
        self.pending_collision_world = np.empty((0, 2), dtype=np.float32)
        self.pending_collision_sectors = set()
        self.collision_replan_streak = 0
        self._endpoint_recheck_count = 0
        # Passage replay suppression is local to one navigation command.  A
        # completed outbound/return trip must not make the next click inherit
        # the previous command's "already attempted here" memory.
        self._last_passage_commit_centre = None
        self._passage_route_active = False
        self._edge_escape_anchor = None
        self.path = [start.copy(), *[np.asarray(point).copy() for point in route]]
        self.stop_event.clear()
        self._navigate_worker(route)

    def _reset_route_transient_state(self) -> None:
        """Remove every obstacle/recovery decision owned by the previous trip."""
        self.dynamic_obstacles_world = np.empty((0, 2), dtype=np.float32)
        self.pending_collision_world = np.empty((0, 2), dtype=np.float32)
        self.pending_collision_sectors = set()
        self.collision_replan_streak = 0
        self._endpoint_recheck_count = 0
        self._last_passage_commit_centre = None
        self._passage_route_active = False
        self._edge_escape_anchor = None
        self._soft_warning_side = None
        self._soft_warning_bias = 0.0
        self._soft_warning_last_seen_at = -math.inf
        self._strong_local_motion_relaxations = 0
        self.global_tracking_cycle = 0
        with self._collision_confirmation_lock:
            self._collision_confirmation.clear()
        self.controller.clear_safety_latch()

    def _fresh_route_boundary_keyframe(self) -> np.ndarray | None:
        """Acquire one genuinely fresh, stationary LiDAR keyframe."""
        after = self.feed.latest()[0]
        for _ in range(3):
            after, frame = self.feed.wait_for_frame_after(
                after_frame_id=after,
                timeout_s=0.8,
                min_frame_advances=1,
            )
            if frame is None:
                continue
            local = _scan_local(frame, self.args)
            if len(local) >= 30:
                return local
        return None

    def _route_boundary_reanchor(self, context: str) -> str | None:
        """Re-establish the saved-map/odometry transform between commands."""
        self.controller.halt()
        with self.pose_lock:
            seed = self.pose
            seed_centre = self._centre(seed)

        frame_id = self.feed.latest()[0]
        candidates: list[tuple[np.ndarray, float, float]] = []
        for _ in range(5):
            frame_id, frame = self.feed.wait_for_frame_after(
                after_frame_id=frame_id,
                timeout_s=0.8,
                min_frame_advances=1,
            )
            if frame is None:
                continue
            local = _scan_local(frame, self.args)
            if len(local) < 30:
                continue
            solved, score = _localize(
                local, self.world, seed, self.args, 0.45, 4.0
            )
            solved = _pose_with_imu_heading(
                solved,
                seed.theta_deg,
                self.lever_m,
                self.forward_offset,
            )
            support = _endpoint_support_ratio(
                _transform_points(local, solved),
                self.world.reference(),
                0.08,
            )
            candidates.append((self._centre(solved), score, support))

        consensus = _route_boundary_reanchor_consensus(
            candidates,
            seed_centre,
            float(self.args.localization_min_score),
        )
        if consensus is not None:
            correction_m = float(np.hypot(*(consensus - seed_centre)))
            anchored = _lidar_pose_from_robot_centre(
                consensus,
                seed.theta_deg,
                self.lever_m,
                self.forward_offset,
            )
            with self.pose_lock:
                self.pose = anchored
            print(
                f"[saved-map] route-boundary {context}: accepted saved-map "
                f"re-anchor of {correction_m * 100:.1f}cm from "
                f"{len(candidates)} stationary scan(s)."
            )
        else:
            best_score = max((item[1] for item in candidates), default=-2.0)
            best_support = max((item[2] for item in candidates), default=0.0)
            print(
                f"[saved-map] route-boundary {context}: saved-map consensus "
                f"unavailable ({len(candidates)} scans, best match "
                f"{best_score:.1f}, support {best_support:.0%}); a complete "
                "angular global localization is required instead of retaining "
                "the previous trip's pose."
            )
            return "needs_global"

        # Always seed from a scan acquired after the consensus calculation.
        # Candidate scans were evaluated at slightly different hypotheses and
        # the old rolling history belongs to the completed command.
        keyframe = self._fresh_route_boundary_keyframe()
        if keyframe is None:
            print(
                f"[saved-map] route-boundary {context}: no fresh LiDAR "
                "keyframe; the next command remains disabled."
            )
            return None
        with self.pose_lock:
            anchored_pose = self.pose
        self.local_odometry.reset(keyframe, anchored_pose)
        self._reset_route_transient_state()
        print(
            f"[saved-map] route-boundary {context}: rolling LiDAR submap "
            "replaced with one fresh keyframe at the saved-map consensus pose."
        )
        return "saved_map"

    def _stationary_local_pose_confirmation(self, context: str) -> bool:
        """Confirm position against the recent session submap while stopped.

        The permanent map can legitimately have weak support in a newly
        observed room.  This front-end check instead matches several fresh,
        stationary scans to the longer rolling LiDAR submap and accepts only
        a tight spatial consensus.  It cannot jump to another globally
        similar corridor because its search remains local and its correction
        is bounded to 25cm.
        """
        self.controller.halt()
        with self.pose_lock:
            seed = self.pose
            seed_centre = self._centre(seed)
        reference = self.local_odometry.reference()
        if len(reference) < 30:
            return False

        after = self.feed.latest()[0]
        candidates: list[tuple[np.ndarray, np.ndarray, float, float]] = []
        for _ in range(5):
            after, frame = self.feed.wait_for_frame_after(
                after_frame_id=after,
                timeout_s=0.8,
                min_frame_advances=1,
            )
            if frame is None:
                continue
            local = _scan_local(frame, self.args)
            if len(local) < 30:
                continue
            solved, score, support = _localize_against_points(
                local,
                reference,
                seed,
                self.args,
                0.30,
                3.0,
            )
            solved = _pose_with_imu_heading(
                solved,
                seed.theta_deg,
                self.lever_m,
                self.forward_offset,
            )
            centre = self._centre(solved)
            shift = float(np.hypot(*(centre - seed_centre)))
            if score >= 4.0 and support >= 0.22 and shift <= 0.25:
                candidates.append((centre, local, float(score), float(support)))

        if len(candidates) < 3:
            print(
                f"[saved-map] stationary local pose confirmation for {context} "
                f"kept the continuous pose ({len(candidates)}/5 supported scans)."
            )
            return False

        centres = np.asarray([item[0] for item in candidates], dtype=np.float64)
        consensus = np.median(centres, axis=0)
        scatter = np.hypot(*(centres - consensus).T)
        if float(np.percentile(scatter, 90.0)) > 0.08:
            print(
                f"[saved-map] stationary local pose confirmation for {context} "
                "rejected spatially inconsistent scan matches."
            )
            return False

        correction_m = float(np.hypot(*(consensus - seed_centre)))
        confirmed = _lidar_pose_from_robot_centre(
            consensus,
            seed.theta_deg,
            self.lever_m,
            self.forward_offset,
        )
        with self.pose_lock:
            self.pose = confirmed
        # One representative stationary keyframe strengthens this location
        # without flooding the rolling history with duplicate scans.
        best = max(candidates, key=lambda item: (item[3], item[2]))
        self.local_odometry.add(best[1], confirmed)
        print(
            f"[saved-map] stationary local pose confirmed for {context}: "
            f"{len(candidates)}/5 scans, correction {correction_m * 100:.1f}cm, "
            f"scatter p90 {float(np.percentile(scatter, 90.0)) * 100:.1f}cm."
        )
        return True

    def _stationary_saved_map_correction(self) -> bool:
        """Apply the slow global SLAM correction only while the base is stopped.

        Continuous motion is tracked by the rolling LiDAR submap.  Running the
        much larger saved-map search in that same moving loop let the chassis
        travel while its map pose was frozen.  This backend correction is
        intentionally stationary and gently fuses only a nearby, supported
        result, matching the normal SLAM front-end/back-end split.
        """
        with self.pose_lock:
            seed = self.pose
        started = time.monotonic()
        seed_centre = self._centre(seed)
        frame_id = self.feed.latest()[0]
        candidates: list[tuple[np.ndarray, float, float]] = []
        accepted_locals: list[np.ndarray] = []
        for _ in range(3):
            frame_id, frame = self.feed.wait_for_frame_after(
                after_frame_id=frame_id,
                timeout_s=0.8,
                min_frame_advances=1,
            )
            if frame is None:
                continue
            local = _scan_local(frame, self.args)
            if len(local) < 30:
                continue
            solved, score = _localize(
                local, self.world, seed, self.args, 0.45, 4.0
            )
            solved = _pose_with_imu_heading(
                solved, seed.theta_deg, self.lever_m, self.forward_offset
            )
            support = _endpoint_support_ratio(
                _transform_points(local, solved),
                self.world.reference(),
                0.08,
            )
            candidates.append((self._centre(solved), score, support))
            accepted_locals.append(local)

        decision = _stationary_correction_consensus(
            candidates,
            seed_centre,
            float(self.args.localization_min_score),
        )
        if decision is not None:
            corrected_centre, correction_kind = decision
            correction_m = float(np.hypot(*(corrected_centre - seed_centre)))
            corrected = _lidar_pose_from_robot_centre(
                corrected_centre,
                seed.theta_deg,
                self.lever_m,
                self.forward_offset,
            )
            with self.pose_lock:
                self.pose = corrected
            if accepted_locals:
                self.local_odometry.add(accepted_locals[-1], corrected)
            print(
                f"[saved-map] stationary saved-map {correction_kind} correction "
                f"{correction_m * 100:.1f}cm from {len(candidates)} fresh scan(s) "
                f"{time.monotonic() - started:.2f}s)."
            )
            return True
        else:
            # A weak backend result never invalidates healthy local odometry.
            best_score = max((item[1] for item in candidates), default=-2.0)
            best_support = max((item[2] for item in candidates), default=0.0)
            nearest_shift = min(
                (
                    float(np.hypot(*(item[0] - seed_centre)))
                    for item in candidates
                ),
                default=float("inf"),
            )
            print(
                "[saved-map] stationary saved-map correction skipped "
                f"({len(candidates)} scans, best match {best_score:.1f}, "
                f"support {best_support:.0%}, nearest shift {nearest_shift:.2f}m; "
                "no spatial consensus)."
            )
            return False

    def _navigate_worker(
        self,
        route: list[np.ndarray],
        backup_failures: int = 0,
    ) -> None:
        self.status_text = "Navigating..."
        imu_anchor = self.imu.deg()
        if imu_anchor is None:
            self.status_text = "IMU unavailable—navigation cancelled"
            return
        weak = 0
        after = self.feed.latest()[0]
        drive_command = _forward_command_above_stiction(
            float(self.args.drive_speed)
        )
        if drive_command != float(self.args.drive_speed):
            print(
                f"[saved-map] requested forward command "
                f"{float(self.args.drive_speed):.2f} is below drivetrain stiction; "
                f"using {drive_command:.2f}."
            )
        with self.pose_lock:
            route_start = self._centre(self.pose)
        route, dropped = _drop_reached_waypoint_prefix(route, route_start)
        if dropped:
            print(
                f"[saved-map] discarded {dropped} already-reached route "
                "connector waypoint(s) before calculating any turn."
            )
        for waypoint_index, waypoint in enumerate(route, start=1):
            if self.stop_event.is_set():
                return
            # Tracking failures are consecutive failures within one straight
            # segment, not a mission-wide budget. A waypoint boundary changes
            # both the commanded motion prior and often the visible LiDAR
            # hemisphere; carrying the previous segment's weak count into the
            # next segment caused a strong 14.0/92% match to trip the 8-scan
            # shutdown before its normal stationary recovery could run.
            weak = 0
            with self.pose_lock:
                centre = self._centre(self.pose)
                physical_heading = self.pose.theta_deg + self.forward_offset
            waypoint_distance = float(
                np.hypot(*(np.asarray(waypoint, dtype=np.float64) - centre))
            )
            if waypoint_distance <= 0.16:
                print(
                    f"[saved-map] waypoint {waypoint_index}/{len(route)} is "
                    f"already reached ({waypoint_distance:.2f}m); skipping it "
                    "without issuing a turn."
                )
                continue
            vector = np.asarray(waypoint) - centre
            desired_heading = math.degrees(math.atan2(vector[1], vector[0]))
            heading_error = (desired_heading - physical_heading + 180.0) % 360.0 - 180.0
            turn_target = float(imu_anchor) + heading_error
            self.controller.clear_safety_latch()
            self.controller.rotate_to(turn_target)
            deadline = time.monotonic() + 12.0
            turn_reached = False
            turn_collision_reason: str | None = None
            while time.monotonic() < deadline and not self.stop_event.is_set():
                yaw = self.imu.deg()
                if yaw is not None and abs(turn_target - yaw) <= 3.0:
                    turn_reached = True
                    break
                if self.controller.safety_latched_reason():
                    turn_collision_reason = (
                        self.controller.safety_latched_reason() or "Turn blocked"
                    )
                    break
                time.sleep(0.05)
            self.controller.halt()
            if turn_collision_reason is not None:
                new_route, backup_failures = self._replan_after_collision(
                    turn_collision_reason,
                    backup_failures,
                )
                if new_route:
                    return self._navigate_worker(new_route, backup_failures)
                return
            if not turn_reached:
                self.status_text = (
                    f"Waypoint {waypoint_index}: turn did not reach its target; stopped"
                )
                print(f"[saved-map] {self.status_text}.")
                return
            yaw_after = self.imu.deg()
            if yaw_after is None:
                return
            with self.pose_lock:
                self.pose = _pose_with_imu_heading(
                    self.pose,
                    self.pose.theta_deg + (yaw_after - imu_anchor),
                    self.lever_m,
                    self.forward_offset,
                )
            imu_anchor = yaw_after

            # The sensor sees only one hemisphere.  A substantial waypoint
            # turn can therefore replace nearly the entire visible scene even
            # though the robot centre has not translated.  Seed the rolling
            # front end at the new IMU-confirmed heading before driving; if we
            # wait until translation begins, the first scan can have almost no
            # overlap with the pre-turn half-view and local odometry starts at
            # an artificial failure (-2 score / near-zero support).
            turn_frame_id, turn_frame = self.feed.wait_for_frame_after(
                after_frame_id=after,
                timeout_s=1.0,
                min_frame_advances=1,
            )
            if turn_frame is not None:
                turn_local = _scan_local(turn_frame, self.args)
                if len(turn_local) >= 30:
                    with self.pose_lock:
                        turn_pose = self.pose
                    self.local_odometry.add(turn_local, turn_pose)
                    # This fresh, IMU-confirmed post-turn scan seeds the new
                    # straight segment. No rejected scan from the preceding
                    # heading is relevant to its consecutive-failure count.
                    weak = 0
                    after = turn_frame_id

            drive_heading = yaw_after
            soft_heading_bias = self._soft_clearance_heading_bias()
            self.controller.clear_safety_latch()
            self.controller.drive_toward(
                drive_command, drive_heading + soft_heading_bias
            )
            segment_started = time.monotonic()
            segment_start = centre.copy()
            last_progress_at = segment_started
            best_progress = 0.0
            clear_no_progress_rescans = 0
            tracking_pause_active = False
            last_pose_update_at = segment_started
            print(
                f"[saved-map] waypoint {waypoint_index}/{len(route)} aligned; "
                f"driving {float(np.hypot(*(np.asarray(waypoint) - centre))):.2f}m "
                f"with forward command {drive_command:.2f}."
            )
            while not self.stop_event.is_set():
                collision_reason = self.controller.safety_latched_reason()
                if collision_reason:
                    new_route, backup_failures = self._replan_after_collision(
                        collision_reason,
                        backup_failures,
                    )
                    if new_route:
                        return self._navigate_worker(new_route, backup_failures)
                    return
                with self.pose_lock:
                    current_pose = self.pose
                    centre = self._centre(current_pose)
                if float(np.hypot(*(np.asarray(waypoint) - centre))) <= 0.16:
                    break
                now = time.monotonic()
                progress = float(np.hypot(*(centre - segment_start)))
                if progress > best_progress + 0.02:
                    best_progress = progress
                    last_progress_at = now
                if now - last_progress_at > 4.0:
                    self.controller.halt()
                    # A frozen pose is not proof of a blocked or stationary
                    # chassis: a healthy scan can be rejected by the bounded
                    # motion prior, leaving the progress watchdog looking at
                    # an old pose. Stop, acquire one genuinely fresh scan, and
                    # let live LiDAR safety decide. If the corridor is clear,
                    # reseed local odometry at the stationary pose and resume
                    # this same straight segment instead of terminating.
                    fresh_id, fresh_frame = self.feed.wait_for_frame_after(
                        after_frame_id=after,
                        timeout_s=1.0,
                        min_frame_advances=1,
                    )
                    if fresh_frame is not None:
                        after = fresh_id
                    live_reason = self._translation_safety()
                    if live_reason is None and clear_no_progress_rescans < 3:
                        clear_no_progress_rescans += 1
                        if fresh_frame is not None:
                            fresh_local = _scan_local(fresh_frame, self.args)
                            if len(fresh_local) >= 30:
                                with self.pose_lock:
                                    stationary_pose = self.pose
                                self.local_odometry.add(
                                    fresh_local, stationary_pose
                                )
                        with self.pose_lock:
                            segment_start = self._centre(self.pose)
                        segment_started = time.monotonic()
                        last_progress_at = segment_started
                        last_pose_update_at = segment_started
                        best_progress = 0.0
                        weak = 0
                        self.controller.clear_safety_latch()
                        self.controller.drive_toward(
                            drive_command,
                            drive_heading + soft_heading_bias,
                        )
                        print(
                            "[saved-map] forward-progress watchdog acquired a "
                            "fresh LiDAR scan and the collision envelope is "
                            f"CLEAR; resumed the same straight segment "
                            f"({clear_no_progress_rescans}/3)."
                        )
                        continue
                    if live_reason is not None:
                        new_route, backup_failures = self._replan_after_collision(
                            live_reason,
                            backup_failures,
                        )
                        if new_route:
                            return self._navigate_worker(
                                new_route, backup_failures
                            )
                        return
                    transport_error = self.controller.command_error()
                    self.status_text = (
                        f"Waypoint {waypoint_index}: no physical forward progress"
                    )
                    detail = transport_error or (
                        "three fresh LiDAR scans were clear but no localized "
                        f"motion followed command {drive_command:.2f}"
                    )
                    print(f"[saved-map] {self.status_text} ({detail}); stopped.")
                    return
                if now - segment_started > 45.0:
                    self.controller.halt()
                    self.status_text = f"Waypoint {waypoint_index}: segment timed out"
                    print(f"[saved-map] {self.status_text}; stopped.")
                    return
                frame_id, frame = self.feed.wait_for_frame_after(
                    after_frame_id=after, timeout_s=0.8, min_frame_advances=1
                )
                if frame is None:
                    continue
                after = frame_id
                # SECONDARY / SOFT ENVELOPE: this never latches safety and
                # never stops translation. Smoothly bias the fast IMU heading
                # hold away from a one-sided wall before it reaches the hard
                # calibrated collision box. If both walls are close, preserve
                # the corridor centreline instead of inventing an oscillation.
                requested_soft_bias = self._soft_clearance_heading_bias()
                soft_heading_bias = (
                    0.65 * soft_heading_bias + 0.35 * requested_soft_bias
                )
                if abs(soft_heading_bias) < 0.25:
                    soft_heading_bias = 0.0
                self.controller.drive_toward(
                    drive_command, drive_heading + soft_heading_bias
                )
                local = _scan_local(frame, self.args)
                if len(local) < 30:
                    continue
                yaw = _imu_yaw_for_scan(self.imu, frame)
                theta_seed = current_pose.theta_deg + (
                    (float(yaw) - float(imu_anchor)) if yaw is not None else 0.0
                )
                seed = _pose_with_imu_heading(
                    current_pose, theta_seed, self.lever_m, self.forward_offset
                )
                # FRONT END: match every fresh scan against the last few
                # accepted scans.  This is the continuous LiDAR odometry that
                # preserves pose through a doorway even when much of the scan
                # is new and therefore cannot match the permanent map yet.
                local_reference = self.local_odometry.reference()
                solved_local, local_score, local_support = _localize_against_points(
                    local,
                    local_reference,
                    seed,
                    self.args,
                    0.45,
                    4.0,
                )
                solved_local = _pose_with_imu_heading(
                    solved_local, theta_seed, self.lever_m, self.forward_offset
                )
                segment_direction = np.asarray(waypoint) - segment_start
                local_step = (
                    self._centre(solved_local) - self._centre(current_pose)
                )
                pose_dt = max(0.05, time.monotonic() - last_pose_update_at)
                maximum_forward_step = min(0.20, max(0.04, 0.40 * pose_dt))
                maximum_lateral_step = min(0.05, max(0.02, 0.08 * pose_dt))
                local_valid = (
                    local_score >= max(
                        4.0, float(self.args.localization_min_score) - 2.0
                    )
                    and local_support >= 0.18
                    and _straight_motion_step_is_consistent(
                        local_step,
                        segment_direction,
                        maximum_forward_m=maximum_forward_step,
                        maximum_lateral_m=maximum_lateral_step,
                    )
                )
                # A rolling local submap is the primary odometry source while
                # crossing into a new room. Strong scan agreement can span a
                # longer interval than the nominal 20cm motion prior after a
                # collision recovery or a slow UI/planning cycle. Rejecting a
                # 97%-supported local solution solely for that tight prior is
                # what caused the recorded run to throw away healthy odometry
                # and launch an ambiguous global search. Retain only
                # physically forward, bounded solutions; lateral teleports and
                # backwards mode switches remain forbidden.
                strong_local_evidence = (
                    local_score >= float(self.args.localization_min_score) + 2.0
                    and local_support >= 0.55
                )
                relaxed_local_motion = _straight_motion_step_is_consistent(
                    local_step,
                    segment_direction,
                    maximum_forward_m=max(0.40, maximum_forward_step),
                    maximum_reverse_m=0.04,
                    maximum_lateral_m=max(0.12, maximum_lateral_step),
                )
                if not local_valid and strong_local_evidence and relaxed_local_motion:
                    local_valid = True
                    self._strong_local_motion_relaxations += 1
                    if self._strong_local_motion_relaxations <= 3:
                        direction_norm = max(
                            1e-9, float(np.hypot(*segment_direction))
                        )
                        direction_unit = segment_direction / direction_norm
                        lateral_unit = np.asarray(
                            [-direction_unit[1], direction_unit[0]],
                            dtype=np.float64,
                        )
                        print(
                            "[saved-map] preserving strong rolling-local LiDAR "
                            f"odometry (match {local_score:.1f}, support "
                            f"{local_support:.0%}): step "
                            f"{float(local_step @ direction_unit):+.2f}m along/"
                            f"{float(local_step @ lateral_unit):+.2f}m lateral "
                            "passed the bounded recovery motion prior."
                        )

                # FRONT END: never run the expensive saved-map search in this
                # moving loop. The rolling submap and IMU own continuous pose;
                # the permanent map is consulted only by the explicitly
                # stationary, lower-rate correction below.
                if not local_valid:
                    # Stop first, then retry against the rolling local submap
                    # with a wider bounded window. Repeated permanent-map
                    # searches here were slow and could jump to another
                    # similar-looking corridor mode.
                    self.controller.halt()
                    tracking_pause_active = True
                    recovered_local, recovered_score, recovered_support = (
                        _localize_against_points(
                            local,
                            local_reference,
                            seed,
                            self.args,
                            0.70,
                            6.0,
                        )
                    )
                    recovered_local = _pose_with_imu_heading(
                        recovered_local,
                        theta_seed,
                        self.lever_m,
                        self.forward_offset,
                    )
                    recovered_step = (
                        self._centre(recovered_local)
                        - self._centre(current_pose)
                    )
                    recovered_valid = (
                        recovered_score
                        >= max(4.0, float(self.args.localization_min_score) - 2.0)
                        and recovered_support >= 0.18
                        and _straight_motion_step_is_consistent(
                            recovered_step,
                            segment_direction,
                            maximum_forward_m=max(0.48, maximum_forward_step),
                            maximum_reverse_m=0.04,
                            maximum_lateral_m=max(0.12, maximum_lateral_step),
                        )
                    )
                    if recovered_valid:
                        solved_local = recovered_local
                        local_score = recovered_score
                        local_support = recovered_support
                        local_valid = True
                        print(
                            "[saved-map] rolling-local LiDAR odometry recovered "
                            "while stationary (match "
                            f"{recovered_score:.1f}, support "
                            f"{recovered_support:.0%}); no global search was run."
                        )

                # The permanent map is a lower-rate stationary drift
                # correction, never an emergency per-frame moving fallback.
                run_global = False
                if run_global:
                    # Emergency global matching may take substantially longer
                    # than one LiDAR period. Stop first so pose and chassis can
                    # never diverge during that computation.
                    self.controller.halt()
                    tracking_pause_active = True
                    solved_global, global_score = _localize(
                        local, self.world, seed, self.args, 0.45, 6.0
                    )
                    solved_global = _pose_with_imu_heading(
                        solved_global, theta_seed, self.lever_m, self.forward_offset
                    )
                    global_support = _endpoint_support_ratio(
                        _transform_points(local, solved_global),
                        self.world.reference(),
                        0.08,
                    )
                    global_step = (
                        self._centre(solved_global)
                        - self._centre(current_pose)
                    )
                    global_valid = (
                        global_score >= float(self.args.localization_min_score)
                        and global_support >= 0.15
                        and _straight_motion_step_is_consistent(
                            global_step,
                            segment_direction,
                            maximum_forward_m=maximum_forward_step,
                            maximum_lateral_m=maximum_lateral_step,
                        )
                    )
                    # A seeded permanent-map search is bounded to this local
                    # 0.45m window.  If it produces overwhelming geometric
                    # agreement, retain it as a front-end re-anchor even when
                    # UI/matcher latency made the normal per-frame 0.20m
                    # motion prior too small.  Directional bounds still reject
                    # backwards mode switches and lateral teleports.
                    strong_global_evidence = (
                        global_score
                        >= float(self.args.localization_min_score) + 2.0
                        and global_support >= 0.55
                    )
                    relaxed_global_motion = _straight_motion_step_is_consistent(
                        global_step,
                        segment_direction,
                        maximum_forward_m=max(0.48, maximum_forward_step),
                        maximum_reverse_m=0.04,
                        maximum_lateral_m=max(0.10, maximum_lateral_step),
                    )
                    if (
                        not global_valid
                        and strong_global_evidence
                        and relaxed_global_motion
                    ):
                        global_valid = True
                        print(
                            "[saved-map] strong nearby saved-map match "
                            f"re-anchored rolling LiDAR odometry (match "
                            f"{global_score:.1f}, support "
                            f"{global_support:.0%})."
                        )
                else:
                    solved_global = seed
                    global_score = 0.0
                    global_support = 0.0
                    global_valid = False

                candidate: Pose2D | None = None
                source = "none"
                if local_valid:
                    candidate = solved_local
                    source = "local"
                    if global_valid:
                        local_centre = self._centre(solved_local)
                        global_centre = self._centre(solved_global)
                        agreement = float(np.hypot(*(global_centre - local_centre)))
                        if agreement <= 0.12:
                            # Preserve the continuous local trajectory and use
                            # a small global correction. Repeated walls cannot
                            # teleport the robot to another saved-map mode.
                            fused_centre = 0.80 * local_centre + 0.20 * global_centre
                            candidate = _lidar_pose_from_robot_centre(
                                fused_centre,
                                theta_seed,
                                self.lever_m,
                                self.forward_offset,
                            )
                            source = "local+global"
                elif global_valid:
                    candidate = solved_global
                    source = "global"

                if candidate is None:
                    weak += 1
                    direction_norm = max(
                        1e-9, float(np.hypot(*segment_direction))
                    )
                    direction_unit = segment_direction / direction_norm
                    lateral_unit = np.asarray(
                        [-direction_unit[1], direction_unit[0]],
                        dtype=np.float64,
                    )
                    rejected_along = float(local_step @ direction_unit)
                    rejected_lateral = float(local_step @ lateral_unit)
                    if weak == 1:
                        print(
                            "[saved-map] rolling-local match rejected by the "
                            "current straight-segment motion prior: match "
                            f"{local_score:.1f}, support {local_support:.0%}, "
                            f"inferred {rejected_along:+.2f}m along/"
                            f"{rejected_lateral:+.2f}m lateral; allowed forward "
                            f"{maximum_forward_step:.2f}m, lateral "
                            f"{maximum_lateral_step:.2f}m. Consecutive count "
                            "starts at 1 for this waypoint only."
                        )
                    # After several consecutive local failures, allow one
                    # stationary back-end correction. Do not repeat a global
                    # saved-map search for every incoming LiDAR frame.
                    if weak == 4 and self._stationary_saved_map_correction():
                        with self.pose_lock:
                            candidate = self.pose
                        source = "stationary-backend"
                        self.local_odometry.reset(local, candidate)
                    if weak >= 8:
                        self.controller.halt()
                        self.localized = False
                        self.status_text = (
                            "Rolling LiDAR odometry "
                            "failed on 8 fresh scans—press Relocalize"
                        )
                        print(
                            "[saved-map] tracking unavailable: "
                            f"local match {local_score:.1f}/support "
                            f"{local_support:.0%}, inferred "
                            f"{rejected_along:+.2f}m along/"
                            f"{rejected_lateral:+.2f}m lateral on the current "
                            "straight segment."
                        )
                        return
                    if candidate is None:
                        continue

                if weak:
                    print(
                        f"[saved-map] continuous LiDAR odometry recovered after "
                        f"{weak} weak scan(s) using {source} tracking."
                    )
                weak = 0
                with self.pose_lock:
                    self.pose = candidate
                self.local_odometry.add(local, candidate)
                last_pose_update_at = time.monotonic()
                imu_anchor = yaw if yaw is not None else imu_anchor
                candidate_centre = self._centre(candidate)
                self.path = [
                    candidate_centre.copy(),
                    *[
                        np.asarray(point).copy()
                        for point in route[waypoint_index - 1 :]
                    ],
                ]
                if tracking_pause_active:
                    self.controller.clear_safety_latch()
                    self.controller.drive_toward(
                        drive_command, drive_heading + soft_heading_bias
                    )
                    tracking_pause_active = False

                # The base executes straight pivot/drive segments, but wheel
                # slip can translate it sideways while its IMU heading remains
                # perfect. Use continuous LiDAR odometry to measure cross-track
                # error. Once outside the corridor, stop and compute a new
                # centreline route instead of blindly continuing parallel to
                # the old path until the shoulder hits something.
                segment_vector = np.asarray(waypoint) - segment_start
                segment_length_sq = float(segment_vector @ segment_vector)
                if segment_length_sq > 1e-9:
                    along = float(
                        (candidate_centre - segment_start) @ segment_vector
                        / segment_length_sq
                    )
                    if -0.10 <= along <= 1.10:
                        closest = segment_start + np.clip(along, 0.0, 1.0) * segment_vector
                        cross_track = float(np.hypot(*(candidate_centre - closest)))
                        if cross_track > float(self.args.max_path_cross_track_m):
                            self.controller.halt()
                            if self.goal is None:
                                return
                            new_route, radius = _plan_saved_map_route(
                                self.world,
                                candidate_centre,
                                np.asarray(self.goal),
                                float(self.args.robot_radius_m),
                                float(self.args.physical_body_radius_m),
                                extra_occupied_xy=self.dynamic_obstacles_world,
                            )
                            if not new_route:
                                self.status_text = (
                                    "Path tracking drifted outside its corridor "
                                    "and no known-free replan exists"
                                )
                                print(f"[saved-map] {self.status_text}.")
                                return
                            self.path = [
                                candidate_centre.copy(),
                                *[np.asarray(point).copy() for point in new_route],
                            ]
                            print(
                                "[saved-map] live LiDAR odometry measured "
                                f"{cross_track:.2f}m cross-track error; stopped "
                                f"the old segment and replanned {len(new_route)} "
                                f"waypoint(s) at radius {radius:.2f}m."
                            )
                            return self._navigate_worker(
                                new_route, backup_failures
                            )
                if self.controller.safety_latched_reason():
                    collision_reason = (
                        self.controller.safety_latched_reason() or "Obstacle stop"
                    )
                    new_route, backup_failures = self._replan_after_collision(
                        collision_reason,
                        backup_failures,
                    )
                    if new_route:
                        return self._navigate_worker(new_route, backup_failures)
                    return
            self.controller.halt()
            self.global_tracking_cycle += 1
            if (
                self.global_tracking_cycle
                % max(1, int(self.args.stationary_global_correction_waypoints))
                == 0
            ):
                self._stationary_saved_map_correction()
            # A completed waypoint is real forward progress. Recovery budgets
            # are local to one obstruction, not cumulative over the mission.
            backup_failures = 0
            self.collision_replan_streak = 0
            self._edge_escape_anchor = None
        # Do not declare arrival solely from integrated moving odometry in a
        # sparsely represented room. Confirm the endpoint against recent
        # session LiDAR geometry while motionless. If that bounded consensus
        # says the robot is still materially short, finish the remaining
        # route instead of letting the next click inherit a wrong start pose.
        self.controller.halt()
        self._stationary_local_pose_confirmation("goal arrival")
        if self.goal is not None:
            with self.pose_lock:
                confirmed_centre = self._centre(self.pose)
            remaining = float(np.hypot(*(np.asarray(self.goal) - confirmed_centre)))
            if remaining > 0.24 and self._endpoint_recheck_count < 2:
                finish_route, finish_radius = _plan_saved_map_route(
                    self.world,
                    confirmed_centre,
                    np.asarray(self.goal),
                    float(self.args.robot_radius_m),
                    float(self.args.physical_body_radius_m),
                    extra_occupied_xy=self.dynamic_obstacles_world,
                )
                if finish_route:
                    self._endpoint_recheck_count += 1
                    self.path = [
                        confirmed_centre.copy(),
                        *[np.asarray(point).copy() for point in finish_route],
                    ]
                    print(
                        "[saved-map] stationary endpoint confirmation found "
                        f"{remaining:.2f}m still remaining; completing "
                        f"{len(finish_route)} corrected waypoint(s) at radius "
                        f"{finish_radius:.2f}m before declaring arrival."
                    )
                    return self._navigate_worker(finish_route, backup_failures)
        self.path = []
        self._passage_route_active = False
        # One edge encounter may select one turn. Another turn is forbidden
        # until rolling LiDAR verifies that the short escape translated away.
        self._edge_escape_anchor: np.ndarray | None = None
        self.status_text = "Goal reached—click another known free point"

    def stop(self) -> None:
        self.stop_event.set()
        self.controller.halt()
        self.path = []
        self._passage_route_active = False
        self._edge_escape_anchor = None
        self.status_text = "Stopped"

    def close(self) -> None:
        if self.closing:
            return
        self.closing = True
        self.stop()
        for cleanup in (
            self._close_collision_diagnostic_video,
            self.controller.shutdown,
            lambda: _send_stop(self.robot),
            self.imu.stop,
            self.feed.stop,
            self.robot.disconnect,
        ):
            with contextlib.suppress(BaseException):
                cleanup()
        self.root.destroy()


class _TimestampedTee:
    """Mirror a stream to the terminal and a timestamped per-run log."""

    def __init__(self, terminal, log_file, lock: threading.Lock) -> None:
        self.terminal = terminal
        self.log_file = log_file
        self.lock = lock
        self.pending = ""

    @property
    def encoding(self):
        return getattr(self.terminal, "encoding", "utf-8")

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())

    def fileno(self) -> int:
        return int(self.terminal.fileno())

    def write(self, value: str) -> int:
        if not value:
            return 0
        with self.lock:
            self.terminal.write(value)
            self.pending += value
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                milliseconds = int((time.time() % 1.0) * 1000.0)
                self.log_file.write(f"[{stamp}.{milliseconds:03d}] {line}\n")
        return len(value)

    def flush(self) -> None:
        with self.lock:
            self.terminal.flush()
            if self.pending:
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                milliseconds = int((time.time() % 1.0) * 1000.0)
                self.log_file.write(
                    f"[{stamp}.{milliseconds:03d}] {self.pending}\n"
                )
                self.pending = ""
            self.log_file.flush()


def _parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-ip", required=True)
    parser.add_argument("--robot-id", default="sourccey_client")
    parser.add_argument("--map", default=str(DEFAULT_SAVED_MAP_PATH))
    parser.add_argument("--lidar-port", type=int, default=8765)
    parser.add_argument("--imu-yaw-port", type=int, default=8770)
    parser.add_argument("--imu-yaw-sign", type=float, default=1.0)
    parser.add_argument(
        "--drive-speed",
        type=float,
        default=0.80,
        help="Forward base command; nonzero values below the measured 0.80 stiction floor are compensated.",
    )
    parser.add_argument("--turn-speed", type=float, default=0.9)
    parser.add_argument("--robot-radius-m", type=float, default=0.31)
    parser.add_argument("--physical-body-radius-m", type=float, default=0.28)
    parser.add_argument("--collision-self-mask-inset-m", type=float, default=0.0)
    parser.add_argument(
        "--soft-collision-margin-m",
        type=float,
        default=0.10,
        help=(
            "Non-latching clearance-warning distance outside the calibrated "
            "hard collision box (default: 0.10m)."
        ),
    )
    parser.add_argument(
        "--soft-steer-max-deg",
        type=float,
        default=10.0,
        help=(
            "Maximum continuous heading bias away from a one-sided soft-box "
            "warning; this never disables hard collision safety (default: 10deg)."
        ),
    )
    parser.add_argument(
        "--max-path-cross-track-m",
        type=float,
        default=0.10,
        help=(
            "Replan instead of continuing blindly when LiDAR odometry places "
            "the robot this far from its straight segment (default: 0.10m)."
        ),
    )
    parser.add_argument(
        "--stationary-global-correction-waypoints",
        type=int,
        default=1,
        help=(
            "Run the slower permanent-map drift correction only after this "
            "many completed waypoints; moving pose updates remain continuous "
            "local LiDAR odometry (default: 1)."
        ),
    )
    parser.add_argument(
        "--collision-confirmation-frames",
        type=int,
        default=3,
        help=(
            "Distinct, spatially consistent LiDAR frames required before a "
            "collision stop (default: 3)."
        ),
    )
    parser.add_argument(
        "--forward-collision-lookahead-m",
        type=float,
        default=0.30,
        help=(
            "Continuously roll the calibrated footprint this far forward "
            "through each fresh LiDAR scan and replan before contact "
            "(default: 0.30m)."
        ),
    )
    parser.add_argument("--collision-box-file", default=str(DEFAULT_COLLISION_BOX_PATH))
    parser.add_argument("--forward-angle-deg", type=float, default=90.0)
    parser.add_argument("--invert-lateral-axis", action="store_true", default=True)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-range-m", type=float, default=0.03)
    parser.add_argument("--max-distance-m", type=float, default=8.0)
    parser.add_argument("--valid-angle-half-width-deg", type=float, default=180.0)
    parser.add_argument("--stitch-resolution-m", type=float, default=0.03)
    parser.add_argument("--match-max-points", type=int, default=700)
    parser.add_argument("--localization-min-score", type=float, default=7.0)
    parser.add_argument(
        "--diagnostic-video-dir",
        default=str(Path("scripts") / "saved_map_diagnostics"),
        help=(
            "Directory for automatic Live LiDAR + Last Stop MP4 recordings "
            "(default: scripts/saved_map_diagnostics)."
        ),
    )
    parser.add_argument(
        "--diagnostic-video-fps",
        type=float,
        default=6.0,
        help="Frame rate for the automatic robot-frame diagnostic MP4.",
    )
    args = parser.parse_args()
    # _localize accesses this only through optional getattr paths today; keep a
    # complete explicit navigation matcher contract for future changes.
    args.track_search_xy_m = 0.45
    args.track_theta_window_deg = 6.0
    return args


def main() -> int:
    args = _parse_args()
    diagnostic_dir = Path(args.diagnostic_video_dir)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = (diagnostic_dir / "latest_run.log").resolve()
    # Deliberately replace this file each launch. The MP4 files remain
    # timestamped, while latest_run.log is the one predictable artifact an
    # operator or debugging agent can inspect after the most recent attempt.
    run_log = run_log_path.open("w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_lock = threading.Lock()
    stdout_tee = _TimestampedTee(original_stdout, run_log, tee_lock)
    stderr_tee = _TimestampedTee(original_stderr, run_log, tee_lock)
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee
    root: tk.Tk | None = None
    navigator: SavedMapNavigator | None = None
    previous_sigint = signal.getsignal(signal.SIGINT)
    try:
        print(f"[saved-map] writing synchronized run log to {run_log_path}")
        configure_candidate_scoring_device("auto")
        root = tk.Tk()
        navigator = SavedMapNavigator(root, args)

        def request_shutdown(_signum=None, _frame=None) -> None:
            if navigator is None or navigator.closing:
                return
            print(
                "[saved-map] Ctrl+C received; stopping the base and "
                "finalizing diagnostics."
            )
            # Tk used to catch KeyboardInterrupt inside a callback and keep
            # running. A real SIGINT handler turns it into an orderly GUI
            # shutdown instead. Setting this event is signal-safe and stops
            # the worker before the scheduled full cleanup runs.
            navigator.stop_event.set()
            with contextlib.suppress(BaseException):
                navigator.controller.halt()
            with contextlib.suppress(tk.TclError):
                root.after(0, navigator.close)

        def callback_exception(exc_type, exc_value, exc_traceback) -> None:
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            print(
                "[saved-map] Tk callback failed; stopping safely and "
                "finalizing diagnostics."
            )
            request_shutdown()

        signal.signal(signal.SIGINT, request_shutdown)
        root.report_callback_exception = callback_exception
        root.mainloop()
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        # A VideoWriter must be released to write the MP4 index.  The window
        # protocol calls close() during an ordinary exit; this also covers a
        # terminal interrupt or an exception escaping Tk's event loop.
        if navigator is not None and not navigator.closing:
            navigator.close()
        stdout_tee.flush()
        stderr_tee.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        run_log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
