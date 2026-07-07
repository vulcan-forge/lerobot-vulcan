from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ldlidar_auto_snapshot_stitch import (
    _ensure_clean_directory,
    _execute_turn_burst,
    _init_rerun,
    _log_rerun_state,
    _send_stop,
    _wait_for_initial_frame,
)
from ldlidar_defaults import (
    DEFAULT_LIDAR_FORWARD_ANGLE_DEG,
    DEFAULT_LIDAR_STOP_BOX_DISTANCE_M,
    DEFAULT_LIDAR_STOP_BOX_HALF_WIDTH_M,
    DEFAULT_LIDAR_STOP_BOX_MIN_DISTANCE_M,
    DEFAULT_LIDAR_STOP_BOX_THICKNESS_M,
)
from ldlidar_direct_snapshot_client import DirectLidarFeed, _save_snapshot, _scan_to_local_points
from ldlidar_direct_snapshot_stitch import (
    HTML_TEMPLATE,
    MotionHint,
    Pose2D,
    Snapshot,
    _advance_pose,
    _append_stitch_snapshot,
    _load_snapshots,
    _build_score_grids,
    _pose_dict,
    _prior_weights_for_hint,
    _score_candidate,
    _search_pose,
    _solve_arc_pose_on_grids,
    _solve_turn_arc_pose,
    _stitch_snapshots,
    _transform_points,
)
from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient


@dataclass(slots=True)
class StopZoneConfig:
    forward_angle_deg: float
    min_distance_m: float
    tripwire_distance_m: float
    tripwire_half_width_m: float
    tripwire_thickness_m: float
    min_points_to_trigger: int
    # Points the lidar permanently sees inside the box (robot's own shell /
    # fixtures), measured at startup. Blocking triggers only on points ABOVE
    # this baseline — otherwise a self-seeing lidar reports "blocked" at every
    # heading and the robot never drives.
    baseline_points: int = 0

    def blocked_trigger_count(self) -> int:
        return int(self.baseline_points) + int(self.min_points_to_trigger)


@dataclass(slots=True)
class FrontierChoice:
    delta_deg: float
    abs_angle_deg: float
    mean_distance_m: float
    width_deg: float
    score: float
    source: str = "live"


@dataclass(slots=True)
class ExploreTarget:
    world_x_m: float
    world_y_m: float
    source: str
    seeded_capture_index: int
    coarse_turns_used: int = 0


@dataclass(slots=True)
class DriveSequenceMeta:
    elapsed_s: float
    iterations: int
    bursts_completed: int
    stopped_by_block: bool
    blocked_points: int
    commanded_forward_speed: float
    steer_theta_vel: float
    host_feedback_updates: int
    host_forward_distance_estimate_m: float
    host_x_vel_peak: float
    host_theta_vel_peak: float
    host_x_vel_last: float
    host_theta_vel_last: float


@dataclass(slots=True)
class DriveBurstMeta:
    elapsed_s: float
    iterations: int
    host_feedback_updates: int
    host_forward_distance_estimate_m: float
    host_x_vel_peak: float
    host_theta_vel_peak: float
    host_x_vel_last: float
    host_theta_vel_last: float


def _normalize_angle_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _turn_lever_arm_local_delta(
    dtheta_deg: float,
    *,
    lidar_offset_forward_m: float,
) -> tuple[float, float]:
    """Expected LiDAR translation (in the pre-turn sensor frame) for an in-place
    robot turn of `dtheta_deg`, given the sensor is mounted `lidar_offset_forward_m`
    forward of the robot's rotation center. The sensor traces an arc around the
    center, so pure robot rotation still translates the sensor by
    2*r*sin(|dtheta|/2) — ~0.32 m for a 90deg turn at r=0.23 m."""
    r = float(lidar_offset_forward_m)
    dtheta_rad = math.radians(float(dtheta_deg))
    return r * (math.cos(dtheta_rad) - 1.0), r * math.sin(dtheta_rad)


def _compose_motion_hints(first: MotionHint, second: MotionHint) -> MotionHint:
    """SE(2)-compose two consecutive expected motions into one hint (used when a
    capture is discarded and its motion must carry into the next capture)."""
    theta_rad = math.radians(float(first.expected_dtheta_deg))
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    dx = float(first.expected_dx_local_m) + c * float(second.expected_dx_local_m) - s * float(second.expected_dy_local_m)
    dy = float(first.expected_dy_local_m) + s * float(second.expected_dx_local_m) + c * float(second.expected_dy_local_m)
    # Two chained turns are still a pure in-place turn (same rotation center);
    # any other mix loses that guarantee and must use the generic solver.
    composed_kind = str(second.kind) if str(first.kind) == str(second.kind) else "mixed"
    return MotionHint(
        kind=composed_kind,
        expected_dx_local_m=float(dx),
        expected_dy_local_m=float(dy),
        expected_dtheta_deg=float(first.expected_dtheta_deg) + float(second.expected_dtheta_deg),
        search_xy_m=max(float(first.search_xy_m), float(second.search_xy_m)) + 0.10,
        search_theta_window_deg=max(float(first.search_theta_window_deg), float(second.search_theta_window_deg)) + 10.0,
        label=f"{first.label}+{second.label}",
    )


def _apply_min_effective_magnitude(value: float, *, minimum_abs: float) -> float:
    minimum_abs = max(float(minimum_abs), 0.0)
    value = float(value)
    if abs(value) <= 1e-9 or minimum_abs <= 1e-9:
        return value
    if abs(value) < minimum_abs:
        return float(minimum_abs if value > 0.0 else -minimum_abs)
    return value


def _poll_remote_base_state(robot: SourcceyClient) -> tuple[dict[str, float], bool]:
    try:
        _frames, remote_state, is_fresh = robot._get_data()
    except Exception:
        remote_state = getattr(robot, "last_remote_state", {}) or {}
        return {
            "x.vel": float(remote_state.get("x.vel", 0.0) or 0.0),
            "y.vel": float(remote_state.get("y.vel", 0.0) or 0.0),
            "theta.vel": float(remote_state.get("theta.vel", 0.0) or 0.0),
        }, False

    remote_state = remote_state or {}
    return {
        "x.vel": float(remote_state.get("x.vel", 0.0) or 0.0),
        "y.vel": float(remote_state.get("y.vel", 0.0) or 0.0),
        "theta.vel": float(remote_state.get("theta.vel", 0.0) or 0.0),
    }, bool(is_fresh)


def _heading_bin_index(theta_deg: float, *, bin_count: int) -> int:
    if int(bin_count) <= 0:
        return 0
    wrapped = float(theta_deg) % 360.0
    bin_width_deg = 360.0 / float(bin_count)
    return int(math.floor(wrapped / bin_width_deg)) % int(bin_count)


def _point_in_stop_zone(angle_deg: float, distance_m: float, cfg: StopZoneConfig) -> bool:
    delta_deg = _normalize_angle_deg(float(angle_deg) - float(cfg.forward_angle_deg))
    theta = math.radians(delta_deg)
    forward_m = float(distance_m) * math.cos(theta)
    lateral_m = float(distance_m) * math.sin(theta)
    near_edge_m = max(0.0, float(cfg.min_distance_m))
    far_edge_m = max(near_edge_m, float(cfg.tripwire_distance_m) + float(cfg.tripwire_thickness_m) / 2.0)
    return (
        forward_m >= near_edge_m
        and forward_m <= far_edge_m
        and abs(lateral_m) <= float(cfg.tripwire_half_width_m)
    )


def _blocked_points_for_frame(frame, zone_cfg: StopZoneConfig) -> int:
    return sum(
        1
        for angle_deg, distance_m, _confidence in frame.points
        if _point_in_stop_zone(float(angle_deg), float(distance_m), zone_cfg)
    )


def _select_frontier_choice(
    frame,
    *,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    max_distance_m: float,
    min_range_m: float,
    min_confidence: int,
    frontier_min_distance_m: float,
    frontier_bin_deg: float,
) -> FrontierChoice | None:
    half_width = max(10.0, float(valid_angle_half_width_deg))
    bin_deg = max(1.0, float(frontier_bin_deg))
    bin_count = max(9, int(math.ceil((half_width * 2.0) / bin_deg)) + 1)
    bin_angles = np.linspace(-half_width, half_width, bin_count, dtype=np.float64)
    max_ranges = np.zeros(bin_count, dtype=np.float64)
    hit_counts = np.zeros(bin_count, dtype=np.int32)

    for angle_deg, distance_m, confidence in frame.points:
        if int(confidence) < int(min_confidence):
            continue
        distance_m = float(distance_m)
        if not (float(min_range_m) <= distance_m <= float(max_distance_m)):
            continue
        delta_deg = _normalize_angle_deg(float(angle_deg) - float(forward_angle_deg))
        if abs(delta_deg) > half_width:
            continue
        idx = int(round((delta_deg + half_width) / bin_deg))
        idx = max(0, min(bin_count - 1, idx))
        if distance_m > max_ranges[idx]:
            max_ranges[idx] = distance_m
        hit_counts[idx] += 1

    if not np.any(hit_counts):
        return None

    # Smooth the range profile so a single noisy return does not dominate heading choice.
    kernel = np.array([0.2, 0.6, 0.2], dtype=np.float64)
    smoothed_ranges = np.convolve(max_ranges, kernel, mode="same")
    open_mask = smoothed_ranges >= float(frontier_min_distance_m)
    if not np.any(open_mask):
        best_idx = int(np.argmax(smoothed_ranges))
        best_delta = float(bin_angles[best_idx])
        best_range = float(smoothed_ranges[best_idx])
        if best_range <= 0.0:
            return None
        return FrontierChoice(
            delta_deg=best_delta,
            abs_angle_deg=float(forward_angle_deg) + best_delta,
            mean_distance_m=best_range,
            width_deg=float(bin_deg),
            score=best_range,
            source="live",
        )

    segments: list[tuple[int, int]] = []
    start_idx: int | None = None
    for idx, is_open in enumerate(open_mask):
        if is_open and start_idx is None:
            start_idx = idx
        elif not is_open and start_idx is not None:
            segments.append((start_idx, idx - 1))
            start_idx = None
    if start_idx is not None:
        segments.append((start_idx, bin_count - 1))

    best_choice: FrontierChoice | None = None
    for start_idx, end_idx in segments:
        seg_ranges = smoothed_ranges[start_idx : end_idx + 1]
        seg_angles = bin_angles[start_idx : end_idx + 1]
        seg_mean = float(np.mean(seg_ranges))
        seg_width = float((end_idx - start_idx + 1) * bin_deg)
        seg_center = int((start_idx + end_idx) // 2)
        seg_delta = float(bin_angles[seg_center])
        # Favor wide, far openings, but mildly penalize large steering angles.
        seg_score = seg_mean + 0.025 * seg_width - 0.003 * abs(seg_delta)
        candidate = FrontierChoice(
            delta_deg=seg_delta,
            abs_angle_deg=float(forward_angle_deg) + seg_delta,
            mean_distance_m=seg_mean,
            width_deg=seg_width,
            score=seg_score,
            source="live",
        )
        if best_choice is None or candidate.score > best_choice.score:
            best_choice = candidate
    return best_choice


def _build_world_occupancy(
    transformed_sets: list[np.ndarray],
    *,
    resolution_m: float,
    padding_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    non_empty_sets = [points for points in transformed_sets if len(points)]
    if not non_empty_sets:
        return None
    world_points_xy = np.concatenate(non_empty_sets, axis=0).astype(np.float32, copy=False)
    mins = np.min(world_points_xy, axis=0) - float(padding_m)
    maxs = np.max(world_points_xy, axis=0) + float(padding_m)
    width = int(math.ceil((maxs[0] - mins[0]) / float(resolution_m))) + 1
    height = int(math.ceil((maxs[1] - mins[1]) / float(resolution_m))) + 1
    if width <= 1 or height <= 1:
        return None

    grid = np.zeros((height, width), dtype=bool)
    ij = np.round((world_points_xy - mins) / float(resolution_m)).astype(np.int32)
    ij[:, 0] = np.clip(ij[:, 0], 0, width - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, height - 1)
    grid[ij[:, 1], ij[:, 0]] = True
    return grid, mins.astype(np.float32), world_points_xy


def _dilate_bool_grid(grid: np.ndarray, radius_cells: int) -> np.ndarray:
    if int(radius_cells) <= 0:
        return grid
    dilated = grid.copy()
    ys, xs = np.nonzero(grid)
    for y, x in zip(ys, xs, strict=False):
        y0 = max(0, int(y) - int(radius_cells))
        y1 = min(grid.shape[0], int(y) + int(radius_cells) + 1)
        x0 = max(0, int(x) - int(radius_cells))
        x1 = min(grid.shape[1], int(x) + int(radius_cells) + 1)
        dilated[y0:y1, x0:x1] = True
    return dilated


def _select_map_frontier_choice(
    *,
    transformed_sets: list[np.ndarray],
    current_pose,
    max_distance_m: float,
    frontier_min_distance_m: float,
    frontier_bin_deg: float,
    resolution_m: float,
) -> FrontierChoice | None:
    occupancy = _build_world_occupancy(
        transformed_sets,
        resolution_m=max(0.03, float(resolution_m)),
        padding_m=max(0.35, float(max_distance_m) * 0.15),
    )
    if occupancy is None:
        return None

    occupied_grid, grid_origin_xy, world_points_xy = occupancy
    dilated_grid = _dilate_bool_grid(
        occupied_grid,
        radius_cells=max(1, int(round(0.08 / max(0.03, float(resolution_m))))),
    )
    pose_xy = np.asarray([float(current_pose.x), float(current_pose.y)], dtype=np.float32)
    theta_rad = math.radians(-float(current_pose.theta_deg))
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    local_rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    local_world_points = (world_points_xy - pose_xy) @ local_rot.T
    local_angles_deg = np.degrees(np.arctan2(local_world_points[:, 1], local_world_points[:, 0]))
    local_ranges_m = np.linalg.norm(local_world_points, axis=1)

    bin_deg = max(4.0, float(frontier_bin_deg))
    step_m = max(0.05, float(resolution_m) * 0.9)
    candidate_deltas = np.arange(-180.0, 180.0 + 1e-6, bin_deg, dtype=np.float32)
    best_choice: FrontierChoice | None = None

    for delta_deg in candidate_deltas:
        heading_world_deg = float(current_pose.theta_deg) + float(delta_deg)
        ray_theta_rad = math.radians(heading_world_deg)
        direction = np.asarray([math.cos(ray_theta_rad), math.sin(ray_theta_rad)], dtype=np.float32)

        clearance_m = 0.0
        exited_known_map = False
        hit_obstacle = False
        distance_m = step_m
        while distance_m <= float(max_distance_m):
            sample_xy = pose_xy + direction * float(distance_m)
            cell_xy = np.round((sample_xy - grid_origin_xy) / float(resolution_m)).astype(np.int32)
            cell_x = int(cell_xy[0])
            cell_y = int(cell_xy[1])
            if (
                cell_x < 0
                or cell_x >= int(dilated_grid.shape[1])
                or cell_y < 0
                or cell_y >= int(dilated_grid.shape[0])
            ):
                exited_known_map = True
                clearance_m = float(distance_m)
                break
            if bool(dilated_grid[cell_y, cell_x]):
                hit_obstacle = True
                clearance_m = float(distance_m)
                break
            clearance_m = float(distance_m)
            distance_m += step_m

        if not exited_known_map and not hit_obstacle:
            clearance_m = float(max_distance_m)

        support_mask = (
            np.abs((local_angles_deg - float(delta_deg) + 180.0) % 360.0 - 180.0)
            <= max(bin_deg * 0.75, 5.0)
        ) & (local_ranges_m <= float(max_distance_m) * 1.25)
        support_count = int(np.count_nonzero(support_mask))

        score = float(clearance_m)
        if exited_known_map:
            score += 1.75
        elif not hit_obstacle:
            score += 0.60
        if support_count <= 1:
            score += 0.25
        score -= 0.0015 * abs(float(delta_deg))
        if clearance_m < float(frontier_min_distance_m) * 0.75:
            score -= 2.5

        candidate = FrontierChoice(
            delta_deg=float(delta_deg),
            abs_angle_deg=float(heading_world_deg),
            mean_distance_m=float(clearance_m),
            width_deg=float(bin_deg),
            score=float(score),
            source="map",
        )
        if best_choice is None or candidate.score > best_choice.score:
            best_choice = candidate

    if best_choice is None:
        return None
    if best_choice.mean_distance_m <= max(0.25, float(frontier_min_distance_m) * 0.40):
        return None
    return best_choice


def _compute_drive_steer_theta_vel(
    frontier_choice: FrontierChoice | None,
    *,
    steer_gain: float,
    steer_max: float,
    steer_deadband_deg: float,
) -> float:
    if frontier_choice is None:
        return 0.0
    delta_deg = float(frontier_choice.delta_deg)
    if abs(delta_deg) <= float(steer_deadband_deg):
        return 0.0
    steer_theta_vel = float(delta_deg) * float(steer_gain)
    steer_limit = max(0.0, float(steer_max))
    return float(max(-steer_limit, min(steer_limit, steer_theta_vel)))


def _target_choice_from_world_point(
    *,
    target_x_m: float,
    target_y_m: float,
    current_pose: Pose2D,
    source: str,
) -> FrontierChoice | None:
    dx_world = float(target_x_m) - float(current_pose.x)
    dy_world = float(target_y_m) - float(current_pose.y)
    distance_m = math.hypot(dx_world, dy_world)
    if distance_m <= 1e-6:
        return None
    abs_angle_deg = math.degrees(math.atan2(dy_world, dx_world))
    delta_deg = _normalize_angle_deg(abs_angle_deg - float(current_pose.theta_deg))
    return FrontierChoice(
        delta_deg=float(delta_deg),
        abs_angle_deg=float(abs_angle_deg),
        mean_distance_m=float(distance_m),
        width_deg=0.0,
        score=float(distance_m),
        source=str(source),
    )


def _seed_explore_target_from_frontier(
    *,
    frontier_choice: FrontierChoice,
    current_pose: Pose2D,
    capture_index: int,
    step_distance_m: float,
) -> ExploreTarget:
    heading_rad = math.radians(float(frontier_choice.abs_angle_deg))
    travel_m = max(0.50, min(float(step_distance_m), float(frontier_choice.mean_distance_m)))
    return ExploreTarget(
        world_x_m=float(current_pose.x) + math.cos(heading_rad) * travel_m,
        world_y_m=float(current_pose.y) + math.sin(heading_rad) * travel_m,
        source=str(frontier_choice.source),
        seeded_capture_index=int(capture_index),
    )


def _target_distance_m(target: ExploreTarget, pose: Pose2D) -> float:
    return float(math.hypot(float(target.world_x_m) - float(pose.x), float(target.world_y_m) - float(pose.y)))


def _capture_snapshot(
    *,
    feed: DirectLidarFeed,
    output_dir: Path,
    request_index: int,
    after_frame_id: int,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    invert_lateral_axis: bool,
    max_distance_m: float,
    min_range_m: float,
    min_confidence: int,
    fresh_frame_timeout_s: float,
    fresh_frame_advances: int,
    capture_config_extra: dict[str, object] | None = None,
) -> tuple[int, object, np.ndarray]:
    armed_wall_ts = time.time()
    print(
        "[capture] waiting for fresh revolution "
        f"(after frame_id={after_frame_id}, min advances={fresh_frame_advances})"
    )
    frame_id, frame = feed.wait_for_frame_after(
        after_frame_id=after_frame_id,
        timeout_s=float(fresh_frame_timeout_s),
        min_frame_advances=int(fresh_frame_advances),
        armed_wall_ts=armed_wall_ts,
    )
    if frame is None:
        raise TimeoutError("Timed out waiting for a fresh LiDAR revolution to capture.")

    local_points_xy = _scan_to_local_points(
        points=frame.points,
        forward_angle_deg=float(forward_angle_deg),
        valid_angle_half_width_deg=float(valid_angle_half_width_deg),
        invert_lateral_axis=bool(invert_lateral_axis),
        max_distance_m=float(max_distance_m),
        min_confidence=int(min_confidence),
        min_range_m=float(min_range_m),
    )
    capture_config = {
        "forward_angle_deg": float(forward_angle_deg),
        "valid_angle_half_width_deg": float(valid_angle_half_width_deg),
        "invert_lateral_axis": bool(invert_lateral_axis),
        "max_distance_m": float(max_distance_m),
        "min_range_m": float(min_range_m),
        "min_confidence": int(min_confidence),
        "capture_mode": "wander_snapshot_stitch",
    }
    if capture_config_extra:
        capture_config.update(capture_config_extra)
    _save_snapshot(
        output_dir=output_dir,
        request_index=int(request_index),
        frame_id=int(frame_id),
        frame=frame,
        local_points_xy=local_points_xy,
        capture_config=capture_config,
    )
    snapshot_json_path = output_dir / f"snapshot_{int(request_index):03d}.json"
    snapshot_metadata = json.loads(snapshot_json_path.read_text(encoding="utf-8"))
    snapshot = Snapshot(
        name=f"snapshot_{int(request_index):03d}",
        points_xy=local_points_xy.astype(np.float32, copy=False),
        metadata=snapshot_metadata,
    )
    return int(frame_id), frame, local_points_xy, snapshot


def _save_motion_hints(output_dir: Path, motion_hints: list[MotionHint]) -> Path:
    payload = {
        "schema": "sourccey.wander_snapshot_hints.v1",
        "motion_hints": [
            {
                "kind": hint.kind,
                "label": hint.label,
                "expected_dx_local_m": round(float(hint.expected_dx_local_m), 6),
                "expected_dy_local_m": round(float(hint.expected_dy_local_m), 6),
                "expected_dtheta_deg": round(float(hint.expected_dtheta_deg), 6),
                "search_xy_m": round(float(hint.search_xy_m), 6),
                "search_theta_window_deg": round(float(hint.search_theta_window_deg), 6),
            }
            for hint in motion_hints
        ],
    }
    path = output_dir / "motion_hints.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_stitch_report(
    *,
    stitch_dir: Path,
    snapshot_dir: Path,
    transformed_sets: list[np.ndarray],
    poses,
    solve_log,
    motion_hints: list[MotionHint],
) -> tuple[Path, Path]:
    from ldlidar_direct_snapshot_stitch import _generate_svg

    svg = _generate_svg(transformed_sets, poses)
    report = {
        "schema": "sourccey.wander_snapshot_stitch.v1",
        "snapshot_dir": str(snapshot_dir.resolve()),
        "output_dir": str(stitch_dir.resolve()),
        "motion_hints": [
            {
                "kind": hint.kind,
                "label": hint.label,
                "expected_dx_local_m": round(float(hint.expected_dx_local_m), 6),
                "expected_dy_local_m": round(float(hint.expected_dy_local_m), 6),
                "expected_dtheta_deg": round(float(hint.expected_dtheta_deg), 6),
                "search_xy_m": round(float(hint.search_xy_m), 6),
                "search_theta_window_deg": round(float(hint.search_theta_window_deg), 6),
            }
            for hint in motion_hints
        ],
        "solve_log": solve_log,
        "final_pose": _pose_dict(poses[-1]),
    }
    html = HTML_TEMPLATE.format(svg=svg, report=json.dumps(report, indent=2))
    html_path = stitch_dir / "latest_stitched_overlay.html"
    report_path = stitch_dir / "latest_stitched_overlay.json"
    html_path.write_text(html, encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return html_path, report_path


def _rebuild_stitch(
    *,
    snapshot_dir: Path,
    stitch_dir: Path,
    motion_hints: list[MotionHint],
    resolution_m: float,
    fallback_search_xy_m: float,
) -> dict[str, object]:
    snapshots = _load_snapshots(snapshot_dir)
    poses, transformed_sets, solve_log = _stitch_snapshots(
        snapshots=snapshots,
        motion_hints=motion_hints,
        resolution_m=float(resolution_m),
        fallback_search_xy_m=float(fallback_search_xy_m),
    )
    html_path, report_path = _write_stitch_report(
        stitch_dir=stitch_dir,
        snapshot_dir=snapshot_dir,
        transformed_sets=transformed_sets,
        poses=poses,
        solve_log=solve_log,
        motion_hints=motion_hints,
    )
    _save_motion_hints(stitch_dir, motion_hints)
    return {
        "snapshots": snapshots,
        "poses": poses,
        "transformed_sets": transformed_sets,
        "solve_log": solve_log,
        "html_path": html_path,
        "report_path": report_path,
    }


def _append_stitch(
    *,
    stitch_dir: Path,
    snapshot_dir: Path,
    stitch_state: dict[str, object],
    motion_hints: list[MotionHint],
    new_snapshot: Snapshot,
    resolution_m: float,
    solved_pose_override: Pose2D | None = None,
    solve_meta_override: dict[str, object] | None = None,
) -> dict[str, object]:
    append_started = time.monotonic()
    if solved_pose_override is None:
        snapshots, poses, transformed_sets, solve_log = _append_stitch_snapshot(
            snapshots=list(stitch_state["snapshots"]),
            poses=list(stitch_state["poses"]),
            transformed_sets=list(stitch_state["transformed_sets"]),
            solve_log=list(stitch_state["solve_log"]),
            new_snapshot=new_snapshot,
            motion_hint=motion_hints[-1],
            resolution_m=float(resolution_m),
        )
    else:
        snapshots = [*list(stitch_state["snapshots"]), new_snapshot]
        poses = [*list(stitch_state["poses"]), solved_pose_override]
        transformed = _transform_points(new_snapshot.points_xy, solved_pose_override)
        transformed_sets = [*list(stitch_state["transformed_sets"]), transformed]
        solve_log = [
            *list(stitch_state["solve_log"]),
            {
                "snapshot": new_snapshot.name,
                "pose": _pose_dict(solved_pose_override),
                "initial_pose": _pose_dict(solved_pose_override),
                "score": None if solve_meta_override is None else solve_meta_override.get("score"),
                "solve_source": "live_relocalize_override"
                if solve_meta_override is None
                else solve_meta_override.get("source", "live_relocalize_override"),
                "local_score": None if solve_meta_override is None else solve_meta_override.get("local_score"),
                "whole_map_score": None if solve_meta_override is None else solve_meta_override.get("whole_map_score"),
                "whole_map_searched": None
                if solve_meta_override is None
                else solve_meta_override.get("whole_map_searched"),
                "timing_s": None if solve_meta_override is None else solve_meta_override.get("timing_s"),
                "motion_hint": {
                    "kind": motion_hints[-1].kind,
                    "label": motion_hints[-1].label,
                    "expected_dx_local_m": round(float(motion_hints[-1].expected_dx_local_m), 4),
                    "expected_dy_local_m": round(float(motion_hints[-1].expected_dy_local_m), 4),
                    "expected_dtheta_deg": round(float(motion_hints[-1].expected_dtheta_deg), 4),
                    "search_xy_m": round(float(motion_hints[-1].search_xy_m), 4),
                    "search_theta_window_deg": round(float(motion_hints[-1].search_theta_window_deg), 4),
                },
                "host_revolution_index": new_snapshot.metadata.get("host_revolution_index"),
                "host_point_digest": new_snapshot.metadata.get("host_point_digest"),
            },
        ]
    solve_elapsed_s = time.monotonic() - append_started
    write_started = time.monotonic()
    html_path, report_path = _write_stitch_report(
        stitch_dir=stitch_dir,
        snapshot_dir=snapshot_dir,
        transformed_sets=transformed_sets,
        poses=poses,
        solve_log=solve_log,
        motion_hints=motion_hints,
    )
    _save_motion_hints(stitch_dir, motion_hints)
    write_elapsed_s = time.monotonic() - write_started
    return {
        "snapshots": snapshots,
        "poses": poses,
        "transformed_sets": transformed_sets,
        "solve_log": solve_log,
        "html_path": html_path,
        "report_path": report_path,
        "timing": {
            "append_total_s": round(time.monotonic() - append_started, 4),
            "solve_phase_s": round(solve_elapsed_s, 4),
            "write_phase_s": round(write_elapsed_s, 4),
        },
    }


def _pose_delta_metrics(reference_pose: Pose2D, candidate_pose: Pose2D) -> tuple[float, float]:
    translation_m = math.hypot(float(candidate_pose.x) - float(reference_pose.x), float(candidate_pose.y) - float(reference_pose.y))
    theta_error_deg = abs(_normalize_angle_deg(float(candidate_pose.theta_deg) - float(reference_pose.theta_deg)))
    return float(translation_m), float(theta_error_deg)


def _estimate_pose_against_stitched_map(
    *,
    points_xy: np.ndarray,
    transformed_sets: list[np.ndarray],
    initial_pose: Pose2D,
    resolution_m: float,
    search_xy_m: float,
    theta_window_deg: float,
    max_translation_from_initial_m: float | None = None,
    prior_translation_weight: float = 0.12,
    prior_theta_weight: float = 0.04,
) -> tuple[Pose2D, dict[str, object]] | None:
    non_empty_sets = [points for points in transformed_sets if len(points)]
    if len(points_xy) < 12 or not non_empty_sets:
        return None

    global_points_xy = np.concatenate(non_empty_sets, axis=0)
    solved_pose, score_meta = _search_pose(
        snapshot_points_xy=points_xy,
        global_points_xy=global_points_xy,
        initial_pose=initial_pose,
        resolution_m=float(resolution_m),
        search_xy_m=max(0.45, float(search_xy_m)),
        coarse_angle_step_deg=4.0,
        fine_angle_step_deg=0.5,
        theta_window_deg=max(24.0, float(theta_window_deg)),
        whole_map_theta_center_deg=float(initial_pose.theta_deg),
        whole_map_theta_window_deg=max(90.0, float(theta_window_deg) * 2.0),
        max_translation_from_initial_m=max_translation_from_initial_m,
        prior_pose=initial_pose,
        prior_translation_weight=float(prior_translation_weight),
        prior_theta_weight=float(prior_theta_weight),
    )
    return solved_pose, score_meta


def _accept_relocalized_pose(
    *,
    label: str,
    candidate_pose: Pose2D,
    score_meta: dict[str, object],
    expected_pose: Pose2D,
    motion_hint: MotionHint | None,
    max_translation_error_m: float,
    max_theta_error_deg: float,
    min_score: float,
) -> bool:
    score = float(score_meta.get("score") or -1e9)
    local_score = float(score_meta.get("local_score") or -1e9)
    whole_map_score = float(score_meta.get("whole_map_score") or -1e9)
    source = str(score_meta.get("source") or "unknown")
    translation_error_m, theta_error_deg = _pose_delta_metrics(expected_pose, candidate_pose)

    rejection_reasons: list[str] = []
    if score < float(min_score):
        rejection_reasons.append(f"score={score:.3f}<min={float(min_score):.3f}")
    if translation_error_m > float(max_translation_error_m):
        rejection_reasons.append(
            f"translation_error={translation_error_m:.3f}m>max={float(max_translation_error_m):.3f}m"
        )
    if theta_error_deg > float(max_theta_error_deg):
        rejection_reasons.append(
            f"theta_error={theta_error_deg:.1f}deg>max={float(max_theta_error_deg):.1f}deg"
        )
    if source == "whole_map" and whole_map_score < (local_score + 0.35):
        rejection_reasons.append(
            f"whole_map_margin_too_small whole={whole_map_score:.3f} local={local_score:.3f}"
        )

    if rejection_reasons:
        hint_label = "none" if motion_hint is None else motion_hint.label
        hint_kind = "none" if motion_hint is None else motion_hint.kind
        print(
            "[wander] relocalization rejected "
            f"(label={label}, source={source}, hint={hint_kind}:{hint_label}, "
            f"candidate=({float(candidate_pose.x):.3f}, {float(candidate_pose.y):.3f}, "
            f"{float(candidate_pose.theta_deg):.1f}deg), expected=({float(expected_pose.x):.3f}, "
            f"{float(expected_pose.y):.3f}, {float(expected_pose.theta_deg):.1f}deg), "
            f"score={score:.3f}, local_score={local_score:.3f}, whole_map_score={whole_map_score:.3f}, "
            f"reasons={'; '.join(rejection_reasons)})"
        )
        return False

    print(
        "[wander] relocalization accepted "
        f"(label={label}, source={source}, score={score:.3f}, local_score={local_score:.3f}, "
        f"whole_map_score={whole_map_score:.3f}, translation_error={translation_error_m:.3f}m, "
        f"theta_error={theta_error_deg:.1f}deg)"
    )
    return True


def _log_live_pose_state(
    rr,
    *,
    capture_index: int,
    pose: Pose2D,
    points_xy: np.ndarray,
    cone_half_width_deg: float,
) -> None:
    rr.set_time("capture_index", sequence=int(capture_index))
    origin = np.asarray([[pose.x, pose.y, 0.0]], dtype=np.float32)
    vector = np.asarray(
        [[0.30 * math.cos(math.radians(pose.theta_deg)), 0.30 * math.sin(math.radians(pose.theta_deg)), 0.0]],
        dtype=np.float32,
    )
    rr.log("world/live_pose/origin", rr.Points3D(origin, colors=[[0, 255, 255]], radii=0.06))
    rr.log("world/live_pose/heading", rr.Arrows3D(origins=origin, vectors=vector, colors=[[0, 255, 255]]))
    arc = np.asarray(
        [
            [pose.x + 0.45 * math.cos(math.radians(pose.theta_deg + delta_deg)),
             pose.y + 0.45 * math.sin(math.radians(pose.theta_deg + delta_deg)),
             0.0]
            for delta_deg in np.linspace(-float(cone_half_width_deg), float(cone_half_width_deg), 36)
        ],
        dtype=np.float32,
    )
    rr.log("world/live_pose/cone", rr.LineStrips3D([arc], colors=[[0, 255, 255]], radii=0.005))
    if len(points_xy):
        world_points_xy = _transform_points(points_xy, pose)
        world_points_xyz = np.column_stack(
            [world_points_xy[:, 0], world_points_xy[:, 1], np.zeros((len(world_points_xy),), dtype=np.float32)]
        )
        rr.log("world/live_scan", rr.Points3D(world_points_xyz, colors=[[130, 200, 255]], radii=0.01))


def _drive_forward_burst(
    *,
    robot: SourcceyClient,
    feed: DirectLidarFeed,
    zone_cfg: StopZoneConfig,
    forward_speed: float,
    burst_s: float,
    steer_theta_vel: float,
    min_effective_move_speed: float,
) -> DriveBurstMeta:
    start = time.monotonic()
    iterations = 0
    host_feedback_updates = 0
    host_forward_distance_estimate_m = 0.0
    host_x_vel_peak = 0.0
    host_theta_vel_peak = 0.0
    host_x_vel_last = 0.0
    host_theta_vel_last = 0.0
    effective_forward_speed = _apply_min_effective_magnitude(
        float(forward_speed),
        minimum_abs=float(min_effective_move_speed),
    )
    while (time.monotonic() - start) < float(burst_s):
        frame_id, frame = feed.latest()
        if frame is not None:
            blocked_points = _blocked_points_for_frame(frame, zone_cfg)
            if blocked_points >= zone_cfg.blocked_trigger_count():
                print(
                    "[drive] stop box became occupied during forward burst "
                    f"(frame_id={frame_id}, blocked_points={blocked_points}, baseline={zone_cfg.baseline_points})"
                )
                break
        robot.send_action(
            {
                "x.vel": float(effective_forward_speed),
                "y.vel": 0.0,
                "theta.vel": float(steer_theta_vel),
                "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
                "untorque_left": True,
                "untorque_right": True,
            }
        )
        iterations += 1
        remote_base_state, is_fresh_state = _poll_remote_base_state(robot)
        host_x_vel_last = float(remote_base_state.get("x.vel", 0.0))
        host_theta_vel_last = float(remote_base_state.get("theta.vel", 0.0))
        host_x_vel_peak = max(host_x_vel_peak, abs(host_x_vel_last))
        host_theta_vel_peak = max(host_theta_vel_peak, abs(host_theta_vel_last))
        if is_fresh_state:
            host_feedback_updates += 1
            host_forward_distance_estimate_m += float(host_x_vel_last) * 0.05
        time.sleep(0.05)
    _send_stop(robot)
    return DriveBurstMeta(
        elapsed_s=float(time.monotonic() - start),
        iterations=int(iterations),
        host_feedback_updates=int(host_feedback_updates),
        host_forward_distance_estimate_m=float(host_forward_distance_estimate_m),
        host_x_vel_peak=float(host_x_vel_peak),
        host_theta_vel_peak=float(host_theta_vel_peak),
        host_x_vel_last=float(host_x_vel_last),
        host_theta_vel_last=float(host_theta_vel_last),
    )


def _run_drive_sequence(
    *,
    robot: SourcceyClient,
    feed: DirectLidarFeed,
    zone_cfg: StopZoneConfig,
    forward_speed: float,
    burst_s: float,
    burst_count: int,
    inter_burst_pause_s: float,
    steer_theta_vel: float,
    min_effective_move_speed: float,
) -> DriveSequenceMeta:
    total_elapsed_s = 0.0
    total_iterations = 0
    bursts_completed = 0
    stopped_by_block = False
    last_blocked_points = 0
    host_feedback_updates = 0
    host_forward_distance_estimate_m = 0.0
    host_x_vel_peak = 0.0
    host_theta_vel_peak = 0.0
    host_x_vel_last = 0.0
    host_theta_vel_last = 0.0
    effective_forward_speed = _apply_min_effective_magnitude(
        float(forward_speed),
        minimum_abs=float(min_effective_move_speed),
    )

    for burst_index in range(max(1, int(burst_count))):
        burst_meta = _drive_forward_burst(
            robot=robot,
            feed=feed,
            zone_cfg=zone_cfg,
            forward_speed=float(forward_speed),
            burst_s=float(burst_s),
            steer_theta_vel=float(steer_theta_vel),
            min_effective_move_speed=float(min_effective_move_speed),
        )
        total_elapsed_s += float(burst_meta.elapsed_s)
        total_iterations += int(burst_meta.iterations)
        bursts_completed += 1
        host_feedback_updates += int(burst_meta.host_feedback_updates)
        host_forward_distance_estimate_m += float(burst_meta.host_forward_distance_estimate_m)
        host_x_vel_peak = max(host_x_vel_peak, float(burst_meta.host_x_vel_peak))
        host_theta_vel_peak = max(host_theta_vel_peak, float(burst_meta.host_theta_vel_peak))
        host_x_vel_last = float(burst_meta.host_x_vel_last)
        host_theta_vel_last = float(burst_meta.host_theta_vel_last)

        frame_id, frame = feed.latest()
        if frame is not None:
            last_blocked_points = _blocked_points_for_frame(frame, zone_cfg)
            print(
                "[drive] burst checkpoint "
                f"(index={burst_index + 1}/{int(burst_count)}, frame_id={frame_id}, blocked_points={last_blocked_points}, "
                f"cmd_x={effective_forward_speed:.3f}, cmd_theta={float(steer_theta_vel):.3f}, "
                f"host_dx_est={burst_meta.host_forward_distance_estimate_m:.3f}, "
                f"host_updates={burst_meta.host_feedback_updates}, host_x_peak={burst_meta.host_x_vel_peak:.3f}, "
                f"host_theta_peak={burst_meta.host_theta_vel_peak:.3f}, host_x_last={burst_meta.host_x_vel_last:.3f}, "
                f"host_theta_last={burst_meta.host_theta_vel_last:.3f})"
            )
            if last_blocked_points >= zone_cfg.blocked_trigger_count():
                stopped_by_block = True
                break
        if burst_index < int(burst_count) - 1 and float(inter_burst_pause_s) > 0.0:
            time.sleep(float(inter_burst_pause_s))

    return DriveSequenceMeta(
        elapsed_s=float(total_elapsed_s),
        iterations=int(total_iterations),
        bursts_completed=int(bursts_completed),
        stopped_by_block=bool(stopped_by_block),
        blocked_points=int(last_blocked_points),
        commanded_forward_speed=float(effective_forward_speed),
        steer_theta_vel=float(steer_theta_vel),
        host_feedback_updates=int(host_feedback_updates),
        host_forward_distance_estimate_m=float(host_forward_distance_estimate_m),
        host_x_vel_peak=float(host_x_vel_peak),
        host_theta_vel_peak=float(host_theta_vel_peak),
        host_x_vel_last=float(host_x_vel_last),
        host_theta_vel_last=float(host_theta_vel_last),
    )


def _turn_with_arc_tracking(
    *,
    robot: SourcceyClient,
    feed: DirectLidarFeed,
    transformed_sets: list[np.ndarray],
    start_pose: Pose2D,
    lidar_offset_forward_m: float,
    resolution_m: float,
    target_turn_deg: float,
    direction_sign: float,
    turn_speed: float,
    turn_burst_s: float,
    turn_settle_s: float,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    invert_lateral_axis: bool,
    max_distance_m: float,
    min_range_m: float,
    min_confidence: int,
    stop_tolerance_deg: float,
    max_bursts: int,
    min_track_score: float = 5.0,
    track_theta_half_window_deg: float = 32.0,
    max_burst_delta_deg: float = 50.0,
    stall_bursts_before_boost: int = 3,
    stall_speed_boost: float = 1.3,
    max_speed_scale: float = 2.0,
    max_consecutive_held: int = 3,
) -> dict[str, float]:
    """Rotate in place while tracking the pose CONTINUOUSLY with the lidar.

    After every burst (robot momentarily at rest) the fresh revolution is
    arc-solved against the stitched map inside a small theta window around the
    last known heading. Between bursts the robot turns at most a few tens of
    degrees, so the solve is unambiguous — the ~90deg symmetry modes of a
    square room are simply outside the window. The turn stops when the
    lidar-tracked rotation reaches the target, so the amount turned is
    measured geometry, not commanded wheel motion."""
    non_empty_sets = [points for points in transformed_sets if len(points)]
    global_points_xy = np.concatenate(non_empty_sets, axis=0) if non_empty_sets else np.zeros((0, 2), np.float32)
    grids = _build_score_grids(global_points_xy, resolution_m=float(resolution_m), padding_m=0.9)
    r = float(lidar_offset_forward_m)
    start_theta_rad = math.radians(float(start_pose.theta_deg))
    arc_center_xy = (
        float(start_pose.x) - r * math.cos(start_theta_rad),
        float(start_pose.y) - r * math.sin(start_theta_rad),
    )

    current_theta_deg = float(start_pose.theta_deg)
    turned_deg = 0.0
    missed_updates = 0
    consecutive_held = 0
    lock_lost = False
    consecutive_timeouts = 0
    stall_count = 0
    speed_scale = 1.0
    last_frame_id = -1
    frame_wait_timeout_s = max(2.0, float(turn_burst_s) + float(turn_settle_s) + 1.5)

    for burst_index in range(1, int(max_bursts) + 1):
        _execute_turn_burst(
            robot=robot,
            direction_sign=float(direction_sign),
            turn_speed=float(turn_speed) * float(speed_scale),
            turn_burst_s=float(turn_burst_s),
            turn_settle_s=float(turn_settle_s),
        )
        frame_id, frame = feed.wait_for_frame_after(
            after_frame_id=int(last_frame_id),
            timeout_s=float(frame_wait_timeout_s),
            min_frame_advances=1,
        )
        if frame is None or int(frame_id) == int(last_frame_id):
            consecutive_timeouts += 1
            print(f"[turn] burst={burst_index:02d} timed out waiting for a fresh revolution")
            if consecutive_timeouts >= 3:
                raise RuntimeError("LiDAR feed stalled while tracking an arc turn.")
            continue
        consecutive_timeouts = 0
        last_frame_id = int(frame_id)

        points_xy = _scan_to_local_points(
            points=frame.points,
            forward_angle_deg=float(forward_angle_deg),
            valid_angle_half_width_deg=float(valid_angle_half_width_deg),
            invert_lateral_axis=bool(invert_lateral_axis),
            max_distance_m=float(max_distance_m),
            min_confidence=int(min_confidence),
            min_range_m=float(min_range_m),
        )
        if len(points_xy) < 12:
            missed_updates += 1
            consecutive_held += 1
            print(f"[turn] burst={burst_index:02d} too few valid points to track; holding")
            if consecutive_held >= max(1, int(max_consecutive_held)):
                lock_lost = True
                print(
                    f"[turn] burst={burst_index:02d} tracking lock lost; "
                    "stopping the turn instead of rotating blind"
                )
                break
        else:
            # Window slightly leads in the commanded direction; per-burst
            # rotation cannot reach the next symmetry mode from here.
            expected_theta_deg = current_theta_deg + float(direction_sign) * 8.0
            solved_pose, solved_score = _solve_arc_pose_on_grids(
                snapshot_points_xy=points_xy,
                grids=grids,
                arc_center_xy=arc_center_xy,
                lidar_offset_forward_m=r,
                expected_theta_deg=float(expected_theta_deg),
                theta_half_window_deg=float(track_theta_half_window_deg),
                theta_step_deg=1.5,
                center_slack_m=0.08,
                slack_step_m=0.04,
                theta_prior_weight_per_deg=0.01,
                refine=False,
            )
            delta_deg = _normalize_angle_deg(float(solved_pose.theta_deg) - current_theta_deg)
            delta_along_deg = float(delta_deg) * float(direction_sign)
            if (
                float(solved_score) >= float(min_track_score)
                and -8.0 <= delta_along_deg <= float(max_burst_delta_deg)
            ):
                current_theta_deg = float(solved_pose.theta_deg)
                turned_deg += float(delta_deg)
                consecutive_held = 0
                if abs(delta_deg) >= 1.5:
                    stall_count = 0
                    speed_scale = 1.0
                else:
                    # Locked but not moving: genuine wheel stall, safe to boost.
                    stall_count += 1
                print(
                    f"[turn] burst={burst_index:02d} tracked={abs(turned_deg):6.1f}deg "
                    f"(delta={delta_along_deg:+5.1f}deg) target={float(target_turn_deg):5.1f}deg "
                    f"score={float(solved_score):0.3f}"
                )
            else:
                # Low-confidence solve: the robot may still be rotating but we
                # cannot see how far. NEVER boost here, and stop the turn after
                # a few held bursts — continuing means rotating blind, which is
                # how the map got corrupted before.
                missed_updates += 1
                consecutive_held += 1
                print(
                    f"[turn] burst={burst_index:02d} tracked={abs(turned_deg):6.1f}deg "
                    f"(solve held: delta={delta_along_deg:+5.1f}deg score={float(solved_score):0.3f}) "
                    f"target={float(target_turn_deg):5.1f}deg"
                )
                if consecutive_held >= max(1, int(max_consecutive_held)):
                    lock_lost = True
                    print(
                        f"[turn] burst={burst_index:02d} tracking lock lost; "
                        "stopping the turn instead of rotating blind"
                    )
                    break

        if stall_count >= max(1, int(stall_bursts_before_boost)) and speed_scale < float(max_speed_scale):
            speed_scale = min(float(max_speed_scale), speed_scale * float(stall_speed_boost))
            stall_count = 0
            print(f"[turn] burst={burst_index:02d} no rotation progress; boosting turn speed (scale={speed_scale:.2f})")

        if abs(turned_deg) >= (float(target_turn_deg) - float(stop_tolerance_deg)):
            break

    _send_stop(robot)
    reached = abs(turned_deg) >= (float(target_turn_deg) - float(stop_tolerance_deg))
    if not reached:
        print(
            "[turn] warning: arc-tracked turn ended before reaching its target "
            f"(tracked={abs(turned_deg):.1f}deg of {float(target_turn_deg):.1f}deg, "
            f"missed_updates={missed_updates}, lock_lost={lock_lost})"
        )
    return {
        "turned_deg": float(turned_deg),
        "final_theta_deg": float(current_theta_deg),
        "missed_updates": float(missed_updates),
        "lock_lost": 1.0 if lock_lost else 0.0,
        "completed": 1.0 if reached else 0.0,
    }


def _solve_drive_step_pose(
    *,
    points_xy: np.ndarray,
    grids: dict[str, object],
    current_pose: Pose2D,
    max_step_m: float = 0.60,
    lateral_slack_m: float = 0.15,
    theta_half_window_deg: float = 12.0,
) -> tuple[Pose2D, float]:
    """Solve one drive-burst pose update: the robot moved at most one burst
    forward from `current_pose`, so search a short strip ahead (with a little
    lateral slack and a small heading window). Small windows keep the solve
    unambiguous, the same principle as the arc-tracked turn."""
    heading_rad = math.radians(float(current_pose.theta_deg))
    c = math.cos(heading_rad)
    s = math.sin(heading_rad)
    best_pose = current_pose
    best_score = -1e9
    for dtheta_deg in np.arange(-theta_half_window_deg, theta_half_window_deg + 1e-6, 2.0, dtype=np.float32):
        for forward_m in np.arange(-0.05, max_step_m + 1e-6, 0.05, dtype=np.float32):
            for lateral_m in np.arange(-lateral_slack_m, lateral_slack_m + 1e-6, 0.05, dtype=np.float32):
                pose = Pose2D(
                    x=float(current_pose.x) + c * float(forward_m) - s * float(lateral_m),
                    y=float(current_pose.y) + s * float(forward_m) + c * float(lateral_m),
                    theta_deg=float(current_pose.theta_deg) + float(dtheta_deg),
                )
                score = _score_candidate(
                    _transform_points(points_xy, pose),
                    use_nearest_penalty=False,
                    **grids,
                )
                if score > best_score:
                    best_score = float(score)
                    best_pose = pose
    return best_pose, float(best_score)


def _drive_with_tracking(
    *,
    robot: SourcceyClient,
    feed: DirectLidarFeed,
    zone_cfg: StopZoneConfig,
    transformed_sets: list[np.ndarray],
    start_pose: Pose2D,
    resolution_m: float,
    forward_speed: float,
    min_effective_move_speed: float,
    burst_s: float,
    burst_count: int,
    inter_burst_pause_s: float,
    steer_theta_vel: float,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    invert_lateral_axis: bool,
    max_distance_m: float,
    min_range_m: float,
    min_confidence: int,
    min_track_score: float = 5.0,
    max_consecutive_held: int = 2,
) -> tuple[Pose2D, dict[str, object]]:
    """Drive forward burst-by-burst while tracking the pose with the lidar
    after every burst. The drive stops when blocked, when the plan completes,
    or when tracking loses lock — the robot never travels more than one burst
    beyond its last confirmed pose, so there is no big-jump solve afterwards."""
    non_empty_sets = [points for points in transformed_sets if len(points)]
    global_points_xy = np.concatenate(non_empty_sets, axis=0) if non_empty_sets else np.zeros((0, 2), np.float32)
    grids = _build_score_grids(global_points_xy, resolution_m=float(resolution_m), padding_m=1.4)

    current_pose = start_pose
    bursts_completed = 0
    stopped_by_block = False
    lock_lost = False
    consecutive_held = 0
    missed_updates = 0
    last_blocked_points = 0
    last_frame_id = -1
    elapsed_total_s = 0.0
    frame_wait_timeout_s = max(2.0, float(burst_s) + 1.5)

    for burst_index in range(max(1, int(burst_count))):
        burst_meta = _drive_forward_burst(
            robot=robot,
            feed=feed,
            zone_cfg=zone_cfg,
            forward_speed=float(forward_speed),
            burst_s=float(burst_s),
            steer_theta_vel=float(steer_theta_vel),
            min_effective_move_speed=float(min_effective_move_speed),
        )
        bursts_completed += 1
        elapsed_total_s += float(burst_meta.elapsed_s)

        frame_id, frame = feed.wait_for_frame_after(
            after_frame_id=int(last_frame_id),
            timeout_s=float(frame_wait_timeout_s),
            min_frame_advances=1,
        )
        if frame is None:
            missed_updates += 1
            consecutive_held += 1
            print(f"[drive] burst={burst_index + 1:02d} no fresh revolution to track against")
        else:
            last_frame_id = int(frame_id)
            points_xy = _scan_to_local_points(
                points=frame.points,
                forward_angle_deg=float(forward_angle_deg),
                valid_angle_half_width_deg=float(valid_angle_half_width_deg),
                invert_lateral_axis=bool(invert_lateral_axis),
                max_distance_m=float(max_distance_m),
                min_confidence=int(min_confidence),
                min_range_m=float(min_range_m),
            )
            solved_pose, solved_score = _solve_drive_step_pose(
                points_xy=points_xy,
                grids=grids,
                current_pose=current_pose,
            )
            step_m = math.hypot(solved_pose.x - current_pose.x, solved_pose.y - current_pose.y)
            if solved_score >= float(min_track_score):
                current_pose = solved_pose
                consecutive_held = 0
                print(
                    f"[drive] burst={burst_index + 1:02d} tracked pose=({current_pose.x:.3f}, "
                    f"{current_pose.y:.3f}, {current_pose.theta_deg:.1f}deg) step={step_m:.3f}m "
                    f"score={solved_score:.3f}"
                )
            else:
                missed_updates += 1
                consecutive_held += 1
                print(
                    f"[drive] burst={burst_index + 1:02d} solve held "
                    f"(score={solved_score:.3f}); not updating pose"
                )
            last_blocked_points = _blocked_points_for_frame(frame, zone_cfg)
            if last_blocked_points >= zone_cfg.blocked_trigger_count():
                stopped_by_block = True
                print(
                    f"[drive] burst={burst_index + 1:02d} stop box occupied "
                    f"(blocked_points={last_blocked_points}, baseline={zone_cfg.baseline_points})"
                )
                break

        if consecutive_held >= max(1, int(max_consecutive_held)):
            lock_lost = True
            print(
                f"[drive] burst={burst_index + 1:02d} tracking lock lost; "
                "stopping the drive instead of moving blind"
            )
            break
        if burst_index < int(burst_count) - 1 and float(inter_burst_pause_s) > 0.0:
            time.sleep(float(inter_burst_pause_s))

    _send_stop(robot)
    return current_pose, {
        "bursts_completed": int(bursts_completed),
        "elapsed_s": float(elapsed_total_s),
        "stopped_by_block": bool(stopped_by_block),
        "blocked_points": int(last_blocked_points),
        "lock_lost": bool(lock_lost),
        "missed_updates": int(missed_updates),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatic wandering LiDAR capture + offline-style stitcher with live Rerun output."
    )
    parser.add_argument("--remote-ip", default="192.168.1.237", help="Sourccey host IP.")
    parser.add_argument("--robot-id", default="sourccey")
    parser.add_argument("--lidar-host", default="192.168.1.237", help="Pi LiDAR stream host.")
    parser.add_argument("--lidar-port", type=int, default=8765)
    parser.add_argument("--output-dir", default="artifacts/wander_snapshot_stitch")
    parser.add_argument(
        "--max-captures",
        type=int,
        default=0,
        help="Maximum snapshots to collect including the initial one. Use 0 for no limit.",
    )
    parser.add_argument("--forward-angle-deg", type=float, default=DEFAULT_LIDAR_FORWARD_ANGLE_DEG)
    parser.add_argument("--valid-angle-half-width-deg", type=float, default=180.0)
    parser.add_argument("--invert-lateral-axis", action="store_true", default=True)
    parser.add_argument("--no-invert-lateral-axis", dest="invert_lateral_axis", action="store_false")
    parser.add_argument("--max-distance-m", type=float, default=8.0)
    parser.add_argument("--min-range-m", type=float, default=0.03)
    parser.add_argument("--min-confidence", type=int, default=0)
    parser.add_argument("--fresh-frame-timeout-s", type=float, default=3.0)
    parser.add_argument("--fresh-frame-advances", type=int, default=1)
    parser.add_argument("--capture-settle-s", type=float, default=0.90)
    parser.add_argument("--tripwire-distance-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_DISTANCE_M)
    parser.add_argument("--tripwire-half-width-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_HALF_WIDTH_M)
    parser.add_argument("--tripwire-thickness-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_THICKNESS_M)
    parser.add_argument("--min-distance-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_MIN_DISTANCE_M)
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument("--move-speed", type=float, default=0.85)
    parser.add_argument(
        "--min-effective-move-speed",
        type=float,
        default=0.75,
        help="Minimum absolute x.vel command to use once a forward burst is requested so wheel stiction is overcome.",
    )
    parser.add_argument("--move-burst-s", type=float, default=1.40)
    parser.add_argument(
        "--drive-bursts-per-capture",
        type=int,
        default=4,
        help="How many forward bursts to chain together before stopping for the next stitched capture.",
    )
    parser.add_argument(
        "--max-drive-bursts-per-capture",
        type=int,
        default=6,
        help="Upper cap on chained forward bursts when the frontier ahead is especially open.",
    )
    parser.add_argument(
        "--inter-burst-pause-s",
        type=float,
        default=0.02,
        help="Short pause between chained forward bursts.",
    )
    parser.add_argument("--move-settle-s", type=float, default=0.70)
    parser.add_argument(
        "--frontier-drive-steer-gain",
        type=float,
        default=0.0035,
        help="Converts small residual frontier angle error in degrees into forward-drive yaw rate after heading alignment.",
    )
    parser.add_argument(
        "--max-drive-steer-theta-vel",
        type=float,
        default=0.10,
        help="Maximum residual yaw rate applied while driving toward a frontier.",
    )
    parser.add_argument(
        "--drive-steer-deadband-deg",
        type=float,
        default=10.0,
        help="If the chosen frontier is within this heading error, drive straight with zero yaw correction.",
    )
    parser.add_argument("--drive-hint-mps", type=float, default=0.75, help="Used only as the initial translation guess for stitching.")
    parser.add_argument("--drive-search-xy-m", type=float, default=0.35)
    parser.add_argument("--drive-theta-window-deg", type=float, default=12.0)
    parser.add_argument("--turn-direction", choices=("ccw", "cw"), default="ccw")
    parser.add_argument("--turn-deg", type=float, default=90.0)
    parser.add_argument("--turn-speed", type=float, default=0.82)
    parser.add_argument("--turn-burst-s", type=float, default=0.14)
    parser.add_argument("--turn-settle-s", type=float, default=0.18)
    parser.add_argument("--rotation-signature-min-distance-m", type=float, default=0.10)
    parser.add_argument("--rotation-signature-bin-deg", type=float, default=3.0)
    parser.add_argument("--stop-tolerance-deg", type=float, default=12.0)
    parser.add_argument("--min-turn-overlap-ratio", type=float, default=0.18)
    parser.add_argument("--max-turn-bursts", type=int, default=32)
    parser.add_argument("--turn-search-xy-m", type=float, default=0.30)
    parser.add_argument("--turn-theta-window-deg", type=float, default=54.0)
    parser.add_argument("--frontier-min-distance-m", type=float, default=1.20)
    parser.add_argument("--frontier-bin-deg", type=float, default=6.0)
    parser.add_argument(
        "--frontier-goal-step-m",
        type=float,
        default=1.35,
        help="World-space distance to step toward a stitched-map frontier before reseeding another exploration goal.",
    )
    parser.add_argument(
        "--frontier-goal-reached-m",
        type=float,
        default=0.45,
        help="Distance threshold for considering a stitched-map exploration target reached.",
    )
    parser.add_argument(
        "--frontier-align-threshold-deg",
        type=float,
        default=45.0,
        help="Only used for blocked/periodic scan turns; smart mode now prefers driving forward when the stop box is clear.",
    )
    parser.add_argument(
        "--wander-mode",
        choices=("smart", "scan_turn_every_capture", "turn_only"),
        default="smart",
        help=(
            "smart: drive when clear and rotate when blocked, with occasional scan turns; "
            "scan_turn_every_capture: drive burst then rotate before every capture; "
            "turn_only: never drive, just rotate/capture."
        ),
    )
    parser.add_argument(
        "--turn-every-capture",
        action="store_true",
        default=None,
        help="Legacy alias for scan_turn_every_capture behavior.",
    )
    parser.add_argument(
        "--no-turn-every-capture",
        dest="turn_every_capture",
        action="store_false",
        default=None,
        help="Legacy alias for smart behavior.",
    )
    parser.add_argument(
        "--scan-turn-interval",
        type=int,
        default=6,
        help="In smart mode, insert a scan turn after this many clear forward captures.",
    )
    parser.add_argument(
        "--bootstrap-turn-captures",
        type=int,
        default=4,
        help="In smart mode, spend the first N captures rotating in place to build an initial stitched room outline before driving.",
    )
    parser.add_argument("--stitch-resolution-m", type=float, default=0.03)
    parser.add_argument(
        "--lidar-offset-forward-m",
        type=float,
        default=0.2286,
        help="How far the LiDAR sits forward of the robot's rotation center (9in default). "
        "Used to predict the sensor translation caused by in-place turns.",
    )
    parser.add_argument(
        "--min-append-score",
        type=float,
        default=5.5,
        help="Minimum stitch-solver score required to append a capture to the map. "
        "Captures scoring below this are discarded and retried instead of corrupting the map.",
    )
    parser.add_argument(
        "--max-consecutive-append-discards",
        type=int,
        default=2,
        help="After this many consecutive discarded captures, the best available pose is "
        "accepted anyway (with a warning) so the run cannot stall forever.",
    )
    parser.add_argument("--rerun-mode", choices=("web", "local"), default="web")
    parser.add_argument("--rerun-grpc-port", type=int, default=9877)
    parser.add_argument("--rerun-web-port", type=int, default=9878)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    snapshot_dir = output_dir / "snapshots"
    stitch_dir = output_dir / "offline_stitch"
    _ensure_clean_directory(output_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stitch_dir.mkdir(parents=True, exist_ok=True)

    rr, viewer_url = _init_rerun(
        session_name="ldlidar_wander_snapshot_stitch",
        mode=args.rerun_mode,
        grpc_port=int(args.rerun_grpc_port),
        web_port=int(args.rerun_web_port),
    )

    zone_cfg = StopZoneConfig(
        forward_angle_deg=float(args.forward_angle_deg),
        min_distance_m=float(args.min_distance_m),
        tripwire_distance_m=float(args.tripwire_distance_m),
        tripwire_half_width_m=float(args.tripwire_half_width_m),
        tripwire_thickness_m=float(args.tripwire_thickness_m),
        min_points_to_trigger=max(int(args.min_points), 1),
    )

    feed = DirectLidarFeed(args.lidar_host, int(args.lidar_port))
    feed.start()

    robot = SourcceyClient(SourcceyClientConfig(id=args.robot_id, remote_ip=args.remote_ip))
    robot.connect()
    _send_stop(robot)

    direction_sign = 1.0 if args.turn_direction == "ccw" else -1.0
    signed_turn_deg = float(args.turn_deg) * float(direction_sign)
    wander_mode = str(args.wander_mode)
    if args.turn_every_capture is True:
        wander_mode = "scan_turn_every_capture"
    elif args.turn_every_capture is False and str(args.wander_mode) == "scan_turn_every_capture":
        wander_mode = "smart"

    print(
        "[wander] startup "
        f"mode={wander_mode} "
        f"move_speed={float(args.move_speed):.2f} "
        f"min_effective_move_speed={float(args.min_effective_move_speed):.2f} "
        f"move_burst_s={float(args.move_burst_s):.2f} "
        f"drive_bursts_per_capture={int(args.drive_bursts_per_capture)} "
        f"drive_steer_gain={float(args.frontier_drive_steer_gain):.3f} "
        f"drive_steer_max={float(args.max_drive_steer_theta_vel):.2f} "
        f"turn_deg={float(args.turn_deg):.1f} "
        f"turn_direction={args.turn_direction} "
        f"bootstrap_turn_captures={int(args.bootstrap_turn_captures)} "
        f"scan_turn_interval={int(args.scan_turn_interval)} "
        f"lidar_offset_forward={float(args.lidar_offset_forward_m):.3f}m "
        f"min_append_score={float(args.min_append_score):.2f} "
        f"frontier=(min_distance={float(args.frontier_min_distance_m):.2f}m, "
        f"bin={float(args.frontier_bin_deg):.1f}deg, "
        f"align_threshold={float(args.frontier_align_threshold_deg):.1f}deg, "
        f"goal_step={float(args.frontier_goal_step_m):.2f}m, "
        f"goal_reached={float(args.frontier_goal_reached_m):.2f}m) "
        f"stop_box=(forward={float(args.forward_angle_deg):.1f}deg, "
        f"min={float(args.min_distance_m):.2f}m, "
        f"depth={float(args.tripwire_distance_m):.2f}m, "
        f"half_width={float(args.tripwire_half_width_m):.2f}m, "
        f"thickness={float(args.tripwire_thickness_m):.2f}m, "
        f"min_points={int(args.min_points)})"
    )
    motion_hints: list[MotionHint] = [
        MotionHint(
            kind="start",
            expected_dx_local_m=0.0,
            expected_dy_local_m=0.0,
            expected_dtheta_deg=0.0,
            search_xy_m=float(args.drive_search_xy_m),
            search_theta_window_deg=float(args.turn_theta_window_deg),
            label="initial_capture",
        )
    ]
    drive_checkpoints_since_scan = 0
    pending_turn_reason: str | None = None
    pending_turn_direction_sign = float(direction_sign)
    pending_turn_deg = float(args.turn_deg)
    force_drive_after_turn = False
    rotation_coverage_bin_count = 4
    rotation_coverage_bins_seen: set[int] = set()
    rotation_coverage_complete = False
    consecutive_turn_captures = 0
    active_explore_target: ExploreTarget | None = None
    active_explore_target_stall_count = 0
    active_explore_target_last_distance_m: float | None = None
    active_explore_target_blocked_count = 0
    force_live_frontier_cycles = 0
    pending_motion_hint: MotionHint | None = None
    consecutive_append_discards = 0

    try:
        last_frame_id, latest_frame = _wait_for_initial_frame(feed, timeout_s=5.0)
        print(
            "[wander] LiDAR feed ready "
            f"(frame_id={last_frame_id}, host_rev={latest_frame.revolution_index}, viewer={viewer_url or 'local'})"
        )

        # Measure how many points the stationary lidar ALWAYS reports inside the
        # stop box (robot shell / mounts). Blocking then triggers only on points
        # above this baseline; otherwise the robot believes it is permanently
        # blocked and spends the whole run rotating in place.
        baseline_samples: list[int] = []
        baseline_frame_id = int(last_frame_id)
        for _sample_index in range(10):
            sample_frame_id, sample_frame = feed.wait_for_frame_after(
                after_frame_id=baseline_frame_id,
                timeout_s=1.0,
                min_frame_advances=1,
            )
            if sample_frame is None:
                break
            baseline_samples.append(_blocked_points_for_frame(sample_frame, zone_cfg))
            baseline_frame_id = int(sample_frame_id)
        if baseline_samples:
            zone_cfg.baseline_points = int(np.median(np.asarray(baseline_samples, dtype=np.int32)))
            last_frame_id = baseline_frame_id
        print(
            "[wander] stop box self-hit baseline "
            f"(samples={baseline_samples}, baseline={zone_cfg.baseline_points}, "
            f"trigger_at={zone_cfg.blocked_trigger_count()} points)"
        )
        if zone_cfg.baseline_points > 0:
            print(
                "[wander] note: the lidar permanently sees part of the robot inside the stop box; "
                "consider recalibrating the stop box geometry"
            )

        frame_id, captured_frame, _, initial_snapshot = _capture_snapshot(
            feed=feed,
            output_dir=snapshot_dir,
            request_index=1,
            after_frame_id=last_frame_id,
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            invert_lateral_axis=bool(args.invert_lateral_axis),
            max_distance_m=float(args.max_distance_m),
            min_range_m=float(args.min_range_m),
            min_confidence=int(args.min_confidence),
            fresh_frame_timeout_s=float(args.fresh_frame_timeout_s),
            fresh_frame_advances=int(args.fresh_frame_advances),
            capture_config_extra={"motion_hint": motion_hints[-1].kind},
        )
        last_frame_id = int(frame_id)
        last_captured_frame = captured_frame

        stitch_state = _append_stitch(
            stitch_dir=stitch_dir,
            snapshot_dir=snapshot_dir,
            stitch_state={"snapshots": [], "poses": [], "transformed_sets": [], "solve_log": []},
            motion_hints=motion_hints,
            new_snapshot=initial_snapshot,
            resolution_m=float(args.stitch_resolution_m),
        )
        _log_rerun_state(
            rr,
            capture_index=1,
            transformed_sets=stitch_state["transformed_sets"],
            poses=stitch_state["poses"],
            solve_log=stitch_state["solve_log"],
            cone_half_width_deg=float(args.valid_angle_half_width_deg),
        )
        initial_pose = stitch_state["poses"][-1]
        initial_heading_bin = _heading_bin_index(
            float(initial_pose.theta_deg),
            bin_count=rotation_coverage_bin_count,
        )
        rotation_coverage_bins_seen.add(initial_heading_bin)
        print(
            "[wander] rotation coverage update "
            f"(capture=1, heading={float(initial_pose.theta_deg):.1f}deg, "
            f"bins={sorted(rotation_coverage_bins_seen)}/{rotation_coverage_bin_count})"
        )
        print(f"[wander] initial stitched map ready: {stitch_state['html_path']}")

        capture_index = 2
        while True:
            if int(args.max_captures) > 0 and capture_index > int(args.max_captures):
                break
            live_frame_id, live_frame = feed.latest()
            if live_frame is None:
                raise RuntimeError("LiDAR feed disappeared during wander loop.")
            current_solved_pose = stitch_state["poses"][-1]
            live_points_xy = _scan_to_local_points(
                points=live_frame.points,
                forward_angle_deg=float(args.forward_angle_deg),
                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                invert_lateral_axis=bool(args.invert_lateral_axis),
                max_distance_m=float(args.max_distance_m),
                min_confidence=int(args.min_confidence),
                min_range_m=float(args.min_range_m),
            )
            live_pose_result = _estimate_pose_against_stitched_map(
                points_xy=live_points_xy,
                transformed_sets=stitch_state["transformed_sets"],
                initial_pose=current_solved_pose,
                resolution_m=float(args.stitch_resolution_m),
                search_xy_m=max(float(args.drive_search_xy_m), 0.55),
                theta_window_deg=max(float(args.drive_theta_window_deg), 36.0),
                max_translation_from_initial_m=max(0.45, float(args.drive_search_xy_m) + 0.20),
                prior_translation_weight=0.20,
                prior_theta_weight=0.08,
            )
            live_pose_accepted = False
            if live_pose_result is not None:
                candidate_live_pose, live_pose_meta = live_pose_result
                if _accept_relocalized_pose(
                    label=f"live_frame_{live_frame_id}",
                    candidate_pose=candidate_live_pose,
                    score_meta=live_pose_meta,
                    expected_pose=current_solved_pose,
                    motion_hint=None,
                    max_translation_error_m=max(0.45, float(args.drive_search_xy_m) + 0.20),
                    max_theta_error_deg=max(24.0, float(args.drive_theta_window_deg) * 0.85),
                    min_score=7.0,
                ):
                    current_live_pose = candidate_live_pose
                    live_pose_accepted = True
                else:
                    current_live_pose = current_solved_pose
                    live_pose_meta = {
                        **live_pose_meta,
                        "source": "rejected_live_fallback",
                    }
                    print(
                        "[wander] live relocalization fallback "
                        f"(frame_id={live_frame_id}, using last stitched pose=({float(current_live_pose.x):.3f}, "
                        f"{float(current_live_pose.y):.3f}, {float(current_live_pose.theta_deg):.1f}deg))"
                    )
            if live_pose_result is None:
                current_live_pose = current_solved_pose
                print(
                    "[wander] live relocalization unavailable; using last stitched pose "
                    f"(frame_id={live_frame_id}, pose=({float(current_live_pose.x):.3f}, "
                    f"{float(current_live_pose.y):.3f}, {float(current_live_pose.theta_deg):.1f}deg))"
                )
            _log_live_pose_state(
                rr,
                capture_index=max(1, capture_index - 1),
                pose=current_live_pose,
                points_xy=live_points_xy,
                cone_half_width_deg=float(args.valid_angle_half_width_deg),
            )

            blocked_points = _blocked_points_for_frame(live_frame, zone_cfg)
            blocked = blocked_points >= zone_cfg.blocked_trigger_count()
            drive_hint_m = 0.0
            should_turn = False
            turn_reason = "none"
            chosen_direction_sign = float(direction_sign)
            chosen_turn_deg = float(args.turn_deg)
            motion_hint: MotionHint | None = None
            settle_s = float(args.move_settle_s)
            turn_capture_theta_window_deg = 80.0
            bootstrap_scan_active = (
                wander_mode == "smart"
                and not rotation_coverage_complete
                and int(capture_index) <= max(1, int(args.bootstrap_turn_captures)) + 4
            )
            forced_drive_due_to_turn_streak = (
                wander_mode == "smart"
                and not blocked
                and consecutive_turn_captures >= max(2, min(4, int(args.bootstrap_turn_captures)))
            )
            if forced_drive_due_to_turn_streak:
                force_drive_after_turn = True
                print(
                    "[wander] forcing forward exploration after repeated turn captures "
                    f"(turn_streak={consecutive_turn_captures}, coverage_complete={rotation_coverage_complete})"
                )

            if pending_turn_reason is not None:
                turn_reason = str(pending_turn_reason)
                chosen_direction_sign = float(pending_turn_direction_sign)
                chosen_turn_deg = float(pending_turn_deg)
                print(
                    "[wander] executing queued turn "
                    f"(reason={turn_reason}, target={chosen_turn_deg:.1f}deg "
                    f"{'ccw' if chosen_direction_sign >= 0.0 else 'cw'})"
                )
                should_turn = True
                pending_turn_reason = None
                pending_turn_direction_sign = float(direction_sign)
                pending_turn_deg = float(args.turn_deg)

            live_frontier_choice = _select_frontier_choice(
                live_frame,
                forward_angle_deg=float(args.forward_angle_deg),
                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                max_distance_m=float(args.max_distance_m),
                min_range_m=float(args.min_range_m),
                min_confidence=int(args.min_confidence),
                frontier_min_distance_m=float(args.frontier_min_distance_m),
                frontier_bin_deg=float(args.frontier_bin_deg),
            )
            if live_frontier_choice is not None:
                print(
                    "[wander] live frontier candidate "
                    f"(frame_id={live_frame_id}, delta={live_frontier_choice.delta_deg:.1f}deg, "
                    f"distance={live_frontier_choice.mean_distance_m:.2f}m, "
                    f"width={live_frontier_choice.width_deg:.1f}deg, score={live_frontier_choice.score:.2f})"
                )
            else:
                print(f"[wander] no live frontier candidate found (frame_id={live_frame_id})")

            map_frontier_choice: FrontierChoice | None = None
            if len(stitch_state["snapshots"]) >= 2:
                map_frontier_choice = _select_map_frontier_choice(
                    transformed_sets=stitch_state["transformed_sets"],
                    current_pose=current_live_pose,
                    max_distance_m=float(args.max_distance_m),
                    frontier_min_distance_m=float(args.frontier_min_distance_m),
                    frontier_bin_deg=float(args.frontier_bin_deg),
                    resolution_m=float(args.stitch_resolution_m),
                )
            if map_frontier_choice is not None:
                print(
                    "[wander] map frontier candidate "
                    f"(delta={map_frontier_choice.delta_deg:.1f}deg, "
                    f"distance={map_frontier_choice.mean_distance_m:.2f}m, "
                    f"width={map_frontier_choice.width_deg:.1f}deg, score={map_frontier_choice.score:.2f}, "
                    f"pose=({float(current_live_pose.x):.3f}, {float(current_live_pose.y):.3f}, {float(current_live_pose.theta_deg):.1f}deg))"
                )
            else:
                print(
                    "[wander] no stitched-map frontier candidate "
                    f"(captures={len(stitch_state['snapshots'])}, pose=({float(current_live_pose.x):.3f}, "
                    f"{float(current_live_pose.y):.3f}, {float(current_live_pose.theta_deg):.1f}deg))"
                )

            target_frontier_choice: FrontierChoice | None = None
            if active_explore_target is not None:
                current_target_distance_m = _target_distance_m(active_explore_target, current_live_pose)
                if current_target_distance_m <= float(args.frontier_goal_reached_m):
                    print(
                        "[wander] stitched-map exploration target reached "
                        f"(distance={current_target_distance_m:.2f}m, source={active_explore_target.source}, "
                        f"seed_capture={active_explore_target.seeded_capture_index})"
                    )
                    active_explore_target = None
                    active_explore_target_stall_count = 0
                    active_explore_target_last_distance_m = None
                    active_explore_target_blocked_count = 0
                    force_live_frontier_cycles = 0
                else:
                    target_frontier_choice = _target_choice_from_world_point(
                        target_x_m=float(active_explore_target.world_x_m),
                        target_y_m=float(active_explore_target.world_y_m),
                        current_pose=current_live_pose,
                        source="target",
                    )
                    if target_frontier_choice is not None:
                        print(
                            "[wander] active stitched-map target "
                            f"(delta={target_frontier_choice.delta_deg:.1f}deg, "
                            f"distance={target_frontier_choice.mean_distance_m:.2f}m, "
                            f"world=({float(active_explore_target.world_x_m):.3f}, "
                            f"{float(active_explore_target.world_y_m):.3f}))"
                        )

            if (
                active_explore_target is None
                and wander_mode == "smart"
                and rotation_coverage_complete
                and map_frontier_choice is not None
            ):
                active_explore_target = _seed_explore_target_from_frontier(
                    frontier_choice=map_frontier_choice,
                    current_pose=current_live_pose,
                    capture_index=capture_index,
                    step_distance_m=float(args.frontier_goal_step_m),
                )
                active_explore_target_stall_count = 0
                active_explore_target_last_distance_m = None
                active_explore_target_blocked_count = 0
                force_live_frontier_cycles = 0
                target_frontier_choice = _target_choice_from_world_point(
                    target_x_m=float(active_explore_target.world_x_m),
                    target_y_m=float(active_explore_target.world_y_m),
                    current_pose=current_live_pose,
                    source="target",
                )
                print(
                    "[wander] seeded stitched-map exploration target "
                    f"(source={map_frontier_choice.source}, delta={map_frontier_choice.delta_deg:.1f}deg, "
                    f"distance={map_frontier_choice.mean_distance_m:.2f}m, target_world=("
                    f"{float(active_explore_target.world_x_m):.3f}, {float(active_explore_target.world_y_m):.3f}))"
                )

            escape_frontier_active = (
                wander_mode == "smart"
                and force_live_frontier_cycles > 0
                and live_frontier_choice is not None
            )
            if escape_frontier_active:
                planning_frontier_choice = live_frontier_choice or map_frontier_choice or target_frontier_choice
                planning_frontier_source = "live_escape"
                print(
                    "[wander] escape frontier override active "
                    f"(cycles_left={force_live_frontier_cycles}, "
                    f"live_delta={live_frontier_choice.delta_deg:.1f}deg, "
                    f"live_distance={live_frontier_choice.mean_distance_m:.2f}m)"
                )
            else:
                planning_frontier_choice = target_frontier_choice or map_frontier_choice or live_frontier_choice
                planning_frontier_source = (
                    planning_frontier_choice.source if planning_frontier_choice is not None else "none"
                )
            if planning_frontier_choice is not None:
                print(
                    "[wander] planning frontier "
                    f"(source={planning_frontier_source}, delta={planning_frontier_choice.delta_deg:.1f}deg, "
                    f"distance={planning_frontier_choice.mean_distance_m:.2f}m, score={planning_frontier_choice.score:.2f})"
                )

            if should_turn:
                pass
            elif bootstrap_scan_active:
                # Aim each bootstrap turn at the CLOSEST heading bin the map has
                # not covered yet, instead of blindly stepping 90deg the same
                # way — stiction makes actual turn sizes erratic, so blind steps
                # revisit the same headings while leaving one bin unseen.
                unseen_bins = [
                    b for b in range(rotation_coverage_bin_count) if b not in rotation_coverage_bins_seen
                ]
                if unseen_bins:
                    bin_width_deg = 360.0 / float(rotation_coverage_bin_count)
                    current_heading_deg = float(current_live_pose.theta_deg)
                    bin_deltas = [
                        (_normalize_angle_deg((b + 0.5) * bin_width_deg - current_heading_deg), b)
                        for b in unseen_bins
                    ]
                    target_delta_deg, target_bin = min(bin_deltas, key=lambda item: abs(item[0]))
                    chosen_direction_sign = 1.0 if target_delta_deg >= 0.0 else -1.0
                    chosen_turn_deg = max(25.0, min(55.0, abs(float(target_delta_deg))))
                    print(
                        "[wander] bootstrap scan capture targeting unseen heading bin "
                        f"(capture={capture_index}, bin={target_bin}, heading={current_heading_deg:.1f}deg, "
                        f"turn={chosen_turn_deg:.1f}deg {'ccw' if chosen_direction_sign >= 0.0 else 'cw'}, "
                        f"unseen={sorted(unseen_bins)})"
                    )
                else:
                    print(
                        "[wander] bootstrap scan capture "
                        f"(capture={capture_index}, target={float(args.turn_deg):.1f}deg {args.turn_direction})"
                    )
                should_turn = True
                turn_reason = "bootstrap_scan"
            elif wander_mode == "turn_only":
                print(
                    "[wander] turn-only mode active; rotating for capture "
                    f"(frame_id={live_frame_id}, blocked_points={blocked_points})"
                )
                should_turn = True
                turn_reason = "turn_only"
            elif blocked:
                print(
                    "[wander] stop box occupied; rotating in place "
                    f"(frame_id={live_frame_id}, blocked_points={blocked_points})"
                )
                should_turn = True
                blocked_frontier = live_frontier_choice or planning_frontier_choice
                if blocked_frontier is not None:
                    chosen_direction_sign = 1.0 if blocked_frontier.delta_deg >= 0.0 else -1.0
                    chosen_turn_deg = max(12.0, min(float(args.turn_deg), abs(float(blocked_frontier.delta_deg))))
                    turn_reason = "blocked_frontier"
                else:
                    turn_reason = "blocked"
            elif (
                wander_mode == "smart"
                and rotation_coverage_complete
                and not force_drive_after_turn
                and (
                    active_explore_target is None
                    or int(active_explore_target.coarse_turns_used) <= 0
                )
                and planning_frontier_choice is not None
                and abs(float(planning_frontier_choice.delta_deg)) >= 115.0
            ):
                chosen_direction_sign = 1.0 if planning_frontier_choice.delta_deg >= 0.0 else -1.0
                chosen_turn_deg = max(22.0, min(55.0, abs(float(planning_frontier_choice.delta_deg)) - 32.0))
                should_turn = True
                turn_reason = "frontier_seek"
                print(
                    "[wander] rotation coverage is complete; performing one coarse frontier seek turn "
                    f"(frame_id={live_frame_id}, source={planning_frontier_source}, "
                    f"delta={planning_frontier_choice.delta_deg:.1f}deg, "
                    f"distance={planning_frontier_choice.mean_distance_m:.2f}m, "
                    f"width={planning_frontier_choice.width_deg:.1f}deg)"
                )
            elif (
                wander_mode == "smart"
                and not force_drive_after_turn
                and not rotation_coverage_complete
                and planning_frontier_choice is not None
                and abs(float(planning_frontier_choice.delta_deg)) >= float(args.frontier_align_threshold_deg)
            ):
                chosen_direction_sign = 1.0 if planning_frontier_choice.delta_deg >= 0.0 else -1.0
                chosen_turn_deg = max(15.0, min(float(args.turn_deg), abs(float(planning_frontier_choice.delta_deg))))
                should_turn = True
                turn_reason = "frontier_align"
                print(
                    "[wander] frontier is far off heading; aligning before drive "
                    f"(frame_id={live_frame_id}, source={planning_frontier_source}, "
                    f"delta={planning_frontier_choice.delta_deg:.1f}deg, "
                    f"distance={planning_frontier_choice.mean_distance_m:.2f}m, "
                    f"width={planning_frontier_choice.width_deg:.1f}deg)"
                )
            else:
                print(
                    "[wander] forward path is clear; driving sequence "
                    f"(frame_id={live_frame_id}, blocked_points={blocked_points})"
                )
                steer_theta_vel = _compute_drive_steer_theta_vel(
                    planning_frontier_choice,
                    steer_gain=float(args.frontier_drive_steer_gain),
                    steer_max=float(args.max_drive_steer_theta_vel),
                    steer_deadband_deg=float(args.drive_steer_deadband_deg),
                )
                planned_bursts = max(1, int(args.drive_bursts_per_capture))
                if planning_frontier_choice is not None:
                    if planning_frontier_choice.mean_distance_m >= 2.4:
                        planned_bursts += 1
                    if planning_frontier_choice.mean_distance_m >= 3.4 and planning_frontier_choice.width_deg >= 24.0:
                        planned_bursts += 1
                planned_bursts = min(planned_bursts, max(1, int(args.max_drive_bursts_per_capture)))
                print(
                    "[wander] drive plan "
                    f"(planned_bursts={planned_bursts}, frontier_delta="
                    f"{None if planning_frontier_choice is None else round(float(planning_frontier_choice.delta_deg), 1)}, "
                    f"frontier_distance={None if planning_frontier_choice is None else round(float(planning_frontier_choice.mean_distance_m), 2)}, "
                    f"frontier_source={planning_frontier_source}, "
                    f"drive_mode={'frontier_follow' if planning_frontier_choice is not None else 'straight_burst'}, "
                    f"steer_theta_vel={steer_theta_vel:.3f})"
                )
                drive_start_pose = current_live_pose
                drive_tracked_pose, drive_track_meta = _drive_with_tracking(
                    robot=robot,
                    feed=feed,
                    zone_cfg=zone_cfg,
                    transformed_sets=stitch_state["transformed_sets"],
                    start_pose=drive_start_pose,
                    resolution_m=float(args.stitch_resolution_m),
                    forward_speed=float(args.move_speed),
                    min_effective_move_speed=float(args.min_effective_move_speed),
                    burst_s=float(args.move_burst_s),
                    burst_count=int(planned_bursts),
                    inter_burst_pause_s=float(args.inter_burst_pause_s),
                    steer_theta_vel=float(steer_theta_vel),
                    forward_angle_deg=float(args.forward_angle_deg),
                    valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                    invert_lateral_axis=bool(args.invert_lateral_axis),
                    max_distance_m=float(args.max_distance_m),
                    min_range_m=float(args.min_range_m),
                    min_confidence=int(args.min_confidence),
                )
                drive_stopped_by_block = bool(drive_track_meta["stopped_by_block"]) or bool(
                    drive_track_meta["lock_lost"]
                )
                # Motion hint straight from the tracked poses — no commanded-
                # speed or wheel-feedback guessing.
                drive_ddx_world = float(drive_tracked_pose.x) - float(drive_start_pose.x)
                drive_ddy_world = float(drive_tracked_pose.y) - float(drive_start_pose.y)
                drive_theta0_rad = math.radians(float(drive_start_pose.theta_deg))
                drive_cos = math.cos(drive_theta0_rad)
                drive_sin = math.sin(drive_theta0_rad)
                drive_hint_dx_local_m = drive_cos * drive_ddx_world + drive_sin * drive_ddy_world
                drive_hint_dy_local_m = -drive_sin * drive_ddx_world + drive_cos * drive_ddy_world
                drive_hint_dtheta_deg = _normalize_angle_deg(
                    float(drive_tracked_pose.theta_deg) - float(drive_start_pose.theta_deg)
                )
                if drive_track_meta["lock_lost"] or int(drive_track_meta["missed_updates"]) > 0:
                    drive_search_xy_m = 0.80
                    drive_theta_window_deg = 30.0
                else:
                    drive_search_xy_m = 0.40
                    drive_theta_window_deg = 16.0
                print(
                    "[wander] forward sequence complete "
                    f"(elapsed={float(drive_track_meta['elapsed_s']):.2f}s "
                    f"bursts={int(drive_track_meta['bursts_completed'])} "
                    f"stopped_by_block={bool(drive_track_meta['stopped_by_block'])} "
                    f"lock_lost={bool(drive_track_meta['lock_lost'])} "
                    f"blocked_points={int(drive_track_meta['blocked_points'])} "
                    f"tracked_move=({drive_hint_dx_local_m:.3f}m, {drive_hint_dy_local_m:.3f}m, "
                    f"{drive_hint_dtheta_deg:.1f}deg))"
                )
                force_drive_after_turn = False
                drive_checkpoints_since_scan += 1
                if wander_mode == "scan_turn_every_capture":
                    pending_turn_reason = "post_drive_scan"
                    pending_turn_direction_sign = float(direction_sign)
                    pending_turn_deg = float(args.turn_deg)
                    should_turn = False
                    turn_reason = "drive_only"
                elif drive_stopped_by_block:
                    should_turn = False
                    if active_explore_target is not None:
                        active_explore_target_blocked_count += 1
                        print(
                            "[wander] stitched-map target blocked during drive "
                            f"(source={active_explore_target.source}, "
                            f"seed_capture={active_explore_target.seeded_capture_index}, "
                            f"blocked_count={active_explore_target_blocked_count})"
                        )
                        force_live_frontier_cycles = max(force_live_frontier_cycles, 2)
                        if active_explore_target_blocked_count >= 3:
                            print(
                                "[wander] abandoning stitched-map target after repeated blocked drives "
                                f"(source={active_explore_target.source}, "
                                f"seed_capture={active_explore_target.seeded_capture_index})"
                            )
                            active_explore_target = None
                            active_explore_target_stall_count = 0
                            active_explore_target_last_distance_m = None
                            active_explore_target_blocked_count = 0
                    recovery_frontier = live_frontier_choice or planning_frontier_choice
                    if recovery_frontier is not None:
                        pending_turn_direction_sign = 1.0 if recovery_frontier.delta_deg >= 0.0 else -1.0
                        pending_turn_deg = max(18.0, min(float(args.turn_deg), abs(float(recovery_frontier.delta_deg))))
                        pending_turn_reason = "drive_blocked_frontier"
                    else:
                        pending_turn_direction_sign = float(direction_sign)
                        pending_turn_deg = float(args.turn_deg)
                        pending_turn_reason = "drive_blocked"
                    turn_reason = "drive_only"
                    drive_checkpoints_since_scan = 0
                elif (
                    wander_mode == "smart"
                    and not rotation_coverage_complete
                    and int(args.scan_turn_interval) > 0
                    and drive_checkpoints_since_scan >= int(args.scan_turn_interval)
                ):
                    should_turn = False
                    if planning_frontier_choice is not None and abs(planning_frontier_choice.delta_deg) >= float(args.frontier_align_threshold_deg):
                        pending_turn_direction_sign = 1.0 if planning_frontier_choice.delta_deg >= 0.0 else -1.0
                        pending_turn_deg = max(15.0, min(float(args.turn_deg), abs(float(planning_frontier_choice.delta_deg))))
                        pending_turn_reason = "periodic_frontier_scan"
                    else:
                        pending_turn_direction_sign = float(direction_sign)
                        pending_turn_deg = float(args.turn_deg)
                        pending_turn_reason = "periodic_scan"
                    turn_reason = "drive_only"
                    drive_checkpoints_since_scan = 0
                else:
                    should_turn = False
                    turn_reason = "drive_only"
                    if force_live_frontier_cycles > 0:
                        force_live_frontier_cycles -= 1
                motion_hint = MotionHint(
                    kind="drive",
                    expected_dx_local_m=float(drive_hint_dx_local_m),
                    expected_dy_local_m=float(drive_hint_dy_local_m),
                    expected_dtheta_deg=float(drive_hint_dtheta_deg),
                    search_xy_m=float(drive_search_xy_m),
                    search_theta_window_deg=float(drive_theta_window_deg),
                    label=f"drive_capture_{capture_index:02d}",
                )
                print(
                    "[wander] drive capture hint "
                    f"(pose_source=lidar_tracked, search_xy_m={drive_search_xy_m:.3f}, "
                    f"expected_dx_local_m={drive_hint_dx_local_m:.3f}, "
                    f"expected_dy_local_m={drive_hint_dy_local_m:.3f}, "
                    f"expected_dtheta_deg={drive_hint_dtheta_deg:.1f})"
                )
                settle_s = float(args.move_settle_s)

            if should_turn:
                if float(chosen_turn_deg) > 55.0:
                    # A capture must land before the view rotates into mostly
                    # unmapped territory: with a ~180deg FOV, 55deg per capture
                    # keeps >=125deg of the previous view in frame, which keeps
                    # solve scores strong. Larger goals just take two captures.
                    print(
                        "[wander] capping turn at 55deg per capture to keep map overlap strong "
                        f"(requested={float(chosen_turn_deg):.1f}deg)"
                    )
                    chosen_turn_deg = 55.0
                print(
                    "[wander] rotating before stitched capture "
                    f"(reason={turn_reason}, target={float(chosen_turn_deg):.1f}deg "
                    f"{'ccw' if chosen_direction_sign >= 0.0 else 'cw'})"
                )
                # Anchor tracking at the dead-reckoned pose INCLUDING any
                # pending (discarded-capture) motion — after a lock-lost
                # discard the robot is physically far from the last stitched
                # pose, and anchoring there makes every retry solve miss.
                # (If live relocalization already re-anchored this cycle, the
                # pending motion is baked into current_live_pose and will be
                # dropped at composition time — don't double-count it here.)
                turn_start_pose = (
                    _advance_pose(current_live_pose, pending_motion_hint)
                    if (pending_motion_hint is not None and not live_pose_accepted)
                    else current_live_pose
                )
                turn_meta = _turn_with_arc_tracking(
                    robot=robot,
                    feed=feed,
                    transformed_sets=stitch_state["transformed_sets"],
                    start_pose=turn_start_pose,
                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                    resolution_m=float(args.stitch_resolution_m),
                    target_turn_deg=float(chosen_turn_deg),
                    direction_sign=float(chosen_direction_sign),
                    turn_speed=float(args.turn_speed),
                    turn_burst_s=float(args.turn_burst_s),
                    turn_settle_s=float(args.turn_settle_s),
                    forward_angle_deg=float(args.forward_angle_deg),
                    valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                    invert_lateral_axis=bool(args.invert_lateral_axis),
                    max_distance_m=float(args.max_distance_m),
                    min_range_m=float(args.min_range_m),
                    min_confidence=int(args.min_confidence),
                    stop_tolerance_deg=min(8.0, float(args.stop_tolerance_deg)),
                    max_bursts=int(args.max_turn_bursts),
                )
                print(
                    "[wander] turn complete "
                    f"(tracked={turn_meta['turned_deg']:+.1f}deg, final_theta={turn_meta['final_theta_deg']:.1f}deg, "
                    f"missed_updates={int(turn_meta['missed_updates'])}, completed={int(turn_meta['completed'])})"
                )
                measured_signed_turn_deg = float(turn_meta["turned_deg"])
                # The capture solve only needs to confirm/refine the tracked
                # heading; widen its window a bit for each burst the tracker
                # could not update on.
                turn_capture_theta_window_deg = min(
                    80.0, 25.0 + 8.0 * float(turn_meta["missed_updates"])
                )
                print(
                    "[wander] turn motion hint "
                    f"(requested_dtheta_deg={float(chosen_turn_deg) * float(chosen_direction_sign):.1f}, "
                    f"measured_dtheta_deg={measured_signed_turn_deg:.1f})"
                )
                if turn_reason in {
                    "blocked",
                    "blocked_frontier",
                    "drive_blocked",
                    "drive_blocked_frontier",
                    "bootstrap_scan",
                    "turn_only",
                    "frontier_seek",
                    "frontier_align",
                    "periodic_frontier_scan",
                    "periodic_scan",
                }:
                    drive_checkpoints_since_scan = 0
                if turn_reason in {"frontier_align", "periodic_frontier_scan", "frontier_seek"}:
                    force_drive_after_turn = True
                    if active_explore_target is not None and turn_reason == "frontier_seek":
                        active_explore_target.coarse_turns_used += 1
                lever_dx_local_m, lever_dy_local_m = _turn_lever_arm_local_delta(
                    float(measured_signed_turn_deg),
                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                )
                motion_hint = MotionHint(
                    kind="turn",
                    expected_dx_local_m=float(lever_dx_local_m),
                    expected_dy_local_m=float(lever_dy_local_m),
                    expected_dtheta_deg=float(measured_signed_turn_deg),
                    search_xy_m=float(args.turn_search_xy_m),
                    search_theta_window_deg=float(args.turn_theta_window_deg),
                    label=f"turn_capture_{capture_index:02d}",
                )
                print(
                    "[wander] turn lever-arm hint "
                    f"(expected_dx_local={lever_dx_local_m:.3f}m, expected_dy_local={lever_dy_local_m:.3f}m, "
                    f"offset={float(args.lidar_offset_forward_m):.3f}m)"
                )
                settle_s = float(args.capture_settle_s)

            print(f"[wander] settling for {settle_s:.2f}s before capture")
            time.sleep(settle_s)

            frame_id, captured_frame, local_points_xy, captured_snapshot = _capture_snapshot(
                feed=feed,
                output_dir=snapshot_dir,
                request_index=capture_index,
                after_frame_id=last_frame_id,
                forward_angle_deg=float(args.forward_angle_deg),
                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                invert_lateral_axis=bool(args.invert_lateral_axis),
                max_distance_m=float(args.max_distance_m),
                min_range_m=float(args.min_range_m),
                min_confidence=int(args.min_confidence),
                fresh_frame_timeout_s=float(args.fresh_frame_timeout_s),
                fresh_frame_advances=int(args.fresh_frame_advances),
                capture_config_extra={
                    "motion_hint": motion_hint.kind,
                    "motion_hint_label": motion_hint.label,
                    "expected_dx_local_m": motion_hint.expected_dx_local_m,
                    "expected_dtheta_deg": motion_hint.expected_dtheta_deg,
                },
            )
            if len(local_points_xy) == 0:
                print(f"[wander] warning: snapshot {capture_index} contains zero local points after filtering")

            if pending_motion_hint is not None:
                if live_pose_accepted:
                    print(
                        "[wander] dropping pending discarded-capture motion; "
                        "live relocalization already re-anchored the pose"
                    )
                else:
                    motion_hint = _compose_motion_hints(pending_motion_hint, motion_hint)
                    print(
                        "[wander] composed pending motion from discarded capture into current hint "
                        f"(expected_dx={motion_hint.expected_dx_local_m:.3f}m, "
                        f"expected_dy={motion_hint.expected_dy_local_m:.3f}m, "
                        f"expected_dtheta={motion_hint.expected_dtheta_deg:.1f}deg)"
                    )
                pending_motion_hint = None

            motion_hints.append(motion_hint)
            rebuild_started = time.monotonic()
            print("[wander] appending stitched capture " f"(capture={capture_index}, snapshots={len(motion_hints)})")
            capture_expected_pose = _advance_pose(current_live_pose, motion_hint)
            capture_live_pose = None
            capture_live_meta = None
            append_solver_pose = None
            append_solver_meta = None
            if motion_hint.kind == "turn":
                # In-place turn: the robot center is pinned, so solve on the
                # lever-arm arc. The measured turn angle only centers a wide
                # search window — it is too unreliable to act as a hard prior.
                arc_theta_half_window_deg = min(
                    110.0,
                    float(turn_capture_theta_window_deg)
                    + (30.0 if "+" in str(motion_hint.label or "") else 0.0),
                )
                arc_result = _solve_turn_arc_pose(
                    snapshot_points_xy=captured_snapshot.points_xy,
                    transformed_sets=stitch_state["transformed_sets"],
                    previous_pose=current_live_pose,
                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                    resolution_m=float(args.stitch_resolution_m),
                    expected_theta_deg=float(capture_expected_pose.theta_deg),
                    theta_half_window_deg=float(arc_theta_half_window_deg),
                )
                if arc_result is not None:
                    arc_solved_pose, arc_meta = arc_result
                    append_solver_pose = arc_solved_pose
                    append_solver_meta = arc_meta
                    arc_score = float(arc_meta.get("score") or -1e9)
                    solved_turn_deg = _normalize_angle_deg(
                        float(arc_solved_pose.theta_deg) - float(current_live_pose.theta_deg)
                    )
                    print(
                        "[wander] turn arc solve "
                        f"(capture={capture_index}, pose=({arc_solved_pose.x:.3f}, {arc_solved_pose.y:.3f}, "
                        f"{arc_solved_pose.theta_deg:.1f}deg), turned={solved_turn_deg:.1f}deg, "
                        f"hinted={float(motion_hint.expected_dtheta_deg):.1f}deg, score={arc_score:.3f}, "
                        f"window=+-{arc_theta_half_window_deg:.0f}deg)"
                    )
                    hint_discrepancy_deg = abs(
                        _normalize_angle_deg(solved_turn_deg - float(motion_hint.expected_dtheta_deg))
                    )
                    if hint_discrepancy_deg > 75.0:
                        print(
                            "[wander] warning: arc solve disagrees strongly with the measured turn "
                            f"(discrepancy={hint_discrepancy_deg:.1f}deg); possible room-symmetry mode"
                        )
                    arc_accept_gate = float(args.min_append_score)
                    if hint_discrepancy_deg <= 8.0:
                        # Geometry and continuous tracking independently agree;
                        # that mutual confirmation outweighs a modest absolute
                        # score (which dips while facing sparsely mapped areas).
                        # Discarding such captures caused blind-motion cascades.
                        arc_accept_gate = min(arc_accept_gate, 3.5)
                    if arc_score >= arc_accept_gate:
                        capture_live_pose = arc_solved_pose
                        capture_live_meta = arc_meta
                    else:
                        print(
                            "[wander] turn arc solve below gate "
                            f"(capture={capture_index}, score={arc_score:.3f}, min={float(args.min_append_score):.2f})"
                        )
                else:
                    print(f"[wander] turn arc solve unavailable (capture={capture_index})")
            else:
                capture_pose_result = _estimate_pose_against_stitched_map(
                    points_xy=captured_snapshot.points_xy,
                    transformed_sets=stitch_state["transformed_sets"],
                    initial_pose=capture_expected_pose,
                    resolution_m=float(args.stitch_resolution_m),
                    search_xy_m=max(float(motion_hint.search_xy_m), 0.65),
                    theta_window_deg=max(float(motion_hint.search_theta_window_deg), 24.0),
                    max_translation_from_initial_m=max(0.75, float(motion_hint.search_xy_m) + 0.25),
                    prior_translation_weight=0.18,
                    prior_theta_weight=0.06,
                )
                if capture_pose_result is not None:
                    candidate_capture_pose, candidate_capture_meta = capture_pose_result
                    if _accept_relocalized_pose(
                        label=f"capture_{capture_index}",
                        candidate_pose=candidate_capture_pose,
                        score_meta=candidate_capture_meta,
                        expected_pose=capture_expected_pose,
                        motion_hint=motion_hint,
                        max_translation_error_m=max(0.75, float(motion_hint.search_xy_m) + 0.25),
                        max_theta_error_deg=max(18.0, float(motion_hint.search_theta_window_deg) * 1.10),
                        min_score=7.5,
                    ):
                        capture_live_pose = candidate_capture_pose
                        capture_live_meta = candidate_capture_meta
                    else:
                        print(
                            "[wander] capture relocalization fallback "
                            f"(capture={capture_index}, using strict append solver around prior stitched map)"
                        )
                if capture_pose_result is None:
                    print(f"[wander] capture relocalization unavailable (capture={capture_index})")

            if capture_live_pose is None and motion_hint.kind != "turn":
                # Run the strict append solver here (instead of inside _append_stitch)
                # so its score can be gated before the capture is committed to the map.
                is_turn_like_hint = motion_hint.kind in {"turn", "bootstrap_turn"}
                append_global_points_xy = np.concatenate(
                    [points for points in stitch_state["transformed_sets"] if len(points)], axis=0
                )
                append_prior_pose, append_prior_tw, append_prior_thw = _prior_weights_for_hint(
                    capture_expected_pose, motion_hint
                )
                append_solver_pose, append_solver_meta = _search_pose(
                    snapshot_points_xy=captured_snapshot.points_xy,
                    global_points_xy=append_global_points_xy,
                    initial_pose=capture_expected_pose,
                    resolution_m=float(args.stitch_resolution_m),
                    search_xy_m=float(motion_hint.search_xy_m),
                    coarse_angle_step_deg=4.0,
                    fine_angle_step_deg=0.5,
                    theta_window_deg=float(motion_hint.search_theta_window_deg),
                    whole_map_theta_center_deg=float(capture_expected_pose.theta_deg),
                    whole_map_theta_window_deg=(
                        float(max(motion_hint.search_theta_window_deg, 36.0))
                        if is_turn_like_hint
                        else float(max(motion_hint.search_theta_window_deg, 45.0))
                    ),
                    # Motion is lidar-tracked burst-by-burst now, so even the
                    # fallback search stays bounded — an unbounded whole-map
                    # drive search is how phantom room-sized jumps got in.
                    max_translation_from_initial_m=(
                        float(max(motion_hint.search_xy_m + 0.20, 0.55))
                        if is_turn_like_hint
                        else float(max(motion_hint.search_xy_m + 0.35, 0.90))
                    ),
                    prior_pose=append_prior_pose,
                    prior_translation_weight=float(append_prior_tw),
                    prior_theta_weight=float(append_prior_thw),
                )
                append_solver_meta = {
                    **append_solver_meta,
                    "source": f"append_{append_solver_meta.get('source', 'unknown')}",
                }
                append_solver_score = float(append_solver_meta.get("score") or -1e9)
                append_translation_err_m, append_theta_err_deg = _pose_delta_metrics(
                    capture_expected_pose, append_solver_pose
                )
                append_gate = float(args.min_append_score)
                if append_translation_err_m <= 0.30 and append_theta_err_deg <= 10.0:
                    # Solver landed where burst-by-burst tracking said we are;
                    # mutual confirmation earns a relaxed absolute gate.
                    append_gate = min(append_gate, 3.5)
                if append_solver_score >= append_gate:
                    capture_live_pose = append_solver_pose
                    capture_live_meta = append_solver_meta
                else:
                    print(
                        "[wander] append solver score below gate "
                        f"(capture={capture_index}, score={append_solver_score:.3f}, "
                        f"min={append_gate:.2f}); attempting wide rescue relocalization"
                    )
                    rescue_result = _estimate_pose_against_stitched_map(
                        points_xy=captured_snapshot.points_xy,
                        transformed_sets=stitch_state["transformed_sets"],
                        initial_pose=capture_expected_pose,
                        resolution_m=float(args.stitch_resolution_m),
                        search_xy_m=1.00,
                        theta_window_deg=80.0,
                        max_translation_from_initial_m=1.40,
                        prior_translation_weight=0.05,
                        prior_theta_weight=0.02,
                    )
                    if rescue_result is not None:
                        rescue_pose, rescue_meta = rescue_result
                        if _accept_relocalized_pose(
                            label=f"capture_{capture_index}_rescue",
                            candidate_pose=rescue_pose,
                            score_meta=rescue_meta,
                            expected_pose=capture_expected_pose,
                            motion_hint=motion_hint,
                            max_translation_error_m=1.40,
                            max_theta_error_deg=85.0,
                            min_score=7.5,
                        ):
                            capture_live_pose = rescue_pose
                            capture_live_meta = {**rescue_meta, "source": f"rescue_{rescue_meta.get('source', 'unknown')}"}

            if capture_live_pose is None:
                consecutive_append_discards += 1
                if consecutive_append_discards <= max(0, int(args.max_consecutive_append_discards)):
                    print(
                        "[wander] discarding capture to protect the map "
                        f"(capture={capture_index}, best_score="
                        f"{float((append_solver_meta or {}).get('score') or -1e9):.3f}, "
                        f"consecutive_discards={consecutive_append_discards}); "
                        "its expected motion will carry into the next capture"
                    )
                    motion_hints.pop()
                    pending_motion_hint = motion_hint
                    last_frame_id = int(frame_id)
                    last_captured_frame = captured_frame
                    continue
                print(
                    "[wander] WARNING: accepting low-confidence pose after "
                    f"{consecutive_append_discards} consecutive discards "
                    f"(capture={capture_index}, score={float((append_solver_meta or {}).get('score') or -1e9):.3f}); "
                    "map quality may degrade"
                )
                capture_live_pose = append_solver_pose
                capture_live_meta = append_solver_meta
            consecutive_append_discards = 0

            stitch_state = _append_stitch(
                stitch_dir=stitch_dir,
                snapshot_dir=snapshot_dir,
                stitch_state=stitch_state,
                motion_hints=motion_hints,
                new_snapshot=captured_snapshot,
                resolution_m=float(args.stitch_resolution_m),
                solved_pose_override=capture_live_pose,
                solve_meta_override=capture_live_meta,
            )
            rebuild_elapsed = time.monotonic() - rebuild_started
            _log_rerun_state(
                rr,
                capture_index=capture_index,
                transformed_sets=stitch_state["transformed_sets"],
                poses=stitch_state["poses"],
                solve_log=stitch_state["solve_log"],
                cone_half_width_deg=float(args.valid_angle_half_width_deg),
            )
            capture_index += 1
            _log_live_pose_state(
                rr,
                capture_index=capture_index,
                pose=stitch_state["poses"][-1],
                points_xy=captured_snapshot.points_xy,
                cone_half_width_deg=float(args.valid_angle_half_width_deg),
            )
            last_frame_id = int(frame_id)
            last_captured_frame = captured_frame

            final_pose = stitch_state["poses"][-1]
            previous_pose = stitch_state["poses"][-2] if len(stitch_state["poses"]) >= 2 else None
            solved_dx = 0.0 if previous_pose is None else float(final_pose.x - previous_pose.x)
            solved_dy = 0.0 if previous_pose is None else float(final_pose.y - previous_pose.y)
            solved_dtheta = 0.0 if previous_pose is None else _normalize_angle_deg(float(final_pose.theta_deg - previous_pose.theta_deg))
            solve_meta = stitch_state["solve_log"][-1] if stitch_state["solve_log"] else {}
            search_timing = solve_meta.get("timing_s") if isinstance(solve_meta, dict) else None
            append_timing = stitch_state.get("timing", {})
            print(
                "[wander] stitched capture complete "
                f"(capture={capture_index}, rebuild={rebuild_elapsed:.2f}s, pose=({final_pose.x:.3f}, {final_pose.y:.3f}, {final_pose.theta_deg:.1f}deg), "
                f"html={stitch_state['html_path']})"
            )
            if previous_pose is not None:
                print(
                    "[wander] solved motion delta "
                    f"(capture={capture_index}, dx={solved_dx:.3f}m, dy={solved_dy:.3f}m, dtheta={solved_dtheta:.1f}deg, "
                    f"source={solve_meta.get('solve_source')}, score={solve_meta.get('score')})"
                )
            if isinstance(search_timing, dict):
                print(
                    "[wander] stitch timing "
                    f"(capture={capture_index}, build={float(search_timing.get('build_occupancy', 0.0)):.2f}s, "
                    f"local={float(search_timing.get('local_search', 0.0)):.2f}s, "
                    f"whole_map={float(search_timing.get('whole_map_search', 0.0)):.2f}s, "
                    f"search_total={float(search_timing.get('total_search', 0.0)):.2f}s, "
                    f"write={float(append_timing.get('write_phase_s', 0.0)):.2f}s)"
                )
            if active_explore_target is not None:
                target_distance_after_capture_m = _target_distance_m(active_explore_target, final_pose)
                previous_target_distance_m = active_explore_target_last_distance_m
                if previous_target_distance_m is not None:
                    target_progress_m = float(previous_target_distance_m) - float(target_distance_after_capture_m)
                    if target_progress_m < 0.08:
                        active_explore_target_stall_count += 1
                        print(
                            "[wander] stitched-map target progress stalled "
                            f"(capture={capture_index}, progress={target_progress_m:.3f}m, "
                            f"distance={target_distance_after_capture_m:.3f}m, "
                            f"stall_count={active_explore_target_stall_count})"
                        )
                    else:
                        active_explore_target_stall_count = 0
                        active_explore_target_blocked_count = 0
                        force_live_frontier_cycles = 0
                        print(
                            "[wander] stitched-map target progress improved "
                            f"(capture={capture_index}, progress={target_progress_m:.3f}m, "
                            f"distance={target_distance_after_capture_m:.3f}m)"
                        )
                active_explore_target_last_distance_m = float(target_distance_after_capture_m)
                if active_explore_target_stall_count >= 2:
                    print(
                        "[wander] dropping stitched-map target after repeated low-progress captures "
                        f"(distance={target_distance_after_capture_m:.3f}m, "
                        f"source={active_explore_target.source}, "
                        f"seed_capture={active_explore_target.seeded_capture_index})"
                    )
                    active_explore_target = None
                    active_explore_target_stall_count = 0
                    active_explore_target_last_distance_m = None
                    active_explore_target_blocked_count = 0
                    force_live_frontier_cycles = max(force_live_frontier_cycles, 2)
            if motion_hint.kind == "turn":
                consecutive_turn_captures += 1
                solved_heading_bin = _heading_bin_index(
                    float(final_pose.theta_deg),
                    bin_count=rotation_coverage_bin_count,
                )
                previous_bin_count = len(rotation_coverage_bins_seen)
                rotation_coverage_bins_seen.add(solved_heading_bin)
                if len(rotation_coverage_bins_seen) != previous_bin_count or capture_index <= 4:
                    print(
                        "[wander] rotation coverage update "
                        f"(capture={capture_index}, heading={float(final_pose.theta_deg):.1f}deg, "
                        f"bins={sorted(rotation_coverage_bins_seen)}/{rotation_coverage_bin_count})"
                    )
                if not rotation_coverage_complete and len(rotation_coverage_bins_seen) >= rotation_coverage_bin_count:
                    rotation_coverage_complete = True
                    force_drive_after_turn = True
                    drive_checkpoints_since_scan = 0
                    print(
                        "[wander] rotational coverage complete; forcing forward exploration "
                        f"(capture={capture_index}, bins={sorted(rotation_coverage_bins_seen)})"
                    )
            else:
                consecutive_turn_captures = 0

        completed_captures = capture_index - 1 if "capture_index" in locals() else 1
        print(f"[wander] complete. captures={completed_captures} viewer={viewer_url or 'local rerun'}")
        print(f"[wander] stitched html: {stitch_state['html_path']}")
        print(f"[wander] stitched report: {stitch_state['report_path']}")
        return 0
    finally:
        try:
            _send_stop(robot)
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        feed.stop()


if __name__ == "__main__":
    raise SystemExit(main())
