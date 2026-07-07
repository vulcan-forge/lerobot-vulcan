from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Direct LiDAR Offline Stitch</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      background: #0f1720;
      color: #e5eef8;
      font-family: Arial, sans-serif;
    }}
    h1 {{
      margin: 0 0 16px 0;
      font-size: 30px;
    }}
    .panel {{
      background: #141f2c;
      border: 1px solid #243346;
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 18px;
    }}
    svg {{
      width: 100%;
      height: auto;
      background: #06090d;
      border-radius: 10px;
      border: 1px solid #203041;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      color: #b8cadb;
      font-size: 13px;
      line-height: 1.45;
      margin: 0;
    }}
  </style>
</head>
<body>
  <h1>Direct LiDAR Offline Stitch</h1>
  <div class="panel">{svg}</div>
  <div class="panel"><pre>{report}</pre></div>
</body>
</html>
"""


COLORS = ["#6ab4ff", "#ffd166", "#7cf29a", "#ff7ce5", "#ff8d6a", "#c299ff"]


@dataclass(slots=True)
class Snapshot:
    name: str
    points_xy: np.ndarray
    metadata: dict[str, object]


@dataclass(slots=True)
class Pose2D:
    x: float
    y: float
    theta_deg: float


@dataclass(slots=True)
class MotionHint:
    kind: str
    expected_dx_local_m: float = 0.0
    expected_dy_local_m: float = 0.0
    expected_dtheta_deg: float = 0.0
    search_xy_m: float = 0.45
    search_theta_window_deg: float = 54.0
    label: str | None = None


def _pose_dict(pose: Pose2D) -> dict[str, float]:
    return {
        "x": round(float(pose.x), 6),
        "y": round(float(pose.y), 6),
        "theta_deg": round(float(pose.theta_deg), 6),
    }


def _rotation_matrix(theta_deg: float) -> np.ndarray:
    theta = math.radians(theta_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    return np.asarray([[c, -s], [s, c]], dtype=np.float32)


def _normalize_angle_deg(angle_deg: float) -> float:
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else float(wrapped)


def _transform_points(points_xy: np.ndarray, pose: Pose2D) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    rot = _rotation_matrix(pose.theta_deg)
    transformed = points_xy @ rot.T
    transformed[:, 0] += pose.x
    transformed[:, 1] += pose.y
    return transformed


def _advance_pose(previous_pose: Pose2D, hint: MotionHint) -> Pose2D:
    local_delta = np.asarray(
        [float(hint.expected_dx_local_m), float(hint.expected_dy_local_m)],
        dtype=np.float32,
    )
    world_delta = _rotation_matrix(previous_pose.theta_deg) @ local_delta
    return Pose2D(
        x=float(previous_pose.x + world_delta[0]),
        y=float(previous_pose.y + world_delta[1]),
        theta_deg=float(previous_pose.theta_deg + float(hint.expected_dtheta_deg)),
    )


def _load_snapshots(snapshot_dir: Path) -> list[Snapshot]:
    snapshots: list[Snapshot] = []
    for json_path in sorted(snapshot_dir.glob("snapshot_*.json")):
        stem = json_path.stem
        if stem.endswith("_raw"):
            continue
        npy_path = snapshot_dir / f"{stem}_local.npy"
        if not npy_path.exists():
            continue
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        points_xy = np.load(npy_path).astype(np.float32, copy=False)
        snapshots.append(Snapshot(name=stem, points_xy=points_xy, metadata=metadata))
    if not snapshots:
        raise SystemExit(f"No snapshot json/npy pairs found in {snapshot_dir}")
    return snapshots


def _build_occupancy(points_xy: np.ndarray, *, resolution_m: float, padding_m: float) -> tuple[np.ndarray, np.ndarray]:
    if len(points_xy) == 0:
        return np.zeros((1, 1), dtype=bool), np.asarray([0.0, 0.0], dtype=np.float32)
    mins = np.min(points_xy, axis=0) - padding_m
    maxs = np.max(points_xy, axis=0) + padding_m
    width = int(math.ceil((maxs[0] - mins[0]) / resolution_m)) + 1
    height = int(math.ceil((maxs[1] - mins[1]) / resolution_m)) + 1
    grid = np.zeros((height, width), dtype=bool)
    ij = np.round((points_xy - mins) / resolution_m).astype(np.int32)
    ij[:, 0] = np.clip(ij[:, 0], 0, width - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, height - 1)
    grid[ij[:, 1], ij[:, 0]] = True
    return grid, mins.astype(np.float32)


def _dilate(grid: np.ndarray, radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        return grid
    dilated = grid.copy()
    ys, xs = np.nonzero(grid)
    for y, x in zip(ys, xs, strict=False):
        y0 = max(0, y - radius_cells)
        y1 = min(grid.shape[0], y + radius_cells + 1)
        x0 = max(0, x - radius_cells)
        x1 = min(grid.shape[1], x + radius_cells + 1)
        dilated[y0:y1, x0:x1] = True
    return dilated


def _score_candidate(
    candidate_points: np.ndarray,
    *,
    exact_grid: np.ndarray,
    dilated_grid: np.ndarray,
    known_grid: np.ndarray,
    grid_origin_xy: np.ndarray,
    resolution_m: float,
    global_sampled_xy: np.ndarray,
    use_nearest_penalty: bool = True,
) -> float:
    if len(candidate_points) == 0:
        return -1e9
    ij = np.round((candidate_points - grid_origin_xy) / resolution_m).astype(np.int32)
    inside = (
        (ij[:, 0] >= 0)
        & (ij[:, 0] < dilated_grid.shape[1])
        & (ij[:, 1] >= 0)
        & (ij[:, 1] < dilated_grid.shape[0])
    )
    if not np.any(inside):
        return -1e9
    inside_points = candidate_points[inside]
    valid = ij[inside]
    # Only judge alignment on points that fall inside the map's observed
    # neighborhood (`known_grid`). Points beyond it are NEW coverage — e.g. the
    # unseen half of the room after a 90deg turn with a ~180deg FOV — and carry
    # no alignment information. Scoring them as misses made every exploratory
    # capture look wrong, which is exactly what stalls a spin-to-map bootstrap.
    known = known_grid[valid[:, 1], valid[:, 0]]
    matchable_count = int(np.count_nonzero(known))
    total_count = int(len(candidate_points))
    if matchable_count < max(12, int(0.15 * total_count)):
        return -1e9
    matchable = valid[known]
    exact_hit_ratio = float(np.count_nonzero(exact_grid[matchable[:, 1], matchable[:, 0]])) / float(matchable_count)
    nearby_hit_ratio = float(np.count_nonzero(dilated_grid[matchable[:, 1], matchable[:, 0]])) / float(matchable_count)
    inside_ratio = float(np.count_nonzero(inside)) / float(total_count)
    miss_ratio = 1.0 - nearby_hit_ratio

    matchable_points = inside_points[known]
    sampled = matchable_points[:: max(1, len(matchable_points) // 24)]
    if use_nearest_penalty and len(global_sampled_xy) and len(sampled):
        deltas = sampled[:, None, :] - global_sampled_xy[None, :, :]
        distances_sq = np.sum(deltas * deltas, axis=2, dtype=np.float32)
        nearest_sq = np.min(distances_sq, axis=1)
        nearest_mean = float(np.sqrt(np.mean(nearest_sq, dtype=np.float32)))
    else:
        nearest_mean = 0.0

    base_score = (
        exact_hit_ratio * 12.0
        + nearby_hit_ratio * 3.0
        + inside_ratio * 1.5
        - miss_ratio * 6.0
        - min(nearest_mean, 0.5) * 10.0
    )
    # Prefer solutions explained by more of the map, but a 50% overlap (the
    # honest value right after a 90deg turn) already earns full weight.
    coverage_ratio = float(matchable_count) / float(total_count)
    coverage_weight = 0.55 + 0.45 * min(1.0, coverage_ratio / 0.50)
    return base_score * coverage_weight if base_score > 0.0 else base_score


def _evaluate_pose_score(
    *,
    snapshot_points_xy: np.ndarray,
    pose: Pose2D,
    exact_grid: np.ndarray,
    dilated_grid: np.ndarray,
    known_grid: np.ndarray,
    grid_origin_xy: np.ndarray,
    resolution_m: float,
    global_sampled_xy: np.ndarray,
    use_nearest_penalty: bool = True,
) -> float:
    candidate_points = _transform_points(snapshot_points_xy, pose)
    return _score_candidate(
        candidate_points,
        exact_grid=exact_grid,
        dilated_grid=dilated_grid,
        known_grid=known_grid,
        grid_origin_xy=grid_origin_xy,
        resolution_m=resolution_m,
        global_sampled_xy=global_sampled_xy,
        use_nearest_penalty=use_nearest_penalty,
    )


def _apply_pose_prior(
    *,
    score: float,
    pose: Pose2D,
    prior_pose: Pose2D | None,
    translation_weight: float,
    theta_weight: float,
) -> float:
    if prior_pose is None:
        return float(score)
    translation_error_m = math.hypot(float(pose.x) - float(prior_pose.x), float(pose.y) - float(prior_pose.y))
    theta_error_deg = abs(_normalize_angle_deg(float(pose.theta_deg) - float(prior_pose.theta_deg)))
    return float(score) - float(translation_weight) * translation_error_m - float(theta_weight) * theta_error_deg


def _prior_weights_for_hint(initial_pose: Pose2D, hint: MotionHint) -> tuple[Pose2D | None, float, float]:
    if hint.kind == "drive":
        # Drive captures must localize from LiDAR geometry, not commanded motion.
        # We still seed the search near the previous pose, but do not bias scoring
        # toward "you probably moved this far".
        return None, 0.0, 0.0
    if hint.kind in {"turn", "bootstrap_turn"}:
        # Turning progress is measured from LiDAR overlap in the caller, so keep
        # only a very light prior here to break ties without overriding geometry.
        return initial_pose, 0.4, 0.02
    return None, 0.0, 0.0


def _refine_pose(
    *,
    snapshot_points_xy: np.ndarray,
    seed_pose: Pose2D,
    best_score: float,
    exact_grid: np.ndarray,
    dilated_grid: np.ndarray,
    known_grid: np.ndarray,
    grid_origin_xy: np.ndarray,
    resolution_m: float,
    global_sampled_xy: np.ndarray,
    theta_half_window_deg: float,
    theta_step_deg: float,
    offset_half_window_m: float,
    offset_step_m: float,
    prior_pose: Pose2D | None = None,
    prior_translation_weight: float = 0.0,
    prior_theta_weight: float = 0.0,
    use_nearest_penalty: bool = True,
) -> tuple[Pose2D, float]:
    best_pose = seed_pose
    theta_candidates = np.arange(
        seed_pose.theta_deg - theta_half_window_deg,
        seed_pose.theta_deg + theta_half_window_deg + 1e-6,
        theta_step_deg,
        dtype=np.float32,
    )
    offset_candidates = np.arange(
        -offset_half_window_m,
        offset_half_window_m + 1e-6,
        offset_step_m,
        dtype=np.float32,
    )
    for theta_deg in theta_candidates:
        for dx in offset_candidates:
            for dy in offset_candidates:
                pose = Pose2D(float(seed_pose.x + dx), float(seed_pose.y + dy), float(theta_deg))
                score = _evaluate_pose_score(
                    snapshot_points_xy=snapshot_points_xy,
                    pose=pose,
                    exact_grid=exact_grid,
                    dilated_grid=dilated_grid,
                    known_grid=known_grid,
                    grid_origin_xy=grid_origin_xy,
                    resolution_m=resolution_m,
                    global_sampled_xy=global_sampled_xy,
                    use_nearest_penalty=use_nearest_penalty,
                )
                score = _apply_pose_prior(
                    score=score,
                    pose=pose,
                    prior_pose=prior_pose,
                    translation_weight=float(prior_translation_weight),
                    theta_weight=float(prior_theta_weight),
                )
                if score > best_score:
                    best_score = score
                    best_pose = pose
    return best_pose, best_score


def _whole_map_seed_poses(
    *,
    snapshot_points_xy: np.ndarray,
    global_points_xy: np.ndarray,
    coarse_angle_step_deg: float,
    resolution_m: float,
    theta_center_deg: float | None = None,
    theta_window_deg: float | None = None,
) -> list[Pose2D]:
    if len(snapshot_points_xy) == 0 or len(global_points_xy) == 0:
        return []

    theta_step_deg = max(float(coarse_angle_step_deg) * 1.5, 8.0)
    if theta_center_deg is None or theta_window_deg is None:
        theta_candidates = np.arange(0.0, 360.0, theta_step_deg, dtype=np.float32)
    else:
        theta_candidates = np.arange(
            float(theta_center_deg) - float(theta_window_deg),
            float(theta_center_deg) + float(theta_window_deg) + 1e-6,
            theta_step_deg,
            dtype=np.float32,
        )
    global_anchor_points = global_points_xy[:: max(1, len(global_points_xy) // 140)]
    snapshot_anchor_points = snapshot_points_xy[:: max(1, len(snapshot_points_xy) // 24)]
    bin_size_m = max(float(resolution_m) * 2.5, 0.08)
    ranked_bins: list[tuple[int, Pose2D]] = []

    for theta_deg in theta_candidates:
        rotated_snapshot = _transform_points(snapshot_anchor_points, Pose2D(0.0, 0.0, float(theta_deg)))
        deltas = global_anchor_points[:, None, :] - rotated_snapshot[None, :, :]
        flat_deltas = deltas.reshape(-1, 2)
        if len(flat_deltas) == 0:
            continue

        bins = np.round(flat_deltas / bin_size_m).astype(np.int32)
        unique_bins, counts = np.unique(bins, axis=0, return_counts=True)
        top_indices = np.argsort(counts)[-8:]
        for top_idx in reversed(top_indices.tolist()):
            seed_xy = unique_bins[top_idx].astype(np.float32) * bin_size_m
            ranked_bins.append(
                (
                    int(counts[top_idx]),
                    Pose2D(x=float(seed_xy[0]), y=float(seed_xy[1]), theta_deg=float(theta_deg)),
                )
            )

    ranked_bins.sort(key=lambda item: item[0], reverse=True)
    deduped: list[Pose2D] = []
    for _votes, pose in ranked_bins:
        if any(
            abs(existing.x - pose.x) < 0.10
            and abs(existing.y - pose.y) < 0.10
            and abs(((existing.theta_deg - pose.theta_deg + 180.0) % 360.0) - 180.0) < 8.0
            for existing in deduped
        ):
            continue
        deduped.append(pose)
        if len(deduped) >= 12:
            break

    return deduped


def _search_pose(
    *,
    snapshot_points_xy: np.ndarray,
    global_points_xy: np.ndarray,
    initial_pose: Pose2D,
    resolution_m: float,
    search_xy_m: float,
    coarse_angle_step_deg: float,
    fine_angle_step_deg: float,
    theta_window_deg: float,
    whole_map_theta_center_deg: float | None = None,
    whole_map_theta_window_deg: float | None = None,
    max_translation_from_initial_m: float | None = None,
    prior_pose: Pose2D | None = None,
    prior_translation_weight: float = 0.0,
    prior_theta_weight: float = 0.0,
) -> tuple[Pose2D, dict[str, object]]:
    build_started = time.monotonic()
    exact_grid, origin = _build_occupancy(global_points_xy, resolution_m=resolution_m, padding_m=search_xy_m + 0.4)
    dilated = _dilate(exact_grid, radius_cells=2)
    # "Known region" of the map: anywhere within ~0.30 m of an observed point.
    # Scan points outside it are new coverage and are excluded from scoring.
    known_grid = _dilate(exact_grid, radius_cells=max(3, int(round(0.30 / max(resolution_m, 1e-3)))))
    global_sampled_xy = global_points_xy[:: max(1, len(global_points_xy) // 128)] if len(global_points_xy) else global_points_xy
    build_elapsed_s = time.monotonic() - build_started

    local_best_pose = initial_pose
    local_best_score = -1e9

    theta_window_deg = max(float(theta_window_deg), float(fine_angle_step_deg))
    coarse_theta_candidates = np.arange(
        initial_pose.theta_deg - theta_window_deg,
        initial_pose.theta_deg + theta_window_deg + 1e-6,
        coarse_angle_step_deg,
        dtype=np.float32,
    )
    coarse_offset_candidates = np.arange(-search_xy_m, search_xy_m + 1e-6, 0.05, dtype=np.float32)

    local_search_started = time.monotonic()
    for theta_deg in coarse_theta_candidates:
        for dx in coarse_offset_candidates:
            for dy in coarse_offset_candidates:
                pose = Pose2D(float(initial_pose.x + dx), float(initial_pose.y + dy), float(theta_deg))
                raw_score = _evaluate_pose_score(
                    snapshot_points_xy=snapshot_points_xy,
                    pose=pose,
                    exact_grid=exact_grid,
                    dilated_grid=dilated,
                    known_grid=known_grid,
                    grid_origin_xy=origin,
                    resolution_m=resolution_m,
                    global_sampled_xy=global_sampled_xy,
                    use_nearest_penalty=False,
                )
                score = _apply_pose_prior(
                    score=raw_score,
                    pose=pose,
                    prior_pose=prior_pose,
                    translation_weight=float(prior_translation_weight),
                    theta_weight=float(prior_theta_weight),
                )
                if score > local_best_score:
                    local_best_score = score
                    local_best_pose = pose

    fine_theta_half_window_deg = max(6.0, min(18.0, theta_window_deg * 0.35))
    local_best_pose, local_best_score = _refine_pose(
        snapshot_points_xy=snapshot_points_xy,
        seed_pose=local_best_pose,
        best_score=local_best_score,
        exact_grid=exact_grid,
        dilated_grid=dilated,
        known_grid=known_grid,
        grid_origin_xy=origin,
        resolution_m=resolution_m,
        global_sampled_xy=global_sampled_xy,
        theta_half_window_deg=fine_theta_half_window_deg,
        theta_step_deg=float(fine_angle_step_deg),
        offset_half_window_m=0.14,
        offset_step_m=0.02,
        prior_pose=prior_pose,
        prior_translation_weight=float(prior_translation_weight),
        prior_theta_weight=float(prior_theta_weight),
        use_nearest_penalty=True,
    )
    local_search_elapsed_s = time.monotonic() - local_search_started

    global_best_pose = local_best_pose
    global_best_score = local_best_score
    run_whole_map_search = local_best_score < 8.0
    whole_map_search_elapsed_s = 0.0
    if run_whole_map_search:
        whole_map_started = time.monotonic()
        global_seed_poses = _whole_map_seed_poses(
            snapshot_points_xy=snapshot_points_xy,
            global_points_xy=global_points_xy,
            coarse_angle_step_deg=float(coarse_angle_step_deg),
            resolution_m=float(resolution_m),
            theta_center_deg=whole_map_theta_center_deg,
            theta_window_deg=whole_map_theta_window_deg,
        )
        for seed_pose in global_seed_poses:
            if max_translation_from_initial_m is not None:
                seed_translation = math.hypot(seed_pose.x - initial_pose.x, seed_pose.y - initial_pose.y)
                if seed_translation > float(max_translation_from_initial_m):
                    continue
            seed_raw_score = _evaluate_pose_score(
                snapshot_points_xy=snapshot_points_xy,
                pose=seed_pose,
                exact_grid=exact_grid,
                dilated_grid=dilated,
                known_grid=known_grid,
                grid_origin_xy=origin,
                resolution_m=resolution_m,
                global_sampled_xy=global_sampled_xy,
                use_nearest_penalty=False,
            )
            seed_score = _apply_pose_prior(
                score=seed_raw_score,
                pose=seed_pose,
                prior_pose=prior_pose,
                translation_weight=float(prior_translation_weight),
                theta_weight=float(prior_theta_weight),
            )
            refined_pose, refined_score = _refine_pose(
                snapshot_points_xy=snapshot_points_xy,
                seed_pose=seed_pose,
                best_score=seed_score,
                exact_grid=exact_grid,
                dilated_grid=dilated,
                known_grid=known_grid,
                grid_origin_xy=origin,
                resolution_m=resolution_m,
                global_sampled_xy=global_sampled_xy,
                theta_half_window_deg=10.0,
                theta_step_deg=1.0,
                offset_half_window_m=0.18,
                offset_step_m=0.03,
                prior_pose=prior_pose,
                prior_translation_weight=float(prior_translation_weight),
                prior_theta_weight=float(prior_theta_weight),
                use_nearest_penalty=True,
            )
            if max_translation_from_initial_m is not None:
                refined_translation = math.hypot(refined_pose.x - initial_pose.x, refined_pose.y - initial_pose.y)
                if refined_translation > float(max_translation_from_initial_m):
                    continue
            if refined_score > global_best_score:
                global_best_score = refined_score
                global_best_pose = refined_pose
        whole_map_search_elapsed_s = time.monotonic() - whole_map_started

    if run_whole_map_search and global_best_score > local_best_score + 0.1:
        best_pose = global_best_pose
        best_score = global_best_score
        source = "whole_map"
    else:
        best_pose = local_best_pose
        best_score = local_best_score
        source = "local"

    return best_pose, {
        "score": round(best_score, 4),
        "local_score": round(local_best_score, 4),
        "whole_map_score": round(global_best_score, 4),
        "source": source,
        "whole_map_searched": run_whole_map_search,
        "timing_s": {
            "build_occupancy": round(build_elapsed_s, 4),
            "local_search": round(local_search_elapsed_s, 4),
            "whole_map_search": round(whole_map_search_elapsed_s, 4),
            "total_search": round(build_elapsed_s + local_search_elapsed_s + whole_map_search_elapsed_s, 4),
        },
    }


def _build_score_grids(
    global_points_xy: np.ndarray,
    *,
    resolution_m: float,
    padding_m: float,
) -> dict[str, object]:
    """Precompute the occupancy/known-region grids used by _score_candidate so
    many poses (e.g. one solve per lidar revolution during a turn) can be
    scored against a static map without rebuilding grids each time."""
    exact_grid, grid_origin_xy = _build_occupancy(
        global_points_xy, resolution_m=float(resolution_m), padding_m=float(padding_m)
    )
    return {
        "exact_grid": exact_grid,
        "dilated_grid": _dilate(exact_grid, radius_cells=2),
        "known_grid": _dilate(exact_grid, radius_cells=max(3, int(round(0.30 / max(float(resolution_m), 1e-3))))),
        "grid_origin_xy": grid_origin_xy,
        "resolution_m": float(resolution_m),
        "global_sampled_xy": global_points_xy[:: max(1, len(global_points_xy) // 128)],
    }


def _solve_arc_pose_on_grids(
    *,
    snapshot_points_xy: np.ndarray,
    grids: dict[str, object],
    arc_center_xy: tuple[float, float],
    lidar_offset_forward_m: float,
    expected_theta_deg: float,
    theta_half_window_deg: float,
    theta_step_deg: float = 1.5,
    center_slack_m: float = 0.10,
    slack_step_m: float = 0.05,
    theta_prior_weight_per_deg: float = 0.022,
    refine: bool = True,
) -> tuple[Pose2D, float]:
    """Score arc-constrained poses against prebuilt grids: for each candidate
    heading the sensor position is center + R(theta)*(offset, 0) plus a small
    slip slack. Returns the best pose and its RAW geometry score (a gentle
    dead-reckoning prior only biases selection between near-tied modes)."""
    r = float(lidar_offset_forward_m)
    center_x = float(arc_center_xy[0])
    center_y = float(arc_center_xy[1])

    def arc_pose(theta_deg: float, dx: float, dy: float) -> Pose2D:
        theta_rad = math.radians(float(theta_deg))
        return Pose2D(
            x=center_x + r * math.cos(theta_rad) + float(dx),
            y=center_y + r * math.sin(theta_rad) + float(dy),
            theta_deg=float(theta_deg),
        )

    def score_pose(pose: Pose2D, use_nearest_penalty: bool) -> float:
        return _score_candidate(
            _transform_points(snapshot_points_xy, pose),
            use_nearest_penalty=use_nearest_penalty,
            **grids,
        )

    prior_w = max(0.0, float(theta_prior_weight_per_deg))

    def theta_penalty(theta_deg: float) -> float:
        return prior_w * abs(_normalize_angle_deg(float(theta_deg) - float(expected_theta_deg)))

    slack = max(0.0, float(center_slack_m))
    half_window = max(float(theta_half_window_deg), float(theta_step_deg))
    coarse_offsets = np.arange(-slack, slack + 1e-6, max(0.01, float(slack_step_m)), dtype=np.float32)
    coarse_thetas = np.arange(
        float(expected_theta_deg) - half_window,
        float(expected_theta_deg) + half_window + 1e-6,
        max(0.5, float(theta_step_deg)),
        dtype=np.float32,
    )
    best_theta_deg = float(expected_theta_deg)
    best_dx = 0.0
    best_dy = 0.0
    best_score = -1e9
    best_adjusted = -1e9
    for theta_deg in coarse_thetas:
        penalty = theta_penalty(float(theta_deg))
        for dx in coarse_offsets:
            for dy in coarse_offsets:
                score = score_pose(arc_pose(float(theta_deg), float(dx), float(dy)), False)
                adjusted = float(score) - penalty
                if adjusted > best_adjusted:
                    best_adjusted = adjusted
                    best_score = float(score)
                    best_theta_deg = float(theta_deg)
                    best_dx = float(dx)
                    best_dy = float(dy)
    best_pose = arc_pose(best_theta_deg, best_dx, best_dy)
    if not refine:
        return best_pose, float(best_score)

    fine_offsets = np.arange(-0.04, 0.04 + 1e-6, 0.02, dtype=np.float32)
    fine_thetas = np.arange(best_theta_deg - 3.0, best_theta_deg + 3.0 + 1e-6, 0.5, dtype=np.float32)
    offset_cap = slack + 0.04
    best_score = -1e9
    best_adjusted = -1e9
    for theta_deg in fine_thetas:
        penalty = theta_penalty(float(theta_deg))
        for ddx in fine_offsets:
            for ddy in fine_offsets:
                dx = max(-offset_cap, min(offset_cap, best_dx + float(ddx)))
                dy = max(-offset_cap, min(offset_cap, best_dy + float(ddy)))
                pose = arc_pose(float(theta_deg), dx, dy)
                score = score_pose(pose, True)
                adjusted = float(score) - penalty
                if adjusted > best_adjusted:
                    best_adjusted = adjusted
                    best_score = float(score)
                    best_pose = pose
    return best_pose, float(best_score)


def _solve_turn_arc_pose(
    *,
    snapshot_points_xy: np.ndarray,
    transformed_sets: list[np.ndarray],
    previous_pose: Pose2D,
    lidar_offset_forward_m: float,
    resolution_m: float,
    expected_theta_deg: float,
    theta_half_window_deg: float = 80.0,
    theta_step_deg: float = 1.5,
    center_slack_m: float = 0.10,
    theta_prior_weight_per_deg: float = 0.022,
) -> tuple[Pose2D, dict[str, object]] | None:
    """Solve an in-place-turn capture with the sensor constrained to the arc it
    physically rides. The robot's rotation center sits `lidar_offset_forward_m`
    BEHIND the sensor and cannot translate during a spin, so each candidate
    heading fully determines the sensor position (up to a small slip slack)."""
    non_empty_sets = [points for points in transformed_sets if len(points)]
    if len(snapshot_points_xy) < 12 or not non_empty_sets:
        return None
    global_points_xy = np.concatenate(non_empty_sets, axis=0)
    build_started = time.monotonic()
    grids = _build_score_grids(
        global_points_xy,
        resolution_m=float(resolution_m),
        padding_m=float(center_slack_m) + 0.8,
    )
    build_elapsed_s = time.monotonic() - build_started

    r = float(lidar_offset_forward_m)
    prev_theta_rad = math.radians(float(previous_pose.theta_deg))
    arc_center_xy = (
        float(previous_pose.x) - r * math.cos(prev_theta_rad),
        float(previous_pose.y) - r * math.sin(prev_theta_rad),
    )
    search_started = time.monotonic()
    best_pose, best_score = _solve_arc_pose_on_grids(
        snapshot_points_xy=snapshot_points_xy,
        grids=grids,
        arc_center_xy=arc_center_xy,
        lidar_offset_forward_m=r,
        expected_theta_deg=float(expected_theta_deg),
        theta_half_window_deg=float(theta_half_window_deg),
        theta_step_deg=float(theta_step_deg),
        center_slack_m=float(center_slack_m),
        theta_prior_weight_per_deg=float(theta_prior_weight_per_deg),
        refine=True,
    )
    search_elapsed_s = time.monotonic() - search_started

    return best_pose, {
        "score": round(best_score, 4),
        "local_score": round(best_score, 4),
        "whole_map_score": round(best_score, 4),
        "source": "turn_arc",
        "whole_map_searched": False,
        "timing_s": {
            "build_occupancy": round(build_elapsed_s, 4),
            "local_search": round(search_elapsed_s, 4),
            "whole_map_search": 0.0,
            "total_search": round(build_elapsed_s + search_elapsed_s, 4),
        },
    }


def _default_turn_hints(
    *,
    snapshot_count: int,
    expected_turn_deg: float,
    turn_sign: str,
    search_xy_m: float,
) -> list[MotionHint]:
    signed_turn = float(expected_turn_deg) * (1.0 if turn_sign == "ccw" else -1.0)
    hints: list[MotionHint] = [
        MotionHint(
            kind="start",
            expected_dx_local_m=0.0,
            expected_dy_local_m=0.0,
            expected_dtheta_deg=0.0,
            search_xy_m=float(search_xy_m),
            search_theta_window_deg=max(float(expected_turn_deg) * 0.6, 10.0),
            label="start",
        )
    ]
    for index in range(1, int(snapshot_count)):
        hints.append(
            MotionHint(
                kind="turn",
                expected_dx_local_m=0.0,
                expected_dy_local_m=0.0,
                expected_dtheta_deg=signed_turn,
                search_xy_m=float(search_xy_m),
                search_theta_window_deg=max(abs(signed_turn) * 0.6, 10.0),
                label=f"turn_{index}",
            )
        )
    return hints


def _stitch_snapshots(
    *,
    snapshots: list[Snapshot],
    motion_hints: list[MotionHint] | None,
    resolution_m: float,
    fallback_search_xy_m: float,
) -> tuple[list[Pose2D], list[np.ndarray], list[dict[str, object]]]:
    hints = list(motion_hints) if motion_hints is not None else _default_turn_hints(
        snapshot_count=len(snapshots),
        expected_turn_deg=90.0,
        turn_sign="ccw",
        search_xy_m=float(fallback_search_xy_m),
    )
    if len(hints) < len(snapshots):
        hints.extend(
            MotionHint(
                kind="unknown",
                search_xy_m=float(fallback_search_xy_m),
                search_theta_window_deg=30.0,
                label=f"fallback_{idx}",
            )
            for idx in range(len(hints), len(snapshots))
        )

    poses: list[Pose2D] = [Pose2D(0.0, 0.0, 0.0)]
    transformed_sets: list[np.ndarray] = [_transform_points(snapshots[0].points_xy, poses[0])]
    solve_log: list[dict[str, object]] = [
        {
            "snapshot": snapshots[0].name,
            "pose": _pose_dict(poses[0]),
            "score": None,
            "motion_hint": {
                "kind": hints[0].kind,
                "label": hints[0].label,
                "expected_dx_local_m": round(float(hints[0].expected_dx_local_m), 4),
                "expected_dy_local_m": round(float(hints[0].expected_dy_local_m), 4),
                "expected_dtheta_deg": round(float(hints[0].expected_dtheta_deg), 4),
            },
            "host_revolution_index": snapshots[0].metadata.get("host_revolution_index"),
            "host_point_digest": snapshots[0].metadata.get("host_point_digest"),
        }
    ]

    for idx in range(1, len(snapshots)):
        hint = hints[idx]
        initial_pose = _advance_pose(poses[-1], hint)
        global_points_xy = np.concatenate(transformed_sets, axis=0)
        is_turn_like = hint.kind in {"turn", "bootstrap_turn"}
        is_drive_like = hint.kind == "drive"
        prior_pose, prior_translation_weight, prior_theta_weight = _prior_weights_for_hint(initial_pose, hint)
        solved_pose, score_meta = _search_pose(
            snapshot_points_xy=snapshots[idx].points_xy,
            global_points_xy=global_points_xy,
            initial_pose=initial_pose,
            resolution_m=float(resolution_m),
            search_xy_m=float(hint.search_xy_m),
            coarse_angle_step_deg=4.0,
            fine_angle_step_deg=0.5,
            theta_window_deg=float(hint.search_theta_window_deg),
            whole_map_theta_center_deg=float(initial_pose.theta_deg),
            whole_map_theta_window_deg=float(max(hint.search_theta_window_deg, 36.0))
            if is_turn_like
            else float(max(hint.search_theta_window_deg, 120.0))
            if is_drive_like
            else None,
            max_translation_from_initial_m=float(max(hint.search_xy_m + 0.20, 0.55))
            if is_turn_like
            else None
            if is_drive_like
            else None,
            prior_pose=prior_pose,
            prior_translation_weight=float(prior_translation_weight),
            prior_theta_weight=float(prior_theta_weight),
        )
        poses.append(solved_pose)
        transformed = _transform_points(snapshots[idx].points_xy, solved_pose)
        transformed_sets.append(transformed)
        solve_log.append(
            {
                "snapshot": snapshots[idx].name,
                "pose": _pose_dict(solved_pose),
                "initial_pose": _pose_dict(initial_pose),
                "score": score_meta["score"],
                "solve_source": score_meta.get("source"),
                "local_score": score_meta.get("local_score"),
                "whole_map_score": score_meta.get("whole_map_score"),
                "whole_map_searched": score_meta.get("whole_map_searched"),
                "timing_s": score_meta.get("timing_s"),
                "motion_hint": {
                    "kind": hint.kind,
                    "label": hint.label,
                    "expected_dx_local_m": round(float(hint.expected_dx_local_m), 4),
                    "expected_dy_local_m": round(float(hint.expected_dy_local_m), 4),
                    "expected_dtheta_deg": round(float(hint.expected_dtheta_deg), 4),
                    "search_xy_m": round(float(hint.search_xy_m), 4),
                    "search_theta_window_deg": round(float(hint.search_theta_window_deg), 4),
                },
                "host_revolution_index": snapshots[idx].metadata.get("host_revolution_index"),
                "host_point_digest": snapshots[idx].metadata.get("host_point_digest"),
            }
        )

    return poses, transformed_sets, solve_log


def _append_stitch_snapshot(
    *,
    snapshots: list[Snapshot],
    poses: list[Pose2D],
    transformed_sets: list[np.ndarray],
    solve_log: list[dict[str, object]],
    new_snapshot: Snapshot,
    motion_hint: MotionHint,
    resolution_m: float,
) -> tuple[list[Snapshot], list[Pose2D], list[np.ndarray], list[dict[str, object]]]:
    if not snapshots:
        initial_pose = Pose2D(0.0, 0.0, 0.0)
        transformed = _transform_points(new_snapshot.points_xy, initial_pose)
        return (
            [new_snapshot],
            [initial_pose],
            [transformed],
            [
                {
                    "snapshot": new_snapshot.name,
                    "pose": _pose_dict(initial_pose),
                    "score": None,
                    "motion_hint": {
                        "kind": motion_hint.kind,
                        "label": motion_hint.label,
                        "expected_dx_local_m": round(float(motion_hint.expected_dx_local_m), 4),
                        "expected_dy_local_m": round(float(motion_hint.expected_dy_local_m), 4),
                        "expected_dtheta_deg": round(float(motion_hint.expected_dtheta_deg), 4),
                    },
                    "host_revolution_index": new_snapshot.metadata.get("host_revolution_index"),
                    "host_point_digest": new_snapshot.metadata.get("host_point_digest"),
                }
            ],
        )

    initial_pose = _advance_pose(poses[-1], motion_hint)
    global_points_xy = np.concatenate(transformed_sets, axis=0)
    is_turn_like = motion_hint.kind in {"turn", "bootstrap_turn"}
    is_drive_like = motion_hint.kind == "drive"
    prior_pose, prior_translation_weight, prior_theta_weight = _prior_weights_for_hint(initial_pose, motion_hint)
    solved_pose, score_meta = _search_pose(
        snapshot_points_xy=new_snapshot.points_xy,
        global_points_xy=global_points_xy,
        initial_pose=initial_pose,
        resolution_m=float(resolution_m),
        search_xy_m=float(motion_hint.search_xy_m),
        coarse_angle_step_deg=4.0,
        fine_angle_step_deg=0.5,
        theta_window_deg=float(motion_hint.search_theta_window_deg),
        whole_map_theta_center_deg=float(initial_pose.theta_deg),
        whole_map_theta_window_deg=float(max(motion_hint.search_theta_window_deg, 36.0))
        if is_turn_like
        else float(max(motion_hint.search_theta_window_deg, 120.0))
        if is_drive_like
        else None,
        max_translation_from_initial_m=float(max(motion_hint.search_xy_m + 0.20, 0.55))
        if is_turn_like
        else None
        if is_drive_like
        else None,
        prior_pose=prior_pose,
        prior_translation_weight=float(prior_translation_weight),
        prior_theta_weight=float(prior_theta_weight),
    )
    transformed = _transform_points(new_snapshot.points_xy, solved_pose)

    snapshots_out = [*snapshots, new_snapshot]
    poses_out = [*poses, solved_pose]
    transformed_out = [*transformed_sets, transformed]
    solve_log_out = [
        *solve_log,
        {
            "snapshot": new_snapshot.name,
            "pose": _pose_dict(solved_pose),
            "initial_pose": _pose_dict(initial_pose),
            "score": score_meta["score"],
            "solve_source": score_meta.get("source"),
            "local_score": score_meta.get("local_score"),
            "whole_map_score": score_meta.get("whole_map_score"),
            "whole_map_searched": score_meta.get("whole_map_searched"),
            "timing_s": score_meta.get("timing_s"),
            "motion_hint": {
                "kind": motion_hint.kind,
                "label": motion_hint.label,
                "expected_dx_local_m": round(float(motion_hint.expected_dx_local_m), 4),
                "expected_dy_local_m": round(float(motion_hint.expected_dy_local_m), 4),
                "expected_dtheta_deg": round(float(motion_hint.expected_dtheta_deg), 4),
                "search_xy_m": round(float(motion_hint.search_xy_m), 4),
                "search_theta_window_deg": round(float(motion_hint.search_theta_window_deg), 4),
            },
            "host_revolution_index": new_snapshot.metadata.get("host_revolution_index"),
            "host_point_digest": new_snapshot.metadata.get("host_point_digest"),
        },
    ]
    return snapshots_out, poses_out, transformed_out, solve_log_out


def _generate_svg(transformed_sets: list[np.ndarray], poses: list[Pose2D], width: int = 1100, height: int = 900) -> str:
    all_points = np.concatenate([points for points in transformed_sets if len(points)], axis=0)
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    center = (mins + maxs) / 2.0
    span = np.maximum(maxs - mins, 1e-6)
    scale = 0.88 * min(width / span[0], height / span[1])

    def project(point_xy: np.ndarray) -> tuple[float, float]:
        x = (point_xy[0] - center[0]) * scale + width / 2.0
        y = height / 2.0 - (point_xy[1] - center[1]) * scale
        return float(x), float(y)

    parts: list[str] = [f'<svg viewBox="0 0 {width} {height}">']
    for frac in np.linspace(0.08, 0.92, 12):
        x = frac * width
        y = frac * height
        parts.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{height}" stroke="#1a2635" stroke-width="1"/>')
        parts.append(f'<line x1="0" y1="{y:.1f}" x2="{width}" y2="{y:.1f}" stroke="#1a2635" stroke-width="1"/>')

    for idx, points in enumerate(transformed_sets):
        color = COLORS[idx % len(COLORS)]
        for point in points:
            x, y = project(point)
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.6" fill="{color}"/>')

    for idx, pose in enumerate(poses):
        color = COLORS[idx % len(COLORS)]
        origin = np.asarray([pose.x, pose.y], dtype=np.float32)
        arrow_tip = origin + (_rotation_matrix(pose.theta_deg) @ np.asarray([0.28, 0.0], dtype=np.float32))
        ox, oy = project(origin)
        tx, ty = project(arrow_tip)
        parts.append(f'<circle cx="{ox:.2f}" cy="{oy:.2f}" r="5" fill="{color}" stroke="#fff7" stroke-width="1.5"/>')
        parts.append(f'<line x1="{ox:.2f}" y1="{oy:.2f}" x2="{tx:.2f}" y2="{ty:.2f}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<text x="{ox + 8:.2f}" y="{oy - 8:.2f}" fill="{color}" font-size="16">{idx + 1}</text>')

    parts.append("</svg>")
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline stitcher for direct LiDAR snapshots.")
    parser.add_argument("--snapshot-dir", default="artifacts/direct_lidar_snapshots")
    parser.add_argument("--output-dir", default="artifacts/direct_lidar_snapshots/offline_stitch")
    parser.add_argument("--resolution-m", type=float, default=0.03)
    parser.add_argument("--search-xy-m", type=float, default=0.45)
    parser.add_argument("--expected-turn-deg", type=float, default=90.0)
    parser.add_argument(
        "--turn-sign",
        choices=("ccw", "cw"),
        default="ccw",
        help="Expected direction the robot was rotated between captures.",
    )
    args = parser.parse_args()

    snapshots = _load_snapshots(Path(args.snapshot_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    motion_hints = _default_turn_hints(
        snapshot_count=len(snapshots),
        expected_turn_deg=float(args.expected_turn_deg),
        turn_sign=str(args.turn_sign),
        search_xy_m=float(args.search_xy_m),
    )
    poses, transformed_sets, solve_log = _stitch_snapshots(
        snapshots=snapshots,
        motion_hints=motion_hints,
        resolution_m=float(args.resolution_m),
        fallback_search_xy_m=float(args.search_xy_m),
    )

    svg = _generate_svg(transformed_sets, poses)
    report = {
        "schema": "sourccey.direct_snapshot_stitch.v1",
        "snapshot_dir": str(Path(args.snapshot_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "parameters": {
            "resolution_m": float(args.resolution_m),
            "search_xy_m": float(args.search_xy_m),
            "expected_turn_deg": float(args.expected_turn_deg),
            "turn_sign": args.turn_sign,
        },
        "solve_log": solve_log,
    }
    html = HTML_TEMPLATE.format(svg=svg, report=json.dumps(report, indent=2))

    html_path = output_dir / "latest_stitched_overlay.html"
    report_path = output_dir / "latest_stitched_overlay.json"
    html_path.write_text(html, encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved stitched preview to {html_path}")
    print(f"Saved stitch report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
