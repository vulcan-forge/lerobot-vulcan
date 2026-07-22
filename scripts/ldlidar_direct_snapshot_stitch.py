from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
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


def _score_candidates_batch(
    cand: np.ndarray,
    *,
    exact_grid: np.ndarray,
    dilated_grid: np.ndarray,
    known_grid: np.ndarray,
    grid_origin_xy: np.ndarray,
    resolution_m: float,
    global_sampled_xy: np.ndarray,
    use_nearest_penalty: bool = True,
) -> np.ndarray:
    """Vectorized replica of _score_candidate for a batch of K candidate
    point sets. cand is K x N x 2 float32 — each row a snapshot already
    rotated AND translated to its candidate pose (built by the callers as
    rotated + offset in float32, matching _transform_points' in-place +=
    on a float32 array).

    NUMERICAL CONTRACT: every accept gate in the stitcher is tuned to the
    scalar scores, so this path reproduces them bit-for-bit on the grid
    terms — same float32 adds, same rounding into grid cells, same count
    ratios. The batched nearest-penalty term differs from the scalar path
    only in float summation order (~1e-6), far below gate sensitivity."""
    n_cand = int(cand.shape[0])
    n_pts = int(cand.shape[1]) if n_cand else 0
    scores = np.full(n_cand, -1e9, dtype=np.float64)
    if n_pts == 0 or n_cand == 0:
        return scores
    if n_cand > 8192:
        # Bound peak memory (the K x N intermediates) on very wide searches.
        for start in range(0, n_cand, 8192):
            scores[start : start + 8192] = _score_candidates_batch(
                cand[start : start + 8192],
                exact_grid=exact_grid,
                dilated_grid=dilated_grid,
                known_grid=known_grid,
                grid_origin_xy=grid_origin_xy,
                resolution_m=resolution_m,
                global_sampled_xy=global_sampled_xy,
                use_nearest_penalty=use_nearest_penalty,
            )
        return scores
    rel = (cand - grid_origin_xy) / resolution_m
    ij = np.round(rel).astype(np.int32)
    height, width = dilated_grid.shape
    inside = (
        (ij[..., 0] >= 0)
        & (ij[..., 0] < width)
        & (ij[..., 1] >= 0)
        & (ij[..., 1] < height)
    )
    jj = np.clip(ij[..., 0], 0, width - 1)
    ii = np.clip(ij[..., 1], 0, height - 1)
    known = known_grid[ii, jj] & inside
    matchable_count = known.sum(axis=1)
    min_matchable = max(12, int(0.15 * n_pts))
    ok = matchable_count >= min_matchable
    if not np.any(ok):
        return scores
    exact_hits = (exact_grid[ii, jj] & known).sum(axis=1)
    nearby_hits = (dilated_grid[ii, jj] & known).sum(axis=1)
    inside_counts = inside.sum(axis=1)

    ok_idx = np.nonzero(ok)[0]
    m_counts = matchable_count[ok_idx].astype(np.float64)
    exact_hit_ratio = exact_hits[ok_idx] / m_counts
    nearby_hit_ratio = nearby_hits[ok_idx] / m_counts
    inside_ratio = inside_counts[ok_idx] / float(n_pts)
    miss_ratio = 1.0 - nearby_hit_ratio

    nearest_mean = np.zeros(len(ok_idx), dtype=np.float64)
    if use_nearest_penalty and len(global_sampled_xy):
        # Fully vectorized replica of the scalar's stride-sampled nearest
        # penalty (the per-candidate loop was the refine stage's whole
        # cost). The scalar takes matchable = points where inside&known in
        # original order, then matchable[::max(1, len//24)] — reproduced
        # here via strided ordinal picks into the row-major nonzero list.
        # Distances use the float64 |a-b|^2 identity via BLAS (chunked to
        # bound memory); deviation from the scalar float32 sums is ~1e-6,
        # far below any gate's sensitivity.
        known_ok = known[ok_idx]
        m_ok = matchable_count[ok_idx].astype(np.int64)
        stride = np.maximum(1, m_ok // 24)
        n_picks = (m_ok + stride - 1) // stride
        max_picks = int(n_picks.max()) if len(n_picks) else 0
        if max_picks > 0:
            n_ok = len(ok_idx)
            pick_ord = np.arange(max_picks, dtype=np.int64)[None, :] * stride[:, None]
            pick_valid = pick_ord < m_ok[:, None]
            _rows, matchable_cols = np.nonzero(known_ok)
            row_start = np.zeros(n_ok, dtype=np.int64)
            np.cumsum(m_ok[:-1], out=row_start[1:])
            flat_idx = row_start[:, None] + np.minimum(
                pick_ord, np.maximum(m_ok[:, None] - 1, 0)
            )
            pick_cols = matchable_cols[flat_idx]
            picked = cand[ok_idx[:, None], pick_cols]
            picked_f64 = picked.astype(np.float64)
            g_f64 = global_sampled_xy.astype(np.float64)
            g_norm_sq = np.sum(g_f64 * g_f64, axis=1)
            nearest_sq = np.empty((n_ok, max_picks), dtype=np.float64)
            chunk = max(1, 2048 // max(1, max_picks // 24))
            for start in range(0, n_ok, chunk):
                p = picked_f64[start : start + chunk]
                p_norm_sq = np.sum(p * p, axis=2)
                cross = p @ g_f64.T
                d2 = p_norm_sq[:, :, None] + g_norm_sq[None, None, :] - 2.0 * cross
                nearest_sq[start : start + chunk] = d2.min(axis=2)
            nearest_sq = np.maximum(nearest_sq, 0.0)
            sums = np.where(pick_valid, nearest_sq, 0.0).sum(axis=1)
            nearest_mean = np.sqrt(sums / n_picks.astype(np.float64))

    base_score = (
        exact_hit_ratio * 12.0
        + nearby_hit_ratio * 3.0
        + inside_ratio * 1.5
        - miss_ratio * 6.0
        - np.minimum(nearest_mean, 0.5) * 10.0
    )
    coverage_ratio = m_counts / float(n_pts)
    coverage_weight = 0.55 + 0.45 * np.minimum(1.0, coverage_ratio / 0.50)
    scores[ok_idx] = np.where(base_score > 0.0, base_score * coverage_weight, base_score)
    return scores


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
    if hint.kind in {"drive", "mixed"}:
        # Drive/mixed hints are LIDAR-TRACKED burst-by-burst (~5cm accurate),
        # not commanded motion — so the expected pose deserves real weight.
        # Without it, a capture taken beside a long straight wall can slide
        # along the wall at zero score cost and paint a phantom duplicate wall.
        return initial_pose, 2.0, 0.10
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
    anchor_pose: Pose2D | None = None,
    max_translation_m: float | None = None,
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
    # Vectorized over the xy grid per theta (was a triple Python loop of
    # ~3k full scoring calls — several seconds per capture). Candidate xy
    # in float32 mirrors the scalar path's float32 arithmetic exactly;
    # iteration order (dx-major, dy-minor, first-strictly-greater wins)
    # is preserved by argmax over the same ordering.
    grid_dx, grid_dy = np.meshgrid(offset_candidates, offset_candidates, indexing="ij")
    offset_pairs = np.column_stack([grid_dx.ravel(), grid_dy.ravel()])
    cand_x = np.float32(seed_pose.x) + offset_pairs[:, 0]
    cand_y = np.float32(seed_pose.y) + offset_pairs[:, 1]
    keep = np.ones(len(offset_pairs), dtype=bool)
    if anchor_pose is not None and max_translation_m is not None:
        # The caller's translation bound is a HARD promise: the refine walk
        # used to add its window on top of the coarse grid, letting a
        # wall-slide exceed the bound by ~40% (field 2026-07-12: a 0.53m
        # jump through a 0.35m cap shifted the whole map half a meter).
        keep = (
            np.hypot(
                cand_x.astype(np.float64) - float(anchor_pose.x),
                cand_y.astype(np.float64) - float(anchor_pose.y),
            )
            <= float(max_translation_m)
        )
    if not np.any(keep):
        return best_pose, best_score
    candidate_xy = np.column_stack([cand_x[keep], cand_y[keep]]).astype(np.float32)
    n_theta = len(theta_candidates)
    n_xy = len(candidate_xy)
    if n_theta == 0 or len(snapshot_points_xy) == 0:
        return best_pose, best_score
    # Score one theta slab at a time. Building theta x xy x points all at once
    # made recovery panoramas allocate hundreds of MB before the scorer's own
    # batching could help. Flattened theta-major ordering is unchanged.
    scores = np.empty(n_theta * n_xy, dtype=np.float64)
    for theta_index, theta in enumerate(theta_candidates):
        rotated = _transform_points(
            snapshot_points_xy, Pose2D(0.0, 0.0, float(theta))
        )
        candidates = rotated[None, :, :] + candidate_xy[:, None, :]
        start = theta_index * n_xy
        scores[start : start + n_xy] = _score_candidates_batch(
            candidates,
            exact_grid=exact_grid,
            dilated_grid=dilated_grid,
            known_grid=known_grid,
            grid_origin_xy=grid_origin_xy,
            resolution_m=resolution_m,
            global_sampled_xy=global_sampled_xy,
            use_nearest_penalty=use_nearest_penalty,
        )
    if prior_pose is not None:
        prior_translation_err = np.hypot(
            candidate_xy[:, 0].astype(np.float64) - float(prior_pose.x),
            candidate_xy[:, 1].astype(np.float64) - float(prior_pose.y),
        )
        theta_errs = np.abs(
            np.array(
                [
                    _normalize_angle_deg(float(t) - float(prior_pose.theta_deg))
                    for t in theta_candidates
                ],
                dtype=np.float64,
            )
        )
        scores = (
            scores
            - float(prior_translation_weight) * np.tile(prior_translation_err, n_theta)
            - float(prior_theta_weight) * np.repeat(theta_errs, n_xy)
        )
    k = int(np.argmax(scores))
    if float(scores[k]) > best_score:
        best_score = float(scores[k])
        best_pose = Pose2D(
            float(candidate_xy[k % n_xy, 0]),
            float(candidate_xy[k % n_xy, 1]),
            float(theta_candidates[k // n_xy]),
        )
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
    # Vectorized over the xy grid per theta (was a triple Python loop of
    # ~10k full scoring calls — this WAS the multi-second "stitching"
    # cost). Same candidate values, same ordering, same scores as the
    # scalar loops it replaces.
    coarse_dx, coarse_dy = np.meshgrid(
        coarse_offset_candidates, coarse_offset_candidates, indexing="ij"
    )
    coarse_pairs = np.column_stack([coarse_dx.ravel(), coarse_dy.ravel()])
    coarse_keep = np.ones(len(coarse_pairs), dtype=bool)
    if max_translation_from_initial_m is not None:
        coarse_keep = (
            np.hypot(
                coarse_pairs[:, 0].astype(np.float64),
                coarse_pairs[:, 1].astype(np.float64),
            )
            <= float(max_translation_from_initial_m)
        )
    coarse_candidate_xy = np.column_stack(
        [
            np.float32(initial_pose.x) + coarse_pairs[coarse_keep, 0],
            np.float32(initial_pose.y) + coarse_pairs[coarse_keep, 1],
        ]
    ).astype(np.float32)
    if len(coarse_candidate_xy) and len(coarse_theta_candidates) and len(snapshot_points_xy):
        n_coarse_theta = len(coarse_theta_candidates)
        n_coarse_xy = len(coarse_candidate_xy)
        coarse_scores = np.empty(n_coarse_theta * n_coarse_xy, dtype=np.float64)
        for theta_index, theta in enumerate(coarse_theta_candidates):
            rotated = _transform_points(
                snapshot_points_xy, Pose2D(0.0, 0.0, float(theta))
            )
            candidates = rotated[None, :, :] + coarse_candidate_xy[:, None, :]
            start = theta_index * n_coarse_xy
            coarse_scores[start : start + n_coarse_xy] = _score_candidates_batch(
                candidates,
                exact_grid=exact_grid,
                dilated_grid=dilated,
                known_grid=known_grid,
                grid_origin_xy=origin,
                resolution_m=resolution_m,
                global_sampled_xy=global_sampled_xy,
                use_nearest_penalty=False,
            )
        if prior_pose is not None:
            coarse_prior_err = np.hypot(
                coarse_candidate_xy[:, 0].astype(np.float64) - float(prior_pose.x),
                coarse_candidate_xy[:, 1].astype(np.float64) - float(prior_pose.y),
            )
            coarse_theta_errs = np.abs(
                np.array(
                    [
                        _normalize_angle_deg(float(t) - float(prior_pose.theta_deg))
                        for t in coarse_theta_candidates
                    ],
                    dtype=np.float64,
                )
            )
            coarse_scores = (
                coarse_scores
                - float(prior_translation_weight) * np.tile(coarse_prior_err, n_coarse_theta)
                - float(prior_theta_weight) * np.repeat(coarse_theta_errs, n_coarse_xy)
            )
        k = int(np.argmax(coarse_scores))
        if float(coarse_scores[k]) > local_best_score:
            local_best_score = float(coarse_scores[k])
            local_best_pose = Pose2D(
                float(coarse_candidate_xy[k % n_coarse_xy, 0]),
                float(coarse_candidate_xy[k % n_coarse_xy, 1]),
                float(coarse_theta_candidates[k // n_coarse_xy]),
            )

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
        anchor_pose=initial_pose,
        max_translation_m=max_translation_from_initial_m,
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
                anchor_pose=initial_pose,
                max_translation_m=max_translation_from_initial_m,
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


def _build_exploration_grid(
    *,
    poses: list[Pose2D],
    transformed_sets: list[np.ndarray],
    resolution_m: float,
    padding_m: float = 0.6,
    robot_clear_radius_m: float = 0.0,
    lidar_offset_forward_m: float = 0.0,
    extra_occupied_xy: np.ndarray | None = None,
) -> dict[str, object] | None:
    """Explicit model of what has been OBSERVED: for every capture, the space
    along each lidar beam (sensor pose -> return) is known-FREE, the return
    cell is OCCUPIED, and everything no beam ever crossed is UNKNOWN. This is
    the map knowledge that makes 'where have I not mapped yet?' a computable
    question instead of a heuristic."""
    paired = [
        (pose, points)
        for pose, points in zip(poses, transformed_sets, strict=False)
        if len(points)
    ]
    if not paired:
        return None
    all_points = np.concatenate([points for _pose, points in paired], axis=0)
    if extra_occupied_xy is not None and len(extra_occupied_xy):
        all_points = np.concatenate(
            [all_points, np.asarray(extra_occupied_xy, dtype=np.float32)], axis=0
        )
    pose_xy = np.asarray([[float(p.x), float(p.y)] for p, _pts in paired], dtype=np.float32)
    res = max(0.03, float(resolution_m))
    mins = np.minimum(all_points.min(axis=0), pose_xy.min(axis=0)) - float(padding_m)
    maxs = np.maximum(all_points.max(axis=0), pose_xy.max(axis=0)) + float(padding_m)
    width = int(math.ceil((maxs[0] - mins[0]) / res)) + 1
    height = int(math.ceil((maxs[1] - mins[1]) / res)) + 1
    if width <= 2 or height <= 2:
        return None
    free_grid = np.zeros((height, width), dtype=bool)
    occupied_grid = np.zeros((height, width), dtype=bool)
    mins32 = mins.astype(np.float32)

    for pose, points in paired:
        origin_xy = np.asarray([float(pose.x), float(pose.y)], dtype=np.float32)
        for point_xy in points:
            delta = point_xy - origin_xy
            beam_len = float(math.hypot(float(delta[0]), float(delta[1])))
            if beam_len < res:
                continue
            n_samples = int(beam_len / res) + 1
            ts = np.arange(n_samples, dtype=np.float32) / float(n_samples)
            samples = origin_xy[None, :] + delta[None, :] * ts[:, None]
            ij = np.round((samples - mins32) / res).astype(np.int32)
            ij[:, 0] = np.clip(ij[:, 0], 0, width - 1)
            ij[:, 1] = np.clip(ij[:, 1], 0, height - 1)
            free_grid[ij[:, 1], ij[:, 0]] = True
        end_ij = np.round((points - mins32) / res).astype(np.int32)
        end_ij[:, 0] = np.clip(end_ij[:, 0], 0, width - 1)
        end_ij[:, 1] = np.clip(end_ij[:, 1], 0, height - 1)
        occupied_grid[end_ij[:, 1], end_ij[:, 0]] = True

    # The robot's own footprint is observed BY OCCUPANCY: it physically stands
    # there, so that space is known-free even though no beam can cross it (the
    # sensor rides a lever arm, so the disk under the body is never swept —
    # without this stamp it stays "unknown" forever and the planner chases a
    # phantom frontier directly beneath the robot).
    clear_cells = int(round(max(0.0, float(robot_clear_radius_m)) / res))
    if clear_cells > 0:
        disk_offsets = [
            (dx, dy)
            for dy in range(-clear_cells, clear_cells + 1)
            for dx in range(-clear_cells, clear_cells + 1)
            if dx * dx + dy * dy <= clear_cells * clear_cells
        ]
        for pose, _points in paired:
            theta_rad = math.radians(float(pose.theta_deg))
            center_x = float(pose.x) - float(lidar_offset_forward_m) * math.cos(theta_rad)
            center_y = float(pose.y) - float(lidar_offset_forward_m) * math.sin(theta_rad)
            ci = int(round((center_x - float(mins32[0])) / res))
            cj = int(round((center_y - float(mins32[1])) / res))
            for dx, dy in disk_offsets:
                gx, gy = ci + dx, cj + dy
                if 0 <= gx < width and 0 <= gy < height:
                    free_grid[gy, gx] = True

    # Camera-confirmed ELEVATED obstacles (table edges etc.) the lidar cannot
    # see: stamped occupied AFTER everything else so no beam or footprint can
    # clear them — the planner must never route through one.
    if extra_occupied_xy is not None and len(extra_occupied_xy):
        extra = np.asarray(extra_occupied_xy, dtype=np.float32)
        extra_ij = np.round((extra - mins32) / res).astype(np.int32)
        extra_ij[:, 0] = np.clip(extra_ij[:, 0], 0, width - 1)
        extra_ij[:, 1] = np.clip(extra_ij[:, 1], 0, height - 1)
        occupied_grid[extra_ij[:, 1], extra_ij[:, 0]] = True
        free_grid[extra_ij[:, 1], extra_ij[:, 0]] = False

    free_grid &= ~occupied_grid
    return {
        "free": free_grid,
        "occupied": occupied_grid,
        "origin_xy": mins32,
        "resolution_m": res,
    }


def _plan_frontier_path(
    *,
    grid: dict[str, object],
    robot_xy: tuple[float, float],
    robot_radius_m: float,
    min_frontier_cells: int = 8,
    preferred_frontier_cells: int = 24,
    goal_standoff_m: float = 0.35,
    waypoint_lookahead_m: float = 1.10,
    target_xy: tuple[float, float] | None = None,
    survey_from_xy: list[tuple[float, float]] | None = None,
    min_survey_spacing_m: float = 0.80,
    observed_from_xy: list[tuple[float, float]] | None = None,
    observation_range_m: float = 2.80,
    robot_theta_deg: float | None = None,
    avoid_face_xy: list[tuple[float, float]] | None = None,
    avoid_radius_m: float = 0.55,
    prefer_face_xy: tuple[float, float] | None = None,
    prefer_radius_m: float = 0.60,
    prefer_bonus_m: float = 1.20,
) -> dict[str, object]:
    """Classic frontier exploration: BFS through known-free space (inflated by
    the robot's radius, so gaps it cannot fit through are simply not
    traversable) to the best boundary between known and unknown space.
    Selection is TWO-TIER: any cluster with >= preferred_frontier_cells (a
    genuinely unexplored region) outranks every smaller scrap regardless of
    distance — nearest-first alone grinds 8-cell slivers next to the robot
    while half the room sits unmapped. Ties within a tier break by
    path + turn cost. Clusters whose centroid falls within avoid_radius_m of
    an avoid_face_xy entry (repeatedly blocked/unreachable goals) are skipped.
    Returns the goal (a standoff pose near the frontier), the next waypoint
    along the actual traversable path, and the frontier centroid to face when
    scanning. status: ok | no_frontier | stuck."""
    free_grid: np.ndarray = grid["free"]
    occupied_grid: np.ndarray = grid["occupied"]
    origin_xy: np.ndarray = grid["origin_xy"]
    res = float(grid["resolution_m"])
    height, width = free_grid.shape

    inflated = _dilate(occupied_grid, radius_cells=max(1, int(round(float(robot_radius_m) / res))))
    traversable = free_grid & ~inflated

    # CLEARANCE FIELD: for every cell, the BFS cell-distance to the nearest
    # NON-traversable cell (wall, inflation band, or off-grid edge). The path
    # extractor uses this to route down the MIDDLE of a corridor rather than
    # along one edge. Field 2026-07-20: with the squeeze radius letting the
    # planner finally route through the 0.635m exit hallway, the robot beelined
    # the door but caught its SHOULDER on the frame because the shortest path
    # hugged one side of the channel. Centering the path gives the shoulder the
    # full ~0.09m/side the doorway actually offers. O(cells), computed once per
    # plan; a multi-source BFS seeded from every blocked cell (edges count as
    # blocked so paths do not skim the grid boundary either).
    clearance = np.full((height, width), -1, dtype=np.int32)
    clearance_q: deque[tuple[int, int]] = deque()
    blocked_ys, blocked_xs = np.where(~traversable)
    for _by, _bx in zip(blocked_ys.tolist(), blocked_xs.tolist(), strict=False):
        clearance[_by, _bx] = 0
        clearance_q.append((_bx, _by))
    while clearance_q:
        cqx, cqy = clearance_q.popleft()
        c_next = clearance[cqy, cqx] + 1
        for nx, ny in ((cqx + 1, cqy), (cqx - 1, cqy), (cqx, cqy + 1), (cqx, cqy - 1)):
            if 0 <= nx < width and 0 <= ny < height and clearance[ny, nx] < 0:
                clearance[ny, nx] = c_next
                clearance_q.append((nx, ny))

    def to_cell(x: float, y: float) -> tuple[int, int]:
        return (
            int(round((float(x) - float(origin_xy[0])) / res)),
            int(round((float(y) - float(origin_xy[1])) / res)),
        )

    def to_world(cx: int, cy: int) -> tuple[float, float]:
        return (
            float(origin_xy[0]) + float(cx) * res,
            float(origin_xy[1]) + float(cy) * res,
        )

    def is_avoided(point_xy: tuple[float, float]) -> bool:
        if not avoid_face_xy:
            return False
        return any(
            math.hypot(point_xy[0] - float(ax), point_xy[1] - float(ay)) <= float(avoid_radius_m)
            for ax, ay in avoid_face_xy
        )

    def is_preferred(point_xy: tuple[float, float]) -> bool:
        """GOAL COMMITMENT: the frontier chosen last cycle keeps priority
        until consumed or blacklisted. Re-picking from scratch every capture
        made the winner flip sides as cluster sizes shifted a few cells —
        the robot spun through 100-165deg turnarounds cycle after cycle
        (field 2026-07-11: ten captures of pure rotation) instead of
        finishing what it was facing."""
        if prefer_face_xy is None:
            return False
        return (
            math.hypot(
                point_xy[0] - float(prefer_face_xy[0]), point_xy[1] - float(prefer_face_xy[1])
            )
            <= float(prefer_radius_m)
        )

    def turn_cost_m(target_world_xy: tuple[float, float]) -> float:
        """Price heading change like travel distance (~0.9m per 90deg) so goal
        selection sweeps by angular continuity: 'nearest unknown first' alone
        flips sides after every scan and the robot ping-pongs through 180deg
        turnarounds between opposite openings."""
        if robot_theta_deg is None:
            return 0.0
        bearing_deg = math.degrees(
            math.atan2(
                float(target_world_xy[1]) - float(robot_xy[1]),
                float(target_world_xy[0]) - float(robot_xy[0]),
            )
        )
        return 0.9 * abs(_normalize_angle_deg(bearing_deg - float(robot_theta_deg))) / 90.0

    seed_x, seed_y = to_cell(robot_xy[0], robot_xy[1])
    seed_x = max(0, min(width - 1, seed_x))
    seed_y = max(0, min(height - 1, seed_y))
    if not traversable[seed_y, seed_x]:
        # The robot itself may stand within the inflation band next to
        # clutter; seed the search from the nearest traversable cell.
        search_r = max(1, int(round(0.60 / res)))
        best = None
        for dy in range(-search_r, search_r + 1):
            for dx in range(-search_r, search_r + 1):
                nx, ny = seed_x + dx, seed_y + dy
                if 0 <= nx < width and 0 <= ny < height and traversable[ny, nx]:
                    d2 = dx * dx + dy * dy
                    if best is None or d2 < best[0]:
                        best = (d2, nx, ny)
        if best is None:
            return {"status": "stuck"}
        seed_x, seed_y = best[1], best[2]

    dist = np.full((height, width), -1, dtype=np.int32)
    dist[seed_y, seed_x] = 0
    queue: deque[tuple[int, int]] = deque([(seed_x, seed_y)])
    while queue:
        cx, cy = queue.popleft()
        d_next = dist[cy, cx] + 1
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if 0 <= nx < width and 0 <= ny < height and traversable[ny, nx] and dist[ny, nx] < 0:
                dist[ny, nx] = d_next
                queue.append((nx, ny))

    def backtrack_path(from_cell: tuple[int, int]) -> list[tuple[int, int]]:
        path: list[tuple[int, int]] = [from_cell]
        px, py = from_cell
        while dist[py, px] > 0:
            d_here = dist[py, px]
            # Among the predecessors on a shortest path (dist == d_here - 1),
            # step to the one with the MOST clearance. This keeps the path the
            # same length but bows it toward the center of any corridor instead
            # of skimming a wall — so a shoulder does not clip the doorframe
            # (field 2026-07-20). Falls back to the first valid predecessor when
            # clearances tie.
            best: tuple[int, int, int] | None = None  # (clearance, nx, ny)
            for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                if 0 <= nx < width and 0 <= ny < height and dist[ny, nx] == d_here - 1:
                    c_here = int(clearance[ny, nx])
                    if best is None or c_here > best[0]:
                        best = (c_here, nx, ny)
            if best is None:
                break
            px, py = best[1], best[2]
            path.append((px, py))
        path.reverse()  # robot -> destination
        return path

    if target_xy is not None:
        # Navigate to an explicit destination (e.g. a survey vantage) through
        # the same traversable space, instead of hunting for a frontier.
        tx, ty = to_cell(float(target_xy[0]), float(target_xy[1]))
        search_r = max(1, int(round(0.60 / res)))
        best_target_cell = None
        for dy in range(-search_r, search_r + 1):
            for dx in range(-search_r, search_r + 1):
                nx, ny = tx + dx, ty + dy
                if 0 <= nx < width and 0 <= ny < height and dist[ny, nx] >= 0:
                    d2 = dx * dx + dy * dy
                    if best_target_cell is None or d2 < best_target_cell[0]:
                        best_target_cell = (d2, nx, ny)
        if best_target_cell is None:
            return {"status": "target_unreachable"}
        target_cell = (best_target_cell[1], best_target_cell[2])
        path_cells = backtrack_path(target_cell)
        goal_xy = to_world(*target_cell)
        waypoint_xy = goal_xy
        for cell in path_cells:
            wx, wy = to_world(*cell)
            if math.hypot(wx - float(robot_xy[0]), wy - float(robot_xy[1])) >= float(waypoint_lookahead_m):
                waypoint_xy = (wx, wy)
                break
        return {
            "status": "ok",
            "goal_xy": goal_xy,
            "waypoint_xy": waypoint_xy,
            "face_xy": (float(target_xy[0]), float(target_xy[1])),
            "path_length_m": float(dist[target_cell[1], target_cell[0]]) * res,
            "frontier_cells": 0,
        }

    def no_frontier_diag() -> dict[str, object]:
        """Why-no-frontier counters: 'no unknown space left' and 'unknown
        space exists but its boundary is unreachable (corridor pinched shut
        in the inflated grid)' print identically in the logs yet demand
        opposite responses — the second means the map, not the room, is the
        obstacle."""
        unreachable = free_grid & adjacent_unknown & (dist < 0)
        out: dict[str, object] = {
            "unknown_cells": int(np.count_nonzero(unknown)),
            "reachable_frontier_cells": int(np.count_nonzero(frontier_mask)),
            "unreachable_frontier_cells": int(np.count_nonzero(unreachable)),
        }
        pts = np.argwhere(unreachable)
        if len(pts):
            out["unreachable_centroid_xy"] = (
                float(origin_xy[0]) + float(np.mean(pts[:, 1])) * res,
                float(origin_xy[1]) + float(np.mean(pts[:, 0])) * res,
            )
        return out

    def survey_or_no_frontier() -> dict[str, object]:
        """No frontier left: optionally propose a SURVEY vantage instead — the
        reachable cell farthest from every previous capture position. Uses the
        same BFS reachability as navigation, so a proposed vantage is
        guaranteed pathable (a separate ray heuristic can silently disagree)."""
        if not survey_from_xy:
            return {"status": "no_frontier", **no_frontier_diag()}
        # Prefer COMFORTABLE vantages: max-distance-from-past-poses alone is
        # always won by a wall corner — the worst place to stand (blocked stop
        # box, poor visibility). Require extra obstacle clearance, falling back
        # to any reachable cell only if nothing comfortable qualifies.
        comfortable_block = _dilate(
            occupied_grid,
            radius_cells=max(1, int(round((float(robot_radius_m) + 0.25) / res))),
        )
        reached_cells = np.argwhere((dist >= 0) & ~comfortable_block)
        if len(reached_cells) == 0:
            reached_cells = np.argwhere(dist >= 0)
        if len(reached_cells) == 0:
            return {"status": "no_frontier", **no_frontier_diag()}
        world_pts = np.column_stack(
            [
                float(origin_xy[0]) + reached_cells[:, 1].astype(np.float32) * res,
                float(origin_xy[1]) + reached_cells[:, 0].astype(np.float32) * res,
            ]
        )
        min_spacing = np.full(len(world_pts), np.inf, dtype=np.float32)
        for sx, sy in survey_from_xy:
            np.minimum(
                min_spacing,
                np.hypot(world_pts[:, 0] - float(sx), world_pts[:, 1] - float(sy)),
                out=min_spacing,
            )
        best_i = int(np.argmax(min_spacing))
        if float(min_spacing[best_i]) < float(min_survey_spacing_m):
            return {
                "status": "no_frontier",
                "best_survey_spacing_m": float(min_spacing[best_i]),
                **no_frontier_diag(),
            }
        survey_cell = (int(reached_cells[best_i][1]), int(reached_cells[best_i][0]))
        path_cells = backtrack_path(survey_cell)
        goal_xy = to_world(*survey_cell)
        waypoint_xy = goal_xy
        for cell in path_cells:
            wx, wy = to_world(*cell)
            if math.hypot(wx - float(robot_xy[0]), wy - float(robot_xy[1])) >= float(waypoint_lookahead_m):
                waypoint_xy = (wx, wy)
                break
        return {
            "status": "survey",
            "goal_xy": goal_xy,
            "waypoint_xy": waypoint_xy,
            "face_xy": goal_xy,
            "path_length_m": float(dist[survey_cell[1], survey_cell[0]]) * res,
            "frontier_cells": 0,
            "survey_spacing_m": float(min_spacing[best_i]),
        }

    def observation_goal_or_none() -> dict[str, object] | None:
        """Frontiers the robot cannot REACH (gaps narrower than its body) can
        still be MAPPED: the lidar only needs line-of-sight, not passage. Find
        unknown-adjacent free cells that BFS cannot reach, cluster them, and
        for the nearest cluster pick a reachable vantage with a clear sight
        line to it — excluding vantages already captured from, so each opening
        is peered through a bounded number of times."""
        if observed_from_xy is None:
            return None
        unreached_frontier = free_grid & adjacent_unknown & (dist < 0)
        if not np.any(unreached_frontier):
            return None
        reached_cells = np.argwhere(dist >= 0)
        if len(reached_cells) == 0:
            return None
        reached_world = np.column_stack(
            [
                float(origin_xy[0]) + reached_cells[:, 1].astype(np.float32) * res,
                float(origin_xy[1]) + reached_cells[:, 0].astype(np.float32) * res,
            ]
        )
        # exclude vantages within 0.45m of any previous capture pose
        vantage_ok = np.ones(len(reached_world), dtype=bool)
        for px, py in observed_from_xy:
            vantage_ok &= (
                np.hypot(reached_world[:, 0] - float(px), reached_world[:, 1] - float(py)) > 0.45
            )
        if not np.any(vantage_ok):
            return None

        # cluster the unreachable frontier cells (8-connected)
        seen = np.zeros_like(unreached_frontier)
        clusters: list[list[tuple[int, int]]] = []
        for fy, fx in np.argwhere(unreached_frontier):
            if seen[fy, fx]:
                continue
            members: list[tuple[int, int]] = []
            stack: deque[tuple[int, int]] = deque([(int(fx), int(fy))])
            seen[fy, fx] = True
            while stack:
                cx, cy = stack.popleft()
                members.append((cx, cy))
                for nx in (cx - 1, cx, cx + 1):
                    for ny in (cy - 1, cy, cy + 1):
                        if (
                            0 <= nx < width
                            and 0 <= ny < height
                            and unreached_frontier[ny, nx]
                            and not seen[ny, nx]
                        ):
                            seen[ny, nx] = True
                            stack.append((nx, ny))
            if len(members) >= int(min_frontier_cells):
                clusters.append(members)
        if not clusters:
            return None

        def centroid_of(members: list[tuple[int, int]]) -> tuple[float, float]:
            return (
                float(np.mean([to_world(cx, cy)[0] for cx, cy in members])),
                float(np.mean([to_world(cx, cy)[1] for cx, cy in members])),
            )

        def line_of_sight_clear(from_xy: tuple[float, float], to_xy: tuple[float, float]) -> bool:
            span = math.hypot(to_xy[0] - from_xy[0], to_xy[1] - from_xy[1])
            n_samples = max(2, int(span / res))
            for t_index in range(1, n_samples):
                t = t_index / float(n_samples)
                sx = from_xy[0] + (to_xy[0] - from_xy[0]) * t
                sy = from_xy[1] + (to_xy[1] - from_xy[1]) * t
                gx = int(round((sx - float(origin_xy[0])) / res))
                gy = int(round((sy - float(origin_xy[1])) / res))
                if 0 <= gx < width and 0 <= gy < height and occupied_grid[gy, gx]:
                    return False
            return True

        def observation_cluster_cost(members: list[tuple[int, int]]) -> float:
            centroid = centroid_of(members)
            travel_m = math.hypot(
                centroid[0] - float(robot_xy[0]), centroid[1] - float(robot_xy[1])
            )
            cost = travel_m + turn_cost_m(centroid)
            if is_preferred(centroid):
                cost = max(0.0, cost - float(prefer_bonus_m))
            return cost

        clusters.sort(key=observation_cluster_cost)
        for members in clusters:
            centroid_xy = centroid_of(members)
            if is_avoided(centroid_xy):
                continue
            d_to_centroid = np.hypot(
                reached_world[:, 0] - centroid_xy[0], reached_world[:, 1] - centroid_xy[1]
            )
            candidate_indices = np.nonzero(
                vantage_ok & (d_to_centroid >= 0.35) & (d_to_centroid <= float(observation_range_m))
            )[0]
            if len(candidate_indices) == 0:
                continue
            travel_cost = (
                dist[reached_cells[candidate_indices, 0], reached_cells[candidate_indices, 1]].astype(
                    np.float32
                )
                * res
                + 0.6 * d_to_centroid[candidate_indices]
            )
            for local_i in np.argsort(travel_cost)[:40]:
                idx = int(candidate_indices[local_i])
                vantage_xy = (float(reached_world[idx, 0]), float(reached_world[idx, 1]))
                if not line_of_sight_clear(vantage_xy, centroid_xy):
                    continue
                vantage_cell = (int(reached_cells[idx][1]), int(reached_cells[idx][0]))
                path_cells = backtrack_path(vantage_cell)
                waypoint_xy = vantage_xy
                for cell in path_cells:
                    wx, wy = to_world(*cell)
                    if (
                        math.hypot(wx - float(robot_xy[0]), wy - float(robot_xy[1]))
                        >= float(waypoint_lookahead_m)
                    ):
                        waypoint_xy = (wx, wy)
                        break
                return {
                    "status": "observe",
                    "goal_xy": vantage_xy,
                    "waypoint_xy": waypoint_xy,
                    "face_xy": centroid_xy,
                    "path_length_m": float(dist[vantage_cell[1], vantage_cell[0]]) * res,
                    "frontier_cells": int(len(members)),
                }
        return None

    def observation_or_fallback() -> dict[str, object]:
        observation_plan = observation_goal_or_none()
        if observation_plan is not None:
            return observation_plan
        return survey_or_no_frontier()

    unknown = ~free_grid & ~occupied_grid
    adjacent_unknown = np.zeros_like(unknown)
    adjacent_unknown[1:, :] |= unknown[:-1, :]
    adjacent_unknown[:-1, :] |= unknown[1:, :]
    adjacent_unknown[:, 1:] |= unknown[:, :-1]
    adjacent_unknown[:, :-1] |= unknown[:, 1:]
    frontier_mask = (dist >= 0) & adjacent_unknown
    if not np.any(frontier_mask):
        return observation_or_fallback()

    # Cluster frontier cells (8-connected); ignore tiny slivers, which are
    # sensor noise or unfittable cracks rather than rooms to explore.
    cluster_ids = np.zeros((height, width), dtype=np.int32)
    clusters: list[dict[str, object]] = []
    frontier_cells = np.argwhere(frontier_mask)
    for fy, fx in frontier_cells:
        if cluster_ids[fy, fx]:
            continue
        cluster_index = len(clusters) + 1
        members: list[tuple[int, int]] = []
        cluster_queue: deque[tuple[int, int]] = deque([(int(fx), int(fy))])
        cluster_ids[fy, fx] = cluster_index
        while cluster_queue:
            cx, cy = cluster_queue.popleft()
            members.append((cx, cy))
            for nx in (cx - 1, cx, cx + 1):
                for ny in (cy - 1, cy, cy + 1):
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and frontier_mask[ny, nx]
                        and not cluster_ids[ny, nx]
                    ):
                        cluster_ids[ny, nx] = cluster_index
                        cluster_queue.append((nx, ny))
        clusters.append({"members": members})

    best_cell = None
    best_cluster = None
    best_cluster_rank: tuple[int, float] = (2, math.inf)
    for cluster in clusters:
        members = cluster["members"]
        if len(members) < int(min_frontier_cells):
            continue
        nearest_cell = min(members, key=lambda c: dist[c[1], c[0]])
        cluster_centroid = (
            float(np.mean([to_world(cx, cy)[0] for cx, cy in members])),
            float(np.mean([to_world(cx, cy)[1] for cx, cy in members])),
        )
        if is_avoided(cluster_centroid):
            continue
        cluster_cost = float(dist[nearest_cell[1], nearest_cell[0]]) * res + turn_cost_m(
            cluster_centroid
        )
        cluster_preferred = is_preferred(cluster_centroid)
        if cluster_preferred:
            cluster_cost = max(0.0, cluster_cost - float(prefer_bonus_m))
        # Tier 0: real unexplored regions (or the committed frontier from
        # last cycle); tier 1: leftover scraps. A big frontier across the
        # room ALWAYS beats an 8-cell sliver at the robot's feet — scraps
        # are mopped up only once nothing big remains.
        cluster_rank = (
            0
            if cluster_preferred or len(members) >= int(preferred_frontier_cells)
            else 1,
            cluster_cost,
        )
        if cluster_rank < best_cluster_rank:
            best_cluster_rank = cluster_rank
            best_cell = nearest_cell
            best_cluster = cluster
    if best_cell is None:
        return observation_or_fallback()

    members = best_cluster["members"]
    centroid_xy = (
        float(np.mean([to_world(cx, cy)[0] for cx, cy in members])),
        float(np.mean([to_world(cx, cy)[1] for cx, cy in members])),
    )

    # Backtrack the BFS distance field from the chosen frontier cell to the
    # robot: this is an actual traversable path, not a straight-line hope.
    path_cells = backtrack_path(best_cell)

    standoff_cells = max(0, int(round(float(goal_standoff_m) / res)))
    goal_index = max(0, len(path_cells) - 1 - standoff_cells)
    goal_xy = to_world(*path_cells[goal_index])

    waypoint_xy = goal_xy
    for cell in path_cells[: goal_index + 1]:
        wx, wy = to_world(*cell)
        if math.hypot(wx - float(robot_xy[0]), wy - float(robot_xy[1])) >= float(waypoint_lookahead_m):
            waypoint_xy = (wx, wy)
            break

    return {
        "status": "ok",
        "goal_xy": goal_xy,
        "waypoint_xy": waypoint_xy,
        "face_xy": centroid_xy,
        "path_length_m": float(dist[best_cell[1], best_cell[0]]) * res,
        "frontier_cells": int(len(members)),
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
