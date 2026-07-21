"""Sourccey EXPLORE — frontier-based room exploration on the spin-map foundation.

The clean-slate SLAM stack, step 2 (2026-07-21). Step 1 (``sourccey_spin_map.py``)
proved the base capability: one continuous 360deg spin, sequentially scan-matched
into a crisp room map. This script builds the intelligent explorer on top of it,
the way professional 2D-lidar SLAM systems (Hector/Cartographer-class frontier
explorers) do:

    1. SPIN     — one fluid 360, sequential scan-to-map alignment (imported ethos,
                  same parameters; the working spin mapper is untouched).
    2. ANALYZE  — ray-cast every aligned scan through an occupancy grid to label
                  the world OCCUPIED / FREE / UNKNOWN. A FRONTIER is free space
                  touching unknown space: the edge of what the map has actually
                  seen. Frontiers are clustered; only SIGNIFICANT clusters
                  survive (span >= a door-ish width, robot-sized clearance) — no
                  itty-bitty crevices.
    3. PLAN     — pick the best frontier (big and near), pull a viewpoint back
                  from it into known-free space, and A* a collision-free path
                  through the free grid (inflated by the robot radius).
    4. DRIVE    — follow the path in short IMU-yaw-held bursts with a live lidar
                  stop-box. After EVERY burst the robot re-localizes by
                  scan-matching against its own map ("Where am I?" is always
                  answered by the map + gyro, never dead-reckoning alone). A weak
                  match triggers a bounded relocalization wiggle — relocalize,
                  don't guess.
    5. SNAPSHOT — at the viewpoint, grab a quick multi-frame snapshot, match it
                  into the map, merge, re-analyze, and go again — until no
                  significant frontier remains or the viewpoint budget is spent.

Run from the repo root:
    uv run --with rerun-sdk python scripts/sourccey_explore.py --remote-ip 192.168.1.237
"""
from __future__ import annotations

import argparse
import heapq
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ldlidar_auto_snapshot_stitch import _init_rerun, _send_stop
from ldlidar_direct_snapshot_client import DirectLidarFeed
from ldlidar_direct_snapshot_stitch import Pose2D, _search_pose, _transform_points
from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient
from sourccey_arm_pose import DEFAULT_POSE_PATH, apply_pose_blocking, hold_action, limp_action, load_pose
from sourccey_spin_map import (
    _MAP_PALETTE,
    _abort,
    _grid_filtered_points,
    _pick_free_ports,
    _scan_local,
    _spin_command,
)
from sourccey_wander.imu_heading import ImuYawClient


# ---------------------------------------------------------------------------
# World map: every scan ever matched in, with the pose that placed it.
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MapScan:
    local_xy: np.ndarray      # (fwd, lat) points in the lidar frame
    pose: Pose2D              # solved LIDAR pose in the world frame
    world_xy: np.ndarray      # local points transformed by pose


class WorldMap:
    """Every aligned scan (spin + continuous tracking) with the pose that placed
    it, plus cached concatenations for matching and rendering."""

    def __init__(self) -> None:
        self.scans: list[MapScan] = []
        self._ref_cache: np.ndarray | None = None

    def add(self, local_xy: np.ndarray, pose: Pose2D) -> None:
        self.scans.append(
            MapScan(local_xy=local_xy, pose=pose, world_xy=_transform_points(local_xy, pose))
        )
        self._ref_cache = None

    def reference(self, max_pts: int = 9000) -> np.ndarray:
        """Subsampled world points for scan-matching against."""
        if self._ref_cache is None:
            sets = [s.world_xy for s in self.scans if len(s.world_xy)]
            self._ref_cache = (
                np.concatenate(sets, axis=0) if sets else np.zeros((0, 2), dtype=np.float32)
            )
        ref = self._ref_cache
        if len(ref) > max_pts:
            ref = ref[:: (len(ref) // max_pts) + 1]
        return ref

    def render_points(self, grid_res_m: float, grid_min_hits: int) -> tuple[np.ndarray, np.ndarray]:
        """(points, scan_ids) for display AND occupancy analysis.

        Every scan is vote-filtered together: a cell survives only if enough
        scans independently put a point there, so real geometry (walls seen by
        many overlapping scans) stays and per-scan noise/misalignment scatter is
        dropped. This does NOT by itself remove a grossly mis-placed scan (a dense
        lidar line self-corroborates); gross misplacement is instead prevented
        upstream — the tracker only integrates a scan whose match clears the score
        gate under tight windows, so the aliased placements that slanted the old
        map never enter here in the first place."""
        sets = [s.world_xy for s in self.scans if len(s.world_xy)]
        if not sets:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        pts = np.concatenate(sets, axis=0)
        ids = np.concatenate([np.full(len(a), i, dtype=np.int64) for i, a in enumerate(sets)])
        min_hits = grid_min_hits if grid_min_hits > 0 else 3
        keep = _grid_filtered_points(pts, grid_res_m, min_hits)
        return pts[keep], ids[keep]


# ---------------------------------------------------------------------------
# Occupancy analysis: occupied / free / unknown -> frontiers.
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FrontierCluster:
    cells_ij: np.ndarray      # (n, 2) grid cells
    centroid_xy: np.ndarray   # world coords
    span_m: float             # bbox diagonal — "how big is the opening"
    size: int


@dataclass(slots=True)
class Analysis:
    origin_xy: np.ndarray     # world coords of grid cell (0, 0)'s corner
    res_m: float
    occupied: np.ndarray      # (H, W) bool
    free: np.ndarray          # (H, W) bool — ray-cast observed empty space
    traversable: np.ndarray   # (H, W) bool — free minus robot-radius inflation
    clusters: list[FrontierCluster] = field(default_factory=list)

    def to_cell(self, xy) -> tuple[int, int]:
        i = int((float(xy[1]) - self.origin_xy[1]) / self.res_m)
        j = int((float(xy[0]) - self.origin_xy[0]) / self.res_m)
        return i, j

    def to_world(self, ij) -> np.ndarray:
        return np.array([
            self.origin_xy[0] + (ij[1] + 0.5) * self.res_m,
            self.origin_xy[1] + (ij[0] + 0.5) * self.res_m,
        ], dtype=np.float64)


def _disk_offsets(radius_cells: int) -> np.ndarray:
    r = int(radius_cells)
    di, dj = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
    keep = di * di + dj * dj <= r * r
    return np.column_stack([di[keep], dj[keep]])


def _inflate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Binary dilation by a disk, without scipy: shift-OR per disk offset."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    for di, dj in _disk_offsets(radius_cells):
        src_i0, src_i1 = max(0, -di), min(h, h - di)
        src_j0, src_j1 = max(0, -dj), min(w, w - dj)
        if src_i0 >= src_i1 or src_j0 >= src_j1:
            continue
        out[src_i0 + di:src_i1 + di, src_j0 + dj:src_j1 + dj] |= mask[src_i0:src_i1, src_j0:src_j1]
    return out


def analyze_map(
    scan_rays: list[tuple[np.ndarray, np.ndarray]],
    occupied_pts: np.ndarray,
    res_m: float,
    robot_radius_m: float,
    min_frontier_span_m: float,
    min_frontier_cells: int,
) -> Analysis:
    """Classify the world and extract significant frontiers.

    ``scan_rays`` is [(sensor_origin_xy, world_points_xy), ...] — every aligned
    scan with the pose it was taken from. Free space is everything a lidar ray
    PASSED THROUGH (ray-cast, stopping short of the hit so walls stay solid).
    ``occupied_pts`` are the (filtered) map points. A frontier cell is free,
    robot-traversable, and 4-adjacent to unknown — the edge of the map's actual
    knowledge. Clusters smaller than ``min_frontier_span_m``/``min_frontier_cells``
    are discarded as insignificant crevices.
    """
    all_pts = [occupied_pts] + [w for _o, w in scan_rays if len(w)]
    concat = np.concatenate([p for p in all_pts if len(p)], axis=0)
    lo = concat.min(axis=0) - 0.6
    hi = concat.max(axis=0) + 0.6
    origin = lo.astype(np.float64)
    w = int(math.ceil((hi[0] - lo[0]) / res_m)) + 1
    h = int(math.ceil((hi[1] - lo[1]) / res_m)) + 1

    occupied = np.zeros((h, w), dtype=bool)
    if len(occupied_pts):
        jj = ((occupied_pts[:, 0] - origin[0]) / res_m).astype(np.int64)
        ii = ((occupied_pts[:, 1] - origin[1]) / res_m).astype(np.int64)
        ok = (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)
        occupied[ii[ok], jj[ok]] = True

    # Ray-cast free space: walk each ray in res-sized steps, stopping short of
    # the hit (2.5 cells) so the wall itself is never carved free.
    free = np.zeros((h, w), dtype=bool)
    step = res_m
    for origin_xy, world in scan_rays:
        if len(world) == 0:
            continue
        pts = world[:: max(1, len(world) // 160)]           # cap rays per scan
        delta = pts.astype(np.float64) - origin_xy[None, :]
        dist = np.hypot(delta[:, 0], delta[:, 1])
        good = dist > 3.0 * step
        if not np.any(good):
            continue
        delta, dist = delta[good], dist[good]
        unit = delta / dist[:, None]
        n_steps = int(np.ceil(dist.max() / step))
        f = (np.arange(n_steps, dtype=np.float64) + 0.5) * step      # distances along ray
        mask = f[None, :] < (dist[:, None] - 2.5 * step)
        px = origin_xy[0] + unit[:, 0:1] * f[None, :]
        py = origin_xy[1] + unit[:, 1:2] * f[None, :]
        jj = ((px[mask] - origin[0]) / res_m).astype(np.int64)
        ii = ((py[mask] - origin[1]) / res_m).astype(np.int64)
        ok = (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)
        free[ii[ok], jj[ok]] = True
    free &= ~occupied

    inflated = _inflate(occupied, int(math.ceil(robot_radius_m / res_m)))
    traversable = free & ~inflated
    unknown = ~free & ~occupied

    # Frontier: traversable AND 4-adjacent to unknown. Traversability is what
    # kills the wall-hugging slivers — a real opening has robot-sized clearance.
    near_unknown = np.zeros((h, w), dtype=bool)
    near_unknown[1:, :] |= unknown[:-1, :]
    near_unknown[:-1, :] |= unknown[1:, :]
    near_unknown[:, 1:] |= unknown[:, :-1]
    near_unknown[:, :-1] |= unknown[:, 1:]
    frontier = traversable & near_unknown

    analysis = Analysis(origin_xy=origin, res_m=res_m, occupied=occupied,
                        free=free, traversable=traversable)

    # Cluster frontier cells (8-connectivity BFS) and keep the significant ones.
    remaining = frontier.copy()
    fi, fj = np.nonzero(remaining)
    for si, sj in zip(fi.tolist(), fj.tolist()):
        if not remaining[si, sj]:
            continue
        stack = [(si, sj)]
        remaining[si, sj] = False
        cells = []
        while stack:
            ci, cj = stack.pop()
            cells.append((ci, cj))
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < h and 0 <= nj < w and remaining[ni, nj]:
                        remaining[ni, nj] = False
                        stack.append((ni, nj))
        arr = np.array(cells, dtype=np.int64)
        span = float(np.hypot(
            (arr[:, 0].max() - arr[:, 0].min()) * res_m,
            (arr[:, 1].max() - arr[:, 1].min()) * res_m,
        ))
        if len(arr) >= min_frontier_cells and span >= min_frontier_span_m:
            centroid = np.array([
                origin[0] + (arr[:, 1].mean() + 0.5) * res_m,
                origin[1] + (arr[:, 0].mean() + 0.5) * res_m,
            ])
            analysis.clusters.append(
                FrontierCluster(cells_ij=arr, centroid_xy=centroid, span_m=span, size=len(arr))
            )
    analysis.clusters.sort(key=lambda c: -c.span_m)
    return analysis


# ---------------------------------------------------------------------------
# A* path planning on the traversable grid.
# ---------------------------------------------------------------------------

_NEIGHBORS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, 1.41421), (-1, 1, 1.41421), (1, -1, 1.41421), (1, 1, 1.41421)]


def _astar(traversable: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = traversable.shape
    si, sj = start
    gi, gj = goal
    if not (0 <= gi < h and 0 <= gj < w) or not traversable[gi, gj]:
        return None

    def _heur(i: int, j: int) -> float:
        di, dj = abs(i - gi), abs(j - gj)
        return max(di, dj) + 0.41421 * min(di, dj)

    g_cost = np.full((h, w), np.inf)
    g_cost[si, sj] = 0.0
    came: dict[tuple[int, int], tuple[int, int]] = {}
    open_heap: list[tuple[float, int, int]] = [(_heur(si, sj), si, sj)]
    while open_heap:
        _f, ci, cj = heapq.heappop(open_heap)
        if (ci, cj) == (gi, gj):
            path = [(ci, cj)]
            while path[-1] in came:
                path.append(came[path[-1]])
            path.reverse()
            return path
        for di, dj, cost in _NEIGHBORS:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < h and 0 <= nj < w and traversable[ni, nj]:
                ng = g_cost[ci, cj] + cost
                if ng < g_cost[ni, nj]:
                    g_cost[ni, nj] = ng
                    came[(ni, nj)] = (ci, cj)
                    heapq.heappush(open_heap, (ng + _heur(ni, nj), ni, nj))
    return None


def _line_free(traversable: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
    for t in np.linspace(0.0, 1.0, n):
        i = int(round(a[0] + (b[0] - a[0]) * t))
        j = int(round(a[1] + (b[1] - a[1]) * t))
        if not traversable[i, j]:
            return False
    return True


def _simplify_path(traversable: np.ndarray, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Greedy line-of-sight shortcutting: keep only the corners that matter."""
    if len(path) <= 2:
        return path
    out = [path[0]]
    k = 0
    while k < len(path) - 1:
        far = k + 1
        for m in range(len(path) - 1, k, -1):
            if _line_free(traversable, path[k], path[m]):
                far = m
                break
        out.append(path[far])
        k = far
    return out


def _pick_target(
    analysis: Analysis,
    robot_centre_xy: np.ndarray,
    visited_xy: list[np.ndarray],
    pullback_m: float,
    visited_skip_m: float,
) -> tuple[FrontierCluster, np.ndarray, list[np.ndarray]] | None:
    """Best (cluster, viewpoint_xy, waypoints) — biggest opening, discounted by
    path length; skips frontiers already visited (no repeat snapshots)."""
    start = analysis.to_cell(robot_centre_xy)
    h, w = analysis.traversable.shape
    si = min(max(start[0], 0), h - 1)
    sj = min(max(start[1], 0), w - 1)
    if not analysis.traversable[si, sj]:
        # The robot's own cell can sit inside the inflation ring; snap to the
        # nearest traversable cell within half a metre.
        best = None
        for di, dj in _disk_offsets(int(0.5 / analysis.res_m)):
            ni, nj = si + di, sj + dj
            if 0 <= ni < h and 0 <= nj < w and analysis.traversable[ni, nj]:
                d = di * di + dj * dj
                if best is None or d < best[0]:
                    best = (d, ni, nj)
        if best is None:
            return None
        si, sj = best[1], best[2]

    scored: list[tuple[float, FrontierCluster, np.ndarray, list[tuple[int, int]]]] = []
    for cluster in analysis.clusters[:6]:
        if any(np.hypot(*(cluster.centroid_xy - v)) < visited_skip_m for v in visited_xy):
            continue
        to_robot = robot_centre_xy - cluster.centroid_xy
        d = float(np.hypot(*to_robot))
        unit = to_robot / d if d > 1e-6 else np.array([1.0, 0.0])
        path = None
        goal_xy = None
        for pull in (pullback_m, pullback_m + 0.3, pullback_m + 0.6):
            desired = cluster.centroid_xy + unit * pull
            gi, gj = analysis.to_cell(desired)
            found = None
            for di, dj in _disk_offsets(int(0.45 / analysis.res_m)):
                ni, nj = gi + di, gj + dj
                if 0 <= ni < h and 0 <= nj < w and analysis.traversable[ni, nj]:
                    dd = di * di + dj * dj
                    if found is None or dd < found[0]:
                        found = (dd, ni, nj)
            if found is None:
                continue
            path = _astar(analysis.traversable, (si, sj), (found[1], found[2]))
            if path is not None:
                goal_xy = analysis.to_world((found[1], found[2]))
                break
        if path is None or goal_xy is None:
            continue
        score = cluster.span_m / (1.0 + 0.25 * len(path) * analysis.res_m)
        scored.append((score, cluster, goal_xy, path))
    if not scored:
        return None
    scored.sort(key=lambda s: -s[0])
    _sc, cluster, goal_xy, path = scored[0]
    waypoints = [analysis.to_world(ij) for ij in _simplify_path(analysis.traversable, path)[1:]]
    if not waypoints or np.hypot(*(waypoints[-1] - goal_xy)) > 0.05:
        waypoints.append(goal_xy)
    # Thin near-duplicate corners (narrow passages simplify into cm-spaced steps);
    # the final viewpoint always survives.
    thinned: list[np.ndarray] = []
    for wp in waypoints[:-1]:
        if not thinned or np.hypot(*(wp - thinned[-1])) >= 0.30:
            thinned.append(wp)
    thinned = [w for w in thinned if np.hypot(*(w - waypoints[-1])) >= 0.30]
    thinned.append(waypoints[-1])
    return cluster, goal_xy, thinned


# ---------------------------------------------------------------------------
# Localization: scan-match "where am I" with a bounded recovery, never a guess.
# ---------------------------------------------------------------------------

def _localize(
    local_xy: np.ndarray,
    world_map: WorldMap,
    seed: Pose2D,
    args,
    search_xy_m: float,
    theta_window_deg: float,
) -> tuple[Pose2D, float]:
    ref = world_map.reference()
    solved, meta = _search_pose(
        snapshot_points_xy=local_xy,
        global_points_xy=ref,
        initial_pose=seed,
        resolution_m=float(args.stitch_resolution_m),
        search_xy_m=float(search_xy_m),
        coarse_angle_step_deg=2.0,
        fine_angle_step_deg=0.35,
        theta_window_deg=float(theta_window_deg),
        whole_map_theta_center_deg=float(seed.theta_deg),
        whole_map_theta_window_deg=float(theta_window_deg) + 4.0,
        max_translation_from_initial_m=float(search_xy_m) + 0.08,
        prior_pose=seed,
        prior_translation_weight=0.05,
        prior_theta_weight=0.04,
    )
    return solved, float(meta.get("score") or 0.0)


# ---------------------------------------------------------------------------
# Base controller: a background thread streams the current velocity command so
# the base never stalls (it has a command watchdog) while the main loop plans and
# scan-matches at its own slower rate — the standard decoupling of control from
# SLAM that lets the robot drive CONTINUOUSLY instead of in stop-start bursts.
#
# ``hold`` is the arm-stow fragment (12 joint targets + untorque False) merged
# into every command, so the parked arms keep their out-of-the-beam pose while
# the base drives. See sourccey_arm_pose.py.
# ---------------------------------------------------------------------------

class BaseController:
    def __init__(self, robot, hold: dict, rate_hz: float = 25.0) -> None:
        self.robot = robot
        self.hold = dict(hold)
        self.dt = 1.0 / float(rate_hz)
        self._cmd = (0.0, 0.0, 0.0)     # (x.vel, y.vel, theta.vel)
        self._lock = threading.Lock()
        self._run = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._run:
            with self._lock:
                x, y, th = self._cmd
            try:
                self.robot.send_action({
                    "x.vel": x, "y.vel": y, "theta.vel": th,
                    "z.pos": getattr(self.robot, "_z_pos_cmd", 100.0),
                    **self.hold,     # arm stow pose (or legacy untorque) every tick
                })
            except Exception:
                pass
            time.sleep(self.dt)

    def set(self, x_vel: float, y_vel: float, theta_vel: float) -> None:
        with self._lock:
            self._cmd = (float(x_vel), float(y_vel), float(theta_vel))

    def halt(self) -> None:
        self.set(0.0, 0.0, 0.0)

    def shutdown(self) -> None:
        self.halt()
        self._run = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            _send_stop(self.robot)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Physical-forward frame helpers.
# ---------------------------------------------------------------------------

# Physical forward = LiDAR angle 270deg (the WORKING teleop stop box gates real
# x.vel>0 motion with forward_angle_deg=270 — ldlidar_defaults.py). The map, by
# contrast, is extracted with forward_angle_deg=90 + inverted lateral, purely for
# a clean picture. So the map frame and the DRIVE frame differ by a fixed
# transform. We recover it WITHOUT a calibration nudge:
#   - the forward OFFSET is analytic (push lidar-270 through the same extraction);
#   - the turn SIGN (does +theta.vel raise or lower the IMU) comes straight from
#     the 360 anchor spin, which is itself a full rotation calibration.
_PHYS_FORWARD_LIDAR_DEG = 270.0


def _phys_forward_offset_deg(args) -> float:
    """Bearing of the robot's physical forward, expressed in the MAP frame.

    Runs the known physical-forward lidar angle (270) through the exact same
    ``_scan_local`` extraction the map is built with, so the reflection from
    ``--invert-lateral-axis`` is accounted for automatically. For the mapper's
    defaults (forward-angle 90, inverted lateral) this returns 180deg — physical
    forward is directly opposite the map's +x. No robot motion required."""
    delta = ((_PHYS_FORWARD_LIDAR_DEG - float(args.forward_angle_deg) + 180.0) % 360.0) - 180.0
    fwd = math.cos(math.radians(delta))
    lat = math.sin(math.radians(delta))
    if bool(args.invert_lateral_axis):
        lat = -lat
    return math.degrees(math.atan2(lat, fwd))


def _to_forward_frame(pts_local: np.ndarray, forward_offset_deg: float) -> np.ndarray:
    """Rotate local (map-frame) scan points so PHYSICAL forward becomes +x.

    Undoes the map's frame offset, so the collider box can be written in intuitive
    'ahead of the body' coordinates: +x is straight ahead, +/-y is left/right."""
    c = math.cos(math.radians(-forward_offset_deg))
    s = math.sin(math.radians(-forward_offset_deg))
    return np.column_stack([
        pts_local[:, 0] * c - pts_local[:, 1] * s,
        pts_local[:, 0] * s + pts_local[:, 1] * c,
    ])


# ---------------------------------------------------------------------------
# Rerun rendering.
# ---------------------------------------------------------------------------

def _log_world(rr, world_map: WorldMap, args, robot_centre, trail: list[np.ndarray],
               analysis: Analysis | None = None, target_xy=None,
               waypoints: list[np.ndarray] | None = None, note: str = "") -> None:
    pts, ids = world_map.render_points(float(args.grid_res_m), int(args.grid_min_hits))
    if len(pts):
        xyz = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(len(pts), dtype=np.float32)])
        colors = _MAP_PALETTE[ids % len(_MAP_PALETTE)]
        rr.log("world/lidar_map", rr.Points3D(xyz.astype(np.float32), colors=colors, radii=0.02))
    if analysis is not None and analysis.clusters:
        f_all = np.concatenate([c.cells_ij for c in analysis.clusters], axis=0)
        fx = analysis.origin_xy[0] + (f_all[:, 1] + 0.5) * analysis.res_m
        fy = analysis.origin_xy[1] + (f_all[:, 0] + 0.5) * analysis.res_m
        fxyz = np.column_stack([fx, fy, np.zeros(len(f_all))]).astype(np.float32)
        rr.log("world/frontiers", rr.Points3D(fxyz, colors=[[255, 70, 70]] * len(fxyz), radii=0.025))
    else:
        rr.log("world/frontiers", rr.Points3D(np.zeros((0, 3), dtype=np.float32)))
    if target_xy is not None:
        rr.log("world/target", rr.Points3D([[float(target_xy[0]), float(target_xy[1]), 0.0]],
                                           colors=[[80, 255, 120]], radii=0.09))
    else:
        rr.log("world/target", rr.Points3D(np.zeros((0, 3), dtype=np.float32)))
    if waypoints:
        strip = [[float(robot_centre[0]), float(robot_centre[1]), 0.0]] + [
            [float(p[0]), float(p[1]), 0.0] for p in waypoints
        ]
        rr.log("world/path", rr.LineStrips3D([strip], colors=[[255, 220, 90]], radii=0.012))
    else:
        rr.log("world/path", rr.LineStrips3D([]))
    if len(trail) >= 2:
        rr.log("world/trail", rr.LineStrips3D(
            [[[float(p[0]), float(p[1]), 0.0] for p in trail]],
            colors=[[255, 150, 70]], radii=0.008))
    rr.log("world/robot", rr.Points3D([[float(robot_centre[0]), float(robot_centre[1]), 0.0]],
                                      colors=[[255, 130, 60]], radii=0.07))
    if note:
        rr.log("explore/status", rr.TextLog(note))


# ---------------------------------------------------------------------------
# Main mission.
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-ip", required=True)
    parser.add_argument("--robot-id", default="sourccey")
    parser.add_argument("--lidar-host", default=None)
    parser.add_argument("--lidar-port", type=int, default=8765)
    parser.add_argument("--imu-host", default=None)
    parser.add_argument("--imu-yaw-port", type=int, default=8770)
    parser.add_argument("--imu-yaw-sign", type=float, default=1.0)
    # Spin (same proven parameters as sourccey_spin_map).
    parser.add_argument("--spin-degrees", type=float, default=360.0)
    parser.add_argument("--spin-speed", type=float, default=0.9)
    parser.add_argument("--max-spin-seconds", type=float, default=60.0)
    parser.add_argument("--snapshots", type=int, default=240)
    parser.add_argument("--stitch-resolution-m", type=float, default=0.03)
    parser.add_argument("--search-xy-m", type=float, default=0.18)
    parser.add_argument("--theta-window-deg", type=float, default=8.0)
    parser.add_argument("--min-match-score", type=float, default=6.0)
    # Rendering.
    parser.add_argument("--grid-res-m", type=float, default=0.03)
    parser.add_argument("--grid-min-hits", type=int, default=0)
    # LiDAR extraction (same conventions as the spin mapper).
    parser.add_argument("--forward-angle-deg", type=float, default=90.0)
    parser.add_argument("--valid-angle-half-width-deg", type=float, default=180.0)
    parser.add_argument("--invert-lateral-axis", action="store_true", default=True)
    parser.add_argument("--max-distance-m", type=float, default=8.0)
    parser.add_argument("--min-range-m", type=float, default=0.03)
    parser.add_argument("--min-confidence", type=int, default=0)
    parser.add_argument("--lidar-offset-forward-m", type=float, default=0.229)
    # Exploration.
    parser.add_argument("--spin-only", action="store_true", default=False,
                        help="Do the anchor spin, build and render the map, then STOP — no driving. "
                             "Use this to perfect the rotation view in isolation.")
    parser.add_argument("--explore-viewpoints", type=int, default=4,
                        help="Maximum viewpoints to drive to and snapshot.")
    parser.add_argument("--frontier-res-m", type=float, default=0.05,
                        help="Occupancy-analysis grid resolution.")
    parser.add_argument("--min-frontier-span-m", type=float, default=0.55,
                        help="Smallest frontier opening worth visiting (~a doorway); anything "
                             "smaller is an insignificant crevice.")
    parser.add_argument("--min-frontier-cells", type=int, default=10)
    parser.add_argument("--viewpoint-pullback-m", type=float, default=0.70,
                        help="How far back from the frontier the snapshot viewpoint sits "
                             "(inside known-free space, looking out).")
    parser.add_argument("--visited-skip-m", type=float, default=0.55,
                        help="Never re-target a frontier this close to one already snapshotted.")
    parser.add_argument("--robot-radius-m", type=float, default=0.31,
                        help="Planning inflation radius (body half-width 0.28 + margin).")
    # Navigation — CONTINUOUS drive: a background thread streams velocity while the
    # main loop scan-matches every iteration (no bursts). The base drives smoothly
    # to the goal and only stops for a NOVEL obstacle in the forward collider box.
    parser.add_argument("--drive-speed", type=float, default=0.85,
                        help="Forward velocity command while driving (must clear wheel stiction ~0.78).")
    parser.add_argument("--turn-speed", type=float, default=0.9,
                        help="Max rotation command while aiming toward the path.")
    parser.add_argument("--control-rate-hz", type=float, default=25.0,
                        help="Rate the background thread re-sends the velocity command (keeps the base's "
                             "watchdog fed so it drives continuously during a scan-match).")
    parser.add_argument("--track-search-xy-m", type=float, default=0.22,
                        help="Translation search window for each continuous tracking match — covers how "
                             "far the base rolls between matches.")
    parser.add_argument("--track-theta-window-deg", type=float, default=8.0,
                        help="Heading search window for the tracking match (gyro seeds it to ~0.5deg).")
    parser.add_argument("--aim-tolerance-deg", type=float, default=18.0,
                        help="Rotate toward the heading until within this, then drive forward.")
    # Arm stow: hold the arms in a saved out-of-the-beam pose (replaces LiDAR arm
    # masking). Capture the pose with scripts/sourccey_capture_arm_pose.py.
    parser.add_argument("--arm-stow-pose", default=str(DEFAULT_POSE_PATH),
                        help="Path to the saved arm stow pose JSON.")
    parser.add_argument("--arm-settle-s", type=float, default=3.0,
                        help="Seconds to hold the stow command so the arms reach the pose before spinning.")
    parser.add_argument("--no-arm-stow", action="store_true", default=False,
                        help="Skip arm stow entirely and leave the arms untorqued (legacy).")
    # Forward collider box (local costmap): the ONLY thing that stops the drive.
    parser.add_argument("--box-near-m", type=float, default=0.10,
                        help="Near edge of the collider box ahead of the LiDAR.")
    parser.add_argument("--box-depth-m", type=float, default=0.75,
                        help="Far edge of the collider box — how far ahead a novel obstacle is watched for.")
    parser.add_argument("--box-half-width-m", type=float, default=0.34,
                        help="Half-width of the collider box (~body half-width + margin).")
    parser.add_argument("--box-min-points", type=int, default=6,
                        help="Novel returns needed inside the box to call it an obstacle.")
    parser.add_argument("--novel-res-m", type=float, default=0.12,
                        help="A box return is NOVEL if the map has no point within ~this of it.")
    parser.add_argument("--hard-stop-m", type=float, default=0.25,
                        help="Last-resort halt if anything is this close on the nose, novel or not — "
                             "guards against localization error into a mapped wall.")
    parser.add_argument("--obstacle-wait-s", type=float, default=4.0,
                        help="How long to stop and watch a novel obstacle before replanning around it.")
    parser.add_argument("--waypoint-tol-m", type=float, default=0.18)
    parser.add_argument("--max-leg-seconds", type=float, default=35.0,
                        help="Give up on a single waypoint leg after this long (something is wedged).")
    parser.add_argument("--max-mission-seconds", type=float, default=600.0)
    # Panorama + rerun.
    parser.add_argument("--panorama", choices=["on", "off"], default="on")
    parser.add_argument("--slam-input-endpoint", default=None)
    parser.add_argument("--panorama-hz", type=float, default=5.0)
    parser.add_argument("--rerun-mode", choices=["web", "local"], default="web")
    parser.add_argument("--rerun-grpc-port", type=int, default=9877)
    parser.add_argument("--rerun-web-port", type=int, default=9878)
    args = parser.parse_args()

    lidar_host = args.lidar_host or args.remote_ip
    imu_host = args.imu_host or args.remote_ip
    lever_m = float(args.lidar_offset_forward_m)

    # ---- LiDAR feed (no self-mask: the arms are stowed out of the beam instead) ----
    feed = DirectLidarFeed(lidar_host, int(args.lidar_port))
    feed.start()
    print(f"[explore] LiDAR feed connecting to {lidar_host}:{args.lidar_port} ...")
    first_id, first_frame = feed.wait_for_frame_after(after_frame_id=-1, timeout_s=8.0, min_frame_advances=1)
    if first_frame is None:
        _abort("No LiDAR frames within 8s (feed not running?).")
    last_frame_id = int(first_id)

    # ---- IMU (mandatory) ----
    imu = ImuYawClient(f"tcp://{imu_host}:{int(args.imu_yaw_port)}", sign=float(args.imu_yaw_sign))
    imu.start()
    yaw0 = None
    imu_deadline = time.time() + 5.0
    while time.time() < imu_deadline:
        yaw0 = imu.deg_fresh(wait_up_to_s=0.5, max_age_s=0.5)
        if yaw0 is not None:
            break
    if yaw0 is None:
        _abort("No IMU yaw within 5s — navigation is gyro-held, a live IMU is required.")
    print(f"[explore] IMU live; start heading = {yaw0:+.1f}deg.")

    # ---- Robot ----
    robot = SourcceyClient(SourcceyClientConfig(id=args.robot_id, remote_ip=args.remote_ip))
    robot.connect()
    _send_stop(robot)
    print(f"[explore] robot connected ({args.remote_ip}).")

    # ---- Arm stow: move the arms to the saved out-of-the-beam pose and hold it
    # for the whole run (this replaces all the LiDAR arm-masking). `hold` is merged
    # into every base command so the torqued arms keep that pose while driving.
    stow_pose = None if bool(args.no_arm_stow) else load_pose(args.arm_stow_pose)
    if stow_pose is not None:
        print(f"[explore] stowing arms to saved pose ({args.arm_stow_pose}) ...")
        apply_pose_blocking(robot, stow_pose, settle_s=float(args.arm_settle_s))
        hold = hold_action(stow_pose)
        print("[explore] arms stowed and held.")
    else:
        hold = limp_action()
        if bool(args.no_arm_stow):
            print("[explore] --no-arm-stow: arms left untorqued.")
        else:
            print(f"[explore] WARNING: no stow pose at {args.arm_stow_pose} — run "
                  "sourccey_capture_arm_pose.py first. Leaving arms untorqued for now.")

    # ---- Rerun ----
    grpc_port, web_port = _pick_free_ports(int(args.rerun_grpc_port), int(args.rerun_web_port))
    rr, _viewer_url = _init_rerun(session_name="sourccey_explore", mode=args.rerun_mode,
                                  grpc_port=grpc_port, web_port=web_port)

    # ---- Panorama (display only, fail-soft) ----
    cam_sub = None
    eye_mosaic = None
    cam_left_key = cam_right_key = None
    if str(args.panorama) == "on":
        try:
            from sourccey_elevated_safety import SlamCameraSubscriber, endpoint_from_remote_ip
            from sourccey_eye_panorama import load_perception_mosaic
            cam_left_key, cam_right_key = "front_left", "front_right"
            cam_endpoint = str(args.slam_input_endpoint or "").strip() or endpoint_from_remote_ip(args.remote_ip)
            cam_sub = SlamCameraSubscriber(endpoint=cam_endpoint, camera_keys=(cam_left_key, cam_right_key))
            cam_sub.start()
            if not cam_sub.wait_for_frames(timeout_s=5.0, required=(cam_left_key, cam_right_key)):
                raise RuntimeError("no camera frames within 5s")
            eye_mosaic = load_perception_mosaic()
            print("[explore] panorama view ON — rerun entity cameras/panorama")
        except Exception as exc:  # noqa: BLE001
            print(f"[explore] panorama unavailable ({type(exc).__name__}: {exc}); continuing without it.")
            if cam_sub is not None:
                try:
                    cam_sub.stop()
                except Exception:
                    pass
            cam_sub = None
            eye_mosaic = None

    def _log_panorama() -> None:
        if cam_sub is None or eye_mosaic is None:
            return
        try:
            left, _al = cam_sub.latest(cam_left_key)
            right, _ar = cam_sub.latest(cam_right_key)
            if left is None or right is None:
                return
            rr.log("cameras/panorama", rr.Image(eye_mosaic.compose(left, right)[:, :, ::-1]))
        except Exception:
            pass

    world_map = WorldMap()
    mission_t0 = time.monotonic()
    controller = BaseController(robot, hold, rate_hz=float(args.control_rate_hz))
    # Novel obstacles the robot met en route (a person, a moved chair) that persisted;
    # fed into the planner as occupied so replanning routes AROUND them.
    dynamic_occupied: list[np.ndarray] = []

    # ================= PHASE 1: the anchor spin =================
    print(f"\n[explore] PHASE 1 — one continuous {float(args.spin_degrees):.0f}deg anchor spin ...")
    revolutions: list[tuple[np.ndarray, float]] = []
    prev_id = last_frame_id
    pano_interval_s = 1.0 / max(0.5, float(args.panorama_hz))
    last_pano = 0.0
    spin_t0 = time.monotonic()
    try:
        while True:
            robot.send_action({**_spin_command(robot, float(args.spin_speed)), **hold})
            now = time.monotonic()
            if now - last_pano >= pano_interval_s:
                last_pano = now
                _log_panorama()
            frame_id, frame = feed.latest()
            yaw_now = imu.deg()
            if frame is not None and int(frame_id) != prev_id and yaw_now is not None:
                prev_id = int(frame_id)
                local_xy = _scan_local(frame, args)
                if len(local_xy) >= 12:
                    revolutions.append((local_xy, float(yaw_now) - float(yaw0)))
            if yaw_now is not None and abs(float(yaw_now) - float(yaw0)) >= float(args.spin_degrees):
                break
            if time.monotonic() - spin_t0 > float(args.max_spin_seconds):
                print("[explore] WARNING: spin time cap hit before the IMU target.")
                break
            time.sleep(0.03)
    finally:
        # Stop turning but KEEP the arms stowed (a plain _send_stop untorques them).
        for _ in range(3):
            robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0,
                               "z.pos": getattr(robot, "_z_pos_cmd", 100.0), **hold})
            time.sleep(0.04)

    # Sequential scan-to-map alignment (the proven spin-map pipeline).
    stride = max(1, math.ceil(len(revolutions) / max(1, int(args.snapshots))))
    picked = revolutions[::stride]
    print(f"[explore] aligning {len(picked)}/{len(revolutions)} spin scans sequentially ...")
    prev_pose: Pose2D | None = None
    prev_heading: float | None = None
    n_matched = 0
    for local_xy, h in picked:
        if prev_pose is None:
            hr = math.radians(h)
            pose = Pose2D(x=lever_m * math.cos(hr), y=lever_m * math.sin(hr), theta_deg=h)
        else:
            theta_seed = float(prev_pose.theta_deg) + (h - prev_heading)
            pr = math.radians(float(prev_pose.theta_deg))
            cx = float(prev_pose.x) - lever_m * math.cos(pr)
            cy = float(prev_pose.y) - lever_m * math.sin(pr)
            sr = math.radians(theta_seed)
            seed = Pose2D(x=cx + lever_m * math.cos(sr), y=cy + lever_m * math.sin(sr),
                          theta_deg=theta_seed)
            solved, score = _localize(local_xy, world_map, seed, args,
                                      float(args.search_xy_m), float(args.theta_window_deg))
            dtheta = ((float(solved.theta_deg) - theta_seed + 180.0) % 360.0) - 180.0
            jump = math.hypot(float(solved.x) - seed.x, float(solved.y) - seed.y)
            if score >= float(args.min_match_score) and abs(dtheta) <= 12.0 and jump <= 0.35:
                pose = solved
                n_matched += 1
            else:
                pose = seed
        world_map.add(local_xy, pose)
        prev_pose = pose
        prev_heading = h
    print(f"[explore] anchor spin mapped: {len(picked)} scans, {n_matched} matched.")

    # ---- Recover the drive frame from what we already have (NO nudge) ----
    # Forward offset: analytic, from the extraction conventions.
    forward_offset = _phys_forward_offset_deg(args)
    # Turn sign: the anchor spin commanded a positive theta.vel throughout, so the
    # sign of the net IMU sweep tells us whether +theta.vel raises or lowers the
    # gyro. This is the rotation calibration, taken for free from the spin.
    net_spin = revolutions[-1][1] if revolutions else 0.0
    turn_sign = 1.0 if net_spin >= 0.0 else -1.0
    print(f"[explore] drive frame recovered (no nudge): physical forward = map heading "
          f"{forward_offset:+.0f}deg; +theta.vel {'raises' if turn_sign > 0 else 'lowers'} IMU "
          f"(net spin {net_spin:+.0f}deg).")

    # Pose tracking: current LIDAR pose + the IMU reading it was solved at.
    cur_pose: Pose2D = prev_pose if prev_pose is not None else Pose2D(x=lever_m, y=0.0, theta_deg=0.0)

    def _robot_centre(pose: Pose2D) -> np.ndarray:
        tr = math.radians(float(pose.theta_deg))
        return np.array([float(pose.x) - lever_m * math.cos(tr),
                         float(pose.y) - lever_m * math.sin(tr)])

    def _track(prev_imu_deg: float | None) -> tuple[float, np.ndarray]:
        """One continuous scan-to-map tracking step — the core of professional 2D
        SLAM, and the same loop that makes the anchor spin crisp.

        Grab the latest LiDAR revolution, seed its pose from the last solved pose
        rotated by the gyro delta, and match it against the WHOLE accumulated map —
        so drift is bounded by the map, not by frame-to-frame error. On a good
        match the pose is updated AND the scan integrated, growing the map as the
        robot moves. A weak match keeps the gyro dead-reckoned rotation but does
        NOT poison the map with a bad scan. Returns (score, the local scan) so the
        caller can reuse the same scan for the collider box without re-extracting."""
        nonlocal cur_pose
        empty = np.zeros((0, 2), dtype=np.float32)
        _fid, frame = feed.latest()
        if frame is None:
            return 0.0, empty
        local = _scan_local(frame, args)
        if len(local) < 12:
            return 0.0, local
        imu_now = imu.deg()
        gyro_delta = (
            float(imu_now) - float(prev_imu_deg)
            if (imu_now is not None and prev_imu_deg is not None) else 0.0
        )
        theta_seed = float(cur_pose.theta_deg) + gyro_delta
        centre = _robot_centre(cur_pose)
        sr = math.radians(theta_seed)
        seed = Pose2D(x=centre[0] + lever_m * math.cos(sr),
                      y=centre[1] + lever_m * math.sin(sr), theta_deg=theta_seed)
        solved, score = _localize(local, world_map, seed, args,
                                  float(args.track_search_xy_m), float(args.track_theta_window_deg))
        if score >= float(args.min_match_score):
            cur_pose = solved
            world_map.add(local, solved)
        else:
            cur_pose = seed        # trust the gyro rotation; integrate nothing bad
        return score, local

    trail: list[np.ndarray] = [_robot_centre(cur_pose)]
    visited: list[np.ndarray] = []
    viewpoints_reached = 0
    spin_scan_count = len(world_map.scans)
    frontiers_left = 0

    def _analysis_now() -> Analysis:
        pts, _ids = world_map.render_points(float(args.grid_res_m), int(args.grid_min_hits))
        if dynamic_occupied:
            pts = np.concatenate([pts] + dynamic_occupied, axis=0)
        rays = [(np.array([float(s.pose.x), float(s.pose.y)]), s.world_xy)
                for s in world_map.scans]
        return analyze_map(rays, pts, float(args.frontier_res_m), float(args.robot_radius_m),
                           float(args.min_frontier_span_m), int(args.min_frontier_cells))

    def _novel_from(local: np.ndarray) -> tuple[bool, np.ndarray | None]:
        """Given the scan at the CURRENT pose, return whether a NOVEL
        obstacle (not on the map) sits in the forward collider box, and its world
        points. Static mapped walls are excluded — only things the map does not
        already explain (a person, a moved chair) count."""
        if local is None or len(local) == 0:
            return False, None
        ff = _to_forward_frame(local, forward_offset)
        inbox = ((ff[:, 0] > float(args.box_near_m)) & (ff[:, 0] < float(args.box_depth_m))
                 & (np.abs(ff[:, 1]) < float(args.box_half_width_m)))
        box_local = local[inbox]
        if len(box_local) < int(args.box_min_points):
            return False, None
        world = _transform_points(box_local, cur_pose)
        res = float(args.novel_res_m)
        ref = world_map.reference(4000)
        occupied = set()
        if len(ref):
            for cx, cy in np.round(ref / res).astype(np.int64):
                occupied.add((int(cx), int(cy)))
        novel_pts = []
        for wx, wy in world:
            ci, cj = int(round(wx / res)), int(round(wy / res))
            if not any((ci + di, cj + dj) in occupied for di in (-1, 0, 1) for dj in (-1, 0, 1)):
                novel_pts.append((wx, wy))
        if len(novel_pts) < int(args.box_min_points):
            return False, None
        return True, np.array(novel_pts, dtype=np.float32)

    def _handle_obstacle() -> str:
        """A novel obstacle is in the box. Stop and watch: if it clears within the
        wait window, resume; if it persists, add it to the planner's occupancy so
        the next plan routes AROUND it, and ask for a replan."""
        controller.halt()
        print("[explore] novel obstacle in the path — stopping to observe ...")
        deadline = time.monotonic() + float(args.obstacle_wait_s)
        last_novel = None
        while time.monotonic() < deadline:
            _fid, frame = feed.latest()
            if frame is not None:
                local = _scan_local(frame, args)
                is_novel, novel = _novel_from(local)
                if not is_novel:
                    print("[explore] path cleared — resuming.")
                    return "clear"
                last_novel = novel
            time.sleep(0.25)
        if last_novel is not None and len(last_novel):
            dynamic_occupied.append(last_novel)
            print(f"[explore] obstacle persisted; marked {len(last_novel)} pts blocked, replanning around it.")
        return "replan"

    def _drive_leg(waypoints, goal_xy, analysis) -> str:
        """Drive the waypoint chain CONTINUOUSLY (no bursts).

        The background controller streams the velocity command so the base never
        stalls; this loop steers toward the path and scan-matches every iteration
        to keep the pose locked and grow the map. A novel return inside the
        forward collider box (something not on the map) is the ONLY thing that
        stops it — then it observes and either resumes or replans around it.
        Returns '' on arrival, else a failure reason."""
        nonlocal cur_pose
        prev_imu = imu.deg()
        for wp in waypoints:
            leg_t0 = time.monotonic()
            while True:
                if time.monotonic() - mission_t0 > float(args.max_mission_seconds):
                    controller.halt()
                    return "mission time budget spent"
                if time.monotonic() - leg_t0 > float(args.max_leg_seconds):
                    controller.halt()
                    return "leg time budget spent"
                centre = _robot_centre(cur_pose)
                vec = wp - centre
                if float(np.hypot(*vec)) <= float(args.waypoint_tol_m):
                    break
                # Continuous localize + map-grow; reuse its scan for the box check.
                score, local = _track(prev_imu)
                prev_imu = imu.deg()
                # Last-resort collision guard (localization error toward a mapped
                # wall): halt if anything is right on the nose.
                if len(local):
                    ff = _to_forward_frame(local, forward_offset)
                    lane = (ff[:, 0] > 0.02) & (np.abs(ff[:, 1]) < float(args.box_half_width_m))
                    if np.any(lane) and float(ff[lane, 0].min()) < float(args.hard_stop_m):
                        controller.halt()
                        return "hard stop (obstacle on the nose)"
                # Novel obstacle in the box?
                is_novel, _n = _novel_from(local)
                if is_novel:
                    if _handle_obstacle() == "replan":
                        controller.halt()
                        return "obstacle - replan"
                    prev_imu = imu.deg()
                    continue
                # Steer: aim the PHYSICAL front (map heading theta+offset) at the
                # waypoint. err drives theta toward (alpha - offset).
                alpha = math.degrees(math.atan2(vec[1], vec[0]))
                desired = alpha - forward_offset
                err = ((desired - float(cur_pose.theta_deg) + 180.0) % 360.0) - 180.0
                if abs(err) > float(args.aim_tolerance_deg):
                    # Rotate toward the heading (continuous; stiction floor on rate).
                    mag = min(float(args.turn_speed), max(0.80, 0.02 * abs(err)))
                    controller.set(0.0, 0.0, float(turn_sign) * math.copysign(mag, err))
                else:
                    theta_corr = max(-0.35, min(0.35, float(turn_sign) * 0.03 * err))
                    controller.set(float(args.drive_speed), 0.0, theta_corr)
                trail.append(_robot_centre(cur_pose))
                _log_panorama()
                _log_world(rr, world_map, args, trail[-1], trail, analysis, goal_xy, waypoints)
                _ = score
        controller.halt()
        return ""

    # ================= PHASE 2..N: explore the frontiers =================
    aborted = ""
    if bool(args.spin_only):
        print("[explore] --spin-only: skipping exploration; rendering the anchor spin map only.")
    else:
        controller.start()   # background velocity streaming for continuous driving
    while not bool(args.spin_only) and viewpoints_reached < int(args.explore_viewpoints):
        if time.monotonic() - mission_t0 > float(args.max_mission_seconds):
            aborted = "mission time budget spent"
            break
        print("\n[explore] analyzing the map for significant frontiers ...")
        analysis = _analysis_now()
        frontiers_left = len(analysis.clusters)
        centre = _robot_centre(cur_pose)
        picked_target = _pick_target(analysis, centre, visited,
                                     float(args.viewpoint_pullback_m), float(args.visited_skip_m))
        if picked_target is None:
            print("[explore] no significant reachable frontier remains — the map is complete.")
            _log_world(rr, world_map, args, centre, trail, analysis, note="map complete")
            break
        cluster, goal_xy, waypoints = picked_target
        print(f"[explore] target frontier: span {cluster.span_m:.2f}m at "
              f"({cluster.centroid_xy[0]:+.2f}, {cluster.centroid_xy[1]:+.2f}), "
              f"viewpoint ({goal_xy[0]:+.2f}, {goal_xy[1]:+.2f}), {len(waypoints)} waypoint(s).")
        _log_world(rr, world_map, args, centre, trail, analysis, goal_xy, waypoints,
                   note=f"driving to frontier (span {cluster.span_m:.2f}m)")

        leg_failed = _drive_leg(waypoints, goal_xy, analysis)
        if leg_failed in ("mission time budget spent", "hard stop (obstacle on the nose)"):
            aborted = leg_failed
            break
        if leg_failed == "obstacle - replan":
            # The obstacle is now in dynamic_occupied; re-plan WITHOUT marking the
            # frontier visited so the planner routes around it (or gives up if the
            # only path is blocked). Whatever was mapped en route is kept.
            print("[explore] re-planning around the obstacle ...")
            continue
        if leg_failed:
            # Other failure (leg budget): drop this frontier and let the map re-decide.
            print(f"[explore] leg abandoned ({leg_failed}); marking frontier visited and re-planning.")
            visited.append(cluster.centroid_xy)
            continue

        # Arrived. The map was extended continuously during the approach — the
        # frontier is already filled in, so there is nothing to 'snapshot'.
        viewpoints_reached += 1
        visited.append(cluster.centroid_xy)
        print(f"[explore] viewpoint {viewpoints_reached} reached; "
              f"map now {len(world_map.scans)} scans.")
        _log_world(rr, world_map, args, _robot_centre(cur_pose), trail, _analysis_now(),
                   note=f"viewpoint {viewpoints_reached} reached")

    _send_stop(robot)
    final_analysis = _analysis_now()
    _log_world(rr, world_map, args, _robot_centre(cur_pose), trail, final_analysis,
               note="exploration finished")

    print("\n=========== EXPLORE COMPLETE ===========")
    if aborted:
        print(f"  stopped early: {aborted} (map preserved).")
    print(f"  {spin_scan_count} spin scans + {len(world_map.scans) - spin_scan_count} "
          f"tracked-while-driving scans = {len(world_map.scans)} total.")
    print(f"  viewpoints reached: {viewpoints_reached}.")
    print(f"  significant frontiers remaining: {len(final_analysis.clusters)} "
          f"(was {frontiers_left} at first analysis).")
    print("  Map = world/lidar_map; frontiers red, target green, path yellow, trail orange.")
    print("  Leave running to keep the viewer up; Ctrl+C to exit.")
    print("========================================")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for _cleanup in (
            controller.shutdown,
            lambda: _send_stop(robot),
            imu.stop,
            feed.stop,
            (cam_sub.stop if cam_sub is not None else (lambda: None)),
            robot.disconnect,
        ):
            try:
                _cleanup()
            except BaseException:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
