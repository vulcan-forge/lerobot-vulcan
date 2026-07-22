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
    4. DRIVE    — follow the path continuously with IMU yaw hold while a 25Hz
                  independent safety loop checks live lidar + mapped full-body
                  clearance. Moving scans correct the odometry/gyro pose against
                  the map; a weak match triggers bounded stationary recovery —
                  relocalize, don't guess.
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
from sourccey_arm_pose import DEFAULT_POSE_PATH, apply_pose_blocking, hold_action, limp_action, load_pose
from sourccey_spin_map import (
    _MAP_PALETTE,
    _abort,
    _pick_free_ports,
    _scan_local,
    _spin_command,
)
from sourccey_wander.imu_heading import ImuYawClient

from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient


# ``Pose2D.theta_deg`` is the heading of the scan's local +x axis. The robot's
# physical forward direction can differ (it is theta+180 degrees with the
# default reflected extraction), so centre<->LiDAR lever-arm conversions must
# include that offset. Omitting it puts the sensor on the opposite side of the
# rotation centre and invents as much as 2*lever displacement across a turn.
def _lidar_pose_from_robot_centre(
    centre_xy: np.ndarray,
    theta_deg: float,
    lever_m: float,
    forward_offset_deg: float,
) -> Pose2D:
    heading = math.radians(float(theta_deg) + float(forward_offset_deg))
    return Pose2D(
        x=float(centre_xy[0]) + float(lever_m) * math.cos(heading),
        y=float(centre_xy[1]) + float(lever_m) * math.sin(heading),
        theta_deg=float(theta_deg),
    )


def _robot_centre_from_lidar_pose(
    pose: Pose2D,
    lever_m: float,
    forward_offset_deg: float,
) -> np.ndarray:
    heading = math.radians(float(pose.theta_deg) + float(forward_offset_deg))
    return np.array(
        [
            float(pose.x) - float(lever_m) * math.cos(heading),
            float(pose.y) - float(lever_m) * math.sin(heading),
        ],
        dtype=np.float64,
    )


def _rotate_lidar_pose_about_robot_centre(
    pose: Pose2D,
    delta_theta_deg: float,
    lever_m: float,
    forward_offset_deg: float,
) -> Pose2D:
    """Advance gyro heading without inventing robot-centre translation."""
    centre = _robot_centre_from_lidar_pose(pose, lever_m, forward_offset_deg)
    return _lidar_pose_from_robot_centre(
        centre,
        float(pose.theta_deg) + float(delta_theta_deg),
        lever_m,
        forward_offset_deg,
    )


def _imu_yaw_for_scan(imu: ImuYawClient, frame) -> float | None:
    """Yaw at scan completion, falling back for an older untimestamped host."""
    scan_ts = getattr(frame, "revolution_completed_ts", None)
    if scan_ts is None:
        scan_ts = getattr(frame, "host_emitted_ts", None)
    if scan_ts is not None:
        yaw = imu.deg_at_wall_time(float(scan_ts))
        if yaw is not None:
            return yaw
    return imu.deg()


def _budget_exhausted(used: float, limit: float) -> bool:
    """Positive limits are finite; zero/negative limits mean unlimited."""
    return float(limit) > 0.0 and float(used) >= float(limit)


# ---------------------------------------------------------------------------
# LOG-ODDS OCCUPANCY GRID — the professional world model (Cartographer/nav2
# style). One probabilistic grid that the planner, renderer, novelty check and
# frontier analysis ALL read. Every integrated scan updates it symmetrically:
# each return RAISES its cell's log-odds (marking), and every ray LOWERS all the
# cells it passed through (clearing). This one mechanism replaces the vote
# filter, the dynamic-obstacle TTL list, and the ad-hoc unknown-ring fixes:
#   * sparse-but-real edges accumulate evidence and stay occupied (the planner
#     can no longer be blind to something the stop-box sees),
#   * a departed person's cells are cleared by the rays that now pass through
#     where they stood (no timers),
#   * a few misaligned points cannot flip cells that hundreds of rays cleared.
# ---------------------------------------------------------------------------

class OccupancyGrid:
    L_HIT = 0.85          # log-odds added per return in a cell   (p≈0.70)
    L_MISS = 0.40         # log-odds removed per ray through a cell (p≈0.40)
    L_MIN, L_MAX = -4.0, 4.0
    OCC_T = 1.2           # occupied when log-odds >= this (≈2 net hits)
    FREE_T = -0.8         # free when log-odds <= this      (≈2 net clears)

    def __init__(self, res_m: float) -> None:
        self.res = float(res_m)
        self.origin = np.array([-6.0, -6.0], dtype=np.float64)   # cell (0,0) corner
        self.L = np.zeros((int(12.0 / self.res), int(12.0 / self.res)), dtype=np.float32)
        self.version = 0

    def _ensure(self, pts_xy: np.ndarray) -> None:
        """Grow the grid (pad + shift origin) so all points fit with margin."""
        lo = pts_xy.min(axis=0) - 1.0
        hi = pts_xy.max(axis=0) + 1.0
        h, w = self.L.shape
        pad_l = max(0, int(math.ceil((self.origin[0] - lo[0]) / self.res)))
        pad_b = max(0, int(math.ceil((self.origin[1] - lo[1]) / self.res)))
        pad_r = max(0, int(math.ceil((hi[0] - (self.origin[0] + w * self.res)) / self.res)))
        pad_t = max(0, int(math.ceil((hi[1] - (self.origin[1] + h * self.res)) / self.res)))
        if pad_l or pad_b or pad_r or pad_t:
            self.L = np.pad(self.L, ((pad_b, pad_t), (pad_l, pad_r)))
            self.origin = self.origin - np.array([pad_l * self.res, pad_b * self.res])

    def _ij(self, pts_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        jj = ((pts_xy[:, 0] - self.origin[0]) / self.res).astype(np.int64)
        ii = ((pts_xy[:, 1] - self.origin[1]) / self.res).astype(np.int64)
        h, w = self.L.shape
        ok = (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)
        return ii[ok], jj[ok]

    def integrate_scan(self, origin_xy: np.ndarray, world_pts: np.ndarray,
                       footprint_clear_m: float = 0.0) -> None:
        """Marking + clearing for one scan taken from ``origin_xy``."""
        if len(world_pts) == 0:
            return
        self._ensure(np.vstack([world_pts, origin_xy[None, :]]))
        delta = world_pts.astype(np.float64) - origin_xy[None, :]
        dist = np.hypot(delta[:, 0], delta[:, 1])
        good = dist > 3.0 * self.res
        if np.any(good):
            unit = delta[good] / dist[good, None]
            d = dist[good]
            n_steps = int(np.ceil(d.max() / self.res))
            f = (np.arange(n_steps, dtype=np.float64) + 0.5) * self.res
            mask = f[None, :] < (d[:, None] - 1.5 * self.res)   # stop short of the hit
            px = origin_xy[0] + unit[:, 0:1] * f[None, :]
            py = origin_xy[1] + unit[:, 1:2] * f[None, :]
            ii, jj = self._ij(np.column_stack([px[mask], py[mask]]))
            np.subtract.at(self.L, (ii, jj), self.L_MISS)
        ii, jj = self._ij(world_pts)
        np.add.at(self.L, (ii, jj), self.L_HIT)
        if footprint_clear_m > 0.0:
            self.assert_free_disk(origin_xy, footprint_clear_m)
        np.clip(self.L, self.L_MIN, self.L_MAX, out=self.L)
        self.version += 1

    def assert_free_disk(self, centre_xy: np.ndarray, radius_m: float) -> None:
        """The robot's own footprint is free space, always."""
        r = int(math.ceil(radius_m / self.res))
        off = _disk_offsets(r)
        ci = int((centre_xy[1] - self.origin[1]) / self.res)
        cj = int((centre_xy[0] - self.origin[0]) / self.res)
        h, w = self.L.shape
        ii = ci + off[:, 0]
        jj = cj + off[:, 1]
        ok = (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)
        self.L[ii[ok], jj[ok]] = np.minimum(self.L[ii[ok], jj[ok]], self.FREE_T - 0.5)

    def mark_hits(self, world_pts: np.ndarray, amount: float = 2.0) -> None:
        """Raise cells toward occupied (planner-facing marks; clearing rays still
        decay them once the obstacle is gone). Applied ONCE PER CELL — per-point
        accumulation let a dozen offender points in the same cell defeat a
        deliberately-provisional amount."""
        if len(world_pts) == 0:
            return
        self._ensure(world_pts)
        ii, jj = self._ij(world_pts)
        if len(ii) == 0:
            return
        cells = np.unique(np.column_stack([ii, jj]), axis=0)
        self.L[cells[:, 0], cells[:, 1]] += float(amount)
        np.clip(self.L, self.L_MIN, self.L_MAX, out=self.L)
        self.version += 1

    def occupied(self) -> np.ndarray:
        return self.L >= self.OCC_T

    def free(self) -> np.ndarray:
        return self.L <= self.FREE_T


# ---------------------------------------------------------------------------
# World map: scans (for the point-cloud matcher) + the occupancy grid.
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MapScan:
    local_xy: np.ndarray      # (fwd, lat) points in the lidar frame
    pose: Pose2D              # solved LIDAR pose in the world frame
    world_xy: np.ndarray      # local points transformed by pose
    gold: bool                # True = part of the LOCALIZATION reference


class WorldMap:
    """Every aligned scan feeds two consumers — with an integrity firewall.

    The matcher's reference is GOLD-ONLY: the anchor spin plus viewpoint batches
    that passed the consistency gate. Provisional (driving) scans may shape the
    planner's grid, but they can NEVER become geometry that localization aligns
    to — one mis-integrated scan in the reference otherwise seeds the next match,
    and the drift compounds until the map is unrecoverable (field 2026-07-21:
    the hallway painted as three crossing ghost copies, then stationary matches
    collapsed to 3.8 — 'once the map is lost it's game over')."""

    def __init__(self, grid_res_m: float, footprint_clear_m: float, *,
                 lidar_offset_m: float = 0.0, forward_offset_deg: float = 0.0) -> None:
        self.scans: list[MapScan] = []
        self.grid = OccupancyGrid(grid_res_m)
        self.footprint_clear_m = float(footprint_clear_m)
        self.lidar_offset_m = float(lidar_offset_m)
        self.forward_offset_deg = float(forward_offset_deg)
        self._ref_cache: np.ndarray | None = None

    def add(self, local_xy: np.ndarray, pose: Pose2D, *, gold: bool = False) -> None:
        world = _transform_points(local_xy, pose)
        self.scans.append(MapScan(local_xy=local_xy, pose=pose, world_xy=world, gold=gold))
        robot_centre = _robot_centre_from_lidar_pose(
            pose, self.lidar_offset_m, self.forward_offset_deg
        )
        self.grid.integrate_scan(
            np.array([float(pose.x), float(pose.y)]), world,
            footprint_clear_m=0.0,
        )
        self.grid.assert_free_disk(robot_centre, self.footprint_clear_m)
        if gold:
            self._ref_cache = None

    def reference(self, max_pts: int = 9000) -> np.ndarray:
        """Subsampled GOLD world points for scan-matching against."""
        if self._ref_cache is None:
            sets = [s.world_xy for s in self.scans if s.gold and len(s.world_xy)]
            self._ref_cache = (
                np.concatenate(sets, axis=0) if sets else np.zeros((0, 2), dtype=np.float32)
            )
        ref = self._ref_cache
        if len(ref) > max_pts:
            ref = ref[:: (len(ref) // max_pts) + 1]
        return ref

    def render_points(self) -> tuple[np.ndarray, np.ndarray]:
        """(points, scan_ids) for display: the colourful raw points, kept only
        where the occupancy grid says OCCUPIED (grid truth decides what is real;
        the palette look stays)."""
        sets = [s.world_xy for s in self.scans if len(s.world_xy)]
        if not sets:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        pts = np.concatenate(sets, axis=0)
        ids = np.concatenate([np.full(len(a), i, dtype=np.int64) for i, a in enumerate(sets)])
        occ = _inflate(self.grid.occupied(), 1)      # 1-cell tolerance for point jitter
        g = self.grid
        jj = ((pts[:, 0] - g.origin[0]) / g.res).astype(np.int64)
        ii = ((pts[:, 1] - g.origin[1]) / g.res).astype(np.int64)
        h, w = occ.shape
        ok = (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)
        keep = np.zeros(len(pts), dtype=bool)
        keep[ok] = occ[ii[ok], jj[ok]]
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
    frontier_cells: np.ndarray | None = None   # (n,2) ALL raw frontier cells, for display
    cost: np.ndarray | None = None  # (H, W) float — 1.0 in open space, rising near obstacles

    def to_cell(self, xy) -> tuple[int, int]:
        i = int((float(xy[1]) - self.origin_xy[1]) / self.res_m)
        j = int((float(xy[0]) - self.origin_xy[0]) / self.res_m)
        return i, j

    def to_world(self, ij) -> np.ndarray:
        return np.array([
            self.origin_xy[0] + (ij[1] + 0.5) * self.res_m,
            self.origin_xy[1] + (ij[0] + 0.5) * self.res_m,
        ], dtype=np.float64)


def _swept_footprint_state(
    analysis: Analysis,
    robot_centre_xy: np.ndarray,
    physical_heading_deg: float,
    lookahead_m: float,
) -> tuple[str, float]:
    """Check the immediate full-body sweep against the inflated map.

    Sampling future robot-centre positions in ``traversable`` catches mapped
    desk corners beside the shoulders, which a rectangle starting in front of
    the forward-mounted LiDAR cannot see. Unknown is reported separately so the
    explorer can stop and map there instead of calling it a physical obstacle.
    """
    heading = math.radians(float(physical_heading_deg))
    direction = np.array([math.cos(heading), math.sin(heading)], dtype=np.float64)
    step = max(0.02, 0.5 * float(analysis.res_m))
    distances = np.arange(0.0, max(0.0, float(lookahead_m)) + 0.5 * step, step)
    h, w = analysis.traversable.shape
    centre_cell = analysis.to_cell(robot_centre_xy)
    ci, cj = centre_cell
    if ci < 0 or ci >= h or cj < 0 or cj >= w or not analysis.free[ci, cj]:
        return "unknown", 0.0

    # A localization correction can place the centre just inside the inflated
    # band even though the physical robot is not colliding. Rejecting distance
    # zero unconditionally deadlocks every subsequent plan. In that state,
    # permit only headings whose short forward sweep gets closer to ANY
    # traversable cell; motion sideways/deeper into inflation remains blocked.
    if not analysis.traversable[ci, cj]:
        traversable_cells = np.column_stack(np.nonzero(analysis.traversable))
        if not len(traversable_cells):
            return "blocked", 0.0
        current_d2 = np.min(
            (traversable_cells[:, 0] - ci) ** 2 + (traversable_cells[:, 1] - cj) ** 2
        )
        for distance in distances[1:]:
            sample = np.asarray(robot_centre_xy, dtype=np.float64) + direction * float(distance)
            i, j = analysis.to_cell(sample)
            if i < 0 or i >= h or j < 0 or j >= w or not analysis.free[i, j]:
                continue
            d2 = np.min(
                (traversable_cells[:, 0] - i) ** 2 + (traversable_cells[:, 1] - j) ** 2
            )
            if d2 < current_d2:
                return "clear", float(distance)
        return "blocked", 0.0

    for distance in distances[1:]:
        sample = np.asarray(robot_centre_xy, dtype=np.float64) + direction * float(distance)
        i, j = analysis.to_cell(sample)
        if i < 0 or i >= h or j < 0 or j >= w or not analysis.free[i, j]:
            return "unknown", float(distance)
        if not analysis.traversable[i, j]:
            return "blocked", float(distance)
    return "clear", float(lookahead_m)


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


def _inflation_radius_cells(radius_m: float, resolution_m: float) -> int:
    """Convert a metric footprint radius without rounding it up a whole cell.

    Occupancy values describe cell centres.  ``ceil(0.31 / 0.05)`` previously
    inflated the robot by seven cells (0.35m), before even accounting for the
    occupied cell's own area.  That silently closed doorways the 0.56m-wide base
    physically fits through.  Include only cell-centre offsets inside the stated
    metric radius; the occupied cell supplies the remaining half-cell coverage.
    """
    if resolution_m <= 0.0:
        raise ValueError("resolution_m must be positive")
    return max(0, int(math.floor(float(radius_m) / float(resolution_m) + 1e-9)))


def analyze_grid(
    grid: OccupancyGrid,
    robot_radius_m: float,
    min_frontier_span_m: float,
    min_frontier_cells: int,
) -> Analysis:
    """Classify the world FROM the log-odds occupancy grid and extract frontiers.

    No re-raycasting: the grid already holds the fused evidence of every scan
    (marking + clearing), so this is a direct read — occupied / free by log-odds
    thresholds, unknown = insufficient evidence either way. The near-wall
    'unknown ring' hack is gone too: clearing rays run right up to the walls, so
    wall-adjacent cells are genuinely FREE and only real unobserved space (beyond
    openings, in shadows) stays unknown. A frontier cell is free and 4-adjacent
    to unknown; clusters below ``min_frontier_span_m``/``min_frontier_cells`` are
    discarded as insignificant crevices."""
    origin = grid.origin.copy()
    res_m = grid.res
    occupied = grid.occupied()
    free = grid.free() & ~occupied
    h, w = occupied.shape
    unknown = ~free & ~occupied

    inflated = _inflate(occupied, _inflation_radius_cells(robot_radius_m, res_m))
    traversable = free & ~inflated

    # COST GRADIENT (nav2-style inflation layer): cells near obstacles cost more,
    # so the cheapest A* path runs down the MIDDLE of corridors instead of
    # hugging the inflation boundary and cutting every corner as close to walls
    # as legally possible (which is what a binary shortest path does — and why
    # the drive needed constant stop-and-adjust near clutter).
    cost = np.ones(occupied.shape, dtype=np.float32)
    rings = max(1, int(round(0.35 / res_m)))
    cur = inflated
    for k in range(rings):
        nxt = _inflate(cur, 1)
        band = nxt & ~cur
        cost[band] += 3.0 * (rings - k) / rings     # +3x right at the boundary, fading out
        cur = nxt

    # Frontier = FREE space 4-adjacent to unknown (the edge of what was seen). We do
    # NOT require the frontier cell to be robot-traversable — that would erase a
    # doorway once robot-radius inflation narrows it. Reachability is checked later
    # in _pick_target, which A*'s to a viewpoint pulled back into open space.
    near_unknown = np.zeros((h, w), dtype=bool)
    near_unknown[1:, :] |= unknown[:-1, :]
    near_unknown[:-1, :] |= unknown[1:, :]
    near_unknown[:, 1:] |= unknown[:, :-1]
    near_unknown[:, :-1] |= unknown[:, 1:]
    frontier = free & near_unknown

    fcells = np.column_stack(np.nonzero(frontier)).astype(np.int64)
    analysis = Analysis(origin_xy=origin, res_m=res_m, occupied=occupied,
                        free=free, traversable=traversable,
                        frontier_cells=fcells if len(fcells) else None,
                        cost=cost)

    # Cluster frontier cells (8-connectivity BFS) and keep the significant ones.
    remaining = frontier.copy()
    fi, fj = np.nonzero(remaining)
    raw_clusters = 0
    biggest_raw = 0.0
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
        raw_clusters += 1
        biggest_raw = max(biggest_raw, span)
        if len(arr) >= min_frontier_cells and span >= min_frontier_span_m:
            centroid = np.array([
                origin[0] + (arr[:, 1].mean() + 0.5) * res_m,
                origin[1] + (arr[:, 0].mean() + 0.5) * res_m,
            ])
            analysis.clusters.append(
                FrontierCluster(cells_ij=arr, centroid_xy=centroid, span_m=span, size=len(arr))
            )
    analysis.clusters.sort(key=lambda c: -c.span_m)
    print(f"[analyze] free={int(free.sum())} unknown={int(unknown.sum())} "
          f"traversable={int(traversable.sum())} frontier_cells={len(fcells)} | "
          f"raw_clusters={raw_clusters} (biggest {biggest_raw:.2f}m) -> "
          f"significant={len(analysis.clusters)} "
          f"(need span>={min_frontier_span_m:.2f}m & >={min_frontier_cells} cells)")
    return analysis


# ---------------------------------------------------------------------------
# A* path planning on the traversable grid.
# ---------------------------------------------------------------------------

_NEIGHBORS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, 1.41421), (-1, 1, 1.41421), (1, -1, 1.41421), (1, 1, 1.41421)]


def _astar(traversable: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
           cost: np.ndarray | None = None) -> list[tuple[int, int]] | None:
    """A* on the traversable grid; with ``cost`` (>=1 everywhere) the step price is
    scaled by the destination cell's cost, so cheap paths run through open space
    (corridor middles) instead of hugging the inflation boundary. The heuristic
    assumes min cost 1.0, so it stays admissible."""
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
        for di, dj, step in _NEIGHBORS:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < h and 0 <= nj < w and traversable[ni, nj]:
                mult = float(cost[ni, nj]) if cost is not None else 1.0
                ng = g_cost[ci, cj] + step * mult
                if ng < g_cost[ni, nj]:
                    g_cost[ni, nj] = ng
                    came[(ni, nj)] = (ci, cj)
                    heapq.heappush(open_heap, (ng + _heur(ni, nj), ni, nj))
    return None


def _decimate_path(path: list[tuple[int, int]], every: int = 3) -> list[tuple[int, int]]:
    """Thin a dense A* cell path WITHOUT cutting corners: keep every Nth cell plus
    the endpoints. Unlike LOS chord-simplification, this preserves the planner's
    centred contour — chords were re-cutting the corners the cost gradient had
    deliberately rounded, dragging the robot back toward the walls."""
    if len(path) <= 2:
        return path
    out = path[:-1:every]
    if out[0] != path[0]:
        out.insert(0, path[0])
    out.append(path[-1])
    return out


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


def _match_active_frontier(
    clusters: list[FrontierCluster],
    active_xy: np.ndarray | None,
    max_shift_m: float = 1.25,
) -> FrontierCluster | None:
    """Associate a moving frontier cluster with the explorer's active objective."""
    if active_xy is None or not clusters:
        return None
    closest = min(
        clusters,
        key=lambda c: float(np.hypot(*(c.centroid_xy - active_xy))),
    )
    if float(np.hypot(*(closest.centroid_xy - active_xy))) > float(max_shift_m):
        return None
    return closest


def _pick_target(
    analysis: Analysis,
    robot_centre_xy: np.ndarray,
    visited_xy: list[np.ndarray],
    pullback_m: float,
    visited_skip_m: float,
    preferred_frontier_xy: np.ndarray | None = None,
    sampled_viewpoints_xy: list[np.ndarray] | None = None,
) -> tuple[FrontierCluster, np.ndarray, list[np.ndarray], bool, np.ndarray, int] | None:
    """Choose a reachable observation pose with line-of-sight to unknown space.

    Returns ``(cluster, goal, path, close_look, observe_xy, expected_gain)``.
    A frontier centroid is only an association label; it is not necessarily
    visible (the centroid of a curved/elongated boundary can lie behind a wall).
    Candidate poses therefore aim at an actual frontier cell and are ranked by
    nearby unknown cells they can reveal, then by path cost.
    """
    start = analysis.to_cell(robot_centre_xy)
    h, w = analysis.traversable.shape
    si = min(max(start[0], 0), h - 1)
    sj = min(max(start[1], 0), w - 1)
    if not analysis.traversable[si, sj]:
        # The robot's own cell can sit inside the inflation ring (or against a
        # provisional mark after a wobbled stop); search generously — a failed
        # start snap once masqueraded as 'map complete' at 0 viewpoints.
        best = None
        for di, dj in _disk_offsets(int(0.9 / analysis.res_m)):
            ni, nj = si + di, sj + dj
            if 0 <= ni < h and 0 <= nj < w and analysis.traversable[ni, nj]:
                d = di * di + dj * dj
                if best is None or d < best[0]:
                    best = (d, ni, nj)
        if best is None:
            return None
        si, sj = best[1], best[2]

    trav_cells = np.column_stack(np.nonzero(analysis.traversable))
    if len(trav_cells) == 0:
        return None
    trav_xy = np.column_stack([
        analysis.origin_xy[0] + (trav_cells[:, 1] + 0.5) * analysis.res_m,
        analysis.origin_xy[1] + (trav_cells[:, 0] + 0.5) * analysis.res_m,
    ])
    # Consider every significant frontier. Runtime is bounded later by ranking
    # observation poses before A*, not by silently discarding smaller openings.
    candidates = analysis.clusters
    if preferred_frontier_xy is not None:
        preferred = _match_active_frontier(analysis.clusters, preferred_frontier_xy)
        candidates = [preferred] if preferred is not None else []

    unknown = ~analysis.free & ~analysis.occupied
    gain_radius = max(1, int(round(1.0 / analysis.res_m)))
    desired_standoff = float(np.clip(pullback_m, 0.40, 0.85))
    sampled = sampled_viewpoints_xy or []
    geometric: list[tuple[float, FrontierCluster, int, np.ndarray, int]] = []

    for cluster in candidates:
        if any(np.hypot(*(cluster.centroid_xy - v)) < visited_skip_m for v in visited_xy):
            continue
        fcells = cluster.cells_ij
        if len(fcells) > 32:
            fcells = fcells[np.linspace(0, len(fcells) - 1, 32, dtype=np.int64)]
        for fi, fj in fcells:
            observe_xy = analysis.to_world((int(fi), int(fj)))
            i0, i1 = max(0, int(fi) - gain_radius), min(h, int(fi) + gain_radius + 1)
            j0, j1 = max(0, int(fj) - gain_radius), min(w, int(fj) + gain_radius + 1)
            expected_gain = int(unknown[i0:i1, j0:j1].sum())
            d_all = np.hypot(trav_xy[:, 0] - observe_xy[0], trav_xy[:, 1] - observe_xy[1])
            eligible = np.flatnonzero((d_all >= 0.30) & (d_all <= 1.0))
            if not len(eligible):
                continue
            order = eligible[np.argsort(np.abs(d_all[eligible] - desired_standoff))[:10]]
            for idx in order:
                goal_xy = trav_xy[idx].astype(np.float64)
                goal_cell = (int(trav_cells[idx, 0]), int(trav_cells[idx, 1]))
                if not _line_free(analysis.free, goal_cell, (int(fi), int(fj))):
                    continue
                repeat_d = min(
                    (float(np.hypot(*(goal_xy - old))) for old in sampled),
                    default=999.0,
                )
                if repeat_d < max(0.35, float(visited_skip_m)):
                    continue
                # Rank cheaply first; only the strongest candidates pay for A*.
                geometric_score = (
                    float(expected_gain)
                    + 30.0 * float(cluster.span_m)
                    - 25.0 * abs(float(d_all[idx]) - desired_standoff)
                )
                geometric.append((geometric_score, cluster, int(idx), observe_xy, expected_gain))
                break

    best = None  # (utility, cluster, goal_xy, path, observe_xy, gain)
    for _geo, cluster, idx, observe_xy, expected_gain in sorted(
        geometric, key=lambda item: item[0], reverse=True
    )[:48]:
        cell = (int(trav_cells[idx, 0]), int(trav_cells[idx, 1]))
        path = _astar(analysis.traversable, (si, sj), cell, cost=analysis.cost)
        if path is None:
            continue
        travel_m = max(analysis.res_m, (len(path) - 1) * analysis.res_m)
        utility = float(expected_gain) / (1.0 + 0.35 * travel_m)
        if best is None or utility > best[0]:
            best = (
                utility,
                cluster,
                trav_xy[idx].astype(np.float64),
                path,
                np.asarray(observe_xy, dtype=np.float64),
                int(expected_gain),
            )
    if best is None:
        return None
    _utility, cluster, goal_xy, path, observe_xy, expected_gain = best
    waypoints = [analysis.to_world(ij) for ij in _decimate_path(path)[1:]]
    if not waypoints or np.hypot(*(waypoints[-1] - goal_xy)) > 0.05:
        waypoints.append(goal_xy)
    close_look = float(np.hypot(*(goal_xy - observe_xy))) <= 0.9
    return cluster, goal_xy, waypoints, close_look, observe_xy, expected_gain


def _snap_traversable(analysis: Analysis, xy: np.ndarray, r_m: float = 0.9) -> tuple[int, int] | None:
    """Nearest traversable cell to a world point, within ``r_m``."""
    h, w = analysis.traversable.shape
    ci, cj = analysis.to_cell(xy)
    ci = min(max(ci, 0), h - 1)
    cj = min(max(cj, 0), w - 1)
    if analysis.traversable[ci, cj]:
        return ci, cj
    best = None
    for di, dj in _disk_offsets(int(r_m / analysis.res_m)):
        ni, nj = ci + di, cj + dj
        if 0 <= ni < h and 0 <= nj < w and analysis.traversable[ni, nj]:
            d = di * di + dj * dj
            if best is None or d < best[0]:
                best = (d, ni, nj)
    return (best[1], best[2]) if best is not None else None


def _plan_waypoints(analysis: Analysis, start_xy: np.ndarray, goal_xy: np.ndarray) -> list[np.ndarray] | None:
    """A* + simplify + thin from start to goal on the CURRENT grid — used for
    in-leg replans after a blockage was marked, so the fresh detour actually
    routes around what just stopped us."""
    start = _snap_traversable(analysis, start_xy, 0.5)
    goal = _snap_traversable(analysis, goal_xy, 0.35)
    if start is None or goal is None:
        return None
    path = _astar(analysis.traversable, start, goal, cost=analysis.cost)
    if path is None:
        return None
    # LOS-simplified corners only — NO extra thinning: dropping corners without
    # re-checking the shortcut segment let the path clip the inflation ring
    # (caught in sim at 0.18m from a marked table). Pure pursuit is happy with
    # dense corners; every kept segment here is verified traversable.
    waypoints: list[np.ndarray] = []
    start_w = analysis.to_world(start)
    start_cell = analysis.to_cell(start_xy)
    h, w = analysis.traversable.shape
    start_was_snapped = not (
        0 <= start_cell[0] < h
        and 0 <= start_cell[1] < w
        and bool(analysis.traversable[start_cell])
    )
    if start_was_snapped and float(np.hypot(*(start_w - start_xy))) > 0.02:
        # Do not hide the snap from the follower.  The old path dropped this
        # first segment, so A* planned from a fictitious safe cell while the
        # guard evaluated from the real pose and rejected every command.
        waypoints.append(start_w)
    waypoints.extend(analysis.to_world(ij) for ij in _decimate_path(path)[1:])
    goal_w = analysis.to_world(goal)
    if not waypoints or np.hypot(*(waypoints[-1] - goal_w)) > 0.05:
        waypoints.append(goal_w)
    return waypoints


def _carrot_on(polyline: np.ndarray, pos: np.ndarray, lookahead_m: float) -> np.ndarray:
    """Pure-pursuit carrot: the point ``lookahead_m`` AHEAD ALONG THE PATH of the
    robot's closest point on it. Steering at the carrot makes the robot TRACE the
    planned contour (left, forward, right around a table) instead of beelining at
    the next corner from wherever it drifted — beelining is how a planned detour
    still drove the robot straight into the table it detoured around."""
    best = (float("inf"), 0, polyline[0])
    for i in range(len(polyline) - 1):
        a, b = polyline[i], polyline[i + 1]
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 < 1e-12 else float(np.clip(((pos - a) @ ab) / L2, 0.0, 1.0))
        proj = a + t * ab
        d = float(np.hypot(*(pos - proj)))
        if d < best[0]:
            best = (d, i, proj)
    _d, idx, cur = best
    rem = float(lookahead_m)
    while idx < len(polyline) - 1:
        seg_rest = float(np.hypot(*(polyline[idx + 1] - cur)))
        if rem <= seg_rest:
            if seg_rest < 1e-9:
                idx += 1
                cur = polyline[idx] if idx < len(polyline) else polyline[-1]
                continue
            return cur + (polyline[idx + 1] - cur) * (rem / seg_rest)
        rem -= seg_rest
        idx += 1
        cur = polyline[idx]
    return polyline[-1]


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
    coarse_angle_step_deg: float = 2.0,
) -> tuple[Pose2D, float]:
    ref = world_map.reference()
    solved, meta = _search_pose(
        snapshot_points_xy=local_xy,
        global_points_xy=ref,
        initial_pose=seed,
        resolution_m=float(args.stitch_resolution_m),
        search_xy_m=float(search_xy_m),
        coarse_angle_step_deg=float(coarse_angle_step_deg),
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
# Base controller: the FAST inner control loop.
#
# Field 2026-07-21: steering computed in the main loop zigzagged badly — each
# main-loop pass blocks ~0.3s on a scan-match (+ rendering), so heading errors
# were up to ~1s stale while the base executed them at 25Hz; rotating ~50deg/s
# on a 1s-old error overshoots by 25-50deg every decision → bang-bang switchbacks
# that also wrecked the pose. The professional structure separates the loops:
# this thread closes the HEADING loop on the gyro directly — every tick reads
# the IMU fresh and servos theta.vel toward a target IMU heading — while the
# slow SLAM loop only updates the TARGET (which waypoint, what bearing) at its
# own pace. The gyro heading is continuous/unwrapped, so targets carry no wrap
# issues. ``hold`` is the arm-stow fragment merged into every command.
# ---------------------------------------------------------------------------

class BaseController:
    def __init__(self, robot, hold: dict, imu, rate_hz: float = 25.0, *,
                 turn_speed: float = 0.9, hold_gain: float = 4.5,
                 hold_max: float = 0.6) -> None:
        self.robot = robot
        self.hold = dict(hold)
        self.imu = imu
        self.turn_sign = 1.0            # set after the anchor spin measures it
        self.turn_speed = float(turn_speed)
        self.hold_gain = float(hold_gain)
        self.hold_max = float(hold_max)
        self.dt = 1.0 / float(rate_hz)
        self._mode: tuple[str, float, float] = ("halt", 0.0, 0.0)  # (kind, x_vel, target_imu_deg)
        self._lock = threading.Lock()
        self._safety_check = None
        self._safety_latched_reason: str | None = None
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
                kind, x_vel, target = self._mode
                safety_check = self._safety_check
            x = th = 0.0
            if kind != "halt":
                yaw = None
                try:
                    yaw = self.imu.deg()
                except Exception:
                    pass
                if yaw is None:
                    x = th = 0.0          # no gyro this tick -> fail safe, stop
                else:
                    err_deg = target - float(yaw)
                    if kind == "rotate":
                        # Closed-loop pivot: fresh gyro every tick means no
                        # overshoot; stiction floor so it always actually turns.
                        if abs(err_deg) > 2.5:
                            mag = min(self.turn_speed, max(0.80, 0.035 * abs(err_deg)))
                            th = self.turn_sign * math.copysign(mag, err_deg)
                    else:                  # "drive"
                        x = float(x_vel)
                        corr = self.hold_gain * math.radians(err_deg)
                        th = self.turn_sign * max(-self.hold_max, min(self.hold_max, corr))
            if kind == "drive" and x > 0.0 and safety_check is not None:
                reason = None
                try:
                    reason = safety_check()
                except Exception:
                    # A broken safety callback fails closed for forward motion.
                    reason = "safety check failed"
                if reason:
                    x = th = 0.0
                    with self._lock:
                        self._safety_latched_reason = str(reason)
                        self._mode = ("halt", 0.0, 0.0)
            try:
                self.robot.send_action({
                    "x.vel": x, "y.vel": 0.0, "theta.vel": th,
                    "z.pos": getattr(self.robot, "_z_pos_cmd", 100.0),
                    **self.hold,     # arm stow pose every tick
                })
            except Exception:
                pass
            time.sleep(self.dt)

    def rotate_to(self, target_imu_deg: float) -> None:
        with self._lock:
            if self._safety_latched_reason is None:
                self._mode = ("rotate", 0.0, float(target_imu_deg))

    def drive_toward(self, x_vel: float, target_imu_deg: float) -> None:
        with self._lock:
            if self._safety_latched_reason is None:
                self._mode = ("drive", float(x_vel), float(target_imu_deg))

    def set_safety_check(self, check) -> None:
        with self._lock:
            self._safety_check = check

    def clear_safety_latch(self) -> None:
        with self._lock:
            self._safety_latched_reason = None

    def safety_latched_reason(self) -> str | None:
        with self._lock:
            return self._safety_latched_reason

    def halt(self) -> None:
        with self._lock:
            self._mode = ("halt", 0.0, 0.0)

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
# Odometry feed: the PROPAGATE half of the professional pose pipeline
# (propagate by odometry, CORRECT by scan matching). Integrates the host's
# measured body-frame forward velocity (wheel-derived, ~m/s) into cumulative
# forward travel. Without propagation the position estimate freezes between
# matches, every pivot/skid becomes drift the matcher must chase — and can
# alias onto with a HIGH score (field 2026-07-21: the robot bumped the real
# table while its confident estimate put it elsewhere, painting the table into
# the map as a phantom and 'blocking' on it forever).
# ---------------------------------------------------------------------------

class OdometryFeed:
    """Integrates RAW ∫x.vel·dt — unit-agnostic. The metres-per-unit scale is
    applied (and continuously calibrated against the scan matcher) by the caller."""

    def __init__(self, robot, rate_hz: float = 12.0) -> None:
        self.robot = robot
        self.dt = 1.0 / float(rate_hz)
        self._s = 0.0
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
        last_t = time.monotonic()
        while self._run:
            v = 0.0
            try:
                obs = self.robot.get_observation()
                v = float(obs.get("x.vel", 0.0) or 0.0)
            except Exception:
                pass
            now = time.monotonic()
            dt = now - last_t
            last_t = now
            # Deadband kills standstill noise; dt guard skips stalled polls.
            if abs(v) > 0.02 and dt < 1.0:
                with self._lock:
                    self._s += v * dt
            time.sleep(self.dt)

    def take_forward_delta(self) -> float:
        """Forward metres travelled since the last call (signed); resets."""
        with self._lock:
            s = self._s
            self._s = 0.0
        return s

    def shutdown(self) -> None:
        self._run = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


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
    pts, ids = world_map.render_points()
    if len(pts):
        xyz = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(len(pts), dtype=np.float32)])
        colors = _MAP_PALETTE[ids % len(_MAP_PALETTE)]
        rr.log("world/lidar_map", rr.Points3D(xyz.astype(np.float32), colors=colors, radii=0.02))
    # ALL raw frontier candidates dim (so we can see near-misses), significant
    # clusters bright red on top.
    if analysis is not None and analysis.frontier_cells is not None:
        rc = analysis.frontier_cells
        rx = analysis.origin_xy[0] + (rc[:, 1] + 0.5) * analysis.res_m
        ry = analysis.origin_xy[1] + (rc[:, 0] + 0.5) * analysis.res_m
        rxyz = np.column_stack([rx, ry, np.zeros(len(rc))]).astype(np.float32)
        rr.log("world/frontier_candidates",
               rr.Points3D(rxyz, colors=[[130, 60, 60]] * len(rc), radii=0.015))
    else:
        rr.log("world/frontier_candidates", rr.Points3D(np.zeros((0, 3), dtype=np.float32)))
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
    footprint_angles = np.linspace(0.0, 2.0 * math.pi, 49)
    footprint = [[
        float(robot_centre[0]) + float(args.robot_radius_m) * math.cos(float(a)),
        float(robot_centre[1]) + float(args.robot_radius_m) * math.sin(float(a)),
        0.0,
    ] for a in footprint_angles]
    rr.log("world/robot_footprint", rr.LineStrips3D(
        [footprint], colors=[[255, 130, 60]], radii=0.006))
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
    parser.add_argument("--integrate-min-score", type=float, default=10.0,
                        help="A scan is WRITTEN INTO THE MAP only when its match scores at least this "
                             "(pose updates still accept --min-match-score). Marginal matches near "
                             "clutter are fine for localization but painting the map from them layered "
                             "ghost copies of walls at small offsets (field 2026-07-21: doubled map).")
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
    parser.add_argument("--explore-viewpoints", type=int, default=0,
                        help="Maximum viewpoints/approach-steps to drive to and map from; "
                             "0 means unlimited.")
    parser.add_argument("--frontier-res-m", type=float, default=0.05,
                        help="Occupancy-analysis grid resolution.")
    parser.add_argument("--min-frontier-span-m", type=float, default=0.40,
                        help="Smallest frontier opening worth visiting; anything smaller is an "
                             "insignificant crevice. Lowered since robot-radius inflation already "
                             "narrows a real opening before it is measured.")
    parser.add_argument("--min-frontier-cells", type=int, default=6)
    parser.add_argument("--self-clear-m", type=float, default=0.30,
                        help="Radius around the robot's own positions asserted as free space, so the "
                             "no-return dead zone under the robot is not a false frontier.")
    parser.add_argument("--viewpoint-pullback-m", type=float, default=0.70,
                        help="How far back from the frontier the snapshot viewpoint sits "
                             "(inside known-free space, looking out).")
    parser.add_argument("--visited-skip-m", type=float, default=0.55,
                        help="Never re-target a frontier this close to one already snapshotted.")
    parser.add_argument("--robot-radius-m", type=float, default=0.31,
                        help="Planning inflation radius (body half-width 0.28 + margin).")
    # Navigation — continuous gyro-held drive. Localization corrects the pose in
    # the slow loop while an independent 25Hz controller checks live LiDAR and
    # mapped full-body clearance before every forward command.
    parser.add_argument("--drive-speed", type=float, default=0.80,
                        help="Forward velocity command while driving (must clear wheel stiction ~0.78).")
    parser.add_argument("--turn-speed", type=float, default=0.9,
                        help="Max rotation command while aiming toward the path.")
    parser.add_argument("--control-rate-hz", type=float, default=25.0,
                        help="Rate of continuous command streaming and independent safety checks.")
    parser.add_argument("--drive-burst-s", type=float, default=0.28,
                        help="Deprecated compatibility option; continuous drive no longer uses bursts.")
    parser.add_argument("--drive-settle-s", type=float, default=0.12,
                        help="Stationary settling time after each drive burst before accepting a "
                             "fresh LiDAR revolution for localization.")
    parser.add_argument("--track-search-xy-m", type=float, default=0.22,
                        help="Translation search window for each continuous tracking match — covers how "
                             "far the base rolls between matches.")
    parser.add_argument("--odom-scale", type=float, default=0.45,
                        help="INITIAL metres-per-unit scale on the host's reported x.vel. The true "
                             "scale is CALIBRATED ONLINE against the scan matcher (every strong match "
                             "measures how far propagation over/undershot) — field 2026-07-22: treating "
                             "x.vel as m/s (scale 1.0) made the pose race ahead of the robot, 'arrive' "
                             "early, and pin the stationary matches outside the search window.")
    parser.add_argument("--max-speed-mps", type=float, default=0.45,
                        help="Physical top speed of the base. Bounds every tracking/relock search "
                             "window (the truth cannot be farther than v_max * elapsed), which is what "
                             "stops the matcher from 'teleporting' the pose along a wall.")
    parser.add_argument("--track-theta-window-deg", type=float, default=8.0,
                        help="Heading search window for the tracking match (gyro seeds it to ~0.5deg).")
    parser.add_argument("--aim-tolerance-deg", type=float, default=10.0,
                        help="Start driving forward once the heading error is within this.")
    parser.add_argument("--lookahead-m", type=float, default=0.20,
                        help="Pure-pursuit carrot distance along the planned path. Smaller = hugs "
                             "corners tighter; larger = smoother but cuts corners more.")
    parser.add_argument("--drive-exit-tol-deg", type=float, default=18.0,
                        help="While driving, fall back to a full pivot only past this heading error — "
                             "below it the heading servo corrects WHILE ROLLING. Kept tight so the "
                             "base aligns with doorway segments instead of curving into a jamb.")
    parser.add_argument("--heading-hold-gain", type=float, default=4.5,
                        help="Proportional gain (per radian of heading error) holding the drive "
                             "heading. High, to counter the mecanum base's curve-while-straight.")
    parser.add_argument("--heading-hold-max", type=float, default=0.6,
                        help="Cap on the heading-hold theta.vel while driving forward.")
    parser.add_argument("--viewpoint-settle-s", type=float, default=0.5,
                        help="Pause at a reached viewpoint before integrating clean stationary scans.")
    # Arms are QUARANTINED by default (hardware damage 2026-07-21): no torque, no
    # targets, ever — unless --arm-stow explicitly opts in.
    parser.add_argument("--arm-stow", action="store_true", default=False,
                        help="OPT IN to driving the arms to the saved stow pose at startup. Off by "
                             "default after the arm-flip incident; leave off until the arm read/write "
                             "scale is verified on hardware.")
    parser.add_argument("--arm-stow-pose", default=str(DEFAULT_POSE_PATH),
                        help="Path to the saved arm stow pose JSON (used only with --arm-stow).")
    parser.add_argument("--arm-settle-s", type=float, default=3.0,
                        help="Seconds to hold the stow command so the arms reach the pose before spinning.")
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
    parser.add_argument("--hard-stop-half-width-m", type=float, default=0.28,
                        help="Live forward lane half-width. This must cover the measured 0.28m body "
                             "half-width; mapped shoulder clearance is enforced separately using "
                             "the full --robot-radius-m swept footprint.")
    parser.add_argument("--swept-guard-lookahead-m", type=float, default=0.12,
                        help="How far ahead to project the full robot-radius footprint on the map. "
                             "A mapped shoulder collision stops/replans; unknown space stops for a "
                             "new stationary viewpoint before proceeding. Keep this short because "
                             "the robot follows curved paths; 25Hz checks cover stopping latency.")
    parser.add_argument("--obstacle-wait-s", type=float, default=4.0,
                        help="How long to stop and watch a novel obstacle before replanning around it. "
                             "(Persisted obstacles are marked into the occupancy grid; clearing rays "
                             "decay them automatically once they are gone — no TTL needed.)")
    parser.add_argument("--waypoint-tol-m", type=float, default=0.18)
    parser.add_argument("--max-leg-seconds", type=float, default=35.0,
                        help="Give up on a single waypoint leg after this long (something is wedged).")
    parser.add_argument("--max-mission-seconds", type=float, default=0.0,
                        help="Mission time limit in seconds; 0 means unlimited.")
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
    forward_offset = _phys_forward_offset_deg(args)
    # This guard projects the CURRENT heading as a straight line; it is not the
    # footprint size. Beyond 12cm that approximation cuts across curves that the
    # path follower will actually turn through and falsely seals doorways. The
    # independent raw-LiDAR stop lane still watches 25cm ahead at 25Hz.
    swept_guard_m = min(0.12, max(0.0, float(args.swept_guard_lookahead_m)))
    if float(args.swept_guard_lookahead_m) > 0.12:
        print(f"[explore] swept guard request {float(args.swept_guard_lookahead_m):.2f}m "
              f"capped to {swept_guard_m:.2f}m (straight-heading guard; path may curve).")

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

    # ---- Arms: QUARANTINED (2026-07-21 — a commanded "hold current position"
    # physically flipped and cracked the left arm; the read-back scale is not
    # trusted against the write path until verified on the hardware). Default is
    # to NEVER torque or target the arms; --arm-stow is an explicit opt-in.
    stow_pose = load_pose(args.arm_stow_pose) if bool(args.arm_stow) else None
    if stow_pose is not None:
        print(f"[explore] --arm-stow OPT-IN: moving arms to {args.arm_stow_pose} ...")
        apply_pose_blocking(robot, stow_pose, settle_s=float(args.arm_settle_s))
        hold = hold_action(stow_pose)
        print("[explore] arms stowed and held for the run.")
    else:
        hold = limp_action()
        if bool(args.arm_stow):
            print(f"[explore] WARNING: --arm-stow set but no pose at {args.arm_stow_pose}; "
                  "arms left untorqued.")
        else:
            print("[explore] arms left untorqued (arm motion is quarantined; --arm-stow to opt in).")

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
            cam_endpoint = (
                str(args.slam_input_endpoint or "").strip()
                or endpoint_from_remote_ip(args.remote_ip)
            )
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

    # ONE world model: the log-odds occupancy grid inside WorldMap. Persisted
    # obstacles (a lingering person, a hard-stop offender) are simply marked into
    # it; once they leave, the rays that pass through their cells clear them —
    # no TTL lists, no separate dynamic-obstacle bookkeeping.
    world_map = WorldMap(grid_res_m=float(args.frontier_res_m),
                         footprint_clear_m=float(args.self_clear_m),
                         lidar_offset_m=lever_m,
                         forward_offset_deg=forward_offset)
    mission_t0 = time.monotonic()
    controller = BaseController(robot, hold, imu, rate_hz=float(args.control_rate_hz),
                                turn_speed=float(args.turn_speed),
                                hold_gain=float(args.heading_hold_gain),
                                hold_max=float(args.heading_hold_max))
    odom = OdometryFeed(robot)
    odom_scale = [float(args.odom_scale)]   # metres per x.vel-unit; calibrated online

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
                    yaw_scan = _imu_yaw_for_scan(imu, frame)
                    if yaw_scan is not None:
                        revolutions.append((local_xy, float(yaw_scan) - float(yaw0)))
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
            pose = _lidar_pose_from_robot_centre(
                np.zeros(2, dtype=np.float64), h, lever_m, forward_offset
            )
        else:
            theta_seed = float(prev_pose.theta_deg) + (h - prev_heading)
            centre = _robot_centre_from_lidar_pose(prev_pose, lever_m, forward_offset)
            seed = _lidar_pose_from_robot_centre(
                centre, theta_seed, lever_m, forward_offset
            )
            solved, score = _localize(local_xy, world_map, seed, args,
                                      float(args.search_xy_m), float(args.theta_window_deg))
            dtheta = ((float(solved.theta_deg) - theta_seed + 180.0) % 360.0) - 180.0
            jump = math.hypot(float(solved.x) - seed.x, float(solved.y) - seed.y)
            if score >= float(args.min_match_score) and abs(dtheta) <= 12.0 and jump <= 0.35:
                pose = solved
                n_matched += 1
            else:
                pose = seed
        world_map.add(local_xy, pose, gold=True)     # the anchor spin IS the reference
        prev_pose = pose
        prev_heading = h
    print(f"[explore] anchor spin mapped: {len(picked)} scans, {n_matched} matched.")

    # ---- Recover the drive frame from what we already have (NO nudge) ----
    # Forward offset: analytic, from the extraction conventions.
    # Turn sign: the anchor spin commanded a positive theta.vel throughout, so the
    # sign of the net IMU sweep tells us whether +theta.vel raises or lowers the
    # gyro. This is the rotation calibration, taken for free from the spin.
    net_spin = revolutions[-1][1] if revolutions else 0.0
    turn_sign = 1.0 if net_spin >= 0.0 else -1.0
    controller.turn_sign = float(turn_sign)
    print(f"[explore] drive frame recovered (no nudge): physical forward = map heading "
          f"{forward_offset:+.0f}deg; +theta.vel {'raises' if turn_sign > 0 else 'lowers'} IMU "
          f"(net spin {net_spin:+.0f}deg).")

    # Pose tracking: current LIDAR pose + the IMU reading it was solved at.
    cur_pose: Pose2D = (
        prev_pose
        if prev_pose is not None
        else _lidar_pose_from_robot_centre(
            np.zeros(2, dtype=np.float64), 0.0, lever_m, forward_offset
        )
    )
    # The final subsampled anchor scan usually predates the instant the spin
    # controller halted. Carry that remaining gyro rotation into the live pose;
    # otherwise the very first drive leg starts from a stale scan heading.
    yaw_after_spin = imu.deg()
    if prev_heading is not None and yaw_after_spin is not None:
        cur_pose = _rotate_lidar_pose_about_robot_centre(
            cur_pose,
            float(yaw_after_spin) - (float(yaw0) + float(prev_heading)),
            lever_m,
            forward_offset,
        )

    def _robot_centre(pose: Pose2D) -> np.ndarray:
        return _robot_centre_from_lidar_pose(pose, lever_m, forward_offset)

    def _track(prev_imu_deg: float | None, *, integrate: bool = True,
               search_xy_m: float | None = None) -> tuple[float, np.ndarray, float | None]:
        """One scan-to-map tracking step: match the latest revolution against the
        WHOLE map (seeded by the previous solved pose + gyro delta) and update the
        pose. ``integrate`` controls whether the scan is ALSO added to the map.

        While DRIVING we localize with integrate=False: a revolution captured in
        motion is smeared (the LiDAR sweeps as the base moves), so folding it into
        the map smudges walls and fills in doorways (they then read as walls). We
        only add scans when stopped (at a viewpoint), where they are clean. Returns
        (score, local scan, IMU pose anchor) so the caller can reuse the scan
        and preserve all yaw accumulated while the matcher was running."""
        nonlocal cur_pose
        empty = np.zeros((0, 2), dtype=np.float32)
        _fid, frame = feed.latest()
        if frame is None:
            return 0.0, empty, prev_imu_deg
        local = _scan_local(frame, args)
        if len(local) < 12:
            return 0.0, local, prev_imu_deg
        # Associate heading with the revolution, not with when this Python
        # thread happened to consume it. Network and matcher latency otherwise
        # turn directly into a heading error while the robot is moving.
        imu_scan = _imu_yaw_for_scan(imu, frame)
        gyro_delta = (
            float(imu_scan) - float(prev_imu_deg)
            if (imu_scan is not None and prev_imu_deg is not None) else 0.0
        )
        theta_seed = float(cur_pose.theta_deg) + gyro_delta
        centre = _robot_centre(cur_pose)
        seed = _lidar_pose_from_robot_centre(
            centre, theta_seed, lever_m, forward_offset
        )
        window = float(search_xy_m) if search_xy_m is not None else float(args.track_search_xy_m)
        # THRESHOLD-SAFE matching: at a doorway much of the scan looks into
        # unmapped space and matches nothing, cratering the score even when the
        # pose is perfect (field: 15.5 -> 4.3 arriving centred at the exit ->
        # "tracking lost" -> the robot turned away from the opening it came for).
        # Solve on the map-SUPPORTED subset; the unsupported remainder is new
        # territory — integrated below so the map grows through the opening.
        sup = _match_support_mask(_transform_points(local, seed))
        match_local = local[sup] if int(sup.sum()) >= 30 else local
        solved, score = _localize(match_local, world_map, seed, args,
                                  window, float(args.track_theta_window_deg))
        if score >= float(args.min_match_score):
            if integrate:
                cur_pose = solved          # stationary: full re-anchor (clean scan)
                if score >= float(args.integrate_min_score):
                    world_map.add(local, solved)
            else:
                # IN MOTION: match owns POSITION, gyro owns ROTATION. A moving
                # scan is smeared, so its solved theta wobbles a few degrees per
                # match — feeding that into the pose made the heading target
                # noisy and the servo faithfully chased the noise (the residual
                # squiggle). The gyro's relative yaw is ~0.5deg-accurate over a
                # whole leg, so keep the gyro-propagated theta and re-anchor
                # rotation only from clean stationary scans at viewpoints.
                cur_pose = Pose2D(x=float(solved.x), y=float(solved.y), theta_deg=theta_seed)
        else:
            cur_pose = seed        # trust the gyro rotation; integrate nothing bad

        # _search_pose can hold the GIL for hundreds of milliseconds while the
        # controller keeps steering. The old caller sampled a new IMU baseline
        # after the solve but left cur_pose at the pre-solve heading, silently
        # dropping all rotation during every match. Carry that yaw through now,
        # rotating the offset LiDAR around the (stationary-for-this-correction)
        # robot centre rather than moving the centre itself.
        imu_after = imu.deg()
        if imu_after is not None and imu_scan is not None:
            cur_pose = _rotate_lidar_pose_about_robot_centre(
                cur_pose,
                float(imu_after) - float(imu_scan),
                lever_m,
                forward_offset,
            )
            pose_imu_anchor = float(imu_after)
        else:
            pose_imu_anchor = imu_scan if imu_scan is not None else prev_imu_deg
        return score, local, pose_imu_anchor

    def _integrate_at_viewpoint() -> int:
        """Stop, settle, and map the new area with CLEAN stationary scans —
        committed as a BATCH only if the batch is self-consistent.

        The robot is standing still, so all 8 scans MUST solve to (nearly) the
        same pose. If the solutions scatter, the matcher is aliasing — however
        high its scores claim to be — and committing would paint ghost copies
        (field 2026-07-21: a 'match 15.6' batch painted the hallway three times
        at crossing angles and the map was lost). Scattered batches are DISCARDED
        whole; consistent ones are promoted to GOLD (the localization reference),
        which is how the reference legitimately grows into new rooms."""
        nonlocal cur_pose
        controller.halt()
        time.sleep(float(args.viewpoint_settle_s))
        known_before = world_map.grid.occupied() | world_map.grid.free()
        origin_before = world_map.grid.origin.copy()
        after = feed.latest()[0]
        batch: list[tuple[np.ndarray, Pose2D, float]] = []
        last_score = 0.0
        for _ in range(8):
            fid, frame = feed.wait_for_frame_after(after_frame_id=after, timeout_s=1.0,
                                                   min_frame_advances=1)
            if frame is None:
                continue
            after = int(fid)
            local = _scan_local(frame, args)
            if len(local) < 12:
                continue
            sup_v = _match_support_mask(_transform_points(local, cur_pose))
            match_v = local[sup_v] if int(sup_v.sum()) >= 30 else local
            solved, last_score = _localize(match_v, world_map, cur_pose, args,
                                           max(0.40, float(args.track_search_xy_m)), 15.0)
            if last_score >= float(args.min_match_score):
                batch.append((local, solved, last_score))
        added = 0
        if len(batch) >= 4:
            px = np.array([float(p.x) for _l, p, _s in batch])
            py = np.array([float(p.y) for _l, p, _s in batch])
            pt = np.array([float(p.theta_deg) for _l, p, _s in batch])
            mx, my, mt = np.median(px), np.median(py), np.median(pt)
            scatter = float(np.max(np.hypot(px - mx, py - my)))
            th_scatter = float(np.max(np.abs(((pt - mt + 180.0) % 360.0) - 180.0)))
            mean_sc = float(np.mean([s for _l, _p, s in batch]))
            if scatter <= 0.06 and th_scatter <= 4.0 and mean_sc >= float(args.integrate_min_score):
                # Consistent: commit the whole batch as GOLD; anchor the pose to
                # the batch member nearest the median.
                k = int(np.argmin(np.hypot(px - mx, py - my)))
                cur_pose = batch[k][1]
                for local_b, pose_b, _s in batch:
                    world_map.add(local_b, pose_b, gold=True)
                added = len(batch)
            else:
                print(f"[explore]   viewpoint batch DISCARDED — stationary solutions scatter "
                      f"{scatter * 100:.0f}cm/{th_scatter:.1f}deg, mean score {mean_sc:.1f}: "
                      "aliasing suspected; map protected.")
        if added == 0 and len(batch) < 4:
            # Too few even matched: stationary bounded relocalization vs GOLD —
            # POSE ONLY, nothing is written to the map from a recovery.
            print(f"[explore]   viewpoint match weak ({last_score:.1f}) — stationary relocalization ...")
            fid2, frame2 = feed.wait_for_frame_after(after_frame_id=after, timeout_s=1.5,
                                                     min_frame_advances=1)
            if frame2 is not None:
                local2 = _scan_local(frame2, args)
                if len(local2) >= 12:
                    sup2 = _match_support_mask(_transform_points(local2, cur_pose))
                    match2 = local2[sup2] if int(sup2.sum()) >= 30 else local2
                    solved2, sc2 = _localize(match2, world_map, cur_pose, args, 0.60, 12.0)
                    if sc2 >= float(args.min_match_score):
                        cur_pose = solved2
                        last_score = sc2
        known_after = world_map.grid.occupied() | world_map.grid.free()
        if (
            known_after.shape == known_before.shape
            and np.allclose(world_map.grid.origin, origin_before)
        ):
            new_known = int(np.count_nonzero(known_after & ~known_before))
        else:
            # Grid growth is rare in a room-scale mission. Net known cells is a
            # conservative fallback when padding changes array coordinates.
            new_known = max(0, int(known_after.sum()) - int(known_before.sum()))
        odom.take_forward_delta()      # absolute fixes above: discard pending odometry
        print(f"[explore]   mapped viewpoint: +{added} scans, {new_known} newly-known cells "
              f"(match {last_score:.1f})")
        vp_last[0] = added
        vp_gain_last[0] = new_known
        return added

    def _recovery_spin() -> bool:
        """Rebuild the pose from a full 360° panorama — the strongest evidence we
        can gather. Single-scan relocks alias exactly when the pose is wrong in
        position AND rotation (field 2026-07-22: robot at the wall, estimate
        mid-room, scores plateaued 9-11 in a wrong-but-consistent basin). A
        complete circular signature pins both at once: spin in place, place every
        scan by its gyro heading around the spin centre, and match that panorama
        rigidly against the GOLD map. Returns True if the pose was re-anchored."""
        nonlocal cur_pose
        controller.halt()
        time.sleep(0.3)
        yaw_start = imu.deg()
        if yaw_start is None:
            return False
        print("[explore] RECOVERY SPIN — rebuilding the pose from a full panorama ...")
        theta_start = float(cur_pose.theta_deg)      # map heading at yaw_start (gyro-carried)
        pano: list[np.ndarray] = []
        prev_id = feed.latest()[0]
        t0 = time.monotonic()
        controller.rotate_to(float(yaw_start) + 370.0)
        try:
            while True:
                yaw_now = imu.deg()
                fid, frame = feed.latest()
                if frame is not None and int(fid) != prev_id and yaw_now is not None:
                    prev_id = int(fid)
                    local = _scan_local(frame, args)
                    if len(local) >= 12:
                        yaw_scan = _imu_yaw_for_scan(imu, frame)
                        if yaw_scan is not None:
                            h = float(yaw_scan) - float(yaw_start)
                            p = _lidar_pose_from_robot_centre(
                                np.zeros(2, dtype=np.float64),
                                h,
                                lever_m,
                                forward_offset,
                            )
                            pano.append(_transform_points(local, p))
                if yaw_now is not None and abs(float(yaw_now) - float(yaw_start)) >= 360.0:
                    break
                if time.monotonic() - t0 > 45.0:
                    break
                time.sleep(0.03)
        finally:
            controller.halt()
        time.sleep(0.3)
        if len(pano) < 30:
            print("[explore]   recovery spin gathered too few scans — failed.")
            return False
        pano_pts = np.concatenate(pano, axis=0)
        # A rigid panorama has massive redundant wall samples. Keeping ~1200
        # evenly distributed points preserves its circular signature while
        # bounding recovery search work and memory.
        if len(pano_pts) > 1200:
            pano_pts = pano_pts[:: len(pano_pts) // 1200 + 1]
        centre = _robot_centre(cur_pose)
        seed = Pose2D(x=float(centre[0]), y=float(centre[1]), theta_deg=theta_start)
        solved, score = _localize(
            pano_pts, world_map, seed, args, 1.20, 25.0, coarse_angle_step_deg=4.0
        )
        if score < 12.0:
            print(f"[explore]   panorama match too weak ({score:.1f}) — recovery failed.")
            return False
        # Panorama frame origin = the SPIN CENTRE; convert back to the lidar pose
        # at the robot's current heading (gyro delta accumulated over the spin).
        yaw_end = imu.deg()
        d_h = (float(yaw_end) - float(yaw_start)) if yaw_end is not None else 360.0
        theta_new = float(solved.theta_deg) + d_h
        cur_pose = _lidar_pose_from_robot_centre(
            np.array([float(solved.x), float(solved.y)], dtype=np.float64),
            theta_new,
            lever_m,
            forward_offset,
        )
        odom.take_forward_delta()
        print(f"[explore]   pose RE-ANCHORED by panorama (match {score:.1f}).")
        return True

    trail: list[np.ndarray] = [_robot_centre(cur_pose)]
    visited: list[np.ndarray] = []
    sampled_viewpoints: list[np.ndarray] = []
    # The global planner owns exactly one frontier until it is observed away,
    # proven unreachable/blocked, or the mission is stopped. Without this state,
    # every localization hiccup re-ran global scoring and could send the robot
    # straight back to a frontier behind it while claiming to retry the same one.
    active_frontier_xy: np.ndarray | None = None
    obstacle_strikes: dict[tuple[int, int], int] = {}   # frontier -> times obstacle-blocked
    lost_strikes: dict[tuple[int, int], int] = {}       # frontier -> times tracking-lost
    viewpoints_reached = 0
    spin_scan_count = len(world_map.scans)
    frontiers_left = 0
    none_retries = 0
    vp_last: list = [None]        # last _integrate_at_viewpoint result
    vp_gain_last: list[int | None] = [None]  # actual unknown -> known cells
    consistency_fails = 0         # consecutive discarded/empty viewpoints

    def _analysis_now() -> Analysis:
        return analyze_grid(world_map.grid, float(args.robot_radius_m),
                            float(args.min_frontier_span_m), int(args.min_frontier_cells))

    # --- TWO support masks, deliberately separate (field 2026-07-21): -------
    # MATCHING support comes ONLY from actually-integrated scan points, NEVER
    # from planner marks. When match-support read the grid, hard-stop marks
    # (including arm self-returns) became geometry the matcher aligned to: poses
    # warped onto phantoms, viewpoints painted real walls at aliased offsets
    # (the fake corridor), marks snowballed (the phantom barrier; traversable
    # 1058->570). Localization must not trust anything it did not observe.
    _match_sup_cache: list = [None, None]   # [n scans, cell set]

    def _match_support_mask(world_pts: np.ndarray) -> np.ndarray:
        res = float(args.novel_res_m)
        key = len(world_map.scans)
        if _match_sup_cache[0] != key:
            occ_set = set()
            ref = world_map.reference(60000)
            if len(ref):
                for cx, cy in np.round(np.asarray(ref, dtype=np.float64) / res).astype(np.int64):
                    occ_set.add((int(cx), int(cy)))
            _match_sup_cache[0] = key
            _match_sup_cache[1] = occ_set
        occ_set = _match_sup_cache[1]
        out = np.empty(len(world_pts), dtype=bool)
        for i in range(len(world_pts)):
            ci = int(round(float(world_pts[i, 0]) / res))
            cj = int(round(float(world_pts[i, 1]) / res))
            out[i] = any((ci + di, cj + dj) in occ_set
                         for di in (-1, 0, 1) for dj in (-1, 0, 1))
        return out

    _support_cache: list = [None, None]     # [grid version, inflated-occupied mask]

    def _support_mask(world_pts: np.ndarray) -> np.ndarray:
        """NOVELTY support (grid-based, marks included — a marked obstacle must
        not read as 'novel' again): which world points are near an OCCUPIED cell
        of the occupancy grid (~0.30m reach). Used ONLY by _novel_from; the
        matcher uses _match_support_mask (observed points only)."""
        g = world_map.grid
        if _support_cache[0] != g.version:
            _support_cache[0] = g.version
            _support_cache[1] = _inflate(g.occupied(), max(1, int(round(0.30 / g.res))))
        sup = _support_cache[1]
        h, w = sup.shape
        jj = ((world_pts[:, 0] - g.origin[0]) / g.res).astype(np.int64)
        ii = ((world_pts[:, 1] - g.origin[1]) / g.res).astype(np.int64)
        ok = (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)
        out = np.zeros(len(world_pts), dtype=bool)
        out[ok] = sup[ii[ok], jj[ok]]
        return out

    def _novel_from(local: np.ndarray) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        """Given the scan at the CURRENT pose, return (is_novel, world_pts,
        body_frame_pts) for any NOVEL obstacle in the forward collider box.

        'Novel' = not explained by the map NOR by already-flagged dynamic
        obstacles — the latter is essential: without it, a persisted obstacle
        re-triggered as 'novel' on every subsequent leg forever (field
        2026-07-21: the same ~32 points marked blocked ~30 times, robot never
        moved). body_frame_pts (physical-forward frame) are returned for the
        self-return diagnostic: something novel at the SAME body position at
        every heading is attached to the robot, not in the room."""
        if local is None or len(local) == 0:
            return False, None, None
        ff = _to_forward_frame(local, forward_offset)
        inbox = ((ff[:, 0] > float(args.box_near_m)) & (ff[:, 0] < float(args.box_depth_m))
                 & (np.abs(ff[:, 1]) < float(args.box_half_width_m)))
        box_local = local[inbox]
        if len(box_local) < int(args.box_min_points):
            return False, None, None
        world = _transform_points(box_local, cur_pose)
        novel_mask = ~_support_mask(world)
        if int(novel_mask.sum()) < int(args.box_min_points):
            return False, None, None
        return True, world[novel_mask].astype(np.float32), ff[inbox][novel_mask]

    def _handle_obstacle() -> str:
        """A novel obstacle is in the box. Stop and watch: if it clears within the
        wait window, resume; if it persists, add it to the planner's occupancy so
        the next plan routes AROUND it, and ask for a replan. Prints the obstacle's
        BODY-FRAME location — the self-return discriminator: an 'obstacle' at the
        same body position at every heading is attached to the robot (arm in the
        scan plane), not something in the room."""
        controller.halt()
        print("[explore] novel obstacle in the path — stopping to observe ...")
        deadline = time.monotonic() + float(args.obstacle_wait_s)
        last_novel = None
        last_body = None
        while time.monotonic() < deadline:
            _fid, frame = feed.latest()
            if frame is not None:
                local = _scan_local(frame, args)
                is_novel, novel, body = _novel_from(local)
                if not is_novel:
                    print("[explore] path cleared — resuming.")
                    return "clear"
                last_novel, last_body = novel, body
            time.sleep(0.25)
        if last_novel is not None and len(last_novel):
            # Mark into THE grid: the planner avoids it now, and if the obstacle
            # leaves, later rays through those cells clear it automatically.
            world_map.grid.mark_hits(last_novel)
            fwd = float(np.mean(last_body[:, 0]))
            lat = float(np.mean(last_body[:, 1]))
            rng = float(np.min(np.hypot(last_body[:, 0], last_body[:, 1])))
            print(f"[explore] obstacle persisted; marked {len(last_novel)} pts blocked "
                  f"(BODY frame: fwd {fwd:+.2f}m lat {lat:+.2f}m nearest {rng:.2f}m — if these "
                  "numbers repeat at every heading, it is attached to the robot: check the arm "
                  "stow height vs the 0.28m scan plane). Replanning around it.")
            xyz = np.column_stack([last_novel[:, 0], last_novel[:, 1], np.zeros(len(last_novel))])
            rr.log("world/obstacle", rr.Points3D(xyz.astype(np.float32),
                                                 colors=[[255, 255, 255]] * len(xyz), radii=0.03))
        return "replan"

    def _drive_leg(waypoints, goal_xy, analysis, face_xy=None) -> str:
        """Drive the leg CONTINUOUSLY along the planned path (pure pursuit).

        Steering aims at a CARROT that slides along the planned polyline
        ~--lookahead-m ahead of the robot's closest point on it, so the robot
        TRACES the planned contour (left, forward, right around a table) instead
        of beelining at the next corner from wherever it drifted — beelining is
        how a planned detour still drove it straight into the table (field
        2026-07-21). Forward motion is continuous, but the controller independently
        checks live LiDAR and mapped full-body clearance at 25Hz; a stop is latched
        until the mission loop explicitly replans. Returns '' on arrival."""
        nonlocal cur_pose

        def _fast_safety_check() -> str | None:
            pose = cur_pose
            centre = _robot_centre(pose)
            state, distance = _swept_footprint_state(
                analysis,
                centre,
                float(pose.theta_deg) + forward_offset,
                swept_guard_m,
            )
            if state == "unknown":
                return f"map edge {distance:.2f}m ahead"
            if state == "blocked":
                return f"mapped full-body clearance {distance:.2f}m ahead"
            _fid, frame = feed.latest()
            if frame is None:
                return None
            local = _scan_local(frame, args)
            if not len(local):
                return None
            ff = _to_forward_frame(local, forward_offset)
            lane = ((ff[:, 0] > 0.02)
                    & (ff[:, 0] < float(args.hard_stop_m))
                    & (np.abs(ff[:, 1]) < float(args.hard_stop_half_width_m)))
            if np.any(lane):
                lane_idx = np.flatnonzero(lane)
                hit = int(lane_idx[np.argmin(ff[lane_idx, 0])])
                return (f"live forward clearance x={ff[hit, 0]:.2f}m "
                        f"y={ff[hit, 1]:+.2f}m")
            return None

        controller.set_safety_check(_fast_safety_check)
        controller.clear_safety_latch()
        prev_imu = imu.deg()
        last_render = 0.0
        last_sent: tuple[bool, float] | None = None    # (driving, target_imu)
        # Tracking-health state. The search window is bounded by PHYSICS: the base
        # cannot exceed --max-speed-mps, so the true position always lies within
        # v_max * (time since the last good fix).
        v_max = float(args.max_speed_mps)
        last_good_t = time.monotonic()
        n_weak = 0
        last_diag = 0.0
        novel_hist: list[tuple[np.ndarray, np.ndarray]] = []   # (obstacle world centroid, robot centre)
        phantom_warned = False
        spin_tried = False           # one recovery spin allowed after actual match failure
        polyline = np.vstack([_robot_centre(cur_pose)[None, :],
                              np.asarray(waypoints, dtype=np.float64)])
        goal = np.asarray(goal_xy, dtype=np.float64)
        leg_t0 = time.monotonic()
        driving = False
        replans = 0

        def _replan_same_frontier(reason: str) -> bool:
            nonlocal analysis, waypoints, polyline, driving, last_sent, n_weak, last_good_t, replans
            controller.halt()
            controller.clear_safety_latch()
            replans += 1
            refreshed = _analysis_now()
            new_waypoints = _plan_waypoints(refreshed, _robot_centre(cur_pose), goal)
            if new_waypoints is None or replans > 3:
                return False
            analysis = refreshed
            waypoints = new_waypoints
            polyline = np.vstack([
                _robot_centre(cur_pose)[None, :],
                np.asarray(new_waypoints, dtype=np.float64),
            ])
            rr.log("world/plan", rr.LineStrips3D(
                [[[float(p[0]), float(p[1]), 0.0] for p in polyline]],
                colors=[[70, 230, 100]], radii=0.014,
            ))
            driving = False
            last_sent = None
            n_weak = 0
            last_good_t = time.monotonic()
            print(f"[drive] {reason} — replanned toward the SAME frontier "
                  f"({replans}/3), no target switching.")
            return True

        if True:     # single-leg loop (keeps the body's indentation stable)
            while True:
                safety_reason = controller.safety_latched_reason()
                if safety_reason is not None:
                    controller.halt()
                    if safety_reason.startswith("map edge"):
                        print(f"[drive] {safety_reason} — stopping to map before continuing.")
                        controller.clear_safety_latch()
                        break
                    if _replan_same_frontier(f"safety stop: {safety_reason}"):
                        continue
                    print(f"[drive] safety stop persisted after same-frontier replans: {safety_reason}.")
                    return f"hard stop ({safety_reason})"
                if _budget_exhausted(
                    time.monotonic() - mission_t0, float(args.max_mission_seconds)
                ):
                    controller.halt()
                    return "mission time budget spent"
                if time.monotonic() - leg_t0 > float(args.max_leg_seconds):
                    controller.halt()
                    return "leg time budget spent"
                # ODOMETRY PROPAGATION: fold measured forward travel since the
                # last iteration into the pose along the physical-forward bearing.
                # The matcher then only CORRECTS a pose that already moved — it no
                # longer has to find (or alias onto) the robot from a frozen guess.
                od = odom.take_forward_delta() * odom_scale[0]
                prop_info = None
                if abs(od) > 1e-4:
                    obr = math.radians(float(cur_pose.theta_deg) + forward_offset)
                    cur_pose = Pose2D(x=float(cur_pose.x) + od * math.cos(obr),
                                      y=float(cur_pose.y) + od * math.sin(obr),
                                      theta_deg=float(cur_pose.theta_deg))
                    prop_info = (od, math.cos(obr), math.sin(obr),
                                 float(cur_pose.x), float(cur_pose.y))
                centre = _robot_centre(cur_pose)
                if float(np.hypot(*(goal - centre))) <= float(args.waypoint_tol_m):
                    break
                centre_cell = analysis.to_cell(centre)
                ch, cw = analysis.traversable.shape
                centre_is_traversable = (
                    0 <= centre_cell[0] < ch
                    and 0 <= centre_cell[1] < cw
                    and bool(analysis.traversable[centre_cell])
                )
                # When a correction places the centre just inside inflation,
                # use a short carrot so the commanded heading follows the local
                # escape segment instead of looking 45cm ahead around a corner.
                pursuit_lookahead = (
                    float(args.lookahead_m)
                    if centre_is_traversable
                    else min(0.12, float(args.lookahead_m))
                )
                carrot = _carrot_on(polyline, centre, pursuit_lookahead)
                vec = carrot - centre
                # Heading to the carrot, from the current estimate.
                alpha = math.degrees(math.atan2(vec[1], vec[0]))
                desired_map = alpha - forward_offset
                err_map = ((desired_map - float(cur_pose.theta_deg) + 180.0) % 360.0) - 180.0
                yaw_now = imu.deg()
                if yaw_now is None:
                    controller.halt()
                    time.sleep(0.1)
                    continue
                # Mode hysteresis: pivot for big errors, drive for small.
                if driving and abs(err_map) > float(args.drive_exit_tol_deg):
                    driving = False
                    last_sent = None
                elif not driving and abs(err_map) <= float(args.aim_tolerance_deg):
                    driving = True
                    last_sent = None
                    # Do not feed the tracker the last revolution captured
                    # during the pivot. Halt, carry the final gyro motion into
                    # the LiDAR lever pose, and wait for a stationary scan.
                    controller.halt()
                    halt_frame_id = feed.latest()[0]
                    time.sleep(max(0.0, float(args.drive_settle_s)))
                    feed.wait_for_frame_after(
                        after_frame_id=halt_frame_id,
                        timeout_s=0.8,
                        min_frame_advances=1,
                    )
                    yaw_settled = imu.deg()
                    if yaw_settled is not None and prev_imu is not None:
                        cur_pose = _rotate_lidar_pose_about_robot_centre(
                            cur_pose,
                            float(yaw_settled) - float(prev_imu),
                            lever_m,
                            forward_offset,
                        )
                    prev_imu = yaw_settled
                    continue
                target_imu = float(yaw_now) + err_map

                if not driving:
                    # PIVOT phase: NO scan-matching. During fast rotation the scan
                    # was captured ~0.1-0.2s before the IMU read that seeds the
                    # match — a 5-10deg heading lie at pivot speed, outside the
                    # search window (field: score 14.9 stationary -> 2.1 the moment
                    # the pivot began). The gyro alone owns the pose here: rotation
                    # is gyro-tracked exactly and the LiDAR follows its lever-arm
                    # arc around the fixed robot centre (the small mecanum
                    # pivot-wander is inside the next drive window). The 25Hz
                    # servo does the actual turning.
                    if last_sent is None or abs(target_imu - last_sent[1]) > 4.0:
                        controller.rotate_to(target_imu)
                        last_sent = (False, target_imu)
                    if prev_imu is not None:
                        cur_pose = _rotate_lidar_pose_about_robot_centre(
                            cur_pose,
                            float(yaw_now) - float(prev_imu),
                            lever_m,
                            forward_offset,
                        )
                    prev_imu = yaw_now
                    time.sleep(0.05)
                else:
                    # DRIVE phase: scan-to-map tracking with a PHYSICS-BOUNDED
                    # window — the base cannot outrun v_max, so no match may claim
                    # a bigger displacement, whatever its score. While driving
                    # straight the yaw-hold keeps rotation slow; the tracker pairs
                    # each scan with its host-timestamped IMU yaw.
                    t_now = time.monotonic()
                    window = min(0.45, v_max * (t_now - last_good_t) + 0.10)
                    score, local, prev_imu = _track(
                        prev_imu, integrate=False, search_xy_m=window
                    )

                    # SAFETY PRECEDES localization recovery. A previous ordering
                    # allowed the mediocre-score detector to start a 360deg spin
                    # while a shoulder was already against a desk. The map guard
                    # checks the complete inflated footprint (including the body
                    # behind the forward-mounted LiDAR); the live lane remains a
                    # pose-independent last resort for returns directly ahead.
                    centre_now = _robot_centre(cur_pose)
                    sweep_state, sweep_distance = _swept_footprint_state(
                        analysis,
                        centre_now,
                        float(cur_pose.theta_deg) + forward_offset,
                        swept_guard_m,
                    )
                    if sweep_state == "unknown":
                        controller.halt()
                        controller.clear_safety_latch()
                        print(
                            f"[drive] map edge {sweep_distance:.2f}m ahead — "
                            "stopping to map before continuing."
                        )
                        break
                    if sweep_state == "blocked":
                        controller.halt()
                        reason = f"full-body clearance blocked {sweep_distance:.2f}m ahead"
                        if _replan_same_frontier(reason):
                            continue
                        return "hard stop (mapped shoulder clearance)"
                    if len(local):
                        ff = _to_forward_frame(local, forward_offset)
                        lane = ((ff[:, 0] > 0.02)
                                & (np.abs(ff[:, 1]) < float(args.hard_stop_half_width_m)))
                        if np.any(lane) and float(ff[lane, 0].min()) < float(args.hard_stop_m):
                            controller.halt()
                            offenders = local[lane & (ff[:, 0] < float(args.hard_stop_m) + 0.25)]
                            if len(offenders):
                                world_map.grid.mark_hits(
                                    _transform_points(offenders, cur_pose), amount=0.6
                                )
                            if _replan_same_frontier("live full-width stop lane blocked"):
                                continue
                            return "hard stop (live forward clearance)"

                    if score >= float(args.min_match_score):
                        last_good_t = t_now
                        n_weak = 0
                        # An accepted match is not a localization failure merely
                        # because its raw score is below 13. Corridors, doorways,
                        # and partial views naturally have less scan support. The
                        # former plateau heuristic spun the robot at score 12.1
                        # while it was making good progress and then abandoned
                        # the objective. Recovery is now reserved for consecutive
                        # matches below --min-match-score in the branch below.
                        # ONLINE ODOMETRY CALIBRATION: a strong match is ground
                        # truth for how far we actually moved. Compare it with the
                        # propagated step to walk the metres-per-unit scale onto
                        # the host's true velocity units (unknown a priori).
                        if score >= 12.0 and prop_info is not None and abs(prop_info[0]) > 0.03:
                            od_i, fx_i, fy_i, px_i, py_i = prop_info
                            c_fwd = ((float(cur_pose.x) - px_i) * fx_i
                                     + (float(cur_pose.y) - py_i) * fy_i)
                            ratio = (od_i + c_fwd) / od_i
                            if 0.3 < ratio < 3.0:
                                odom_scale[0] = float(np.clip(
                                    0.85 * odom_scale[0] + 0.15 * odom_scale[0] * ratio,
                                    0.05, 1.5))
                    else:
                        n_weak += 1
                        if n_weak >= 2:
                            # Lost while moving: stop; relock STATIONARY with a
                            # window still bounded by how far we could physically
                            # have gone since the last good fix, and a tight theta
                            # (the gyro is right). No teleport license.
                            controller.halt()
                            time.sleep(0.35)
                            win_r = min(0.60, v_max * (time.monotonic() - last_good_t) + 0.20)
                            relocked = False
                            after_r = feed.latest()[0]
                            for _attempt in range(3):     # patience: fresh frame each try
                                fid_r, frame_r = feed.wait_for_frame_after(
                                    after_frame_id=after_r, timeout_s=1.5, min_frame_advances=2)
                                if frame_r is None:
                                    continue
                                after_r = int(fid_r)
                                local_r = _scan_local(frame_r, args)
                                if len(local_r) < 12:
                                    continue
                                sup_r = _match_support_mask(_transform_points(local_r, cur_pose))
                                match_r = local_r[sup_r] if int(sup_r.sum()) >= 30 else local_r
                                solved_r, sc_r = _localize(match_r, world_map, cur_pose, args,
                                                           win_r, 10.0)
                                if sc_r >= float(args.min_match_score):
                                    cur_pose = solved_r
                                    odom.take_forward_delta()   # absolute fix: discard pending odometry
                                    relocked = True
                                    print(f"[drive] tracking slipped — re-locked stationary "
                                          f"(match {sc_r:.1f}, window {win_r:.2f}m).")
                                    break
                            if not relocked:
                                # Single-scan relock failed — escalate to the
                                # strongest evidence we have (once per leg).
                                controller.halt()
                                if not spin_tried and _recovery_spin():
                                    spin_tried = True
                                    relocked = True
                                else:
                                    return "tracking lost"
                            n_weak = 0
                            last_good_t = time.monotonic()
                            last_sent = None
                            driving = False       # re-aim from the corrected pose
                            prev_imu = imu.deg()
                            continue
                    if t_now - last_diag >= 2.0:
                        last_diag = t_now
                        c_dbg = _robot_centre(cur_pose)
                        print(f"[drive] score {max(score, -9.9):4.1f} window {window:.2f}m "
                              f"pos ({c_dbg[0]:+.2f},{c_dbg[1]:+.2f}) dist {float(np.hypot(*vec)):.2f}m "
                              f"heading_err {err_map:+.1f}deg odo x{odom_scale[0]:.2f}")
                    # Novel returns in the box? Do NOT stop yet — first apply the
                    # physical discriminator, which only works WHILE MOVING:
                    # a REAL object stays fixed in the world as the robot
                    # approaches; a PHANTOM attached to the robot (arm flex, near-
                    # field vibration artifact) travels along with it. The parked
                    # observe-window can never tell these apart (both look static
                    # when the robot is still) — which is how a self-artifact
                    # blocked the exit twice (field: user confirmed nothing there).
                    # The full-width live lane above still guards during confirmation.
                    is_novel, novel_w, _b = _novel_from(local)
                    if is_novel and novel_w is not None:
                        novel_hist.append((np.mean(novel_w, axis=0), _robot_centre(cur_pose)))
                        if len(novel_hist) > 4:
                            novel_hist.pop(0)
                        if len(novel_hist) >= 3:
                            robot_moved = float(np.hypot(*(novel_hist[-1][1] - novel_hist[0][1])))
                            if robot_moved >= 0.08:
                                obs_moved = float(np.hypot(*(novel_hist[-1][0] - novel_hist[0][0])))
                                if obs_moved < max(0.12, 0.5 * robot_moved):
                                    # World-fixed while we approached: REAL.
                                    novel_hist.clear()
                                    if _handle_obstacle() == "replan":
                                        controller.halt()
                                        return "obstacle - replan"
                                    prev_imu = imu.deg()
                                    driving = False
                                    last_sent = None
                                    continue
                                if not phantom_warned:
                                    phantom_warned = True
                                    print("[explore] near-field returns MOVE WITH the robot — "
                                          "self-artifact (arm flex / vibration), ignoring; "
                                          "not a world obstacle.")
                                # Phantom: keep driving.
                    else:
                        novel_hist.clear()
                    # Continuous drive: the fast controller keeps streaming and
                    # holding IMU yaw while its independent safety callback runs
                    # before every command. The slow loop only refreshes the
                    # pursuit target and corrects pose from moving LiDAR scans.
                    if last_sent is None or abs(target_imu - last_sent[1]) > 4.0:
                        controller.drive_toward(float(args.drive_speed), target_imu)
                        last_sent = (True, target_imu)
                trail.append(_robot_centre(cur_pose))
                # Throttle rendering — it is expensive and adds steering latency.
                now = time.monotonic()
                if now - last_render >= 0.5:
                    last_render = now
                    _log_panorama()
                    _log_world(rr, world_map, args, trail[-1], trail, analysis, goal_xy, waypoints)
        # Arrived: FACE THE FRONTIER before mapping. The body occludes ~90deg
        # behind and the stowed arms shadow their sectors — an arbitrary arrival
        # heading can blind exactly the area we came to observe (field: stops
        # 1.2m from the doorway 'learned nothing' twice, then retired the exit).
        controller.halt()
        if face_xy is not None:
            vec_f = np.asarray(face_xy, dtype=np.float64) - _robot_centre(cur_pose)
            if float(np.hypot(*vec_f)) > 0.05:
                alpha_f = math.degrees(math.atan2(vec_f[1], vec_f[0]))
                err_f = ((alpha_f - forward_offset - float(cur_pose.theta_deg) + 180.0) % 360.0) - 180.0
                yaw_f = imu.deg()
                if yaw_f is not None and abs(err_f) > 10.0:
                    controller.rotate_to(float(yaw_f) + err_f)
                    t_f = time.monotonic()
                    while time.monotonic() - t_f < 6.0:
                        y_now = imu.deg()
                        if y_now is not None and abs((float(yaw_f) + err_f) - float(y_now)) <= 3.0:
                            break
                        time.sleep(0.05)
                    controller.halt()
                    y_now = imu.deg()
                    if y_now is not None:
                        cur_pose = _rotate_lidar_pose_about_robot_centre(
                            cur_pose,
                            float(y_now) - float(yaw_f),
                            lever_m,
                            forward_offset,
                        )
        # Now map here with CLEAN stationary scans, looking AT the frontier.
        _integrate_at_viewpoint()
        return ""

    # ================= PHASE 2..N: explore the frontiers =================
    aborted = ""
    if bool(args.spin_only):
        print("[explore] --spin-only: skipping exploration; rendering the anchor spin map only.")
    else:
        controller.start()   # background velocity streaming for continuous driving
        odom.start()         # propagate-by-odometry between scan corrections
        odom.take_forward_delta()   # discard anything accumulated during startup
    while (
        not bool(args.spin_only)
        and not _budget_exhausted(viewpoints_reached, int(args.explore_viewpoints))
    ):
        if _budget_exhausted(
            time.monotonic() - mission_t0, float(args.max_mission_seconds)
        ):
            aborted = "mission time budget spent"
            break
        print("\n[explore] analyzing the map for significant frontiers ...")
        analysis = _analysis_now()
        frontiers_left = len(analysis.clusters)
        centre = _robot_centre(cur_pose)
        if active_frontier_xy is not None:
            active_cluster = _match_active_frontier(analysis.clusters, active_frontier_xy)
            if active_cluster is None:
                print("[explore] active frontier disappeared after mapping — objective resolved.")
                active_frontier_xy = None
            else:
                # Follow the frontier as its centroid advances into newly seen space.
                active_frontier_xy = active_cluster.centroid_xy.copy()
        picked_target = _pick_target(analysis, centre, visited,
                                     float(args.viewpoint_pullback_m), float(args.visited_skip_m),
                                     preferred_frontier_xy=active_frontier_xy,
                                     sampled_viewpoints_xy=sampled_viewpoints)
        if picked_target is None:
            # 'No target' has TWO very different meanings, and conflating them
            # once ended a mission at 0 viewpoints with live frontiers: either
            # the map is genuinely done, or PLANNING failed from the current
            # (possibly wobbled) pose. If frontiers remain, re-anchor the pose
            # stationary and retry before ever declaring completion.
            live = [c for c in analysis.clusters
                    if not any(np.hypot(*(c.centroid_xy - v)) < float(args.visited_skip_m)
                               for v in visited)]
            if live and none_retries < 2:
                none_retries += 1
                print(f"[explore] {len(live)} live frontier(s) but planning failed from here — "
                      f"re-anchoring and retrying ({none_retries}/2) ...")
                controller.halt()
                time.sleep(0.4)
                fid_n, frame_n = feed.wait_for_frame_after(after_frame_id=feed.latest()[0],
                                                           timeout_s=1.5, min_frame_advances=2)
                if frame_n is not None:
                    local_n = _scan_local(frame_n, args)
                    if len(local_n) >= 12:
                        sup_n = _match_support_mask(_transform_points(local_n, cur_pose))
                        match_n = local_n[sup_n] if int(sup_n.sum()) >= 30 else local_n
                        solved_n, sc_n = _localize(match_n, world_map, cur_pose, args, 0.60, 12.0)
                        if sc_n >= float(args.min_match_score):
                            cur_pose = solved_n
                            odom.take_forward_delta()   # absolute fix: discard pending odometry
                            print(f"[explore]   re-anchored (match {sc_n:.1f}).")
                continue
            if live:
                if active_frontier_xy is not None:
                    print("[explore] active path ended at the known-space boundary — "
                          "taking one snapshot, then selecting from the updated map.")
                    _integrate_at_viewpoint()
                    sampled_viewpoints.append(_robot_centre(cur_pose).copy())
                    viewpoints_reached += 1
                    active_frontier_xy = None
                    vp_last[0] = None
                    vp_gain_last[0] = None
                    none_retries = 0
                    continue
                print(f"[explore] {len(live)} frontier(s) remain but are unreachable from anywhere "
                      "the robot can stand — stopping honestly (NOT 'complete').")
                aborted = f"{len(live)} frontier(s) remain unreachable"
            else:
                if analysis.clusters:
                    print(f"[explore] {len(analysis.clusters)} significant frontier(s) remain, "
                          "but all were retired as blocked — stopping, NOT complete.")
                    aborted = f"{len(analysis.clusters)} blocked frontier(s) remain"
                else:
                    print("[explore] no significant frontier remains — the map is complete.")
            _log_world(rr, world_map, args, centre, trail, analysis, note="exploration finished")
            break
        none_retries = 0
        cluster, goal_xy, waypoints, close_look, observe_xy, expected_gain = picked_target
        continuing_objective = active_frontier_xy is not None
        active_frontier_xy = cluster.centroid_xy.copy()
        kind = "CLOSE LOOK at" if close_look else "ADVANCE toward"
        ownership = "continuing active" if continuing_objective else "new active"
        print(f"[explore] {ownership} frontier: span {cluster.span_m:.2f}m at "
              f"({cluster.centroid_xy[0]:+.2f}, {cluster.centroid_xy[1]:+.2f}), "
              f"{kind} visible cell ({observe_xy[0]:+.2f}, {observe_xy[1]:+.2f}) "
              f"from ({goal_xy[0]:+.2f}, {goal_xy[1]:+.2f}), "
              f"expected gain {expected_gain} cells, {len(waypoints)} waypoint(s).")
        # PLANNED path, logged ONCE per plan as a static green polyline — compare it
        # against the orange executed trail to see intent vs reality.
        plan_line = [[float(centre[0]), float(centre[1]), 0.0]] + [
            [float(p[0]), float(p[1]), 0.0] for p in waypoints
        ]
        rr.log("world/plan", rr.LineStrips3D([plan_line], colors=[[70, 230, 100]], radii=0.014))
        _log_world(rr, world_map, args, centre, trail, analysis, observe_xy, waypoints,
                   note=f"driving to frontier (span {cluster.span_m:.2f}m)")

        leg_failed = _drive_leg(waypoints, goal_xy, analysis, face_xy=observe_xy)
        if leg_failed == "mission time budget spent":
            aborted = leg_failed
            break
        if leg_failed == "obstacle - replan":
            # The obstacle is now marked in the occupancy grid (and excluded from future
            # novelty checks); re-plan around it. Progress guarantee: if the SAME
            # frontier gets obstacle-blocked twice, give up on it — endless
            # replan-loops against one target are how the mission dies in place.
            fkey = (int(round(cluster.centroid_xy[0] / 0.3)), int(round(cluster.centroid_xy[1] / 0.3)))
            obstacle_strikes[fkey] = obstacle_strikes.get(fkey, 0) + 1
            if obstacle_strikes[fkey] >= 2:
                print("[explore] frontier blocked twice — giving up on it, trying the next.")
                visited.append(cluster.centroid_xy)
                active_frontier_xy = None
            else:
                print("[explore] re-planning around the obstacle ...")
            continue
        if leg_failed.startswith("hard stop"):
            # A clearance boundary is itself a useful observation pose. Scan
            # once, merge, and let the updated map select the next viewpoint.
            # Never reverse automatically: the rear LiDAR sector is occluded.
            controller.halt()
            controller.clear_safety_latch()
            print("[explore] clearance boundary reached — taking a stationary map here.")
            _integrate_at_viewpoint()
            leg_failed = ""
        if leg_failed == "tracking lost":
            # Localization hiccup, NOT a bad target — don't blacklist the exit for
            # it (field: robot reached the doorway, confidence dipped, and the old
            # policy marked THE EXIT visited and turned away). Retry once; only a
            # second loss on the same frontier gives up on it.
            fkey = (int(round(cluster.centroid_xy[0] / 0.3)), int(round(cluster.centroid_xy[1] / 0.3)))
            lost_strikes[fkey] = lost_strikes.get(fkey, 0) + 1
            if lost_strikes[fkey] >= 2:
                print("[explore] tracking lost twice while pursuing the active frontier — "
                      "stopping instead of silently choosing a different destination.")
                aborted = "localization lost while pursuing active frontier"
                break
            else:
                print("[explore] tracking lost — re-anchoring and retrying the same frontier.")
            continue
        if leg_failed == "leg time budget spent":
            # A leg timer is a scheduling slice, not evidence that the frontier
            # is bad. Start another slice toward the same persistent objective.
            print("[explore] leg time slice spent — continuing the same active frontier.")
            continue
        if leg_failed:
            print(f"[explore] leg failed ({leg_failed}) — stopping with the active objective "
                  "preserved instead of switching destinations.")
            aborted = leg_failed
            break

        # DIVERGENCE WATCHDOG: two consecutive viewpoints that mapped nothing
        # (weak or discarded batches) means the map and reality have parted ways.
        # Recover once against the GOLD reference; if that fails, stop honestly
        # instead of continuing to paint garbage ('once the map is lost it's
        # game over' — the user's words, and correct).
        if vp_last[0] == 0:
            consistency_fails += 1
            if consistency_fails >= 2:
                print("[explore] map-consistency watchdog: two failed viewpoints — recovery spin.")
                if _recovery_spin():
                    consistency_fails = 0
                else:
                    aborted = "map consistency lost (watchdog)"
                    break
        elif vp_last[0] is not None and vp_last[0] > 0:
            consistency_fails = 0
        vp_last[0] = None

        # Snapshot complete: release the travel objective. The next selection is
        # made from the updated map and this scan's measured information gain.
        # Persistence applies only while travelling/replanning, never after a
        # photograph has completed.
        viewpoints_reached += 1
        here = _robot_centre(cur_pose).copy()
        sampled_viewpoints.append(here)
        actual_gain = int(vp_gain_last[0] or 0)
        vp_gain_last[0] = None
        active_frontier_xy = None
        kind_msg = "close look" if close_look else "advance"
        print(f"[explore] {kind_msg} {viewpoints_reached} done; map now {len(world_map.scans)} "
              f"scans, actual gain {actual_gain} newly-known cells; selecting next viewpoint.")
        _log_world(rr, world_map, args, _robot_centre(cur_pose), trail, _analysis_now(),
                   note=f"stop {viewpoints_reached} mapped")

    _send_stop(robot)
    final_analysis = _analysis_now()
    _log_world(rr, world_map, args, _robot_centre(cur_pose), trail, final_analysis,
               note="exploration finished")

    genuinely_complete = not aborted and len(final_analysis.clusters) == 0
    if genuinely_complete:
        final_heading = "EXPLORE COMPLETE"
    elif aborted:
        final_heading = "EXPLORE STOPPED"
    else:
        final_heading = "EXPLORE LIMIT REACHED"
    print(f"\n=========== {final_heading} ===========")
    if aborted:
        print(f"  stopped early: {aborted} (map preserved).")
    print(f"  {spin_scan_count} spin scans + {len(world_map.scans) - spin_scan_count} "
          f"tracked-while-driving scans = {len(world_map.scans)} total.")
    print(f"  viewpoints reached: {viewpoints_reached}.")
    print(f"  significant frontiers remaining: {len(final_analysis.clusters)} "
          f"(was {frontiers_left} at first analysis).")
    print("  Map = world/lidar_map; frontiers red, target green sphere, PLANNED path "
          "green line (world/plan), executed trail orange (world/trail).")
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
            odom.shutdown,
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
