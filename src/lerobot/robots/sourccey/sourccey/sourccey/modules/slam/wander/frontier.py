"""Wander PATH GENERATION — deciding WHERE there is unmapped space worth going to.

Two sources of "frontier" (the boundary between mapped and unknown): the LIVE
LiDAR scan (``_select_frontier_choice`` — the widest/nearest open gap the sensor
sees right now) and the STITCHED occupancy grid (``_select_map_frontier_choice``
— BFS to the nearest reachable unknown cell on the inflated grid). Plus the
supporting machinery: building/dilating the world occupancy grid, picking a
survey vantage, converting a world point into a steer command, and seeding an
exploration target. This module chooses direction only; it never commands motion.
"""
from __future__ import annotations

import math

import numpy as np

from ..lidar.direct_snapshot_stitch import (
    Pose2D,
)
from .wander_types import (
    ExploreTarget,
    FrontierChoice,
    _normalize_angle_deg,
)

def _doorway_mouth_and_axis(
    frame,
    *,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    max_distance_m: float,
    min_range_m: float,
    min_confidence: int,
    opening_delta_deg: float,
    opening_width_deg: float,
    opening_depth_m: float,
    frontier_bin_deg: float = 6.0,
) -> dict | None:
    """Locate the two JAMBS flanking a chosen opening and return the mouth
    centre + hall axis, so the exit run can approach a doorway SQUARE-ON and
    CENTRED instead of blind-probing on a single bearing (field 2026-07-20: the
    robot reached the hallway mouth off to one side, the bearing probe aimed
    diagonally, it clipped/stalled and never entered).

    Geometry is returned in the robot-LOCAL polar convention the rest of the
    planner uses (bearing measured the same way as ``FrontierChoice.delta_deg``
    — world bearing = ``pose.theta_deg + delta``, so the caller converts with a
    plain addition):
      {
        'mouth_delta_deg', 'mouth_range_m',   # polar to the gap centre
        'axis_delta_deg',                     # local bearing pointing INTO the hall
        'gap_width_m',
        'jamb_left_xy', 'jamb_right_xy',      # local (x fwd, y lateral), for tests
      }
    A jamb is the nearest wall return just OUTSIDE each edge of the open gap.
    Returns None if either jamb cannot be found (open on that side / no wall)."""
    half_width = max(10.0, float(valid_angle_half_width_deg))
    bin_deg = max(1.0, float(frontier_bin_deg))
    bin_count = max(9, int(math.ceil((half_width * 2.0) / bin_deg)) + 1)
    bin_angles = np.linspace(-half_width, half_width, bin_count, dtype=np.float64)
    nearest = np.full(bin_count, np.nan, dtype=np.float64)
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
        if math.isnan(nearest[idx]) or distance_m < nearest[idx]:
            nearest[idx] = distance_m

    # A jamb is a WALL post: a return markedly CLOSER than the depth seen
    # through the gap. Threshold below that depth so a through-gap return (which
    # bins near the edge can still hold) is never mistaken for the frame.
    wall_threshold_m = max(0.20, float(opening_depth_m) * 0.70)
    center_idx = int(round((float(opening_delta_deg) + half_width) / bin_deg))
    center_idx = max(0, min(bin_count - 1, center_idx))

    def _jamb(outward_sign: float) -> tuple[float, float] | None:
        # Walk OUTWARD from the gap CENTRE; skip open (through-gap) bins; the
        # first bin closer than the wall threshold is the doorframe post on this
        # side. Returns (delta_deg, range_m).
        for step in range(1, bin_count):
            idx = center_idx + int(outward_sign) * step
            if idx < 0 or idx >= bin_count:
                return None
            r = nearest[idx]
            if not math.isnan(r) and float(r) < wall_threshold_m:
                return float(bin_angles[idx]), float(r)
        return None

    left = _jamb(-1.0)
    right = _jamb(+1.0)
    if left is None or right is None:
        return None

    def _to_xy(delta_deg: float, range_m: float) -> tuple[float, float]:
        a = math.radians(delta_deg)
        return (range_m * math.cos(a), range_m * math.sin(a))

    lx, ly = _to_xy(*left)
    rx, ry = _to_xy(*right)
    mx, my = (lx + rx) / 2.0, (ly + ry) / 2.0
    mouth_range = math.hypot(mx, my)
    if mouth_range < 1e-3:
        return None
    mouth_delta = math.degrees(math.atan2(my, mx))
    gap_width = math.hypot(rx - lx, ry - ly)
    # Axis = perpendicular to the jamb line, pointing AWAY from the robot (into
    # the hall). Pick the perpendicular whose dot with the mouth-centre vector
    # is positive.
    jx, jy = (rx - lx), (ry - ly)
    perp_a = (-jy, jx)
    perp_b = (jy, -jx)
    axis_vec = perp_a if (perp_a[0] * mx + perp_a[1] * my) >= 0.0 else perp_b
    axis_delta = math.degrees(math.atan2(axis_vec[1], axis_vec[0]))
    return {
        "mouth_delta_deg": float(mouth_delta),
        "mouth_range_m": float(mouth_range),
        "axis_delta_deg": float(axis_delta),
        "gap_width_m": float(gap_width),
        "jamb_left_xy": (float(lx), float(ly)),
        "jamb_right_xy": (float(rx), float(ry)),
    }


def _select_exit_doorway_choice(
    frame,
    *,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    max_distance_m: float,
    min_range_m: float,
    min_confidence: int,
    frontier_min_distance_m: float,
    frontier_bin_deg: float,
    min_gap_width_m: float = 0.50,
    max_gap_width_m: float = 1.50,
) -> FrontierChoice | None:
    """Pick the best real DOORWAY (a gap flanked by two walls) from the live
    scan — for the exit run, so it commits to an actual door instead of the
    widest OPEN-ROOM direction (field 2026-07-20: the exit run kept probing
    into 80-120deg-wide open floor, the new-space gate correctly rejected it as
    'a wall inside the room, not a door', and the robot milled near the centre
    forever instead of turning to the narrow doorway). A doorway is an open
    angular segment whose OUTER edges both have a wall return (a jamb) and whose
    physical gap is doorway-plausible (``min_gap_width_m``..``max_gap_width_m``).
    Open-room directions have no flanking jambs and are rejected. Returns the
    best doorway as a ``FrontierChoice`` (deepest / most door-like wins), or
    None if no real door is visible from this heading — in which case the caller
    should ROTATE to keep searching rather than drive into open floor."""
    half_width = max(10.0, float(valid_angle_half_width_deg))
    bin_deg = max(1.0, float(frontier_bin_deg))
    bin_count = max(9, int(math.ceil((half_width * 2.0) / bin_deg)) + 1)
    bin_angles = np.linspace(-half_width, half_width, bin_count, dtype=np.float64)
    nearest_ranges = np.zeros(bin_count, dtype=np.float64)
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
        if hit_counts[idx] == 0 or distance_m < nearest_ranges[idx]:
            nearest_ranges[idx] = distance_m
        hit_counts[idx] += 1
    if not np.any(hit_counts):
        return None
    kernel = np.array([0.2, 0.6, 0.2], dtype=np.float64)
    smoothed = np.convolve(nearest_ranges, kernel, mode="same")
    open_mask = smoothed >= float(frontier_min_distance_m)
    if not np.any(open_mask):
        return None
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

    best: FrontierChoice | None = None
    for start_idx, end_idx in segments:
        seg_center = int((start_idx + end_idx) // 2)
        seg_delta = float(bin_angles[seg_center])
        seg_width = float((end_idx - start_idx + 1) * bin_deg)
        seg_mean = float(np.mean(smoothed[start_idx : end_idx + 1]))
        geom = _doorway_mouth_and_axis(
            frame,
            forward_angle_deg=forward_angle_deg,
            valid_angle_half_width_deg=valid_angle_half_width_deg,
            max_distance_m=max_distance_m,
            min_range_m=min_range_m,
            min_confidence=min_confidence,
            opening_delta_deg=seg_delta,
            opening_width_deg=seg_width,
            opening_depth_m=seg_mean,
            frontier_bin_deg=bin_deg,
        )
        if geom is None:
            continue
        gap_w = float(geom["gap_width_m"])
        if not (float(min_gap_width_m) <= gap_w <= float(max_gap_width_m)):
            continue
        # Score: reward depth (a real door leads OUT to deep space) and being
        # close to a typical door width (~0.8m); a plain forward heading is a
        # mild tiebreak so it does not spin toward a marginally-deeper side door.
        score = (
            float(seg_mean)
            + 1.5 * (1.0 - min(1.0, abs(gap_w - 0.80) / 0.80))
            - 0.004 * abs(seg_delta)
        )
        candidate = FrontierChoice(
            delta_deg=seg_delta,
            abs_angle_deg=float(forward_angle_deg) + seg_delta,
            mean_distance_m=seg_mean,
            width_deg=seg_width,
            score=float(score),
            source="live_doorway",
        )
        if best is None or candidate.score > best.score:
            best = candidate
    return best


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
    min_gap_width_m: float = 0.0,
    corridor_veto=None,
) -> FrontierChoice | None:
    """Pick the best "drive here" opening straight from the LIVE LiDAR scan.

    Bins the forward field of view by bearing, and in each bin records how far
    away the nearest return is — a far/absent return means open space. It then
    finds the widest contiguous run of open bins that is deep enough
    (``frontier_min_distance_m``) and wide enough (``min_gap_width_m`` so the body
    fits), optionally rejecting any opening whose corridor is vetoed (e.g. crosses
    a known table edge). Returns the winning gap as a ``FrontierChoice`` (bearing
    delta, mean depth, angular width, a score), or ``None`` if nothing opens up.
    This is the reflexive "I can see somewhere to go" sense, independent of the map.
    """
    half_width = max(10.0, float(valid_angle_half_width_deg))
    bin_deg = max(1.0, float(frontier_bin_deg))
    bin_count = max(9, int(math.ceil((half_width * 2.0) / bin_deg)) + 1)
    bin_angles = np.linspace(-half_width, half_width, bin_count, dtype=np.float64)
    # Openness of a direction is its NEAREST return, not its farthest: a bin
    # with a chair leg at 0.3m and a wall at 1.4m is NOT 1.4m of open space.
    # Using the max here made the planner steer the robot nose-first into
    # near obstacles it could plainly see.
    nearest_ranges = np.zeros(bin_count, dtype=np.float64)
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
        if hit_counts[idx] == 0 or distance_m < nearest_ranges[idx]:
            nearest_ranges[idx] = distance_m
        hit_counts[idx] += 1

    if not np.any(hit_counts):
        return None

    # Smooth the range profile so a single noisy return does not dominate heading choice.
    kernel = np.array([0.2, 0.6, 0.2], dtype=np.float64)
    smoothed_ranges = np.convolve(nearest_ranges, kernel, mode="same")
    open_mask = smoothed_ranges >= float(frontier_min_distance_m)
    if not np.any(open_mask):
        if float(min_gap_width_m) > 0.0:
            # Caller wants drivable openings only. The least-bad-bin fallback
            # below is a single-bin (~6deg) pointer — following it is how the
            # robot noses into corner slots it cannot fit through.
            return None
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
        seg_mean = float(np.mean(seg_ranges))
        seg_width = float((end_idx - start_idx + 1) * bin_deg)
        # A narrow angular slot is not a drivable opening: a 12deg-wide gap at
        # 1.4m is ~0.3m across — the robot's body cannot pass. Chasing such
        # slots is how the robot wedges itself into clutter fields.
        gap_width_m = 2.0 * seg_mean * math.sin(math.radians(min(seg_width, 178.0)) / 2.0)
        if float(min_gap_width_m) > 0.0 and gap_width_m < float(min_gap_width_m):
            continue
        seg_center = int((start_idx + end_idx) // 2)
        seg_delta = float(bin_angles[seg_center])
        # A 2D scan at ~20cm height sees UNDER beds and tables: a wide, deep
        # "opening" whose corridor the eye cameras have mapped as elevated
        # furniture is an under-furniture tunnel, not a frontier. Field
        # 2026-07-17: the robot pirouetted in the bed/dresser corner for a
        # whole run committing probes into exactly these tunnels while every
        # safety gate (correctly) refused the drive.
        if corridor_veto is not None and corridor_veto(seg_delta, seg_mean):
            continue
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
    """Rasterize the stitched point cloud into a boolean occupancy grid.

    Concatenates every snapshot's world points, finds their bounding box (plus a
    padding margin), and stamps each point into a 2D boolean grid at
    ``resolution_m`` per cell (True = something is there). Returns
    ``(grid, origin_mins, world_points)`` — the grid the frontier/clearance BFS
    runs on, the world coordinate of cell (0,0), and the raw points — or ``None``
    if the map is empty or degenerate.
    """
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
    """Fatten every occupied cell by ``radius_cells`` in all directions.

    For each True cell it sets the surrounding square block True. Dilating the
    obstacle grid by the robot's radius (in cells) turns "where is a wall" into
    "where would the robot's CENTER collide," so a point-sized path search on the
    dilated grid yields body-safe routes. Radius 0 returns the grid unchanged.
    """
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
    robot_radius_m: float = 0.30,
) -> FrontierChoice | None:
    """Find the nearest reachable unmapped region by searching the STITCHED map.

    Builds the world occupancy grid, dilates it by the robot radius (so routes
    are body-safe), and runs a breadth-first flood from the robot's cell through
    free space to find frontier cells — free cells that border the unknown. It
    picks the closest reachable one that's far enough away to be worth a trip and
    returns it as a ``FrontierChoice``. Unlike ``_select_frontier_choice`` (which
    only sees the live scan), this reasons over the whole accumulated map, so it
    can send the robot back to an opening it noticed earlier but didn't visit.
    Returns ``None`` if the map is empty or no reachable frontier remains.
    """
    occupancy = _build_world_occupancy(
        transformed_sets,
        resolution_m=max(0.03, float(resolution_m)),
        padding_m=max(0.35, float(max_distance_m) * 0.15),
    )
    if occupancy is None:
        return None

    occupied_grid, grid_origin_xy, world_points_xy = occupancy
    # Inflate obstacles by the ROBOT's radius, not a token margin: a gap the
    # map shows as open but narrower than the robot must read as blocked, or
    # the planner keeps steering into corners the robot cannot fit through.
    dilated_grid = _dilate_bool_grid(
        occupied_grid,
        radius_cells=max(1, int(round(max(0.08, float(robot_radius_m)) / max(0.03, float(resolution_m))))),
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
        # The robot's own footprint may sit inside the inflated region (its
        # center can legitimately be one robot-radius from a wall). Within
        # that footprint only EXACT wall cells block; inflation applies beyond.
        inflation_starts_m = float(robot_radius_m) + step_m
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
            blocking = (
                bool(dilated_grid[cell_y, cell_x])
                if float(distance_m) > inflation_starts_m
                else bool(occupied_grid[cell_y, cell_x])
            )
            if blocking:
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


def _select_survey_target(
    *,
    transformed_sets: list[np.ndarray],
    current_pose: Pose2D,
    capture_poses: list[Pose2D],
    resolution_m: float,
    robot_radius_m: float,
    max_distance_m: float,
    min_pose_spacing_m: float = 1.0,
) -> tuple[float, float] | None:
    """When nothing nearby is worth mapping, pick a vantage the robot has NOT
    camped at yet: ray-cast through the robot-inflated map and choose the
    reachable point that is most open and farthest from every previous capture
    pose. This is what sends the robot across the room to photograph the other
    corner instead of milling around the current one."""
    occupancy = _build_world_occupancy(
        transformed_sets,
        resolution_m=max(0.03, float(resolution_m)),
        padding_m=0.5,
    )
    if occupancy is None:
        return None
    occupied_grid, grid_origin_xy, _world_points = occupancy
    inflated_grid = _dilate_bool_grid(
        occupied_grid,
        radius_cells=max(1, int(round(float(robot_radius_m) / max(0.03, float(resolution_m))))),
    )
    pose_xy = np.asarray([float(current_pose.x), float(current_pose.y)], dtype=np.float32)
    step_m = 0.06
    inflation_starts_m = float(robot_radius_m) + step_m
    best_score = 0.0
    best_target: tuple[float, float] | None = None
    for heading_deg in np.arange(-180.0, 180.0, 10.0, dtype=np.float32):
        heading_rad = math.radians(float(heading_deg))
        direction = np.asarray([math.cos(heading_rad), math.sin(heading_rad)], dtype=np.float32)
        clearance_m = 0.0
        distance_m = step_m
        while distance_m <= float(max_distance_m):
            sample_xy = pose_xy + direction * float(distance_m)
            cell_xy = np.round((sample_xy - grid_origin_xy) / max(0.03, float(resolution_m))).astype(np.int32)
            cell_x = int(cell_xy[0])
            cell_y = int(cell_xy[1])
            if (
                cell_x < 0
                or cell_x >= int(inflated_grid.shape[1])
                or cell_y < 0
                or cell_y >= int(inflated_grid.shape[0])
            ):
                break
            blocking = (
                bool(inflated_grid[cell_y, cell_x])
                if float(distance_m) > inflation_starts_m
                else bool(occupied_grid[cell_y, cell_x])
            )
            if blocking:
                break
            clearance_m = float(distance_m)
            distance_m += step_m
        candidate_distance_m = min(clearance_m * 0.7, clearance_m - 0.35)
        if candidate_distance_m < 0.60:
            continue
        candidate_xy = pose_xy + direction * float(candidate_distance_m)
        pose_spacing_m = min(
            (
                math.hypot(float(candidate_xy[0]) - float(p.x), float(candidate_xy[1]) - float(p.y))
                for p in capture_poses
            ),
            default=1e9,
        )
        if pose_spacing_m < float(min_pose_spacing_m):
            continue
        score = float(pose_spacing_m) + 0.3 * float(clearance_m)
        if score > best_score:
            best_score = score
            best_target = (float(candidate_xy[0]), float(candidate_xy[1]))
    return best_target


def _compute_drive_steer_theta_vel(
    frontier_choice: FrontierChoice | None,
    *,
    steer_gain: float,
    steer_max: float,
    steer_deadband_deg: float,
) -> float:
    """Turn a target bearing into a gentle in-motion steering rate.

    While driving toward a frontier the robot curves toward it rather than
    stopping to turn: this maps the bearing error (``delta_deg``) to a rotational
    velocity via ``steer_gain``, ignores errors inside a small dead band, and
    clamps the result to ``steer_max``. Returns 0 when there's no target. It's the
    "aim while you roll" nudge, not a full turn.
    """
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
    """Re-express a fixed WORLD point as a frontier relative to the robot NOW.

    Given an absolute (x, y) goal in the map and the robot's current pose, it
    computes the bearing and distance from the robot to that point and packs them
    into a ``FrontierChoice`` (with zero width, since a point isn't a gap). This
    lets a remembered map target be fed through the same "drive toward a choice"
    machinery as a live opening. Returns ``None`` if the robot is already on top
    of the point.
    """
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
    """Turn a chosen frontier direction into a concrete world waypoint to hold.

    Projects out from the robot's current pose along the frontier's absolute
    heading by a capped distance (at least 0.5m, at most the step size or the
    frontier's own depth) and returns that world point as an ``ExploreTarget``
    tagged with the capture index. Committing to a fixed world point — instead of
    re-picking a live bearing each cycle — stops the robot oscillating between two
    openings and never actually translating.
    """
    heading_rad = math.radians(float(frontier_choice.abs_angle_deg))
    travel_m = max(0.50, min(float(step_distance_m), float(frontier_choice.mean_distance_m)))
    return ExploreTarget(
        world_x_m=float(current_pose.x) + math.cos(heading_rad) * travel_m,
        world_y_m=float(current_pose.y) + math.sin(heading_rad) * travel_m,
        source=str(frontier_choice.source),
        seeded_capture_index=int(capture_index),
    )


def _target_distance_m(target: ExploreTarget, pose: Pose2D) -> float:
    """Straight-line distance from the robot's current pose to an ExploreTarget's
    world point — used to tell whether it has arrived or stalled."""
    return float(math.hypot(float(target.world_x_m) - float(pose.x), float(target.world_y_m) - float(pose.y)))


