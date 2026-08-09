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

Code layout (mirrors ``ExploreSystem.md``):

    EXPLORE PHASE 0   connect hardware and prepare the mission
    EXPLORE PHASE 1   attempt the required initial 360-degree body spin
    EXPLORE PHASE 1.1 collision fallback: one partial map, one escape route,
                      then replace it with a complete 360-degree map
    EXPLORE PHASE 1.2 construct and select the initial map
    EXPLORE PHASE 1.3 recover the physical drive frame
    EXPLORE PHASE 1.4 validate the initial map
    EXPLORE PHASE 2   analyze the map and select an unknown-area target
    EXPLORE PHASE 3   follow the selected path
    EXPLORE PHASE 4   face the unknown area, scan it, and update the map
    EXPLORE PHASE 5   classify the result and repeat Phase 2

Search this file for ``EXPLORE PHASE`` to follow the behavior in execution
order. The phase labels organize the existing implementation; they are not a
second state machine.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import heapq
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from ldlidar_auto_snapshot_stitch import _init_rerun, _send_stop
from ldlidar_direct_snapshot_client import DirectLidarFeed
from ldlidar_direct_snapshot_stitch import (
    Pose2D,
    _search_pose,
    _transform_points,
    configure_candidate_scoring_device,
)
from sourccey_arm_pose import DEFAULT_POSE_PATH, apply_pose_blocking, load_pose
from sourccey_bottom_camera import (
    BottomCameraGroundOdometry,
    wait_for_live_bottom_camera,
)
from sourccey_pose_graph import (
    PoseGraph2D,
    axis_snap_delta_deg,
    dominant_axis_deg,
    wall_direction_masses,
)
from sourccey_surface_motion import SurfaceMotionAtlas, ground_texture_signature
from sourccey_collision_box import (
    DEFAULT_COLLISION_BOX_PATH,
    collision_box_rotation_violation as _collision_box_rotation_violation,
    collision_box_violation as _collision_box_violation,
    effective_ranges as _collision_box_effective_ranges,
    load_collision_box as _load_collision_box,
    physical_body_self_return_mask as _physical_body_self_return_mask,
)
from sourccey_saved_map import DEFAULT_SAVED_MAP_PATH, save_world_map
from sourccey_spin_map import (
    _MAP_PALETTE,
    _abort,
    _pick_free_ports,
    _scan_local,
    _spin_command,
)
from sourccey_visual_color_anchors import bottom_upper_color_signature, color_signature
from sourccey_wander.imu_heading import ImuYawClient

from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient


# ===========================================================================
# SHARED GEOMETRY — USED BY PHASES 1 THROUGH 5
# ===========================================================================
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


def _translate_lidar_pose_in_body_frame(
    pose: Pose2D,
    forward_m: float,
    left_m: float,
    lever_m: float,
    forward_offset_deg: float,
) -> Pose2D:
    """Propagate a robot-centre translation while retaining LiDAR heading."""
    centre = _robot_centre_from_lidar_pose(pose, lever_m, forward_offset_deg)
    heading = math.radians(float(pose.theta_deg) + float(forward_offset_deg))
    centre += np.array(
        [
            float(forward_m) * math.cos(heading) - float(left_m) * math.sin(heading),
            float(forward_m) * math.sin(heading) + float(left_m) * math.cos(heading),
        ],
        dtype=np.float64,
    )
    return _lidar_pose_from_robot_centre(
        centre,
        float(pose.theta_deg),
        lever_m,
        forward_offset_deg,
    )


def _pose_with_imu_heading(
    solved_pose: Pose2D,
    imu_theta_deg: float,
    lever_m: float,
    forward_offset_deg: float,
) -> Pose2D:
    """Keep LiDAR-matched translation while making the IMU own heading.

    A 2D scan matcher can obtain an excellent score at a rotated copy of a long
    wall.  On this robot the gyro observes the short-horizon yaw directly, so a
    professional LiDAR-inertial front end uses scan matching for translation and
    the synchronized IMU delta for rotation instead of letting that wall alias
    rewrite heading.
    """
    return _lidar_pose_from_robot_centre(
        _robot_centre_from_lidar_pose(
            solved_pose,
            lever_m,
            forward_offset_deg,
        ),
        float(imu_theta_deg),
        lever_m,
        forward_offset_deg,
    )


def _stationary_pose_consensus(
    poses: list[Pose2D],
    reference_theta_deg: float,
    lever_m: float,
    forward_offset_deg: float,
) -> tuple[Pose2D, float, float]:
    """Fuse stationary solves into one rigid pose and report their scatter."""
    if not poses:
        raise ValueError("stationary pose consensus requires at least one pose")
    centres = np.asarray([_robot_centre_from_lidar_pose(pose, lever_m, forward_offset_deg) for pose in poses])
    median_centre = np.median(centres, axis=0)
    position_scatter = float(
        np.max(
            np.hypot(
                centres[:, 0] - median_centre[0],
                centres[:, 1] - median_centre[1],
            )
        )
    )
    theta_deltas = np.asarray(
        [((float(pose.theta_deg) - float(reference_theta_deg) + 180.0) % 360.0) - 180.0 for pose in poses]
    )
    median_delta = float(np.median(theta_deltas))
    consensus_theta = float(reference_theta_deg) + median_delta
    heading_scatter = float(np.max(np.abs(theta_deltas - median_delta)))
    consensus = _lidar_pose_from_robot_centre(
        median_centre,
        consensus_theta,
        lever_m,
        forward_offset_deg,
    )
    return consensus, position_scatter, heading_scatter


def _stationary_pose_inlier_consensus(
    poses: list[Pose2D],
    reference_theta_deg: float,
    lever_m: float,
    forward_offset_deg: float,
    max_position_residual_m: float,
    max_heading_residual_deg: float,
    min_inliers: int = 4,
    reference_centre_xy: np.ndarray | None = None,
) -> tuple[Pose2D, list[int], float, float]:
    """RANSAC-style stationary pose mode and its coherent scan indices.

    Scan matching near repeated walls can produce one or two aliased solves in
    an otherwise coherent stationary batch. Rejecting all eight scans because
    the single worst residual crosses a hard threshold turns a recoverable
    observation into a navigation deadlock. Select the largest compact pose
    mode, recompute its median consensus, and integrate only that mode.

    Equally supported disjoint modes remain ambiguous and return no inliers;
    this preserves the map instead of arbitrarily choosing one copy of a wall.
    """
    if not poses:
        raise ValueError("stationary pose consensus requires at least one pose")
    min_required = max(1, int(min_inliers))
    position_gate = max(1e-6, float(max_position_residual_m))
    heading_gate = max(1e-6, float(max_heading_residual_deg))
    centres = np.asarray([_robot_centre_from_lidar_pose(pose, lever_m, forward_offset_deg) for pose in poses])
    theta_deltas = np.asarray(
        [((float(pose.theta_deg) - float(reference_theta_deg) + 180.0) % 360.0) - 180.0 for pose in poses]
    )
    position_pair = np.hypot(
        centres[:, None, 0] - centres[None, :, 0],
        centres[:, None, 1] - centres[None, :, 1],
    )
    heading_pair = np.abs((theta_deltas[:, None] - theta_deltas[None, :] + 180.0) % 360.0 - 180.0)

    prior_centre = (
        np.asarray(reference_centre_xy, dtype=np.float64) if reference_centre_xy is not None else None
    )
    unique_supports: dict[tuple[int, ...], tuple[int, float, float]] = {}
    for seed in range(len(poses)):
        indices = tuple(
            np.flatnonzero(
                (position_pair[seed] <= position_gate) & (heading_pair[seed] <= heading_gate)
            ).tolist()
        )
        compactness = float(
            np.mean(
                position_pair[seed, list(indices)] / position_gate
                + heading_pair[seed, list(indices)] / heading_gate
            )
        )
        support_centre = np.median(centres[list(indices)], axis=0)
        prior_cost = (
            float(np.hypot(*(support_centre - prior_centre))) / position_gate
            if prior_centre is not None
            else 0.0
        )
        prior = unique_supports.get(indices)
        if prior is None or compactness < prior[1]:
            unique_supports[indices] = (seed, compactness, prior_cost)

    ranked = sorted(
        unique_supports.items(),
        key=lambda item: (-len(item[0]), item[1][2], item[1][1]),
    )
    fallback, fallback_pos, fallback_heading = _stationary_pose_consensus(
        poses,
        reference_theta_deg,
        lever_m,
        forward_offset_deg,
    )
    if not ranked or len(ranked[0][0]) < min_required:
        return fallback, [], fallback_pos, fallback_heading

    best_indices = set(ranked[0][0])
    # Two equally large, mostly-disjoint modes are a real aliasing ambiguity.
    # Do not choose one merely because its numeric residual is microscopically
    # smaller.
    best_prior_cost = ranked[0][1][2]
    for indices, meta in ranked[1:]:
        if len(indices) != len(best_indices):
            break
        other = set(indices)
        overlap = len(best_indices & other) / max(1, len(best_indices | other))
        # A trusted motion prior resolves repeated-wall aliases the same way a
        # probabilistic SLAM filter does. Without a discriminating prior, two
        # equally supported disjoint modes remain unsafe to merge.
        prior_separation = abs(float(meta[2]) - float(best_prior_cost))
        if overlap < 0.5 and (prior_centre is None or prior_separation < 0.25):
            return fallback, [], fallback_pos, fallback_heading

    selected = sorted(best_indices)
    # Refit at the robust median and admit any additional solutions consistent
    # with that consensus. Two passes are enough for this eight-scan batch.
    for _ in range(2):
        consensus, _scatter, _heading_scatter = _stationary_pose_consensus(
            [poses[idx] for idx in selected],
            reference_theta_deg,
            lever_m,
            forward_offset_deg,
        )
        consensus_centre = _robot_centre_from_lidar_pose(consensus, lever_m, forward_offset_deg)
        consensus_delta = ((float(consensus.theta_deg) - float(reference_theta_deg) + 180.0) % 360.0) - 180.0
        position_residual = np.hypot(
            centres[:, 0] - consensus_centre[0],
            centres[:, 1] - consensus_centre[1],
        )
        heading_residual = np.abs((theta_deltas - consensus_delta + 180.0) % 360.0 - 180.0)
        refined = np.flatnonzero(
            (position_residual <= position_gate) & (heading_residual <= heading_gate)
        ).tolist()
        if len(refined) < min_required or refined == selected:
            break
        selected = refined

    if len(selected) < min_required:
        return fallback, [], fallback_pos, fallback_heading
    consensus, scatter, heading_scatter = _stationary_pose_consensus(
        [poses[idx] for idx in selected],
        reference_theta_deg,
        lever_m,
        forward_offset_deg,
    )
    return consensus, selected, scatter, heading_scatter


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


def _anchor_motion_is_complete(
    net_sweep_deg: float,
    direction_consistency: float,
    requested_sweep_deg: float,
) -> bool:
    """Require physical observability before a capture can initialize SLAM.

    Duplicate stationary scans can match perfectly while covering only one
    sensor heading. That is not a map-quality opinion; it is a failed actuator
    motion and cannot initialize an observable SLAM state.
    """
    return (
        abs(float(net_sweep_deg)) >= 0.90 * abs(float(requested_sweep_deg))
        and float(direction_consistency) >= 0.80
    )


def _anchor_pass_is_self_consistent(
    accepted_count: int,
    accepted_ratio: float,
    quality: float,
    net_sweep_deg: float,
    requested_sweep_deg: float,
) -> bool:
    """Whether a complete, coherently matched sweep can initialize SLAM.

    A post-spin stationary matcher is useful additional evidence, but repeated
    geometry or a view dominated by new space can make it inconclusive. It must
    not veto a physically complete sweep whose own sequential scan chain is
    dense and internally consistent.
    """
    return (
        int(accepted_count) >= 12
        and float(accepted_ratio) >= 0.90
        and float(quality) >= 0.75
        and abs(float(net_sweep_deg)) >= 0.90 * abs(float(requested_sweep_deg))
    )


def _anchor_failure_consumes_motion_budget(stop_reason: str) -> bool:
    """Safety interruptions are local-planner events, not actuator failures."""
    return str(stop_reason) != "safety_blocked"


def _anchor_interruption_uses_stationary_fallback(stop_reason: str) -> bool:
    """A footprint stop ends optional body motion, never the SLAM mission."""
    return str(stop_reason) == "safety_blocked"


def _keyframe_batch_is_committable(
    inlier_count: int,
    mean_match_score: float,
    minimum_match_score: float,
    mean_known_support: float,
    minimum_known_support: float,
    position_correction_m: float,
    maximum_position_correction_m: float,
    heading_correction_deg: float,
    maximum_heading_correction_deg: float,
    minimum_inliers: int = 4,
) -> bool:
    """Gate irreversible map writes using measurement and motion-prior evidence."""
    return (
        int(inlier_count) >= max(1, int(minimum_inliers))
        and float(mean_match_score) >= float(minimum_match_score)
        and float(mean_known_support) >= float(minimum_known_support)
        and float(position_correction_m) <= float(maximum_position_correction_m)
        and float(heading_correction_deg) <= float(maximum_heading_correction_deg)
    )


def _keyframe_heading_correction_limit_deg(
    configured_limit_deg: float,
    inlier_count: int,
    batch_count: int,
    mean_match_score: float,
    minimum_match_score: float,
    mean_known_support: float,
) -> float:
    """Permit only a small extra LiDAR yaw correction with strong evidence.

    Relative IMU yaw owns short-horizon heading. Allowing a stationary matcher
    to rewrite it by 15 degrees produced visibly rotated wall fragments even
    when the batch was internally consistent. Larger hypotheses remain valid
    for pose-only recovery, never irreversible map writes.
    """
    required_inliers = max(3, int(math.ceil(0.75 * max(1, int(batch_count)))))
    high_confidence = (
        int(inlier_count) >= required_inliers
        and float(mean_match_score) >= max(12.0, float(minimum_match_score))
        and float(mean_known_support) >= 0.50
    )
    if high_confidence:
        return max(float(configured_limit_deg), 6.0)
    return float(configured_limit_deg)


def _keyframe_match_commit_threshold(
    configured_score: float,
    inlier_count: int,
    batch_count: int,
    mean_known_support: float,
    *,
    sequential_overlap_keyframe: bool = False,
) -> float:
    """Lower the score floor only for a strongly supported stationary batch.

    Scan scores fall when a correct observation contains substantial new space.
    High committed-map overlap plus a >=75% compact pose mode supplies the
    missing confidence, allowing mapping to grow without accepting weak lost-
    tracking batches.
    """
    required_inliers = max(3, int(math.ceil(0.75 * max(1, int(batch_count)))))
    strongly_supported = int(inlier_count) >= required_inliers and float(mean_known_support) >= 0.50
    if sequential_overlap_keyframe and strongly_supported:
        # At a doorway most returns legitimately point into new space, so the
        # global occupied-grid score is not on the same scale as ordinary
        # in-room localization.  The professional SLAM criterion here is a
        # compact multi-scan pose mode plus overlap with the preceding trusted
        # keyframe, not an unrelated global score floor.  Keep a modest score
        # floor as an aliasing guard; the existing support, innovation,
        # position and heading gates still all apply before GOLD is changed.
        return min(float(configured_score), 6.5)
    if strongly_supported:
        return min(float(configured_score), 12.0)
    return float(configured_score)


def _stationary_pose_is_recoverable(
    inlier_count: int,
    mean_match_score: float,
    minimum_match_score: float,
    mean_known_support: float,
    minimum_known_support: float,
    position_correction_m: float,
    maximum_position_correction_m: float,
    heading_correction_deg: float,
    maximum_heading_correction_deg: float,
    minimum_inliers: int = 4,
) -> bool:
    """Accept a coherent pose-only correction without authorizing a map write."""
    return (
        int(inlier_count) >= max(1, int(minimum_inliers))
        and float(mean_match_score) >= float(minimum_match_score)
        and float(mean_known_support) >= float(minimum_known_support)
        and float(position_correction_m) <= float(maximum_position_correction_m)
        and float(heading_correction_deg) <= float(maximum_heading_correction_deg)
    )


def _bidirectional_anchor_is_sufficient(
    bidirectional_verified: bool,
    accepted_count: int,
    accepted_ratio: float,
    max_heading_gap_deg: float,
    minimum_ratio: float,
    maximum_gap_deg: float,
) -> bool:
    """A coherent reverse pass outranks an unavailable stationary refiner.

    ``bidirectional_verified`` already means two independently constructed
    trajectories agreed under the dedicated residual/ratio thresholds. A
    modest gap in one pass is unobserved space, not contradictory geometry;
    requiring that pass to also satisfy the single-pass coverage gate turned a
    38/38, 2cm-consistent map into four needless reacquisitions and an abort.
    """
    del max_heading_gap_deg, maximum_gap_deg
    return (
        bool(bidirectional_verified)
        and int(accepted_count) >= 12
        and float(accepted_ratio) >= float(minimum_ratio)
    )


def _bidirectional_consensus_is_acceptable(
    match_ratio: float,
    centre_p90_m: float,
    heading_p90_deg: float,
    minimum_ratio: float,
    maximum_centre_p90_m: float,
    maximum_heading_p90_deg: float,
    *,
    linear_quantization_slack_m: float = 0.0,
    angular_numeric_slack_deg: float = 0.05,
) -> bool:
    """Resolution-aware dual-pass consistency predicate."""
    return (
        float(match_ratio) + 1e-9 >= float(minimum_ratio)
        and float(centre_p90_m) <= float(maximum_centre_p90_m) + max(0.0, float(linear_quantization_slack_m))
        and float(heading_p90_deg)
        <= float(maximum_heading_p90_deg) + max(0.0, float(angular_numeric_slack_deg))
    )


def _anchor_pass_is_eligible(
    accepted_count: int,
    accepted_ratio: float,
    largest_heading_gap_deg: float,
    minimum_ratio: float,
    maximum_heading_gap_deg: float,
) -> bool:
    """Whether one spin pass has enough coverage to become an anchor candidate."""
    return (
        int(accepted_count) > 0
        and float(accepted_ratio) >= float(minimum_ratio)
        and float(largest_heading_gap_deg) <= float(maximum_heading_gap_deg)
    )


def _stationary_anchor_is_accepted(
    inlier_count: int,
    minimum_inliers: int,
    centre_scatter_m: float,
    maximum_centre_scatter_m: float,
    heading_scatter_deg: float,
    maximum_heading_scatter_deg: float,
) -> bool:
    """Independent stationary-consensus predicate that releases navigation."""
    return (
        int(inlier_count) >= int(minimum_inliers)
        and math.isfinite(float(centre_scatter_m))
        and float(centre_scatter_m) <= float(maximum_centre_scatter_m)
        and math.isfinite(float(heading_scatter_deg))
        and float(heading_scatter_deg) <= float(maximum_heading_scatter_deg)
    )


def _high_confidence_anchor_validation_index(
    scores: list[float],
    reciprocal_supports: list[float],
    minimum_score: float,
    minimum_reciprocal_support: float,
) -> int | None:
    """Best full-view observation strong enough to validate without voting."""
    eligible = [
        index
        for index, (score, support) in enumerate(zip(scores, reciprocal_supports, strict=True))
        if float(score) >= float(minimum_score) and float(support) >= float(minimum_reciprocal_support)
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda index: (reciprocal_supports[index], scores[index]),
    )


def _next_anchor_spin_speed(
    current_speed: float,
    maximum_speed: float,
    step: float = 0.10,
) -> float:
    """Return one bounded stiction-escalation step for an anchor turn."""
    current = abs(float(current_speed))
    maximum = max(current, abs(float(maximum_speed)))
    return min(maximum, current + max(0.01, abs(float(step))))


def _latched_arm_torque_state(stowed: bool) -> dict[str, bool]:
    """Arm flags for base-only streaming after the one-time stow command.

    Position servos retain their last target while torque remains enabled. Do
    not resend all twelve joint targets with every wheel command: on the host,
    arm bus writes precede the base write, so one arm exception otherwise drops
    the entire base command before it reaches the wheels.
    """
    if stowed:
        # Stowing already enabled torque and wrote the target once. Omitting arm
        # fields afterwards preserves that hardware state while keeping every
        # navigation command strictly base-only on the wire.
        return {}
    return {"untorque_left": True, "untorque_right": True}


def _collision_depth_m(hit) -> float:
    """Positive envelope penetration represented by a collision result."""
    if hit is None:
        return 0.0
    return max(0.0, float(hit[4]) - float(hit[3]))


def _filter_isolated_lidar_specks(points_xy: np.ndarray) -> np.ndarray:
    """Remove unsupported single-return specks while preserving surfaces.

    The live LD LiDAR occasionally emits one high-confidence return with no
    angularly adjacent surface support. Those points appeared as transient green
    dots and could distort a one-sweep startup costmap. A real wall/edge has at
    least one nearby return; allow the support radius to grow mildly with range
    so distant walls are not thinned by their larger angular sample spacing.
    """
    points = np.asarray(points_xy)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3:
        return points.copy()
    finite = np.all(np.isfinite(points), axis=1)
    clean = points[finite]
    if len(clean) < 3:
        return clean.copy()
    ranges = np.hypot(clean[:, 0], clean[:, 1])
    support_radius = np.clip(0.055 + 0.018 * ranges, 0.06, 0.16)
    delta = clean[:, None, :] - clean[None, :, :]
    distance_sq = np.sum(delta * delta, axis=2)
    np.fill_diagonal(distance_sq, np.inf)
    supported = np.any(
        distance_sq
        <= np.maximum(
            support_radius[:, None],
            support_radius[None, :],
        )
        ** 2,
        axis=1,
    )
    return clean[supported].astype(points.dtype, copy=False)


def _continuous_passage_keyframe_is_eligible(
    *,
    fresh_scan: bool,
    pose_accepted: bool,
    local_match_valid: bool,
    local_support: float,
    local_innovation_m: float,
    min_support: float = 0.65,
    max_innovation_m: float = 0.08,
) -> bool:
    """Tight, local-only gate for a live passage keyframe.

    This intentionally has no global-map score: doorway scans contain new
    space. Pose continuity comes from the immediately preceding rolling submap
    plus IMU yaw, while bounded innovation prevents a bad relative match from
    entering permanent geometry.
    """
    return (
        bool(fresh_scan)
        and bool(pose_accepted)
        and bool(local_match_valid)
        and float(local_support) >= float(min_support)
        and float(local_innovation_m) <= float(max_innovation_m)
    )


def _moving_keyframe_motion_is_eligible(
    translation_m: float,
    heading_change_deg: float,
    *,
    minimum_translation_m: float = 0.20,
    maximum_heading_change_deg: float = 5.0,
) -> bool:
    """Allow permanent moving keyframes only on well-spaced straight motion.

    A full LiDAR revolution collected through a large turn is not rigid.  The
    explorer therefore promotes a moving revolution only after useful forward
    translation and while its immutable IMU-held segment heading stayed nearly
    constant.  Turning geometry is acquired by stopped angular keyframes.
    """
    return float(translation_m) >= float(minimum_translation_m) and abs(float(heading_change_deg)) <= float(
        maximum_heading_change_deg
    )


def _obstacle_pose_fix_is_corroborated(
    *,
    fresh_scan: bool,
    pose_accepted: bool,
    local_match_valid: bool,
    global_match_valid: bool,
    local_innovation_m: float,
    local_global_agreement_m: float,
    max_local_innovation_m: float = 0.08,
    max_local_global_agreement_m: float = 0.12,
) -> bool:
    """Whether one tracking update may restore world-frame obstacle trust.

    Planner obstacles are transformed through the current SLAM pose.  A good
    LiDAR return at a bad pose is therefore *not* evidence of a new obstacle.
    Require the rolling local odometry and the permanent-map solve to agree
    before allowing a pose to contribute to the temporary obstacle costmap.
    """
    return (
        bool(fresh_scan)
        and bool(pose_accepted)
        and bool(local_match_valid)
        and bool(global_match_valid)
        and float(local_innovation_m) <= float(max_local_innovation_m)
        and float(local_global_agreement_m) <= float(max_local_global_agreement_m)
    )


def _new_known_cells(
    known_before: np.ndarray,
    origin_before: np.ndarray,
    known_after: np.ndarray,
    origin_after: np.ndarray,
    _resolution_m: float,
) -> int:
    """Count occupancy knowledge added by one map transaction.

    Grid expansion shifts array coordinates, so a direct Boolean subtraction is
    valid only when shape and origin stayed fixed. Total known-cell growth is
    the conservative equivalent when the grid resized.
    """
    if known_after.shape == known_before.shape and np.allclose(origin_after, origin_before):
        return int(np.count_nonzero(known_after & ~known_before))
    return max(0, int(known_after.sum()) - int(known_before.sum()))


def _translation_escape_is_safe(
    points_forward_xy: np.ndarray,
    profile: dict | None,
    translation_forward_xy: np.ndarray,
    *,
    lidar_offset_forward_m: float,
    physical_body_radius_m: float,
    self_mask_inset_m: float,
    samples: int = 8,
) -> tuple[bool, float]:
    """Check that a short body translation monotonically exits an intrusion.

    This is a local trajectory rollout over the calibrated footprint. Static
    world points move opposite the proposed robot translation. Returns attached
    to the chassis are removed before rollout so they are not incorrectly
    treated as stationary obstacles.
    """
    points = np.asarray(points_forward_xy, dtype=np.float64)
    delta = np.asarray(translation_forward_xy, dtype=np.float64)
    if (
        not profile
        or points.ndim != 2
        or points.shape[1:] != (2,)
        or not len(points)
        or delta.shape != (2,)
        or float(np.hypot(*delta)) < 0.01
    ):
        return False, 0.0
    if float(physical_body_radius_m) > 0.0:
        self_returns = _physical_body_self_return_mask(
            points,
            lidar_offset_forward_m=lidar_offset_forward_m,
            physical_body_radius_m=physical_body_radius_m,
            self_mask_inset_m=self_mask_inset_m,
        )
        points = points[~self_returns]
    if not len(points):
        return False, 0.0
    geometry = {
        "lidar_offset_forward_m": float(lidar_offset_forward_m),
        "physical_body_radius_m": float(physical_body_radius_m),
        "self_mask_inset_m": float(self_mask_inset_m),
    }
    current = _collision_box_violation(points, profile, **geometry)
    current_depth = _collision_depth_m(current)
    if current is None or current_depth <= 0.0:
        return False, 0.0
    final_depth = current_depth
    for fraction in np.linspace(1.0 / max(2, int(samples)), 1.0, max(2, int(samples))):
        projected = points - float(fraction) * delta
        hit = _collision_box_violation(projected, profile, **geometry)
        depth = _collision_depth_m(hit)
        # Never approve a maneuver that first drives farther into contact.
        if depth > current_depth + 0.003:
            return False, current_depth - depth
        final_depth = depth
    improvement = current_depth - final_depth
    # A motion parallel to the obstacle can be a valid first step in a local
    # recovery sequence. Its execution must be verified by scan odometry; the
    # footprint rollout's job is to prove that it never worsens penetration.
    return final_depth <= current_depth + 0.003, improvement


def _translation_trajectory_is_safe(
    points_forward_xy: np.ndarray,
    profile: dict | None,
    translation_forward_xy: np.ndarray,
    *,
    lidar_offset_forward_m: float,
    physical_body_radius_m: float,
    self_mask_inset_m: float,
    samples: int = 8,
) -> bool:
    """Collision-check a short body translation from clear or intruding state."""
    if not profile:
        return False
    points = np.asarray(points_forward_xy, dtype=np.float64)
    delta = np.asarray(translation_forward_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or not len(points) or delta.shape != (2,):
        return False
    geometry = {
        "lidar_offset_forward_m": float(lidar_offset_forward_m),
        "physical_body_radius_m": float(physical_body_radius_m),
        "self_mask_inset_m": float(self_mask_inset_m),
    }
    initial_depth = _collision_depth_m(_collision_box_violation(points, profile, **geometry))
    permitted_depth = initial_depth + 0.003
    for fraction in np.linspace(1.0 / max(2, int(samples)), 1.0, max(2, int(samples))):
        projected = points - float(fraction) * delta
        depth = _collision_depth_m(_collision_box_violation(projected, profile, **geometry))
        if depth > permitted_depth:
            return False
    return True


def _points_in_rotated_body_frame(
    points_forward_xy: np.ndarray,
    body_turn_deg: float,
    *,
    lidar_offset_forward_m: float = 0.0,
) -> np.ndarray:
    """Express stationary world returns after a turn about the robot centre."""
    points = np.asarray(points_forward_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,):
        return np.empty((0, 2), dtype=np.float64)
    angle = math.radians(float(body_turn_deg))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    # LiDAR-frame +x becomes robot-centre +x after adding the forward lever.
    # Rotate stationary world points by the inverse body turn about that centre,
    # then express them from the offset LiDAR at the new body heading.
    centred_x = points[:, 0] + float(lidar_offset_forward_m)
    turned = np.column_stack(
        (
            cosine * centred_x + sine * points[:, 1],
            -sine * centred_x + cosine * points[:, 1],
        )
    )
    turned[:, 0] -= float(lidar_offset_forward_m)
    return turned


def _widest_visible_corridor_bearing_deg(
    points_forward_xy: np.ndarray,
    *,
    lidar_offset_forward_m: float,
    corridor_half_width_m: float,
    maximum_range_m: float = 2.0,
) -> tuple[float, float] | None:
    """Return the widest forward-hemisphere LiDAR corridor and its range."""
    points = np.asarray(points_forward_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or not len(points):
        return None
    centred = points.copy()
    centred[:, 0] += float(lidar_offset_forward_m)
    corridor_half_width = max(0.05, float(corridor_half_width_m))
    maximum_range = max(0.20, float(maximum_range_m))
    ranked: list[tuple[float, float, float]] = []
    for bearing in range(-90, 91, 5):
        angle = math.radians(float(bearing))
        direction = np.array([math.cos(angle), math.sin(angle)])
        lateral_axis = np.array([-direction[1], direction[0]])
        longitudinal = centred @ direction
        lateral = np.abs(centred @ lateral_axis)
        in_corridor = (
            (longitudinal > 0.03) & (longitudinal <= maximum_range) & (lateral <= corridor_half_width)
        )
        visible_range = float(np.min(longitudinal[in_corridor])) if np.any(in_corridor) else maximum_range
        ranked.append((visible_range, -abs(float(bearing)), float(bearing)))
    if not ranked:
        return None
    visible_range, _forward_preference, bearing = max(ranked)
    return bearing, visible_range


def _startup_escape_velocity(
    translation_forward_xy: np.ndarray,
    requested_speed: float,
    *,
    stiction_floor: float = 0.80,
) -> np.ndarray:
    """Actuator-aware direction-preserving velocity for a startup escape.

    Normalizing a diagonal 0.80 command produces 0.57 on each body axis,
    below Sourccey's measured wheel stiction threshold. The planner's startup
    escape is scaled by its largest body-axis component instead. This preserves
    arbitrary travel bearings while ensuring at least one component reaches the
    effective drivetrain magnitude.
    """
    delta = np.asarray(translation_forward_xy, dtype=np.float64)
    if delta.shape != (2,) or float(np.hypot(*delta)) < 1e-9:
        return np.zeros(2, dtype=np.float64)
    magnitude = min(
        1.0,
        max(abs(float(requested_speed)), abs(float(stiction_floor))),
    )
    return delta / float(np.max(np.abs(delta))) * magnitude


def _full_pivot_clearance_margin_m(
    points_forward_xy: np.ndarray,
    profile: dict | None,
    *,
    lidar_offset_forward_m: float,
    physical_body_radius_m: float,
    self_mask_inset_m: float,
) -> float:
    """Robust radial clearance beyond the envelope swept by a full pivot."""
    if not profile:
        return -math.inf
    points = np.asarray(points_forward_xy, dtype=np.float64)
    ranges = _collision_box_effective_ranges(profile)
    if points.ndim != 2 or points.shape[1:] != (2,) or not len(points) or not len(ranges):
        return -math.inf
    bin_size = float(profile.get("bin_size_deg", 0.0))
    if bin_size <= 0.0:
        return -math.inf
    angles = np.radians(-180.0 + (np.arange(len(ranges)) + 0.5) * bin_size)
    valid = np.isfinite(ranges)
    if not np.any(valid):
        return -math.inf
    boundary_x = ranges[valid] * np.cos(angles[valid]) + float(lidar_offset_forward_m)
    boundary_y = ranges[valid] * np.sin(angles[valid])
    radial_padding = max(
        0.0,
        float(profile.get("safety_margin_m", 0.02)) - float(profile.get("noise_tolerance_m", 0.02)),
    )
    swept_radius = float(np.max(np.hypot(boundary_x, boundary_y))) + radial_padding
    self_returns = _physical_body_self_return_mask(
        points,
        lidar_offset_forward_m=lidar_offset_forward_m,
        physical_body_radius_m=physical_body_radius_m,
        self_mask_inset_m=self_mask_inset_m,
    )
    body_distance = np.hypot(
        points[:, 0] + float(lidar_offset_forward_m),
        points[:, 1],
    )
    environmental = np.sort(body_distance[~self_returns])
    evidence_points = max(
        2,
        min(
            int(profile.get("min_violation_points", 3)),
            int(profile.get("side_min_violation_points", 2)),
        ),
    )
    if len(environmental) < evidence_points:
        return math.inf
    return float(environmental[evidence_points - 1] - swept_radius)


def _keyframe_view_is_diverse(
    centre_xy: np.ndarray,
    heading_deg: float,
    previous_centre_xy: np.ndarray,
    previous_heading_deg: float | None,
    min_translation_m: float = 0.12,
    min_heading_deg: float = 12.0,
) -> bool:
    """Whether a stationary view adds useful parallax or angular coverage."""
    if previous_heading_deg is None:
        return True
    translated = float(
        np.hypot(
            *(np.asarray(centre_xy, dtype=np.float64) - np.asarray(previous_centre_xy, dtype=np.float64))
        )
    )
    turned = abs((float(heading_deg) - float(previous_heading_deg) + 180.0) % 360.0 - 180.0)
    return translated >= float(min_translation_m) or turned >= float(min_heading_deg)


def _representative_keyframe_indices(
    scores: list[float],
    inlier_indices: list[int],
    *,
    max_keyframes: int = 2,
) -> list[int]:
    """Select the strongest compact subset from a validated stationary batch.

    Several revolutions at one stopped pose are useful for consensus, but adding
    all of them to the permanent localization map creates no parallax and
    overweights one viewpoint.  Two high-score representatives provide the two
    occupancy hits needed by the log-odds grid while keeping the GOLD reference
    compact, as a keyframe/submap SLAM front end would.
    """
    valid = [int(index) for index in inlier_indices if 0 <= int(index) < len(scores)]
    limit = max(1, int(max_keyframes))
    return sorted(valid, key=lambda index: float(scores[index]), reverse=True)[:limit]


def _bounded_heading_step(error_deg: float, max_step_deg: float) -> float:
    """One signed, non-overshooting step for an overlap-preserving turn."""
    error = float(error_deg)
    limit = max(1.0, abs(float(max_step_deg)))
    if abs(error) <= limit:
        return error
    return math.copysign(limit, error)


def _single_startup_escape_waypoint(
    waypoints: list[np.ndarray],
    *,
    maximum_distance_m: float = 0.45,
) -> list[np.ndarray]:
    """Collapse a temporary-map route to one bounded initial straight segment.

    ExploreSystem's constrained-start contract is not general navigation. The
    temporary map only needs to move the chassis once into more open space so a
    complete replacement spin can begin. Following the entire A* route here
    previously produced several forward movements before the initial map even
    existed.
    """
    if not waypoints:
        return []
    first = np.asarray(waypoints[0], dtype=np.float64)
    if first.shape != (2,):
        return []
    distance = float(np.hypot(*first))
    if not math.isfinite(distance) or distance < 1e-6:
        return []
    bounded_distance = min(distance, max(0.05, float(maximum_distance_m)))
    return [first * (bounded_distance / distance)]


def _active_localization_rotation_stalled(progress_deg: float) -> bool:
    """Whether a commanded probe failed to acquire useful angular parallax."""
    return float(progress_deg) < 2.0


def _bounded_turn_status(
    commanded_delta_deg: float,
    actual_delta_deg: float,
    *,
    wrong_way_limit_deg: float = 3.0,
    completion_tolerance_deg: float = 3.0,
    overshoot_margin_deg: float = 5.0,
) -> str:
    """Classify signed progress of a deliberately bounded pivot.

    A timeout alone is not a sufficient motion watchdog: if a host interprets
    the angular command with the opposite sign, waiting for the timeout makes
    the base rotate *away* from its target for several seconds.  Stop after a
    few degrees of signed disagreement and let the caller correct the measured
    command sign.  This check is independent of scan matching and therefore
    remains valid while localization itself is uncertain.
    """
    command = float(commanded_delta_deg)
    actual = float(actual_delta_deg)
    if abs(command) < 1e-6:
        return "complete"
    signed_progress = math.copysign(1.0, command) * actual
    if signed_progress <= -abs(float(wrong_way_limit_deg)):
        return "wrong_direction"
    if signed_progress >= abs(command) + abs(float(overshoot_margin_deg)):
        return "overshoot"
    if signed_progress >= max(
        0.0,
        abs(command) - abs(float(completion_tolerance_deg)),
    ):
        return "complete"
    return "turning"


def _pivot_progress_stalled(
    now_s: float,
    last_progress_s: float,
    timeout_s: float = 2.5,
) -> bool:
    """Navigation2-style progress timeout for a commanded in-place turn."""
    return float(now_s) - float(last_progress_s) >= max(0.25, float(timeout_s))


def _max_heading_gap_deg(headings_deg: list[float]) -> float:
    """Largest uncovered arc in a circular set of scan headings."""
    angles = np.sort(np.mod(np.asarray(headings_deg, dtype=np.float64), 360.0))
    if len(angles) < 2:
        return 360.0
    gaps = np.diff(np.concatenate([angles, angles[:1] + 360.0]))
    return float(np.max(gaps))


def _anchor_heading_gap_deg(
    headings_deg: list[float],
    *,
    stationary_full_revolutions: bool = False,
    observable_sensor_arc_deg: float = 180.0,
) -> float:
    """Unobserved angular gap after accounting for one LiDAR revolution.

    The LD LiDAR mechanically reports a full revolution, but on Sourccey the
    chassis blocks roughly half of that view. Repeating stationary revolutions
    proves measurement consistency; it does not reveal the occluded half of the
    room. A collision-aware fan therefore needs about 180 degrees of distinct
    body headings, while one stationary heading still has about 180 degrees of
    genuinely unobserved geometry.
    """
    heading_gap = _max_heading_gap_deg(headings_deg)
    if stationary_full_revolutions and headings_deg:
        return max(
            0.0,
            heading_gap - min(360.0, max(0.0, float(observable_sensor_arc_deg))),
        )
    return heading_gap


def _anchor_validation_support_ratio(
    scan_to_map_support: float,
    map_to_scan_support: float,
    *,
    body_occluded_sensor: bool = True,
) -> float:
    """Observation support used for independent anchor validation.

    A stationary chassis-occluded scan should be explained by the completed
    submap, but it cannot be required to reproduce the portions of that submap
    observed from other body headings. Requiring reciprocal whole-map support
    rejects exactly the complete multi-heading anchors we want. An unobstructed
    360-degree sensor may still use the stricter reciprocal score.
    """
    scan_support = float(scan_to_map_support)
    if body_occluded_sensor:
        return scan_support
    return min(scan_support, float(map_to_scan_support))


def _p90_and_max(values: np.ndarray | list[float]) -> tuple[float, float]:
    """Robust consensus residual plus its diagnostic worst case."""
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return float("inf"), float("inf")
    return float(np.percentile(array, 90.0)), float(np.max(array))


# ---------------------------------------------------------------------------
# LOG-ODDS OCCUPANCY GRID — the professional world model (Cartographer/nav2
# style). One probabilistic grid that the planner, renderer, novelty check and
# frontier analysis ALL read. Every integrated scan updates it symmetrically:
# each return RAISES its cell's log-odds (marking), and every ray LOWERS all the
# cells it passed through (clearing). This one mechanism replaces the vote
# filter and the ad-hoc unknown-ring fixes. Dynamic obstacles remain in a
# separate decaying planner layer so they cannot corrupt SLAM geometry:
#   * sparse-but-real edges accumulate evidence and stay occupied (the planner
#     can no longer be blind to something the stop-box sees),
#   * a departed person's cells are cleared by the rays that now pass through
#     where they stood (no timers),
#   * a few misaligned points cannot flip cells that hundreds of rays cleared.
# ===========================================================================
# MAP MODEL — WRITTEN BY PHASES 1.2 AND 4, READ BY PHASES 2 AND 3
# ===========================================================================


class OccupancyGrid:
    L_HIT = 0.85  # log-odds added per return in a cell   (p≈0.70)
    L_MISS = 0.40  # log-odds removed per ray through a cell (p≈0.40)
    L_MIN, L_MAX = -4.0, 4.0
    OCC_T = 1.2  # occupied when log-odds >= this (≈2 net hits)
    FREE_T = -0.8  # free when log-odds <= this      (≈2 net clears)
    # Range-weighted MARKING: this lidar's returns at 4 m are several times
    # sparser and noisier than at 1 m (angular spread, beam divergence, shallow
    # wall incidence), so far snapshots paint fat fuzzy walls that a later
    # close pass then has to out-vote.  Scaling hit log-odds by the return's
    # range makes one full-strength clearing ray dominate any far-range mark
    # by construction, while far-only cells stay on the map (weak but present
    # — supersede, not discard).  Clearing stays FULL strength: free space
    # along a ray is real regardless of endpoint noise, and range-weighting
    # the misses starved the planner of traversable cells (field 2026-08-05:
    # repeated "map edge ahead" stalls with only 1-2 committed scans per
    # checkpoint).
    RANGE_W_NEAR_M = 1.5  # full-strength marking inside this range
    RANGE_W_MIN = 0.25  # marking floor for far-range returns
    MARK_MAX_RANGE_M = 5.0  # beyond this a return clears free space, marks nothing

    def __init__(self, res_m: float) -> None:
        self.res = float(res_m)
        self.origin = np.array([-6.0, -6.0], dtype=np.float64)  # cell (0,0) corner
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

    def _ij(self, pts_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        jj = ((pts_xy[:, 0] - self.origin[0]) / self.res).astype(np.int64)
        ii = ((pts_xy[:, 1] - self.origin[1]) / self.res).astype(np.int64)
        h, w = self.L.shape
        ok = (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)
        return ii[ok], jj[ok], ok

    def integrate_scan(
        self,
        origin_xy: np.ndarray,
        world_pts: np.ndarray,
        footprint_clear_m: float = 0.0,
        preserve_confirmed_occupied: bool = True,
    ) -> None:
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
            mask = f[None, :] < (d[:, None] - 1.5 * self.res)  # stop short of the hit
            px = origin_xy[0] + unit[:, 0:1] * f[None, :]
            py = origin_xy[1] + unit[:, 1:2] * f[None, :]
            ii, jj, ok = self._ij(np.column_stack([px[mask], py[mask]]))
            # A later, slightly displaced scan must not erase a wall that the
            # verified anchor already established.  The old log-odds update
            # let three otherwise valid viewpoints ray-clear a wall whenever
            # their pose differed by a few cells, producing the "vanished
            # wall" map seen on 2026-08-04.  Dynamic/live obstacle evidence
            # is handled outside this permanent map; confirmed static geometry
            # is monotonic within one exploration run.
            if preserve_confirmed_occupied and len(ii):
                clear = self.L[ii, jj] < self.OCC_T
                ii, jj = ii[clear], jj[clear]
            np.subtract.at(self.L, (ii, jj), self.L_MISS)
        near = dist <= self.MARK_MAX_RANGE_M
        if np.any(near):
            w_hit = np.clip(
                self.RANGE_W_NEAR_M / np.maximum(dist[near], 1e-9),
                self.RANGE_W_MIN,
                1.0,
            )
            ii, jj, ok = self._ij(world_pts[near])
            np.add.at(self.L, (ii, jj), self.L_HIT * w_hit[ok])
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
        ii, jj, _ = self._ij(world_pts)
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
    local_xy: np.ndarray  # (fwd, lat) points in the lidar frame
    pose: Pose2D  # solved LIDAR pose in the world frame
    world_xy: np.ndarray  # local points transformed by pose
    gold: bool  # True = part of the LOCALIZATION reference
    # Pose-graph attachment: the scan rides rigidly on this node.  When the
    # graph optimizer moves the node, the scan's pose/world points move by
    # the same rigid delta and the grid is re-rendered.  node_id -1 = not
    # attached (anchor-spin scans: the fixed gauge that everything else is
    # optimized against).
    node_id: int = -1
    node_pose: Pose2D | None = None  # node pose at the moment of attachment


class WorldMap:
    """Every aligned scan feeds two consumers — with an integrity firewall.

    The matcher's reference is GOLD-ONLY: the anchor spin plus viewpoint batches
    that passed the consistency gate. Provisional (driving) scans may shape the
    planner's grid, but they can NEVER become geometry that localization aligns
    to — one mis-integrated scan in the reference otherwise seeds the next match,
    and the drift compounds until the map is unrecoverable (field 2026-07-21:
    the hallway painted as three crossing ghost copies, then stationary matches
    collapsed to 3.8 — 'once the map is lost it's game over')."""

    def __init__(
        self,
        grid_res_m: float,
        footprint_clear_m: float,
        *,
        lidar_offset_m: float = 0.0,
        forward_offset_deg: float = 0.0,
    ) -> None:
        self.scans: list[MapScan] = []
        self.grid = OccupancyGrid(grid_res_m)
        self.footprint_clear_m = float(footprint_clear_m)
        self.lidar_offset_m = float(lidar_offset_m)
        self.forward_offset_deg = float(forward_offset_deg)
        self._ref_cache: np.ndarray | None = None
        # A clean keyframe is collected at a stationary checkpoint or after a
        # short, bounded translation.  A multi-metre jump between adjacent
        # map poses is therefore never a real motion; it is a bad scan-match
        # hypothesis.  Keep this firewall at the irreversible map-write point
        # so one erroneous correction cannot seed every later localization or
        # smear the displayed occupancy map.
        self.max_gold_pose_step_m = 0.80
        self.max_moving_heading_step_deg = 55.0
        # Bumped whenever the pose-graph optimizer retroactively moves scan
        # poses.  Consumers caching anything derived from scan POSES (not
        # just scan COUNT) must key on this as well.
        self.pose_revision = 0
        # Optional external heading aligner, set by the explorer once the
        # anchor is validated.  Called with the candidate pose right before
        # the irreversible map write; returns the pose to commit.  This is
        # the IMU's role at the one chokepoint that matters: the gyro is the
        # directional reference for where new segments land on the board, so
        # a slightly rotated scan-match is straightened rather than allowed
        # to slant the map (and rather than being refused — the map must keep
        # growing).
        self.heading_align = None

    def add(
        self,
        local_xy: np.ndarray,
        pose: Pose2D,
        *,
        gold: bool = False,
        stationary: bool = False,
        verified_stationary_gap: bool = False,
    ) -> bool:
        """Integrate one scan after its pose has been validated.

        ``stationary`` marks a pose which has just passed the independent
        multi-scan stopped-batch validation.  Such a keyframe may legitimately
        be a metre away from the prior keyframe and face a different heading;
        applying a *moving* scan continuity test to it would reject valid map
        growth after every turn. ``verified_stationary_gap`` is deliberately narrow: it is allowed only
        for a non-GOLD scan captured while the base is stopped, after an
        independent multi-scan full-view LiDAR consensus has already verified
        that pose.  It lets the occupancy map grow when the robot crosses from
        an old room into genuinely new geometry.  The normal adjacent-scan
        jump firewall remains in force for every moving and GOLD scan, and the
        resulting scan can never enter the localization reference.
        """
        if gold and verified_stationary_gap:
            raise ValueError("A verified stationary gap scan must remain non-GOLD")
        if self.scans:
            previous_scan = self.scans[-1]
            if previous_scan is not None:
                current_centre = _robot_centre_from_lidar_pose(
                    pose,
                    self.lidar_offset_m,
                    self.forward_offset_deg,
                )
                previous_centre = _robot_centre_from_lidar_pose(
                    previous_scan.pose,
                    self.lidar_offset_m,
                    self.forward_offset_deg,
                )
                step_m = float(np.hypot(*(current_centre - previous_centre)))
                heading_step_deg = abs(
                    (float(pose.theta_deg) - float(previous_scan.pose.theta_deg) + 180.0) % 360.0
                    - 180.0
                )
                if not math.isfinite(step_m) or (
                    not stationary
                    and not verified_stationary_gap
                    and step_m > self.max_gold_pose_step_m
                ):
                    print(
                        f"[map] rejected {'GOLD' if gold else 'provisional'} scan: "
                        "impossible pose jump "
                        f"{step_m:.2f}m (limit {self.max_gold_pose_step_m:.2f}m); "
                        "permanent occupancy/reference unchanged."
                    )
                    return False
                if (
                    not stationary
                    and not verified_stationary_gap
                    and
                    step_m > 0.25
                    and heading_step_deg > self.max_moving_heading_step_deg
                ):
                    print(
                        f"[map] rejected {'GOLD' if gold else 'provisional'} scan: "
                        f"inconsistent moving pose ({step_m:.2f}m/{heading_step_deg:.1f}deg); "
                        "permanent occupancy/reference unchanged."
                    )
                    return False
        if self.heading_align is not None:
            if not stationary and not verified_stationary_gap:
                # Stops-only mapping (user directive 2026-08-05): once the
                # anchor is established (heading_align attached), scans taken
                # while the base is MOVING serve odometry only.  The 22
                # moving keyframes of the evening run are exactly where the
                # transposed doorframes and hallway drift entered the map —
                # a moving scan cannot be paired with a same-instant yaw.
                self._moving_refusals = getattr(self, "_moving_refusals", 0) + 1
                if self._moving_refusals <= 3 or self._moving_refusals % 20 == 0:
                    print(
                        "[map] moving scan kept out of the permanent map "
                        f"(#{self._moving_refusals}); occupancy grows only from "
                        "stationary stop-scan-go captures."
                    )
                return False
            pose = self.heading_align(pose, local_xy)
        world = _transform_points(local_xy, pose)
        self.scans.append(MapScan(local_xy=local_xy, pose=pose, world_xy=world, gold=gold))
        robot_centre = _robot_centre_from_lidar_pose(pose, self.lidar_offset_m, self.forward_offset_deg)
        self.grid.integrate_scan(
            np.array([float(pose.x), float(pose.y)]),
            world,
            footprint_clear_m=0.0,
        )
        self.grid.assert_free_disk(robot_centre, self.footprint_clear_m)
        if gold:
            self._ref_cache = None
        return True

    def attach_recent_scans_to_node(
        self, node_id: int, node_pose: Pose2D, since_index: int
    ) -> int:
        """Bind scans committed since ``since_index`` to a pose-graph node.

        The scans ride rigidly on the node from now on: when the optimizer
        moves the node, ``apply_node_poses`` moves them by the same delta.
        """
        attached = 0
        for index in range(max(0, int(since_index)), len(self.scans)):
            scan = self.scans[index]
            if scan.node_id >= 0:
                continue
            scan.node_id = int(node_id)
            scan.node_pose = node_pose
            attached += 1
        return attached

    def apply_node_poses(
        self, node_poses: dict[int, tuple[float, float, float]]
    ) -> float:
        """Retroactively move attached scans to their nodes' optimized poses.

        This is the heart of the pose-graph architecture: scan poses are no
        longer irreversible facts — the grid and render points are re-derived
        from the corrected scan list.  Anchor scans (node_id -1) never move;
        they are the gauge.  Returns the largest position delta applied (m).
        """
        deltas: dict[int, tuple[np.ndarray, float, np.ndarray, Pose2D]] = {}
        for node_id, (nx, ny, nth) in node_poses.items():
            reference = next(
                (s.node_pose for s in self.scans if s.node_id == node_id), None
            )
            if reference is None:
                continue
            d_theta = ((nth - float(reference.theta_deg) + 180.0) % 360.0) - 180.0
            rad = math.radians(d_theta)
            rot = np.asarray(
                [[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]]
            )
            old_xy = np.asarray([float(reference.x), float(reference.y)])
            new_xy = np.asarray([float(nx), float(ny)])
            if float(np.hypot(*(new_xy - old_xy))) < 1e-4 and abs(d_theta) < 5e-3:
                continue
            deltas[node_id] = (rot, d_theta, new_xy - rot @ old_xy, Pose2D(nx, ny, nth))
        if not deltas:
            return 0.0
        moved_max = 0.0
        gold_touched = False
        for index, scan in enumerate(self.scans):
            entry = deltas.get(scan.node_id)
            if entry is None:
                continue
            rot, d_theta, shift, new_node_pose = entry
            old = np.asarray([float(scan.pose.x), float(scan.pose.y)])
            new = rot @ old + shift
            corrected = Pose2D(
                float(new[0]), float(new[1]), float(scan.pose.theta_deg) + d_theta
            )
            moved_max = max(moved_max, float(np.hypot(*(new - old))))
            self.scans[index] = MapScan(
                local_xy=scan.local_xy,
                pose=corrected,
                world_xy=_transform_points(scan.local_xy, corrected),
                gold=scan.gold,
                node_id=scan.node_id,
                node_pose=new_node_pose,
            )
            gold_touched = gold_touched or scan.gold
        self._rebuild_grid_from_scans()
        if gold_touched:
            self._ref_cache = None
        self.pose_revision += 1
        return moved_max

    def _rebuild_grid_from_scans(self) -> None:
        """Re-render the occupancy grid from the (possibly re-posed) scans.

        Planner marks (``mark_hits``) are not replayed: a re-render follows a
        retroactive pose improvement, under which previously inserted marks'
        world positions are exactly what was corrected.
        """
        rebuilt = OccupancyGrid(self.grid.res)
        for scan in self.scans:
            rebuilt.integrate_scan(
                np.array([float(scan.pose.x), float(scan.pose.y)]),
                scan.world_xy,
                footprint_clear_m=0.0,
            )
            rebuilt.assert_free_disk(
                _robot_centre_from_lidar_pose(
                    scan.pose, self.lidar_offset_m, self.forward_offset_deg
                ),
                self.footprint_clear_m,
            )
        rebuilt.version = self.grid.version + 1
        self.grid = rebuilt

    def reference(self, max_pts: int = 9000) -> np.ndarray:
        """Subsampled GOLD world points for scan-matching against."""
        if self._ref_cache is None:
            sets = [s.world_xy for s in self.scans if s.gold and len(s.world_xy)]
            self._ref_cache = np.concatenate(sets, axis=0) if sets else np.zeros((0, 2), dtype=np.float32)
        ref = self._ref_cache
        if len(ref) > max_pts:
            ref = ref[:: (len(ref) // max_pts) + 1]
        return ref

    def local_gold_reference(
        self,
        centre_xy: np.ndarray,
        *,
        pose_radius_m: float,
        max_pts: int = 9000,
    ) -> np.ndarray:
        """Return permanent geometry observed from the current pose bubble.

        A recovery after a 180-degree turn is *not* a global kidnapped-robot
        search.  The robot has a trustworthy recent position prior, and the
        useful reference is therefore the GOLD scans previously captured from
        nearby robot positions.  Keeping that local observation set intact is
        important: a distinctive doorway behind the robot must compete with
        its matching doorway points, rather than with every repeated hallway
        wall in the whole building.

        ``pose_radius_m`` applies to the *LiDAR origins* of GOLD scans, not to
        their endpoints.  A scan taken near the robot may legitimately contain
        a wall or door several metres away, so trimming its endpoints would
        throw away exactly the distant shape needed for orientation recovery.
        """
        centre = np.asarray(centre_xy, dtype=np.float64)
        radius = max(0.0, float(pose_radius_m))
        sets: list[np.ndarray] = []
        for scan in self.scans:
            if not scan.gold or not len(scan.world_xy):
                continue
            scan_centre = _robot_centre_from_lidar_pose(
                scan.pose,
                self.lidar_offset_m,
                self.forward_offset_deg,
            )
            if float(np.hypot(*(scan_centre - centre))) <= radius:
                sets.append(scan.world_xy)
        if not sets:
            return np.zeros((0, 2), dtype=np.float32)
        ref = np.concatenate(sets, axis=0)
        if len(ref) > max_pts:
            ref = ref[:: (len(ref) // max_pts) + 1]
        return ref

    def nearest_gold_scan(self, centre_xy: np.ndarray) -> tuple[float, np.ndarray | None]:
        """Distance to, and robot centre of, the nearest GOLD scan origin.

        Recovery needs to know WHERE trustworthy reference geometry actually
        is: deep in freshly explored space the map is all provisional
        snapshots, and a recovery walk bounded by anything other than the real
        GOLD distance can exhaust itself without ever seeing the reference.
        """
        centre = np.asarray(centre_xy, dtype=np.float64)
        best_distance = float("inf")
        best_centre: np.ndarray | None = None
        for scan in self.scans:
            if not scan.gold or not len(scan.world_xy):
                continue
            scan_centre = _robot_centre_from_lidar_pose(
                scan.pose,
                self.lidar_offset_m,
                self.forward_offset_deg,
            )
            distance = float(np.hypot(*(scan_centre - centre)))
            if distance < best_distance:
                best_distance = distance
                best_centre = scan_centre
        return best_distance, best_centre

    def render_points(self) -> tuple[np.ndarray, np.ndarray]:
        """(points, scan_ids) for display: the colourful raw points, kept only
        where the occupancy grid says OCCUPIED (grid truth decides what is real;
        the palette look stays)."""
        sets = [s.world_xy for s in self.scans if len(s.world_xy)]
        if not sets:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        pts = np.concatenate(sets, axis=0)
        ids = np.concatenate([np.full(len(a), i, dtype=np.int64) for i, a in enumerate(sets)])
        occ = _inflate(self.grid.occupied(), 1)  # 1-cell tolerance for point jitter
        g = self.grid
        jj = ((pts[:, 0] - g.origin[0]) / g.res).astype(np.int64)
        ii = ((pts[:, 1] - g.origin[1]) / g.res).astype(np.int64)
        h, w = occ.shape
        ok = (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)
        keep = np.zeros(len(pts), dtype=bool)
        keep[ok] = occ[ii[ok], jj[ok]]
        return pts[keep], ids[keep]


@dataclass(slots=True)
class LocalOdomScan:
    """One scan retained only by the continuous local-odometry front end."""

    local_xy: np.ndarray
    pose: Pose2D
    world_xy: np.ndarray


class RollingLocalSubmap:
    """Small replaceable scan submap used for continuous LiDAR odometry.

    This is deliberately separate from ``WorldMap``. Moving scans are useful
    for estimating relative motion but are never authoritative enough to alter
    permanent occupancy. Keeping the last few overlapping scans gives the pose
    estimator a nearby reference even when a new doorway view has little
    overlap with the global anchor map.
    """

    def __init__(self, max_scans: int = 10, max_points: int = 7000) -> None:
        self.max_scans = max(2, int(max_scans))
        self.max_points = max(500, int(max_points))
        self.scans: deque[LocalOdomScan] = deque(maxlen=self.max_scans)

    def clear(self) -> None:
        self.scans.clear()

    def add(self, local_xy: np.ndarray, pose: Pose2D) -> None:
        local = np.asarray(local_xy, dtype=np.float32)
        if len(local) < 12:
            return
        self.scans.append(LocalOdomScan(local, pose, _transform_points(local, pose)))

    def seed_from_world_map(self, world_map: WorldMap) -> None:
        self.clear()
        for scan in [scan for scan in world_map.scans if scan.gold][-self.max_scans :]:
            self.scans.append(LocalOdomScan(scan.local_xy, scan.pose, scan.world_xy))

    def reset(self, local_xy: np.ndarray, pose: Pose2D) -> None:
        self.clear()
        self.add(local_xy, pose)

    def translate(self, delta_xy: np.ndarray) -> None:
        """Move the complete local odometry frame without discarding history.

        A saved-map backend correction changes the map-to-local-odometry
        translation; it does not invalidate the relative geometry between the
        recent scans.  Reprojecting every retained scan by the same bounded
        translation preserves doorway/wall overlap across navigation commands
        while keeping the rolling reference in the corrected map frame.
        """

        delta = np.asarray(delta_xy, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(delta)) or float(np.hypot(*delta)) < 1e-9:
            return
        for scan in self.scans:
            scan.pose = Pose2D(
                float(scan.pose.x + delta[0]),
                float(scan.pose.y + delta[1]),
                float(scan.pose.theta_deg),
            )
            scan.world_xy = (
                np.asarray(scan.world_xy, dtype=np.float64) + delta
            ).astype(np.float32)

    def reference(self) -> np.ndarray:
        if not self.scans:
            return np.zeros((0, 2), dtype=np.float32)
        points = np.concatenate([scan.world_xy for scan in self.scans], axis=0)
        if len(points) > self.max_points:
            indices = np.linspace(0, len(points) - 1, self.max_points, dtype=np.int64)
            points = points[indices]
        return points


def _endpoint_support_ratio(
    world_points: np.ndarray,
    reference_points: np.ndarray,
    resolution_m: float,
) -> float:
    """Fraction of endpoints supported by a nearby local-submap endpoint."""
    query = np.asarray(world_points, dtype=np.float64)
    reference = np.asarray(reference_points, dtype=np.float64)
    if not len(query) or not len(reference):
        return 0.0
    resolution = max(0.01, float(resolution_m))
    occupied = {(int(x), int(y)) for x, y in np.round(reference / resolution).astype(np.int64)}
    cells = np.round(query / resolution).astype(np.int64)
    supported = 0
    for x, y in cells:
        supported += int(
            any((int(x) + dx, int(y) + dy) in occupied for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        )
    return float(supported) / float(len(cells))


# ---------------------------------------------------------------------------
# EXPLORE PHASE 2 SUPPORT — OCCUPIED/FREE/UNKNOWN FRONTIER EXTRACTION
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FrontierCluster:
    cells_ij: np.ndarray  # (n, 2) grid cells
    centroid_xy: np.ndarray  # world coords
    span_m: float  # bbox diagonal — "how big is the opening"
    size: int
    passable: bool = False  # wide enough for the inflated robot to cross


def _viewpoint_completes_frontier(cluster: FrontierCluster, scans_added: int) -> bool:
    """A successful look completes a non-crossable frontier.

    The robot cannot gain a new viewing angle by repeatedly selecting the same
    standoff in front of a gap it has already classified as too narrow to enter.
    Passable frontiers are deliberately not completed here: their boundary is
    expected to advance as the robot travels through it into a new room.
    """
    return not cluster.passable and int(scans_added) > 0


def _behind_completed_transition(
    point_xy: np.ndarray,
    transitions: list[tuple[np.ndarray, np.ndarray]],
    slack_m: float,
) -> bool:
    """Whether a target lies back across any completed doorway plane."""
    point = np.asarray(point_xy, dtype=np.float64)
    slack = max(0.0, float(slack_m))
    return any(float(np.dot(point - anchor, outward)) < -slack for anchor, outward in transitions)


def _corridor_centering_steer_deg(forward_frame_xy: np.ndarray) -> float:
    """Live-LiDAR wall-centering steer for corridor traverses.

    Positive return = steer left (toward +y in the forward frame, which maps
    to +IMU degrees through the same err_map chain the segment aim uses).
    Finds the left and right corridor walls in a bounded window around the
    robot, aligns the heading with their axis, and pulls toward the
    centreline.  Returns 0.0 whenever the view does not look like a corridor
    (a missing wall, too wide a gap, or non-wall-aligned clutter), so open
    rooms and doorways are never steered by this term.
    """
    if len(forward_frame_xy) < 20:
        return 0.0
    window = forward_frame_xy[
        (forward_frame_xy[:, 0] > -0.20)
        & (forward_frame_xy[:, 0] < 1.60)
        & (np.abs(forward_frame_xy[:, 1]) < 1.50)
    ]
    left = window[window[:, 1] > 0.12]
    right = window[window[:, 1] < -0.12]
    if len(left) < 8 or len(right) < 8:
        return 0.0
    distance_left = float(np.median(left[:, 1]))
    distance_right = float(-np.median(right[:, 1]))
    if distance_left <= 0.0 or distance_right <= 0.0 or distance_left + distance_right > 2.60:
        return 0.0
    # Corridor axis from the denser wall via PCA, expressed relative to the
    # robot's forward direction.
    wall = left if len(left) >= len(right) else right
    centred = wall - wall.mean(axis=0)
    covariance = centred.T @ centred
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if float(axis[0]) < 0.0:
        axis = -axis
    heading_error_deg = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    if abs(heading_error_deg) > 25.0:
        # The dominant structure is not roughly wall-aligned (corner or
        # clutter); steering on it would be wrong more often than right.
        return 0.0
    lateral_error_m = 0.5 * (distance_left - distance_right)
    steer = 0.8 * heading_error_deg + 18.0 * lateral_error_m
    return float(np.clip(steer, -6.0, 6.0))


def _route_progress_class(
    robot_xy: np.ndarray,
    first_step_xy: np.ndarray,
    goal_xy: np.ndarray,
    current_heading_deg: float | None,
    completed_transitions: list[tuple[np.ndarray, np.ndarray]],
    *,
    forward_turn_limit_deg: float = 95.0,
    doorway_backtrack_tolerance_m: float = 0.15,
) -> int:
    """Return 0 for forward exploration and 1 for an explicit return route.

    A route is a return when it begins with a large reversal or moves back
    toward the most recently crossed doorway. Return routes remain legal, but
    target selection considers them only after every useful forward route has
    been exhausted.
    """
    robot = np.asarray(robot_xy, dtype=np.float64)
    first = np.asarray(first_step_xy, dtype=np.float64)
    goal = np.asarray(goal_xy, dtype=np.float64)
    reverse_turn = False
    initial = first - robot
    if current_heading_deg is not None and float(np.hypot(*initial)) > 0.01:
        first_heading = math.degrees(math.atan2(float(initial[1]), float(initial[0])))
        turn_deg = abs((first_heading - float(current_heading_deg) + 180.0) % 360.0 - 180.0)
        reverse_turn = turn_deg > max(0.0, float(forward_turn_limit_deg))

    doorway_backtrack = False
    if completed_transitions:
        _anchor, outward = completed_transitions[-1]
        doorway_backtrack = float(np.dot(goal - robot, outward)) < -max(
            0.0,
            float(doorway_backtrack_tolerance_m),
        )
    # A goal that lies physically AHEAD of the robot is forward progress, no
    # matter what the doorway plane or the route's first detour step says.
    # Field 2026-08-05 night: a bent map frame rotated a committed doorway
    # plane across the corridor, classifying dead-ahead frontiers as return
    # routes — the robot announced "no useful forward route remains" and
    # retreated out of an open, unmapped corridor.
    if current_heading_deg is not None:
        to_goal = goal - robot
        if float(np.hypot(*to_goal)) > 0.05:
            goal_bearing = abs(
                (
                    math.degrees(math.atan2(float(to_goal[1]), float(to_goal[0])))
                    - float(current_heading_deg)
                    + 180.0
                )
                % 360.0
                - 180.0
            )
            if goal_bearing <= 60.0:
                return 0
    return int(reverse_turn or doorway_backtrack)


def _doorway_transition(
    anchor_xy: np.ndarray,
    objective_start_xy: np.ndarray,
    objective_end_xy: np.ndarray,
    observe_xy: np.ndarray,
    scans_added: int,
    newly_known_cells: int,
    min_gain_cells: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Create a directed room-transition constraint from verified map growth.

    Frontier exploration is most stable as a depth-first traversal of connected
    rooms: once a doorway observation reveals substantial new space, continue
    exploring on that side instead of immediately selecting an attractive old
    frontier behind the robot.  This is deliberately based on accepted scans
    and measured information gain, not merely on the frontier's width.
    """
    if int(scans_added) <= 0 or int(newly_known_cells) < int(min_gain_cells):
        return None
    anchor = np.asarray(anchor_xy, dtype=np.float64)
    start = np.asarray(objective_start_xy, dtype=np.float64)
    end = np.asarray(objective_end_xy, dtype=np.float64)
    outward = np.asarray(observe_xy, dtype=np.float64) - anchor
    outward_norm = float(np.hypot(*outward))
    if outward_norm < 0.30:
        return None
    outward /= outward_norm
    # Information gain alone does not prove a doorway crossing. The previous
    # implementation armed the one-way room constraint while the base was still
    # over a metre short of the frontier, then immediately released it and sent
    # the planner backward. Require the executed robot-centre trajectory to cross
    # the frontier plane and finish measurably on its outward side.
    translated = float(np.hypot(*(end - start)))
    start_progress = float(np.dot(start - anchor, outward))
    end_progress = float(np.dot(end - anchor, outward))
    if translated < 0.25 or start_progress > 0.10 or end_progress < 0.10:
        return None
    return anchor.copy(), outward


def _crossed_passable_frontier(
    analysis: Analysis,
    path_xy: list[np.ndarray] | np.ndarray,
    *,
    max_path_distance_m: float = 0.35,
    min_beyond_m: float = 0.10,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Detect a directed crossing of a passable frontier by an executed path.

    Frontier exploration normally records room transitions while executing an
    explicitly selected doorway objective.  A coverage/patrol route can also
    cross the same boundary.  Treating that as ordinary free-space motion loses
    the topological constraint and permits a later scan-match alias to send the
    pose and planner back into the previous room.

    The crossing is geometric: the executed path must pass close to an actual
    cell of a passable frontier and finish beyond its local boundary plane.
    """
    path = np.asarray(path_xy, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        return None
    start = path[0]
    end = path[-1]
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for cluster in analysis.clusters:
        if not cluster.passable or not len(cluster.cells_ij):
            continue
        frontier_points = np.asarray(
            [analysis.to_world((int(cell[0]), int(cell[1]))) for cell in cluster.cells_ij], dtype=np.float64
        )
        for anchor in frontier_points:
            outward = anchor - start
            norm = float(np.hypot(*outward))
            if norm < 0.20:
                continue
            outward /= norm
            if float(np.dot(end - anchor, outward)) < float(min_beyond_m):
                continue
            distance = math.inf
            for segment_start, segment_end in zip(path[:-1], path[1:], strict=True):
                segment = segment_end - segment_start
                denom = float(segment @ segment)
                if denom <= 1e-12:
                    projection = segment_start
                else:
                    fraction = float(
                        np.clip(
                            ((anchor - segment_start) @ segment) / denom,
                            0.0,
                            1.0,
                        )
                    )
                    projection = segment_start + fraction * segment
                distance = min(distance, float(np.hypot(*(anchor - projection))))
            if distance <= float(max_path_distance_m) and (best is None or distance < best[0]):
                best = (distance, anchor.copy(), outward.copy())
    if best is None:
        return None
    return best[1], best[2]


@dataclass(slots=True)
class Analysis:
    origin_xy: np.ndarray  # world coords of grid cell (0, 0)'s corner
    res_m: float
    occupied: np.ndarray  # (H, W) bool
    free: np.ndarray  # (H, W) bool — ray-cast observed empty space
    traversable: np.ndarray  # (H, W) bool — free minus robot-radius inflation
    clusters: list[FrontierCluster] = field(default_factory=list)
    frontier_cells: np.ndarray | None = None  # (n,2) ALL raw frontier cells, for display
    cost: np.ndarray | None = None  # (H, W) float — 1.0 in open space, rising near obstacles

    def to_cell(self, xy) -> tuple[int, int]:
        i = int((float(xy[1]) - self.origin_xy[1]) / self.res_m)
        j = int((float(xy[0]) - self.origin_xy[0]) / self.res_m)
        return i, j

    def to_world(self, ij) -> np.ndarray:
        return np.array(
            [
                self.origin_xy[0] + (ij[1] + 0.5) * self.res_m,
                self.origin_xy[1] + (ij[0] + 0.5) * self.res_m,
            ],
            dtype=np.float64,
        )


def _swept_footprint_state(
    analysis: Analysis,
    robot_centre_xy: np.ndarray,
    physical_heading_deg: float,
    lookahead_m: float,
    *,
    planned_start_xy: np.ndarray | None = None,
    planned_start_tolerance_m: float = 0.25,
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
    robot_centre = np.asarray(robot_centre_xy, dtype=np.float64)
    centre_cell = analysis.to_cell(robot_centre)
    ci, cj = centre_cell
    if ci < 0 or ci >= h or cj < 0 or cj >= w or not analysis.free[ci, cj]:
        # _plan_waypoints() starts A* from the nearest validated traversable
        # cell. A small scan-matching correction can leave the raw pose one grid
        # cell outside ``free`` even though the committed path begins from that
        # snapped cell. Use the exact same start reference for the immediate
        # sweep, but only within a tightly bounded localization tolerance. This
        # removes false ``map edge 0.00m`` stops without treating genuinely
        # unknown starting positions as clear.
        if planned_start_xy is None:
            return "unknown", 0.0
        planned_start = np.asarray(planned_start_xy, dtype=np.float64)
        if planned_start.shape != (2,) or float(np.hypot(*(planned_start - robot_centre))) > max(
            0.0, float(planned_start_tolerance_m)
        ):
            return "unknown", 0.0
        pi, pj = analysis.to_cell(planned_start)
        if (
            pi < 0
            or pi >= h
            or pj < 0
            or pj >= w
            or not analysis.free[pi, pj]
            or not analysis.traversable[pi, pj]
        ):
            return "unknown", 0.0
        robot_centre = planned_start
        ci, cj = pi, pj

    # A localization correction can place the centre just inside the inflated
    # band even though the physical robot is not colliding. Rejecting distance
    # zero unconditionally deadlocks every subsequent plan. In that state,
    # permit only headings whose short forward sweep gets closer to ANY
    # traversable cell; motion sideways/deeper into inflation remains blocked.
    if not analysis.traversable[ci, cj]:
        traversable_cells = np.column_stack(np.nonzero(analysis.traversable))
        if not len(traversable_cells):
            return "blocked", 0.0
        current_d2 = np.min((traversable_cells[:, 0] - ci) ** 2 + (traversable_cells[:, 1] - cj) ** 2)
        for distance in distances[1:]:
            sample = robot_centre + direction * float(distance)
            i, j = analysis.to_cell(sample)
            if i < 0 or i >= h or j < 0 or j >= w or not analysis.free[i, j]:
                continue
            d2 = np.min((traversable_cells[:, 0] - i) ** 2 + (traversable_cells[:, 1] - j) ** 2)
            if d2 < current_d2:
                return "clear", float(distance)
        return "blocked", 0.0

    for distance in distances[1:]:
        sample = robot_centre + direction * float(distance)
        i, j = analysis.to_cell(sample)
        if i < 0 or i >= h or j < 0 or j >= w or not analysis.free[i, j]:
            return "unknown", float(distance)
        if not analysis.traversable[i, j]:
            return "blocked", float(distance)
    return "clear", float(lookahead_m)


def _map_edge_is_pose_disagreement(distance_m: float, resolution_m: float) -> bool:
    """A zero-distance map edge means pose/planner disagreement, not arrival."""
    return float(distance_m) <= max(0.02, 0.5 * float(resolution_m))


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
        out[src_i0 + di : src_i1 + di, src_j0 + dj : src_j1 + dj] |= mask[src_i0:src_i1, src_j0:src_j1]
    return out


def _grid_endpoint_support_ratio(
    world_xy: np.ndarray,
    support_grid: np.ndarray,
    grid_origin_xy: np.ndarray,
    resolution_m: float,
) -> float:
    """Fraction of scan endpoints landing on independently mapped geometry."""
    points = np.asarray(world_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or not len(points):
        return 0.0
    jj = np.floor((points[:, 0] - float(grid_origin_xy[0])) / float(resolution_m)).astype(np.int64)
    ii = np.floor((points[:, 1] - float(grid_origin_xy[1])) / float(resolution_m)).astype(np.int64)
    inside = (ii >= 0) & (ii < support_grid.shape[0]) & (jj >= 0) & (jj < support_grid.shape[1])
    return float(np.count_nonzero(support_grid[ii[inside], jj[inside]])) / float(len(points))


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
    passable_opening_min_m: float = 0.90,
    extra_occupied_xy: np.ndarray | None = None,
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
    occupied = grid.occupied().copy()
    # Dynamic/hard-stop returns belong to a decaying local costmap layer, not
    # the SLAM occupancy posterior. Mixing them into ``grid.L`` manufactured
    # persistent invisible walls and changed frontier geometry even though no
    # validated keyframe had observed a wall there.
    if extra_occupied_xy is not None and len(extra_occupied_xy):
        extra = np.asarray(extra_occupied_xy, dtype=np.float64)
        jj = ((extra[:, 0] - origin[0]) / res_m).astype(np.int64)
        ii = ((extra[:, 1] - origin[1]) / res_m).astype(np.int64)
        valid = (ii >= 0) & (ii < occupied.shape[0]) & (jj >= 0) & (jj < occupied.shape[1])
        occupied[ii[valid], jj[valid]] = True
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
        cost[band] += 3.0 * (rings - k) / rings  # +3x right at the boundary, fading out
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
    analysis = Analysis(
        origin_xy=origin,
        res_m=res_m,
        occupied=occupied,
        free=free,
        traversable=traversable,
        frontier_cells=fcells if len(fcells) else None,
        cost=cost,
    )

    # Cluster frontier cells (8-connectivity BFS) and keep the significant ones.
    remaining = frontier.copy()
    fi, fj = np.nonzero(remaining)
    raw_clusters = 0
    biggest_raw = 0.0
    required_passage_m = max(
        float(passable_opening_min_m),
        2.0 * float(robot_radius_m) + float(res_m),
    )
    for si, sj in zip(fi.tolist(), fj.tolist(), strict=True):
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
        span = float(
            np.hypot(
                (arr[:, 0].max() - arr[:, 0].min()) * res_m,
                (arr[:, 1].max() - arr[:, 1].min()) * res_m,
            )
        )
        raw_clusters += 1
        biggest_raw = max(biggest_raw, span)
        if len(arr) >= min_frontier_cells and span >= min_frontier_span_m:
            centroid = np.array(
                [
                    origin[0] + (arr[:, 1].mean() + 0.5) * res_m,
                    origin[1] + (arr[:, 0].mean() + 0.5) * res_m,
                ]
            )
            analysis.clusters.append(
                FrontierCluster(
                    cells_ij=arr,
                    centroid_xy=centroid,
                    span_m=span,
                    size=len(arr),
                    passable=span >= required_passage_m,
                )
            )
    analysis.clusters.sort(key=lambda c: -c.span_m)
    narrow_count = sum(not cluster.passable for cluster in analysis.clusters)
    passable_count = len(analysis.clusters) - narrow_count
    print(
        f"[analyze] free={int(free.sum())} unknown={int(unknown.sum())} "
        f"traversable={int(traversable.sum())} frontier_cells={len(fcells)} | "
        f"raw_clusters={raw_clusters} (biggest {biggest_raw:.2f}m) -> "
        f"significant={len(analysis.clusters)} "
        f"(narrow-look={narrow_count}, passable={passable_count}) "
        f"(frontier>={min_frontier_span_m:.2f}m, passage>={required_passage_m:.2f}m, "
        f">={min_frontier_cells} cells)"
    )
    return analysis


# ---------------------------------------------------------------------------
# A* path planning on the traversable grid.
# ---------------------------------------------------------------------------

_NEIGHBORS = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, 1.41421),
    (-1, 1, 1.41421),
    (1, -1, 1.41421),
    (1, 1, 1.41421),
]


def _astar(
    traversable: np.ndarray, start: tuple[int, int], goal: tuple[int, int], cost: np.ndarray | None = None
) -> list[tuple[int, int]] | None:
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


def _route_cost_field(
    traversable: np.ndarray,
    start: tuple[int, int],
    cost: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One Dijkstra expansion reused for every frontier candidate.

    Frontier selection previously ran as many as 48 independent A* searches
    over the same costmap. A single-source cost field is the standard global
    planner primitive: it gives reachability, route cost, and a predecessor
    tree for every candidate in one pass.
    """
    h, w = traversable.shape
    distance = np.full((h, w), np.inf, dtype=np.float64)
    parent_i = np.full((h, w), -1, dtype=np.int32)
    parent_j = np.full((h, w), -1, dtype=np.int32)
    si, sj = start
    if not (0 <= si < h and 0 <= sj < w) or not traversable[si, sj]:
        return distance, parent_i, parent_j
    distance[si, sj] = 0.0
    open_heap: list[tuple[float, int, int]] = [(0.0, si, sj)]
    while open_heap:
        current, ci, cj = heapq.heappop(open_heap)
        if current > float(distance[ci, cj]) + 1e-9:
            continue
        for di, dj, step in _NEIGHBORS:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < h and 0 <= nj < w and traversable[ni, nj]):
                continue
            multiplier = float(cost[ni, nj]) if cost is not None else 1.0
            candidate = current + float(step) * multiplier
            if candidate < float(distance[ni, nj]):
                distance[ni, nj] = candidate
                parent_i[ni, nj] = ci
                parent_j[ni, nj] = cj
                heapq.heappush(open_heap, (candidate, ni, nj))
    return distance, parent_i, parent_j


def _path_from_cost_field(
    start: tuple[int, int],
    goal: tuple[int, int],
    distance: np.ndarray,
    parent_i: np.ndarray,
    parent_j: np.ndarray,
) -> list[tuple[int, int]] | None:
    """Recover one route from a reusable single-source predecessor tree."""
    gi, gj = goal
    if not (
        0 <= gi < distance.shape[0] and 0 <= gj < distance.shape[1] and math.isfinite(float(distance[gi, gj]))
    ):
        return None
    path = [(gi, gj)]
    limit = int(distance.size) + 1
    while path[-1] != start and len(path) <= limit:
        ci, cj = path[-1]
        pi, pj = int(parent_i[ci, cj]), int(parent_j[ci, cj])
        if pi < 0 or pj < 0:
            return None
        path.append((pi, pj))
    if path[-1] != start:
        return None
    path.reverse()
    return path


def _line_free(traversable: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Return whether a grid chord stays inside the traversable costmap.

    This is called both by route smoothing and frontier planning.  Keeping the
    sample generation vectorized matters: a long, winding route previously
    made route smoothing quadratic enough that Ctrl+C could land in this tiny
    predicate rather than in the drive loop.
    """
    h, w = traversable.shape
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
    t = np.linspace(0.0, 1.0, n)
    ii = np.rint(a[0] + (b[0] - a[0]) * t).astype(np.intp)
    jj = np.rint(a[1] + (b[1] - a[1]) * t).astype(np.intp)
    if np.any(ii < 0) or np.any(ii >= h) or np.any(jj < 0) or np.any(jj >= w):
        return False
    return bool(np.all(traversable[ii, jj]))


def _line_clear_of_known_occupancy(
    occupied: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
) -> bool:
    """Visibility test for a frontier observation pose.

    A frontier is intentionally adjacent to *unknown* space.  Requiring a
    sightline to be fully ``free`` therefore rejects the exact doorway or
    corner the LiDAR is meant to observe.  Unknown cells do not prove a wall;
    only a confirmed occupied cell can occlude this line of sight.  Movement is
    still planned exclusively through ``traversable`` cells elsewhere.
    """
    h, w = occupied.shape
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
    t = np.linspace(0.0, 1.0, n)
    ii = np.rint(a[0] + (b[0] - a[0]) * t).astype(np.intp)
    jj = np.rint(a[1] + (b[1] - a[1]) * t).astype(np.intp)
    if np.any(ii < 0) or np.any(ii >= h) or np.any(jj < 0) or np.any(jj >= w):
        return False
    # The final sample is the free frontier cell itself.  Checking every
    # preceding sample is enough and avoids a numerical boundary artefact at
    # the endpoint.
    return bool(not np.any(occupied[ii[:-1], jj[:-1]]))


def _straight_segment_path(
    traversable: np.ndarray,
    path: list[tuple[int, int]],
    max_deviation_cells: float = 1.5,
) -> list[tuple[int, int]]:
    """Compress A* into collision-free straight runs without cutting its contour.

    A chord is accepted only when every sampled cell is traversable *and* the
    original cost-aware A* path stays close to it. Long corridor runs collapse
    to one segment, while real corners remain explicit pivot points.
    """
    if len(path) <= 2:
        return path

    def _deviation(a_idx: int, b_idx: int) -> float:
        a = np.asarray(path[a_idx], dtype=np.float64)
        b = np.asarray(path[b_idx], dtype=np.float64)
        ab = b - a
        length = float(np.hypot(*ab))
        if length < 1e-9:
            return 0.0
        points = np.asarray(path[a_idx : b_idx + 1], dtype=np.float64)
        rel = points - a
        return float(np.max(np.abs(rel[:, 0] * ab[1] - rel[:, 1] * ab[0]) / length))

    out = [path[0]]
    start = 0
    while start < len(path) - 1:
        far = start + 1
        # A very long A* path does not need an O(n^2) all-the-way-to-the-goal
        # chord test at every vertex.  At the default 5cm grid this still lets
        # one straight segment span 12.8m, which is beyond the mapped rooms.
        candidate_limit = min(len(path) - 1, start + 256)
        for candidate in range(candidate_limit, start, -1):
            if _line_free(traversable, path[start], path[candidate]) and _deviation(
                start, candidate
            ) <= float(max_deviation_cells):
                far = candidate
                break
        out.append(path[far])
        start = far
    return out


def _segment_complete(
    position_xy: np.ndarray,
    segment_start_xy: np.ndarray,
    segment_end_xy: np.ndarray,
    tolerance_m: float,
) -> bool:
    """A segment ends on proximity or once localization places us past its end."""
    position = np.asarray(position_xy, dtype=np.float64)
    start = np.asarray(segment_start_xy, dtype=np.float64)
    end = np.asarray(segment_end_xy, dtype=np.float64)
    if float(np.hypot(*(end - position))) <= float(tolerance_m):
        return True
    direction = end - start
    return float(direction @ direction) > 1e-9 and float((position - end) @ direction) >= 0.0


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


def _record_distinct_blocked_approach(
    history: dict[tuple[int, int], list[np.ndarray]],
    frontier_key: tuple[int, int],
    approach_xy: np.ndarray,
    *,
    distinct_distance_m: float = 0.30,
    max_distinct_attempts: int = 2,
) -> tuple[int, bool]:
    """Record independent local-planner failures for one frontier.

    One collision-limited observation pose does not make a frontier unreachable,
    while repeatedly selecting nearly the same pose is not new evidence.  After
    multiple geometrically distinct failures, the global planner should defer
    this objective and map somewhere else.
    """
    point = np.asarray(approach_xy, dtype=np.float64).copy()
    attempts = history.setdefault(frontier_key, [])
    separation = max(0.05, float(distinct_distance_m))
    if not any(float(np.hypot(*(point - old))) < separation for old in attempts):
        attempts.append(point)
    count = len(attempts)
    return count, count >= max(1, int(max_distinct_attempts))


def _wait_for_navigation_yaw(imu, *, wait_up_to_s: float = 2.0) -> float | None:
    """Wait a bounded time for fresh gyro data at a navigation boundary."""
    try:
        yaw = imu.deg()
    except Exception:  # noqa: BLE001
        yaw = None
    if yaw is not None:
        return float(yaw)

    fresh = getattr(imu, "deg_fresh", None)
    if callable(fresh):
        try:
            yaw = fresh(
                wait_up_to_s=max(0.0, float(wait_up_to_s)),
                max_age_s=0.75,
            )
        except Exception:  # noqa: BLE001
            yaw = None
        return None if yaw is None else float(yaw)

    deadline = time.monotonic() + max(0.0, float(wait_up_to_s))
    while time.monotonic() < deadline:
        time.sleep(0.02)
        try:
            yaw = imu.deg()
        except Exception:  # noqa: BLE001
            yaw = None
        if yaw is not None:
            return float(yaw)
    return None


# ---------------------------------------------------------------------------
# EXPLORE PHASE 2 — PRIORITIZE AND SELECT ONE UNKNOWN-AREA TARGET
# ---------------------------------------------------------------------------
def _pick_target(
    analysis: Analysis,
    robot_centre_xy: np.ndarray,
    visited_xy: list[np.ndarray],
    pullback_m: float,
    visited_skip_m: float,
    preferred_frontier_xy: np.ndarray | None = None,
    sampled_viewpoints_xy: list[np.ndarray] | None = None,
    blocked_viewpoints_xy: list[np.ndarray] | None = None,
    blocked_viewpoint_skip_m: float = 0.45,
    completed_transitions: list[tuple[np.ndarray, np.ndarray]] | None = None,
    transition_backtrack_slack_m: float = 0.25,
    current_heading_deg: float | None = None,
    turn_cost_per_deg: float = 1.0,
    forward_turn_limit_deg: float = 95.0,
    allow_return_routes: bool = True,
) -> tuple[FrontierCluster, np.ndarray, list[np.ndarray], bool, np.ndarray, int] | None:
    """Choose a reachable observation pose with line-of-sight to unknown space.

    Returns ``(cluster, goal, path, close_look, observe_xy, expected_gain)``.
    A frontier centroid is only an association label; it is not necessarily
    visible (the centroid of a curved/elongated boundary can lie behind a wall).
    Candidate poses therefore aim at an actual frontier cell and are ranked by
    nearby unknown cells they can reveal, then by path cost.
    """
    planning_started = time.monotonic()
    transitions = completed_transitions or []
    traversable = analysis.traversable
    if transitions:
        # A completed doorway is a one-way topological boundary for this
        # mission. Mask the old-room side from A* itself, not merely from target
        # ranking, so a path to a forward goal cannot loop back through it.
        cell_i, cell_j = np.indices(traversable.shape)
        cell_x = analysis.origin_xy[0] + (cell_j + 0.5) * analysis.res_m
        cell_y = analysis.origin_xy[1] + (cell_i + 0.5) * analysis.res_m
        allowed = np.ones_like(traversable)
        slack = max(0.0, float(transition_backtrack_slack_m))
        for anchor, outward in transitions:
            progress = (cell_x - float(anchor[0])) * float(outward[0]) + (cell_y - float(anchor[1])) * float(
                outward[1]
            )
            allowed &= progress >= -slack
        traversable = traversable & allowed

    start = analysis.to_cell(robot_centre_xy)
    h, w = traversable.shape
    si = min(max(start[0], 0), h - 1)
    sj = min(max(start[1], 0), w - 1)
    if not traversable[si, sj]:
        # The robot's own cell can sit inside the inflation ring (or against a
        # provisional mark after a wobbled stop); search generously — a failed
        # start snap once masqueraded as 'map complete' at 0 viewpoints.
        best = None
        for di, dj in _disk_offsets(int(0.9 / analysis.res_m)):
            ni, nj = si + di, sj + dj
            if 0 <= ni < h and 0 <= nj < w and traversable[ni, nj]:
                d = di * di + dj * dj
                if best is None or d < best[0]:
                    best = (d, ni, nj)
        if best is None:
            return None
        si, sj = best[1], best[2]

    route_started = time.monotonic()
    route_distance, route_parent_i, route_parent_j = _route_cost_field(
        traversable,
        (si, sj),
        cost=analysis.cost,
    )
    reachable = np.isfinite(route_distance)
    route_elapsed = time.monotonic() - route_started

    trav_cells = np.column_stack(np.nonzero(traversable))
    if len(trav_cells) == 0:
        return None
    trav_xy = np.column_stack(
        [
            analysis.origin_xy[0] + (trav_cells[:, 1] + 0.5) * analysis.res_m,
            analysis.origin_xy[1] + (trav_cells[:, 0] + 0.5) * analysis.res_m,
        ]
    )
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
    blocked = blocked_viewpoints_xy or []
    # Tier 0: narrow/non-crossable gaps. Observe these from the current room so
    # shelves, alcoves, and occluded wall edges are completed before the robot
    # commits through a doorway. Tier 1: passable openings into another space.
    geometric: list[tuple[int, float, FrontierCluster, int, np.ndarray, int, float]] = []
    skipped_visited = 0
    skipped_transition = 0
    frontier_cells_tested = 0
    cells_without_reachable_standoff = 0
    sightlines_rejected = 0
    blocked_viewpoints_rejected = 0
    extended_standoff_candidates = 0

    for cluster in candidates:
        if any(np.hypot(*(cluster.centroid_xy - v)) < visited_skip_m for v in visited_xy):
            skipped_visited += 1
            continue
        if _behind_completed_transition(
            cluster.centroid_xy,
            transitions,
            transition_backtrack_slack_m,
        ):
            skipped_transition += 1
            continue
        fcells = cluster.cells_ij
        if len(fcells) > 32:
            fcells = fcells[np.linspace(0, len(fcells) - 1, 32, dtype=np.int64)]
        for fi, fj in fcells:
            frontier_cells_tested += 1
            observe_xy = analysis.to_world((int(fi), int(fj)))
            i0, i1 = max(0, int(fi) - gain_radius), min(h, int(fi) + gain_radius + 1)
            j0, j1 = max(0, int(fj) - gain_radius), min(w, int(fj) + gain_radius + 1)
            expected_gain = int(unknown[i0:i1, j0:j1].sum())
            d_all = np.hypot(trav_xy[:, 0] - observe_xy[0], trav_xy[:, 1] - observe_xy[1])
            reachable_goals = reachable[trav_cells[:, 0], trav_cells[:, 1]]
            # Frontier planners project a boundary onto the robot's reachable
            # configuration space.  Do not sample only the ten cells nearest
            # one arbitrary standoff: in an alcove/doorway those ten can all be
            # hidden by the same wall edge while a perfectly valid viewpoint a
            # few cells around the corner is visible.  Search the complete
            # normal band first, then a bounded long-range observation band.
            # The latter lets LiDAR observe through a long narrow gap that the
            # inflated robot footprint cannot enter.
            normal = np.flatnonzero((d_all >= 0.30) & (d_all <= 1.0) & reachable_goals)
            extended = np.flatnonzero(
                (d_all >= 0.15) & (d_all <= 2.50) & reachable_goals & ~((d_all >= 0.30) & (d_all <= 1.0))
            )

            def _nearest_standoff(
                indices: np.ndarray,
                limit: int = 96,
                distances: np.ndarray = d_all,
                desired: float = desired_standoff,
            ) -> np.ndarray:
                if not len(indices):
                    return indices
                residual = np.abs(distances[indices] - desired)
                if len(indices) > limit:
                    keep = np.argpartition(residual, limit - 1)[:limit]
                    indices = indices[keep]
                    residual = residual[keep]
                return indices[np.argsort(residual)]

            # Bound line-of-sight work per frontier cell. Testing every free
            # cell in a 2.5m disk made candidate generation—not routing—the
            # dominant 2.4s pause. Ninety-six representatives from each band
            # preserve angular/standoff diversity while keeping latency fixed.
            eligible = np.concatenate(
                (
                    _nearest_standoff(normal),
                    _nearest_standoff(extended),
                )
            )
            if not len(eligible):
                cells_without_reachable_standoff += 1
                continue
            found_visible = False
            for idx in eligible:
                goal_xy = trav_xy[idx].astype(np.float64)
                goal_cell = (int(trav_cells[idx, 0]), int(trav_cells[idx, 1]))
                if any(
                    float(np.hypot(*(goal_xy - old))) < max(0.10, float(blocked_viewpoint_skip_m))
                    for old in blocked
                ):
                    blocked_viewpoints_rejected += 1
                    continue
                if not _line_clear_of_known_occupancy(
                    analysis.occupied,
                    goal_cell,
                    (int(fi), int(fj)),
                ):
                    sightlines_rejected += 1
                    continue
                found_visible = True
                if float(d_all[idx]) > 1.0 or float(d_all[idx]) < 0.30:
                    extended_standoff_candidates += 1
                repeat_d = min(
                    (float(np.hypot(*(goal_xy - old))) for old in sampled),
                    default=999.0,
                )
                # Revisiting nearly the same pose is undesirable, but making it
                # an absolute rejection caused eleven live frontiers to be
                # reported unreachable after one large doorway scan. Penalize
                # it instead; a reachable repeated pose beats no plan at all.
                repeat_penalty = 350.0 if repeat_d < max(0.35, float(visited_skip_m)) else 0.0
                turn_deg = 0.0
                if current_heading_deg is not None:
                    delta = goal_xy - robot_centre_xy
                    if float(np.hypot(*delta)) > 0.05:
                        target_heading = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
                        turn_deg = abs((target_heading - float(current_heading_deg) + 180.0) % 360.0 - 180.0)
                geometric_score = (
                    float(expected_gain)
                    + 30.0 * float(cluster.span_m)
                    - 25.0 * abs(float(d_all[idx]) - desired_standoff)
                    - repeat_penalty
                    - max(0.0, float(turn_cost_per_deg)) * turn_deg
                )
                geometric.append(
                    (
                        1 if cluster.passable else 0,
                        geometric_score,
                        cluster,
                        int(idx),
                        observe_xy,
                        expected_gain,
                        repeat_penalty,
                    )
                )
                break
            if not found_visible:
                cells_without_reachable_standoff += 1

    # Route every bounded geometric finalist once, then select lexicographically:
    # forward progress before return, room-detail tier before room transition,
    # and utility within that class. This prevents a high-gain fragment behind
    # the robot from commanding a full turn while useful map growth exists in
    # front, without making an unresolved return corridor unreachable forever.
    routed: list[
        tuple[
            int,
            int,
            float,
            FrontierCluster,
            np.ndarray,
            list[tuple[int, int]],
            np.ndarray,
            int,
        ]
    ] = []
    available_tiers = sorted({item[0] for item in geometric})
    for tier in available_tiers:
        tier_candidates = sorted(
            (item for item in geometric if item[0] == tier),
            key=lambda item: item[1],
            reverse=True,
        )[:48]
        for (
            _tier,
            _geo,
            cluster,
            idx,
            observe_xy,
            expected_gain,
            repeat_penalty,
        ) in tier_candidates:
            cell = (int(trav_cells[idx, 0]), int(trav_cells[idx, 1]))
            path = _path_from_cost_field(
                (si, sj),
                cell,
                route_distance,
                route_parent_i,
                route_parent_j,
            )
            if path is None:
                continue
            travel_m = max(
                analysis.res_m,
                float(route_distance[cell]) * analysis.res_m,
            )
            turn_deg = 0.0
            if current_heading_deg is not None and len(path) > 1:
                first_xy = analysis.to_world(path[1])
                delta = first_xy - robot_centre_xy
                if float(np.hypot(*delta)) > 0.01:
                    first_heading = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
                    turn_deg = abs((first_heading - float(current_heading_deg) + 180.0) % 360.0 - 180.0)
            # Standard frontier utility: information gain minus travel and turn
            # effort. Heading is a soft cost, never a reachability gate.
            utility = (
                float(expected_gain) / (1.0 + 0.35 * travel_m)
                - repeat_penalty
                - max(0.0, float(turn_cost_per_deg)) * turn_deg
            )
            goal_xy = trav_xy[idx].astype(np.float64)
            first_step_xy = analysis.to_world(path[1]) if len(path) > 1 else goal_xy
            progress_class = _route_progress_class(
                robot_centre_xy,
                first_step_xy,
                goal_xy,
                current_heading_deg,
                transitions,
                forward_turn_limit_deg=forward_turn_limit_deg,
            )
            routed.append(
                (
                    progress_class,
                    int(tier),
                    utility,
                    cluster,
                    goal_xy,
                    path,
                    np.asarray(observe_xy, dtype=np.float64),
                    int(expected_gain),
                )
            )
    if not routed:
        print(
            f"[plan] no reachable observation pose among {len(geometric)} candidates "
            f"across {len(available_tiers)} priority tier(s)."
        )
        print(
            "[plan] candidate diagnostics: "
            f"clusters={len(candidates)}, visited={skipped_visited}, "
            f"doorway-masked={skipped_transition}, frontier-cells={frontier_cells_tested}, "
            f"without-visible-reachable-standoff={cells_without_reachable_standoff}, "
            f"blocked-viewpoints={blocked_viewpoints_rejected}, "
            f"rejected-sightlines={sightlines_rejected}."
        )
        print(
            f"[perf] frontier planning {time.monotonic() - planning_started:.3f}s "
            f"(shared route field {route_elapsed:.3f}s; no route selected)."
        )
        return None
    selected_progress_class = min(item[0] for item in routed)
    selected_tier = min(item[1] for item in routed if item[0] == selected_progress_class)
    eligible_routed = [
        item for item in routed if item[0] == selected_progress_class and item[1] == selected_tier
    ]
    best_routed = max(eligible_routed, key=lambda item: item[2])
    (
        _progress_class,
        _tier,
        _utility,
        cluster,
        goal_xy,
        path,
        observe_xy,
        expected_gain,
    ) = best_routed
    if selected_progress_class != 0 and not allow_return_routes:
        print(
            "[plan] only rearward routes are currently available; deferring "
            "them while the map receives another forward observation."
        )
        print(
            f"[perf] frontier planning {time.monotonic() - planning_started:.3f}s "
            f"(shared route field {route_elapsed:.3f}s, "
            f"{len(geometric)} geometric candidate(s); return deferred)."
        )
        return None
    if extended_standoff_candidates:
        print(
            f"[plan] used {extended_standoff_candidates} long-range/close-boundary "
            "frontier viewpoint candidate(s) after normal standoff visibility was exhausted."
        )
    tier_name = "narrow/non-crossable" if selected_tier == 0 else "passable opening"
    if selected_progress_class == 0:
        print(
            "[plan] forward-progress policy selected a target without a return "
            "trip; rearward candidates remain deferred."
        )
    else:
        print(
            "[plan] no useful forward route remains; explicitly allowing a "
            "return route to an unresolved frontier."
        )
    print(f"[plan] selected {tier_name} tier; lower-priority room transitions deferred.")
    segment_path = _straight_segment_path(
        traversable,
        path,
        max_deviation_cells=max(1.0, 0.08 / analysis.res_m),
    )
    waypoints = [analysis.to_world(ij) for ij in segment_path[1:]]
    if not waypoints or np.hypot(*(waypoints[-1] - goal_xy)) > 0.05:
        waypoints.append(goal_xy)
    close_look = float(np.hypot(*(goal_xy - observe_xy))) <= 0.9
    print(
        f"[perf] frontier planning {time.monotonic() - planning_started:.3f}s "
        f"(shared route field {route_elapsed:.3f}s, "
        f"{len(geometric)} geometric candidate(s))."
    )
    return cluster, goal_xy, waypoints, close_look, observe_xy, expected_gain


def _pick_target_batch(
    analysis: Analysis,
    robot_centre_xy: np.ndarray,
    visited_xy: list[np.ndarray],
    pullback_m: float,
    visited_skip_m: float,
    *,
    batch_size: int,
    sampled_viewpoints_xy: list[np.ndarray] | None = None,
    blocked_viewpoints_xy: list[np.ndarray] | None = None,
    blocked_viewpoint_skip_m: float = 0.45,
    completed_transitions: list[tuple[np.ndarray, np.ndarray]] | None = None,
    transition_backtrack_slack_m: float = 0.25,
    current_heading_deg: float | None = None,
    turn_cost_per_deg: float = 1.0,
    forward_turn_limit_deg: float = 95.0,
    allow_return_routes: bool = False,
) -> list[tuple[FrontierCluster, np.ndarray, list[np.ndarray], bool, np.ndarray, int]]:
    """Create a genuine fixed observation batch from one frozen map revision.

    Selection is simulated from each preceding marker so the three poses form a
    coherent forward tour.  We deliberately keep their *coordinates* fixed
    after this call: arriving snapshots may change the occupancy grid, but they
    must not cause the explorer to forget marker 2 and invent a different task.
    Only the collision-safe route to the next marker is allowed to be replanned.
    """
    selected: list[tuple[FrontierCluster, np.ndarray, list[np.ndarray], bool, np.ndarray, int]] = []
    working_visited = [np.asarray(point, dtype=np.float64).copy() for point in visited_xy]
    working_sampled = [
        np.asarray(point, dtype=np.float64).copy() for point in (sampled_viewpoints_xy or [])
    ]
    virtual_xy = np.asarray(robot_centre_xy, dtype=np.float64).copy()
    virtual_heading = current_heading_deg

    for _slot in range(max(1, int(batch_size))):
        picked = _pick_target(
            analysis,
            virtual_xy,
            working_visited,
            pullback_m,
            visited_skip_m,
            sampled_viewpoints_xy=working_sampled,
            blocked_viewpoints_xy=blocked_viewpoints_xy,
            blocked_viewpoint_skip_m=blocked_viewpoint_skip_m,
            completed_transitions=completed_transitions,
            transition_backtrack_slack_m=transition_backtrack_slack_m,
            current_heading_deg=virtual_heading,
            turn_cost_per_deg=turn_cost_per_deg,
            forward_turn_limit_deg=forward_turn_limit_deg,
            allow_return_routes=allow_return_routes,
        )
        if picked is None:
            break
        cluster, goal_xy, waypoints, close_look, observe_xy, expected_gain = picked
        selected.append((cluster, goal_xy.copy(), waypoints, close_look, observe_xy.copy(), expected_gain))
        working_visited.append(cluster.centroid_xy.copy())
        working_sampled.append(goal_xy.copy())
        if waypoints:
            delta = np.asarray(waypoints[-1], dtype=np.float64) - virtual_xy
            if float(np.hypot(*delta)) > 0.05:
                virtual_heading = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
        virtual_xy = goal_xy.copy()
    return selected


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


# ---------------------------------------------------------------------------
# EXPLORE PHASE 3 SUPPORT — COLLISION-INFLATED PATH PLANNING
# ---------------------------------------------------------------------------
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
        0 <= start_cell[0] < h and 0 <= start_cell[1] < w and bool(analysis.traversable[start_cell])
    )
    if start_was_snapped and float(np.hypot(*(start_w - start_xy))) > 0.02:
        # Do not hide the snap from the follower.  The old path dropped this
        # first segment, so A* planned from a fictitious safe cell while the
        # guard evaluated from the real pose and rejected every command.
        waypoints.append(start_w)
    segment_path = _straight_segment_path(
        analysis.traversable,
        path,
        max_deviation_cells=max(1.0, 0.08 / analysis.res_m),
    )
    waypoints.extend(analysis.to_world(ij) for ij in segment_path[1:])
    goal_w = analysis.to_world(goal)
    if not waypoints or np.hypot(*(waypoints[-1] - goal_w)) > 0.05:
        waypoints.append(goal_w)
    return waypoints


def _plan_passable_frontier_crossing(
    analysis: Analysis,
    robot_xy: np.ndarray,
    cluster: FrontierCluster,
    *,
    minimum_beyond_m: float = 0.15,
    maximum_beyond_m: float = 1.20,
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Plan directly through a photographed passable frontier.

    A frontier observation pose can legitimately be the robot's current cell.
    Re-running generic viewpoint selection from there merely photographs the
    doorway repeatedly.  Once an overlap snapshot has exposed free space, pick
    the deepest collision-inflated traversable cell in a narrow corridor beyond
    the frontier and commit one A* route to it.
    """
    robot = np.asarray(robot_xy, dtype=np.float64)
    anchor = np.asarray(cluster.centroid_xy, dtype=np.float64)
    outward = anchor - robot
    frontier_distance = float(np.hypot(*outward))
    if frontier_distance < 0.10:
        return None
    outward /= frontier_distance

    cells = np.column_stack(np.nonzero(analysis.traversable))
    if not len(cells):
        return None
    points = np.asarray([analysis.to_world((int(i), int(j))) for i, j in cells])
    relative = points - robot
    progress = relative @ outward
    lateral = np.abs(relative[:, 0] * -outward[1] + relative[:, 1] * outward[0])
    corridor_half_width = max(0.25, min(0.55, 0.5 * float(cluster.span_m)))
    eligible = np.flatnonzero(
        (progress >= frontier_distance + float(minimum_beyond_m))
        & (progress <= frontier_distance + float(maximum_beyond_m))
        & (lateral <= corridor_half_width)
    )
    if not len(eligible):
        return None

    # Try the deepest evidence first. Limit A* attempts so passage commitment
    # has a deterministic calculation bound.
    # Prefer maximum forward depth, then the corridor centreline so the
    # shoulders retain equal clearance through the frame.
    order = eligible[np.lexsort((lateral[eligible], -progress[eligible]))][:24]
    for index in order:
        goal = points[int(index)].astype(np.float64)
        if float(np.hypot(*(goal - robot))) < 0.30:
            continue
        route = _plan_waypoints(analysis, robot, goal)
        if route:
            return goal, route
    return None


def _pick_patrol_route(
    analysis: Analysis,
    robot_xy: np.ndarray,
    observation_history: list[np.ndarray],
    blocked_xy: list[np.ndarray] | None = None,
    min_translation_m: float = 0.50,
    completed_transitions: list[tuple[np.ndarray, np.ndarray]] | None = None,
    current_heading_deg: float | None = None,
    transition_backtrack_slack_m: float = 0.25,
    forward_turn_limit_deg: float = 95.0,
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Maximin coverage patrol over reachable free space.

    A patrol goal is a navigation action, not a stationary observation.  Goals
    inside the progress-checker's translation radius, and approaches that have
    just failed, are therefore ineligible.  This is the same separation used by
    navigation stacks between global coverage selection and the local
    controller's temporary failure costmap.
    """
    start = _snap_traversable(analysis, robot_xy, 0.9)
    if start is None:
        return None
    reachable = np.zeros_like(analysis.traversable)
    reachable[start] = True
    stack = [start]
    while stack:
        ci, cj = stack.pop()
        for di, dj, _step in _NEIGHBORS:
            ni, nj = ci + di, cj + dj
            if (
                0 <= ni < reachable.shape[0]
                and 0 <= nj < reachable.shape[1]
                and analysis.traversable[ni, nj]
                and not reachable[ni, nj]
            ):
                reachable[ni, nj] = True
                stack.append((ni, nj))
    cells = np.column_stack(np.nonzero(reachable))
    if len(cells) < 2:
        return None
    if len(cells) > 512:
        cells = cells[np.linspace(0, len(cells) - 1, 512, dtype=np.int64)]
    points = np.asarray([analysis.to_world((int(cell[0]), int(cell[1]))) for cell in cells])
    distance_from_robot = np.hypot(
        points[:, 0] - float(robot_xy[0]),
        points[:, 1] - float(robot_xy[1]),
    )
    minimum = max(0.20, float(min_translation_m))
    eligible = distance_from_robot >= max(0.75, minimum)
    if not np.any(eligible):
        eligible = distance_from_robot >= minimum
    for blocked in blocked_xy or []:
        eligible &= (
            np.hypot(
                points[:, 0] - float(blocked[0]),
                points[:, 1] - float(blocked[1]),
            )
            >= 0.45
        )
    transitions = completed_transitions or []
    if transitions:
        eligible &= np.asarray(
            [
                not _behind_completed_transition(
                    point,
                    transitions,
                    float(transition_backtrack_slack_m),
                )
                for point in points
            ],
            dtype=bool,
        )
    if not np.any(eligible):
        return None
    recent = observation_history[-32:]
    if recent:
        history = np.asarray(recent, dtype=np.float64)
        novelty = np.min(
            np.hypot(
                points[:, None, 0] - history[None, :, 0],
                points[:, None, 1] - history[None, :, 1],
            ),
            axis=1,
        )
    else:
        novelty = distance_from_robot
    score = novelty + 0.15 * distance_from_robot
    score[~eligible] = -math.inf
    for index in np.argsort(score)[::-1][:24]:
        goal = points[int(index)].astype(np.float64)
        route = _plan_waypoints(analysis, robot_xy, goal)
        if route:
            first_step = np.asarray(route[0], dtype=np.float64)
            if (
                _route_progress_class(
                    robot_xy,
                    first_step,
                    goal,
                    current_heading_deg,
                    transitions,
                    forward_turn_limit_deg=forward_turn_limit_deg,
                    doorway_backtrack_tolerance_m=transition_backtrack_slack_m,
                )
                != 0
            ):
                continue
            return goal, route
    return None


def _navigation_action_made_progress(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    scans_added: int,
    newly_known_cells: int,
    min_translation_m: float = 0.12,
) -> bool:
    """Return whether a navigation/observation action changed useful state.

    A controller result alone is not progress: a collision-blocked pivot used
    to be counted as a completed patrol even when the base never moved and the
    map transaction was rejected.  Navigation2-style progress checking accepts
    either measurable base displacement or a committed observation that reveals
    new cells.
    """
    translated = float(np.hypot(*(np.asarray(end_xy) - np.asarray(start_xy))))
    mapped_new_space = int(scans_added) > 0 and int(newly_known_cells) > 0
    return translated >= max(0.01, float(min_translation_m)) or mapped_new_space


def _relocalization_candidate_is_continuous(
    prior_centre_xy: np.ndarray,
    solved_centre_xy: np.ndarray,
    score: float,
    min_score: float,
    support_ratio: float,
    min_support_ratio: float,
    max_pose_innovation_m: float,
) -> bool:
    """Gate ordinary tracking recovery by quality and odometric continuity."""
    innovation = float(
        np.hypot(
            *(np.asarray(solved_centre_xy, dtype=np.float64) - np.asarray(prior_centre_xy, dtype=np.float64))
        )
    )
    return (
        float(score) >= float(min_score)
        and float(support_ratio) >= float(min_support_ratio)
        and innovation <= float(max_pose_innovation_m)
    )


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
    allow_whole_map_search: bool = False,
    fine_angle_step_deg: float = 0.35,
    fine_theta_half_window_deg: float | None = None,
    use_nearest_penalty: bool = True,
) -> tuple[Pose2D, float]:
    ref = world_map.reference()
    # Preserve angular coverage while bounding candidate-scoring cost. The full
    # scan is still integrated into the map after the reduced set solves its pose.
    max_match_points = max(100, int(getattr(args, "match_max_points", 700)))
    match_xy = np.asarray(local_xy)
    if len(match_xy) > max_match_points:
        indices = np.linspace(0, len(match_xy) - 1, max_match_points, dtype=np.int64)
        match_xy = match_xy[indices]
    solved, meta = _search_pose(
        snapshot_points_xy=match_xy,
        global_points_xy=ref,
        initial_pose=seed,
        resolution_m=float(args.stitch_resolution_m),
        search_xy_m=float(search_xy_m),
        coarse_angle_step_deg=float(coarse_angle_step_deg),
        fine_angle_step_deg=float(fine_angle_step_deg),
        theta_window_deg=float(theta_window_deg),
        whole_map_theta_center_deg=float(seed.theta_deg),
        whole_map_theta_window_deg=float(theta_window_deg) + 4.0,
        max_translation_from_initial_m=float(search_xy_m) + 0.08,
        prior_pose=seed,
        prior_translation_weight=0.05,
        prior_theta_weight=0.04,
        allow_whole_map_search=bool(allow_whole_map_search),
        local_fine_theta_half_window_deg=fine_theta_half_window_deg,
        local_use_nearest_penalty=bool(use_nearest_penalty),
    )
    return solved, float(meta.get("score") or 0.0)


def _localize_against_points(
    local_xy: np.ndarray,
    reference_xy: np.ndarray,
    seed: Pose2D,
    args,
    search_xy_m: float,
    theta_window_deg: float,
) -> tuple[Pose2D, float, float]:
    """Match one scan to a rolling local reference without touching the map."""
    reference = np.asarray(reference_xy, dtype=np.float32)
    if len(reference) < 30 or len(local_xy) < 12:
        return seed, 0.0, 0.0
    max_match_points = max(100, int(getattr(args, "match_max_points", 700)))
    match_xy = np.asarray(local_xy)
    if len(match_xy) > max_match_points:
        indices = np.linspace(0, len(match_xy) - 1, max_match_points, dtype=np.int64)
        match_xy = match_xy[indices]
    solved, meta = _search_pose(
        snapshot_points_xy=match_xy,
        global_points_xy=reference,
        initial_pose=seed,
        resolution_m=float(args.stitch_resolution_m),
        search_xy_m=float(search_xy_m),
        coarse_angle_step_deg=2.0,
        fine_angle_step_deg=0.35,
        theta_window_deg=float(theta_window_deg),
        max_translation_from_initial_m=float(search_xy_m) + 0.08,
        prior_pose=seed,
        prior_translation_weight=0.08,
        prior_theta_weight=0.08,
        allow_whole_map_search=False,
        local_fine_theta_half_window_deg=min(2.0, float(theta_window_deg)),
        local_use_nearest_penalty=True,
    )
    score = float(meta.get("score") or 0.0)
    support = _endpoint_support_ratio(
        _transform_points(local_xy, solved),
        reference,
        max(0.04, 2.0 * float(args.stitch_resolution_m)),
    )
    return solved, score, support


# ---------------------------------------------------------------------------
# EXPLORE PHASE 3 SUPPORT — BASE COMMAND STREAM AND LIVE SAFETY
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
    def __init__(
        self,
        robot,
        hold: dict,
        imu,
        rate_hz: float = 25.0,
        *,
        turn_speed: float = 0.9,
        hold_gain: float = 2.0,
        hold_max: float = 0.25,
        hold_deadband_deg: float = 1.5,
    ) -> None:
        self.robot = robot
        self.hold = dict(hold)
        self.imu = imu
        self.turn_sign = 1.0  # set after the anchor spin measures it
        self.turn_speed = float(turn_speed)
        self.hold_gain = float(hold_gain)
        self.hold_max = float(hold_max)
        self.hold_deadband_deg = float(hold_deadband_deg)
        self.dt = 1.0 / float(rate_hz)
        # (kind, x_vel, y_vel, target_imu_deg). ``translate`` supports the
        # collision-checked backup/strafe recovery used by the local planner.
        self._mode: tuple[str, float, float, float] = ("halt", 0.0, 0.0, 0.0)
        self._lock = threading.Lock()
        self._safety_check = None
        self._rotation_safety_check = None
        self._safety_latched_reason: str | None = None
        self._run = False
        self._thread: threading.Thread | None = None
        self._command_error: str | None = None
        self._legacy_host_untorque_sentinel = False

    def _open_thread_owned_command_socket(self):
        """Create the real-time command transport in its owning thread.

        ZeroMQ sockets are thread-affine.  ``SourcceyClient.connect()`` creates
        its general command socket on the main thread, while this controller
        runs on a dedicated 25Hz thread.  Reusing that socket here produced
        apparently valid rotate/translate modes with no physical wheel command.
        A dedicated PUSH socket keeps one writer, one owner, and one predictable
        control cadence.  Test doubles and non-remote robots retain the normal
        ``send_action`` path.
        """
        endpoint_host = getattr(self.robot, "remote_ip", None)
        endpoint_port = getattr(self.robot, "port_zmq_cmd", None)
        converter = getattr(self.robot, "protobuf_converter", None)
        if endpoint_host is None or endpoint_port is None or converter is None:
            return None
        try:
            import zmq

            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
            socket.setsockopt(zmq.SNDHWM, 1)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDTIMEO, 100)
            socket.connect(f"tcp://{endpoint_host}:{int(endpoint_port)}")
            print(
                "[drive] real-time base command transport active "
                f"(thread-owned tcp://{endpoint_host}:{int(endpoint_port)})."
            )
            return context, socket, converter
        except Exception as exc:  # noqa: BLE001
            self._command_error = f"control transport setup failed: {exc}"
            print(
                f"[drive] WARNING: {self._command_error}; falling back to the "
                "robot client's general command path."
            )
            return None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        transport = self._open_thread_owned_command_socket()
        try:
            while self._run:
                with self._lock:
                    kind, x_vel, y_vel, target = self._mode
                    safety_check = self._safety_check
                    rotation_safety_check = self._rotation_safety_check
                x = y = th = 0.0
                err_deg = 0.0
                if kind != "halt":
                    yaw = None
                    with contextlib.suppress(Exception):
                        yaw = self.imu.deg()
                    if yaw is None:
                        x = th = 0.0  # no gyro this tick -> fail safe, stop
                    else:
                        err_deg = target - float(yaw)
                        if kind == "rotate":
                            # Closed-loop pivot: fresh gyro every tick means no
                            # overshoot; stiction floor so it always actually turns.
                            if abs(err_deg) > 2.5:
                                mag = min(self.turn_speed, max(0.80, 0.035 * abs(err_deg)))
                                th = self.turn_sign * math.copysign(mag, err_deg)
                        else:  # "drive" or collision-recovery translation
                            x = float(x_vel)
                            y = float(y_vel)
                            if abs(err_deg) > self.hold_deadband_deg:
                                corr = self.hold_gain * math.radians(err_deg)
                                th = self.turn_sign * max(-self.hold_max, min(self.hold_max, corr))
                if kind == "drive" and abs(x) > 0.0 and safety_check is not None:
                    reason = None
                    try:
                        reason = safety_check()
                    except Exception:
                        # A broken safety callback fails closed for translation
                        # in either direction. Callers performing a bounded
                        # reverse recovery install a rear-facing guard.
                        reason = "safety check failed"
                    if reason:
                        x = th = 0.0
                        with self._lock:
                            self._safety_latched_reason = str(reason)
                            self._mode = ("halt", 0.0, 0.0, 0.0)
                if kind == "rotate" and abs(th) > 0.0 and rotation_safety_check is not None:
                    reason = None
                    try:
                        reason = rotation_safety_check(err_deg)
                    except Exception:
                        # A broken rotational safety callback fails closed too.
                        reason = "rotational safety check failed"
                    if reason:
                        x = th = 0.0
                        with self._lock:
                            self._safety_latched_reason = str(reason)
                            self._mode = ("halt", 0.0, 0.0, 0.0)
                action = {
                    "x.vel": x,
                    "y.vel": y,
                    "theta.vel": th,
                    "z.pos": getattr(self.robot, "_z_pos_cmd", 100.0),
                    **self.hold,
                }
                if self._legacy_host_untorque_sentinel:
                    # Older Pi hosts decode absent proto3 booleans as explicit
                    # torque requests and enter the arm bus before writing the
                    # wheels. The proven spin packet uses these flags to mean
                    # "base only". Keep this compatibility mode off unless IMU
                    # proves the normal packet produced no base motion.
                    action["untorque_left"] = True
                    action["untorque_right"] = True
                try:
                    if transport is None:
                        self.robot.send_action(action)
                    else:
                        _context, socket, converter = transport
                        message = converter.action_to_protobuf(action)
                        socket.send(message.SerializeToString())
                    self._command_error = None
                except Exception as exc:  # noqa: BLE001
                    self._command_error = f"control command send failed: {exc}"
                time.sleep(self.dt)
        finally:
            if transport is not None:
                context, socket, _converter = transport
                try:
                    socket.close(0)
                finally:
                    context.term()

    def rotate_to(self, target_imu_deg: float) -> None:
        with self._lock:
            if self._safety_latched_reason is None:
                self._mode = ("rotate", 0.0, 0.0, float(target_imu_deg))

    def drive_toward(self, x_vel: float, target_imu_deg: float) -> None:
        with self._lock:
            if self._safety_latched_reason is None:
                self._mode = ("drive", float(x_vel), 0.0, float(target_imu_deg))

    def translate_body(self, x_vel: float, y_vel: float, target_imu_deg: float) -> None:
        """Translate in the body frame while the fast gyro loop holds heading."""
        with self._lock:
            if self._safety_latched_reason is None:
                self._mode = (
                    "translate",
                    float(x_vel),
                    float(y_vel),
                    float(target_imu_deg),
                )

    def set_safety_check(self, check) -> None:
        with self._lock:
            self._safety_check = check

    def set_rotation_safety_check(self, check) -> None:
        with self._lock:
            self._rotation_safety_check = check

    def clear_safety_latch(self) -> None:
        with self._lock:
            self._safety_latched_reason = None

    def safety_latched_reason(self) -> str | None:
        with self._lock:
            return self._safety_latched_reason

    def command_error(self) -> str | None:
        return self._command_error

    def enable_legacy_host_untorque_sentinel(self) -> None:
        self._legacy_host_untorque_sentinel = True

    def legacy_host_untorque_sentinel_enabled(self) -> bool:
        return bool(self._legacy_host_untorque_sentinel)

    def halt(self) -> None:
        with self._lock:
            self._mode = ("halt", 0.0, 0.0, 0.0)

    def shutdown(self) -> None:
        self.halt()
        self._run = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with contextlib.suppress(Exception):
            _send_stop(self.robot)


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
    return np.column_stack(
        [
            pts_local[:, 0] * c - pts_local[:, 1] * s,
            pts_local[:, 0] * s + pts_local[:, 1] * c,
        ]
    )


# ---------------------------------------------------------------------------
# Rerun rendering.
# ---------------------------------------------------------------------------


def _log_world(
    rr,
    world_map: WorldMap,
    args,
    robot_centre,
    trail: list[np.ndarray],
    analysis: Analysis | None = None,
    target_xy=None,
    waypoints: list[np.ndarray] | None = None,
    note: str = "",
) -> None:
    pts, ids = world_map.render_points()
    if len(pts):
        xyz = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(len(pts), dtype=np.float32)])
        colors = _MAP_PALETTE[ids % len(_MAP_PALETTE)]
        rr.log("world/lidar_map", rr.Points3D(xyz.astype(np.float32), colors=colors, radii=0.02))
        with contextlib.suppress(Exception):
            rr.log("world2d/lidar_map", rr.Points2D(pts.astype(np.float32), colors=colors, radii=0.035))
    else:
        with contextlib.suppress(Exception):
            rr.log("world2d/lidar_map", rr.Points2D(np.zeros((0, 2), dtype=np.float32)))
    # Frontier cells are planner state, not LiDAR returns. Drawing every cell as
    # a red dotted wall made newly exposed UNKNOWN boundaries look like corrupt
    # map geometry. Keep the map layer pure and show one amber marker per
    # significant frontier instead.
    rr.log("world/frontier_candidates", rr.Points3D(np.zeros((0, 3), dtype=np.float32)))
    with contextlib.suppress(Exception):
        rr.log("world2d/frontier_candidates", rr.Points2D(np.zeros((0, 2), dtype=np.float32)))
    if analysis is not None and analysis.clusters:
        fxy = np.asarray(
            [[float(cluster.centroid_xy[0]), float(cluster.centroid_xy[1])] for cluster in analysis.clusters],
            dtype=np.float32,
        )
        fxyz = np.column_stack([fxy[:, 0], fxy[:, 1], np.full(len(fxy), 0.015, dtype=np.float32)])
        rr.log(
            "world/frontiers",
            rr.Points3D(
                fxyz,
                colors=[[255, 180, 40]] * len(fxyz),
                radii=0.045,
            ),
        )
        with contextlib.suppress(Exception):
            rr.log("world2d/frontiers", rr.Points2D(fxy, colors=[[255, 180, 40]] * len(fxy), radii=0.08))
    else:
        rr.log("world/frontiers", rr.Points3D(np.zeros((0, 3), dtype=np.float32)))
        with contextlib.suppress(Exception):
            rr.log("world2d/frontiers", rr.Points2D(np.zeros((0, 2), dtype=np.float32)))
    if target_xy is not None:
        target2 = np.asarray([[float(target_xy[0]), float(target_xy[1])]], dtype=np.float32)
        rr.log(
            "world/target",
            rr.Points3D(
                [[float(target_xy[0]), float(target_xy[1]), 0.0]], colors=[[80, 255, 120]], radii=0.09
            ),
        )
        with contextlib.suppress(Exception):
            rr.log("world2d/target", rr.Points2D(target2, colors=[[80, 255, 120]], radii=0.12))
    else:
        rr.log("world/target", rr.Points3D(np.zeros((0, 3), dtype=np.float32)))
        with contextlib.suppress(Exception):
            rr.log("world2d/target", rr.Points2D(np.zeros((0, 2), dtype=np.float32)))
    if waypoints:
        strip = [[float(robot_centre[0]), float(robot_centre[1]), 0.0]] + [
            [float(p[0]), float(p[1]), 0.0] for p in waypoints
        ]
        rr.log("world/path", rr.LineStrips3D([strip], colors=[[255, 220, 90]], radii=0.012))
        with contextlib.suppress(Exception):
            strip2 = [[float(robot_centre[0]), float(robot_centre[1])]] + [
                [float(p[0]), float(p[1])] for p in waypoints
            ]
            rr.log("world2d/path", rr.LineStrips2D([strip2], colors=[[255, 220, 90]], radii=0.025))
    else:
        rr.log("world/path", rr.LineStrips3D([]))
        with contextlib.suppress(Exception):
            rr.log("world2d/path", rr.LineStrips2D([]))
    if len(trail) >= 2:
        rr.log(
            "world/trail",
            rr.LineStrips3D(
                [[[float(p[0]), float(p[1]), 0.0] for p in trail]], colors=[[255, 150, 70]], radii=0.008
            ),
        )
        with contextlib.suppress(Exception):
            rr.log(
                "world2d/trail",
                rr.LineStrips2D([[[float(p[0]), float(p[1])] for p in trail]], colors=[[255, 150, 70]], radii=0.018),
            )
    rr.log(
        "world/robot",
        rr.Points3D(
            [[float(robot_centre[0]), float(robot_centre[1]), 0.0]], colors=[[255, 130, 60]], radii=0.07
        ),
    )
    with contextlib.suppress(Exception):
        rr.log(
            "world2d/robot",
            rr.Points2D([[float(robot_centre[0]), float(robot_centre[1])]], colors=[[255, 130, 60]], radii=0.13),
        )
    footprint_angles = np.linspace(0.0, 2.0 * math.pi, 49)
    footprint = [
        [
            float(robot_centre[0]) + float(args.robot_radius_m) * math.cos(float(a)),
            float(robot_centre[1]) + float(args.robot_radius_m) * math.sin(float(a)),
            0.0,
        ]
        for a in footprint_angles
    ]
    rr.log("world/robot_footprint", rr.LineStrips3D([footprint], colors=[[255, 130, 60]], radii=0.006))
    with contextlib.suppress(Exception):
        rr.log("world2d/robot_footprint", rr.LineStrips2D([[p[:2] for p in footprint]], colors=[[255, 130, 60]], radii=0.014))
    if note:
        rr.log("explore/status", rr.TextLog(note))


def _configure_rerun_layout(rr) -> None:
    """Keep world mapping and robot-frame LiDAR in different Rerun panes.

    ``world`` is metric map geometry. ``live_lidar`` is deliberately *not*
    transformed into that frame: it is the current raw sensor view around the
    robot. Letting Rerun's automatic 3D view combine both made the live fan
    look like corrupt map geometry.  Pin two origin-scoped views instead.
    """
    try:
        import rerun.blueprint as rrb

        world_bounds = rrb.VisualBounds2D(x_range=[-6.0, 6.0], y_range=[-6.0, 6.0])
        live_bounds = rrb.VisualBounds2D(x_range=[-1.0, 5.0], y_range=[-3.0, 3.0])
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(
                    name="World map",
                    origin="/world2d",
                    visual_bounds=world_bounds,
                ),
                rrb.Vertical(
                    rrb.Spatial2DView(
                        name="Live LiDAR (robot frame)",
                        origin="/live_lidar2d",
                        visual_bounds=live_bounds,
                    ),
                    rrb.Spatial2DView(name="Camera panorama", origin="/cameras/panorama"),
                ),
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint, make_active=True, make_default=True)
        print("[rerun] layout pinned as stable 2D floorplan panes: world map left; robot-frame LiDAR right.")
    except Exception as exc:  # noqa: BLE001 - Rerun blueprint API is optional.
        print(f"[rerun] could not pin the split viewer layout ({exc}); data remains under world/ and live_lidar/.")


def _log_observation_batch_markers(
    rr,
    batch: list[tuple[FrontierCluster, np.ndarray, list[np.ndarray], bool, np.ndarray, int]],
    *,
    completed: int = 0,
) -> None:
    """Draw the frozen observation queue as numbered world-map markers."""
    if not batch:
        rr.log("world/observation_batch", rr.Points3D(np.empty((0, 3), dtype=np.float32)))
        return
    xyz = np.asarray(
        [[float(item[1][0]), float(item[1][1]), 0.08] for item in batch],
        dtype=np.float32,
    )
    rr.log(
        "world/observation_batch",
        rr.Points3D(
            xyz,
            colors=[[90, 230, 255]] * len(xyz),
            radii=0.10,
            labels=[str(completed + index + 1) for index in range(len(xyz))],
        ),
    )


# ===========================================================================
# EXPLORE PHASE 0 — CONNECT HARDWARE AND PREPARE THE MISSION
#
# Configuration, sensor/robot connections, calibrated collision-envelope
# loading, and visualization setup. Exploration begins at the explicit Phase 1
# execution block later in main().
# ===========================================================================


class _TeeLog:
    """Mirror console output into a run log file, flushing every write so a
    crash or battery death never loses the tail of the log."""

    def __init__(self, stream, file):
        self._stream = stream
        self._file = file

    def write(self, text):
        self._stream.write(text)
        try:
            self._file.write(text)
            self._file.flush()
        except Exception:
            pass
        return len(text)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass


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
    parser.add_argument(
        "--spin-speed",
        type=float,
        default=0.80,
        help="Slow anchor-spin command; 0.80 is the measured rotation-stiction floor.",
    )
    parser.add_argument("--max-spin-seconds", type=float, default=75.0, help="Per-pass anchor-spin timeout.")
    parser.add_argument(
        "--anchor-spin-max-speed",
        type=float,
        default=1.0,
        help="Maximum bounded theta.vel used only when IMU feedback proves the "
        "requested slow anchor turn is stuck below drivetrain stiction.",
    )
    parser.add_argument(
        "--anchor-stall-seconds",
        type=float,
        default=2.5,
        help="IMU motion-watchdog window during anchor acquisition.",
    )
    parser.add_argument(
        "--anchor-motion-retries",
        type=int,
        default=2,
        help="Automatic reacquisitions after an incomplete physical anchor turn. "
        "Invalid stationary captures are discarded, never mapped.",
    )
    parser.add_argument(
        "--anchor-safety-recoveries",
        type=int,
        default=6,
        help="Maximum collision-triggered relocate-and-restart cycles per anchor "
        "pass. Safety interruptions do not consume motor-failure retries.",
    )
    parser.add_argument(
        "--anchor-map-retries",
        type=int,
        default=3,
        help="Complete bidirectional anchor-map rebuilds after independent "
        "stationary validation rejects a candidate. The robot never drives "
        "on rejected geometry.",
    )
    parser.add_argument(
        "--startup-escape-distance-m",
        type=float,
        default=0.12,
        help="Nominal calibrated-footprint escape step. Constrained-start open-"
        "space planning expands this into bounded 0.18-0.65m candidates.",
    )
    parser.add_argument(
        "--startup-escape-speed",
        type=float,
        default=1.0,
        help="Body-axis velocity command for the bounded startup escape. Active "
        "axes are kept above the measured 0.80 drivetrain stiction floor.",
    )
    parser.add_argument(
        "--startup-escape-max-steps",
        type=int,
        default=4,
        help="Maximum verified short translations in one startup recovery. "
        "Recovery continues until the footprint is clear, not merely improved.",
    )
    parser.add_argument(
        "--startup-pivot-relocation-max-steps",
        type=int,
        default=6,
        help="Maximum scan-odometry-verified short translations used to move away "
        "from a startup obstruction before reacquiring the anchor spin.",
    )
    parser.add_argument(
        "--anchor-passes",
        type=int,
        choices=(1, 2),
        default=2,
        help="Independent anchor spins. Two uses opposite directions, ranks both "
        "maps, and continues with the stronger available pass.",
    )
    parser.add_argument(
        "--anchor-deskew",
        choices=("auto", "on", "off"),
        default="auto",
        help="In auto mode, independently build raw and timestamp-deskewed anchor "
        "candidates and retain the one with better scan-match consistency.",
    )
    parser.add_argument(
        "--anchor-deskew-reverse",
        action="store_true",
        default=False,
        help="Reverse LiDAR point acquisition order for deskewing if required by a different host driver.",
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        default=72,
        help="Maximum anchor keyframes stitched per spin pass. Full LiDAR scans are "
        "retained; 72 gives 5-degree heading coverage without solving hundreds "
        "of nearly identical adjacent revolutions.",
    )
    parser.add_argument("--stitch-resolution-m", type=float, default=0.03)
    parser.add_argument("--search-xy-m", type=float, default=0.18)
    parser.add_argument("--theta-window-deg", type=float, default=8.0)
    parser.add_argument("--min-match-score", type=float, default=6.0)
    parser.add_argument(
        "--match-max-points",
        type=int,
        default=700,
        help="Maximum evenly sampled LiDAR points used per pose solve. The full "
        "scan is still written to the map after localization.",
    )
    parser.add_argument(
        "--matcher-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Pose-candidate scoring device. Auto uses CUDA when available and falls back safely to CPU.",
    )
    parser.add_argument(
        "--integrate-min-score",
        type=float,
        default=13.0,
        help="A scan is WRITTEN INTO THE MAP only when its match scores at least this "
        "or when a stationary batch has at least 75 percent compact pose consensus and at least 50 percent "
        "support from committed geometry (adaptive floor 12). Marginal matches "
        "remain localization-only to prevent doubled walls.",
    )
    parser.add_argument(
        "--anchor-min-accept-ratio",
        type=float,
        default=0.75,
        help="Minimum sequential scan acceptance required before an anchor can enter independent validation.",
    )
    parser.add_argument(
        "--anchor-max-heading-gap-deg",
        type=float,
        default=25.0,
        help="Maximum genuinely unobserved anchor arc after accounting for the "
        "body-occluded LiDAR view; incomplete coverage cannot initialize SLAM.",
    )
    parser.add_argument(
        "--anchor-validation-score",
        type=float,
        default=10.0,
        help="Minimum score for each independent stationary anchor-validation scan. "
        "Weak validation discards and automatically rebuilds the anchor.",
    )
    parser.add_argument(
        "--anchor-validation-scans",
        type=int,
        default=5,
        help="Fresh stationary LiDAR revolutions used to validate a completed anchor.",
    )
    parser.add_argument(
        "--anchor-validation-min-inliers",
        type=int,
        default=4,
        help="Minimum mutually consistent stationary solves required to accept an anchor.",
    )
    parser.add_argument(
        "--anchor-validation-min-endpoint-support",
        type=float,
        default=0.60,
        help="Minimum fraction of a stationary scan's endpoints supported by the "
        "candidate occupancy map; prevents a wall subset from scoring as a map.",
    )
    parser.add_argument(
        "--anchor-validation-high-confidence-support",
        type=float,
        default=0.85,
        help="Reciprocal scan-to-map and map-to-scan endpoint support that permits "
        "one exceptionally strong full-360 validation frame.",
    )
    parser.add_argument(
        "--anchor-validation-centre-scatter-m",
        type=float,
        default=0.06,
        help="Maximum robot-centre scatter in the stationary anchor-validation mode.",
    )
    parser.add_argument(
        "--anchor-validation-heading-scatter-deg",
        type=float,
        default=3.0,
        help="Maximum heading scatter in the stationary anchor-validation mode.",
    )
    parser.add_argument(
        "--anchor-consensus-score",
        type=float,
        default=8.0,
        help="Minimum cross-pass match score for bidirectional anchor consensus.",
    )
    parser.add_argument(
        "--anchor-consensus-ratio",
        type=float,
        default=0.75,
        help="Diagnostic fraction of sampled scans expected to agree across passes.",
    )
    parser.add_argument(
        "--anchor-consensus-centre-p90-m",
        type=float,
        default=0.10,
        help="Diagnostic 90th-percentile translation residual across anchor passes.",
    )
    parser.add_argument(
        "--anchor-consensus-heading-p90-deg",
        type=float,
        default=3.0,
        help="Diagnostic 90th-percentile heading residual between anchor passes.",
    )
    # LiDAR extraction (same conventions as the spin mapper).
    parser.add_argument("--forward-angle-deg", type=float, default=90.0)
    parser.add_argument("--valid-angle-half-width-deg", type=float, default=180.0)
    parser.add_argument("--invert-lateral-axis", action="store_true", default=True)
    parser.add_argument("--max-distance-m", type=float, default=8.0)
    parser.add_argument("--min-range-m", type=float, default=0.03)
    parser.add_argument("--min-confidence", type=int, default=0)
    parser.add_argument("--lidar-offset-forward-m", type=float, default=0.229)
    # Exploration.
    parser.add_argument(
        "--spin-only",
        action="store_true",
        default=False,
        help="Do the anchor spin, build and render the map, then STOP — no driving. "
        "Use this to perfect the rotation view in isolation.",
    )
    parser.add_argument(
        "--exit-after-spin-only",
        action="store_true",
        default=False,
        help=(
            "When used with --spin-only, save the initial 360deg map and exit "
            "instead of parking forever to keep the viewer open. Intended for "
            "map-maker handoff scripts."
        ),
    )
    parser.add_argument(
        "--explore-viewpoints",
        type=int,
        default=0,
        help="Maximum viewpoints/approach-steps to drive to and map from; 0 means unlimited.",
    )
    parser.add_argument(
        "--frontier-res-m", type=float, default=0.05, help="Occupancy-analysis grid resolution."
    )
    parser.add_argument(
        "--min-frontier-span-m",
        type=float,
        default=0.40,
        help="Smallest frontier opening worth visiting; anything smaller is an "
        "insignificant crevice. Lowered since robot-radius inflation already "
        "narrows a real opening before it is measured.",
    )
    parser.add_argument("--min-frontier-cells", type=int, default=6)
    parser.add_argument(
        "--self-clear-m",
        type=float,
        default=0.30,
        help="Radius around the robot's own positions asserted as free space, so the "
        "no-return dead zone under the robot is not a false frontier.",
    )
    parser.add_argument(
        "--viewpoint-pullback-m",
        type=float,
        default=0.70,
        help="How far back from the frontier the snapshot viewpoint sits "
        "(inside known-free space, looking out).",
    )
    parser.add_argument(
        "--visited-skip-m",
        type=float,
        default=0.55,
        help="Never re-target a frontier this close to one already snapshotted.",
    )
    parser.add_argument(
        "--robot-radius-m",
        type=float,
        default=0.31,
        help="Planning inflation radius (body half-width 0.28 + margin).",
    )
    parser.add_argument(
        "--physical-body-radius-m",
        type=float,
        default=0.28,
        help="Measured physical chassis radius used only to reject impossible "
        "LiDAR self-returns inside the body; unlike --robot-radius-m this "
        "contains no navigation margin.",
    )
    parser.add_argument(
        "--collision-self-mask-inset-m",
        type=float,
        default=0.0,
        help="Inset from the measured chassis edge used by the LiDAR self-filter. "
        "Zero rejects every physically impossible return inside the measured "
        "body while preserving all external collision points.",
    )
    parser.add_argument(
        "--passable-opening-min-m",
        type=float,
        default=0.90,
        help="Minimum frontier span treated as a crossable room transition. "
        "Smaller gaps are mapped from this side as NARROW-LOOK targets.",
    )
    parser.add_argument(
        "--doorway-ratchet-min-gain-cells",
        type=int,
        default=300,
        help="Newly-known cells required before a passable frontier is recorded as "
        "a completed room transition.",
    )
    parser.add_argument(
        "--doorway-ratchet-slack-m",
        type=float,
        default=0.25,
        help="Distance behind a completed doorway plane still allowed for localization "
        "noise; farther-back targets in completed rooms are excluded.",
    )
    # Navigation — pivot at verified path corners, then drive each straight run
    # on one fixed gyro heading. Localization corrects position without steering;
    # an independent 25Hz controller checks clearance before every command.
    parser.add_argument(
        "--drive-speed",
        type=float,
        default=0.80,
        help="Forward velocity command while driving (must clear wheel stiction ~0.78).",
    )
    parser.add_argument(
        "--drive-command-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=-1.0,
        help="Sign applied to the autonomous x.vel drive command. This robot drives forward with -1; use +1 if another base is wired the opposite way.",
    )
    parser.add_argument(
        "--auto-correct-drive-command-sign",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop and flip --drive-command-sign once if the first straight leg clearly moves away from its waypoint.",
    )
    parser.add_argument(
        "--reverse-progress-trigger-m",
        type=float,
        default=0.18,
        help="Reverse progress distance that triggers the autonomous drive sign auto-correction.",
    )
    parser.add_argument(
        "--turn-speed", type=float, default=0.9, help="Max rotation command while aiming toward the path."
    )
    parser.add_argument(
        "--control-rate-hz",
        type=float,
        default=25.0,
        help="Rate of continuous command streaming and independent safety checks.",
    )
    parser.add_argument(
        "--drive-burst-s",
        type=float,
        default=0.28,
        help="Deprecated compatibility option; continuous drive no longer uses bursts.",
    )
    parser.add_argument(
        "--drive-settle-s",
        type=float,
        default=0.12,
        help="Stationary settling time after each pivot before driving straight.",
    )
    parser.add_argument(
        "--active-yaw-dither",
        action="store_true",
        help=(
            "While a long, collision-clear straight segment is driving, apply a "
            "small bounded yaw oscillation so the 180-degree LiDAR gets side-wall "
            "parallax for rolling localization. Disabled by default; never used "
            "during pivots or short doorway segments."
        ),
    )
    parser.add_argument(
        "--active-yaw-dither-deg",
        type=float,
        default=6.0,
        help="Peak active-perception yaw dither in degrees (default: 6).",
    )
    parser.add_argument(
        "--active-yaw-dither-period-s",
        type=float,
        default=2.5,
        help="Full period of the active-perception yaw dither (default: 2.5s).",
    )
    parser.add_argument(
        "--active-yaw-dither-min-segment-m",
        type=float,
        default=0.80,
        help="Minimum straight segment length before yaw dither is allowed.",
    )
    parser.add_argument(
        "--track-search-xy-m",
        type=float,
        default=0.22,
        help="Translation search window for each continuous tracking match — covers how "
        "far the base rolls between matches.",
    )
    parser.add_argument(
        "--drive-command-distance-scale",
        "--odom-scale",
        dest="drive_command_distance_scale",
        type=float,
        default=0.45,
        help=(
            "Estimated metres per command-unit-second, used only to time bounded "
            "startup/recovery translations. This is not wheel odometry and is "
            "never integrated into pose. --odom-scale remains an alias."
        ),
    )
    parser.add_argument(
        "--bottom-odometry",
        choices=("on", "off", "required"),
        default="on",
        help=(
            "Use validated bottom-camera floor flow as a short translation prior "
            "for LiDAR odometry (default: on, fail-soft)."
        ),
    )
    parser.add_argument(
        "--surface-motion-calibration",
        choices=("on", "off"),
        default="on",
        help=(
            "Use bottom-camera ground texture to select a LiDAR-trained "
            "command-to-motion predictor. This is separate from optical flow; "
            "LiDAR remains the pose authority (default: on)."
        ),
    )
    parser.add_argument(
        "--surface-texture-similarity",
        type=float,
        default=0.88,
        help=(
            "Descriptor similarity required to reuse a known ground surface "
            "instead of creating a new texture region (default: 0.88)."
        ),
    )
    parser.add_argument(
        "--max-speed-mps",
        type=float,
        default=0.45,
        help="Physical top speed of the base. Bounds every tracking/relock search "
        "window (the truth cannot be farther than v_max * elapsed), which is what "
        "stops the matcher from 'teleporting' the pose along a wall.",
    )
    parser.add_argument(
        "--track-theta-window-deg",
        type=float,
        default=8.0,
        help="Heading search window for the tracking match (gyro seeds it to ~0.5deg).",
    )
    parser.add_argument(
        "--aim-tolerance-deg",
        type=float,
        default=3.0,
        help="Start driving forward once the heading error is within this.",
    )
    parser.add_argument(
        "--lookahead-m",
        type=float,
        default=0.20,
        help="Deprecated compatibility option; segmented driving has no carrot.",
    )
    parser.add_argument(
        "--drive-exit-tol-deg",
        type=float,
        default=18.0,
        help="Deprecated compatibility option; segment headings never retarget in motion.",
    )
    parser.add_argument(
        "--heading-hold-gain",
        type=float,
        default=2.0,
        help="Gentle proportional gain holding one fixed straight-segment heading.",
    )
    parser.add_argument(
        "--heading-hold-max",
        type=float,
        default=0.25,
        help="Cap on the heading-hold theta.vel while driving forward.",
    )
    parser.add_argument(
        "--heading-hold-deadband-deg",
        type=float,
        default=1.5,
        help="Do not steer inside this IMU error band; prevents left/right hunting.",
    )
    parser.add_argument(
        "--viewpoint-settle-s",
        type=float,
        default=1.0,
        help="Full stop before every stationary mapping capture. A one-second "
        "settle lets the base finish moving AND lets the IMU stream catch up, "
        "so the LiDAR snapshot and its yaw reference are taken from the same "
        "stationary instant before driving on.",
    )
    parser.add_argument(
        "--viewpoint-max-position-scatter-m",
        type=float,
        default=0.08,
        help="Maximum robot-centre scatter across a stationary mapping batch.",
    )
    parser.add_argument(
        "--viewpoint-max-heading-scatter-deg",
        type=float,
        default=4.0,
        help="Maximum solve-heading scatter before a stationary batch is discarded. "
        "Accepted scans are fused at one median pose, so this cannot smear walls.",
    )
    parser.add_argument(
        "--viewpoint-min-known-ratio",
        type=float,
        default=0.20,
        help="Minimum mean overlap with previously committed LiDAR geometry before "
        "a stationary keyframe batch may extend the map.",
    )
    parser.add_argument(
        "--viewpoint-max-pose-correction-m",
        type=float,
        default=0.25,
        help="Maximum stationary keyframe correction from the propagated pose; "
        "larger innovations are localization hypotheses, not map writes.",
    )
    parser.add_argument(
        "--viewpoint-max-pose-correction-deg",
        type=float,
        default=5.0,
        help="Ordinary stationary heading correction limit. A 6deg correction "
        "is allowed only when a multi-scan batch has strong match score, "
        "at least 50 percent committed-map support, and at least 75 percent pose consensus.",
    )
    parser.add_argument(
        "--relocalization-max-pose-correction-m",
        type=float,
        default=0.15,
        help="Maximum local stationary pose-only correction. Larger innovations "
        "must use multi-scan continuity-constrained localization rather than "
        "teleporting the local pose onto a repeated wall.",
    )
    parser.add_argument(
        "--relocalization-max-pose-correction-deg",
        type=float,
        default=6.0,
        help="Maximum local stationary heading correction; larger corrections "
        "are deferred to IMU-constrained multi-scan global localization.",
    )
    parser.add_argument(
        "--global-relocalization-scans",
        type=int,
        default=3,
        help="Fresh stationary scans voted during wider lost-tracking localization.",
    )
    parser.add_argument(
        "--global-relocalization-search-m",
        type=float,
        default=2.5,
        help="Requested GPU recovery-search radius against GOLD geometry; ordinary "
        "tracking recovery is capped by the odometric continuity bound.",
    )
    parser.add_argument(
        "--global-relocalization-max-pose-correction-m",
        type=float,
        default=0.60,
        help="Maximum pose innovation accepted during ordinary lost-tracking "
        "recovery. Larger whole-map matches are kidnapped-robot hypotheses "
        "and cannot replace continuous odometry without loop-closure proof.",
    )
    parser.add_argument(
        "--global-relocalization-min-known-ratio",
        type=float,
        default=0.35,
        help="Minimum committed-map endpoint support for a whole-map recovery "
        "hypothesis. This is intentionally stricter than normal keyframing.",
    )
    parser.add_argument(
        "--active-localization-probe-deg",
        type=float,
        default=15.0,
        help="Step size for the one-direction collision-checked localization "
        "sweep used when stationary/global localization are ambiguous.",
    )
    parser.add_argument(
        "--active-localization-max-sweep-deg",
        type=float,
        default=30.0,
        help="Maximum one-direction active-localization sweep before acquiring "
        "parallax from a short collision-checked translation.",
    )
    parser.add_argument(
        "--active-localization-translation-m",
        type=float,
        default=0.12,
        help="Short collision-checked translation used after an inconclusive "
        "localization sweep; zero disables translation recovery.",
    )
    parser.add_argument(
        "--frontier-handoff-turn-step-deg",
        type=float,
        default=30.0,
        help="Maximum pivot between overlap-verified stationary snapshots while "
        "turning into a passable frontier.",
    )
    parser.add_argument(
        "--frontier-keyframe-fan-half-deg",
        type=float,
        default=30.0,
        help="Deprecated compatibility option. Doorway mapping now follows one "
        "monotonic turn toward unexplored space and never sweeps back and forth.",
    )
    parser.add_argument(
        "--frontier-keyframe-batch-scans",
        type=int,
        default=3,
        help="Stationary revolutions per routine doorway angle. Three coherent "
        "scans provide a fast pose/map transaction; ambiguous recovery "
        "paths explicitly request five scans.",
    )
    parser.add_argument(
        "--frontier-handoff-sweep-deg",
        type=float,
        default=120.0,
        help="Maximum monotonic turn-and-snapshot arc toward a passable frontier. "
        "This is never a full recovery spin or a bidirectional fan.",
    )
    parser.add_argument(
        "--frontier-turn-cost-per-deg",
        type=float,
        default=4.0,
        help="Soft frontier-utility cost for initial turning. Turning around remains "
        "legal when necessary; it is never treated as unreachable.",
    )
    parser.add_argument(
        "--frontier-forward-turn-limit-deg",
        type=float,
        default=95.0,
        help="Largest initial route turn treated as forward progress. Routes beyond "
        "this remain legal but are selected only when no forward frontier is "
        "reachable.",
    )
    parser.add_argument(
        "--frontier-return-deferral-cycles",
        type=int,
        default=4,
        help="Consecutive planning cycles with no forward route required before a "
        "new rearward objective may be selected. Committed objectives are "
        "exempt so collision replans can finish their current passage.",
    )
    parser.add_argument(
        "--frontier-batch-size",
        type=int,
        default=3,
        help="Number of forward observation targets to complete before beginning "
        "a fresh forward-selection batch. Each stop still replans its safe route "
        "from the newly verified map; stale waypoints are never blindly reused.",
    )
    parser.add_argument(
        "--allow-return-frontiers",
        action="store_true",
        help="Permit automatic selection of a rearward frontier after the forward "
        "batch is exhausted. Disabled by default so exploration does not meander "
        "back through already mapped space.",
    )
    parser.add_argument(
        "--frontier-handoff-min-known-ratio",
        type=float,
        default=0.20,
        help="Minimum fraction of a doorway-handoff scan that must overlap the "
        "trusted map before that scan may extend the map.",
    )
    parser.add_argument(
        "--frontier-handoff-checkpoint-m",
        type=float,
        default=0.35,
        help="Distance between overlap-verified stationary mapping checkpoints "
        "on every long path. Checkpoints grow the trusted map before the robot "
        "can outrun LiDAR overlap; they are not limited to doorway targets.",
    )
    parser.add_argument(
        "--frontier-doorway-keyframe-m",
        type=float,
        default=0.30,
        help="Translation between overlap keyframes while crossing a "
        "passable opening. Kept shorter than ordinary path checkpoints so "
        "the previous room remains visible during the handoff.",
    )
    parser.add_argument(
        "--room-entry-keyframes",
        type=int,
        default=4,
        help="Number of subsequent observation legs that keep building dense, "
        "multi-angle rolling submaps after crossing into newly revealed space.",
    )
    # Arms are QUARANTINED by default (hardware damage 2026-07-21): no torque, no
    # targets, ever — unless --arm-stow explicitly opts in.
    parser.add_argument(
        "--arm-stow",
        action="store_true",
        default=False,
        help="OPT IN to driving the arms to the saved stow pose at startup. Off by "
        "default after the arm-flip incident; leave off until the arm read/write "
        "scale is verified on hardware.",
    )
    parser.add_argument(
        "--arm-stow-pose",
        default=str(DEFAULT_POSE_PATH),
        help="Path to the saved arm stow pose JSON (used only with --arm-stow).",
    )
    parser.add_argument(
        "--arm-settle-s",
        type=float,
        default=3.0,
        help="Seconds to hold the stow command so the arms reach the pose before spinning.",
    )
    # Forward collider box (local costmap): the ONLY thing that stops the drive.
    parser.add_argument(
        "--box-near-m", type=float, default=0.10, help="Near edge of the collider box ahead of the LiDAR."
    )
    parser.add_argument(
        "--box-depth-m",
        type=float,
        default=0.75,
        help="Far edge of the collider box — how far ahead a novel obstacle is watched for.",
    )
    parser.add_argument(
        "--box-half-width-m",
        type=float,
        default=0.34,
        help="Half-width of the collider box (~body half-width + margin).",
    )
    parser.add_argument(
        "--box-min-points",
        type=int,
        default=6,
        help="Novel returns needed inside the box to call it an obstacle.",
    )
    parser.add_argument(
        "--novel-res-m",
        type=float,
        default=0.12,
        help="A box return is NOVEL if the map has no point within ~this of it.",
    )
    parser.add_argument(
        "--hard-stop-m",
        type=float,
        default=0.25,
        help="Last-resort halt if anything is this close on the nose, novel or not — "
        "guards against localization error into a mapped wall.",
    )
    parser.add_argument(
        "--hard-stop-half-width-m",
        type=float,
        default=0.28,
        help="Live forward lane half-width. This must cover the measured 0.28m body "
        "half-width; mapped shoulder clearance is enforced separately using "
        "the full --robot-radius-m swept footprint.",
    )
    parser.add_argument(
        "--collision-box-file",
        default=str(DEFAULT_COLLISION_BOX_PATH),
        help="Saved full-angle LiDAR collision-envelope calibration.",
    )
    parser.add_argument(
        "--collision-box-mode",
        choices=("auto", "off", "required"),
        default="auto",
        help="auto loads a calibration when present; required refuses to explore without one.",
    )
    parser.add_argument(
        "--swept-guard-lookahead-m",
        type=float,
        default=0.12,
        help="How far ahead to project the full robot-radius footprint on the map. "
        "A mapped shoulder collision stops/replans; unknown space stops for a "
        "new stationary viewpoint before proceeding. Keep this short because "
        "the robot follows curved paths; 25Hz checks cover stopping latency.",
    )
    parser.add_argument(
        "--lidar-safety-max-age-s",
        type=float,
        default=1.50,
        help="Stop only when the LiDAR feed is genuinely unavailable, not for a "
        "single delayed revolution. Forward motion resumes on the same "
        "straight command as soon as one fresh revolution arrives.",
    )
    parser.add_argument(
        "--obstacle-wait-s",
        type=float,
        default=4.0,
        help="How long to stop and watch a novel obstacle before replanning around it. "
        "Confirmed returns enter only the temporary planner costmap.",
    )
    parser.add_argument(
        "--obstacle-memory-s",
        type=float,
        default=15.0,
        help="Lifetime of confirmed dynamic returns in the temporary planner layer. "
        "They never modify SLAM occupancy geometry.",
    )
    parser.add_argument("--waypoint-tol-m", type=float, default=0.18)
    parser.add_argument(
        "--max-leg-seconds",
        type=float,
        default=35.0,
        help="Give up on a single waypoint leg after this long (something is wedged).",
    )
    parser.add_argument(
        "--max-mission-seconds",
        type=float,
        default=0.0,
        help="Mission time limit in seconds; 0 means unlimited.",
    )
    parser.add_argument(
        "--saved-map",
        default=None,
        help=(
            "Output NPZ map path. If omitted, a new timestamped map is written "
            "beside the default map; existing maps are never overwritten."
        ),
    )
    parser.add_argument(
        "--no-save-map",
        action="store_true",
        help="Disable automatic saved-map persistence for this run.",
    )
    # Panorama + rerun.
    parser.add_argument("--panorama", choices=["on", "off"], default="on")
    parser.add_argument("--slam-input-endpoint", default=None)
    parser.add_argument("--panorama-hz", type=float, default=5.0)
    parser.add_argument("--rerun-mode", choices=["web", "local"], default="web")
    parser.add_argument("--rerun-grpc-port", type=int, default=9877)
    parser.add_argument("--rerun-web-port", type=int, default=9878)
    args = parser.parse_args()

    # Mirror EVERYTHING the console prints (including tracebacks on stderr)
    # into one file that is overwritten each run, so a run can be analysed
    # without copying the terminal.  faulthandler additionally dumps thread
    # stacks into the same file on a hard crash.
    import faulthandler
    import os as _os
    import sys as _sys

    run_log_path = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "sourccey_explore_last_run.log",
    )
    try:
        run_log_file = open(run_log_path, "w", encoding="utf-8", errors="replace")
        run_log_file.write(
            f"=== sourccey_explore run started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            f"args: {vars(args)}\n\n"
        )
        run_log_file.flush()
        _sys.stdout = _TeeLog(_sys.stdout, run_log_file)
        _sys.stderr = _TeeLog(_sys.stderr, run_log_file)
        faulthandler.enable(file=run_log_file)
        print(f"[log] console output is mirrored to {run_log_path} (overwritten each run).")
    except OSError as exc:
        print(f"[log] WARNING: could not open run log file ({exc}); console only.")
    if args.saved_map is None:
        # Include microseconds so two launches in the same second cannot
        # target the same output file.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dated = DEFAULT_SAVED_MAP_PATH.with_name(
            f"{DEFAULT_SAVED_MAP_PATH.stem}_{timestamp}{DEFAULT_SAVED_MAP_PATH.suffix}"
        )
        args.saved_map = str(dated)
        print(f"[map] timestamped output selected: {dated}")
    if not 0.0 < float(args.physical_body_radius_m) <= float(args.robot_radius_m):
        parser.error("--physical-body-radius-m must be positive and no larger than --robot-radius-m")
    if not 0.0 <= float(args.collision_self_mask_inset_m) < float(args.physical_body_radius_m):
        parser.error("--collision-self-mask-inset-m must be nonnegative and smaller than the body radius")
    if not 0.0 <= float(args.viewpoint_min_known_ratio) <= 1.0:
        parser.error("--viewpoint-min-known-ratio must be in [0, 1]")
    if float(args.viewpoint_max_pose_correction_m) <= 0.0:
        parser.error("--viewpoint-max-pose-correction-m must be positive")
    if float(args.viewpoint_max_pose_correction_deg) <= 0.0:
        parser.error("--viewpoint-max-pose-correction-deg must be positive")
    if float(args.relocalization_max_pose_correction_m) <= 0.0:
        parser.error("--relocalization-max-pose-correction-m must be positive")
    if float(args.relocalization_max_pose_correction_deg) <= 0.0:
        parser.error("--relocalization-max-pose-correction-deg must be positive")
    if int(args.global_relocalization_scans) < 2:
        parser.error("--global-relocalization-scans must be at least 2")
    if int(args.frontier_batch_size) < 1:
        parser.error("--frontier-batch-size must be at least 1")
    if float(args.global_relocalization_search_m) <= 0.0:
        parser.error("--global-relocalization-search-m must be positive")
    if float(args.global_relocalization_max_pose_correction_m) <= 0.0:
        parser.error("--global-relocalization-max-pose-correction-m must be positive")
    if not 0.0 < float(args.global_relocalization_min_known_ratio) <= 1.0:
        parser.error("--global-relocalization-min-known-ratio must be in (0, 1]")
    if int(args.room_entry_keyframes) < 0:
        parser.error("--room-entry-keyframes must be nonnegative")
    if not 0.0 <= float(args.active_localization_probe_deg) <= 45.0:
        parser.error("--active-localization-probe-deg must be in [0, 45]")
    if (
        not float(args.active_localization_probe_deg)
        <= float(args.active_localization_max_sweep_deg)
        <= 120.0
    ):
        parser.error("--active-localization-max-sweep-deg must be at least one probe step and <= 120")
    if not 0.0 <= float(args.active_localization_translation_m) <= 0.30:
        parser.error("--active-localization-translation-m must be in [0, 0.30]")
    if float(args.obstacle_memory_s) <= 0.0:
        parser.error("--obstacle-memory-s must be positive")
    if float(args.anchor_stall_seconds) <= 0.0:
        parser.error("--anchor-stall-seconds must be positive")
    if int(args.anchor_motion_retries) < 0:
        parser.error("--anchor-motion-retries must be nonnegative")
    if int(args.anchor_safety_recoveries) < 0:
        parser.error("--anchor-safety-recoveries must be nonnegative")
    if int(args.anchor_map_retries) < 0:
        parser.error("--anchor-map-retries must be nonnegative")
    if int(args.anchor_validation_scans) < 1:
        parser.error("--anchor-validation-scans must be positive")
    if int(args.anchor_validation_min_inliers) < 1:
        parser.error("--anchor-validation-min-inliers must be positive")
    if int(args.anchor_validation_min_inliers) > int(args.anchor_validation_scans):
        parser.error("--anchor-validation-min-inliers cannot exceed --anchor-validation-scans")
    if not 0.0 < float(args.anchor_validation_min_endpoint_support) <= 1.0:
        parser.error("--anchor-validation-min-endpoint-support must be in (0, 1]")
    if not 0.0 < float(args.anchor_validation_high_confidence_support) <= 1.0:
        parser.error("--anchor-validation-high-confidence-support must be in (0, 1]")
    if int(args.startup_pivot_relocation_max_steps) < 0:
        parser.error("--startup-pivot-relocation-max-steps must be nonnegative")
    if float(args.anchor_spin_max_speed) <= 0.0:
        parser.error("--anchor-spin-max-speed must be positive")
    if float(args.startup_escape_distance_m) <= 0.0:
        parser.error("--startup-escape-distance-m must be positive")
    if float(args.startup_escape_speed) <= 0.0:
        parser.error("--startup-escape-speed must be positive")
    if int(args.startup_escape_max_steps) < 1:
        parser.error("--startup-escape-max-steps must be at least one")
    if float(args.frontier_doorway_keyframe_m) <= 0.0:
        parser.error("--frontier-doorway-keyframe-m must be positive")
    if not 3 <= int(args.frontier_keyframe_batch_scans) <= 8:
        parser.error("--frontier-keyframe-batch-scans must be between 3 and 8")
    if not 0.0 < float(args.active_yaw_dither_deg) <= 12.0:
        parser.error("--active-yaw-dither-deg must be in (0, 12]")
    if float(args.active_yaw_dither_period_s) <= 0.5:
        parser.error("--active-yaw-dither-period-s must be greater than 0.5")
    if float(args.active_yaw_dither_min_segment_m) <= 0.0:
        parser.error("--active-yaw-dither-min-segment-m must be positive")
    if float(args.reverse_progress_trigger_m) <= 0.0:
        parser.error("--reverse-progress-trigger-m must be positive")
    if not 0.50 <= float(args.surface_texture_similarity) <= 0.99:
        parser.error("--surface-texture-similarity must be in [0.50, 0.99]")
    configure_candidate_scoring_device(str(args.matcher_device))

    lidar_host = args.lidar_host or args.remote_ip
    imu_host = args.imu_host or args.remote_ip
    lever_m = float(args.lidar_offset_forward_m)
    forward_offset = _phys_forward_offset_deg(args)
    drive_command_sign = [1.0 if float(args.drive_command_sign) >= 0.0 else -1.0]
    drive_command_sign_flips = [0]
    print(
        f"[explore] autonomous drive x.vel sign {drive_command_sign[0]:+.0f} "
        f"(auto-correct {'on' if bool(args.auto_correct_drive_command_sign) else 'off'})."
    )
    # This guard projects the CURRENT heading as a straight line; it is not the
    # footprint size. Beyond 12cm that approximation cuts across curves that the
    # path follower will actually turn through and falsely seals doorways. The
    # independent raw-LiDAR stop lane still watches 25cm ahead at 25Hz.
    swept_guard_m = min(0.12, max(0.0, float(args.swept_guard_lookahead_m)))
    if float(args.swept_guard_lookahead_m) > 0.12:
        print(
            f"[explore] swept guard request {float(args.swept_guard_lookahead_m):.2f}m "
            f"capped to {swept_guard_m:.2f}m (straight-heading guard; path may curve)."
        )

    # ---- LiDAR feed (no self-mask: the arms are stowed out of the beam instead) ----
    feed = DirectLidarFeed(lidar_host, int(args.lidar_port))
    feed.start()
    print(f"[explore] LiDAR feed connecting to {lidar_host}:{args.lidar_port} ...")
    first_frame = None
    for _attempt in range(2):
        _first_id, first_frame = feed.wait_for_frame_after(
            after_frame_id=-1,
            timeout_s=5.0,
            min_frame_advances=1,
        )
        if first_frame is not None:
            break
    if first_frame is None:
        _abort(
            "No complete LiDAR revolution arrived within 10s; ExploreSystem "
            "requires a live LiDAR stream and will not wait indefinitely."
        )

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
    robot = SourcceyClient(
        SourcceyClientConfig(
            id=args.robot_id,
            remote_ip=args.remote_ip,
            host_session_mode="slam_mapping",
        )
    )
    robot.connect()
    _send_stop(robot)
    print(f"[explore] robot connected ({args.remote_ip}).")

    # ---- Arms: QUARANTINED (2026-07-21 — a commanded "hold current position"
    # physically flipped and cracked the left arm; the read-back scale is not
    # trusted against the write path until verified on the hardware). Default is
    # to NEVER torque or target the arms; --arm-stow is an explicit opt-in.
    remote_arms_available = bool(getattr(robot, "remote_arms_available", True))
    stow_pose = load_pose(args.arm_stow_pose) if bool(args.arm_stow) and remote_arms_available else None
    if stow_pose is not None:
        print(f"[explore] --arm-stow OPT-IN: moving arms to {args.arm_stow_pose} ...")
        apply_pose_blocking(robot, stow_pose, settle_s=float(args.arm_settle_s))
        # The servo controller holds its latched target. Stream torque state
        # only from here onward so an arm-bus write can never suppress the wheel
        # command that follows it on the host.
        hold = _latched_arm_torque_state(stowed=True)
        print(
            "[explore] arms stowed; target latched and torque held without "
            "resending arm bus writes in the base-control stream."
        )
    else:
        hold = _latched_arm_torque_state(stowed=False)
        if bool(args.arm_stow) and not remote_arms_available:
            print(
                "[explore] --arm-stow ignored: Pi host reports follower-arm "
                "hardware disabled/removed; exploration remains base-only."
            )
        elif bool(args.arm_stow):
            print(
                f"[explore] WARNING: --arm-stow set but no pose at {args.arm_stow_pose}; arms left untorqued."
            )
        else:
            print("[explore] arms left untorqued (arm motion is quarantined; --arm-stow to opt in).")

    collision_profile: dict | None = None
    if str(args.collision_box_mode) != "off":
        try:
            collision_profile = _load_collision_box(args.collision_box_file)
        except (OSError, ValueError, TypeError) as exc:
            if str(args.collision_box_mode) == "required":
                _abort(f"Invalid collision-box calibration: {exc}")
            print(f"[explore] WARNING: collision-box calibration ignored: {exc}")
        if collision_profile is None and str(args.collision_box_mode) == "required":
            _abort(f"Collision-box calibration required but not found: {args.collision_box_file}")
        if collision_profile is not None:
            learned = collision_profile["ranges_m"]
            self_return_half_extent = max(
                0.0,
                float(args.physical_body_radius_m) - float(args.collision_self_mask_inset_m),
            )
            print(
                f"[explore] calibrated collision box ENABLED "
                f"({sum(v is not None for v in learned)}/{len(learned)} angular bins; "
                "side intrusions are single-bin hard stops; "
                f"rounded-square self-return half-extent "
                f"{self_return_half_extent:.2f}m "
                f"about the robot centre is excluded)."
            )
        else:
            print("[explore] no collision-box calibration; using legacy forward stop lane.")

    # ---- Rerun ----
    grpc_port, web_port = _pick_free_ports(int(args.rerun_grpc_port), int(args.rerun_web_port))
    rr, _viewer_url = _init_rerun(
        session_name="sourccey_explore", mode=args.rerun_mode, grpc_port=grpc_port, web_port=web_port
    )
    _configure_rerun_layout(rr)

    # ---- Camera stream: panorama display plus bottom-camera ground odometry ----
    cam_sub = None
    eye_mosaic = None
    cam_left_key = cam_right_key = None
    ground_odom = None
    surface_texture_enabled = str(args.surface_motion_calibration) != "off"
    surface_texture_ready = False
    if str(args.panorama) == "on" or str(args.bottom_odometry) != "off" or surface_texture_enabled:
        try:
            from sourccey_elevated_safety import SlamCameraSubscriber, endpoint_from_remote_ip

            cam_left_key, cam_right_key = "front_left", "front_right"
            cam_endpoint = str(args.slam_input_endpoint or "").strip() or endpoint_from_remote_ip(
                args.remote_ip
            )
            camera_keys: list[str] = []
            if str(args.panorama) == "on":
                camera_keys.extend((cam_left_key, cam_right_key))
                # The upper half of the downward-facing camera provides a
                # cleaner long-range colour cue than the slanted eye cameras.
                # It is optional: a missing bottom stream must not disable
                # LiDAR mapping or the fused panorama.
                camera_keys.append("bottom")
            if str(args.bottom_odometry) != "off":
                camera_keys.append("bottom")
            if surface_texture_enabled:
                camera_keys.append("bottom")
            cam_sub = SlamCameraSubscriber(
                endpoint=cam_endpoint,
                camera_keys=tuple(dict.fromkeys(camera_keys)),
            )
            cam_sub.start()
            if str(args.bottom_odometry) != "off":
                bottom_health = wait_for_live_bottom_camera(
                    cam_sub,
                    timeout_s=5.0,
                )
                if not bottom_health.ready and str(args.bottom_odometry) == "required":
                    _abort(
                        "Bottom-camera odometry required but its live stream "
                        f"failed the health check: {bottom_health.reason}."
                    )
                if bottom_health.ready:
                    ground_odom = BottomCameraGroundOdometry(cam_sub, imu)
                    ground_odom.start()
                    # Texture classification consumes the same live frame, but
                    # remains logically independent from the optional flow
                    # worker.  In particular, a flow measurement must never be
                    # mistaken for a surface identity.
                    surface_texture_ready = bool(surface_texture_enabled)
                    print(
                        "[explore] bottom-camera ground odometry ENABLED; "
                        "validated optical flow supplies translation priors only "
                        f"({bottom_health.reason})."
                    )
                else:
                    print(
                        "[explore] WARNING: bottom camera unavailable; continuing "
                        "with LiDAR local odometry + IMU yaw only "
                        f"({bottom_health.reason})."
                    )
            elif surface_texture_enabled:
                bottom_health = wait_for_live_bottom_camera(
                    cam_sub,
                    timeout_s=5.0,
                )
                surface_texture_ready = bool(bottom_health.ready)
                if surface_texture_ready:
                    print(
                        "[explore] ground-texture motion calibration ENABLED; "
                        "texture selects a LiDAR-trained command-motion prior "
                        f"({bottom_health.reason})."
                    )
                else:
                    print(
                        "[explore] WARNING: ground-texture calibration unavailable; "
                        "continuing with the neutral motion model "
                        f"({bottom_health.reason})."
                    )
            else:
                surface_texture_ready = ground_odom is not None
            if str(args.panorama) == "on":
                from sourccey_eye_panorama import load_perception_mosaic

                if not cam_sub.wait_for_frames(timeout_s=5.0, required=(cam_left_key, cam_right_key)):
                    raise RuntimeError("no eye-camera frames within 5s")
                eye_mosaic = load_perception_mosaic()
                print("[explore] panorama view ON — rerun entity cameras/panorama")
        except Exception as exc:  # noqa: BLE001
            capability = (
                "panorama unavailable; bottom odometry remains active"
                if ground_odom is not None
                else "camera assistance unavailable"
            )
            print(f"[explore] {capability} ({type(exc).__name__}: {exc}).")
            if str(args.bottom_odometry) == "required" and ground_odom is None:
                raise
            eye_mosaic = None

    # Keep texture regions and their LiDAR-trained motion scales completely
    # separate from the optional optical-flow worker above. The surface atlas
    # may supply a *prediction* between scans, but cannot alter a pose until a
    # fresh LiDAR registration accepts it.
    surface_motion_atlas = SurfaceMotionAtlas()
    surface_motion_atlas.similarity_threshold = float(args.surface_texture_similarity)
    surface_texture_last_at = [-math.inf]
    surface_texture_last_id: list[int | None] = [None]
    # This is intentionally a command *prediction*, not wheel odometry.  It
    # only places the next LiDAR solve in the right small bubble; a fresh
    # accepted LiDAR registration is required before any of it becomes a
    # learned surface calibration.
    surface_prior_last_at = [time.monotonic()]
    surface_prior_reference_pose: list[Pose2D | None] = [None]
    surface_prior_command_body = [np.zeros(2, dtype=np.float64)]
    surface_prior_uncertainty_m = [0.0]

    def _observe_ground_texture(*, force: bool = False) -> int | None:
        """Classify the current floor texture without assigning a map pose."""
        if not surface_texture_enabled or not surface_texture_ready or cam_sub is None:
            return None
        now = time.monotonic()
        if not force and now - surface_texture_last_at[0] < 0.25:
            return surface_texture_last_id[0]
        try:
            frame, age_s = cam_sub.latest("bottom")
            if frame is None or age_s is None or float(age_s) > 0.75:
                return surface_texture_last_id[0]
            signature = ground_texture_signature(frame)
            surface_id, changed, similarity = surface_motion_atlas.observe_texture(signature)
            if surface_id is not None:
                if changed:
                    print(
                        f"[surface] ground texture {'new' if similarity >= 0.999 else 'changed'} "
                        f"to surface {surface_id} (similarity {similarity:.2f}); "
                        "awaiting LiDAR-confirmed motion before calibration."
                    )
                surface_texture_last_id[0] = surface_id
                surface_texture_last_at[0] = now
            return surface_id
        except Exception:
            return surface_texture_last_id[0]

    def _reset_surface_motion_prior(pose: Pose2D | None = None) -> None:
        """Start a new LiDAR-validated surface-motion interval.

        Calling this at a pivot/stop prevents a turn or a stalled command from
        being learned as carpet slip.  IMU remains responsible for heading.
        """
        surface_prior_last_at[0] = time.monotonic()
        surface_prior_reference_pose[0] = pose
        surface_prior_command_body[0] = np.zeros(2, dtype=np.float64)
        surface_prior_uncertainty_m[0] = 0.0

    def _apply_surface_motion_prediction(
        pose: Pose2D,
        *,
        forward_command: float,
        left_command: float = 0.0,
    ) -> Pose2D:
        """Advance only the next LiDAR seed using a texture-calibrated command.

        The controller command is converted through the conservative baseline
        speed model, then through the currently recognized floor model.  This
        function never writes a pose acceptance or the map by itself.
        """
        if not surface_texture_enabled:
            return pose
        now = time.monotonic()
        elapsed = min(0.16, max(0.0, now - surface_prior_last_at[0]))
        surface_prior_last_at[0] = now
        if elapsed <= 1e-3 or (abs(float(forward_command)) < 1e-4 and abs(float(left_command)) < 1e-4):
            return pose
        _observe_ground_texture()
        # ``drive_command_distance_scale`` is deliberately only a bounded
        # control prediction.  The surface model corrects it later from a
        # confirmed LiDAR displacement, rather than claiming fake wheel data.
        baseline_body = float(args.drive_command_distance_scale) * elapsed * np.asarray(
            [float(forward_command), float(left_command)], dtype=np.float64
        )
        prediction_body, uncertainty = surface_motion_atlas.predict_body_translation(
            float(baseline_body[0]),
            float(baseline_body[1]),
        )
        heading = math.radians(float(pose.theta_deg) + forward_offset)
        rotation = np.asarray(
            [[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]],
            dtype=np.float64,
        )
        centre = _robot_centre_from_lidar_pose(pose, lever_m, forward_offset)
        if surface_prior_reference_pose[0] is None:
            surface_prior_reference_pose[0] = pose
        surface_prior_command_body[0] += baseline_body
        surface_prior_uncertainty_m[0] = min(
            0.35,
            float(surface_prior_uncertainty_m[0]) + float(uncertainty),
        )
        return _lidar_pose_from_robot_centre(
            centre + rotation @ prediction_body,
            float(pose.theta_deg),
            lever_m,
            forward_offset,
        )

    def _learn_surface_motion_from_lidar(accepted_pose: Pose2D) -> None:
        """Teach the active surface only from one bounded LiDAR correction."""
        reference = surface_prior_reference_pose[0]
        command = surface_prior_command_body[0]
        if reference is None or float(np.hypot(*command)) < 0.04:
            _reset_surface_motion_prior(accepted_pose)
            return
        if not bool(obstacle_pose_trust["trusted"]):
            # A displacement measured between two poses is only as good as the
            # LiDAR->map transform that produced them.  While localization is
            # quarantined the interval is discarded, not taught (field
            # 2026-08-05: intervals collected during a quarantine taught
            # forward x1.44; local odometry immediately disagreed by 63cm and
            # tracking was lost for the rest of the run).
            _reset_surface_motion_prior(accepted_pose)
            return
        ref_centre = _robot_centre_from_lidar_pose(reference, lever_m, forward_offset)
        actual_map = _robot_centre_from_lidar_pose(accepted_pose, lever_m, forward_offset) - ref_centre
        heading = math.radians(float(reference.theta_deg) + forward_offset)
        # Map-vector to the reference body frame: [forward, left].
        actual_body = np.asarray(
            [
                math.cos(heading) * actual_map[0] + math.sin(heading) * actual_map[1],
                -math.sin(heading) * actual_map[0] + math.cos(heading) * actual_map[1],
            ],
            dtype=np.float64,
        )
        turn_delta = ((float(accepted_pose.theta_deg) - float(reference.theta_deg) + 180.0) % 360.0) - 180.0
        learned = surface_motion_atlas.update_from_lidar(
            float(command[0]),
            float(command[1]),
            float(actual_body[0]),
            float(actual_body[1]),
            turn_delta_deg=turn_delta,
        )
        if learned:
            model = surface_motion_atlas.model()
            if model is not None and model.observations in {1, 4, 8, 16}:
                print(
                    f"[surface] LiDAR calibrated surface {model.surface_id}: "
                    f"forward x{model.forward_scale:.2f}, lateral x{model.lateral_scale:.2f} "
                    f"from {model.observations} trusted motion interval(s)."
                )
        _reset_surface_motion_prior(accepted_pose)

    def _record_surface_region_at_trusted_pose(pose: Pose2D) -> None:
        """Persist a floor-region/boundary label only after LiDAR trusts pose."""
        if _observe_ground_texture() is not None:
            surface_motion_atlas.record_trusted_landmark((pose.x, pose.y, pose.theta_deg))

    # Appearance keyframes are deliberately coarse and optional.  They are
    # stored alongside accepted LiDAR poses and later used only to break ties
    # between geometrically plausible headings in symmetric rooms.
    visual_color_landmarks: list[dict[str, object]] = []
    visual_last_keyframe_at = [-math.inf]

    def _log_panorama() -> None:
        if cam_sub is None or eye_mosaic is None:
            return
        try:
            left, _al = cam_sub.latest(cam_left_key)
            right, _ar = cam_sub.latest(cam_right_key)
            if left is None or right is None:
                return
            panorama = eye_mosaic.compose(left, right)
            rr.log("cameras/panorama", rr.Image(panorama[:, :, ::-1]))
            now = time.monotonic()
            # One keyframe per second is enough for a directional appearance
            # cue and avoids turning the saved map into a video archive.
            if now - visual_last_keyframe_at[0] < 1.0:
                return
            try:
                pose = cur_pose
                if not world_map.scans or not any(scan.gold for scan in world_map.scans):
                    return
            except (NameError, UnboundLocalError):
                return
            signature = color_signature(panorama)
            bottom_frame, _bottom_age = cam_sub.latest("bottom")
            bottom_signature = (
                bottom_upper_color_signature(bottom_frame)
                if bottom_frame is not None
                else []
            )
            if not signature and not bottom_signature:
                return
            if visual_color_landmarks:
                previous = visual_color_landmarks[-1]
                prev_pose = np.asarray(previous.get("pose", ()), dtype=np.float64)
                if prev_pose.shape == (3,) and np.hypot(float(pose.x) - prev_pose[0], float(pose.y) - prev_pose[1]) < 0.08:
                    heading_delta = abs((float(pose.theta_deg) - prev_pose[2] + 180.0) % 360.0 - 180.0)
                    if heading_delta < 12.0:
                        return
            visual_color_landmarks.append(
                {
                    "version": 1,
                    "pose": [float(pose.x), float(pose.y), float(pose.theta_deg)],
                    "signature": signature,
                    "bottom_signature": bottom_signature,
                }
            )
            if len(visual_color_landmarks) > 512:
                del visual_color_landmarks[:-512]
            visual_last_keyframe_at[0] = now
        except Exception:
            pass

    live_lidar_log_state = {"frame_id": -1, "time": 0.0}

    def _log_live_lidar(frame=None) -> None:
        """Publish the unstitched, isolated-speck-filtered LiDAR and envelope."""
        frame_id, latest_frame = feed.latest()
        if frame is None:
            frame = latest_frame
        if frame is None:
            return
        now = time.monotonic()
        if (
            int(frame_id) == int(live_lidar_log_state["frame_id"])
            or now - float(live_lidar_log_state["time"]) < 0.08
        ):
            return
        local = _scan_local(frame, args)
        if not len(local):
            return
        points = _filter_isolated_lidar_specks(_to_forward_frame(local, forward_offset))
        if not len(points):
            return
        xyz = np.column_stack(
            (
                points[:, 0],
                points[:, 1],
                np.zeros(len(points), dtype=np.float64),
            )
        ).astype(np.float32)
        rr.log(
            "live_lidar/returns",
            rr.Points3D(xyz, colors=[[80, 255, 120]], radii=0.012),
        )
        with contextlib.suppress(Exception):
            rr.log("live_lidar2d/returns", rr.Points2D(points.astype(np.float32), colors=[[80, 255, 120]], radii=0.025))
        if collision_profile is not None:
            hit = _collision_box_violation(
                points,
                collision_profile,
                lidar_offset_forward_m=lever_m,
                physical_body_radius_m=float(args.physical_body_radius_m),
                self_mask_inset_m=float(args.collision_self_mask_inset_m),
            )
            if hit is not None and np.asarray(hit[0]).shape == (len(points),):
                violating = xyz[np.asarray(hit[0], dtype=bool)]
                rr.log(
                    "live_lidar/collision_returns",
                    rr.Points3D(
                        violating,
                        colors=[[255, 55, 40]],
                        radii=0.022,
                    ),
                )
                with contextlib.suppress(Exception):
                    rr.log("live_lidar2d/collision_returns", rr.Points2D(points[np.asarray(hit[0], dtype=bool)].astype(np.float32), colors=[[255, 55, 40]], radii=0.04))
            else:
                rr.log(
                    "live_lidar/collision_returns",
                    rr.Points3D(np.empty((0, 3), dtype=np.float32)),
                )
                with contextlib.suppress(Exception):
                    rr.log("live_lidar2d/collision_returns", rr.Points2D(np.zeros((0, 2), dtype=np.float32)))
            ranges = _collision_box_effective_ranges(collision_profile)
            bin_size = float(collision_profile["bin_size_deg"])
            angles = np.radians(-180.0 + (np.arange(len(ranges)) + 0.5) * bin_size)
            valid = np.isfinite(ranges)
            outline = np.column_stack(
                (
                    ranges[valid] * np.cos(angles[valid]),
                    ranges[valid] * np.sin(angles[valid]),
                    np.zeros(int(np.count_nonzero(valid)), dtype=np.float64),
                )
            ).astype(np.float32)
            if len(outline) >= 2:
                outline = np.vstack((outline, outline[:1]))
                rr.log(
                    "live_lidar/collision_envelope",
                    rr.LineStrips3D(
                        [outline],
                        colors=[[0, 210, 255]],
                        radii=0.008,
                    ),
                )
                with contextlib.suppress(Exception):
                    rr.log("live_lidar2d/collision_envelope", rr.LineStrips2D([outline[:, :2].tolist()], colors=[[0, 210, 255]], radii=0.018))
        robot_origin = np.array([[-lever_m, 0.0, 0.0]], dtype=np.float32)
        rr.log(
            "live_lidar/robot_centre",
            rr.Points3D(robot_origin, colors=[[255, 170, 30]], radii=0.035),
        )
        with contextlib.suppress(Exception):
            rr.log("live_lidar2d/robot_centre", rr.Points2D(robot_origin[:, :2], colors=[[255, 170, 30]], radii=0.055))
        rr.log(
            "live_lidar/physical_forward",
            rr.Arrows3D(
                origins=robot_origin,
                vectors=np.array([[0.50, 0.0, 0.0]], dtype=np.float32),
                colors=[[255, 255, 255]],
            ),
        )
        with contextlib.suppress(Exception):
            rr.log(
                "live_lidar2d/physical_forward",
                rr.LineStrips2D([[[float(-lever_m), 0.0], [float(-lever_m) + 0.50, 0.0]]], colors=[[255, 255, 255]], radii=0.018),
            )
        live_lidar_log_state["frame_id"] = int(frame_id)
        live_lidar_log_state["time"] = now

    _log_live_lidar(first_frame)
    live_lidar_publisher_running = [True]

    def _live_lidar_publisher_loop() -> None:
        while live_lidar_publisher_running[0]:
            with contextlib.suppress(Exception):
                _log_live_lidar()
            time.sleep(0.05)

    live_lidar_publisher = threading.Thread(
        target=_live_lidar_publisher_loop,
        name="rerun-live-lidar",
        daemon=True,
    )
    live_lidar_publisher.start()

    def _stop_live_lidar_publisher() -> None:
        live_lidar_publisher_running[0] = False
        live_lidar_publisher.join(timeout=1.0)

    print(
        "[rerun] live LiDAR view is pinned in the robot-frame side pane: "
        "live_lidar/returns (green, isolated specks filtered), collision "
        "envelope (cyan), current intrusions (red), physical forward (white)."
    )

    # The SLAM occupancy posterior contains validated keyframes only. Dynamic
    # obstacles and hard-stop offenders are kept in a separate decaying planner
    # layer below; navigation evidence must never become localization geometry.
    world_map = WorldMap(
        grid_res_m=float(args.frontier_res_m),
        footprint_clear_m=float(args.self_clear_m),
        lidar_offset_m=lever_m,
        forward_offset_deg=forward_offset,
    )
    mission_t0 = time.monotonic()
    controller = BaseController(
        robot,
        hold,
        imu,
        rate_hz=float(args.control_rate_hz),
        turn_speed=float(args.turn_speed),
        hold_gain=float(args.heading_hold_gain),
        hold_max=float(args.heading_hold_max),
        hold_deadband_deg=float(args.heading_hold_deadband_deg),
    )

    def _rotation_safety_check(remaining_yaw_deg: float) -> str | None:
        """Validate the requested rotational footprint trajectory."""
        if collision_profile is None:
            return None
        _frame_id, frame = feed.latest()
        if frame is None:
            return "LiDAR safety unavailable during pivot (no scan)"
        frame_age_s = time.time() - float(getattr(frame, "received_wall_ts", 0.0) or 0.0)
        if not math.isfinite(frame_age_s) or frame_age_s > float(args.lidar_safety_max_age_s):
            return f"LiDAR safety stale during pivot ({max(0.0, frame_age_s):.2f}s old)"
        local = _scan_local(frame, args)
        if not len(local):
            return "LiDAR safety unavailable during pivot (empty scan)"
        forward_points = _to_forward_frame(local, forward_offset)
        try:
            sweep_result = _collision_box_rotation_violation(
                forward_points,
                collision_profile,
                float(remaining_yaw_deg),
                lidar_offset_forward_m=lever_m,
                physical_body_radius_m=float(args.physical_body_radius_m),
                self_mask_inset_m=float(args.collision_self_mask_inset_m),
            )
        except Exception as exc:
            # A safety subsystem fault is fail-closed operational state, not an
            # uncaught mission exception. The anchor/recovery state machine can
            # wait for another revolution or retry; it must never drive without
            # a valid footprint result.
            return f"rotational safety evaluation unavailable ({type(exc).__name__}: {exc})"
        if sweep_result is None:
            return None
        hit, blocked_at_deg = sweep_result
        mask, sector, angle, hit_range, limit = hit
        evidence_points = int(np.count_nonzero(mask))
        if np.asarray(mask).shape == (len(forward_points),):
            evidence_bins = int(
                np.unique(
                    np.floor(
                        (
                            np.degrees(
                                np.arctan2(
                                    forward_points[mask, 1],
                                    forward_points[mask, 0],
                                )
                            )
                            + 180.0
                        )
                        / float(collision_profile["bin_size_deg"])
                    ).astype(np.int64)
                ).size
            )
        else:
            # Safety results must never crash motion merely while formatting a
            # diagnostic. The collision remains a fail-closed stop even if an
            # alternate profile implementation violates the mask contract.
            evidence_bins = max(1, evidence_points)
        return (
            f"rotational collision box {sector} at sweep {blocked_at_deg:+.0f}deg: "
            f"{hit_range:.2f}m "
            f"at {angle:+.0f}deg (limit {limit:.2f}m; "
            f"{evidence_points} points/{evidence_bins} bins)"
        )

    controller.set_rotation_safety_check(_rotation_safety_check)

    def _discard_translation_prior() -> None:
        if ground_odom is not None:
            ground_odom.discard()

    def _apply_ground_translation_prior(pose: Pose2D) -> tuple[Pose2D, object | None]:
        """Propagate translation from floor flow; LiDAR will correct this seed."""
        if ground_odom is None:
            return pose, None
        measurement = ground_odom.take_delta()
        yaw_now = imu.deg()
        if measurement is None or yaw_now is None:
            return pose, measurement
        delta_imu = np.asarray(
            [measurement.forward_imu_m, measurement.left_imu_m],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(delta_imu)) or float(np.hypot(*delta_imu)) > 0.45:
            return pose, None
        # The accumulated flow vector is expressed in the IMU's continuous
        # yaw-zero frame. Align that frame to the current map physical-forward
        # heading, then advance the robot centre (not the offset LiDAR origin).
        map_minus_imu = math.radians(float(pose.theta_deg) + forward_offset - float(yaw_now))
        rotation = np.asarray(
            [
                [math.cos(map_minus_imu), -math.sin(map_minus_imu)],
                [math.sin(map_minus_imu), math.cos(map_minus_imu)],
            ],
            dtype=np.float64,
        )
        delta_map = rotation @ delta_imu
        centre = _robot_centre_from_lidar_pose(pose, lever_m, forward_offset)
        propagated = _lidar_pose_from_robot_centre(
            centre + delta_map,
            float(pose.theta_deg),
            lever_m,
            forward_offset,
        )
        return propagated, measurement

    # =======================================================================
    # EXPLORE PHASE 1 SUPPORT — INITIAL-SPIN SENSING AND COLLISION SAFETY
    #
    # These definitions support the Phase 1 execution block below and remain
    # inside main() because they share its live sensor/controller state.
    # =======================================================================
    print(
        f"\n[explore] PHASE 1 — up to {int(args.anchor_passes)} slow anchor "
        f"pass(es), {float(args.spin_degrees):.0f}deg each; a constrained "
        "start first escapes into open space, then performs the full spin ..."
    )
    pano_interval_s = 1.0 / max(0.5, float(args.panorama_hz))

    def _startup_forward_scan() -> np.ndarray | None:
        _frame_id, frame = feed.latest()
        if frame is None:
            return None
        age_s = time.time() - float(getattr(frame, "received_wall_ts", 0.0) or 0.0)
        if not math.isfinite(age_s) or age_s > float(args.lidar_safety_max_age_s):
            return None
        local = _scan_local(frame, args)
        if not len(local):
            return None
        _log_live_lidar(frame)
        points = _filter_isolated_lidar_specks(_to_forward_frame(local, forward_offset))
        return points if len(points) else None

    def _measure_startup_translation(
        before_points: np.ndarray,
        after_points: np.ndarray | None,
        expected_delta: np.ndarray,
    ) -> tuple[float, float, float]:
        """Measure a recovery primitive with scan-to-scan LiDAR odometry.

        Clearance is not odometry: sliding parallel to a chair leaves minimum
        range unchanged despite real motion. Build a one-scan temporary local
        submap and solve the fresh scan against it, without touching the mission
        map that does not exist yet.
        """
        if after_points is None or len(before_points) < 20 or len(after_points) < 20:
            return 0.0, 0.0, -math.inf
        temporary_map = WorldMap(
            grid_res_m=float(args.frontier_res_m),
            footprint_clear_m=0.0,
            lidar_offset_m=0.0,
            forward_offset_deg=0.0,
        )
        origin = Pose2D(0.0, 0.0, 0.0)
        temporary_map.add(before_points, origin, gold=True)
        delta = np.asarray(expected_delta, dtype=np.float64)
        seed = Pose2D(float(delta[0]), float(delta[1]), 0.0)
        solved, score = _localize(
            after_points,
            temporary_map,
            seed,
            args,
            max(0.18, float(np.hypot(*delta)) + 0.08),
            6.0,
        )
        measured = np.array([float(solved.x), float(solved.y)], dtype=np.float64)
        expected_norm = max(1e-6, float(np.hypot(*delta)))
        unit = delta / expected_norm
        along = float(np.dot(measured, unit))
        cross = abs(float(unit[0] * measured[1] - unit[1] * measured[0]))
        return along, cross, float(score)

    def _attempt_startup_clearance_escape() -> str:
        """Advance one local-recovery step: ``clear``, ``moved``, or ``blocked``."""
        if collision_profile is None:
            print(
                "[explore]   startup recovery unavailable: no calibrated footprint "
                "is loaded, so an escape translation cannot be proven safe."
            )
            return "blocked"
        points = _startup_forward_scan()
        if points is None:
            print("[explore]   startup recovery unavailable: no fresh LiDAR safety scan.")
            return "blocked"
        geometry = {
            "lidar_offset_forward_m": lever_m,
            "physical_body_radius_m": float(args.physical_body_radius_m),
            "self_mask_inset_m": float(args.collision_self_mask_inset_m),
        }
        current = _collision_box_violation(points, collision_profile, **geometry)
        if current is None:
            print("[explore]   startup local recovery: calibrated footprint is now clear.")
            return "clear"
        _mask, sector, angle, hit_range, limit = current
        distance = float(args.startup_escape_distance_m)
        cardinal = {
            "front": np.array([-distance, 0.0]),
            "rear": np.array([distance, 0.0]),
            "left side": np.array([0.0, -distance]),
            "right side": np.array([0.0, distance]),
        }
        primary = cardinal[sector]
        candidates = [
            primary,
            np.array([-distance, 0.0]),
            np.array([distance, 0.0]),
            np.array([0.0, -distance]),
            np.array([0.0, distance]),
        ]
        ranked: list[tuple[float, np.ndarray]] = []
        seen: set[tuple[float, float]] = set()
        for delta in candidates:
            key = (float(delta[0]), float(delta[1]))
            if key in seen:
                continue
            seen.add(key)
            safe, improvement = _translation_escape_is_safe(
                points,
                collision_profile,
                delta,
                **geometry,
            )
            if safe:
                ranked.append((float(improvement), delta))
        if not ranked:
            print(
                f"[explore]   startup obstacle at {sector} ({hit_range:.2f}m, "
                f"limit {limit:.2f}m, {angle:+.0f}deg), but no short translation "
                "monotonically exits it; refusing blind motion."
            )
            return "blocked"
        ranked.sort(key=lambda item: item[0], reverse=True)
        speed = abs(float(args.startup_escape_speed))
        baseline_depth = _collision_depth_m(current)
        for candidate_index, (expected_improvement, delta) in enumerate(ranked, start=1):
            norm = float(np.hypot(*delta))
            command_xy = _startup_escape_velocity(delta, speed)
            command_effort = float(np.max(np.abs(command_xy)))
            # Command scale estimates duration only; it is never pose odometry.
            duration_s = float(
                np.clip(
                    norm / max(0.05, float(args.drive_command_distance_scale) * command_effort),
                    0.45,
                    1.20,
                )
            )
            print(
                f"[explore]   startup local recovery candidate "
                f"{candidate_index}/{len(ranked)}: obstacle at {sector}; commanding "
                f"({delta[0]:+.2f}m forward, {delta[1]:+.2f}m left), expected "
                f"clearance gain {expected_improvement * 100:.1f}cm."
            )
            started_escape = time.monotonic()
            worsened = False
            worsening_since: float | None = None
            recovery_heading = imu.deg()
            if controller._run:
                controller.clear_safety_latch()
            try:
                while time.monotonic() - started_escape < duration_s:
                    if controller._run:
                        if recovery_heading is not None:
                            controller.translate_body(
                                float(command_xy[0]),
                                float(command_xy[1]),
                                float(recovery_heading),
                            )
                    else:
                        robot.send_action(
                            {
                                "x.vel": float(command_xy[0]),
                                "y.vel": float(command_xy[1]),
                                "theta.vel": 0.0,
                                "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
                                **hold,
                            }
                        )
                    live_points = _startup_forward_scan()
                    if live_points is not None:
                        live_hit = _collision_box_violation(live_points, collision_profile, **geometry)
                        if _collision_depth_m(live_hit) > baseline_depth + 0.02:
                            if worsening_since is None:
                                worsening_since = time.monotonic()
                            elif time.monotonic() - worsening_since >= 0.16:
                                worsened = True
                                break
                        else:
                            worsening_since = None
                    time.sleep(0.04)
            finally:
                if controller._run:
                    controller.halt()
                    time.sleep(0.12)
                else:
                    for _ in range(3):
                        robot.send_action(
                            {
                                "x.vel": 0.0,
                                "y.vel": 0.0,
                                "theta.vel": 0.0,
                                "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
                                **hold,
                            }
                        )
                        time.sleep(0.04)
            time.sleep(0.30)
            final_points = _startup_forward_scan()
            final_hit = (
                _collision_box_violation(final_points, collision_profile, **geometry)
                if final_points is not None
                else current
            )
            actual_improvement = baseline_depth - _collision_depth_m(final_hit)
            measured_along, measured_cross, motion_score = _measure_startup_translation(
                points, final_points, delta
            )
            motion_verified = (
                motion_score >= float(args.min_match_score)
                and measured_along >= 0.025
                and measured_cross <= 0.10
            )
            if worsened or actual_improvement < -0.005:
                print(
                    f"[explore]   startup candidate worsened clearance "
                    f"({actual_improvement * 100:+.1f}cm); stopping local recovery."
                )
                return "blocked"
            if motion_verified:
                print(
                    f"[explore]   startup escape verified: clearance "
                    f"{actual_improvement * 100:+.1f}cm, LiDAR odometry "
                    f"{measured_along * 100:+.1f}cm along/"
                    f"{measured_cross * 100:.1f}cm cross (match {motion_score:.1f})."
                )
                return "moved"
            print(
                f"[explore]   startup candidate produced no measured motion "
                f"(clearance {actual_improvement * 100:+.1f}cm, LiDAR odometry "
                f"{measured_along * 100:+.1f}cm along/"
                f"{measured_cross * 100:.1f}cm cross, match {motion_score:.1f}); "
                "rejecting that primitive "
                "and trying the next safe command."
            )
        print(
            "[explore]   all LiDAR-safe startup translation primitives were "
            "non-responsive; the base did not execute the requested motion."
        )
        return "blocked"

    def _startup_full_pivot_is_clear() -> bool:
        """Check whether enhanced body-spin acquisition is safe, without moving."""
        if collision_profile is None:
            return True
        points = _startup_forward_scan()
        if points is None:
            print(
                "[explore] no fresh LiDAR scan for full-pivot preflight; "
                "using stationary local-submap initialization."
            )
            return False
        geometry = {
            "lidar_offset_forward_m": lever_m,
            "physical_body_radius_m": float(args.physical_body_radius_m),
            "self_mask_inset_m": float(args.collision_self_mask_inset_m),
        }
        current_hit = _collision_box_violation(points, collision_profile, **geometry)
        positive_sweep = _collision_box_rotation_violation(
            points, collision_profile, float(args.spin_degrees), **geometry
        )
        negative_sweep = _collision_box_rotation_violation(
            points, collision_profile, -float(args.spin_degrees), **geometry
        )
        margin = _full_pivot_clearance_margin_m(points, collision_profile, **geometry)
        clear = current_hit is None and positive_sweep is None and negative_sweep is None
        if clear:
            print(
                "[explore] active initialization: full bidirectional pivot "
                f"footprint is clear (radial margin {margin * 100:+.1f}cm)."
            )
        else:
            print(
                "[explore] full-body pivot preflight is constrained "
                f"(radial margin {margin * 100:+.1f}cm); selecting a short "
                "collision-checked escape translation before mapping."
            )
        return clear

    def _ensure_startup_pivot_clearance(
        *,
        allow_reciprocal_half_turn: bool = False,
        require_translation: bool = False,
    ) -> str:
        """Relocate before mapping until a useful collision-free sweep exists.

        A 180-degree body sweep followed immediately by the same sweep in
        reverse exposes the LiDAR sector hidden by the chassis and restores the
        original travel heading.  It therefore supplies complete angular
        coverage without demanding clearance for a full 360-degree body pivot.
        """
        if collision_profile is None:
            return "clear"
        geometry = {
            "lidar_offset_forward_m": lever_m,
            "physical_body_radius_m": float(args.physical_body_radius_m),
            "self_mask_inset_m": float(args.collision_self_mask_inset_m),
        }
        maximum_steps = max(0, int(args.startup_pivot_relocation_max_steps))
        distance = max(0.05, float(args.startup_escape_distance_m))
        speed = abs(float(args.startup_escape_speed))
        directions = [
            np.array(
                [
                    math.cos(math.radians(angle_deg)) * distance,
                    math.sin(math.radians(angle_deg)) * distance,
                ]
            )
            for angle_deg in range(0, 360, 45)
        ]

        for relocation_step in range(maximum_steps + 1):
            points = _startup_forward_scan()
            if points is None:
                print(
                    "[explore] active initialization cannot verify pivot clearance: "
                    "no fresh LiDAR safety scan."
                )
                return "blocked"
            current_hit = _collision_box_violation(points, collision_profile, **geometry)
            positive_sweep = _collision_box_rotation_violation(
                points,
                collision_profile,
                float(args.spin_degrees),
                **geometry,
            )
            negative_sweep = _collision_box_rotation_violation(
                points,
                collision_profile,
                -float(args.spin_degrees),
                **geometry,
            )
            positive_half = _collision_box_rotation_violation(
                points,
                collision_profile,
                180.0,
                **geometry,
            )
            negative_half = _collision_box_rotation_violation(
                points,
                collision_profile,
                -180.0,
                **geometry,
            )
            current_margin = _full_pivot_clearance_margin_m(
                points,
                collision_profile,
                **geometry,
            )
            translation_required_now = bool(require_translation and relocation_step == 0)
            if (
                not translation_required_now
                and current_hit is None
                and positive_sweep is None
                and negative_sweep is None
            ):
                print(
                    f"[explore] active initialization: full bidirectional pivot "
                    f"footprint is clear (radial margin {current_margin * 100:+.1f}cm)."
                )
                return "clear"
            if (
                allow_reciprocal_half_turn
                and not translation_required_now
                and current_hit is None
                and (positive_half is None or negative_half is None)
            ):
                print(
                    "[explore] active initialization: a collision-free reciprocal "
                    "180deg sweep is available; it will acquire the hidden sector "
                    "and return to the original travel heading."
                )
                return "half_turn_clear"
            if relocation_step >= maximum_steps:
                print(
                    f"[explore] active initialization could not create full-pivot "
                    f"clearance after {maximum_steps} verified translation step(s) "
                    f"(radial margin {current_margin * 100:+.1f}cm)."
                )
                return "blocked"

            ranked: list[tuple[int, int, float, float, float, np.ndarray]] = []
            for delta in directions:
                if not _translation_trajectory_is_safe(
                    points,
                    collision_profile,
                    delta,
                    **geometry,
                ):
                    continue
                projected = points - delta
                projected_hit = _collision_box_violation(projected, collision_profile, **geometry)
                projected_positive = _collision_box_rotation_violation(
                    projected,
                    collision_profile,
                    float(args.spin_degrees),
                    **geometry,
                )
                projected_negative = _collision_box_rotation_violation(
                    projected,
                    collision_profile,
                    -float(args.spin_degrees),
                    **geometry,
                )
                projected_positive_half = _collision_box_rotation_violation(
                    projected,
                    collision_profile,
                    180.0,
                    **geometry,
                )
                projected_negative_half = _collision_box_rotation_violation(
                    projected,
                    collision_profile,
                    -180.0,
                    **geometry,
                )
                projected_margin = _full_pivot_clearance_margin_m(
                    projected,
                    collision_profile,
                    **geometry,
                )
                fully_clear = int(
                    projected_hit is None and projected_positive is None and projected_negative is None
                )
                half_turn_clear = int(
                    projected_hit is None
                    and (projected_positive_half is None or projected_negative_half is None)
                )
                improvement = projected_margin - current_margin
                forward_preference = float(delta[0] / max(distance, 1e-6))
                if (
                    fully_clear
                    or (allow_reciprocal_half_turn and half_turn_clear)
                    or improvement >= 0.01
                    # The behavioral contract for constrained initialization is
                    # turn/scan, move straight into verified open space, then
                    # acquire the missing half and turn back.  A safe forward
                    # step remains useful even when one 12cm rollout does not
                    # by itself make the complete reciprocal sweep clear.
                    or (require_translation and forward_preference >= 0.70)
                ):
                    # Among equally useful sweep outcomes, prefer forward body
                    # motion.  It preserves the intended travel direction and
                    # avoids an unnecessary lateral corner shuffle.
                    ranked.append(
                        (
                            fully_clear,
                            half_turn_clear,
                            forward_preference,
                            projected_margin,
                            improvement,
                            delta,
                        )
                    )
            ranked.sort(
                # A reciprocal half-turn is already sufficient coverage. Once
                # either usable sweep is available, prefer the translation
                # closest to straight forward; use full-pivot capability only
                # as the next tiebreaker.
                key=lambda item: (
                    max(item[0], item[1] if allow_reciprocal_half_turn else 0),
                    item[2],
                    item[0],
                    item[3],
                    item[4],
                ),
                reverse=True,
            )
            if not ranked:
                print(
                    "[explore] active initialization found no collision-free short "
                    "translation that increases full-pivot clearance; refusing to "
                    "spin or map while corner-constrained."
                )
                return "blocked"

            moved = False
            for candidate_index, (
                fully_clear,
                half_turn_clear,
                _forward_preference,
                projected_margin,
                _improvement,
                delta,
            ) in enumerate(ranked, start=1):
                norm = float(np.hypot(*delta))
                command_xy = _startup_escape_velocity(delta, speed)
                command_effort = float(np.max(np.abs(command_xy)))
                duration_s = float(
                    np.clip(
                        norm / max(0.05, float(args.drive_command_distance_scale) * command_effort),
                        0.45,
                        1.20,
                    )
                )
                print(
                    f"[explore] active initialization relocation "
                    f"{relocation_step + 1}/{maximum_steps}, candidate "
                    f"{candidate_index}/{len(ranked)}: commanding "
                    f"({delta[0]:+.2f}m forward, {delta[1]:+.2f}m left); "
                    f"pivot margin {current_margin * 100:+.1f} -> "
                    f"{projected_margin * 100:+.1f}cm"
                    f"{' (full pivot predicted clear)' if fully_clear else ''}"
                    f"{' (reciprocal half-turn predicted clear)' if half_turn_clear and not fully_clear else ''}."
                )
                baseline_depth = _collision_depth_m(current_hit)
                worsened = False
                worsening_since: float | None = None
                started_relocation = time.monotonic()
                try:
                    while time.monotonic() - started_relocation < duration_s:
                        robot.send_action(
                            {
                                "x.vel": float(command_xy[0]),
                                "y.vel": float(command_xy[1]),
                                "theta.vel": 0.0,
                                "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
                                **hold,
                            }
                        )
                        live_points = _startup_forward_scan()
                        if live_points is not None:
                            live_depth = _collision_depth_m(
                                _collision_box_violation(live_points, collision_profile, **geometry)
                            )
                            # One angular bin is intentionally enough for a
                            # hard stop during normal navigation, but it is not
                            # enough evidence to cancel a pre-validated corner
                            # escape: close-range returns flicker as the robot
                            # starts moving. Require a continuously worsening
                            # envelope before cancelling the primitive.
                            if live_depth > baseline_depth + 0.02:
                                if worsening_since is None:
                                    worsening_since = time.monotonic()
                                elif time.monotonic() - worsening_since >= 0.16:
                                    worsened = True
                                    break
                            else:
                                worsening_since = None
                        time.sleep(0.04)
                finally:
                    for _ in range(3):
                        robot.send_action(
                            {
                                "x.vel": 0.0,
                                "y.vel": 0.0,
                                "theta.vel": 0.0,
                                "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
                                **hold,
                            }
                        )
                        time.sleep(0.04)
                time.sleep(0.30)
                final_points = _startup_forward_scan()
                final_margin = (
                    _full_pivot_clearance_margin_m(
                        final_points,
                        collision_profile,
                        **geometry,
                    )
                    if final_points is not None
                    else -math.inf
                )
                final_current = (
                    _collision_box_violation(
                        final_points,
                        collision_profile,
                        **geometry,
                    )
                    if final_points is not None
                    else current_hit
                )
                final_positive = (
                    _collision_box_rotation_violation(
                        final_points,
                        collision_profile,
                        float(args.spin_degrees),
                        **geometry,
                    )
                    if final_points is not None
                    else current_hit
                )
                final_negative = (
                    _collision_box_rotation_violation(
                        final_points,
                        collision_profile,
                        -float(args.spin_degrees),
                        **geometry,
                    )
                    if final_points is not None
                    else current_hit
                )
                measured_along, measured_cross, motion_score = _measure_startup_translation(
                    points, final_points, delta
                )
                motion_verified = (
                    not worsened
                    and motion_score >= float(args.min_match_score)
                    and measured_along >= 0.025
                    and measured_cross <= 0.10
                )
                if (
                    not worsened
                    and final_current is None
                    and final_positive is None
                    and final_negative is None
                ):
                    print(
                        "[explore] active initialization relocation established "
                        "full-pivot clearance on the fresh LiDAR scan; restarting "
                        "the complete anchor spin from this pose."
                    )
                    return "clear"
                if motion_verified:
                    print(
                        f"[explore] active initialization relocation verified: "
                        f"LiDAR odometry {measured_along * 100:+.1f}cm along/"
                        f"{measured_cross * 100:.1f}cm cross "
                        f"(match {motion_score:.1f})."
                    )
                    moved = True
                    break
                if not worsened and final_margin >= current_margin + 0.01:
                    # Before the map exists, absolute displacement is irrelevant:
                    # the new anchor defines its own origin. A clear improvement
                    # in the fresh footprint rollout is sufficient evidence to
                    # replan the next short escape from the new observation,
                    # even when scan-to-scan odometry aliases on a repeated wall.
                    print(
                        "[explore] active initialization relocation improved "
                        f"full-pivot clearance by "
                        f"{(final_margin - current_margin) * 100:.1f}cm; "
                        "replanning the next short step from the fresh scan "
                        f"(odometry diagnostic {measured_along * 100:+.1f}cm/"
                        f"{measured_cross * 100:.1f}cm, match {motion_score:.1f})."
                    )
                    moved = True
                    break
                if worsened:
                    print(
                        "[explore] active initialization relocation stopped after "
                        "a sustained 160ms worsening collision envelope; trying "
                        "the next pre-validated escape direction."
                    )
                print(
                    f"[explore] active initialization relocation was not verified "
                    f"(along {measured_along * 100:+.1f}cm, cross "
                    f"{measured_cross * 100:.1f}cm, match {motion_score:.1f}); "
                    "trying the next safe candidate."
                )
            if not moved:
                return "blocked"
        return "blocked"

    startup_partial_captures: list[tuple[np.ndarray, np.ndarray, float]] = []
    anchor_stop_state: list[tuple[bool, float | None]] = [(False, None)]
    verified_live_anchor_scan: list[np.ndarray | None] = [None]

    # -----------------------------------------------------------------------
    # EXPLORE PHASE 1 — CAPTURE ONE INITIAL 360-DEGREE BODY-SPIN PASS
    # -----------------------------------------------------------------------
    def _phase_1_request_acknowledged_stop(label: str) -> None:
        """Stop and require proof that this exact command reached the wheels."""
        stop_action = {
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0,
            "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
            **hold,
        }
        # The acknowledgement socket can briefly stall while the LiDAR
        # publisher reconnects.  A single five-second wait used to turn that
        # transient transport gap into a false startup abort, even though the
        # watchdog had already received repeated zero commands.  Keep sending
        # the zero packet while allowing the acknowledgement channel to
        # recover; the base remains stopped throughout this boundary.
        waiter = getattr(robot, "wait_for_base_stop_ack", None)
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            for _ in range(3):
                robot.send_action(stop_action.copy())
                time.sleep(0.04)
            if callable(waiter):
                try:
                    if waiter(timeout_s=min(2.0, max(0.1, deadline - time.monotonic()))):
                        return
                except Exception as exc:
                    print(
                        f"[explore] {label}: stop acknowledgement transport "
                        f"temporarily unavailable ({type(exc).__name__}); retrying."
                    )
            else:
                # Older hosts have no acknowledgement endpoint.  The repeated
                # zero command plus watchdog is their documented stop proof.
                return
            time.sleep(0.15)
        _abort(
            f"Host did not acknowledge zero wheel velocity for {label} within "
            "20s. Map processing was not started while the base might be moving."
        )

    def _phase_1_reacquire_stationary_imu(label: str) -> float | None:
        """Hold zero velocity while waiting for a genuinely fresh yaw sample."""
        _phase_1_request_acknowledged_stop(label)
        deadline = time.monotonic() + 15.0
        next_status_at = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            yaw = imu.deg_fresh(wait_up_to_s=0.25, max_age_s=0.75)
            if yaw is not None:
                return float(yaw)
            now = time.monotonic()
            if now >= next_status_at:
                age = imu.sample_age_s()
                age_text = "no sample received" if age is None else f"last sample {age:.1f}s old"
                receiver_text = "running" if imu.receiver_running() else "stopped"
                print(
                    f"[explore] {label}: waiting stationary for fresh IMU yaw "
                    f"({age_text}; receiver {receiver_text})."
                )
                next_status_at = now + 3.0
            time.sleep(0.05)
        return None

    def _phase_1_hold_base_stopped_before_processing() -> None:
        """Refresh zero velocity until the IMU confirms the pivot has settled."""
        _phase_1_request_acknowledged_stop("anchor processing boundary")
        deadline = time.monotonic() + 4.0
        previous_yaw = imu.deg_fresh(wait_up_to_s=0.15, max_age_s=0.4)
        stop_start_yaw = previous_yaw
        stable_samples = 0
        while time.monotonic() < deadline:
            time.sleep(0.10)
            current_yaw = imu.deg_fresh(wait_up_to_s=0.08, max_age_s=0.4)
            if current_yaw is None or previous_yaw is None:
                stable_samples = 0
            elif abs(float(current_yaw) - float(previous_yaw)) <= 0.6:
                stable_samples += 1
            else:
                stable_samples = 0
            previous_yaw = current_yaw
            if stable_samples >= 4:
                anchor_stop_state[0] = (True, current_yaw)
                if stop_start_yaw is not None and current_yaw is not None:
                    residual = abs(float(current_yaw) - float(stop_start_yaw))
                    if residual > 5.0:
                        print(
                            f"[explore] WARNING: base rotated {residual:.1f}deg "
                            "after stop was first commanded; host command latency "
                            "or queued motor commands delayed the physical stop."
                        )
                return
        anchor_stop_state[0] = (False, previous_yaw)
        print(
            "[explore] WARNING: base did not produce four stationary IMU samples "
            "after the spin; zero-velocity packets were refreshed for 4.0s and "
            "the host watchdog remains the final motion fail-safe."
        )

    def _phase_1_capture_initial_spin_pass(
        direction: float,
        label: str,
        *,
        sweep_degrees: float | None = None,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], float, float, float, str]:
        """Capture one continuous spin in the original yaw coordinate system."""
        requested_sweep = (
            abs(float(args.spin_degrees)) if sweep_degrees is None else max(5.0, abs(float(sweep_degrees)))
        )
        start_yaw = _phase_1_reacquire_stationary_imu(f"anchor {label} sensor reacquisition")
        if start_yaw is None:
            _abort(
                f"IMU stream did not deliver a fresh yaw during the 15s "
                f"stationary reacquisition before anchor {label}."
            )
        scans: list[tuple[np.ndarray, np.ndarray, float]] = []
        scan_yaws: list[float] = []
        deskew_timed = 0
        previous_id = feed.latest()[0]
        previous_scan_yaw: float | None = None
        last_pano = 0.0
        started = time.monotonic()
        requested_speed = abs(float(args.spin_speed))
        maximum_speed = max(requested_speed, abs(float(args.anchor_spin_max_speed)))
        command_speed = requested_speed
        command = math.copysign(command_speed, float(direction))
        motion_check_at = started
        motion_check_yaw = float(start_yaw)
        safety_check_at = 0.0
        safety_blocked_since: float | None = None
        safety_blocked_reason: str | None = None
        safety_blocked_frames = 0
        safety_last_frame_id: int | None = None
        last_command_at = 0.0
        stop_reason = "timeout"
        try:
            while True:
                now = time.monotonic()
                yaw_now = imu.deg()
                if now - safety_check_at >= 0.08:
                    safety_check_at = now
                    safety_frame_id = int(feed.latest()[0])
                    swept_deg = abs(float(yaw_now) - float(start_yaw)) if yaw_now is not None else 0.0
                    remaining_deg = max(0.0, requested_sweep - swept_deg)
                    safety_reason = _rotation_safety_check(math.copysign(remaining_deg, command))
                    if safety_reason is not None:
                        # The collision envelope is evaluated from successive
                        # mechanical revolutions. One near-field speck must not
                        # turn a clear preflight into an incomplete anchor, but
                        # a persistent return still stops the slow pivot well
                        # before contact.
                        if safety_blocked_since is None:
                            safety_blocked_since = now
                            safety_blocked_reason = safety_reason
                        if safety_last_frame_id != safety_frame_id:
                            safety_last_frame_id = safety_frame_id
                            safety_blocked_frames += 1
                        if safety_blocked_frames >= 3 and now - safety_blocked_since >= 0.12:
                            print(
                                f"[explore]   {label}: anchor sweep stopped by "
                                "persistent full-footprint safety: "
                                f"{safety_blocked_reason or safety_reason}."
                            )
                            stop_reason = "safety_blocked"
                            break
                    else:
                        safety_blocked_since = None
                        safety_blocked_reason = None
                        safety_blocked_frames = 0
                        safety_last_frame_id = None
                if now - last_command_at >= 0.07:
                    robot.send_action({**_spin_command(robot, command), **hold})
                    last_command_at = now
                if now - last_pano >= pano_interval_s:
                    last_pano = now
                    _log_panorama()
                frame_id, frame = feed.latest()
                if frame is not None:
                    _log_live_lidar(frame)
                if frame is not None and int(frame_id) != int(previous_id):
                    previous_id = int(frame_id)
                    yaw_end = _imu_yaw_for_scan(imu, frame)
                    if yaw_end is not None:
                        yaw_start_scan = None
                        timed_this_scan = False
                        start_ts = getattr(frame, "revolution_started_ts", None)
                        if start_ts is not None:
                            yaw_start_scan = imu.deg_at_wall_time(float(start_ts))
                        if yaw_start_scan is not None:
                            heading_change = float(yaw_end) - float(yaw_start_scan)
                            timed_this_scan = True
                        elif previous_scan_yaw is not None:
                            heading_change = float(yaw_end) - float(previous_scan_yaw)
                        else:
                            heading_change = 0.0
                        raw_xy = _scan_local(frame, args)
                        deskewed_xy = _scan_local(
                            frame,
                            args,
                            heading_change_deg=heading_change,
                            reverse_sweep=bool(args.anchor_deskew_reverse),
                        )
                        previous_scan_yaw = float(yaw_end)
                        if len(raw_xy) >= 12 and len(deskewed_xy) >= 12:
                            scans.append(
                                (
                                    raw_xy,
                                    deskewed_xy,
                                    float(yaw_end) - float(yaw0),
                                )
                            )
                            scan_yaws.append(float(yaw_end))
                            deskew_timed += int(timed_this_scan)
                if yaw_now is not None and abs(float(yaw_now) - float(start_yaw)) >= requested_sweep:
                    stop_reason = "complete"
                    break
                if yaw_now is not None and now - motion_check_at >= max(
                    0.5, float(args.anchor_stall_seconds)
                ):
                    window_motion = abs(float(yaw_now) - float(motion_check_yaw))
                    if window_motion < 5.0:
                        faster = _next_anchor_spin_speed(command_speed, maximum_speed)
                        if faster > command_speed + 1e-6:
                            print(
                                f"[explore]   {label}: only {window_motion:.1f}deg IMU motion "
                                f"in {now - motion_check_at:.1f}s; raising theta.vel "
                                f"{command_speed:.2f} -> {faster:.2f} to clear drivetrain stiction."
                            )
                            command_speed = faster
                            command = math.copysign(command_speed, float(direction))
                            # Idle LiDAR revolutions are not anchor keyframes.
                            # Discard them when reacquiring physical motion.
                            scans.clear()
                            scan_yaws.clear()
                            deskew_timed = 0
                            previous_scan_yaw = None
                        else:
                            print(
                                f"[explore]   {label}: anchor motion watchdog found only "
                                f"{window_motion:.1f}deg in {now - motion_check_at:.1f}s at "
                                f"maximum theta.vel {command_speed:.2f}."
                            )
                            stop_reason = "motion_stall"
                            break
                    motion_check_at = now
                    motion_check_yaw = float(yaw_now)
                if time.monotonic() - started > float(args.max_spin_seconds):
                    break
                time.sleep(0.03)
        finally:
            _phase_1_hold_base_stopped_before_processing()
        end_yaw = imu.deg_fresh(wait_up_to_s=0.8, max_age_s=0.4)
        net = (float(end_yaw) - float(start_yaw)) if end_yaw is not None else 0.0
        yaw_steps = np.diff(np.asarray(scan_yaws, dtype=np.float64))
        directional_steps = yaw_steps[np.abs(yaw_steps) >= 0.05]
        monotonic_ratio = (
            float(np.mean(np.sign(directional_steps) == np.sign(net)))
            if len(directional_steps) and abs(net) > 1.0
            else 0.0
        )
        timed_ratio = float(deskew_timed) / max(1, len(scans))
        print(
            f"[explore]   {label}: captured {len(scans)} scans, IMU sweep {net:+.1f}deg, "
            f"direction consistency {monotonic_ratio:.1%}, timed deskew {timed_ratio:.1%}."
        )
        if abs(net) < 0.95 * requested_sweep:
            if _anchor_interruption_uses_stationary_fallback(stop_reason):
                print(
                    f"[explore]   WARNING: {label} completed only {net:+.1f}deg; "
                    "its scans may be used only in the disposable startup "
                    "costmap that plans a route to full-pivot clearance."
                )
            else:
                print(
                    f"[explore]   WARNING: {label} completed only {net:+.1f}deg; "
                    "the motion watchdog will discard and reacquire this capture."
                )
        if monotonic_ratio < 0.95:
            print(
                f"[explore]   WARNING: {label} IMU direction consistency was "
                f"{monotonic_ratio:.1%}; incomplete motion will not initialize SLAM."
            )
        if str(args.anchor_deskew) == "on" and timed_ratio < 0.80:
            print(
                f"[explore]   WARNING: {label} timestamped deskew coverage was only "
                f"{timed_ratio:.1%}; the pass remains eligible."
            )
        if stop_reason == "motion_stall":
            print(f"[explore]   WARNING: {label} was physically stalled; its scans are invalid.")
        time.sleep(0.40)
        return scans, net, timed_ratio, monotonic_ratio, stop_reason

    anchor_turn_observation: list[tuple[float, float] | None] = [None]

    def _phase_1_capture_verified_initial_spin_pass(
        direction: float,
        label: str,
        *,
        required: bool,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], float, float, float] | None:
        """Acquire a real sweep, retrying motion rather than scoring duplicates."""
        motion_budget = max(1, int(args.anchor_motion_retries) + 1)
        motion_failures = 0
        safety_interruptions = 0
        capture_index = 0
        while motion_failures < motion_budget and safety_interruptions < 2:
            capture_index += 1
            attempt_direction = float(direction) * (1.0 if capture_index % 2 else -1.0)
            scans, net, timed_ratio, monotonic_ratio, stop_reason = _phase_1_capture_initial_spin_pass(
                attempt_direction,
                label,
            )
            if abs(net) >= 10.0 and monotonic_ratio >= 0.80:
                # Even an interrupted collision-safe sweep is valid actuator
                # calibration: retain the observed command/IMU sign for the
                # stationary-submap fallback without mapping partial geometry.
                anchor_turn_observation[0] = (float(net), float(attempt_direction))
            if _anchor_motion_is_complete(net, monotonic_ratio, float(args.spin_degrees)):
                return scans, net, timed_ratio, attempt_direction
            if scans:
                # Retain interrupted-spin geometry only for the temporary
                # startup costmap. It is never eligible for the permanent SLAM
                # anchor, but it is exactly the evidence needed to plan a stable
                # escape instead of choosing a new reactive gap every frame.
                startup_partial_captures.extend(scans)
                if len(startup_partial_captures) > 360:
                    del startup_partial_captures[:-360]

            if stop_reason == "safety_blocked":
                safety_interruptions += 1
                if safety_interruptions < 2:
                    print(
                        f"[explore]   {label}: one pivot direction was persistently "
                        "blocked; discarding its partial scans and immediately "
                        "trying the opposite direction."
                    )
                    continue
                print(
                    f"[explore]   {label}: both full-pivot directions were "
                    "persistently constrained. Their scans remain isolated from "
                    "SLAM and will now form one temporary startup costmap."
                )
                return None

            motion_failures += int(_anchor_failure_consumes_motion_budget(stop_reason))
            if motion_failures < motion_budget:
                escape_state = _ensure_startup_pivot_clearance()
                print(
                    f"[explore]   {label}: incomplete physical sweep "
                    f"({net:+.1f}deg, {monotonic_ratio:.1%} direction consistency); "
                    f"discarding it and reacquiring in the opposite turn direction "
                    f"({motion_failures + 1}/{motion_budget}; full-pivot recovery "
                    f"{escape_state})."
                )
                time.sleep(0.5)
        message = (
            f"Anchor {label} could not produce a complete physical rotation after "
            f"{motion_failures} drivetrain failure(s)."
        )
        if required:
            _abort(message)
        print(f"[explore] WARNING: {message}")
        return None

    # =======================================================================
    # EXPLORE PHASE 1.1 — INITIAL SPIN BLOCKED: PARTIAL SPIN AND ESCAPE
    #
    # ExploreSystem.md requires this single path:
    #   partial sweep -> temporary map -> route to open space -> move there ->
    #   discard temporary map -> complete replacement 360-degree spin.
    # =======================================================================
    def _phase_1_1_turn_toward_partial_map_route(
        requested_body_turn_deg: float,
        label: str,
    ) -> bool:
        """Slowly face a visible gap while allowing an existing margin overlap.

        A calibrated safety envelope is intentionally larger than the physical
        chassis. If startup is already inside that *margin*, rejecting both
        turn directions forever is not useful. This controller permits the
        chosen turn only while live LiDAR proves collision penetration is not
        worsening; a sustained increase still stops immediately.
        """
        requested = float(np.clip(requested_body_turn_deg, -90.0, 90.0))
        if abs(requested) < 8.0:
            return True
        start_yaw = imu.deg_fresh(wait_up_to_s=0.8, max_age_s=0.4)
        before = _startup_forward_scan()
        if start_yaw is None or before is None:
            return False
        geometry = {
            "lidar_offset_forward_m": lever_m,
            "physical_body_radius_m": float(args.physical_body_radius_m),
            "self_mask_inset_m": float(args.collision_self_mask_inset_m),
        }
        baseline_depth = _collision_depth_m(_collision_box_violation(before, collision_profile, **geometry))
        worsening_frames = [0]

        def _non_worsening_turn_guard(_remaining_deg: float) -> str | None:
            live = _startup_forward_scan()
            if live is None:
                return "live LiDAR unavailable during startup gap turn"
            depth = _collision_depth_m(_collision_box_violation(live, collision_profile, **geometry))
            if depth > baseline_depth + 0.035:
                worsening_frames[0] += 1
            else:
                worsening_frames[0] = 0
            if worsening_frames[0] >= 3:
                return (
                    f"startup gap turn worsened collision penetration {baseline_depth:.2f}m -> {depth:.2f}m"
                )
            return None

        target_yaw = float(start_yaw) + requested
        original_turn_sign = float(controller.turn_sign)
        controller.clear_safety_latch()
        controller.set_rotation_safety_check(_non_worsening_turn_guard)
        controller.start()
        controller.rotate_to(target_yaw)
        reversed_once = False
        turn_started = time.monotonic()
        deadline = turn_started + 14.0
        previous_yaw = float(start_yaw)
        angular_travel = 0.0
        angular_travel_limit = abs(requested) + 18.0
        success = False
        try:
            while time.monotonic() < deadline:
                current_yaw = imu.deg_fresh(wait_up_to_s=0.15, max_age_s=0.4)
                if current_yaw is None:
                    continue
                angular_travel += abs(float(current_yaw) - previous_yaw)
                previous_yaw = float(current_yaw)
                if angular_travel > angular_travel_limit:
                    print(
                        f"[explore] {label}: stopping bounded gap turn after "
                        f"{angular_travel:.1f}deg of IMU travel; the request was "
                        f"only {requested:+.1f}deg."
                    )
                    break
                progress = float(current_yaw) - float(start_yaw)
                turn_status = _bounded_turn_status(requested, progress)
                if turn_status == "wrong_direction" and not reversed_once:
                    controller.turn_sign *= -1.0
                    controller.clear_safety_latch()
                    controller.rotate_to(target_yaw)
                    reversed_once = True
                    print(
                        f"[explore] {label}: measured startup turn sign was "
                        "opposite the requested visible gap; corrected it from "
                        "live IMU feedback."
                    )
                    continue
                if turn_status == "wrong_direction":
                    print(
                        f"[explore] {label}: corrected command still moved away "
                        "from the committed heading; stopping this route segment."
                    )
                    break
                if turn_status == "overshoot":
                    print(
                        f"[explore] {label}: bounded gap turn crossed its "
                        f"{requested:+.1f}deg target; stopping immediately."
                    )
                    break
                if turn_status == "complete":
                    success = True
                    break
                if controller.safety_latched_reason() is not None:
                    break
                if time.monotonic() - turn_started >= 3.0 and abs(progress) < 3.0:
                    break
                time.sleep(0.04)
        finally:
            controller.halt()
            time.sleep(0.15)
            controller.shutdown()
            controller.set_rotation_safety_check(_rotation_safety_check)
            if not success and not reversed_once:
                controller.turn_sign = original_turn_sign
        final_yaw = imu.deg()
        turned = 0.0 if final_yaw is None else float(final_yaw) - float(start_yaw)
        if success:
            print(
                f"[explore] {label}: faced visible open space with a controlled "
                f"{turned:+.1f}deg turn; selecting forward travel from a fresh scan."
            )
            return True
        failure_reason = controller.safety_latched_reason()
        if failure_reason is None and not controller.legacy_host_untorque_sentinel_enabled():
            controller.enable_legacy_host_untorque_sentinel()
            print(
                f"[explore] {label}: threaded base turn ended at {turned:+.1f}deg "
                "without a collision stop. Enabling the legacy Pi-host base-only "
                "sentinel and retrying the SAME committed turn now."
            )
            return _phase_1_1_turn_toward_partial_map_route(requested, label)
        if failure_reason is None:
            # Eliminate the remaining transport variable. This is the exact
            # main-thread packet shape used by the historically working anchor
            # spin. Use it for wrong-direction responses too, not only a total
            # lack of motion, while retaining the bounded IMU/collision guards.
            direct_start = imu.deg_fresh(wait_up_to_s=0.8, max_age_s=0.4)
            if direct_start is not None:
                direct_target = float(direct_start) + requested
                command_sign = float(controller.turn_sign)
                direct_started = time.monotonic()
                direct_success = False
                direct_previous_yaw = float(direct_start)
                direct_angular_travel = 0.0
                direct_angular_limit = abs(requested) + 18.0
                direct_reversed_once = False
                print(
                    f"[explore] {label}: legacy controller stream did not complete "
                    "the signed turn; retrying that same turn through the proven "
                    "main-thread anchor command path."
                )
                try:
                    while time.monotonic() - direct_started < 14.0:
                        direct_yaw = imu.deg_fresh(
                            wait_up_to_s=0.15,
                            max_age_s=0.4,
                        )
                        if direct_yaw is None:
                            continue
                        direct_angular_travel += abs(float(direct_yaw) - direct_previous_yaw)
                        direct_previous_yaw = float(direct_yaw)
                        if direct_angular_travel > direct_angular_limit:
                            print(
                                f"[explore] {label}: stopping proven-packet gap "
                                f"turn after {direct_angular_travel:.1f}deg of IMU "
                                f"travel; the request was only {requested:+.1f}deg."
                            )
                            break
                        error = direct_target - float(direct_yaw)
                        progress = float(direct_yaw) - float(direct_start)
                        turn_status = _bounded_turn_status(requested, progress)
                        if turn_status == "complete":
                            direct_success = True
                            break
                        if turn_status == "overshoot":
                            print(
                                f"[explore] {label}: proven-packet turn crossed "
                                f"its {requested:+.1f}deg target; stopping now."
                            )
                            break
                        if turn_status == "wrong_direction":
                            if direct_reversed_once:
                                print(
                                    f"[explore] {label}: proven-packet command "
                                    "still moved away after its one sign "
                                    "correction; stopping."
                                )
                                break
                            command_sign *= -1.0
                            controller.turn_sign *= -1.0
                            direct_start = float(direct_yaw)
                            direct_target = float(direct_yaw) + requested
                            direct_previous_yaw = float(direct_yaw)
                            direct_reversed_once = True
                            continue
                        guard_reason = _non_worsening_turn_guard(error)
                        if guard_reason is not None:
                            print(
                                f"[explore] {label}: proven-packet gap turn "
                                f"stopped by live guard: {guard_reason}."
                            )
                            break
                        theta_command = command_sign * math.copysign(
                            max(0.80, abs(float(args.spin_speed))),
                            error,
                        )
                        robot.send_action(
                            {
                                **_spin_command(robot, theta_command),
                                **hold,
                            }
                        )
                        if time.monotonic() - direct_started >= 3.0 and abs(progress) < 3.0:
                            break
                        time.sleep(0.04)
                finally:
                    for _ in range(3):
                        _send_stop(robot)
                        time.sleep(0.04)
                if direct_success:
                    final_direct_yaw = imu.deg()
                    direct_turn = (
                        requested
                        if final_direct_yaw is None
                        else float(final_direct_yaw) - float(direct_start)
                    )
                    print(
                        f"[explore] {label}: proven anchor command path faced "
                        f"the visible opening ({direct_turn:+.1f}deg)."
                    )
                    return True
        print(
            f"[explore] {label}: controlled gap turn stopped after {turned:+.1f}deg"
            f" ({controller.safety_latched_reason() or 'no physical turn response'})."
        )
        return False

    def _phase_1_1_build_partial_map_and_escape_route(
        label: str,
    ) -> tuple[list[np.ndarray], float] | None:
        """Phase 1.1 step A: build the temporary map and freeze one route."""
        if not startup_partial_captures:
            return None
        reference_heading = float(startup_partial_captures[0][2])
        temporary_map = WorldMap(
            grid_res_m=float(args.frontier_res_m),
            footprint_clear_m=float(args.self_clear_m),
            lidar_offset_m=lever_m,
            forward_offset_deg=forward_offset,
        )
        stride = max(1, len(startup_partial_captures) // 120)
        for raw_xy, _deskewed_xy, heading in startup_partial_captures[::stride]:
            raw_xy = _filter_isolated_lidar_specks(raw_xy)
            if len(raw_xy) < 12:
                continue
            relative_heading = float(heading) - reference_heading
            pose = _lidar_pose_from_robot_centre(
                np.zeros(2, dtype=np.float64),
                relative_heading,
                lever_m,
                forward_offset,
            )
            temporary_map.add(raw_xy, pose, gold=False)
        analysis = analyze_grid(
            temporary_map.grid,
            # Startup escape is a one-shot local maneuver from a constrained
            # pose. Inflate by the measured body plus one grid cell, not the
            # larger mission-navigation comfort radius; the latter erased the
            # only reachable corridor in field runs that still had a clear,
            # significant passable frontier.
            robot_radius_m=(float(args.physical_body_radius_m) + float(args.frontier_res_m)),
            min_frontier_span_m=float(args.min_frontier_span_m),
            min_frontier_cells=int(args.min_frontier_cells),
            passable_opening_min_m=float(args.passable_opening_min_m),
        )
        patrol = _pick_patrol_route(
            analysis,
            np.zeros(2, dtype=np.float64),
            [],
            min_translation_m=0.18,
        )
        route_kind = "A*"
        if patrol is None:
            # A very short collision-bounded sweep may not contain enough free
            # cells for inflated A*. The sweep still directly observed which
            # side is more open. Freeze that one observation into one straight
            # route; do not request another sweep or run a reactive selector.
            last_raw, _last_deskewed, last_heading = startup_partial_captures[-1]
            last_forward = _filter_isolated_lidar_specks(_to_forward_frame(last_raw, forward_offset))
            self_returns = _physical_body_self_return_mask(
                last_forward,
                lidar_offset_forward_m=lever_m,
                physical_body_radius_m=float(args.physical_body_radius_m),
                self_mask_inset_m=float(args.collision_self_mask_inset_m),
            )
            opening = _widest_visible_corridor_bearing_deg(
                last_forward[~self_returns],
                lidar_offset_forward_m=lever_m,
                corridor_half_width_m=float(args.physical_body_radius_m) + 0.03,
            )
            if opening is None:
                return None
            bearing_deg, visible_range_m = opening
            travel_m = float(
                np.clip(
                    visible_range_m - float(args.physical_body_radius_m) - 0.12,
                    0.18,
                    0.60,
                )
            )
            map_heading_deg = (
                float(last_heading) - reference_heading + float(forward_offset) + float(bearing_deg)
            )
            goal = travel_m * np.array(
                [
                    math.cos(math.radians(map_heading_deg)),
                    math.sin(math.radians(map_heading_deg)),
                ]
            )
            waypoints = [goal]
            route_kind = "frozen more-open-side"
            print(
                f"[explore] {label}: inflated A* start was constrained; the "
                f"same frozen sweep shows its more-open side at {bearing_deg:+.0f}deg "
                f"for {visible_range_m:.2f}m. Committing one {travel_m:.2f}m "
                "forward route toward that observation."
            )
        else:
            goal, waypoints = patrol
        if not waypoints:
            return None
        planned_waypoint_count = len(waypoints)
        waypoints = _single_startup_escape_waypoint(
            waypoints,
            maximum_distance_m=0.45,
        )
        if not waypoints:
            return None
        if planned_waypoint_count > 1:
            print(
                f"[explore] {label}: temporary A* route contained "
                f"{planned_waypoint_count} waypoints; Phase 1.1 keeps only one "
                "bounded forward escape segment before the replacement 360deg spin."
            )
        goal = np.asarray(waypoints[0], dtype=np.float64)
        world_sets = [scan.world_xy for scan in temporary_map.scans if len(scan.world_xy)]
        world_points = (
            np.concatenate(world_sets, axis=0) if world_sets else np.zeros((0, 2), dtype=np.float32)
        )
        rr.log(
            "startup_partial_map/returns",
            rr.Points3D(
                np.column_stack(
                    (
                        world_points[:, 0],
                        world_points[:, 1],
                        np.zeros(len(world_points)),
                    )
                ).astype(np.float32),
                colors=[[170, 140, 255]],
                radii=0.012,
            ),
        )
        plan_line = np.column_stack(
            (
                np.asarray([np.zeros(2), *waypoints])[:, 0],
                np.asarray([np.zeros(2), *waypoints])[:, 1],
                np.zeros(len(waypoints) + 1),
            )
        ).astype(np.float32)
        rr.log(
            "startup_partial_map/plan",
            rr.LineStrips3D([plan_line], colors=[[40, 255, 120]], radii=0.016),
        )
        print(
            f"[explore] {label}: temporary partial map contains "
            f"{len(temporary_map.scans)} IMU-aligned scans; committed one "
            f"{route_kind} "
            f"route with {len(waypoints)} waypoint(s) to open-space goal "
            f"({goal[0]:+.2f}, {goal[1]:+.2f})."
        )
        return [np.asarray(point, dtype=np.float64) for point in waypoints], (float(yaw0) + reference_heading)

    def _phase_1_1_follow_partial_map_escape_route(
        waypoints: list[np.ndarray],
        reference_yaw_deg: float,
        label: str,
    ) -> bool:
        """Phase 1.1 step B: execute the frozen route as turn, then forward."""
        # Defensive enforcement of the Phase 1.1 contract: even if a future
        # planner accidentally returns a general multi-waypoint route, startup
        # may issue at most one forward translation before the replacement spin.
        waypoints = list(waypoints[:1])
        estimated_xy = np.zeros(2, dtype=np.float64)
        for waypoint_index, waypoint in enumerate(waypoints, start=1):
            delta = np.asarray(waypoint, dtype=np.float64) - estimated_xy
            distance = float(np.hypot(*delta))
            if distance < 0.05:
                estimated_xy = np.asarray(waypoint, dtype=np.float64)
                continue
            desired_world_heading = math.degrees(math.atan2(delta[1], delta[0]))
            # PHASE 1.1 STEP B1 — FACE THIS FROZEN WAYPOINT.
            # Commit to this heading. The 25deg commands are control
            # subdivisions, not route attempts; a target behind the robot may
            # legitimately require eight of them. Progress watchdogs catch an
            # unresponsive actuator without inventing a 75deg reach limit.
            heading_started = time.monotonic()
            heading_last_progress_at = heading_started
            best_abs_error = float("inf")
            while True:
                current_yaw = imu.deg_fresh(wait_up_to_s=0.6, max_age_s=0.75)
                if current_yaw is None:
                    current_yaw = _phase_1_reacquire_stationary_imu(f"{label} waypoint heading")
                    if current_yaw is None:
                        print(
                            f"[explore] {label}: IMU stream unavailable for 15s "
                            "while stationary; the committed route cannot be "
                            "executed safely."
                        )
                        return False
                current_world_heading = float(current_yaw) - float(reference_yaw_deg) + float(forward_offset)
                turn_error = ((desired_world_heading - current_world_heading + 180.0) % 360.0) - 180.0
                # Keep this equal to the small-turn no-op threshold in
                # _phase_1_1_turn_toward_partial_map_route(). The previous
                # 7deg/8deg mismatch repeated a +7.5deg request forever.
                if abs(turn_error) < 8.0:
                    break
                now = time.monotonic()
                abs_error = abs(float(turn_error))
                if abs_error <= best_abs_error - 2.0:
                    best_abs_error = abs_error
                    heading_last_progress_at = now
                if now - heading_last_progress_at > 10.0:
                    print(
                        f"[explore] {label}: committed heading made no measurable "
                        f"IMU progress for 10s (remaining {turn_error:+.1f}deg); "
                        "stopping because the actuator/transport is unresponsive, "
                        "not because the mapped route is blocked."
                    )
                    return False
                if now - heading_started > 45.0:
                    print(
                        f"[explore] {label}: committed heading did not finish "
                        f"within 45s (remaining {turn_error:+.1f}deg); stopping "
                        "the actuator without declaring an obstacle."
                    )
                    return False
                turn_step = _bounded_heading_step(turn_error, 25.0)
                print(
                    f"[explore] {label}: partial-map waypoint {waypoint_index} "
                    f"requires {turn_error:+.1f}deg; turning {turn_step:+.1f}deg "
                    "and rescanning."
                )
                if not _phase_1_1_turn_toward_partial_map_route(turn_step, label):
                    if controller.safety_latched_reason() is None:
                        print(
                            f"[explore] {label}: actuator sign/transport did not "
                            "complete the turn; retaining the same waypoint and "
                            "retrying its remaining IMU heading instead of aborting."
                        )
                        time.sleep(0.20)
                        continue
                    return False

            before = _startup_forward_scan()
            heading_hold = _phase_1_reacquire_stationary_imu(f"{label} before escape translation")
            if before is None or heading_hold is None:
                return False
            body_delta = np.array([distance, 0.0], dtype=np.float64)
            geometry = {
                "lidar_offset_forward_m": lever_m,
                "physical_body_radius_m": float(args.physical_body_radius_m),
                "self_mask_inset_m": float(args.collision_self_mask_inset_m),
            }
            # PHASE 1.1 STEP B2 — VERIFY, THEN MOVE STRAIGHT FORWARD.
            if not _translation_trajectory_is_safe(
                before,
                collision_profile,
                body_delta,
                **geometry,
            ):
                print(
                    f"[explore] {label}: temporary-map segment "
                    f"{waypoint_index}/{len(waypoints)} is no longer clear in "
                    "the fresh forward scan; requesting a scan-turn replan."
                )
                return False
            baseline_depth = _collision_depth_m(
                _collision_box_violation(before, collision_profile, **geometry)
            )
            speed = max(0.80, min(1.0, abs(float(args.startup_escape_speed))))
            duration_s = float(
                np.clip(
                    distance / max(0.05, float(args.drive_command_distance_scale) * speed),
                    0.35,
                    3.0,
                )
            )
            controller.clear_safety_latch()
            controller.start()
            controller.translate_body(
                speed,
                0.0,
                float(heading_hold),
            )
            started = time.monotonic()
            worsening_since: float | None = None
            blocked = False
            try:
                while time.monotonic() - started < duration_s:
                    live = _startup_forward_scan()
                    if live is not None:
                        depth = _collision_depth_m(
                            _collision_box_violation(live, collision_profile, **geometry)
                        )
                        if depth > baseline_depth + 0.025:
                            if worsening_since is None:
                                worsening_since = time.monotonic()
                            elif time.monotonic() - worsening_since >= 0.16:
                                blocked = True
                                break
                        else:
                            worsening_since = None
                    time.sleep(0.04)
            finally:
                controller.halt()
                time.sleep(0.12)
                controller.shutdown()
            if blocked:
                print(
                    f"[explore] {label}: committed partial-map route stopped at "
                    f"waypoint {waypoint_index}; live collision depth worsened."
                )
                return False
            estimated_xy = np.asarray(waypoint, dtype=np.float64)
            print(
                f"[explore] {label}: completed committed partial-map waypoint "
                f"{waypoint_index}/{len(waypoints)} ({distance:.2f}m forward)."
            )
        return True

    # Legacy startup escape implementation retained for behavior-preserving
    # archaeology. It currently has no caller. Phase 1.1 uses the frozen-map
    # route functions above and the coordinator below.
    def _legacy_unused_escape_startup_to_open_area(label: str) -> str:
        """Turn-scan toward one observed opening, then drive forward into it.

        Off-axis strafe is not reliable on every deployed base host. Commit to
        one visible opening in IMU coordinates, acquire it with bounded turns
        and fresh scans, then issue only a forward translation. A half-view is
        never allowed to declare full-pivot clearance until forward displacement
        has actually been verified.
        """
        if collision_profile is None:
            return "clear"
        geometry = {
            "lidar_offset_forward_m": lever_m,
            "physical_body_radius_m": float(args.physical_body_radius_m),
            "self_mask_inset_m": float(args.collision_self_mask_inset_m),
        }
        base_distance = max(0.12, float(args.startup_escape_distance_m))
        distances = sorted(
            {
                min(0.65, max(0.45, 3.0 * base_distance)),
                min(0.50, max(0.30, 2.0 * base_distance)),
                min(0.35, max(0.18, base_distance)),
            },
            reverse=True,
        )
        # Use only the forward-visible hemisphere. The chassis contaminates or
        # occludes much of the rear sector; treating missing rear returns as
        # free space caused blind backwards escapes. Sideways motion remains
        # available at +/-90 degrees on the mecanum base.
        heading_candidates = [float(angle) for angle in range(-90, 91, 15)]
        decision_index = 0
        verified_forward_moves = 0
        committed_gap_yaw: float | None = None
        search_turn_direction = 1.0

        def _full_pivot_clear(points: np.ndarray) -> bool:
            return (
                _collision_box_violation(points, collision_profile, **geometry) is None
                and _collision_box_rotation_violation(
                    points, collision_profile, float(args.spin_degrees), **geometry
                )
                is None
                and _collision_box_rotation_violation(
                    points, collision_profile, -float(args.spin_degrees), **geometry
                )
                is None
            )

        while True:
            decision_index += 1
            points = _startup_forward_scan()
            if points is None:
                return "lidar_unavailable"
            if _full_pivot_clear(points) and verified_forward_moves > 0:
                print(
                    f"[explore] {label}: reached open space after "
                    f"{verified_forward_moves} verified forward move(s); full "
                    "360deg pivot is clear in the fresh scan."
                )
                return "clear"

            if committed_gap_yaw is not None:
                current_yaw = imu.deg_fresh(wait_up_to_s=0.6, max_age_s=0.4)
                if current_yaw is None:
                    return "lidar_unavailable"
                error = ((float(committed_gap_yaw) - float(current_yaw) + 180.0) % 360.0) - 180.0
                if abs(error) > 8.0:
                    turn_step = _bounded_heading_step(error, 25.0)
                    print(
                        f"[explore] {label}: committed opening has {error:+.1f}deg "
                        f"remaining; turning {turn_step:+.1f}deg, then rescanning."
                    )
                    if not _phase_1_1_turn_toward_partial_map_route(turn_step, label):
                        committed_gap_yaw = None
                        search_turn_direction *= -1.0
                    continue
                committed_gap_yaw = None

            candidates: list[tuple[int, float, float, float, float, np.ndarray]] = []
            initial_hit = _collision_box_violation(points, collision_profile, **geometry)
            for travel_deg in heading_candidates:
                travel_rad = math.radians(travel_deg)
                for distance in distances:
                    body_delta = distance * np.array(
                        [math.cos(travel_rad), math.sin(travel_rad)],
                        dtype=np.float64,
                    )
                    if initial_hit is None:
                        translation_safe = _translation_trajectory_is_safe(
                            points,
                            collision_profile,
                            body_delta,
                            **geometry,
                        )
                        improvement = 0.0
                    else:
                        translation_safe, improvement = _translation_escape_is_safe(
                            points,
                            collision_profile,
                            body_delta,
                            **geometry,
                        )
                    if not translation_safe:
                        continue
                    projected = points - body_delta
                    final_clear = int(_full_pivot_clear(projected))
                    final_margin = _full_pivot_clearance_margin_m(projected, collision_profile, **geometry)
                    candidates.append(
                        (
                            final_clear,
                            float(final_margin),
                            float(improvement),
                            -abs(float(travel_deg)),
                            float(distance),
                            body_delta,
                        )
                    )
            if not candidates:
                self_returns = _physical_body_self_return_mask(
                    points,
                    lidar_offset_forward_m=lever_m,
                    physical_body_radius_m=float(args.physical_body_radius_m),
                    self_mask_inset_m=float(args.collision_self_mask_inset_m),
                )
                visible_gap = _widest_visible_corridor_bearing_deg(
                    points[~self_returns],
                    lidar_offset_forward_m=lever_m,
                    corridor_half_width_m=(float(args.physical_body_radius_m) + 0.06),
                )
                if visible_gap is not None:
                    gap_bearing, gap_range = visible_gap
                    print(
                        f"[explore] {label}: calibrated envelope rejects every "
                        "translation, but raw LiDAR shows the widest visible "
                        f"corridor at {gap_bearing:+.0f}deg for {gap_range:.2f}m. "
                        "Committing to that corridor for bounded turn-scan steps."
                    )
                    if abs(gap_bearing) >= 10.0:
                        current_yaw = imu.deg_fresh(
                            wait_up_to_s=0.6,
                            max_age_s=0.4,
                        )
                        if current_yaw is None:
                            return "lidar_unavailable"
                        committed_gap_yaw = float(current_yaw) + float(gap_bearing)
                        continue
                    else:
                        forced_distance = min(
                            0.25,
                            max(
                                0.12,
                                gap_range - float(args.physical_body_radius_m) - 0.10,
                            ),
                        )
                        candidates.append(
                            (
                                0,
                                -math.inf,
                                0.0,
                                0.0,
                                forced_distance,
                                np.array([forced_distance, 0.0], dtype=np.float64),
                            )
                        )
                if not candidates:
                    print(
                        f"[explore] {label}: no forward corridor is visible; "
                        "taking one bounded "
                        f"{'counterclockwise' if search_turn_direction > 0 else 'clockwise'} "
                        "scan step instead of aborting."
                    )
                    if not _phase_1_1_turn_toward_partial_map_route(
                        20.0 * search_turn_direction,
                        label,
                    ):
                        search_turn_direction *= -1.0
                    continue

            candidates.sort(
                key=lambda item: (item[0], item[1], item[2], item[3], item[4]),
                reverse=True,
            )
            _full_clear, predicted_margin, _improvement, _heading_cost, distance, body_delta = candidates[0]
            travel_deg = math.degrees(math.atan2(body_delta[1], body_delta[0]))
            print(
                f"[explore] {label} escape decision {decision_index}: "
                f"visible open-space bearing {travel_deg:+.0f}deg, target advance "
                f"{distance:.2f}m (predicted pivot margin "
                f"{predicted_margin * 100:+.1f}cm)."
            )

            if abs(travel_deg) >= 10.0:
                current_yaw = imu.deg_fresh(wait_up_to_s=0.6, max_age_s=0.4)
                if current_yaw is None:
                    return "lidar_unavailable"
                committed_gap_yaw = float(current_yaw) + float(travel_deg)
                print(
                    f"[explore] {label}: committing to the {travel_deg:+.0f}deg "
                    "opening; it will be centered before forward movement."
                )
                continue

            # The selected corridor is now physically forward. Do not send an
            # off-axis command: deployed hosts may silently ignore y velocity.
            body_delta = np.array([distance, 0.0], dtype=np.float64)

            before = _startup_forward_scan()
            if before is None:
                return "lidar_unavailable"
            before_margin = _full_pivot_clearance_margin_m(before, collision_profile, **geometry)
            before_hit = _collision_box_violation(before, collision_profile, **geometry)
            baseline_depth = _collision_depth_m(before_hit)
            requested_speed = max(
                0.80,
                min(1.0, abs(float(args.startup_escape_speed))),
            )
            duration_s = float(
                np.clip(
                    distance / max(0.05, float(args.drive_command_distance_scale) * requested_speed),
                    0.55,
                    3.0,
                )
            )
            worsening_since: float | None = None
            started = time.monotonic()
            heading_hold = imu.deg_fresh(wait_up_to_s=0.5, max_age_s=0.4)
            if heading_hold is None:
                return "lidar_unavailable"
            controller.clear_safety_latch()
            controller.set_rotation_safety_check(_rotation_safety_check)
            controller.start()
            controller.translate_body(
                requested_speed,
                0.0,
                float(heading_hold),
            )
            try:
                while time.monotonic() - started < duration_s:
                    live = _startup_forward_scan()
                    if live is not None:
                        live_depth = _collision_depth_m(
                            _collision_box_violation(live, collision_profile, **geometry)
                        )
                        if live_depth > baseline_depth + 0.02:
                            if worsening_since is None:
                                worsening_since = time.monotonic()
                            elif time.monotonic() - worsening_since >= 0.16:
                                print(
                                    f"[explore] {label}: straight escape stopped "
                                    "because the live collision envelope worsened."
                                )
                                break
                        else:
                            worsening_since = None
                    time.sleep(0.04)
            finally:
                controller.halt()
                time.sleep(0.15)
                controller.shutdown()
            time.sleep(0.30)
            after = _startup_forward_scan()
            if after is None:
                return "lidar_unavailable"
            after_margin = _full_pivot_clearance_margin_m(after, collision_profile, **geometry)
            after_depth = _collision_depth_m(_collision_box_violation(after, collision_profile, **geometry))
            measured_along, measured_cross, motion_score = _measure_startup_translation(
                before,
                after,
                body_delta,
            )
            motion_verified = (
                (
                    motion_score >= float(args.min_match_score)
                    and measured_along >= 0.025
                    and measured_cross <= 0.10
                )
                or after_margin >= before_margin + 0.015
                or after_depth <= baseline_depth - 0.010
            )
            if not motion_verified:
                print(
                    f"[explore] {label}: scan odometry was inconclusive after "
                    "the escape command; this is not an anchor or actuator "
                    "failure. Re-selecting visible free space from the live scan "
                    f"(diagnostic LiDAR odometry {measured_along * 100:+.1f}cm/"
                    f"{measured_cross * 100:.1f}cm, match {motion_score:.1f}, "
                    f"pivot margin {before_margin * 100:+.1f} -> "
                    f"{after_margin * 100:+.1f}cm)."
                )
                continue
            print(
                f"[explore] {label}: straight escape verified; pivot margin "
                f"{before_margin * 100:+.1f} -> {after_margin * 100:+.1f}cm."
            )
            verified_forward_moves += 1

    startup_stationary_bootstrap = False

    def _phase_1_1_partial_spin_move_to_open_space_then_full_spin(
        label: str,
        pre_captured_partial: tuple[
            list[tuple[np.ndarray, np.ndarray, float]],
            float,
            float,
            float,
            float,
        ]
        | None = None,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], float, float, float]:
        """Phase 1.1 coordinator: partial map, escape route, replacement spin."""
        if pre_captured_partial is None:
            points = _startup_forward_scan()
            if points is None:
                _abort(f"{label} cannot acquire the one required partial sweep: no LiDAR.")
            geometry = {
                "lidar_offset_forward_m": lever_m,
                "physical_body_radius_m": float(args.physical_body_radius_m),
                "self_mask_inset_m": float(args.collision_self_mask_inset_m),
            }
            positive = _collision_box_rotation_violation(
                points,
                collision_profile,
                float(args.spin_degrees),
                **geometry,
            )
            negative = _collision_box_rotation_violation(
                points,
                collision_profile,
                -float(args.spin_degrees),
                **geometry,
            )

            def _available_sweep(result) -> float:
                return float(args.spin_degrees) if result is None else abs(float(result[1]))

            positive_available = _available_sweep(positive)
            negative_available = _available_sweep(negative)
            partial_direction = 1.0 if positive_available >= negative_available else -1.0
            direction_reason = "larger predicted collision-free sweep"
            if positive_available <= 8.0 and negative_available <= 8.0:
                self_returns = _physical_body_self_return_mask(
                    points,
                    lidar_offset_forward_m=lever_m,
                    physical_body_radius_m=float(args.physical_body_radius_m),
                    self_mask_inset_m=float(args.collision_self_mask_inset_m),
                )
                visible_opening = _widest_visible_corridor_bearing_deg(
                    points[~self_returns],
                    lidar_offset_forward_m=lever_m,
                    corridor_half_width_m=(float(args.physical_body_radius_m) + 0.03),
                )
                if visible_opening is not None and abs(visible_opening[0]) >= 5.0:
                    partial_direction = math.copysign(1.0, visible_opening[0])
                    direction_reason = (
                        f"both pivot directions constrained; wider observed side "
                        f"is {visible_opening[0]:+.0f}deg"
                    )
            print(
                f"[explore]   {label}: ONE partial sweep selected "
                f"({'counterclockwise' if partial_direction > 0 else 'clockwise'}, "
                f"predicted safe arc "
                f"{max(positive_available, negative_available):.0f}deg; "
                f"{direction_reason}). Its scans "
                "will form one frozen temporary map and will never enter the "
                "permanent SLAM map."
            )
            startup_partial_captures.clear()
            scans, net, timed_ratio, monotonic_ratio, _stop_reason = _phase_1_capture_initial_spin_pass(
                partial_direction,
                f"{label} single partial sweep",
            )
        else:
            scans, net, timed_ratio, monotonic_ratio, partial_direction = pre_captured_partial
            print(
                f"[explore]   {label}: the already captured interrupted outbound "
                f"sweep ({net:+.1f}deg, {len(scans)} scans) is the ONE partial "
                "sweep. No opposite-direction probe or second partial capture "
                "will run."
            )
            startup_partial_captures.clear()
        if _anchor_motion_is_complete(net, monotonic_ratio, float(args.spin_degrees)):
            print(
                f"[explore]   {label}: the one sweep completed 360deg; using it directly as the full anchor."
            )
            return scans, net, timed_ratio, partial_direction
        startup_partial_captures.extend(scans)
        partial_plan = _phase_1_1_build_partial_map_and_escape_route(label)
        if partial_plan is None:
            _abort(
                f"{label} single partial sweep did not expose a reachable open-space "
                "route; no second sweep or reactive turn loop is permitted."
            )
        route, route_reference_yaw = partial_plan
        print(
            f"[explore]   {label}: partial sweep is now frozen; executing its ONE "
            "route without selecting another target."
        )
        # PHASE 1.1 STEP C — ATTEMPT THE ONE FROZEN ESCAPE SEGMENT ONCE.
        # Whether it completes or live safety stops it, Phase 1.1 never issues a
        # second forward translation. Control proceeds directly to the required
        # replacement 360-degree spin from the robot's resulting pose.
        escape_completed = _phase_1_1_follow_partial_map_escape_route(
            route,
            route_reference_yaw,
            label,
        )
        if not escape_completed:
            print(
                f"[explore]   {label}: the single escape segment ended early under "
                "live safety. It will NOT be repeated; starting the replacement "
                "360deg spin from the current pose."
            )

        print(
            f"[explore]   {label}: temporary-map route complete. Discarding the "
            "partial map and capturing the replacement full 360deg anchor now."
        )
        startup_partial_captures.clear()
        replacement_attempt = 1
        completed = _phase_1_capture_verified_initial_spin_pass(
            partial_direction,
            f"{label} replacement full 360deg anchor",
            required=True,
        )
        while completed is None:
            # A collision-interrupted required spin is a normal Phase 1.1 state,
            # not an impossible value. The failed clockwise/counterclockwise
            # captures were deliberately isolated in startup_partial_captures by
            # _phase_1_capture_verified_initial_spin_pass(). Use exactly that
            # evidence for one more disposable route, move once, then retry the
            # complete spin. Never assert/crash and never execute multiple route
            # segments between spin attempts.
            replacement_attempt += 1
            print(
                f"[explore]   {label}: replacement 360deg spin remained "
                "collision-constrained in both directions. Building one new "
                "disposable partial map from those scans, moving once toward "
                f"open space, then retrying the full spin (attempt "
                f"{replacement_attempt})."
            )
            retry_plan = _phase_1_1_build_partial_map_and_escape_route(
                f"{label} replacement recovery {replacement_attempt}"
            )
            if retry_plan is None:
                _abort(
                    f"{label} replacement spin was collision-constrained and its "
                    "captured LiDAR evidence contained no reachable open-space "
                    "escape segment."
                )
            retry_route, retry_reference_yaw = retry_plan
            retry_escape_completed = _phase_1_1_follow_partial_map_escape_route(
                retry_route,
                retry_reference_yaw,
                f"{label} replacement recovery {replacement_attempt}",
            )
            if not retry_escape_completed:
                print(
                    f"[explore]   {label}: replacement recovery "
                    f"{replacement_attempt} movement ended early under live "
                    "safety. It will not be repeated before the next full-spin "
                    "attempt."
                )
            startup_partial_captures.clear()
            completed = _phase_1_capture_verified_initial_spin_pass(
                partial_direction,
                f"{label} replacement full 360deg anchor attempt {replacement_attempt}",
                required=True,
            )
        # The temporary map has no lifecycle beyond relocation. Ensure even an
        # interrupted first direction inside full-anchor acquisition leaves no
        # startup scans available to any later planner or mapper.
        startup_partial_captures.clear()
        rr.log(
            "startup_partial_map/returns",
            rr.Points3D(np.empty((0, 3), dtype=np.float32)),
        )
        rr.log("startup_partial_map/plan", rr.LineStrips3D([]))
        return completed

    # =======================================================================
    # EXPLORE PHASE 1.2 — BUILD AND SELECT THE INITIAL MAP
    # =======================================================================
    def _phase_1_2_build_initial_map_candidate(
        scans: list[tuple[np.ndarray, np.ndarray, float]],
        label: str,
        *,
        deskew: bool,
    ) -> tuple[WorldMap, list[tuple[np.ndarray, float, Pose2D]], float, float, float]:
        """Build one pass sequentially while gyro owns rotation.

        Scan matching may move the robot centre because a mecanum base really
        does wander during a pivot. It may not rewrite heading: the continuous
        IMU delta owns that, preventing accumulated rotational forgetting.
        """
        pass_map = WorldMap(
            grid_res_m=float(args.frontier_res_m),
            footprint_clear_m=float(args.self_clear_m),
            lidar_offset_m=lever_m,
            forward_offset_deg=forward_offset,
        )
        stride = max(1, math.ceil(len(scans) / max(1, int(args.snapshots))))
        selected = scans[::stride]
        accepted: list[tuple[np.ndarray, float, Pose2D]] = []
        matched = 0
        rejected = 0
        score_sum = 0.0
        exhaustive_retries = 0
        build_started = time.monotonic()
        prev_pose: Pose2D | None = None
        prev_heading: float | None = None
        for raw_xy, deskewed_xy, heading in selected:
            local_xy = deskewed_xy if deskew else raw_xy
            if prev_pose is None:
                seed = _lidar_pose_from_robot_centre(
                    np.zeros(2, dtype=np.float64), heading, lever_m, forward_offset
                )
                pose = seed
                keep = True
            else:
                theta_seed = float(prev_pose.theta_deg) + float(heading) - float(prev_heading)
                previous_centre = _robot_centre_from_lidar_pose(prev_pose, lever_m, forward_offset)
                seed = _lidar_pose_from_robot_centre(previous_centre, theta_seed, lever_m, forward_offset)
                # The IMU owns anchor heading, so try a tight angular solve
                # first. Strong geometry accepts immediately; ambiguous scans
                # fall back to the original exhaustive search below.
                solved, score = _localize(
                    local_xy,
                    pass_map,
                    seed,
                    args,
                    float(args.search_xy_m),
                    min(2.0, float(args.theta_window_deg)),
                    coarse_angle_step_deg=1.0,
                    fine_angle_step_deg=0.5,
                    fine_theta_half_window_deg=2.0,
                )
                if score < 12.0:
                    exhaustive_retries += 1
                    full_solved, full_score = _localize(
                        local_xy,
                        pass_map,
                        seed,
                        args,
                        float(args.search_xy_m),
                        min(5.0, float(args.theta_window_deg)),
                    )
                    if full_score > score:
                        solved, score = full_solved, full_score
                solved_centre = _robot_centre_from_lidar_pose(solved, lever_m, forward_offset)
                centre_step = float(np.hypot(*(solved_centre - previous_centre)))
                theta_error = abs(((float(solved.theta_deg) - float(seed.theta_deg) + 180.0) % 360.0) - 180.0)
                keep = (
                    score >= float(args.min_match_score)
                    and centre_step <= float(args.search_xy_m) + 0.03
                    and theta_error <= 5.0
                )
                if keep:
                    # Translation comes from LiDAR; rotation remains the
                    # gyro-propagated seed. This tracks real pivot drift without
                    # letting scan matching accumulate angular deformation.
                    pose = _lidar_pose_from_robot_centre(solved_centre, theta_seed, lever_m, forward_offset)
                    matched += 1
                    score_sum += float(score)
                else:
                    pose = seed
            if keep:
                pass_map.add(local_xy, pose, gold=True)
                accepted.append((local_xy, heading, pose))
            else:
                rejected += 1
            prev_pose = pose
            prev_heading = heading
        attempted = max(1, len(selected) - 1)
        ratio = float(matched) / float(attempted)
        mean_score = score_sum / max(1, matched)
        max_heading_gap = _anchor_heading_gap_deg(
            [item[1] for item in accepted],
            stationary_full_revolutions=startup_stationary_bootstrap,
        )
        mode = "stationary" if startup_stationary_bootstrap else ("deskewed" if deskew else "raw")
        build_elapsed = time.monotonic() - build_started
        print(
            f"[explore]   {label} {mode} candidate: accepted "
            f"{len(accepted)}/{len(selected)} scans; rejected {rejected}, "
            f"validation {ratio:.1%}, mean match {mean_score:.1f}, "
            f"largest heading gap {max_heading_gap:.1f}deg, "
            f"{build_elapsed:.1f}s ({exhaustive_retries} exhaustive retries)."
        )
        return pass_map, accepted, ratio, mean_score, max_heading_gap

    def _phase_1_2_choose_initial_map_candidate(
        scans: list[tuple[np.ndarray, np.ndarray, float]],
        label: str,
        timed_ratio: float,
        *,
        permit_auto_deskew: bool = True,
    ) -> tuple[
        WorldMap,
        list[tuple[np.ndarray, float, Pose2D]],
        float,
        float,
        float,
    ]:
        mode = str(args.anchor_deskew)
        candidates: list[
            tuple[
                str,
                tuple[
                    WorldMap,
                    list[tuple[np.ndarray, float, Pose2D]],
                    float,
                    float,
                    float,
                ],
            ]
        ] = []
        if mode in ("auto", "off"):
            candidates.append(
                (
                    "raw",
                    _phase_1_2_build_initial_map_candidate(
                        scans,
                        label,
                        deskew=False,
                    ),
                )
            )
        build_deskewed = mode == "on"
        if mode == "auto" and timed_ratio >= 0.80 and permit_auto_deskew:
            raw = candidates[0][1]
            raw_is_strong = (
                raw[2] >= 0.95 and raw[3] >= 12.0 and raw[4] <= float(args.anchor_max_heading_gap_deg)
            )
            build_deskewed = not raw_is_strong
            if raw_is_strong:
                print(
                    f"[explore]   {label}: raw anchor is already strong; skipping redundant deskewed rebuild."
                )
        elif mode == "auto" and not permit_auto_deskew:
            print(
                f"[explore]   {label}: stronger pass already available; "
                "using this pass for raw bidirectional verification only."
            )
        if build_deskewed:
            candidates.append(
                (
                    "deskewed",
                    _phase_1_2_build_initial_map_candidate(
                        scans,
                        label,
                        deskew=True,
                    ),
                )
            )
        if not candidates:
            _abort(
                f"Anchor {label} cannot evaluate deskew because timestamp coverage "
                f"was only {timed_ratio:.1%}."
            )
        # Raw capture is the proven field behavior. Auto mode adopts deskew only
        # for a material, unambiguous improvement rather than selecting it on
        # one scan or a fraction of a score point.
        chosen_mode, chosen = candidates[0]
        if len(candidates) == 2:
            _raw_mode, raw = candidates[0]
            deskew_mode, deskewed = candidates[1]
            ratio_gain = float(deskewed[2]) - float(raw[2])
            score_gain = float(deskewed[3]) - float(raw[3])
            if deskewed[2] >= raw[2] and ((ratio_gain >= 0.03 and score_gain >= 0.0) or score_gain >= 0.75):
                chosen_mode, chosen = deskew_mode, deskewed
        pass_map, accepted, ratio, mean_score, max_heading_gap = chosen
        print(
            f"[explore]   {label}: selected {chosen_mode} candidate "
            f"({ratio:.1%}, mean match {mean_score:.1f}, "
            f"largest gap {max_heading_gap:.1f}deg)."
        )
        if (
            len(accepted) < 40
            or ratio < float(args.anchor_min_accept_ratio)
            or max_heading_gap > float(args.anchor_max_heading_gap_deg)
        ):
            print(
                f"[explore]   WARNING: {label} is below the former quality gate "
                f"({len(accepted)} scans, "
                f"{ratio:.1%} validation, largest heading gap "
                f"{max_heading_gap:.1f}deg); it will be ranked against the other pass."
            )
        return pass_map, accepted, ratio, mean_score, max_heading_gap

    def _phase_1_2_initial_map_candidate_quality(
        accepted: list[tuple[np.ndarray, float, Pose2D]],
        ratio: float,
        mean_score: float,
        max_heading_gap: float,
    ) -> float:
        """Rank available anchor passes without turning diagnostics into gates."""
        coverage = max(0.0, min(1.0, 1.0 - float(max_heading_gap) / 360.0))
        density = min(1.0, len(accepted) / 80.0)
        match = max(0.0, min(1.0, float(mean_score) / 16.0))
        return coverage * max(0.0, float(ratio)) * (0.70 + 0.20 * density + 0.10 * match)

    # =======================================================================
    # EXPLORE PHASE 1.4 — VALIDATE THE INITIAL MAP
    # =======================================================================
    def _phase_1_4_validate_initial_map(
        candidate_map: WorldMap,
        seed_pose: Pose2D,
    ) -> tuple[Pose2D | None, float, float, int, int, float, float]:
        """Independently recognize an anchor from a stationary multi-scan batch.

        Sequential spin matching is correlated evidence: adjacent scans can all
        agree with the same locally warped map. This batch is captured only after
        the base has stopped and requires a compact pose mode across fresh full
        LiDAR revolutions before navigation may begin.
        """
        _phase_1_hold_base_stopped_before_processing()
        requested = max(1, int(args.anchor_validation_scans))
        if not anchor_stop_state[0][0]:
            return None, -1.0, 0.0, 0, requested, float("inf"), float("inf")
        minimum = max(1, int(args.anchor_validation_min_inliers))
        poses: list[Pose2D] = []
        scores: list[float] = []
        supports: list[float] = []
        accepted_locals: list[np.ndarray] = []
        observed_scores: list[float] = []
        observed_supports: list[float] = []
        occupied_support = _inflate(
            candidate_map.grid.occupied(),
            max(1, int(math.ceil(0.10 / candidate_map.grid.res))),
        )
        after_id = feed.latest()[0]
        time.sleep(0.30)
        for _ in range(requested):
            frame_id, frame = feed.wait_for_frame_after(
                after_frame_id=after_id,
                timeout_s=2.0,
                min_frame_advances=1,
            )
            if frame is None:
                continue
            after_id = int(frame_id)
            local_xy = _scan_local(frame, args)
            if len(local_xy) < 12:
                continue
            solved, score = _localize(
                local_xy,
                candidate_map,
                seed_pose,
                args,
                0.12,
                4.0,
            )
            world_xy = _transform_points(local_xy, solved)
            endpoint_support = _grid_endpoint_support_ratio(
                world_xy,
                occupied_support,
                candidate_map.grid.origin,
                candidate_map.grid.res,
            )
            scan_endpoint_grid = np.zeros_like(occupied_support)
            scan_jj = np.floor(
                (world_xy[:, 0] - candidate_map.grid.origin[0]) / candidate_map.grid.res
            ).astype(np.int64)
            scan_ii = np.floor(
                (world_xy[:, 1] - candidate_map.grid.origin[1]) / candidate_map.grid.res
            ).astype(np.int64)
            scan_inside = (
                (scan_ii >= 0)
                & (scan_ii < scan_endpoint_grid.shape[0])
                & (scan_jj >= 0)
                & (scan_jj < scan_endpoint_grid.shape[1])
            )
            scan_endpoint_grid[scan_ii[scan_inside], scan_jj[scan_inside]] = True
            scan_endpoint_support = _inflate(
                scan_endpoint_grid,
                max(1, int(math.ceil(0.10 / candidate_map.grid.res))),
            )
            map_endpoint_support = _grid_endpoint_support_ratio(
                candidate_map.reference(),
                scan_endpoint_support,
                candidate_map.grid.origin,
                candidate_map.grid.res,
            )
            validation_support = _anchor_validation_support_ratio(
                endpoint_support,
                map_endpoint_support,
                body_occluded_sensor=True,
            )
            observed_scores.append(float(score))
            observed_supports.append(validation_support)
            if score >= float(args.anchor_validation_score) and validation_support >= float(
                args.anchor_validation_min_endpoint_support
            ):
                poses.append(solved)
                scores.append(float(score))
                supports.append(validation_support)
                accepted_locals.append(local_xy)
        # One excellent revolution is still one measurement.  It previously
        # bypassed the configured 4/5 stationary consensus, which is exactly
        # how this run declared an anchor "validated" from 1/5 scans after the
        # two spin passes had already disagreed by 18.6cm.  A valid initial
        # map must have the same independent stopped-batch vote demanded by
        # every later mapping checkpoint.
        if len(poses) < minimum:
            mean_score = float(np.mean(observed_scores)) if observed_scores else -1.0
            mean_support = float(np.mean(observed_supports)) if observed_supports else 0.0
            return (
                None,
                mean_score,
                mean_support,
                len(poses),
                requested,
                float("inf"),
                float("inf"),
            )

        consensus, inliers, centre_scatter, heading_scatter = _stationary_pose_inlier_consensus(
            poses,
            reference_theta_deg=float(seed_pose.theta_deg),
            lever_m=lever_m,
            forward_offset_deg=forward_offset,
            max_position_residual_m=float(args.anchor_validation_centre_scatter_m),
            max_heading_residual_deg=float(args.anchor_validation_heading_scatter_deg),
            min_inliers=minimum,
            reference_centre_xy=_robot_centre_from_lidar_pose(seed_pose, lever_m, forward_offset),
        )
        inlier_scores = [scores[index] for index in inliers]
        inlier_supports = [supports[index] for index in inliers]
        mean_score = float(np.mean(inlier_scores)) if inlier_scores else -1.0
        mean_support = float(np.mean(inlier_supports)) if inlier_supports else 0.0
        accepted = _stationary_anchor_is_accepted(
            len(inliers),
            minimum,
            centre_scatter,
            float(args.anchor_validation_centre_scatter_m),
            heading_scatter,
            float(args.anchor_validation_heading_scatter_deg),
        )
        if accepted and inliers:
            strongest = max(inliers, key=lambda index: scores[index])
            verified_live_anchor_scan[0] = accepted_locals[strongest].copy()
        return (
            consensus if accepted else None,
            mean_score,
            mean_support,
            len(inliers),
            requested,
            float(centre_scatter),
            float(heading_scatter),
        )

    def _phase_1_4_register_terminal_anchor_pose(
        candidate_map: WorldMap,
        accepted_pass: list[tuple[np.ndarray, float, Pose2D]],
        seed_pose: Pose2D,
    ) -> tuple[Pose2D | None, float, float, int]:
        """Register the physically stopped robot to the end of its anchor pass.

        Agreement between two constructed maps validates geometry, not the live
        robot pose. Match fresh stationary revolutions first to the terminal
        local portion of the selected pass, where overlap is strongest, and
        require a compact multi-scan mode before navigation is released.
        """
        if not anchor_stop_state[0][0] or not accepted_pass:
            return None, 0.0, 0.0, 0
        terminal = accepted_pass[-10:]
        terminal_reference = np.concatenate(
            [_transform_points(local_xy, pose) for local_xy, _heading, pose in terminal], axis=0
        )
        after_id = feed.latest()[0]
        yaw_reference = imu.deg_fresh(wait_up_to_s=0.5, max_age_s=0.4)
        candidates: list[Pose2D] = []
        scores: list[float] = []
        supports: list[float] = []
        locals_used: list[np.ndarray] = []
        for _ in range(max(4, int(args.anchor_validation_scans))):
            frame_id, frame = feed.wait_for_frame_after(
                after_frame_id=after_id,
                timeout_s=1.5,
                min_frame_advances=1,
            )
            if frame is None:
                continue
            after_id = int(frame_id)
            local_xy = _scan_local(frame, args)
            if len(local_xy) < 12:
                continue
            scan_yaw = _imu_yaw_for_scan(imu, frame)
            theta_seed = float(seed_pose.theta_deg)
            if yaw_reference is not None and scan_yaw is not None:
                theta_seed += float(scan_yaw) - float(yaw_reference)
            scan_seed = _lidar_pose_from_robot_centre(
                _robot_centre_from_lidar_pose(seed_pose, lever_m, forward_offset),
                theta_seed,
                lever_m,
                forward_offset,
            )
            solved, score, support = _localize_against_points(
                local_xy,
                terminal_reference,
                scan_seed,
                args,
                search_xy_m=0.30,
                theta_window_deg=4.0,
            )
            solved = _pose_with_imu_heading(solved, theta_seed, lever_m, forward_offset)
            if score >= max(4.0, float(args.min_match_score) - 2.0) and support >= 0.18:
                candidates.append(solved)
                scores.append(float(score))
                supports.append(float(support))
                locals_used.append(local_xy)
        if len(candidates) < 2:
            return (
                None,
                float(np.mean(scores)) if scores else 0.0,
                float(np.mean(supports)) if supports else 0.0,
                len(candidates),
            )
        consensus, inliers, scatter, heading_scatter = _stationary_pose_inlier_consensus(
            candidates,
            float(seed_pose.theta_deg),
            lever_m,
            forward_offset,
            max_position_residual_m=0.10,
            max_heading_residual_deg=4.0,
            min_inliers=2,
            reference_centre_xy=_robot_centre_from_lidar_pose(seed_pose, lever_m, forward_offset),
        )
        if len(inliers) < 2 or scatter > 0.10 or heading_scatter > 4.0:
            return None, float(np.mean(scores)), float(np.mean(supports)), len(inliers)
        strongest = max(inliers, key=lambda index: scores[index])
        verified_live_anchor_scan[0] = locals_used[strongest].copy()
        return (
            consensus,
            float(np.mean([scores[index] for index in inliers])),
            float(np.mean([supports[index] for index in inliers])),
            len(inliers),
        )

    # =======================================================================
    # EXPLORE PHASE 1 — INITIAL 360-DEGREE SPIN (EXECUTION STARTS HERE)
    #
    # Normal path:
    #   capture initial spin -> optionally capture reverse verification spin ->
    #   continue to Phase 1.2.
    #
    # Collision path:
    #   enter Phase 1.1 -> build one disposable partial map -> move to its open
    #   space -> capture a replacement complete spin -> return to Phase 1.2.
    # =======================================================================
    # The mechanical LiDAR revolution is chassis-occluded. A full body spin is
    # preferred. A constrained start uses raw live LiDAR to face a visible gap,
    # advances into open space under a non-worsening collision guard, and then
    # captures the complete 360-degree anchor required by ExploreSystem.
    initialization_state = "clear" if _startup_full_pivot_is_clear() else "blocked"
    if initialization_state != "clear":
        startup_stationary_bootstrap = True
        print(
            "[explore] full-body pivot is constrained; using live LiDAR to move "
            "toward the widest visible opening, advance into it, then start the "
            "complete 360deg anchor."
        )

    # Capture both directions back-to-back. No scan matching or map construction
    # is allowed between them: the robot stops briefly for the direction reversal,
    # immediately performs the return sweep, and only then does CPU-heavy work.
    if startup_stationary_bootstrap:
        # >>> EXPLORE PHASE 1.1: preflight says the initial spin is blocked.
        first_capture = _phase_1_1_partial_spin_move_to_open_space_then_full_spin("constrained-start anchor")
        # Relocation returns only the later full-360 capture; its single partial
        # sweep is never passed to permanent anchor-map construction.
        startup_stationary_bootstrap = False
    else:
        outbound_scans, outbound_net, outbound_timed, outbound_monotonic, _reason = (
            _phase_1_capture_initial_spin_pass(+1.0, "outbound pass")
        )
        if _anchor_motion_is_complete(
            outbound_net,
            outbound_monotonic,
            float(args.spin_degrees),
        ):
            first_capture = (
                outbound_scans,
                outbound_net,
                outbound_timed,
                1.0,
            )
        else:
            # >>> EXPLORE PHASE 1.1: the spin started but collision safety
            # interrupted it. Its exact capture becomes the one partial sweep.
            print(
                "[explore] outbound body spin was interrupted; that capture is "
                "now the ONE frozen partial sweep. No opposite-direction probe "
                "will run before relocation."
            )
            first_capture = _phase_1_1_partial_spin_move_to_open_space_then_full_spin(
                "collision-safe anchor recovery",
                pre_captured_partial=(
                    outbound_scans,
                    outbound_net,
                    outbound_timed,
                    outbound_monotonic,
                    1.0,
                ),
            )
    assert first_capture is not None
    first_scans, first_net_spin, first_timed_ratio, first_command_direction = first_capture
    second_capture: (
        tuple[
            list[tuple[np.ndarray, np.ndarray, float]],
            float,
            float,
            float,
        ]
        | None
    ) = None
    if int(args.anchor_passes) == 2 and not startup_stationary_bootstrap:
        second_capture = _phase_1_capture_verified_initial_spin_pass(
            -first_command_direction, "return pass", required=False
        )
        if second_capture is not None and first_net_spin * second_capture[1] >= 0.0:
            print(
                "[explore] WARNING: the two anchor commands produced the same IMU "
                "rotation direction; both captures will still be ranked."
            )

    # =======================================================================
    # EXPLORE PHASE 1.2 — CONSTRUCT AND SELECT THE INITIAL MAP
    # =======================================================================
    print("[explore] anchor motion complete; processing captured passes ...")
    first_map, first_accepted, first_ratio, first_mean, first_gap = _phase_1_2_choose_initial_map_candidate(
        first_scans,
        "outbound pass",
        first_timed_ratio,
        permit_auto_deskew=not startup_stationary_bootstrap,
    )
    first_quality = _phase_1_2_initial_map_candidate_quality(
        first_accepted, first_ratio, first_mean, first_gap
    )
    base_map = first_map
    base_pass = first_accepted
    base_label = "outbound"
    base_quality = first_quality
    base_ratio = first_ratio
    base_gap = first_gap
    base_net_spin = first_net_spin
    latest_pass = base_pass
    bidirectional_verified = False
    anchor_bidirectional_conflict = False

    if second_capture is not None:
        second_scans, second_net_spin, second_timed_ratio, _second_direction = second_capture
        second_map, second_accepted, second_ratio, second_mean, second_gap = (
            _phase_1_2_choose_initial_map_candidate(
                second_scans,
                "return pass",
                second_timed_ratio,
                permit_auto_deskew=first_quality < 0.85,
            )
        )
        second_quality = _phase_1_2_initial_map_candidate_quality(
            second_accepted, second_ratio, second_mean, second_gap
        )
        print(f"[explore]   anchor pass ranking: outbound {first_quality:.3f}, return {second_quality:.3f}.")
        if second_quality > first_quality:
            base_map = second_map
            base_pass = second_accepted
            base_label = "return"
            base_quality = second_quality
            base_ratio = second_ratio
            base_gap = second_gap
            base_net_spin = second_net_spin
            latest_pass = second_accepted
            print(
                "[explore] return pass is stronger; using it as the anchor instead "
                "of allowing the weaker outbound pass to stop or degrade the run."
            )

        # Align the independently built reverse pass to pass one with ONE rigid
        # correction. Per-scan corrections are forbidden: their scatter is the
        # quality metric that protects against a smeared or rotationally lost pass.
        comparison_pass = second_accepted if base_label == "outbound" else first_accepted
        sample_step = max(1, len(comparison_pass) // 24)
        source_centres: list[np.ndarray] = []
        solved_centres: list[np.ndarray] = []
        theta_offsets: list[float] = []
        consensus_scores: list[float] = []
        for local_xy, _heading, pass_pose in comparison_pass[::sample_step]:
            solved, score = _localize(local_xy, base_map, pass_pose, args, 0.15, 5.0)
            if score < float(args.anchor_consensus_score):
                continue
            source_centres.append(_robot_centre_from_lidar_pose(pass_pose, lever_m, forward_offset))
            solved_centres.append(_robot_centre_from_lidar_pose(solved, lever_m, forward_offset))
            theta_offsets.append(
                ((float(solved.theta_deg) - float(pass_pose.theta_deg) + 180.0) % 360.0) - 180.0
            )
            consensus_scores.append(score)
        sampled_count = len(comparison_pass[::sample_step])
        consensus_ratio = len(consensus_scores) / max(1, sampled_count)
        if source_centres:
            median_theta = float(np.median(np.asarray(theta_offsets, dtype=np.float64)))
            theta_residuals = np.abs(np.asarray(theta_offsets, dtype=np.float64) - median_theta)
            theta_p90, theta_max = _p90_and_max(theta_residuals)
            rotation_rad = math.radians(median_theta)
            rotation = np.array(
                [
                    [math.cos(rotation_rad), -math.sin(rotation_rad)],
                    [math.sin(rotation_rad), math.cos(rotation_rad)],
                ]
            )
            source_array = np.asarray(source_centres, dtype=np.float64)
            solved_array = np.asarray(solved_centres, dtype=np.float64)
            rotated_sources = source_array @ rotation.T
            translations = solved_array - rotated_sources
            median_translation = np.median(translations, axis=0)
            residuals = solved_array - (rotated_sources + median_translation)
            centre_p90, centre_max = _p90_and_max(np.hypot(residuals[:, 0], residuals[:, 1]))
        else:
            median_theta = 0.0
            median_translation = np.zeros(2, dtype=np.float64)
            rotation = np.eye(2, dtype=np.float64)
            centre_p90 = centre_max = float("inf")
            theta_p90 = theta_max = float("inf")
        print(
            f"[explore]   bidirectional consensus: {len(consensus_scores)}/{sampled_count} "
            f"matches ({consensus_ratio:.1%}), centre residual "
            f"p90/max {centre_p90 * 100:.1f}/{centre_max * 100:.1f}cm, "
            f"heading residual p90/max {theta_p90:.1f}/{theta_max:.1f}deg."
        )
        consensus_ok = _bidirectional_consensus_is_acceptable(
            consensus_ratio,
            centre_p90,
            theta_p90,
            float(args.anchor_consensus_ratio),
            float(args.anchor_consensus_centre_p90_m),
            float(args.anchor_consensus_heading_p90_deg),
            linear_quantization_slack_m=0.5 * float(args.frontier_res_m),
        )
        if not consensus_ok:
            anchor_bidirectional_conflict = True
            print(
                "[explore] WARNING: bidirectional anchor maps disagree "
                f"(ratio {consensus_ratio:.1%}, p90 residual "
                f"{centre_p90 * 100:.1f}cm/{theta_p90:.1f}deg); "
                f"continuing with the stronger {base_label} pass."
            )
        aligned_comparison: list[tuple[np.ndarray, float, Pose2D]] = []
        for local_xy, heading, pass_pose in comparison_pass:
            pass_centre = _robot_centre_from_lidar_pose(pass_pose, lever_m, forward_offset)
            aligned_centre = rotation @ pass_centre + median_translation
            aligned_pose = _lidar_pose_from_robot_centre(
                aligned_centre,
                float(pass_pose.theta_deg) + median_theta,
                lever_m,
                forward_offset,
            )
            aligned_comparison.append((local_xy, heading, aligned_pose))
        # The return pass is independent evidence and supplies the robot's
        # current pose. Do not paint its duplicate observations into an already
        # dense outbound map: residual cross-pass error would only thicken walls.
        if consensus_ok:
            bidirectional_verified = True
            if base_label == "outbound":
                latest_pass = aligned_comparison
            print(
                f"[explore] the other pass verified the {base_label} anchor; "
                "preserving the stronger map instead of merging duplicate scans."
            )

    if not base_pass:
        _abort("Neither anchor spin produced a usable LiDAR scan; no map can be initialized.")

    # Use one coherent pass. Quality measurements above select the best evidence
    # but never veto an otherwise usable run.
    world_map = WorldMap(
        grid_res_m=float(args.frontier_res_m),
        footprint_clear_m=float(args.self_clear_m),
        lidar_offset_m=lever_m,
        forward_offset_deg=forward_offset,
    )
    for local_xy, _heading, pose in base_pass:
        world_map.add(local_xy, pose, gold=True)
    last_pass = latest_pass
    prev_pose = last_pass[-1][2]
    prev_heading = last_pass[-1][1]
    anchor_kind = "bidirectionally verified" if bidirectional_verified else "best available"
    print(
        f"[explore] {anchor_kind} {base_label} anchor selected "
        f"(quality {base_quality:.3f}): {len(world_map.scans)} total scans."
    )

    # =======================================================================
    # EXPLORE PHASE 1.3 — RECOVER THE PHYSICAL DRIVE FRAME (NO NUDGE)
    # =======================================================================
    # Forward offset: analytic, from the extraction conventions.
    # Turn sign: the anchor spin commanded a positive theta.vel throughout, so the
    # sign of the net IMU sweep tells us whether +theta.vel raises or lowers the
    # gyro. This is the rotation calibration, taken for free from the spin.
    net_spin = first_net_spin
    if startup_stationary_bootstrap:
        if anchor_turn_observation[0] is not None:
            observed_net, observed_direction = anchor_turn_observation[0]
            turn_sign = 1.0 if observed_net * observed_direction >= 0.0 else -1.0
            controller.turn_sign = float(turn_sign)
            print(
                "[explore] drive frame calibrated from the completed safe portion "
                f"of the anchor sweep: physical forward = map heading "
                f"{forward_offset:+.0f}deg; +theta.vel "
                f"{'raises' if turn_sign > 0 else 'lowers'} IMU "
                f"(observed sweep {observed_net:+.0f}deg)."
            )
        else:
            # Do not probe the turn sign by moving when the footprint is blocked;
            # live IMU feedback still closes every subsequent turn.
            turn_sign = float(controller.turn_sign)
            print(
                "[explore] drive frame initialized without an unsafe nudge: physical "
                f"forward = map heading {forward_offset:+.0f}deg; turn sign retained "
                "from the configured base convention and will be IMU-closed-loop."
            )
    else:
        # Normalize the measured sweep to the response of a positive theta command;
        # startup recovery may have acquired the first valid pass in the opposite
        # command direction.
        turn_sign = 1.0 if net_spin * first_command_direction >= 0.0 else -1.0
        controller.turn_sign = float(turn_sign)
        print(
            f"[explore] drive frame recovered (no nudge): physical forward = map heading "
            f"{forward_offset:+.0f}deg; +theta.vel {'raises' if turn_sign > 0 else 'lowers'} IMU "
            f"(net spin {net_spin:+.0f}deg)."
        )

    # Pose tracking: current LIDAR pose + the IMU reading it was solved at.
    cur_pose: Pose2D = (
        prev_pose
        if prev_pose is not None
        else _lidar_pose_from_robot_centre(np.zeros(2, dtype=np.float64), 0.0, lever_m, forward_offset)
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

    # =======================================================================
    # EXPLORE PHASE 1.4 — VALIDATE THE INITIAL MAP
    # =======================================================================
    # Independent post-spin refinement: several fresh stationary revolutions
    # try to recognize one compact pose in the completed map. Bidirectional
    # consensus is already independent map evidence and remains authoritative
    # when this secondary single-view comparison is merely inconclusive.
    validation_overridden_by_bidirectional = False
    validation_accepted_from_complete_pass = False
    anchor_coverage_eligible = _anchor_pass_is_eligible(
        len(base_pass),
        base_ratio,
        base_gap,
        float(args.anchor_min_accept_ratio),
        float(args.anchor_max_heading_gap_deg),
    )
    if not anchor_coverage_eligible:
        print(
            f"[explore] selected anchor pass has {base_ratio:.1%} accepted scans "
            f"and a {base_gap:.1f}deg heading gap; attempting the independent "
            "stationary cross-check despite sparse angular sampling."
        )
    (
        validation_pose,
        validation_score,
        validation_support,
        validation_inliers,
        validation_requested,
        validation_centre_scatter,
        validation_heading_scatter,
    ) = _phase_1_4_validate_initial_map(world_map, cur_pose)
    if validation_pose is None and _bidirectional_anchor_is_sufficient(
        bidirectional_verified,
        len(base_pass),
        base_ratio,
        base_gap,
        float(args.anchor_min_accept_ratio),
        float(args.anchor_max_heading_gap_deg),
    ):
        terminal_pose, terminal_score, terminal_support, terminal_inliers = (
            _phase_1_4_register_terminal_anchor_pose(
                world_map,
                base_pass,
                cur_pose,
            )
        )
        if terminal_pose is not None:
            validation_pose = terminal_pose
            validation_score = terminal_score
            validation_support = terminal_support
            validation_inliers = terminal_inliers
            validation_overridden_by_bidirectional = True
            print(
                "[explore] bidirectional consensus verified map geometry and "
                f"{terminal_inliers} fresh terminal-view scans registered the "
                "stopped robot inside that map; navigation pose initialized."
            )
        else:
            print(
                "[explore] bidirectional consensus verified map geometry, but "
                "fresh terminal-view scans did not verify the robot's current "
                f"pose ({terminal_inliers} inliers, match {terminal_score:.1f}, "
                f"support {terminal_support:.1%}). Map quality will not be used "
                "as a substitute for live-pose registration."
            )
    if (
        validation_pose is None
        and not anchor_bidirectional_conflict
        and _anchor_pass_is_self_consistent(
        len(base_pass),
        base_ratio,
        base_quality,
        base_net_spin,
        float(args.spin_degrees),
        )
    ):
        validation_pose = cur_pose
        validation_accepted_from_complete_pass = True
        verified_live_anchor_scan[0] = base_pass[-1][0].copy()
        print(
            "[explore] stationary anchor cross-check was inconclusive, but the "
            f"physical sweep completed {base_net_spin:+.1f}deg with "
            f"{len(base_pass)} coherent scans, {base_ratio:.1%} acceptance, and "
            f"quality {base_quality:.3f}. Preserving that internally consistent "
            "anchor and its terminal pose; no rebuild loop is required."
        )
    rebuild_round = 0
    while validation_pose is None and rebuild_round < int(args.anchor_map_retries):
        rebuild_round += 1
        reacquisition_kind = "a complete 360deg anchor from open space"
        print(
            f"[explore] anchor candidate REJECTED by independent stationary validation "
            f"({validation_inliers}/{validation_requested} coherent scans, "
            f"mean match {validation_score:.1f}, endpoint support "
            f"{validation_support:.1%}, scatter "
            f"{validation_centre_scatter * 100:.1f}cm/"
            f"{validation_heading_scatter:.1f}deg); discarding all candidate geometry "
            f"and reacquiring {reacquisition_kind} "
            f"({rebuild_round}/"
            f"{int(args.anchor_map_retries)})."
        )

        reacquire_direction = 1.0 if rebuild_round % 2 else -1.0
        retry_first = _phase_1_capture_verified_initial_spin_pass(
            reacquire_direction,
            f"anchor rebuild {rebuild_round} outbound pass",
            required=False,
        )
        if retry_first is None:
            print(
                f"[explore] anchor rebuild {rebuild_round}: full sweep was "
                "interrupted; sensing an escape route, moving into open space, "
                "and restarting the complete 360deg capture."
            )
            retry_first = _phase_1_1_partial_spin_move_to_open_space_then_full_spin(
                f"collision-safe anchor rebuild {rebuild_round}"
            )
        # The collision-aware routine returns a complete relocated spin. Its
        # reactive escape scans are never mapping candidates.
        startup_stationary_bootstrap = False
        assert retry_first is not None
        (
            retry_first_scans,
            retry_first_net,
            retry_first_timed,
            retry_first_direction,
        ) = retry_first
        retry_second = None
        if int(args.anchor_passes) == 2 and not startup_stationary_bootstrap:
            retry_second = _phase_1_capture_verified_initial_spin_pass(
                -retry_first_direction,
                f"anchor rebuild {rebuild_round} return pass",
                required=False,
            )
        print(
            f"[explore] anchor rebuild {rebuild_round} full-spin motion complete; "
            "processing captured passes ..."
        )

        retry_candidates: list[
            tuple[
                float,
                str,
                WorldMap,
                list[tuple[np.ndarray, float, Pose2D]],
            ]
        ] = []
        retry_first_map, retry_first_pass, retry_first_ratio, retry_first_mean, retry_first_gap = (
            _phase_1_2_choose_initial_map_candidate(
                retry_first_scans,
                f"anchor rebuild {rebuild_round} outbound pass",
                retry_first_timed,
                permit_auto_deskew=not startup_stationary_bootstrap,
            )
        )
        retry_first_quality = _phase_1_2_initial_map_candidate_quality(
            retry_first_pass,
            retry_first_ratio,
            retry_first_mean,
            retry_first_gap,
        )
        if _anchor_pass_is_eligible(
            len(retry_first_pass),
            retry_first_ratio,
            retry_first_gap,
            float(args.anchor_min_accept_ratio),
            float(args.anchor_max_heading_gap_deg),
        ):
            retry_candidates.append(
                (
                    retry_first_quality,
                    "outbound",
                    retry_first_map,
                    retry_first_pass,
                )
            )

        if retry_second is not None:
            retry_second_scans, _retry_second_net, retry_second_timed, _retry_second_direction = retry_second
            (
                retry_second_map,
                retry_second_pass,
                retry_second_ratio,
                retry_second_mean,
                retry_second_gap,
            ) = _phase_1_2_choose_initial_map_candidate(
                retry_second_scans,
                f"anchor rebuild {rebuild_round} return pass",
                retry_second_timed,
                permit_auto_deskew=retry_first_quality < 0.85,
            )
            retry_second_quality = _phase_1_2_initial_map_candidate_quality(
                retry_second_pass,
                retry_second_ratio,
                retry_second_mean,
                retry_second_gap,
            )
            if _anchor_pass_is_eligible(
                len(retry_second_pass),
                retry_second_ratio,
                retry_second_gap,
                float(args.anchor_min_accept_ratio),
                float(args.anchor_max_heading_gap_deg),
            ):
                retry_candidates.append(
                    (
                        retry_second_quality,
                        "return",
                        retry_second_map,
                        retry_second_pass,
                    )
                )

        if not retry_candidates:
            validation_pose = None
            validation_score = -1.0
            validation_support = 0.0
            validation_inliers = 0
            validation_requested = int(args.anchor_validation_scans)
            validation_centre_scatter = float("inf")
            validation_heading_scatter = float("inf")
            print(
                f"[explore] anchor rebuild {rebuild_round} had no pass with both "
                f">={float(args.anchor_min_accept_ratio):.0%} accepted scans and "
                f"<={float(args.anchor_max_heading_gap_deg):.1f}deg heading gap; "
                "nothing from this rebuild was mapped."
            )
            continue

        base_quality, base_label, _retry_map, base_pass = max(
            retry_candidates,
            key=lambda candidate: candidate[0],
        )
        world_map = WorldMap(
            grid_res_m=float(args.frontier_res_m),
            footprint_clear_m=float(args.self_clear_m),
            lidar_offset_m=lever_m,
            forward_offset_deg=forward_offset,
        )
        for local_xy, _heading, pose in base_pass:
            world_map.add(local_xy, pose, gold=True)
        prev_pose = base_pass[-1][2]
        prev_heading = base_pass[-1][1]
        first_net_spin = retry_first_net
        first_command_direction = retry_first_direction
        if not startup_stationary_bootstrap:
            turn_sign = 1.0 if first_net_spin * first_command_direction >= 0.0 else -1.0
            controller.turn_sign = float(turn_sign)
        cur_pose = prev_pose
        yaw_after_spin = imu.deg()
        if yaw_after_spin is not None:
            cur_pose = _rotate_lidar_pose_about_robot_centre(
                cur_pose,
                float(yaw_after_spin) - (float(yaw0) + float(prev_heading)),
                lever_m,
                forward_offset,
            )
        print(
            f"[explore] anchor rebuild {rebuild_round} selected {base_label} pass "
            f"(quality {base_quality:.3f}, {len(world_map.scans)} scans); "
            "running independent stationary validation."
        )
        (
            validation_pose,
            validation_score,
            validation_support,
            validation_inliers,
            validation_requested,
            validation_centre_scatter,
            validation_heading_scatter,
        ) = _phase_1_4_validate_initial_map(world_map, cur_pose)

    if validation_pose is None:
        _abort(
            f"No coherent anchor map after {int(args.anchor_map_retries) + 1} "
            f"complete acquisition(s); final validation had "
            f"{validation_inliers}/{validation_requested} coherent scans, "
            f"mean match {validation_score:.1f}, scan-to-map endpoint support "
            f"{validation_support:.1%}. The robot remained stationary and no rejected "
            "geometry was used; inspect LiDAR/IMU timing or scan dropout before retrying."
        )
    cur_pose = validation_pose
    if validation_accepted_from_complete_pass:
        validation_method = "complete self-consistent sequential LiDAR/IMU sweep"
    elif validation_overridden_by_bidirectional:
        validation_method = "bidirectional map consensus plus terminal-view pose registration"
    else:
        validation_method = (
            "high-confidence supported stationary view"
            if validation_inliers == 1
            else "multi-scan stationary consensus"
        )
    if validation_accepted_from_complete_pass:
        print(
            f"[explore] anchor initialized by {validation_method}; the optional "
            "stationary cross-check was inconclusive but did not contradict the "
            "completed sweep. Exploration enabled."
        )
    elif validation_overridden_by_bidirectional:
        print(
            f"[explore] anchor independently validated by {validation_method}; "
            "both map geometry and the robot's current live pose are verified. "
            "Exploration enabled."
        )
    else:
        print(
            f"[explore] anchor independently validated by {validation_method}: "
            f"{validation_inliers}/{validation_requested} coherent stationary scans, "
            f"mean match {validation_score:.1f}, endpoint support "
            f"{validation_support:.1%}, scatter "
            f"{validation_centre_scatter * 100:.1f}cm/"
            f"{validation_heading_scatter:.1f}deg. Exploration enabled."
        )

    # IMU/map heading ledger.  The gyro cannot say WHERE the robot is, but
    # between strong fixes it says very accurately how much the heading has
    # CHANGED.  Every scan entering the permanent map is therefore PLACED at
    # the gyro-expected heading whenever the scan-match heading disagrees —
    # the map keeps growing, but a slightly rotated match can no longer slant
    # the board (field 2026-08-05: skewed corridor walls degraded every later
    # fit to score 7-8).  Accepted multi-view stationary consensuses re-sync
    # the ledger, so a verified LiDAR heading fix stays authoritative.
    imu_heading_ref: dict[str, float | int | None] = {
        "imu": None,
        "theta": None,
        "aligned_count": 0,
        # Same-signed refused-correction streak (drift detector).  A LiDAR
        # heading alias is geometry-dependent and sign-random; the gyro
        # reference drifting away from the map produces the OPPOSITE
        # signature — consecutive validated consensuses proposing the same
        # direction of correction (field 2026-08-05 night: +9.9..+12.9deg
        # refused eight times while the map curved and every batch died).
        "clamp_streak": 0,
        "clamp_sign": 0,
    }

    def _rezero_gyro_heading_reference(theta_deg: float, reason: str) -> None:
        """Make ``theta_deg`` the gyro ledger's truth at the current yaw.

        Every accepted absolute re-anchor that may rotate the pose MUST pass
        through here: an untouched ledger keeps enforcing the pre-correction
        offset at every later commit, which is exactly the split-heading state
        that curved the 2026-08-05 map.  A stale yaw sample cannot form a
        valid pair, so the ledger is dropped instead and re-seeds on the next
        fresh agreeing commit.
        """
        imu_heading_ref["clamp_streak"] = 0
        imu_heading_ref["clamp_sign"] = 0
        yaw_now = imu.deg_fresh(wait_up_to_s=0.3, max_age_s=0.4)
        if yaw_now is not None:
            imu_heading_ref["imu"] = float(yaw_now)
            imu_heading_ref["theta"] = float(theta_deg)
        else:
            imu_heading_ref["imu"] = None
            imu_heading_ref["theta"] = None
        print(f"[localize] gyro heading reference re-zeroed ({reason}).")

    # =======================================================================
    # POSE GRAPH + MANHATTAN AXIS COMPASS — the professional core.
    #
    # Stationary keyframe poses are graph VARIABLES connected by measured
    # constraints (odometry between stops, anisotropic scan-match fixes,
    # absolute axis headings from the building's walls); the occupancy grid
    # is a RENDERING of the optimized scan list, re-derived whenever poses
    # improve.  The anchor spin is the fixed gauge.  Loop closures are
    # deliberately absent for now: in a corridor of identical doorways a
    # false closure warps the whole map, so closures wait until they can be
    # multi-view verified against distinctive GOLD anchors.
    # =======================================================================
    building_axis: list[float | None] = [None]
    _axis_estimate = dominant_axis_deg(world_map.reference())
    if _axis_estimate is not None:
        building_axis[0] = float(_axis_estimate[0])
        print(
            f"[graph] building axis locked from the anchor map: "
            f"{building_axis[0]:.1f}deg (mod 90, wall mass {_axis_estimate[1]:.0%}); "
            "stationary headings snap to it within 7deg — the walls, not the "
            "gyro, are now the heading reference."
        )
    else:
        print(
            "[graph] anchor map has no dominant rectilinear axis; Manhattan "
            "compass stays silent and the gyro remains the heading referee."
        )

    pose_graph = PoseGraph2D()
    _anchor_node = pose_graph.add_node(
        float(cur_pose.x),
        float(cur_pose.y),
        float(cur_pose.theta_deg),
        fixed=True,  # the anchor is the gauge
    )
    graph_state: dict[str, object] = {
        "last_node": _anchor_node,
        "last_pose": cur_pose,
        "axis_snap_count": 0,
        # Scans below this index are anchor/gauge scans (or already attached);
        # everything committed after it rides on the NEXT registered node.
        "scans_watermark": len(world_map.scans),
        # Cumulative driven distance at each node — the loop-closure
        # guardrail's drift budget is proportional to this.
        "node_odo": {_anchor_node: 0.0},
    }

    def _axis_snap_pose(pose: Pose2D, local_xy: np.ndarray | None) -> Pose2D | None:
        """Manhattan compass: absolute heading from the building's walls.

        Returns the snapped pose when this scan confidently sees rectilinear
        structure within 7deg of the building axis; None when the compass has
        nothing trustworthy to say (sparse view, non-Manhattan geometry, or a
        disagreement too large to be drift).
        """
        if building_axis[0] is None or local_xy is None or len(local_xy) < 60:
            return None
        world = _transform_points(local_xy, pose)
        delta = axis_snap_delta_deg(world, float(building_axis[0]), tolerance_deg=7.0)
        if delta is None:
            return None
        snapped = _rotate_lidar_pose_about_robot_centre(
            pose, float(delta), lever_m, forward_offset
        )
        count = int(graph_state["axis_snap_count"]) + 1
        graph_state["axis_snap_count"] = count
        if abs(delta) >= 0.5 and (count <= 5 or count % 10 == 0):
            print(
                f"[graph] axis compass: heading snapped {float(delta):+.1f}deg to the "
                f"building axis (snap #{count})."
            )
        return snapped

    def _scan_match_position_info(
        world_pts: np.ndarray, mean_score: float
    ) -> np.ndarray:
        """2x2 world-frame position information for a stationary consensus.

        Anisotropic by observability: a wall constrains translation
        PERPENDICULAR to itself, so a corridor (walls parallel to axis0)
        yields strong lateral information and weak along-corridor
        information — the graph then lets odometry own the along-corridor
        coordinate instead of letting a parallel-wall alias fight it.
        """
        sigma_base = float(np.clip(0.9 / max(4.0, float(mean_score)), 0.05, 0.25))
        base_info = 1.0 / sigma_base**2
        axis = building_axis[0]
        masses = (
            wall_direction_masses(world_pts, float(axis)) if axis is not None else None
        )
        if masses is None:
            return np.eye(2) * base_info
        mass_axis0, mass_axis1 = masses
        # Walls parallel to axis1 constrain translation ALONG axis0.
        strength = np.array([max(0.05, mass_axis1), max(0.05, mass_axis0)])
        strength = strength / float(strength.max())
        rad = math.radians(float(axis))
        rot = np.array(
            [[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]]
        )
        return rot @ np.diag(strength * base_info) @ rot.T

    def _register_stationary_fix(
        predicted_pose: Pose2D,
        accepted_pose: Pose2D,
        local_xy: np.ndarray | None,
        mean_score: float,
        *,
        axis_locked: bool,
        odometry_broken: bool = False,
    ) -> None:
        """Register an accepted stop as a graph node; optimize; re-render.

        ``odometry_broken`` marks recovery fixes where the carried pose chain
        is known-bad (tracking was lost): the between factor is then wide
        open so the graph does not average a broken chain into good history.
        """
        nonlocal cur_pose
        last_node = int(graph_state["last_node"])  # type: ignore[arg-type]
        last_pose: Pose2D = graph_state["last_pose"]  # type: ignore[assignment]
        # Odometry between stops, expressed in the previous node's frame.
        d_world = np.array(
            [
                float(predicted_pose.x) - float(last_pose.x),
                float(predicted_pose.y) - float(last_pose.y),
            ]
        )
        rad = math.radians(float(last_pose.theta_deg))
        rot_t = np.array(
            [[math.cos(rad), math.sin(rad)], [-math.sin(rad), math.cos(rad)]]
        )
        d_local = rot_t @ d_world
        d_theta = (
            (float(predicted_pose.theta_deg) - float(last_pose.theta_deg) + 180.0)
            % 360.0
        ) - 180.0
        distance = float(np.hypot(*d_world))
        if odometry_broken:
            sigma_xy, sigma_theta = max(0.50, distance), 25.0
        else:
            sigma_xy = 0.05 + 0.12 * distance
            sigma_theta = 1.5 + 1.0 * distance
        node = pose_graph.add_node(
            float(accepted_pose.x),
            float(accepted_pose.y),
            float(accepted_pose.theta_deg),
        )
        pose_graph.add_between(
            last_node,
            node,
            float(d_local[0]),
            float(d_local[1]),
            d_theta,
            sigma_xy_m=sigma_xy,
            sigma_theta_deg=sigma_theta,
        )
        world_pts = (
            _transform_points(local_xy, accepted_pose)
            if local_xy is not None and len(local_xy)
            else np.zeros((0, 2))
        )
        pose_graph.add_pose_prior(
            node,
            float(accepted_pose.x),
            float(accepted_pose.y),
            float(accepted_pose.theta_deg),
            position_info=_scan_match_position_info(world_pts, mean_score),
            # The axis compass owns heading when it fired; otherwise the
            # scan-match consensus heading is used at moderate confidence.
            sigma_theta_deg=None if axis_locked else 3.0,
        )
        if axis_locked:
            pose_graph.add_heading_prior(
                node, float(accepted_pose.theta_deg), sigma_deg=1.0
            )
        node_odo: dict[int, float] = graph_state["node_odo"]  # type: ignore[assignment]
        node_odo[node] = node_odo.get(last_node, 0.0) + distance
        # ---- LOOP CLOSURE, odometry-gated -------------------------------
        # Geometry PROPOSES, odometry DISPOSES.  A candidate is an old node
        # (>=5 stops back — a revisit, not a continuation) that the current
        # graph already places within 0.9m.  The guardrail: the scan-match
        # correction implied by the closure must fit inside the accumulated
        # drift budget (15cm + 3% of distance driven between the visits).
        # A hallway that merely LOOKS like this one fails that budget — by
        # our own odometry we KNOW it is not the same place — and is
        # refused loudly instead of warping the map.
        if local_xy is not None and len(local_xy) >= 60:
            closure_candidate = None
            closure_separation = 0.90
            for cand in range(max(0, node - 4)):
                cand_x, cand_y, _cand_th = pose_graph.node_pose_deg(cand)
                separation = float(
                    np.hypot(
                        float(accepted_pose.x) - cand_x,
                        float(accepted_pose.y) - cand_y,
                    )
                )
                if separation < closure_separation:
                    closure_candidate, closure_separation = cand, separation
            if closure_candidate is not None:
                cand_sets = [
                    s.world_xy
                    for s in world_map.scans
                    if s.node_id == closure_candidate and len(s.world_xy)
                ]
                cand_pts = (
                    np.concatenate(cand_sets, axis=0)
                    if cand_sets
                    else np.zeros((0, 2))
                )
                driven = abs(
                    node_odo[node] - node_odo.get(closure_candidate, 0.0)
                )
                drift_gate = 0.15 + 0.03 * driven
                if len(cand_pts) >= 150:
                    solved_lc, score_lc, support_lc = _localize_against_points(
                        local_xy,
                        cand_pts,
                        accepted_pose,
                        args,
                        search_xy_m=0.45,
                        theta_window_deg=4.0,
                    )
                    innovation_lc = float(
                        np.hypot(
                            float(solved_lc.x) - float(accepted_pose.x),
                            float(solved_lc.y) - float(accepted_pose.y),
                        )
                    )
                    strong_lc = score_lc >= max(11.0, float(args.min_match_score))
                    if (
                        strong_lc
                        and float(support_lc) >= 0.55
                        and innovation_lc <= drift_gate
                    ):
                        cand_x, cand_y, cand_th = pose_graph.node_pose_deg(
                            closure_candidate
                        )
                        rad_c = math.radians(cand_th)
                        rot_c = np.array(
                            [
                                [math.cos(rad_c), math.sin(rad_c)],
                                [-math.sin(rad_c), math.cos(rad_c)],
                            ]
                        )
                        d_lc = rot_c @ np.array(
                            [
                                float(solved_lc.x) - cand_x,
                                float(solved_lc.y) - cand_y,
                            ]
                        )
                        pose_graph.add_between(
                            closure_candidate,
                            node,
                            float(d_lc[0]),
                            float(d_lc[1]),
                            (
                                (float(solved_lc.theta_deg) - cand_th + 180.0)
                                % 360.0
                            )
                            - 180.0,
                            sigma_xy_m=0.06,
                            sigma_theta_deg=1.5,
                        )
                        print(
                            f"[graph] loop closure: node {node} <-> node "
                            f"{closure_candidate} (separation "
                            f"{closure_separation * 100:.0f}cm, {driven:.1f}m "
                            f"driven between visits): match {score_lc:.1f}, "
                            f"support {float(support_lc):.0%}, innovation "
                            f"{innovation_lc * 100:.0f}cm within the "
                            f"{drift_gate * 100:.0f}cm odometry gate."
                        )
                    elif strong_lc and innovation_lc > drift_gate:
                        print(
                            "[graph] loop-closure candidate REFUSED by the "
                            f"odometry guardrail: looks like node "
                            f"{closure_candidate} (match {score_lc:.1f}) but "
                            f"wants a {innovation_lc * 100:.0f}cm correction vs "
                            f"the {drift_gate * 100:.0f}cm accumulated-drift "
                            f"budget ({driven:.1f}m driven) — a similar-looking "
                            "place, not the same place."
                        )
        # -----------------------------------------------------------------
        world_map.attach_recent_scans_to_node(
            node, accepted_pose, int(graph_state["scans_watermark"])  # type: ignore[arg-type]
        )
        graph_state["scans_watermark"] = len(world_map.scans)
        pose_graph.optimize()
        optimized = {
            k: pose_graph.node_pose_deg(k) for k in range(len(pose_graph.nodes))
        }
        moved = world_map.apply_node_poses(optimized)
        ox, oy, oth = optimized[node]
        refined = Pose2D(ox, oy, oth)
        refinement = float(np.hypot(ox - float(accepted_pose.x), oy - float(accepted_pose.y)))
        if refinement > 0.005 or moved > 0.01:
            print(
                f"[graph] node {node} registered (odo {distance:.2f}m): "
                f"{len(pose_graph.nodes)} nodes optimized, live pose refined "
                f"{refinement * 100:.0f}cm, largest retroactive map correction "
                f"{moved * 100:.0f}cm; grid re-rendered."
            )
        cur_pose = refined
        graph_state["last_node"] = node
        graph_state["last_pose"] = refined

    # Persistent-offset ledger (position twin of the heading drift escape).
    # A single stationary batch proposing a >cap correction is treated as a
    # possible alias and discarded.  But when TWO consecutive stationary
    # batches independently land on the SAME absolute pose (<=8cm/2deg apart)
    # with committable-grade score, that is corroboration, not aliasing —
    # refusing it made recoveries accept only PART of the offset (30 of
    # 46cm, field 2026-08-05 19:35), and snapshots painted at half-corrected
    # poses are exactly the doubled ghost walls seen that run.
    persistent_offset_state: dict[str, np.ndarray | float | None] = {
        "centre": None,
        "theta": None,
    }

    def _imu_map_heading_align(pose: Pose2D, local_xy: np.ndarray | None = None) -> Pose2D:
        # MANHATTAN COMPASS FIRST: when the scan confidently sees the
        # building's rectilinear structure, the walls supply an absolute
        # heading that cannot drift — strictly better than the gyro delta.
        # The gyro ledger is re-synced to the snapped result so the veto
        # machinery below stays consistent for the scans the compass cannot
        # judge (sparse views, non-Manhattan corners).
        snapped = _axis_snap_pose(pose, local_xy)
        if snapped is not None:
            yaw_now = imu.deg_fresh(wait_up_to_s=0.15, max_age_s=0.35)
            if yaw_now is not None:
                imu_heading_ref["imu"] = float(yaw_now)
                imu_heading_ref["theta"] = float(snapped.theta_deg)
            return snapped
        # FRESH yaw only: imu.deg() returns the last RECEIVED sample, which
        # lags real heading by seconds around pivots and stream hiccups.  One
        # stale pair poisons the ledger and then every alignment enforces the
        # stale offset (field 2026-08-05 16:26: persistent unanimous +8-12deg
        # LiDAR votes were discarded for minutes until an 18deg-window
        # recovery finally re-fixed heading).  No fresh yaw -> no alignment.
        yaw_now = imu.deg_fresh(wait_up_to_s=0.15, max_age_s=0.35)
        if yaw_now is None:
            return pose  # advisory sensor stale/absent: LiDAR owns placement
        if imu_heading_ref["imu"] is None:
            imu_heading_ref["imu"] = float(yaw_now)
            imu_heading_ref["theta"] = float(pose.theta_deg)
            return pose
        expected_theta = float(imu_heading_ref["theta"]) + (
            float(yaw_now) - float(imu_heading_ref["imu"])
        )
        disagreement = (
            (float(pose.theta_deg) - expected_theta + 180.0) % 360.0
        ) - 180.0
        if abs(disagreement) <= 0.6:
            # Agreeing STATIONARY commit: re-sync the reference pair.  Every
            # commit reaching this point is now a stop-scan-go multi-scan
            # batch representative (moving scans are refused upstream), so
            # re-syncing cannot walk with in-motion bias — while NOT
            # re-syncing let raw gyro drift age the reference for a whole
            # mission and rotate the frame into a curved corridor (field
            # 2026-08-05 night: zero alignments fired; the ledger pair was
            # mission-old).
            imu_heading_ref["imu"] = float(yaw_now)
            imu_heading_ref["theta"] = float(pose.theta_deg)
            return pose
        aligned = Pose2D(
            float(pose.x),
            float(pose.y),
            float(pose.theta_deg) - float(disagreement),
        )
        count = int(imu_heading_ref["aligned_count"]) + 1
        imu_heading_ref["aligned_count"] = count
        if count <= 3 or count % 10 == 0:
            print(
                f"[map] scan heading aligned to the gyro reference before commit: "
                f"scan-match {float(pose.theta_deg):+.1f}deg -> "
                f"{float(aligned.theta_deg):+.1f}deg "
                f"({disagreement:+.1f}deg slant prevented; alignment #{count})."
            )
        return aligned

    world_map.heading_align = _imu_map_heading_align

    def _gyro_clamped_consensus(
        pose: Pose2D, local_xy: np.ndarray | None = None
    ) -> Pose2D:
        """Clamp an ACCEPTED consensus heading to the gyro expectation.

        Heading truth must be ONE value shared by the live pose and every
        committed scan.  Field 2026-08-05: a stationary re-anchor accepted a
        7deg LiDAR heading 'correction' that the gyro contradicted; commits
        were gyro-aligned but the live pose kept the 7deg, so map and pose
        disagreed by ~6deg from then on and every later batch was discarded
        at exactly -6deg.  A stationary gyro cannot invent rotation, so a
        LiDAR heading delta beyond a small tolerance of the (continuously
        re-synced) gyro expectation is an alias: keep the LiDAR position,
        refuse the rotated heading.
        """
        # Manhattan compass first: an absolute wall-derived heading beats
        # both the gyro expectation and the scan-match theta.
        snapped = _axis_snap_pose(pose, local_xy)
        if snapped is not None:
            imu_heading_ref["clamp_streak"] = 0
            imu_heading_ref["clamp_sign"] = 0
            yaw_axis = imu.deg_fresh(wait_up_to_s=0.15, max_age_s=0.35)
            if yaw_axis is not None:
                imu_heading_ref["imu"] = float(yaw_axis)
                imu_heading_ref["theta"] = float(snapped.theta_deg)
            return snapped
        yaw_now = imu.deg_fresh(wait_up_to_s=0.15, max_age_s=0.35)
        if yaw_now is None:
            # LiDAR owns placement while the advisory sensor is stale — but
            # the accepted consensus may include a heading correction the old
            # ledger pair would later fight forever (a 12.1deg re-anchor slid
            # through exactly this hole on 2026-08-05 and split map heading
            # from gyro heading for the rest of the run).  Drop the pair; the
            # next fresh agreeing commit re-seeds it.
            imu_heading_ref["imu"] = None
            imu_heading_ref["theta"] = None
            imu_heading_ref["clamp_streak"] = 0
            imu_heading_ref["clamp_sign"] = 0
            return pose
        if imu_heading_ref["imu"] is None:
            imu_heading_ref["imu"] = float(yaw_now)
            imu_heading_ref["theta"] = float(pose.theta_deg)
            return pose
        expected_theta = float(imu_heading_ref["theta"]) + (
            float(yaw_now) - float(imu_heading_ref["imu"])
        )
        disagreement = (
            (float(pose.theta_deg) - expected_theta + 180.0) % 360.0
        ) - 180.0
        if abs(disagreement) <= 1.5:
            imu_heading_ref["imu"] = float(yaw_now)
            imu_heading_ref["theta"] = float(pose.theta_deg)
            imu_heading_ref["clamp_streak"] = 0
            imu_heading_ref["clamp_sign"] = 0
            return pose
        # Drift escape (hard directive #9: LiDAR owns mapping, the gyro is a
        # veto for single-event jumps).  Three consecutive validated
        # consensuses proposing a same-signed correction >= 2.0deg is the
        # gyro reference drifting away from the map, not three independent
        # aliases: re-zero the ledger to the LiDAR consensus instead of
        # clamping the LiDAR to a wrong reference.
        sign = 1 if disagreement > 0.0 else -1
        if abs(disagreement) >= 2.0 and sign == int(imu_heading_ref["clamp_sign"]):
            streak = int(imu_heading_ref["clamp_streak"]) + 1
        else:
            streak = 1 if abs(disagreement) >= 2.0 else 0
        imu_heading_ref["clamp_streak"] = streak
        imu_heading_ref["clamp_sign"] = sign if streak else 0
        if streak >= 3:
            print(
                "[localize] gyro heading reference drift detected: "
                f"{streak} consecutive validated consensuses proposed same-signed "
                f"corrections (latest {disagreement:+.1f}deg); accepting the LiDAR "
                "heading and re-zeroing the gyro reference to it."
            )
            imu_heading_ref["imu"] = float(yaw_now)
            imu_heading_ref["theta"] = float(pose.theta_deg)
            imu_heading_ref["clamp_streak"] = 0
            imu_heading_ref["clamp_sign"] = 0
            return pose
        clamped = _rotate_lidar_pose_about_robot_centre(
            pose,
            -float(disagreement),
            lever_m,
            forward_offset,
        )
        print(
            "[localize] consensus heading clamped to the gyro reference: "
            f"{float(pose.theta_deg):+.1f}deg -> {float(clamped.theta_deg):+.1f}deg "
            f"({disagreement:+.1f}deg LiDAR-heading alias refused; position kept)."
        )
        imu_heading_ref["imu"] = float(yaw_now)
        imu_heading_ref["theta"] = float(clamped.theta_deg)
        return clamped

    completed_transitions: list[tuple[np.ndarray, np.ndarray]] = []
    transition_pose_rejections = [0]
    room_entry_keyframes_remaining = [0]

    # =======================================================================
    # EXPLORE PHASES 2–5 SUPPORT
    #
    # The definitions below implement repeated frontier selection, navigation,
    # snapshot integration, doorway handoff, and recovery. The numbered mission
    # loop that calls them appears after _drive_leg().
    # =======================================================================
    def _record_completed_transition(
        anchor_xy: np.ndarray,
        outward_xy: np.ndarray,
        source: str,
    ) -> bool:
        """Commit one directed doorway edge and start a rolling local submap."""
        anchor = np.asarray(anchor_xy, dtype=np.float64).copy()
        outward = np.asarray(outward_xy, dtype=np.float64).copy()
        norm = float(np.hypot(*outward))
        if norm < 1e-6:
            return False
        outward /= norm
        duplicate = any(
            float(np.hypot(*(anchor - old_anchor))) < 0.75 and float(np.dot(outward, old_outward)) > 0.70
            for old_anchor, old_outward in completed_transitions
        )
        if not duplicate:
            completed_transitions.append((anchor, outward))
        room_entry_keyframes_remaining[0] = max(
            room_entry_keyframes_remaining[0],
            int(args.room_entry_keyframes),
        )
        print(
            f"[explore] {source}: doorway edge committed at "
            f"({anchor[0]:+.2f}, {anchor[1]:+.2f}); old-room targets and "
            "backward route choices are now discouraged while a rolling "
            f"local submap is built for {room_entry_keyframes_remaining[0]} leg(s)."
        )
        return not duplicate

    def _robot_centre(pose: Pose2D) -> np.ndarray:
        return _robot_centre_from_lidar_pose(pose, lever_m, forward_offset)

    def _pose_stays_beyond_completed_doorways(pose: Pose2D) -> bool:
        behind = _behind_completed_transition(
            _robot_centre(pose),
            completed_transitions,
            float(args.doorway_ratchet_slack_m),
        )
        if behind:
            transition_pose_rejections[0] += 1
            if transition_pose_rejections[0] <= 3 or transition_pose_rejections[0] % 10 == 0:
                print(
                    "[localize] doorway-ratchet warning: pose is behind a "
                    "completed doorway plane, but LiDAR localization remains "
                    f"eligible (warning #{transition_pose_rejections[0]})."
                )
        # A completed doorway is useful navigation memory, not proof of pose.
        # Treat it as a planning/target-selection bias only.  Hard-rejecting
        # localization here caused the robot to discard valid 180-degree
        # recovery scans that were intentionally looking back at known doorway
        # geometry.
        return True

    local_odometry = RollingLocalSubmap(max_scans=10, max_points=7000)
    if verified_live_anchor_scan[0] is not None:
        local_odometry.reset(verified_live_anchor_scan[0], cur_pose)
        print(
            "[localize] rolling local odometry seeded from the fresh stationary "
            "scan that verified the robot's current anchor-map pose."
        )
    else:
        # This is reachable only for legacy/bootstrap paths whose current pose
        # was independently established without retaining the source scan.
        local_odometry.seed_from_world_map(world_map)
    last_tracking_frame_id: list[int | None] = [None]
    last_tracking_score = [float(args.min_match_score)]
    global_tracking_cycle = [0]
    # A pair of parallel hallway walls constrains lateral position and heading,
    # but not progress along the hallway.  Every long, straight passable
    # transit—not merely a post-turn recovery—therefore carries its pose with
    # IMU + local scan odometry.  Global wall matching is still sampled to
    # notice the next distinctive landmark, but it may not bend the carried
    # trajectory sideways or declare the known parallel-wall view lost.
    featureless_corridor_state: dict[str, object] = {
        "active": False,
        "origin_xy": None,
        "heading_deg": None,
        "max_distance_m": 4.0,
        "skip_logged": False,
        # A corridor is deliberately carried by local LiDAR/IMU odometry, but
        # that must not mean that the map is left behind.  These fields request
        # sparse *stationary* provisional snapshots at the carried pose.  They
        # are never used as GOLD localization geometry until independently
        # confirmed from another view.
        "checkpoint_due": False,
        "checkpoint_reason": "",
        "last_checkpoint_distance_m": 0.0,
        "ignored_cross_track_updates": 0,
        # Set only by the 180-degree landmark recovery.  It means the robot
        # has just re-anchored by looking backward and must now continue
        # forward through the original parallel hallway by carried odometry,
        # not fall back into generic frontier target selection.
        "pending_carry_after_reanchor": False,
        # Consecutive carry legs that neither translated the base nor grew
        # the map.  Two in a row means the beeline has reached whatever ends
        # the corridor (or cannot progress); the carry must terminate rather
        # than replay the same leg.
        "carry_stalled_legs": 0,
    }
    # A local rolling submap is excellent at estimating short-range motion but
    # cannot, by itself, prove that it has not slowly slid along a repeated
    # hallway wall.  Two consecutive disagreements with the bounded permanent
    # map solve force a stopped full-view verification before more map geometry
    # or driving is allowed.
    local_global_disagreement_streak = [0]
    last_tracking_health: dict[str, float | bool] = {
        "fresh": False,
        "accepted": False,
        "local_valid": False,
        "local_support": 0.0,
        "local_innovation_m": math.inf,
        "global_valid": False,
        "local_global_agreement_m": math.inf,
    }
    # A temporary planner obstacle is meaningful only when the transform from
    # LiDAR to the permanent map is trusted.  Keep this confidence separate
    # from collision safety: raw body-frame collision checks remain active even
    # while world-frame obstacle insertion is quarantined.
    obstacle_pose_trust: dict[str, int | bool | str] = {
        "trusted": True,
        "corroborated_fixes": 3,
        "reason": "anchor pose",
        "warning_emitted": False,
    }

    def _quarantine_world_obstacles(reason: str) -> None:
        obstacle_pose_trust.update(
            trusted=False,
            corroborated_fixes=0,
            reason=str(reason),
            warning_emitted=False,
        )

    def _trust_world_obstacles(reason: str) -> None:
        obstacle_pose_trust.update(
            trusted=True,
            corroborated_fixes=3,
            reason=str(reason),
            warning_emitted=False,
        )

    def _update_world_obstacle_pose_trust() -> None:
        health = last_tracking_health
        if not bool(health["fresh"]):
            return
        if _obstacle_pose_fix_is_corroborated(
            fresh_scan=bool(health["fresh"]),
            pose_accepted=bool(health["accepted"]),
            local_match_valid=bool(health["local_valid"]),
            global_match_valid=bool(health["global_valid"]),
            local_innovation_m=float(health["local_innovation_m"]),
            local_global_agreement_m=float(health["local_global_agreement_m"]),
        ):
            fixes = int(obstacle_pose_trust["corroborated_fixes"]) + 1
            obstacle_pose_trust["corroborated_fixes"] = fixes
            if fixes >= 3 and not bool(obstacle_pose_trust["trusted"]):
                _trust_world_obstacles("three consecutive local/global pose agreements")
                print(
                    "[localize] pose is re-corroborated; world-frame dynamic obstacle insertion re-enabled."
                )
        elif bool(health["global_valid"]):
            # A fresh permanent-map solve that disagrees with local continuity
            # is direct evidence that the world transform is ambiguous.
            _quarantine_world_obstacles("local odometry and permanent-map pose disagree")

    def _reset_local_odometry(local_xy: np.ndarray, pose: Pose2D) -> None:
        """Re-anchor the rolling submap after a verified stationary solution."""
        local_odometry.reset(local_xy, pose)
        last_tracking_frame_id[0] = None
        last_tracking_score[0] = float(args.min_match_score)
        global_tracking_cycle[0] = 0
        # A stationary, permanent-map-confirmed pose is a new baseline.  Do
        # not carry a pre-stop local/global disagreement into the next drive
        # leg, or the first otherwise-good rolling scan can immediately trip a
        # false "lost" state.
        local_global_disagreement_streak[0] = 0

    def _track(
        prev_imu_deg: float | None,
        *,
        integrate: bool = True,
        search_xy_m: float | None = None,
        global_match_every: int = 2,
    ) -> tuple[float, np.ndarray, float | None]:
        """Continuous local-submap odometry plus guarded global-map correction.

        Every fresh revolution first matches the rolling nearby scan submap.
        Global-map matching may gently correct that trajectory when both agree,
        but a rejected permanent-map hypothesis can no longer freeze translation.
        ``integrate`` controls only permanent occupancy writes.

        While DRIVING we localize with integrate=False: a revolution captured in
        motion is smeared (the LiDAR sweeps as the base moves), so folding it into
        the map smudges walls and fills in doorways (they then read as walls). We
        only add scans when stopped (at a viewpoint), where they are clean. Returns
        (score, local scan, IMU pose anchor) so the caller can reuse the scan
        and preserve all yaw accumulated while the matcher was running."""
        nonlocal cur_pose
        empty = np.zeros((0, 2), dtype=np.float32)
        last_tracking_health.update(
            fresh=False,
            accepted=False,
            local_valid=False,
            local_support=0.0,
            local_innovation_m=math.inf,
            global_valid=False,
            local_global_agreement_m=math.inf,
        )
        frame_id, frame = feed.latest()
        if frame is None:
            return 0.0, empty, prev_imu_deg
        if last_tracking_frame_id[0] == int(frame_id):
            # Never reinterpret the same physical revolution as another motion
            # measurement merely because the navigation loop runs faster than
            # the LiDAR. Odometry propagation remains active in the caller.
            return last_tracking_score[0], empty, prev_imu_deg
        last_tracking_frame_id[0] = int(frame_id)
        global_tracking_cycle[0] += 1
        last_tracking_health["fresh"] = True
        local = _scan_local(frame, args)
        if len(local) < 12:
            return 0.0, local, prev_imu_deg
        # Associate heading with the revolution, not with when this Python
        # thread happened to consume it. Network and matcher latency otherwise
        # turn directly into a heading error while the robot is moving.
        imu_scan = _imu_yaw_for_scan(imu, frame)
        gyro_delta = (
            float(imu_scan) - float(prev_imu_deg)
            if (imu_scan is not None and prev_imu_deg is not None)
            else 0.0
        )
        theta_seed = float(cur_pose.theta_deg) + gyro_delta
        centre = _robot_centre(cur_pose)
        seed = _lidar_pose_from_robot_centre(centre, theta_seed, lever_m, forward_offset)
        corridor_active = bool(featureless_corridor_state["active"])
        corridor_heading = featureless_corridor_state["heading_deg"]
        if corridor_active and corridor_heading is not None:
            heading_departure = abs(
                (float(theta_seed) - float(corridor_heading) + 180.0) % 360.0 - 180.0
            )
            if heading_departure > 12.0:
                featureless_corridor_state["active"] = False
                featureless_corridor_state["pending_carry_after_reanchor"] = False
                corridor_active = False
                print(
                    "[localize] featureless-corridor traverse ended because the "
                    f"route changed heading by {heading_departure:.1f}deg."
                )
        window = float(search_xy_m) if search_xy_m is not None else float(args.track_search_xy_m)
        # THRESHOLD-SAFE matching: at a doorway much of the scan looks into
        # unmapped space and matches nothing, cratering the score even when the
        # pose is perfect (field: 15.5 -> 4.3 arriving centred at the exit ->
        # "tracking lost" -> the robot turned away from the opening it came for).
        # Solve on the map-SUPPORTED subset; the unsupported remainder is new
        # territory — integrated below so the map grows through the opening.
        sup = _match_support_mask(_transform_points(local, seed))
        # Whole-view scoring uses every currently visible wall when enough of
        # it overlaps the permanent map.  This gives distant geometry a vote
        # in the bounded local refit; a sparse doorway still uses supported
        # rays so genuinely new space is not penalized as a mismatch.
        support_at_seed = float(np.mean(sup)) if len(sup) else 0.0
        match_local = local if support_at_seed >= 0.35 else (local[sup] if int(sup.sum()) >= 30 else local)
        # Local odometry runs on every fresh revolution. Global alignment is a
        # lower-rate correction, as in a conventional SLAM front/back end; it
        # must not double matcher latency on every control iteration. During a
        # deliberately carried parallel-corridor transit it is disabled: the
        # current view is known to be ambiguous along the hallway and a global
        # query can only create a self-triggered checkpoint at the start.
        match_period = max(1, int(global_match_every))
        run_global_match = (
            not corridor_active and global_tracking_cycle[0] % match_period == 1
        )
        if run_global_match:
            solved_global, global_score = _localize(
                match_local,
                world_map,
                seed,
                args,
                window,
                float(args.track_theta_window_deg),
            )
            solved_global = _pose_with_imu_heading(solved_global, theta_seed, lever_m, forward_offset)
            global_support = (
                float(np.mean(_match_support_mask(_transform_points(local, solved_global))))
                if len(local)
                else 0.0
            )
        else:
            solved_global = seed
            global_score = 0.0
            global_support = 0.0

        local_reference = local_odometry.reference()
        solved_local, local_score, local_support = _localize_against_points(
            local,
            local_reference,
            seed,
            args,
            window,
            min(4.0, float(args.track_theta_window_deg)),
        )
        solved_local = _pose_with_imu_heading(solved_local, theta_seed, lever_m, forward_offset)
        local_innovation = float(np.hypot(*(_robot_centre(solved_local) - _robot_centre(seed))))
        local_valid = (
            local_score >= max(4.0, float(args.min_match_score) - 2.0)
            and local_support >= 0.18
            and local_innovation <= window + 0.08
        )
        global_valid = global_score >= float(args.min_match_score) and global_support >= max(
            0.12, float(args.viewpoint_min_known_ratio) * 0.75
        )
        # A high-scoring repeated wall can still produce a globally plausible
        # alias several metres from the continuously propagated pose.  Never
        # let that hypothesis teleport the robot or get written into the map:
        # the true displacement since the last LiDAR/IMU seed is bounded by the
        # tracking window.  This is the root cause of the radial/ghosted maps
        # seen after entering the hallway.
        global_correction_m = float(
            np.hypot(*(_robot_centre(solved_global) - _robot_centre(seed)))
        )
        global_pose_bounded = global_valid and global_correction_m <= max(0.65, window + 0.15)
        if global_valid and not global_pose_bounded:
            print(
                "[localize] rejected global alias during motion: "
                f"correction {global_correction_m:.2f}m exceeds the "
                f"{max(0.65, window + 0.15):.2f}m continuity bound; "
                "local/IMU pose remains authoritative."
            )
            global_valid = False
        local_global_agreement = (
            float(np.hypot(*(_robot_centre(solved_global) - _robot_centre(solved_local))))
            if local_valid and global_valid
            else math.inf
        )
        last_tracking_health.update(
            local_valid=bool(local_valid),
            local_support=float(local_support),
            local_innovation_m=float(local_innovation),
            global_valid=bool(global_valid),
            local_global_agreement_m=float(local_global_agreement),
        )

        global_disagrees_with_local = (
            local_valid
            and global_valid
            and local_global_agreement > 0.12
        )
        if global_disagrees_with_local and not corridor_active:
            local_global_disagreement_streak[0] += 1
        elif local_valid and run_global_match:
            local_global_disagreement_streak[0] = 0

        candidate: Pose2D | None = None
        if local_valid:
            if corridor_active:
                # A pair of parallel walls is an excellent measurement of
                # heading and cross-track error, but it contains almost no
                # reliable information about progress *along* the hallway.
                # Keep the already anchored cross-track coordinate instead of
                # letting an aliased scan-to-scan fit bend the world trail.
                # The local registration is allowed to make only a small
                # along-corridor correction; IMU/surface propagation remains
                # responsible for the travelled distance between checkpoints.
                locked_theta = float(corridor_heading if corridor_heading is not None else theta_seed)
                axis_rad = math.radians(locked_theta + float(forward_offset))
                axis = np.array([math.cos(axis_rad), math.sin(axis_rad)], dtype=np.float64)
                local_delta = _robot_centre(solved_local) - centre
                along = float(np.dot(local_delta, axis))
                cross = float(local_delta[0] * -axis[1] + local_delta[1] * axis[0])
                # Registration of repeated hallway walls routinely aliases by
                # tens of centimetres, so lateral updates stay tightly
                # bounded.  They can no longer be discarded outright: the
                # wall-centering steer commands REAL lateral motion inside
                # corridors, and zeroing it painted every checkpoint wall
                # 9-15cm off — the splotchy doubled corridor of 2026-08-05.
                # A ±6cm-per-update trim tracks genuine centering drift
                # across updates while a one-off alias contributes at most
                # 6cm that later trims correct.
                along = float(np.clip(along, -0.10, 0.10))
                cross_trim = float(np.clip(cross, -0.06, 0.06))
                cross_axis = np.array([-axis[1], axis[0]], dtype=np.float64)
                candidate = _lidar_pose_from_robot_centre(
                    centre + along * axis + cross_trim * cross_axis,
                    theta_seed,
                    lever_m,
                    forward_offset,
                )
                if abs(cross) >= 0.08:
                    ignored = int(featureless_corridor_state["ignored_cross_track_updates"]) + 1
                    featureless_corridor_state["ignored_cross_track_updates"] = ignored
                    if ignored <= 3 or ignored % 10 == 0:
                        print(
                            "[localize] corridor pose carry trimmed "
                            f"{cross_trim * 100:+.0f}cm of a {cross * 100:+.0f}cm lateral "
                            f"offset and {along * 100:+.0f}cm along-corridor; the "
                            "remainder is treated as parallel-wall alias."
                        )
            else:
                candidate = solved_local
            if global_valid and not corridor_active:
                local_centre = _robot_centre(solved_local)
                global_centre = _robot_centre(solved_global)
                if float(np.hypot(*(global_centre - local_centre))) <= 0.12:
                    # The local trajectory owns continuity; a corroborating
                    # global solve removes slow drift without allowing a
                    # repeated wall to teleport the robot.
                    fused_centre = 0.80 * local_centre + 0.20 * global_centre
                    candidate = _lidar_pose_from_robot_centre(
                        fused_centre,
                        theta_seed,
                        lever_m,
                        forward_offset,
                    )
                elif not corridor_active and local_global_disagreement_streak[0] >= 2:
                    # Do not keep integrating a self-consistent local scan
                    # chain after the current full map view says it is no
                    # longer at the same place.  The caller stops and runs the
                    # multi-scan stationary re-fit, which has enough evidence
                    # to choose the correct small translation instead of
                    # letting one local alias paint a parallel hallway.
                    print(
                        "[localize] local rolling odometry disagreed with the "
                        f"bounded full-map LiDAR fit by {local_global_agreement * 100:.0f}cm "
                        "for two fresh scans; stopping for a stationary full-view re-fit."
                    )
                    candidate = None
        elif global_valid and not corridor_active:
            candidate = solved_global

        accepted_tracking = candidate is not None and _pose_stays_beyond_completed_doorways(candidate)
        score = max(float(global_score), float(local_score))
        if candidate is None and global_disagrees_with_local:
            score = min(score, float(args.min_match_score) - 0.1)
        if not accepted_tracking and not local_valid and not global_pose_bounded:
            # Do not report the rejected global score as healthy tracking.  The
            # caller must enter its bounded stationary recovery rather than
            # continuing to drive while believing an alias is valid.
            score = min(score, float(args.min_match_score) - 0.1)
        if accepted_tracking:
            cur_pose = candidate
            local_odometry.add(local, cur_pose)
            last_tracking_health["accepted"] = True
            _learn_surface_motion_from_lidar(cur_pose)
            _record_surface_region_at_trusted_pose(cur_pose)
            if corridor_active:
                corridor_origin = featureless_corridor_state["origin_xy"]
                if corridor_origin is not None:
                    travelled = float(
                        np.hypot(*(_robot_centre(cur_pose) - np.asarray(corridor_origin, dtype=np.float64)))
                    )
                    # Map the corridor from the carried pose rather than
                    # waiting for an ambiguous parallel-wall scan to pass a
                    # global-map transaction. A fresh stopped snapshot every
                    # ~75cm bounds length error. Do not use an immediately
                    # coherent global match here: at departure it is merely
                    # the unchanged start view and used to create an endless
                    # stop/re-aim/restart loop at 0.00m.
                    if not bool(featureless_corridor_state["checkpoint_due"]):
                        last_checkpoint = float(
                            featureless_corridor_state["last_checkpoint_distance_m"]
                        )
                        if travelled - last_checkpoint >= 0.75:
                            featureless_corridor_state["checkpoint_due"] = True
                            featureless_corridor_state["checkpoint_reason"] = "odometry interval"
                        elif travelled >= float(featureless_corridor_state["max_distance_m"]):
                            featureless_corridor_state["checkpoint_due"] = True
                            featureless_corridor_state["checkpoint_reason"] = "bounded corridor endpoint"
            # Local scan odometry has different score statistics from matching
            # an accumulated global map. Report it as healthy once its overlap
            # and bounded innovation tests pass so the caller does not invoke a
            # false global relocalization sequence.
            score = max(score, float(args.min_match_score))
            if integrate and global_valid and global_score >= float(args.integrate_min_score):
                world_map.add(local, cur_pose)
        else:
            cur_pose = seed

        if (
            integrate
            and global_valid
            and not accepted_tracking
            and _pose_stays_beyond_completed_doorways(solved_global)
        ):
            # Retain the former stationary behavior only when the global solve
            # itself is valid and the local tracker was not initialized yet.
            cur_pose = solved_global
            if global_score >= float(args.integrate_min_score):
                world_map.add(local, solved_global)
            local_odometry.add(local, cur_pose)
            _learn_surface_motion_from_lidar(cur_pose)
            _record_surface_region_at_trusted_pose(cur_pose)
            score = max(score, float(args.min_match_score))

        if False:  # Kept structurally unreachable for a readable diff boundary.
            # Legacy global-only tracking lived here. It intentionally remains
            # disabled: rejected global matches must never freeze local motion.
            solved = solved_global
            if integrate:
                if _pose_stays_beyond_completed_doorways(solved):
                    cur_pose = solved  # stationary: full re-anchor (clean scan)
                    if score >= float(args.integrate_min_score):
                        world_map.add(local, solved)
                else:
                    cur_pose = seed
                    score = 0.0
            else:
                # IN MOTION: match owns POSITION, gyro owns ROTATION. A moving
                # scan is smeared, so its solved theta wobbles a few degrees per
                # match — feeding that into the pose made the heading target
                # noisy and the servo faithfully chased the noise (the residual
                # squiggle). The gyro's relative yaw is ~0.5deg-accurate over a
                # whole leg, so keep the gyro-propagated theta and re-anchor
                # rotation only from clean stationary scans at viewpoints.
                candidate = Pose2D(
                    x=float(solved.x),
                    y=float(solved.y),
                    theta_deg=theta_seed,
                )
                if _pose_stays_beyond_completed_doorways(candidate):
                    cur_pose = candidate
                else:
                    cur_pose = seed
                    score = 0.0
        else:
            pass

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
        last_tracking_score[0] = float(score)
        return score, local, pose_imu_anchor

    def _integrate_at_viewpoint(
        *,
        require_known_overlap: bool = False,
        scan_count: int = 3,
    ) -> int:
        """Stop, settle, and map the new area with CLEAN stationary scans —
        committed as a BATCH only if the batch is self-consistent.

        The robot is standing still, so accepted scans must belong to one
        compact pose mode. A robust consensus removes isolated aliased solves;
        equally supported disjoint modes remain ambiguous and are discarded.
        A coherent batch may recover the pose, but only its two strongest
        representative revolutions are promoted to GOLD. This keeps one stopped
        pose from dominating the reference map or amplifying tiny residuals."""
        nonlocal cur_pose
        integration_started = time.monotonic()
        requested_scans = max(3, int(scan_count))
        minimum_inliers = 3 if requested_scans <= 3 else 4
        vp_pose_ok_last[0] = False
        controller.halt()
        time.sleep(float(args.viewpoint_settle_s))
        known_before = world_map.grid.occupied() | world_map.grid.free()
        origin_before = world_map.grid.origin.copy()
        # A zero command does not guarantee that the physical base has stopped.
        # The 27--37cm scatter in the failed run was motion being treated as a
        # stationary LiDAR batch. Require two small scan-to-scan registrations
        # before any frame is permitted to affect pose or map.
        settle_after = feed.latest()[0]
        settle_previous: np.ndarray | None = None
        settle_previous_yaw: float | None = None
        stable_pairs = 0
        last_settle_motion = math.inf
        last_settle_yaw = math.inf
        settle_deadline = time.monotonic() + max(1.20, float(args.viewpoint_settle_s) + 0.80)
        while time.monotonic() < settle_deadline and stable_pairs < 2:
            settle_id, settle_frame = feed.wait_for_frame_after(
                after_frame_id=settle_after,
                timeout_s=0.55,
                min_frame_advances=1,
            )
            if settle_frame is None:
                continue
            settle_after = int(settle_id)
            settle_local = _scan_local(settle_frame, args)
            settle_yaw = _imu_yaw_for_scan(imu, settle_frame)
            if settle_previous is not None and len(settle_local) >= 30 and len(settle_previous) >= 30:
                relative, settle_score, settle_support = _localize_against_points(
                    settle_local,
                    settle_previous,
                    Pose2D(0.0, 0.0, 0.0),
                    args,
                    search_xy_m=0.12,
                    theta_window_deg=2.0,
                )
                last_settle_motion = float(math.hypot(float(relative.x), float(relative.y)))
                last_settle_yaw = (
                    abs(float(settle_yaw) - float(settle_previous_yaw))
                    if settle_yaw is not None and settle_previous_yaw is not None
                    else 0.0
                )
                stable = (
                    settle_score >= max(4.0, float(args.min_match_score) - 2.0)
                    and settle_support >= 0.35
                    and last_settle_motion <= 0.08
                    and last_settle_yaw <= 1.5
                )
                stable_pairs = stable_pairs + 1 if stable else 0
            settle_previous = settle_local
            settle_previous_yaw = settle_yaw
        if stable_pairs < 2:
            print(
                "[explore] stationary checkpoint deferred: LiDAR still reports "
                f"{last_settle_motion * 100:.0f}cm/{last_settle_yaw:.1f}deg "
                "between consecutive frames after halt; no snapshot was taken."
            )
            return 0
        after = int(settle_after)
        batch: list[tuple[np.ndarray, Pose2D, float, float]] = []
        inlier_indices: list[int] = []
        last_score = 0.0
        batch_frames = 0
        batch_score_rejects = 0
        batch_doorway_rejects = 0
        stationary_pose_reference = cur_pose
        stationary_yaw_reference = imu.deg()
        for _ in range(requested_scans):
            fid, frame = feed.wait_for_frame_after(after_frame_id=after, timeout_s=1.0, min_frame_advances=1)
            if frame is None:
                continue
            after = int(fid)
            local = _scan_local(frame, args)
            if len(local) < 12:
                continue
            batch_frames += 1
            scan_yaw = _imu_yaw_for_scan(imu, frame)
            theta_seed = float(stationary_pose_reference.theta_deg)
            if stationary_yaw_reference is not None and scan_yaw is not None:
                theta_seed += float(scan_yaw) - float(stationary_yaw_reference)
            scan_seed = _lidar_pose_from_robot_centre(
                _robot_centre(stationary_pose_reference),
                theta_seed,
                lever_m,
                forward_offset,
            )
            sup_v = _match_support_mask(_transform_points(local, scan_seed))
            if require_known_overlap:
                min_known = max(
                    30,
                    int(math.ceil(float(args.frontier_handoff_min_known_ratio) * len(local))),
                )
                if int(sup_v.sum()) < min_known:
                    continue
            # A stationary confirmation is where the robot must actively
            # challenge its predicted pose with everything it can currently
            # see.  When at least 35% of the present view is already explained
            # by the map, include the full near-and-distant scan in the solve.
            # This lets a doorway, pillar, or far wall disprove a wrong local
            # track.  Retain the supported subset only at genuinely new map
            # edges where the unknown room would otherwise dominate scoring.
            support_at_seed = float(np.mean(sup_v)) if len(sup_v) else 0.0
            match_v = local if support_at_seed >= 0.35 else (local[sup_v] if int(sup_v.sum()) >= 30 else local)
            solved, last_score = _localize(
                match_v,
                world_map,
                scan_seed,
                args,
                max(0.40, float(args.track_search_xy_m)),
                3.0,
                coarse_angle_step_deg=1.0,
                fine_angle_step_deg=0.35,
                fine_theta_half_window_deg=1.5,
            )
            # During motion the IMU owns heading.  At a *stopped* checkpoint,
            # however, three complete, high-support LiDAR views are also a
            # legitimate way to detect a small IMU/map yaw offset.  Probe a
            # wider local heading window and retain it only when the full
            # present view strongly and repeatedly improves the fit.  This
            # prevents the former failure where a clear 12deg LiDAR correction
            # was overwritten by the IMU seed, then rejected as "lost".
            imu_locked = _pose_with_imu_heading(
                solved,
                theta_seed,
                lever_m,
                forward_offset,
            )
            heading_delta = (
                (float(solved.theta_deg) - float(theta_seed) + 180.0) % 360.0 - 180.0
            )
            use_lidar_heading = False
            if support_at_seed >= 0.35:
                wider, wider_score = _localize(
                    local,
                    world_map,
                    scan_seed,
                    args,
                    max(0.40, float(args.track_search_xy_m)),
                    15.0,
                    coarse_angle_step_deg=1.0,
                    fine_angle_step_deg=0.35,
                    fine_theta_half_window_deg=2.0,
                )
                wider_support = float(
                    np.mean(_match_support_mask(_transform_points(local, wider)))
                )
                wider_delta = (
                    (float(wider.theta_deg) - float(theta_seed) + 180.0) % 360.0 - 180.0
                )
                if (
                    wider_score >= last_score + 0.5
                    and wider_support >= 0.85
                    and 5.0 <= abs(wider_delta) <= 15.0
                ):
                    solved = wider
                    last_score = float(wider_score)
                    heading_delta = float(wider_delta)
                    use_lidar_heading = True
            if not use_lidar_heading:
                solved = imu_locked
            if last_score < float(args.min_match_score):
                batch_score_rejects += 1
                continue
            if not _pose_stays_beyond_completed_doorways(solved):
                batch_doorway_rejects += 1
                continue
            solved_support = _match_support_mask(_transform_points(local, solved))
            support_ratio = float(np.mean(solved_support)) if len(local) else 0.0
            batch.append((local, solved, last_score, support_ratio))
        added = 0
        if len(batch) >= minimum_inliers:
            consensus_pose, inlier_indices, scatter, th_scatter = _stationary_pose_inlier_consensus(
                [pose for _local, pose, _score, _support in batch],
                float(cur_pose.theta_deg),
                lever_m,
                forward_offset,
                float(args.viewpoint_max_position_scatter_m),
                float(args.viewpoint_max_heading_scatter_deg),
                min_inliers=minimum_inliers,
                reference_centre_xy=_robot_centre(cur_pose),
            )
            mean_sc = float(np.mean([batch[idx][2] for idx in inlier_indices])) if inlier_indices else 0.0
            mean_support = (
                float(np.mean([batch[idx][3] for idx in inlier_indices])) if inlier_indices else 0.0
            )
            prior_centre = _robot_centre(cur_pose)
            consensus_centre = _robot_centre(consensus_pose)
            predicted_pose_snapshot = cur_pose
            representative_local = (
                batch[max(inlier_indices, key=lambda i: float(batch[i][2]))][0]
                if inlier_indices
                else None
            )
            position_correction = float(np.hypot(*(consensus_centre - prior_centre)))
            heading_correction = abs(
                (float(consensus_pose.theta_deg) - float(cur_pose.theta_deg) + 180.0) % 360.0 - 180.0
            )
            match_commit_threshold = _keyframe_match_commit_threshold(
                float(args.integrate_min_score),
                len(inlier_indices),
                len(batch),
                mean_support,
                sequential_overlap_keyframe=bool(require_known_overlap),
            )
            heading_commit_limit = _keyframe_heading_correction_limit_deg(
                float(args.viewpoint_max_pose_correction_deg),
                len(inlier_indices),
                len(batch),
                mean_sc,
                float(args.integrate_min_score),
                mean_support,
            )
            committable = _keyframe_batch_is_committable(
                len(inlier_indices),
                mean_sc,
                match_commit_threshold,
                mean_support,
                float(args.viewpoint_min_known_ratio),
                position_correction,
                float(args.viewpoint_max_pose_correction_m),
                heading_correction,
                heading_commit_limit,
                minimum_inliers,
            )
            pose_recoverable = _stationary_pose_is_recoverable(
                len(inlier_indices),
                mean_sc,
                max(8.0, float(args.min_match_score)),
                mean_support,
                max(0.15, float(args.viewpoint_min_known_ratio)),
                position_correction,
                # This is an ordinary local viewpoint update, not a kidnapped-
                # robot/global relocalization event. The former use of the
                # largest configured radius (normally 60 cm) let repeated-wall
                # aliases teleport the navigation pose by 36-47 cm even though
                # the irreversible map transaction was correctly rejected.
                # Large corrections belong exclusively to
                # _continuity_relocalize_stationary(), which collects a fresh
                # multi-scan vote in a separately managed recovery state.
                float(args.relocalization_max_pose_correction_m),
                heading_correction,
                float(args.relocalization_max_pose_correction_deg),
                minimum_inliers,
            )
            # This is the exact doorway-slip case: several fresh stationary
            # LiDAR scans agree tightly on one nearby pose and the *full live
            # view* has strong permanent-map support, but the correction is
            # larger than the deliberately conservative 15cm routine-update
            # cap.  It is not safe to write geometry from that result, but it
            # is less safe to ignore it and continue a local-only drive.
            # Accept it as a pose-only local-bubble re-anchor. The same batch
            # already supplies the required independent three-view check.
            # A routine stop-and-check is only allowed to make a *micro*
            # correction.  A 35--43cm result is not a harmless LiDAR update;
            # it says that the rolling pose chain and permanent map have
            # separated.
            strong_full_view_reanchor = (
                len(inlier_indices) >= minimum_inliers
                and mean_sc >= max(8.0, float(args.min_match_score))
                and mean_support >= max(0.35, float(args.global_relocalization_min_known_ratio))
                and position_correction <= min(
                    0.20,
                    float(args.global_relocalization_max_pose_correction_m),
                )
                and heading_correction <= max(
                    float(args.relocalization_max_pose_correction_deg),
                    float(args.track_theta_window_deg),
                )
            )
            # A large correction normally belongs in the explicit recovery
            # state.  There is one important exception: three independent
            # stopped scans can all place the robot at the *same nearby pose*
            # while explaining almost all of the full current LiDAR view.
            # That is not an alias; it is the concrete "the robot slipped,
            # but the room says exactly where it is" case.  Refusing that
            # evidence was the root cause of the August 4 doorway failure:
            # 3/3 scans agreed on a 35cm correction at 90% support, yet the
            # stale pose was retained until the later, weaker fit failed.
            #
            # Re-anchor pose only inside a local bubble.  It can write one
            # provisional snapshot, but never promotes this correction to
            # GOLD map geometry.
            # ``match_commit_threshold`` is deliberately support-aware: a
            # stopped scan that explains essentially the entire permanent map
            # needs less raw correlation than one with weak overlap.  The old
            # re-anchor gate ignored that calculation and instead required
            # ``integrate_min_score`` unconditionally.  That made the code
            # reject a 3/3, 100%-supported 12.5/12.0 LiDAR correction solely
            # because the raw integration setting was 13.  In other words the
            # map knew the robot had slipped ~40 cm down the hallway, but the
            # estimator threw the result away and escalated to a 180-degree
            # recovery.  Pose-only recovery must use the same threshold that
            # the stationary batch itself has already derived.
            full_view_reanchor_score_floor = max(8.0, float(match_commit_threshold))
            full_view_continuity_reanchor = (
                len(inlier_indices) >= minimum_inliers
                and scatter <= min(0.08, float(args.viewpoint_max_position_scatter_m))
                and th_scatter <= min(2.0, float(args.viewpoint_max_heading_scatter_deg))
                and mean_sc >= full_view_reanchor_score_floor
                and mean_support >= 0.85
                and position_correction <= min(
                    0.45,
                    float(args.global_relocalization_max_pose_correction_m),
                )
                and heading_correction <= min(
                    5.0,
                    float(args.relocalization_max_pose_correction_deg),
                )
            )
            # Same local-bubble rule, but with a LiDAR-observed stationary yaw
            # correction.  This is intentionally unavailable while driving:
            # the scan must be a compact stopped batch with near-complete
            # permanent-map support before LiDAR may correct the IMU/map yaw
            # relationship.
            full_view_heading_reanchor = (
                len(inlier_indices) >= minimum_inliers
                and scatter <= min(0.08, float(args.viewpoint_max_position_scatter_m))
                and th_scatter <= 2.0
                and mean_sc >= full_view_reanchor_score_floor
                and mean_support >= 0.85
                and position_correction <= min(
                    0.45,
                    float(args.global_relocalization_max_pose_correction_m),
                )
                and 5.0 <= heading_correction <= 15.0
            )
            # Position twin of the heading drift escape.  One over-cap
            # correction is a possible alias and stays refused; but when two
            # CONSECUTIVE stationary batches independently land on the SAME
            # absolute pose (<=8cm/2deg apart) at recoverable-grade score,
            # that is corroboration — an alias would need to reproduce the
            # identical wrong answer from fresh scans twice in a row.
            # Refusing it made recoveries accept only PART of the offset
            # (30 of 46cm, 2026-08-05 19:35 run) and snapshots painted at
            # half-corrected poses became doubled ghost walls.
            persistent_offset_corroborated = False
            offset_candidate = (
                len(inlier_indices) >= minimum_inliers
                and mean_sc >= max(8.0, float(args.min_match_score))
                and mean_support >= 0.35
                and position_correction > float(args.viewpoint_max_pose_correction_m)
                and position_correction
                <= max(0.60, float(args.global_relocalization_max_pose_correction_m))
                and heading_correction
                <= max(6.0, float(args.relocalization_max_pose_correction_deg))
            )
            if offset_candidate:
                # Compare absolute robot centres only: the two batches may
                # face different directions (a pivot between snapshot
                # attempts), but a real offset places the robot at the same
                # spot both times.  Each batch's own heading correction is
                # already bounded by ``offset_candidate``.
                previous_centre = persistent_offset_state["centre"]
                if (
                    previous_centre is not None
                    and float(np.hypot(*(consensus_centre - previous_centre))) <= 0.08
                ):
                    persistent_offset_corroborated = True
                    print(
                        "[localize] persistent-offset acceptance: two consecutive "
                        "stationary batches independently agreed on the same "
                        f"{position_correction * 100:.0f}cm/"
                        f"{heading_correction:.1f}deg correction; accepting the "
                        "full offset instead of smearing partial re-anchors."
                    )
                persistent_offset_state["centre"] = consensus_centre.copy()
                persistent_offset_state["theta"] = float(consensus_pose.theta_deg)
            else:
                persistent_offset_state["centre"] = None
                persistent_offset_state["theta"] = None
            if committable and _pose_stays_beyond_completed_doorways(consensus_pose):
                # The base was stationary: every revolution belongs at one
                # consensus pose. This prevents harmless solve-angle wobble
                # from painting several rotated copies of the same walls.
                consensus_pose = _gyro_clamped_consensus(
                    consensus_pose, representative_local
                )
                cur_pose = consensus_pose
                vp_pose_ok_last[0] = True
                _record_surface_region_at_trusted_pose(cur_pose)
                _trust_world_obstacles("validated stationary map transaction")
                commit_indices = _representative_keyframe_indices(
                    [item[2] for item in batch],
                    inlier_indices,
                    max_keyframes=2,
                )
                for idx in commit_indices:
                    local_b, _pose_b, _s, _support = batch[idx]
                    if world_map.add(
                        local_b,
                        consensus_pose,
                        gold=True,
                        stationary=True,
                    ):
                        added += 1
                if len(inlier_indices) < len(batch):
                    print(
                        f"[explore]   robust stationary consensus retained "
                        f"{len(inlier_indices)}/{len(batch)} scans; rejected "
                        f"{len(batch) - len(inlier_indices)} aliased pose outlier(s)."
                    )
                if added < len(inlier_indices):
                    print(
                        f"[explore]   keyframe compaction committed the {added} "
                        f"strongest representatives from {len(inlier_indices)} "
                        "stationary inliers; redundant revolutions were not added "
                        "to GOLD geometry."
                    )
                if added:
                    print(
                        f"[explore]   map transaction committed: match {mean_sc:.1f}, "
                        f"known support {mean_support:.1%}, correction "
                        f"{position_correction * 100:.0f}cm/{heading_correction:.1f}deg."
                    )
                else:
                    # ``committable`` should normally imply writes.  Keep the
                    # viewer/map state truthful if the final write firewall
                    # rejects every representative for an unexpected reason.
                    vp_pose_ok_last[0] = False
                    print(
                        "[explore]   stationary batch localized but no keyframe was "
                        "accepted by the map-write firewall; map growth is held."
                    )
                _register_stationary_fix(
                    predicted_pose_snapshot,
                    consensus_pose,
                    representative_local,
                    mean_sc,
                    axis_locked=_axis_snap_pose(consensus_pose, representative_local)
                    is not None,
                )
            elif (
                pose_recoverable
                or strong_full_view_reanchor
                or full_view_continuity_reanchor
                or full_view_heading_reanchor
                or persistent_offset_corroborated
            ) and _pose_stays_beyond_completed_doorways(consensus_pose):
                # The batch is coherent enough to recover state estimation but
                # not strong enough to add GOLD reference geometry.  It may,
                # however, be strong enough to add one *provisional* stationary
                # occupancy keyframe.  Previously this branch was pose-only:
                # the robot would prove it had reached the next room, then
                # discard the same verified LiDAR view and continue planning
                # against the old room.  That is exactly how a healthy local
                # track became a stale-map navigation failure at the unknown
                # boundary.
                consensus_pose = _gyro_clamped_consensus(
                    consensus_pose, representative_local
                )
                cur_pose = consensus_pose
                vp_pose_ok_last[0] = True
                _record_surface_region_at_trusted_pose(cur_pose)
                provisional_written = False
                provisional_spacing_m = 0.18
                previous_provisional = last_provisional_stationary_centre[0]
                previous_provisional_heading = last_provisional_stationary_heading[0]
                heading_from_provisional = (
                    180.0
                    if previous_provisional_heading is None
                    else abs(
                        (float(consensus_pose.theta_deg) - float(previous_provisional_heading) + 180.0)
                        % 360.0
                        - 180.0
                    )
                )
                enough_new_position = (
                    previous_provisional is None
                    or float(np.hypot(*(_robot_centre(consensus_pose) - previous_provisional)))
                    >= provisional_spacing_m
                    # The 180-degree LiDAR can expose a completely new room
                    # from the same centre after a pivot.  Position-only
                    # coalescing used to discard exactly those useful map
                    # snapshots.
                    or heading_from_provisional >= 20.0
                )
                # A routine small correction still needs a higher map-support
                # bar.  A strong full-view re-anchor has already met the
                # independent multi-scan/full-LiDAR test, so it can bridge a
                # larger unmapped gap while remaining non-GOLD.
                provisional_supported = (
                    strong_full_view_reanchor
                    or full_view_continuity_reanchor
                    or full_view_heading_reanchor
                    or persistent_offset_corroborated
                    or (
                    pose_recoverable
                    and mean_support >= max(0.55, float(args.viewpoint_min_known_ratio))
                    )
                )
                if provisional_supported and enough_new_position:
                    representative_index = max(
                        inlier_indices,
                        key=lambda index: (float(batch[index][3]), float(batch[index][2])),
                    )
                    provisional_local = _filter_isolated_lidar_specks(
                        batch[representative_index][0]
                    )
                    if len(provisional_local) >= 12 and world_map.add(
                        provisional_local,
                        consensus_pose,
                        gold=False,
                        verified_stationary_gap=True,
                    ):
                        provisional_written = True
                        added = 1
                        last_provisional_stationary_centre[0] = _robot_centre(consensus_pose).copy()
                        last_provisional_stationary_heading[0] = float(consensus_pose.theta_deg)
                    elif len(provisional_local) < 12:
                        print(
                            "[explore]   stationary re-anchor had too few clean LiDAR returns "
                            "for its provisional occupancy snapshot."
                        )
                if position_correction <= 0.15 and heading_correction <= 5.0:
                    _trust_world_obstacles("bounded stationary localization consensus")
                else:
                    _quarantine_world_obstacles("stationary recovery required a large pose correction")
                print(
                    "[explore]   stationary pose "
                    f"{'RE-ANCHORED from full live LiDAR view' if (strong_full_view_reanchor or full_view_continuity_reanchor or full_view_heading_reanchor) else 'RECOVERED'} "
                    f"{'and committed one provisional occupancy snapshot' if provisional_written else 'without map commit'}: "
                    f"{len(inlier_indices)}/{len(batch)} inliers, match {mean_sc:.1f}, "
                    f"known support {mean_support:.1%}, correction "
                    f"{position_correction * 100:.0f}cm/{heading_correction:.1f}deg. "
                    "GOLD localization geometry was left unchanged."
                )
                _register_stationary_fix(
                    predicted_pose_snapshot,
                    consensus_pose,
                    representative_local,
                    mean_sc,
                    axis_locked=_axis_snap_pose(consensus_pose, representative_local)
                    is not None,
                    # A re-anchor exists BECAUSE the carried chain needed a
                    # large correction; widen the odometry edge accordingly.
                    odometry_broken=position_correction > 0.25,
                )
            else:
                if position_correction > 0.15 or heading_correction > 5.0:
                    _quarantine_world_obstacles("stationary scan disagreed with the current map pose")
                print(
                    "[explore]   viewpoint batch DISCARDED — map transaction "
                    f"failed validation: {len(inlier_indices)}/{len(batch)} inliers, "
                    f"scatter {scatter * 100:.0f}cm/{th_scatter:.1f}deg, "
                    f"mean match {mean_sc:.1f}/{match_commit_threshold:.1f}, "
                    f"known support {mean_support:.1%}/"
                    f"{float(args.viewpoint_min_known_ratio):.1%}, pose innovation "
                    f"{position_correction * 100:.0f}cm/{heading_correction:.1f}deg. "
                    "No occupancy cells were changed."
                )
                if batch:
                    prior_debug_centre = _robot_centre(cur_pose)
                    candidate_debug = []
                    for _local_d, pose_d, score_d, support_d in batch:
                        centre_d = _robot_centre(pose_d)
                        delta_d = centre_d - prior_debug_centre
                        heading_d = (
                            (float(pose_d.theta_deg) - float(cur_pose.theta_deg) + 180.0) % 360.0
                            - 180.0
                        )
                        candidate_debug.append(
                            f"d=({float(delta_d[0]):+.2f},{float(delta_d[1]):+.2f})m/"
                            f"{heading_d:+.1f}deg score={float(score_d):.1f} support={float(support_d):.0%}"
                        )
                    print(
                        "[explore]   rejected stationary candidate offsets from the predicted pose: "
                        + " | ".join(candidate_debug)
                    )
        if len(batch) < minimum_inliers:
            print(
                "[explore]   stationary batch evidence: "
                f"{batch_frames} fresh frames, {len(batch)} pose candidates, "
                f"{batch_score_rejects} score rejects, "
                f"{batch_doorway_rejects} doorway-ratchet rejects."
            )
        if added == 0 and len(batch) < minimum_inliers:
            # Too few even matched: stationary bounded relocalization vs GOLD —
            # POSE ONLY, nothing is written to the map from a recovery.
            print(f"[explore]   viewpoint match weak ({last_score:.1f}) — stationary relocalization ...")
            fid2, frame2 = feed.wait_for_frame_after(
                after_frame_id=after, timeout_s=1.5, min_frame_advances=1
            )
            if frame2 is not None:
                local2 = _scan_local(frame2, args)
                if len(local2) >= 12:
                    sup2 = _match_support_mask(_transform_points(local2, cur_pose))
                    if require_known_overlap:
                        min_known2 = max(
                            30,
                            int(math.ceil(float(args.frontier_handoff_min_known_ratio) * len(local2))),
                        )
                        if int(sup2.sum()) < min_known2:
                            local2 = np.empty((0, 2), dtype=np.float64)
                    if len(local2) >= 12:
                        support_at_seed2 = float(np.mean(sup2)) if len(sup2) else 0.0
                        match2 = (
                            local2
                            if support_at_seed2 >= 0.35
                            else (local2[sup2] if int(sup2.sum()) >= 30 else local2)
                        )
                        solved2, sc2 = _localize(match2, world_map, cur_pose, args, 0.60, 12.0)
                        last_score = sc2
                        solved_support2 = _match_support_mask(_transform_points(local2, solved2))
                        support_ratio2 = float(np.mean(solved_support2)) if len(local2) else 0.0
                        prior_centre2 = _robot_centre(cur_pose)
                        solved_centre2 = _robot_centre(solved2)
                        correction2 = float(np.hypot(*(solved_centre2 - prior_centre2)))
                        heading2 = abs(
                            (float(solved2.theta_deg) - float(cur_pose.theta_deg) + 180.0) % 360.0 - 180.0
                        )
                        if sc2 >= float(args.min_match_score):
                            print(
                                "[explore]   single-scan pose hypothesis retained only "
                                f"as a diagnostic (match {sc2:.1f}, support "
                                f"{support_ratio2:.1%}, innovation "
                                f"{correction2 * 100:.0f}cm/{heading2:.1f}deg); "
                                "multi-scan consensus is required before pose recovery."
                            )
        if vp_pose_ok_last[0] and batch:
            reset_index = inlier_indices[-1] if inlier_indices else len(batch) - 1
            _reset_local_odometry(batch[reset_index][0], cur_pose)

        known_after = world_map.grid.occupied() | world_map.grid.free()
        if known_after.shape == known_before.shape and np.allclose(world_map.grid.origin, origin_before):
            new_known = int(np.count_nonzero(known_after & ~known_before))
        else:
            # Grid growth is rare in a room-scale mission. Net known cells is a
            # conservative fallback when padding changes array coordinates.
            new_known = max(0, int(known_after.sum()) - int(known_before.sum()))
        _discard_translation_prior()  # absolute stationary fix owns this pose
        print(
            f"[perf] stationary integration "
            f"{time.monotonic() - integration_started:.3f}s "
            f"({len(batch)}/{requested_scans} matched, {added} committed)."
        )
        print(
            f"[explore]   mapped viewpoint: +{added} scans, {new_known} newly-known cells "
            f"(match {last_score:.1f})"
        )
        vp_last[0] = added
        vp_gain_last[0] = new_known
        return added

    def _commit_featureless_corridor_checkpoint() -> bool:
        """Write one stationary scan at the pose carried through a hallway.

        A parallel corridor cannot say *where along its length* the robot is,
        but it does not erase the pose that was established at the previous
        doorway/pillar.  This transaction therefore deliberately does **not**
        run another global pose search.  It proves that the base is stationary
        from three fresh scan-to-scan registrations, then attaches one
        provisional scan at the IMU/local-LiDAR carried pose.  A later view can
        promote corroborated geometry to GOLD; the immediate goal is to ensure
        that a newly seen end wall or doorway actually exists on the map at the
        measured corridor distance.
        """
        nonlocal cur_pose
        if not bool(featureless_corridor_state["active"]):
            return False
        reason = str(featureless_corridor_state["checkpoint_reason"] or "odometry interval")
        controller.halt()
        time.sleep(float(args.viewpoint_settle_s))
        known_before = world_map.grid.occupied() | world_map.grid.free()
        origin_before = world_map.grid.origin.copy()
        after = feed.latest()[0]
        samples: list[np.ndarray] = []
        previous: np.ndarray | None = None
        stable_pairs = 0
        for _ in range(5):
            fid, frame = feed.wait_for_frame_after(
                after_frame_id=after,
                timeout_s=1.0,
                min_frame_advances=1,
            )
            if frame is None:
                continue
            after = int(fid)
            local = _scan_local(frame, args)
            if len(local) < 30:
                continue
            if previous is not None:
                relative, score, support = _localize_against_points(
                    local,
                    previous,
                    Pose2D(0.0, 0.0, 0.0),
                    args,
                    search_xy_m=0.10,
                    theta_window_deg=2.0,
                )
                stationary = (
                    score >= max(4.0, float(args.min_match_score) - 2.0)
                    and support >= 0.30
                    and math.hypot(float(relative.x), float(relative.y)) <= 0.08
                    and abs(float(relative.theta_deg)) <= 1.5
                )
                stable_pairs = stable_pairs + 1 if stationary else 0
            samples.append(local)
            previous = local
            if len(samples) >= 3 and stable_pairs >= 2:
                break
        if len(samples) < 3 or stable_pairs < 2:
            print(
                "[explore] featureless-corridor checkpoint deferred: the base "
                "did not hold still across three fresh LiDAR views; pose and map remain unchanged."
            )
            return False
        representative = max(samples, key=len)
        added = world_map.add(
            _filter_isolated_lidar_specks(representative),
            cur_pose,
            gold=False,
            stationary=True,
            verified_stationary_gap=True,
        )
        known_after = world_map.grid.occupied() | world_map.grid.free()
        new_known = _new_known_cells(
            known_before,
            origin_before,
            known_after,
            world_map.grid.origin,
            world_map.grid.res,
        )
        corridor_origin = featureless_corridor_state["origin_xy"]
        travelled = (
            0.0
            if corridor_origin is None
            else float(
                np.hypot(*(_robot_centre(cur_pose) - np.asarray(corridor_origin, dtype=np.float64)))
            )
        )
        featureless_corridor_state["last_checkpoint_distance_m"] = travelled
        featureless_corridor_state["checkpoint_due"] = False
        featureless_corridor_state["checkpoint_reason"] = ""
        _reset_local_odometry(representative, cur_pose)
        vp_last[0] = int(bool(added))
        vp_gain_last[0] = int(new_known)
        leave_corridor = reason in {
            "coherent landmark view",
            "bounded corridor endpoint",
            "straight-segment endpoint",
        }
        if leave_corridor:
            featureless_corridor_state["active"] = False
            featureless_corridor_state["pending_carry_after_reanchor"] = False
        print(
            "[explore] featureless-corridor checkpoint "
            f"({reason}): attached {'one provisional LiDAR snapshot' if added else 'no new occupancy cells'} "
            f"at the carried {travelled:.2f}m corridor distance "
            f"({new_known} newly-known cells); "
            + (
                "normal landmark matching resumes."
                if leave_corridor
                else "continuing the bounded corridor traverse."
            )
        )
        return True

    active_localization_state = {
        "direction": 0.0,
        "swept_deg": 0.0,
        "translation_attempted": False,
        "rotation_stalled": False,
    }

    def _reset_active_localization_recovery() -> None:
        active_localization_state["direction"] = 0.0
        active_localization_state["swept_deg"] = 0.0
        active_localization_state["translation_attempted"] = False
        active_localization_state["rotation_stalled"] = False

    def _continuity_relocalize_stationary(
        *,
        bubble_radius_m: float | None = None,
    ) -> bool:
        """Recover tracking by voting fresh scans in the local odometric basin.

        Ordinary SLAM tracking loss is not a kidnapped-robot event.  Search the
        trusted submap around the propagated pose and require multi-scan
        consensus; never let a repeated wall elsewhere in the global map replace
        continuous odometry.  This changes pose only and cannot add geometry.
        """
        nonlocal cur_pose
        controller.halt()
        requested = max(2, int(args.global_relocalization_scans))
        after_id = feed.latest()[0]
        candidates: list[Pose2D] = []
        scores: list[float] = []
        supports: list[float] = []
        prior_centre = _robot_centre(cur_pose).copy()
        pose_reference = cur_pose
        yaw_reference = imu.deg()
        max_innovation = min(
            float(args.global_relocalization_max_pose_correction_m),
            float(bubble_radius_m)
            if bubble_radius_m is not None
            else float(args.global_relocalization_max_pose_correction_m),
        )
        # This routine authorizes a new drive/replan after tracking was lost.
        # It is deliberately stricter than normal rolling tracking: a weak
        # repeated-wall fit (the 6--8 score / 45--52% support fits in the
        # failed hallway run) is localization evidence only, never permission
        # to change the route or write another part of the room.  Map growth
        # itself already requires roughly 13, so recovery must be close to
        # that standard too.
        recovery_min_score = max(10.0, float(args.integrate_min_score) - 2.0)
        min_global_support = max(
            0.65,
            float(args.viewpoint_min_known_ratio),
            float(args.global_relocalization_min_known_ratio),
        )
        required_votes = min(3, requested)
        rejected_discontinuous = 0
        for _ in range(requested):
            frame_id, frame = feed.wait_for_frame_after(
                after_frame_id=after_id,
                timeout_s=2.0,
                min_frame_advances=1,
            )
            if frame is None:
                continue
            after_id = int(frame_id)
            local = _scan_local(frame, args)
            if len(local) < 30:
                continue
            scan_yaw = _imu_yaw_for_scan(imu, frame)
            theta_seed = float(pose_reference.theta_deg)
            if yaw_reference is not None and scan_yaw is not None:
                theta_seed += float(scan_yaw) - float(yaw_reference)
            scan_seed = _lidar_pose_from_robot_centre(
                prior_centre,
                theta_seed,
                lever_m,
                forward_offset,
            )
            # Challenge the predicted pose with the entire current view once
            # enough of it is explained by the local map.  Scoring only the
            # already-supported returns makes a wrong corridor hypothesis
            # self-confirming: the distinctive doorway/pillar that disproves it
            # is silently removed before matching.
            prior_support = _match_support_mask(_transform_points(local, scan_seed))
            support_at_seed = float(np.mean(prior_support)) if len(prior_support) else 0.0
            match_local = (
                local
                if support_at_seed >= 0.35
                else (local[prior_support] if int(np.count_nonzero(prior_support)) >= 30 else local)
            )
            solved, score = _localize(
                match_local,
                world_map,
                scan_seed,
                args,
                min(
                    float(args.global_relocalization_search_m),
                    max_innovation,
                ),
                3.0,
                coarse_angle_step_deg=1.0,
                allow_whole_map_search=False,
                fine_angle_step_deg=0.35,
                fine_theta_half_window_deg=1.5,
            )
            solved = _pose_with_imu_heading(
                solved,
                theta_seed,
                lever_m,
                forward_offset,
            )
            support = _match_support_mask(_transform_points(local, solved))
            support_ratio = float(np.mean(support)) if len(support) else 0.0
            innovation = float(np.hypot(*(_robot_centre(solved) - prior_centre)))
            # A pose at the edge of explored space CANNOT reach the normal
            # support floor: half the view falls in unknown cells no matter
            # how correct the fit is (field 2026-08-05: 3/3 votes, scatter
            # 0cm, match 14+, support 55% — discarded until the run died).
            # Admit that case only with a clearly higher score so the old
            # weak repeated-wall fits (score 6-8 at 45-52% support) remain
            # rejected.
            if (
                _relocalization_candidate_is_continuous(
                    prior_centre,
                    _robot_centre(solved),
                    score,
                    recovery_min_score,
                    support_ratio,
                    min_global_support,
                    max_innovation,
                )
                or _relocalization_candidate_is_continuous(
                    prior_centre,
                    _robot_centre(solved),
                    score,
                    recovery_min_score + 3.0,
                    support_ratio,
                    0.50,
                    max_innovation,
                )
            ) and _pose_stays_beyond_completed_doorways(solved):
                candidates.append(solved)
                scores.append(float(score))
                supports.append(support_ratio)
            elif innovation > max_innovation:
                rejected_discontinuous += 1
        if len(candidates) < required_votes:
            print(
                f"[localize] continuity-constrained stationary vote rejected: "
                f"{len(candidates)}/{requested} strong permanent-map hypotheses "
                f"(need {required_votes}, score >= {recovery_min_score:.1f}, "
                f"known support >= {min_global_support:.0%})"
                + (
                    f"; rejected {rejected_discontinuous} alias(es) beyond "
                    f"the {max_innovation:.2f}m odometric continuity bound."
                    if rejected_discontinuous
                    else "."
                )
            )
            return False
        consensus, inliers, scatter, heading_scatter = _stationary_pose_inlier_consensus(
            candidates,
            float(cur_pose.theta_deg),
            lever_m,
            forward_offset,
            max_position_residual_m=0.15,
            max_heading_residual_deg=5.0,
            min_inliers=required_votes,
            reference_centre_xy=prior_centre,
        )
        if len(inliers) < required_votes or scatter > 0.15 or heading_scatter > 5.0:
            print(
                f"[localize] local-submap hypotheses disagree: {len(inliers)}/"
                f"{len(candidates)} inliers, scatter "
                f"{scatter * 100:.0f}cm/{heading_scatter:.1f}deg."
            )
            return False
        predicted_recovery_pose = cur_pose
        cur_pose = consensus
        _discard_translation_prior()
        vp_pose_ok_last[0] = True
        _reset_active_localization_recovery()
        _rezero_gyro_heading_reference(
            float(consensus.theta_deg), "continuity-constrained consensus accepted"
        )
        mean_score = float(np.mean([scores[index] for index in inliers]))
        _register_stationary_fix(
            predicted_recovery_pose,
            consensus,
            None,
            mean_score,
            axis_locked=False,
            odometry_broken=True,
        )
        mean_support = float(np.mean([supports[index] for index in inliers]))
        innovation = float(np.hypot(*(_robot_centre(consensus) - prior_centre)))
        if innovation <= 0.15 and heading_scatter <= 5.0:
            _trust_world_obstacles("bounded continuity-constrained stationary consensus")
        else:
            _quarantine_world_obstacles("continuity recovery required a large pose correction")
        print(
            f"[localize] continuity-constrained pose recovered from {len(inliers)}/"
            f"{requested} stationary scans (match {mean_score:.1f}, "
            f"known support {mean_support:.1%}, scatter "
            f"{scatter * 100:.0f}cm/{heading_scatter:.1f}deg, pose innovation "
            f"{innovation * 100:.0f}cm); map unchanged."
        )
        return True

    def _reacquire_opposite_view_in_local_bubble() -> bool:
        """Use the post-turn *full* LiDAR view to reacquire nearby GOLD geometry.

        This is deliberately separate from ordinary rolling tracking.  It runs
        only after a measured turnaround, when the robot has exposed the
        previously mapped doorway/corner behind it.  The IMU supplies a narrow
        physical prior, but it does not overwrite a three-view LiDAR heading
        solution: that would discard the very orientation evidence this
        procedure was asked to recover.
        """
        nonlocal cur_pose
        controller.halt()
        requested = max(3, int(args.global_relocalization_scans))
        required_votes = min(3, requested)
        prior = cur_pose
        prior_centre = _robot_centre(prior).copy()
        max_innovation = max(
            float(args.global_relocalization_max_pose_correction_m),
            0.80,
        )
        # Scan origins slightly beyond the position bubble can still have seen
        # the same doorway from the other side of the threshold.  This is not a
        # search expansion: candidate robot centres remain inside the bubble.
        local_ref_radius = max_innovation + 1.25
        local_reference = world_map.local_gold_reference(
            prior_centre,
            pose_radius_m=local_ref_radius,
        )
        if len(local_reference) < 80:
            print(
                "[localize] opposite-view bubble reacquisition has too little nearby "
                f"permanent geometry ({len(local_reference)} points within "
                f"{local_ref_radius:.2f}m); retaining the normal local-bubble vote."
            )
            return False

        # This recovery is entered only after a measured 180-degree turn and
        # still requires three fresh scans to agree.  In that specific state,
        # high known-map support is more important than the raw integration
        # score: hallway/doorway geometry can be sparse but decisive.  The
        # previous 10.0 floor rejected 95%-supported recovery scans at 9.1.
        recovery_min_score = max(8.5, float(args.integrate_min_score) - 4.0)
        min_support = max(
            0.65,
            float(args.viewpoint_min_known_ratio),
            float(args.global_relocalization_min_known_ratio),
        )
        after_id = feed.latest()[0]
        yaw_reference = imu.deg_fresh(wait_up_to_s=0.5, max_age_s=0.4)
        candidates: list[Pose2D] = []
        scores: list[float] = []
        supports: list[float] = []
        accepted_locals: list[np.ndarray] = []
        rejected: list[str] = []
        # Votes whose ONLY defect is exceeding the fixed continuity bubble.
        # A unanimous high-support set of these is a confident fix that the
        # bubble radius happens to be a few centimetres too small for — the
        # 2026-08-05 field run rejected twelve consecutive identical fixes at
        # 81-87cm against an 80cm bubble until the run died.
        wide_candidates: list[Pose2D] = []
        wide_scores: list[float] = []
        wide_supports: list[float] = []
        wide_locals: list[np.ndarray] = []
        # Votes pinned at the gyro-seeded theta window edge with high map
        # support.  The seed heading comes from the gyro ledger; if that
        # reference has drifted, every solve clamps at the window edge,
        # cannot finish rotating, and the misalignment ALSO depresses the
        # score — the exact signature that killed the 2026-08-05 run (eight
        # consecutive unanimous fixes at yaw 18-20deg, support 85-92%,
        # rejected on score/heading until the recovery walk exhausted).
        drift_retry: list[tuple[np.ndarray, Pose2D]] = []
        for _ in range(requested):
            frame_id, frame = feed.wait_for_frame_after(
                after_frame_id=after_id,
                timeout_s=2.0,
                min_frame_advances=1,
            )
            if frame is None:
                continue
            after_id = int(frame_id)
            local = _scan_local(frame, args)
            if len(local) < 30:
                continue
            scan_yaw = _imu_yaw_for_scan(imu, frame)
            theta_seed = float(prior.theta_deg)
            if yaw_reference is not None and scan_yaw is not None:
                theta_seed += float(scan_yaw) - float(yaw_reference)
            seed = _lidar_pose_from_robot_centre(
                prior_centre,
                theta_seed,
                lever_m,
                forward_offset,
            )
            # The entire present view participates.  In particular, do not
            # pre-filter away the doorway/corner merely because the previous
            # forward-facing hallway hypothesis could not explain it.
            solved, score, local_support = _localize_against_points(
                local,
                local_reference,
                seed,
                args,
                search_xy_m=max_innovation,
                # IMU still limits this to a physical post-turn neighborhood,
                # while allowing LiDAR to correct a modest map/gyro yaw bias.
                theta_window_deg=18.0,
            )
            global_support = float(
                np.mean(_match_support_mask(_transform_points(local, solved)))
            ) if len(local) else 0.0
            support = min(float(local_support), global_support)
            innovation = float(np.hypot(*(_robot_centre(solved) - prior_centre)))
            heading_delta = abs(
                (float(solved.theta_deg) - float(theta_seed) + 180.0) % 360.0 - 180.0
            )
            accepted = (
                score >= recovery_min_score
                and support >= min_support
                and innovation <= max_innovation
                and heading_delta <= 18.0
                and _pose_stays_beyond_completed_doorways(solved)
            )
            if accepted:
                candidates.append(solved)
                scores.append(float(score))
                supports.append(float(support))
                accepted_locals.append(local.copy())
            else:
                why: list[str] = []
                if score < recovery_min_score:
                    why.append("score")
                if support < min_support:
                    why.append("support")
                if innovation > max_innovation:
                    why.append("bubble")
                if heading_delta > 18.0:
                    why.append("heading")
                if not _pose_stays_beyond_completed_doorways(solved):
                    why.append("doorway")
                if why == ["bubble"] and (
                    score >= recovery_min_score + 2.0
                    and support >= max(0.85, min_support)
                    and innovation <= 2.0 * max_innovation
                ):
                    wide_candidates.append(solved)
                    wide_scores.append(float(score))
                    wide_supports.append(float(support))
                    wide_locals.append(local.copy())
                if (
                    heading_delta >= 14.0
                    and support >= min_support
                    and innovation <= 2.0 * max_innovation
                ):
                    drift_retry.append((local.copy(), seed))
                rejected.append(
                    f"match {score:.1f}, support {support:.0%}, "
                    f"innovation {innovation * 100:.0f}cm, yaw {heading_delta:.1f}deg "
                    f"({'/'.join(why) or 'unknown'})"
                )
        if len(candidates) < required_votes and len(wide_candidates) >= required_votes:
            # The bubble is a continuity prior, not physics.  When every fresh
            # view independently proposes the SAME high-support pose a bounded
            # distance beyond it, the accumulated odometric error is simply
            # larger than the bubble assumed; refusing the unanimous fix and
            # dying instead is the wrong trade.  Require a tighter mutual
            # agreement than the normal path before widening.
            wide_consensus, wide_inliers, wide_scatter, wide_heading_scatter = (
                _stationary_pose_inlier_consensus(
                    wide_candidates,
                    float(prior.theta_deg),
                    lever_m,
                    forward_offset,
                    max_position_residual_m=0.10,
                    max_heading_residual_deg=3.0,
                    min_inliers=required_votes,
                    reference_centre_xy=prior_centre,
                )
            )
            if len(wide_inliers) >= required_votes and wide_scatter <= 0.10 and wide_heading_scatter <= 3.0:
                wide_innovation = float(np.hypot(*(_robot_centre(wide_consensus) - prior_centre)))
                print(
                    "[localize] widened-bubble acceptance: "
                    f"{len(wide_inliers)}/{requested} unanimous fresh views agree on one pose "
                    f"{wide_innovation * 100:.0f}cm beyond the {max_innovation:.2f}m continuity "
                    f"bubble (mean match {float(np.mean([wide_scores[i] for i in wide_inliers])):.1f}, "
                    f"support {float(np.mean([wide_supports[i] for i in wide_inliers])):.0%}, "
                    f"scatter {wide_scatter * 100:.0f}cm/{wide_heading_scatter:.1f}deg); "
                    "accepting the consistent fix instead of exhausting the recovery."
                )
                candidates = wide_candidates
                scores = wide_scores
                supports = wide_supports
                accepted_locals = wide_locals
        if len(candidates) < required_votes and len(drift_retry) >= required_votes:
            # Gyro-reference drift retry: every fresh view proposed a
            # high-support fix pinned at the gyro-seeded window edge.  A LiDAR
            # alias is sign-random and geometry-dependent; a unanimous
            # same-direction rotation is the gyro reference having drifted
            # away from the map.  Re-solve the SAME views with a window wide
            # enough to finish the rotation, and only accept if the full
            # solve now clears the NORMAL score/support gates and all views
            # still agree on one pose.
            retry_candidates: list[Pose2D] = []
            retry_scores: list[float] = []
            retry_supports: list[float] = []
            retry_locals: list[np.ndarray] = []
            for local_r, seed_r in drift_retry:
                solved_r, score_r, local_support_r = _localize_against_points(
                    local_r,
                    local_reference,
                    seed_r,
                    args,
                    search_xy_m=max_innovation,
                    theta_window_deg=35.0,
                )
                global_support_r = float(
                    np.mean(_match_support_mask(_transform_points(local_r, solved_r)))
                ) if len(local_r) else 0.0
                support_r = min(float(local_support_r), global_support_r)
                innovation_r = float(np.hypot(*(_robot_centre(solved_r) - prior_centre)))
                if (
                    score_r >= recovery_min_score
                    and support_r >= min_support
                    and innovation_r <= 2.0 * max_innovation
                    and _pose_stays_beyond_completed_doorways(solved_r)
                ):
                    retry_candidates.append(solved_r)
                    retry_scores.append(float(score_r))
                    retry_supports.append(float(support_r))
                    retry_locals.append(local_r)
            if len(retry_candidates) >= required_votes:
                retry_consensus, retry_inliers, retry_scatter, retry_heading_scatter = (
                    _stationary_pose_inlier_consensus(
                        retry_candidates,
                        float(prior.theta_deg),
                        lever_m,
                        forward_offset,
                        max_position_residual_m=0.10,
                        max_heading_residual_deg=3.0,
                        min_inliers=required_votes,
                        reference_centre_xy=prior_centre,
                    )
                )
                if (
                    len(retry_inliers) >= required_votes
                    and retry_scatter <= 0.10
                    and retry_heading_scatter <= 3.0
                ):
                    retry_yaw = (
                        (float(retry_consensus.theta_deg) - float(prior.theta_deg) + 180.0)
                        % 360.0
                    ) - 180.0
                    print(
                        "[localize] gyro-drift acceptance: "
                        f"{len(retry_inliers)}/{requested} fresh views unanimously "
                        f"propose the same {retry_yaw:+.1f}deg heading fix beyond the "
                        "gyro-seeded window (mean match "
                        f"{float(np.mean([retry_scores[i] for i in retry_inliers])):.1f}, "
                        f"support {float(np.mean([retry_supports[i] for i in retry_inliers])):.0%}, "
                        f"scatter {retry_scatter * 100:.0f}cm/{retry_heading_scatter:.1f}deg); "
                        "treating the disagreement as gyro-reference drift, not aliasing."
                    )
                    candidates = retry_candidates
                    scores = retry_scores
                    supports = retry_supports
                    accepted_locals = retry_locals
        if len(candidates) < required_votes:
            detail = "; ".join(rejected[-3:]) if rejected else "no usable fresh scans"
            print(
                "[localize] opposite-view nearby-geometry reacquisition rejected: "
                f"{len(candidates)}/{requested} votes using {len(local_reference)} nearby GOLD points; {detail}."
            )
            return False
        consensus, inliers, scatter, heading_scatter = _stationary_pose_inlier_consensus(
            candidates,
            float(prior.theta_deg),
            lever_m,
            forward_offset,
            max_position_residual_m=0.12,
            max_heading_residual_deg=4.0,
            min_inliers=required_votes,
            reference_centre_xy=prior_centre,
        )
        if len(inliers) < required_votes or scatter > 0.12 or heading_scatter > 4.0:
            print(
                "[localize] opposite-view nearby-geometry votes disagree: "
                f"{len(inliers)}/{len(candidates)} inliers, scatter "
                f"{scatter * 100:.0f}cm/{heading_scatter:.1f}deg."
            )
            return False
        predicted_recovery_pose = cur_pose
        cur_pose = consensus
        _discard_translation_prior()
        vp_pose_ok_last[0] = True
        _reset_active_localization_recovery()
        # An accepted three-view consensus is authorised to correct heading
        # beyond the gyro expectation; the veto measures future drift from
        # THIS fix.
        _rezero_gyro_heading_reference(
            float(consensus.theta_deg), "opposite-view GOLD reacquisition accepted"
        )
        # This view is the distinctive doorway/pillar geometry that resolved
        # the parallel-wall alias.  Preserve one strongest member of the
        # three-view consensus as GOLD so a later turnaround has the same
        # locally observed landmark available again.  The saved *pre-turn*
        # hallway scan remains non-GOLD because parallel walls alone are not a
        # reliable localization reference.
        strongest = max(inliers, key=lambda index: scores[index])
        landmark_added = world_map.add(
            accepted_locals[strongest],
            consensus,
            gold=True,
            stationary=True,
        )
        if landmark_added:
            _reset_local_odometry(accepted_locals[strongest], consensus)
        _register_stationary_fix(
            predicted_recovery_pose,
            consensus,
            accepted_locals[strongest],
            float(np.mean([scores[i] for i in inliers])),
            axis_locked=_axis_snap_pose(consensus, accepted_locals[strongest])
            is not None,
            odometry_broken=True,
        )
        innovation = float(np.hypot(*(_robot_centre(consensus) - prior_centre)))
        imu_heading_error = abs(
            (float(consensus.theta_deg) - float(prior.theta_deg) + 180.0) % 360.0 - 180.0
        )
        print(
            "[localize] opposite-view nearby GOLD geometry reacquired pose from "
            f"{len(inliers)}/{requested} full LiDAR views (match "
            f"{float(np.mean([scores[i] for i in inliers])):.1f}, support "
            f"{float(np.mean([supports[i] for i in inliers])):.0%}, innovation "
            f"{innovation * 100:.0f}cm, LiDAR heading correction {imu_heading_error:.1f}deg); "
            + (
                "committed the verified opposite-view landmark as GOLD."
                if landmark_added
                else "the map integrity firewall rejected the landmark commit."
            )
        )
        return True

    def _active_localization_translation(
        distance_m: float | None = None,
        direction_body_deg: float | None = None,
    ) -> bool:
        """Acquire parallax with one short, pre-validated body translation.

        A turn changes scan orientation but not the observation origin. Repeated
        rotational probes therefore cannot resolve translational aliases along
        similar walls. Once a bounded monotonic scan fan is exhausted, choose a
        short local trajectory that the live calibrated footprint proves clear,
        execute it with continuous envelope monitoring, and verify the physical
        displacement by scan-to-scan odometry. This is pose-only active
        localization: it never writes uncertain geometry to the occupancy map.
        """
        nonlocal cur_pose
        distance = (
            float(args.active_localization_translation_m)
            if distance_m is None
            else float(distance_m)
        )
        if distance < 0.01 or collision_profile is None:
            return False
        before_points = _startup_forward_scan()
        if before_points is None:
            print("[localize] parallax recovery unavailable: no fresh LiDAR safety scan.")
            return False
        geometry = {
            "lidar_offset_forward_m": lever_m,
            "physical_body_radius_m": float(args.physical_body_radius_m),
            "self_mask_inset_m": float(args.collision_self_mask_inset_m),
        }
        initial_hit = _collision_box_violation(before_points, collision_profile, **geometry)
        ranked: list[tuple[float, float, float, np.ndarray]] = []
        # A caller may request one specific body-frame direction (arrival trim
        # drives the measured offset to the planned point); otherwise fan out
        # over eight candidates and prefer safe forward parallax.
        if direction_body_deg is None:
            candidate_angles = [float(a) for a in range(0, 360, 45)]
        else:
            candidate_angles = [float(direction_body_deg)]
        for angle_deg in candidate_angles:
            delta = distance * np.array(
                [
                    math.cos(math.radians(angle_deg)),
                    math.sin(math.radians(angle_deg)),
                ]
            )
            if initial_hit is None:
                safe = _translation_trajectory_is_safe(
                    before_points,
                    collision_profile,
                    delta,
                    **geometry,
                )
                improvement = 0.0
            else:
                safe, improvement = _translation_escape_is_safe(
                    before_points,
                    collision_profile,
                    delta,
                    **geometry,
                )
            if not safe:
                continue
            projected = before_points - delta
            projected_hit = _collision_box_violation(projected, collision_profile, **geometry)
            if initial_hit is None and projected_hit is not None:
                continue
            pivot_margin = _full_pivot_clearance_margin_m(projected, collision_profile, **geometry)
            # Recovery is entered while the base is aligned with the current
            # straight path segment. Prefer safe forward parallax so recovery
            # continues that plan instead of introducing an arbitrary strafe.
            # Clearance remains a hard prerequisite above, not a ranking hint.
            forward_preference = float(delta[0] / max(distance, 1e-6))
            ranked.append(
                (
                    forward_preference,
                    float(pivot_margin),
                    float(improvement),
                    delta,
                )
            )
        if not ranked:
            print(
                "[localize] bounded sweep exhausted, but no short parallax trajectory "
                "is collision-free; holding position without reversing the sweep."
            )
            return False
        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        _forward_preference, _margin, _improvement, delta = ranked[0]
        unit = delta / max(1e-6, float(np.hypot(*delta)))
        # Field measurement: this base does not reliably break drivetrain
        # stiction below about 0.8 command units.  The previous 0.12-0.35
        # recovery range therefore produced a valid-looking command with zero
        # physical response and fed an endless plan/pivot/replan loop.
        command_xy = _startup_escape_velocity(
            delta,
            abs(float(args.startup_escape_speed)),
        )
        command_effort = float(np.max(np.abs(command_xy)))
        duration_s = float(
            np.clip(
                distance / max(0.05, float(args.drive_command_distance_scale) * command_effort),
                0.45,
                max(1.20, 10.0 * distance),
            )
        )
        heading = imu.deg()
        if heading is None:
            return False
        baseline_depth = _collision_depth_m(initial_hit)
        worsened_since: float | None = None
        controller.clear_safety_latch()
        if direction_body_deg is None:
            print(
                "[localize] monotonic sweep exhausted; acquiring parallax with a "
                f"collision-checked ({delta[0]:+.2f}m forward, {delta[1]:+.2f}m left) "
                "translation."
            )
        else:
            print(
                f"[drive] collision-checked trim translation ({delta[0]:+.2f}m forward, "
                f"{delta[1]:+.2f}m left)."
            )
        started = time.monotonic()
        try:
            while time.monotonic() - started < duration_s:
                controller.translate_body(
                    float(command_xy[0]),
                    float(command_xy[1]),
                    float(heading),
                )
                live_points = _startup_forward_scan()
                if live_points is not None:
                    live_depth = _collision_depth_m(
                        _collision_box_violation(live_points, collision_profile, **geometry)
                    )
                    if live_depth > baseline_depth + 0.01:
                        if worsened_since is None:
                            worsened_since = time.monotonic()
                        elif time.monotonic() - worsened_since >= 0.12:
                            print(
                                "[localize] parallax translation stopped because the "
                                "live collision envelope began closing."
                            )
                            break
                    else:
                        worsened_since = None
                time.sleep(0.04)
        finally:
            controller.halt()
            time.sleep(0.30)
        after_points = _startup_forward_scan()
        measured_along, measured_cross, motion_score = _measure_startup_translation(
            before_points, after_points, delta
        )
        motion_verified = (
            motion_score >= float(args.min_match_score) and measured_along >= 0.025 and measured_cross <= 0.10
        )
        if not motion_verified:
            print(
                "[localize] parallax command was not accepted as odometry "
                f"({measured_along * 100:+.1f}cm along, "
                f"{measured_cross * 100:.1f}cm cross, match {motion_score:.1f}); "
                "pose and map remain unchanged."
            )
            return False
        body_delta = unit * measured_along
        cur_pose = _translate_lidar_pose_in_body_frame(
            cur_pose,
            float(body_delta[0]),
            float(body_delta[1]),
            lever_m,
            forward_offset,
        )
        _discard_translation_prior()
        trail.append(_robot_centre(cur_pose).copy())
        active_localization_state["swept_deg"] = 0.0
        active_localization_state["translation_attempted"] = True
        active_localization_state["rotation_stalled"] = False
        print(
            f"[localize] parallax translation verified ({measured_along * 100:.1f}cm, "
            f"match {motion_score:.1f}); retrying global localization from the new view."
        )
        return True

    def _staged_bubble_relocalize_stationary() -> bool:
        """Refit the full live LiDAR view near the predicted pose first.

        Losing tracking is normally a small accumulated slip, not a teleport.
        Search the smallest plausible bubble before admitting a wider one so a
        distant repeated wall cannot win merely because it has a slightly
        higher raw correlation.  Every stage collects a new stationary
        three-view vote; no geometry is written and the robot remains halted
        throughout this decision.
        """
        maximum = float(args.global_relocalization_max_pose_correction_m)
        radii: list[float] = []
        for radius in (0.20, 0.40, maximum):
            bounded = min(maximum, float(radius))
            if bounded > 0.0 and all(abs(bounded - prior) > 0.01 for prior in radii):
                radii.append(bounded)
        for index, radius in enumerate(radii, start=1):
            print(
                f"[localize] full-view local-bubble refit {index}/{len(radii)}: "
                f"searching {radius:.2f}m around the propagated pose with fresh LiDAR votes."
            )
            if _continuity_relocalize_stationary(bubble_radius_m=radius):
                return True
        return False

    def _turnaround_relocalize_once() -> bool:
        """Turn around once, then reacquire from the opposite-looking room.

        When the bounded stationary/local-submap attempt is inconclusive, the
        current forward view may simply be a low-parallax hallway edge.  A
        single collision-checked 180-degree turn exposes the geometry behind
        the robot; if that is still insufficient, one safe body-forward move
        in the new heading retraces the previous approach before a final
        continuity vote.  This is deliberately one-shot so recovery cannot
        oscillate forever.
        """
        nonlocal cur_pose
        controller.halt()
        # A fresh recovery supersedes any previous featureless-corridor
        # allowance.  It is re-enabled only after the opposite-facing
        # landmark has passed its independent three-view consensus.
        featureless_corridor_state["active"] = False
        time.sleep(0.25)
        # Preserve the forward hallway view that led to this recovery.  It may
        # be geometrically ambiguous by itself (two parallel walls), but once
        # the opposite-facing doorway view has re-localized the robot, the
        # measured physical turn supplies its exact map pose.  This is the
        # safe way to extend the map into the hallway without treating the
        # hallway scan itself as a localization authority.
        preturn_frame = feed.latest()[1]
        preturn_local = _scan_local(preturn_frame, args) if preturn_frame is not None else None
        if preturn_local is not None and len(preturn_local) < 30:
            preturn_local = None
        # Accumulated collision-checked translation executed after the 180deg
        # turn while walking back toward mapped geometry to reacquire the
        # reference.  The pre-turn hallway scan and the local-odometry seed
        # were captured at the pre-walk position, so both are only reused when
        # this stays small.
        walkback_m = [0.0]

        def _commit_preturn_forward_view(actual_turn_deg: float) -> None:
            if preturn_local is None:
                print(
                    "[map] opposite-view recovery succeeded, but no usable pre-turn "
                    "hallway scan was available to place."
                )
                return
            if walkback_m[0] > 0.30:
                print(
                    "[map] pre-turn hallway scan skipped: the reference walk moved "
                    f"{walkback_m[0]:.2f}m from where it was captured; the corridor "
                    "will be remapped by the carried-pose checkpoints instead."
                )
                return
            # ``cur_pose`` is the accepted post-turn LiDAR pose. Rotate it
            # back about the physical robot centre by the *measured* IMU turn,
            # rather than trusting the old predicted pose. The saved forward
            # scan consequently lands where it was actually observed before
            # the 180-degree recovery turn.
            preturn_pose = _rotate_lidar_pose_about_robot_centre(
                cur_pose,
                -float(actual_turn_deg),
                lever_m,
                forward_offset,
            )
            known_before = world_map.grid.occupied() | world_map.grid.free()
            origin_before = world_map.grid.origin.copy()
            added = world_map.add(
                preturn_local,
                preturn_pose,
                gold=False,
                stationary=True,
                verified_stationary_gap=True,
            )
            if not added:
                print(
                    "[map] recovered pre-turn hallway scan was rejected by the map "
                    "integrity firewall; permanent map unchanged."
                )
                return
            known_after = world_map.grid.occupied() | world_map.grid.free()
            if known_after.shape == known_before.shape and np.allclose(world_map.grid.origin, origin_before):
                newly_known = int(np.count_nonzero(known_after & ~known_before))
            else:
                newly_known = max(0, int(known_after.sum()) - int(known_before.sum()))
            print(
                "[map] placed the pre-turn forward hallway scan from the accepted "
                f"opposite-view pose ({len(preturn_local)} returns, "
                f"{newly_known} newly-known cells); it remains non-GOLD until "
                "independent stationary confirmation."
            )

        def _return_to_featureless_corridor(actual_turn_deg: float) -> bool:
            """Face the original hallway again without discarding the re-anchor.

            The first turnaround exists only to expose a distinctive landmark.
            Once that landmark has fixed the map pose, return to the original
            parallel-wall direction and carry the now-known pose through a
            bounded, IMU/local-odometry corridor interval.
            """
            nonlocal cur_pose
            yaw_before_return = imu.deg_fresh(wait_up_to_s=0.6, max_age_s=0.4)
            if yaw_before_return is None:
                print("[localize] landmark recovery accepted, but no fresh IMU yaw was available to face the corridor again.")
                return False
            requested = -float(actual_turn_deg)
            blocked = _rotation_safety_check(requested)
            if blocked is not None:
                print(
                    "[localize] landmark recovery accepted, but the return turn to "
                    f"the featureless corridor is collision-blocked: {blocked}."
                )
                return False
            controller.clear_safety_latch()
            controller.rotate_to(float(yaw_before_return) + requested)
            started = time.monotonic()
            try:
                while time.monotonic() - started < 14.0:
                    if controller.safety_latched_reason() is not None:
                        break
                    yaw_now = imu.deg()
                    if yaw_now is not None and abs(float(yaw_now) - float(yaw_before_return)) >= 174.0:
                        break
                    time.sleep(0.05)
            finally:
                controller.halt()
                time.sleep(0.25)
            yaw_after_return = imu.deg_fresh(wait_up_to_s=0.6, max_age_s=0.4)
            if yaw_after_return is None:
                return False
            actual_return = float(yaw_after_return) - float(yaw_before_return)
            if abs(actual_return) < 120.0 or controller.safety_latched_reason() is not None:
                print(
                    "[localize] landmark recovery remains valid, but the corridor "
                    f"return turn completed only {actual_return:+.1f}deg; normal planning will choose the next heading."
                )
                return False
            cur_pose = _rotate_lidar_pose_about_robot_centre(
                cur_pose,
                actual_return,
                lever_m,
                forward_offset,
            )
            _discard_translation_prior()
            if preturn_local is not None and walkback_m[0] <= 0.30:
                _reset_local_odometry(preturn_local, cur_pose)
            featureless_corridor_state.update(
                active=True,
                origin_xy=_robot_centre(cur_pose).copy(),
                heading_deg=float(cur_pose.theta_deg),
                # A short 20–40cm allowance merely re-enters the same
                # parallel-wall ambiguity. Carry the re-anchored pose for a
                # meaningful hallway interval (subject to the normal live
                # collision envelope) so the robot has a realistic chance to
                # expose a doorway, pillar, or end wall. Sparse stationary
                # provisional scans keep that measured interval on the map.
                # The walk back toward the reference must be re-traversed on
                # top of the normal hallway allowance for the beeline to have
                # a realistic chance of reaching the far end of the corridor.
                max_distance_m=max(4.00, 10.0 * float(args.robot_radius_m)) + float(walkback_m[0]),
                skip_logged=False,
                checkpoint_due=False,
                checkpoint_reason="",
                last_checkpoint_distance_m=0.0,
                ignored_cross_track_updates=0,
                pending_carry_after_reanchor=True,
                carry_stalled_legs=0,
            )
            print(
                "[localize] returned to the re-anchored parallel corridor; "
                f"holding IMU/local-LiDAR odometry for up to "
                f"{float(featureless_corridor_state['max_distance_m']):.2f}m "
                "until a distinct landmark is available for a normal map refit."
            )
            return True

        def _rotate_and_track(delta_deg: float) -> float | None:
            """Collision-checked in-place rotation with a measured pose update."""
            nonlocal cur_pose
            yaw_before = imu.deg_fresh(wait_up_to_s=0.6, max_age_s=0.4)
            if yaw_before is None or _rotation_safety_check(float(delta_deg)) is not None:
                return None
            controller.clear_safety_latch()
            controller.rotate_to(float(yaw_before) + float(delta_deg))
            started = time.monotonic()
            try:
                while time.monotonic() - started < 14.0:
                    if controller.safety_latched_reason() is not None:
                        break
                    yaw_now = imu.deg()
                    if yaw_now is not None and abs(float(yaw_now) - float(yaw_before)) >= abs(float(delta_deg)) - 6.0:
                        break
                    time.sleep(0.05)
            finally:
                controller.halt()
                time.sleep(0.30)
            yaw_after = imu.deg_fresh(wait_up_to_s=0.6, max_age_s=0.4)
            if yaw_after is None:
                return None
            actual_rotation = float(yaw_after) - float(yaw_before)
            if abs(actual_rotation) > 2.0:
                cur_pose = _rotate_lidar_pose_about_robot_centre(
                    cur_pose,
                    actual_rotation,
                    lever_m,
                    forward_offset,
                )
                _discard_translation_prior()
                trail.append(_robot_centre(cur_pose).copy())
            return actual_rotation

        def _reference_walk(actual_turn_deg: float, turned: bool) -> bool:
            """Walk to the nearest GOLD reference, voting whenever it is in reach.

            The walk budget and direction come from where the trustworthy
            reference ACTUALLY is.  Field 2026-08-05 (morning): a corridor-
            derived budget exhausted itself without ever seeing GOLD.  Field
            2026-08-05 (afternoon): the walk commanded a backward step when
            GOLD lay behind — a 180deg-FOV scanner can neither watch nor
            verify reverse motion, so the step was refused and the run died.
            The walk now physically faces the reference before stepping.
            """
            nearest_gold_m, nearest_gold_xy = world_map.nearest_gold_scan(
                _robot_centre(cur_pose)
            )
            max_walkback_m = float(
                np.clip(
                    (nearest_gold_m + 1.00) if math.isfinite(nearest_gold_m) else 4.00,
                    1.50,
                    6.00,
                )
            )
            failed_steps = 0
            while True:
                centre_now = _robot_centre(cur_pose)
                nearest_gold_m, nearest_gold_xy = world_map.nearest_gold_scan(centre_now)
                gold_in_reach = math.isfinite(nearest_gold_m) and nearest_gold_m <= 2.00
                reacquired = False
                if gold_in_reach:
                    reacquired = _reacquire_opposite_view_in_local_bubble()
                    if not reacquired:
                        print(
                            "[localize] opposite-view nearby-geometry match was "
                            "inconclusive; trying the standard local-bubble vote here."
                        )
                        reacquired = _staged_bubble_relocalize_stationary()
                else:
                    print(
                        f"[localize] nearest GOLD reference is {nearest_gold_m:.2f}m away; "
                        "walking to it before spending stationary votes against "
                        "provisional geometry."
                    )
                if reacquired:
                    if turned:
                        _commit_preturn_forward_view(actual_turn_deg)
                        _return_to_featureless_corridor(actual_turn_deg)
                    return True
                remaining = max_walkback_m - walkback_m[0]
                if remaining < 0.15 or failed_steps >= 2:
                    break
                bearing_deg = None
                if nearest_gold_xy is not None:
                    to_gold = nearest_gold_xy - centre_now
                    if float(np.hypot(*to_gold)) > 0.05:
                        bearing_deg = (
                            (
                                math.degrees(math.atan2(float(to_gold[1]), float(to_gold[0])))
                                - (float(cur_pose.theta_deg) + forward_offset)
                                + 180.0
                            )
                            % 360.0
                        ) - 180.0
                if bearing_deg is not None and abs(bearing_deg) > 60.0:
                    print(
                        f"[localize] GOLD reference bears {bearing_deg:+.0f}deg; "
                        "rotating to face it before stepping."
                    )
                    if _rotate_and_track(bearing_deg) is not None:
                        bearing_deg = 0.0
                    else:
                        print(
                            "[localize] facing rotation was blocked; keeping the "
                            "current heading for a fan-selected safe step."
                        )
                        bearing_deg = None
                step_m = min(0.45 if nearest_gold_m > 2.5 else 0.30, remaining)
                print(
                    "[localize] no confident reference at this position; walking "
                    f"{step_m:.2f}m toward the GOLD reference "
                    f"({walkback_m[0]:.2f}/{max_walkback_m:.2f}m used)."
                )
                active_localization_state["translation_attempted"] = False
                centre_before_step = _robot_centre(cur_pose).copy()
                moved = _active_localization_translation(
                    distance_m=step_m,
                    direction_body_deg=bearing_deg,
                )
                if not moved and bearing_deg is not None:
                    # The direct line to the reference is blocked; let the
                    # eight-direction fan pick any safe parallax step so the
                    # walk can slide around the obstruction.
                    moved = _active_localization_translation(distance_m=step_m)
                if moved:
                    failed_steps = 0
                    walkback_m[0] += float(
                        np.hypot(*(_robot_centre(cur_pose) - centre_before_step))
                    )
                else:
                    failed_steps += 1
            print(
                "[localize] reference walk exhausted without a confident fix "
                f"({walkback_m[0]:.2f}m of {max_walkback_m:.2f}m walked)."
            )
            return False

        # A silent IMU is a transient transport problem, not a localization
        # verdict — the ZMQ streams auto-reconnect.  Hold in place for the
        # stream instead of aborting the run (field 2026-08-05: a ~5s dropout
        # at exactly this decision point ended an otherwise healthy run with
        # "turnaround recovery unavailable: IMU yaw is stale").
        if imu.deg_fresh(wait_up_to_s=0.6, max_age_s=0.4) is None:
            print(
                "[localize] IMU yaw is stale at the recovery decision; holding "
                "stationary up to 25s for the stream to reconnect."
            )
            if _wait_for_navigation_yaw(imu, wait_up_to_s=25.0) is None:
                print("[localize] turnaround recovery unavailable: IMU yaw is stale.")
                return False

        # If the trusted reference already lies AHEAD, a 180deg turn is
        # counterproductive: it points the forward-only scanner away from the
        # geometry that can fix the pose, and the subsequent walk would have
        # to command unverifiable backward motion (field 2026-08-05: the route
        # had already reversed the robot, the recovery turned it away from
        # GOLD, and both backward steps were refused).  Walk to it directly.
        centre_start = _robot_centre(cur_pose)
        gold_distance_m, gold_centre_xy = world_map.nearest_gold_scan(centre_start)
        if gold_centre_xy is not None and gold_distance_m > 0.30:
            to_gold_start = gold_centre_xy - centre_start
            start_bearing_deg = (
                (
                    math.degrees(math.atan2(float(to_gold_start[1]), float(to_gold_start[0])))
                    - (float(cur_pose.theta_deg) + forward_offset)
                    + 180.0
                )
                % 360.0
            ) - 180.0
            if abs(start_bearing_deg) <= 75.0:
                print(
                    f"[localize] trusted GOLD geometry already lies ahead "
                    f"({gold_distance_m:.2f}m at {start_bearing_deg:+.0f}deg); skipping "
                    "the 180deg turnaround and walking to it directly."
                )
                return _reference_walk(0.0, turned=False)

        # The blind 180deg opposite-view turnaround is ELIMINATED (user
        # directive 2026-08-05 evening).  The robot has odometry, the
        # continuity bubble, and the GOLD reference walk — which itself
        # rotates to FACE the reference whenever it bears more than 60deg
        # off and walks to it in collision-checked steps.  Recovery is
        # simply: go to the nearest trusted geometry and vote there.  The
        # turn machinery below is intentionally unreachable and kept only
        # until the next cleanup pass.
        return _reference_walk(0.0, turned=False)

        yaw_initial = imu.deg_fresh(wait_up_to_s=0.6, max_age_s=0.4)
        if yaw_initial is None:
            print("[localize] turnaround recovery unavailable: IMU yaw is stale.")
            return False
        positive_clear = _rotation_safety_check(180.0) is None
        negative_clear = _rotation_safety_check(-180.0) is None
        if not positive_clear and not negative_clear:
            print(
                "[localize] turnaround recovery blocked: both 180-degree "
                "rotational footprints are occupied."
            )
            return False
        direction = 1.0 if positive_clear else -1.0
        # A collision latch can interrupt the first 180-degree command after
        # only part of the turn (the previous implementation then abandoned
        # the run even though the opposite direction was still available).
        # Try the other direction once, carrying the measured partial yaw into
        # the pose.  There is no ping-pong: at most two bounded turn commands.
        for turn_attempt in range(2):
            yaw_before = imu.deg_fresh(wait_up_to_s=0.6, max_age_s=0.4)
            if yaw_before is None:
                print("[localize] turnaround recovery failed: no fresh pre-turn yaw.")
                return False
            if _rotation_safety_check(direction * 180.0) is not None:
                if turn_attempt == 0:
                    direction = -direction
                    continue
                print("[localize] turnaround recovery blocked in both turn directions.")
                return False
            target = float(yaw_before) + direction * 180.0
            print(
                f"[localize] bounded turnaround recovery: rotating 180deg "
                f"{'counterclockwise' if direction > 0 else 'clockwise'} "
                f"(attempt {turn_attempt + 1}/2) to reacquire the opposite room view."
            )
            controller.clear_safety_latch()
            controller.rotate_to(target)
            started = time.monotonic()
            try:
                while time.monotonic() - started < 14.0:
                    if controller.safety_latched_reason() is not None:
                        break
                    yaw_now = imu.deg()
                    if yaw_now is not None and abs(float(yaw_now) - float(yaw_before)) >= 174.0:
                        break
                    time.sleep(0.05)
            finally:
                controller.halt()
                time.sleep(0.30)
            yaw_after = imu.deg_fresh(wait_up_to_s=0.6, max_age_s=0.4)
            if yaw_after is None:
                print("[localize] turnaround recovery failed: no fresh post-turn yaw.")
                return False
            actual = float(yaw_after) - float(yaw_before)
            if abs(actual) > 2.0:
                cur_pose = _rotate_lidar_pose_about_robot_centre(
                    cur_pose,
                    actual,
                    lever_m,
                    forward_offset,
                )
                _discard_translation_prior()
                trail.append(_robot_centre(cur_pose).copy())
                active_localization_state["swept_deg"] = 0.0
                active_localization_state["direction"] = 0.0
            if abs(actual) < 120.0:
                print(
                    f"[localize] turnaround attempt {turn_attempt + 1} made only "
                    f"{actual:+.1f}deg; trying the opposite turn direction once."
                )
                direction = -direction
                continue
            print(
                f"[localize] turnaround completed ({actual:+.1f}deg); "
                "matching fresh full LiDAR views to nearby remembered GOLD geometry."
            )
            # Mid-corridor there is often no distinctive GOLD geometry within
            # the local bubble at all — the reference that can fix the pose is
            # the previous room's doorway/corner, a bounded distance back
            # along the approach.  Facing that way now, walk toward it in
            # collision-checked steps, retrying the opposite-view
            # reacquisition and the standard bubble vote after each step.
            # Once the reference is recovered, turn back around and carry the
            # re-anchored pose down the corridor in one continuous odometry
            # beeline instead of replanning in place.
            return _reference_walk(actual, turned=True)
        return False

    def _active_localization_probe(_attempt: int) -> bool:
        """Advance a bounded one-direction scan fan; never ping-pong headings."""
        nonlocal cur_pose
        probe = abs(float(args.active_localization_probe_deg))
        maximum = abs(float(args.active_localization_max_sweep_deg))
        # Bracket the maneuver with fresh samples. Heavy GPU scan matching can
        # leave ``deg()`` valid but several seconds old; a stale baseline makes
        # a bounded physical turn look arbitrarily large.
        yaw_before = imu.deg_fresh(wait_up_to_s=0.5, max_age_s=0.3)
        if yaw_before is None:
            return False
        if bool(active_localization_state["rotation_stalled"]):
            if not bool(active_localization_state["translation_attempted"]):
                active_localization_state["translation_attempted"] = True
                return _active_localization_translation()
            print(
                "[localize] rotational probe and one parallax trajectory are "
                "exhausted; retaining the continuous pose prior for a fresh "
                "supported-overlap vote instead of repeating a zero-motion turn."
            )
            return False
        if probe < 1.0:
            if not bool(active_localization_state["translation_attempted"]):
                active_localization_state["translation_attempted"] = True
                return _active_localization_translation()
            return False
        remaining = maximum - float(active_localization_state["swept_deg"])
        if remaining < 2.0:
            if not bool(active_localization_state["translation_attempted"]):
                active_localization_state["translation_attempted"] = True
                return _active_localization_translation()
            print(
                "[localize] active-localization maneuver is exhausted; keeping the "
                "base still rather than oscillating over the same headings."
            )
            return False
        direction = float(active_localization_state["direction"])
        if direction == 0.0:
            positive_clear = _rotation_safety_check(maximum) is None
            negative_clear = _rotation_safety_check(-maximum) is None
            if positive_clear:
                direction = 1.0
            elif negative_clear:
                direction = -1.0
            else:
                positive_clear = _rotation_safety_check(probe) is None
                negative_clear = _rotation_safety_check(-probe) is None
                if positive_clear:
                    direction = 1.0
                elif negative_clear:
                    direction = -1.0
                else:
                    print(
                        "[localize] no collision-free localization sweep direction; "
                        "attempting a safe parallax translation instead."
                    )
                    active_localization_state["translation_attempted"] = True
                    return _active_localization_translation()
            active_localization_state["direction"] = direction
            print(
                f"[localize] starting a bounded {'left' if direction > 0 else 'right'} "
                f"localization sweep (up to {maximum:.0f}deg); heading direction is "
                "latched until recovery completes."
            )
        step = min(probe, remaining)
        target = float(yaw_before) + direction * step
        controller.clear_safety_latch()
        controller.rotate_to(target)
        started = time.monotonic()
        turn_status = "turning"
        while time.monotonic() - started < 4.0:
            if controller.safety_latched_reason() is not None:
                break
            yaw_now = imu.deg()
            if yaw_now is not None:
                turn_status = _bounded_turn_status(
                    direction * step,
                    float(yaw_now) - float(yaw_before),
                )
                if turn_status != "turning":
                    break
            time.sleep(0.05)
        controller.halt()
        yaw_after = imu.deg_fresh(wait_up_to_s=0.5, max_age_s=0.3)
        if yaw_after is None:
            return False
        actual = float(yaw_after) - float(yaw_before)
        cur_pose = _rotate_lidar_pose_about_robot_centre(
            cur_pose,
            actual,
            lever_m,
            forward_offset,
        )
        if turn_status == "wrong_direction":
            # The IMU is the source of truth for the physical response. Repair
            # the command polarity learned at startup and stop here; never let
            # an erroneous 15-degree probe become a 100+ degree spin.
            controller.turn_sign *= -1.0
            active_localization_state["direction"] = 0.0
            active_localization_state["rotation_stalled"] = False
            print(
                f"[localize] recovery turn moved the wrong way ({actual:+.1f}deg); "
                "halted immediately and corrected the runtime angular-command "
                "polarity. No further recovery turn is issued this cycle."
            )
            return True
        if turn_status == "overshoot":
            active_localization_state["rotation_stalled"] = True
            print(
                f"[localize] recovery turn exceeded its {step:.0f}deg bound "
                f"({actual:+.1f}deg measured); halted and disabled further "
                "rotation for this recovery cycle."
            )
            return True
        progress = max(0.0, direction * actual)
        active_localization_state["swept_deg"] = min(
            maximum,
            float(active_localization_state["swept_deg"]) + progress,
        )
        print(
            f"[localize] monotonic localization sweep turned {actual:+.1f}deg "
            f"({float(active_localization_state['swept_deg']):.1f}/"
            f"{maximum:.0f}deg) within the live collision envelope."
        )
        if _active_localization_rotation_stalled(progress):
            active_localization_state["rotation_stalled"] = True
            print(
                "[localize] rotational localization probe made no physical "
                "progress; switching immediately to collision-checked parallax "
                "instead of retrying the same turn."
            )
            if not bool(active_localization_state["translation_attempted"]):
                active_localization_state["translation_attempted"] = True
                return _active_localization_translation()
            return False
        return True

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
        theta_start = float(cur_pose.theta_deg)  # map heading at yaw_start (gyro-carried)
        pano: list[np.ndarray] = []
        prev_id = feed.latest()[0]
        t0 = time.monotonic()
        controller.rotate_to(float(yaw_start) + 370.0)
        try:
            while True:
                yaw_now = imu.deg()
                blocked_reason = controller.safety_latched_reason()
                if blocked_reason is not None:
                    print(
                        f"[explore]   recovery spin refused: {blocked_reason}; "
                        "remaining at the current safe heading."
                    )
                    return False
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
        # At a doorway most panorama returns can belong to the new room and are
        # absent from GOLD by definition. Match only rays that the current seed
        # places near already-observed geometry; otherwise new information
        # overwhelms the overlap and makes a good recovery score look terrible.
        known_support = _match_support_mask(_transform_points(pano_pts, seed))
        if int(known_support.sum()) >= 100:
            print(
                f"[explore]   recovery using {int(known_support.sum())}/{len(pano_pts)} "
                "known-map panorama points; new-room returns ignored."
            )
            pano_pts = pano_pts[known_support]
        solved, score = _localize(
            pano_pts,
            world_map,
            seed,
            args,
            1.20,
            25.0,
            coarse_angle_step_deg=4.0,
            allow_whole_map_search=True,
        )
        if score < 12.0:
            print(f"[explore]   panorama match too weak ({score:.1f}) — recovery failed.")
            return False
        # Panorama frame origin = the SPIN CENTRE; convert back to the lidar pose
        # at the robot's current heading (gyro delta accumulated over the spin).
        yaw_end = imu.deg()
        d_h = (float(yaw_end) - float(yaw_start)) if yaw_end is not None else 360.0
        theta_new = float(solved.theta_deg) + d_h
        recovered_pose = _lidar_pose_from_robot_centre(
            np.array([float(solved.x), float(solved.y)], dtype=np.float64),
            theta_new,
            lever_m,
            forward_offset,
        )
        if not _pose_stays_beyond_completed_doorways(recovered_pose):
            print("[explore]   panorama solution crossed a completed doorway — rejected.")
            return False
        predicted_recovery_pose = cur_pose
        cur_pose = recovered_pose
        _discard_translation_prior()
        _rezero_gyro_heading_reference(
            float(recovered_pose.theta_deg), "panorama re-anchor"
        )
        _register_stationary_fix(
            predicted_recovery_pose,
            recovered_pose,
            None,
            float(score),
            axis_locked=False,
            odometry_broken=True,
        )
        print(f"[explore]   pose RE-ANCHORED by panorama (match {score:.1f}).")
        return True

    trail: list[np.ndarray] = [_robot_centre(cur_pose)]
    # The first surface label is attached only now, after the initial LiDAR
    # anchor has passed validation.  This makes the map's texture regions as
    # trustworthy as its initial geometry.
    _record_surface_region_at_trusted_pose(cur_pose)
    saved_map_revision = [-1]

    def _save_current_map() -> None:
        if bool(args.no_save_map) or not world_map.scans:
            return
        revision = int(world_map.grid.version)
        if revision == saved_map_revision[0]:
            return
        try:
            destination = save_world_map(
                args.saved_map,
                world_map,
                cur_pose,
                trail=trail,
                sensor_config={
                    "forward_angle_deg": float(args.forward_angle_deg),
                    "valid_angle_half_width_deg": float(args.valid_angle_half_width_deg),
                    "invert_lateral_axis": bool(args.invert_lateral_axis),
                    "max_distance_m": float(args.max_distance_m),
                    "min_range_m": float(args.min_range_m),
                    "min_confidence": float(args.min_confidence),
                    "stitch_resolution_m": float(args.stitch_resolution_m),
                    "match_max_points": int(args.match_max_points),
                },
                navigation_config={
                    "robot_radius_m": float(args.robot_radius_m),
                    "physical_body_radius_m": float(args.physical_body_radius_m),
                    "collision_self_mask_inset_m": float(args.collision_self_mask_inset_m),
                },
                visual_color_landmarks=visual_color_landmarks,
                surface_motion_atlas=surface_motion_atlas.to_metadata(),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[map] WARNING: could not save persistent map: {exc}")
            return
        saved_map_revision[0] = revision
        print(f"[map] saved {len(world_map.scans)} scans and current pose to {destination}.")

    # Covers Ctrl+C and ordinary interpreter shutdown. The explicit call after
    # the mission loop saves before the viewer is left open; this callback is a
    # final guard for an interruption during exploration.
    atexit.register(_save_current_map)
    # Frontier completion is observation memory, not a permanent blacklist.
    # A frontier can remain detectable after one weak/occluded view or move as
    # unknown space is revealed.  Keep completed/failed objectives on a bounded
    # cooldown, then make unresolved geometry eligible again.
    retired_frontiers: list[tuple[np.ndarray, int]] = []
    # Multiple collision-checked approaches can prove a frontier temporarily
    # unreachable in the current costmap. Keep that state separate from normal
    # observation cooldowns so the indefinite explorer cannot immediately age it
    # out and drive back into the same corner.
    deferred_frontiers: list[tuple[np.ndarray, int]] = []
    sampled_viewpoints: list[np.ndarray] = []
    transient_obstacles: list[tuple[np.ndarray, float]] = []
    transient_obstacle_revision = [0]
    analysis_cache: list[object | None] = [None, None]
    # Local-planner failure memory. A collision-limited navigation goal is not
    # evidence that its whole frontier is invalid, but immediately selecting
    # the same goal again creates an infinite stop/snapshot loop. Keep the
    # frontier and exclude only the failed approach pose.
    # Failed local approaches are transient costmap evidence, not permanent
    # obstacles. Entries expire after several global replans so changed sensor
    # geometry or a successful recovery can reopen the approach.
    blocked_viewpoints: list[tuple[np.ndarray, int]] = []
    planning_epoch = 0

    def _retire_frontier(frontier_xy: np.ndarray, cooldown_epochs: int = 12) -> None:
        point = np.asarray(frontier_xy, dtype=np.float64).copy()
        retired_frontiers.append((point, planning_epoch + max(2, int(cooldown_epochs))))
        if len(retired_frontiers) > 64:
            del retired_frontiers[:-64]

    def _defer_frontier(frontier_xy: np.ndarray, cooldown_epochs: int = 24) -> None:
        point = np.asarray(frontier_xy, dtype=np.float64).copy()
        deferred_frontiers.append(
            (
                point,
                planning_epoch + max(6, int(cooldown_epochs)),
            )
        )
        if len(deferred_frontiers) > 64:
            del deferred_frontiers[:-64]

    # The global planner owns exactly one frontier until it is observed away,
    # proven unreachable/blocked, or the mission is stopped. Without this state,
    # every localization hiccup re-ran global scoring and could send the robot
    # straight back to a frontier behind it while claiming to retry the same one.
    active_frontier_xy: np.ndarray | None = None
    # One batch is a frozen, numbered set of observation poses.  It is made
    # immediately after the anchor (and after each completed batch), then the
    # robot visits marker 1, 2, and 3 in that order.  Mapping may replan the
    # route to a marker, but it may not replace the remaining markers midway.
    observation_batch: list[
        tuple[FrontierCluster, np.ndarray, list[np.ndarray], bool, np.ndarray, int]
    ] = []
    observation_batch_completed = 0
    forward_batch_remaining = 0

    def _invalidate_pre_recovery_navigation_objectives(reason: str) -> None:
        """Discard routes selected before a recovered map pose was available.

        A stationary recovery can legitimately shift the robot by a few
        decimetres.  A waypoint list or frozen frontier selected from the old
        pose then encodes the wrong approach side of a doorway.  Continuing it
        is not ``preserving the objective``; it is executing stale geometry.
        Recompute the frontier set from the recovered pose before another base
        command is sent.
        """
        nonlocal active_frontier_xy, observation_batch, observation_batch_completed
        nonlocal forward_batch_remaining
        discarded_markers = len(observation_batch)
        had_frontier = active_frontier_xy is not None
        observation_batch = []
        observation_batch_completed = 0
        forward_batch_remaining = 0
        active_frontier_xy = None
        _log_observation_batch_markers(rr, observation_batch)
        print(
            "[plan] discarded "
            f"{discarded_markers} frozen observation marker(s)"
            + (" and the active frontier" if had_frontier else "")
            + f" after {reason}; rebuilding targets from the recovered pose and map."
        )

    obstacle_strikes: dict[tuple[int, int], int] = {}  # frontier -> times obstacle-blocked
    lost_strikes: dict[tuple[int, int], int] = {}  # frontier -> times tracking-lost
    zero_motion_strikes: dict[tuple[int, int], int] = {}
    blocked_approach_history: dict[tuple[int, int], list[np.ndarray]] = {}
    # Bound retries even when every collision occurs at the exact same pose.
    # Distinct-approach history intentionally ignores duplicates, which would
    # otherwise permit an endless plan/scan/replan cycle at one doorway.
    rotation_block_strikes: dict[tuple[int, int], int] = {}
    viewpoints_reached = 0
    spin_scan_count = len(world_map.scans)
    frontiers_left = 0
    none_retries = 0
    # A stationary scan can reveal a little additional geometry, but it must
    # not turn "no route" into an endless scan/replan loop at an otherwise
    # open pose.  After two such refreshes we force the normal patrol/return
    # decision below instead of pretending another identical scan is motion.
    no_route_stationary_refreshes = 0
    rear_route_deferrals = 0
    vp_last: list = [None]  # last _integrate_at_viewpoint result
    vp_gain_last: list[int | None] = [None]  # actual unknown -> known cells
    vp_pose_ok_last = [False]  # stationary batch or fallback localized successfully
    # Provisional stationary scans are useful for occupancy growth, but a
    # stopped robot must not write duplicate copies of the same view every
    # checkpoint.  GOLD scans remain the only localization reference.
    last_provisional_stationary_centre: list[np.ndarray | None] = [None]
    last_provisional_stationary_heading: list[float | None] = [None]
    consistency_fails = 0  # consecutive discarded/empty viewpoints

    def _active_transient_obstacle_points() -> np.ndarray:
        if not bool(obstacle_pose_trust["trusted"]):
            if transient_obstacles:
                transient_obstacles.clear()
                transient_obstacle_revision[0] += 1
                rr.log(
                    "world/obstacle",
                    rr.Points3D(np.empty((0, 3), dtype=np.float32)),
                )
            return np.empty((0, 2), dtype=np.float32)
        now = time.monotonic()
        previous_count = len(transient_obstacles)
        transient_obstacles[:] = [
            (points, expires_at) for points, expires_at in transient_obstacles if expires_at > now
        ]
        if len(transient_obstacles) != previous_count:
            transient_obstacle_revision[0] += 1
        active = [points for points, _expires_at in transient_obstacles if len(points)]
        if not active:
            return np.empty((0, 2), dtype=np.float32)
        return np.concatenate(active, axis=0).astype(np.float32, copy=False)

    def _remember_transient_obstacle(world_points: np.ndarray) -> None:
        # Defense in depth: every caller, including hard-stop paths, must obey
        # the same pose-confidence contract.  Raw body-frame collision safety
        # may stop the base, but an uncertain world transform may never paint a
        # temporary wall into the planner.
        if not bool(obstacle_pose_trust["trusted"]):
            return
        points = np.asarray(world_points, dtype=np.float32)
        if not len(points):
            return
        # Bound the local costmap layer and deduplicate returns at grid-cell
        # resolution. It expires independently of the immutable SLAM posterior.
        cells = np.round(points / float(args.frontier_res_m)).astype(np.int64)
        _unique, indices = np.unique(cells, axis=0, return_index=True)
        transient_obstacles.append(
            (
                points[np.sort(indices)],
                time.monotonic() + float(args.obstacle_memory_s),
            )
        )
        transient_obstacle_revision[0] += 1
        if len(transient_obstacles) > 12:
            del transient_obstacles[:-12]

    def _analysis_now() -> Analysis:
        dynamic = _active_transient_obstacle_points()
        cache_key = (
            int(world_map.grid.version),
            int(transient_obstacle_revision[0]),
        )
        if analysis_cache[0] == cache_key and isinstance(analysis_cache[1], Analysis):
            return analysis_cache[1]
        analysis_started = time.monotonic()
        if len(dynamic):
            xyz = np.column_stack([dynamic[:, 0], dynamic[:, 1], np.zeros(len(dynamic), dtype=np.float32)])
            rr.log(
                "world/obstacle",
                rr.Points3D(
                    xyz.astype(np.float32),
                    colors=[[255, 255, 255]] * len(xyz),
                    radii=0.03,
                ),
            )
        else:
            rr.log(
                "world/obstacle",
                rr.Points3D(np.empty((0, 3), dtype=np.float32)),
            )
        result = analyze_grid(
            world_map.grid,
            float(args.robot_radius_m),
            float(args.min_frontier_span_m),
            int(args.min_frontier_cells),
            float(args.passable_opening_min_m),
            extra_occupied_xy=dynamic,
        )
        analysis_cache[:] = [cache_key, result]
        print(
            f"[perf] occupancy/frontier analysis "
            f"{time.monotonic() - analysis_started:.3f}s "
            f"(grid revision {cache_key[0]}, obstacle revision {cache_key[1]})."
        )
        return result

    # --- TWO support masks, deliberately separate (field 2026-07-21): -------
    # MATCHING support comes ONLY from actually-integrated scan points, NEVER
    # from planner marks. When match-support read the grid, hard-stop marks
    # (including arm self-returns) became geometry the matcher aligned to: poses
    # warped onto phantoms, viewpoints painted real walls at aliased offsets
    # (the fake corridor), marks snowballed (the phantom barrier; traversable
    # 1058->570). Localization must not trust anything it did not observe.
    _match_sup_cache: list = [None, None]  # [n scans, cell set]

    def _match_support_mask(world_pts: np.ndarray) -> np.ndarray:
        res = float(args.novel_res_m)
        # Key on pose_revision too: the graph optimizer can move scan poses
        # without changing the scan count.
        key = (len(world_map.scans), world_map.pose_revision)
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
            out[i] = any((ci + di, cj + dj) in occ_set for di in (-1, 0, 1) for dj in (-1, 0, 1))
        return out

    _support_cache: list = [None, None]  # [grid version, inflated-occupied mask]

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
        dynamic = _active_transient_obstacle_points()
        if len(dynamic) and len(world_pts):
            # The layer is small and short-lived; chunked Euclidean support is
            # clearer and safer here than mutating the SLAM grid/cache.
            for start in range(0, len(world_pts), 512):
                query = world_pts[start : start + 512]
                near = np.any(
                    np.sum((query[:, None, :] - dynamic[None, :, :]) ** 2, axis=2) <= 0.30**2,
                    axis=1,
                )
                out[start : start + len(query)] |= near
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
        if not bool(obstacle_pose_trust["trusted"]):
            if not bool(obstacle_pose_trust["warning_emitted"]):
                obstacle_pose_trust["warning_emitted"] = True
                print(
                    "[explore] world-frame obstacle insertion QUARANTINED while "
                    "localization is uncertain ("
                    f"{obstacle_pose_trust['reason']}). Raw collision-envelope "
                    "safety remains active; no phantom planner wall can be created."
                )
            return False, None, None
        if local is None or len(local) == 0:
            return False, None, None
        ff = _to_forward_frame(local, forward_offset)
        inbox = (
            (ff[:, 0] > float(args.box_near_m))
            & (ff[:, 0] < float(args.box_depth_m))
            & (np.abs(ff[:, 1]) < float(args.box_half_width_m))
        )
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
            # Dynamic returns belong to the local planner layer. They must not
            # modify the SLAM posterior: doing so created invisible walls,
            # changed frontier topology, and could strand the robot behind a
            # non-existent obstacle even though no keyframe saw permanent
            # geometry there.
            _remember_transient_obstacle(last_novel)
            fwd = float(np.mean(last_body[:, 0]))
            lat = float(np.mean(last_body[:, 1]))
            rng = float(np.min(np.hypot(last_body[:, 0], last_body[:, 1])))
            print(
                f"[explore] obstacle persisted; placed {len(last_novel)} pts in the "
                f"temporary planner layer for {float(args.obstacle_memory_s):.0f}s "
                f"(BODY frame: fwd {fwd:+.2f}m lat {lat:+.2f}m nearest {rng:.2f}m — if these "
                "numbers repeat at every heading, it is attached to the robot: check the arm "
                "stow height vs the 0.28m scan plane). SLAM geometry was unchanged; "
                "replanning around it."
            )
            xyz = np.column_stack([last_novel[:, 0], last_novel[:, 1], np.zeros(len(last_novel))])
            rr.log(
                "world/obstacle",
                rr.Points3D(xyz.astype(np.float32), colors=[[255, 255, 255]] * len(xyz), radii=0.03),
            )
        return "replan"

    def _face_snapshot_target(target_xy: np.ndarray, label: str) -> bool:
        """Pivot so physical forward points directly at the requested map point."""
        nonlocal cur_pose
        controller.halt()
        target = np.asarray(target_xy, dtype=np.float64)
        vec = target - _robot_centre(cur_pose)
        if float(np.hypot(*vec)) <= 0.05:
            return True
        alpha = math.degrees(math.atan2(vec[1], vec[0]))
        error = ((alpha - forward_offset - float(cur_pose.theta_deg) + 180.0) % 360.0) - 180.0
        if abs(error) <= 3.0:
            return True
        yaw_before = imu.deg()
        if yaw_before is None:
            return False
        target_yaw = float(yaw_before) + error
        print(f"[explore] facing {label}: pivot {error:+.0f}deg before snapshot.")
        controller.rotate_to(target_yaw)
        started = time.monotonic()
        blocked_reason = None
        while time.monotonic() - started < 6.0:
            blocked_reason = controller.safety_latched_reason()
            if blocked_reason is not None:
                break
            yaw_now = imu.deg()
            if yaw_now is not None and abs(target_yaw - float(yaw_now)) <= 3.0:
                break
            time.sleep(0.05)
        controller.halt()
        yaw_after = imu.deg()
        if yaw_after is None:
            return False
        actual = float(yaw_after) - float(yaw_before)
        cur_pose = _rotate_lidar_pose_about_robot_centre(
            cur_pose,
            actual,
            lever_m,
            forward_offset,
        )
        if blocked_reason is not None:
            print(
                f"[explore] snapshot pivot refused: {blocked_reason}; "
                "mapping from the current safe heading instead."
            )
            return False
        return abs(target_yaw - float(yaw_after)) <= 5.0

    last_leg_mapping: dict[str, object] = {
        "scans": 0,
        "gain": 0,
        "best_gain": 0,
        "best_pose": None,
    }

    # =======================================================================
    # EXPLORE PHASE 3 — FOLLOW THE SELECTED PATH
    #
    # This function owns turn/straight-drive execution for one selected target.
    # Its successful arrival flows directly into Phase 4 at the bottom.
    # =======================================================================
    def _drive_leg(
        waypoints,
        goal_xy,
        analysis,
        face_xy=None,
        *,
        overlap_handoff: bool = False,
        rolling_submap: bool = False,
        continuing_doorway: bool = False,
        corridor_carry_only: bool = False,
    ) -> str:
        """Execute a path as PIVOT -> STRAIGHT -> PIVOT -> STRAIGHT segments.

        Each straight run holds one immutable IMU heading. Scan matching corrects
        position but never retargets steering mid-segment, eliminating the
        pure-pursuit micro-adjustment zig-zags. Safety remains live at 25Hz.
        """
        nonlocal cur_pose
        last_leg_mapping.update(scans=0, gain=0, best_gain=0, best_pose=None)
        pending_collision_local: list[np.ndarray | None] = [None]
        collision_streak = {"frame_id": None, "sector": None, "count": 0}
        handoff_added = 0
        handoff_gain = 0
        handoff_pose_ok = False
        handoff_attempted = False
        checkpoint_failures = 0
        leg_start_centre = _robot_centre(cur_pose).copy()
        last_handoff_centre = leg_start_centre.copy()
        last_handoff_heading: float | None = None
        handoff_keyframes: list[tuple[np.ndarray, float]] = []

        def _record_mapping_result() -> tuple[int, int]:
            added_now = int(vp_last[0] or 0)
            gain_now = int(vp_gain_last[0] or 0)
            last_leg_mapping["scans"] = int(last_leg_mapping["scans"]) + added_now
            last_leg_mapping["gain"] = int(last_leg_mapping["gain"]) + gain_now
            if gain_now > int(last_leg_mapping["best_gain"]):
                last_leg_mapping["best_gain"] = gain_now
                last_leg_mapping["best_pose"] = _robot_centre(cur_pose).copy()
            return added_now, gain_now

        def _capture_handoff(
            *,
            force: bool = False,
            doorway_keyframe: bool = False,
        ) -> bool:
            nonlocal handoff_added, handoff_gain, handoff_pose_ok, handoff_attempted
            nonlocal last_handoff_centre, last_handoff_heading
            centre_now = _robot_centre(cur_pose)
            heading_now = float(cur_pose.theta_deg) + forward_offset
            duplicate_view = next(
                (
                    (old_centre, old_heading)
                    for old_centre, old_heading in reversed(handoff_keyframes)
                    if not _keyframe_view_is_diverse(
                        centre_now,
                        heading_now,
                        old_centre,
                        old_heading,
                    )
                ),
                None,
            )
            if not force and duplicate_view is not None:
                old_centre, old_heading = duplicate_view
                turned = abs((heading_now - old_heading + 180.0) % 360.0 - 180.0)
                translated = float(np.hypot(*(centre_now - old_centre)))
                print(
                    f"[explore]   doorway keyframe skipped: only "
                    f"{translated * 100:.0f}cm/{turned:.0f}deg from the last "
                    "accepted view."
                )
                return handoff_pose_ok
            handoff_attempted = True
            _integrate_at_viewpoint(
                # A doorway handoff deliberately demands overlap with the old
                # GOLD map so it can extend the permanent reference chain.
                # A normal map-growth checkpoint must not impose that same
                # restriction: its job is precisely to record the newly seen
                # geometry once a fresh stationary full-view solve has
                # verified the current pose.
                require_known_overlap=bool(doorway_keyframe),
                scan_count=(int(args.frontier_keyframe_batch_scans) if doorway_keyframe else 3),
            )
            added_now, gain_now = _record_mapping_result()
            handoff_added += added_now
            handoff_gain += gain_now
            handoff_pose_ok = handoff_pose_ok or bool(vp_pose_ok_last[0])
            if vp_pose_ok_last[0]:
                last_handoff_centre = _robot_centre(cur_pose).copy()
                last_handoff_heading = float(cur_pose.theta_deg) + forward_offset
                handoff_keyframes.append(
                    (
                        last_handoff_centre.copy(),
                        last_handoff_heading,
                    )
                )
                if len(handoff_keyframes) > 64:
                    del handoff_keyframes[:-64]
                if doorway_keyframe:
                    keyframe_xyz = np.asarray(
                        [[float(point[0]), float(point[1]), 0.04] for point, _heading in handoff_keyframes],
                        dtype=np.float32,
                    )
                    rr.log(
                        "world/doorway_keyframes",
                        rr.Points3D(
                            keyframe_xyz,
                            colors=[[60, 220, 255]] * len(keyframe_xyz),
                            radii=0.035,
                        ),
                    )
            return bool(vp_pose_ok_last[0])

        def _turn_handoff_to(
            target_yaw: float,
            label: str,
            *,
            capture_steps: bool,
        ) -> bool:
            """Collision-checked bounded turn, carrying the LiDAR lever pose."""
            nonlocal cur_pose
            max_step = max(
                10.0,
                min(45.0, float(args.frontier_handoff_turn_step_deg)),
            )
            for _step_index in range(12):
                yaw_before = imu.deg()
                if yaw_before is None:
                    return False
                remaining = float(target_yaw) - float(yaw_before)
                if abs(remaining) <= 3.0:
                    return True
                step = _bounded_heading_step(remaining, max_step)
                step_target = float(yaw_before) + step
                controller.clear_safety_latch()
                controller.rotate_to(step_target)
                started = time.monotonic()
                blocked_reason = None
                while time.monotonic() - started < 6.0:
                    blocked_reason = controller.safety_latched_reason()
                    if blocked_reason is not None:
                        break
                    yaw_now = imu.deg()
                    if yaw_now is not None and abs(step_target - float(yaw_now)) <= 3.0:
                        break
                    time.sleep(0.05)
                controller.halt()
                yaw_after = imu.deg()
                if yaw_after is None:
                    return False
                actual = float(yaw_after) - float(yaw_before)
                cur_pose = _rotate_lidar_pose_about_robot_centre(
                    cur_pose,
                    actual,
                    lever_m,
                    forward_offset,
                )
                if blocked_reason is not None:
                    print(f"[explore] {label} skipped unsafe heading: {blocked_reason}.")
                    controller.clear_safety_latch()
                    return False
                if abs(actual) < 2.0:
                    print(f"[explore] {label} turn made no progress; keeping the last keyframe.")
                    return False
                if capture_steps and not _capture_handoff(doorway_keyframe=True):
                    print(
                        f"[explore] {label} stopped: the next angular keyframe "
                        "lacked trusted local-submap overlap."
                    )
                    return False
            return abs(float(target_yaw) - float(imu.deg() or target_yaw)) <= 5.0

        def _capture_overlap_turn(
            label: str,
            target_xy: np.ndarray | None,
            *,
            max_centre_arc_deg: float,
        ) -> bool:
            """Build overlap keyframes along one monotonic turn toward the target.

            The first keyframe overlaps the established map. Subsequent views
            are matched after the preceding accepted keyframe has joined GOLD,
            forming a sequential local-submap chain through the doorway. The
            old left-centre-right-centre fan revisited old headings, caused
            visible backtracking, and repeatedly overweighted the same wall.
            """
            controller.halt()
            print(f"[explore] {label}: building monotonic overlap keyframes.")
            accepted_before = handoff_added
            if not _capture_handoff(
                force=last_handoff_heading is None,
                doorway_keyframe=True,
            ):
                print(f"[explore] {label}: starting keyframe lacked trusted overlap.")
                return False
            yaw_start = imu.deg()
            if yaw_start is None:
                return handoff_added > accepted_before

            if target_xy is not None:
                target = np.asarray(target_xy, dtype=np.float64)
                vector = target - _robot_centre(cur_pose)
                if float(np.hypot(*vector)) > 0.05:
                    desired_map = math.degrees(math.atan2(vector[1], vector[0])) - forward_offset
                    error = ((desired_map - float(cur_pose.theta_deg) + 180.0) % 360.0) - 180.0
                    limited = float(
                        np.clip(
                            error,
                            -abs(float(max_centre_arc_deg)),
                            abs(float(max_centre_arc_deg)),
                        )
                    )
                    if abs(limited) >= 3.0:
                        _turn_handoff_to(
                            float(yaw_start) + limited,
                            f"{label} target sweep",
                            capture_steps=True,
                        )
            print(
                f"[explore] {label}: accepted "
                f"{handoff_added - accepted_before} scan(s) along one outward turn."
            )
            return handoff_added > accepted_before

        navigation_start_cell = _snap_traversable(
            analysis,
            _robot_centre(cur_pose),
            r_m=0.25,
        )
        navigation_start_reference = (
            analysis.to_world(navigation_start_cell) if navigation_start_cell is not None else None
        )
        corridor_translation_active = [False]

        def _fast_safety_check() -> str | None:
            pose = cur_pose
            centre = _robot_centre(pose)
            state, distance = _swept_footprint_state(
                analysis,
                centre,
                float(pose.theta_deg) + forward_offset,
                swept_guard_m,
                planned_start_xy=navigation_start_reference,
            )
            if state == "unknown":
                return f"map edge {distance:.2f}m ahead"
            # With a calibrated envelope, the raw LiDAR is the immediate
            # physical-clearance authority. Inflated-map corrections can put the
            # estimated centre a few centimetres inside its own wall and caused
            # the old guard to stop repeatedly at 0.00-0.10m. The map still owns
            # unknown-space stopping and global path planning.
            if state == "blocked" and collision_profile is None:
                return f"mapped full-body clearance {distance:.2f}m ahead"
            frame_id, frame = feed.latest()
            if frame is None:
                return "LiDAR safety unavailable (no scan)"
            frame_age_s = time.time() - float(getattr(frame, "received_wall_ts", 0.0) or 0.0)
            if not math.isfinite(frame_age_s) or frame_age_s > float(args.lidar_safety_max_age_s):
                return f"LiDAR safety stale ({max(0.0, frame_age_s):.2f}s old)"
            local = _scan_local(frame, args)
            if not len(local):
                return "LiDAR safety unavailable (empty scan)"
            ff = _to_forward_frame(local, forward_offset)
            calibrated_hit = _collision_box_violation(
                ff,
                collision_profile,
                lidar_offset_forward_m=lever_m,
                physical_body_radius_m=float(args.physical_body_radius_m),
                self_mask_inset_m=float(args.collision_self_mask_inset_m),
            )
            if calibrated_hit is not None:
                if corridor_translation_active[0] and _translation_trajectory_is_safe(
                    ff,
                    collision_profile,
                    np.array([0.10, 0.0], dtype=np.float64),
                    lidar_offset_forward_m=lever_m,
                    physical_body_radius_m=float(args.physical_body_radius_m),
                    self_mask_inset_m=float(args.collision_self_mask_inset_m),
                ):
                    # Permit only a non-worsening straight doorway translation.
                    # The same calibrated rollout is rechecked on every fresh
                    # revolution; rotation and motion deeper into contact remain
                    # prohibited.
                    return None
                mask, sector, angle, hit_range, limit = calibrated_hit
                # Count only fresh LiDAR revolutions. Re-reading one frame at
                # 25Hz must not turn a single glint/self-return into a stop.
                if frame_id != collision_streak["frame_id"]:
                    if sector == collision_streak["sector"]:
                        collision_streak["count"] += 1
                    else:
                        collision_streak["sector"] = sector
                        collision_streak["count"] = 1
                    collision_streak["frame_id"] = frame_id
                # The violation function has already required multiple points
                # or angular bins. Waiting for a second revolution at the nose
                # or shoulders permits avoidable contact.
                required_frames = (
                    1
                    if sector in ("front", "left side", "right side")
                    else int(collision_profile.get("confirmation_frames", 2))
                )
                if collision_streak["count"] < required_frames:
                    return None
                pending_collision_local[0] = local[mask].copy()
                evidence_points = int(np.count_nonzero(mask))
                evidence_bins = int(
                    np.unique(
                        np.floor(
                            (np.degrees(np.arctan2(ff[mask, 1], ff[mask, 0])) + 180.0)
                            / float(collision_profile["bin_size_deg"])
                        ).astype(np.int64)
                    ).size
                )
                return (
                    f"calibrated collision box {sector}: {hit_range:.2f}m "
                    f"at {angle:+.0f}deg (limit {limit:.2f}m; "
                    f"{evidence_points} points/{evidence_bins} bins)"
                )
            if collision_profile is not None:
                if frame_id != collision_streak["frame_id"]:
                    collision_streak.update(frame_id=frame_id, sector=None, count=0)
                return None
            lane = (
                (ff[:, 0] > 0.02)
                & (ff[:, 0] < float(args.hard_stop_m))
                & (np.abs(ff[:, 1]) < float(args.hard_stop_half_width_m))
            )
            if np.any(lane):
                lane_idx = np.flatnonzero(lane)
                hit = int(lane_idx[np.argmin(ff[lane_idx, 0])])
                return f"live forward clearance x={ff[hit, 0]:.2f}m y={ff[hit, 1]:+.2f}m"
            return None

        controller.set_safety_check(_fast_safety_check)
        controller.clear_safety_latch()
        prev_imu = imu.deg()
        last_render = 0.0
        last_sent: tuple[bool, float] | None = None  # (driving, target_imu)
        # Tracking-health state. The search window is bounded by PHYSICS: the base
        # cannot exceed --max-speed-mps, so the true position always lies within
        # v_max * (time since the last good fix).
        v_max = float(args.max_speed_mps)
        last_good_t = time.monotonic()
        n_weak = 0
        last_diag = 0.0
        novel_hist: list[tuple[np.ndarray, np.ndarray]] = []  # (obstacle world centroid, robot centre)
        phantom_warned = False
        polyline = np.vstack([_robot_centre(cur_pose)[None, :], np.asarray(waypoints, dtype=np.float64)])
        goal = np.asarray(goal_xy, dtype=np.float64)
        leg_t0 = time.monotonic()
        driving = False
        segment_index = 0
        segment_start = _robot_centre(cur_pose).copy()
        segment_target_imu: float | None = None
        pivot_target_imu: float | None = None
        pivot_last_yaw: float | None = None
        pivot_last_progress_t = time.monotonic()
        dither_segment_started = time.monotonic()
        dither_logged = False
        replans = 0

        def _replan_same_frontier(reason: str) -> bool:
            nonlocal analysis, waypoints, polyline, driving, last_sent, n_weak
            nonlocal last_good_t, replans, segment_index, segment_start, segment_target_imu
            nonlocal pivot_target_imu, pivot_last_yaw, pivot_last_progress_t
            nonlocal navigation_start_reference
            nonlocal dither_segment_started, dither_logged
            controller.halt()
            controller.clear_safety_latch()
            replans += 1
            refreshed = _analysis_now()
            new_waypoints = _plan_waypoints(refreshed, _robot_centre(cur_pose), goal)
            if new_waypoints is None or replans > 3:
                return False
            analysis = refreshed
            waypoints = new_waypoints
            refreshed_start_cell = _snap_traversable(
                analysis,
                _robot_centre(cur_pose),
                r_m=0.25,
            )
            navigation_start_reference = (
                analysis.to_world(refreshed_start_cell) if refreshed_start_cell is not None else None
            )
            polyline = np.vstack(
                [
                    _robot_centre(cur_pose)[None, :],
                    np.asarray(new_waypoints, dtype=np.float64),
                ]
            )
            rr.log(
                "world/plan",
                rr.LineStrips3D(
                    [[[float(p[0]), float(p[1]), 0.0] for p in polyline]],
                    colors=[[70, 230, 100]],
                    radii=0.014,
                ),
            )
            driving = False
            last_sent = None
            segment_index = 0
            segment_start = _robot_centre(cur_pose).copy()
            segment_target_imu = None
            pivot_target_imu = None
            pivot_last_yaw = None
            pivot_last_progress_t = time.monotonic()
            dither_segment_started = time.monotonic()
            dither_logged = False
            n_weak = 0
            last_good_t = time.monotonic()
            print(
                f"[drive] {reason} — replanned toward the SAME frontier ({replans}/3), no target switching."
            )
            return True

        if overlap_handoff and not continuing_doorway:
            print(
                "[explore] passable-frontier transit: taking one stationary "
                "overlap reference, then holding one passage heading while "
                "live LiDAR keyframes grow the map continuously."
            )
            handoff_ready = _capture_handoff(
                force=True,
                doorway_keyframe=True,
            )
            if not handoff_ready:
                print(
                    "[explore] pre-door overlap keyframe was not accepted; this "
                    "optional snapshot will not suppress the collision-checked "
                    "navigation leg. Continuing toward the doorway and requiring "
                    "fresh overlap again at the next translated checkpoint."
                )
                # A rejected optional keyframe does not mean the continuously
                # tracked pose was lost. The former return here created a loop of
                # in-place snapshots/relocalizations and never executed the plan.
                # Reset attempt state so translated checkpoints and arrival can
                # acquire the missing overlap later.
                handoff_attempted = False
                handoff_pose_ok = False
            prev_imu = imu.deg()
            # Mapping time is not navigation failure time. Start the leg slice
            # after the deliberate stationary handoff has completed.
            leg_t0 = time.monotonic()
        elif overlap_handoff:
            print(
                "[explore] continuing committed doorway transit: reusing the "
                "existing overlap reference until the base translates; no "
                "duplicate pre-door scan batch."
            )

        if True:  # single-leg loop (keeps the body's indentation stable)
            while True:
                safety_reason = controller.safety_latched_reason()
                if safety_reason is not None:
                    controller.halt()
                    if safety_reason.startswith("LiDAR safety"):
                        stale_id = feed.latest()[0]
                        print(
                            f"[drive] {safety_reason} — forward motion inhibited; "
                            "waiting for a fresh LiDAR revolution."
                        )
                        fresh_id, fresh_frame = feed.wait_for_frame_after(
                            after_frame_id=stale_id,
                            timeout_s=3.0,
                            min_frame_advances=1,
                        )
                        fresh_age_s = (
                            time.time() - float(getattr(fresh_frame, "received_wall_ts", 0.0) or 0.0)
                            if fresh_frame is not None
                            else float("inf")
                        )
                        if (
                            fresh_frame is None
                            or int(fresh_id) <= int(stale_id)
                            or not math.isfinite(fresh_age_s)
                            or fresh_age_s > float(args.lidar_safety_max_age_s)
                        ):
                            print("[drive] fresh LiDAR did not return within 3.0s; remaining stopped.")
                            return "navigation lidar unavailable"
                        collision_streak.update(frame_id=None, sector=None, count=0)
                        controller.clear_safety_latch()
                        # Safety latching changed the fast controller to HALT.
                        # Clearing the reason alone does not restore its drive
                        # mode, so resend the unchanged straight command.
                        last_sent = None
                        print("[drive] fresh LiDAR restored — resuming the same straight command.")
                        continue
                    if safety_reason.startswith("map edge"):
                        print(f"[drive] {safety_reason} — stopping to map before continuing.")
                        controller.clear_safety_latch()
                        break
                    if safety_reason.startswith("calibrated collision box"):
                        offenders = pending_collision_local[0]
                        pending_collision_local[0] = None
                        if offenders is not None and len(offenders):
                            world_map.grid.mark_hits(_transform_points(offenders, cur_pose), amount=2.0)
                        # The physical envelope has proven this approach pose is
                        # blocked. The mission loop photographs the boundary
                        # once, retires it, and chooses another frontier.
                        print(
                            f"[drive] {safety_reason} — clearance boundary reached; "
                            "ending this approach without same-target retries."
                        )
                        return f"hard stop ({safety_reason})"
                    if safety_reason.startswith("rotational collision box"):
                        if overlap_handoff and segment_index < len(waypoints):
                            centre_now = _robot_centre(cur_pose)
                            segment_end_now = np.asarray(waypoints[segment_index], dtype=np.float64)
                            desired_vector = segment_end_now - centre_now
                            desired_heading = math.degrees(
                                math.atan2(float(desired_vector[1]), float(desired_vector[0]))
                            )
                            physical_heading = float(cur_pose.theta_deg) + forward_offset
                            heading_error = ((desired_heading - physical_heading + 180.0) % 360.0) - 180.0
                            _frame_id, latest_frame = feed.latest()
                            latest_local = (
                                _scan_local(latest_frame, args)
                                if latest_frame is not None
                                else np.empty((0, 2))
                            )
                            latest_forward = _to_forward_frame(latest_local, forward_offset)
                            step_m = min(
                                0.20,
                                max(0.12, float(np.hypot(*desired_vector))),
                            )
                            translation_safe = _translation_trajectory_is_safe(
                                latest_forward,
                                collision_profile,
                                np.array([step_m, 0.0], dtype=np.float64),
                                lidar_offset_forward_m=lever_m,
                                physical_body_radius_m=float(args.physical_body_radius_m),
                                self_mask_inset_m=float(args.collision_self_mask_inset_m),
                            )
                            if abs(heading_error) <= 25.0 and translation_safe:
                                heading_rad = math.radians(physical_heading)
                                escape_end = centre_now + step_m * np.array(
                                    [math.cos(heading_rad), math.sin(heading_rad)]
                                )
                                waypoints.insert(segment_index, escape_end)
                                segment_start = centre_now.copy()
                                segment_target_imu = imu.deg()
                                driving = segment_target_imu is not None
                                corridor_translation_active[0] = driving
                                last_sent = None
                                pivot_target_imu = None
                                pivot_last_yaw = None
                                controller.clear_safety_latch()
                                if driving:
                                    print(
                                        f"[drive] doorway pivot constrained at "
                                        f"{heading_error:+.1f}deg; calibrated "
                                        f"straight rollout is safe, translating "
                                        f"{step_m:.2f}m before reconsidering the turn."
                                    )
                                    continue
                        print(
                            f"[drive] {safety_reason} - pivot prohibited; "
                            "mapping this clearance boundary without rotating."
                        )
                        return f"hard stop ({safety_reason})"
                    if _replan_same_frontier(f"safety stop: {safety_reason}"):
                        continue
                    print(f"[drive] safety stop persisted after same-frontier replans: {safety_reason}.")
                    return f"hard stop ({safety_reason})"
                if _budget_exhausted(time.monotonic() - mission_t0, float(args.max_mission_seconds)):
                    controller.halt()
                    return "mission time budget spent"
                if time.monotonic() - leg_t0 > float(args.max_leg_seconds):
                    controller.halt()
                    return "leg time budget spent"
                ground_motion = None
                centre = _robot_centre(cur_pose)
                while segment_index < len(waypoints):
                    segment_end = np.asarray(waypoints[segment_index], dtype=np.float64)
                    if not _segment_complete(
                        centre,
                        segment_start,
                        segment_end,
                        float(args.waypoint_tol_m),
                    ):
                        break
                    controller.halt()
                    # Before a completed passable straight segment turns, add
                    # its stationary endpoint at the IMU/local-LiDAR carried
                    # pose.  This keeps a doorway or end wall at the measured
                    # distance instead of asking ambiguous parallel walls for
                    # another global pose estimate.
                    if bool(featureless_corridor_state["active"]):
                        featureless_corridor_state["checkpoint_due"] = True
                        featureless_corridor_state["checkpoint_reason"] = "straight-segment endpoint"
                        print(
                            "[drive] straight passable segment completed; attaching "
                            "the endpoint LiDAR view at the carried map pose before turning."
                        )
                        if not _commit_featureless_corridor_checkpoint():
                            return "tracking lost"
                        added_now, gain_now = _record_mapping_result()
                        handoff_added += added_now
                        handoff_gain += gain_now
                        handoff_pose_ok = True
                        last_handoff_centre = _robot_centre(cur_pose).copy()
                        last_handoff_heading = float(cur_pose.theta_deg) + forward_offset
                        prev_imu = imu.deg()
                    segment_index += 1
                    segment_start = centre.copy()
                    corridor_translation_active[0] = False
                    segment_target_imu = None
                    pivot_target_imu = None
                    pivot_last_yaw = None
                    dither_segment_started = time.monotonic()
                    dither_logged = False
                    driving = False
                    last_sent = None
                if segment_index >= len(waypoints):
                    break

                segment_end = np.asarray(waypoints[segment_index], dtype=np.float64)
                vec = segment_end - centre
                yaw_now = _wait_for_navigation_yaw(imu, wait_up_to_s=2.0)
                if yaw_now is None:
                    controller.halt()
                    print(
                        "[drive] IMU yaw remained unavailable for 2.0s; "
                        "leaving this leg in bounded sensor-recovery hold instead "
                        "of silently sleeping forever."
                    )
                    return "navigation sensor unavailable"

                if not driving:
                    # Aim once at this segment endpoint. During the subsequent
                    # translation this target is frozen; localization corrections
                    # are deliberately forbidden from steering the base.
                    alpha = math.degrees(math.atan2(vec[1], vec[0]))
                    desired_map = alpha - forward_offset
                    err_map = ((desired_map - float(cur_pose.theta_deg) + 180.0) % 360.0) - 180.0
                    target_imu = float(yaw_now) + err_map
                    if abs(err_map) <= float(args.aim_tolerance_deg):
                        last_sent = None
                        segment_target_imu = target_imu
                        pivot_target_imu = None
                        pivot_last_yaw = None
                        dither_segment_started = time.monotonic()
                        dither_logged = False
                        corridor_translation_active[0] = bool(overlap_handoff)
                        segment_distance = float(np.hypot(*vec))
                        if (
                            bool(overlap_handoff)
                            and segment_distance >= 0.50
                            and not bool(featureless_corridor_state["active"])
                        ):
                            # On every long passable straight, the robot has a
                            # known start pose and a known heading.  Parallel
                            # walls cannot safely produce a lateral global-map
                            # correction, so carry this pose until the next
                            # stationary endpoint/landmark snapshot.
                            featureless_corridor_state.update(
                                active=True,
                                origin_xy=_robot_centre(cur_pose).copy(),
                                heading_deg=float(cur_pose.theta_deg),
                                max_distance_m=max(4.0, segment_distance + 0.30),
                                skip_logged=False,
                                checkpoint_due=False,
                                checkpoint_reason="",
                                last_checkpoint_distance_m=0.0,
                                ignored_cross_track_updates=0,
                            )
                            print(
                                "[localize] long straight passable segment: locking the "
                                "anchored lateral map coordinate and carrying IMU/local "
                                f"odometry for {segment_distance:.2f}m."
                            )
                        print(
                            f"[drive] segment {segment_index + 1}/{len(waypoints)} aligned — "
                            f"straight {segment_distance:.2f}m."
                        )
                        driving = True
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
                        _discard_translation_prior()
                        _reset_surface_motion_prior(cur_pose)
                        continue
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
                        _reset_surface_motion_prior(cur_pose)
                        controller.rotate_to(target_imu)
                        last_sent = (False, target_imu)
                        pivot_target_imu = target_imu
                        pivot_last_yaw = float(yaw_now)
                        pivot_last_progress_t = time.monotonic()
                    elif pivot_target_imu is None or abs(target_imu - pivot_target_imu) > 4.0:
                        pivot_target_imu = target_imu
                        pivot_last_yaw = float(yaw_now)
                        pivot_last_progress_t = time.monotonic()
                    if pivot_last_yaw is None or abs(float(yaw_now) - pivot_last_yaw) >= 2.0:
                        pivot_last_yaw = float(yaw_now)
                        pivot_last_progress_t = time.monotonic()
                    elif _pivot_progress_stalled(
                        time.monotonic(),
                        pivot_last_progress_t,
                    ):
                        controller.halt()
                        controller.clear_safety_latch()
                        print(
                            "[drive] pivot progress timeout: the commanded base turn "
                            "changed IMU yaw by less than 2deg for 2.5s. Resolving "
                            "the blocked orientation locally instead of selecting "
                            "another destination from the same physical pose."
                        )
                        if _active_localization_translation():
                            prev_imu = imu.deg()
                            if _replan_same_frontier("short collision-checked clearance move completed"):
                                continue
                        print(
                            "[drive] pivot remained physically unavailable after "
                            "the bounded clearance maneuver; treating this pose as "
                            "a mapped clearance boundary instead of entering a "
                            "plan/pivot/replan loop."
                        )
                        return "hard stop (pivot physically stalled)"
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
                    assert segment_target_imu is not None
                    target_imu = float(segment_target_imu)
                    err_map = target_imu - float(yaw_now)
                    # Predict only the next LiDAR seed from the *classified
                    # surface's LiDAR-trained command scale*.  This is not a
                    # wheel/flow pose measurement: the following fresh LiDAR
                    # registration must accept or discard the prediction.
                    signed_drive_command = float(args.drive_speed) * float(drive_command_sign[0])
                    cur_pose = _apply_surface_motion_prediction(
                        cur_pose,
                        forward_command=signed_drive_command,
                    )
                    # Legacy optical flow can still be explicitly enabled for
                    # comparison, but surface calibration is independent of it.
                    cur_pose, ground_motion = _apply_ground_translation_prior(cur_pose)
                    # DRIVE phase: scan-to-map tracking with a PHYSICS-BOUNDED
                    # window — the base cannot outrun v_max, so no match may claim
                    # a bigger displacement, whatever its score. While driving
                    # straight the yaw-hold keeps rotation slow; the tracker pairs
                    # each scan with its host-timestamped IMU yaw.
                    t_now = time.monotonic()
                    window = min(
                        0.55,
                        max(
                            v_max * (t_now - last_good_t) + 0.10,
                            0.12 + float(surface_prior_uncertainty_m[0]),
                        ),
                    )
                    score, local, prev_imu = _track(
                        prev_imu,
                        integrate=False,
                        search_xy_m=window,
                        # Local scan-to-submap odometry owns passage
                        # continuity. Global correction remains available at a
                        # lower rate without blocking every control cycle.
                        global_match_every=(6 if overlap_handoff else 2),
                    )
                    _update_world_obstacle_pose_trust()

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
                        planned_start_xy=navigation_start_reference,
                    )
                    if sweep_state == "unknown":
                        controller.halt()
                        controller.clear_safety_latch()
                        if _map_edge_is_pose_disagreement(sweep_distance, analysis.res_m):
                            print(
                                "[drive] planner/localization disagreement at the "
                                "current cell; this is NOT viewpoint arrival. "
                                "Preserving the objective and requesting a "
                                "stationary pose consensus without rotating."
                            )
                            return "tracking lost"
                        print(
                            f"[drive] map edge {sweep_distance:.2f}m ahead — "
                            "stopping to map before continuing."
                        )
                        break
                    if sweep_state == "blocked" and collision_profile is None:
                        controller.halt()
                        reason = f"full-body clearance blocked {sweep_distance:.2f}m ahead"
                        if _replan_same_frontier(reason):
                            continue
                        return "hard stop (mapped shoulder clearance)"
                    corridor_steer_deg = 0.0
                    if len(local):
                        ff = _to_forward_frame(local, forward_offset)
                        if bool(featureless_corridor_state["active"]):
                            corridor_steer_deg = _corridor_centering_steer_deg(ff)
                        if collision_profile is not None:
                            lane = np.zeros(len(ff), dtype=bool)
                        else:
                            lane = (ff[:, 0] > 0.02) & (np.abs(ff[:, 1]) < float(args.hard_stop_half_width_m))
                        if np.any(lane) and float(ff[lane, 0].min()) < float(args.hard_stop_m):
                            controller.halt()
                            offenders = local[lane & (ff[:, 0] < float(args.hard_stop_m) + 0.25)]
                            if len(offenders):
                                _remember_transient_obstacle(_transform_points(offenders, cur_pose))
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
                            relock_candidates: list[tuple[Pose2D, float, float]] = []
                            after_r = feed.latest()[0]
                            for _attempt in range(3):  # patience: fresh frame each try
                                fid_r, frame_r = feed.wait_for_frame_after(
                                    after_frame_id=after_r, timeout_s=1.5, min_frame_advances=2
                                )
                                if frame_r is None:
                                    continue
                                after_r = int(fid_r)
                                local_r = _scan_local(frame_r, args)
                                if len(local_r) < 12:
                                    continue
                                sup_r = _match_support_mask(_transform_points(local_r, cur_pose))
                                support_at_seed_r = float(np.mean(sup_r)) if len(sup_r) else 0.0
                                match_r = (
                                    local_r
                                    if support_at_seed_r >= 0.35
                                    else (local_r[sup_r] if int(sup_r.sum()) >= 30 else local_r)
                                )
                                solved_r, sc_r = _localize(match_r, world_map, cur_pose, args, win_r, 10.0)
                                solved_support_r = _match_support_mask(_transform_points(local_r, solved_r))
                                support_r = float(np.mean(solved_support_r)) if len(local_r) else 0.0
                                correction_r = float(
                                    np.hypot(*(_robot_centre(solved_r) - _robot_centre(cur_pose)))
                                )
                                heading_r = abs(
                                    (float(solved_r.theta_deg) - float(cur_pose.theta_deg) + 180.0) % 360.0
                                    - 180.0
                                )
                                if (
                                    sc_r >= max(8.0, float(args.min_match_score))
                                    and support_r >= max(0.15, float(args.viewpoint_min_known_ratio))
                                    and correction_r <= float(args.relocalization_max_pose_correction_m)
                                    and heading_r <= float(args.relocalization_max_pose_correction_deg)
                                    and _pose_stays_beyond_completed_doorways(solved_r)
                                ):
                                    relock_candidates.append((solved_r, float(sc_r), support_r))
                            if len(relock_candidates) >= 2:
                                consensus_r, inliers_r, scatter_r, heading_scatter_r = (
                                    _stationary_pose_inlier_consensus(
                                        [item[0] for item in relock_candidates],
                                        float(cur_pose.theta_deg),
                                        lever_m,
                                        forward_offset,
                                        max_position_residual_m=0.10,
                                        max_heading_residual_deg=4.0,
                                        min_inliers=2,
                                        reference_centre_xy=_robot_centre(cur_pose),
                                    )
                                )
                                if len(inliers_r) >= 2:
                                    cur_pose = consensus_r
                                    _discard_translation_prior()
                                    relocked = True
                                    mean_relock_score = float(
                                        np.mean([relock_candidates[index][1] for index in inliers_r])
                                    )
                                    print(
                                        "[drive] tracking slipped — re-locked by stationary "
                                        f"consensus ({len(inliers_r)}/{len(relock_candidates)} "
                                        f"inliers, match {mean_relock_score:.1f}, scatter "
                                        f"{scatter_r * 100:.0f}cm/"
                                        f"{heading_scatter_r:.1f}deg, window {win_r:.2f}m)."
                                    )
                            if not relocked:
                                controller.halt()
                                print(
                                    "[drive] stationary relock failed; mapping this boundary "
                                    "once without performing a recovery spin."
                                )
                                _integrate_at_viewpoint(scan_count=5)
                                if vp_pose_ok_last[0]:
                                    print(
                                        "[drive] boundary localization accepted; continuing "
                                        "the same plan (map geometry changes only if its "
                                        "stricter transaction gate also passed)."
                                    )
                                    relocked = True
                                else:
                                    print(
                                        "[drive] boundary localization still weak; "
                                        "remaining stationary instead of spinning."
                                    )
                                    return "tracking lost"
                            n_weak = 0
                            last_good_t = time.monotonic()
                            last_sent = None
                            driving = False  # re-aim from the corrected pose
                            segment_start = _robot_centre(cur_pose).copy()
                            segment_target_imu = None
                            prev_imu = imu.deg()
                            continue
                    segment_length = float(np.hypot(*(segment_end - segment_start)))
                    segment_progress_m = 0.0
                    if segment_length > 1e-6:
                        segment_axis = (segment_end - segment_start) / segment_length
                        segment_progress_m = float(np.dot(_robot_centre(cur_pose) - segment_start, segment_axis))
                    if (
                        bool(args.auto_correct_drive_command_sign)
                        and drive_command_sign_flips[0] < 1
                        and segment_progress_m < -float(args.reverse_progress_trigger_m)
                    ):
                        controller.halt()
                        old_sign = float(drive_command_sign[0])
                        drive_command_sign[0] = -old_sign
                        drive_command_sign_flips[0] += 1
                        last_sent = None
                        driving = False
                        segment_start = _robot_centre(cur_pose).copy()
                        segment_target_imu = None
                        _discard_translation_prior()
                        _reset_surface_motion_prior(cur_pose)
                        print(
                            "[drive] commanded straight motion moved AWAY from the waypoint "
                            f"by {-segment_progress_m:.2f}m; flipping autonomous x.vel sign "
                            f"{old_sign:+.0f}->{drive_command_sign[0]:+.0f} and re-aiming."
                        )
                        continue
                    if segment_progress_m < -max(0.40, 2.5 * float(args.reverse_progress_trigger_m)):
                        controller.halt()
                        return "hard stop (autonomous drive moved opposite planned waypoint)"
                    if bool(featureless_corridor_state["checkpoint_due"]):
                        checkpoint_reason = str(
                            featureless_corridor_state["checkpoint_reason"] or "odometry interval"
                        )
                        print(
                            "[drive] featureless-corridor "
                            f"{checkpoint_reason}: stopping to attach the live LiDAR "
                            "view at the odometry-carried map pose."
                        )
                        if not _commit_featureless_corridor_checkpoint():
                            return "tracking lost"
                        added_now, gain_now = _record_mapping_result()
                        handoff_added += added_now
                        handoff_gain += gain_now
                        handoff_pose_ok = True
                        last_handoff_centre = _robot_centre(cur_pose).copy()
                        last_handoff_heading = float(cur_pose.theta_deg) + forward_offset
                        prev_imu = imu.deg()
                        last_good_t = time.monotonic()
                        last_sent = None
                        driving = False
                        segment_start = _robot_centre(cur_pose).copy()
                        segment_target_imu = None
                        continue
                    passage_checkpoint_due = (
                        bool(last_tracking_health["fresh"])
                        and bool(last_tracking_health["accepted"])
                        and score >= float(args.min_match_score)
                        and float(np.hypot(*(_robot_centre(cur_pose) - last_handoff_centre)))
                        >= (
                            max(0.20, float(args.frontier_doorway_keyframe_m))
                            if overlap_handoff or rolling_submap
                            # Plain transit through already-known space: the
                            # live scan-to-map tracker, the rolling-odometry
                            # disagreement stop, and the map-edge guard remain
                            # active, so a ~1.8s stationary batch every leg
                            # fraction only slowed the traverse.  Verify at a
                            # long interval and rely on the arrival re-fit for
                            # the final position check.
                            else max(2.50, 3.0 * float(args.frontier_handoff_checkpoint_m))
                        )
                    )
                    # The post-turn recovery has a verified pose but may be
                    # facing back into the same featureless parallel-wall
                    # corridor that caused the original ambiguity.  Stopping
                    # every 20cm here to demand a *global* stationary
                    # confirmation simply recreates the loss condition.  Hold
                    # that checkpoint until the bounded local/IMU corridor
                    # traverse has reached a distinctive view (or its hard
                    # distance budget), then immediately return to ordinary
                    # mapping rules.  Collision checks remain active.
                    if passage_checkpoint_due and bool(featureless_corridor_state["active"]):
                        if not bool(featureless_corridor_state["skip_logged"]):
                            print(
                                "[explore] featureless-corridor traverse: deferring "
                                "stationary map checkpoints while carrying the verified "
                                "turnaround pose toward a distinctive landmark."
                            )
                            featureless_corridor_state["skip_logged"] = True
                        passage_checkpoint_due = False
                    if passage_checkpoint_due and (overlap_handoff or rolling_submap):
                        # Moving scans are useful for local odometry but were
                        # not safe map evidence here: the old code promoted
                        # them to GOLD, then continued through several failed
                        # stopped checks. That is the direct source of the
                        # ghosted hallway in the attached run. Every passage
                        # interval now stops for a compact full-view LiDAR
                        # batch before it may extend either map.
                        controller.halt()
                        print(
                            "[explore] passage map-growth checkpoint — taking a clean "
                            "stationary full-view LiDAR batch before continuing."
                        )
                        if not _capture_handoff(force=True, doorway_keyframe=False):
                            print(
                                "[drive] stationary full-view checkpoint could not verify "
                                "the current map pose; refusing to dead-reckon farther "
                                "through this passage."
                            )
                            return "tracking lost"
                        checkpoint_failures = 0
                        prev_imu = imu.deg()
                        last_good_t = time.monotonic()
                        last_sent = None
                        driving = False
                        segment_start = _robot_centre(cur_pose).copy()
                        segment_target_imu = None
                        continue
                    if passage_checkpoint_due and not (overlap_handoff or rolling_submap):
                        controller.halt()
                        print(
                            "[explore] map-growth checkpoint — snapshotting before continuing the same path."
                        )
                        checkpoint_ok = _capture_handoff()
                        if not checkpoint_ok:
                            checkpoint_failures += 1
                            print(
                                "[drive] stationary map-growth checkpoint did not verify "
                                "the present full LiDAR view; stopping this leg before any "
                                "further unverified movement."
                            )
                            return "tracking lost"
                        checkpoint_failures = 0
                        prev_imu = imu.deg()
                        last_good_t = time.monotonic()
                        last_sent = None
                        leg_t0 = time.monotonic()
                    if t_now - last_diag >= 2.0:
                        last_diag = t_now
                        c_dbg = _robot_centre(cur_pose)
                        ground_label = (
                            "ground --"
                            if ground_motion is None
                            else (f"ground {ground_motion.samples}x/{ground_motion.residual_m * 100:.1f}cm")
                        )
                        wall_label = (
                            f" wall {corridor_steer_deg:+.1f}deg"
                            if corridor_steer_deg != 0.0
                            else ""
                        )
                        print(
                            f"[drive] score {max(score, -9.9):4.1f} window {window:.2f}m "
                            f"pos ({c_dbg[0]:+.2f},{c_dbg[1]:+.2f}) dist {float(np.hypot(*vec)):.2f}m "
                            f"heading_err {err_map:+.1f}deg {ground_label}{wall_label}"
                        )
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
                                    segment_start = _robot_centre(cur_pose).copy()
                                    segment_target_imu = None
                                    continue
                                if not phantom_warned:
                                    phantom_warned = True
                                    print(
                                        "[explore] near-field returns MOVE WITH the robot — "
                                        "self-artifact (arm flex / vibration), ignoring; "
                                        "not a world obstacle."
                                    )
                                # Phantom: keep driving.
                    else:
                        novel_hist.clear()
                    # Straight drive: the fast controller keeps streaming while
                    # safety runs before every command.  With a forward-only
                    # 180-degree scanner, a perfectly straight segment can
                    # remain geometrically ambiguous (parallel walls look the
                    # same after a small slip).  An optional, very small yaw
                    # dither supplies parallax without introducing a lateral
                    # zig-zag: the base still translates forward and the
                    # collision callback guards every updated heading.
                    command_target = target_imu
                    # Corridor traverse: continuously re-centre between the
                    # live walls and align with their axis instead of holding
                    # a heading frozen at segment start — an IMU-held heading
                    # lets small aim errors integrate into a wall-hugging
                    # drift over a long hallway.  Bounded to ±6deg; zero
                    # whenever the live view does not look like a corridor.
                    if corridor_steer_deg != 0.0:
                        command_target += corridor_steer_deg
                    remaining_segment_m = float(
                        np.hypot(*(segment_end - _robot_centre(cur_pose)))
                    )
                    dither_allowed = (
                        bool(args.active_yaw_dither)
                        and segment_length >= float(args.active_yaw_dither_min_segment_m)
                        and remaining_segment_m > 0.35
                        and sweep_state == "clear"
                        and controller.safety_latched_reason() is None
                    )
                    if dither_allowed:
                        if not dither_logged:
                            print(
                                f"[drive] active yaw-parallax dither enabled for "
                                f"{segment_length:.2f}m segment (±{float(args.active_yaw_dither_deg):.1f}deg, "
                                f"period {float(args.active_yaw_dither_period_s):.1f}s)."
                            )
                            dither_logged = True
                        phase = (
                            2.0
                            * math.pi
                            * (time.monotonic() - dither_segment_started)
                            / float(args.active_yaw_dither_period_s)
                        )
                        command_target += float(args.active_yaw_dither_deg) * math.sin(phase)
                    if last_sent is None or abs(command_target - last_sent[1]) > 1.0 or len(last_sent) < 3 or abs(float(last_sent[2]) - signed_drive_command) > 1e-6:
                        controller.drive_toward(signed_drive_command, command_target)
                        last_sent = (True, command_target, signed_drive_command)
                trail.append(_robot_centre(cur_pose))
                # Throttle rendering — it is expensive and adds steering latency.
                now = time.monotonic()
                if now - last_render >= 0.5:
                    last_render = now
                    _log_panorama()
                    _log_world(rr, world_map, args, trail[-1], trail, analysis, goal_xy, waypoints)
        # ===================================================================
        # EXPLORE PHASE 4 — FACE THE UNKNOWN AREA, SCAN, AND UPDATE THE MAP
        # ===================================================================
        if corridor_carry_only:
            controller.halt()
            # This used to only *claim* the endpoint view was attached.  When a
            # map-edge stop ended the leg ~0.1m out, nothing was written, the
            # unknown boundary never receded, and the pending carry replayed
            # the identical leg forever.  Commit the stationary endpoint
            # checkpoint for real so the beeline can actually extend the map.
            featureless_corridor_state["checkpoint_reason"] = "carry endpoint"
            committed = _commit_featureless_corridor_checkpoint()
            if not committed:
                vp_last[0] = 0
                vp_gain_last[0] = 0
            vp_pose_ok_last[0] = True
            print(
                "[explore] featureless-corridor carry leg complete "
                f"({'endpoint view attached, %d newly-known cells' % int(vp_gain_last[0] or 0) if committed else 'endpoint view could not be attached'}); "
                "continuing from the odometry-carried pose."
            )
            return ""

        # Arrived: FACE THE FRONTIER before mapping. The body occludes ~90deg
        # behind and the stowed arms shadow their sectors — an arbitrary arrival
        # heading can blind exactly the area we came to observe (field: stops
        # 1.2m from the doorway 'learned nothing' twice, then retired the exit).
        controller.halt()

        leg_translation = float(np.hypot(*(_robot_centre(cur_pose) - leg_start_centre)))
        if overlap_handoff or rolling_submap:
            minimum_translation = 0.20 if overlap_handoff else 0.12
            if leg_translation >= minimum_translation:
                survey_target = (
                    np.asarray(face_xy, dtype=np.float64)
                    if face_xy is not None
                    else np.asarray(goal_xy, dtype=np.float64)
                )
                if overlap_handoff:
                    _capture_overlap_turn(
                        "doorway-arrival local submap",
                        survey_target,
                        max_centre_arc_deg=min(
                            120.0,
                            max(30.0, float(args.frontier_handoff_sweep_deg)),
                        ),
                    )
                else:
                    # A stopped, forward-facing full revolution already supplies
                    # the LiDAR's entire unobscured front hemisphere. Do not fan
                    # back and forth: face the unknown once, then compact several
                    # stationary revolutions into clean permanent geometry.
                    print(
                        "[explore] room survey: facing the unknown once, then "
                        "capturing one stationary forward-hemisphere batch."
                    )
                    _face_snapshot_target(survey_target, "room survey")
                    _capture_handoff(force=True, doorway_keyframe=True)
            else:
                print(
                    f"[explore] arrival mapping skipped: base translated only "
                    f"{leg_translation * 100:.0f}cm; refusing to rescan and turn "
                    "at the same observation pose."
                )
        elif face_xy is not None:
            _face_snapshot_target(
                np.asarray(face_xy, dtype=np.float64),
                "frontier",
            )
        if overlap_handoff or rolling_submap:
            # If no pre-arrival handoff ran, acquire one overlap-verified view.
            if not handoff_attempted and leg_translation >= 0.12:
                _capture_handoff(doorway_keyframe=True)
            elif not handoff_attempted:
                print(
                    "[explore] doorway handoff retry skipped: the base did not "
                    "translate, so another stationary batch would duplicate the "
                    "same failed observation."
                )
            vp_last[0] = handoff_added
            vp_gain_last[0] = handoff_gain
            vp_pose_ok_last[0] = handoff_pose_ok
            print(
                f"[explore] rolling submap handoff complete: +{handoff_added} scans, "
                f"{handoff_gain} newly-known cells across overlapping views."
            )
        else:
            _integrate_at_viewpoint()
            _record_mapping_result()
            # ARRIVAL TRIM: fast transit skips most mid-leg checkpoints, so
            # position error surfaces at this stationary arrival re-fit.  If
            # the verified pose landed a bounded distance off the planned
            # observation point, drive the measured offset directly (live
            # collision envelope enforced throughout) and re-verify with one
            # more stationary batch — at most two corrective moves, so this
            # can never oscillate.
            for _trim in range(2):
                if goal_xy is None or not vp_pose_ok_last[0]:
                    break
                offset_world = np.asarray(goal_xy, dtype=np.float64) - _robot_centre(cur_pose)
                offset_m = float(np.hypot(*offset_world))
                if offset_m <= 0.25 or offset_m > 1.00:
                    break
                heading_deg = float(cur_pose.theta_deg) + forward_offset
                trim_direction_deg = (
                    math.degrees(math.atan2(float(offset_world[1]), float(offset_world[0])))
                    - heading_deg
                )
                print(
                    f"[drive] arrival trim: verified pose is {offset_m * 100:.0f}cm from "
                    "the planned observation point; driving the measured offset and "
                    "re-verifying."
                )
                if not _active_localization_translation(
                    distance_m=min(0.60, offset_m),
                    direction_body_deg=trim_direction_deg,
                ):
                    break
                _integrate_at_viewpoint()
                _record_mapping_result()
        return ""

    # =======================================================================
    # EXPLORE PHASES 2–5 — REPEATING WANDER/MAP LOOP (EXECUTION)
    #
    # Phase 2 selects an unknown-area target. Phase 3 drives there. Phase 4
    # captures/integrates the view. Phase 5 records the outcome and loops back
    # to Phase 2 until a configured budget or terminal condition is reached.
    # =======================================================================
    aborted = ""
    localization_hold = False
    localization_hold_attempts = 0
    turnaround_recovery_attempted = False
    actuator_no_motion_attempts = 0
    if bool(args.spin_only):
        print("[explore] --spin-only: skipping exploration; rendering the anchor spin map only.")
    else:
        controller.start()  # background velocity streaming for continuous driving
        _discard_translation_prior()  # startup pivots are not translation priors
    while not bool(args.spin_only) and not _budget_exhausted(
        viewpoints_reached, int(args.explore_viewpoints)
    ):
        if _budget_exhausted(time.monotonic() - mission_t0, float(args.max_mission_seconds)):
            aborted = "mission time budget spent"
            break
        if localization_hold:
            controller.halt()
            localization_hold_attempts += 1
            print(
                "\n[explore] bounded localization recovery: stationary batch, "
                "local-submap consensus, then one collision-checked parallax move."
            )
            _integrate_at_viewpoint(scan_count=5)
            if vp_pose_ok_last[0]:
                localization_hold = False
                localization_hold_attempts = 0
                turnaround_recovery_attempted = False
                _reset_active_localization_recovery()
                _invalidate_pre_recovery_navigation_objectives(
                    "stationary localization recovery"
                )
                print(
                    "[explore] stationary localization recovered; selecting a fresh "
                    "route rather than resuming the pre-recovery plan."
                )
            else:
                print(
                    "[explore] stationary pose remains ambiguous; running the "
                    "single bounded active-localization transition."
                )
                recovered = _staged_bubble_relocalize_stationary()
                if not recovered:
                    active_localization_state["translation_attempted"] = True
                    moved = _active_localization_translation()
                    recovered = moved and _staged_bubble_relocalize_stationary()
                if not recovered and not turnaround_recovery_attempted:
                    turnaround_recovery_attempted = True
                    recovered = _turnaround_relocalize_once()
                if not recovered:
                    aborted = (
                        "localization remained ambiguous after the bounded "
                        "stationary/local-submap/parallax/turnaround recovery"
                    )
                    print(
                        f"[explore] LOCALIZATION EXIT: {aborted}. No uncertain "
                        "geometry was written and no endless hold was entered."
                    )
                    break
                else:
                    localization_hold = False
                    localization_hold_attempts = 0
                    turnaround_recovery_attempted = False
                    none_retries = 0
                    if bool(featureless_corridor_state["pending_carry_after_reanchor"]):
                        print(
                            "[explore] permanent-map localization recovered by the "
                            "180-degree landmark view; preserving the re-anchored "
                            "featureless-corridor carry instead of rebuilding "
                            "frontier targets."
                        )
                    else:
                        _invalidate_pre_recovery_navigation_objectives(
                            "permanent-map localization recovery"
                        )
                        print(
                            "[explore] stationary permanent-map localization recovered; "
                            "rebuilding the frontier candidates from the recovered pose."
                        )
                    continue
        print("\n[explore] analyzing the map for significant frontiers ...")
        planning_epoch += 1
        retired_frontiers[:] = [item for item in retired_frontiers if item[1] > planning_epoch]
        deferred_frontiers[:] = [item for item in deferred_frontiers if item[1] > planning_epoch]
        visited = [item[0] for item in retired_frontiers]
        visited.extend(item[0] for item in deferred_frontiers)
        blocked_viewpoints[:] = [item for item in blocked_viewpoints if item[1] > planning_epoch]
        # -------------------------------------------------------------------
        # EXPLORE PHASE 2 — ANALYZE MAP AND SELECT NEXT UNKNOWN-AREA TARGET
        # _pick_target() prioritizes narrow/non-crossable frontiers before
        # passable transitions into another room.
        # -------------------------------------------------------------------
        analysis = _analysis_now()
        frontiers_left = len(analysis.clusters)
        centre = _robot_centre(cur_pose)
        # A doorway ratchet exists to discourage going BACK while the robot is
        # still on the far side.  Once the verified pose is back on a plane's
        # origin side (deliberate return, fallback patrol, or recovery), that
        # ratchet is spent: keeping it armed only spams pose warnings and
        # starves the planner of legitimate routes — the direct cause of the
        # repeated no-route deadlocks.  Release such planes here, at planning
        # time, when the pose is stationary and freshly verified.
        if completed_transitions:
            spent_indices = [
                index
                for index, (anchor, outward) in enumerate(completed_transitions)
                if float(np.dot(centre - anchor, outward))
                < -(max(0.0, float(args.doorway_ratchet_slack_m)) + 0.10)
            ]
            if spent_indices:
                for index in reversed(spent_indices):
                    completed_transitions.pop(index)
                transition_pose_rejections[0] = 0
                print(
                    f"[plan] released {len(spent_indices)} spent doorway ratchet(s): "
                    "the robot is back on their origin side, so backward-route "
                    "discouragement and pose warnings for them no longer apply."
                )
        if (
            bool(featureless_corridor_state["active"])
            and bool(featureless_corridor_state["pending_carry_after_reanchor"])
        ):
            heading_value = featureless_corridor_state["heading_deg"]
            heading_deg = float(cur_pose.theta_deg if heading_value is None else heading_value)
            axis_rad = math.radians(heading_deg + float(forward_offset))
            corridor_axis = np.array(
                [math.cos(axis_rad), math.sin(axis_rad)],
                dtype=np.float64,
            )
            carry_distance = max(
                1.20,
                min(
                    float(featureless_corridor_state["max_distance_m"]),
                    2.00,
                ),
            )
            carry_goal = centre + carry_distance * corridor_axis
            carry_waypoints = [carry_goal.copy()]
            rr.log(
                "world/plan",
                rr.LineStrips3D(
                    [
                        [
                            [float(centre[0]), float(centre[1]), 0.0],
                            [float(carry_goal[0]), float(carry_goal[1]), 0.0],
                        ]
                    ],
                    colors=[[70, 230, 100]],
                    radii=0.014,
                ),
            )
            _log_world(
                rr,
                world_map,
                args,
                centre,
                trail,
                analysis,
                carry_goal,
                carry_waypoints,
                note="carrying re-anchored featureless corridor",
            )
            print(
                "[explore] executing re-anchored featureless-corridor carry: "
                f"driving {carry_distance:.2f}m straight from the recovered pose, "
                "using IMU/local LiDAR odometry for distance and attaching the "
                "endpoint scan at the predicted map pose."
            )
            carry_result = _drive_leg(
                carry_waypoints,
                carry_goal,
                analysis,
                face_xy=carry_goal,
                corridor_carry_only=True,
            )
            if carry_result == "tracking lost":
                localization_hold = True
                localization_hold_attempts = 0
                turnaround_recovery_attempted = False
                _reset_active_localization_recovery()
                print(
                    "[explore] re-anchored corridor carry lost tracking; entering "
                    "stationary recovery from the carried pose."
                )
                continue
            if carry_result:
                controller.halt()
                aborted = str(carry_result)
                print(f"[explore] CORRIDOR CARRY EXIT: {aborted}.")
                break
            viewpoints_reached += 1
            sampled_viewpoints.append(_robot_centre(cur_pose).copy())
            # Bound the beeline: a leg that neither moved the base nor grew
            # the map is a stall (end wall, blocked edge, or failed endpoint
            # attach).  Two stalls in a row — or spending the corridor's
            # distance allowance — ends the carry so normal frontier planning
            # resumes instead of replaying the identical leg forever.
            leg_progress_m = float(np.hypot(*(_robot_centre(cur_pose) - centre)))
            leg_gain = int(vp_gain_last[0] or 0)
            if leg_progress_m >= 0.25 or leg_gain > 0:
                featureless_corridor_state["carry_stalled_legs"] = 0
            else:
                featureless_corridor_state["carry_stalled_legs"] = (
                    int(featureless_corridor_state["carry_stalled_legs"]) + 1
                )
            corridor_origin = featureless_corridor_state["origin_xy"]
            carried_total_m = (
                0.0
                if corridor_origin is None
                else float(
                    np.hypot(*(_robot_centre(cur_pose) - np.asarray(corridor_origin, dtype=np.float64)))
                )
            )
            carry_done_reason = ""
            if carried_total_m >= float(featureless_corridor_state["max_distance_m"]) - 0.05:
                carry_done_reason = (
                    f"carried {carried_total_m:.2f}m of the "
                    f"{float(featureless_corridor_state['max_distance_m']):.2f}m allowance"
                )
            elif int(featureless_corridor_state["carry_stalled_legs"]) >= 2:
                carry_done_reason = (
                    "two consecutive legs made no forward progress and grew no map"
                )
            vp_last[0] = None
            vp_gain_last[0] = None
            if observation_batch:
                observation_batch = []
                observation_batch_completed = 0
                forward_batch_remaining = 0
                _log_observation_batch_markers(rr, observation_batch)
            if carry_done_reason:
                featureless_corridor_state["active"] = False
                featureless_corridor_state["pending_carry_after_reanchor"] = False
                featureless_corridor_state["carry_stalled_legs"] = 0
                print(
                    f"[explore] featureless-corridor carry ended: {carry_done_reason}; "
                    "normal frontier planning resumes from here."
                )
            else:
                print(
                    "[explore] re-anchored featureless-corridor carry leg done "
                    f"({leg_progress_m:.2f}m progress, {leg_gain} new cells, "
                    f"{carried_total_m:.2f}m carried); reanalyzing before the next leg."
                )
            continue
        directed_passage_target = None
        batch_active = bool(observation_batch)
        # A frozen batch owns the next observation.  Do not let the old
        # doorway-continuation mechanism replace marker 2 or 3 with a newly
        # discovered passage objective.
        if batch_active:
            active_frontier_xy = None
        if not batch_active and active_frontier_xy is not None:
            active_cluster = _match_active_frontier(analysis.clusters, active_frontier_xy)
            if active_cluster is None:
                print("[explore] active frontier disappeared after mapping — objective resolved.")
                active_frontier_xy = None
            else:
                # Follow the frontier as its centroid advances into newly seen space.
                active_frontier_xy = active_cluster.centroid_xy.copy()
                if active_cluster.passable:
                    crossing_plan = _plan_passable_frontier_crossing(
                        analysis,
                        centre,
                        active_cluster,
                    )
                    if crossing_plan is not None:
                        crossing_goal, crossing_waypoints = crossing_plan
                        directed_passage_target = (
                            active_cluster,
                            crossing_goal,
                            crossing_waypoints,
                            False,
                            active_cluster.centroid_xy.copy(),
                            0,
                        )
                        print(
                            "[plan] active doorway already has its overlap "
                            "snapshot; committing a direct collision-checked "
                            f"route {float(np.hypot(*(crossing_goal - centre))):.2f}m "
                            "through the opening without another viewpoint search."
                        )
        picked_target = None
        if observation_batch:
            cluster, goal_xy, _old_waypoints, close_look, observe_xy, expected_gain = observation_batch[0]
            waypoints = _plan_waypoints(analysis, centre, goal_xy)
            if waypoints is None:
                dropped_marker = observation_batch_completed + 1
                observation_batch.pop(0)
                forward_batch_remaining = len(observation_batch)
                _log_observation_batch_markers(
                    rr, observation_batch, completed=observation_batch_completed + 1
                )
                print(
                    f"[plan] batch marker {dropped_marker} is no longer reachable on the "
                    "current collision-inflated map; dropping only that marker and keeping "
                    "the rest of the frozen batch."
                )
                continue
            picked_target = (cluster, goal_xy, waypoints, close_look, observe_xy, expected_gain)
            print(
                f"[plan] executing frozen observation marker "
                f"{observation_batch_completed + 1}/"
                f"{observation_batch_completed + len(observation_batch)}."
            )
        elif directed_passage_target is not None:
            picked_target = directed_passage_target
        elif active_frontier_xy is not None:
            picked_target = _pick_target(
                analysis,
                centre,
                visited,
                float(args.viewpoint_pullback_m),
                float(args.visited_skip_m),
                preferred_frontier_xy=active_frontier_xy,
                sampled_viewpoints_xy=sampled_viewpoints,
                blocked_viewpoints_xy=[item[0] for item in blocked_viewpoints],
                blocked_viewpoint_skip_m=max(0.75, 2.0 * float(args.robot_radius_m)),
                completed_transitions=completed_transitions,
                transition_backtrack_slack_m=float(args.doorway_ratchet_slack_m),
                current_heading_deg=float(cur_pose.theta_deg) + forward_offset,
                turn_cost_per_deg=float(args.frontier_turn_cost_per_deg),
                forward_turn_limit_deg=float(args.frontier_forward_turn_limit_deg),
                allow_return_routes=True,
            )
        else:
            observation_batch = _pick_target_batch(
                analysis,
                centre,
                visited,
                float(args.viewpoint_pullback_m),
                float(args.visited_skip_m),
                batch_size=int(args.frontier_batch_size),
                sampled_viewpoints_xy=sampled_viewpoints,
                blocked_viewpoints_xy=[item[0] for item in blocked_viewpoints],
                blocked_viewpoint_skip_m=max(0.75, 2.0 * float(args.robot_radius_m)),
                completed_transitions=completed_transitions,
                transition_backtrack_slack_m=float(args.doorway_ratchet_slack_m),
                current_heading_deg=float(cur_pose.theta_deg) + forward_offset,
                turn_cost_per_deg=float(args.frontier_turn_cost_per_deg),
                forward_turn_limit_deg=float(args.frontier_forward_turn_limit_deg),
                # This is a deliberate, frozen three-marker tour rather than
                # an opportunistic return.  Permit the batch builder to use
                # any reachable marker if fewer than three lie forward, then
                # execute that declared order without re-scoring mid-batch.
                allow_return_routes=True,
            )
            if observation_batch:
                # The batch builder pads with reachable rearward markers when
                # fewer than three lie forward.  Executing that literally
                # walks the corridor and then doubles back mid-mission (field
                # 2026-08-05: marker 1 down-hall, markers 2-3 behind the
                # start).  While any forward marker exists, unexplored forward
                # space owns the batch; rearward fillers wait for a pass in
                # which nothing lies ahead.
                batch_heading = float(cur_pose.theta_deg) + forward_offset
                forward_markers = []
                for batch_item in observation_batch:
                    item_goal = np.asarray(batch_item[1], dtype=np.float64)
                    item_waypoints = batch_item[2]
                    item_first = (
                        np.asarray(item_waypoints[0], dtype=np.float64)
                        if item_waypoints is not None and len(item_waypoints)
                        else item_goal
                    )
                    if (
                        _route_progress_class(
                            centre,
                            item_first,
                            item_goal,
                            batch_heading,
                            completed_transitions,
                            forward_turn_limit_deg=float(args.frontier_forward_turn_limit_deg),
                        )
                        == 0
                    ):
                        forward_markers.append(batch_item)
                if forward_markers and len(forward_markers) < len(observation_batch):
                    print(
                        f"[plan] dropped {len(observation_batch) - len(forward_markers)} "
                        "rearward filler marker(s) from the frozen batch; forward "
                        "exploration is not interrupted by a scheduled round trip."
                    )
                    observation_batch = forward_markers
                observation_batch_completed = 0
                forward_batch_remaining = len(observation_batch)
                _log_observation_batch_markers(rr, observation_batch)
                marker_text = ", ".join(
                    f"{index + 1}=({item[1][0]:+.2f},{item[1][1]:+.2f})"
                    for index, item in enumerate(observation_batch)
                )
                print(
                    f"[plan] froze {len(observation_batch)} numbered observation marker(s): "
                    f"{marker_text}. They will be visited in order before replanning."
                )
                cluster, goal_xy, _old_waypoints, close_look, observe_xy, expected_gain = observation_batch[0]
                waypoints = _plan_waypoints(analysis, centre, goal_xy)
                if waypoints is not None:
                    picked_target = (cluster, goal_xy, waypoints, close_look, observe_xy, expected_gain)
        executing_batch_marker = picked_target is not None and bool(observation_batch)
        if picked_target is None:
            no_route_stationary_refreshes += 1
            rear_route_deferrals = min(
                max(1, int(args.frontier_return_deferral_cycles)),
                rear_route_deferrals + 1,
            )
            # 'No target' has TWO very different meanings: either the map is
            # genuinely done, or the current costmap has no visible reachable
            # observation pose. This is a PLANNING result, not evidence that
            # localization is wrong. Refresh the stationary map/costmap once;
            # do not repeat pose solves against the same unchanged geometry.
            unconstrained_live = [
                c
                for c in analysis.clusters
                if not any(np.hypot(*(c.centroid_xy - v)) < float(args.visited_skip_m) for v in visited)
            ]
            live = [
                c
                for c in unconstrained_live
                if not _behind_completed_transition(
                    c.centroid_xy,
                    completed_transitions,
                    float(args.doorway_ratchet_slack_m),
                )
            ]
            if analysis.clusters and not unconstrained_live and retired_frontiers:
                # Every currently visible frontier was suppressed by observation
                # cooldown.  That does not mean exploration is complete.  Release
                # the oldest unresolved objective and let frontier utility choose
                # a fresh observation pose (sampled-viewpoint penalties still
                # favor a different angle).  This is frontier blacklist aging,
                # standard in long-running exploration systems.
                oldest_index = min(
                    range(len(retired_frontiers)),
                    key=lambda index: retired_frontiers[index][1],
                )
                oldest = retired_frontiers.pop(oldest_index)
                none_retries = 0
                print(
                    "[plan] all live frontiers are in observation cooldown; "
                    f"reactivating the oldest unresolved frontier at "
                    f"({oldest[0][0]:+.2f}, {oldest[0][1]:+.2f}) instead of "
                    "falling into coverage patrol."
                )
                continue
            if (
                active_frontier_xy is None
                and completed_transitions
                and unconstrained_live
                and room_entry_keyframes_remaining[0] <= 0
            ):
                completed_transitions.pop()
                none_retries = 0
                print(
                    "[explore] current room has no reachable frontier; releasing the "
                    "most recent doorway commitment and returning to the parent area."
                )
                continue
            if live and none_retries < 1 and no_route_stationary_refreshes <= 2:
                none_retries += 1
                print(
                    f"[explore] {len(live)} live frontier(s) but the current costmap has no "
                    "visible reachable viewpoint — taking one stationary map refresh before "
                    "replanning."
                )
                controller.halt()
                time.sleep(0.4)
                _integrate_at_viewpoint()
                # This is a planner refresh, not completion of an objective.
                # Do not let its scan count/gain leak into the next target's
                # completion or doorway-commitment decision.
                vp_last[0] = None
                vp_gain_last[0] = None
                continue
            unlimited_patrol = int(args.explore_viewpoints) <= 0 and float(args.max_mission_seconds) <= 0.0
            if unlimited_patrol:
                patrol = _pick_patrol_route(
                    analysis,
                    centre,
                    sampled_viewpoints,
                    blocked_xy=[item[0] for item in blocked_viewpoints],
                    completed_transitions=completed_transitions,
                    current_heading_deg=float(cur_pose.theta_deg) + forward_offset,
                    transition_backtrack_slack_m=float(args.doorway_ratchet_slack_m),
                    forward_turn_limit_deg=float(args.frontier_forward_turn_limit_deg),
                )
                if patrol is not None:
                    patrol_goal, patrol_waypoints = patrol
                    active_frontier_xy = None
                    none_retries = 0
                    print(
                        f"[patrol] no frontier route is currently actionable; "
                        f"unlimited exploration remains active. Patrolling to "
                        f"({patrol_goal[0]:+.2f}, {patrol_goal[1]:+.2f}) for a "
                        "new observation and frontier refresh."
                    )
                    patrol_start = _robot_centre(cur_pose).copy()
                    patrol_trail_start = len(trail)
                    patrol_rolling_submap = room_entry_keyframes_remaining[0] > 0
                    patrol_result = _drive_leg(
                        patrol_waypoints,
                        patrol_goal,
                        analysis,
                        face_xy=patrol_goal,
                        rolling_submap=patrol_rolling_submap,
                    )
                    if patrol_result:
                        controller.halt()
                        _integrate_at_viewpoint()
                    if patrol_result == "tracking lost":
                        localization_hold = True
                        localization_hold_attempts = 0
                        turnaround_recovery_attempted = False
                        _reset_active_localization_recovery()
                    here = _robot_centre(cur_pose).copy()
                    patrol_scans = max(
                        int(vp_last[0] or 0),
                        int(last_leg_mapping["scans"]),
                    )
                    patrol_gain = max(
                        int(vp_gain_last[0] or 0),
                        int(last_leg_mapping["gain"]),
                    )
                    made_progress = _navigation_action_made_progress(
                        patrol_start,
                        here,
                        patrol_scans,
                        patrol_gain,
                    )
                    if not made_progress:
                        blocked_viewpoints.append(
                            (
                                np.asarray(patrol_goal, dtype=np.float64).copy(),
                                planning_epoch + 6,
                            )
                        )
                        if len(blocked_viewpoints) > 32:
                            del blocked_viewpoints[:-32]
                        if patrol_result.startswith("hard stop (rotational collision box"):
                            print(
                                "[patrol] initial pivot was collision-constrained; "
                                "running the collision-checked local clearance "
                                "recovery before replanning."
                            )
                            recovery_state = _attempt_startup_clearance_escape()
                            if recovery_state == "moved":
                                _integrate_at_viewpoint()
                                here = _robot_centre(cur_pose).copy()
                                made_progress = _navigation_action_made_progress(
                                    patrol_start,
                                    here,
                                    int(vp_last[0] or 0),
                                    int(vp_gain_last[0] or 0),
                                )
                        if not made_progress:
                            print(
                                f"[patrol] progress checker rejected this action: "
                                f"the base moved {float(np.hypot(*(here - patrol_start))) * 100:.0f}cm "
                                f"and committed {patrol_gain} newly-known cells. "
                                "The failed goal is cooling down; replanning without "
                                "counting a completed observation."
                            )
                            vp_last[0] = None
                            vp_gain_last[0] = None
                            continue
                    executed_patrol = [patrol_start]
                    executed_patrol.extend(
                        np.asarray(point, dtype=np.float64).copy() for point in trail[patrol_trail_start:]
                    )
                    executed_patrol.append(here)
                    crossing = _crossed_passable_frontier(
                        analysis,
                        executed_patrol,
                    )
                    if (
                        crossing is not None
                        and patrol_scans > 0
                        and patrol_gain >= int(args.doorway_ratchet_min_gain_cells)
                    ):
                        anchor, outward = crossing
                        _record_completed_transition(
                            anchor,
                            outward,
                            "overlap-verified patrol crossing",
                        )
                    elif patrol_rolling_submap and patrol_scans > 0 and room_entry_keyframes_remaining[0] > 0:
                        room_entry_keyframes_remaining[0] -= 1
                    sampled_viewpoints.append(here)
                    viewpoints_reached += 1
                    vp_last[0] = None
                    vp_gain_last[0] = None
                    # A patrol leg that moved and mapped is real progress: give
                    # the next planning pass its full route ladder back instead
                    # of leaving the no-route escalation latched.
                    no_route_stationary_refreshes = 0
                    print(
                        f"[patrol] observation {viewpoints_reached} complete; "
                        "re-running SLAM frontier detection."
                    )
                    continue
                controller.halt()
                if no_route_stationary_refreshes <= 3:
                    print(
                        "[patrol] no collision-free patrol translation is currently "
                        "available; taking one final stationary observation before "
                        "making a bounded completion decision."
                    )
                    _integrate_at_viewpoint()
                    final_gain = int(vp_gain_last[0] or 0)
                    vp_last[0] = None
                    vp_gain_last[0] = None
                    if final_gain > 0 and no_route_stationary_refreshes <= 2:
                        print(
                            f"[patrol] final observation revealed {final_gain} new cells; replanning immediately."
                        )
                        continue
                else:
                    # The refresh budget is spent: repeated in-place scans were
                    # only shaving single-digit noise cells while never
                    # creating a route.  Stop observing and make a movement
                    # decision instead.
                    final_gain = 0
                if unconstrained_live and no_route_stationary_refreshes <= 3:
                    # No forward coverage translation exists, but unresolved
                    # frontiers still do. In unlimited exploration this means
                    # the forward-only preference has exhausted the current
                    # heading; it does not mean the mission is complete or
                    # physically stuck. Mature the bounded return-route gate so
                    # the next global planning pass deliberately selects an
                    # unresolved frontier. This is distinct from patrol: the
                    # destination must provide actual frontier information.
                    rear_route_deferrals = max(
                        1,
                        int(args.frontier_return_deferral_cycles),
                    )
                    none_retries = 0
                    print(
                        "[patrol] no forward patrol motion remains, but live "
                        "frontiers still exist; admitting one explicit "
                        "frontier-directed return route instead of stopping."
                    )
                    continue
                if unconstrained_live:
                    # The return-route admission has been offered for several
                    # consecutive planning passes and never produced a route —
                    # looping on it means standing still scanning forever.
                    # Fall through to the fallback coverage patrol below: it
                    # releases the doorway ratchets and allows a full 180deg
                    # turn, so the robot physically drives back through mapped
                    # space toward the unresolved frontiers instead of
                    # deliberating in place.
                    print(
                        "[patrol] return-route admission produced no route for "
                        f"{no_route_stationary_refreshes} consecutive planning passes; "
                        "releasing doorway ratchets and turning around for a physical "
                        "fallback patrol instead of scanning in place again."
                    )
                if final_gain > 0:
                    print(
                        "[patrol] stationary refresh budget is spent; the map changed but "
                        "did not create a route, so selecting a physical patrol/return "
                        "decision rather than scanning in place again."
                    )
                # No frontier is not a mission-completion condition in an
                # unlimited wander.  The map may be temporarily saturated,
                # or the robot may have crossed into a room whose frontier
                # cells are still outside the current observation window.
                # Release doorway ratchets and run a coverage patrol over any
                # reachable known-free space instead of stopping at the room
                # boundary.  This is deliberately a bounded fallback: it
                # still uses the normal collision-checked planner and local
                # LiDAR watchdog, but is allowed to return through a mapped
                # doorway to discover the next opening.
                if unlimited_patrol:
                    fallback_patrol = _pick_patrol_route(
                        analysis,
                        centre,
                        sampled_viewpoints,
                        blocked_xy=[item[0] for item in blocked_viewpoints],
                        completed_transitions=[],
                        current_heading_deg=float(cur_pose.theta_deg) + forward_offset,
                        transition_backtrack_slack_m=float(args.doorway_ratchet_slack_m),
                        forward_turn_limit_deg=180.0,
                    )
                    if fallback_patrol is not None:
                        patrol_goal, patrol_waypoints = fallback_patrol
                        active_frontier_xy = None
                        none_retries = 0
                        print(
                            "[patrol] no new frontier is currently visible; "
                            "continuing mapped-space coverage toward "
                            f"({patrol_goal[0]:+.2f}, {patrol_goal[1]:+.2f}) "
                            "until another frontier appears."
                        )
                        patrol_start = _robot_centre(cur_pose).copy()
                        patrol_trail_start = len(trail)
                        patrol_result = _drive_leg(
                            patrol_waypoints,
                            patrol_goal,
                            analysis,
                            face_xy=patrol_goal,
                            rolling_submap=False,
                        )
                        if patrol_result:
                            controller.halt()
                            _integrate_at_viewpoint()
                        here = _robot_centre(cur_pose).copy()
                        patrol_scans = max(int(vp_last[0] or 0), int(last_leg_mapping["scans"]))
                        patrol_gain = max(int(vp_gain_last[0] or 0), int(last_leg_mapping["gain"]))
                        if not _navigation_action_made_progress(
                            patrol_start, here, patrol_scans, patrol_gain
                        ):
                            blocked_viewpoints.append(
                                (np.asarray(patrol_goal, dtype=np.float64).copy(), planning_epoch + 3)
                            )
                        else:
                            # Productive fallback leg: unlatch the no-route
                            # escalation so the next pass can use observations
                            # and return routes again.
                            no_route_stationary_refreshes = 0
                        sampled_viewpoints.append(here)
                        viewpoints_reached += 1
                        vp_last[0] = None
                        vp_gain_last[0] = None
                        continue
                    if no_route_stationary_refreshes <= 6:
                        controller.halt()
                        _integrate_at_viewpoint()
                        vp_last[0] = None
                        vp_gain_last[0] = None
                        print(
                            "[patrol] no reachable coverage route in the current "
                            "costmap; refreshed LiDAR and continuing the wander loop."
                        )
                        time.sleep(0.25)
                        continue
                aborted = "no collision-free motion or new map information remains"
                print(
                    "[patrol] exploration cannot make physical or informational "
                    "progress; exiting instead of entering an idle retry loop."
                )
                break
            if live:
                if blocked_viewpoints:
                    released = len(blocked_viewpoints)
                    blocked_viewpoints.clear()
                    none_retries = 0
                    print(
                        f"[explore] global planning exhausted while {released} "
                        "temporary failed-approach cost(s) were active; clearing "
                        "that local failure memory and replanning before declaring "
                        "the frontiers unreachable."
                    )
                    continue
                if active_frontier_xy is not None:
                    print(
                        "[explore] active path ended at the known-space boundary — "
                        "taking one snapshot, then selecting from the updated map."
                    )
                    _integrate_at_viewpoint()
                    sampled_viewpoints.append(_robot_centre(cur_pose).copy())
                    viewpoints_reached += 1
                    active_frontier_xy = None
                    vp_last[0] = None
                    vp_gain_last[0] = None
                    none_retries = 0
                    continue
                print(
                    f"[explore] {len(live)} frontier(s) remain but are unreachable from anywhere "
                    "the robot can stand — stopping honestly (NOT 'complete')."
                )
                aborted = f"{len(live)} frontier(s) remain unreachable"
            else:
                if analysis.clusters:
                    print(
                        f"[explore] {len(analysis.clusters)} significant frontier(s) remain, "
                        "but all were retired as blocked — stopping, NOT complete."
                    )
                    aborted = f"{len(analysis.clusters)} blocked frontier(s) remain"
                else:
                    print("[explore] no significant frontier remains — the map is complete.")
            _log_world(rr, world_map, args, centre, trail, analysis, note="exploration finished")
            break
        none_retries = 0
        no_route_stationary_refreshes = 0
        rear_route_deferrals = 0
        cluster, goal_xy, waypoints, close_look, observe_xy, expected_gain = picked_target
        if active_frontier_xy is None:
            if forward_batch_remaining <= 0:
                forward_batch_remaining = max(1, int(args.frontier_batch_size))
                print(
                    f"[plan] starting a {forward_batch_remaining}-target forward observation batch; "
                    "rearward frontiers remain disabled until this batch is complete."
                )
            print(
                f"[plan] forward batch target "
                f"{max(1, int(args.frontier_batch_size)) - forward_batch_remaining + 1}/"
                f"{max(1, int(args.frontier_batch_size))} selected."
            )
        objective_start_xy = centre.copy()
        objective_trail_start = len(trail)
        continuing_objective = active_frontier_xy is not None
        active_frontier_xy = cluster.centroid_xy.copy()
        kind = "CLOSE LOOK at" if close_look else "ADVANCE toward"
        ownership = "continuing active" if continuing_objective else "new active"
        opening_kind = "PASSABLE" if cluster.passable else "NARROW-LOOK"
        print(
            f"[explore] {ownership} {opening_kind} frontier: span {cluster.span_m:.2f}m at "
            f"({cluster.centroid_xy[0]:+.2f}, {cluster.centroid_xy[1]:+.2f}), "
            f"{kind} visible cell ({observe_xy[0]:+.2f}, {observe_xy[1]:+.2f}) "
            f"from ({goal_xy[0]:+.2f}, {goal_xy[1]:+.2f}), "
            f"expected gain {expected_gain} cells, {len(waypoints)} waypoint(s)."
        )
        # PLANNED path, logged ONCE per plan as a static green polyline — compare it
        # against the orange executed trail to see intent vs reality.
        # -------------------------------------------------------------------
        # EXPLORE PHASE 3 — COMMIT AND FOLLOW THIS TARGET'S PATH
        # -------------------------------------------------------------------
        plan_line = [[float(centre[0]), float(centre[1]), 0.0]] + [
            [float(p[0]), float(p[1]), 0.0] for p in waypoints
        ]
        rr.log("world/plan", rr.LineStrips3D([plan_line], colors=[[70, 230, 100]], radii=0.014))
        _log_world(
            rr,
            world_map,
            args,
            centre,
            trail,
            analysis,
            observe_xy,
            waypoints,
            note=f"driving to frontier (span {cluster.span_m:.2f}m)",
        )

        fkey = (int(round(cluster.centroid_xy[0] / 0.3)), int(round(cluster.centroid_xy[1] / 0.3)))
        rolling_submap_active = room_entry_keyframes_remaining[0] > 0
        leg_failed = _drive_leg(
            waypoints,
            goal_xy,
            analysis,
            face_xy=observe_xy,
            overlap_handoff=bool(cluster.passable),
            rolling_submap=rolling_submap_active,
            continuing_doorway=bool(cluster.passable and continuing_objective),
        )
        objective_translation = float(np.hypot(*(_robot_centre(cur_pose) - objective_start_xy)))
        zero_motion_without_scan = (
            not leg_failed
            and objective_translation < 0.05
            and int(vp_last[0] or 0) == 0
            and int(last_leg_mapping["scans"]) == 0
        )
        if zero_motion_without_scan:
            zero_motion_strikes[fkey] = zero_motion_strikes.get(fkey, 0) + 1
        else:
            zero_motion_strikes.pop(fkey, None)

        if not leg_failed and cluster.passable and zero_motion_strikes.get(fkey, 0) >= 2:
            # The route solver can legitimately return an observation pose that
            # is already under the robot. It is not a new doorway approach and
            # cannot generate a diverse keyframe. Retaining topological ownership
            # here created an unbounded zero-waypoint planning loop. Treat two
            # consecutive zero-motion selections as a temporarily exhausted
            # viewpoint, release ownership, and let the global planner advance.
            _defer_frontier(cluster.centroid_xy, cooldown_epochs=8)
            blocked_viewpoints.append(
                (
                    np.asarray(goal_xy, dtype=np.float64).copy(),
                    planning_epoch + 8,
                )
            )
            if len(blocked_viewpoints) > 32:
                del blocked_viewpoints[:-32]
            active_frontier_xy = None
            zero_motion_strikes.pop(fkey, None)
            vp_last[0] = None
            vp_gain_last[0] = None
            consistency_fails = 0
            print(
                "[plan] active doorway selected the robot's current observation "
                "pose twice without motion or a new scan; releasing it for a "
                "bounded cooldown and selecting another forward frontier."
            )
            continue
        clearance_limited = leg_failed.startswith("hard stop")
        rotation_limited = leg_failed.startswith("hard stop (rotational collision box")
        actuator_no_motion = leg_failed.startswith("hard stop (pivot physically stalled)")
        if leg_failed == "mission time budget spent":
            aborted = leg_failed
            break
        if leg_failed == "navigation sensor unavailable":
            controller.halt()
            aborted = "IMU stream unavailable; heading-controlled exploration cannot continue"
            print(f"[explore] SENSOR EXIT: {aborted}.")
            break
        if leg_failed == "navigation lidar unavailable":
            controller.halt()
            aborted = "LiDAR stream unavailable; ExploreSystem requires fresh scans"
            print(f"[explore] SENSOR EXIT: {aborted}.")
            break
        if leg_failed == "pivot stalled":
            # Defensive compatibility for any older drive path that still
            # reports this result.  Re-selecting viewpoints cannot cure a turn
            # that did not move from the current pose; doing so was the source
            # of the unbounded planning loop.  Convert it to the same bounded
            # clearance-boundary mapping transition used by physical obstacles.
            controller.halt()
            controller.clear_safety_latch()
            print(
                "[explore] zero-progress pivot is a local clearance boundary; "
                "mapping it once and advancing the exploration state."
            )
            leg_failed = "hard stop (pivot physically stalled)"
            clearance_limited = True
        if leg_failed == "obstacle - replan":
            # The obstacle is now marked in the occupancy grid (and excluded from future
            # novelty checks); re-plan around it. Progress guarantee: if the SAME
            # frontier gets obstacle-blocked twice, give up on it — endless
            # replan-loops against one target are how the mission dies in place.
            obstacle_strikes[fkey] = obstacle_strikes.get(fkey, 0) + 1
            if obstacle_strikes[fkey] >= 2:
                print("[explore] frontier blocked twice — giving up on it, trying the next.")
                _retire_frontier(cluster.centroid_xy)
                active_frontier_xy = None
            else:
                print("[explore] re-planning around the obstacle ...")
            continue
        if actuator_no_motion:
            # This is not a mapped boundary and not a completed observation.
            # The previous implementation photographed the unchanged starting
            # pose, retired a distant frontier, and repeated that fiction for
            # every target. Restart the command owner once, then use the direct
            # collision-checked escape path as an independent actuator check.
            controller.shutdown()
            actuator_no_motion_attempts += 1
            recovery_state = _attempt_startup_clearance_escape()
            controller.start()
            if recovery_state == "moved":
                actuator_no_motion_attempts = 0
                print(
                    "[drive] direct collision-checked motion verified after the "
                    "control-channel restart; preserving the objective and "
                    "replanning from the measured new pose."
                )
                continue
            if actuator_no_motion_attempts == 1:
                print(
                    "[drive] navigation command channel restarted after zero "
                    "physical response; retrying the SAME objective once with "
                    "the fresh thread-owned transport."
                )
                continue
            aborted = (
                controller.command_error()
                or "base actuator accepted neither pivot nor collision-checked translation"
            )
            print(
                f"[drive] ACTUATOR FAULT: {aborted}. The objective was not marked "
                "observed and no false map snapshot was committed."
            )
            break
        if leg_failed.startswith("hard stop"):
            # A clearance boundary is itself a useful observation pose. Scan
            # once, merge, and let the updated map select the next viewpoint.
            # Never reverse automatically: the rear LiDAR sector is occluded.
            controller.halt()
            controller.clear_safety_latch()
            print("[explore] clearance boundary reached — taking a stationary map here.")
            _face_snapshot_target(
                np.asarray(observe_xy, dtype=np.float64),
                "clearance-boundary frontier",
            )
            _integrate_at_viewpoint()
            leg_failed = ""
        else:
            actuator_no_motion_attempts = 0
        if leg_failed == "tracking lost":
            # Localization hiccup, NOT a bad target — don't blacklist the exit for
            # it (field: robot reached the doorway, confidence dipped, and the old
            # policy marked THE EXIT visited and turned away). Keep the objective
            # and retry stationary localization without repeating recovery spins.
            lost_strikes[fkey] = lost_strikes.get(fkey, 0) + 1
            localization_hold = True
            localization_hold_attempts = 0
            turnaround_recovery_attempted = False
            _reset_active_localization_recovery()
            print(
                "[explore] tracking lost; preserving the active frontier and entering "
                "stationary localization hold."
            )
            continue
        if leg_failed == "leg time budget spent":
            # A leg timer is a scheduling slice, not evidence that the frontier
            # is bad. Start another slice toward the same persistent objective.
            print("[explore] leg time slice spent — continuing the same active frontier.")
            continue
        if leg_failed:
            print(
                f"[explore] leg failed ({leg_failed}) — stopping with the active objective "
                "preserved instead of switching destinations."
            )
            aborted = leg_failed
            break

        # DIVERGENCE WATCHDOG: two consecutive viewpoints that mapped nothing
        # (weak or discarded batches) means the map and reality have parted ways.
        # Recover once against the GOLD reference; if that fails, stop honestly
        # instead of continuing to paint garbage ('once the map is lost it's
        # game over' — the user's words, and correct).
        # -------------------------------------------------------------------
        # EXPLORE PHASE 4 — VERIFY THE SNAPSHOT/MAP UPDATE FROM _drive_leg()
        # -------------------------------------------------------------------
        viewpoint_scans_added = max(
            int(vp_last[0] or 0),
            int(last_leg_mapping["scans"]),
        )
        if viewpoint_scans_added == 0:
            if clearance_limited:
                # No usable scan at a physical clearance boundary is a blocked
                # target result, not evidence that global localization failed.
                consistency_fails = 0
            else:
                consistency_fails += 1
            if not clearance_limited and consistency_fails >= 2:
                if not cluster.passable:
                    # A failed photograph at one non-crossable target is not
                    # evidence that the entire map or navigation pose is lost.
                    _retire_frontier(cluster.centroid_xy)
                    active_frontier_xy = None
                    consistency_fails = 0
                    vp_last[0] = None
                    vp_gain_last[0] = None
                    print(
                        "[explore] narrow look target produced two unusable batches - "
                        "retiring it and continuing to the next frontier."
                    )
                    continue
                print("[explore] map-consistency watchdog: two failed viewpoints.")
                # A rejected map transaction is not evidence that a physical
                # doorway stopped existing.  Retiring it here was the exact
                # path that sent the robot back to old-room photography after
                # a healthy, strongly tracked crossing. Preserve topological
                # ownership and force a fresh translated keyframe attempt.
                active_frontier_xy = cluster.centroid_xy.copy()
                consistency_fails = 0
                vp_last[0] = None
                vp_gain_last[0] = None
                print(
                    "[explore] passable target produced two conservative map "
                    "transactions — retaining doorway ownership and continuing "
                    "outward from a translated pose; no old-room target switch."
                )
                continue
        elif viewpoint_scans_added > 0:
            consistency_fails = 0
        vp_last[0] = None

        # Snapshot complete: release the travel objective. The next selection is
        # made from the updated map and this scan's measured information gain.
        # Persistence applies only while travelling/replanning, never after a
        # photograph has completed.
        # -------------------------------------------------------------------
        # EXPLORE PHASE 5 — CLASSIFY RESULT, THEN RETURN TO PHASE 2
        # -------------------------------------------------------------------
        viewpoints_reached += 1
        if executing_batch_marker and observation_batch:
            completed_marker = observation_batch_completed + 1
            observation_batch.pop(0)
            observation_batch_completed += 1
            forward_batch_remaining = len(observation_batch)
            _log_observation_batch_markers(
                rr,
                observation_batch,
                completed=observation_batch_completed,
            )
            print(
                f"[plan] observation marker {completed_marker} mapped; "
                f"{len(observation_batch)} frozen marker(s) remain."
            )
            if not observation_batch:
                print(
                    "[plan] observation batch complete; rebuilding the frontier set "
                    "before choosing the next three markers."
                )
        elif not continuing_objective and forward_batch_remaining > 0:
            forward_batch_remaining -= 1
            if forward_batch_remaining == 0:
                print(
                    "[plan] forward observation batch complete; rebuilding the "
                    "frontier set from the newly verified map before selecting "
                    "the next batch."
                )
        here = _robot_centre(cur_pose).copy()
        sampled_viewpoints.append(here)
        frontier_deferred_by_clearance = False
        if rotation_limited:
            rotation_block_strikes[fkey] = rotation_block_strikes.get(fkey, 0) + 1
            print(
                "[nav] initial turn is collision-constrained; running a "
                "collision-checked backup/strafe recovery before global replanning."
            )
            recovery_state = _attempt_startup_clearance_escape()
            if recovery_state == "moved":
                # Register the verified body translation in the SLAM pose/map
                # before asking the global planner for another path.
                _integrate_at_viewpoint()
                print(
                    "[nav] local clearance recovery moved the base and refreshed "
                    "SLAM; prior failed-approach costs remain active for replanning."
                )
            blocked_viewpoints.append(
                (
                    np.asarray(goal_xy, dtype=np.float64).copy(),
                    planning_epoch + 6,
                )
            )
            # Bound transient local failure memory independently of scan count.
            if len(blocked_viewpoints) > 32:
                del blocked_viewpoints[:-32]
            blocked_count, distinct_approaches_exhausted = _record_distinct_blocked_approach(
                blocked_approach_history,
                fkey,
                np.asarray(goal_xy, dtype=np.float64),
                distinct_distance_m=0.30,
                max_distinct_attempts=2,
            )
            repeated_same_approach = rotation_block_strikes[fkey] >= 2
            frontier_deferred_by_clearance = distinct_approaches_exhausted or repeated_same_approach
            if frontier_deferred_by_clearance and cluster.passable:
                # Two spatially distinct collision-checked approaches have now
                # shown that this objective cannot currently be entered. Keeping
                # permanent ownership made the planner alternate translations
                # and refused pivots forever. Defer (never permanently discard)
                # the doorway so another forward frontier can add map evidence;
                # the cooldown makes this opening eligible again later.
                _defer_frontier(cluster.centroid_xy)
                blocked_approach_history.pop(fkey, None)
                total_blocked_attempts = rotation_block_strikes.pop(fkey, 0)
                print(
                    f"[plan] passable-frontier approach ({goal_xy[0]:+.2f}, "
                    f"{goal_xy[1]:+.2f}) failed after {total_blocked_attempts} "
                    f"blocked attempt(s) from {blocked_count} distinct "
                    "collision-checked approaches; releasing doorway ownership "
                    "for a bounded cooldown so exploration continues elsewhere."
                )
            elif frontier_deferred_by_clearance:
                _defer_frontier(cluster.centroid_xy)
                # The cooldown permits new map evidence and a new set of
                # approaches later; do not make today's failures permanent.
                blocked_approach_history.pop(fkey, None)
                total_blocked_attempts = rotation_block_strikes.pop(fkey, 0)
                print(
                    f"[plan] collision-blocked approach ({goal_xy[0]:+.2f}, "
                    f"{goal_xy[1]:+.2f}) blacklisted; the frontier failed after "
                    f"{total_blocked_attempts} blocked attempt(s) from "
                    f"{blocked_count} distinct "
                    "collision-checked approaches; deferring it while the explorer "
                    "maps another frontier/patrol area."
                )
            else:
                print(
                    f"[plan] collision-blocked approach ({goal_xy[0]:+.2f}, "
                    f"{goal_xy[1]:+.2f}) blacklisted; the frontier remains eligible "
                    "from another viewpoint."
                )
        else:
            rotation_block_strikes.pop(fkey, None)
        actual_gain = max(
            int(vp_gain_last[0] or 0),
            int(last_leg_mapping["gain"]),
        )
        vp_gain_last[0] = None
        active_frontier_xy = None
        transition = None
        if cluster.passable and not rotation_limited:
            transition = _doorway_transition(
                cluster.centroid_xy,
                objective_start_xy,
                here,
                observe_xy,
                viewpoint_scans_added,
                actual_gain,
                int(args.doorway_ratchet_min_gain_cells),
            )
            if transition is None:
                executed_leg = [objective_start_xy]
                executed_leg.extend(
                    np.asarray(point, dtype=np.float64).copy() for point in trail[objective_trail_start:]
                )
                executed_leg.append(here)
                geometric_crossing = _crossed_passable_frontier(
                    analysis,
                    executed_leg,
                )
                if geometric_crossing is not None:
                    # Room topology and map insertion confidence are separate
                    # state estimates. A collision-checked, continuously
                    # localized trajectory can prove the chassis crossed a
                    # doorway even when a conservative stationary map write is
                    # withheld. Record that crossing so the global planner may
                    # not immediately choose the old side of the doorway.
                    transition = geometric_crossing
                    print(
                        "[explore] passable-frontier crossing verified from the "
                        "executed trajectory; committing room topology "
                        "independently of the stationary map transaction."
                    )
        if transition is not None:
            anchor, outward = transition
            _record_completed_transition(
                anchor,
                outward,
                "overlap-verified passable-frontier crossing",
            )
            _retire_frontier(cluster.centroid_xy)
            qualifier = "clearance-limited " if clearance_limited else ""
            print(
                f"[explore] {qualifier}passable frontier revealed a new room - "
                "doorway commitment armed; continuing outward before any return."
            )
        elif cluster.passable and not rotation_limited and viewpoint_scans_added > 0:
            # A photographed passable frontier is not yet a completed doorway.
            # Keep the global objective committed to this opening until the
            # chassis crosses it. Selecting side/rear frontiers while the body
            # occupies the doorway caused repeated refused pivots and visible
            # left/right meandering. Only the exact observation region is
            # retained as the overlap anchor; the next planning cycle bypasses
            # generic viewpoint search and commits a route through the opening.
            active_frontier_xy = cluster.centroid_xy.copy()
            print(
                "[explore] passable frontier photographed without an executed "
                "crossing - retaining doorway transit ownership; the next "
                "action is a direct route through the photographed opening."
            )
        elif cluster.passable and not rotation_limited:
            # Even a conservative/no-commit snapshot does not cancel a valid
            # doorway route. Continue the same topological transition; mapping
            # checkpoints along the translated leg will reacquire overlap.
            active_frontier_xy = cluster.centroid_xy.copy()
            print(
                "[explore] doorway snapshot was conservative; retaining the "
                "same passable-frontier transit instead of selecting a side target."
            )
        elif not rotation_limited and _viewpoint_completes_frontier(cluster, viewpoint_scans_added):
            _retire_frontier(cluster.centroid_xy)
            print(
                "[explore] narrow/non-crossable frontier observed successfully — "
                "retiring this look target so planning advances."
            )
        elif rotation_limited and not frontier_deferred_by_clearance:
            if cluster.passable:
                active_frontier_xy = cluster.centroid_xy.copy()
                print(
                    "[explore] doorway pivot was constrained, but the passage "
                    "remains the committed objective; replanning a translated "
                    "low-turn approach instead of turning back."
                )
            else:
                print(
                    "[explore] target was not reached because its initial rotational "
                    "trajectory was blocked; leaving the frontier eligible for a "
                    "different approach."
                )
        elif rotation_limited and frontier_deferred_by_clearance:
            active_frontier_xy = None
            print(
                "[explore] repeatedly rotation-constrained doorway deferred; "
                "selecting a different forward mapping objective."
            )
        elif clearance_limited:
            _retire_frontier(cluster.centroid_xy)
            print(
                "[explore] blocked approach was mapped once — retiring this "
                "viewpoint so the planner selects a different frontier."
            )
        if (
            transition is None
            and rolling_submap_active
            and viewpoint_scans_added > 0
            and room_entry_keyframes_remaining[0] > 0
        ):
            room_entry_keyframes_remaining[0] -= 1
        if executing_batch_marker:
            # The next queued marker—not a newly photographed doorway—owns the
            # next global action. Its route may be replanned, but the marker
            # itself is not replaced by an opportunistic side objective.
            active_frontier_xy = None
        # NEARBY GAP FILL (user directive 2026-08-05): before leaving a
        # completed viewpoint, sweep up to two frontier gaps within easy
        # reach — one collision-checked pivot plus one stationary stop-scan
        # each — so small holes beside the robot are closed with a single
        # turn now instead of surviving as ragged map edges or becoming
        # dedicated route targets later.
        gap_analysis = _analysis_now()
        here_now = _robot_centre(cur_pose)
        nearby_gaps = sorted(
            (
                c
                for c in gap_analysis.clusters
                if float(np.hypot(*(c.centroid_xy - here_now))) <= 1.60
            ),
            key=lambda c: float(np.hypot(*(c.centroid_xy - here_now))),
        )[:2]
        for gap in nearby_gaps:
            if not _face_snapshot_target(
                np.asarray(gap.centroid_xy, dtype=np.float64), "nearby gap"
            ):
                continue
            _integrate_at_viewpoint()
            gap_gain = int(vp_gain_last[0] or 0)
            vp_last[0] = None
            vp_gain_last[0] = None
            print(
                f"[explore] nearby-gap snapshot at ({gap.centroid_xy[0]:+.2f}, "
                f"{gap.centroid_xy[1]:+.2f}) gained {gap_gain} cells before moving on."
            )
        kind_msg = "close look" if close_look else "advance"
        print(
            f"[explore] {kind_msg} {viewpoints_reached} done; map now {len(world_map.scans)} "
            f"scans, actual gain {actual_gain} newly-known cells; selecting next viewpoint."
        )
        _log_world(
            rr,
            world_map,
            args,
            _robot_centre(cur_pose),
            trail,
            _analysis_now(),
            note=f"stop {viewpoints_reached} mapped",
        )

    _send_stop(robot)
    final_analysis = _analysis_now()
    _log_world(
        rr, world_map, args, _robot_centre(cur_pose), trail, final_analysis, note="exploration finished"
    )
    _save_current_map()

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
    print(
        f"  {spin_scan_count} spin scans + {len(world_map.scans) - spin_scan_count} "
        f"tracked-while-driving scans = {len(world_map.scans)} total."
    )
    print(f"  viewpoints reached: {viewpoints_reached}.")
    print(
        f"  significant frontiers remaining: {len(final_analysis.clusters)} "
        f"(was {frontiers_left} at first analysis)."
    )
    print(
        "  Map = world/lidar_map; frontier centroids amber, target green sphere, PLANNED path "
        "green line (world/plan), executed trail orange (world/trail)."
    )
    handoff_exit = bool(args.spin_only) and bool(args.exit_after_spin_only)
    if handoff_exit:
        print("  Handoff mode: initial spin map saved; closing so the navigator can start.")
    else:
        print("  Leave running to keep the viewer up; Ctrl+C to exit.")
    print("========================================")

    try:
        while not handoff_exit:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for _cleanup in (
            controller.shutdown,
            (ground_odom.stop if ground_odom is not None else (lambda: None)),
            lambda: _send_stop(robot),
            _stop_live_lidar_publisher,
            imu.stop,
            feed.stop,
            (cam_sub.stop if cam_sub is not None else (lambda: None)),
            robot.disconnect,
        ):
            with contextlib.suppress(BaseException):
                _cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

