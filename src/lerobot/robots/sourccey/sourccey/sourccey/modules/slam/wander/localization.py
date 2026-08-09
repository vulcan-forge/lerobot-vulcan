"""Relocalize the robot against the stitched map, and judge the result.

``_estimate_pose_against_stitched_map`` scan-matches the current LiDAR points
against the accumulated map to solve a pose; ``_accept_relocalized_pose`` decides
whether that candidate is trustworthy (score + how far it strays from the
expected pose); ``_pose_delta_metrics`` measures that stray; ``_log_live_pose_state``
prints the running pose-health line. This is the "where am I" half of SLAM — the
map-growing half is ``mapping.py``.
"""
from __future__ import annotations

import math

import numpy as np

from ..lidar.direct_snapshot_stitch import (
    MotionHint,
    Pose2D,
    _search_pose,
    _transform_points,
)
from .wander_types import _normalize_angle_deg

def _pose_delta_metrics(reference_pose: Pose2D, candidate_pose: Pose2D) -> tuple[float, float]:
    """How far apart two poses are, split into distance and heading.

    Returns ``(translation_m, theta_error_deg)``: the straight-line XY distance
    between the poses, and the absolute wrapped heading difference. Used to ask
    "how far did this candidate pose stray from where we expected to be?" when
    deciding whether to trust a relocalization.
    """
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
    """Scan-match the current LiDAR points against the whole stitched map to
    solve for the robot's pose.

    Gathers every non-empty prior snapshot cloud into one big reference point
    set, then hands the live points, that reference, and an ``initial_pose``
    guess to the engine's ``_search_pose``. The search sweeps translation
    (``search_xy_m``) and rotation (``theta_window_deg``, with a wider whole-map
    theta sweep centered on the guess) to find the best-scoring alignment, gently
    biased toward the prior pose so it doesn't jump without evidence. Returns
    ``(solved_pose, score_meta)``, or ``None`` if there aren't enough points to
    even try (fewer than 12, or an empty map).
    """
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
    """Decide whether a solved candidate pose is trustworthy enough to adopt.

    Pulls the match score(s) and source out of ``score_meta`` and measures how
    far the candidate strayed from ``expected_pose``. It REJECTS (returns False,
    logging every reason) if: the score is below ``min_score``, the translation
    or heading error exceeds the caps, or a whole-map match didn't beat the local
    match by a clear margin (a weak whole-map win is how the room's symmetry
    produces a confident-looking wrong pose). Otherwise it logs an accept line and
    returns True. It only judges — the caller decides what to do with the verdict.
    """
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
    body_radius_m: float = 0.29,
) -> None:
    """Draw the robot's current pose into the Rerun viewer for live debugging.

    Logs a cyan dot + heading arrow at the pose, two footprint rings (the bright
    planning-collision radius and the dimmer squeeze radius it will actually
    thread), and the live scan transformed into world coordinates so you can see
    what the robot sees from where it thinks it is. Purely visualization — it
    changes no state and feeds no decision.
    """
    rr.set_time("capture_index", sequence=int(capture_index))
    origin = np.asarray([[pose.x, pose.y, 0.0]], dtype=np.float32)
    vector = np.asarray(
        [[0.30 * math.cos(math.radians(pose.theta_deg)), 0.30 * math.sin(math.radians(pose.theta_deg)), 0.0]],
        dtype=np.float32,
    )
    rr.log("world/live_pose/origin", rr.Points3D(origin, colors=[[0, 255, 255]], radii=0.06))
    rr.log("world/live_pose/heading", rr.Arrows3D(origins=origin, vectors=vector, colors=[[0, 255, 255]]))

    def _circle(radius_m: float) -> np.ndarray:
        """48 XYZ points evenly around a horizontal circle of the given radius,
        centered on the pose — the vertices for one footprint ring."""
        return np.asarray(
            [
                [pose.x + radius_m * math.cos(a), pose.y + radius_m * math.sin(a), 0.0]
                for a in np.linspace(0.0, 2.0 * math.pi, 48)
            ],
            dtype=np.float32,
        )

    # Body-footprint gauge: bright cyan = the PLANNING collision radius (what
    # the planner treats the robot as — a gap must exceed this DIAMETER to be
    # "reachable"); dim teal = the SQUEEZE radius it will actually attempt to
    # thread (radius - 2cm; was -5cm until 2026-07-18 — that margin turned the
    # robot into tables at dangerous closeness, and the doorway pass proved
    # the extra 3cm was never what unblocked it).
    squeeze_radius_m = max(0.25, float(body_radius_m) - 0.02)
    rr.log(
        "world/live_pose/body_radius",
        rr.LineStrips3D([_circle(float(body_radius_m))], colors=[[0, 255, 255]], radii=0.006),
    )
    rr.log(
        "world/live_pose/squeeze_radius",
        rr.LineStrips3D([_circle(squeeze_radius_m)], colors=[[0, 140, 150]], radii=0.004),
    )
    if len(points_xy):
        world_points_xy = _transform_points(points_xy, pose)
        world_points_xyz = np.column_stack(
            [world_points_xy[:, 0], world_points_xy[:, 1], np.zeros((len(world_points_xy),), dtype=np.float32)]
        )
        rr.log("world/live_scan", rr.Points3D(world_points_xyz, colors=[[130, 200, 255]], radii=0.01))


