"""Rotation-snapshot capture and map STITCHING.

``_capture_snapshot`` grabs one fresh LiDAR revolution and saves it as a
``Snapshot``. ``_append_stitch``/``_rebuild_stitch`` fold a new snapshot into the
growing stitched map (scan-matching it against the accumulated cloud) and rewrite
the debug HTML overlay; ``_save_motion_hints``/``_write_stitch_report`` persist
the bookkeeping. This is the "grow the map" half of the SLAM loop — localization
(the "where am I in that map" half) lives in ``localization.py``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..lidar.direct_snapshot_client import DirectLidarFeed, _save_snapshot, _scan_to_local_points
from ..lidar.direct_snapshot_stitch import (
    HTML_TEMPLATE,
    MotionHint,
    Pose2D,
    Snapshot,
    _append_stitch_snapshot,
    _load_snapshots,
    _pose_dict,
    _stitch_snapshots,
    _transform_points,
)


class LidarBoxedInError(RuntimeError):
    """The lidar feed is healthy but almost every return is within the near-cutoff
    (the robot is nose-against a wall / surrounded). This is RECOVERABLE — the
    caller should back out / rotate to open space and retry, NEVER crash. Distinct
    from a degraded feed (a genuine host problem)."""


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
    """Wait for one clean LiDAR revolution, filter it to usable points, and save
    it as a ``Snapshot``.

    Loops up to 6 times asking the feed for a fresh full revolution newer than
    ``after_frame_id``, converting each to local points. It stops as soon as a
    revolution has enough usable points; if it keeps coming up short it
    distinguishes the two causes and says which — a DEGRADED FEED (too few raw
    points, a host/USB problem) versus BOXED IN (a full revolution arrives but
    almost everything is within the near-cutoff, i.e. the robot is physically
    wedged) — and raises with the right diagnosis if neither recovers. On success
    it writes the snapshot JSON + point cloud to disk and returns
    ``(frame_id, frame, local_points_xy, snapshot)``.
    """
    print(
        "[capture] waiting for fresh revolution "
        f"(after frame_id={after_frame_id}, min advances={fresh_frame_advances})"
    )
    frame = None
    frame_id = int(after_frame_id)
    local_points_xy = None
    # Two failure modes produce a capture too sparse to stitch, and they are
    # NOT the same problem:
    #   (1) DEGRADED FEED — the host serves partial revolutions (raw far
    #       below the ~240 norm), e.g. USB stutter / spin-up. A host issue.
    #   (2) BOXED IN — a FULL revolution arrives (raw ~240) but almost all
    #       returns fall within min_range_m (self-hits / walls right against
    #       the lidar), so few survive filtering. The robot is physically
    #       wedged / something is draped within ~min_range of the lidar. NOT
    #       a host issue (field 2026-07-18: baseline 168 self-hits, raw 241,
    #       usable 49 — misreported as a feed fault).
    # Either way the capture is refused (it would found a garbage map), but
    # the diagnosis must be accurate so the operator fixes the right thing.
    MIN_HEALTHY_LOCAL_POINTS = 60
    MIN_HEALTHY_RAW_POINTS = 150
    last_raw = 0
    last_usable = 0
    for attempt_index in range(6):
        armed_wall_ts = time.time()
        frame_id, frame = feed.wait_for_frame_after(
            after_frame_id=after_frame_id,
            timeout_s=float(fresh_frame_timeout_s),
            min_frame_advances=int(fresh_frame_advances),
            armed_wall_ts=armed_wall_ts,
        )
        if frame is None:
            # Transient feed dropouts self-heal (the client auto-reconnects);
            # give it a few chances before treating this as fatal.
            print(
                f"[capture] no fresh revolution yet (attempt {attempt_index + 1}/6); "
                "waiting for feed to recover"
            )
            continue
        local_points_xy = _scan_to_local_points(
            points=frame.points,
            forward_angle_deg=float(forward_angle_deg),
            valid_angle_half_width_deg=float(valid_angle_half_width_deg),
            invert_lateral_axis=bool(invert_lateral_axis),
            max_distance_m=float(max_distance_m),
            min_confidence=int(min_confidence),
            min_range_m=float(min_range_m),
        )
        last_raw = len(frame.points)
        last_usable = len(local_points_xy)
        if len(local_points_xy) >= MIN_HEALTHY_LOCAL_POINTS:
            break
        if last_raw < MIN_HEALTHY_RAW_POINTS:
            print(
                f"[capture] revolution {frame_id} DEGRADED FEED (raw={last_raw} < "
                f"{MIN_HEALTHY_RAW_POINTS} — host serving partial revolutions); waiting "
                f"(attempt {attempt_index + 1}/6)"
            )
        else:
            print(
                f"[capture] revolution {frame_id}: FULL revolution (raw={last_raw}) but only "
                f"{last_usable} points beyond {float(min_range_m):.2f}m — the robot is BOXED IN "
                f"(surrounded by returns within {float(min_range_m):.2f}m); waiting "
                f"(attempt {attempt_index + 1}/6)"
            )
        after_frame_id = int(frame_id)
        frame = None
        local_points_xy = None
        time.sleep(0.4)
    if frame is None or local_points_xy is None:
        if last_raw >= MIN_HEALTHY_RAW_POINTS:
            raise LidarBoxedInError(
                f"lidar HEALTHY (~{last_raw} pts/rev) but only ~{last_usable} beyond "
                f"{float(min_range_m):.2f}m — the robot is BOXED IN (nose against a wall / "
                "surrounded / draped material)"
            )
        raise RuntimeError(
            f"LiDAR feed is delivering PARTIAL revolutions (raw ~{last_raw}, far below the ~240 "
            "norm) and did not recover within 6 attempts. Check the lidar stream host on the Pi "
            "(USB contention / power) before rerunning."
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
    """Persist the running list of motion hints to ``motion_hints.json``.

    Serializes each hint (kind, label, expected dx/dy/dtheta, search windows) to
    a rounded JSON record and writes it into ``output_dir``. These hints are the
    per-step "expected motion" the stitcher uses to seed each solve, so saving
    them lets a run be re-stitched offline. Returns the written path.
    """
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
    """Render the current stitched map to the debug HTML overlay (and JSON).

    Builds an SVG of all the transformed snapshot clouds + capture poses, wraps
    it and a JSON report (motion hints, per-solve log, final pose) into the HTML
    template, and writes both ``latest_stitched_overlay.html`` and ``.json`` into
    ``stitch_dir``. This is the file you open to SEE the map as it grows. Returns
    the two written paths.
    """
    from ..lidar.direct_snapshot_stitch import _generate_svg

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
    """Re-stitch the ENTIRE map from scratch, from all snapshots on disk.

    Loads every saved snapshot, replays the motion hints through the engine's
    ``_stitch_snapshots`` to re-solve all poses and transformed clouds, rewrites
    the overlay, and re-saves the hints. Returns the fresh stitch state dict
    (snapshots, poses, transformed_sets, solve_log, paths). Heavier than
    ``_append_stitch`` — used when the incremental state must be rebuilt whole.
    """
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
    """Fold ONE new snapshot into the existing map (the incremental hot path).

    Takes the current stitch state and a new snapshot and appends it: normally by
    scan-matching the new cloud onto the accumulated map (seeded by the latest
    motion hint), or — when ``solved_pose_override`` is given — by trusting a pose
    the caller already solved (e.g. a turn-arc or recovery solve) instead of
    re-solving. Then it rewrites the overlay and returns the updated stitch state.
    This is what runs every capture; ``_rebuild_stitch`` is the from-scratch cousin.
    """
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


