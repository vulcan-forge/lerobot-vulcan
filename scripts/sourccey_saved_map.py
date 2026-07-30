"""Versioned, pickle-free persistence for Sourccey's 2D SLAM maps."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_NAME = "sourccey-slam-map"
FORMAT_VERSION = 1
DEFAULT_SAVED_MAP_PATH = Path(__file__).with_name("sourccey_saved_map.npz")


@dataclass(slots=True)
class SavedMap:
    metadata: dict
    scan_points: np.ndarray
    scan_offsets: np.ndarray
    scan_poses: np.ndarray
    scan_gold: np.ndarray
    occupancy_log_odds: np.ndarray
    occupancy_origin_xy: np.ndarray
    current_pose: np.ndarray
    trail_xy: np.ndarray

    def local_scan(self, index: int) -> np.ndarray:
        start = int(self.scan_offsets[index])
        end = int(self.scan_offsets[index + 1])
        return self.scan_points[start:end]


def _validate(saved: SavedMap) -> None:
    if saved.metadata.get("format") != FORMAT_NAME:
        raise ValueError("not a Sourccey SLAM map")
    if int(saved.metadata.get("version", -1)) != FORMAT_VERSION:
        raise ValueError(
            f"unsupported Sourccey map version {saved.metadata.get('version')!r}"
        )
    count = len(saved.scan_poses)
    if saved.scan_poses.shape != (count, 3):
        raise ValueError("scan_poses must have shape (N, 3)")
    if saved.scan_gold.shape != (count,):
        raise ValueError("scan_gold must have shape (N,)")
    if saved.scan_offsets.shape != (count + 1,):
        raise ValueError("scan_offsets must have N+1 entries")
    if int(saved.scan_offsets[0]) != 0 or int(saved.scan_offsets[-1]) != len(
        saved.scan_points
    ):
        raise ValueError("scan offsets do not cover the saved point array")
    if np.any(np.diff(saved.scan_offsets) < 0):
        raise ValueError("scan offsets are not monotonic")
    if saved.scan_points.ndim != 2 or saved.scan_points.shape[1:] != (2,):
        raise ValueError("scan_points must have shape (M, 2)")
    if saved.occupancy_log_odds.ndim != 2:
        raise ValueError("occupancy_log_odds must be a 2D grid")
    if saved.occupancy_origin_xy.shape != (2,):
        raise ValueError("occupancy_origin_xy must have shape (2,)")
    if saved.current_pose.shape != (3,):
        raise ValueError("current_pose must have shape (3,)")
    if count <= 0 or len(saved.scan_points) <= 0:
        raise ValueError("saved map contains no LiDAR scans")
    known_cells = np.count_nonzero(np.abs(saved.occupancy_log_odds) >= 0.5)
    if int(known_cells) <= 0:
        raise ValueError("saved map contains no known occupancy cells")


def save_world_map(
    path: str | Path,
    world_map,
    current_pose,
    *,
    trail: list[np.ndarray] | None = None,
    sensor_config: dict | None = None,
    navigation_config: dict | None = None,
) -> Path:
    """Atomically save the exact localization and occupancy state."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    scans = list(world_map.scans)
    lengths = np.asarray([len(scan.local_xy) for scan in scans], dtype=np.int64)
    offsets = np.concatenate([
        np.zeros(1, dtype=np.int64),
        np.cumsum(lengths, dtype=np.int64),
    ])
    points = (
        np.concatenate(
            [np.asarray(scan.local_xy, dtype=np.float32) for scan in scans], axis=0
        )
        if int(offsets[-1])
        else np.empty((0, 2), dtype=np.float32)
    )
    poses = np.asarray(
        [[scan.pose.x, scan.pose.y, scan.pose.theta_deg] for scan in scans],
        dtype=np.float64,
    ).reshape((-1, 3))
    gold = np.asarray([bool(scan.gold) for scan in scans], dtype=np.bool_)
    metadata = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "grid_resolution_m": float(world_map.grid.res),
        "footprint_clear_m": float(world_map.footprint_clear_m),
        "lidar_offset_m": float(world_map.lidar_offset_m),
        "forward_offset_deg": float(world_map.forward_offset_deg),
        "scan_count": len(scans),
        "gold_scan_count": int(np.count_nonzero(gold)),
        "sensor_config": dict(sensor_config or {}),
        "navigation_config": dict(navigation_config or {}),
    }
    current = np.asarray(
        [current_pose.x, current_pose.y, current_pose.theta_deg], dtype=np.float64
    )
    trail_xy = np.asarray(trail or [], dtype=np.float64).reshape((-1, 2))

    # Refuse to atomically replace a previously useful map with an incomplete
    # shutdown artifact. A saved navigation map must contain both scan history
    # and occupancy evidence; pose-only files cannot be displayed or routed.
    candidate = SavedMap(
        metadata=metadata,
        scan_points=points,
        scan_offsets=offsets,
        scan_poses=poses,
        scan_gold=gold,
        occupancy_log_odds=np.asarray(world_map.grid.L, dtype=np.float32),
        occupancy_origin_xy=np.asarray(world_map.grid.origin, dtype=np.float64),
        current_pose=current,
        trail_xy=trail_xy,
    )
    _validate(candidate)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-", suffix=".npz", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                scan_points=points,
                scan_offsets=offsets,
                scan_poses=poses,
                scan_gold=gold,
                occupancy_log_odds=candidate.occupancy_log_odds,
                occupancy_origin_xy=candidate.occupancy_origin_xy,
                current_pose=current,
                trail_xy=trail_xy,
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_saved_map(path: str | Path) -> SavedMap:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        saved = SavedMap(
            metadata=json.loads(str(data["metadata_json"].item())),
            scan_points=np.asarray(data["scan_points"], dtype=np.float32),
            scan_offsets=np.asarray(data["scan_offsets"], dtype=np.int64),
            scan_poses=np.asarray(data["scan_poses"], dtype=np.float64),
            scan_gold=np.asarray(data["scan_gold"], dtype=np.bool_),
            occupancy_log_odds=np.asarray(data["occupancy_log_odds"], dtype=np.float32),
            occupancy_origin_xy=np.asarray(data["occupancy_origin_xy"], dtype=np.float64),
            current_pose=np.asarray(data["current_pose"], dtype=np.float64),
            trail_xy=np.asarray(data["trail_xy"], dtype=np.float64).reshape((-1, 2)),
        )
    _validate(saved)
    return saved
