"""Stationary LiDAR calibration for Sourccey's collision envelope."""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
from ldlidar_direct_snapshot_client import DirectLidarFeed
from sourccey_collision_box import (
    DEFAULT_COLLISION_BOX_PATH,
    calibrate_collision_box,
    save_collision_box,
)
from sourccey_spin_map import _scan_local

_PHYS_FORWARD_LIDAR_DEG = 270.0


def _physical_forward_offset_deg(args) -> float:
    delta = ((_PHYS_FORWARD_LIDAR_DEG - float(args.forward_angle_deg) + 180.0) % 360.0) - 180.0
    lateral = math.sin(math.radians(delta))
    if bool(args.invert_lateral_axis):
        lateral = -lateral
    return math.degrees(math.atan2(lateral, math.cos(math.radians(delta))))


def _to_forward_frame(points: np.ndarray, offset_deg: float) -> np.ndarray:
    c = math.cos(math.radians(-offset_deg))
    s = math.sin(math.radians(-offset_deg))
    return np.column_stack([
        points[:, 0] * c - points[:, 1] * s,
        points[:, 0] * s + points[:, 1] * c,
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-ip", required=True)
    parser.add_argument("--lidar-host", default=None)
    parser.add_argument("--lidar-port", type=int, default=8765)
    parser.add_argument("--scans", type=int, default=40)
    parser.add_argument("--bin-size-deg", type=float, default=4.0)
    parser.add_argument("--max-boundary-m", type=float, default=1.5)
    parser.add_argument("--output", default=str(DEFAULT_COLLISION_BOX_PATH))
    parser.add_argument("--forward-angle-deg", type=float, default=90.0)
    parser.add_argument("--valid-angle-half-width-deg", type=float, default=180.0)
    parser.add_argument("--invert-lateral-axis", action="store_true", default=True)
    parser.add_argument("--max-distance-m", type=float, default=8.0)
    parser.add_argument("--min-range-m", type=float, default=0.03)
    parser.add_argument("--min-confidence", type=int, default=0)
    args = parser.parse_args()

    host = args.lidar_host or args.remote_ip
    feed = DirectLidarFeed(host, int(args.lidar_port))
    feed.start()
    print(f"[calibrate] LiDAR connecting to {host}:{args.lidar_port} ...")
    scans: list[np.ndarray] = []
    previous_id = -1
    deadline = time.monotonic() + 30.0
    try:
        while len(scans) < max(8, int(args.scans)):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"received only {len(scans)} usable scans in 30 seconds")
            frame_id, frame = feed.wait_for_frame_after(
                after_frame_id=previous_id,
                timeout_s=1.0,
                min_frame_advances=1,
            )
            if frame is None:
                continue
            previous_id = int(frame_id)
            local = _scan_local(frame, args)
            if len(local) >= 12:
                scans.append(_to_forward_frame(local, _physical_forward_offset_deg(args)))

        profile = calibrate_collision_box(
            scans,
            bin_size_deg=float(args.bin_size_deg),
            min_scans_per_bin=max(4, len(scans) // 3),
            max_boundary_m=float(args.max_boundary_m),
        )
        output = save_collision_box(profile, args.output)
        learned = profile["ranges_m"]
        print(f"[calibrate] learned {sum(v is not None for v in learned)}/{len(learned)} bins.")
        print(f"[calibrate] saved {output}")
        print("[calibrate] Remove the physical box before starting the explorer.")
        return 0
    finally:
        feed.stop()


if __name__ == "__main__":
    raise SystemExit(main())
