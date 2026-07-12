"""Save snapshots of what Sourccey's cameras are seeing.

Grabs frames from the slam_input.v1 stream (front_left / front_right eyes,
plus the bottom camera when the host publishes it) and writes them to disk as
PNGs — raw frames, no detector overlays — so they can be fed through
candidate perception models (Florence-2, Depth Anything, ...) offline and
compared against what the current edge detector decided.

One-shot (default):
    uv run python scripts/sourccey_eye_snapshot.py --remote-ip 192.168.1.237

Timed series (e.g. while teleoping the robot around furniture):
    uv run python scripts/sourccey_eye_snapshot.py --remote-ip 192.168.1.237 \
        --every 2.0 --count 30

NEVER touches the robot base or arms: this is a passive camera listener.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from sourccey_elevated_safety import SlamCameraSubscriber, endpoint_from_remote_ip

CAMERA_KEYS = ("front_left", "front_right", "bottom")


def save_eye_snapshot(
    subscriber: SlamCameraSubscriber,
    output_dir: Path,
    *,
    tag: str = "",
    max_age_s: float = 1.0,
) -> list[Path]:
    """Write the freshest frame from every available camera to output_dir.

    Returns the paths written. Frames older than max_age_s are skipped so a
    stalled stream cannot masquerade as a current view.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    written: list[Path] = []
    for cam_name in CAMERA_KEYS:
        frame, age_s = subscriber.latest(cam_name)
        if frame is None:
            continue
        if age_s is not None and age_s > float(max_age_s):
            print(f"[eyes] {cam_name}: frame is {age_s:.1f}s old, skipping (stream stalled?)")
            continue
        suffix = f"_{tag}" if tag else ""
        path = output_dir / f"{stamp}{suffix}_{cam_name}.png"
        if cv2.imwrite(str(path), frame):
            written.append(path)
        else:
            print(f"[eyes] failed to write {path}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save raw snapshots from Sourccey's cameras for offline model evaluation."
    )
    parser.add_argument("--remote-ip", type=str, required=True)
    parser.add_argument(
        "--slam-input-endpoint",
        type=str,
        default="",
        help="slam_input.v1 endpoint (default tcp://<remote-ip>:5560).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/eye_snapshots",
        help="Output directory (created if missing).",
    )
    parser.add_argument(
        "--every",
        type=float,
        default=0.0,
        help="Seconds between snapshots. 0 (default) = take one snapshot and exit.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Stop after this many snapshot rounds in --every mode. 0 = until Ctrl+C.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional label baked into the filenames (e.g. 'table_headon').",
    )
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoint = str(args.slam_input_endpoint).strip() or endpoint_from_remote_ip(args.remote_ip)
    subscriber = SlamCameraSubscriber(endpoint=endpoint, camera_keys=CAMERA_KEYS)
    subscriber.start()
    print(f"[eyes] listening on {endpoint} ...")
    try:
        if not subscriber.wait_for_frames(timeout_s=6.0, required=("front_left", "front_right")):
            print(
                "[eyes] ERROR: no eye frames within 6s — is the host running and "
                "publishing slam_input on port 5560?"
            )
            return 1
        bottom_present = subscriber.latest("bottom")[0] is not None
        print(
            f"[eyes] cameras ready (front_left, front_right, "
            f"bottom={'present' if bottom_present else 'absent'})"
        )

        rounds_taken = 0
        while True:
            written = save_eye_snapshot(subscriber, output_dir, tag=str(args.tag))
            rounds_taken += 1
            if written:
                names = ", ".join(p.name for p in written)
                print(f"[eyes] snapshot {rounds_taken}: {names}")
            else:
                print(f"[eyes] snapshot {rounds_taken}: nothing fresh to save")
            if float(args.every) <= 0.0:
                break
            if int(args.count) > 0 and rounds_taken >= int(args.count):
                break
            time.sleep(float(args.every))
    except KeyboardInterrupt:
        print("\n[eyes] stopped")
    finally:
        subscriber.stop()
    print(f"[eyes] frames in {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
