#!/usr/bin/env python

"""
Verify a specific video file and time range in a LeRobot dataset.

Use this to pinpoint corrupt frames, e.g. front_right file-038 between 5m10s and 5m35s.
"""

import argparse
import logging
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import load_episodes
from lerobot.datasets.video_utils import (
    decode_video_frames,
    get_safe_default_codec,
)
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging


def _scalar(val) -> float | int:
    if hasattr(val, "item"):
        return val.item()
    return val


def _decode_with_both_backends(
    path: str | Path,
    timestamps: list[float],
    tolerance_s: float,
    preferred_backend: str | None,
) -> tuple[bool, str | None]:
    """
    Try preferred backend (or torchcodec), then fallback (pyav). Return (success, error_message).
    Mirrors training: both backends must succeed or we report the failure(s).
    """
    primary = preferred_backend or get_safe_default_codec()
    fallback = "pyav" if primary == "torchcodec" else "torchcodec"
    errors: list[str] = []
    for backend in (primary, fallback):
        try:
            decode_video_frames(str(path), timestamps, tolerance_s, backend=backend)
            # This backend succeeded; if we had a previous failure, the frame is still bad for training
            # which uses primary then fallback. So we only report success if primary succeeded.
            if backend == primary:
                return True, None
            # Fallback succeeded but primary failed - report primary failure so user knows training would fail
            return False, f"{primary}: {errors[0]}" if errors else "unknown"
        except Exception as e:
            errors.append(f"{backend}: {type(e).__name__}: {e!s}")
    return False, " ; ".join(errors)


def _find_video_file_path(
    meta: LeRobotDatasetMetadata,
    video_key: str,
    file_index: int,
) -> Path | None:
    """Return absolute path to the video file (video_key, file_index). chunk_index from episodes."""
    root = Path(meta.root)
    episodes = meta.episodes
    if episodes is None:
        episodes = load_episodes(root)
    for ep_idx in range(len(episodes)):
        ep = episodes[ep_idx]
        try:
            fi = int(_scalar(ep[f"videos/{video_key}/file_index"]))
            if fi != file_index:
                continue
            chunk_idx = int(_scalar(ep[f"videos/{video_key}/chunk_index"]))
            rel = meta.video_path.format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_index,
            )
            path = root / rel
            if path.exists():
                return path
        except (KeyError, TypeError, ValueError):
            continue
    return None


def verify_specific(
    repo_id: str,
    dataset_folder: str,
    video_key: str,
    file_index: int,
    start_s: float,
    end_s: float,
    step_s: float = 0.5,
    root: Path | None = None,
    tolerance_s: float = 1e-4,
    backend: str | None = None,
    video_path: Path | None = None,
    fps_override: float | None = None,
) -> list[dict]:
    """
    Decode frames in [start_s, end_s] for the given video file. Return list of errors.
    Tries torchcodec then pyav (like training) so corrupt frames are guaranteed to be found.
    """
    root_base = root or HF_LEROBOT_HOME
    dataset_root = root_base / repo_id / dataset_folder
    dataset_name = f"{repo_id}/{dataset_folder}"

    if video_path is not None:
        path = Path(video_path)
        if not path.exists():
            return [{"error": "file_not_found", "video_key": video_key, "file_index": file_index}]
        fps = fps_override if fps_override is not None else 30.0
    else:
        meta = LeRobotDatasetMetadata(repo_id=dataset_name, root=dataset_root)
        fps = meta.fps
        path = _find_video_file_path(meta, video_key, file_index)
        if path is None:
            return [{"error": "file_not_found", "video_key": video_key, "file_index": file_index}]

    use_both_backends = backend is None
    errors: list[dict] = []
    t = start_s
    while t <= end_s:
        if abs(t - round(t)) < 1e-9:
            logging.info("Checking %.1f s ...", t)
        frame_index = int(round(t * fps))
        if use_both_backends:
            ok, msg = _decode_with_both_backends(path, [t], tolerance_s, preferred_backend=backend)
            if not ok and msg:
                errors.append({
                    "timestamp_s": t,
                    "frame_index": frame_index,
                    "message": msg,
                })
                break
        else:
            try:
                decode_video_frames(str(path), [t], tolerance_s, backend=backend)
            except Exception as e:
                errors.append({
                    "timestamp_s": t,
                    "frame_index": frame_index,
                    "message": f"{type(e).__name__}: {e!s}",
                })
                break
        t += step_s

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a specific video file and time range (e.g. find corrupt frames).",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="sourccey-012",
        help="Repo id (default: sourccey-012).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sourccey-012__shirt-fold__cmb__0008__chrism",
        help="Dataset folder name (default: sourccey-012__shirt-fold__cmb__0008__chrism).",
    )
    parser.add_argument(
        "--video_key",
        type=str,
        default="observation.images.front_right",
        help="Video key (default: observation.images.front_right).",
    )
    parser.add_argument(
        "--file_index",
        type=int,
        default=38,
        help="File index, e.g. 38 for file-038.mp4 (default: 38).",
    )
    parser.add_argument(
        "--start_s",
        type=float,
        default=310.0,
        help="Start time in seconds (default: 310 = 5m10s).",
    )
    parser.add_argument(
        "--end_s",
        type=float,
        default=340.0,
        help="End time in seconds (default: 340 = 5m40s).",
    )
    parser.add_argument(
        "--step_s",
        type=float,
        default=0.5,
        help="Step between checked timestamps in seconds (default: 0.5). Use 1/30 for every frame at 30fps.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Direct path to the .mp4 file (overrides repo_id/dataset path resolution). Use when path layout differs (e.g. combine/0008/...).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS when using --video_path (default: 30). Ignored when resolving path from dataset.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Override cache root (default: HF_LEROBOT_HOME).",
    )
    parser.add_argument(
        "--tolerance_s",
        type=float,
        default=1e-4,
        help="Timestamp tolerance in seconds (default: 1e-4).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["torchcodec", "pyav", "video_reader"],
        help="Video decoder backend (default: auto).",
    )

    args = parser.parse_args()
    init_logging()

    root_path = Path(args.root) if args.root else None
    video_path = Path(args.video_path) if args.video_path else None
    errors = verify_specific(
        repo_id=args.repo_id,
        dataset_folder=args.dataset,
        video_key=args.video_key,
        file_index=args.file_index,
        start_s=args.start_s,
        end_s=args.end_s,
        step_s=args.step_s,
        root=root_path,
        tolerance_s=args.tolerance_s,
        backend=args.backend,
        video_path=video_path,
        fps_override=args.fps,
    )

    if errors and errors[0].get("error") == "file_not_found":
        logging.error("Video file not found: %s file-%03d", args.video_key, args.file_index)
        return

    for err in errors:
        logging.error(
            "ERROR %s file-%03d timestamp_s=%.3f frame_index=%s message=%s",
            args.video_key,
            args.file_index,
            err["timestamp_s"],
            err["frame_index"],
            err["message"],
        )

    if errors:
        logging.warning(
            "Verification found %d error(s) in the range %.1f–%.1f s.",
            len(errors),
            args.start_s,
            args.end_s,
        )
    else:
        logging.info("No errors in the range %.1f–%.1f s.", args.start_s, args.end_s)


if __name__ == "__main__":
    main()
