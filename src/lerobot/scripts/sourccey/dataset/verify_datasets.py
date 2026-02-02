#!/usr/bin/env python

"""
Verify video frames in LeRobot datasets.

Assumes datasets live under the HuggingFace LeRobot cache (HF_LEROBOT_HOME).
Each dataset is given as repo_id/name (e.g. sourccey-012/my_dataset).

Decodes sample frames from each unique video file and reports any broken/corrupt
frames so they can be fixed or excluded before training fails.
"""

import argparse
import json
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import load_episodes
from lerobot.datasets.video_utils import (
    decode_video_frames,
    get_safe_default_codec,
)
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging


@dataclass
class FrameError:
    """A single frame verification error."""

    dataset_name: str
    file_name: str
    video_key: str
    timestamp_s: float
    frame_index: int
    message: str


# Directory names considered temp/tmp (e.g. left over from interrupted recording)
TEMP_FOLDER_PREFIXES = ("tmp", "temp")


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
            if backend == primary:
                return True, None
            return False, (errors[0] if errors else "unknown")
        except Exception as e:
            errors.append(f"{backend}: {type(e).__name__}: {e!s}")
    return False, " ; ".join(errors)


def _is_tolerance_error_message(msg: str) -> bool:
    """True if the error is the timestamp-tolerance AssertionError (skip; not corruption)."""
    if not msg:
        return False
    m = msg.lower()
    return "tolerance" in m and ("violate" in m or "query timestamps" in m)


def _find_temp_folders(dataset_root: Path) -> list[Path]:
    """Return immediate children of dataset_root that look like temp/tmp folders."""
    if not dataset_root.exists() or not dataset_root.is_dir():
        return []
    found: list[Path] = []
    for child in dataset_root.iterdir():
        if child.is_dir() and child.name.lower().startswith(TEMP_FOLDER_PREFIXES):
            found.append(child)
    return sorted(found)


def _verify_temp_folder_videos(
    dataset_name: str,
    temp_dir: Path,
    tolerance_s: float,
    fps: int,
    backend: str | None,
) -> list[FrameError]:
    """Try to decode first frame of each .mp4 in a temp folder; return errors for failures."""
    errors: list[FrameError] = []
    for mp4_path in temp_dir.rglob("*.mp4"):
        if backend is None:
            ok, msg = _decode_with_both_backends(mp4_path, [0.0], tolerance_s, preferred_backend=None)
            if not ok and msg:
                if _is_tolerance_error_message(msg):
                    continue
                errors.append(
                    FrameError(
                        dataset_name=dataset_name,
                        file_name=mp4_path.name,
                        video_key=f"(temp folder: {temp_dir.name})",
                        timestamp_s=0.0,
                        frame_index=0,
                        message=msg,
                    )
                )
        else:
            try:
                decode_video_frames(str(mp4_path), [0.0], tolerance_s, backend=backend)
            except Exception as e:
                msg = f"{type(e).__name__}: {e!s}"
                if _is_tolerance_error_message(msg):
                    continue
                errors.append(
                    FrameError(
                        dataset_name=dataset_name,
                        file_name=mp4_path.name,
                        video_key=f"(temp folder: {temp_dir.name})",
                        timestamp_s=0.0,
                        frame_index=0,
                        message=msg,
                    )
                )
    return errors


def _parse_datasets(datasets_arg: str) -> list[str]:
    """
    Parse --datasets argument: list of "repo_id/name" (e.g. sourccey-012/my_dataset).
    Supports JSON array format or space-separated.
    """
    if datasets_arg.strip().startswith("[") and datasets_arg.strip().endswith("]"):
        try:
            cleaned_arg = datasets_arg.strip().replace(",]", "]").replace(", ]", "]")
            parsed = json.loads(cleaned_arg)
            return [p for p in parsed if p and isinstance(p, str) and p.strip()]
        except json.JSONDecodeError:
            pass

    parsed = shlex.split(datasets_arg)
    return [p for p in parsed if p and p.strip() and p not in ("[", "]")]


def _scalar(val) -> float | int:
    """Normalize parquet value to Python scalar."""
    if hasattr(val, "item"):
        return val.item()
    return val


def _get_unique_video_files_with_timestamps(
    meta: LeRobotDatasetMetadata,
    num_sample_frames: int = 5,
    step_s: float | None = None,
) -> list[tuple[str, Path, list[float]]]:
    """
    For a dataset, return list of (video_key, absolute_path, list of timestamps to check).

    When step_s is set, timestamps are file_from_ts, file_from_ts + step_s, ... until file_to_ts.
    Otherwise timestamps are frame-aligned (file_from_ts + frame_index/fps) using num_sample_frames.
    """
    if not meta.video_keys or meta.video_path is None:
        return []

    root = Path(meta.root)
    episodes = meta.episodes
    if episodes is None:
        episodes = load_episodes(root)
    n_episodes = len(episodes)

    # Unique (video_key, chunk_idx, file_idx) -> (from_ts_min, to_ts_max) over episodes using this file
    file_bounds: dict[tuple[str, int, int], tuple[float, float]] = {}

    for video_key in meta.video_keys:
        for ep_idx in range(n_episodes):
            try:
                ep = episodes[ep_idx]
                chunk_idx = int(_scalar(ep[f"videos/{video_key}/chunk_index"]))
                file_idx = int(_scalar(ep[f"videos/{video_key}/file_index"]))
                from_ts = float(_scalar(ep[f"videos/{video_key}/from_timestamp"]))
                to_ts = float(_scalar(ep[f"videos/{video_key}/to_timestamp"]))
            except (KeyError, TypeError, ValueError):
                continue

            key = (video_key, chunk_idx, file_idx)
            if key not in file_bounds:
                file_bounds[key] = (from_ts, to_ts)
            else:
                lo, hi = file_bounds[key]
                file_bounds[key] = (min(lo, from_ts), max(hi, to_ts))

    result: list[tuple[str, Path, list[float]]] = []
    for (video_key, chunk_idx, file_idx), (file_from_ts, file_to_ts) in file_bounds.items():
        rel_path = meta.video_path.format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        abs_path = root / rel_path
        if not abs_path.exists():
            continue
        duration = file_to_ts - file_from_ts
        fps = meta.fps
        if step_s is not None and step_s > 0:
            # Sample at file_from_ts, file_from_ts + step_s, ... until <= file_to_ts
            timestamps = []
            t = file_from_ts
            while t <= file_to_ts:
                timestamps.append(t)
                t += step_s
            if not timestamps:
                timestamps = [file_from_ts]
        else:
            # Number of frames in this file (floor so we stay in range)
            num_frames = int(duration * fps)
            if num_frames <= 0:
                timestamps = [file_from_ts]
            else:
                # Sample frame indices 0, ..., num_frames-1 (never num_frames → avoids IndexError)
                step = max(1, (num_frames - 1) // max(1, num_sample_frames - 1))
                frame_indices = [min(i * step, num_frames - 1) for i in range(num_sample_frames)]
                frame_indices = sorted(set(frame_indices))  # unique, sorted
                # Frame-aligned timestamps (same grid training uses)
                timestamps = [file_from_ts + idx / fps for idx in frame_indices]
        result.append((video_key, abs_path, timestamps))

    return result


def _verify_one_file(
    dataset_name: str,
    video_key: str,
    file_path: Path,
    timestamps: list[float],
    tolerance_s: float,
    fps: int,
    backend: str | None,
) -> list[FrameError]:
    """Decode sample frames at the given timestamps. Return list of errors (one per failed frame)."""
    errors: list[FrameError] = []
    file_name = file_path.name
    rel_path_str = str(file_path)

    for i, ts in enumerate(timestamps):
        frame_index = int(round(ts * fps))
        if backend is None:
            ok, msg = _decode_with_both_backends(
                rel_path_str, [ts], tolerance_s, preferred_backend=None
            )
            if not ok and msg:
                if _is_tolerance_error_message(msg):
                    continue
                errors.append(
                    FrameError(
                        dataset_name=dataset_name,
                        file_name=file_name,
                        video_key=video_key,
                        timestamp_s=ts,
                        frame_index=frame_index,
                        message=msg,
                    )
                )
                break
            continue
        try:
            frames = decode_video_frames(
                rel_path_str,
                [ts],
                tolerance_s,
                backend=backend,
            )
            if frames is None or frames.numel() == 0:
                errors.append(
                    FrameError(
                        dataset_name=dataset_name,
                        file_name=file_name,
                        video_key=video_key,
                        timestamp_s=ts,
                        frame_index=frame_index,
                        message="Decode returned empty or null frames",
                    )
                )
                break
            if not frames.shape or len(frames.shape) < 3:
                errors.append(
                    FrameError(
                        dataset_name=dataset_name,
                        file_name=file_name,
                        video_key=video_key,
                        timestamp_s=ts,
                        frame_index=frame_index,
                        message=f"Invalid frame shape: {frames.shape}",
                    )
                )
                break
        except AssertionError as e:
            msg = f"AssertionError: {e!s}"
            if _is_tolerance_error_message(msg):
                continue
            errors.append(
                FrameError(
                    dataset_name=dataset_name,
                    file_name=file_name,
                    video_key=video_key,
                    timestamp_s=ts,
                    frame_index=frame_index,
                    message=msg,
                )
            )
            break
        except Exception as e:
            msg = f"{type(e).__name__}: {e!s}"
            if _is_tolerance_error_message(msg):
                continue
            errors.append(
                FrameError(
                    dataset_name=dataset_name,
                    file_name=file_name,
                    video_key=video_key,
                    timestamp_s=ts,
                    frame_index=frame_index,
                    message=msg,
                )
            )
            break

    return errors


def verify_datasets(
    datasets: list[str],
    root: Path | None = None,
    tolerance_s: float = 1e-4,
    video_backend: str | None = None,
    num_sample_frames: int = 5,
    step_s: float | None = None,
) -> list[FrameError]:
    """
    Verify video frames for LeRobot datasets under the HuggingFace LeRobot cache.

    Each item in datasets is "repo_id/name" (e.g. sourccey-012/my_dataset).
    Resolved as: root / repo_id / name (e.g. HF_LEROBOT_HOME / sourccey-012 / my_dataset).

    Args:
        datasets: List of "repo_id/name" (e.g. ["sourccey-012/ds1", "sourccey-012/ds2"]).
        root: Cache root (default: HF_LEROBOT_HOME).
        tolerance_s: Timestamp tolerance in seconds (same as training).
        video_backend: Decoder backend ('torchcodec' or 'pyav'). Default: auto.
        num_sample_frames: Number of sample timestamps per file when step_s is not set.
        step_s: If set, sample at this step in seconds (file_from_ts, +step_s, ...). Overrides num_sample_frames.

    Returns:
        List of FrameError for every broken frame found.
    """
    all_errors: list[FrameError] = []
    root_base = root or HF_LEROBOT_HOME

    for spec in datasets:
        spec = spec.strip()
        if "/" not in spec:
            logging.warning("Skipping invalid dataset spec (expected repo_id/name): %s", spec)
            continue
        repo_id, dataset_folder = spec.split("/", 1)
        repo_id = repo_id.strip()
        dataset_folder = dataset_folder.strip()
        dataset_root = root_base / repo_id / dataset_folder
        dataset_name = f"{repo_id}/{dataset_folder}"

        logging.info("Verifying dataset: %s (root=%s)", dataset_name, dataset_root)

        try:
            meta = LeRobotDatasetMetadata(repo_id=dataset_name, root=dataset_root)
        except Exception as e:
            logging.error("Failed to load metadata for %s: %s", dataset_name, e)
            all_errors.append(
                FrameError(
                    dataset_name=dataset_name,
                    file_name="(meta)",
                    video_key="",
                    timestamp_s=0.0,
                    frame_index=0,
                    message=f"Failed to load metadata: {e!s}",
                )
            )
            continue

        if not meta.video_keys:
            logging.info("Dataset %s has no video keys, skipping.", dataset_name)
            continue

        fps = meta.fps
        files_with_ts = _get_unique_video_files_with_timestamps(
            meta, num_sample_frames=num_sample_frames, step_s=step_s
        )

        # Check for temp/tmp folders (should not be present; often contain broken videos)
        for temp_dir in _find_temp_folders(dataset_root):
            all_errors.append(
                FrameError(
                    dataset_name=dataset_name,
                    file_name=temp_dir.name,
                    video_key="(temp folder)",
                    timestamp_s=0.0,
                    frame_index=0,
                    message=f"Dataset contains temp/tmp folder: {temp_dir.name}",
                )
            )
            temp_errs = _verify_temp_folder_videos(
                dataset_name, temp_dir, tolerance_s, fps, video_backend
            )
            all_errors.extend(temp_errs)
            if temp_errs:
                return all_errors

        for video_key, file_path, timestamps in files_with_ts:
            logging.info("\nAnalyzing video: %s (%s)", file_path, video_key)
            errs = _verify_one_file(
                dataset_name=dataset_name,
                video_key=video_key,
                file_path=file_path,
                timestamps=timestamps,
                tolerance_s=tolerance_s,
                fps=fps,
                backend=video_backend,
            )
            all_errors.extend(errs)
            if errs:
                return all_errors

    return all_errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify video frames in LeRobot datasets under the HuggingFace cache. "
        "Reports dataset name (repo_id/dataset), file name, and frame for every error.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Datasets as repo_id/name. JSON array (e.g. '[\"sourccey-012/name\"]' or '[\"sourccey-012/ds1\", \"sourccey-012/ds2\"]') or space-separated.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Override cache root (default: HF_LEROBOT_HOME, e.g. ~/.cache/huggingface/lerobot).",
    )
    parser.add_argument(
        "--tolerance_s",
        type=float,
        default=1e-4,
        help="Timestamp tolerance in seconds (default: 1e-4).",
    )
    parser.add_argument(
        "--video_backend",
        type=str,
        default=None,
        choices=["torchcodec", "pyav", "video_reader"],
        help="Video decoder backend (default: auto).",
    )
    parser.add_argument(
        "--num_sample_frames",
        type=int,
        default=5,
        help="Number of sample frames to check per video file when --step_s is not set (default: 5).",
    )
    parser.add_argument(
        "--step_s",
        type=float,
        default=None,
        help="Step in seconds between checked timestamps per file (e.g. 0.5, 1.0). When set, overrides --num_sample_frames.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write errors as JSON (one object per line).",
    )

    args = parser.parse_args()
    init_logging()

    datasets_list = _parse_datasets(args.datasets)
    if not datasets_list:
        logging.error("No datasets provided.")
        return

    root = Path(args.root) if args.root else None
    errors = verify_datasets(
        datasets=datasets_list,
        root=root,
        tolerance_s=args.tolerance_s,
        video_backend=args.video_backend or None,
        num_sample_frames=args.num_sample_frames,
        step_s=args.step_s,
    )

    for err in errors:
        logging.error(
            "ERROR dataset=%s file=%s video_key=%s frame_index=%s timestamp_s=%.4f message=%s",
            err.dataset_name,
            err.file_name,
            err.video_key,
            err.frame_index,
            err.timestamp_s,
            err.message,
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for err in errors:
                line = json.dumps(
                    {
                        "dataset_name": err.dataset_name,
                        "file_name": err.file_name,
                        "video_key": err.video_key,
                        "frame_index": err.frame_index,
                        "timestamp_s": err.timestamp_s,
                        "message": err.message,
                    }
                ) + "\n"
                f.write(line)
        logging.info("Wrote %d errors to %s", len(errors), out_path)

    if errors:
        logging.warning("Verification found %d frame error(s).", len(errors))
    else:
        logging.info("Verification completed with no errors.")


if __name__ == "__main__":
    main()
