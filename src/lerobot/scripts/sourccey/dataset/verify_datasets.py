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
import re
import shlex
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import multiprocessing

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import load_episodes
from lerobot.datasets.video_utils import decode_video_frames
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


def _worker_label() -> str:
    """Return a short label for the current process (e.g. 'worker 1') for logging."""
    name = multiprocessing.current_process().name
    m = re.search(r"(\d+)$", name)
    if m:
        return f"worker {m.group(1)}"
    return "worker 0" if name == "MainProcess" else name


def _is_tolerance_error_message(msg: str) -> bool:
    """True if the error is the timestamp-tolerance AssertionError (skip; not corruption)."""
    if not msg:
        return False
    m = msg.lower()
    return "tolerance" in m and ("violate" in m or "query timestamps" in m)


def _is_end_of_stream_error(msg: str) -> bool:
    """True if the error is from requesting a frame past the end of the video (metadata boundary)."""
    if not msg:
        return False
    m = msg.lower()
    return (
        "no more frames" in m
        or ("invalid frame index" in m and "must be less than" in m)
    )


def _is_invalid_packet_error(msg: str) -> bool:
    """True if the error is decoder 'invalid data' packet (skip; not necessarily corruption)."""
    if not msg:
        return False
    m = msg.lower()
    return (
        "invalid data found when processing input" in m
        or "could not push packet to decoder" in m
    )


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
    backend: str,
) -> list[FrameError]:
    """Try to decode first frame of each .mp4 in a temp folder; return errors for failures."""
    errors: list[FrameError] = []
    for mp4_path in temp_dir.rglob("*.mp4"):
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
            # Sample at file_from_ts, file_from_ts + step_s, ... but never at/past last frame (0-based).
            # Use an exclusive upper bound so we never request a frame that doesn't exist when the
            # actual file has fewer frames than metadata (avoids "no more frames" / "Invalid frame index").
            # Cap at 2 seconds before file_to_ts so we never sample near the end.
            num_frames_from_meta = int(duration * fps)
            num_frames = max(1, num_frames_from_meta - 2)
            last_valid_from_frames = file_from_ts + (num_frames - 1) / fps
            last_valid_from_end = file_to_ts - 2.0  # 2 seconds before metadata end
            # Exclusive upper bound: we add t only when t < last_valid_ts (never include boundary).
            last_valid_ts = max(
                file_from_ts,
                min(last_valid_from_frames, last_valid_from_end),
            )
            timestamps = []
            t = file_from_ts
            while t <= file_to_ts:
                if t < last_valid_ts:
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
    backend: str,
    decode_batch_size: int = 1,
) -> list[FrameError]:
    """Decode sample frames at the given timestamps. Return list of errors (one per failed frame)."""
    errors: list[FrameError] = []
    file_name = file_path.name
    rel_path_str = str(file_path)
    batch_size = max(1, decode_batch_size)

    for chunk_start in range(0, len(timestamps), batch_size):
        chunk = timestamps[chunk_start : chunk_start + batch_size]
        first_ts = chunk[0]
        frame_index = int(round(first_ts * fps))

        try:
            frames = decode_video_frames(
                rel_path_str,
                chunk,
                tolerance_s,
                backend=backend,
            )
            if frames is None or frames.numel() == 0:
                errors.append(
                    FrameError(
                        dataset_name=dataset_name,
                        file_name=file_name,
                        video_key=video_key,
                        timestamp_s=first_ts,
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
                        timestamp_s=first_ts,
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
                    timestamp_s=first_ts,
                    frame_index=frame_index,
                    message=msg,
                )
            )
            break
        except Exception as e:
            msg = f"{type(e).__name__}: {e!s}"
            if _is_tolerance_error_message(msg):
                continue
            if _is_end_of_stream_error(msg):
                # Metadata said we had more frames than the file; skip this chunk, not a data error.
                logging.debug(
                    "Skipping chunk at ts=%.2f (end-of-stream): %s",
                    first_ts,
                    msg[:80],
                )
                continue
            if _is_invalid_packet_error(msg):
                # Decoder rejected a packet; often a decoder quirk, not real corruption.
                logging.debug(
                    "Skipping chunk at ts=%.2f (invalid packet): %s",
                    first_ts,
                    msg[:80],
                )
                continue
            errors.append(
                FrameError(
                    dataset_name=dataset_name,
                    file_name=file_name,
                    video_key=video_key,
                    timestamp_s=first_ts,
                    frame_index=frame_index,
                    message=msg,
                )
            )
            break

    return errors


def _init_worker_logging() -> None:
    """Initialize logging in each pool worker so INFO logs from workers are visible."""
    init_logging()


def _verify_one_file_worker(
    args: tuple,
) -> list[FrameError]:
    """Picklable worker for ProcessPoolExecutor."""
    (
        dataset_name,
        video_key,
        file_path,
        timestamps,
        tolerance_s,
        fps,
        backend,
        decode_batch_size,
    ) = args
    file_name = Path(file_path).name if isinstance(file_path, str) else Path(file_path).name
    logging.info("[%s] Analyzing video: %s (%s)", _worker_label(), file_name, video_key)
    return _verify_one_file(
        dataset_name=dataset_name,
        video_key=video_key,
        file_path=Path(file_path) if isinstance(file_path, str) else file_path,
        timestamps=timestamps,
        tolerance_s=tolerance_s,
        fps=fps,
        backend=backend,
        decode_batch_size=decode_batch_size,
    )


def verify_datasets(
    datasets: list[str],
    root: Path | None = None,
    tolerance_s: float = 1e-4,
    video_backend: str | None = None,
    num_sample_frames: int = 5,
    step_s: float | None = None,
    decode_batch_size: int = 1,
    workers: int = 1,
    stop_on_first_error: bool = True,
) -> list[FrameError]:
    """
    Verify video frames for LeRobot datasets under the HuggingFace LeRobot cache.

    Each item in datasets is "repo_id/name" (e.g. sourccey-012/my_dataset).
    Resolved as: root / repo_id / name (e.g. HF_LEROBOT_HOME / sourccey-012 / my_dataset).

    Args:
        datasets: List of "repo_id/name" (e.g. ["sourccey-012/ds1", "sourccey-012/ds2"]).
        root: Cache root (default: HF_LEROBOT_HOME).
        tolerance_s: Timestamp tolerance in seconds (same as training).
        video_backend: Decoder backend (default: torchcodec). No pyav fallback.
        num_sample_frames: Number of sample timestamps per file when step_s is not set.
        step_s: If set, sample at this step in seconds (file_from_ts, +step_s, ...). Overrides num_sample_frames.
        decode_batch_size: Timestamps per decode call (larger = faster, approximate error location).
        workers: Number of files to verify in parallel.
        stop_on_first_error: If True, return as soon as any error is found. If False, collect all errors.

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

        backend = video_backend if video_backend is not None else "torchcodec"
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
                dataset_name, temp_dir, tolerance_s, fps, backend
            )
            all_errors.extend(temp_errs)
            if temp_errs and stop_on_first_error:
                return all_errors

        file_tasks = [
            (
                dataset_name,
                video_key,
                str(file_path),
                timestamps,
                tolerance_s,
                fps,
                backend,
                decode_batch_size,
            )
            for video_key, file_path, timestamps in files_with_ts
        ]

        if workers <= 1:
            for (video_key, file_path, timestamps), task in zip(
                files_with_ts, file_tasks, strict=True
            ):
                logging.info(
                    "[%s] Analyzing video: %s (%s)",
                    _worker_label(),
                    file_path.name,
                    video_key,
                )
                errs = _verify_one_file(
                    dataset_name=task[0],
                    video_key=task[1],
                    file_path=Path(task[2]),
                    timestamps=task[3],
                    tolerance_s=task[4],
                    fps=task[5],
                    backend=task[6],
                    decode_batch_size=task[7],
                )
                all_errors.extend(errs)
                if errs and stop_on_first_error:
                    return all_errors
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker_logging,
            ) as executor:
                future_to_task = {
                    executor.submit(_verify_one_file_worker, task): task
                    for task in file_tasks
                }
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        errs = future.result()
                    except Exception as e:
                        logging.exception("Worker failed for %s: %s", task[2], e)
                        all_errors.append(
                            FrameError(
                                dataset_name=task[0],
                                file_name=Path(task[2]).name,
                                video_key=task[1],
                                timestamp_s=0.0,
                                frame_index=0,
                                message=f"Worker failed: {e!s}",
                            )
                        )
                        if stop_on_first_error:
                            return all_errors
                    if errs:
                        all_errors.extend(errs)
                        if stop_on_first_error:
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
        help="Video decoder backend (default: torchcodec). No pyav fallback.",
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
        help="Step in seconds between checked timestamps per file. When set, overrides --num_sample_frames.",
    )
    parser.add_argument(
        "--corruption_check",
        action="store_true",
        help="Use step-based sampling for corruption detection (step_s=1.0s if --step_s not set).",
    )
    parser.add_argument(
        "--decode_batch_size",
        type=int,
        default=1,
        help="Decode this many timestamps per call (default: 1). Larger values are faster but report approximate error location.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of files to verify in parallel (default: 1).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write errors as JSON (one object per line).",
    )
    parser.add_argument(
        "--no-stop-on-first-error",
        action="store_true",
        help="Collect and report all errors instead of stopping after the first.",
    )

    args = parser.parse_args()
    init_logging()

    datasets_list = _parse_datasets(args.datasets)
    if not datasets_list:
        logging.error("No datasets provided.")
        return

    root = Path(args.root) if args.root else None
    step_s = args.step_s
    if args.corruption_check and step_s is None:
        step_s = 1.0
    errors = verify_datasets(
        datasets=datasets_list,
        root=root,
        tolerance_s=args.tolerance_s,
        video_backend=args.video_backend or None,
        num_sample_frames=args.num_sample_frames,
        step_s=step_s,
        decode_batch_size=args.decode_batch_size,
        workers=args.workers,
        stop_on_first_error=not args.no_stop_on_first_error,
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
