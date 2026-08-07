#!/usr/bin/env python

import contextlib
import os
import time
from pathlib import Path

DEFAULT_DATA_ERROR_LOG_PATH = "training_logs/training_data_errors.txt"


class VideoDecodeError(RuntimeError):
    def __init__(
        self,
        *,
        video_path: Path,
        video_key: str,
        ep_idx: int,
        shifted_query_ts: list[float],
        cause: Exception,
    ):
        super().__init__(str(cause))
        self.video_path = video_path
        self.video_key = video_key
        self.ep_idx = ep_idx
        self.shifted_query_ts = shifted_query_ts
        self.cause = cause


def is_known_data_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "could not push packet to decoder" in msg
        or "invalid data found when processing input" in msg
        or "no more frames available" in msg
        or ("invalid frame index" in msg and "must be less than" in msg)
        or "timestamp mismatch" in msg
    )


def append_data_error_log(
    *,
    repo_id: str,
    requested_idx: int,
    replacement_idx: int,
    attempt: int,
    exc: Exception,
) -> None:
    log_path = os.environ.get("LEROBOT_DATA_ERROR_LOG", DEFAULT_DATA_ERROR_LOG_PATH)
    log_path_obj = Path(log_path)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(exc, VideoDecodeError):
        line = (
            f"{ts} kind=video_decode repo_id={repo_id} requested_idx={requested_idx} "
            f"replacement_idx={replacement_idx} attempt={attempt} ep_idx={exc.ep_idx} "
            f"video_key={exc.video_key} video_path={exc.video_path} "
            f"query_ts={exc.shifted_query_ts} error={exc}\n"
        )
    else:
        line = (
            f"{ts} kind=data_error repo_id={repo_id} requested_idx={requested_idx} "
            f"replacement_idx={replacement_idx} attempt={attempt} error={type(exc).__name__}: {exc}\n"
        )
    with contextlib.suppress(Exception):
        log_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
