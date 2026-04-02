#!/usr/bin/env python

"""
Trace training decode errors back to source datasets and source video files.

This script reads `training_data_errors.txt` entries (`kind=video_decode`) and
maps each failing sample from:
  combined dataset index -> source dataset range -> source episode/video file.

It works best when the combined dataset contains `meta/source_datasets.json`
(written by the combine pipeline). If that manifest is not available, pass the
source dataset list explicitly via `--source_repo_ids`.
"""

import argparse
import ast
import csv
import json
import logging
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep HF datasets lock/cache files in a local writable folder by default.
os.environ.setdefault("HF_DATASETS_CACHE", str(Path.cwd() / ".hf_datasets_cache"))

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging


@dataclass
class DecodeErrorEntry:
    logged_at: str
    repo_id: str
    requested_idx: int
    replacement_idx: int
    attempt: int
    ep_idx: int
    video_key: str
    video_path: str
    query_ts: list[float]
    error_message: str
    raw_line: str


@dataclass
class SourceRange:
    repo_id: str
    frame_start: int
    frame_end_exclusive: int
    episode_start: int
    episode_end_exclusive: int


def _parse_dataset_paths(arg: str | None) -> list[str]:
    if not arg:
        return []
    arg = arg.strip()
    if not arg:
        return []

    if arg.startswith("[") and arg.endswith("]"):
        try:
            cleaned = arg.replace(",]", "]").replace(", ]", "]")
            parsed = json.loads(cleaned)
            return [p for p in parsed if isinstance(p, str) and p.strip()]
        except json.JSONDecodeError:
            pass

    parsed = shlex.split(arg)
    return [p for p in parsed if p and p not in ["[", "]"]]


def _load_source_repo_ids_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            return [p for p in parsed if isinstance(p, str) and p.strip()]
        except json.JSONDecodeError:
            pass

    repo_ids: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Allow JSON-style trailing commas in line-based files.
        if line.endswith(","):
            line = line[:-1].strip()
        # Strip optional quotes.
        if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
            line = line[1:-1]
        if line:
            repo_ids.append(line)
    return repo_ids


def _parse_decode_error_line(line: str) -> DecodeErrorEntry | None:
    line = line.strip()
    if " kind=video_decode " not in line:
        return None

    try:
        timestamp = line[:19]
        remainder = line[20:]
        fields: dict[str, str] = {}
        keys = [
            "kind",
            "repo_id",
            "requested_idx",
            "replacement_idx",
            "attempt",
            "ep_idx",
            "video_key",
            "video_path",
            "query_ts",
            "error",
        ]

        for i, key in enumerate(keys):
            marker = f"{key}="
            start = remainder.find(marker)
            if start < 0:
                continue
            start += len(marker)

            end = len(remainder)
            for next_key in keys[i + 1 :]:
                next_marker = f" {next_key}="
                pos = remainder.find(next_marker, start)
                if pos >= 0:
                    end = min(end, pos)
            fields[key] = remainder[start:end].strip()

        if fields.get("kind") != "video_decode":
            return None

        query_ts = ast.literal_eval(fields["query_ts"]) if "query_ts" in fields else []
        if not isinstance(query_ts, list):
            query_ts = []

        return DecodeErrorEntry(
            logged_at=timestamp,
            repo_id=fields["repo_id"],
            requested_idx=int(fields["requested_idx"]),
            replacement_idx=int(fields["replacement_idx"]),
            attempt=int(fields["attempt"]),
            ep_idx=int(fields["ep_idx"]),
            video_key=fields["video_key"],
            video_path=fields["video_path"],
            query_ts=[float(x) for x in query_ts],
            error_message=fields.get("error", ""),
            raw_line=line,
        )
    except Exception:
        return None


def _load_decode_errors(error_log_path: Path) -> list[DecodeErrorEntry]:
    errors: list[DecodeErrorEntry] = []
    with open(error_log_path, encoding="utf-8") as f:
        for line in f:
            parsed = _parse_decode_error_line(line)
            if parsed is not None:
                errors.append(parsed)
    return errors


def _dataset_root_for(repo_id: str, hf_root: Path | None) -> Path | None:
    if hf_root is None:
        return None
    return hf_root / repo_id


def _load_manifest_ranges(manifest_path: Path) -> list[SourceRange]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    ranges = []
    for entry in payload.get("source_ranges", []):
        ranges.append(
            SourceRange(
                repo_id=entry["repo_id"],
                frame_start=int(entry["frame_start"]),
                frame_end_exclusive=int(entry["frame_end_exclusive"]),
                episode_start=int(entry["episode_start"]),
                episode_end_exclusive=int(entry["episode_end_exclusive"]),
            )
        )
    return ranges


def _build_ranges_from_sources(
    source_repo_ids: list[str],
    hf_root: Path | None,
) -> tuple[list[SourceRange], dict[str, LeRobotDatasetMetadata]]:
    ranges: list[SourceRange] = []
    metas: dict[str, LeRobotDatasetMetadata] = {}
    frame_start = 0
    episode_start = 0

    for repo_id in source_repo_ids:
        root = _dataset_root_for(repo_id, hf_root)
        meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
        metas[repo_id] = meta

        frame_end = frame_start + meta.total_frames
        episode_end = episode_start + meta.total_episodes
        ranges.append(
            SourceRange(
                repo_id=repo_id,
                frame_start=frame_start,
                frame_end_exclusive=frame_end,
                episode_start=episode_start,
                episode_end_exclusive=episode_end,
            )
        )
        frame_start = frame_end
        episode_start = episode_end

    return ranges, metas


def _find_source_range(ranges: list[SourceRange], replacement_idx: int) -> SourceRange | None:
    for r in ranges:
        if r.frame_start <= replacement_idx < r.frame_end_exclusive:
            return r
    return None


def _resolve_combined_repo_id(errors: list[DecodeErrorEntry], cli_repo_id: str | None) -> str:
    if cli_repo_id:
        return cli_repo_id
    unique = sorted({e.repo_id for e in errors})
    if len(unique) != 1:
        raise ValueError(
            "Could not infer a single combined repo_id from error log. "
            "Pass --combined_repo_id explicitly."
        )
    return unique[0]


def _classify_decode_error(error_message: str) -> str:
    msg = error_message.lower()
    if "invalid data found when processing input" in msg or "could not push packet to decoder" in msg:
        return "invalid_packet"
    if "violate the tolerance" in msg or "query timestamps unexpectedly violate the tolerance" in msg:
        return "timestamp_tolerance"
    return "other_decode_error"


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace decode errors to source dataset video files.")
    parser.add_argument(
        "--error_log",
        type=str,
        default="training_logs/training_data_errors.txt",
        help="Path to training_data_errors.txt",
    )
    parser.add_argument(
        "--combined_repo_id",
        type=str,
        default=None,
        help="Combined dataset repo_id (defaults to value inferred from error log).",
    )
    parser.add_argument(
        "--hf_root",
        type=str,
        default=None,
        help=f"Override HF root (default: {HF_LEROBOT_HOME}).",
    )
    parser.add_argument(
        "--source_manifest",
        type=str,
        default=None,
        help="Path to source manifest JSON. Defaults to <combined>/meta/source_datasets.json.",
    )
    parser.add_argument(
        "--source_repo_ids",
        type=str,
        default=None,
        help=(
            "Explicit source repo IDs (space-separated or JSON array) when manifest is missing."
        ),
    )
    parser.add_argument(
        "--source_repo_ids_file",
        type=str,
        default=None,
        help=(
            "Path to a file containing source repo IDs (JSON array or one repo_id per line). "
            "Used when manifest is missing."
        ),
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="training_logs/decode_error_source_trace.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    init_logging()

    error_log_path = Path(args.error_log)
    if not error_log_path.exists():
        raise FileNotFoundError(f"Error log not found: {error_log_path}")

    errors = _load_decode_errors(error_log_path)
    if not errors:
        logging.warning("No kind=video_decode entries found in %s", error_log_path)
        return

    hf_root = Path(args.hf_root) if args.hf_root else HF_LEROBOT_HOME
    combined_repo_id = _resolve_combined_repo_id(errors, args.combined_repo_id)
    combined_root = _dataset_root_for(combined_repo_id, hf_root)
    combined_meta = LeRobotDatasetMetadata(repo_id=combined_repo_id, root=combined_root)

    manifest_path = (
        Path(args.source_manifest)
        if args.source_manifest
        else (combined_meta.root / "meta" / "source_datasets.json")
    )

    source_ranges: list[SourceRange] = []
    source_metas: dict[str, LeRobotDatasetMetadata] = {}

    if manifest_path.exists():
        source_ranges = _load_manifest_ranges(manifest_path)
        logging.info("Using source manifest: %s", manifest_path)
    else:
        source_repo_ids = _parse_dataset_paths(args.source_repo_ids)
        if not source_repo_ids and args.source_repo_ids_file:
            source_repo_ids = _load_source_repo_ids_file(Path(args.source_repo_ids_file))
        if source_repo_ids:
            logging.info("Source manifest not found; using --source_repo_ids with %d entries.", len(source_repo_ids))
            source_ranges, source_metas = _build_ranges_from_sources(source_repo_ids, hf_root)
        else:
            logging.warning(
                "No source manifest found at %s and no --source_repo_ids provided. "
                "Output will include combined-level mapping only.",
                manifest_path,
            )

    rows: list[dict[str, Any]] = []

    for err in errors:
        row: dict[str, Any] = {
            "logged_at": err.logged_at,
            "combined_repo_id": err.repo_id,
            "requested_idx": err.requested_idx,
            "replacement_idx": err.replacement_idx,
            "attempt": err.attempt,
            "combined_episode_index": err.ep_idx,
            "video_key": err.video_key,
            "combined_video_path_from_error": err.video_path,
            "combined_query_ts": json.dumps(err.query_ts),
            "decode_error_message": err.error_message,
            "decode_error_class": _classify_decode_error(err.error_message),
            "mapping_status": "combined_only",
        }

        if not (0 <= err.ep_idx < len(combined_meta.episodes)):
            row["mapping_status"] = "invalid_combined_episode_index"
            rows.append(row)
            continue

        combined_ep = combined_meta.episodes[err.ep_idx]
        combined_chunk = int(combined_ep[f"videos/{err.video_key}/chunk_index"])
        combined_file = int(combined_ep[f"videos/{err.video_key}/file_index"])
        combined_from_ts = float(combined_ep[f"videos/{err.video_key}/from_timestamp"])
        combined_to_ts = float(combined_ep[f"videos/{err.video_key}/to_timestamp"])

        row["combined_video_chunk_index"] = combined_chunk
        row["combined_video_file_index"] = combined_file
        row["combined_video_from_timestamp"] = combined_from_ts
        row["combined_video_to_timestamp"] = combined_to_ts

        if not source_ranges:
            rows.append(row)
            continue

        source_range = _find_source_range(source_ranges, err.replacement_idx)
        if source_range is None:
            row["mapping_status"] = "source_range_not_found"
            rows.append(row)
            continue

        source_repo_id = source_range.repo_id
        row["source_repo_id"] = source_repo_id
        row["source_frame_index"] = err.replacement_idx - source_range.frame_start
        source_ep_idx = err.ep_idx - source_range.episode_start
        row["source_episode_index"] = source_ep_idx

        if source_repo_id not in source_metas:
            source_root = _dataset_root_for(source_repo_id, hf_root)
            source_metas[source_repo_id] = LeRobotDatasetMetadata(repo_id=source_repo_id, root=source_root)
        source_meta = source_metas[source_repo_id]

        if not (0 <= source_ep_idx < len(source_meta.episodes)):
            row["mapping_status"] = "source_episode_index_out_of_range"
            rows.append(row)
            continue

        source_ep = source_meta.episodes[source_ep_idx]
        source_chunk = int(source_ep[f"videos/{err.video_key}/chunk_index"])
        source_file = int(source_ep[f"videos/{err.video_key}/file_index"])
        source_from_ts = float(source_ep[f"videos/{err.video_key}/from_timestamp"])
        source_to_ts = float(source_ep[f"videos/{err.video_key}/to_timestamp"])

        source_video_path = source_meta.root / source_meta.get_video_file_path(source_ep_idx, err.video_key)
        row["source_video_chunk_index"] = source_chunk
        row["source_video_file_index"] = source_file
        row["source_video_path"] = str(source_video_path)
        row["source_video_from_timestamp"] = source_from_ts
        row["source_video_to_timestamp"] = source_to_ts

        offset = combined_from_ts - source_from_ts
        source_query_ts = [q - offset for q in err.query_ts]
        row["source_query_ts"] = json.dumps(source_query_ts)
        row["combined_source_timestamp_offset"] = offset
        row["mapping_status"] = "mapped_to_source"

        rows.append(row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    total = len(rows)
    mapped = sum(1 for r in rows if r.get("mapping_status") == "mapped_to_source")
    logging.info("Wrote %d rows to %s", total, output_csv)
    logging.info("Mapped to source: %d / %d", mapped, total)


if __name__ == "__main__":
    main()
