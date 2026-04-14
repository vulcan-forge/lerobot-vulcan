#!/usr/bin/env python

"""
Find bad episodes in a LeRobot dataset.

This script checks:
1) Missing/zero-byte referenced parquet files (meta/data)
2) Episode length vs video timestamp mismatch
3) Video decode integrity per unique video file (optional, parallel via ffmpeg)

It outputs a sorted episode index list and a ready-to-run `lerobot-edit-dataset`
delete command.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lerobot.utils.constants import HF_LEROBOT_HOME


@dataclass(frozen=True)
class VideoRef:
    video_key: str
    chunk_index: int
    file_index: int


def _load_episodes_df(dataset_root: Path) -> pd.DataFrame:
    ep_dir = dataset_root / "meta" / "episodes"
    parquet_files = sorted(ep_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found under: {ep_dir}")
    return pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)


def _video_keys(df: pd.DataFrame) -> list[str]:
    keys: list[str] = []
    for col in df.columns:
        if col.startswith("videos/") and col.endswith("/file_index"):
            keys.append(col[len("videos/") : -len("/file_index")])
    return sorted(set(keys))


def _video_path(root: Path, ref: VideoRef) -> Path:
    return root / "videos" / ref.video_key / f"chunk-{ref.chunk_index:03d}" / f"file-{ref.file_index:03d}.mp4"


def _data_path(root: Path, chunk_idx: int, file_idx: int) -> Path:
    return root / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"


def _meta_episodes_path(root: Path, chunk_idx: int, file_idx: int) -> Path:
    return root / "meta" / "episodes" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"


def _ffmpeg_decode_ok(path: Path) -> tuple[bool, str]:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    stderr = (res.stderr or "").strip()
    # Some ffmpeg builds can print hard decode errors but still return 0.
    # With `-v error`, any stderr output indicates a real decode problem.
    if res.returncode == 0 and not stderr:
        return True, ""
    if stderr:
        return False, stderr
    return False, f"ffmpeg exited with non-zero status: {res.returncode}"


def _collect_episode_refs(df: pd.DataFrame, video_key: str, chunk_idx: int, file_idx: int) -> list[int]:
    ccol = f"videos/{video_key}/chunk_index"
    fcol = f"videos/{video_key}/file_index"
    out = df[(df[ccol] == chunk_idx) & (df[fcol] == file_idx)]["episode_index"].tolist()
    return sorted(int(x) for x in out)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find bad episodes in a LeRobot dataset.")
    parser.add_argument("--repo_id", required=True, type=str, help="Dataset repo id, e.g. Combination/my_dataset")
    parser.add_argument(
        "--new_repo_id",
        default=None,
        type=str,
        help="Output repo id for the suggested delete command. Defaults to '<repo_id>_clean'.",
    )
    parser.add_argument(
        "--root",
        default=None,
        type=str,
        help="Dataset root path. Defaults to $HF_LEROBOT_HOME/repo_id",
    )
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        help="Parallel workers for decode checks (default: 8)",
    )
    parser.add_argument(
        "--skip_decode_check",
        action="store_true",
        help="Skip ffmpeg decode check and only run metadata checks.",
    )
    parser.add_argument(
        "--output_prefix",
        default=None,
        type=str,
        help="Optional output prefix for artifacts (.json/.txt).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.root) if args.root else HF_LEROBOT_HOME / args.repo_id

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info file: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = int(info["fps"])

    df = _load_episodes_df(dataset_root)
    vkeys = _video_keys(df)
    if not vkeys:
        print("No video keys found. Running parquet checks only.")

    bad_episodes: set[int] = set()
    reasons_per_episode: dict[int, list[str]] = defaultdict(list)
    reason_counts: Counter[str] = Counter()
    bad_video_files: list[dict] = []

    def mark_eps(episodes: list[int], reason: str) -> None:
        if not episodes:
            return
        reason_counts[reason] += len(episodes)
        for ep in episodes:
            bad_episodes.add(ep)
            reasons_per_episode[ep].append(reason)

    # 1) Missing/zero-byte referenced meta/episodes parquet files.
    if {"meta/episodes/chunk_index", "meta/episodes/file_index"}.issubset(df.columns):
        refs = df[["episode_index", "meta/episodes/chunk_index", "meta/episodes/file_index"]]
        for (chunk_idx, file_idx), g in refs.groupby(["meta/episodes/chunk_index", "meta/episodes/file_index"]):
            epath = _meta_episodes_path(dataset_root, int(chunk_idx), int(file_idx))
            if not epath.exists():
                mark_eps(sorted(int(x) for x in g["episode_index"].tolist()), "missing_meta_episodes_parquet")
            elif epath.stat().st_size == 0:
                mark_eps(sorted(int(x) for x in g["episode_index"].tolist()), "zero_byte_meta_episodes_parquet")

    # 2) Missing/zero-byte referenced data parquet files.
    if {"data/chunk_index", "data/file_index"}.issubset(df.columns):
        refs = df[["episode_index", "data/chunk_index", "data/file_index"]]
        for (chunk_idx, file_idx), g in refs.groupby(["data/chunk_index", "data/file_index"]):
            dpath = _data_path(dataset_root, int(chunk_idx), int(file_idx))
            if not dpath.exists():
                mark_eps(sorted(int(x) for x in g["episode_index"].tolist()), "missing_data_parquet")
            elif dpath.stat().st_size == 0:
                mark_eps(sorted(int(x) for x in g["episode_index"].tolist()), "zero_byte_data_parquet")

    # 3) length mismatch checks for each camera.
    for key in vkeys:
        from_col = f"videos/{key}/from_timestamp"
        to_col = f"videos/{key}/to_timestamp"
        if not {"episode_index", "length", from_col, to_col}.issubset(df.columns):
            continue
        expected = (df[to_col] * fps).round().astype(int) - (df[from_col] * fps).round().astype(int)
        bad = df[df["length"].astype(int) != expected.astype(int)]["episode_index"].tolist()
        mark_eps(sorted(int(x) for x in bad), f"length_timestamp_mismatch:{key}")

    # 4) decode check each unique video file and map failures to episodes.
    if not args.skip_decode_check and vkeys:
        unique_refs: list[VideoRef] = []
        for key in vkeys:
            ccol = f"videos/{key}/chunk_index"
            fcol = f"videos/{key}/file_index"
            if not {ccol, fcol}.issubset(df.columns):
                continue
            uniq = df[[ccol, fcol]].drop_duplicates()
            for _, r in uniq.iterrows():
                unique_refs.append(VideoRef(key, int(r[ccol]), int(r[fcol])))

        print(f"Decode-checking {len(unique_refs)} unique video files with {args.workers} workers...")

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            fut_to_ref = {ex.submit(_ffmpeg_decode_ok, _video_path(dataset_root, ref)): ref for ref in unique_refs}
            done = 0
            for fut in as_completed(fut_to_ref):
                done += 1
                if done % 100 == 0:
                    print(f"  checked {done}/{len(unique_refs)}")
                ref = fut_to_ref[fut]
                ok, err = fut.result()
                if ok:
                    continue
                path = _video_path(dataset_root, ref)
                eps = _collect_episode_refs(df, ref.video_key, ref.chunk_index, ref.file_index)
                mark_eps(eps, f"video_decode_error:{ref.video_key}")
                bad_video_files.append(
                    {
                        "video_key": ref.video_key,
                        "chunk_index": ref.chunk_index,
                        "file_index": ref.file_index,
                        "path": str(path),
                        "episodes": eps,
                        "error": err.splitlines()[:3],
                    }
                )

    bad_sorted = sorted(bad_episodes)
    print("\nSummary")
    print(f"- dataset_root: {dataset_root}")
    print(f"- total_episodes: {len(df)}")
    print(f"- bad_episodes: {len(bad_sorted)}")
    if reason_counts:
        print("- reason_counts:")
        for reason, count in sorted(reason_counts.items()):
            print(f"  - {reason}: {count}")

    print("\nBad episode indices:")
    print(bad_sorted)

    quoted = ",".join(str(x) for x in bad_sorted)
    suggested_new_repo_id = args.new_repo_id or f"{args.repo_id}_clean"
    print("\nDelete command:")
    print(
        "uv run lerobot-edit-dataset "
        f'--repo_id="{args.repo_id}" '
        f'--new_repo_id="{suggested_new_repo_id}" '
        '--operation.type=delete_episodes '
        f'--operation.episode_indices="[{quoted}]"'
    )

    if args.output_prefix:
        prefix = Path(args.output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        (prefix.with_suffix(".txt")).write_text(",".join(str(x) for x in bad_sorted) + "\n", encoding="utf-8")
        payload = {
            "repo_id": args.repo_id,
            "suggested_new_repo_id": suggested_new_repo_id,
            "dataset_root": str(dataset_root),
            "bad_episode_count": len(bad_sorted),
            "bad_episodes": bad_sorted,
            "reasons_per_episode": {str(k): v for k, v in sorted(reasons_per_episode.items())},
            "bad_video_files": bad_video_files,
        }
        prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote artifacts:\n- {prefix.with_suffix('.txt')}\n- {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
