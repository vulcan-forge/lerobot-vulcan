#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 Vulcan Robotics, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Plot one episode from SARM progress parquet for quick ranking debugging."""

import argparse
import csv
import importlib.util
import logging
from pathlib import Path

import numpy as np

_RANK_SCRIPT = Path(__file__).resolve().parent / "sourccey_rank_sarm_progress.py"
_RANK_SPEC = importlib.util.spec_from_file_location("sourccey_rank_sarm_progress_mod", _RANK_SCRIPT)
if _RANK_SPEC is None or _RANK_SPEC.loader is None:
    raise RuntimeError(f"Failed to load ranking script from: {_RANK_SCRIPT}")
_RANK_MODULE = importlib.util.module_from_spec(_RANK_SPEC)
_RANK_SPEC.loader.exec_module(_RANK_MODULE)

EpisodeMetrics = _RANK_MODULE.EpisodeMetrics
_load_episode_progress = _RANK_MODULE._load_episode_progress
_save_episode_chart = _RANK_MODULE._save_episode_chart
compute_episode_metrics = _RANK_MODULE.compute_episode_metrics
resolve_progress_path = _RANK_MODULE.resolve_progress_path


def _default_episode_metrics(episode_index: int, n_frames: int) -> EpisodeMetrics:
    nan = float("nan")
    return EpisodeMetrics(
        episode_index=episode_index,
        n_frames=n_frames,
        valid_ratio=nan,
        final_progress=nan,
        max_progress=nan,
        mean_progress=nan,
        upward_ratio=nan,
        max_drawdown=nan,
        largest_single_drop=nan,
        stability_score=nan,
        quality_score=nan,
        rank=-1,
    )


def _load_episode_from_ranking_csv(ranking_csv: Path, episode_index: int, n_frames: int) -> EpisodeMetrics | None:
    if not ranking_csv.exists():
        raise FileNotFoundError(f"Ranking CSV not found: {ranking_csv}")

    with ranking_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "episode_index" not in row:
                continue
            try:
                row_episode_idx = int(row["episode_index"])
            except (TypeError, ValueError):
                continue
            if row_episode_idx != episode_index:
                continue

            rank_str = row.get("rank") or row.get("overall_rank") or "-1"
            score_str = row.get("quality_score") or row.get("score") or "nan"
            try:
                rank = int(rank_str)
            except (TypeError, ValueError):
                rank = -1
            try:
                quality_score = float(score_str)
            except (TypeError, ValueError):
                quality_score = float("nan")

            m = _default_episode_metrics(episode_index=episode_index, n_frames=n_frames)
            m.rank = rank
            m.quality_score = quality_score
            return m

    return None


def _compute_episode_metrics_from_progress(
    progress_path: Path,
    progress_column: str,
    episode_index: int,
    weight_max: float,
    weight_mean: float,
    weight_upward: float,
    weight_stability: float,
) -> EpisodeMetrics:
    metrics = compute_episode_metrics(
        progress_path=progress_path,
        progress_column=progress_column,
        weight_max=weight_max,
        weight_mean=weight_mean,
        weight_upward=weight_upward,
        weight_stability=weight_stability,
    )
    by_episode = {m.episode_index: m for m in metrics}
    if episode_index not in by_episode:
        raise ValueError(f"Episode {episode_index} not found in {progress_path}")
    return by_episode[episode_index]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot a single episode chart from sarm_progress.parquet for ranking debug.",
    )
    parser.add_argument("--progress-path", type=str, default=None, help="Path to sarm_progress.parquet")
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=None,
        help="Dataset repo id (used to auto-resolve parquet if --progress-path is omitted)",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="Episode index to chart (0-indexed)",
    )
    parser.add_argument(
        "--head-mode",
        type=str,
        default="sparse",
        choices=["sparse", "dense"],
        help="Which progress column to plot (progress_sparse/progress_dense)",
    )
    parser.add_argument(
        "--ranking-csv",
        type=str,
        default=None,
        help="Optional ranking CSV to read rank/score from (e.g., episode_ranking_progress.csv)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="PNG output path (default: outputs/sarm_episode_ranking_progress/episode_<idx>_<head>.png)",
    )
    parser.add_argument(
        "--chart-rolling-window",
        type=int,
        default=61,
        help="Moving-average window for smoothing (1 disables smoothing)",
    )
    parser.add_argument(
        "--chart-dpi",
        type=int,
        default=160,
        help="DPI for PNG output",
    )
    parser.add_argument("--weight-max", type=float, default=0.40, help="Weight for max progress")
    parser.add_argument("--weight-mean", type=float, default=0.25, help="Weight for mean progress")
    parser.add_argument("--weight-upward", type=float, default=0.20, help="Weight for upward ratio")
    parser.add_argument("--weight-stability", type=float, default=0.15, help="Weight for stability score")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    progress_path = resolve_progress_path(args.progress_path, args.dataset_repo_id)
    progress_column = f"progress_{args.head_mode}"
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else (Path("outputs/sarm_episode_ranking_progress") / f"episode_{args.episode_index:04d}_{args.head_mode}.png")
    )

    frame_index, progress = _load_episode_progress(progress_path, args.episode_index, progress_column)

    episode_metrics = None
    if args.ranking_csv:
        ranking_csv = Path(args.ranking_csv).expanduser().resolve()
        episode_metrics = _load_episode_from_ranking_csv(
            ranking_csv=ranking_csv,
            episode_index=args.episode_index,
            n_frames=int(len(frame_index)),
        )
        if episode_metrics is None:
            logging.warning(
                "Episode %d not found in ranking CSV %s. Recomputing rank/score from parquet.",
                args.episode_index,
                ranking_csv,
            )

    if episode_metrics is None:
        episode_metrics = _compute_episode_metrics_from_progress(
            progress_path=progress_path,
            progress_column=progress_column,
            episode_index=args.episode_index,
            weight_max=args.weight_max,
            weight_mean=args.weight_mean,
            weight_upward=args.weight_upward,
            weight_stability=args.weight_stability,
        )

    _save_episode_chart(
        output_path=output_path,
        episode=episode_metrics,
        frame_index=frame_index,
        progress=progress,
        head_mode=args.head_mode,
        rolling_window=max(1, args.chart_rolling_window),
        dpi=max(72, args.chart_dpi),
    )

    logging.info("Saved chart: %s", output_path)
    if np.isfinite(episode_metrics.quality_score):
        logging.info(
            "Episode %d (%s): rank=%d score=%.6f",
            args.episode_index,
            args.head_mode,
            episode_metrics.rank,
            episode_metrics.quality_score,
        )
    else:
        logging.info(
            "Episode %d (%s): rank/score unavailable (ranking CSV omitted or incompatible).",
            args.episode_index,
            args.head_mode,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
