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
"""Custom Sourccey utility: rank episodes by progress stability (no final-score term).

This script is for curation when final-frame values are noisy or misleading.
It scores episodes using:
- max_progress
- mean_progress
- upward_ratio
- stability (low drawdown + low sudden drops)

Importantly, `final_progress` is NOT used in the ranking score.
"""

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class EpisodeMetrics:
    episode_index: int
    n_frames: int
    valid_ratio: float
    final_progress: float
    max_progress: float
    mean_progress: float
    upward_ratio: float
    max_drawdown: float
    largest_single_drop: float
    stability_score: float
    quality_score: float
    rank: int = -1


def resolve_progress_path(progress_path: str | None, dataset_repo_id: str | None) -> Path:
    if progress_path:
        resolved = Path(progress_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Progress file not found: {resolved}")
        return resolved

    if not dataset_repo_id:
        raise ValueError("Provide either --progress-path or --dataset-repo-id.")

    dataset = LeRobotDataset(dataset_repo_id, download_videos=False)
    resolved = dataset.root / "sarm_progress.parquet"
    if not resolved.exists():
        raise FileNotFoundError(
            f"Could not find sarm_progress.parquet at {resolved}. "
            "Run compute_rabc_weights.py first or pass --progress-path."
        )
    return resolved


def _compute_stability(valid_progress: np.ndarray) -> tuple[float, float, float]:
    """Return (stability_score, max_drawdown, largest_single_drop)."""
    if valid_progress.size < 2:
        return 1.0, 0.0, 0.0

    deltas = np.diff(valid_progress)
    running_max = np.maximum.accumulate(valid_progress)
    drawdowns = running_max - valid_progress
    max_drawdown = float(np.max(drawdowns))

    largest_single_drop = float(max(0.0, -np.min(deltas)))
    stability_score = 1.0 - (0.5 * min(max_drawdown, 1.0) + 0.5 * min(largest_single_drop, 1.0))
    stability_score = float(np.clip(stability_score, 0.0, 1.0))
    return stability_score, max_drawdown, largest_single_drop


def compute_episode_metrics(
    progress_path: Path,
    progress_column: str,
    weight_max: float,
    weight_mean: float,
    weight_upward: float,
    weight_stability: float,
) -> list[EpisodeMetrics]:
    table = pq.read_table(progress_path, columns=["index", "episode_index", progress_column])
    if progress_column not in table.column_names:
        raise ValueError(
            f"Column '{progress_column}' not found in {progress_path}. "
            f"Available columns: {sorted(table.column_names)}"
        )

    global_index = np.asarray(table["index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    episode_index = np.asarray(table["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    progress = np.asarray(table[progress_column].to_numpy(zero_copy_only=False), dtype=np.float64)

    order = np.lexsort((global_index, episode_index))
    episode_index = episode_index[order]
    progress = progress[order]

    split_points = np.where(np.diff(episode_index) != 0)[0] + 1
    starts = np.concatenate(([0], split_points))
    ends = np.concatenate((split_points, [len(episode_index)]))

    metrics: list[EpisodeMetrics] = []
    for start, end in zip(starts, ends, strict=True):
        ep_idx = int(episode_index[start])
        ep_progress = progress[start:end]
        n_frames = int(end - start)
        valid_mask = np.isfinite(ep_progress)
        valid_values = ep_progress[valid_mask]
        valid_ratio = float(valid_mask.mean()) if n_frames > 0 else 0.0

        if valid_values.size > 0:
            final_progress = float(valid_values[-1])
            max_progress = float(np.max(valid_values))
            mean_progress = float(np.mean(valid_values))
            if valid_values.size > 1:
                deltas = np.diff(valid_values)
                upward_ratio = float((deltas > 0).mean())
            else:
                upward_ratio = float("nan")
            stability_score, max_drawdown, largest_single_drop = _compute_stability(valid_values)
        else:
            final_progress = float("nan")
            max_progress = float("nan")
            mean_progress = float("nan")
            upward_ratio = float("nan")
            stability_score = 0.0
            max_drawdown = float("nan")
            largest_single_drop = float("nan")

        quality_score = (
            weight_max * (max_progress if np.isfinite(max_progress) else 0.0)
            + weight_mean * (mean_progress if np.isfinite(mean_progress) else 0.0)
            + weight_upward * (upward_ratio if np.isfinite(upward_ratio) else 0.0)
            + weight_stability * stability_score
        )
        metrics.append(
            EpisodeMetrics(
                episode_index=ep_idx,
                n_frames=n_frames,
                valid_ratio=valid_ratio,
                final_progress=final_progress,
                max_progress=max_progress,
                mean_progress=mean_progress,
                upward_ratio=upward_ratio,
                max_drawdown=max_drawdown,
                largest_single_drop=largest_single_drop,
                stability_score=stability_score,
                quality_score=float(quality_score),
            )
        )

    metrics.sort(key=lambda x: x.quality_score, reverse=True)
    for rank, entry in enumerate(metrics, start=1):
        entry.rank = rank

    return metrics


def write_ranking_csv(metrics: list[EpisodeMetrics], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "episode_index",
        "quality_score",
        "max_progress",
        "mean_progress",
        "upward_ratio",
        "stability_score",
        "max_drawdown",
        "largest_single_drop",
        "final_progress",
        "valid_ratio",
        "n_frames",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics:
            writer.writerow(
                {
                    "rank": m.rank,
                    "episode_index": m.episode_index,
                    "quality_score": f"{m.quality_score:.8f}",
                    "max_progress": f"{m.max_progress:.8f}" if np.isfinite(m.max_progress) else "nan",
                    "mean_progress": f"{m.mean_progress:.8f}" if np.isfinite(m.mean_progress) else "nan",
                    "upward_ratio": f"{m.upward_ratio:.8f}" if np.isfinite(m.upward_ratio) else "nan",
                    "stability_score": f"{m.stability_score:.8f}",
                    "max_drawdown": f"{m.max_drawdown:.8f}" if np.isfinite(m.max_drawdown) else "nan",
                    "largest_single_drop": (
                        f"{m.largest_single_drop:.8f}" if np.isfinite(m.largest_single_drop) else "nan"
                    ),
                    "final_progress": f"{m.final_progress:.8f}" if np.isfinite(m.final_progress) else "nan",
                    "valid_ratio": f"{m.valid_ratio:.8f}",
                    "n_frames": m.n_frames,
                }
            )


def _resolve_selection_size(total_episodes: int, k_value: int) -> int:
    """Interpret k as absolute count."""
    if total_episodes <= 0 or k_value <= 0:
        return 0
    return min(total_episodes, k_value)


def write_selection_csv(metrics: list[EpisodeMetrics], output_path: Path, selection_label: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "selection_rank",
        "overall_rank",
        "episode_index",
        "selection",
        "score",
        "max",
        "mean",
        "up",
        "stab",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for selection_rank, m in enumerate(metrics, start=1):
            writer.writerow(
                {
                    "selection_rank": selection_rank,
                    "overall_rank": m.rank,
                    "episode_index": m.episode_index,
                    "selection": selection_label,
                    "score": f"{m.quality_score:.6f}",
                    "max": f"{m.max_progress:.4f}" if np.isfinite(m.max_progress) else "nan",
                    "mean": f"{m.mean_progress:.4f}" if np.isfinite(m.mean_progress) else "nan",
                    "up": f"{m.upward_ratio:.4f}" if np.isfinite(m.upward_ratio) else "nan",
                    "stab": f"{m.stability_score:.4f}",
                }
            )


def select_episodes_to_drop(
    metrics: list[EpisodeMetrics],
    quality_below: float | None,
    bottom_quality_quantile: float | None,
    max_drawdown_above: float | None,
) -> list[int]:
    drop_set: set[int] = set()

    if quality_below is not None:
        for m in metrics:
            if m.quality_score < quality_below:
                drop_set.add(m.episode_index)

    if bottom_quality_quantile is not None:
        if not 0.0 < bottom_quality_quantile < 1.0:
            raise ValueError("--bottom-quality-quantile must be in (0, 1).")
        scores = np.array([m.quality_score for m in metrics], dtype=np.float64)
        threshold = float(np.quantile(scores, bottom_quality_quantile))
        for m in metrics:
            if m.quality_score <= threshold:
                drop_set.add(m.episode_index)

    if max_drawdown_above is not None:
        for m in metrics:
            if np.isfinite(m.max_drawdown) and m.max_drawdown > max_drawdown_above:
                drop_set.add(m.episode_index)

    return sorted(drop_set)


def _print_preview(metrics: list[EpisodeMetrics], top_k: int, bottom_k: int) -> None:
    def fmt(m: EpisodeMetrics) -> str:
        return (
            f"ep={m.episode_index:4d} rank={m.rank:4d} score={m.quality_score:.6f} "
            f"max={m.max_progress:.4f} mean={m.mean_progress:.4f} up={m.upward_ratio:.4f} "
            f"stab={m.stability_score:.4f} drawdown={m.max_drawdown:.4f}"
        )

    logging.info("Top %d episodes:", top_k)
    for m in metrics[:top_k]:
        logging.info("  %s", fmt(m))

    logging.info("Bottom %d episodes:", bottom_k)
    for m in metrics[-bottom_k:]:
        logging.info("  %s", fmt(m))


def _moving_average_with_nans(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    weights = np.ones(window, dtype=np.float64)
    valid = np.isfinite(values)
    safe_values = np.where(valid, values, 0.0)
    numerator = np.convolve(safe_values, weights, mode="same")
    denominator = np.convolve(valid.astype(np.float64), weights, mode="same")
    smoothed = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    return smoothed


def _load_episode_progress(
    progress_path: Path, episode_index: int, progress_column: str
) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(
        progress_path,
        columns=["index", "frame_index", progress_column],
        filters=[("episode_index", "=", episode_index)],
    )
    if table.num_rows == 0:
        raise ValueError(f"Episode {episode_index} not found in {progress_path}")
    global_index = np.asarray(table["index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    frame_index = np.asarray(table["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    progress = np.asarray(table[progress_column].to_numpy(zero_copy_only=False), dtype=np.float64)
    order = np.argsort(global_index)
    return frame_index[order], progress[order]


def _save_episode_chart(
    output_path: Path,
    episode: EpisodeMetrics,
    frame_index: np.ndarray,
    progress: np.ndarray,
    head_mode: str,
    rolling_window: int,
    dpi: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required to export charts. Install with extras that include matplotlib."
        ) from e

    smoothed = _moving_average_with_nans(progress, rolling_window)
    valid = np.isfinite(progress)
    valid_values = progress[valid]
    valid_frames = frame_index[valid]
    deltas = np.diff(valid_values) if valid_values.size > 1 else np.array([], dtype=np.float64)
    delta_frames = valid_frames[1:] if valid_values.size > 1 else np.array([], dtype=np.int64)

    fig, (ax_progress, ax_delta) = plt.subplots(
        2,
        1,
        figsize=(13, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    ax_progress.plot(frame_index, progress, color="#1f77b4", alpha=0.45, linewidth=1.2, label="Raw")
    if rolling_window > 1:
        ax_progress.plot(
            frame_index,
            smoothed,
            color="#d62728",
            linewidth=2.0,
            label=f"Smoothed (w={rolling_window})",
        )
    ax_progress.set_title(
        f"Episode {episode.episode_index} | rank={episode.rank} | score={episode.quality_score:.6f} | {head_mode}"
    )
    ax_progress.set_ylabel("Progress")
    ax_progress.set_ylim(-0.05, 1.05)
    ax_progress.grid(alpha=0.3)
    ax_progress.legend(loc="lower right", fontsize=9)

    if deltas.size > 0:
        ax_delta.plot(delta_frames, deltas, color="#ff7f0e", linewidth=1.0, label="Δ Progress")
        min_idx = int(np.argmin(deltas))
        ax_delta.scatter(
            [delta_frames[min_idx]],
            [deltas[min_idx]],
            color="red",
            s=24,
            zorder=5,
            label=f"Min Δ={deltas[min_idx]:.4f}",
        )
    ax_delta.axhline(0.0, color="black", linewidth=0.9, alpha=0.5)
    ax_delta.set_ylabel("Δ")
    ax_delta.set_xlabel("Frame Index")
    ax_delta.grid(alpha=0.3)
    if deltas.size > 0:
        ax_delta.legend(loc="lower right", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _export_ranked_charts(
    *,
    metrics: list[EpisodeMetrics],
    progress_path: Path,
    progress_column: str,
    head_mode: str,
    chart_dir: Path,
    chart_top_k: int,
    chart_bottom_k: int,
    chart_rolling_window: int,
    chart_dpi: int,
) -> None:
    if chart_top_k <= 0 and chart_bottom_k <= 0:
        return

    selections: list[tuple[str, EpisodeMetrics]] = []
    if chart_top_k > 0:
        selections.extend(("top", m) for m in metrics[:chart_top_k])
    if chart_bottom_k > 0:
        selections.extend(("bottom", m) for m in metrics[-chart_bottom_k:])

    for kind, m in selections:
        frame_index, progress = _load_episode_progress(progress_path, m.episode_index, progress_column)
        chart_path = chart_dir / f"{kind}_rank{m.rank:04d}_ep{m.episode_index:04d}_{head_mode}.png"
        _save_episode_chart(
            output_path=chart_path,
            episode=m,
            frame_index=frame_index,
            progress=progress,
            head_mode=head_mode,
            rolling_window=chart_rolling_window,
            dpi=chart_dpi,
        )
    logging.info(
        "Saved %d chart(s) to %s",
        len(selections),
        chart_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Custom Sourccey ranking using progress+stability only (no final-score term)",
    )
    parser.add_argument("--progress-path", type=str, default=None, help="Path to sarm_progress.parquet")
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=None,
        help="Dataset repo id (used to auto-resolve parquet if --progress-path is omitted)",
    )
    parser.add_argument(
        "--head-mode",
        type=str,
        default="sparse",
        choices=["sparse", "dense"],
        help="Which progress column to rank on (progress_sparse/progress_dense)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sarm_episode_ranking_progress",
        help="Directory where ranking artifacts are written",
    )

    # Score weights (no final term by design)
    parser.add_argument("--weight-max", type=float, default=0.40, help="Weight for max progress")
    parser.add_argument("--weight-mean", type=float, default=0.25, help="Weight for mean progress")
    parser.add_argument("--weight-upward", type=float, default=0.20, help="Weight for upward ratio")
    parser.add_argument("--weight-stability", type=float, default=0.15, help="Weight for stability score")

    parser.add_argument(
        "--preview-top-k",
        "--top-k",
        dest="preview_top_k",
        type=int,
        default=25,
        help="How many top episodes to print in console preview (alias: --top-k)",
    )
    parser.add_argument(
        "--preview-bottom-k",
        "--bottom-k",
        dest="preview_bottom_k",
        type=int,
        default=25,
        help="How many bottom episodes to print in console preview (alias: --bottom-k)",
    )
    parser.add_argument(
        "--file-top-k",
        type=int,
        default=300,
        help="How many top episodes to export in top-selection CSV",
    )
    parser.add_argument(
        "--file-bottom-k",
        type=int,
        default=300,
        help="How many bottom episodes to export in bottom-selection CSV",
    )
    parser.add_argument(
        "--save-charts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to save per-episode progress charts for ranked top/bottom episodes",
    )
    parser.add_argument(
        "--chart-dir",
        type=str,
        default=None,
        help="Directory for chart PNGs (default: <output-dir>/charts)",
    )
    parser.add_argument(
        "--chart-top-k",
        type=int,
        default=0,
        help="Number of top-ranked episodes to export as charts (0 disables)",
    )
    parser.add_argument(
        "--chart-bottom-k",
        type=int,
        default=0,
        help="Number of bottom-ranked episodes to export as charts (0 disables)",
    )
    parser.add_argument(
        "--chart-rolling-window",
        type=int,
        default=61,
        help="Moving-average window for chart smoothing (1 disables smoothing)",
    )
    parser.add_argument(
        "--chart-dpi",
        type=int,
        default=160,
        help="DPI used when saving chart PNGs",
    )

    # Drop criteria (all optional)
    parser.add_argument(
        "--quality-below",
        type=float,
        default=None,
        help="Mark episodes with quality score below this threshold",
    )
    parser.add_argument(
        "--bottom-quality-quantile",
        type=float,
        default=None,
        help="Mark bottom quantile episodes by quality score (e.g., 0.1 = worst 10%%)",
    )
    parser.add_argument(
        "--max-drawdown-above",
        type=float,
        default=None,
        help="Mark episodes with max drawdown above this threshold",
    )
    parser.add_argument(
        "--new-repo-id",
        type=str,
        default=None,
        help="Optional: used to print a ready-to-run delete command preview",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    progress_path = resolve_progress_path(args.progress_path, args.dataset_repo_id)
    progress_column = f"progress_{args.head_mode}"
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Ranking episodes from %s (%s)", progress_path, progress_column)
    metrics = compute_episode_metrics(
        progress_path=progress_path,
        progress_column=progress_column,
        weight_max=args.weight_max,
        weight_mean=args.weight_mean,
        weight_upward=args.weight_upward,
        weight_stability=args.weight_stability,
    )

    ranking_csv = output_dir / "episode_ranking_progress.csv"
    write_ranking_csv(metrics, ranking_csv)
    logging.info("Saved ranking CSV: %s", ranking_csv)

    preview_top_count = _resolve_selection_size(len(metrics), args.preview_top_k)
    preview_bottom_count = _resolve_selection_size(len(metrics), args.preview_bottom_k)

    top_selection_count = _resolve_selection_size(len(metrics), args.file_top_k)
    bottom_selection_count = _resolve_selection_size(len(metrics), args.file_bottom_k)

    top_selection = metrics[:top_selection_count]
    bottom_selection = list(reversed(metrics[-bottom_selection_count:])) if bottom_selection_count > 0 else []

    top_selection_csv = output_dir / f"episodes_top{top_selection_count}_progress.csv"
    bottom_selection_csv = output_dir / f"episodes_bottom{bottom_selection_count}_progress.csv"
    write_selection_csv(top_selection, top_selection_csv, selection_label="top")
    write_selection_csv(bottom_selection, bottom_selection_csv, selection_label="bottom")
    logging.info(
        "Saved top-selection CSV (%d episodes, from --file-top-k=%d): %s",
        top_selection_count,
        args.file_top_k,
        top_selection_csv,
    )
    logging.info(
        "Saved bottom-selection CSV (%d episodes, from --file-bottom-k=%d): %s",
        bottom_selection_count,
        args.file_bottom_k,
        bottom_selection_csv,
    )

    _print_preview(metrics, top_k=max(preview_top_count, 1), bottom_k=max(preview_bottom_count, 1))

    if args.save_charts:
        chart_dir = (
            Path(args.chart_dir).expanduser().resolve()
            if args.chart_dir
            else (output_dir / "charts").resolve()
        )
        _export_ranked_charts(
            metrics=metrics,
            progress_path=progress_path,
            progress_column=progress_column,
            head_mode=args.head_mode,
            chart_dir=chart_dir,
            chart_top_k=max(0, args.chart_top_k),
            chart_bottom_k=max(0, args.chart_bottom_k),
            chart_rolling_window=max(1, args.chart_rolling_window),
            chart_dpi=max(72, args.chart_dpi),
        )

    episodes_to_drop = select_episodes_to_drop(
        metrics=metrics,
        quality_below=args.quality_below,
        bottom_quality_quantile=args.bottom_quality_quantile,
        max_drawdown_above=args.max_drawdown_above,
    )
    if episodes_to_drop:
        drop_json = output_dir / "episodes_to_drop_progress.json"
        drop_json.write_text(json.dumps(episodes_to_drop))
        logging.info("Marked %d episodes to drop", len(episodes_to_drop))
        logging.info("Saved: %s", drop_json)

        if args.dataset_repo_id:
            new_repo_id = args.new_repo_id or f"{args.dataset_repo_id}_filtered_progress"
            logging.info("Delete command preview:")
            logging.info(
                "uv run lerobot-edit-dataset --repo_id \"%s\" --new_repo_id \"%s\" "
                "--operation.type delete_episodes --operation.episode_indices \"%s\"",
                args.dataset_repo_id,
                new_repo_id,
                episodes_to_drop,
            )
    else:
        logging.info("No drop thresholds specified (or no episodes matched). Ranking CSV is ready.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
