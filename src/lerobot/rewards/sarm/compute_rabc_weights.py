#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Compute SARM progress values for RA-BC (Reward-Aware Behavior Cloning) weighting.

This script processes all frames in a dataset with SARM to compute progress values [0, 1].
The results are saved as a parquet file that can be loaded during training for RA-BC weighting.

Uses multi-output extraction: each SARM query returns progress for 9 frames, so we only
need ~num_frames/30 queries instead of one per frame (~30x speedup).

Usage:
    # Full RA-BC computation with visualizations
    python src/lerobot/rewards/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path <USER>/sarm_single_uni4

    # Faster computation with stride (compute every 5 frames, interpolate the rest)
    python src/lerobot/rewards/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path <USER>/sarm_single_uni4 \\
        --stride 5

    # Visualize predictions only (no RA-BC computation)
    python src/lerobot/rewards/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path <USER>/sarm_single_uni4 \\
        --visualize-only \\
        --num-visualizations 5

    # Visualize specific episodes only
    python src/lerobot/rewards/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path <USER>/sarm_single_uni4 \\
        --visualize-only \\
        --visualize-episode-indices 32,44

The output is saved to the dataset's local cache directory as 'sarm_progress.parquet'.
"""

import argparse
import csv
import json
import logging
import re
import shutil
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset
from lerobot.rewards.sarm.modeling_sarm import SARMRewardModel
from lerobot.rewards.sarm.processor_sarm import make_sarm_pre_post_processors
from lerobot.rewards.sarm.sarm_utils import normalize_stage_tau


def get_reward_model_path_from_parquet(parquet_path: Path) -> str | None:
    """Read reward_model_path from parquet metadata if available."""
    if not parquet_path.exists():
        return None
    try:
        metadata = pq.read_metadata(parquet_path).schema.to_arrow_schema().metadata
        if metadata and b"reward_model_path" in metadata:
            return metadata[b"reward_model_path"].decode()
    except Exception:  # nosec B110
        return None
    return None


def load_sarm_resources(
    dataset_repo_id: str,
    reward_model_path: str,
    device: str = "cuda",
) -> tuple[LeRobotDataset, SARMRewardModel, any]:
    """
    Load SARM model, dataset, and preprocessor.

    Returns:
        Tuple of (dataset, reward_model, preprocessor)
    """
    logging.info(f"Loading model: {reward_model_path}")
    reward_model = SARMRewardModel.from_pretrained(reward_model_path)
    reward_model.config.device = device
    reward_model.to(device).eval()

    image_keys = reward_model.config.camera_keys
    state_key = reward_model.config.state_key
    delta_indices = reward_model.config.observation_delta_indices

    logging.info(f"Loading dataset: {dataset_repo_id}")
    temp_dataset = LeRobotDataset(dataset_repo_id, download_videos=True)
    fps = temp_dataset.fps

    delta_timestamps = {key: [idx / fps for idx in delta_indices] for key in image_keys}
    delta_timestamps[state_key] = [idx / fps for idx in delta_indices]
    dataset = LeRobotDataset(dataset_repo_id, delta_timestamps=delta_timestamps)
    logging.info(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    logging.info(f"SARM cameras: {image_keys}")

    preprocess, _ = make_sarm_pre_post_processors(
        config=reward_model.config,
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
    )

    return dataset, reward_model, preprocess


def to_numpy_image(img) -> np.ndarray:
    """Convert image tensor to numpy uint8 (H, W, C)."""
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.ndim == 4:
        # Take center frame for bidirectional sampling
        img = img[img.shape[0] // 2]
    if img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        # Handle normalized images (may have negative values or values > 1)
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # Normalize to [0, 1]
        img = (img * 255).astype(np.uint8)
    return img


def visualize_episode(
    frames, progress_preds, stage_preds, title, output_path, stage_labels, gt_progress=None, gt_stages=None
):
    """Create visualization with progress plot, stage probabilities, and sample frames.

    Same as sarm_inference_visualization.py
    """
    num_stages = stage_preds.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, num_stages))
    frame_indices = np.arange(len(progress_preds))

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    ax_progress, ax_stages, ax_frames = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    # Progress plot
    ax_progress.plot(frame_indices, progress_preds, linewidth=2, color="#2E86AB", label="Predicted")
    ax_progress.fill_between(frame_indices, 0, progress_preds, alpha=0.3, color="#2E86AB")
    if gt_progress is not None:
        ax_progress.plot(
            frame_indices, gt_progress, linewidth=2, color="#28A745", linestyle="--", label="Ground Truth"
        )
    ax_progress.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax_progress.set_ylabel("Progress")
    ax_progress.set_title(f'Task: "{title}"', fontweight="bold")
    ax_progress.set_ylim(-0.05, 1.1)
    ax_progress.legend(loc="upper left")
    ax_progress.grid(True, alpha=0.3)

    # Stage predictions
    ax_stages.stackplot(
        frame_indices,
        *[stage_preds[:, i] for i in range(num_stages)],
        colors=colors,
        alpha=0.8,
        labels=stage_labels,
    )
    if gt_stages is not None:
        for change_idx in np.where(np.diff(gt_stages) != 0)[0] + 1:
            ax_stages.axvline(x=change_idx, color="black", linestyle="-", alpha=0.7, linewidth=1.5)
    ax_stages.set_xlabel("Frame")
    ax_stages.set_ylabel("Stage Probability")
    ax_stages.set_ylim(0, 1)
    ax_stages.legend(loc="upper left", ncol=min(num_stages, 5), fontsize=8)
    ax_stages.grid(True, alpha=0.3)

    # Sample frames
    ax_frames.axis("off")
    num_sample = 8
    sample_indices = np.linspace(0, len(frames) - 1, num_sample, dtype=int)
    h, w = frames[0].shape[:2]
    combined = np.zeros((h, w * num_sample, 3), dtype=np.uint8)
    for i, idx in enumerate(sample_indices):
        frame = frames[idx]
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        combined[:, i * w : (i + 1) * w] = frame
        stage_name = stage_labels[np.argmax(stage_preds[idx])][:12]
        ax_frames.text(
            i * w + w / 2,
            -10,
            f"Frame {idx}\n{progress_preds[idx]:.2f}\n{stage_name}",
            ha="center",
            va="top",
            fontsize=7,
        )
    ax_frames.imshow(combined)
    ax_frames.set_title("Sample Frames", pad=20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def visualize_sarm_predictions(
    dataset: LeRobotDataset,
    reward_model: SARMRewardModel,
    preprocess,
    episode_indices: list[int],
    head_mode: str,
    output_dir: Path,
    num_display_frames: int = 5,
    stride: int = 1,
):
    """
    Visualize SARM predictions for multiple episodes.

    Computes predictions for every frame by default. With stride > 1, computes predictions
    every N frames and interpolates (progress + stage probabilities) for visualization.

    Args:
        dataset: LeRobotDataset with delta_timestamps configured
        reward_model: Loaded SARM model
        preprocess: Preprocessor from make_sarm_pre_post_processors
        episode_indices: List of episode indices to visualize
        head_mode: "sparse", "dense", or "both"
        output_dir: Directory to save visualizations
        num_display_frames: Number of frames to display in thumbnail strip (default: 5)
        stride: Compute predictions every N frames, interpolate the rest (default: 1)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_keys = reward_model.config.camera_keys
    primary_image_key = image_keys[0]
    state_key = reward_model.config.state_key
    dual_mode = reward_model.config.uses_dual_heads
    device = reward_model.device

    # Center frame index for bidirectional sampling
    target_idx = reward_model.config.n_obs_steps // 2

    # Determine which heads to visualize
    schemes_to_viz = []
    if head_mode in ("sparse", "both") or not dual_mode:
        schemes_to_viz.append("sparse")
    if head_mode in ("dense", "both") and dual_mode:
        schemes_to_viz.append("dense")

    # Set preprocessor to eval mode to disable augmentations
    if hasattr(preprocess, "eval"):
        preprocess.eval()
    for step in preprocess.steps:
        if hasattr(step, "eval"):
            step.eval()

    for episode_idx in episode_indices:
        ep = dataset.meta.episodes[episode_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        task = dataset[ep_start].get("task", "perform the task")
        num_frames = ep_end - ep_start

        # Select frames for display thumbnails (evenly sampled from begin to end)
        display_indices = set(
            [
                ep_start + int(i * (num_frames - 1) / (num_display_frames - 1))
                for i in range(num_display_frames)
            ]
            if num_frames >= num_display_frames
            else list(range(ep_start, ep_end))
        )
        viz_frames = {}

        # Load display frames up-front (stride mode might skip them otherwise).
        for frame_idx in display_indices:
            sample = dataset[frame_idx]
            viz_frames[frame_idx] = to_numpy_image(sample[primary_image_key])

        # Initialize storage for each scheme
        scheme_data = {}
        for scheme in schemes_to_viz:
            num_stages = getattr(reward_model.config, f"num_{scheme}_stages")
            scheme_data[scheme] = {
                "viz_progress": np.full(num_frames, np.nan),
                "viz_stages": np.full((num_frames, num_stages), np.nan),
                "viz_gt_progress": np.full(num_frames, np.nan),
                "viz_gt_stages": np.full(num_frames, np.nan),
                "target_key": f"{scheme}_targets",
                "num_stages": num_stages,
                "temporal_props": getattr(reward_model.config, f"{scheme}_temporal_proportions"),
                "subtask_names": getattr(reward_model.config, f"{scheme}_subtask_names"),
            }

        if stride > 1:
            logging.info(f"Visualization stride={stride}: inferring every {stride} frames and interpolating")

        # Process frames one at a time to avoid memory buildup
        frame_indices = list(range(ep_start, ep_end, stride))
        if (ep_end - 1) not in frame_indices:
            frame_indices.append(ep_end - 1)
        frame_indices = sorted(set(frame_indices))

        for frame_idx in tqdm(frame_indices, desc=f"Episode {episode_idx}", leave=False):
            local_idx = frame_idx - ep_start
            sample = dataset[frame_idx]

            batch = {
                "task": task,
                "index": frame_idx,
                "episode_index": episode_idx,
            }
            for key in image_keys:
                batch[key] = sample[key]
            if state_key in sample:
                batch[state_key] = sample[state_key]

            with torch.no_grad():
                processed = preprocess(batch)
                video_features = processed["video_features"].to(device)
                text_features = processed["text_features"].to(device)
                state_features = processed.get("state_features")
                if state_features is not None:
                    state_features = state_features.to(device)
                lengths = processed.get("lengths")

                for scheme in schemes_to_viz:
                    sd = scheme_data[scheme]

                    # Ground truth
                    # In stride visualization mode, ground-truth plots can be misleading
                    # (only sparse points are available), so we skip GT.
                    if stride == 1 and sd["target_key"] in processed:
                        gt_target = processed[sd["target_key"]][0, target_idx].cpu().item()
                        sd["viz_gt_stages"][local_idx] = int(gt_target)
                        sd["viz_gt_progress"][local_idx] = normalize_stage_tau(
                            gt_target,
                            num_stages=sd["num_stages"],
                            temporal_proportions=sd["temporal_props"],
                            subtask_names=sd["subtask_names"],
                        )

                    # Predictions
                    reward, stage_probs = reward_model.calculate_rewards(
                        text_embeddings=text_features,
                        video_embeddings=video_features,
                        state_features=state_features,
                        lengths=lengths,
                        return_all_frames=True,
                        return_stages=True,
                        head_mode=scheme,
                    )

                    # Handle both tensor and numpy outputs
                    if isinstance(reward, torch.Tensor):
                        reward = reward.cpu().numpy()
                        stage_probs = stage_probs.cpu().numpy()

                    if reward.ndim == 2:
                        sd["viz_progress"][local_idx] = reward[0, target_idx]
                        sd["viz_stages"][local_idx] = stage_probs[0, target_idx, :]
                    else:
                        sd["viz_progress"][local_idx] = reward[target_idx]
                        sd["viz_stages"][local_idx] = stage_probs[target_idx, :]

                # Clear GPU memory after each frame
                del processed, video_features, text_features
                if state_features is not None:
                    del state_features

            torch.cuda.empty_cache()

        # Interpolate predictions back to per-frame arrays for smooth visualization.
        if stride > 1:
            all_local = np.arange(num_frames)
            for scheme in schemes_to_viz:
                sd = scheme_data[scheme]

                valid = np.isfinite(sd["viz_progress"])
                valid_idx = np.where(valid)[0]
                if valid_idx.size >= 1:
                    sd["viz_progress"] = interpolate_progress(
                        valid_idx, sd["viz_progress"][valid_idx], all_local
                    )

                    stage_interp = np.zeros_like(sd["viz_stages"], dtype=np.float32)
                    for s in range(sd["num_stages"]):
                        stage_interp[:, s] = interpolate_progress(
                            valid_idx, sd["viz_stages"][valid_idx, s], all_local
                        )

                    stage_interp = np.clip(stage_interp, 0.0, 1.0)
                    row_sums = stage_interp.sum(axis=1, keepdims=True)
                    nz = row_sums.squeeze(-1) > 0
                    stage_interp[nz] = stage_interp[nz] / row_sums[nz]
                    sd["viz_stages"] = stage_interp
                else:
                    # No valid points: keep NaNs/zeros; visualization will be empty.
                    sd["viz_stages"] = np.nan_to_num(sd["viz_stages"], nan=0.0)

        # Generate visualization for each head
        ordered_viz_frames = [viz_frames[idx] for idx in sorted(display_indices)]
        for scheme in schemes_to_viz:
            sd = scheme_data[scheme]
            stage_labels = sd["subtask_names"] or [f"Stage {i + 1}" for i in range(sd["num_stages"])]
            viz_path = output_dir / f"sarm_prediction_ep{episode_idx}_{scheme}.png"

            visualize_episode(
                frames=np.array(ordered_viz_frames),
                progress_preds=sd["viz_progress"],
                stage_preds=sd["viz_stages"],
                title=f"{task} (Episode {episode_idx})",
                output_path=viz_path,
                stage_labels=stage_labels,
                gt_progress=sd["viz_gt_progress"] if not np.all(np.isnan(sd["viz_gt_progress"])) else None,
                gt_stages=sd["viz_gt_stages"] if not np.all(np.isnan(sd["viz_gt_stages"])) else None,
            )

        # Clear memory between episodes
        torch.cuda.empty_cache()

    logging.info(f"Visualizations saved to: {output_dir.absolute()}")


def _get_parts_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_parts"


def _parts_manifest_path(parts_dir: Path) -> Path:
    return parts_dir / "manifest.json"


def _episode_part_path(parts_dir: Path, episode_idx: int) -> Path:
    return parts_dir / f"episode_{episode_idx:06d}.parquet"


def _atomic_write_table(table: pa.Table, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    pq.write_table(table, tmp_path)
    tmp_path.replace(target)


def _write_parts_manifest(
    parts_dir: Path,
    *,
    dataset_repo_id: str,
    reward_model_path: str,
    head_mode: str,
    stride: int,
    compute_sparse: bool,
    compute_dense: bool,
    num_episodes: int,
    reset_parts: bool,
) -> None:
    if reset_parts and parts_dir.exists():
        shutil.rmtree(parts_dir)

    parts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _parts_manifest_path(parts_dir)
    expected = {
        "dataset_repo_id": dataset_repo_id,
        "reward_model_path": reward_model_path,
        "head_mode": head_mode,
        "stride": int(stride),
        "compute_sparse": bool(compute_sparse),
        "compute_dense": bool(compute_dense),
        "num_episodes": int(num_episodes),
    }

    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
        except Exception:  # nosec B110
            existing = None
        if existing != expected:
            logging.warning("Existing SARM parts manifest differs from current run. Resetting parts directory.")
            shutil.rmtree(parts_dir)
            parts_dir.mkdir(parents=True, exist_ok=True)

    manifest_path.write_text(json.dumps(expected, indent=2))


def _list_completed_episodes(parts_dir: Path) -> set[int]:
    completed: set[int] = set()
    if not parts_dir.exists():
        return completed
    pattern = re.compile(r"^episode_(\d{6})\.parquet$")
    for path in parts_dir.iterdir():
        if not path.is_file():
            continue
        m = pattern.match(path.name)
        if m:
            completed.add(int(m.group(1)))
    return completed


def _merge_parts_to_output(
    *,
    parts_dir: Path,
    output_path: Path,
    reward_model_path: str,
    expected_num_episodes: int,
) -> pa.Table:
    part_paths = sorted(parts_dir.glob("episode_*.parquet"))
    if not part_paths:
        raise RuntimeError(f"No episode part files found in {parts_dir}")

    if len(part_paths) < expected_num_episodes:
        logging.warning(
            "Merging incomplete parts: found %d / expected %d episodes",
            len(part_paths),
            expected_num_episodes,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    metadata = {b"reward_model_path": reward_model_path.encode()}
    writer = None
    try:
        for part_path in part_paths:
            table = pq.read_table(part_path)
            table = table.replace_schema_metadata(metadata)
            if writer is None:
                writer = pq.ParquetWriter(tmp_output, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    tmp_output.replace(output_path)
    return pq.read_table(output_path)


def generate_all_frame_indices(ep_start: int, ep_end: int, frame_gap: int = 30) -> list[int]:
    """Generate all frame indices, ordered by offset for cache-friendly access.

    Orders frames as: [0, 30, 60...], [1, 31, 61...], ..., [29, 59, 89...]
    This groups frames that share similar temporal windows together.
    """
    num_frames = ep_end - ep_start
    indices = []
    for offset in range(frame_gap):
        for frame_rel in range(offset, num_frames, frame_gap):
            indices.append(ep_start + frame_rel)
    return indices


def interpolate_progress(
    computed_indices: np.ndarray,
    computed_values: np.ndarray,
    all_indices: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate values to fill in gaps (robust to NaNs / edge cases)."""
    computed_indices = np.asarray(computed_indices)
    computed_values = np.asarray(computed_values)
    all_indices = np.asarray(all_indices)

    mask = np.isfinite(computed_values)
    if mask.sum() == 0:
        return np.full(all_indices.shape, np.nan, dtype=np.float32)
    if mask.sum() == 1:
        return np.full(all_indices.shape, float(computed_values[mask][0]), dtype=np.float32)

    out = np.interp(all_indices, computed_indices[mask], computed_values[mask])
    return out.astype(np.float32)


def parse_visualization_episode_indices(
    raw_values: list[str] | None, num_episodes: int | None = None
) -> list[int] | None:
    """Parse and validate episode indices from CLI strings.

    Accepts comma-separated values and/or repeated flags.
    Example: ["32"] or ["32,44", "100"].
    """
    if not raw_values:
        return None

    parsed: list[int] = []
    seen: set[int] = set()

    for raw in raw_values:
        for piece in raw.split(","):
            token = piece.strip()
            if not token:
                continue
            try:
                episode_idx = int(token)
            except ValueError as e:
                raise ValueError(
                    f"Invalid episode index '{token}' in --visualize-episode-indices. "
                    "Use integers like: --visualize-episode-indices 32,44"
                ) from e

            if episode_idx < 0:
                raise ValueError(f"Episode index {episode_idx} is invalid. Episode indices must be >= 0.")

            if num_episodes is not None and episode_idx >= num_episodes:
                raise ValueError(
                    f"Episode index {episode_idx} is out of range. "
                    f"Valid range is [0, {num_episodes - 1}]"
                )

            if episode_idx not in seen:
                parsed.append(episode_idx)
                seen.add(episode_idx)

    if not parsed:
        return None
    return parsed


def parse_visualization_ranking_indices(
    ranking_csv: str | None,
    top_k: int,
    bottom_k: int,
    num_episodes: int | None = None,
) -> list[int] | None:
    """Parse episode indices from a ranking CSV by taking top/bottom K rows."""
    top_selection, bottom_selection = parse_visualization_ranking_groups(
        ranking_csv=ranking_csv,
        top_k=top_k,
        bottom_k=bottom_k,
        num_episodes=num_episodes,
    )
    if not top_selection and not bottom_selection:
        return None

    selected: list[int] = []
    seen: set[int] = set()
    for episode_idx in top_selection + bottom_selection:
        if episode_idx not in seen:
            selected.append(episode_idx)
            seen.add(episode_idx)
    return selected if selected else None


def parse_visualization_ranking_groups(
    ranking_csv: str | None,
    top_k: int,
    bottom_k: int,
    num_episodes: int | None = None,
) -> tuple[list[int], list[int]]:
    """Parse top/bottom episode-index groups from a ranking CSV."""
    if not ranking_csv:
        if top_k > 0 or bottom_k > 0:
            raise ValueError(
                "--visualize-top-k/--visualize-bottom-k require --visualize-ranking-csv."
            )
        return [], []

    if top_k < 0 or bottom_k < 0:
        raise ValueError("--visualize-top-k and --visualize-bottom-k must be >= 0.")

    if top_k == 0 and bottom_k == 0:
        raise ValueError(
            "When --visualize-ranking-csv is provided, set --visualize-top-k and/or "
            "--visualize-bottom-k to a value > 0."
        )

    ranking_path = Path(ranking_csv).expanduser().resolve()
    if not ranking_path.exists():
        raise FileNotFoundError(f"Ranking CSV not found: {ranking_path}")

    episode_indices: list[int] = []
    with ranking_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "episode_index" not in reader.fieldnames:
            raise ValueError(
                f"Ranking CSV must contain an 'episode_index' column: {ranking_path}"
            )

        for line_no, row in enumerate(reader, start=2):
            raw_ep = (row.get("episode_index") or "").strip()
            if not raw_ep:
                continue
            try:
                episode_idx = int(raw_ep)
            except ValueError as e:
                raise ValueError(
                    f"Invalid episode_index '{raw_ep}' at line {line_no} in {ranking_path}"
                ) from e

            if episode_idx < 0:
                raise ValueError(f"Episode index {episode_idx} is invalid. Episode indices must be >= 0.")
            if num_episodes is not None and episode_idx >= num_episodes:
                raise ValueError(
                    f"Episode index {episode_idx} from ranking CSV is out of range. "
                    f"Valid range is [0, {num_episodes - 1}]"
                )
            episode_indices.append(episode_idx)

    if not episode_indices:
        raise ValueError(f"No episode_index values found in ranking CSV: {ranking_path}")

    top_count = min(top_k, len(episode_indices)) if top_k > 0 else 0
    bottom_count = min(bottom_k, len(episode_indices)) if bottom_k > 0 else 0

    top_selection = episode_indices[:top_count]
    bottom_selection = episode_indices[-bottom_count:] if bottom_count > 0 else []
    return top_selection, bottom_selection


def resolve_visualization_episode_selection(
    explicit_episode_indices_raw: list[str] | None,
    ranking_csv: str | None,
    ranking_top_k: int,
    ranking_bottom_k: int,
    num_episodes: int | None = None,
) -> list[int] | None:
    """Resolve final visualization episode list from explicit and ranking-derived sources."""
    explicit = parse_visualization_episode_indices(explicit_episode_indices_raw, num_episodes) or []
    ranked = parse_visualization_ranking_indices(
        ranking_csv=ranking_csv,
        top_k=ranking_top_k,
        bottom_k=ranking_bottom_k,
        num_episodes=num_episodes,
    ) or []

    if not explicit and not ranked:
        return None

    # Preserve user-specified ordering first, then append ranked episodes.
    combined: list[int] = []
    seen: set[int] = set()
    for episode_idx in explicit + ranked:
        if episode_idx not in seen:
            combined.append(episode_idx)
            seen.add(episode_idx)
    return combined


def compute_sarm_progress(
    dataset_repo_id: str,
    reward_model_path: str,
    output_path: str | None = None,
    head_mode: str = "sparse",
    device: str = "cuda",
    num_visualizations: int = 5,
    output_dir: str = "./sarm_viz",
    stride: int = 1,
    reset_parts: bool = False,
    visualize_episode_indices: list[int] | None = None,
):
    """
    Compute SARM progress predictions for all frames in a dataset.

    Args:
        dataset_repo_id: HuggingFace dataset repo ID or local path
        reward_model_path: Path to pretrained SARM model
        output_path: Path to save results. If None, saves to dataset's cache directory
        head_mode: SARM head to use ("sparse", "dense", or "both")
        device: Device to use for inference
        num_visualizations: Number of episodes to visualize (0 to skip)
        output_dir: Directory to save visualizations
        stride: Compute progress every N frames, interpolate the rest (default: 1 = every frame)
        reset_parts: If True, clears previous partial results and starts from scratch
        visualize_episode_indices: Optional explicit episode indices to visualize
    """
    dataset, reward_model, preprocess = load_sarm_resources(dataset_repo_id, reward_model_path, device)

    # Set preprocessor to eval mode to disable augmentations
    if hasattr(preprocess, "eval"):
        preprocess.eval()
    for step in preprocess.steps:
        if hasattr(step, "eval"):
            step.eval()

    image_keys = reward_model.config.camera_keys
    state_key = reward_model.config.state_key
    frame_gap = reward_model.config.frame_gap
    num_episodes = dataset.num_episodes
    total_frames = dataset.num_frames
    logging.info(f"Processing {total_frames} frames across {num_episodes} episodes")

    if visualize_episode_indices:
        invalid = [idx for idx in visualize_episode_indices if idx < 0 or idx >= num_episodes]
        if invalid:
            raise ValueError(
                f"Requested visualize episode index/indices out of range: {invalid}. "
                f"Valid range is [0, {num_episodes - 1}]"
            )

    # Determine which heads to compute
    dual_mode = reward_model.config.uses_dual_heads
    compute_sparse = head_mode in ("sparse", "both") or not dual_mode
    compute_dense = head_mode in ("dense", "both") and dual_mode

    # Determine output path and setup resumable episode parts.
    output_path = Path(dataset.root) / "sarm_progress.parquet" if output_path is None else Path(output_path)
    parts_dir = _get_parts_dir(output_path)
    _write_parts_manifest(
        parts_dir,
        dataset_repo_id=dataset_repo_id,
        reward_model_path=reward_model_path,
        head_mode=head_mode,
        stride=stride,
        compute_sparse=compute_sparse,
        compute_dense=compute_dense,
        num_episodes=num_episodes,
        reset_parts=reset_parts,
    )
    completed_episodes = _list_completed_episodes(parts_dir)
    if completed_episodes:
        logging.info(
            "Resuming from partial results in %s (%d/%d episodes already completed)",
            parts_dir,
            len(completed_episodes),
            num_episodes,
        )
    else:
        logging.info("No prior partial results found. Starting fresh.")

    if stride > 1:
        logging.info(f"Using stride={stride}: computing every {stride} frames, interpolating the rest")

    # Process all episodes
    for episode_idx in tqdm(range(num_episodes), desc="Episodes"):
        if episode_idx in completed_episodes:
            continue

        ep = dataset.meta.episodes[episode_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        # Get task description
        task = dataset[ep_start].get("task", "perform the task")

        # Generate frames to compute (with stride applied)
        all_ep_indices = generate_all_frame_indices(ep_start, ep_end, frame_gap)
        if stride > 1:
            # Only compute every stride-th frame (relative to episode start)
            compute_indices = [idx for idx in all_ep_indices if (idx - ep_start) % stride == 0]
            # Always include last frame for better interpolation at episode end
            last_frame = ep_end - 1
            if last_frame not in compute_indices:
                compute_indices.append(last_frame)
            compute_indices = sorted(set(compute_indices))
        else:
            compute_indices = all_ep_indices

        center_idx = reward_model.config.n_obs_steps // 2  # Center of bidirectional window

        # Dictionary to collect results
        frame_results = {}

        for query_idx in tqdm(compute_indices, desc=f"  Ep {episode_idx}", leave=False):
            try:
                sample = dataset[query_idx]

                batch = {
                    "task": task,
                    "index": query_idx,
                    "episode_index": episode_idx,
                }
                for key in image_keys:
                    batch[key] = sample[key]
                if state_key in sample:
                    batch[state_key] = sample[state_key]

                with torch.no_grad():
                    processed = preprocess(batch)
                    video_features = processed["video_features"].to(device)
                    text_features = processed["text_features"].to(device)
                    state_features = processed.get("state_features")
                    if state_features is not None:
                        state_features = state_features.to(device)
                    lengths = processed.get("lengths")

                    sparse_val = np.nan
                    dense_val = np.nan

                    # Compute sparse prediction for center frame
                    if compute_sparse:
                        sparse_progress = reward_model.calculate_rewards(
                            text_embeddings=text_features,
                            video_embeddings=video_features,
                            state_features=state_features,
                            lengths=lengths,
                            return_all_frames=True,
                            head_mode="sparse",
                        )
                        sparse_val = float(
                            sparse_progress[0, center_idx]
                            if sparse_progress.ndim == 2
                            else sparse_progress[center_idx]
                        )

                    # Compute dense prediction for center frame
                    if compute_dense:
                        dense_progress = reward_model.calculate_rewards(
                            text_embeddings=text_features,
                            video_embeddings=video_features,
                            state_features=state_features,
                            lengths=lengths,
                            return_all_frames=True,
                            head_mode="dense",
                        )
                        dense_val = float(
                            dense_progress[0, center_idx]
                            if dense_progress.ndim == 2
                            else dense_progress[center_idx]
                        )

                    frame_results[query_idx] = (sparse_val, dense_val)

            except Exception as e:
                logging.warning(f"Failed to process frame {query_idx}: {e}")

        # Interpolate to get values for all frames
        computed_indices = np.array(sorted(frame_results.keys()))
        computed_sparse = (
            np.array([frame_results[i][0] for i in computed_indices]) if compute_sparse else None
        )
        computed_dense = np.array([frame_results[i][1] for i in computed_indices]) if compute_dense else None

        # All frame indices for this episode
        all_frame_idx_array = np.arange(ep_start, ep_end)

        if stride > 1 and len(computed_indices) > 1:
            # Interpolate progress values
            if compute_sparse:
                interp_sparse = interpolate_progress(computed_indices, computed_sparse, all_frame_idx_array)
            if compute_dense:
                interp_dense = interpolate_progress(computed_indices, computed_dense, all_frame_idx_array)
        else:
            # No interpolation needed
            interp_sparse = computed_sparse if compute_sparse else None
            interp_dense = computed_dense if compute_dense else None

        # Episode-local storage.
        episode_indices: list[int] = []
        episode_episode_indices: list[int] = []
        episode_frame_indices: list[int] = []
        episode_progress_sparse: list[float] = []
        episode_progress_dense: list[float] = []

        # Store results for all frames
        for i, frame_idx in enumerate(all_frame_idx_array):
            local_idx = frame_idx - ep_start
            episode_indices.append(frame_idx)
            episode_episode_indices.append(episode_idx)
            episode_frame_indices.append(local_idx)
            if compute_sparse:
                if stride > 1 and len(computed_indices) > 1:
                    episode_progress_sparse.append(float(interp_sparse[i]))
                elif frame_idx in frame_results:
                    episode_progress_sparse.append(frame_results[frame_idx][0])
                else:
                    episode_progress_sparse.append(np.nan)
            if compute_dense:
                if stride > 1 and len(computed_indices) > 1:
                    episode_progress_dense.append(float(interp_dense[i]))
                elif frame_idx in frame_results:
                    episode_progress_dense.append(frame_results[frame_idx][1])
                else:
                    episode_progress_dense.append(np.nan)

        episode_table_data = {
            "index": np.array(episode_indices, dtype=np.int64),
            "episode_index": np.array(episode_episode_indices, dtype=np.int64),
            "frame_index": np.array(episode_frame_indices, dtype=np.int64),
        }
        if compute_sparse:
            episode_table_data["progress_sparse"] = np.array(episode_progress_sparse, dtype=np.float32)
        if compute_dense:
            episode_table_data["progress_dense"] = np.array(episode_progress_dense, dtype=np.float32)

        episode_table = pa.table(episode_table_data)
        metadata = {b"reward_model_path": reward_model_path.encode()}
        episode_table = episode_table.replace_schema_metadata(metadata)
        _atomic_write_table(episode_table, _episode_part_path(parts_dir, episode_idx))

    # Merge all episode parts into final output parquet.
    final_table = _merge_parts_to_output(
        parts_dir=parts_dir,
        output_path=output_path,
        reward_model_path=reward_model_path,
        expected_num_episodes=num_episodes,
    )
    logging.info("Saved %d frame progress values to %s", len(final_table), output_path)

    # Print statistics
    if "progress_sparse" in final_table.column_names:
        sparse_values = final_table["progress_sparse"].to_numpy(zero_copy_only=False)
        valid = sparse_values[np.isfinite(sparse_values)]
        if valid.size > 0:
            logging.info(
                "Sparse progress: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                float(valid.mean()),
                float(valid.std()),
                float(valid.min()),
                float(valid.max()),
            )

    if "progress_dense" in final_table.column_names:
        dense_values = final_table["progress_dense"].to_numpy(zero_copy_only=False)
        valid = dense_values[np.isfinite(dense_values)]
        if valid.size > 0:
            logging.info(
                "Dense progress: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                float(valid.mean()),
                float(valid.std()),
                float(valid.min()),
                float(valid.max()),
            )

    # Visualize episodes after processing
    if visualize_episode_indices:
        viz_episodes = visualize_episode_indices
        logging.info(f"Generating visualizations for requested episodes: {viz_episodes}")
        visualize_sarm_predictions(
            dataset=dataset,
            reward_model=reward_model,
            preprocess=preprocess,
            episode_indices=viz_episodes,
            head_mode=head_mode,
            output_dir=Path(output_dir),
            stride=stride,
        )
    elif num_visualizations > 0:
        viz_episodes = list(range(min(num_visualizations, num_episodes)))
        logging.info(f"Generating {len(viz_episodes)} visualizations...")
        visualize_sarm_predictions(
            dataset=dataset,
            reward_model=reward_model,
            preprocess=preprocess,
            episode_indices=viz_episodes,
            head_mode=head_mode,
            output_dir=Path(output_dir),
            stride=stride,
        )

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute SARM progress values for RA-BC weighting or visualize SARM predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full RA-BC computation with visualizations
    python src/lerobot/rewards/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path <USER>/sarm_single_uni4

    # Visualize predictions only (no RA-BC computation)
    python src/lerobot/rewards/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path <USER>/sarm_single_uni4 \\
        --visualize-only \\
        --num-visualizations 10

    # Visualize specific episodes only
    python src/lerobot/rewards/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path <USER>/sarm_single_uni4 \\
        --visualize-only \\
        --visualize-episode-indices 32,44

    # Visualize top/bottom episodes from ranking CSV
    python src/lerobot/rewards/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path <USER>/sarm_single_uni4 \\
        --visualize-only \\
        --visualize-ranking-csv outputs/sarm_episode_ranking_progress/episode_ranking_progress.csv \\
        --visualize-top-k 25 \\
        --visualize-bottom-k 25
        """,
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="HuggingFace dataset repo ID or local path",
    )
    parser.add_argument(
        "--reward-model-path",
        type=str,
        default=None,
        help="Path to pretrained SARM model (reads from existing parquet metadata if not provided)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for parquet. If not set, saves to dataset's cache directory",
    )
    parser.add_argument(
        "--head-mode",
        type=str,
        default="sparse",
        choices=["sparse", "dense", "both"],
        help="SARM head to use (default: sparse)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    # Visualization options
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only visualize SARM predictions (no RA-BC computation)",
    )
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=5,
        help="Number of episodes to visualize (default: 5, set to 0 to skip)",
    )
    parser.add_argument(
        "--visualize-episode-indices",
        action="append",
        default=None,
        help=(
            "Specific episode indices to visualize (comma-separated; can repeat). "
            "Example: --visualize-episode-indices 32,44"
        ),
    )
    parser.add_argument(
        "--visualize-ranking-csv",
        type=str,
        default=None,
        help="Ranking CSV path (e.g., episode_ranking_progress.csv) used for top/bottom episode selection.",
    )
    parser.add_argument(
        "--visualize-top-k",
        type=int,
        default=0,
        help="From --visualize-ranking-csv, include top K ranked episodes (0 disables).",
    )
    parser.add_argument(
        "--visualize-bottom-k",
        type=int,
        default=0,
        help="From --visualize-ranking-csv, include bottom K ranked episodes (0 disables).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./sarm_viz",
        help="Output directory for visualizations (default: ./sarm_viz)",
    )
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload progress file to the dataset repo on HuggingFace Hub (default: disabled)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Compute progress every N frames, interpolate the rest (default: 1 = every frame)",
    )
    parser.add_argument(
        "--reset-parts",
        action="store_true",
        help="Discard any existing episode part checkpoints and recompute from scratch",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Try to get reward_model_path from parquet metadata if not provided
    reward_model_path = args.reward_model_path
    if reward_model_path is None:
        # Load dataset to find parquet path
        temp_dataset = LeRobotDataset(args.dataset_repo_id, download_videos=False)
        parquet_path = Path(temp_dataset.root) / "sarm_progress.parquet"
        reward_model_path = get_reward_model_path_from_parquet(parquet_path)
        if reward_model_path:
            logging.info(f"Using reward model from parquet metadata: {reward_model_path}")
        else:
            raise ValueError(
                "--reward-model-path is required (no existing parquet with model metadata found)"
            )

    # Handle visualize-only mode
    if args.visualize_only:
        dataset, reward_model, preprocess = load_sarm_resources(
            args.dataset_repo_id, reward_model_path, args.device
        )
        explicit_viz_episodes = parse_visualization_episode_indices(
            args.visualize_episode_indices, dataset.num_episodes
        ) or []
        ranking_top_episodes, ranking_bottom_episodes = parse_visualization_ranking_groups(
            ranking_csv=args.visualize_ranking_csv,
            top_k=args.visualize_top_k,
            bottom_k=args.visualize_bottom_k,
            num_episodes=dataset.num_episodes,
        )

        ranking_mode = bool(ranking_top_episodes or ranking_bottom_episodes)
        if ranking_mode:
            if explicit_viz_episodes:
                manual_dir = Path(args.output_dir) / "manual"
                logging.info(
                    "Visualization-only mode: visualizing explicit episodes %s to %s",
                    explicit_viz_episodes,
                    manual_dir,
                )
                visualize_sarm_predictions(
                    dataset=dataset,
                    reward_model=reward_model,
                    preprocess=preprocess,
                    episode_indices=explicit_viz_episodes,
                    head_mode=args.head_mode,
                    output_dir=manual_dir,
                    stride=args.stride,
                )

            if ranking_top_episodes:
                top_dir = Path(args.output_dir) / "top"
                logging.info(
                    "Visualization-only mode: visualizing top-ranked episodes %s to %s",
                    ranking_top_episodes,
                    top_dir,
                )
                visualize_sarm_predictions(
                    dataset=dataset,
                    reward_model=reward_model,
                    preprocess=preprocess,
                    episode_indices=ranking_top_episodes,
                    head_mode=args.head_mode,
                    output_dir=top_dir,
                    stride=args.stride,
                )

            if ranking_bottom_episodes:
                bottom_dir = Path(args.output_dir) / "bottom"
                logging.info(
                    "Visualization-only mode: visualizing bottom-ranked episodes %s to %s",
                    ranking_bottom_episodes,
                    bottom_dir,
                )
                visualize_sarm_predictions(
                    dataset=dataset,
                    reward_model=reward_model,
                    preprocess=preprocess,
                    episode_indices=ranking_bottom_episodes,
                    head_mode=args.head_mode,
                    output_dir=bottom_dir,
                    stride=args.stride,
                )
        else:
            viz_episodes = explicit_viz_episodes if explicit_viz_episodes else None
            if viz_episodes is None:
                logging.info(f"Visualization-only mode: visualizing {args.num_visualizations} episodes")
                viz_episodes = list(range(min(args.num_visualizations, dataset.num_episodes)))
            else:
                logging.info(f"Visualization-only mode: visualizing requested episodes {viz_episodes}")
            visualize_sarm_predictions(
                dataset=dataset,
                reward_model=reward_model,
                preprocess=preprocess,
                episode_indices=viz_episodes,
                head_mode=args.head_mode,
                output_dir=Path(args.output_dir),
                stride=args.stride,
            )

        print(f"\nVisualizations saved to: {Path(args.output_dir).absolute()}")
        return

    # Full RABC computation (compute_sarm_progress loads model/dataset itself)
    visualize_episode_indices = resolve_visualization_episode_selection(
        explicit_episode_indices_raw=args.visualize_episode_indices,
        ranking_csv=args.visualize_ranking_csv,
        ranking_top_k=args.visualize_top_k,
        ranking_bottom_k=args.visualize_bottom_k,
    )
    output_path = compute_sarm_progress(
        dataset_repo_id=args.dataset_repo_id,
        reward_model_path=reward_model_path,
        output_path=args.output_path,
        head_mode=args.head_mode,
        device=args.device,
        num_visualizations=args.num_visualizations,
        output_dir=args.output_dir,
        stride=args.stride,
        reset_parts=args.reset_parts,
        visualize_episode_indices=visualize_episode_indices,
    )

    print(f"\nSARM progress values saved to: {output_path}")

    # Upload to Hub if requested
    if args.push_to_hub:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

        api = HfApi()
        hub_path = "sarm_progress.parquet"

        try:
            print(f"\nUploading to Hub: {args.dataset_repo_id}/{hub_path}")
            api.upload_file(
                path_or_fileobj=str(output_path),
                path_in_repo=hub_path,
                repo_id=args.dataset_repo_id,
                repo_type="dataset",
            )
            print(
                f"Successfully uploaded to: https://huggingface.co/datasets/{args.dataset_repo_id}/blob/main/{hub_path}"
            )

            print("\nTo use in training, add to your config:")
            print("  use_rabc: true")
            print(f"  rabc_progress_path: hf://datasets/{args.dataset_repo_id}/{hub_path}")
            print("  rabc_head_mode: sparse  # or dense")
        except RepositoryNotFoundError as e:
            logging.error("Upload skipped: dataset repo not found (%s)", args.dataset_repo_id)
            logging.error("Hub error: %s", e)
            print("\nLocal progress file is already saved and can be used directly.")
            print("To use in training, add to your config:")
            print("  use_rabc: true")
            print(f"  rabc_progress_path: {output_path}")
            print("  rabc_head_mode: sparse  # or dense")
        except HfHubHTTPError as e:
            logging.error("Upload failed due to Hub HTTP error: %s", e)
            print("\nLocal progress file is already saved and can be used directly.")
            print("To use in training, add to your config:")
            print("  use_rabc: true")
            print(f"  rabc_progress_path: {output_path}")
            print("  rabc_head_mode: sparse  # or dense")
    else:
        print("\nTo use in training, add to your config:")
        print("  use_rabc: true")
        print(f"  rabc_progress_path: {output_path}")
        print("  rabc_head_mode: sparse  # or dense")


if __name__ == "__main__":
    main()
