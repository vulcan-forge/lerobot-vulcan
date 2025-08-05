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
import logging
from pathlib import Path
from pprint import pformat
import shutil

import torch
import pandas as pd
import cv2  # Add this import at the top

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.transforms import ImageTransforms

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "next.reward" and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == "action" and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith("observation.") and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
        )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


def combine_datasets(datasets: list[LeRobotDataset], root: str, repo_id: str) -> LeRobotDataset:
    """
    Combines multiple LeRobotDatasets into a single dataset.
    Uses the chunk size and structure from the first dataset as the template.
    """
    if not datasets:
        raise ValueError("No datasets provided to combine")

    # Validate that all datasets have compatible configurations
    first_dataset = datasets[0]
    for dataset in datasets[1:]:
        if dataset.fps != first_dataset.fps:
            raise ValueError(f"Incompatible FPS: {dataset.fps} vs {first_dataset.fps}")
        if dataset.features != first_dataset.features:
            raise ValueError("Datasets have incompatible features")
        if dataset.video_backend != first_dataset.video_backend:
            raise ValueError("Datasets have incompatible video backends")

    # Delete existing dataset directory if it exists
    output_root = Path(root) if root else HF_LEROBOT_HOME / repo_id
    if output_root.exists():
        logging.info(f"Removing existing dataset directory: {output_root}")
        shutil.rmtree(output_root)

    # Create the combined dataset using create()
    combined_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=first_dataset.fps,
        root=root,
        robot_type=first_dataset.meta.robot_type,
        features=first_dataset.features,
        use_videos=len(first_dataset.meta.video_keys) > 0,
        video_backend=first_dataset.video_backend
    )

    # Use first dataset's chunk size as the template
    chunk_size = first_dataset.meta.chunks_size
    total_episodes = sum(dataset.meta.total_episodes for dataset in datasets)
    total_chunks = (total_episodes + chunk_size - 1) // chunk_size

    # Initialize offsets
    episode_offset = 0
    task_content_to_index = {}  # Map task content to its index

    # First, combine all tasks from all datasets, deduplicating by task content
    for dataset in datasets:
        for task_idx, task in dataset.meta.tasks.items():
            if task not in task_content_to_index:
                # If this is a new task, add it with the next available index
                task_content_to_index[task] = len(task_content_to_index)
            # Update the task index in the dataset's episodes
            for ep_idx in dataset.meta.episodes:
                if dataset.meta.episodes[ep_idx].get("task_index") == task_idx:
                    dataset.meta.episodes[ep_idx]["task_index"] = task_content_to_index[task]

    # Add all unique tasks to the combined dataset
    combined_dataset.meta.tasks = {idx: task for task, idx in task_content_to_index.items()}

    # Then combine episodes and stats
    for dataset_idx, dataset in enumerate(datasets):
        logging.info(f"Processing dataset {dataset_idx + 1} with root: {dataset.meta.root}")

        # Get the episode indices for this dataset
        dataset_episodes = sorted(dataset.meta.episodes.keys())

        for ep_idx in dataset_episodes:
            # Calculate the new episode index based on which dataset we're processing
            new_ep_idx = ep_idx + (dataset_idx * len(dataset_episodes))

            # Get the original episode data
            episode = dataset.meta.episodes[ep_idx].copy()
            episode_stats = dataset.meta.episodes_stats[ep_idx].copy()

            # Update episode with new task indices and episode index
            if "task_index" in episode:
                episode["task_index"] = task_content_to_index[dataset.meta.tasks[episode["task_index"]]]
            episode["episode_index"] = new_ep_idx

            # Calculate new chunk and episode numbers
            new_chunk = new_ep_idx // chunk_size

            # Copy data files with new sequential naming
            src_data_path = dataset.meta.root / dataset.meta.get_data_file_path(ep_idx)
            dst_data_path = combined_dataset.meta.root / f"data/chunk-{new_chunk:03d}/episode_{new_ep_idx:06d}.parquet"

            if src_data_path.exists():
                dst_data_path.parent.mkdir(parents=True, exist_ok=True)

                # Read the parquet file
                df = pd.read_parquet(src_data_path)

                # Update frame indices to match the new episode indices
                df['frame_index'] = df['frame_index'] + (dataset_idx * len(dataset_episodes))

                # Update episode indices
                df['episode_index'] = new_ep_idx

                # Write the updated data back to the new parquet file
                df.to_parquet(dst_data_path)
            else:
                logging.warning(f"Missing data file for episode {ep_idx} in dataset {dataset.meta.root}")

            # Copy video files if they exist
            for vid_key in dataset.meta.video_keys:
                src_video_path = dataset.meta.root / dataset.meta.get_video_file_path(ep_idx, vid_key)
                dst_video_path = combined_dataset.meta.root / f"videos/chunk-{new_chunk:03d}/{vid_key}/episode_{new_ep_idx:06d}.mp4"

                if src_video_path.exists():
                    dst_video_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_video_path, dst_video_path)
                else:
                    logging.warning(f"Missing video file for episode {ep_idx}, camera {vid_key} in dataset {dataset.meta.root}")

            # Add episode and stats to combined dataset
            combined_dataset.meta.episodes[new_ep_idx] = episode
            combined_dataset.meta.episodes_stats[new_ep_idx] = episode_stats

    # Calculate combined statistics
    total_frames = sum(dataset.meta.info["total_frames"] for dataset in datasets)
    total_tasks = len(task_content_to_index)  # This is the number of unique tasks
    total_videos = sum(dataset.meta.info["total_videos"] for dataset in datasets)

    # Update metadata with correct statistics
    combined_dataset.meta.info.update({
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": chunk_size,
        "splits": {
            "train": f"0:{total_episodes}"  # Update splits to cover all episodes
        }
    })

    # Save all metadata files
    write_info(combined_dataset.meta.info, combined_dataset.meta.root)

    # Save tasks.jsonl
    tasks_data = [{"task_index": idx, "task": task} for idx, task in combined_dataset.meta.tasks.items()]
    write_jsonlines(tasks_data, combined_dataset.meta.root / "meta" / "tasks.jsonl")

    # Save episodes.jsonl
    episodes_data = list(combined_dataset.meta.episodes.values())
    write_jsonlines(episodes_data, combined_dataset.meta.root / "meta" / "episodes.jsonl")

    # Save episodes_stats.jsonl
    episodes_stats_data = [
        {"episode_index": idx, "stats": serialize_dict(stats)}
        for idx, stats in combined_dataset.meta.episodes_stats.items()
    ]
    write_jsonlines(episodes_stats_data, combined_dataset.meta.root / "meta" / "episodes_stats.jsonl")

    # Verify that data files exist before loading
    data_dir = combined_dataset.meta.root / "data"
    if not data_dir.exists():
        raise ValueError(f"Data directory {data_dir} does not exist")

    parquet_files = list(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    # Load the combined dataset
    combined_dataset.hf_dataset = combined_dataset.load_hf_dataset()
    combined_dataset.episode_data_index = get_episode_data_index(
        combined_dataset.meta.episodes,
        list(combined_dataset.meta.episodes.keys())
    )

    return combined_dataset
