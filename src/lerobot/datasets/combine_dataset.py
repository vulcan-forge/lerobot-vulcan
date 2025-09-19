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

import argparse
import logging
import shutil
import gc
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd
import numpy as np
from huggingface_hub import HfApi

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    create_lerobot_dataset_card,
    write_info,
    write_stats,
    write_tasks,
    write_episodes,
    load_info,
    load_stats,
    load_tasks,
    load_episodes,
    combine_feature_dicts,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_PATH,
    DEFAULT_EPISODES_PATH,
    update_chunk_file_indices,
    get_parquet_file_size_in_mb,
    get_video_size_in_mb,
    flatten_dict,
    unflatten_dict,
    to_parquet_with_hf_images,
)
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.video_utils import concatenate_video_files
from lerobot.utils.utils import init_logging


def combine_v3_datasets(
    dataset_paths: List[Path],
    output_path: Path,
    output_repo_id: str,
    data_file_size_in_mb: int = None,
    video_file_size_in_mb: int = None,
    chunk_size: int = None,
    push_to_hub: bool = False,
    private: bool = False,
    tags: List[str] = None,
    license: str = None,
) -> LeRobotDataset:
    """
    Combine multiple LeRobot v3.0 datasets into a single dataset.

    Args:
        dataset_paths: List of paths to the datasets to combine
        output_path: Path where the combined dataset will be saved
        output_repo_id: Repository ID for the combined dataset
        data_file_size_in_mb: Maximum size for data files in MB
        video_file_size_in_mb: Maximum size for video files in MB
        chunk_size: Maximum number of files per chunk
        push_to_hub: Whether to push the dataset to Hugging Face Hub
        private: Whether the repository should be private
        tags: List of tags for the dataset
        license: License for the dataset

    Returns:
        The combined LeRobotDataset
    """
    if not dataset_paths:
        raise ValueError("No dataset paths provided")

    if data_file_size_in_mb is None:
        data_file_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    if video_file_size_in_mb is None:
        video_file_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    logging.info(f"Combining {len(dataset_paths)} datasets...")

    # Load all datasets
    datasets = []
    for path in dataset_paths:
        logging.info(f"Loading dataset from {path}")
        dataset = LeRobotDataset(repo_id="local", root=path)
        datasets.append(dataset)

    # Validate compatibility
    first_dataset = datasets[0]
    for i, dataset in enumerate(datasets[1:], 1):
        if dataset.fps != first_dataset.fps:
            raise ValueError(f"Dataset {i+1} has incompatible FPS: {dataset.fps} vs {first_dataset.fps}")
        if dataset.features != first_dataset.features:
            raise ValueError(f"Dataset {i+1} has incompatible features")
        if dataset.video_backend != first_dataset.video_backend:
            raise ValueError(f"Dataset {i+1} has incompatible video backend")

    # Create output directory
    if output_path.exists():
        logging.info(f"Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create combined dataset metadata
    combined_meta = LeRobotDatasetMetadata.create(
        repo_id=output_repo_id,
        fps=first_dataset.fps,
        features=first_dataset.features,
        robot_type=first_dataset.meta.robot_type,
        root=output_path,
        use_videos=len(first_dataset.meta.video_keys) > 0,
    )

    # Update chunk settings
    combined_meta.update_chunk_settings(
        chunks_size=chunk_size,
        data_files_size_in_mb=data_file_size_in_mb,
        video_files_size_in_mb=video_file_size_in_mb,
    )

    # Combine tasks
    all_tasks = {}
    task_content_to_index = {}
    for dataset in datasets:
        # tasks is a DataFrame with task strings as index and task_index as column
        for task_string in dataset.meta.tasks.index.tolist():
            if task_string not in task_content_to_index:
                new_idx = len(task_content_to_index)
                task_content_to_index[task_string] = new_idx
                all_tasks[new_idx] = task_string

    # Write tasks
    tasks_df = pd.DataFrame({"task_index": list(all_tasks.keys())}, index=list(all_tasks.values()))
    write_tasks(tasks_df, output_path)

    # Process each dataset with proper chunking
    episode_offset = 0
    frame_offset = 0
    video_timestamp_offset = 0.0  # Track accumulated video duration
    all_episodes_data = []
    all_stats = []

    # Initialize chunking state
    data_chunk_idx, data_file_idx = 0, 0
    video_chunk_idx, video_file_idx = 0, 0
    episodes_chunk_idx, episodes_file_idx = 0, 0

    for dataset_idx, dataset in enumerate(datasets):
        logging.info(f"Processing dataset {dataset_idx + 1}/{len(datasets)}")

        # Load episodes metadata
        episodes_dataset = load_episodes(dataset.meta.root)

        for episode_data in episodes_dataset:
            # Create new episode data
            new_episode_data = dict(episode_data)

            # Update episode index - FIXED: Use consecutive episode indices
            old_episode_idx = episode_data['episode_index']
            new_episode_idx = episode_offset  # Use consecutive indexing
            new_episode_data['episode_index'] = new_episode_idx

            # Update task index - FIXED: Get task string from tasks DataFrame
            if 'task_index' in new_episode_data:
                old_task_idx = new_episode_data['task_index']
                old_task_string = dataset.meta.tasks.iloc[old_task_idx].name
                new_episode_data['task_index'] = task_content_to_index[old_task_string]

            # Update data indices
            new_episode_data['dataset_from_index'] = frame_offset
            new_episode_data['dataset_to_index'] = frame_offset + episode_data['length']

            # Update chunk and file indices for data
            new_episode_data['data/chunk_index'] = data_chunk_idx
            new_episode_data['data/file_index'] = data_file_idx

            # Update video indices if videos exist
            for video_key in dataset.meta.video_keys:
                new_episode_data[f'videos/{video_key}/chunk_index'] = video_chunk_idx
                new_episode_data[f'videos/{video_key}/file_index'] = video_file_idx

                # Update video timestamps to account for accumulated video duration
                if f'videos/{video_key}/from_timestamp' in new_episode_data:
                    new_episode_data[f'videos/{video_key}/from_timestamp'] += video_timestamp_offset
                if f'videos/{video_key}/to_timestamp' in new_episode_data:
                    new_episode_data[f'videos/{video_key}/to_timestamp'] += video_timestamp_offset

            # Update meta/episodes indices
            new_episode_data['meta/episodes/chunk_index'] = episodes_chunk_idx
            new_episode_data['meta/episodes/file_index'] = episodes_file_idx

            all_episodes_data.append(new_episode_data)

            # Load and update episode stats
            episode_stats = _load_episode_stats(dataset.meta.root, old_episode_idx)
            if episode_stats:
                all_stats.append(episode_stats)

            episode_offset += 1
            frame_offset += episode_data['length']

        # Update video timestamp offset for next dataset
        if dataset.meta.video_keys:
            # Calculate total video duration for this dataset
            dataset_video_duration = 0.0
            for episode_data in episodes_dataset:
                for video_key in dataset.meta.video_keys:
                    if f'videos/{video_key}/to_timestamp' in episode_data:
                        dataset_video_duration = max(dataset_video_duration, episode_data[f'videos/{video_key}/to_timestamp'])
            video_timestamp_offset += dataset_video_duration

    # Write episodes metadata with proper chunking
    _write_episodes_with_chunking(all_episodes_data, output_path, episodes_chunk_idx, episodes_file_idx, chunk_size)

    # Combine and write stats - FIXED: Always write stats
    if all_stats:
        combined_stats = aggregate_stats(all_stats)
        write_stats(combined_stats, output_path)
    else:
        # Create empty stats if no episode stats were loaded
        empty_stats = {}
        write_stats(empty_stats, output_path)

    # Copy and combine data files with proper chunking - FIXED: Pass episode offset mapping
    _combine_data_files_with_chunking(datasets, output_path, chunk_size, data_file_size_in_mb, first_dataset.meta.image_keys)

    # Copy and combine video files with proper chunking
    if first_dataset.meta.video_keys:
        _combine_video_files_with_chunking(datasets, output_path, chunk_size, video_file_size_in_mb)

    # Update final info
    combined_meta.info.update({
        "total_episodes": episode_offset,
        "total_frames": frame_offset,
        "total_tasks": len(all_tasks),
    })
    write_info(combined_meta.info, output_path)

    # Create dataset card
    if push_to_hub:
        card = create_lerobot_dataset_card(
            tags=tags,
            dataset_info=combined_meta.info,
            license=license,
        )
        card.push_to_hub(output_repo_id, repo_type="dataset")

    # Load and return the combined dataset
    combined_dataset = LeRobotDataset(
        repo_id=output_repo_id,
        root=output_path,
    )

    if push_to_hub:
        logging.info(f"Pushing dataset to Hugging Face Hub: {output_repo_id}")
        combined_dataset.push_to_hub()

    return combined_dataset


def _write_episodes_with_chunking(
    episodes_data: List[Dict],
    output_path: Path,
    chunk_idx: int,
    file_idx: int,
    chunk_size: int
) -> None:
    """Write episodes metadata with proper chunking based on file size limits."""
    logging.info("Writing episodes metadata with proper chunking...")

    episodes_df = pd.DataFrame(episodes_data)
    episodes_dataset = episodes_df.to_dict('records')
    from datasets import Dataset
    episodes_hf_dataset = Dataset.from_list(episodes_dataset)

    # Write episodes with size-aware chunking
    _append_or_create_parquet_file(
        df=episodes_df,
        src_path=None,  # No source path for episodes
        idx={"chunk": chunk_idx, "file": file_idx},
        max_mb=DEFAULT_DATA_FILE_SIZE_IN_MB,
        chunk_size=chunk_size,
        default_path=DEFAULT_EPISODES_PATH,
        contains_images=False,
        aggr_root=output_path,
    )


def _combine_data_files_with_chunking(
    datasets: List[LeRobotDataset],
    output_path: Path,
    chunk_size: int,
    data_file_size_in_mb: int,
    image_keys: List[str]
) -> None:
    """Combine data files with proper size-aware chunking."""
    logging.info("Combining data files with proper chunking...")

    chunk_idx, file_idx = 0, 0
    current_df = None
    episode_offset = 0

    for dataset in datasets:
        data_dir = dataset.meta.root / "data"
        for parquet_file in sorted(data_dir.rglob("*.parquet")):
            df = pd.read_parquet(parquet_file)

            # Update episode indices in the data - FIXED: Use consecutive episode indices
            if 'episode_index' in df.columns:
                # Map old episode indices to new consecutive indices
                unique_old_indices = sorted(df['episode_index'].unique())
                episode_mapping = {old_idx: episode_offset + i for i, old_idx in enumerate(unique_old_indices)}
                df['episode_index'] = df['episode_index'].map(episode_mapping)

            if current_df is None:
                current_df = df
            else:
                current_df = pd.concat([current_df, df], ignore_index=True)

            # Check if we need to write current file and start a new one
            current_size_mb = get_parquet_file_size_in_mb(parquet_file)  # Estimate based on source
            if current_size_mb >= data_file_size_in_mb:
                # Write current file
                _write_data_file(current_df, output_path, chunk_idx, file_idx, image_keys)

                # Move to next file
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunk_size)
                current_df = None

        # Update episode offset for next dataset
        episode_offset += dataset.meta.total_episodes

    # Write remaining data if any
    if current_df is not None:
        _write_data_file(current_df, output_path, chunk_idx, file_idx, image_keys)


def _write_data_file(df: pd.DataFrame, output_path: Path, chunk_idx: int, file_idx: int, image_keys: List[str]) -> None:
    """Write a data file with proper handling for images."""
    path = output_path / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(image_keys) > 0:
        to_parquet_with_hf_images(df, path)
    else:
        df.to_parquet(path, index=False)


def _combine_video_files_with_chunking(
    datasets: List[LeRobotDataset],
    output_path: Path,
    chunk_size: int,
    video_file_size_in_mb: int
) -> None:
    """Combine video files with proper size-aware chunking."""
    logging.info("Combining video files with proper chunking...")

    # Get all video keys from the first dataset
    video_keys = datasets[0].meta.video_keys

    for video_key in video_keys:
        chunk_idx, file_idx = 0, 0
        current_video_files = []
        current_size_mb = 0.0

        for dataset in datasets:
            video_dir = dataset.meta.root / "videos" / video_key
            if video_dir.exists():
                for video_file in sorted(video_dir.rglob("*.mp4")):
                    video_size_mb = get_video_size_in_mb(video_file)

                    # Check if adding this video would exceed size limit
                    if current_size_mb + video_size_mb >= video_file_size_in_mb and current_video_files:
                        # Write current batch
                        _write_video_file(current_video_files, output_path, video_key, chunk_idx, file_idx)

                        # Move to next file
                        chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunk_size)
                        current_video_files = []
                        current_size_mb = 0.0

                    current_video_files.append(video_file)
                    current_size_mb += video_size_mb

        # Write remaining videos
        if current_video_files:
            _write_video_file(current_video_files, output_path, video_key, chunk_idx, file_idx)


def _write_video_file(video_files: List[Path], output_path: Path, video_key: str, chunk_idx: int, file_idx: int) -> None:
    """Write a concatenated video file."""
    if not video_files:
        return

    output_path_video = output_path / DEFAULT_VIDEO_PATH.format(
        video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
    )
    output_path_video.parent.mkdir(parents=True, exist_ok=True)

    if len(video_files) == 1:
        shutil.copy2(video_files[0], output_path_video)
    else:
        concatenate_video_files(video_files, output_path_video)


def _append_or_create_parquet_file(
    df: pd.DataFrame,
    src_path: Path,
    idx: Dict[str, int],
    max_mb: float,
    chunk_size: int,
    default_path: str,
    contains_images: bool = False,
    aggr_root: Path = None,
) -> Dict[str, int]:
    """Appends data to an existing parquet file or creates a new one based on size constraints.

    This is adapted from the LeRobot aggregate.py implementation.
    """
    dst_path = aggr_root / default_path.format(chunk_index=idx["chunk"], file_index=idx["file"])

    if not dst_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if contains_images:
            to_parquet_with_hf_images(df, dst_path)
        else:
            df.to_parquet(dst_path, index=False)
        return idx

    # Check if we need a new file due to size constraints
    if src_path and src_path.exists():
        src_size = get_parquet_file_size_in_mb(src_path)
        dst_size = get_parquet_file_size_in_mb(dst_path)

        if dst_size + src_size >= max_mb:
            # Size limit would be exceeded, create new file
            idx["chunk"], idx["file"] = update_chunk_file_indices(idx["chunk"], idx["file"], chunk_size)
            new_path = aggr_root / default_path.format(chunk_index=idx["chunk"], file_index=idx["file"])
            new_path.parent.mkdir(parents=True, exist_ok=True)
            final_df = df
            target_path = new_path
        else:
            # Append to existing file
            existing_df = pd.read_parquet(dst_path)
            final_df = pd.concat([existing_df, df], ignore_index=True)
            target_path = dst_path
    else:
        # No source path, just append to existing
        existing_df = pd.read_parquet(dst_path)
        final_df = pd.concat([existing_df, df], ignore_index=True)
        target_path = dst_path

    # Write the final dataframe
    if contains_images:
        to_parquet_with_hf_images(final_df, target_path)
    else:
        final_df.to_parquet(target_path, index=False)

    return idx


def _load_episode_stats(dataset_root: Path, episode_idx: int) -> Dict[str, Any]:
    """Load episode statistics from the dataset."""
    # This would need to be implemented based on how stats are stored
    # in the v3.0 format - for now return empty dict
    return {}


def main():
    parser = argparse.ArgumentParser(description="Combine multiple LeRobot v3.0 datasets into a single dataset")
    parser.add_argument(
        "--dataset_paths",
        type=str,
        nargs="+",
        required=True,
        help="List of paths to the datasets to combine (e.g., `path/to/dataset1 path/to/dataset2`).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where the combined dataset will be saved.",
    )
    parser.add_argument(
        "--output_repo_id",
        type=str,
        required=True,
        help="Repository ID for the combined dataset (e.g., `local/combined_dataset`).",
    )
    parser.add_argument(
        "--data_file_size_in_mb",
        type=int,
        default=DEFAULT_DATA_FILE_SIZE_IN_MB,
        help="Maximum size for data files in MB.",
    )
    parser.add_argument(
        "--video_file_size_in_mb",
        type=int,
        default=DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        help="Maximum size for video files in MB.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum number of files per chunk.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
        help="Upload to Hugging Face hub. Defaults to False for local dataset combination.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="If set, the repository on the Hub will be private",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        help="List of tags to apply to the dataset on the Hub",
    )
    parser.add_argument(
        "--license",
        type=str,
        default=None,
        help="License to use for the dataset on the Hub"
    )

    args = parser.parse_args()

    # Convert paths to Path objects
    dataset_paths = [Path(p) for p in args.dataset_paths]
    output_path = Path(args.output_path)

    # Run the combination
    combined_dataset = combine_v3_datasets(
        dataset_paths=dataset_paths,
        output_path=output_path,
        output_repo_id=args.output_repo_id,
        data_file_size_in_mb=args.data_file_size_in_mb,
        video_file_size_in_mb=args.video_file_size_in_mb,
        chunk_size=args.chunk_size,
        push_to_hub=args.push_to_hub,
        private=args.private,
        tags=args.tags,
        license=args.license,
    )

    logging.info(f"Successfully combined datasets into {args.output_repo_id}")
    logging.info(f"Total episodes: {combined_dataset.meta.total_episodes}")
    logging.info(f"Total frames: {combined_dataset.meta.total_frames}")


if __name__ == "__main__":
    init_logging()
    main()
