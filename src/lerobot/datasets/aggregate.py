#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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
import shutil
from pathlib import Path

import pandas as pd
import tqdm

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    get_file_size_in_mb,
    get_parquet_file_size_in_mb,
    to_parquet_with_hf_images,
    update_chunk_file_indices,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s


def validate_all_metadata(all_metadata: list[LeRobotDatasetMetadata]):
    """Validates that all dataset metadata have consistent properties.

    Ensures all datasets have the same fps, robot_type, and features to guarantee
    compatibility when aggregating them into a single dataset.

    Args:
        all_metadata: List of LeRobotDatasetMetadata objects to validate.

    Returns:
        tuple: A tuple containing (fps, robot_type, features) from the first metadata.

    Raises:
        ValueError: If any metadata has different fps, robot_type, or features
                   than the first metadata in the list.
    """

    fps = all_metadata[0].fps
    robot_type = all_metadata[0].robot_type
    features = all_metadata[0].features

    for meta in tqdm.tqdm(all_metadata, desc="Validate all meta data"):
        if fps != meta.fps:
            raise ValueError(f"Same fps is expected, but got fps={meta.fps} instead of {fps}.")
        if robot_type != meta.robot_type:
            raise ValueError(
                f"Same robot_type is expected, but got robot_type={meta.robot_type} instead of {robot_type}."
            )
        if features != meta.features:
            raise ValueError(
                f"Same features is expected, but got features={meta.features} instead of {features}."
            )

    return fps, robot_type, features


def update_data_df(df, src_meta, dst_meta):
    """Updates a data DataFrame with new indices and task mappings for aggregation.

    Adjusts episode indices, frame indices, and task indices to account for
    previously aggregated data in the destination dataset.

    Args:
        df: DataFrame containing the data to be updated.
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.

    Returns:
        pd.DataFrame: Updated DataFrame with adjusted indices.
    """

    df["episode_index"] = df["episode_index"] + dst_meta.info["total_episodes"]
    df["index"] = df["index"] + dst_meta.info["total_frames"]

    src_task_names = src_meta.tasks.index.take(df["task_index"].to_numpy())
    df["task_index"] = dst_meta.tasks.loc[src_task_names, "task_index"].to_numpy()

    return df


def update_meta_data(
    df,
    dst_meta,
    meta_dst_idx,
    data_src_to_dst,
    videos_idx,
):
    """Updates metadata DataFrame with new chunk, file, and timestamp indices.

    Adjusts all indices and timestamps to account for previously aggregated
    data and videos in the destination dataset.

    Args:
        df: DataFrame containing the metadata to be updated.
        dst_meta: Destination dataset metadata.
        meta_dst_idx: Destination metadata chunk and file indices for this parquet file.
        data_src_to_dst: Mapping from source data chunk/file pairs to destination chunk/file pairs.
        videos_idx: Dictionary containing current video indices and timestamps.

    Returns:
        pd.DataFrame: Updated DataFrame with adjusted indices and timestamps.
    """

    df["meta/episodes/chunk_index"] = meta_dst_idx["chunk"]
    df["meta/episodes/file_index"] = meta_dst_idx["file"]

    df["_orig_data_chunk"] = df["data/chunk_index"].copy()
    df["_orig_data_file"] = df["data/file_index"].copy()
    for idx in df.index:
        src_key = (df.at[idx, "_orig_data_chunk"], df.at[idx, "_orig_data_file"])
        dst_chunk_idx, dst_file_idx = data_src_to_dst[src_key]
        df.at[idx, "data/chunk_index"] = dst_chunk_idx
        df.at[idx, "data/file_index"] = dst_file_idx
    df = df.drop(columns=["_orig_data_chunk", "_orig_data_file"])

    for key, video_idx in videos_idx.items():
        # Store original video file indices before updating
        orig_chunk_col = f"videos/{key}/chunk_index"
        orig_file_col = f"videos/{key}/file_index"
        df["_orig_chunk"] = df[orig_chunk_col].copy()
        df["_orig_file"] = df[orig_file_col].copy()

        # Apply the destination file mapping and timestamp offset for each source file.
        src_to_dst = video_idx["src_to_dst"]
        for idx in df.index:
            src_key = (df.at[idx, "_orig_chunk"], df.at[idx, "_orig_file"])
            dst_chunk_idx, dst_file_idx, offset = src_to_dst[src_key]
            df.at[idx, orig_chunk_col] = dst_chunk_idx
            df.at[idx, orig_file_col] = dst_file_idx
            df.at[idx, f"videos/{key}/from_timestamp"] += offset
            df.at[idx, f"videos/{key}/to_timestamp"] += offset

        # Clean up temporary columns
        df = df.drop(columns=["_orig_chunk", "_orig_file"])

    df["dataset_from_index"] = df["dataset_from_index"] + dst_meta.info["total_frames"]
    df["dataset_to_index"] = df["dataset_to_index"] + dst_meta.info["total_frames"]
    df["episode_index"] = df["episode_index"] + dst_meta.info["total_episodes"]

    return df


def aggregate_datasets(
    repo_ids: list[str],
    aggr_repo_id: str,
    roots: list[Path] | None = None,
    aggr_root: Path | None = None,
    data_files_size_in_mb: float | None = None,
    video_files_size_in_mb: float | None = None,
    chunk_size: int | None = None,
):
    """Aggregates multiple LeRobot datasets into a single unified dataset.

    This is the main function that orchestrates the aggregation process by:
    1. Loading and validating all source dataset metadata
    2. Creating a new destination dataset with unified tasks
    3. Aggregating videos, data, and metadata from all source datasets
    4. Finalizing the aggregated dataset with proper statistics

    Args:
        repo_ids: List of repository IDs for the datasets to aggregate.
        aggr_repo_id: Repository ID for the aggregated output dataset.
        roots: Optional list of root paths for the source datasets.
        aggr_root: Optional root path for the aggregated dataset.
        data_files_size_in_mb: Maximum size for data files in MB (defaults to DEFAULT_DATA_FILE_SIZE_IN_MB)
        video_files_size_in_mb: Maximum size for video files in MB (defaults to DEFAULT_VIDEO_FILE_SIZE_IN_MB)
        chunk_size: Maximum number of files per chunk (defaults to DEFAULT_CHUNK_SIZE)
    """
    logging.info("Start aggregate_datasets")

    if data_files_size_in_mb is None:
        data_files_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    if video_files_size_in_mb is None:
        video_files_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    all_metadata = (
        [LeRobotDatasetMetadata(repo_id) for repo_id in repo_ids]
        if roots is None
        else [
            LeRobotDatasetMetadata(repo_id, root=root) for repo_id, root in zip(repo_ids, roots, strict=False)
        ]
    )
    fps, robot_type, features = validate_all_metadata(all_metadata)
    video_keys = [key for key in features if features[key]["dtype"] == "video"]

    dst_meta = LeRobotDatasetMetadata.create(
        repo_id=aggr_repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=aggr_root,
        use_videos=len(video_keys) > 0,
        chunks_size=chunk_size,
        data_files_size_in_mb=data_files_size_in_mb,
        video_files_size_in_mb=video_files_size_in_mb,
    )

    logging.info("Find all tasks")
    unique_tasks = pd.concat([m.tasks for m in all_metadata]).index.unique()
    dst_meta.tasks = pd.DataFrame({"task_index": range(len(unique_tasks))}, index=unique_tasks)

    meta_idx = {"chunk": 0, "file": 0}
    data_idx = {"chunk": 0, "file": 0, "src_to_dst": {}}
    videos_idx = {
        key: {"chunk": 0, "file": 0, "episode_duration": 0, "episode_offset": 0, "src_to_dst": {}}
        for key in video_keys
    }

    dst_meta.episodes = {}

    for src_meta in tqdm.tqdm(all_metadata, desc="Copy data and videos"):
        videos_idx = aggregate_videos(src_meta, dst_meta, videos_idx, video_files_size_in_mb, chunk_size)
        data_idx = aggregate_data(src_meta, dst_meta, data_idx, data_files_size_in_mb, chunk_size)
        meta_idx = aggregate_metadata(src_meta, dst_meta, meta_idx, data_idx, videos_idx)

        dst_meta.info["total_episodes"] += src_meta.total_episodes
        dst_meta.info["total_frames"] += src_meta.total_frames

    finalize_aggregation(dst_meta, all_metadata)
    logging.info("Aggregation complete.")


def aggregate_videos(src_meta, dst_meta, videos_idx, video_files_size_in_mb, chunk_size):
    """Aggregates video chunks from a source dataset into the destination dataset.

    Handles video file concatenation and rotation based on file size limits.
    Creates new video files when size limits are exceeded.

    Args:
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.
        videos_idx: Dictionary tracking video chunk and file indices.
        video_files_size_in_mb: Maximum size for video files in MB (defaults to DEFAULT_VIDEO_FILE_SIZE_IN_MB)
        chunk_size: Maximum number of files per chunk (defaults to DEFAULT_CHUNK_SIZE)

    Returns:
        dict: Updated videos_idx with current chunk and file indices.
    """
    for key in videos_idx:
        videos_idx[key]["episode_duration"] = 0
        # Track the destination file and offset for each source (chunk, file) pair.
        videos_idx[key]["src_to_dst"] = {}

    for key, video_idx in videos_idx.items():
        unique_chunk_file_pairs = {
            (chunk, file)
            for chunk, file in zip(
                src_meta.episodes[f"videos/{key}/chunk_index"],
                src_meta.episodes[f"videos/{key}/file_index"],
                strict=False,
            )
        }
        unique_chunk_file_pairs = sorted(unique_chunk_file_pairs)

        chunk_idx = video_idx["chunk"]
        file_idx = video_idx["file"]
        current_offset = video_idx["episode_offset"]

        for src_chunk_idx, src_file_idx in unique_chunk_file_pairs:
            src_path = src_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=src_chunk_idx,
                file_index=src_file_idx,
            )

            dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )

            src_duration = get_video_duration_in_s(src_path)

            if not dst_path.exists():
                videos_idx[key]["src_to_dst"][(src_chunk_idx, src_file_idx)] = (
                    chunk_idx,
                    file_idx,
                    current_offset,
                )
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src_path), str(dst_path))

                current_offset += src_duration
                videos_idx[key]["episode_offset"] = current_offset
                videos_idx[key]["episode_duration"] += src_duration
                continue

            # Check file sizes before appending
            src_size = get_file_size_in_mb(src_path)
            dst_size = get_file_size_in_mb(dst_path)

            if dst_size + src_size >= video_files_size_in_mb:
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunk_size)
                dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                    video_key=key,
                    chunk_index=chunk_idx,
                    file_index=file_idx,
                )
                videos_idx[key]["src_to_dst"][(src_chunk_idx, src_file_idx)] = (
                    chunk_idx,
                    file_idx,
                    0,
                )
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src_path), str(dst_path))
                # Reset offset for next file
                current_offset = src_duration
            else:
                # Append to existing video file and preserve the current accumulated offset.
                videos_idx[key]["src_to_dst"][(src_chunk_idx, src_file_idx)] = (
                    chunk_idx,
                    file_idx,
                    current_offset,
                )
                concatenate_video_files(
                    [dst_path, src_path],
                    dst_path,
                )
                current_offset += src_duration

            videos_idx[key]["episode_offset"] = current_offset
            videos_idx[key]["episode_duration"] += src_duration

        videos_idx[key]["chunk"] = chunk_idx
        videos_idx[key]["file"] = file_idx

    return videos_idx


def aggregate_data(src_meta, dst_meta, data_idx, data_files_size_in_mb, chunk_size):
    """Aggregates data chunks from a source dataset into the destination dataset.

    Reads source data files, updates indices to match the aggregated dataset,
    and writes them to the destination with proper file rotation.

    Args:
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.
        data_idx: Dictionary tracking data chunk and file indices.

    Returns:
        dict: Updated data_idx with current chunk and file indices.
    """
    data_idx["src_to_dst"] = {}

    unique_chunk_file_ids = {
        (c, f)
        for c, f in zip(
            src_meta.episodes["data/chunk_index"], src_meta.episodes["data/file_index"], strict=False
        )
    }

    unique_chunk_file_ids = sorted(unique_chunk_file_ids)

    for src_chunk_idx, src_file_idx in unique_chunk_file_ids:
        src_path = src_meta.root / DEFAULT_DATA_PATH.format(
            chunk_index=src_chunk_idx, file_index=src_file_idx
        )
        target_chunk_idx, target_file_idx = get_parquet_target_indices(
            src_path,
            data_idx,
            data_files_size_in_mb,
            chunk_size,
            DEFAULT_DATA_PATH,
            dst_meta.root,
        )
        data_idx["src_to_dst"][(src_chunk_idx, src_file_idx)] = (target_chunk_idx, target_file_idx)

        df = pd.read_parquet(src_path)
        df = update_data_df(df, src_meta, dst_meta)

        write_parquet_to_target(
            df,
            target_chunk_idx,
            target_file_idx,
            DEFAULT_DATA_PATH,
            contains_images=len(dst_meta.image_keys) > 0,
            aggr_root=dst_meta.root,
        )
        data_idx["chunk"] = target_chunk_idx
        data_idx["file"] = target_file_idx

    return data_idx


def aggregate_metadata(src_meta, dst_meta, meta_idx, data_idx, videos_idx):
    """Aggregates metadata from a source dataset into the destination dataset.

    Reads source metadata files, updates all indices and timestamps,
    and writes them to the destination with proper file rotation.

    Args:
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.
        meta_idx: Dictionary tracking metadata chunk and file indices.
        data_idx: Dictionary tracking data chunk and file indices.
        videos_idx: Dictionary tracking video indices and timestamps.

    Returns:
        dict: Updated meta_idx with current chunk and file indices.
    """
    chunk_file_ids = {
        (c, f)
        for c, f in zip(
            src_meta.episodes["meta/episodes/chunk_index"],
            src_meta.episodes["meta/episodes/file_index"],
            strict=False,
        )
    }

    chunk_file_ids = sorted(chunk_file_ids)
    for chunk_idx, file_idx in chunk_file_ids:
        src_path = src_meta.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        target_chunk_idx, target_file_idx = get_parquet_target_indices(
            src_path,
            meta_idx,
            DEFAULT_DATA_FILE_SIZE_IN_MB,
            DEFAULT_CHUNK_SIZE,
            DEFAULT_EPISODES_PATH,
            dst_meta.root,
        )
        df = pd.read_parquet(src_path)
        df = update_meta_data(
            df,
            dst_meta,
            {"chunk": target_chunk_idx, "file": target_file_idx},
            data_idx["src_to_dst"],
            videos_idx,
        )

        write_parquet_to_target(
            df,
            target_chunk_idx,
            target_file_idx,
            DEFAULT_EPISODES_PATH,
            contains_images=False,
            aggr_root=dst_meta.root,
        )
        meta_idx["chunk"] = target_chunk_idx
        meta_idx["file"] = target_file_idx

    return meta_idx


def get_parquet_target_indices(
    src_path: Path,
    idx: dict[str, int],
    max_mb: float,
    chunk_size: int,
    default_path: str,
    aggr_root: Path = None,
):
    """Determines the destination parquet file for the next appended source file.

    Manages file rotation when size limits are exceeded to prevent individual files
    from becoming too large.

    Args:
        src_path: Path to the source file (used for size estimation).
        idx: Dictionary containing current 'chunk' and 'file' indices.
        max_mb: Maximum allowed file size in MB before rotation.
        chunk_size: Maximum number of files per chunk before incrementing chunk index.
        default_path: Format string for generating file paths.
        aggr_root: Root path for the aggregated dataset.

    Returns:
        tuple[int, int]: Destination (chunk, file) indices for this source file.
    """
    dst_path = aggr_root / default_path.format(chunk_index=idx["chunk"], file_index=idx["file"])

    if not dst_path.exists():
        return idx["chunk"], idx["file"]

    src_size = get_parquet_file_size_in_mb(src_path)
    dst_size = get_parquet_file_size_in_mb(dst_path)

    if dst_size + src_size >= max_mb:
        return update_chunk_file_indices(idx["chunk"], idx["file"], chunk_size)

    return idx["chunk"], idx["file"]


def write_parquet_to_target(
    df: pd.DataFrame,
    chunk_idx: int,
    file_idx: int,
    default_path: str,
    contains_images: bool = False,
    aggr_root: Path = None,
):
    """Writes a parquet DataFrame to a specific destination file, appending if it already exists."""
    target_path = aggr_root / default_path.format(chunk_index=chunk_idx, file_index=file_idx)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        existing_df = pd.read_parquet(target_path)
        final_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        final_df = df

    if contains_images:
        to_parquet_with_hf_images(final_df, target_path)
    else:
        final_df.to_parquet(target_path)


def finalize_aggregation(aggr_meta, all_metadata):
    """Finalizes the dataset aggregation by writing summary files and statistics.

    Writes the tasks file, info file with total counts and splits, and
    aggregated statistics from all source datasets.

    Args:
        aggr_meta: Aggregated dataset metadata.
        all_metadata: List of all source dataset metadata objects.
    """
    logging.info("write tasks")
    write_tasks(aggr_meta.tasks, aggr_meta.root)

    logging.info("write info")
    aggr_meta.info.update(
        {
            "total_tasks": len(aggr_meta.tasks),
            "total_episodes": sum(m.total_episodes for m in all_metadata),
            "total_frames": sum(m.total_frames for m in all_metadata),
            "splits": {"train": f"0:{sum(m.total_episodes for m in all_metadata)}"},
        }
    )
    write_info(aggr_meta.info, aggr_meta.root)

    logging.info("write stats")
    aggr_meta.stats = aggregate_stats([m.stats for m in all_metadata])
    write_stats(aggr_meta.stats, aggr_meta.root)
