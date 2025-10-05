"""
LeRobot Dataset Combination Module

This module provides functionality to combine multiple LeRobot v3.0 datasets
into a single dataset with proper metadata handling, chunking, and validation.
"""

import argparse
import logging
import shutil
import json
from pathlib import Path
from typing import Dict, List, Any

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.combine_dataset.combine_metadata.combine_info import create_combined_info_metadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    create_lerobot_dataset_card,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
)
from lerobot.utils.utils import init_logging

# Import our modular functions - use absolute imports
from lerobot.datasets.combine_dataset.combine_metadata.combine_episodes import combine_episodes_metadata, update_episode_video_indices
from lerobot.datasets.combine_dataset.combine_metadata.combine_tasks import combine_tasks_metadata
from lerobot.datasets.combine_dataset.combine_metadata.combine_stats import combine_stats_metadata
from lerobot.datasets.combine_dataset.combine_parquet.combine_parquet import combine_parquet_files
from lerobot.datasets.combine_dataset.combine_video.combine_video import combine_video_files


def combine_v3_datasets(
    dataset_paths: List[str],
    output_path: str,
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
        dataset_paths: List of dataset repo IDs to combine (e.g., "repo_id/dataset_name")
        output_path: Output repo ID for the combined dataset (e.g., "repo_id/dataset_name")
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
        data_file_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB* 1
    if video_file_size_in_mb is None:
        video_file_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB* 1
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE* 1

    logging.info(f"Combining {len(dataset_paths)} datasets...")

    # Load all datasets from local cache
    datasets = []
    for repo_id in dataset_paths:
        logging.info(f"Loading dataset from local cache: {repo_id}")
        
        # Load from local cache - LeRobotDataset will look in HF_LEROBOT_HOME / repo_id
        dataset = LeRobotDataset(repo_id=repo_id, force_cache_sync=False)
        datasets.append(dataset)

    # Validate compatibility
    _validate_dataset_compatibility(datasets)

    # Determine output path - construct local cache path for output
    output_repo_id = output_path
    output_local_path = HF_LEROBOT_HOME / output_path

    # Create output directory
    if output_local_path.exists():
        shutil.rmtree(output_local_path)
    output_local_path.mkdir(parents=True, exist_ok=True)

    # Create combined dataset metadata
    first_dataset = datasets[0]
    combined_meta = LeRobotDatasetMetadata.create(
        repo_id=output_repo_id,
        fps=first_dataset.fps,
        features=first_dataset.features,
        robot_type=first_dataset.meta.robot_type,
        root=output_local_path,
        use_videos=len(first_dataset.meta.video_keys) > 0,
    )

    # Update chunk settings
    combined_meta.update_chunk_settings(
        chunks_size=chunk_size,
        data_files_size_in_mb=data_file_size_in_mb,
        video_files_size_in_mb=video_file_size_in_mb,
    )

    # Step 1: Combine tasks metadata
    logging.info("Step 1: Combining tasks metadata...")
    task_content_to_index = combine_tasks_metadata(datasets, output_local_path)
    logging.info("")

    # Step 2: Combine episodes metadata (WITHOUT video file indices - will be updated later)
    logging.info("Step 2: Combining episodes metadata...")
    episode_offset, frame_offset, video_timestamp_offset = combine_episodes_metadata(
        datasets=datasets,
        output_path=output_local_path,
        task_content_to_index=task_content_to_index,
        data_file_size_in_mb=data_file_size_in_mb,
        update_video_indices=False,  # Don't update video indices yet
    )
    logging.info("")

    # Step 3: Combine statistics metadata
    logging.info("Step 3: Combining statistics metadata...")
    combine_stats_metadata(datasets, output_local_path)
    logging.info("")

    # Step 4: Create combined info metadata
    logging.info("Step 4: Creating combined info metadata...")
    create_combined_info_metadata(
        datasets=datasets,
        output_path=output_local_path,
        total_episodes=episode_offset,
        total_frames=frame_offset,
        total_tasks=len(task_content_to_index),
        data_file_size_in_mb=data_file_size_in_mb,
        video_file_size_in_mb=video_file_size_in_mb,
        chunk_size=chunk_size,
    )
    logging.info("")

    # Step 5: Combine parquet data files
    logging.info("Step 5: Combining parquet data files...")
    combine_parquet_files(
        datasets=datasets,
        output_path=output_local_path,
        chunk_size=chunk_size,
        data_file_size_in_mb=data_file_size_in_mb,
        image_keys=first_dataset.meta.image_keys,
    )
    logging.info("")
    
    # Step 6: Combine video files (if any)
    if first_dataset.meta.video_keys:
        logging.info("Step 6: Combining video files...")
        combine_video_files(
            datasets=datasets,
            output_path=output_local_path,
            chunk_size=chunk_size,
            video_file_size_in_mb=video_file_size_in_mb,
        )
        
        # Step 6b: Update episode metadata with correct video file indices
        logging.info("Step 6b: Updating episode metadata with correct video file indices...")
        update_episode_video_indices(output_local_path, datasets)
    else:
        logging.info("Step 6: No video files to combine")
    logging.info("")

    # Create dataset card
    if push_to_hub:
        logging.info("Creating dataset card...")
        card = create_lerobot_dataset_card(
            tags=tags,
            dataset_info=combined_meta.info,
            license=license,
        )
        card.push_to_hub(output_repo_id, repo_type="dataset")

    # Load and return the combined dataset
    combined_dataset = LeRobotDataset(
        repo_id=output_repo_id,
        root=output_local_path,
    )

    if push_to_hub:
        logging.info(f"Pushing dataset to Hugging Face Hub: {output_repo_id}")
        combined_dataset.push_to_hub()

    logging.info("Dataset combination completed successfully!")
    logging.info(f"Combined dataset: {output_repo_id}")
    logging.info(f"Total episodes: {episode_offset}")
    logging.info(f"Total frames: {frame_offset}")
    logging.info(f"Total tasks: {len(task_content_to_index)}")

    return combined_dataset


def _validate_dataset_compatibility(datasets: List[LeRobotDataset]) -> None:
    """
    Validate that all datasets are compatible for combination.
    
    Args:
        datasets: List of LeRobotDataset objects
        
    Raises:
        ValueError: If datasets are incompatible
    """
    if not datasets:
        raise ValueError("No datasets provided")
    
    if len(datasets) == 1:
        logging.info("Only one dataset provided, no compatibility check needed")
        return

    logging.info("Validating dataset compatibility...")
    
    first_dataset = datasets[0]
    
    for i, dataset in enumerate(datasets[1:], 1):
        # Check FPS compatibility
        if dataset.fps != first_dataset.fps:
            raise ValueError(f"Dataset {i+1} has incompatible FPS: {dataset.fps} vs {first_dataset.fps}")
        
        # Check features compatibility
        if dataset.features != first_dataset.features:
            raise ValueError(f"Dataset {i+1} has incompatible features")
        
        # Check video backend compatibility
        if dataset.video_backend != first_dataset.video_backend:
            raise ValueError(f"Dataset {i+1} has incompatible video backend")
        
        # Check robot type compatibility
        if dataset.meta.robot_type != first_dataset.meta.robot_type:
            raise ValueError(f"Dataset {i+1} has incompatible robot type: {dataset.meta.robot_type} vs {first_dataset.meta.robot_type}")
    
    logging.info("All datasets validated as compatible ✓")


def _parse_dataset_paths(dataset_paths_arg: str) -> List[str]:
    """
    Parse dataset paths from command line argument.
    Supports both space-separated format and JSON array format.
    
    Args:
        dataset_paths_arg: String containing dataset paths
        
    Returns:
        List of dataset path strings
    """
    # Try to parse as JSON array first
    if dataset_paths_arg.strip().startswith('[') and dataset_paths_arg.strip().endswith(']'):
        try:
            # Clean up the JSON string by removing trailing commas
            cleaned_arg = dataset_paths_arg.strip()
            # Remove trailing comma before the closing bracket
            cleaned_arg = cleaned_arg.replace(',]', ']').replace(', ]', ']')
            
            parsed_paths = json.loads(cleaned_arg)
            # Filter out empty strings and None values
            filtered_paths = [path for path in parsed_paths if path and isinstance(path, str) and path.strip()]
            return filtered_paths
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse as JSON array: {e}")
            # If JSON parsing fails, fall back to space-separated parsing
            pass
    
    # Parse as space-separated arguments
    import shlex
    parsed_paths = shlex.split(dataset_paths_arg)
    # Filter out empty strings and brackets
    filtered_paths = [path for path in parsed_paths if path and path.strip() and path not in ['[', ']']]
    return filtered_paths


def main():
    parser = argparse.ArgumentParser(description="Combine multiple LeRobot v3.0 datasets into a single dataset")
    parser.add_argument(
        "--dataset_paths",
        type=str,
        required=True,
        help="Dataset paths to combine. Can be space-separated (e.g., 'repo1/dataset1 repo2/dataset2') or JSON array format (e.g., '[\"repo1/dataset1\", \"repo2/dataset2\"]').",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path/repo ID for the combined dataset (e.g., `repo_id/combined_dataset`).",
    )
    parser.add_argument(
        "--data_file_size_in_mb",
        type=int,
        default=DEFAULT_DATA_FILE_SIZE_IN_MB* 1,
        help="Maximum size for data files in MB.",
    )
    parser.add_argument(
        "--video_file_size_in_mb",
        type=int,
        default=DEFAULT_VIDEO_FILE_SIZE_IN_MB* 1,
        help="Maximum size for video files in MB.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE* 1,
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

    # Parse dataset paths
    dataset_paths = _parse_dataset_paths(args.dataset_paths)
    
    logging.info(f"Parsed dataset paths: {dataset_paths}")

    # Run the combination
    combined_dataset = combine_v3_datasets(
        dataset_paths=dataset_paths,
        output_path=args.output_path,
        data_file_size_in_mb=args.data_file_size_in_mb,
        video_file_size_in_mb=args.video_file_size_in_mb,
        chunk_size=args.chunk_size,
        push_to_hub=args.push_to_hub,
        private=args.private,
        tags=args.tags,
        license=args.license,
    )

    logging.info(f"Successfully combined datasets into {args.output_path}")
    logging.info(f"Total episodes: {combined_dataset.meta.total_episodes}")
    logging.info(f"Total frames: {combined_dataset.meta.total_frames}")


if __name__ == "__main__":
    init_logging()
    main()
