import logging
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

from lerobot.datasets.utils import (
    load_nested_dataset,
    DEFAULT_EPISODES_PATH,
    get_parquet_file_size_in_mb,
    update_chunk_file_indices,
)


def combine_episodes_metadata(
    datasets: List[Any],
    output_path: Path,
    task_content_to_index: Dict[str, int],
    episode_offset: int = 0,
    frame_offset: int = 0,
    video_timestamp_offset: float = 0.0,
    data_file_size_in_mb: int = 100,  # Default file size limit
) -> tuple[int, int, float]:
    """
    Combine episodes metadata from multiple datasets.
    
    Args:
        datasets: List of LeRobotDataset objects
        output_path: Output directory path
        task_content_to_index: Mapping from task strings to indices
        episode_offset: Starting episode index offset
        frame_offset: Starting frame index offset
        video_timestamp_offset: Starting video timestamp offset
        data_file_size_in_mb: Maximum file size in MB for episodes files
        
    Returns:
        Tuple of updated offsets: (episode_offset, frame_offset, video_timestamp_offset)
    """
    logging.info("Combining episodes metadata...")
    
    all_episodes_data = []
    current_episode_offset = episode_offset
    current_frame_offset = frame_offset
    current_video_timestamp_offset = video_timestamp_offset
    
    for dataset_idx, dataset in enumerate(datasets):
        logging.info(f"Processing episodes for dataset {dataset_idx + 1}/{len(datasets)}")
        
        # Load episodes metadata WITHOUT filtering out stats columns
        # This ensures we get all the data that contributes to file size
        episodes_dataset = load_nested_dataset(dataset.meta.root / "meta" / "episodes")
        
        logging.info(f"Dataset {dataset_idx + 1} has {len(episodes_dataset)} episodes")
        logging.info(f"Dataset {dataset_idx + 1} episodes columns: {list(episodes_dataset.features.keys())}")
        
        for episode_idx, episode_data in enumerate(episodes_dataset):
            # Create new episode data
            new_episode_data = dict(episode_data)
            
            # Map episode index - use consecutive indexing
            new_episode_data['episode_index'] = current_episode_offset
            
            # Map task index
            if 'task_index' in new_episode_data:
                old_task_idx = new_episode_data['task_index']
                # Get task string from the dataset's tasks DataFrame
                if old_task_idx < len(dataset.meta.tasks):
                    old_task_string = dataset.meta.tasks.iloc[old_task_idx].name
                    if old_task_string in task_content_to_index:
                        new_episode_data['task_index'] = task_content_to_index[old_task_string]
                    else:
                        logging.warning(f"Task string '{old_task_string}' not found in combined tasks")
                        new_episode_data['task_index'] = 0  # Default to first task
                else:
                    logging.warning(f"Task index {old_task_idx} out of range for dataset {dataset_idx}")
                    new_episode_data['task_index'] = 0
            
            # Map frame indices - ensure no overlap
            new_episode_data['dataset_from_index'] = current_frame_offset
            new_episode_data['dataset_to_index'] = current_frame_offset + episode_data['length']
            
            # Update video timestamps to account for accumulated video duration
            for video_key in dataset.meta.video_keys:
                if f'videos/{video_key}/from_timestamp' in new_episode_data:
                    new_episode_data[f'videos/{video_key}/from_timestamp'] += current_video_timestamp_offset
                if f'videos/{video_key}/to_timestamp' in new_episode_data:
                    new_episode_data[f'videos/{video_key}/to_timestamp'] += current_video_timestamp_offset
            
            all_episodes_data.append(new_episode_data)
            
            current_episode_offset += 1
            current_frame_offset += episode_data['length']
        
        # Update video timestamp offset for next dataset
        if dataset.meta.video_keys:
            dataset_video_duration = 0.0
            for episode_data in episodes_dataset:
                for video_key in dataset.meta.video_keys:
                    if f'videos/{video_key}/to_timestamp' in episode_data:
                        dataset_video_duration = max(dataset_video_duration, episode_data[f'videos/{video_key}/to_timestamp'])
            current_video_timestamp_offset += dataset_video_duration
    
    logging.info(f"Total episodes collected: {len(all_episodes_data)}")
    
    # Validate episode data
    _validate_episode_data(all_episodes_data)
    
    # Write episodes metadata with proper chunking
    _write_episodes_with_chunking(all_episodes_data, output_path, data_file_size_in_mb)
    
    logging.info(f"Combined {len(all_episodes_data)} episodes successfully")
    
    return (
        current_episode_offset,
        current_frame_offset,
        current_video_timestamp_offset,
    )


def _write_episodes_with_chunking(
    episodes_data: List[Dict],
    output_path: Path,
    data_file_size_in_mb: int,
) -> None:
    """Write episodes metadata with proper chunking based on file size limits."""
    logging.info("Writing episodes metadata with chunking...")
    
    if not episodes_data:
        logging.warning("No episodes data to write")
        return
    
    # Convert to DataFrame
    episodes_df = pd.DataFrame(episodes_data)
    
    logging.info(f"Episodes DataFrame shape: {episodes_df.shape}")
    logging.info(f"Episodes DataFrame columns: {list(episodes_df.columns)}")
    
    # Initialize chunking variables
    chunk_idx = 0
    file_idx = 0
    current_df = pd.DataFrame()
    
    for _, episode_row in episodes_df.iterrows():
        # Add episode to current dataframe
        if current_df.empty:
            current_df = episode_row.to_frame().T
        else:
            current_df = pd.concat([current_df, episode_row.to_frame().T], ignore_index=True)
        
        # Check if we need to write the current file
        temp_file = output_path / "temp_episodes.parquet"
        current_df.to_parquet(temp_file, index=False)
        current_size_mb = get_parquet_file_size_in_mb(temp_file)
        temp_file.unlink()  # Clean up temp file
        
        if current_size_mb >= data_file_size_in_mb:
            # Write current dataframe to file
            _write_episodes_file(current_df, output_path, chunk_idx, file_idx)
            
            # Reset for next file
            current_df = pd.DataFrame()
            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, 1000)  # Use large chunk size for episodes
    
    # Write remaining episodes if any
    if not current_df.empty:
        _write_episodes_file(current_df, output_path, chunk_idx, file_idx)
    
    logging.info(f"Episodes metadata written across {chunk_idx + 1} chunks")


def _write_episodes_file(
    episodes_df: pd.DataFrame,
    output_path: Path,
    chunk_idx: int,
    file_idx: int,
) -> None:
    """Write a single episodes file."""
    episodes_dir = output_path / "meta" / "episodes" / f"chunk-{chunk_idx:03d}"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    
    episodes_file = episodes_dir / f"file-{file_idx:03d}.parquet"
    episodes_df.to_parquet(episodes_file, index=False)
    
    file_size_mb = get_parquet_file_size_in_mb(episodes_file)
    logging.info(f"Episodes metadata written to {episodes_file} ({len(episodes_df)} episodes, {file_size_mb:.2f} MB)")


def _validate_episode_data(episodes_data: List[Dict]) -> None:
    """Validate that episode data is properly combined."""
    logging.info("Validating combined episode data...")
    
    if not episodes_data:
        logging.warning("No episodes data to validate")
        return
    
    # Check episode indices are consecutive
    episode_indices = [ep['episode_index'] for ep in episodes_data]
    expected_indices = list(range(len(episodes_data)))
    
    if episode_indices != expected_indices:
        logging.error(f"Episode indices not consecutive! Got {episode_indices}, expected {expected_indices}")
        raise ValueError("Episode indices are not properly mapped")
    
    # Check frame indices don't overlap
    frame_ranges = [(ep['dataset_from_index'], ep['dataset_to_index']) for ep in episodes_data]
    for i in range(len(frame_ranges)):
        for j in range(i + 1, len(frame_ranges)):
            start1, end1 = frame_ranges[i]
            start2, end2 = frame_ranges[j]
            if not (end1 <= start2 or end2 <= start1):
                logging.error(f"Overlapping frame ranges: episode {i} [{start1}, {end1}) and episode {j} [{start2}, {end2})")
                raise ValueError("Frame ranges overlap")
    
    # Check that episode lengths are consistent
    for i, ep in enumerate(episodes_data):
        expected_length = ep['dataset_to_index'] - ep['dataset_from_index']
        if ep['length'] != expected_length:
            logging.error(f"Episode {i} length mismatch: stored={ep['length']}, calculated={expected_length}")
            raise ValueError("Episode length mismatch")
    
    logging.info("Episode data validation passed ✓")
