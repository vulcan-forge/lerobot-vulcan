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
    update_video_indices: bool = True,  # New parameter
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
        update_video_indices: Whether to update video file indices (default: True)
        
    Returns:
        Tuple of updated offsets: (episode_offset, frame_offset, video_timestamp_offset)
    """
    all_episodes_data = []
    current_episode_offset = episode_offset
    current_frame_offset = frame_offset
    current_video_timestamp_offset = video_timestamp_offset
    
    for dataset_idx, dataset in enumerate(datasets):        
        # Load episodes metadata WITHOUT filtering out stats columns
        # This ensures we get all the data that contributes to file size
        episodes_dataset = load_nested_dataset(dataset.meta.root / "meta" / "episodes")
        
        # Log progress every 100 datasets
        if dataset_idx % 100 == 0:            
            logging.info(f"Processing episodes for datasets {dataset_idx + 1} - {len(datasets)}")
        
        episode_count = 0
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
                
                # Keep original video file indices - they will be updated later after video combination
                # Don't modify chunk_index and file_index here
            
            all_episodes_data.append(new_episode_data)
            
            current_episode_offset += 1
            current_frame_offset += episode_data['length']
            episode_count += 1
        
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
    
    if not episodes_data:
        logging.warning("No episodes data to write")
        return
    
    # Convert to DataFrame
    episodes_df = pd.DataFrame(episodes_data)
    
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

def _validate_episode_data(episodes_data: List[Dict]) -> None:
    """Validate that episode data is properly combined."""
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

def update_episode_video_indices(output_path: Path, datasets: List[Any]) -> None:
    """
    Update episode metadata with correct video file indices based on actual video file organization.
    
    This function analyzes the actual video files created during combination and updates
    the episode metadata to point to the correct video files based on timestamp ranges.
    
    Args:
        output_path: Output directory path
        datasets: List of LeRobotDataset objects (for reference)
    """
    if not datasets:
        logging.info("No datasets provided")
        return
    
    # Get video keys from first dataset
    video_keys = datasets[0].meta.video_keys
    if not video_keys:
        logging.info("No video keys found")
        return
    
    # Load all episodes metadata
    episodes_dir = output_path / "meta" / "episodes"
    if not episodes_dir.exists():
        logging.warning(f"Episodes directory not found: {episodes_dir}")
        return
    
    # Find all episode files
    episode_files = []
    for chunk_dir in episodes_dir.iterdir():
        if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk-"):
            for file_path in chunk_dir.iterdir():
                if file_path.is_file() and file_path.name.startswith("file-") and file_path.suffix == ".parquet":
                    episode_files.append(file_path)
    
    episode_files.sort()
    
    # Process each video key
    for video_key in video_keys:
        logging.info(f"Processing video key: {video_key}")
        
        # Get actual video files that were created
        video_dir = output_path / "videos" / video_key
        if not video_dir.exists():
            logging.warning(f"Video directory not found: {video_dir}")
            continue
        
        # Find all video files
        video_files = []
        for chunk_dir in video_dir.iterdir():
            if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk-"):
                for file_path in chunk_dir.iterdir():
                    if file_path.is_file() and file_path.name.startswith("file-") and file_path.suffix == ".mp4":
                        video_files.append(file_path)
        
        video_files.sort()
        
        if not video_files:
            logging.warning(f"No video files found for {video_key}")
            continue
        
        # Create mapping from timestamp ranges to video files
        video_file_mapping = _create_video_file_mapping(video_files, video_key)
        
        # Update episode metadata for this video key
        _update_episodes_for_video_key(episode_files, video_key, video_file_mapping)
    
    logging.info("✓ Episode metadata updated with correct video file indices")


def _create_video_file_mapping(video_files: List[Path], video_key: str) -> List[Dict]:
    """
    Create mapping from timestamp ranges to video files based on actual episode data.
    
    Args:
        video_files: List of video file paths
        video_key: Video key name
        
    Returns:
        List of mappings with timestamp ranges and file indices
    """
    from lerobot.datasets.video_utils import get_video_duration_in_s
    
    mappings = []
    
    for video_file in video_files:
        # Extract chunk and file indices from path
        parts = video_file.parts
        chunk_part = None
        file_part = None
        
        for part in parts:
            if part.startswith("chunk-"):
                chunk_part = part
            elif part.startswith("file-"):
                file_part = part
        
        if not chunk_part or not file_part:
            logging.warning(f"Could not parse video file path: {video_file}")
            continue
        
        chunk_idx = int(chunk_part.split("-")[1])
        file_idx = int(file_part.split("-")[1].split(".")[0])
        
        # Get video duration
        try:
            duration = get_video_duration_in_s(video_file)
        except Exception as e:
            logging.warning(f"Could not get duration for {video_file}: {e}")
            duration = 0.0
        
        mapping = {
            'chunk_index': chunk_idx,
            'file_index': file_idx,
            'timestamp_start': None,  # Will be calculated based on episodes
            'timestamp_end': None,    # Will be calculated based on episodes
            'video_file': video_file,
            'duration': duration
        }
        mappings.append(mapping)
        
    return mappings


def _update_episodes_for_video_key(
    episode_files: List[Path], 
    video_key: str, 
    video_file_mapping: List[Dict]
) -> None:
    """
    Update episode metadata for a specific video key.
    
    Args:
        episode_files: List of episode file paths
        video_key: Video key name
        video_file_mapping: Mapping from timestamp ranges to video files
    """
    if not video_file_mapping:
        return
    
    # First pass: collect all episodes sorted by timestamp
    all_episodes = []
    for episode_file in episode_files:
        episodes_df = pd.read_parquet(episode_file)
        
        chunk_key = f'videos/{video_key}/chunk_index'
        file_key = f'videos/{video_key}/file_index'
        from_timestamp_key = f'videos/{video_key}/from_timestamp'
        to_timestamp_key = f'videos/{video_key}/to_timestamp'
        
        if chunk_key not in episodes_df.columns:
            continue
        
        for idx, episode_row in episodes_df.iterrows():
            if pd.isna(episode_row[from_timestamp_key]):
                continue
            
            episode_data = {
                'episode_index': episode_row['episode_index'],
                'chunk_index': episode_row[chunk_key],
                'file_index': episode_row[file_key],
                'from_timestamp': episode_row[from_timestamp_key],
                'to_timestamp': episode_row[to_timestamp_key],
                'episode_file': episode_file,
                'episode_row_idx': idx
            }
            all_episodes.append(episode_data)
    
    # Sort all episodes by timestamp
    all_episodes.sort(key=lambda x: x['from_timestamp'])
    
    # Calculate cumulative video durations to determine episode-to-file mapping
    cumulative_duration = 0.0
    episode_idx = 0
    
    for mapping in video_file_mapping:
        video_duration = mapping['duration']
        
        # Find episodes that belong to this video file based on cumulative duration
        episodes_in_file = []
        while (episode_idx < len(all_episodes) and 
               all_episodes[episode_idx]['from_timestamp'] < cumulative_duration + video_duration):
            episodes_in_file.append(all_episodes[episode_idx])
            episode_idx += 1
        
        if episodes_in_file:
            # Calculate timestamp range for this video file
            min_timestamp = min(ep['from_timestamp'] for ep in episodes_in_file)
            max_timestamp = max(ep['to_timestamp'] for ep in episodes_in_file)
            
            mapping['timestamp_start'] = min_timestamp
            mapping['timestamp_end'] = max_timestamp
        else:
            logging.warning(f"No episodes found for video file chunk={mapping['chunk_index']}, file={mapping['file_index']}")
        
        cumulative_duration += video_duration
    
    # Second pass: update episode metadata with correct file indices and timestamps
    updated_count = 0
    for episode_file in episode_files:
        episodes_df = pd.read_parquet(episode_file)
        
        chunk_key = f'videos/{video_key}/chunk_index'
        file_key = f'videos/{video_key}/file_index'
        from_timestamp_key = f'videos/{video_key}/from_timestamp'
        to_timestamp_key = f'videos/{video_key}/to_timestamp'
        
        if chunk_key not in episodes_df.columns:
            continue
        
        updated = False
        for idx, episode_row in episodes_df.iterrows():
            if pd.isna(episode_row[from_timestamp_key]):
                continue
            
            episode_from_timestamp = episode_row[from_timestamp_key]
            episode_to_timestamp = episode_row[to_timestamp_key]
            
            # Find which video file this episode belongs to based on timestamp
            target_mapping = None
            for mapping in video_file_mapping:
                if (mapping['timestamp_start'] is not None and 
                    mapping['timestamp_start'] <= episode_from_timestamp < mapping['timestamp_end']):
                    target_mapping = mapping
                    break
            
            if target_mapping:
                # Update the episode metadata
                old_chunk = episodes_df.at[idx, chunk_key]
                old_file = episodes_df.at[idx, file_key]
                new_chunk = target_mapping['chunk_index']
                new_file = target_mapping['file_index']
                
                episodes_df.at[idx, chunk_key] = new_chunk
                episodes_df.at[idx, file_key] = new_file
                
                # CRITICAL FIX: Adjust timestamps to be relative to the video file start
                video_file_start = target_mapping['timestamp_start']
                episodes_df.at[idx, from_timestamp_key] = episode_from_timestamp - video_file_start
                episodes_df.at[idx, to_timestamp_key] = episode_to_timestamp - video_file_start
                
                updated = True
                updated_count += 1
            else:
                logging.warning(f"No video file mapping found for episode {episode_row['episode_index']} with timestamp {episode_from_timestamp:.2f}s")
        
        # Save updated episodes if any changes were made
        if updated:
            episodes_df.to_parquet(episode_file, index=False)
            logging.debug(f"Updated episode file: {episode_file}")
    
    logging.info(f"Updated {updated_count} episodes for {video_key}")
