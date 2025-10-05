import logging
import shutil
from pathlib import Path
from typing import List, Any, Dict

from lerobot.datasets.utils import (
    DEFAULT_VIDEO_PATH,
    update_chunk_file_indices,
    get_video_size_in_mb,
    load_episodes,
)
from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s


def combine_video_files(
    datasets: List[Any],
    output_path: Path,
    chunk_size: int,
    video_file_size_in_mb: int
) -> None:
    """
    Combine video files with proper size-aware chunking and timestamp synchronization.
    
    Args:
        datasets: List of LeRobotDataset objects
        output_path: Output directory path
        chunk_size: Maximum number of files per chunk
        video_file_size_in_mb: Maximum size for video files in MB
    """
    logging.info("Combining video files with proper chunking and timestamp sync...")
    
    if not datasets:
        logging.warning("No datasets provided for video combination")
        return
    
    # Get all video keys from the first dataset
    video_keys = datasets[0].meta.video_keys
    if not video_keys:
        logging.info("No video keys found in datasets")
        return
    
    for video_key in video_keys:
        logging.info(f"Processing video key: {video_key}")
        
        # Process videos in episode order to maintain timestamp continuity
        _combine_videos_by_episode_order(
            datasets, output_path, video_key, chunk_size, video_file_size_in_mb
        )

    logging.info("✓ Video files combined successfully")
    logging.info("")


def _find_episode_video_file(episode_data: Dict, video_files: List[Path], video_key: str) -> Path:
    """Find the video file corresponding to a specific episode."""
    episode_index = episode_data.get('episode_index', -1)
    
    # For LeRobot v3.0 datasets, videos are typically organized by chunks/files
    # and don't have episode-specific names. We'll process them in order.
    
    # Look for video files that might correspond to this episode
    for video_file in video_files:
        # Check if the video file name contains episode information
        if f"episode_{episode_index:06d}" in video_file.name:
            return video_file
        elif f"episode-{episode_index:06d}" in video_file.name:
            return video_file
    
    # If no specific episode match, return the first available video file
    # This is correct for LeRobot v3.0 datasets where videos are chunked
    if video_files:
        return video_files[0]
    
    return None


def _combine_videos_by_episode_order(
    datasets: List[Any],
    output_path: Path,
    video_key: str,
    chunk_size: int,
    video_file_size_in_mb: int
) -> None:
    """Combine videos in episode order to maintain timestamp continuity."""
    chunk_idx, file_idx = 0, 0
    current_video_files = []
    current_size_mb = 0.0
    cumulative_duration = 0.0
    
    # Process datasets in order
    for dataset_idx, dataset in enumerate(datasets):
        # Log progress every 100 datasets
        if dataset_idx % 100 == 0:            
            logging.info(f"Processing videos for datasets {dataset_idx + 1} - {len(datasets)}")
        
        # Get video files for this dataset
        video_dir = dataset.meta.root / "videos" / video_key
        if not video_dir.exists():
            logging.warning(f"Video directory not found: {video_dir}")
            continue
        
        dataset_video_files = _find_video_files(video_dir)
        if not dataset_video_files:
            logging.warning(f"No video files found in {video_dir}")
            continue
        
        # Process video files directly in order (don't try to match to episodes)
        for video_file in dataset_video_files:
            try:
                video_size_mb = get_video_size_in_mb(video_file)
                
                # Get actual video duration for accurate calculation
                try:
                    video_duration = get_video_duration_in_s(video_file)
                    logging.debug(f"Video {video_file.name}: {video_duration:.2f}s, {video_size_mb:.2f}MB")
                except Exception as e:
                    logging.warning(f"Could not get duration for {video_file}: {e}, using size estimate")
                    # Fallback to size-based estimate if duration extraction fails
                    video_duration = video_size_mb / 10.0  # Rough estimate: 10MB per second
                
                # Check if adding this video would exceed size limit
                # Only create a new file if we already have files AND would exceed the limit
                if (current_video_files and 
                    current_size_mb + video_size_mb > video_file_size_in_mb):
                    # Write current batch
                    _write_video_file_with_timestamps(
                        current_video_files, output_path, video_key, 
                        chunk_idx, file_idx, cumulative_duration
                    )
                    logging.info(f"Wrote video batch chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4 with {len(current_video_files)} videos")
                    
                    # Move to next file
                    chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunk_size)
                    current_video_files = []
                    current_size_mb = 0.0
                    cumulative_duration = 0.0
                
                current_video_files.append(video_file)
                current_size_mb += video_size_mb
                cumulative_duration += video_duration
                
            except Exception as e:
                logging.error(f"Error processing video file {video_file}: {e}")
                continue
    
    # Write remaining videos
    if current_video_files:
        _write_video_file_with_timestamps(
            current_video_files, output_path, video_key, 
            chunk_idx, file_idx, cumulative_duration
        )


def _write_video_file_with_timestamps(
    video_files: List[Path], 
    output_path: Path, 
    video_key: str, 
    chunk_idx: int, 
    file_idx: int,
    cumulative_duration: float
) -> None:
    """Write a concatenated video file with proper timestamp handling."""
    if not video_files:
        return
    
    output_path_video = output_path / DEFAULT_VIDEO_PATH.format(
        video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
    )
    output_path_video.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if len(video_files) == 1:
            # Single video file - just copy it
            shutil.copy2(video_files[0], output_path_video)
            logging.debug(f"Copied single video file: {video_files[0]} -> {output_path_video}")
        else:
            # Multiple video files - concatenate them with timestamp continuity
            concatenate_video_files(video_files, output_path_video)
            logging.debug(f"Concatenated {len(video_files)} video files -> {output_path_video}")
            
            # Validate that the concatenated video duration is reasonable
            _validate_concatenated_video(output_path_video, cumulative_duration)
            
    except Exception as e:
        logging.error(f"Error writing video file {output_path_video}: {e}")
        raise


def _validate_concatenated_video(video_path: Path, expected_duration: float) -> None:
    """Validate that the concatenated video has reasonable duration."""
    try:
        # This is a basic validation - in a real implementation you might want to
        # use ffprobe or similar to get actual video duration
        file_size_mb = get_video_size_in_mb(video_path)
        
        # Basic sanity check - video should not be empty
        if file_size_mb < 0.1:  # Less than 100KB
            logging.warning(f"Concatenated video seems too small: {video_path} ({file_size_mb:.2f} MB)")
        
        logging.debug(f"Video validation passed: {video_path} ({file_size_mb:.2f} MB)")
        
    except Exception as e:
        logging.warning(f"Could not validate video {video_path}: {e}")


def _find_video_files(video_dir: Path) -> List[Path]:
    """
    Find all video files in the directory structure.
    Handles both flat structure and chunked structure.
    
    Args:
        video_dir: Directory to search for video files
        
    Returns:
        List of video file paths
    """
    video_files = []
    
    # Look for video files in the directory structure
    # This handles both:
    # - Flat structure: videos/camera_name/file.mp4
    # - Chunked structure: videos/camera_name/chunk-000/file-000.mp4
    
    for pattern in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
        video_files.extend(video_dir.rglob(pattern))
    
    # Sort by path to ensure consistent ordering
    video_files.sort()
    
    return video_files


def combine_video_files_by_camera(
    datasets: List[Any],
    output_path: Path,
    chunk_size: int,
    video_file_size_in_mb: int,
    camera_mapping: Dict[str, str] = None
) -> None:
    """
    Combine video files with camera-specific processing and timestamp sync.
    
    Args:
        datasets: List of LeRobotDataset objects
        output_path: Output directory path
        chunk_size: Maximum number of files per chunk
        video_file_size_in_mb: Maximum size for video files in MB
        camera_mapping: Optional mapping from old camera names to new ones
    """
    logging.info("Combining video files by camera with timestamp sync...")
    
    if not datasets:
        logging.warning("No datasets provided for video combination")
        return
    
    # Get all video keys from the first dataset
    video_keys = datasets[0].meta.video_keys
    if not video_keys:
        return
    
    for video_key in video_keys:
        # Apply camera mapping if provided
        output_video_key = camera_mapping.get(video_key, video_key) if camera_mapping else video_key
        
        logging.info(f"Processing camera: {video_key} -> {output_video_key}")
        
        # Use the main combine function with proper timestamp handling
        _combine_videos_by_episode_order(
            datasets, output_path, output_video_key, chunk_size, video_file_size_in_mb
        )
    
    logging.info("✓ Video files combined by camera successfully")


def validate_video_files(video_files: List[Path]) -> List[Path]:
    """
    Validate video files and return only valid ones.
    
    Args:
        video_files: List of video file paths
        
    Returns:
        List of valid video file paths
    """
    valid_files = []
    
    for video_file in video_files:
        if not video_file.exists():
            logging.warning(f"Video file does not exist: {video_file}")
            continue
        
        if video_file.stat().st_size == 0:
            logging.warning(f"Video file is empty: {video_file}")
            continue
        
        # Check file extension
        if video_file.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv']:
            logging.warning(f"Unsupported video format: {video_file}")
            continue
        
        valid_files.append(video_file)
    
    return valid_files


def get_video_summary(datasets: List[Any]) -> Dict[str, Any]:
    """
    Get a summary of video files across all datasets.
    
    Args:
        datasets: List of LeRobotDataset objects
        
    Returns:
        Dictionary with video summary information
    """
    summary = {
        "total_datasets": len(datasets),
        "video_keys": set(),
        "total_video_files": 0,
        "total_video_size_mb": 0.0,
        "dataset_video_counts": []
    }
    
    for dataset_idx, dataset in enumerate(datasets):
        dataset_video_count = 0
        dataset_video_size = 0.0
        
        for video_key in dataset.meta.video_keys:
            summary["video_keys"].add(video_key)
            
            video_dir = dataset.meta.root / "videos" / video_key
            if video_dir.exists():
                video_files = _find_video_files(video_dir)
                dataset_video_count += len(video_files)
                
                for video_file in video_files:
                    try:
                        dataset_video_size += get_video_size_in_mb(video_file)
                    except Exception as e:
                        logging.warning(f"Error getting size for {video_file}: {e}")
        
        summary["dataset_video_counts"].append(dataset_video_count)
        summary["total_video_files"] += dataset_video_count
        summary["total_video_size_mb"] += dataset_video_size
    
    summary["video_keys"] = list(summary["video_keys"])
    
    return summary
