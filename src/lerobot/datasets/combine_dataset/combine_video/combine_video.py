import logging
import shutil
from pathlib import Path
from typing import List, Any, Dict

from lerobot.datasets.utils import (
    DEFAULT_VIDEO_PATH,
    update_chunk_file_indices,
    get_video_size_in_mb,
)
from lerobot.datasets.video_utils import concatenate_video_files


def combine_video_files(
    datasets: List[Any],
    output_path: Path,
    chunk_size: int,
    video_file_size_in_mb: int
) -> None:
    """
    Combine video files with proper size-aware chunking.
    
    Args:
        datasets: List of LeRobotDataset objects
        output_path: Output directory path
        chunk_size: Maximum number of files per chunk
        video_file_size_in_mb: Maximum size for video files in MB
    """
    logging.info("Combining video files with proper chunking...")
    
    if not datasets:
        logging.warning("No datasets provided for video combination")
        return
    
    # Get all video keys from the first dataset
    video_keys = datasets[0].meta.video_keys
    if not video_keys:
        logging.info("No video keys found in datasets")
        return
    
    logging.info(f"Found video keys: {video_keys}")
    
    for video_key in video_keys:
        logging.info(f"Processing video key: {video_key}")
        
        chunk_idx, file_idx = 0, 0
        current_video_files = []
        current_size_mb = 0.0
        total_videos_processed = 0
        
        for dataset_idx, dataset in enumerate(datasets):
            logging.info(f"Processing videos for dataset {dataset_idx + 1}/{len(datasets)}")
            
            video_dir = dataset.meta.root / "videos" / video_key
            if not video_dir.exists():
                logging.warning(f"Video directory not found: {video_dir}")
                continue
            
            # Find all video files in the directory structure
            video_files = _find_video_files(video_dir)
            if not video_files:
                logging.warning(f"No video files found in {video_dir}")
                continue
            
            logging.info(f"Found {len(video_files)} video files in {video_dir}")
            
            for video_file in sorted(video_files):
                try:
                    video_size_mb = get_video_size_in_mb(video_file)
                    logging.debug(f"Processing video file: {video_file} ({video_size_mb:.2f} MB)")
                    
                    # Check if adding this video would exceed size limit
                    if current_size_mb + video_size_mb >= video_file_size_in_mb and current_video_files:
                        # Write current batch
                        _write_video_file(current_video_files, output_path, video_key, chunk_idx, file_idx)
                        logging.info(f"Wrote video batch chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4 with {len(current_video_files)} videos")
                        
                        # Move to next file
                        chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunk_size)
                        current_video_files = []
                        current_size_mb = 0.0
                    
                    current_video_files.append(video_file)
                    current_size_mb += video_size_mb
                    total_videos_processed += 1
                    
                except Exception as e:
                    logging.error(f"Error processing video file {video_file}: {e}")
                    continue
        
        # Write remaining videos
        if current_video_files:
            _write_video_file(current_video_files, output_path, video_key, chunk_idx, file_idx)
            logging.info(f"Wrote final video batch chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4 with {len(current_video_files)} videos")
        
        logging.info(f"Completed video key '{video_key}': processed {total_videos_processed} videos")
    
    logging.info("Video files combined successfully")


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


def _write_video_file(video_files: List[Path], output_path: Path, video_key: str, chunk_idx: int, file_idx: int) -> None:
    """Write a concatenated video file."""
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
            # Multiple video files - concatenate them
            concatenate_video_files(video_files, output_path_video)
            logging.debug(f"Concatenated {len(video_files)} video files -> {output_path_video}")
            
    except Exception as e:
        logging.error(f"Error writing video file {output_path_video}: {e}")
        raise


def combine_video_files_by_camera(
    datasets: List[Any],
    output_path: Path,
    chunk_size: int,
    video_file_size_in_mb: int,
    camera_mapping: Dict[str, str] = None
) -> None:
    """
    Combine video files with camera-specific processing.
    
    Args:
        datasets: List of LeRobotDataset objects
        output_path: Output directory path
        chunk_size: Maximum number of files per chunk
        video_file_size_in_mb: Maximum size for video files in MB
        camera_mapping: Optional mapping from old camera names to new ones
    """
    logging.info("Combining video files by camera...")
    
    if not datasets:
        logging.warning("No datasets provided for video combination")
        return
    
    # Get all video keys from the first dataset
    video_keys = datasets[0].meta.video_keys
    if not video_keys:
        logging.info("No video keys found in datasets")
        return
    
    for video_key in video_keys:
        # Apply camera mapping if provided
        output_video_key = camera_mapping.get(video_key, video_key) if camera_mapping else video_key
        
        logging.info(f"Processing camera: {video_key} -> {output_video_key}")
        
        chunk_idx, file_idx = 0, 0
        current_video_files = []
        current_size_mb = 0.0
        
        for dataset_idx, dataset in enumerate(datasets):
            video_dir = dataset.meta.root / "videos" / video_key
            if not video_dir.exists():
                continue
            
            video_files = _find_video_files(video_dir)
            
            for video_file in sorted(video_files):
                try:
                    video_size_mb = get_video_size_in_mb(video_file)
                    
                    # Check if adding this video would exceed size limit
                    if current_size_mb + video_size_mb >= video_file_size_in_mb and current_video_files:
                        _write_video_file(current_video_files, output_path, output_video_key, chunk_idx, file_idx)
                        chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunk_size)
                        current_video_files = []
                        current_size_mb = 0.0
                    
                    current_video_files.append(video_file)
                    current_size_mb += video_size_mb
                    
                except Exception as e:
                    logging.error(f"Error processing video file {video_file}: {e}")
                    continue
        
        # Write remaining videos
        if current_video_files:
            _write_video_file(current_video_files, output_path, output_video_key, chunk_idx, file_idx)
    
    logging.info("Video files combined by camera successfully")


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
