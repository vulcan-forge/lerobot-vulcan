import logging
from pathlib import Path
from typing import Dict, List, Any

from lerobot.datasets.utils import write_info


def combine_info_metadata(
    datasets: List[Any], 
    output_path: Path, 
    combined_meta: Any,
    total_episodes: int,
    total_frames: int,
    total_tasks: int
) -> None:
    """
    Combine info metadata from multiple datasets.
    
    Args:
        datasets: List of LeRobotDataset objects
        output_path: Output directory path
        combined_meta: Combined dataset metadata object
        total_episodes: Total number of episodes
        total_frames: Total number of frames
        total_tasks: Total number of tasks
    """
    logging.info("Combining info metadata...")
    
    # Update final info with combined totals
    combined_meta.info.update({
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
    })
    
    # Write the combined info
    write_info(combined_meta.info, output_path)
    
    logging.info(f"Info metadata updated: {total_episodes} episodes, {total_frames} frames, {total_tasks} tasks")


def create_combined_info_metadata(
    datasets: List[Any],
    output_path: Path,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    data_file_size_in_mb: int = None,
    video_file_size_in_mb: int = None,
    chunk_size: int = None,
) -> Dict[str, Any]:
    """
    Create combined info metadata from multiple datasets.
    
    Args:
        datasets: List of LeRobotDataset objects
        output_path: Output directory path
        total_episodes: Total number of episodes
        total_frames: Total number of frames
        total_tasks: Total number of tasks
        data_file_size_in_mb: Maximum size for data files in MB
        video_file_size_in_mb: Maximum size for video files in MB
        chunk_size: Maximum number of files per chunk
        
    Returns:
        Combined info metadata dictionary
    """
    logging.info("Creating combined info metadata...")
    
    # Use the first dataset as the base for metadata
    first_dataset = datasets[0]
    
    # Start with the first dataset's info
    combined_info = dict(first_dataset.meta.info)
    
    # Update totals
    combined_info.update({
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
    })
    
    # Update chunking settings if provided
    if data_file_size_in_mb is not None:
        combined_info["data_files_size_in_mb"] = data_file_size_in_mb
    if video_file_size_in_mb is not None:
        combined_info["video_files_size_in_mb"] = video_file_size_in_mb
    if chunk_size is not None:
        combined_info["chunks_size"] = chunk_size
    
    # Update splits to cover all episodes
    combined_info["splits"] = {
        "train": f"0:{total_episodes}"
    }
    
    # Validate that all datasets have compatible features
    _validate_compatible_features(datasets)
    
    # Features should be the same across all datasets (already validated)
    # Keep the features from the first dataset
    logging.info("Features validated as compatible across all datasets")
    
    # Write the combined info
    write_info(combined_info, output_path)
    
    logging.info(f"Combined info metadata created successfully")
    logging.info(f"  - Total episodes: {total_episodes}")
    logging.info(f"  - Total frames: {total_frames}")
    logging.info(f"  - Total tasks: {total_tasks}")
    logging.info(f"  - Robot type: {combined_info.get('robot_type', 'N/A')}")
    logging.info(f"  - FPS: {combined_info.get('fps', 'N/A')}")
    
    return combined_info


def _validate_compatible_features(datasets: List[Any]) -> None:
    """
    Validate that all datasets have compatible features.
    
    Args:
        datasets: List of LeRobotDataset objects
        
    Raises:
        ValueError: If datasets have incompatible features
    """
    if not datasets:
        return
    
    first_dataset = datasets[0]
    first_features = first_dataset.meta.info.get("features", {})
    
    for i, dataset in enumerate(datasets[1:], 1):
        dataset_features = dataset.meta.info.get("features", {})
        
        # Check if features are the same
        if first_features != dataset_features:
            logging.error(f"Dataset {i+1} has incompatible features")
            logging.error(f"First dataset features: {list(first_features.keys())}")
            logging.error(f"Dataset {i+1} features: {list(dataset_features.keys())}")
            raise ValueError(f"Dataset {i+1} has incompatible features")
        
        # Check other compatibility requirements
        if dataset.fps != first_dataset.fps:
            raise ValueError(f"Dataset {i+1} has incompatible FPS: {dataset.fps} vs {first_dataset.fps}")
        
        if dataset.meta.robot_type != first_dataset.meta.robot_type:
            raise ValueError(f"Dataset {i+1} has incompatible robot type: {dataset.meta.robot_type} vs {first_dataset.meta.robot_type}")
    
    logging.info("All datasets validated as compatible ✓")


def update_info_totals(
    info_dict: Dict[str, Any],
    total_episodes: int,
    total_frames: int,
    total_tasks: int
) -> Dict[str, Any]:
    """
    Update info dictionary with new totals.
    
    Args:
        info_dict: Info dictionary to update
        total_episodes: Total number of episodes
        total_frames: Total number of frames
        total_tasks: Total number of tasks
        
    Returns:
        Updated info dictionary
    """
    updated_info = dict(info_dict)
    updated_info.update({
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
    })
    
    # Update splits to cover all episodes
    updated_info["splits"] = {
        "train": f"0:{total_episodes}"
    }
    
    return updated_info