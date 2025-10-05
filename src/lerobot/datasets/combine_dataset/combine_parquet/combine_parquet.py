import logging
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

from lerobot.datasets.utils import (
    load_episodes,
    DEFAULT_DATA_PATH,
    update_chunk_file_indices,
    get_parquet_file_size_in_mb,
    to_parquet_with_hf_images,
)


def combine_parquet_files(
    datasets: List[Any],
    output_path: Path,
    chunk_size: int,
    data_file_size_in_mb: int,
    image_keys: List[str]
) -> None:
    """
    Combine parquet data files with proper size-aware chunking.
    
    Args:
        datasets: List of LeRobotDataset objects
        output_path: Output directory path
        chunk_size: Maximum number of files per chunk
        data_file_size_in_mb: Maximum size for data files in MB
        image_keys: List of image keys that require special handling
    """
    logging.info("Combining parquet data files with proper chunking...")
    
    chunk_idx, file_idx = 0, 0
    current_df = None
    episode_offset = 0
    frame_offset = 0
    
    for dataset_idx, dataset in enumerate(datasets):
        # Log progress every 100 datasets
        if dataset_idx % 100 == 0:            
            logging.info(f"Processing parquet files for datasets {dataset_idx + 1} - {len(datasets)}")
        
        data_dir = dataset.meta.root / "data"
        if not data_dir.exists():
            logging.warning(f"Data directory not found for dataset {dataset_idx + 1}: {data_dir}")
            continue
        
        parquet_files = list(data_dir.rglob("*.parquet"))
        if not parquet_files:
            logging.warning(f"No parquet files found in dataset {dataset_idx + 1}: {data_dir}")
            continue
        
        for parquet_file in sorted(parquet_files):
            try:
                df = pd.read_parquet(parquet_file)
                logging.debug(f"Loaded parquet file {parquet_file} with {len(df)} rows")
                
                # Update episode indices to be consecutive across datasets
                if 'episode_index' in df.columns:
                    df['episode_index'] = df['episode_index'] + episode_offset
                
                # Update frame indices to be consecutive across datasets
                if 'index' in df.columns:
                    df['index'] = df['index'] + frame_offset
                
                # Concatenate with current dataframe
                if current_df is None:
                    current_df = df
                else:
                    current_df = pd.concat([current_df, df], ignore_index=True)
                
                # Check if we need to write current file and start a new one
                current_size_mb = get_parquet_file_size_in_mb(parquet_file)
                if current_size_mb >= data_file_size_in_mb:
                    # Write current file
                    _write_data_file(current_df, output_path, chunk_idx, file_idx, image_keys)

                    # Move to next file
                    chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunk_size)
                    current_df = None
                    
            except Exception as e:
                logging.error(f"Error processing parquet file {parquet_file}: {e}")
                continue
        
        # Update offsets for next dataset
        episode_offset += dataset.meta.total_episodes
        frame_offset += dataset.meta.total_frames

    # Write remaining data if any
    if current_df is not None:
        _write_data_file(current_df, output_path, chunk_idx, file_idx, image_keys)

    logging.info("✓ Parquet files combined successfully")

def _write_data_file(df: pd.DataFrame, output_path: Path, chunk_idx: int, file_idx: int, image_keys: List[str]) -> None:
    """Write a data file with proper handling for images."""
    path = output_path / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if len(image_keys) > 0:
            to_parquet_with_hf_images(df, path)
        else:
            df.to_parquet(path, index=False)
    except Exception as e:
        logging.error(f"Error writing data file {path}: {e}")
        raise


def combine_parquet_files_with_episode_mapping(
    datasets: List[Any],
    output_path: Path,
    chunk_size: int,
    data_file_size_in_mb: int,
    image_keys: List[str],
    episode_mapping: Dict[int, int] = None
) -> None:
    """
    Combine parquet data files with explicit episode mapping.
    
    Args:
        datasets: List of LeRobotDataset objects
        output_path: Output directory path
        chunk_size: Maximum number of files per chunk
        data_file_size_in_mb: Maximum size for data files in MB
        image_keys: List of image keys that require special handling
        episode_mapping: Optional mapping from old episode indices to new ones
    """
    logging.info("Combining parquet data files with episode mapping...")
    
    chunk_idx, file_idx = 0, 0
    current_df = None
    frame_offset = 0
    
    for dataset_idx, dataset in enumerate(datasets):
        # Log progress every 100 datasets
        if dataset_idx % 100 == 0:            
            logging.info(f"Processing parquet files for datasets {dataset_idx + 1} - {len(datasets)}")
        
        data_dir = dataset.meta.root / "data"
        if not data_dir.exists():
            logging.warning(f"Data directory not found for dataset {dataset_idx + 1}: {data_dir}")
            continue
        
        parquet_files = list(data_dir.rglob("*.parquet"))
        if not parquet_files:
            logging.warning(f"No parquet files found in dataset {dataset_idx + 1}: {data_dir}")
            continue
        
        for parquet_file in sorted(parquet_files):
            try:
                df = pd.read_parquet(parquet_file)
                
                # Apply episode mapping if provided
                if episode_mapping and 'episode_index' in df.columns:
                    df['episode_index'] = df['episode_index'].map(episode_mapping)
                
                # Update frame indices
                if 'index' in df.columns:
                    df['index'] = df['index'] + frame_offset
                
                # Concatenate with current dataframe
                if current_df is None:
                    current_df = df
                else:
                    current_df = pd.concat([current_df, df], ignore_index=True)
                
                # Check if we need to write current file and start a new one
                current_size_mb = get_parquet_file_size_in_mb(parquet_file)
                if current_size_mb >= data_file_size_in_mb:
                    _write_data_file(current_df, output_path, chunk_idx, file_idx, image_keys)
                    chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunk_size)
                    current_df = None
                    
            except Exception as e:
                logging.error(f"Error processing parquet file {parquet_file}: {e}")
                continue
        
        # Update frame offset for next dataset
        frame_offset += dataset.meta.total_frames
    
    # Write remaining data if any
    if current_df is not None:
        _write_data_file(current_df, output_path, chunk_idx, file_idx, image_keys)
    
    logging.info("Parquet files combined successfully with episode mapping")


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


def validate_parquet_data(df: pd.DataFrame, expected_columns: List[str] = None) -> bool:
    """
    Validate parquet data structure.
    
    Args:
        df: DataFrame to validate
        expected_columns: List of expected column names
        
    Returns:
        True if valid, False otherwise
    """
    if df is None or len(df) == 0:
        logging.warning("DataFrame is empty or None")
        return False
    
    if expected_columns:
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            logging.warning(f"Missing expected columns: {missing_columns}")
            return False
    
    # Check for required LeRobot columns
    required_columns = ['episode_index', 'index']
    missing_required = set(required_columns) - set(df.columns)
    if missing_required:
        logging.warning(f"Missing required columns: {missing_required}")
        return False
    
    return True
