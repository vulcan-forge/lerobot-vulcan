import logging
import json
from pathlib import Path
from typing import Dict, List, Any

from lerobot.datasets.utils import write_stats


def combine_stats_metadata(datasets: List[Any], output_path: Path) -> None:
    """
    Combine statistics metadata from multiple datasets.
    
    Args:
        datasets: List of LeRobotDataset objects
        output_path: Output directory path
    """
    all_dataset_stats = []
    
    for dataset in datasets:
        # Load dataset stats
        dataset_stats = _load_dataset_stats(dataset.meta.root)
        if dataset_stats:
            all_dataset_stats.append(dataset_stats)
    
    # Combine and write stats
    if all_dataset_stats:
        combined_stats = _combine_dataset_stats(all_dataset_stats)
        write_stats(combined_stats, output_path)
    else:
        # Create empty stats if no dataset stats were loaded
        empty_stats = {}
        write_stats(empty_stats, output_path)
    
    logging.info("✓ Statistics metadata combined successfully")


def _load_dataset_stats(dataset_root: Path) -> Dict[str, Any]:
    """Load dataset statistics from the dataset."""
    stats_path = dataset_root / "meta" / "stats.json"
    if stats_path.exists():
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            logging.debug(f"Loaded stats from {stats_path}")
            return stats
        except Exception as e:
            logging.warning(f"Failed to load stats from {stats_path}: {e}")
            return {}
    else:
        logging.debug(f"No stats file found at {stats_path}")
        return {}


def _combine_dataset_stats(dataset_stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine statistics from multiple datasets."""
    if not dataset_stats_list:
        return {}
    
    if len(dataset_stats_list) == 1:
        return dataset_stats_list[0]
    
    combined_stats = {}
    
    # Get all unique keys across all datasets
    all_keys = set()
    for stats in dataset_stats_list:
        all_keys.update(stats.keys())
    
    for key in all_keys:
        # Collect all values for this key across datasets
        key_values = []
        for stats in dataset_stats_list:
            if key in stats:
                key_values.append(stats[key])
        
        if not key_values:
            continue
            
        # Combine statistics for this key
        combined_key_stats = {}
        
        # Get all sub-keys (min, max, mean, std, count)
        sub_keys = set()
        for values in key_values:
            sub_keys.update(values.keys())
        
        for sub_key in sub_keys:
            sub_values = []
            for values in key_values:
                if sub_key in values:
                    sub_values.append(values[sub_key])
            
            if sub_key == "min":
                # For min, take the minimum across all datasets
                combined_key_stats[sub_key] = _combine_min_values(sub_values)
            elif sub_key == "max":
                # For max, take the maximum across all datasets
                combined_key_stats[sub_key] = _combine_max_values(sub_values)
            elif sub_key == "mean":
                # For mean, compute weighted average based on counts
                combined_key_stats[sub_key] = _combine_mean_values(sub_values, key_values)
            elif sub_key == "std":
                # For std, compute combined standard deviation
                combined_key_stats[sub_key] = _combine_std_values(sub_values, key_values)
            elif sub_key == "count":
                # For count, sum all counts
                combined_key_stats[sub_key] = _combine_count_values(sub_values)
        
        combined_stats[key] = combined_key_stats
    
    return combined_stats


def _combine_min_values(values_list: List[List]) -> List:
    """Combine min values by taking the minimum across datasets."""
    if not values_list:
        return []
    
    # Check if this is nested image data (like observation.images.* with shape [height, width, channels])
    # vs flat data (like action with 16 dimensions)
    is_nested_image_data = (
        isinstance(values_list[0], list) and 
        len(values_list[0]) > 0 and 
        isinstance(values_list[0][0], list) and
        len(values_list[0][0]) > 0 and
        isinstance(values_list[0][0][0], list)
    )
    
    if is_nested_image_data:
        # This is nested image data (e.g., observation.images.*)
        result = []
        for i in range(len(values_list[0])):
            if isinstance(values_list[0][i], list):
                # Handle nested lists
                nested_result = []
                for j in range(len(values_list[0][i])):
                    if isinstance(values_list[0][i][j], list):
                        # Handle doubly nested lists
                        doubly_nested_result = []
                        for k in range(len(values_list[0][i][j])):
                            min_val = min(values[i][j][k] for values in values_list 
                                        if i < len(values) and j < len(values[i]) and k < len(values[i][j]))
                            doubly_nested_result.append(min_val)
                        nested_result.append(doubly_nested_result)
                    else:
                        # Handle single values in nested lists
                        min_val = min(values[i][j] for values in values_list 
                                    if i < len(values) and j < len(values[i]))
                        nested_result.append(min_val)
                result.append(nested_result)
            else:
                # Handle single values
                min_val = min(values[i] for values in values_list if i < len(values))
                result.append(min_val)
        return result
    else:
        # This is flat data (like action, observation.state) - preserve the structure
        result = []
        for i in range(len(values_list[0])):
            min_val = min(values[i] for values in values_list if i < len(values))
            result.append(min_val)
        return result


def _combine_max_values(values_list: List[List]) -> List:
    """Combine max values by taking the maximum across datasets."""
    if not values_list:
        return []
    
    # Check if this is nested image data (like observation.images.* with shape [height, width, channels])
    # vs flat data (like action with 16 dimensions)
    is_nested_image_data = (
        isinstance(values_list[0], list) and 
        len(values_list[0]) > 0 and 
        isinstance(values_list[0][0], list) and
        len(values_list[0][0]) > 0 and
        isinstance(values_list[0][0][0], list)
    )
    
    if is_nested_image_data:
        # This is nested image data (e.g., observation.images.*)
        result = []
        for i in range(len(values_list[0])):
            if isinstance(values_list[0][i], list):
                # Handle nested lists
                nested_result = []
                for j in range(len(values_list[0][i])):
                    if isinstance(values_list[0][i][j], list):
                        # Handle doubly nested lists
                        doubly_nested_result = []
                        for k in range(len(values_list[0][i][j])):
                            max_val = max(values[i][j][k] for values in values_list 
                                        if i < len(values) and j < len(values[i]) and k < len(values[i][j]))
                            doubly_nested_result.append(max_val)
                        nested_result.append(doubly_nested_result)
                    else:
                        # Handle single values in nested lists
                        max_val = max(values[i][j] for values in values_list 
                                    if i < len(values) and j < len(values[i]))
                        nested_result.append(max_val)
                result.append(nested_result)
            else:
                # Handle single values
                max_val = max(values[i] for values in values_list if i < len(values))
                result.append(max_val)
        return result
    else:
        # This is flat data (like action, observation.state) - preserve the structure
        result = []
        for i in range(len(values_list[0])):
            max_val = max(values[i] for values in values_list if i < len(values))
            result.append(max_val)
        return result


def _combine_mean_values(mean_values: List[List], all_stats: List[Dict]) -> List:
    """Combine mean values using weighted average based on counts."""
    if not mean_values or not all_stats:
        return []
    
    # Get counts for weighting
    counts = []
    for stats in all_stats:
        if "count" in stats:
            counts.append(stats["count"])
        else:
            counts.append([1])  # Default count of 1
    
    # Compute weighted average
    total_count = sum(count[0] if isinstance(count, list) else count for count in counts)
    if total_count == 0:
        return mean_values[0] if mean_values else []
    
    # Check if this is nested image data (like observation.images.* with shape [height, width, channels])
    # vs flat data (like action with 16 dimensions)
    is_nested_image_data = (
        isinstance(mean_values[0], list) and 
        len(mean_values[0]) > 0 and 
        isinstance(mean_values[0][0], list) and
        len(mean_values[0][0]) > 0 and
        isinstance(mean_values[0][0][0], list)
    )
    
    if is_nested_image_data:
        # This is nested image data (e.g., observation.images.*)
        result = []
        for i in range(len(mean_values[0])):
            if isinstance(mean_values[0][i], list):
                # Handle nested lists
                nested_result = []
                for j in range(len(mean_values[0][i])):
                    if isinstance(mean_values[0][i][j], list):
                        # Handle doubly nested lists
                        doubly_nested_result = []
                        for k in range(len(mean_values[0][i][j])):
                            weighted_sum = 0
                            for l, mean_val in enumerate(mean_values):
                                count = counts[l][0] if isinstance(counts[l], list) else counts[l]
                                if i < len(mean_val) and j < len(mean_val[i]) and k < len(mean_val[i][j]):
                                    weighted_sum += mean_val[i][j][k] * count
                            doubly_nested_result.append(weighted_sum / total_count)
                        nested_result.append(doubly_nested_result)
                    else:
                        # Handle single values in nested lists
                        weighted_sum = 0
                        for l, mean_val in enumerate(mean_values):
                            count = counts[l][0] if isinstance(counts[l], list) else counts[l]
                            if i < len(mean_val) and j < len(mean_val[i]):
                                weighted_sum += mean_val[i][j] * count
                        nested_result.append(weighted_sum / total_count)
                result.append(nested_result)
            else:
                # Handle single values
                weighted_sum = 0
                for l, mean_val in enumerate(mean_values):
                    count = counts[l][0] if isinstance(counts[l], list) else counts[l]
                    if i < len(mean_val):
                        weighted_sum += mean_val[i] * count
                result.append(weighted_sum / total_count)
        return result
    else:
        # This is flat data (like action, observation.state) - preserve the structure
        result = []
        for i in range(len(mean_values[0])):
            weighted_sum = 0
            for j, mean_val in enumerate(mean_values):
                count = counts[j][0] if isinstance(counts[j], list) else counts[j]
                if i < len(mean_val):
                    weighted_sum += mean_val[i] * count
            result.append(weighted_sum / total_count)
        return result


def _combine_std_values(std_values: List[List], all_stats: List[Dict]) -> List:
    """Combine standard deviation values."""
    # For simplicity, we'll use the maximum std across datasets
    # A more sophisticated approach would compute the combined std using pooled variance
    if not std_values:
        return []
    
    # Check if this is nested image data (like observation.images.* with shape [height, width, channels])
    # vs flat data (like action with 16 dimensions)
    is_nested_image_data = (
        isinstance(std_values[0], list) and 
        len(std_values[0]) > 0 and 
        isinstance(std_values[0][0], list) and
        len(std_values[0][0]) > 0 and
        isinstance(std_values[0][0][0], list)
    )
    
    if is_nested_image_data:
        # This is nested image data (e.g., observation.images.*)
        result = []
        for i in range(len(std_values[0])):
            if isinstance(std_values[0][i], list):
                # Handle nested lists
                nested_result = []
                for j in range(len(std_values[0][i])):
                    if isinstance(std_values[0][i][j], list):
                        # Handle doubly nested lists
                        doubly_nested_result = []
                        for k in range(len(std_values[0][i][j])):
                            max_std = max(values[i][j][k] for values in std_values 
                                        if i < len(values) and j < len(values[i]) and k < len(values[i][j]))
                            doubly_nested_result.append(max_std)
                        nested_result.append(doubly_nested_result)
                    else:
                        # Handle single values in nested lists
                        max_std = max(values[i][j] for values in std_values 
                                    if i < len(values) and j < len(values[i]))
                        nested_result.append(max_std)
                result.append(nested_result)
            else:
                # Handle single values
                max_std = max(values[i] for values in std_values if i < len(values))
                result.append(max_std)
        return result
    else:
        # This is flat data (like action, observation.state) - preserve the structure
        result = []
        for i in range(len(std_values[0])):
            max_std = max(std_val[i] for std_val in std_values if i < len(std_val))
            result.append(max_std)
        return result


def _combine_count_values(count_values: List[List]) -> List:
    """Combine count values by summing them."""
    if not count_values:
        return []
    
    total_count = sum(count[0] if isinstance(count, list) else count for count in count_values)
    return [total_count]
