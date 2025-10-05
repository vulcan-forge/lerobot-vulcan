import logging
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

from lerobot.datasets.utils import write_tasks


def combine_tasks_metadata(datasets: List[Any], output_path: Path) -> Dict[str, int]:
    """
    Combine tasks metadata from multiple datasets.
    
    Args:
        datasets: List of LeRobotDataset objects
        output_path: Output directory path
        
    Returns:
        Mapping from task strings to task indices
    """    
    if not datasets:
        logging.warning("No datasets provided for task combination")
        return {}
    
    all_tasks = {}
    task_content_to_index = {}
    dataset_task_counts = []
    
    for dataset_idx, dataset in enumerate(datasets):

        # Log progress every 100 datasets
        if dataset_idx % 100 == 0:            
            logging.info(f"Processing tasks for datasets {dataset_idx + 1} - {len(datasets)}")
        
        # Validate that dataset has tasks
        if not hasattr(dataset.meta, 'tasks') or dataset.meta.tasks is None:
            logging.warning(f"Dataset {dataset_idx + 1} has no tasks metadata")
            dataset_task_counts.append(0)
            continue
        
        # Get tasks from this dataset
        dataset_tasks = dataset.meta.tasks
        if hasattr(dataset_tasks, 'index'):
            task_strings = dataset_tasks.index.tolist()
        else:
            # Handle case where tasks might be a different format
            task_strings = list(dataset_tasks.keys()) if isinstance(dataset_tasks, dict) else []
        
        dataset_task_counts.append(len(task_strings))
        
        # Process each task string
        for task_string in task_strings:
            if task_string not in task_content_to_index:
                new_idx = len(task_content_to_index)
                task_content_to_index[task_string] = new_idx
                all_tasks[new_idx] = task_string
                logging.debug(f"Added new task '{task_string}' with index {new_idx}")
            else:
                logging.debug(f"Task '{task_string}' already exists with index {task_content_to_index[task_string]}")
    
    # Validate that we have tasks to combine
    if not all_tasks:
        logging.warning("No tasks found across all datasets")
        # Create empty tasks file
        empty_tasks_df = pd.DataFrame({"task_index": []}, index=[])
        write_tasks(empty_tasks_df, output_path)
        return {}
    
    # Create the combined tasks DataFrame
    tasks_df = pd.DataFrame({"task_index": list(all_tasks.keys())}, index=list(all_tasks.values()))
    
    # Write tasks
    write_tasks(tasks_df, output_path)
    
    logging.info(f"Tasks combination completed successfully")
    
    return task_content_to_index


def validate_task_compatibility(datasets: List[Any]) -> bool:
    """
    Validate that all datasets have compatible task structures.
    
    Args:
        datasets: List of LeRobotDataset objects
        
    Returns:
        True if compatible, False otherwise
    """
    if not datasets:
        return True
    
    # Check that all datasets have the same task structure
    first_dataset = datasets[0]
    if not hasattr(first_dataset.meta, 'tasks') or first_dataset.meta.tasks is None:
        logging.warning("First dataset has no tasks metadata")
        return False
    
    first_tasks = first_dataset.meta.tasks
    
    for i, dataset in enumerate(datasets[1:], 1):
        if not hasattr(dataset.meta, 'tasks') or dataset.meta.tasks is None:
            logging.warning(f"Dataset {i+1} has no tasks metadata")
            continue
        
        dataset_tasks = dataset.meta.tasks
        
        # Check if task structures are compatible
        if hasattr(first_tasks, 'index') and hasattr(dataset_tasks, 'index'):
            # Both are DataFrames with index
            if not isinstance(first_tasks.index, type(dataset_tasks.index)):
                logging.error(f"Dataset {i+1} has incompatible task index type")
                return False
        elif isinstance(first_tasks, dict) and isinstance(dataset_tasks, dict):
            # Both are dictionaries
            pass  # Compatible
        else:
            logging.error(f"Dataset {i+1} has incompatible task structure")
            return False
    
    return True


def get_task_summary(datasets: List[Any]) -> Dict[str, Any]:
    """
    Get a summary of tasks across all datasets.
    
    Args:
        datasets: List of LeRobotDataset objects
        
    Returns:
        Dictionary with task summary information
    """
    summary = {
        "total_datasets": len(datasets),
        "datasets_with_tasks": 0,
        "total_unique_tasks": 0,
        "task_strings": set(),
        "dataset_task_counts": []
    }
    
    for dataset_idx, dataset in enumerate(datasets):
        if not hasattr(dataset.meta, 'tasks') or dataset.meta.tasks is None:
            summary["dataset_task_counts"].append(0)
            continue
        
        dataset_tasks = dataset.meta.tasks
        if hasattr(dataset_tasks, 'index'):
            task_strings = dataset_tasks.index.tolist()
        else:
            task_strings = list(dataset_tasks.keys()) if isinstance(dataset_tasks, dict) else []
        
        summary["datasets_with_tasks"] += 1
        summary["dataset_task_counts"].append(len(task_strings))
        summary["task_strings"].update(task_strings)
    
    summary["total_unique_tasks"] = len(summary["task_strings"])
    summary["task_strings"] = list(summary["task_strings"])  # Convert set to list for JSON serialization
    
    return summary
