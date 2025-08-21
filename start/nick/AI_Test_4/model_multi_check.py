#!/usr/bin/env python3
"""
Model Check Function for ACT Policy

This script loads data from a local dataset directory and uses an ACT policy
to predict actions from the loaded observations, following the same pattern
as the recording code.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Import LeRobot components
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.control_utils import predict_action
from lerobot.datasets.utils import build_dataset_frame
from lerobot.utils.utils import get_safe_torch_device

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_act_policy(policy_path: str, device: str = "cpu") -> ACTPolicy:
    """
    Load an ACT policy from a local path.

    Args:
        policy_path: Path to the directory containing the policy files
        device: Device to load the policy on ('cpu' or 'cuda')

    Returns:
        Loaded ACT policy
    """
    try:
        logger.info(f"Loading ACT policy from: {policy_path}")

        # Load the policy directly - it will handle the config internally
        policy = ACTPolicy.from_pretrained(policy_path)

        # Move to the specified device
        policy.to(device)

        logger.info(f"Successfully loaded ACT policy on device: {device}")
        logger.info(f"Policy input features: {list(policy.config.input_features.keys())}")
        logger.info(f"Policy output features: {list(policy.config.output_features.keys())}")

        # Add detailed logging of input/output shapes
        logger.info("Policy input shapes:")
        for key, feature in policy.config.input_features.items():
            logger.info(f"  {key}: {feature.shape}")

        logger.info("Policy output shapes:")
        for key, feature in policy.config.output_features.items():
            logger.info(f"  {key}: {feature.shape}")

        # Also log the normalization mapping to understand what gets normalized
        logger.info("Policy normalization mapping:")
        for key, mode in policy.config.normalization_mapping.items():
            logger.info(f"  {key}: {mode}")

        return policy

    except Exception as e:
        logger.error(f"Failed to load ACT policy: {e}")
        raise

def load_dataset_from_directory(directory_path: str) -> LeRobotDataset:
    """
    Load a dataset from a local directory.

    Args:
        directory_path: Path to the dataset directory

    Returns:
        Loaded LeRobotDataset
    """
    try:
        logger.info(f"Loading dataset from: {directory_path}")

        # For local loading, we need to go up one level from the chunk directory
        # to the main dataset directory that contains the meta/ folder
        dataset_root = Path(directory_path).parent.parent

        # Extract the dataset name from the path
        # The path structure is: .../sourccey-002__stiction_tape-test-a__set001__chrism/data/chunk-000
        # We want: sourccey-002__stiction_tape-test-a__set001__chrism
        dataset_name = dataset_root.name

        # Load the dataset
        dataset = LeRobotDataset(
            repo_id=dataset_name,
            root=dataset_root
        )

        logger.info(f"Successfully loaded dataset with {len(dataset)} frames")
        logger.info(f"Dataset features: {list(dataset.features.keys())}")
        logger.info(f"Dataset FPS: {dataset.fps}")

        return dataset

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

def predict_multiple_actions(policy, dataset, num_actions: int = 100, start_frame: int = 0):
    """
    Predict multiple actions from consecutive frames in the dataset.

    Args:
        policy: Loaded ACT policy
        dataset: LeRobotDataset instance
        num_actions: Number of actions to predict
        start_frame: Starting frame index

    Returns:
        List of predicted actions
    """
    actions = []
    device = get_safe_torch_device(policy.config.device)

    logger.info(f"Predicting {num_actions} actions starting from frame {start_frame}")

    for i in range(num_actions):
        frame_idx = start_frame + i

        # Check if we have enough frames
        if frame_idx >= len(dataset):
            logger.warning(f"Reached end of dataset at frame {frame_idx}, stopping prediction")
            break

        try:
            # Load observation for current frame
            observation = dataset[frame_idx]

            # Build observation frame for the policy
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

            # Convert observation frame tensors to numpy arrays for predict_action
            numpy_observation = {}
            for k, v in observation_frame.items():
                if torch.is_tensor(v):
                    # For images, we need to convert from [C, H, W] to [H, W, C] format
                    if "image" in k:
                        # Convert from [C, H, W] to [H, W, C] for numpy conversion
                        v = v.permute(1, 2, 0)
                    numpy_observation[k] = v.cpu().numpy()
                else:
                    numpy_observation[k] = v

            # Predict action using the policy
            action = predict_action(
                numpy_observation,
                policy,
                device,
                use_amp=False,
                task="Grab the tape and put it in the cup",
                robot_type="sourccey"
            )

            actions.append({
                'frame': frame_idx,
                'action': action,
                'observation_keys': list(observation.keys())
            })

            # Print progress every 10 actions
            if (i + 1) % 10 == 0:
                logger.info(f"Predicted {i + 1}/{num_actions} actions")

        except Exception as e:
            logger.error(f"Failed to predict action for frame {frame_idx}: {e}")
            actions.append({
                'frame': frame_idx,
                'action': None,
                'error': str(e)
            })

    return actions

def print_actions_summary(actions):
    """
    Print a summary of the predicted actions.

    Args:
        actions: List of action dictionaries
    """
    logger.info("\n" + "="*80)
    logger.info("PREDICTED ACTIONS SUMMARY")
    logger.info("="*80)

    successful_predictions = [a for a in actions if a['action'] is not None]
    failed_predictions = [a for a in actions if a['action'] is None]

    logger.info(f"Total predictions: {len(actions)}")
    logger.info(f"Successful predictions: {len(successful_predictions)}")
    logger.info(f"Failed predictions: {len(failed_predictions)}")

    if successful_predictions:
        logger.info("\n" + "-"*80)
        logger.info("ALL PREDICTED ACTIONS (100 frames)")
        logger.info("-"*80)

        # Print all 100 frames in a nicely formatted table
        for i, action_data in enumerate(successful_predictions):
            frame_num = action_data['frame']
            action = action_data['action']

            # Format the action values nicely
            if hasattr(action, 'cpu'):  # If it's a tensor
                action_values = action.cpu().numpy()
            else:
                action_values = np.array(action)

            # Print frame header
            logger.info(f"\n{'='*60}")
            logger.info(f"FRAME {frame_num:3d} (Action {i+1:3d}/100)")
            logger.info(f"{'='*60}")

            # Print action shape and values
            logger.info(f"Action Shape: {action_values.shape}")
            logger.info(f"Action Values:")

            # Format action values in a readable way
            if len(action_values.shape) == 1:
                # 1D array - print in rows of 6 for readability
                for j in range(0, len(action_values), 6):
                    row_values = action_values[j:j+6]
                    row_str = "  ".join([f"{val:8.3f}" for val in row_values])
                    logger.info(f"  [{j:2d}:{j+len(row_values)-1:2d}] {row_str}")
            else:
                # Multi-dimensional - print as is
                logger.info(f"  {action_values}")

            # Print observation keys (truncated if too long)
            obs_keys = action_data['observation_keys']
            if len(obs_keys) > 8:
                obs_keys_str = ", ".join(obs_keys[:8]) + f" ... (+{len(obs_keys)-8} more)"
            else:
                obs_keys_str = ", ".join(obs_keys)

            logger.info(f"Observation Keys: {obs_keys_str}")

            # Add a small separator between frames
            if i < len(successful_predictions) - 1:  # Don't add separator after last frame
                logger.info("-" * 40)

    if failed_predictions:
        logger.info("\n" + "="*80)
        logger.info("FAILED PREDICTIONS")
        logger.info("="*80)
        for action_data in failed_predictions:
            logger.info(f"Frame {action_data['frame']}: {action_data['error']}")

    # Print statistics for successful actions
    if successful_predictions:
        logger.info("\n" + "="*80)
        logger.info("ACTION STATISTICS")
        logger.info("="*80)

        action_arrays = []
        for a in successful_predictions:
            if hasattr(a['action'], 'cpu'):
                action_arrays.append(a['action'].cpu().numpy())
            else:
                action_arrays.append(np.array(a['action']))

        action_array = np.array(action_arrays)

        logger.info(f"Action Array Shape: {action_array.shape}")
        logger.info(f"Mean Action Values: {np.mean(action_array, axis=0)}")
        logger.info(f"Std Action Values:  {np.std(action_array, axis=0)}")
        logger.info(f"Min Action Values:  {np.min(action_array, axis=0)}")
        logger.info(f"Max Action Values:  {np.max(action_array, axis=0)}")

        # Print per-dimension statistics in a table format
        logger.info(f"\nPer-Dimension Statistics:")
        logger.info(f"{'Dim':>3} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        logger.info(f"{'-'*50}")
        for dim in range(action_array.shape[1]):
            mean_val = np.mean(action_array[:, dim])
            std_val = np.std(action_array[:, dim])
            min_val = np.min(action_array[:, dim])
            max_val = np.max(action_array[:, dim])
            logger.info(f"{dim:3d} {mean_val:10.3f} {std_val:10.3f} {min_val:10.3f} {max_val:10.3f}")

def main():
    # Paths - you'll need to update these to your actual paths
    dataset_directory = r"C:\Users\Nicholas\.cache\huggingface\lerobot\local\sourccey-001__ai_test_6_shoulder-fast_chrism_combined\data\chunk-000"
    num_actions = 250  # Number of actions to predict
    start_frame = 0    # Starting frame index

    try:
        # Load the dataset
        dataset = load_dataset_from_directory(dataset_directory)

        # Load the ACT policy
        logger.info("Loading ACT policy...")
        policy_path = "outputs/train/act__sourccey-001__ai_test_6_shoulder-fast_chrism_combined/checkpoints/040000/pretrained_model"
        policy = load_act_policy(policy_path, device="cuda")

        # Predict multiple actions
        actions = predict_multiple_actions(policy, dataset, num_actions, start_frame)

        # Print summary of all actions
        print_actions_summary(actions)

    except Exception as e:
        logger.error(f"Failed to complete action prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
