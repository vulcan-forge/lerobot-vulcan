#!/usr/bin/env python3
"""
Model Check Function for ACT Policy

This script loads data from a local dataset directory and uses an ACT policy
to predict actions from the loaded observations.
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
        # The path structure is: .../sourccey_v3beta-002__stiction_tape-test-a__set001__chrism/data/chunk-000
        # We want: sourccey_v3beta-002__stiction_tape-test-a__set001__chrism
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

def prepare_observation_from_dataset(dataset: LeRobotDataset, frame_index: int = 0) -> Dict[str, np.ndarray]:
    """
    Prepare an observation dictionary from a dataset frame.

    Args:
        dataset: The loaded dataset
        frame_index: Index of the frame to load

    Returns:
        Dictionary containing the observation data
    """
    try:
        logger.info(f"Loading frame {frame_index} from dataset")

        # Get the frame data
        frame_data = dataset[frame_index]

        # Convert to numpy arrays and prepare observation dictionary
        observation = {}

        for key, value in frame_data.items():
            if isinstance(value, torch.Tensor):
                # Convert torch tensor to numpy array
                observation[key] = value.numpy()
            elif isinstance(value, np.ndarray):
                # Keep numpy arrays as is
                observation[key] = value
            # Skip non-array values (like strings, paths, etc.)

        logger.info(f"Loaded observation with keys: {list(observation.keys())}")

        # Log shapes of key components
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                logger.info(f"  {key}: shape {value.shape}, dtype {value.dtype}")

        return observation

    except Exception as e:
        logger.error(f"Failed to prepare observation: {e}")
        raise

def run_model_check(
    dataset_directory: str,
    policy_path: str,
    device: str = "cpu",
    frame_index: int = 0,
    use_amp: bool = False
) -> Dict[str, Any]:
    """
    Main function to run the model check.

    Args:
        dataset_directory: Path to the dataset directory
        policy_path: Path to the policy directory
        device: Device to run inference on ('cpu' or 'cuda')
        frame_index: Index of the frame to test
        use_amp: Whether to use automatic mixed precision

    Returns:
        Dictionary containing the results
    """
    logger.info("=" * 60)
    logger.info("Starting Model Check")
    logger.info("=" * 60)

    # try:
    #     # Load the dataset
    #     dataset = load_dataset_from_directory(dataset_directory)

    #     # Load the ACT policy
    #     policy = load_act_policy(policy_path, device)

    #     # Prepare observation from dataset
    #     observation = prepare_observation_from_dataset(dataset, frame_index)

    #     # Convert device string to torch device
    #     torch_device = torch.device(device)

    #     logger.info("=" * 60)
    #     logger.info("Running Action Prediction")
    #     logger.info("=" * 60)

    #     # Use predict_action function to get action
    #     action = predict_action(
    #         observation=observation,
    #         policy=policy,
    #         device=torch_device,
    #         use_amp=use_amp,
    #         task="Grab the tape and put it in the cup",  # No specific task for this check
    #         robot_type="sourccey_v3beta"  # Based on the dataset name
    #     )

    #     # Convert action to numpy for easier handling
    #     if isinstance(action, torch.Tensor):
    #         action_np = action.numpy()
    #     else:
    #         action_np = action

    #     logger.info(f"Successfully predicted action with shape: {action_np.shape}")
    #     logger.info(f"Action values: {action_np}")

    #     # Prepare results
    #     results = {
    #         "success": True,
    #         "dataset_info": {
    #             "num_frames": len(dataset),
    #             "features": list(dataset.features.keys()),
    #             "fps": dataset.fps
    #         },
    #         "policy_info": {
    #             "type": "act",
    #             "device": device,
    #             "input_features": list(policy.config.input_features.keys()),
    #             "output_features": list(policy.config.output_features.keys())
    #         },
    #         "observation_info": {
    #             "keys": list(observation.keys()),
    #             "shapes": {k: v.shape if hasattr(v, 'shape') else str(type(v))
    #                       for k, v in observation.items()}
    #         },
    #         "action_info": {
    #             "shape": action_np.shape,
    #             "values": action_np.tolist(),
    #             "dtype": str(action_np.dtype)
    #         }
    #     }

    #     logger.info("=" * 60)
    #     logger.info("Model Check Completed Successfully!")
    #     logger.info("=" * 60)

    #     return results

    # except Exception as e:
    #     logger.error(f"Model check failed: {e}")
    #     import traceback
    #     traceback.print_exc()

    #     return {
    #         "success": False,
    #         "error": str(e),
    #         "traceback": traceback.format_exc()
    #     }

def main():
    # Paths - you'll need to update these to your actual paths
    dataset_directory = r"C:\Users\Nicholas\.cache\huggingface\lerobot\local\sourccey_v3beta-002__stiction_tape-test-a__set001__chrism\data\chunk-000"  # Update this
    frame_index = 0  # Start with first frame

    try:
        # Load the dataset
        dataset = load_dataset_from_directory(dataset_directory)

        # Prepare observation from dataset
        observation = prepare_observation_from_dataset(dataset, frame_index)
        print(f"observation.state: {observation['observation.state']}")

        # For now, let's just verify we can load the policy without running inference
        logger.info("Loading ACT policy to verify it works...")
        policy_path = "outputs/train/act__sourccey_v3beta-002__stiction_tape-test-a__set001__chrism/checkpoints/020000/pretrained_model"
        policy = load_act_policy(policy_path, device="cuda")
        logger.info("Policy loaded successfully!")

    except Exception as e:
        logger.error(f"Failed to load dataset or observation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
