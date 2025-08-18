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

def main():
    # Paths - you'll need to update these to your actual paths
    dataset_directory = r"C:\Users\Nicholas\.cache\huggingface\lerobot\local\sourccey_v3beta-002__stiction_tape-test-a__set001__chrism\data\chunk-000"
    frame_index = 300 # Start with first frame

    try:
        # Load the dataset
        dataset = load_dataset_from_directory(dataset_directory)

        # Prepare observation from dataset
        observation = dataset[frame_index]
        logger.info(f"Loaded observation with keys: {list(observation.keys())}")

        # Build the observation frame for the policy
        observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
        logger.info(f"Built observation frame with keys: {list(observation_frame.keys())}")
        logger.info(f"Observation frame state: {observation_frame['observation.state']}")

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

        # Load the ACT policy
        logger.info("Loading ACT policy...")
        policy_path = "outputs/train/act__sourccey_v3beta-002__stiction_tape-test-a__set001__chrism/checkpoints/020000/pretrained_model"
        policy = load_act_policy(policy_path, device="cuda")

        device = get_safe_torch_device(policy.config.device)

        # Predict action using the policy
        action = predict_action(numpy_observation, policy, device, use_amp=False, task="Grab the tape and put it in the cup", robot_type="sourccey_v3beta")
        logger.info(f"Predicted action: {action}")
    except Exception as e:
        logger.error(f"Failed to load dataset or observation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
