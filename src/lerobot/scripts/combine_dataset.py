#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi

from lerobot.datasets.factory import combine_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import create_lerobot_dataset_card
from lerobot.utils.utils import init_logging


def main():
    parser = argparse.ArgumentParser(description="Combine multiple LeRobot datasets into a single dataset")
    parser.add_argument(
        "--repo_ids",
        type=str,
        nargs="+",
        required=True,
        help="List of Hugging Face repository IDs containing LeRobotDatasets to combine (e.g., `local/sourccey_v1beta_towel_010_a local/sourccey_v1beta_towel_010_b`).",
    )
    parser.add_argument(
        "--output_repo_id",
        type=str,
        required=True,
        help="Name of the output Hugging Face repository for the combined dataset (e.g., `local/sourccey_v1beta_towel_010_a_b_combined`).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally. By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--push_to_hub",
        type=int,
        default=1,
        help="Upload to Hugging Face hub.",
    )
    parser.add_argument(
        "--private",
        type=int,
        default=0,
        help="If set, the repository on the Hub will be private",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        help="List of tags to apply to the dataset on the Hub",
    )
    parser.add_argument(
        "--license",
        type=str,
        default=None,
        help="License to use for the dataset on the Hub"
    )
    args = parser.parse_args()

    # Load all datasets
    logging.info(f"Loading {len(args.repo_ids)} datasets...")
    datasets = []
    total_episodes = 0
    for repo_id in args.repo_ids:
        dataset = LeRobotDataset(repo_id=repo_id, root=args.root)
        datasets.append(dataset)
        total_episodes += dataset.meta.total_episodes
        logging.info(f"Loaded {repo_id}: {dataset.meta.total_episodes} episodes")

    # Combine datasets
    logging.info(f"Combining {len(datasets)} datasets into {args.output_repo_id}...")
    combined_dataset = combine_datasets(datasets, root=args.root, repo_id=args.output_repo_id)
    logging.info(f"Successfully combined datasets into {args.output_repo_id}")
    logging.info(f"Total episodes: {combined_dataset.meta.total_episodes}")

    if args.push_to_hub:
        logging.info(f"Pushing dataset to Hugging Face Hub: {args.output_repo_id}")
        # TODO: Implement push to hub functionality
        pass

if __name__ == "__main__":
    init_logging()
    main()
