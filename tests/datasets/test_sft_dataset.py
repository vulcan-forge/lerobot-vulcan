# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from types import SimpleNamespace

import pytest
import torch

from lerobot.scripts.sourccey.train.configs.sft import SFTDatasetConfig, SFTDatasetSourceConfig
from lerobot.scripts.sourccey.train.datasets.sft import (
    SFT_SOURCE_INDEX,
    SFTMixtureDataset,
    SFTMixtureSampler,
)


class _FakeDataset:
    def __init__(self, repo_id: str, episode_lengths: list[int]):
        self.repo_id = repo_id
        self.episodes = None
        start = 0
        episodes = []
        for length in episode_lengths:
            episodes.append({"dataset_from_index": start, "dataset_to_index": start + length})
            start += length
        self.meta = SimpleNamespace(total_episodes=len(episodes), episodes=episodes)
        self.num_episodes = len(episodes)
        self._length = start

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return {"index": torch.tensor(index)}


def test_sft_dataset_config_validates_sources():
    with pytest.raises(ValueError, match="at least one"):
        SFTDatasetConfig(sources=[])
    with pytest.raises(ValueError, match="weight"):
        SFTDatasetSourceConfig(repo_id="org/corrections", weight=0)
    with pytest.raises(ValueError, match="unique repo_ids"):
        SFTDatasetConfig(
            sources=[
                SFTDatasetSourceConfig(repo_id="org/data"),
                SFTDatasetSourceConfig(repo_id="org/data"),
            ]
        )


def test_sft_mixture_sampler_respects_source_weights():
    torch.manual_seed(123)
    datasets = [_FakeDataset("org/base", [100]), _FakeDataset("org/corrections", [100])]
    sampler = SFTMixtureSampler(datasets, weights=[0.7, 0.3], num_samples=20_000)

    samples = torch.tensor(list(sampler))
    base_fraction = (samples < 100).float().mean().item()

    assert base_fraction == pytest.approx(0.7, abs=0.015)


def test_sft_mixture_sampler_drops_episode_tails():
    torch.manual_seed(123)
    dataset = _FakeDataset("org/base", [5, 4])
    sampler = SFTMixtureSampler([dataset], weights=[1.0], num_samples=1_000, drop_n_last_frames=2)

    assert set(sampler).issubset({0, 1, 2, 5, 6})


def test_sft_mixture_dataset_adds_source_index():
    datasets = [_FakeDataset("org/base", [2]), _FakeDataset("org/corrections", [3])]
    mixture = SFTMixtureDataset(datasets, weights=[0.8, 0.2])

    assert len(mixture) == 5
    assert mixture.num_episodes == 2
    assert mixture[0][SFT_SOURCE_INDEX].item() == 0
    assert mixture[2][SFT_SOURCE_INDEX].item() == 1
