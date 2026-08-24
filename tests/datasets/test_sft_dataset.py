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

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.sft import SFTDatasetConfig, SFTDatasetSourceConfig
from lerobot.datasets.sft import (
    SFT_MASK_PADDED_ACTIONS,
    SFT_SOURCE_INDEX,
    SFTMixtureDataset,
    SFTMixtureSampler,
    _resolve_source,
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
    with pytest.raises(ValueError, match="requires root"):
        SFTDatasetSourceConfig(
            repo_id="org/corrections",
            subdataset_glob="*",
        )


def test_sft_source_discovers_local_subdatasets(tmp_path):
    for name in ["set-b", "set-a"]:
        (tmp_path / name / "meta").mkdir(parents=True)
        (tmp_path / name / "meta" / "info.json").touch()
    (tmp_path / "not-a-dataset").mkdir()

    source = SFTDatasetSourceConfig(
        repo_id="local/corrections",
        root=str(tmp_path),
        subdataset_glob="*",
        weight=0.25,
    )

    resolved = _resolve_source(source, source_index=1)

    assert [Path(item.root).name for item in resolved] == ["set-a", "set-b"]
    assert [item.source_index for item in resolved] == [1, 1]
    assert [item.weight for item in resolved] == pytest.approx([0.25, 0.25])


def test_sft_mixture_sampler_respects_source_weights():
    torch.manual_seed(123)
    datasets = [_FakeDataset("org/base", [100]), _FakeDataset("org/corrections", [100])]
    sampler = SFTMixtureSampler(datasets, weights=[0.7, 0.3], num_samples=20_000)

    samples = torch.tensor(list(sampler))
    base_fraction = (samples < 100).float().mean().item()

    assert base_fraction == pytest.approx(0.7, abs=0.015)


def test_sft_mixture_sampler_visits_every_source_frame_before_repeating():
    torch.manual_seed(123)
    dataset = _FakeDataset("org/corrections", [3, 2])
    sampler = SFTMixtureSampler([dataset], weights=[1.0], num_samples=10)

    samples = list(sampler)

    assert set(samples[:5]) == set(range(5))
    assert set(samples[5:]) == set(range(5))


def test_sft_mixture_dataset_adds_source_index():
    datasets = [_FakeDataset("org/base", [2]), _FakeDataset("org/corrections", [3])]
    mixture = SFTMixtureDataset(datasets, weights=[0.8, 0.2])

    assert len(mixture) == 5
    assert mixture.num_episodes == 2
    assert mixture[0][SFT_SOURCE_INDEX].item() == 0
    assert mixture[2][SFT_SOURCE_INDEX].item() == 1
    assert mixture.weights == pytest.approx([0.8, 0.2])


def test_sft_mixture_dataset_opt_in_adds_padding_mask_flag():
    mixture = SFTMixtureDataset(
        [_FakeDataset("org/base", [2])],
        weights=[1.0],
        mask_padded_actions=True,
    )

    assert mixture[0][SFT_MASK_PADDED_ACTIONS].item() is True


def test_sft_mixture_dataset_aggregates_child_dataset_source_indices():
    datasets = [
        _FakeDataset("org/base", [2]),
        _FakeDataset("local/corrections/a", [3]),
        _FakeDataset("local/corrections/b", [4]),
    ]
    mixture = SFTMixtureDataset(
        datasets,
        weights=[0.75, 0.1, 0.15],
        source_names=["org/base", "local/corrections"],
        dataset_source_indices=[0, 1, 1],
    )

    assert mixture[0][SFT_SOURCE_INDEX].item() == 0
    assert mixture[2][SFT_SOURCE_INDEX].item() == 1
    assert mixture[5][SFT_SOURCE_INDEX].item() == 1
    assert mixture.weights == pytest.approx([0.75, 0.25])
