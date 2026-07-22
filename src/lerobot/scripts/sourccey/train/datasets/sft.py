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

"""Weighted multi-dataset support for supervised fine-tuning."""

import logging
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.sourccey.train.configs.sft import SFTPipelineConfig
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_PREFIX

logger = logging.getLogger(__name__)

SFT_SOURCE_INDEX = "sft_source_index"


def _feature_signature(feature: dict[str, Any]) -> tuple[Any, ...]:
    shape = feature.get("shape")
    return (
        feature.get("dtype"),
        tuple(shape) if shape is not None else None,
        repr(feature.get("names")),
    )


def _validate_compatible_metadata(metadata: Sequence[LeRobotDatasetMetadata]) -> None:
    """Reject schema differences that could change policy inputs or targets."""
    reference = metadata[0]
    reference_keys = {key for key in reference.features if key == ACTION or key.startswith(OBS_PREFIX)}

    for candidate in metadata[1:]:
        candidate_keys = {key for key in candidate.features if key == ACTION or key.startswith(OBS_PREFIX)}
        if candidate.fps != reference.fps:
            raise ValueError(
                f"SFT datasets must use the same FPS: {reference.repo_id} uses {reference.fps}, "
                f"but {candidate.repo_id} uses {candidate.fps}."
            )
        if candidate_keys != reference_keys:
            missing = sorted(reference_keys - candidate_keys)
            extra = sorted(candidate_keys - reference_keys)
            raise ValueError(
                f"SFT dataset feature mismatch for {candidate.repo_id}: missing={missing}, extra={extra}."
            )
        for key in sorted(reference_keys):
            expected = _feature_signature(reference.features[key])
            actual = _feature_signature(candidate.features[key])
            if actual != expected:
                raise ValueError(
                    f"SFT dataset feature '{key}' differs between {reference.repo_id} and "
                    f"{candidate.repo_id}: expected {expected}, got {actual}."
                )
        if candidate.has_language_columns != reference.has_language_columns:
            raise ValueError(
                "All SFT datasets must either declare language columns or omit them consistently."
            )


def _eligible_local_indices(dataset: LeRobotDataset, drop_n_last_frames: int) -> torch.Tensor:
    episodes = dataset.episodes
    if episodes is None:
        episodes = list(range(dataset.meta.total_episodes))

    indices: list[int] = []
    local_start = 0
    for episode_index in episodes:
        episode = dataset.meta.episodes[episode_index]
        episode_length = episode["dataset_to_index"] - episode["dataset_from_index"]
        usable_length = episode_length - drop_n_last_frames
        if usable_length > 0:
            indices.extend(range(local_start, local_start + usable_length))
        else:
            logger.warning(
                "Skipping SFT episode %s from %s because drop_n_last_frames=%s removes all %s frames.",
                episode_index,
                dataset.repo_id,
                drop_n_last_frames,
                episode_length,
            )
        local_start += episode_length

    if not indices:
        raise ValueError(
            f"No valid frames remain in {dataset.repo_id} after dropping {drop_n_last_frames} trailing frames."
        )
    return torch.tensor(indices, dtype=torch.int64)


class SFTMixtureSampler(Sampler[int]):
    """Sample a dataset by configured probability, then a frame uniformly within it."""

    def __init__(
        self,
        datasets: Sequence[LeRobotDataset],
        weights: Sequence[float],
        num_samples: int,
        drop_n_last_frames: int = 0,
    ) -> None:
        if len(datasets) != len(weights) or not datasets:
            raise ValueError("datasets and weights must have the same non-zero length.")
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0.")
        if drop_n_last_frames < 0:
            raise ValueError("drop_n_last_frames must be >= 0.")

        self.weights = torch.tensor(weights, dtype=torch.float64)
        if not torch.isfinite(self.weights).all() or (self.weights <= 0).any():
            raise ValueError("All SFT dataset weights must be finite and > 0.")
        self.weights /= self.weights.sum()
        self.num_samples = num_samples
        self.offsets: list[int] = []
        offset = 0
        self.eligible_indices: list[torch.Tensor] = []
        for dataset in datasets:
            self.offsets.append(offset)
            self.eligible_indices.append(_eligible_local_indices(dataset, drop_n_last_frames))
            offset += len(dataset)

    def __iter__(self) -> Iterator[int]:
        # Bound temporary memory even when an epoch represents millions of draws.
        draw_chunk_size = 65_536
        for chunk_start in range(0, self.num_samples, draw_chunk_size):
            chunk_size = min(draw_chunk_size, self.num_samples - chunk_start)
            source_indices = torch.multinomial(self.weights, chunk_size, replacement=True)
            sampled_indices = torch.empty(chunk_size, dtype=torch.int64)
            for source_index, eligible in enumerate(self.eligible_indices):
                positions = torch.where(source_indices == source_index)[0]
                if positions.numel() == 0:
                    continue
                draws = torch.randint(len(eligible), (positions.numel(),))
                sampled_indices[positions] = eligible[draws] + self.offsets[source_index]
            yield from sampled_indices.tolist()

    def __len__(self) -> int:
        return self.num_samples


class SFTMixtureDataset(Dataset):
    """Concatenated datasets carrying source IDs and a weighted sampler factory."""

    def __init__(
        self,
        datasets: Sequence[LeRobotDataset],
        weights: Sequence[float],
        samples_per_epoch: int | None = None,
    ) -> None:
        if len(datasets) != len(weights) or not datasets:
            raise ValueError("datasets and weights must have the same non-zero length.")
        self.datasets = list(datasets)
        self.weights = list(weights)
        self.samples_per_epoch = samples_per_epoch or sum(len(dataset) for dataset in datasets)
        self.offsets: list[int] = []
        offset = 0
        for dataset in datasets:
            self.offsets.append(offset)
            offset += len(dataset)
        self._length = offset

        # Policy construction requires one LeRobotDatasetMetadata object. Compatibility
        # checks guarantee that the first source accurately describes every model feature.
        self.meta = self.datasets[0].meta
        self.episodes = None
        self.source_names = [dataset.repo_id for dataset in self.datasets]

    @property
    def num_frames(self) -> int:
        return self._length

    @property
    def num_episodes(self) -> int:
        return sum(dataset.num_episodes for dataset in self.datasets)

    def make_sampler(self, drop_n_last_frames: int = 0) -> SFTMixtureSampler:
        return SFTMixtureSampler(
            self.datasets,
            self.weights,
            self.samples_per_epoch,
            drop_n_last_frames=drop_n_last_frames,
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(f"SFT mixture index {index} is out of bounds for length {len(self)}.")
        source_index = bisect_right(self.offsets, index) - 1
        item = self.datasets[source_index][index - self.offsets[source_index]]
        item[SFT_SOURCE_INDEX] = torch.tensor(source_index, dtype=torch.int64)
        return item


def make_sft_dataset(cfg: SFTPipelineConfig) -> SFTMixtureDataset:
    """Load, validate, and combine every configured SFT source."""
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )
    metadata = [
        LeRobotDatasetMetadata(source.repo_id, root=source.root, revision=source.revision)
        for source in cfg.dataset.sources
    ]
    _validate_compatible_metadata(metadata)

    datasets = []
    for source, source_meta in zip(cfg.dataset.sources, metadata, strict=True):
        delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, source_meta)
        datasets.append(
            LeRobotDataset(
                source.repo_id,
                root=source.root,
                episodes=source.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=source.revision,
                video_backend=cfg.dataset.video_backend,
                return_uint8=True,
                tolerance_s=cfg.tolerance_s,
            )
        )

    return SFTMixtureDataset(
        datasets,
        [source.weight for source in cfg.dataset.sources],
        samples_per_epoch=cfg.dataset.samples_per_epoch,
    )
