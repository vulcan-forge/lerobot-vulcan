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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from lerobot.configs.sft import SFTDatasetSourceConfig, SFTPipelineConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_PREFIX

logger = logging.getLogger(__name__)

SFT_SOURCE_INDEX = "sft_source_index"
SFT_MASK_PADDED_ACTIONS = "sft_mask_padded_actions"


@dataclass(frozen=True)
class _ResolvedSFTSource:
    """One concrete LeRobot dataset belonging to a configured logical source."""

    source_index: int
    source_name: str
    repo_id: str
    weight: float
    root: str | None
    episodes: list[int] | None
    revision: str | None


def _resolve_source(
    source: SFTDatasetSourceConfig,
    source_index: int,
) -> list[_ResolvedSFTSource]:
    if source.subdataset_glob is None:
        return [
            _ResolvedSFTSource(
                source_index=source_index,
                source_name=source.repo_id,
                repo_id=source.repo_id,
                weight=source.weight,
                root=source.root,
                episodes=source.episodes,
                revision=source.revision,
            )
        ]

    root = Path(source.root).expanduser()  # type: ignore[arg-type]
    if not root.is_dir():
        raise FileNotFoundError(f"SFT subdataset root does not exist or is not a directory: {root}")

    child_roots = sorted(
        path
        for path in root.glob(source.subdataset_glob)
        if path.is_dir() and (path / "meta" / "info.json").is_file()
    )
    if not child_roots:
        raise FileNotFoundError(
            f"SFT source {source.repo_id} found no LeRobot datasets under {root} "
            f"matching {source.subdataset_glob!r}."
        )

    logger.info(
        "Discovered %d LeRobot subdatasets for SFT source %s under %s.",
        len(child_roots),
        source.repo_id,
        root,
    )
    return [
        _ResolvedSFTSource(
            source_index=source_index,
            source_name=source.repo_id,
            repo_id=f"{source.repo_id}/{child_root.relative_to(root).as_posix()}",
            weight=source.weight,
            root=str(child_root),
            episodes=None,
            revision=source.revision,
        )
        for child_root in child_roots
    ]


def _resolve_sources(cfg: SFTPipelineConfig) -> list[_ResolvedSFTSource]:
    return [
        resolved
        for source_index, source in enumerate(cfg.dataset.sources)
        for resolved in _resolve_source(source, source_index)
    ]


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


class SFTMixtureSampler(Sampler[int]):
    """Sample logical sources by weight and cycle through their shuffled frames."""

    def __init__(
        self,
        datasets: Sequence[LeRobotDataset],
        weights: Sequence[float],
        num_samples: int,
        dataset_source_indices: Sequence[int] | None = None,
    ) -> None:
        if len(datasets) != len(weights) or not datasets:
            raise ValueError("datasets and weights must have the same non-zero length.")
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0.")
        dataset_weights = torch.tensor(weights, dtype=torch.float64)
        if not torch.isfinite(dataset_weights).all() or (dataset_weights <= 0).any():
            raise ValueError("All SFT dataset weights must be finite and > 0.")
        if dataset_source_indices is None:
            dataset_source_indices = list(range(len(datasets)))
        if len(dataset_source_indices) != len(datasets):
            raise ValueError("dataset_source_indices must contain one entry per dataset.")
        if any(source_index < 0 for source_index in dataset_source_indices):
            raise ValueError("dataset_source_indices must be non-negative.")

        num_sources = max(dataset_source_indices) + 1
        if set(dataset_source_indices) != set(range(num_sources)):
            raise ValueError("dataset_source_indices must form a contiguous range starting at zero.")

        self.weights = torch.zeros(num_sources, dtype=torch.float64)
        self.weights.scatter_add_(
            0,
            torch.tensor(dataset_source_indices, dtype=torch.int64),
            dataset_weights,
        )
        self.weights /= self.weights.sum()
        self.num_samples = num_samples
        source_eligible_indices: list[list[torch.Tensor]] = [[] for _ in range(num_sources)]
        offset = 0
        for dataset, source_index in zip(datasets, dataset_source_indices, strict=True):
            eligible = torch.arange(len(dataset))
            source_eligible_indices[source_index].append(eligible + offset)
            offset += len(dataset)
        self.eligible_indices = [
            torch.cat(indices) if len(indices) > 1 else indices[0] for indices in source_eligible_indices
        ]

    def __iter__(self) -> Iterator[int]:
        # Each logical source gets an independently shuffled frame pool. A pool
        # is exhausted before it is reshuffled, so sufficiently long SFT runs
        # see every correction frame instead of repeatedly missing random ones.
        shuffled_indices = [eligible[torch.randperm(len(eligible))] for eligible in self.eligible_indices]
        cursors = [0] * len(self.eligible_indices)

        def draw_from_source(source_index: int, count: int) -> torch.Tensor:
            draws = torch.empty(count, dtype=torch.int64)
            draw_offset = 0
            while draw_offset < count:
                shuffled = shuffled_indices[source_index]
                cursor = cursors[source_index]
                available = len(shuffled) - cursor
                take = min(count - draw_offset, available)
                draws[draw_offset : draw_offset + take] = shuffled[cursor : cursor + take]
                draw_offset += take
                cursor += take
                if cursor == len(shuffled):
                    shuffled = self.eligible_indices[source_index][
                        torch.randperm(len(self.eligible_indices[source_index]))
                    ]
                    shuffled_indices[source_index] = shuffled
                    cursor = 0
                cursors[source_index] = cursor
            return draws

        # Bound temporary memory even when an epoch represents millions of draws.
        draw_chunk_size = 65_536
        for chunk_start in range(0, self.num_samples, draw_chunk_size):
            chunk_size = min(draw_chunk_size, self.num_samples - chunk_start)
            source_indices = torch.multinomial(self.weights, chunk_size, replacement=True)
            sampled_indices = torch.empty(chunk_size, dtype=torch.int64)
            for source_index in range(len(self.eligible_indices)):
                positions = torch.where(source_indices == source_index)[0]
                if positions.numel() == 0:
                    continue
                sampled_indices[positions] = draw_from_source(source_index, positions.numel())
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
        source_names: Sequence[str] | None = None,
        dataset_source_indices: Sequence[int] | None = None,
        mask_padded_actions: bool = False,
    ) -> None:
        if len(datasets) != len(weights) or not datasets:
            raise ValueError("datasets and weights must have the same non-zero length.")
        if any(len(dataset) == 0 for dataset in datasets):
            raise ValueError("SFT datasets must contain at least one frame.")
        self.datasets = list(datasets)
        self.dataset_weights = list(weights)
        self.samples_per_epoch = samples_per_epoch or sum(len(dataset) for dataset in datasets)
        self.mask_padded_actions = mask_padded_actions
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
        self.source_names = (
            list(source_names) if source_names is not None else [dataset.repo_id for dataset in self.datasets]
        )
        self.dataset_source_indices = (
            list(dataset_source_indices)
            if dataset_source_indices is not None
            else list(range(len(self.datasets)))
        )
        if len(self.dataset_source_indices) != len(self.datasets):
            raise ValueError("dataset_source_indices must contain one entry per dataset.")
        if not self.source_names:
            raise ValueError("source_names must not be empty.")
        if any(
            source_index < 0 or source_index >= len(self.source_names)
            for source_index in self.dataset_source_indices
        ):
            raise ValueError("dataset_source_indices contains an out-of-range source index.")
        self.weights = [0.0] * len(self.source_names)
        for weight, source_index in zip(self.dataset_weights, self.dataset_source_indices, strict=True):
            self.weights[source_index] += weight

    @property
    def num_frames(self) -> int:
        return self._length

    @property
    def num_episodes(self) -> int:
        return sum(dataset.num_episodes for dataset in self.datasets)

    def make_sampler(self) -> SFTMixtureSampler:
        return SFTMixtureSampler(
            self.datasets,
            self.dataset_weights,
            self.samples_per_epoch,
            dataset_source_indices=self.dataset_source_indices,
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(f"SFT mixture index {index} is out of bounds for length {len(self)}.")
        dataset_index = bisect_right(self.offsets, index) - 1
        item = self.datasets[dataset_index][index - self.offsets[dataset_index]]
        item[SFT_SOURCE_INDEX] = torch.tensor(self.dataset_source_indices[dataset_index], dtype=torch.int64)
        if self.mask_padded_actions:
            item[SFT_MASK_PADDED_ACTIONS] = torch.tensor(True)
        return item


def make_sft_dataset(cfg: SFTPipelineConfig) -> SFTMixtureDataset:
    """Load, validate, and combine every configured SFT source."""
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )
    resolved_sources = _resolve_sources(cfg)
    metadata = [
        LeRobotDatasetMetadata(source.repo_id, root=source.root, revision=source.revision)
        for source in resolved_sources
    ]
    _validate_compatible_metadata(metadata)

    datasets = []
    for source, source_meta in zip(resolved_sources, metadata, strict=True):
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

    # A globbed folder is one logical source. Divide that source's probability
    # across its children by frame count so sampling is uniform over all frames
    # in the folder rather than uniform over recording sessions.
    empty_datasets = [
        source.repo_id
        for dataset, source in zip(datasets, resolved_sources, strict=True)
        if len(dataset) == 0
    ]
    if empty_datasets:
        raise ValueError(f"SFT datasets must contain at least one frame: {empty_datasets}.")
    source_frame_counts = [0] * len(cfg.dataset.sources)
    for dataset, source in zip(datasets, resolved_sources, strict=True):
        source_frame_counts[source.source_index] += len(dataset)
    weights = [
        source.weight * len(dataset) / source_frame_counts[source.source_index]
        for dataset, source in zip(datasets, resolved_sources, strict=True)
    ]

    return SFTMixtureDataset(
        datasets,
        weights,
        samples_per_epoch=cfg.dataset.samples_per_epoch,
        source_names=[source.repo_id for source in cfg.dataset.sources],
        dataset_source_indices=[source.source_index for source in resolved_sources],
        mask_padded_actions=cfg.dataset.mask_padded_actions,
    )
