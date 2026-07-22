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

"""Configuration for supervised fine-tuning from a pretrained policy."""

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs.train import TrainPipelineConfig
from lerobot.transforms import ImageTransformsConfig
from lerobot.utils.import_utils import get_safe_default_video_backend


@dataclass
class SFTDatasetSourceConfig:
    """One dataset participating in an SFT sampling mixture."""

    repo_id: str
    weight: float = 1.0
    root: str | None = None
    episodes: list[int] | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        if not self.repo_id:
            raise ValueError("SFT dataset repo_id must not be empty.")
        if self.weight <= 0:
            raise ValueError(f"SFT dataset weight must be > 0, got {self.weight} for {self.repo_id}.")
        if self.episodes is not None:
            if any(episode < 0 for episode in self.episodes):
                raise ValueError(f"Episode indices must be non-negative, got {self.episodes}.")
            if len(self.episodes) != len(set(self.episodes)):
                raise ValueError(f"Episode indices must not contain duplicates, got {self.episodes}.")


@dataclass
class SFTDatasetConfig:
    """A weighted collection of compatible LeRobot datasets."""

    sources: list[SFTDatasetSourceConfig]
    normalization: str = "checkpoint"
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    video_backend: str = field(default_factory=get_safe_default_video_backend)
    samples_per_epoch: int | None = None
    streaming: bool = False

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("SFT requires at least one dataset source.")
        source_ids = [source.repo_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(f"SFT dataset sources must have unique repo_ids, got {source_ids}.")
        if self.normalization != "checkpoint":
            raise ValueError("The initial SFT implementation only supports normalization='checkpoint'.")
        if self.samples_per_epoch is not None and self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be > 0 when provided.")
        if self.streaming:
            raise ValueError("Streaming datasets are not supported by weighted SFT.")

    @property
    def repo_id(self) -> str:
        """Primary dataset ID used for model-card compatibility."""
        return self.sources[0].repo_id

    @property
    def root(self) -> str | None:
        """Primary root retained for compatibility with shared training utilities."""
        return self.sources[0].root


@dataclass
class SFTPipelineConfig(TrainPipelineConfig):
    """Train-pipeline configuration specialized for supervised fine-tuning."""

    dataset: SFTDatasetConfig
    policy_preset_lr_scale: float = 0.1

    def validate(self) -> None:
        output_dir_was_default = self.output_dir is None
        super().validate()

        if self.is_reward_model_training:
            raise ValueError("lerobot-sft supports policies only, not reward-model training.")
        if self.policy is None or self.policy.pretrained_path is None:
            raise ValueError("lerobot-sft requires a pretrained policy supplied with --policy.path.")
        if self.sample_weighting is not None:
            raise ValueError("sample_weighting cannot be combined with weighted SFT datasets.")
        if self.policy_preset_lr_scale <= 0:
            raise ValueError("policy_preset_lr_scale must be > 0.")

        # Policy presets are designed for initial training. Keep their parameter-group
        # behavior while using a safer learning rate for a new SFT run.
        if not self.resume and self.use_policy_training_preset and self.optimizer is not None:
            self.optimizer.lr *= self.policy_preset_lr_scale

        if output_dir_was_default and self.output_dir is not None:
            try:
                suffix = self.output_dir.relative_to(Path("outputs/train"))
            except ValueError:
                pass
            else:
                self.output_dir = Path("outputs/sft") / suffix
