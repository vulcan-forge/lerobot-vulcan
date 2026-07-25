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

import draccus
import pytest
import yaml

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.sourccey.train.configs.sft import (
    SFTDatasetConfig,
    SFTDatasetSourceConfig,
    SFTPipelineConfig,
)

SFT_RECIPE_DIR = (
    Path(__file__).parents[2] / "src" / "lerobot" / "scripts" / "sourccey" / "train" / "configs" / "sft_recipes"
)


@pytest.mark.parametrize("recipe_name", ["example.yaml", "shirt_fold_c_009.yaml"])
def test_sft_recipes_are_valid_yaml(recipe_name: str) -> None:
    recipe = yaml.safe_load((SFT_RECIPE_DIR / recipe_name).read_text(encoding="utf-8"))

    assert recipe["policy"]["path"]
    assert recipe["dataset"]["sources"]


def test_sft_config_parses_weighted_sources_from_yaml(tmp_path: Path):
    config_path = tmp_path / "sft.yaml"
    config_path.write_text(
        """
dataset:
  sources:
    - repo_id: org/base
      weight: 0.7
    - repo_id: org/corrections
      weight: 0.3
policy:
  type: act
  device: cpu
  push_to_hub: false
output_dir: outputs/sft/test
""".strip()
    )

    cfg = draccus.parse(SFTPipelineConfig, config_path=config_path, args=[])

    assert [source.repo_id for source in cfg.dataset.sources] == ["org/base", "org/corrections"]
    assert [source.weight for source in cfg.dataset.sources] == pytest.approx([0.7, 0.3])


def test_sft_config_checkpoint_round_trip(tmp_path: Path):
    config_dir = tmp_path / "pretrained_model"
    cfg = SFTPipelineConfig(
        dataset=SFTDatasetConfig(
            sources=[
                SFTDatasetSourceConfig(repo_id="org/base", weight=0.7),
                SFTDatasetSourceConfig(repo_id="org/corrections", weight=0.3),
            ]
        ),
        policy=ACTConfig(
            device="cpu",
            push_to_hub=False,
            pretrained_path=tmp_path / "base_policy",
        ),
        output_dir=tmp_path / "sft",
    )

    cfg.save_pretrained(config_dir)
    restored = SFTPipelineConfig.from_pretrained(config_dir, cli_args=[])

    assert [source.repo_id for source in restored.dataset.sources] == [
        "org/base",
        "org/corrections",
    ]
    assert restored.dataset.normalization == "checkpoint"


def test_sft_requires_a_pretrained_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("sys.argv", ["sourccey/train/sft.py"])
    cfg = SFTPipelineConfig(
        dataset=SFTDatasetConfig(sources=[SFTDatasetSourceConfig(repo_id="org/data")]),
        policy=ACTConfig(device="cpu", push_to_hub=False),
        output_dir=tmp_path / "sft",
    )

    with pytest.raises(ValueError, match="requires a pretrained policy"):
        cfg.validate()


def test_sft_scales_policy_preset_learning_rate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("sys.argv", ["sourccey/train/sft.py"])
    policy = ACTConfig(
        device="cpu",
        push_to_hub=False,
        pretrained_path=tmp_path / "pretrained",
    )
    preset_lr = policy.get_optimizer_preset().lr
    cfg = SFTPipelineConfig(
        dataset=SFTDatasetConfig(sources=[SFTDatasetSourceConfig(repo_id="org/data")]),
        policy=policy,
        output_dir=tmp_path / "sft",
        policy_preset_lr_scale=0.1,
    )

    cfg.validate()

    assert cfg.optimizer is not None
    assert cfg.optimizer.lr == pytest.approx(preset_lr * 0.1)
