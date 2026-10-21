#!/usr/bin/env python

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

import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig


def test_resume_sets_checkpoint_path_even_with_policy_path_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    policy_dir = tmp_path / "checkpoints" / "last" / "pretrained_model"
    policy_dir.mkdir(parents=True)
    config_path = policy_dir / "train_config.json"
    config_path.write_text("{}")

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="dummy/repo"),
        policy=ACTConfig(push_to_hub=False, pretrained_path=Path("user/base-model")),
        output_dir=tmp_path,
        resume=True,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "lerobot_train.py",
            "--resume=true",
            f"--config_path={config_path}",
            "--policy.path=lerobot/xvla-base",
        ],
    )
    cfg.validate()

    assert cfg.checkpoint_path == policy_dir.parent
    assert cfg.policy is not None
    assert cfg.policy.pretrained_path == Path("user/base-model")
