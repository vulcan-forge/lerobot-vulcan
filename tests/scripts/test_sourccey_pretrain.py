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

from unittest.mock import patch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.sourccey.train import pretrain


def test_pretrain_wrapper_uses_private_training_engine():
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="org/data"),
        policy=ACTConfig(device="cpu", push_to_hub=False),
    )

    with patch.object(pretrain, "run_training", return_value=None) as run_training:
        pretrain.pretrain(cfg)

    run_training.assert_called_once_with(cfg, accelerator=None)
