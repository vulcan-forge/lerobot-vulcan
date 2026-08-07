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
from unittest.mock import patch

from lerobot.scripts.sourccey.train.retry import _extract_output_dir
from lerobot.scripts.sourccey.train.sft_retry import main as sft_retry_main


def test_sft_retry_targets_sourccey_sft_module():
    with patch("lerobot.scripts.sourccey.train.sft_retry.retry_main", return_value=0) as retry_main:
        assert sft_retry_main() == 0

    retry_main.assert_called_once_with(
        training_module="lerobot.scripts.sourccey.train.sft",
        training_name="sourccey/train/sft.py",
    )


def test_extract_output_dir_from_yaml_config(tmp_path: Path):
    config_path = tmp_path / "sft.yaml"
    config_path.write_text("output_dir: outputs/sft/correction-run\n")

    output_dir = _extract_output_dir([f"--config_path={config_path}"])

    assert output_dir == Path("outputs/sft/correction-run")
