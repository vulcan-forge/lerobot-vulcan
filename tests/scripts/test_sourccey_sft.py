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

from lerobot.configs import parser
from lerobot.scripts.sourccey.train import sft
from lerobot.scripts.sourccey.train.datasets.sft import make_sft_dataset


def test_sft_entrypoint_parses_policy_path_and_weighted_sources(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "sft.yaml"
    config_path.write_text(
        f"""
policy:
  path: {tmp_path.as_posix()}/base-policy
  push_to_hub: false
dataset:
  sources:
    - repo_id: org/base
      weight: 0.7
    - repo_id: org/corrections
      weight: 0.3
output_dir: {tmp_path.as_posix()}/output
""".strip()
    )
    monkeypatch.setattr("sys.argv", ["sourccey/train/sft.py", f"--config_path={config_path}"])
    parser._config_path_args.clear()
    parser._config_yaml_overrides.clear()

    try:
        with patch.object(sft, "run_training", return_value=None) as run_training:
            sft.sft()
    finally:
        parser._config_path_args.clear()
        parser._config_yaml_overrides.clear()

    cfg = run_training.call_args.args[0]
    assert [source.repo_id for source in cfg.dataset.sources] == ["org/base", "org/corrections"]
    assert run_training.call_args.kwargs["dataset_factory"] is make_sft_dataset
