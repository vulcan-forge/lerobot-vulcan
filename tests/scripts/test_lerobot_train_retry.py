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

from lerobot.scripts.lerobot_train_retry import _build_resume_args


def test_build_resume_args_drops_policy_path_flags():
    train_args = [
        "--dataset.repo_id=dummy/repo",
        "--output_dir=outputs/train/test-run",
        "--policy.path=lerobot/xvla-base",
        "--resume=false",
    ]

    resume_args = _build_resume_args(
        train_args, Path("outputs/train/test-run/checkpoints/last/pretrained_model/train_config.json")
    )

    assert all(not arg.startswith("--policy.path") for arg in resume_args)
    assert "--resume=true" in resume_args
    assert any(arg.startswith("--config_path=") for arg in resume_args)
