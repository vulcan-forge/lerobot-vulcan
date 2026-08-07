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

"""Supervised fine-tune a pretrained policy on a weighted dataset mixture."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from accelerate import Accelerator

from lerobot.configs import parser
from lerobot.utils.import_utils import register_third_party_plugins

from .configs.sft import SFTPipelineConfig
from .datasets.sft import make_sft_dataset
from .sft_xvla import enable_sft_xvla_padding_mask
from .trainer import run_training


@parser.wrap()
def sft(cfg: SFTPipelineConfig, accelerator: "Accelerator | None" = None):
    """Run behavior-cloning SFT with fresh optimizer state from a pretrained policy."""
    return run_training(
        cfg,
        accelerator=accelerator,
        dataset_factory=make_sft_dataset,
        policy_setup=enable_sft_xvla_padding_mask,
    )


def main():
    register_third_party_plugins()
    sft()


if __name__ == "__main__":
    main()
