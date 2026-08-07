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

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.xvla.configuration_xvla import XVLAConfig


@PreTrainedConfig.register_subclass("xvla_light")
@dataclass
class XVLALightConfig(XVLAConfig):
    """XVLA with a smaller policy transformer and the same Florence-2 backbone."""

    depth: int = 12
    mlp_ratio: float = 2.0
    len_soft_prompts: int = 8

    # Training-only, lazy feature cache. Rollout/evaluation always runs Florence
    # online because incoming robot observations have no stable dataset index.
    cache_florence_features: bool = False
    florence_cache_path: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.cache_florence_features:
            return
        if not self.florence_cache_path:
            raise ValueError("`florence_cache_path` is required when `cache_florence_features=True`.")
        if not self.freeze_vision_encoder or not self.freeze_language_encoder:
            raise ValueError(
                "Florence caching requires both `freeze_vision_encoder=True` and "
                "`freeze_language_encoder=True`."
            )
