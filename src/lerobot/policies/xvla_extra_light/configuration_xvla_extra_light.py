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
from lerobot.policies.xvla_light.configuration_xvla_light import XVLALightConfig


@PreTrainedConfig.register_subclass("xvla_extra_light")
@dataclass
class XVLAExtraLightConfig(XVLALightConfig):
    """XVLA-light with a 512-wide residual stream and eight attention heads."""

    hidden_size: int = 512
    num_heads: int = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"`hidden_size` ({self.hidden_size}) must be divisible by `num_heads` ({self.num_heads})."
            )
