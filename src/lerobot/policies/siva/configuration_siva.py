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

"""Configuration for the experimental Slot-Intent VLA policy.

SIVA deliberately inherits XVLA's observation contract and Florence-2
configuration.  That keeps existing XVLA datasets and processors usable while
making the action architecture an isolated experiment.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.xvla.configuration_xvla import XVLAConfig


@PreTrainedConfig.register_subclass("siva")
@dataclass
class SIVAConfig(XVLAConfig):
    """Slot-Intent VLA configuration.

    The defaults target a first practical prototype, not a claimed optimum.
    Florence-2 remains the representation source, but its long token sequence is
    compressed to ``num_context_slots`` before action generation.
    """

    # The robot in this repository uses XVLA's automatic action layout.  Unlike
    # XVLA's fixed EE6D default, this avoids spending capacity on unused padded
    # action channels when training a new head.
    action_mode: str = "auto"

    # Compact context/action networks.  XVLA-base uses 24 x 1024 transformer
    # blocks over every multimodal and action token; SIVA routes to 16 slots and
    # uses a smaller action-only stack with cross-attention.
    hidden_size: int = 512
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 3.0
    num_context_slots: int = 16
    slot_depth: int = 2
    vision_token_dropout: float = 0.1

    # Structured source distribution.  Each mode is a low-frequency trajectory
    # represented by a handful of control points.  The flow decoder learns local
    # residual corrections instead of transporting arbitrary Gaussian noise all
    # the way to the demonstration.
    num_motion_modes: int = 8
    num_control_points: int = 6
    domain_adapter_rank: int = 4
    prior_init_scale: float = 0.02
    prior_noise_scale: float = 0.15
    noise_temporal_correlation: float = 0.8
    routing_temperature: float = 1.0

    # Use the same ten-step operating point as XVLA for a stable baseline.
    # Shorter Euler solves remain useful latency ablations once action quality
    # has been established.
    num_denoising_steps: int = 10
    deterministic_inference: bool = True

    # Loss weights are exposed so every architectural claim can be ablated.
    flow_loss_weight: float = 1.0
    endpoint_loss_weight: float = 1.0
    prior_loss_weight: float = 0.05
    router_loss_weight: float = 0.1
    router_balance_loss_weight: float = 0.01
    smoothness_loss_weight: float = 0.001

    # Freeze the complete Florence boundary only after transferring a pretrained
    # XVLA VLM.  Keeping this false by default prevents accidentally freezing a
    # randomly initialized Florence model in from-scratch tests.
    freeze_vlm: bool = False

    # Optional, training-only disk cache for the frozen Florence output.  This
    # is deliberately opt-in because it trades local disk space for throughput
    # and is only correct when both the samples and Florence are deterministic.
    cache_florence_features: bool = False
    florence_cache_path: str | None = None

    # Set only by the XVLA-to-SIVA converter.  It records provenance and explains
    # why the first load is allowed to have newly initialized action modules.
    xvla_init_source: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("`hidden_size` must be divisible by `num_heads`.")
        if self.num_context_slots <= 0:
            raise ValueError("`num_context_slots` must be positive.")
        if self.slot_depth <= 0 or self.depth <= 0:
            raise ValueError("SIVA transformer depths must be positive.")
        if self.num_motion_modes < 2:
            raise ValueError("`num_motion_modes` must be at least 2 to model multimodal behavior.")
        if not 2 <= self.num_control_points <= self.chunk_size:
            raise ValueError("`num_control_points` must be in [2, chunk_size].")
        if not 0.0 <= self.noise_temporal_correlation <= 1.0:
            raise ValueError("`noise_temporal_correlation` must be in [0, 1].")
        if not 0.0 <= self.vision_token_dropout < 1.0:
            raise ValueError("`vision_token_dropout` must be in [0, 1).")
        if self.domain_adapter_rank <= 0:
            raise ValueError("`domain_adapter_rank` must be positive.")
        if self.prior_noise_scale <= 0.0:
            raise ValueError("`prior_noise_scale` must be positive.")
        if self.routing_temperature <= 0.0:
            raise ValueError("`routing_temperature` must be positive.")
        if self.cache_florence_features:
            if not self.florence_cache_path:
                raise ValueError("`florence_cache_path` is required when `cache_florence_features=True`.")
            if not self.freeze_vlm:
                raise ValueError("SIVA Florence caching requires `freeze_vlm=True`.")
