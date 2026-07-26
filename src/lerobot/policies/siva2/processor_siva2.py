# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""SIVA2 processors retain the XVLA/SIVA observation normalization boundary.

The shared tokenizer already emits optional subtask tokens.  SIVA2 metadata is
kept as complementary batch data and consumed directly by the policy.
"""

from typing import Any

import torch

from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.types import PolicyAction

from .configuration_siva2 import SIVA2Config


def make_siva2_pre_post_processors(
    config: SIVA2Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return make_xvla_pre_post_processors(config=config, dataset_stats=dataset_stats)
