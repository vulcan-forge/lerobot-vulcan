# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""SIVA processors.

The first experiment intentionally uses XVLA's exact preprocessing.  Changing
both the representation and the inputs would make a sample-efficiency comparison
impossible to interpret.
"""

from typing import Any

import torch

from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.types import PolicyAction

from .configuration_siva import SIVAConfig


def make_siva_pre_post_processors(
    config: SIVAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return make_xvla_pre_post_processors(config=config, dataset_stats=dataset_stats)
