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

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.xvla_light.configuration_xvla_light import XVLALightConfig
from lerobot.policies.xvla_light.convert_checkpoint import (
    convert_checkpoint,
    convert_xvla_state_dict,
    select_evenly_spaced_layers,
)


def _make_source_state() -> dict[str, torch.Tensor]:
    state = {
        "model.transformer.pos_emb": torch.arange(12).reshape(1, 6, 2).float(),
        "model.transformer.soft_prompt_hub.weight": torch.arange(16).reshape(2, 8).float(),
    }
    for layer in range(4):
        prefix = f"model.transformer.blocks.{layer}."
        fc1_weight = torch.tensor([[1.0, 0.0], [10.0, 0.0], [3.0, 0.0], [8.0, 0.0]]) + layer
        state.update(
            {
                f"{prefix}norm1.weight": torch.full((2,), float(layer)),
                f"{prefix}mlp.fc1.weight": fc1_weight,
                f"{prefix}mlp.fc1.bias": torch.arange(4).float() + layer,
                f"{prefix}mlp.fc2.weight": torch.zeros(2, 4),
                f"{prefix}mlp.fc2.bias": torch.full((2,), float(layer)),
            }
        )
    return state


def test_select_evenly_spaced_layers_includes_stack_endpoints():
    assert select_evenly_spaced_layers(24, 12) == [0, 2, 4, 6, 8, 10, 13, 15, 17, 19, 21, 23]


def test_convert_state_dict_maps_layers_prunes_mlp_and_averages_prompts():
    converted, layer_map = convert_xvla_state_dict(
        _make_source_state(),
        source_depth=4,
        target_depth=2,
        hidden_size=2,
        source_mlp_ratio=2.0,
        target_mlp_ratio=1.0,
        source_prompt_length=4,
        target_prompt_length=2,
    )

    assert layer_map == [0, 3]
    assert not any(key.startswith("model.transformer.blocks.2.") for key in converted)
    assert torch.equal(converted["model.transformer.blocks.1.norm1.weight"], torch.full((2,), 3.0))
    assert torch.equal(
        converted["model.transformer.blocks.0.mlp.fc1.weight"],
        torch.tensor([[10.0, 0.0], [8.0, 0.0]]),
    )
    assert torch.equal(
        converted["model.transformer.soft_prompt_hub.weight"],
        torch.tensor([[1.0, 2.0, 5.0, 6.0], [9.0, 10.0, 13.0, 14.0]]),
    )


def test_convert_checkpoint_writes_light_config_weights_and_processor_files(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_config = {
        "type": "xvla",
        "hidden_size": 2,
        "depth": 4,
        "mlp_ratio": 2.0,
        "len_soft_prompts": 4,
        "pretrained_path": "stale/source/path",
    }
    (source_dir / "config.json").write_text(json.dumps(source_config))
    (source_dir / "policy_preprocessor.json").write_text("{}")
    save_file(_make_source_state(), source_dir / "model.safetensors")

    output_dir = tmp_path / "converted"
    layer_map = convert_checkpoint(
        str(source_dir),
        output_dir,
        target_depth=2,
        target_mlp_ratio=1.0,
        target_prompt_length=2,
    )

    target_config = json.loads((output_dir / "config.json").read_text())
    target_state = load_file(output_dir / "model.safetensors")
    assert layer_map == [0, 3]
    assert target_config | {} == {
        "type": "xvla_light",
        "hidden_size": 2,
        "depth": 2,
        "mlp_ratio": 1.0,
        "len_soft_prompts": 2,
        "pretrained_path": None,
    }
    assert target_state["model.transformer.blocks.1.mlp.fc1.weight"].shape == (2, 2)
    assert (output_dir / "policy_preprocessor.json").is_file()
    assert isinstance(PreTrainedConfig.from_pretrained(output_dir), XVLALightConfig)


def test_converter_refuses_to_expand_architecture():
    with pytest.raises(ValueError, match="cannot exceed source depth"):
        select_evenly_spaced_layers(4, 5)


def test_xvla_light_has_compressed_defaults():
    config = XVLALightConfig(device="cpu")

    assert config.type == "xvla_light"
    assert config.hidden_size == 1024
    assert config.depth == 12
    assert config.num_heads == 16
    assert config.mlp_ratio == 2.0
    assert config.len_soft_prompts == 8
