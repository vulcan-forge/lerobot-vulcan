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

import torch
from safetensors.torch import load_file, save_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.xvla.soft_transformer import SoftPromptedTransformer
from lerobot.policies.xvla_extra_light.configuration_xvla_extra_light import XVLAExtraLightConfig
from lerobot.policies.xvla_extra_light.convert_checkpoint import convert_checkpoint
from lerobot.policies.xvla_extra_light.modeling_xvla_extra_light import XVLAExtraLightPolicy
from lerobot.policies.xvla_light.configuration_xvla_light import XVLALightConfig
from lerobot.policies.xvla_light.convert_checkpoint import convert_xvla_state_dict
from lerobot.policies.xvla_light.modeling_xvla_light import XVLALightPolicy


def _make_source_state() -> dict[str, torch.Tensor]:
    hidden_size = 4
    mlp_width = 8
    num_domains = 2
    action_input_size = 3
    action_output_size = 2
    state = {
        "model.transformer.pos_emb": torch.arange(24).reshape(1, 6, hidden_size).float(),
        "model.transformer.norm.weight": torch.arange(hidden_size).float(),
        "model.transformer.norm.bias": torch.arange(hidden_size).float(),
        "model.transformer.vlm_proj.weight": torch.arange(24).reshape(hidden_size, 6).float(),
        "model.transformer.vlm_proj.bias": torch.arange(hidden_size).float(),
        "model.transformer.aux_visual_proj.weight": torch.arange(24).reshape(hidden_size, 6).float(),
        "model.transformer.aux_visual_proj.bias": torch.arange(hidden_size).float(),
        "model.transformer.action_encoder.fc.weight": torch.arange(
            num_domains * action_input_size * hidden_size
        )
        .reshape(num_domains, -1)
        .float(),
        "model.transformer.action_encoder.bias.weight": torch.arange(num_domains * hidden_size)
        .reshape(num_domains, hidden_size)
        .float(),
        "model.transformer.action_decoder.fc.weight": torch.arange(
            num_domains * hidden_size * action_output_size
        )
        .reshape(num_domains, -1)
        .float(),
        "model.transformer.action_decoder.bias.weight": torch.zeros(num_domains, action_output_size),
        "model.transformer.soft_prompt_hub.weight": torch.arange(num_domains * 4 * hidden_size)
        .reshape(num_domains, -1)
        .float(),
    }
    for layer in range(2):
        prefix = f"model.transformer.blocks.{layer}."
        qkv_weight = torch.ones(3 * hidden_size, hidden_size)
        for section in range(3):
            qkv_weight[section * hidden_size + 2 : section * hidden_size + 4] = 10
        projection = torch.ones(hidden_size, hidden_size)
        projection[:, 2:] = 10
        state.update(
            {
                f"{prefix}norm1.weight": torch.arange(hidden_size).float(),
                f"{prefix}norm1.bias": torch.arange(hidden_size).float(),
                f"{prefix}norm2.weight": torch.arange(hidden_size).float(),
                f"{prefix}norm2.bias": torch.arange(hidden_size).float(),
                f"{prefix}attn.qkv.weight": qkv_weight,
                f"{prefix}attn.qkv.bias": torch.arange(3 * hidden_size).float(),
                f"{prefix}attn.proj.weight": projection,
                f"{prefix}attn.proj.bias": torch.arange(hidden_size).float(),
                f"{prefix}mlp.fc1.weight": torch.arange(mlp_width * hidden_size)
                .reshape(mlp_width, hidden_size)
                .float(),
                f"{prefix}mlp.fc1.bias": torch.arange(mlp_width).float(),
                f"{prefix}mlp.fc2.weight": torch.arange(hidden_size * mlp_width)
                .reshape(hidden_size, mlp_width)
                .float(),
                f"{prefix}mlp.fc2.bias": torch.arange(hidden_size).float(),
            }
        )
    return state


def test_extra_light_inherits_xvla_and_light_behavior():
    assert issubclass(XVLAExtraLightConfig, XVLALightConfig)
    assert issubclass(XVLAExtraLightConfig, XVLAConfig)
    assert issubclass(XVLAExtraLightPolicy, XVLALightPolicy)
    assert issubclass(XVLAExtraLightPolicy, XVLAPolicy)

    config = XVLAExtraLightConfig(device="cpu")
    assert config.type == "xvla_extra_light"
    assert config.hidden_size == 512
    assert config.num_heads == 8
    assert config.depth == 12
    assert config.mlp_ratio == 2.0
    assert config.len_soft_prompts == 8


def test_hidden_conversion_selects_complete_head_and_resizes_every_policy_projection():
    converted, layer_map = convert_xvla_state_dict(
        _make_source_state(),
        source_depth=2,
        target_depth=1,
        hidden_size=4,
        target_hidden_size=2,
        source_num_heads=2,
        target_num_heads=1,
        source_mlp_ratio=2.0,
        target_mlp_ratio=1.0,
        source_prompt_length=4,
        target_prompt_length=2,
    )

    assert layer_map == [1]
    assert converted["model.transformer.pos_emb"].shape == (1, 6, 2)
    assert torch.equal(converted["model.transformer.norm.weight"], torch.tensor([2.0, 3.0]))
    assert converted["model.transformer.vlm_proj.weight"].shape == (2, 6)
    assert converted["model.transformer.action_encoder.fc.weight"].shape == (2, 6)
    assert converted["model.transformer.action_decoder.fc.weight"].shape == (2, 4)
    assert converted["model.transformer.soft_prompt_hub.weight"].shape == (2, 4)
    assert converted["model.transformer.blocks.0.attn.qkv.weight"].shape == (6, 2)
    assert converted["model.transformer.blocks.0.attn.proj.weight"].shape == (2, 2)
    assert converted["model.transformer.blocks.0.mlp.fc1.weight"].shape == (2, 2)
    assert converted["model.transformer.blocks.0.mlp.fc2.weight"].shape == (2, 2)

    target = SoftPromptedTransformer(
        hidden_size=2,
        multi_modal_input_size=6,
        depth=1,
        num_heads=1,
        mlp_ratio=1.0,
        num_domains=2,
        dim_action=2,
        dim_propio=0,
        dim_time=1,
        len_soft_prompts=2,
        max_len_seq=6,
    )
    transformer_state = {
        key.removeprefix("model.transformer."): value
        for key, value in converted.items()
        if key.startswith("model.transformer.")
    }
    target.load_state_dict(transformer_state, strict=True)


def test_hidden_conversion_resizes_soft_prompts_when_prompt_length_is_unchanged():
    converted, _ = convert_xvla_state_dict(
        _make_source_state(),
        source_depth=2,
        target_depth=1,
        hidden_size=4,
        target_hidden_size=2,
        source_num_heads=2,
        target_num_heads=1,
        source_mlp_ratio=2.0,
        target_mlp_ratio=1.0,
        source_prompt_length=4,
        target_prompt_length=4,
    )

    assert converted["model.transformer.soft_prompt_hub.weight"].shape == (2, 8)


def test_extra_light_converter_writes_registered_checkpoint(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_config = {
        "type": "xvla_light",
        "hidden_size": 4,
        "num_heads": 2,
        "depth": 2,
        "mlp_ratio": 2.0,
        "len_soft_prompts": 4,
        "pretrained_path": None,
    }
    (source_dir / "config.json").write_text(json.dumps(source_config))
    (source_dir / "policy_preprocessor.json").write_text("{}")
    save_file(_make_source_state(), source_dir / "model.safetensors")

    output_dir = tmp_path / "extra-light"
    convert_checkpoint(
        str(source_dir),
        output_dir,
        target_depth=1,
        target_hidden_size=2,
        target_num_heads=1,
        target_mlp_ratio=1.0,
        target_prompt_length=2,
    )

    config = json.loads((output_dir / "config.json").read_text())
    state = load_file(output_dir / "model.safetensors")
    assert config["type"] == "xvla_extra_light"
    assert config["hidden_size"] == 2
    assert config["num_heads"] == 1
    assert state["model.transformer.blocks.0.attn.qkv.weight"].shape == (6, 2)
    assert isinstance(PreTrainedConfig.from_pretrained(output_dir), XVLAExtraLightConfig)
    assert (output_dir / "policy_preprocessor.json").is_file()
