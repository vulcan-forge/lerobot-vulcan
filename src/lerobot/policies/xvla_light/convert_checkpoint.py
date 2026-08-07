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

"""Utilities for transferring an XVLA checkpoint into XVLA-light."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file, save_file

_BLOCK_PREFIX = "model.transformer.blocks."
_SOFT_PROMPT_KEY = "model.transformer.soft_prompt_hub.weight"
_COMPANION_PREFIXES = ("policy_preprocessor", "policy_postprocessor")


def select_evenly_spaced_layers(source_depth: int, target_depth: int) -> list[int]:
    """Choose target layers spread from the start to the end of the source stack."""
    if source_depth <= 0 or target_depth <= 0:
        raise ValueError("Source and target depth must both be positive.")
    if target_depth > source_depth:
        raise ValueError(f"Target depth ({target_depth}) cannot exceed source depth ({source_depth}).")
    if target_depth == 1:
        return [source_depth - 1]
    return torch.linspace(0, source_depth - 1, target_depth).round().to(torch.int64).tolist()


def _select_mlp_neurons(
    fc1_weight: torch.Tensor, fc2_weight: torch.Tensor, target_width: int
) -> torch.Tensor:
    source_width = fc1_weight.shape[0]
    if fc2_weight.shape[1] != source_width:
        raise ValueError("fc1 and fc2 do not have matching MLP widths.")
    if not 0 < target_width <= source_width:
        raise ValueError(f"Target MLP width must be in [1, {source_width}], got {target_width}.")
    if target_width == source_width:
        return torch.arange(source_width, device=fc1_weight.device)

    importance = torch.linalg.vector_norm(fc1_weight.float(), dim=1)
    importance += torch.linalg.vector_norm(fc2_weight.float(), dim=0)
    # Preserve the source ordering of the retained neurons after ranking by importance.
    return importance.topk(target_width, largest=True, sorted=False).indices.sort().values


def _compress_soft_prompts(
    weight: torch.Tensor,
    *,
    source_length: int,
    target_length: int,
    source_hidden_size: int,
    hidden_indices: torch.Tensor,
) -> torch.Tensor:
    if target_length <= 0 or target_length > source_length:
        raise ValueError(f"Target soft-prompt length must be in [1, {source_length}], got {target_length}.")
    expected_width = source_length * source_hidden_size
    if weight.ndim != 2 or weight.shape[1] != expected_width:
        raise ValueError(
            f"Expected soft prompts shaped [num_domains, {expected_width}], got {tuple(weight.shape)}."
        )
    prompts = weight.reshape(weight.shape[0], source_length, source_hidden_size)
    if target_length != source_length:
        prompts = prompts.transpose(1, 2)
        prompts = F.adaptive_avg_pool1d(prompts.float(), target_length).to(weight.dtype)
        prompts = prompts.transpose(1, 2)
    compressed = prompts.index_select(2, hidden_indices)
    return compressed.reshape(weight.shape[0], target_length * hidden_indices.numel())


def _select_attention_heads(
    source_state: dict[str, torch.Tensor],
    *,
    layer_map: list[int],
    source_hidden_size: int,
    target_hidden_size: int,
    source_num_heads: int,
    target_num_heads: int,
) -> tuple[torch.Tensor, list[int]]:
    """Select complete source attention heads and their residual-channel groups."""
    if source_hidden_size % source_num_heads != 0 or target_hidden_size % target_num_heads != 0:
        raise ValueError("Source and target hidden sizes must be divisible by their attention-head counts.")
    source_head_dim = source_hidden_size // source_num_heads
    target_head_dim = target_hidden_size // target_num_heads
    if source_head_dim != target_head_dim:
        raise ValueError(
            "Hidden-size conversion preserves complete attention heads and therefore requires an unchanged "
            f"head dimension, got source={source_head_dim} and target={target_head_dim}."
        )
    if target_num_heads > source_num_heads:
        raise ValueError(
            f"Target head count ({target_num_heads}) cannot exceed source head count ({source_num_heads})."
        )
    if target_hidden_size == source_hidden_size and target_num_heads == source_num_heads:
        return torch.arange(source_hidden_size), list(range(source_num_heads))

    importance = torch.zeros(source_num_heads, dtype=torch.float32)
    for source_layer in layer_map:
        prefix = f"{_BLOCK_PREFIX}{source_layer}.attn."
        qkv = (
            source_state[f"{prefix}qkv.weight"]
            .float()
            .reshape(3, source_num_heads, source_head_dim, source_hidden_size)
        )
        projection = (
            source_state[f"{prefix}proj.weight"]
            .float()
            .reshape(source_hidden_size, source_num_heads, source_head_dim)
        )
        importance += qkv.square().sum(dim=(0, 2, 3)).sqrt()
        importance += projection.square().sum(dim=(0, 2)).sqrt()

    selected_heads = importance.topk(target_num_heads, largest=True, sorted=False).indices.sort().values
    hidden_indices = torch.cat(
        [
            torch.arange(head * source_head_dim, (head + 1) * source_head_dim)
            for head in selected_heads.tolist()
        ]
    )
    return hidden_indices, selected_heads.tolist()


def _convert_outer_transformer_tensors(
    converted: dict[str, torch.Tensor],
    source_state: dict[str, torch.Tensor],
    *,
    source_hidden_size: int,
    hidden_indices: torch.Tensor,
) -> None:
    """Resize non-block policy-transformer tensors along the residual stream."""
    simple_dim0_keys = (
        "model.transformer.norm.weight",
        "model.transformer.norm.bias",
        "model.transformer.vlm_proj.bias",
        "model.transformer.aux_visual_proj.bias",
    )
    for key in simple_dim0_keys:
        if key in source_state:
            converted[key] = source_state[key].index_select(0, hidden_indices)

    if "model.transformer.pos_emb" in source_state:
        converted["model.transformer.pos_emb"] = source_state["model.transformer.pos_emb"].index_select(
            2, hidden_indices
        )
    for key in ("model.transformer.vlm_proj.weight", "model.transformer.aux_visual_proj.weight"):
        if key in source_state:
            converted[key] = source_state[key].index_select(0, hidden_indices)

    # Domain-aware VLM projections store [domain, input_size * hidden_size].
    for stem in ("model.transformer.vlm_proj", "model.transformer.aux_visual_proj"):
        weight_key = f"{stem}.fc.weight"
        bias_key = f"{stem}.bias.weight"
        if weight_key in source_state:
            weight = source_state[weight_key]
            input_size = weight.shape[1] // source_hidden_size
            converted[weight_key] = (
                weight.reshape(weight.shape[0], input_size, source_hidden_size)
                .index_select(2, hidden_indices)
                .reshape(weight.shape[0], -1)
            )
            converted[bias_key] = source_state[bias_key].index_select(1, hidden_indices)

    action_encoder_weight = source_state["model.transformer.action_encoder.fc.weight"]
    action_input_size = action_encoder_weight.shape[1] // source_hidden_size
    converted["model.transformer.action_encoder.fc.weight"] = (
        action_encoder_weight.reshape(action_encoder_weight.shape[0], action_input_size, source_hidden_size)
        .index_select(2, hidden_indices)
        .reshape(action_encoder_weight.shape[0], -1)
    )
    converted["model.transformer.action_encoder.bias.weight"] = source_state[
        "model.transformer.action_encoder.bias.weight"
    ].index_select(1, hidden_indices)

    action_decoder_weight = source_state["model.transformer.action_decoder.fc.weight"]
    action_output_size = action_decoder_weight.shape[1] // source_hidden_size
    converted["model.transformer.action_decoder.fc.weight"] = (
        action_decoder_weight.reshape(action_decoder_weight.shape[0], source_hidden_size, action_output_size)
        .index_select(1, hidden_indices)
        .reshape(action_decoder_weight.shape[0], -1)
    )


def convert_xvla_state_dict(
    source_state: dict[str, torch.Tensor],
    *,
    source_depth: int,
    target_depth: int,
    hidden_size: int,
    target_hidden_size: int | None = None,
    source_num_heads: int | None = None,
    target_num_heads: int | None = None,
    source_mlp_ratio: float,
    target_mlp_ratio: float,
    source_prompt_length: int,
    target_prompt_length: int,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    """Map an XVLA state dict to a smaller transformer without constructing either model."""
    target_hidden_size = target_hidden_size or hidden_size
    source_num_heads = source_num_heads or target_num_heads
    target_num_heads = target_num_heads or source_num_heads
    if source_num_heads is None or target_num_heads is None:
        if target_hidden_size != hidden_size:
            raise ValueError("Source and target head counts are required when changing hidden size.")
        source_num_heads = target_num_heads = 1

    source_mlp_width = int(hidden_size * source_mlp_ratio)
    target_mlp_width = int(target_hidden_size * target_mlp_ratio)
    if target_mlp_width > source_mlp_width:
        raise ValueError(
            f"Target MLP width ({target_mlp_width}) cannot exceed source width ({source_mlp_width})."
        )

    layer_map = select_evenly_spaced_layers(source_depth, target_depth)
    hidden_indices, _ = _select_attention_heads(
        source_state,
        layer_map=layer_map,
        source_hidden_size=hidden_size,
        target_hidden_size=target_hidden_size,
        source_num_heads=source_num_heads,
        target_num_heads=target_num_heads,
    )
    converted = {key: value for key, value in source_state.items() if not key.startswith(_BLOCK_PREFIX)}
    shrinking_hidden = target_hidden_size != hidden_size
    if shrinking_hidden:
        _convert_outer_transformer_tensors(
            converted,
            source_state,
            source_hidden_size=hidden_size,
            hidden_indices=hidden_indices,
        )

    for target_layer, source_layer in enumerate(layer_map):
        source_prefix = f"{_BLOCK_PREFIX}{source_layer}."
        target_prefix = f"{_BLOCK_PREFIX}{target_layer}."
        layer_keys = [key for key in source_state if key.startswith(source_prefix)]
        if not layer_keys:
            raise KeyError(f"No checkpoint tensors found for source transformer layer {source_layer}.")

        transformed_suffixes = {"mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight"}
        if shrinking_hidden:
            transformed_suffixes.update(
                {
                    "norm1.weight",
                    "norm1.bias",
                    "norm2.weight",
                    "norm2.bias",
                    "attn.qkv.weight",
                    "attn.qkv.bias",
                    "attn.proj.weight",
                    "attn.proj.bias",
                    "mlp.fc2.bias",
                }
            )
        for source_key in layer_keys:
            suffix = source_key.removeprefix(source_prefix)
            if suffix in transformed_suffixes:
                continue
            converted[f"{target_prefix}{suffix}"] = source_state[source_key]

        if shrinking_hidden:
            for norm in ("norm1", "norm2"):
                for parameter in ("weight", "bias"):
                    suffix = f"{norm}.{parameter}"
                    converted[f"{target_prefix}{suffix}"] = source_state[
                        f"{source_prefix}{suffix}"
                    ].index_select(0, hidden_indices)

            qkv_rows = torch.cat([hidden_indices + section * hidden_size for section in range(3)])
            converted[f"{target_prefix}attn.qkv.weight"] = (
                source_state[f"{source_prefix}attn.qkv.weight"]
                .index_select(0, qkv_rows)
                .index_select(1, hidden_indices)
            )
            converted[f"{target_prefix}attn.qkv.bias"] = source_state[
                f"{source_prefix}attn.qkv.bias"
            ].index_select(0, qkv_rows)
            converted[f"{target_prefix}attn.proj.weight"] = (
                source_state[f"{source_prefix}attn.proj.weight"]
                .index_select(0, hidden_indices)
                .index_select(1, hidden_indices)
            )
            converted[f"{target_prefix}attn.proj.bias"] = source_state[
                f"{source_prefix}attn.proj.bias"
            ].index_select(0, hidden_indices)

        fc1_weight = source_state[f"{source_prefix}mlp.fc1.weight"]
        fc1_bias = source_state[f"{source_prefix}mlp.fc1.bias"]
        fc2_weight = source_state[f"{source_prefix}mlp.fc2.weight"]
        if fc1_weight.shape[0] != source_mlp_width:
            raise ValueError(
                f"Layer {source_layer} has MLP width {fc1_weight.shape[0]}, but its config declares "
                f"{source_mlp_width}."
            )
        neurons = _select_mlp_neurons(fc1_weight, fc2_weight, target_mlp_width)
        converted[f"{target_prefix}mlp.fc1.weight"] = fc1_weight.index_select(0, neurons).index_select(
            1, hidden_indices
        )
        converted[f"{target_prefix}mlp.fc1.bias"] = fc1_bias.index_select(0, neurons)
        converted[f"{target_prefix}mlp.fc2.weight"] = fc2_weight.index_select(0, hidden_indices).index_select(
            1, neurons
        )
        if shrinking_hidden:
            converted[f"{target_prefix}mlp.fc2.bias"] = source_state[
                f"{source_prefix}mlp.fc2.bias"
            ].index_select(0, hidden_indices)

    if target_prompt_length == 0:
        converted.pop(_SOFT_PROMPT_KEY, None)
    else:
        if _SOFT_PROMPT_KEY not in source_state:
            raise KeyError(f"{_SOFT_PROMPT_KEY} is missing from the source checkpoint.")
        converted[_SOFT_PROMPT_KEY] = _compress_soft_prompts(
            source_state[_SOFT_PROMPT_KEY],
            source_length=source_prompt_length,
            target_length=target_prompt_length,
            source_hidden_size=hidden_size,
            hidden_indices=hidden_indices,
        )

    return converted, layer_map


def make_xvla_light_config(
    source_config: dict[str, Any],
    *,
    target_depth: int,
    target_mlp_ratio: float,
    target_prompt_length: int,
    target_hidden_size: int | None = None,
    target_num_heads: int | None = None,
    target_policy_type: str = "xvla_light",
) -> dict[str, Any]:
    """Return the serialized config for the converted checkpoint."""
    if source_config.get("type") not in {"xvla", "xvla_light", "xvla_extra_light"}:
        raise ValueError(f"Expected an XVLA checkpoint, got policy type {source_config.get('type')!r}.")

    target_config = dict(source_config)
    target_config.update(
        {
            "type": target_policy_type,
            "depth": target_depth,
            "mlp_ratio": target_mlp_ratio,
            "len_soft_prompts": target_prompt_length,
            "pretrained_path": None,
        }
    )
    if target_hidden_size is not None:
        target_config["hidden_size"] = target_hidden_size
    if target_num_heads is not None:
        target_config["num_heads"] = target_num_heads
    return target_config


def _resolve_source(source: str, *, revision: str | None, token: str | None) -> tuple[Path, Path, list[Path]]:
    local_source = Path(source).expanduser()
    if local_source.is_dir():
        config_path = local_source / "config.json"
        model_path = local_source / "model.safetensors"
        companions = [
            path
            for path in local_source.iterdir()
            if path.is_file() and path.name.startswith(_COMPANION_PREFIXES)
        ]
    else:
        config_path = Path(hf_hub_download(source, "config.json", revision=revision, token=token))
        model_path = Path(hf_hub_download(source, "model.safetensors", revision=revision, token=token))
        repo_files = list_repo_files(source, revision=revision, token=token)
        companions = [
            Path(hf_hub_download(source, filename, revision=revision, token=token))
            for filename in repo_files
            if "/" not in filename and filename.startswith(_COMPANION_PREFIXES)
        ]

    for required_path in (config_path, model_path):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)
    return config_path, model_path, companions


def convert_checkpoint(
    source: str,
    output_dir: Path,
    *,
    target_depth: int = 12,
    target_mlp_ratio: float = 2.0,
    target_prompt_length: int = 8,
    target_hidden_size: int | None = None,
    target_num_heads: int | None = None,
    target_policy_type: str = "xvla_light",
    target_model_name: str = "XVLA-light",
    revision: str | None = None,
    token: str | None = None,
) -> list[int]:
    """Convert a local or Hub XVLA checkpoint and save a loadable XVLA-light checkpoint."""
    config_path, model_path, companions = _resolve_source(source, revision=revision, token=token)
    with config_path.open() as config_file:
        source_config = json.load(config_file)

    source_depth = int(source_config["depth"])
    source_hidden_size = int(source_config["hidden_size"])
    source_num_heads = int(source_config["num_heads"])
    source_mlp_ratio = float(source_config["mlp_ratio"])
    source_prompt_length = int(source_config["len_soft_prompts"])
    target_hidden_size = target_hidden_size or source_hidden_size
    target_num_heads = target_num_heads or source_num_heads
    if target_prompt_length > source_prompt_length:
        raise ValueError(
            f"Target soft-prompt length ({target_prompt_length}) cannot exceed source length "
            f"({source_prompt_length})."
        )
    if (
        target_depth == source_depth
        and target_hidden_size == source_hidden_size
        and target_num_heads == source_num_heads
        and target_mlp_ratio == source_mlp_ratio
        and target_prompt_length == source_prompt_length
    ):
        raise ValueError(
            f"At least one {target_model_name} dimension must be smaller than the source checkpoint."
        )
    if target_hidden_size > source_hidden_size:
        raise ValueError(
            f"Target hidden size ({target_hidden_size}) cannot exceed source hidden size ({source_hidden_size})."
        )

    existing_files = list(output_dir.iterdir()) if output_dir.exists() else []
    if existing_files:
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading source tensors from %s", model_path)
    source_state = load_file(model_path, device="cpu")
    converted_state, layer_map = convert_xvla_state_dict(
        source_state,
        source_depth=source_depth,
        target_depth=target_depth,
        hidden_size=source_hidden_size,
        target_hidden_size=target_hidden_size,
        source_num_heads=source_num_heads,
        target_num_heads=target_num_heads,
        source_mlp_ratio=source_mlp_ratio,
        target_mlp_ratio=target_mlp_ratio,
        source_prompt_length=source_prompt_length,
        target_prompt_length=target_prompt_length,
    )
    target_config = make_xvla_light_config(
        source_config,
        target_depth=target_depth,
        target_mlp_ratio=target_mlp_ratio,
        target_prompt_length=target_prompt_length,
        target_hidden_size=target_hidden_size,
        target_num_heads=target_num_heads,
        target_policy_type=target_policy_type,
    )

    save_file(
        converted_state,
        output_dir / "model.safetensors",
        metadata={
            "format": "pt",
            "source_checkpoint": source,
            "xvla_layer_map": ",".join(map(str, layer_map)),
            "xvla_hidden_size": str(target_hidden_size),
            "xvla_num_heads": str(target_num_heads),
        },
    )
    with (output_dir / "config.json").open("w") as config_file:
        json.dump(target_config, config_file, indent=4)
        config_file.write("\n")
    for companion in companions:
        shutil.copy2(companion, output_dir / companion.name)

    logging.info("Saved %s checkpoint to %s", target_model_name, output_dir)
    return layer_map


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert an XVLA checkpoint into a smaller, strictly loadable XVLA-light checkpoint."
    )
    parser.add_argument("--source", required=True, help="Local pretrained_model directory or Hub repo ID.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=12, help="Target policy-transformer depth.")
    parser.add_argument("--mlp-ratio", type=float, default=2.0, help="Target policy-transformer MLP ratio.")
    parser.add_argument("--len-soft-prompts", type=int, default=8, help="Target soft-prompt count.")
    parser.add_argument("--revision", default=None, help="Optional Hub revision for a remote source.")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token for a private source.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    layer_map = convert_checkpoint(
        args.source,
        args.output_dir,
        target_depth=args.depth,
        target_mlp_ratio=args.mlp_ratio,
        target_prompt_length=args.len_soft_prompts,
        revision=args.revision,
        token=args.token,
    )
    logging.info("Layer map (target <- source): %s", list(enumerate(layer_map)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
