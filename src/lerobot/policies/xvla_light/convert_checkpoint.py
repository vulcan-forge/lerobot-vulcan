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
    hidden_size: int,
) -> torch.Tensor:
    if target_length <= 0 or target_length > source_length:
        raise ValueError(f"Target soft-prompt length must be in [1, {source_length}], got {target_length}.")
    expected_width = source_length * hidden_size
    if weight.ndim != 2 or weight.shape[1] != expected_width:
        raise ValueError(
            f"Expected soft prompts shaped [num_domains, {expected_width}], got {tuple(weight.shape)}."
        )
    if target_length == source_length:
        return weight

    prompts = weight.reshape(weight.shape[0], source_length, hidden_size).transpose(1, 2)
    compressed = F.adaptive_avg_pool1d(prompts.float(), target_length).to(weight.dtype)
    return compressed.transpose(1, 2).reshape(weight.shape[0], target_length * hidden_size)


def convert_xvla_state_dict(
    source_state: dict[str, torch.Tensor],
    *,
    source_depth: int,
    target_depth: int,
    hidden_size: int,
    source_mlp_ratio: float,
    target_mlp_ratio: float,
    source_prompt_length: int,
    target_prompt_length: int,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    """Map an XVLA state dict to a smaller transformer without constructing either model."""
    source_mlp_width = int(hidden_size * source_mlp_ratio)
    target_mlp_width = int(hidden_size * target_mlp_ratio)
    if target_mlp_width > source_mlp_width:
        raise ValueError(
            f"Target MLP width ({target_mlp_width}) cannot exceed source width ({source_mlp_width})."
        )

    layer_map = select_evenly_spaced_layers(source_depth, target_depth)
    converted = {key: value for key, value in source_state.items() if not key.startswith(_BLOCK_PREFIX)}

    for target_layer, source_layer in enumerate(layer_map):
        source_prefix = f"{_BLOCK_PREFIX}{source_layer}."
        target_prefix = f"{_BLOCK_PREFIX}{target_layer}."
        layer_keys = [key for key in source_state if key.startswith(source_prefix)]
        if not layer_keys:
            raise KeyError(f"No checkpoint tensors found for source transformer layer {source_layer}.")

        for source_key in layer_keys:
            suffix = source_key.removeprefix(source_prefix)
            if suffix in {"mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight"}:
                continue
            converted[f"{target_prefix}{suffix}"] = source_state[source_key]

        fc1_weight = source_state[f"{source_prefix}mlp.fc1.weight"]
        fc1_bias = source_state[f"{source_prefix}mlp.fc1.bias"]
        fc2_weight = source_state[f"{source_prefix}mlp.fc2.weight"]
        if fc1_weight.shape[0] != source_mlp_width:
            raise ValueError(
                f"Layer {source_layer} has MLP width {fc1_weight.shape[0]}, but its config declares "
                f"{source_mlp_width}."
            )
        neurons = _select_mlp_neurons(fc1_weight, fc2_weight, target_mlp_width)
        converted[f"{target_prefix}mlp.fc1.weight"] = fc1_weight.index_select(0, neurons)
        converted[f"{target_prefix}mlp.fc1.bias"] = fc1_bias.index_select(0, neurons)
        converted[f"{target_prefix}mlp.fc2.weight"] = fc2_weight.index_select(1, neurons)

    if target_prompt_length == 0:
        converted.pop(_SOFT_PROMPT_KEY, None)
    else:
        if _SOFT_PROMPT_KEY not in source_state:
            raise KeyError(f"{_SOFT_PROMPT_KEY} is missing from the source checkpoint.")
        converted[_SOFT_PROMPT_KEY] = _compress_soft_prompts(
            source_state[_SOFT_PROMPT_KEY],
            source_length=source_prompt_length,
            target_length=target_prompt_length,
            hidden_size=hidden_size,
        )

    return converted, layer_map


def make_xvla_light_config(
    source_config: dict[str, Any], *, target_depth: int, target_mlp_ratio: float, target_prompt_length: int
) -> dict[str, Any]:
    """Return the serialized config for the converted checkpoint."""
    if source_config.get("type") not in {"xvla", "xvla_light"}:
        raise ValueError(f"Expected an XVLA checkpoint, got policy type {source_config.get('type')!r}.")

    target_config = dict(source_config)
    target_config.update(
        {
            "type": "xvla_light",
            "depth": target_depth,
            "mlp_ratio": target_mlp_ratio,
            "len_soft_prompts": target_prompt_length,
            "pretrained_path": None,
        }
    )
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
    revision: str | None = None,
    token: str | None = None,
) -> list[int]:
    """Convert a local or Hub XVLA checkpoint and save a loadable XVLA-light checkpoint."""
    config_path, model_path, companions = _resolve_source(source, revision=revision, token=token)
    with config_path.open() as config_file:
        source_config = json.load(config_file)

    source_depth = int(source_config["depth"])
    source_mlp_ratio = float(source_config["mlp_ratio"])
    source_prompt_length = int(source_config["len_soft_prompts"])
    if target_prompt_length > source_prompt_length:
        raise ValueError(
            f"Target soft-prompt length ({target_prompt_length}) cannot exceed source length "
            f"({source_prompt_length})."
        )
    if (
        target_depth == source_depth
        and target_mlp_ratio == source_mlp_ratio
        and target_prompt_length == source_prompt_length
    ):
        raise ValueError("At least one XVLA-light dimension must be smaller than the source checkpoint.")

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
        hidden_size=int(source_config["hidden_size"]),
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
    )

    save_file(
        converted_state,
        output_dir / "model.safetensors",
        metadata={
            "format": "pt",
            "source_checkpoint": source,
            "xvla_layer_map": ",".join(map(str, layer_map)),
        },
    )
    with (output_dir / "config.json").open("w") as config_file:
        json.dump(target_config, config_file, indent=4)
        config_file.write("\n")
    for companion in companions:
        shutil.copy2(companion, output_dir / companion.name)

    logging.info("Saved XVLA-light checkpoint to %s", output_dir)
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
