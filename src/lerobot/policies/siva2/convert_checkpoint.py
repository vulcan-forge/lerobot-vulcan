# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Create a VLM-initialized SIVA2 checkpoint from a local XVLA checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

SUPPORTED_SOURCE_TYPES = {"xvla", "xvla_light", "xvla_extra_light"}
VLM_PREFIX = "model.vlm."


def _copy_processor_bundle(source: Path, output: Path) -> list[str]:
    """Copy processor JSON plus every state file referenced by those JSON files."""
    copied: list[str] = []
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        config_path = source / filename
        if not config_path.is_file():
            continue
        shutil.copy2(config_path, output / filename)
        copied.append(filename)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for step in config.get("steps", []):
            state_file = step.get("state_file")
            if state_file and state_file not in copied:
                state_path = source / state_file
                if not state_path.is_file():
                    raise FileNotFoundError(
                        f"Processor {filename} references missing state file {state_file}."
                    )
                shutil.copy2(state_path, output / state_file)
                copied.append(state_file)
    return copied


def convert_xvla_checkpoint(
    source: str | Path, output: str | Path, *, freeze_vlm: bool = True
) -> dict[str, object]:
    """Write a VLM-only SIVA2 initialization checkpoint and manifest."""
    source = Path(source).resolve()
    output = Path(output).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"XVLA checkpoint directory does not exist: {source}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    config_path = source / "config.json"
    weights_path = source / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("Source must contain config.json and model.safetensors.")
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    source_type = source_config.get("type")
    if source_type not in SUPPORTED_SOURCE_TYPES:
        raise ValueError(f"Expected an XVLA checkpoint, got type={source_type!r}.")

    target = dict(source_config)
    target.update(
        {
            "type": "siva2",
            "pretrained_path": None,
            "freeze_vlm": freeze_vlm,
            "xvla_init_source": str(source),
            "action_mode": "auto",
            "n_action_steps": min(
                int(source_config.get("n_action_steps", source_config.get("chunk_size", 32))),
                int(source_config.get("chunk_size", 32)),
            ),
            "n_obs_steps": 4,
            "hidden_size": 512,
            "depth": 8,
            "num_heads": 8,
            "mlp_ratio": 3.0,
            "scene_slots": 16,
            "memory_slots": 12,
            "goal_slots": 6,
            "subgoal_slots": 6,
            "slot_depth": 2,
            "fusion_depth": 2,
            "memory_depth": 2,
            "num_motion_primitives": 8,
            "num_strategies": 4,
            "num_control_points": min(6, int(source_config.get("chunk_size", 32))),
            "domain_adapter_rank": 4,
            "num_denoising_steps": 3,
        }
    )
    target.pop("cache_florence_features", None)
    target.pop("florence_cache_path", None)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(target, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    transferred = {}
    skipped = 0
    with safe_open(weights_path, framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():  # noqa: SIM118
            target_key = key if key.startswith("model.") else f"model.{key}"
            if target_key.startswith(VLM_PREFIX):
                transferred[target_key] = checkpoint.get_tensor(key)
            else:
                skipped += 1
    if not transferred:
        raise ValueError("The checkpoint contains no model.vlm.* weights to transfer.")
    save_file(transferred, output / "model.safetensors")
    copied = _copy_processor_bundle(source, output)
    manifest: dict[str, object] = {
        "format": 1,
        "source": str(source),
        "source_type": source_type,
        "transferred_prefixes": [VLM_PREFIX],
        "transferred_tensors": len(transferred),
        "skipped_tensors": skipped,
        "new_module_prefixes": [
            "model.scene_bank.",
            "model.memory.",
            "model.goal_bank.",
            "model.subgoal_bank.",
            "model.behavior.",
            "model.context_fusion.",
            "model.action_head.",
            "model.value_head.",
        ],
        "copied_processor_files": copied,
    }
    (output / "siva2_initialization.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Local XVLA pretrained_model directory.")
    parser.add_argument("--output-dir", required=True, help="New, empty SIVA2 initialization directory.")
    parser.add_argument("--train-vlm", action="store_true", help="Leave transferred Florence trainable.")
    args = parser.parse_args()
    manifest = convert_xvla_checkpoint(args.source, args.output_dir, freeze_vlm=not args.train_vlm)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
