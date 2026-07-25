# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Create a SIVA initialization checkpoint from a local XVLA checkpoint.

Only `model.vlm.*` is architecture-compatible.  The converter never guesses a
mapping for XVLA's action transformer; SIVA's slots, router, motion priors, and
flow decoder must be initialized and trained as new modules.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

SUPPORTED_SOURCE_TYPES = {"xvla", "xvla_light", "xvla_extra_light"}
VLM_PREFIX = "model.vlm."


def convert_xvla_checkpoint(
    source: str | Path,
    output: str | Path,
    *,
    freeze_vlm: bool = True,
) -> dict[str, object]:
    """Write a VLM-only SIVA initialization checkpoint and return its manifest."""
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

    target_config = dict(source_config)
    target_config.update(
        {
            "type": "siva",
            "pretrained_path": None,
            "freeze_vlm": freeze_vlm,
            "xvla_init_source": str(source),
            # SIVA architecture defaults.  Keeping these explicit in the emitted
            # config makes the initialization reproducible if defaults evolve.
            "hidden_size": 512,
            "depth": 8,
            "num_heads": 8,
            "mlp_ratio": 3.0,
            "num_context_slots": 16,
            "slot_depth": 2,
            "num_motion_modes": 8,
            "num_control_points": min(6, int(source_config.get("chunk_size", 32))),
            "domain_adapter_rank": 4,
            "num_denoising_steps": 3,
        }
    )
    # These belong to XVLA-light's feature cache, not to the SIVA config.
    target_config.pop("cache_florence_features", None)
    target_config.pop("florence_cache_path", None)

    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(target_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    transferred = {}
    skipped_count = 0
    with safe_open(weights_path, framework="pt", device="cpu") as checkpoint:
        # `safe_open` exposes keys() but is not itself iterable like a Mapping.
        for key in checkpoint.keys():  # noqa: SIM118
            target_key = key if key.startswith("model.") else f"model.{key}"
            if target_key.startswith(VLM_PREFIX):
                transferred[target_key] = checkpoint.get_tensor(key)
            else:
                skipped_count += 1
    if not transferred:
        raise ValueError("The checkpoint contains no model.vlm.* weights to transfer.")
    save_file(transferred, output / "model.safetensors")

    copied_files = []
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        candidate = source / filename
        if candidate.is_file():
            shutil.copy2(candidate, output / filename)
            copied_files.append(filename)

    manifest: dict[str, object] = {
        "format": 1,
        "source": str(source),
        "source_type": source_type,
        "transferred_prefixes": [VLM_PREFIX],
        "transferred_tensors": len(transferred),
        "skipped_tensors": skipped_count,
        "new_module_prefixes": ["model.context.", "model.action_head."],
        "copied_processor_files": copied_files,
    }
    (output / "siva_initialization.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Local XVLA pretrained_model directory.")
    parser.add_argument("--output-dir", required=True, help="New, empty SIVA initialization directory.")
    parser.add_argument(
        "--train-vlm",
        action="store_true",
        help="Leave Florence trainable instead of freezing the transferred representation.",
    )
    args = parser.parse_args()
    manifest = convert_xvla_checkpoint(args.source, args.output_dir, freeze_vlm=not args.train_vlm)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
