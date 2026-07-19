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

"""Convert XVLA-family checkpoints into XVLA-extra-light."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lerobot.policies.xvla_light.convert_checkpoint import convert_checkpoint as convert_xvla_checkpoint


def convert_checkpoint(
    source: str,
    output_dir: Path,
    *,
    target_depth: int = 12,
    target_hidden_size: int = 512,
    target_num_heads: int = 8,
    target_mlp_ratio: float = 2.0,
    target_prompt_length: int = 8,
    revision: str | None = None,
    token: str | None = None,
) -> list[int]:
    """Save a strictly loadable XVLA-extra-light checkpoint."""
    return convert_xvla_checkpoint(
        source,
        output_dir,
        target_depth=target_depth,
        target_hidden_size=target_hidden_size,
        target_num_heads=target_num_heads,
        target_mlp_ratio=target_mlp_ratio,
        target_prompt_length=target_prompt_length,
        target_policy_type="xvla_extra_light",
        target_model_name="XVLA-extra-light",
        revision=revision,
        token=token,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert an XVLA-family checkpoint into a strictly loadable XVLA-extra-light checkpoint."
    )
    parser.add_argument("--source", required=True, help="Local pretrained_model directory or Hub repo ID.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--len-soft-prompts", type=int, default=8)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    layer_map = convert_checkpoint(
        args.source,
        args.output_dir,
        target_depth=args.depth,
        target_hidden_size=args.hidden_size,
        target_num_heads=args.num_heads,
        target_mlp_ratio=args.mlp_ratio,
        target_prompt_length=args.len_soft_prompts,
        revision=args.revision,
        token=args.token,
    )
    logging.info("Layer map (target <- source): %s", list(enumerate(layer_map)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
