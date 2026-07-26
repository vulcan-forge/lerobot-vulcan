# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

from pathlib import Path

import pytest
import yaml

RECIPE_DIR = (
    Path(__file__).parents[2]
    / "src"
    / "lerobot"
    / "scripts"
    / "sourccey"
    / "train"
    / "configs"
    / "siva_recipes"
)


@pytest.mark.parametrize(
    ("filename", "policy_name"),
    [
        ("siva_shirt_fold_c_010.yaml", "siva"),
        ("siva2_shirt_fold_c_010.yaml", "siva2"),
    ],
)
def test_siva_c010_training_recipe(filename: str, policy_name: str):
    recipe = yaml.safe_load((RECIPE_DIR / filename).read_text(encoding="utf-8"))

    assert recipe["dataset"]["repo_id"] == "Combination/sourccey-shirt-fold-c-010"
    if policy_name == "siva":
        assert recipe["dataset"]["root"] == (
            "/home/sourccey/.cache/huggingface/lerobot/Combination/sourccey-shirt-fold-c-010"
        )
        assert recipe["output_dir"] == "outputs/train/siva_sourccey-shirt-fold-c-010"
        assert recipe["job_name"] == "siva_sourccey-shirt-fold-c-010"
        assert recipe["policy"]["cache_florence_features"] is True
        assert recipe["policy"]["florence_cache_path"].endswith("florence_features.sqlite")
        assert recipe["dataset"]["image_transforms"]["enable"] is False
    assert recipe["policy"]["path"].endswith(f"to-{policy_name}")
    assert recipe["policy"]["dtype"] == "bfloat16"
    assert recipe["policy"]["input_features"] is None
    assert "action_mode" not in recipe["policy"]
    assert "max_action_dim" not in recipe["policy"]
    assert recipe["steps"] == 1_000_000
    assert recipe["save_freq"] == 50_000
    assert recipe["batch_size"] == 8
