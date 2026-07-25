# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.siva.configuration_siva import SIVAConfig
from lerobot.policies.siva2.architecture_siva2 import (
    BehaviorConditionEncoder,
    BehaviorConditions,
    FactoredMotionPriorLibrary,
    TemporalMemoryEncoder,
)
from lerobot.policies.siva2.configuration_siva2 import SIVA2Config
from lerobot.policies.siva2.convert_checkpoint import convert_xvla_checkpoint
from lerobot.policies.siva2.modeling_siva2 import SIVA2ActionHead, SIVA2Policy
from lerobot.policies.xvla.action_hub import build_action_space
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def _tiny_config() -> SIVA2Config:
    return SIVA2Config(
        device="cpu",
        n_obs_steps=3,
        chunk_size=6,
        n_action_steps=4,
        max_action_dim=4,
        max_state_dim=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        scene_slots=4,
        memory_slots=3,
        goal_slots=2,
        subgoal_slots=2,
        slot_depth=1,
        fusion_depth=1,
        memory_depth=1,
        memory_spatial_queries=2,
        max_history_frames=3,
        num_motion_primitives=3,
        num_strategies=2,
        num_control_points=3,
        domain_adapter_rank=2,
        num_domains=2,
        vision_token_dropout=0.0,
        condition_dropout=0.0,
        subgoal_dropout=0.0,
    )


def test_siva2_is_separately_registered_and_siva_registration_is_unchanged():
    siva = make_policy_config("siva", device="cpu")
    siva2 = make_policy_config("siva2", device="cpu")

    assert isinstance(siva, SIVAConfig)
    assert not isinstance(siva, SIVA2Config)
    assert isinstance(siva2, SIVA2Config)
    assert isinstance(PreTrainedConfig.get_choice_class("siva2")(), SIVA2Config)
    assert get_policy_class("siva2") is SIVA2Policy


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"hidden_size": 30}, "divisible"),
        ({"memory_slots": 0}, "slot counts"),
        ({"num_strategies": 1}, "at least two"),
        ({"condition_dropout": 1.0}, r"in \[0, 1\)"),
    ],
)
def test_siva2_rejects_invalid_architecture_settings(override, message):
    kwargs = {
        "device": "cpu",
        "chunk_size": 6,
        "n_action_steps": 6,
        "num_control_points": 3,
        "hidden_size": 32,
        "num_heads": 4,
    }
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        SIVA2Config(**kwargs)


def test_temporal_memory_has_fixed_output_for_different_histories():
    encoder = TemporalMemoryEncoder(
        input_size=12,
        hidden_size=32,
        num_slots=5,
        spatial_queries=2,
        max_frames=4,
        num_heads=4,
        depth=1,
        mlp_ratio=2.0,
    ).eval()
    short = encoder(torch.randn(2, 2, 7, 12), torch.ones(2, 2, 7, dtype=torch.bool))
    long = encoder(torch.randn(2, 6, 7, 12), torch.ones(2, 6, 7, dtype=torch.bool))

    assert short.shape == long.shape == (2, 5, 32)


def test_behavior_encoder_distinguishes_missing_from_desired_quality():
    encoder = BehaviorConditionEncoder(16, num_control_modes=3, num_data_sources=4, dropout=0.0).eval()
    base = {
        "quality": torch.ones(2),
        "speed": torch.zeros(2),
        "mistake": torch.zeros(2),
        "advantage": torch.ones(2),
        "progress": torch.zeros(2),
        "control_mode": torch.zeros(2, dtype=torch.long),
        "data_source": torch.zeros(2, dtype=torch.long),
    }
    missing = encoder(BehaviorConditions(**base, present=torch.zeros(2, 7, dtype=torch.bool)))
    present = encoder(BehaviorConditions(**base, present=torch.ones(2, 7, dtype=torch.bool)))

    assert missing.shape == present.shape == (2, 7, 16)
    assert not torch.allclose(missing, present)


def test_factored_prior_shares_geometry_but_has_strategy_and_domain_offsets():
    library = FactoredMotionPriorLibrary(
        num_primitives=3,
        num_strategies=2,
        num_control_points=3,
        action_dim=4,
        horizon=6,
        num_domains=2,
        adapter_rank=2,
        init_scale=0.02,
        noise_scale=0.1,
    )
    domains = torch.tensor([0, 1])
    initial = library.all_trajectories(domains)
    assert initial.shape == (2, 3, 2, 6, 4)
    assert torch.allclose(initial[0], initial[1])
    assert not torch.allclose(initial[0, 0, 0], initial[0, 0, 1])

    with torch.no_grad():
        library.domain_coefficients.weight[1, 0] = 1.0
    assert not torch.allclose(library.all_trajectories(domains)[0], library.all_trajectories(domains)[1])


def test_action_head_trains_primitive_strategy_prior_and_flow_paths():
    torch.manual_seed(5)
    config = _tiny_config()
    action_space = build_action_space("auto", real_dim=4, max_dim=4)
    head = SIVA2ActionHead(config, action_dim=4, action_space=action_space)
    slots = torch.randn(3, 19, config.hidden_size, requires_grad=True)
    actions = torch.randn(3, config.chunk_size, 4)
    padding = torch.tensor(
        [
            [False] * 6,
            [False, False, False, True, True, True],
            [False, True, True, True, True, True],
        ]
    )
    losses = head(slots, actions, torch.tensor([0, 1, 0]), padding)
    sum(losses.values()).backward()

    assert {
        "flow_loss",
        "prior_loss",
        "primitive_router_loss",
        "strategy_router_loss",
    } <= losses.keys()
    assert head.flow.output_projection.weight.grad is not None
    assert head.priors.primitive_controls.grad is not None
    assert head.priors.strategy_offsets.grad is not None
    assert head.primitive_router.weight.grad is not None
    assert head.strategy_router.weight.grad is not None
    assert slots.grad is not None


def test_converter_copies_vlm_and_referenced_processor_state(tmp_path):
    source = tmp_path / "xvla"
    source.mkdir()
    (source / "config.json").write_text('{"type":"xvla","chunk_size":6}')
    state_name = "policy_preprocessor_step_4_normalizer.safetensors"
    (source / "policy_preprocessor.json").write_text(json.dumps({"steps": [{"state_file": state_name}]}))
    save_file({"mean": torch.ones(2)}, source / state_name)
    save_file(
        {
            "model.vlm.encoder.weight": torch.ones(2, 2),
            "model.transformer.action.weight": torch.zeros(3, 3),
        },
        source / "model.safetensors",
    )

    output = tmp_path / "siva2"
    manifest = convert_xvla_checkpoint(source, output)
    config = PreTrainedConfig.from_pretrained(output)
    weights = load_file(output / "model.safetensors")

    assert isinstance(config, SIVA2Config)
    assert config.freeze_vlm
    assert set(weights) == {"model.vlm.encoder.weight"}
    assert (output / state_name).is_file()
    assert state_name in manifest["copied_processor_files"]
    assert (output / "siva2_initialization.json").is_file()


def test_tiny_policy_runs_history_subgoal_value_forward_and_inference():
    torch.manual_seed(11)
    florence_config = {
        "vision_config": {
            "dim_embed": [8, 16, 24, 32],
            "num_heads": [1, 2, 3, 4],
            "num_groups": [1, 2, 3, 4],
            "depths": [1, 1, 1, 1],
            "window_size": 2,
            "projection_dim": 32,
            "drop_path_rate": 0.0,
        },
        "text_config": {
            "vocab_size": 64,
            "d_model": 32,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "encoder_ffn_dim": 64,
            "decoder_ffn_dim": 64,
            "encoder_attention_heads": 4,
            "decoder_attention_heads": 4,
            "max_position_embeddings": 64,
        },
        "projection_dim": 32,
        "vocab_size": 64,
        "pad_token_id": 1,
        "bos_token_id": 0,
        "eos_token_id": 2,
    }
    image_key = f"{OBS_IMAGES}.front"
    subgoal_key = "observation.subgoal.front"
    config = _tiny_config()
    config.florence_config = florence_config
    config.input_features = {
        image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        subgoal_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))}
    policy = SIVA2Policy(config).eval()
    with torch.no_grad():
        for parameter in policy.model.vlm.parameters():
            parameter.zero_()
    batch = {
        image_key: torch.rand(2, config.n_obs_steps, 3, 32, 32),
        f"{image_key}_is_pad": torch.tensor([[True, False, False], [False, False, False]]),
        subgoal_key: torch.rand(2, 3, 32, 32),
        OBS_STATE: torch.randn(2, config.n_obs_steps, 4),
        OBS_LANGUAGE_TOKENS: torch.randint(3, 64, (2, 8)),
        OBS_LANGUAGE_SUBTASK_TOKENS: torch.randint(3, 64, (2, 8)),
        OBS_LANGUAGE_SUBTASK_ATTENTION_MASK: torch.ones(2, 8, dtype=torch.long),
        ACTION: torch.randn(2, config.chunk_size, 4),
        "domain_id": torch.tensor([0, 1]),
        "action_is_pad": torch.tensor([[False] * 6, [False] * 4 + [True] * 2]),
        "siva2.quality": torch.tensor([5.0, 3.0]),
        "siva2.advantage": torch.tensor([1.0, -1.0]),
        "siva2.success": torch.tensor([1.0, 0.0]),
        "siva2.progress": torch.tensor([0.8, 0.3]),
        "siva2.time_to_completion": torch.tensor([100.0, 700.0]),
        "siva2.intervention": torch.tensor([0.0, 1.0]),
    }

    policy.train()
    loss, log = policy(batch)
    loss.backward()
    policy.eval()
    inference = {key: value for key, value in batch.items() if key not in (ACTION, "action_is_pad")}
    action = policy.predict_action_chunk(inference)
    values = policy.predict_values(inference)

    assert torch.isfinite(loss)
    assert action.shape == (2, config.chunk_size, 4)
    assert values.keys() == {
        "success_logit",
        "progress",
        "time_to_completion",
        "intervention_logit",
    }
    assert "value_success_loss" in log
    assert "value_progress_loss" in log
    assert policy.model.action_head.flow.output_projection.weight.grad is not None
    assert policy.model.value_head.output.weight.grad is not None
