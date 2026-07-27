# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import pytest
import torch
from safetensors.torch import load_file, save_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.siva.configuration_siva import SIVAConfig
from lerobot.policies.siva.convert_checkpoint import convert_xvla_checkpoint
from lerobot.policies.siva.florence_cache import SIVAFlorenceFeatureCache
from lerobot.policies.siva.modeling_siva import SIVAActionHead, SIVAModel, SIVAPolicy
from lerobot.policies.siva.structured_flow import MotionPriorLibrary, SlotCompressor
from lerobot.policies.xvla.action_hub import build_action_space
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_LANGUAGE_TOKENS, OBS_STATE


def _tiny_config() -> SIVAConfig:
    return SIVAConfig(
        device="cpu",
        chunk_size=6,
        n_action_steps=4,
        max_action_dim=4,
        max_state_dim=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        num_context_slots=4,
        slot_depth=1,
        num_motion_modes=3,
        num_control_points=3,
        domain_adapter_rank=2,
        num_domains=2,
        vision_token_dropout=0.0,
    )


def test_siva_is_registered_without_importing_a_florence_checkpoint():
    config = make_policy_config("siva", device="cpu")

    assert isinstance(config, SIVAConfig)
    assert isinstance(PreTrainedConfig.get_choice_class("siva")(), SIVAConfig)
    assert get_policy_class("siva") is SIVAPolicy


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"hidden_size": 30}, "divisible"),
        ({"num_motion_modes": 1}, "at least 2"),
        ({"noise_temporal_correlation": 1.1}, r"in \[0, 1\]"),
        ({"prior_noise_scale": 0.0}, "positive"),
    ],
)
def test_siva_rejects_invalid_architecture_settings(override, message):
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
        SIVAConfig(**kwargs)


def test_siva_cache_is_optional_and_requires_a_frozen_vlm(tmp_path):
    assert not _tiny_config().cache_florence_features

    with pytest.raises(ValueError, match="florence_cache_path"):
        SIVAConfig(
            device="cpu",
            chunk_size=6,
            n_action_steps=6,
            num_control_points=3,
            hidden_size=32,
            num_heads=4,
            freeze_vlm=True,
            cache_florence_features=True,
        )
    with pytest.raises(ValueError, match="freeze_vlm=True"):
        SIVAConfig(
            device="cpu",
            chunk_size=6,
            n_action_steps=6,
            num_control_points=3,
            hidden_size=32,
            num_heads=4,
            cache_florence_features=True,
            florence_cache_path=str(tmp_path / "features.sqlite"),
        )


def test_siva_florence_cache_reuses_complete_context(tmp_path):
    # Build only the cache-facing slice of SIVAModel; the cache behavior does
    # not require constructing Florence for this focused unit test.
    model = SIVAModel.__new__(SIVAModel)
    torch.nn.Module.__init__(model)
    model.config = type("CacheConfig", (), {"cache_florence_features": True})()
    model._florence_cache = SIVAFlorenceFeatureCache(tmp_path / "features.sqlite", "test-signature")
    model._cache_requests = 0
    model._cache_hits = 0
    model.training = True
    online_calls = 0

    def fake_online(input_ids, pixel_values, image_mask):
        nonlocal online_calls
        online_calls += 1
        batch_size = input_ids.shape[0]
        sample_value = input_ids[:, :1].to(torch.float32)
        return {
            "vlm_features": sample_value[:, :, None].expand(batch_size, 3, 5).clone(),
            "auxiliary_visual_features": sample_value[:, :, None].expand(batch_size, 2, 5).clone(),
            "context_mask": image_mask[:, :1].expand(batch_size, 5).bool().clone(),
        }

    model._forward_vlm_online = fake_online
    input_ids = torch.tensor([[11, 1], [22, 1]])
    pixel_values = torch.randn(2, 2, 3, 4, 4)
    image_mask = torch.ones(2, 2, dtype=torch.bool)
    cache_keys = torch.tensor([101, 202])

    first = model.forward_vlm(input_ids, pixel_values, image_mask, cache_keys=cache_keys)
    second = model.forward_vlm(input_ids, pixel_values, image_mask, cache_keys=cache_keys)

    assert online_calls == 1
    assert model._cache_hits == 2
    assert model._florence_cache.count() == 2
    assert first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)


def test_slot_compressor_has_fixed_output_size_for_different_input_lengths():
    compressor = SlotCompressor(
        input_size=12,
        hidden_size=32,
        proprio_size=4,
        num_slots=5,
        num_heads=4,
        depth=1,
        mlp_ratio=2.0,
        num_domains=2,
        token_dropout=0.0,
    )
    proprio = torch.randn(2, 4)
    domain = torch.tensor([0, 1])

    short = compressor(torch.randn(2, 7, 12), torch.randn(2, 3, 12), proprio, domain)
    long = compressor(torch.randn(2, 31, 12), torch.randn(2, 11, 12), proprio, domain)

    assert short.shape == long.shape == (2, 5, 32)


def test_slot_compressor_ignores_masked_camera_tokens():
    torch.manual_seed(3)
    compressor = SlotCompressor(
        input_size=8,
        hidden_size=16,
        proprio_size=0,
        num_slots=3,
        num_heads=4,
        depth=1,
        mlp_ratio=2.0,
        num_domains=1,
        token_dropout=0.0,
    ).eval()
    primary = torch.randn(1, 5, 8)
    masked_camera = torch.randn(1, 2, 8)
    mask = torch.tensor([[True, True, True, True, True, False, False]])

    first = compressor(primary, masked_camera, torch.empty(1, 0), torch.zeros(1, dtype=torch.long), mask)
    second = compressor(
        primary,
        masked_camera * 1_000,
        torch.empty(1, 0),
        torch.zeros(1, dtype=torch.long),
        mask,
    )

    assert torch.allclose(first, second)


def test_motion_prior_shares_base_and_applies_domain_adapter():
    library = MotionPriorLibrary(
        num_modes=3,
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
    assert initial.shape == (2, 3, 6, 4)
    assert torch.allclose(initial[0], initial[1])

    with torch.no_grad():
        library.domain_coefficients.weight[1, 0] = 1.0
    adapted = library.all_trajectories(domains)
    assert not torch.allclose(adapted[0], adapted[1])


def test_action_head_trains_all_three_paths_and_masks_padding():
    torch.manual_seed(7)
    config = _tiny_config()
    action_space = build_action_space("auto", real_dim=4, max_dim=4)
    head = SIVAActionHead(config, action_dim=4, action_space=action_space)
    slots = torch.randn(3, config.num_context_slots, config.hidden_size, requires_grad=True)
    actions = torch.randn(3, config.chunk_size, 4)
    action_is_pad = torch.tensor(
        [
            [False, False, False, False, False, False],
            [False, False, False, True, True, True],
            [False, True, True, True, True, True],
        ]
    )

    losses = head(slots, actions, torch.tensor([0, 1, 0]), action_is_pad)
    total = sum(losses.values())
    total.backward()

    assert torch.isfinite(total)
    assert {"flow_loss", "prior_loss", "router_loss", "router_balance_loss"} <= losses.keys()
    assert head.flow.output_projection.weight.grad is not None
    assert head.priors.control_points.grad is not None
    assert head.router[-1].weight.grad is not None
    assert slots.grad is not None


def test_action_head_ignores_auto_action_padding():
    torch.manual_seed(17)
    config = _tiny_config()
    action_space = build_action_space("auto", real_dim=4, max_dim=6)
    head = SIVAActionHead(config, action_dim=6, action_space=action_space)
    slots = torch.randn(2, config.num_context_slots, config.hidden_size)
    actions = torch.randn(2, config.chunk_size, 6)
    domains = torch.tensor([0, 1])

    changed_padding = actions.clone()
    changed_padding[..., 4:] = 10_000 * torch.randn_like(changed_padding[..., 4:])

    torch.manual_seed(23)
    original_losses = head(slots, actions, domains)
    torch.manual_seed(23)
    changed_losses = head(slots, changed_padding, domains)

    assert head.real_action_dim == 4
    assert torch.equal(head.flow_weights[4:], torch.zeros(2))
    for name in original_losses:
        assert torch.allclose(original_losses[name], changed_losses[name])


def test_action_head_generation_is_deterministic_and_has_requested_horizon():
    config = _tiny_config()
    action_space = build_action_space("auto", real_dim=4, max_dim=4)
    head = SIVAActionHead(config, action_dim=4, action_space=action_space).eval()
    slots = torch.randn(2, config.num_context_slots, config.hidden_size)
    domains = torch.tensor([0, 1])

    first = head.generate(slots, domains, steps=3)
    second = head.generate(slots, domains, steps=3)

    assert first.shape == (2, config.chunk_size, 4)
    assert torch.equal(first, second)


def test_xvla_converter_transfers_only_vlm_and_records_new_modules(tmp_path):
    source = tmp_path / "xvla"
    source.mkdir()
    (source / "config.json").write_text(
        '{"type":"xvla_light","cache_florence_features":true,"florence_cache_path":"cache"}'
    )
    state_name = "policy_preprocessor_step_4_normalizer.safetensors"
    (source / "policy_preprocessor.json").write_text(f'{{"steps":[{{"state_file":"{state_name}"}}]}}')
    save_file({"mean": torch.ones(2)}, source / state_name)
    save_file(
        {
            "model.vlm.encoder.weight": torch.ones(2, 2),
            "model.transformer.action.weight": torch.zeros(3, 3),
        },
        source / "model.safetensors",
    )

    output = tmp_path / "siva"
    manifest = convert_xvla_checkpoint(source, output)
    config = PreTrainedConfig.from_pretrained(output)
    weights = load_file(output / "model.safetensors")

    assert isinstance(config, SIVAConfig)
    assert config.freeze_vlm
    assert config.xvla_init_source == str(source.resolve())
    assert "cache_florence_features" not in (output / "config.json").read_text()
    assert set(weights) == {"model.vlm.encoder.weight"}
    assert manifest["transferred_tensors"] == 1
    assert manifest["skipped_tensors"] == 1
    assert (output / "policy_preprocessor.json").is_file()
    assert (output / state_name).is_file()
    assert (output / "siva_initialization.json").is_file()


def test_tiny_policy_runs_florence_forward_backward_and_inference():
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
    config = _tiny_config()
    config.florence_config = florence_config
    config.input_features = {
        image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))}
    # Florence's tiny, nonstandard random configuration is only a shape fixture.
    # Zeroing its representation weights avoids flaky numerical spikes that do
    # not occur with the pretrained checkpoint used by SIVA.
    policy = SIVAPolicy(config).eval()
    with torch.no_grad():
        for parameter in policy.model.vlm.parameters():
            parameter.zero_()
    batch = {
        image_key: torch.rand(2, 3, 32, 32),
        OBS_STATE: torch.randn(2, 4),
        OBS_LANGUAGE_TOKENS: torch.randint(3, 64, (2, 8)),
        ACTION: torch.randn(2, config.chunk_size, 4),
        "domain_id": torch.tensor([0, 1]),
        "action_is_pad": torch.tensor([[False] * config.chunk_size, [False] * 4 + [True] * 2]),
    }

    loss, log = policy(batch)
    loss.backward()
    inference_batch = {key: value for key, value in batch.items() if key not in (ACTION, "action_is_pad")}
    action = policy.predict_action_chunk(inference_batch)

    assert torch.isfinite(loss)
    assert action.shape == (2, config.chunk_size, 4)
    assert "flow_loss" in log
    assert policy.model.action_head.flow.output_projection.weight.grad is not None
