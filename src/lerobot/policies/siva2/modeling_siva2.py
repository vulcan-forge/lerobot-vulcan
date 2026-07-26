# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""SIVA2: temporal, steerable, value-aware structured action generation."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.xvla.action_hub import BaseActionSpace, build_action_space
from lerobot.policies.xvla.configuration_florence2 import Florence2Config
from lerobot.policies.xvla.modeling_florence2 import Florence2ForConditionalGeneration
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy, resize_with_pad
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
)

from .architecture_siva2 import (
    BehaviorConditionEncoder,
    BehaviorConditions,
    FactoredMotionPriorLibrary,
    ResidualFlowDecoderV2,
    SlotBank,
    TemporalMemoryEncoder,
    TypedContextEncoder,
    ValueProgressHead,
    temporally_correlated_noise,
)
from .configuration_siva2 import SIVA2Config
from .florence_cache import SIVA2FlorenceFeatureCache


def _make_cache_signature(config: SIVA2Config, vlm: nn.Module) -> str:
    """Fingerprint SIVA2's Florence inputs and representative loaded weights."""
    settings = {
        "cache_boundary": "siva2_florence_context_v1",
        "florence_config": config.florence_config,
        "tokenizer_name": config.tokenizer_name,
        "tokenizer_max_length": config.tokenizer_max_length,
        "tokenizer_padding_side": config.tokenizer_padding_side,
        "pad_language_to": config.pad_language_to,
        "resize_imgs_with_padding": config.resize_imgs_with_padding,
        "num_image_views": config.num_image_views,
        "image_features": list(config.image_features),
        "n_obs_steps": config.n_obs_steps,
        "subgoal_feature_prefix": config.subgoal_feature_prefix,
        "dtype": config.dtype,
    }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode())
    for name, parameter in vlm.named_parameters():
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(str(parameter.dtype).encode())
        flattened = parameter.detach().reshape(-1)
        if flattened.numel() > 0:
            indices = torch.tensor(
                [0, flattened.numel() // 2, flattened.numel() - 1], device=flattened.device
            ).unique()
            digest.update(flattened.index_select(0, indices).float().cpu().numpy().tobytes())
    return digest.hexdigest()


def _masked_mean(value: Tensor, valid: Tensor) -> Tensor:
    while valid.ndim < value.ndim:
        valid = valid.unsqueeze(-1)
    valid = valid.expand_as(value)
    return value.masked_select(valid).sum() / valid.sum().clamp_min(1)


class SIVA2ActionHead(nn.Module):
    """Factored intent routing plus structured residual flow generation."""

    def __init__(self, config: SIVA2Config, action_dim: int, action_space: BaseActionSpace) -> None:
        super().__init__()
        self.config = config
        self.action_space = action_space
        self.router_trunk = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(approximate="tanh"),
        )
        self.primitive_router = nn.Linear(config.hidden_size, config.num_motion_primitives)
        self.strategy_router = nn.Linear(config.hidden_size, config.num_strategies)
        self.priors = FactoredMotionPriorLibrary(
            num_primitives=config.num_motion_primitives,
            num_strategies=config.num_strategies,
            num_control_points=config.num_control_points,
            action_dim=action_dim,
            horizon=config.chunk_size,
            num_domains=config.num_domains,
            adapter_rank=config.domain_adapter_rank,
            init_scale=config.prior_init_scale,
            noise_scale=config.prior_noise_scale,
        )
        self.flow = ResidualFlowDecoderV2(
            action_dim=action_dim,
            horizon=config.chunk_size,
            hidden_size=config.hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_primitives=config.num_motion_primitives,
            num_strategies=config.num_strategies,
            num_domains=config.num_domains,
        )
        flow_weights = torch.ones(action_dim)
        for index in action_space.gripper_idx:
            if index < action_dim:
                flow_weights[index] = 0.25
        self.register_buffer("flow_weights", flow_weights, persistent=False)

    def _route(self, slots: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.router_trunk(slots.mean(dim=1))
        temperature = self.config.routing_temperature
        return self.primitive_router(hidden) / temperature, self.strategy_router(hidden) / temperature

    def _assign(self, actions: Tensor, trajectories: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        error = (trajectories - actions[:, None, None]).square()
        weights = self.flow_weights.view(1, 1, 1, 1, -1)
        mask = valid[:, None, None, :, None]
        distance = (error * weights * mask).sum(dim=(3, 4))
        denominator = (mask * weights).sum(dim=(3, 4)).clamp_min(1)
        flat = (distance / denominator).flatten(1).argmin(dim=1)
        primitive = torch.div(flat, self.config.num_strategies, rounding_mode="floor")
        strategy = flat.remainder(self.config.num_strategies)
        return primitive, strategy

    def _endpoint_losses(self, predicted: Tensor, target: Tensor, valid: Tensor) -> dict[str, Tensor]:
        predicted_valid = predicted[valid].unsqueeze(1)
        target_valid = target[valid].unsqueeze(1)
        if predicted_valid.shape[0] == 0:
            return {"endpoint_loss": predicted.sum() * 0.0}
        return {
            f"endpoint_{name}": loss * self.config.endpoint_loss_weight
            for name, loss in self.action_space.compute_loss(predicted_valid, target_valid).items()
        }

    def forward(
        self,
        slots: Tensor,
        actions: Tensor,
        domain_id: Tensor,
        action_is_pad: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch_size, horizon, _ = actions.shape
        valid = (
            torch.ones((batch_size, horizon), dtype=torch.bool, device=actions.device)
            if action_is_pad is None
            else ~action_is_pad.to(device=actions.device, dtype=torch.bool)
        )
        trajectories = self.priors.all_trajectories(domain_id)
        primitive, strategy = self._assign(actions, trajectories, valid)
        primitive, strategy = primitive.detach(), strategy.detach()
        selected_prior = self.priors.select(trajectories, primitive, strategy)
        primitive_logits, strategy_logits = self._route(slots)

        scales = self.priors.select(self.priors.all_scales(batch_size), primitive, strategy)
        noise = temporally_correlated_noise(actions, self.config.noise_temporal_correlation)
        source = selected_prior.detach() + scales * noise
        time = (
            torch.rand(1, device=actions.device, dtype=actions.dtype)
            + torch.arange(batch_size, device=actions.device, dtype=actions.dtype) / batch_size
        ) % (1.0 - 1e-5)
        interpolated = source * (1.0 - time[:, None, None]) + actions * time[:, None, None]
        velocity = self.flow(interpolated, time, slots, primitive, strategy, domain_id)
        target_velocity = actions - source

        losses = {
            "flow_loss": _masked_mean((velocity - target_velocity).square() * self.flow_weights, valid)
            * self.config.flow_loss_weight,
            "prior_loss": _masked_mean((selected_prior - actions).square(), valid)
            * self.config.prior_loss_weight,
            "primitive_router_loss": F.cross_entropy(primitive_logits, primitive)
            * self.config.primitive_router_loss_weight,
            "strategy_router_loss": F.cross_entropy(strategy_logits, strategy)
            * self.config.strategy_router_loss_weight,
        }
        primitive_probability = primitive_logits.softmax(dim=-1).mean(dim=0)
        strategy_probability = strategy_logits.softmax(dim=-1).mean(dim=0)
        balance = (
            self.config.num_motion_primitives * primitive_probability.square().sum()
            + self.config.num_strategies * strategy_probability.square().sum()
            - 2.0
        )
        losses["router_balance_loss"] = balance * self.config.router_balance_loss_weight

        predicted_endpoint = interpolated + (1.0 - time[:, None, None]) * velocity
        losses.update(self._endpoint_losses(predicted_endpoint, actions, valid))
        if horizon >= 3:
            acceleration = (
                predicted_endpoint[:, 2:] - 2 * predicted_endpoint[:, 1:-1] + predicted_endpoint[:, :-2]
            )
            acceleration_valid = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
            losses["smoothness_loss"] = (
                _masked_mean(acceleration.square(), acceleration_valid) * self.config.smoothness_loss_weight
            )
        return losses

    @torch.no_grad()
    def generate(
        self,
        slots: Tensor,
        domain_id: Tensor,
        steps: int,
        noise: Tensor | None = None,
    ) -> Tensor:
        primitive_logits, strategy_logits = self._route(slots)
        primitive = primitive_logits.argmax(dim=-1)
        strategy = strategy_logits.argmax(dim=-1)
        trajectory = self.priors.select(self.priors.all_trajectories(domain_id), primitive, strategy)
        scale = self.priors.select(self.priors.all_scales(slots.shape[0]), primitive, strategy)
        if noise is None:
            noise = (
                torch.zeros_like(trajectory)
                if self.config.deterministic_inference
                else torch.randn_like(trajectory)
            )
        if noise.shape != trajectory.shape:
            raise ValueError(f"Expected inference noise {tuple(trajectory.shape)}, got {tuple(noise.shape)}.")
        action = trajectory + scale * noise
        steps = max(1, int(steps))
        for index in range(steps):
            time = torch.full((slots.shape[0],), index / steps, device=action.device, dtype=action.dtype)
            action = action + self.flow(action, time, slots, primitive, strategy, domain_id) / steps
        return action


class SIVA2Model(nn.Module):
    """Florence boundary plus typed context, memory, value, and action experts."""

    def __init__(self, config: SIVA2Config, florence_config: Florence2Config, proprio_dim: int) -> None:
        super().__init__()
        self.config = config
        self.chunk_size = config.chunk_size
        self.dim_proprio = proprio_dim
        if config.action_mode.lower() == "auto":
            real_dim = config.action_feature.shape[-1] if config.action_feature else config.max_action_dim
            self.action_space = build_action_space("auto", real_dim=real_dim, max_dim=config.max_action_dim)
        else:
            self.action_space = build_action_space(config.action_mode.lower())
        self.dim_action = self.action_space.dim_action

        # This name preserves direct compatibility with model.vlm.* XVLA tensors.
        self.vlm = Florence2ForConditionalGeneration(florence_config)
        if hasattr(self.vlm, "language_model"):
            language_model = self.vlm.language_model
            if hasattr(language_model, "model") and hasattr(language_model.model, "decoder"):
                del language_model.model.decoder
            if hasattr(language_model, "lm_head"):
                del language_model.lm_head
        projection_dim = getattr(self.vlm.config, "projection_dim", None)
        if projection_dim is None:
            raise ValueError("Florence2 config must provide `projection_dim`.")
        text_dim = self.vlm.get_input_embeddings().weight.shape[-1]

        self.proprio_projection = nn.Linear(proprio_dim, projection_dim) if proprio_dim > 0 else None
        self.scene_bank = SlotBank(
            projection_dim,
            config.hidden_size,
            config.scene_slots,
            config.num_heads,
            config.slot_depth,
            config.mlp_ratio,
            config.vision_token_dropout,
        )
        self.memory = TemporalMemoryEncoder(
            projection_dim,
            config.hidden_size,
            config.memory_slots,
            config.memory_spatial_queries,
            config.max_history_frames,
            config.num_heads,
            config.memory_depth,
            config.mlp_ratio,
        )
        self.goal_bank = SlotBank(
            text_dim,
            config.hidden_size,
            config.goal_slots,
            config.num_heads,
            config.slot_depth,
            config.mlp_ratio,
        )
        self.subgoal_bank = SlotBank(
            projection_dim,
            config.hidden_size,
            config.subgoal_slots,
            config.num_heads,
            config.slot_depth,
            config.mlp_ratio,
        )
        self.behavior = BehaviorConditionEncoder(
            config.hidden_size,
            config.num_control_modes,
            config.num_data_sources,
            config.condition_dropout,
        )
        self.domain_context = nn.Embedding(config.num_domains, config.hidden_size)
        self.context_fusion = TypedContextEncoder(
            config.hidden_size, config.num_heads, config.fusion_depth, config.mlp_ratio
        )
        self.action_head = SIVA2ActionHead(config, self.dim_action, self.action_space)
        self.value_head = ValueProgressHead(config.hidden_size)
        self._florence_cache: SIVA2FlorenceFeatureCache | None = None
        self._cache_requests = 0
        self._cache_hits = 0

        if config.freeze_vlm:
            self.vlm.requires_grad_(False)
            self.vlm.eval()
        else:
            if config.freeze_vision_encoder and hasattr(self.vlm, "vision_tower"):
                self.vlm.vision_tower.requires_grad_(False)
            if config.freeze_language_encoder and hasattr(self.vlm, "language_model"):
                self.vlm.language_model.requires_grad_(False)
        self.to(dtype=self._get_target_dtype())
        if config.cache_florence_features:
            logging.info("SIVA2 Florence feature cache enabled at %s", config.florence_cache_path)

    def _get_target_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32

    def train(self, mode: bool = True) -> SIVA2Model:
        super().train(mode)
        if self.config.freeze_vlm:
            self.vlm.eval()
        return self

    def _get_florence_cache(self) -> SIVA2FlorenceFeatureCache:
        # Construct after from_pretrained has loaded the transferred XVLA VLM.
        if self._florence_cache is None:
            self._florence_cache = SIVA2FlorenceFeatureCache(
                self.config.florence_cache_path,
                signature=_make_cache_signature(self.config, self.vlm),
            )
        return self._florence_cache

    def _encode_images(self, images: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Encode [B,...,C,H,W], retaining every leading grouping dimension."""
        leading = images.shape[:-3]
        flat_mask = mask.reshape(-1).bool()
        valid_images = images.reshape(-1, *images.shape[-3:])[flat_mask]
        if valid_images.shape[0] == 0:
            empty = images.new_zeros((*leading, 1, self.vlm.config.projection_dim))
            return empty, mask.new_zeros((*leading, 1))
        valid_features = self.vlm._encode_image(valid_images)
        token_count, hidden = valid_features.shape[1:]
        all_features = valid_features.new_zeros(mask.numel(), token_count, hidden)
        all_features[flat_mask] = valid_features
        features = all_features.view(*leading, token_count, hidden)
        token_mask = mask[..., None].expand(*leading, token_count).bool()
        return features, token_mask

    def _encode_scene_tokens(
        self, input_ids: Tensor, images: Tensor, image_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        image_features, image_token_mask = self._encode_images(images, image_mask)
        text_embeddings = self.vlm.get_input_embeddings()(input_ids)
        merged, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0], text_embeddings
        )
        pad_token_id = getattr(self.vlm.config, "pad_token_id", None)
        if pad_token_id is not None:
            attention_mask[:, -input_ids.shape[1] :] = input_ids.ne(pad_token_id).to(attention_mask.dtype)
        encoded = self.vlm.language_model.model.encoder(attention_mask=attention_mask, inputs_embeds=merged)[
            0
        ]
        auxiliary = image_features[:, 1:].flatten(1, 2)
        auxiliary_mask = image_token_mask[:, 1:].flatten(1, 2)
        tokens = torch.cat((encoded, auxiliary), dim=1)
        mask = torch.cat((attention_mask.bool(), auxiliary_mask), dim=1)
        return tokens, mask

    def _forward_florence_online(
        self,
        input_ids: Tensor,
        image_input: Tensor,
        image_mask: Tensor,
        history_input: Tensor,
        history_mask: Tensor,
        subgoal_input: Tensor,
        subgoal_mask: Tensor,
        subtask_ids: Tensor | None = None,
        subtask_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        scene_tokens, scene_mask = self._encode_scene_tokens(input_ids, image_input, image_mask)

        history_features, history_token_mask = self._encode_images(history_input, history_mask)
        history_features = history_features.flatten(2, 3)
        history_token_mask = history_token_mask.flatten(2, 3)

        goal_ids = input_ids if subtask_ids is None else subtask_ids
        goal_embeddings = self.vlm.get_input_embeddings()(goal_ids)
        if subtask_mask is None:
            pad_token_id = getattr(self.vlm.config, "pad_token_id", None)
            goal_mask = (
                torch.ones_like(goal_ids, dtype=torch.bool)
                if pad_token_id is None
                else goal_ids.ne(pad_token_id)
            )
        else:
            goal_mask = subtask_mask.bool()

        subgoal_features, subgoal_token_mask = self._encode_images(subgoal_input, subgoal_mask)
        return {
            "scene_tokens": scene_tokens,
            "scene_mask": scene_mask,
            "history_features": history_features,
            "history_token_mask": history_token_mask,
            "goal_embeddings": goal_embeddings,
            "goal_mask": goal_mask,
            "subgoal_features": subgoal_features.flatten(1, 2),
            "subgoal_token_mask": subgoal_token_mask.flatten(1, 2),
        }

    def forward_florence(
        self,
        input_ids: Tensor,
        image_input: Tensor,
        image_mask: Tensor,
        history_input: Tensor,
        history_mask: Tensor,
        subgoal_input: Tensor,
        subgoal_mask: Tensor,
        subtask_ids: Tensor | None = None,
        subtask_mask: Tensor | None = None,
        cache_keys: Tensor | None = None,
    ) -> dict[str, Tensor]:
        arguments = {
            "input_ids": input_ids,
            "image_input": image_input,
            "image_mask": image_mask,
            "history_input": history_input,
            "history_mask": history_mask,
            "subgoal_input": subgoal_input,
            "subgoal_mask": subgoal_mask,
            "subtask_ids": subtask_ids,
            "subtask_mask": subtask_mask,
        }
        if not self.config.cache_florence_features or cache_keys is None or not self.training:
            return self._forward_florence_online(**arguments)

        sample_indices = [int(index) for index in cache_keys.detach().cpu().reshape(-1).tolist()]
        if len(sample_indices) != input_ids.shape[0]:
            raise ValueError(
                f"Expected one SIVA2 cache key per sample, got {len(sample_indices)} keys for "
                f"batch size {input_ids.shape[0]}."
            )
        cache = self._get_florence_cache()
        cached = cache.get_many(sample_indices)
        missing_positions = [
            position for position, sample_index in enumerate(sample_indices) if sample_index not in cached
        ]
        online: dict[int, dict[str, Tensor]] = {}
        computed: dict[int, dict[str, Tensor]] = {}
        if missing_positions:
            missing = torch.tensor(missing_positions, device=input_ids.device, dtype=torch.long)
            missing_arguments = {
                name: value.index_select(0, missing) if value is not None else None
                for name, value in arguments.items()
            }
            with torch.no_grad():
                missing_features = self._forward_florence_online(**missing_arguments)
            for offset, position in enumerate(missing_positions):
                sample = {name: tensor[offset] for name, tensor in missing_features.items()}
                online[position] = sample
                computed[sample_indices[position]] = sample
            cache.put_many(computed)

        device = input_ids.device
        resolved: dict[str, list[Tensor]] = {}
        for position, sample_index in enumerate(sample_indices):
            sample = online.get(position, cached.get(sample_index))
            if sample is None:
                raise RuntimeError(f"Failed to resolve SIVA2 Florence features for index {sample_index}.")
            for name, tensor in sample.items():
                resolved.setdefault(name, []).append(tensor.to(device, non_blocking=True))

        self._cache_requests += len(sample_indices)
        self._cache_hits += len(sample_indices) - len(missing_positions)
        if self._cache_requests % 10_000 < len(sample_indices):
            logging.info(
                "SIVA2 Florence cache hit rate: %.1f%% (%d cached samples)",
                100 * self._cache_hits / self._cache_requests,
                cache.count(),
            )
        return {name: torch.stack(tensors) for name, tensors in resolved.items()}

    def encode_context(
        self,
        input_ids: Tensor,
        image_input: Tensor,
        image_mask: Tensor,
        history_input: Tensor,
        history_mask: Tensor,
        subgoal_input: Tensor,
        subgoal_mask: Tensor,
        domain_id: Tensor,
        proprio: Tensor,
        conditions: BehaviorConditions,
        subtask_ids: Tensor | None = None,
        subtask_mask: Tensor | None = None,
        cache_keys: Tensor | None = None,
        include_value_context: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=dtype)
        history_input = history_input.to(dtype=dtype)
        subgoal_input = subgoal_input.to(dtype=dtype)
        proprio = proprio.to(dtype=dtype)
        proprio_model, _ = self.action_space.preprocess(
            proprio, proprio.new_zeros((proprio.shape[0], self.chunk_size, self.dim_action))
        )
        features = self.forward_florence(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            history_input=history_input,
            history_mask=history_mask,
            subgoal_input=subgoal_input,
            subgoal_mask=subgoal_mask,
            subtask_ids=subtask_ids,
            subtask_mask=subtask_mask,
            cache_keys=cache_keys,
        )
        scene_tokens = features["scene_tokens"]
        scene_mask = features["scene_mask"]
        if self.proprio_projection is not None:
            scene_tokens = torch.cat(
                (scene_tokens, self.proprio_projection(proprio_model).unsqueeze(1)), dim=1
            )
            scene_mask = F.pad(scene_mask, (0, 1), value=True)
        scene = self.scene_bank(scene_tokens, scene_mask)
        memory = self.memory(features["history_features"], features["history_token_mask"])
        goal = self.goal_bank(features["goal_embeddings"], features["goal_mask"])

        subgoal_token_mask = features["subgoal_token_mask"]
        if self.training and self.config.subgoal_dropout > 0.0:
            keep = (
                torch.rand(subgoal_mask.shape[0], device=subgoal_mask.device) >= self.config.subgoal_dropout
            )
            subgoal_token_mask = subgoal_token_mask & keep.unsqueeze(1)
        subgoal = self.subgoal_bank(features["subgoal_features"], subgoal_token_mask)

        domain_context = self.domain_context(domain_id).unsqueeze(1)
        metadata = torch.cat((self.behavior(conditions), domain_context), dim=1)
        policy_slots = self.context_fusion(scene, memory, goal, subgoal, metadata)
        if not include_value_context:
            return policy_slots
        # A value function must judge the observed state, not copy desired
        # quality or the policy's advantage label.  It therefore receives the
        # same grounded context plus embodiment, but no behavior-condition tokens.
        value_slots = self.context_fusion(scene, memory, goal, subgoal, domain_context)
        return policy_slots, value_slots

    def _value_losses(self, predictions: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        losses: dict[str, Tensor] = {}
        specifications = (
            ("success", "success_logit", self.config.value_success_loss_weight, True),
            ("progress", "progress", self.config.value_progress_loss_weight, False),
            ("time_to_completion", "time_to_completion", self.config.value_time_loss_weight, False),
            ("intervention", "intervention_logit", self.config.value_intervention_loss_weight, True),
        )
        for target_name, prediction_name, weight, binary in specifications:
            if target_name not in targets:
                continue
            target = targets[target_name].to(dtype=predictions[prediction_name].dtype)
            valid = torch.isfinite(target)
            if not valid.any():
                continue
            prediction = predictions[prediction_name][valid]
            target = target[valid]
            loss = (
                F.binary_cross_entropy_with_logits(prediction, target)
                if binary
                else F.mse_loss(prediction, target)
            )
            losses[f"value_{target_name}_loss"] = loss * weight
        return losses

    def forward(
        self,
        action: Tensor,
        action_is_pad: Tensor | None = None,
        value_targets: dict[str, Tensor] | None = None,
        **context_inputs,
    ) -> dict[str, Tensor]:
        encoded = self.encode_context(**context_inputs, include_value_context=bool(value_targets))
        slots, value_slots = encoded if isinstance(encoded, tuple) else (encoded, None)
        losses = self.action_head(
            slots,
            action.to(dtype=self._get_target_dtype()),
            context_inputs["domain_id"],
            action_is_pad,
        )
        if value_targets and value_slots is not None:
            losses.update(self._value_losses(self.value_head(value_slots), value_targets))
        return losses

    @torch.no_grad()
    def generate_actions(self, steps: int, noise: Tensor | None = None, **context_inputs) -> Tensor:
        slots = self.encode_context(**context_inputs)
        action = self.action_head.generate(slots, context_inputs["domain_id"], steps=steps, noise=noise)
        return self.action_space.postprocess(action)

    @torch.no_grad()
    def predict_values(self, **context_inputs) -> dict[str, Tensor]:
        _, value_slots = self.encode_context(**context_inputs, include_value_context=True)
        return self.value_head(value_slots)


def _prepare_padding_mask(batch: dict[str, Tensor], targets: Tensor) -> Tensor:
    mask = batch.get("action_is_pad")
    if mask is None:
        return torch.zeros(targets.shape[:2], dtype=torch.bool, device=targets.device)
    mask = mask.to(device=targets.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[1] > targets.shape[1]:
        mask = mask[:, : targets.shape[1]]
    elif mask.shape[1] < targets.shape[1]:
        mask = F.pad(mask, (0, targets.shape[1] - mask.shape[1]), value=True)
    if mask.shape != targets.shape[:2]:
        raise ValueError(f"action_is_pad {tuple(mask.shape)} != actions {tuple(targets.shape[:2])}.")
    return mask


class SIVA2Policy(XVLAPolicy):
    """Registered LeRobot policy; SIVA remains a completely separate baseline."""

    config_class = SIVA2Config
    name = "siva2"

    def __init__(self, config: SIVA2Config, **kwargs) -> None:
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.model = SIVA2Model(
            config,
            config.get_florence_config(),
            config.max_state_dim if config.use_proprio else 0,
        )
        self.reset()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, strict: bool = False, **kwargs):
        return PreTrainedPolicy.from_pretrained.__func__(
            cls, pretrained_name_or_path, strict=strict, **kwargs
        )

    @property
    def _scene_image_keys(self) -> list[str]:
        return [
            key
            for key in self.config.image_features
            if not key.startswith(self.config.subgoal_feature_prefix)
        ]

    @property
    def _subgoal_image_keys(self) -> list[str]:
        return [
            key for key in self.config.image_features if key.startswith(self.config.subgoal_feature_prefix)
        ]

    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}
        for key in self._scene_image_keys:
            self._queues[key] = deque(maxlen=self.config.n_obs_steps)

    def _update_history_queues(self, batch: dict[str, Tensor]) -> None:
        """Append one live frame without fabricating repeated startup history."""
        for key in self._scene_image_keys:
            if key in batch and batch[key].ndim == 4:
                self._queues[key].append(batch[key])

    def _resize_sequence(self, images: Tensor) -> Tensor:
        if self.config.resize_imgs_with_padding is None:
            return images
        leading = images.shape[:-3]
        resized = resize_with_pad(
            images.reshape(-1, *images.shape[-3:]), *self.config.resize_imgs_with_padding
        )
        return resized.reshape(*leading, *resized.shape[-3:])

    def _image_sequence(self, batch: dict[str, Tensor], key: str) -> Tensor:
        value = batch[key]
        if value.ndim == 5:
            return value[:, -self.config.max_history_frames :]
        queue = self._queues.get(key)
        if queue and len(queue) > 0:
            return torch.stack(tuple(queue), dim=1)[:, -self.config.max_history_frames :]
        return value.unsqueeze(1)

    def _prepare_scene_and_history(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        keys = [key for key in self._scene_image_keys if key in batch]
        if not keys:
            raise ValueError(f"SIVA2 batch has no current camera from {self._scene_image_keys}.")
        sequences = [self._resize_sequence(self._image_sequence(batch, key)) for key in keys]
        frames = min(sequence.shape[1] for sequence in sequences)
        sequences = [sequence[:, -frames:] for sequence in sequences]
        masks = []
        for key, sequence in zip(keys, sequences, strict=True):
            pad = batch.get(f"{key}_is_pad")
            if pad is not None and batch[key].ndim == 5:
                valid = ~pad.to(device=sequence.device, dtype=torch.bool)[:, -frames:]
            else:
                valid = torch.ones(sequence.shape[:2], dtype=torch.bool, device=sequence.device)
            masks.append(valid)
        history = torch.stack(sequences, dim=2)
        history_mask = torch.stack(masks, dim=2)
        current = history[:, -1]
        current_mask = history_mask[:, -1]

        # The current images are already encoded with language by the scene
        # branch.  Memory receives only earlier frames, avoiding duplicate VLM
        # work while preserving a clean past-versus-present distinction.
        if history.shape[1] > 1:
            history = history[:, :-1]
            history_mask = history_mask[:, :-1]
        else:
            history_mask = torch.zeros_like(history_mask)

        total_views = max(self.config.num_image_views or current.shape[1], current.shape[1])
        if total_views > current.shape[1]:
            pad_views = total_views - current.shape[1]
            current = torch.cat(
                (current, current.new_zeros(current.shape[0], pad_views, *current.shape[2:])), dim=1
            )
            current_mask = F.pad(current_mask, (0, pad_views), value=False)
            history = torch.cat(
                (
                    history,
                    history.new_zeros(history.shape[0], history.shape[1], pad_views, *history.shape[3:]),
                ),
                dim=2,
            )
            history_mask = F.pad(history_mask, (0, pad_views), value=False)
        return current, current_mask, history, history_mask

    def _prepare_subgoals(self, batch: dict[str, Tensor], current: Tensor) -> tuple[Tensor, Tensor]:
        keys = [key for key in self._subgoal_image_keys if key in batch]
        if not keys:
            return current.new_zeros(current.shape[0], 1, *current.shape[2:]), torch.zeros(
                current.shape[0], 1, dtype=torch.bool, device=current.device
            )
        images = []
        for key in keys:
            value = batch[key][:, -1] if batch[key].ndim == 5 else batch[key]
            images.append(self._resize_sequence(value))
        stacked = torch.stack(images, dim=1)
        return stacked, torch.ones(stacked.shape[:2], dtype=torch.bool, device=stacked.device)

    @staticmethod
    def _batch_vector(
        batch: dict[str, Tensor], key: str, batch_size: int, device: torch.device, default: float
    ) -> tuple[Tensor, Tensor]:
        if key not in batch:
            return torch.full((batch_size,), default, device=device), torch.zeros(
                batch_size, dtype=torch.bool, device=device
            )
        value = batch[key]
        value = value if isinstance(value, Tensor) else torch.as_tensor(value)
        value = value.to(device=device).reshape(batch_size, -1)[:, -1].float()
        return value, torch.isfinite(value)

    def _prepare_conditions(
        self, batch: dict[str, Tensor], batch_size: int, device: torch.device, inference: bool
    ) -> BehaviorConditions:
        config = self.config
        quality, quality_present = self._batch_vector(
            batch,
            config.quality_key,
            batch_size,
            device,
            config.inference_quality if inference else config.default_quality,
        )
        speed, speed_present = self._batch_vector(batch, config.speed_key, batch_size, device, 0.0)
        mistake, mistake_present = self._batch_vector(batch, config.mistake_key, batch_size, device, 0.0)
        advantage, advantage_present = self._batch_vector(
            batch,
            config.advantage_key,
            batch_size,
            device,
            config.inference_advantage if inference else config.default_advantage,
        )
        desired_progress, progress_present = self._batch_vector(
            batch, config.desired_progress_key, batch_size, device, 0.0
        )
        control, control_present = self._batch_vector(batch, config.control_mode_key, batch_size, device, 0.0)
        source, source_present = self._batch_vector(batch, config.data_source_key, batch_size, device, 0.0)
        if inference or config.assume_demonstration_if_unlabeled:
            quality_present = torch.ones_like(quality_present)
            advantage_present = torch.ones_like(advantage_present)
            mistake_present = torch.ones_like(mistake_present)
        quality = ((quality - 1.0) / 4.0).clamp(0.0, 1.0)
        speed = torch.log1p(speed.clamp_min(0.0)) / torch.log1p(speed.new_tensor(config.speed_normalizer))
        present = torch.stack(
            (
                quality_present,
                speed_present,
                mistake_present,
                advantage_present,
                progress_present,
                control_present,
                source_present,
            ),
            dim=1,
        )
        return BehaviorConditions(
            quality=quality,
            speed=speed,
            mistake=mistake.clamp(0.0, 1.0),
            advantage=advantage.clamp(-1.0, 1.0),
            progress=desired_progress.clamp(0.0, 1.0),
            control_mode=control.long().clamp(0, config.num_control_modes - 1),
            data_source=source.long().clamp(0, config.num_data_sources - 1),
            present=present,
        )

    def _prepare_value_targets(
        self, batch: dict[str, Tensor], batch_size: int, device: torch.device
    ) -> dict[str, Tensor]:
        targets = {}
        specifications = (
            ("success", self.config.success_key, 1.0),
            ("progress", self.config.progress_key, 1.0),
            ("time_to_completion", self.config.time_to_completion_key, self.config.time_normalizer),
            ("intervention", self.config.intervention_key, 1.0),
        )
        for name, key, normalizer in specifications:
            if key in batch:
                value, _ = self._batch_vector(batch, key, batch_size, device, float("nan"))
                targets[name] = value / normalizer
        return targets

    def _build_siva2_inputs(self, batch: dict[str, Tensor], *, inference: bool) -> dict:
        input_ids = batch[OBS_LANGUAGE_TOKENS]
        batch_size = input_ids.shape[0]
        current, current_mask, history, history_mask = self._prepare_scene_and_history(batch)
        subgoal, subgoal_mask = self._prepare_subgoals(batch, current)
        inputs = {
            "input_ids": input_ids,
            "image_input": current,
            "image_mask": current_mask,
            "history_input": history,
            "history_mask": history_mask,
            "subgoal_input": subgoal,
            "subgoal_mask": subgoal_mask,
            "domain_id": self._get_domain_id(batch, batch_size, current.device),
            "proprio": self._prepare_state(batch, batch_size, current.device),
            "conditions": self._prepare_conditions(batch, batch_size, current.device, inference),
        }
        if OBS_LANGUAGE_SUBTASK_TOKENS in batch:
            inputs["subtask_ids"] = batch[OBS_LANGUAGE_SUBTASK_TOKENS]
            inputs["subtask_mask"] = batch.get(OBS_LANGUAGE_SUBTASK_ATTENTION_MASK)
        elif OBS_LANGUAGE_ATTENTION_MASK in batch:
            inputs["subtask_mask"] = batch[OBS_LANGUAGE_ATTENTION_MASK]
        return inputs

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        inputs = self._build_siva2_inputs(batch, inference=False)
        if self.config.cache_florence_features and self.training:
            if "index" not in batch:
                raise KeyError("SIVA2 Florence caching requires a stable dataset `index` in every batch.")
            inputs["cache_keys"] = batch["index"]
        targets = self._prepare_action_targets(batch)
        action_is_pad = _prepare_padding_mask(batch, targets)
        targets = targets.masked_fill(action_is_pad.unsqueeze(-1), 0.0)
        value_targets = self._prepare_value_targets(batch, targets.shape[0], targets.device)
        losses = self.model(
            action=targets,
            action_is_pad=action_is_pad,
            value_targets=value_targets,
            **inputs,
        )
        total_loss = sum(losses.values())
        log = {name: value.detach().item() for name, value in losses.items()}
        log["loss"] = total_loss.detach().item()
        return total_loss, log

    def _get_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        inputs = self._build_siva2_inputs(batch, inference=True)
        return self.model.generate_actions(**inputs, steps=self.config.num_denoising_steps, noise=noise)

    @torch.no_grad()
    def predict_values(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        self.eval()
        values = self.model.predict_values(**self._build_siva2_inputs(batch, inference=True))
        # The head trains in normalized units for numerical stability; callers
        # should receive the same physical step/time unit used by the dataset.
        values["time_to_completion"] = values["time_to_completion"] * self.config.time_normalizer
        return values

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()
        self._update_history_queues(batch)
        return self._get_action_chunk(batch, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()
        self._update_history_queues(batch)
        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()
