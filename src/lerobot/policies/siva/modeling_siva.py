# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""SIVA: a structured-prior, slot-compressed VLA policy.

This is an experimental architecture, not a claim of benchmark superiority.
The implementation is deliberately compatible with XVLA's Florence-2 input
boundary so comparisons can hold data, preprocessing, and visual features fixed.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.policies.xvla.action_hub import BaseActionSpace, build_action_space
from lerobot.policies.xvla.configuration_florence2 import Florence2Config
from lerobot.policies.xvla.modeling_florence2 import Florence2ForConditionalGeneration
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.utils.constants import ACTION

from .configuration_siva import SIVAConfig
from .structured_flow import (
    MotionPriorLibrary,
    ResidualFlowDecoder,
    SlotCompressor,
    temporally_correlated_noise,
)


def _masked_mean(value: Tensor, valid: Tensor) -> Tensor:
    """Mean over [batch, time, ...], returning a differentiable zero if empty."""
    while valid.ndim < value.ndim:
        valid = valid.unsqueeze(-1)
    valid = valid.expand_as(value)
    return value.masked_select(valid).sum() / valid.sum().clamp_min(1)


class SIVAActionHead(nn.Module):
    """Observation router + structured motion library + residual flow decoder."""

    def __init__(self, config: SIVAConfig, action_dim: int, action_space: BaseActionSpace) -> None:
        super().__init__()
        self.config = config
        self.action_space = action_space
        self.router = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.hidden_size, config.num_motion_modes),
        )
        self.priors = MotionPriorLibrary(
            num_modes=config.num_motion_modes,
            num_control_points=config.num_control_points,
            action_dim=action_dim,
            horizon=config.chunk_size,
            num_domains=config.num_domains,
            adapter_rank=config.domain_adapter_rank,
            init_scale=config.prior_init_scale,
            noise_scale=config.prior_noise_scale,
        )
        self.flow = ResidualFlowDecoder(
            action_dim=action_dim,
            horizon=config.chunk_size,
            hidden_size=config.hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_modes=config.num_motion_modes,
            num_domains=config.num_domains,
        )

        flow_weights = torch.ones(action_dim)
        # Gripper state is trained primarily through the action-space endpoint
        # loss (usually BCE).  Keeping a small flow weight still gives the
        # trajectory model a signal without treating binary transitions exactly
        # like Cartesian motion.
        for index in action_space.gripper_idx:
            if index < action_dim:
                flow_weights[index] = 0.25
        self.register_buffer("flow_weights", flow_weights, persistent=False)

    def _assign_modes(self, actions: Tensor, trajectories: Tensor, valid: Tensor) -> Tensor:
        squared_error = (trajectories - actions.unsqueeze(1)).square()
        weights = self.flow_weights.view(1, 1, 1, -1)
        mask = valid[:, None, :, None]
        distance = (squared_error * weights * mask).sum(dim=(2, 3))
        denominator = (mask * weights).sum(dim=(2, 3)).clamp_min(1)
        return (distance / denominator).argmin(dim=1)

    def _endpoint_losses(self, predicted: Tensor, target: Tensor, valid: Tensor) -> dict[str, Tensor]:
        # Action-space losses know which channels are Cartesian, rotational, or
        # discrete.  Flattening valid time steps lets us reuse that knowledge
        # without allowing episode padding into the objective.
        predicted_valid = predicted[valid].unsqueeze(1)
        target_valid = target[valid].unsqueeze(1)
        if predicted_valid.shape[0] == 0:
            zero = predicted.sum() * 0.0
            return {"endpoint_loss": zero}
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
        # Nearest-prior assignment acts like online trajectory clustering.  It
        # creates a self-supervised "intent" label from every demonstration,
        # while the observation router learns to predict that label before acting.
        mode = self._assign_modes(actions, trajectories, valid).detach()
        selected_prior = self.priors.select(trajectories, mode)
        router_logits = self.router(slots.mean(dim=1)) / self.config.routing_temperature

        scales = self.priors.select(self.priors.all_scales(batch_size), mode)
        noise = temporally_correlated_noise(actions, self.config.noise_temporal_correlation)
        # Stop gradients through the prior on the flow path.  The explicit prior
        # reconstruction objective behaves like stable mini-batch k-means; the
        # flow cannot improve its own loss by moving the source underneath itself.
        source = selected_prior.detach() + scales * noise

        # Source is t=0 and the demonstrated action is t=1.
        time = (
            torch.rand(1, device=actions.device, dtype=actions.dtype)
            + torch.arange(batch_size, device=actions.device, dtype=actions.dtype) / batch_size
        ) % (1.0 - 1e-5)
        interpolated = source * (1.0 - time[:, None, None]) + actions * time[:, None, None]
        velocity = self.flow(interpolated, time, slots, mode, domain_id)
        target_velocity = actions - source

        weighted_flow_error = (velocity - target_velocity).square() * self.flow_weights
        losses: dict[str, Tensor] = {
            "flow_loss": _masked_mean(weighted_flow_error, valid) * self.config.flow_loss_weight,
            "prior_loss": _masked_mean((selected_prior - actions).square(), valid)
            * self.config.prior_loss_weight,
            "router_loss": F.cross_entropy(router_logits, mode) * self.config.router_loss_weight,
        }

        routing_probability = router_logits.softmax(dim=-1).mean(dim=0)
        # This is zero at uniform batch utilization and positive when modes die.
        losses["router_balance_loss"] = (
            self.config.num_motion_modes * routing_probability.square().sum() - 1.0
        ) * self.config.router_balance_loss_weight

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
        batch_size = slots.shape[0]
        mode = self.router(slots.mean(dim=1)).argmax(dim=-1)
        trajectory = self.priors.select(self.priors.all_trajectories(domain_id), mode)
        scale = self.priors.select(self.priors.all_scales(batch_size), mode)
        if noise is None:
            noise = (
                torch.zeros_like(trajectory)
                if self.config.deterministic_inference
                else torch.randn_like(trajectory)
            )
        if noise.shape != trajectory.shape:
            raise ValueError(
                f"Expected inference noise shape {tuple(trajectory.shape)}, got {tuple(noise.shape)}."
            )
        action = trajectory + scale * noise

        steps = max(1, int(steps))
        step_size = 1.0 / steps
        for index in range(steps):
            time = torch.full((batch_size,), index / steps, device=action.device, dtype=action.dtype)
            velocity = self.flow(action, time, slots, mode, domain_id)
            action = action + step_size * velocity
        return action


class SIVAModel(nn.Module):
    """Florence-2 representation boundary plus the SIVA action architecture."""

    def __init__(self, config: SIVAConfig, florence_config: Florence2Config, proprio_dim: int) -> None:
        super().__init__()
        self.config = config
        self.chunk_size = config.chunk_size
        self.use_proprio = config.use_proprio
        self.dim_proprio = proprio_dim

        if config.action_mode.lower() == "auto":
            real_dim = config.action_feature.shape[-1] if config.action_feature else config.max_action_dim
            self.action_space = build_action_space("auto", real_dim=real_dim, max_dim=config.max_action_dim)
        else:
            self.action_space = build_action_space(config.action_mode.lower())
        self.dim_action = self.action_space.dim_action

        # Keeping the attribute name `vlm` makes XVLA Florence weights directly
        # transferable by key.  Only the new context/action modules start fresh.
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
        self.context = SlotCompressor(
            input_size=projection_dim,
            hidden_size=config.hidden_size,
            proprio_size=proprio_dim,
            num_slots=config.num_context_slots,
            num_heads=config.num_heads,
            depth=config.slot_depth,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            token_dropout=config.vision_token_dropout,
        )
        self.action_head = SIVAActionHead(config, self.dim_action, self.action_space)

        if config.freeze_vlm:
            self.vlm.requires_grad_(False)
            self.vlm.eval()
        else:
            if config.freeze_vision_encoder and hasattr(self.vlm, "vision_tower"):
                self.vlm.vision_tower.requires_grad_(False)
            if config.freeze_language_encoder and hasattr(self.vlm, "language_model"):
                self.vlm.language_model.requires_grad_(False)
        self.to(dtype=self._get_target_dtype())

    def _get_target_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32

    def train(self, mode: bool = True) -> SIVAModel:
        super().train(mode)
        if self.config.freeze_vlm:
            self.vlm.eval()
        return self

    def forward_vlm(self, input_ids: Tensor, pixel_values: Tensor, image_mask: Tensor) -> dict[str, Tensor]:
        batch_size, num_views = pixel_values.shape[:2]
        flat_mask = image_mask.reshape(-1).bool()
        valid_images = pixel_values.flatten(0, 1)[flat_mask]
        if valid_images.shape[0] == 0:
            raise ValueError("At least one image view must be valid.")

        valid_features = self.vlm._encode_image(valid_images)
        tokens_per_view, hidden_dim = valid_features.shape[1:]
        image_features = valid_features.new_zeros(batch_size * num_views, tokens_per_view, hidden_dim)
        image_features[flat_mask] = valid_features
        image_features = image_features.view(batch_size, num_views, tokens_per_view, hidden_dim)

        text_embeddings = self.vlm.get_input_embeddings()(input_ids)
        merged, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0], text_embeddings
        )
        # Florence's helper marks every text position valid.  The tokenizer pads
        # XVLA prompts to a fixed length, so explicitly hide padding from both the
        # language encoder and the downstream slot read.
        pad_token_id = getattr(self.vlm.config, "pad_token_id", None)
        if pad_token_id is not None:
            attention_mask[:, -input_ids.shape[1] :] = input_ids.ne(pad_token_id).to(attention_mask.dtype)
        encoded = self.vlm.language_model.model.encoder(attention_mask=attention_mask, inputs_embeds=merged)[
            0
        ]
        auxiliary = image_features[:, 1:].reshape(batch_size, -1, hidden_dim)
        auxiliary_mask = (
            image_mask[:, 1:, None].expand(-1, -1, tokens_per_view).reshape(batch_size, -1).bool()
        )
        context_mask = torch.cat((attention_mask.bool(), auxiliary_mask), dim=1)
        return {
            "vlm_features": encoded,
            "auxiliary_visual_features": auxiliary,
            "context_mask": context_mask,
        }

    def _encode_context(
        self,
        input_ids: Tensor,
        image_input: Tensor,
        image_mask: Tensor,
        domain_id: Tensor,
        proprio: Tensor,
    ) -> Tensor:
        dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=dtype)
        proprio = proprio.to(dtype=dtype)
        features = self.forward_vlm(input_ids, image_input, image_mask)
        proprio_model, _ = self.action_space.preprocess(
            proprio, proprio.new_zeros((proprio.shape[0], self.chunk_size, self.dim_action))
        )
        return self.context(domain_id=domain_id, proprio=proprio_model, **features)

    def forward(
        self,
        input_ids: Tensor,
        image_input: Tensor,
        image_mask: Tensor,
        domain_id: Tensor,
        proprio: Tensor,
        action: Tensor,
        action_is_pad: Tensor | None = None,
    ) -> dict[str, Tensor]:
        slots = self._encode_context(input_ids, image_input, image_mask, domain_id, proprio)
        return self.action_head(
            slots, action.to(dtype=self._get_target_dtype()), domain_id, action_is_pad=action_is_pad
        )

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: Tensor,
        image_input: Tensor,
        image_mask: Tensor,
        domain_id: Tensor,
        proprio: Tensor,
        steps: int,
        noise: Tensor | None = None,
    ) -> Tensor:
        slots = self._encode_context(input_ids, image_input, image_mask, domain_id, proprio)
        action = self.action_head.generate(slots, domain_id, steps=steps, noise=noise)
        return self.action_space.postprocess(action)


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
        raise ValueError(
            f"action_is_pad {tuple(mask.shape)} does not match actions {tuple(targets.shape[:2])}."
        )
    return mask


class SIVAPolicy(XVLAPolicy):
    """LeRobot policy wrapper that intentionally reuses XVLA batch preparation."""

    config_class = SIVAConfig
    name = "siva"

    def __init__(self, config: SIVAConfig, **kwargs) -> None:
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.model = SIVAModel(
            config=config,
            florence_config=config.get_florence_config(),
            proprio_dim=config.max_state_dim if config.use_proprio else 0,
        )
        self.reset()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, strict: bool = False, **kwargs):
        """Use LeRobot's standard loader, including for VLM-only initialization checkpoints.

        XVLA overrides this loader with unconditional strict loading.  SIVA's
        conversion checkpoint intentionally omits every new head parameter, so
        the standard non-strict loader is required for the first training run.
        Fully trained SIVA checkpoints can opt into ``strict=True``.
        """
        return PreTrainedPolicy.from_pretrained.__func__(
            cls, pretrained_name_or_path, strict=strict, **kwargs
        )

    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        inputs = self._build_model_inputs(batch)
        targets = self._prepare_action_targets(batch)
        action_is_pad = _prepare_padding_mask(batch, targets)
        # Invalid boundary values must be sanitized before interpolation; masking
        # an infinity after squaring still produces NaN.
        targets = targets.masked_fill(action_is_pad.unsqueeze(-1), 0.0)
        losses = self.model(action=targets, action_is_pad=action_is_pad, **inputs)
        total_loss = sum(losses.values())
        log_dict = {name: value.detach().item() for name, value in losses.items()}
        log_dict["loss"] = total_loss.detach().item()
        return total_loss, log_dict

    def _get_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        inputs = self._build_model_inputs(batch)
        return self.model.generate_actions(**inputs, steps=self.config.num_denoising_steps, noise=noise)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        return self._get_action_chunk(batch, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()
