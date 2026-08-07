# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""The small, independently testable networks used by SIVA.

The comments in this module document architectural rationale and tensor
contracts.  They are intentionally kept separate from the Florence wrapper so
we can iterate on the action model without downloading a VLM for every unit
test.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def sinusoidal_embedding(value: Tensor, dim: int, max_period: float = 10_000.0) -> Tensor:
    """Embed a continuous scalar without learning a discretized time table."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=value.device, dtype=value.dtype) / max(half - 1, 1)
    )
    angles = value[:, None] * frequencies[None]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResidualMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float, dropout: float = 0.0) -> None:
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.norm = nn.LayerNorm(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(self.norm(x))


class SlotCompressor(nn.Module):
    """Compress an arbitrarily long multimodal sequence into fixed policy slots.

    Full self-attention over Florence tokens makes the action stack pay for the
    VLM sequence at every denoising step.  Learned queries instead perform one
    cross-attention read, after which all policy computation is bounded by the
    slot count.  Proprioception and embodiment enter as tokens/queries rather
    than being repeated at every visual position.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proprio_size: int,
        num_slots: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float,
        num_domains: int,
        token_dropout: float,
    ) -> None:
        super().__init__()
        self.token_dropout = token_dropout
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.slot_queries = nn.Parameter(torch.randn(1, num_slots, hidden_size) * 0.02)
        self.domain_embedding = nn.Embedding(num_domains, hidden_size)
        self.proprio_projection = nn.Linear(proprio_size, hidden_size) if proprio_size > 0 else None
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.cross_mlp = ResidualMLP(hidden_size, mlp_ratio)
        self.slot_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=int(hidden_size * mlp_ratio),
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        vlm_features: Tensor,
        auxiliary_visual_features: Tensor,
        proprio: Tensor,
        domain_id: Tensor,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        tokens = torch.cat((vlm_features, auxiliary_visual_features), dim=1)
        tokens = self.input_norm(self.input_projection(tokens))
        if context_mask is None:
            context_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            context_mask = context_mask.to(device=tokens.device, dtype=torch.bool)

        if self.training and self.token_dropout > 0.0:
            # Drop complete tokens, not individual channels.  This pressures the
            # slots to combine cameras and language instead of memorizing one
            # brittle token position.
            keep = torch.rand(tokens.shape[:2], device=tokens.device) >= self.token_dropout
            keep[:, 0] = True  # Never allow an entirely empty context.
            context_mask = context_mask & keep

        if self.proprio_projection is not None:
            tokens = torch.cat((tokens, self.proprio_projection(proprio).unsqueeze(1)), dim=1)
            context_mask = F.pad(context_mask, (0, 1), value=True)

        domain = self.domain_embedding(domain_id)
        slots = self.slot_queries.expand(tokens.shape[0], -1, -1) + domain.unsqueeze(1)
        attended, _ = self.cross_attention(
            self.cross_norm(slots),
            tokens,
            tokens,
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        slots = self.cross_mlp(slots + attended)
        for block in self.slot_blocks:
            slots = block(slots)
        return self.output_norm(slots)


class MotionPriorLibrary(nn.Module):
    """A shared library of coarse trajectories with low-rank domain adapters.

    Control points impose a low-frequency bias.  Linear interpolation is used on
    purpose for the first prototype: it is stable, differentiable, and makes the
    effect of the prior easy to inspect.  A cubic basis can be swapped in later
    without changing the rest of SIVA.
    """

    def __init__(
        self,
        num_modes: int,
        num_control_points: int,
        action_dim: int,
        horizon: int,
        num_domains: int,
        adapter_rank: int,
        init_scale: float,
        noise_scale: float,
    ) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.horizon = horizon
        self.control_points = nn.Parameter(
            torch.randn(num_modes, num_control_points, action_dim) * init_scale
        )
        # Shared priors carry reusable behavior.  A small low-rank offset absorbs
        # kinematic conventions without allocating an independent library per robot.
        self.domain_coefficients = nn.Embedding(num_domains, adapter_rank)
        self.domain_basis = nn.Parameter(
            torch.randn(adapter_rank, num_modes, num_control_points, action_dim) * init_scale
        )
        nn.init.zeros_(self.domain_coefficients.weight)
        self.log_noise_scale = nn.Parameter(torch.full((num_modes, action_dim), math.log(noise_scale)))

    def all_trajectories(self, domain_id: Tensor) -> Tensor:
        offsets = torch.einsum("br,rmkd->bmkd", self.domain_coefficients(domain_id), self.domain_basis)
        controls = self.control_points.unsqueeze(0) + offsets
        batch_size, num_modes, num_controls, action_dim = controls.shape
        controls = controls.reshape(batch_size * num_modes, num_controls, action_dim).transpose(1, 2)
        trajectories = F.interpolate(controls, size=self.horizon, mode="linear", align_corners=True)
        return trajectories.transpose(1, 2).reshape(batch_size, num_modes, self.horizon, action_dim)

    def all_scales(self, batch_size: int) -> Tensor:
        # Clamping prevents a mode from escaping its reconstruction objective by
        # making its source distribution arbitrarily broad or narrow.
        scale = self.log_noise_scale.clamp(math.log(1e-3), math.log(2.0)).exp()
        return scale.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, self.horizon, -1)

    @staticmethod
    def select(values: Tensor, mode: Tensor) -> Tensor:
        batch = torch.arange(values.shape[0], device=values.device)
        return values[batch, mode]


class FlowDecoderBlock(nn.Module):
    """Action self-attention followed by a read from fixed context slots."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.mlp = ResidualMLP(hidden_size, mlp_ratio)

    def forward(self, actions: Tensor, slots: Tensor) -> Tensor:
        normalized = self.self_norm(actions)
        attended, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        actions = actions + attended
        attended, _ = self.cross_attention(self.cross_norm(actions), slots, slots, need_weights=False)
        return self.mlp(actions + attended)


class ResidualFlowDecoder(nn.Module):
    """Predict a velocity field from a structured motion source to an action chunk."""

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        num_modes: int,
        num_domains: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_projection = nn.Linear(action_dim, hidden_size)
        self.position_embedding = nn.Parameter(torch.randn(1, horizon, hidden_size) * 0.02)
        self.mode_embedding = nn.Embedding(num_modes, hidden_size)
        self.domain_embedding = nn.Embedding(num_domains, hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.blocks = nn.ModuleList(
            [FlowDecoderBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, action_dim)
        # Zero initialization makes the initial model fall back to the interpretable
        # motion prior, then gradually learns residual dynamics.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        action: Tensor,
        time: Tensor,
        slots: Tensor,
        mode: Tensor,
        domain_id: Tensor,
    ) -> Tensor:
        conditioning = (
            self.time_mlp(sinusoidal_embedding(time, self.hidden_size))
            + self.mode_embedding(mode)
            + self.domain_embedding(domain_id)
        )
        hidden = self.action_projection(action) + self.position_embedding + conditioning.unsqueeze(1)
        for block in self.blocks:
            hidden = block(hidden, slots)
        return self.output_projection(self.output_norm(hidden))


def temporally_correlated_noise(reference: Tensor, correlation: float) -> Tensor:
    """Mix white and smoothed noise while preserving per-sample stochasticity."""
    white = torch.randn_like(reference)
    if correlation == 0.0 or reference.shape[1] < 3:
        return white
    smooth = F.avg_pool1d(white.transpose(1, 2), kernel_size=5, stride=1, padding=2).transpose(1, 2)
    mixed = (1.0 - correlation) * white + correlation * smooth
    return mixed / mixed.square().mean(dim=1, keepdim=True).add(1e-6).sqrt()
