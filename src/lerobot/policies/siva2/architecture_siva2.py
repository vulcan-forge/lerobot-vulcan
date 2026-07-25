# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Small, VLM-independent building blocks for SIVA2.

The code makes the research hypotheses explicit and unit-testable: typed slot
banks, attention-based temporal memory, prompt dropout, factored motion priors,
and a progress/value model all live outside the Florence wrapper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def sinusoidal_embedding(value: Tensor, dim: int, max_period: float = 10_000.0) -> Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=value.device, dtype=value.dtype) / max(half - 1, 1)
    )
    angles = value[:, None] * frequencies[None]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    return F.pad(embedding, (0, dim - embedding.shape[-1]))


class ResidualMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.net(self.norm(value))


class SlotBank(nn.Module):
    """Read a variable token sequence once into a fixed, semantically typed bank."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_slots: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float,
        token_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_dropout = token_dropout
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.queries = nn.Parameter(torch.randn(1, num_slots, hidden_size) * 0.02)
        # A permanently valid null token prevents all-masked attention for an
        # optional modality while retaining a learned representation of absence.
        self.null_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.mlp = ResidualMLP(hidden_size, mlp_ratio)
        self.blocks = nn.ModuleList(
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

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        tokens = self.input_norm(self.input_projection(tokens))
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            mask = mask.to(device=tokens.device, dtype=torch.bool)
        if self.training and self.token_dropout > 0.0 and tokens.shape[1] > 0:
            mask = mask & (torch.rand(mask.shape, device=mask.device) >= self.token_dropout)
        tokens = torch.cat((tokens, self.null_token.expand(tokens.shape[0], -1, -1)), dim=1)
        mask = F.pad(mask, (0, 1), value=True)
        slots = self.queries.expand(tokens.shape[0], -1, -1)
        attended, _ = self.cross_attention(
            self.cross_norm(slots), tokens, tokens, key_padding_mask=~mask, need_weights=False
        )
        slots = self.mlp(slots + attended)
        for block in self.blocks:
            slots = block(slots)
        return self.output_norm(slots)


class TemporalMemoryEncoder(nn.Module):
    """Compress arbitrary camera history into a fixed number of memory slots.

    Spatial queries retain more than frame averaging, then temporal attention
    lets every retained region reason about motion and ordering.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_slots: int,
        spatial_queries: int,
        max_frames: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.max_frames = max_frames
        self.spatial_queries = nn.Parameter(torch.randn(1, spatial_queries, hidden_size) * 0.02)
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.null_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.spatial_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.temporal_position = nn.Parameter(torch.randn(1, max_frames, 1, hidden_size) * 0.02)
        self.empty_memory_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.temporal_blocks = nn.ModuleList(
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
        self.output_bank = SlotBank(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_slots=num_slots,
            num_heads=num_heads,
            depth=1,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, frame_tokens: Tensor, frame_mask: Tensor) -> Tensor:
        """Encode ``[B,T,L,D]`` features, with ``frame_mask`` shaped ``[B,T,L]``."""
        if frame_tokens.ndim != 4 or frame_mask.shape != frame_tokens.shape[:3]:
            raise ValueError("Temporal memory expects tokens [B,T,L,D] and mask [B,T,L].")
        batch, frames, tokens_per_frame, _ = frame_tokens.shape
        if frames > self.max_frames:
            frame_tokens = frame_tokens[:, -self.max_frames :]
            frame_mask = frame_mask[:, -self.max_frames :]
            frames = self.max_frames

        projected = self.input_projection(frame_tokens).flatten(0, 1)
        mask = frame_mask.flatten(0, 1).bool()
        null = self.null_token.expand(batch * frames, -1, -1)
        projected = torch.cat((projected, null), dim=1)
        mask = F.pad(mask, (0, 1), value=True)
        queries = self.spatial_queries.expand(batch * frames, -1, -1)
        spatial, _ = self.spatial_attention(
            queries, projected, projected, key_padding_mask=~mask, need_weights=False
        )
        spatial = spatial.view(batch, frames, -1, spatial.shape[-1])
        spatial = spatial + self.temporal_position[:, :frames]
        temporal = spatial.flatten(1, 2)
        valid_frame = frame_mask.any(dim=-1)
        temporal_mask = valid_frame.unsqueeze(-1).expand(-1, -1, spatial.shape[2]).flatten(1, 2)
        # Live inference begins before a past frame exists.  The sentinel keeps
        # temporal attention finite while representing "no history yet".
        temporal = torch.cat((temporal, self.empty_memory_token.expand(batch, -1, -1)), dim=1)
        temporal_mask = F.pad(temporal_mask, (0, 1), value=True)
        for block in self.temporal_blocks:
            temporal = block(temporal, src_key_padding_mask=~temporal_mask)
        return self.output_bank(temporal, temporal_mask)


@dataclass
class BehaviorConditions:
    quality: Tensor
    speed: Tensor
    mistake: Tensor
    advantage: Tensor
    progress: Tensor
    control_mode: Tensor
    data_source: Tensor
    present: Tensor


class BehaviorConditionEncoder(nn.Module):
    """Turn performance/control metadata into maskable typed tokens."""

    NUM_CONDITIONS = 7

    def __init__(
        self,
        hidden_size: int,
        num_control_modes: int,
        num_data_sources: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.scalar_mlps = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(1, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
                for _ in range(5)
            ]
        )
        self.control_embedding = nn.Embedding(num_control_modes, hidden_size)
        self.source_embedding = nn.Embedding(num_data_sources, hidden_size)
        self.type_embedding = nn.Parameter(torch.randn(1, self.NUM_CONDITIONS, hidden_size) * 0.02)
        self.missing_embedding = nn.Parameter(torch.randn(1, self.NUM_CONDITIONS, hidden_size) * 0.02)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, conditions: BehaviorConditions) -> Tensor:
        scalars = (
            conditions.quality,
            conditions.speed,
            conditions.mistake,
            conditions.advantage,
            conditions.progress,
        )
        tokens = [
            encoder(value.unsqueeze(-1)) for encoder, value in zip(self.scalar_mlps, scalars, strict=True)
        ]
        tokens.extend(
            (
                self.control_embedding(conditions.control_mode),
                self.source_embedding(conditions.data_source),
            )
        )
        encoded = torch.stack(tokens, dim=1) + self.type_embedding
        present = conditions.present.bool()
        if self.training and self.dropout > 0.0:
            present = present & (torch.rand(present.shape, device=present.device) >= self.dropout)
        encoded = torch.where(present.unsqueeze(-1), encoded, self.missing_embedding.expand_as(encoded))
        return self.norm(encoded)


class TypedContextEncoder(nn.Module):
    """Fuse typed slot banks without erasing their modality identity."""

    NUM_TYPES = 5  # scene, memory, goal, subgoal, metadata

    def __init__(self, hidden_size: int, num_heads: int, depth: int, mlp_ratio: float) -> None:
        super().__init__()
        self.type_embedding = nn.Parameter(torch.randn(self.NUM_TYPES, hidden_size) * 0.02)
        self.blocks = nn.ModuleList(
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
        scene: Tensor,
        memory: Tensor,
        goal: Tensor,
        subgoal: Tensor,
        metadata: Tensor,
    ) -> Tensor:
        banks = (scene, memory, goal, subgoal, metadata)
        typed = [bank + self.type_embedding[index] for index, bank in enumerate(banks)]
        slots = torch.cat(typed, dim=1)
        for block in self.blocks:
            slots = block(slots)
        return self.output_norm(slots)


class FactoredMotionPriorLibrary(nn.Module):
    """Coarse trajectory = geometric primitive + strategy + embodiment offset."""

    def __init__(
        self,
        num_primitives: int,
        num_strategies: int,
        num_control_points: int,
        action_dim: int,
        horizon: int,
        num_domains: int,
        adapter_rank: int,
        init_scale: float,
        noise_scale: float,
    ) -> None:
        super().__init__()
        self.num_primitives = num_primitives
        self.num_strategies = num_strategies
        self.horizon = horizon
        self.primitive_controls = nn.Parameter(
            torch.randn(num_primitives, num_control_points, action_dim) * init_scale
        )
        self.strategy_offsets = nn.Parameter(
            torch.randn(num_strategies, num_control_points, action_dim) * init_scale
        )
        self.domain_coefficients = nn.Embedding(num_domains, adapter_rank)
        self.domain_basis = nn.Parameter(
            torch.randn(adapter_rank, num_primitives, num_control_points, action_dim) * init_scale
        )
        nn.init.zeros_(self.domain_coefficients.weight)
        self.log_noise_scale = nn.Parameter(
            torch.full((num_primitives, num_strategies, action_dim), math.log(noise_scale))
        )

    def all_trajectories(self, domain_id: Tensor) -> Tensor:
        domain = torch.einsum("br,rpkd->bpkd", self.domain_coefficients(domain_id), self.domain_basis)
        controls = (
            self.primitive_controls[None, :, None] + self.strategy_offsets[None, None, :] + domain[:, :, None]
        )
        batch, primitives, strategies, control_points, action_dim = controls.shape
        flat = controls.reshape(batch * primitives * strategies, control_points, action_dim).transpose(1, 2)
        trajectories = F.interpolate(flat, size=self.horizon, mode="linear", align_corners=True)
        return trajectories.transpose(1, 2).reshape(batch, primitives, strategies, self.horizon, action_dim)

    def all_scales(self, batch_size: int) -> Tensor:
        scale = self.log_noise_scale.clamp(math.log(1e-3), math.log(2.0)).exp()
        return scale[None, :, :, None].expand(batch_size, -1, -1, self.horizon, -1)

    @staticmethod
    def select(values: Tensor, primitive: Tensor, strategy: Tensor) -> Tensor:
        batch = torch.arange(values.shape[0], device=values.device)
        return values[batch, primitive, strategy]


class FlowDecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.mlp = ResidualMLP(hidden_size, mlp_ratio)

    def forward(self, actions: Tensor, slots: Tensor) -> Tensor:
        value = self.self_norm(actions)
        attended, _ = self.self_attention(value, value, value, need_weights=False)
        actions = actions + attended
        attended, _ = self.cross_attention(self.cross_norm(actions), slots, slots, need_weights=False)
        return self.mlp(actions + attended)


class ResidualFlowDecoderV2(nn.Module):
    def __init__(
        self,
        action_dim: int,
        horizon: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        num_primitives: int,
        num_strategies: int,
        num_domains: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_projection = nn.Linear(action_dim, hidden_size)
        self.position_embedding = nn.Parameter(torch.randn(1, horizon, hidden_size) * 0.02)
        self.primitive_embedding = nn.Embedding(num_primitives, hidden_size)
        self.strategy_embedding = nn.Embedding(num_strategies, hidden_size)
        self.domain_embedding = nn.Embedding(num_domains, hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2), nn.SiLU(), nn.Linear(hidden_size * 2, hidden_size)
        )
        self.blocks = nn.ModuleList(
            [FlowDecoderBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, action_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        action: Tensor,
        time: Tensor,
        slots: Tensor,
        primitive: Tensor,
        strategy: Tensor,
        domain_id: Tensor,
    ) -> Tensor:
        conditioning = (
            self.time_mlp(sinusoidal_embedding(time, self.hidden_size))
            + self.primitive_embedding(primitive)
            + self.strategy_embedding(strategy)
            + self.domain_embedding(domain_id)
        )
        hidden = self.action_projection(action) + self.position_embedding + conditioning.unsqueeze(1)
        for block in self.blocks:
            hidden = block(hidden, slots)
        return self.output_projection(self.output_norm(hidden))


class ValueProgressHead(nn.Module):
    """Predict deployment outcomes used to estimate RECAP advantages."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden_size, 4)

    def forward(self, slots: Tensor) -> dict[str, Tensor]:
        values = self.output(self.trunk(slots.mean(dim=1)))
        return {
            "success_logit": values[:, 0],
            "progress": values[:, 1].sigmoid(),
            "time_to_completion": F.softplus(values[:, 2]),
            "intervention_logit": values[:, 3],
        }


def temporally_correlated_noise(reference: Tensor, correlation: float) -> Tensor:
    white = torch.randn_like(reference)
    if correlation == 0.0 or reference.shape[1] < 3:
        return white
    smooth = F.avg_pool1d(white.transpose(1, 2), kernel_size=5, stride=1, padding=2).transpose(1, 2)
    mixed = (1.0 - correlation) * white + correlation * smooth
    return mixed / mixed.square().mean(dim=1, keepdim=True).add(1e-6).sqrt()
