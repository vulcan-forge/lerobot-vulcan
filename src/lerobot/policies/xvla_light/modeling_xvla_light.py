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

from __future__ import annotations

import hashlib
import json
import logging
from collections import deque

import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.xvla.modeling_xvla import XVLAModel, XVLAPolicy
from lerobot.utils.constants import ACTION

from .configuration_xvla_light import XVLALightConfig
from .florence_cache import FlorenceFeatureCache


def _make_cache_signature(config: XVLALightConfig, vlm: torch.nn.Module) -> str:
    """Fingerprint cache-boundary settings and representative loaded Florence weights."""
    signature_config = {
        "florence_config": config.florence_config,
        "tokenizer_name": config.tokenizer_name,
        "tokenizer_max_length": config.tokenizer_max_length,
        "tokenizer_padding_side": config.tokenizer_padding_side,
        "pad_language_to": config.pad_language_to,
        "resize_imgs_with_padding": config.resize_imgs_with_padding,
        "num_image_views": config.num_image_views,
        "image_features": list(config.image_features),
        "dtype": config.dtype,
    }
    digest = hashlib.sha256(json.dumps(signature_config, sort_keys=True, default=str).encode())
    # Sampling three values from every tensor avoids copying the complete Florence
    # checkpoint back to CPU while still detecting a reused path from a different
    # fine-tuned checkpoint with overwhelming probability.
    for name, parameter in vlm.named_parameters():
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(str(parameter.dtype).encode())
        flattened = parameter.detach().reshape(-1)
        if flattened.numel() > 0:
            indices = torch.tensor(
                [0, flattened.numel() // 2, flattened.numel() - 1],
                device=flattened.device,
            ).unique()
            sample = flattened.index_select(0, indices).float().cpu().numpy()
            digest.update(sample.tobytes())
    return digest.hexdigest()


class XVLALightModel(XVLAModel):
    """XVLA model with an optional training-only Florence feature cache."""

    config: XVLALightConfig

    def __init__(self, config: XVLALightConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self._florence_cache: FlorenceFeatureCache | None = None
        self._cache_requests = 0
        self._cache_hits = 0

        if config.cache_florence_features:
            # XVLA's two legacy freeze flags do not cover image_projection,
            # image_proj_norm, image_pos_embed, or visual_temporal_embed. The final
            # feature cache boundary requires every Florence parameter to be fixed.
            for parameter in self.vlm.parameters():
                parameter.requires_grad = False
            self.vlm.eval()
            logging.info("XVLA-light Florence feature cache enabled at %s", config.florence_cache_path)

    def _get_florence_cache(self) -> FlorenceFeatureCache:
        # This must be lazy: from_pretrained constructs the model before loading
        # checkpoint weights, and the signature must describe the loaded weights.
        if self._florence_cache is None:
            self._florence_cache = FlorenceFeatureCache(
                self.config.florence_cache_path,
                signature=_make_cache_signature(self.config, self.vlm),
            )
        return self._florence_cache

    def train(self, mode: bool = True) -> XVLALightModel:
        super().train(mode)
        if self.config.cache_florence_features:
            # Frozen Florence must remain deterministic even while the policy
            # transformer is in training mode.
            self.vlm.eval()
        return self

    def forward_vlm(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
        cache_keys: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self.config.cache_florence_features or cache_keys is None or not self.training:
            return super().forward_vlm(input_ids, pixel_values, image_mask)
        cache = self._get_florence_cache()

        sample_indices = [int(index) for index in cache_keys.detach().cpu().reshape(-1).tolist()]
        if len(sample_indices) != input_ids.shape[0]:
            raise ValueError(
                f"Expected one Florence cache key per sample, got {len(sample_indices)} keys for "
                f"batch size {input_ids.shape[0]}."
            )

        cached = cache.get_many(sample_indices)
        missing_positions = [
            position for position, sample_index in enumerate(sample_indices) if sample_index not in cached
        ]
        computed: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        online: dict[int, dict[str, torch.Tensor]] = {}
        if missing_positions:
            missing = torch.tensor(missing_positions, device=input_ids.device, dtype=torch.long)
            with torch.no_grad():
                missing_features = super().forward_vlm(
                    input_ids.index_select(0, missing),
                    pixel_values.index_select(0, missing),
                    image_mask.index_select(0, missing),
                )
            for offset, position in enumerate(missing_positions):
                sample_index = sample_indices[position]
                sample_features = {
                    "vlm_features": missing_features["vlm_features"][offset],
                    "aux_visual_inputs": missing_features["aux_visual_inputs"][offset],
                }
                online[position] = sample_features
                computed[sample_index] = (
                    sample_features["vlm_features"],
                    sample_features["aux_visual_inputs"],
                )
            cache.put_many(computed)

        device = input_ids.device
        vlm_features = []
        aux_visual_inputs = []
        for position, sample_index in enumerate(sample_indices):
            features = online.get(position, cached.get(sample_index))
            if features is None:
                raise RuntimeError(f"Failed to resolve Florence features for dataset index {sample_index}.")
            vlm_features.append(features["vlm_features"].to(device=device, non_blocking=True))
            aux_visual_inputs.append(features["aux_visual_inputs"].to(device=device, non_blocking=True))

        self._cache_requests += len(sample_indices)
        self._cache_hits += len(sample_indices) - len(missing_positions)
        if self._cache_requests % 10_000 < len(sample_indices):
            logging.info(
                "Florence cache hit rate: %.1f%% (%d cached samples)",
                100 * self._cache_hits / self._cache_requests,
                cache.count(),
            )
        return {
            "vlm_features": torch.stack(vlm_features),
            "aux_visual_inputs": torch.stack(aux_visual_inputs),
        }

    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        cache_keys: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)
        action = action.to(dtype=target_dtype)
        enc = self.forward_vlm(input_ids, image_input, image_mask, cache_keys=cache_keys)

        batch_size = input_ids.shape[0]
        t = (
            torch.rand(1, device=input_ids.device, dtype=target_dtype)
            + torch.arange(batch_size, device=input_ids.device, dtype=target_dtype) / batch_size
        ) % (1 - 1e-5)
        action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)
        pred_action = self.transformer(
            domain_id=domain_id,
            action_with_noise=action_noisy_m,
            t=t,
            proprio=proprio_m,
            **enc,
        )
        return self.action_space.compute_loss(pred_action, action)


class XVLALightPolicy(XVLAPolicy):
    """XVLA policy wrapper using :class:`XVLALightConfig` defaults."""

    config_class = XVLALightConfig
    name = "xvla_light"

    def __init__(self, config: XVLALightConfig, **kwargs) -> None:
        # XVLAPolicy constructs XVLAModel directly, so build the cache-aware
        # subclass here while retaining all inherited input/action behavior.
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        florence_config = config.get_florence_config()
        proprio_dim = config.max_state_dim if config.use_proprio else 0
        self.model = XVLALightModel(
            config=config,
            florence_config=florence_config,
            proprio_dim=proprio_dim,
        )
        self._queues = {ACTION: deque(maxlen=config.n_action_steps)}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        inputs = self._build_model_inputs(batch)
        if self.config.cache_florence_features:
            if "index" not in batch:
                raise KeyError("Florence caching requires the dataset's stable `index` field in every batch.")
            inputs["cache_keys"] = batch["index"]
        targets = self._prepare_action_targets(batch)
        losses = self.model(action=targets, **inputs)
        total_loss = sum(losses.values())
        log_dict = {key: value.detach().item() for key, value in losses.items()}
        log_dict["loss"] = total_loss.detach().item()
        return total_loss, log_dict
