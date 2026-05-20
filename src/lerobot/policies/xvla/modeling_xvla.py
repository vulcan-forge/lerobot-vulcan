#!/usr/bin/env python

# ------------------------------------------------------------------------------
# Copyright 2025 The HuggingFace Inc. team and 2toINF (https://github.com/2toINF)
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
# ------------------------------------------------------------------------------

from __future__ import annotations

import builtins
import concurrent.futures
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

from ..pretrained import PreTrainedPolicy, T
from ..utils import populate_queues
from .action_hub import build_action_space
from .configuration_xvla import XVLAConfig
from .soft_transformer import SoftPromptedTransformer

# Florence2 config and modeling depend on transformers
if TYPE_CHECKING or _transformers_available:
    from .configuration_florence2 import Florence2Config
    from .modeling_florence2 import Florence2ForConditionalGeneration
else:
    Florence2Config = None
    Florence2ForConditionalGeneration = None


class XVLAModel(nn.Module):
    """
    XVLA backbone that stitches Florence-2 embeddings with the temporal/action transformer head.
    """

    def __init__(
        self,
        config: XVLAConfig,
        florence_config: Florence2Config,
        proprio_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.chunk_size: int = config.chunk_size
        self.use_proprio: bool = config.use_proprio

        # Build action space with auto-detection for "auto" mode
        if config.action_mode.lower() == "auto":
            # Auto-detect real action dim from config.action_feature
            real_dim = (
                config.action_feature.shape[-1]
                if config.action_feature is not None
                else config.max_action_dim
            )
            self.action_space = build_action_space(
                config.action_mode.lower(),
                real_dim=real_dim,
                max_dim=config.max_action_dim,
            )
        else:
            self.action_space = build_action_space(config.action_mode.lower())

        self.dim_action = self.action_space.dim_action
        self.dim_proprio = proprio_dim

        self.vlm = Florence2ForConditionalGeneration(florence_config)
        if hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
                del lm.model.decoder
            if hasattr(lm, "lm_head"):
                del lm.lm_head

        projection_dim = getattr(self.vlm.config, "projection_dim", None)
        if projection_dim is None:
            raise ValueError("Florence2 config must provide `projection_dim` for multimodal fusion.")

        self.transformer = SoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=self.dim_action,
            dim_propio=self.dim_proprio,
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
        )

        # Apply freezing based on config
        self._apply_freezing()

        # Apply dtype casting based on config
        self._apply_dtype()
        self._perf: dict[str, float] = {
            "xvla_model_generate_calls": 0.0,
            "xvla_model_forward_vlm_avg_ms": 0.0,
            "xvla_model_forward_vlm_max_ms": 0.0,
            "xvla_model_denoise_avg_ms": 0.0,
            "xvla_model_denoise_max_ms": 0.0,
            "xvla_model_postprocess_avg_ms": 0.0,
            "xvla_model_postprocess_max_ms": 0.0,
            "xvla_model_total_avg_ms": 0.0,
            "xvla_model_total_max_ms": 0.0,
            "xvla_model_last_forward_vlm_ms": 0.0,
            "xvla_model_last_denoise_ms": 0.0,
            "xvla_model_last_postprocess_ms": 0.0,
            "xvla_model_last_total_ms": 0.0,
            "xvla_model_last_step_avg_ms": 0.0,
            "xvla_model_last_step_max_ms": 0.0,
        }

    def _get_target_dtype(self) -> torch.dtype:
        """Get the target dtype based on config."""
        if self.config.dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def _apply_dtype(self) -> None:
        """
        Apply dtype casting to model components based on config.
        """
        target_dtype = self._get_target_dtype()
        self.to(dtype=target_dtype)

    def _apply_freezing(self) -> None:
        """
        Freeze VLM vision and language encoders based on config options.
        Keep only policy transformer and soft prompts trainable.
        """
        # Freeze vision encoder
        if self.config.freeze_vision_encoder and hasattr(self.vlm, "vision_tower"):
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = False

        # Freeze language encoder
        if self.config.freeze_language_encoder and hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            # Freeze encoder
            if hasattr(lm, "model") and hasattr(lm.model, "encoder"):
                for param in lm.model.encoder.parameters():
                    param.requires_grad = False
            # Freeze shared embeddings
            if hasattr(lm, "model") and hasattr(lm.model, "shared"):
                for param in lm.model.shared.parameters():
                    param.requires_grad = False

        # Freeze or unfreeze policy transformer
        if not self.config.train_policy_transformer:
            for name, param in self.transformer.named_parameters():
                if "soft_prompts" not in name:
                    param.requires_grad = False

        # Freeze or unfreeze soft prompts
        if not self.config.train_soft_prompts and hasattr(self.transformer, "soft_prompt_hub"):
            for param in self.transformer.soft_prompt_hub.parameters():
                param.requires_grad = False

    def forward_vlm(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Encode text and multi-view images via Florence2 encoder.
        """
        batch_size, num_views = pixel_values.shape[:2]
        flat_mask = image_mask.view(-1).to(dtype=torch.bool)
        flat_images = pixel_values.flatten(0, 1)
        num_valid = int(flat_mask.sum().item())
        if num_valid == 0:
            raise ValueError("At least one image view must be valid per batch.")

        valid_images = flat_images[flat_mask]
        valid_feats = self.vlm._encode_image(valid_images)
        tokens_per_view, hidden_dim = valid_feats.shape[1:]

        image_features = valid_feats.new_zeros((batch_size * num_views, tokens_per_view, hidden_dim))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(batch_size, num_views, tokens_per_view, hidden_dim)
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)
        merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0],
            inputs_embeds,
        )

        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]

        aux_visual_inputs = image_features[:, 1:].reshape(batch_size, -1, hidden_dim)
        return {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs}

    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        reduction: str = "mean",
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for the XVLA model.
        """
        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)
        action = action.to(dtype=target_dtype)

        enc = self.forward_vlm(input_ids, image_input, image_mask)

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
        return self.action_space.compute_loss(pred_action, action, reduction=reduction)

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        self.eval()
        total_start = time.perf_counter()

        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)

        vlm_start = time.perf_counter()
        enc = self.forward_vlm(input_ids, image_input, image_mask)
        forward_vlm_s = time.perf_counter() - vlm_start

        batch_size = input_ids.shape[0]
        action_dim = self.dim_action

        x1 = torch.randn(batch_size, self.chunk_size, action_dim, device=proprio.device, dtype=target_dtype)
        action = torch.zeros_like(x1)

        steps = max(1, int(steps))
        denoise_total_s = 0.0
        denoise_max_s = 0.0
        for i in range(steps, 0, -1):
            step_start = time.perf_counter()
            t = torch.full((batch_size,), i / steps, device=proprio.device, dtype=target_dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            action = self.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc,
            )
            step_s = time.perf_counter() - step_start
            denoise_total_s += step_s
            denoise_max_s = max(denoise_max_s, step_s)
        postprocess_start = time.perf_counter()
        action = self.action_space.postprocess(action)
        postprocess_s = time.perf_counter() - postprocess_start
        total_s = time.perf_counter() - total_start

        calls = self._perf["xvla_model_generate_calls"] + 1.0
        self._perf["xvla_model_generate_calls"] = calls

        def _update_avg(key: str, sample_ms: float) -> None:
            prev_calls = calls - 1.0
            prev_avg = self._perf[key]
            self._perf[key] = ((prev_avg * prev_calls) + sample_ms) / calls

        forward_vlm_ms = forward_vlm_s * 1000.0
        denoise_ms = denoise_total_s * 1000.0
        postprocess_ms = postprocess_s * 1000.0
        total_ms = total_s * 1000.0
        step_avg_ms = (denoise_total_s / steps * 1000.0) if steps > 0 else 0.0
        step_max_ms = denoise_max_s * 1000.0

        _update_avg("xvla_model_forward_vlm_avg_ms", forward_vlm_ms)
        _update_avg("xvla_model_denoise_avg_ms", denoise_ms)
        _update_avg("xvla_model_postprocess_avg_ms", postprocess_ms)
        _update_avg("xvla_model_total_avg_ms", total_ms)
        self._perf["xvla_model_forward_vlm_max_ms"] = max(
            self._perf["xvla_model_forward_vlm_max_ms"], forward_vlm_ms
        )
        self._perf["xvla_model_denoise_max_ms"] = max(self._perf["xvla_model_denoise_max_ms"], denoise_ms)
        self._perf["xvla_model_postprocess_max_ms"] = max(
            self._perf["xvla_model_postprocess_max_ms"], postprocess_ms
        )
        self._perf["xvla_model_total_max_ms"] = max(self._perf["xvla_model_total_max_ms"], total_ms)
        self._perf["xvla_model_last_forward_vlm_ms"] = forward_vlm_ms
        self._perf["xvla_model_last_denoise_ms"] = denoise_ms
        self._perf["xvla_model_last_postprocess_ms"] = postprocess_ms
        self._perf["xvla_model_last_total_ms"] = total_ms
        self._perf["xvla_model_last_step_avg_ms"] = step_avg_ms
        self._perf["xvla_model_last_step_max_ms"] = step_max_ms
        return action

    def get_perf_snapshot(self) -> dict[str, float]:
        return dict(self._perf)


class XVLAPolicy(PreTrainedPolicy):
    """LeRobot-compliant wrapper built around the XVLA model."""

    config_class = XVLAConfig
    name = "xvla"

    def __init__(self, config: XVLAConfig, **kwargs):
        require_package("transformers", extra="xvla")
        super().__init__(config)
        config.validate_features()
        florence_config = config.get_florence_config()
        proprio_dim = config.max_state_dim if config.use_proprio else 0
        self.model = XVLAModel(config=config, florence_config=florence_config, proprio_dim=proprio_dim)
        self._prefetch_enabled = bool(config.prefetch_refill_enabled)
        self._prefetch_low_watermark = int(config.prefetch_low_watermark)
        self._chunk_lock = threading.Lock()
        self._prefetch_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._prefetch_future: concurrent.futures.Future[Tensor] | None = None
        if self._prefetch_enabled:
            self._prefetch_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="xvla_prefetch",
            )
        self._perf: dict[str, float] = {}
        self.reset()

    def reset(self) -> None:
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        self._perf = {
            "xvla_select_calls": 0.0,
            "xvla_refill_calls": 0.0,
            "xvla_queue_hit_calls": 0.0,
            "xvla_select_avg_ms": 0.0,
            "xvla_select_max_ms": 0.0,
            "xvla_refill_avg_ms": 0.0,
            "xvla_refill_max_ms": 0.0,
            "xvla_last_select_ms": 0.0,
            "xvla_last_refill_ms": 0.0,
            "xvla_last_populate_ms": 0.0,
            "xvla_last_get_chunk_ms": 0.0,
            "xvla_last_queue_extend_ms": 0.0,
            "xvla_last_popleft_ms": 0.0,
            "xvla_last_queue_len_before": 0.0,
            "xvla_last_queue_len_after": 0.0,
            "xvla_last_did_refill": 0.0,
            "xvla_prefetch_enabled": 1.0 if self._prefetch_enabled else 0.0,
            "xvla_prefetch_low_watermark": float(self._prefetch_low_watermark),
            "xvla_prefetch_submit_calls": 0.0,
            "xvla_prefetch_ready_hits": 0.0,
            "xvla_prefetch_wait_fallback_calls": 0.0,
            "xvla_prefetch_inflight": 0.0,
            "xvla_last_prefetch_submit_ms": 0.0,
            "xvla_last_prefetch_wait_ms": 0.0,
            "xvla_last_used_prefetched_chunk": 0.0,
            "xvla_config_n_action_steps": float(self.config.n_action_steps),
            "xvla_config_chunk_size": float(self.config.chunk_size),
            "xvla_config_num_denoising_steps": float(self.config.num_denoising_steps),
        }
        self._prefetch_future = None

    def _clone_batch_for_prefetch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        cloned: dict[str, Tensor] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                cloned[key] = value.detach().clone()
            else:
                cloned[key] = value
        return cloned

    def get_optim_params(self) -> dict:
        """Return trainable named parameters for optimization.

        Returns a dict of name -> param for all trainable parameters.
        This enables the xvla-adamw optimizer to apply differential learning rates
        based on parameter names (e.g., 1/10 LR for VLM components).
        """
        return dict(filter(lambda kv: kv[1].requires_grad, self.named_parameters()))

    def _prepare_state(self, batch: dict[str, Tensor], batch_size: int, device: torch.device) -> Tensor:
        if not self.config.use_proprio or OBS_STATE not in batch:
            return torch.zeros(batch_size, 0, device=device)
        state = batch[OBS_STATE]
        if state.ndim > 2:
            state = state[:, -1, :]
        return pad_vector(state, self.model.dim_proprio)

    def _prepare_images(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        present_img_keys = [key for key in self.config.image_features if key in batch]
        if len(present_img_keys) == 0:
            raise ValueError(
                "All image features are missing from the batch. "
                f"Batch keys: {list(batch.keys())}, expected at least one of {list(self.config.image_features)}."
            )

        images = []
        masks = []
        for key in present_img_keys:
            img = batch[key][:, -1] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding)
            images.append(img)
            masks.append(torch.ones(img.size(0), dtype=torch.bool, device=img.device))

        stacked_imgs = torch.stack(images, dim=1)
        stacked_masks = torch.stack(masks, dim=1)

        total_views = self.config.num_image_views or stacked_imgs.size(1)
        total_views = max(total_views, stacked_imgs.size(1))
        num_pad = total_views - stacked_imgs.size(1)
        if num_pad > 0:
            pad_shape = (stacked_imgs.size(0), num_pad, *stacked_imgs.shape[2:])
            pad_imgs = stacked_imgs.new_zeros(pad_shape)
            pad_masks = stacked_masks.new_zeros((stacked_masks.size(0), num_pad))
            stacked_imgs = torch.cat([stacked_imgs, pad_imgs], dim=1)
            stacked_masks = torch.cat([stacked_masks, pad_masks], dim=1)

        return stacked_imgs, stacked_masks

    def _get_domain_id(self, batch: dict[str, Tensor], batch_size: int, device: torch.device) -> Tensor:
        candidate = None
        if self.config.domain_feature_key and self.config.domain_feature_key in batch:
            candidate = batch[self.config.domain_feature_key]
        elif "domain_id" in batch:
            candidate = batch["domain_id"]

        if candidate is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        if not isinstance(candidate, torch.Tensor):
            candidate = torch.as_tensor(candidate, device=device)
        else:
            candidate = candidate.to(device=device)

        if candidate.ndim == 0:
            candidate = candidate.expand(batch_size)
        if candidate.ndim > 1:
            candidate = candidate.view(candidate.shape[0], -1)[:, 0]
        if candidate.shape[0] != batch_size:
            candidate = candidate.expand(batch_size)
        return candidate.to(dtype=torch.long)

    def _prepare_action_targets(self, batch: dict[str, Tensor]) -> Tensor:
        if ACTION not in batch:
            raise ValueError("Batch is missing action targets required for training.")
        actions = batch[ACTION]
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        actions = pad_tensor_along_dim(actions, self.config.chunk_size, dim=1)
        if actions.shape[-1] != self.model.dim_action:
            actions = pad_vector(actions, self.model.dim_action)
        return actions

    def _build_model_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        input_ids = batch[OBS_LANGUAGE_TOKENS]
        batch_size = input_ids.shape[0]
        images, image_mask = self._prepare_images(batch)
        domain_id = self._get_domain_id(batch, batch_size, images.device)
        proprio = self._prepare_state(batch, batch_size, images.device)
        return {
            "input_ids": input_ids,
            "image_input": images,
            "image_mask": image_mask,
            "domain_id": domain_id,
            "proprio": proprio,
        }

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        inputs = self._build_model_inputs(batch)
        targets = self._prepare_action_targets(batch)
        losses = self.model(action=targets, reduction=reduction, **inputs)
        total_loss = sum(losses.values())

        if reduction == "none":
            log_dict = {k: v.detach().mean().item() for k, v in losses.items()}
            log_dict["loss"] = total_loss.detach().mean().item()
        else:
            log_dict = {k: v.detach().item() for k, v in losses.items()}
            log_dict["loss"] = total_loss.detach().item()
        return total_loss, log_dict

    def _get_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        inputs_start = time.perf_counter()
        inputs = self._build_model_inputs(batch)
        build_inputs_ms = (time.perf_counter() - inputs_start) * 1000.0
        gen_start = time.perf_counter()
        actions = self.model.generate_actions(**inputs, steps=self.config.num_denoising_steps)
        generate_ms = (time.perf_counter() - gen_start) * 1000.0
        self._perf["xvla_last_build_model_inputs_ms"] = build_inputs_ms
        self._perf["xvla_last_model_generate_ms"] = generate_ms
        return actions

    def _get_action_chunk_threadsafe(self, batch: dict[str, Tensor]) -> Tensor:
        with self._chunk_lock:
            return self._get_action_chunk(batch)

    def _maybe_submit_prefetch(self, batch: dict[str, Tensor]) -> None:
        if not self._prefetch_enabled or self._prefetch_executor is None:
            return
        if self._prefetch_future is not None:
            self._perf["xvla_prefetch_inflight"] = 0.0 if self._prefetch_future.done() else 1.0
            return
        submit_start = time.perf_counter()
        prefetch_batch = self._clone_batch_for_prefetch(batch)
        self._prefetch_future = self._prefetch_executor.submit(
            self._get_action_chunk_threadsafe, prefetch_batch
        )
        self._perf["xvla_prefetch_submit_calls"] = self._perf["xvla_prefetch_submit_calls"] + 1.0
        self._perf["xvla_last_prefetch_submit_ms"] = (time.perf_counter() - submit_start) * 1000.0
        self._perf["xvla_prefetch_inflight"] = 1.0

    def _maybe_consume_ready_prefetch(self) -> bool:
        if self._prefetch_future is None:
            self._perf["xvla_prefetch_inflight"] = 0.0
            return False
        if not self._prefetch_future.done():
            self._perf["xvla_prefetch_inflight"] = 1.0
            return False
        actions = self._prefetch_future.result()
        self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        self._prefetch_future = None
        self._perf["xvla_prefetch_ready_hits"] = self._perf["xvla_prefetch_ready_hits"] + 1.0
        self._perf["xvla_prefetch_inflight"] = 0.0
        return True

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:  # noqa: ARG002
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        return self._get_action_chunk(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:  # noqa: ARG002
        self.eval()
        total_start = time.perf_counter()
        queue_len_before = len(self._queues[ACTION])
        populate_start = time.perf_counter()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        populate_ms = (time.perf_counter() - populate_start) * 1000.0
        did_refill = 0.0
        used_prefetched_chunk = 0.0

        if len(self._queues[ACTION]) <= self._prefetch_low_watermark:
            self._maybe_submit_prefetch(batch)

        if len(self._queues[ACTION]) == 0 and self._maybe_consume_ready_prefetch():
            used_prefetched_chunk = 1.0

        if len(self._queues[ACTION]) == 0:
            did_refill = 1.0
            refill_start = time.perf_counter()
            if self._prefetch_enabled and self._prefetch_future is not None:
                wait_start = time.perf_counter()
                actions = self._prefetch_future.result()
                wait_ms = (time.perf_counter() - wait_start) * 1000.0
                self._prefetch_future = None
                self._perf["xvla_prefetch_wait_fallback_calls"] = (
                    self._perf["xvla_prefetch_wait_fallback_calls"] + 1.0
                )
                self._perf["xvla_last_prefetch_wait_ms"] = wait_ms
                get_chunk_ms = (time.perf_counter() - refill_start) * 1000.0
            else:
                actions = self._get_action_chunk_threadsafe(batch)
                get_chunk_ms = (time.perf_counter() - refill_start) * 1000.0
            extend_start = time.perf_counter()
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
            extend_ms = (time.perf_counter() - extend_start) * 1000.0
            refill_ms = get_chunk_ms + extend_ms
        else:
            get_chunk_ms = 0.0
            extend_ms = 0.0
            refill_ms = 0.0

        popleft_start = time.perf_counter()
        action = self._queues[ACTION].popleft()
        popleft_ms = (time.perf_counter() - popleft_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        calls = self._perf["xvla_select_calls"] + 1.0
        self._perf["xvla_select_calls"] = calls
        prev_calls = calls - 1.0
        self._perf["xvla_select_avg_ms"] = ((self._perf["xvla_select_avg_ms"] * prev_calls) + total_ms) / calls
        self._perf["xvla_select_max_ms"] = max(self._perf["xvla_select_max_ms"], total_ms)
        if did_refill == 1.0:
            refill_calls = self._perf["xvla_refill_calls"] + 1.0
            prev_refill_calls = refill_calls - 1.0
            self._perf["xvla_refill_calls"] = refill_calls
            self._perf["xvla_refill_avg_ms"] = (
                (self._perf["xvla_refill_avg_ms"] * prev_refill_calls) + refill_ms
            ) / refill_calls
            self._perf["xvla_refill_max_ms"] = max(self._perf["xvla_refill_max_ms"], refill_ms)
        else:
            self._perf["xvla_queue_hit_calls"] = self._perf["xvla_queue_hit_calls"] + 1.0

        self._perf["xvla_last_select_ms"] = total_ms
        self._perf["xvla_last_refill_ms"] = refill_ms
        self._perf["xvla_last_populate_ms"] = populate_ms
        self._perf["xvla_last_get_chunk_ms"] = get_chunk_ms
        self._perf["xvla_last_queue_extend_ms"] = extend_ms
        self._perf["xvla_last_popleft_ms"] = popleft_ms
        self._perf["xvla_last_queue_len_before"] = float(queue_len_before)
        self._perf["xvla_last_queue_len_after"] = float(len(self._queues[ACTION]))
        self._perf["xvla_last_did_refill"] = did_refill
        self._perf["xvla_last_used_prefetched_chunk"] = used_prefetched_chunk
        return action

    def get_perf_snapshot(self) -> dict[str, float]:
        snapshot = dict(self._perf)
        snapshot.update(self.model.get_perf_snapshot())
        return snapshot

    def __del__(self):
        executor = getattr(self, "_prefetch_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ):
        """
        Loads XVLA model weights with:
        - automatic prefix 'model.' added to all keys
        - skip list for layers that should remain randomly initialized
        """
        import safetensors.torch

        # step 1: load config
        # TODO: jadechoghari, fix this
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        # step 2: locate model.safetensors
        if os.path.isdir(model_id):
            logging.info("Loading weights from local directory")
            model_file = os.path.join(model_id, "model.safetensors")
        else:
            try:
                from huggingface_hub import hf_hub_download
                from huggingface_hub.utils import HfHubHTTPError

                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename="model.safetensors",
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(f"model.safetensors not found on the Hub at {model_id}") from e

        logging.info(f"Loading checkpoint from {model_file}")
        # step 3: load state dict
        state_dict = safetensors.torch.load_file(model_file)
        encoder_key = "model.vlm.language_model.model.encoder.embed_tokens.weight"
        shared_key = "model.vlm.language_model.model.shared.weight"
        if encoder_key in state_dict:
            state_dict[shared_key] = state_dict[encoder_key]
            # or deepcopy
        # step 4: load into instance
        instance.load_state_dict(state_dict, strict=True)
        logging.info("Loaded XVLA checkpoint")
        # step 5: finalize
        # Reapply dtype after loading state dict
        instance.model._apply_dtype()
        instance.to(config.device)
        instance.eval()
        return instance


def resize_with_pad(img: torch.Tensor, height: int, width: int, pad_value: float = 0.0) -> torch.Tensor:
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but got {img.shape}")

    current_height, current_width = img.shape[2:]
    if current_height == height and current_width == width:
        return img

    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, height - resized_height)
    pad_width = max(0, width - resized_width)
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    if vector.shape[-1] == new_dim:
        return vector
    if new_dim == 0:
        shape = list(vector.shape)
        shape[-1] = 0
        return vector.new_zeros(*shape)
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = vector.new_zeros(*shape)
    length = min(current_dim, new_dim)
    new_vector[..., :length] = vector[..., :length]
    return new_vector


def pad_tensor_along_dim(tensor: Tensor, target_len: int, dim: int = 1) -> Tensor:
    current_len = tensor.size(dim)
    if current_len == target_len:
        return tensor
    if current_len > target_len:
        slices = [slice(None)] * tensor.dim()
        slices[dim] = slice(0, target_len)
        return tensor[tuple(slices)]
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_len - current_len
    pad_tensor = tensor.new_zeros(pad_shape)
    return torch.cat([tensor, pad_tensor], dim=dim)
