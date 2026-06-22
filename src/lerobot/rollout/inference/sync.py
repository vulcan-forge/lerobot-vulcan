# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Synchronous inference engine: inline policy call per control tick."""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from copy import copy

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline

from .base import InferenceEngine

logger = logging.getLogger(__name__)


# TODO(Steven): support relative-action policies.  The per-tick flow refreshes
# ``RelativeActionsProcessorStep._last_state`` every call, so cached chunk
# actions popped on later ticks get reanchored to the *current* robot state and
# absolute targets drift through the chunk.  Relative-action policies are
# rejected at context-build time today; RTC postprocesses the whole chunk and
# is unaffected.
#
# Candidate fix: drive the policy via ``predict_action_chunk`` and serve a
# local FIFO of postprocessed actions.  Eliminates drift by construction and
# saves per-tick pre/post work, but bypasses ``select_action`` — needs
# fallbacks for SAC (raises), ACT temporal ensembling (ensembler lives in
# ``select_action``), and Diffusion-family (obs-history queues populated as a
# side effect of ``select_action``).


class SyncInferenceEngine(InferenceEngine):
    """Inline synchronous inference: compute one action per call.

    ``get_action`` runs the full policy pipeline (pre/post-processor +
    ``select_action``) on the given observation frame and returns a
    CPU action tensor reordered to match the dataset action keys.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        fps: float,
        device: str | None,
        robot_type: str,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._fps = float(fps)
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        self._time_aware_chunking_enabled = (
            getattr(policy, "name", None) == "xvla"
            and hasattr(policy, "predict_action_chunk")
            and getattr(policy.config, "n_action_steps", 1) > 1
            and self._fps > 0
        )
        self._chunk_step_interval_s = 1.0 / self._fps if self._fps > 0 else 0.0
        self._cached_chunk_actions: list[torch.Tensor] = []
        self._cached_chunk_started_at: float | None = None
        logger.info(
            "SyncInferenceEngine initialized (device=%s, action_keys=%d)",
            self._device,
            len(ordered_action_keys),
        )
        if self._time_aware_chunking_enabled:
            logger.info("SyncInferenceEngine using time-aware chunk playback for %s", policy.name)

    def start(self) -> None:
        """No background resources to start."""
        logger.info("SyncInferenceEngine started (inline mode — no background thread)")

    def stop(self) -> None:
        """No background resources to stop."""
        logger.info("SyncInferenceEngine stopped")

    def reset(self) -> None:
        """Reset the policy and pre/post-processors."""
        logger.info("Resetting sync inference state (policy + processors)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        self._cached_chunk_actions = []
        self._cached_chunk_started_at = None

    @property
    def time_aware_chunking_enabled(self) -> bool:
        """Whether this engine plays chunked actions against wall-clock time."""
        return self._time_aware_chunking_enabled

    def _prepare_observation(self, obs_frame: dict) -> dict:
        observation = copy(obs_frame)
        observation = prepare_observation_for_inference(
            observation, self._device, self._task, self._robot_type
        )
        return self._preprocessor(observation)

    def _reorder_action_tensor(self, action: torch.Tensor) -> torch.Tensor:
        action_tensor = action.cpu()
        action_dict = make_robot_action(action_tensor, self._dataset_features)
        return torch.tensor([action_dict[k] for k in self._ordered_action_keys])

    def _populate_time_aware_chunk(self, obs_frame: dict) -> None:
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            observation = self._prepare_observation(obs_frame)
            action_chunk = self._policy.predict_action_chunk(observation)

            if action_chunk.ndim == 2:
                action_chunk = action_chunk.unsqueeze(1)
            if action_chunk.ndim != 3:
                raise ValueError(
                    f"Expected predict_action_chunk to return rank-3 tensor, got shape {tuple(action_chunk.shape)}"
                )

            n_action_steps = int(getattr(self._policy.config, "n_action_steps", action_chunk.shape[1]))
            action_chunk = action_chunk[:, :n_action_steps]

            self._cached_chunk_actions = [
                self._reorder_action_tensor(self._postprocessor(action_chunk[:, step_idx, :]))
                for step_idx in range(action_chunk.shape[1])
            ]
            self._cached_chunk_started_at = time.monotonic()

    def _get_time_aware_chunk_action(self, obs_frame: dict) -> torch.Tensor | None:
        if not self._cached_chunk_actions or self._cached_chunk_started_at is None:
            self._populate_time_aware_chunk(obs_frame)

        if not self._cached_chunk_actions or self._cached_chunk_started_at is None:
            return None

        elapsed_s = max(0.0, time.monotonic() - self._cached_chunk_started_at)
        desired_index = int(elapsed_s / self._chunk_step_interval_s) if self._chunk_step_interval_s > 0 else 0

        if desired_index >= len(self._cached_chunk_actions):
            self._populate_time_aware_chunk(obs_frame)
            desired_index = 0

        return self._cached_chunk_actions[desired_index]

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Run the full inference pipeline on ``obs_frame`` and return an action tensor."""
        if obs_frame is None:
            return None
        if self._time_aware_chunking_enabled:
            return self._get_time_aware_chunk_action(obs_frame)

        # Shallow copy is intentional: the caller (`send_next_action`) builds
        # ``obs_frame`` fresh per tick via ``build_dataset_frame``, so the
        # tensor/array values are not shared with any other reader.
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            observation = self._prepare_observation(obs_frame)
            action = self._policy.select_action(observation)
            action = self._postprocessor(action)
        return self._reorder_action_tensor(action)
