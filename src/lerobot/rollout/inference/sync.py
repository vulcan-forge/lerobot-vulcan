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
from contextlib import nullcontext
from copy import copy
from datetime import datetime, timezone
from pathlib import Path
import time

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline

from .base import InferenceEngine

logger = logging.getLogger(__name__)


class _SyncInferencePerfReporter:
    """Overwrite-style perf log for sync inference rollouts."""

    def __init__(self, path: str | None = None, flush_interval_s: float = 5.0) -> None:
        self.path = Path(path or "~/lerobot_sync_inference_perf.txt").expanduser()
        self.flush_interval_s = max(float(flush_interval_s), 0.5)
        self.start_monotonic = time.monotonic()
        self.last_flush_ts = self.start_monotonic

        self.calls = 0
        self.none_obs_calls = 0
        self.prepare_total_s = 0.0
        self.prepare_max_s = 0.0
        self.preprocess_total_s = 0.0
        self.preprocess_max_s = 0.0
        self.policy_total_s = 0.0
        self.policy_max_s = 0.0
        self.postprocess_total_s = 0.0
        self.postprocess_max_s = 0.0
        self.cpu_total_s = 0.0
        self.cpu_max_s = 0.0
        self.reorder_total_s = 0.0
        self.reorder_max_s = 0.0
        self.total_total_s = 0.0
        self.total_max_s = 0.0
        self.last_prepare_s = 0.0
        self.last_preprocess_s = 0.0
        self.last_policy_s = 0.0
        self.last_postprocess_s = 0.0
        self.last_cpu_s = 0.0
        self.last_reorder_s = 0.0
        self.last_total_s = 0.0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self._render(status="starting"))

    def record(
        self,
        *,
        total_s: float,
        prepare_s: float,
        preprocess_s: float,
        policy_s: float,
        postprocess_s: float,
        cpu_s: float,
        reorder_s: float,
    ) -> None:
        self.calls += 1
        self.last_total_s = total_s
        self.last_prepare_s = prepare_s
        self.last_preprocess_s = preprocess_s
        self.last_policy_s = policy_s
        self.last_postprocess_s = postprocess_s
        self.last_cpu_s = cpu_s
        self.last_reorder_s = reorder_s
        self.total_total_s += total_s
        self.total_max_s = max(self.total_max_s, total_s)
        self.prepare_total_s += prepare_s
        self.prepare_max_s = max(self.prepare_max_s, prepare_s)
        self.preprocess_total_s += preprocess_s
        self.preprocess_max_s = max(self.preprocess_max_s, preprocess_s)
        self.policy_total_s += policy_s
        self.policy_max_s = max(self.policy_max_s, policy_s)
        self.postprocess_total_s += postprocess_s
        self.postprocess_max_s = max(self.postprocess_max_s, postprocess_s)
        self.cpu_total_s += cpu_s
        self.cpu_max_s = max(self.cpu_max_s, cpu_s)
        self.reorder_total_s += reorder_s
        self.reorder_max_s = max(self.reorder_max_s, reorder_s)
        self._maybe_flush()

    def record_none_obs(self) -> None:
        self.none_obs_calls += 1
        self._maybe_flush()

    def finalize(self, status: str = "stopped") -> None:
        self.path.write_text(self._render(status=status))

    def _maybe_flush(self) -> None:
        now = time.monotonic()
        if now - self.last_flush_ts < self.flush_interval_s:
            return
        self.last_flush_ts = now
        self.path.write_text(self._render(status="running"))

    def _render(self, *, status: str) -> str:
        uptime_s = time.monotonic() - self.start_monotonic
        effective_hz = self.calls / uptime_s if uptime_s > 0 else 0.0

        def avg_ms(total_s: float) -> float:
            return (total_s / self.calls * 1000.0) if self.calls else 0.0

        def max_ms(value_s: float) -> float:
            return value_s * 1000.0

        lines = [
            f"status: {status}",
            f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}",
            f"uptime_s: {uptime_s:.3f}",
            f"calls: {self.calls}",
            f"none_obs_calls: {self.none_obs_calls}",
            f"effective_hz: {effective_hz:.2f}",
            f"total_avg_ms: {avg_ms(self.total_total_s):.3f}",
            f"total_max_ms: {max_ms(self.total_max_s):.3f}",
            f"prepare_avg_ms: {avg_ms(self.prepare_total_s):.3f}",
            f"prepare_max_ms: {max_ms(self.prepare_max_s):.3f}",
            f"preprocess_avg_ms: {avg_ms(self.preprocess_total_s):.3f}",
            f"preprocess_max_ms: {max_ms(self.preprocess_max_s):.3f}",
            f"policy_avg_ms: {avg_ms(self.policy_total_s):.3f}",
            f"policy_max_ms: {max_ms(self.policy_max_s):.3f}",
            f"postprocess_avg_ms: {avg_ms(self.postprocess_total_s):.3f}",
            f"postprocess_max_ms: {max_ms(self.postprocess_max_s):.3f}",
            f"cpu_avg_ms: {avg_ms(self.cpu_total_s):.3f}",
            f"cpu_max_ms: {max_ms(self.cpu_max_s):.3f}",
            f"reorder_avg_ms: {avg_ms(self.reorder_total_s):.3f}",
            f"reorder_max_ms: {max_ms(self.reorder_max_s):.3f}",
        ]
        return "\n".join(lines) + "\n"

    def snapshot(self) -> dict[str, float]:
        def avg_ms(total_s: float) -> float:
            return (total_s / self.calls * 1000.0) if self.calls else 0.0

        return {
            "sync_calls": float(self.calls),
            "sync_none_obs_calls": float(self.none_obs_calls),
            "sync_total_avg_ms": avg_ms(self.total_total_s),
            "sync_total_max_ms": self.total_max_s * 1000.0,
            "sync_last_total_ms": self.last_total_s * 1000.0,
            "sync_prepare_avg_ms": avg_ms(self.prepare_total_s),
            "sync_last_prepare_ms": self.last_prepare_s * 1000.0,
            "sync_preprocess_avg_ms": avg_ms(self.preprocess_total_s),
            "sync_last_preprocess_ms": self.last_preprocess_s * 1000.0,
            "sync_policy_avg_ms": avg_ms(self.policy_total_s),
            "sync_policy_max_ms": self.policy_max_s * 1000.0,
            "sync_last_policy_ms": self.last_policy_s * 1000.0,
            "sync_postprocess_avg_ms": avg_ms(self.postprocess_total_s),
            "sync_last_postprocess_ms": self.last_postprocess_s * 1000.0,
            "sync_cpu_avg_ms": avg_ms(self.cpu_total_s),
            "sync_last_cpu_ms": self.last_cpu_s * 1000.0,
            "sync_reorder_avg_ms": avg_ms(self.reorder_total_s),
            "sync_last_reorder_ms": self.last_reorder_s * 1000.0,
        }


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
        device: str | None,
        robot_type: str,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        self._perf = _SyncInferencePerfReporter()
        logger.info(
            "SyncInferenceEngine initialized (device=%s, action_keys=%d)",
            self._device,
            len(ordered_action_keys),
        )

    def start(self) -> None:
        """No background resources to start."""
        logger.info("SyncInferenceEngine started (inline mode — no background thread)")

    def stop(self) -> None:
        """No background resources to stop."""
        self._perf.finalize(status="stopped")
        logger.info("SyncInferenceEngine stopped")

    def reset(self) -> None:
        """Reset the policy and pre/post-processors."""
        logger.info("Resetting sync inference state (policy + processors)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()

    def get_perf_snapshot(self) -> dict:
        snapshot = self._perf.snapshot()
        policy_snapshot = getattr(self._policy, "get_perf_snapshot", lambda: {})()
        snapshot.update(policy_snapshot)
        return snapshot

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Run the full inference pipeline on ``obs_frame`` and return an action tensor."""
        if obs_frame is None:
            self._perf.record_none_obs()
            return None
        # Shallow copy is intentional: the caller (`send_next_action`) builds
        # ``obs_frame`` fresh per tick via ``build_dataset_frame``, so the
        # tensor/array values are not shared with any other reader.
        total_start = time.perf_counter()
        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            prepare_start = time.perf_counter()
            observation = prepare_observation_for_inference(
                observation, self._device, self._task, self._robot_type
            )
            prepare_s = time.perf_counter() - prepare_start
            preprocess_start = time.perf_counter()
            observation = self._preprocessor(observation)
            preprocess_s = time.perf_counter() - preprocess_start
            policy_start = time.perf_counter()
            action = self._policy.select_action(observation)
            policy_s = time.perf_counter() - policy_start
            postprocess_start = time.perf_counter()
            action = self._postprocessor(action)
            postprocess_s = time.perf_counter() - postprocess_start
        cpu_start = time.perf_counter()
        action_tensor = action.squeeze(0).cpu()
        cpu_s = time.perf_counter() - cpu_start

        # Reorder to match dataset action ordering so the caller can treat
        # the returned tensor uniformly across backends.
        reorder_start = time.perf_counter()
        action_dict = make_robot_action(action_tensor, self._dataset_features)
        ordered = torch.tensor([action_dict[k] for k in self._ordered_action_keys])
        reorder_s = time.perf_counter() - reorder_start
        total_s = time.perf_counter() - total_start
        self._perf.record(
            total_s=total_s,
            prepare_s=prepare_s,
            preprocess_s=preprocess_s,
            policy_s=policy_s,
            postprocess_s=postprocess_s,
            cpu_s=cpu_s,
            reorder_s=reorder_s,
        )
        return ordered
