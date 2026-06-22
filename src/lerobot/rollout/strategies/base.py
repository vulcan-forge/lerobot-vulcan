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

"""Base rollout strategy: autonomous policy execution with no data recording."""

from __future__ import annotations

import logging
import time

from lerobot.utils.robot_utils import precise_sleep

from ..context import RolloutContext
from .core import RolloutStrategy, send_next_action

logger = logging.getLogger(__name__)


class BaseStrategy(RolloutStrategy):
    """Autonomous policy rollout with no data recording.

    All actions flow through the ``robot_action_processor`` pipeline
    before reaching the robot.
    """

    def setup(self, ctx: RolloutContext) -> None:
        """Initialise the inference engine."""
        self._init_engine(ctx)
        self._sync_action_plan_started_at: float | None = None
        self._sync_action_plan_horizon_s = self._get_sync_action_plan_horizon_s(ctx)
        self._last_stale_reset_reason: str | None = None
        self._last_stale_reset_log_time: float = 0.0
        self._suppressed_stale_reset_logs: int = 0
        if self._sync_action_plan_horizon_s is not None:
            logger.info(
                "Sync stale-action guard enabled (max plan age %.3fs)",
                self._sync_action_plan_horizon_s,
            )
        logger.info("Base strategy ready")

    def _get_sync_action_plan_horizon_s(self, ctx: RolloutContext) -> float | None:
        """Return the intended wall-clock lifetime of one sync action chunk."""
        if ctx.runtime.cfg.inference.type != "sync":
            return None

        n_action_steps = getattr(ctx.policy.policy.config, "n_action_steps", 1)
        if not isinstance(n_action_steps, int) or n_action_steps <= 1:
            return None

        fps = float(ctx.runtime.cfg.fps)
        if fps <= 0:
            return None

        return n_action_steps / fps

    def _reset_stale_rollout_state(self, reason: str) -> None:
        """Flush queued policy/interpolator state so the next tick replans fresh."""
        now = time.monotonic()
        should_log = (
            reason != self._last_stale_reset_reason or (now - self._last_stale_reset_log_time) >= 5.0
        )
        if should_log:
            if self._suppressed_stale_reset_logs > 0 and reason == self._last_stale_reset_reason:
                logger.warning(
                    "Flushing stale rollout state: %s (suppressed %d similar messages)",
                    reason,
                    self._suppressed_stale_reset_logs,
                )
            else:
                logger.warning("Flushing stale rollout state: %s", reason)
            self._last_stale_reset_reason = reason
            self._last_stale_reset_log_time = now
            self._suppressed_stale_reset_logs = 0
        else:
            self._suppressed_stale_reset_logs += 1
        self._engine.reset()
        self._engine.resume()
        self._interpolator.reset()
        self._cached_obs_processed = None
        self._sync_action_plan_started_at = None

    def run(self, ctx: RolloutContext) -> None:
        """Run the autonomous control loop until shutdown or duration expires."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)

        start_time = time.perf_counter()
        engine.resume()
        logger.info("Base strategy control loop started")

        while not ctx.runtime.shutdown_event.is_set():
            loop_start = time.perf_counter()

            if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            if (
                self._sync_action_plan_horizon_s is not None
                and self._sync_action_plan_started_at is not None
                and (loop_start - self._sync_action_plan_started_at) >= self._sync_action_plan_horizon_s
            ):
                self._reset_stale_rollout_state(
                    "sync action chunk exceeded its wall-clock horizon; forcing replanning from fresh observation"
                )

            try:
                obs = robot.get_observation()
            except TimeoutError as exc:
                self._reset_stale_rollout_state(str(exc))
                dt = time.perf_counter() - loop_start
                if (sleep_t := control_interval - dt) > 0:
                    precise_sleep(sleep_t)
                continue
            obs_processed = self._process_observation_and_notify(ctx.processors, obs)

            if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                continue

            action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
            if action_dict is not None and self._sync_action_plan_started_at is None:
                self._sync_action_plan_started_at = time.perf_counter()
            self._log_telemetry(obs_processed, action_dict, ctx.runtime)

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            else:
                logger.warning(
                    f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
                )
                if self._sync_action_plan_horizon_s is not None and dt >= self._sync_action_plan_horizon_s:
                    self._reset_stale_rollout_state(
                        f"control loop iteration took {dt:.3f}s, exceeding the sync action horizon of {self._sync_action_plan_horizon_s:.3f}s"
                    )

    def teardown(self, ctx: RolloutContext) -> None:
        """Disconnect hardware and stop inference."""
        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("Base strategy teardown complete")
