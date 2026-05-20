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

"""Rollout strategy ABC and shared action-dispatch helper."""

from __future__ import annotations

import abc
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import log_rerun_data

from ..inference import InferenceEngine

if TYPE_CHECKING:
    from ..configs import RolloutStrategyConfig
    from ..context import HardwareContext, ProcessorContext, RolloutContext, RuntimeContext

logger = logging.getLogger(__name__)


def _store_last_action_perf(runtime_ctx, snapshot: dict[str, float | str]) -> None:
    runtime_ctx.last_action_perf = snapshot


class _LoopPerfReporter:
    """Overwrite-style rollout loop perf report with slow-loop snapshots."""

    def __init__(self, path: str | None = None, flush_interval_s: float = 5.0) -> None:
        self.path = Path(path or "~/lerobot_rollout_loop_perf.txt").expanduser()
        self.flush_interval_s = max(float(flush_interval_s), 0.5)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start_monotonic = time.monotonic()
        self.last_flush_ts = self.start_monotonic
        self.calls = 0
        self.loop_total_s = 0.0
        self.loop_max_s = 0.0
        self.obs_total_s = 0.0
        self.obs_max_s = 0.0
        self.process_total_s = 0.0
        self.process_max_s = 0.0
        self.action_total_s = 0.0
        self.action_max_s = 0.0
        self.telemetry_total_s = 0.0
        self.telemetry_max_s = 0.0
        self.slow_events = 0
        self.last_slow_snapshot: dict[str, float | str] = {}
        self.path.write_text(self._render(status="starting"))

    def record(
        self,
        *,
        loop_s: float,
        obs_s: float,
        process_s: float,
        action_s: float,
        telemetry_s: float,
        slow: bool,
        slow_snapshot: dict[str, float | str] | None = None,
    ) -> None:
        self.calls += 1
        self.loop_total_s += loop_s
        self.loop_max_s = max(self.loop_max_s, loop_s)
        self.obs_total_s += obs_s
        self.obs_max_s = max(self.obs_max_s, obs_s)
        self.process_total_s += process_s
        self.process_max_s = max(self.process_max_s, process_s)
        self.action_total_s += action_s
        self.action_max_s = max(self.action_max_s, action_s)
        self.telemetry_total_s += telemetry_s
        self.telemetry_max_s = max(self.telemetry_max_s, telemetry_s)
        if slow:
            self.slow_events += 1
            self.last_slow_snapshot = slow_snapshot or {}
            self.path.write_text(self._render(status="slow_loop"))
            self.last_flush_ts = time.monotonic()
            return
        now = time.monotonic()
        if now - self.last_flush_ts >= self.flush_interval_s:
            self.last_flush_ts = now
            self.path.write_text(self._render(status="running"))

    def finalize(self, status: str = "stopped") -> None:
        self.path.write_text(self._render(status=status))

    def _render(self, *, status: str) -> str:
        uptime_s = time.monotonic() - self.start_monotonic

        def avg_ms(total_s: float) -> float:
            return (total_s / self.calls * 1000.0) if self.calls else 0.0

        lines = [
            f"status: {status}",
            f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}",
            f"uptime_s: {uptime_s:.3f}",
            f"calls: {self.calls}",
            f"effective_hz: {(self.calls / uptime_s) if uptime_s > 0 else 0.0:.2f}",
            f"slow_events: {self.slow_events}",
            f"loop_avg_ms: {avg_ms(self.loop_total_s):.3f}",
            f"loop_max_ms: {self.loop_max_s * 1000.0:.3f}",
            f"obs_avg_ms: {avg_ms(self.obs_total_s):.3f}",
            f"obs_max_ms: {self.obs_max_s * 1000.0:.3f}",
            f"process_avg_ms: {avg_ms(self.process_total_s):.3f}",
            f"process_max_ms: {self.process_max_s * 1000.0:.3f}",
            f"action_avg_ms: {avg_ms(self.action_total_s):.3f}",
            f"action_max_ms: {self.action_max_s * 1000.0:.3f}",
            f"telemetry_avg_ms: {avg_ms(self.telemetry_total_s):.3f}",
            f"telemetry_max_ms: {self.telemetry_max_s * 1000.0:.3f}",
        ]
        for key in sorted(self.last_slow_snapshot):
            value = self.last_slow_snapshot[key]
            if isinstance(value, float):
                lines.append(f"{key}: {value:.3f}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines) + "\n"


class RolloutStrategy(abc.ABC):
    """Abstract base for rollout execution strategies.

    Each concrete strategy implements a self-contained control loop with
    its own recording/interaction semantics.  Strategies are mutually
    exclusive — only one runs per session.
    """

    def __init__(self, config: RolloutStrategyConfig) -> None:
        self.config = config
        self._engine: InferenceEngine | None = None
        self._interpolator: ActionInterpolator | None = None
        self._warmup_flushed: bool = False
        self._cached_obs_processed: dict | None = None
        self._loop_perf = _LoopPerfReporter()

    def _init_engine(self, ctx: RolloutContext) -> None:
        """Attach the inference engine and action interpolator, then start the backend.

        Creates an :class:`ActionInterpolator` from the config's
        ``interpolation_multiplier`` and starts the inference engine.
        Call this from ``setup()`` so strategies share identical
        initialisation without duplicating code.
        """
        self._interpolator = ActionInterpolator(multiplier=ctx.runtime.cfg.interpolation_multiplier)
        self._engine = ctx.policy.inference
        logger.info("Starting inference engine...")
        self._engine.reset()
        self._engine.start()
        self._warmup_flushed = False
        self._cached_obs_processed = None
        logger.info("Inference engine started")

    def _process_observation_and_notify(self, processors: ProcessorContext, obs_raw: dict) -> dict:
        """Run the observation processor and notify the engine — throttled to policy ticks.

        Callers are responsible for calling ``robot.get_observation()`` every loop
        iteration so ``obs_raw`` stays fresh for the action post-processor.  This
        helper gates only the comparatively expensive bits — the processor pipeline
        and ``engine.notify_observation`` — to fire when the interpolator signals
        it needs a new action (once per ``interpolation_multiplier`` ticks).  On
        interpolated ticks the cached ``obs_processed`` is reused.

        With ``interpolation_multiplier == 1`` this is equivalent to the unthrottled
        path: ``needs_new_action()`` is True every tick.

        The cache is implicitly invalidated whenever ``interpolator.reset()`` is
        called (warmup completion, DAgger phase transitions back to AUTONOMOUS),
        because reset makes ``needs_new_action()`` return True on the next call.
        """
        if self._cached_obs_processed is None or self._interpolator.needs_new_action():
            obs_processed = processors.robot_observation_processor(obs_raw)
            self._engine.notify_observation(obs_processed)
            self._cached_obs_processed = obs_processed
        return self._cached_obs_processed

    def _handle_warmup(self, use_torch_compile: bool, loop_start: float, control_interval: float) -> bool:
        """Handle torch.compile warmup phase.

        Returns ``True`` if the caller should ``continue`` (still warming
        up).  On the first post-warmup iteration the engine and
        interpolator are reset so stale warmup state is discarded.
        """
        engine = self._engine
        interpolator = self._interpolator
        if not use_torch_compile:
            return False
        if not engine.ready:
            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            return True
        if not self._warmup_flushed:
            logger.info("Warmup complete — flushing stale state and resuming engine")
            engine.reset()
            interpolator.reset()
            self._warmup_flushed = True
            engine.resume()
        return False

    def _teardown_hardware(self, hw: HardwareContext, return_to_initial_position: bool = True) -> None:
        """Stop the inference engine, optionally return robot to initial position, and disconnect hardware."""
        if self._engine is not None:
            logger.info("Stopping inference engine...")
            self._engine.stop()
        self._loop_perf.finalize(status="stopped")
        robot = hw.robot_wrapper.inner
        if robot.is_connected:
            if return_to_initial_position and hw.initial_position:
                logger.info("Returning robot to initial position before shutdown...")
                self._return_to_initial_position(hw)
            elif not return_to_initial_position:
                logger.info(
                    "Skipping return-to-initial-position (disabled by config); leaving robot in final pose."
                )
            logger.info("Disconnecting robot...")
            robot.disconnect()
        teleop = hw.teleop
        if teleop is not None and teleop.is_connected:
            logger.info("Disconnecting teleoperator...")
            teleop.disconnect()

    @staticmethod
    def _return_to_initial_position(hw: HardwareContext, duration_s: float = 3.0, fps: int = 50) -> None:
        """Smoothly interpolate the robot back to its initial position."""
        robot = hw.robot_wrapper
        target = hw.initial_position
        try:
            current_obs = robot.get_observation()
            current_pos = {k: v for k, v in current_obs.items() if k in target}
            steps = max(int(duration_s * fps), 1)
            for step in range(1, steps + 1):
                t = step / steps
                interp = {}
                for k in current_pos:
                    interp[k] = current_pos[k] * (1 - t) + target[k] * t
                robot.send_action(interp)
                precise_sleep(1 / fps)
        except Exception as e:
            logger.warning("Could not return to initial position: %s", e)

    @staticmethod
    def _log_telemetry(
        obs_processed: dict | None,
        action_dict: dict | None,
        runtime_ctx: RuntimeContext,
    ) -> None:
        """Log observation/action telemetry to Rerun if display_data is enabled."""
        cfg = runtime_ctx.cfg
        if not cfg.display_data:
            return
        log_rerun_data(
            observation=obs_processed,
            action=action_dict,
            compress_images=cfg.display_compressed_images,
        )

    def _record_loop_perf(
        self,
        *,
        ctx: RolloutContext,
        loop_s: float,
        obs_s: float,
        process_s: float,
        action_s: float,
        telemetry_s: float,
        target_fps: float,
        slow: bool,
    ) -> None:
        snapshot: dict[str, float | str] | None = None
        if slow:
            snapshot = {
                "slow_loop_hz": (1.0 / loop_s) if loop_s > 0 else 0.0,
                "slow_loop_target_fps": target_fps,
                "slow_obs_ms": obs_s * 1000.0,
                "slow_process_ms": process_s * 1000.0,
                "slow_action_ms": action_s * 1000.0,
                "slow_telemetry_ms": telemetry_s * 1000.0,
            }
            action_snapshot = getattr(ctx.runtime, "last_action_perf", {})
            robot_snapshot = getattr(ctx.hardware.robot_wrapper.inner, "get_perf_snapshot", lambda: {})()
            engine_snapshot = ctx.policy.inference.get_perf_snapshot()
            snapshot.update(action_snapshot)
            snapshot.update(robot_snapshot)
            snapshot.update(engine_snapshot)
            logger.warning(
                "Slow loop breakdown: hz=%.1f target=%.1f obs=%.1fms process=%.1fms action=%.1fms telemetry=%.1fms",
                (1.0 / loop_s) if loop_s > 0 else 0.0,
                target_fps,
                obs_s * 1000.0,
                process_s * 1000.0,
                action_s * 1000.0,
                telemetry_s * 1000.0,
            )
        self._loop_perf.record(
            loop_s=loop_s,
            obs_s=obs_s,
            process_s=process_s,
            action_s=action_s,
            telemetry_s=telemetry_s,
            slow=slow,
            slow_snapshot=snapshot,
        )

    @abc.abstractmethod
    def setup(self, ctx: RolloutContext) -> None:
        """Strategy-specific initialisation (keyboard listeners, buffers, etc.)."""

    @abc.abstractmethod
    def run(self, ctx: RolloutContext) -> None:
        """Main rollout loop.  Returns when shutdown is requested or duration expires."""

    @abc.abstractmethod
    def teardown(self, ctx: RolloutContext) -> None:
        """Cleanup: save dataset, stop threads, disconnect hardware."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def safe_push_to_hub(dataset, tags=None, private=False) -> bool:
    """Push dataset to hub, skipping if no episodes have been saved.

    Returns ``True`` if the push was attempted, ``False`` if skipped.
    """
    if dataset.num_episodes == 0:
        logger.warning("No episodes saved — skipping push to hub")
        return False
    dataset.push_to_hub(tags=tags, private=private)
    return True


def estimate_max_episode_seconds(
    dataset_features: dict,
    fps: float,
    target_size_mb: float = DEFAULT_VIDEO_FILE_SIZE_IN_MB,
) -> float:
    """Conservatively estimate how many seconds of video will exceed *target_size_mb*.

    Each camera produces its own video file, so the episode duration is
    driven by the **slowest** camera to fill ``target_size_mb`` — i.e.
    the one with the fewest pixels per frame (lowest bitrate).

    Uses a deliberately **low** bits-per-pixel estimate so the computed
    duration is *longer* than reality.  By the time the timer fires the
    actual video file is guaranteed to have crossed the target size,
    which aligns episode boundaries with the dataset's video-file
    chunking — each ``push_to_hub`` uploads complete files rather than
    re-uploading a still-growing one.

    The estimate ignores codec-specific settings (CRF, preset) on purpose:
    we only need a rough lower bound on bitrate, not a precise prediction.

    Falls back to 300 s (5 min) when no video features are present.
    """
    # 0.1 bits-per-pixel is a *low* estimate for CRF-30 streaming video of
    # robot footage (real-world is typically 0.1 – 0.3 bpp).  Under-
    # estimating the bitrate over-estimates the time → the episode will be
    # *larger* than target_size_mb when we save, which is what we want.
    conservative_bpp = 0.1

    # Collect per-camera pixel counts — each camera has its own video file.
    camera_pixels = []
    for feat in dataset_features.values():
        if feat.get("dtype") == "video":
            shape = feat.get("shape", ())

            # (H, W, C) — bits-per-pixel is a per-spatial-pixel metric,
            # so we exclude the channel dimension from the count.
            if len(shape) == 3:
                pixels = shape[0] * shape[1]
                camera_pixels.append(pixels)
            else:
                raise ValueError(f"Unexpected video feature shape: {shape}")

    if not camera_pixels:
        return 300.0

    # Use the smallest camera: it produces the lowest bitrate and therefore
    # takes the longest to reach the target — the conservative choice.
    min_pixels = min(camera_pixels)
    bits_per_frame = min_pixels * conservative_bpp
    bytes_per_second = (bits_per_frame * fps) / 8

    # Guard against division by zero just in case
    if bytes_per_second <= 0:
        return 300.0

    return (target_size_mb * 1024 * 1024) / bytes_per_second


# ---------------------------------------------------------------------------
# Shared action-dispatch helper
# ---------------------------------------------------------------------------


def send_next_action(
    obs_processed: dict,
    obs_raw: dict,
    ctx: RolloutContext,
    interpolator: ActionInterpolator,
) -> dict | None:
    """Dispatch the next action to the robot.

    Pulls the next action tensor from the inference engine, feeds the
    interpolator, and sends the interpolated action through the
    ``robot_action_processor`` to the robot.  Works identically for
    sync and async backends — the rollout strategy never needs to branch.

    Returns the action dict that was sent, or ``None`` if no action was
    ready (e.g. empty async queue, interpolator not yet primed).
    """
    total_start = time.perf_counter()
    engine = ctx.policy.inference
    features = ctx.data.dataset_features
    ordered_keys = ctx.data.ordered_action_keys
    perf: dict[str, float | str] = {
        "action_needs_new_action": 1.0 if interpolator.needs_new_action() else 0.0,
        "action_none": 0.0,
    }

    if perf["action_needs_new_action"] == 1.0:
        build_start = time.perf_counter()
        obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
        perf["action_build_frame_ms"] = (time.perf_counter() - build_start) * 1000.0
        infer_start = time.perf_counter()
        action_tensor = engine.get_action(obs_frame)
        perf["action_engine_get_ms"] = (time.perf_counter() - infer_start) * 1000.0
        if action_tensor is not None:
            add_start = time.perf_counter()
            interpolator.add(action_tensor.cpu())
            perf["action_interpolator_add_ms"] = (time.perf_counter() - add_start) * 1000.0
        else:
            perf["action_interpolator_add_ms"] = 0.0
    else:
        perf["action_build_frame_ms"] = 0.0
        perf["action_engine_get_ms"] = 0.0
        perf["action_interpolator_add_ms"] = 0.0

    interp_start = time.perf_counter()
    interp = interpolator.get()
    perf["action_interpolator_get_ms"] = (time.perf_counter() - interp_start) * 1000.0
    if interp is None:
        perf["action_none"] = 1.0
        perf["action_total_ms"] = (time.perf_counter() - total_start) * 1000.0
        _store_last_action_perf(ctx.runtime, perf)
        return None

    if len(interp) != len(ordered_keys):
        raise ValueError(f"Interpolated tensor length ({len(interp)}) != action keys ({len(ordered_keys)})")
    dict_start = time.perf_counter()
    action_dict = {k: interp[i].item() for i, k in enumerate(ordered_keys)}
    perf["action_dict_ms"] = (time.perf_counter() - dict_start) * 1000.0
    process_start = time.perf_counter()
    processed = ctx.processors.robot_action_processor((action_dict, obs_raw))
    perf["action_robot_process_ms"] = (time.perf_counter() - process_start) * 1000.0
    send_start = time.perf_counter()
    ctx.hardware.robot_wrapper.send_action(processed)
    perf["action_robot_send_ms"] = (time.perf_counter() - send_start) * 1000.0
    perf["action_total_ms"] = (time.perf_counter() - total_start) * 1000.0
    _store_last_action_perf(ctx.runtime, perf)
    return action_dict
