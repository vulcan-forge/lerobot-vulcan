from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(slots=True)
class FixedRateJointLimit:
    step_deg: float = 2.0
    deadband_deg: float = 0.0


@dataclass(slots=True)
class JointPostprocessConfig:
    lowpass_alpha: float = 0.25
    delta_scale: dict[int, float] = field(default_factory=lambda: {0: 0.5, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})
    wrist_roll_bias_enabled: bool = True
    wrist_roll_index: int = 4
    wrist_roll_target_rad: float = 0.0
    wrist_roll_blend: float = 0.05
    elbow_soft_stop_enabled: bool = True
    elbow_soft_stop_index: int = 2
    elbow_soft_stop_fraction_from_lower: float = 0.25
    elbow_soft_stop_below_small_step_deg: float = 3.0
    elbow_soft_stop_below_margin_deg: float = 8.0
    elbow_soft_stop_above_max_down_deg: float = 25.0
    elbow_back_block_enabled: bool = True
    elbow_back_block_index: int = 2
    elbow_back_block_direction: str = "increase"
    elbow_back_block_tolerance_deg: float = 2.0
    fixed_rate_enabled: bool = True
    fixed_rate_limits: dict[int, FixedRateJointLimit] = field(
        default_factory=lambda: {0: FixedRateJointLimit(step_deg=2.0, deadband_deg=0.2)}
    )
    bypass_all_mods: bool = False

    @classmethod
    def from_legacy_tune(cls, tune: Mapping[str, Any]) -> "JointPostprocessConfig":
        fixed_rate_raw = tune.get("fixed_rate", {})
        joints_raw = fixed_rate_raw.get("joints", {})
        fixed_rate_limits = {
            int(joint_idx): FixedRateJointLimit(
                step_deg=float(joint_cfg.get("step_deg", 2.0)),
                deadband_deg=float(joint_cfg.get("deadband_deg", 0.0)),
            )
            for joint_idx, joint_cfg in joints_raw.items()
        }

        wrist_roll_bias = tune.get("wrist_roll_overhand_bias", {})
        elbow_soft_stop = tune.get("elbow_soft_stop", {})
        elbow_back_block = tune.get("elbow_back_block", {})

        return cls(
            lowpass_alpha=float(tune.get("lowpass_alpha", 0.25)),
            delta_scale={int(idx): float(scale) for idx, scale in tune.get("delta_scale", {}).items()},
            wrist_roll_bias_enabled=bool(wrist_roll_bias.get("enabled", True)),
            wrist_roll_index=int(wrist_roll_bias.get("index", 4)),
            wrist_roll_target_rad=float(wrist_roll_bias.get("target", 0.0)),
            wrist_roll_blend=float(wrist_roll_bias.get("blend", 0.05)),
            elbow_soft_stop_enabled=bool(elbow_soft_stop.get("enabled", True)),
            elbow_soft_stop_index=int(elbow_soft_stop.get("index", 2)),
            elbow_soft_stop_fraction_from_lower=float(elbow_soft_stop.get("fraction_from_lower", 0.25)),
            elbow_soft_stop_below_small_step_deg=float(elbow_soft_stop.get("below_small_step_deg", 3.0)),
            elbow_soft_stop_below_margin_deg=float(elbow_soft_stop.get("below_margin_deg", 8.0)),
            elbow_soft_stop_above_max_down_deg=float(elbow_soft_stop.get("above_max_down_deg", 25.0)),
            elbow_back_block_enabled=bool(elbow_back_block.get("enabled", True)),
            elbow_back_block_index=int(elbow_back_block.get("index", 2)),
            elbow_back_block_direction=str(elbow_back_block.get("direction", "increase")),
            elbow_back_block_tolerance_deg=float(elbow_back_block.get("tolerance_deg", 2.0)),
            fixed_rate_enabled=bool(fixed_rate_raw.get("enabled", True)),
            fixed_rate_limits=fixed_rate_limits or {0: FixedRateJointLimit(step_deg=2.0, deadband_deg=0.2)},
            bypass_all_mods=bool(tune.get("bypass_all_mods", False)),
        )


class JointPostprocessor:
    def __init__(self, config: JointPostprocessConfig) -> None:
        self.config = config
        self._prev_q: np.ndarray | None = None
        self._elbow_back_limit: float | None = None

    def sync_from_legacy_tune(self, tune: Mapping[str, Any]) -> None:
        self.config = JointPostprocessConfig.from_legacy_tune(tune)

    def reset(self) -> None:
        self._prev_q = None
        self._elbow_back_limit = None

    def apply(self, solution_rad: np.ndarray, *, elbow_soft_stop: float | None) -> np.ndarray:
        solution = np.array(solution_rad, dtype=float, copy=True)

        if self.config.bypass_all_mods:
            self._prev_q = solution.copy()
            return solution

        if self._prev_q is None:
            self._prev_q = solution.copy()

        prev_q = self._prev_q.copy()
        solution = self.config.lowpass_alpha * solution + (1.0 - self.config.lowpass_alpha) * prev_q

        if self.config.elbow_soft_stop_enabled:
            solution = self._apply_elbow_soft_stop(solution, prev_q, elbow_soft_stop)

        if self.config.wrist_roll_bias_enabled:
            wrist_idx = self.config.wrist_roll_index
            if 0 <= wrist_idx < len(solution):
                blend = self.config.wrist_roll_blend
                solution[wrist_idx] = (1.0 - blend) * solution[wrist_idx] + blend * self.config.wrist_roll_target_rad

        for joint_idx, scale in self.config.delta_scale.items():
            if 0 <= joint_idx < len(solution) and scale != 1.0:
                delta = solution[joint_idx] - prev_q[joint_idx]
                solution[joint_idx] = prev_q[joint_idx] + scale * delta

        if self.config.fixed_rate_enabled:
            for joint_idx, limit in self.config.fixed_rate_limits.items():
                if 0 <= joint_idx < len(solution):
                    solution[joint_idx] = self._apply_fixed_rate_limit(solution[joint_idx], prev_q[joint_idx], limit)

        if self.config.elbow_back_block_enabled:
            solution = self._apply_elbow_back_block(solution)

        self._prev_q = solution.copy()
        return solution

    def _apply_elbow_soft_stop(
        self,
        solution: np.ndarray,
        prev_q: np.ndarray,
        elbow_soft_stop: float | None,
    ) -> np.ndarray:
        elbow_idx = self.config.elbow_soft_stop_index
        if not 0 <= elbow_idx < len(solution):
            return solution

        if elbow_soft_stop is not None:
            if solution[elbow_idx] < elbow_soft_stop:
                small_step = np.deg2rad(self.config.elbow_soft_stop_below_small_step_deg)
                margin = np.deg2rad(self.config.elbow_soft_stop_below_margin_deg)
                allowed = max(solution[elbow_idx], prev_q[elbow_idx] - small_step)
                solution[elbow_idx] = max(allowed, elbow_soft_stop - margin)
            else:
                max_down_per_call = np.deg2rad(self.config.elbow_soft_stop_above_max_down_deg)
                solution[elbow_idx] = max(solution[elbow_idx], prev_q[elbow_idx] - max_down_per_call)
            return solution

        max_down_per_call = np.deg2rad(self.config.elbow_soft_stop_above_max_down_deg)
        solution[elbow_idx] = max(solution[elbow_idx], prev_q[elbow_idx] - max_down_per_call)
        return solution

    def _apply_fixed_rate_limit(self, target: float, previous: float, limit: FixedRateJointLimit) -> float:
        step_rad = np.deg2rad(limit.step_deg)
        deadband_rad = np.deg2rad(limit.deadband_deg)
        error = target - previous
        if abs(error) <= deadband_rad:
            return previous
        direction = 1.0 if error > 0 else -1.0
        return previous + direction * min(step_rad, abs(error))

    def _apply_elbow_back_block(self, solution: np.ndarray) -> np.ndarray:
        elbow_idx = self.config.elbow_back_block_index
        if not 0 <= elbow_idx < len(solution):
            return solution

        if self._elbow_back_limit is None:
            self._elbow_back_limit = float(solution[elbow_idx])

        back_margin = np.deg2rad(self.config.elbow_back_block_tolerance_deg)
        direction = self.config.elbow_back_block_direction.lower()
        if direction == "decrease":
            solution[elbow_idx] = max(solution[elbow_idx], self._elbow_back_limit - back_margin)
        else:
            solution[elbow_idx] = min(solution[elbow_idx], self._elbow_back_limit + back_margin)
        return solution
