from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_vector(
    value: Sequence[Any] | np.ndarray | None,
    *,
    size: int,
    default: Sequence[float],
) -> np.ndarray:
    if value is None:
        return np.array(default, dtype=float)
    try:
        seq = list(value)
    except TypeError:
        return np.array(default, dtype=float)
    if len(seq) != size:
        return np.array(default, dtype=float)
    try:
        return np.array([float(item) for item in seq], dtype=float)
    except (TypeError, ValueError):
        return np.array(default, dtype=float)


@dataclass(frozen=True, slots=True)
class BaseMotionCommand:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    active: bool = False

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "BaseMotionCommand":
        if payload is None:
            return cls()
        x = _coerce_float(payload.get("x.vel", 0.0))
        y = _coerce_float(payload.get("y.vel", 0.0))
        theta = _coerce_float(payload.get("theta.vel", 0.0))
        active = bool(payload.get("active", False)) or any(abs(val) > 0.0 for val in (x, y, theta))
        return cls(x=x, y=y, theta=theta, active=active)


@dataclass(frozen=True, slots=True)
class VRTeleopSample:
    position: np.ndarray
    rotation_wxyz: np.ndarray
    gripper_value: float = 0.0
    teleop_active: bool = False
    precision_mode: bool = False
    reset_mapping: bool = False
    is_resetting: bool = False
    start_episode: bool = False
    stop_episode: bool = False
    rerecord_episode: bool = False
    mark_success: bool = False
    base: BaseMotionCommand = field(default_factory=BaseMotionCommand)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "VRTeleopSample | None":
        if payload is None:
            return None
        return cls(
            position=_coerce_vector(payload.get("position"), size=3, default=(0.0, 0.0, 0.0)),
            rotation_wxyz=_coerce_vector(payload.get("rotation"), size=4, default=(1.0, 0.0, 0.0, 0.0)),
            gripper_value=_coerce_float(payload.get("gripper_value", 0.0)),
            teleop_active=bool(payload.get("switch", False)),
            precision_mode=bool(payload.get("precision", False)),
            reset_mapping=bool(payload.get("reset_mapping", False)),
            is_resetting=bool(payload.get("is_resetting", False)),
            start_episode=bool(payload.get("start_episode", False)),
            stop_episode=bool(payload.get("stop_episode", False)),
            rerecord_episode=bool(payload.get("rerecord_episode", False)),
            mark_success=bool(payload.get("mark_success", False)),
            base=BaseMotionCommand.from_payload(payload.get("base")),
        )


@dataclass(frozen=True, slots=True)
class ControlledArmObservation:
    arm_side: str
    joint_names: tuple[str, ...]
    joint_keys: tuple[str, ...]
    joint_positions_deg: list[float]
