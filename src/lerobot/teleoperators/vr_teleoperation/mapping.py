from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy

_scipy_version = tuple(map(int, scipy.__version__.split(".")[:2]))
_supports_scalar_first = _scipy_version >= (1, 7)


def rotation_from_quat(quat: np.ndarray, *, scalar_first: bool = True) -> R:
    if _supports_scalar_first:
        return R.from_quat(quat, scalar_first=scalar_first)
    if scalar_first:
        quat = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)
    return R.from_quat(quat)


def quat_as_scalar_first(rotation: R) -> np.ndarray:
    if _supports_scalar_first:
        return rotation.as_quat(scalar_first=True)
    quat_xyzw = rotation.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)


def _coerce_axis_scale(
    value: tuple[float, float, float] | list[float] | np.ndarray | None,
    *,
    default: tuple[float, float, float],
) -> np.ndarray:
    if value is None:
        return np.array(default, dtype=float)
    try:
        arr = np.array(value, dtype=float)
    except (TypeError, ValueError):
        return np.array(default, dtype=float)
    if arr.shape != (3,):
        return np.array(default, dtype=float)
    return arr


class PoseMapper:
    def __init__(
        self,
        *,
        initial_robot_position: np.ndarray,
        initial_robot_wxyz: np.ndarray,
        sensitivity_normal: float,
        sensitivity_precision: float,
        rotation_sensitivity: float,
        mapping_gain: float,
        translation_axis_scale_normal: tuple[float, float, float] | list[float] | np.ndarray | None = None,
        translation_axis_scale_precision: tuple[float, float, float] | list[float] | np.ndarray | None = None,
        rotation_axis_scale: tuple[float, float, float] | list[float] | np.ndarray | None = None,
        translation_deadband_m: float = 0.0,
        rotation_deadband_rad: float = 0.0,
        incremental_mode: bool = True,
    ) -> None:
        self.sensitivity_normal = float(sensitivity_normal)
        self.sensitivity_precision = float(sensitivity_precision)
        self.rotation_sensitivity = float(rotation_sensitivity)
        self.mapping_gain = float(mapping_gain)
        self.translation_axis_scale_normal = _coerce_axis_scale(
            translation_axis_scale_normal,
            default=(1.0, 1.0, 1.0),
        )
        self.translation_axis_scale_precision = _coerce_axis_scale(
            translation_axis_scale_precision,
            default=(1.0, 1.0, 1.0),
        )
        self.rotation_axis_scale = _coerce_axis_scale(rotation_axis_scale, default=(1.0, 1.0, 1.0))
        self.translation_deadband_m = max(0.0, float(translation_deadband_m))
        self.rotation_deadband_rad = max(0.0, float(rotation_deadband_rad))
        self.incremental_mode = bool(incremental_mode)

        self.current_robot_position = np.array(initial_robot_position, dtype=float)
        self.current_robot_wxyz = np.array(initial_robot_wxyz, dtype=float)
        self.initial_input_position: np.ndarray | None = None
        self.initial_input_wxyz: np.ndarray | None = None
        self.mapping_rotation: R | None = None
        self.mapping_translation: np.ndarray | None = None
        self.last_precision_mode = False

    def _update_mapping_from_current_pose(self, input_position: np.ndarray, input_wxyz: np.ndarray) -> None:
        robot_rotation = rotation_from_quat(self.current_robot_wxyz, scalar_first=True)
        input_rotation = rotation_from_quat(input_wxyz, scalar_first=True)
        self.mapping_rotation = robot_rotation * input_rotation.inv()
        self.mapping_translation = self.current_robot_position - self.mapping_rotation.apply(input_position)

    def _apply_translation_deadband(self, delta: np.ndarray) -> np.ndarray:
        if self.translation_deadband_m <= 0.0:
            return delta
        out = delta.copy()
        out[np.abs(out) < self.translation_deadband_m] = 0.0
        return out

    def _apply_rotation_deadband(self, rotvec: np.ndarray) -> np.ndarray:
        if self.rotation_deadband_rad <= 0.0:
            return rotvec
        if np.linalg.norm(rotvec) < self.rotation_deadband_rad:
            return np.zeros(3, dtype=float)
        return rotvec

    def set_robot_pose(self, position: np.ndarray, wxyz: np.ndarray) -> None:
        self.current_robot_position = np.array(position, dtype=float)
        self.current_robot_wxyz = np.array(wxyz, dtype=float)
        if self.initial_input_position is not None and self.initial_input_wxyz is not None:
            self._update_mapping_from_current_pose(self.initial_input_position, self.initial_input_wxyz)

    def open_session(self, input_position: np.ndarray, input_wxyz: np.ndarray) -> None:
        input_position = np.array(input_position, dtype=float)
        input_wxyz = np.array(input_wxyz, dtype=float)

        robot_rotation = rotation_from_quat(self.current_robot_wxyz, scalar_first=True)
        input_rotation = rotation_from_quat(input_wxyz, scalar_first=True)

        self.initial_input_position = input_position.copy()
        self.initial_input_wxyz = input_wxyz.copy()
        self.mapping_rotation = robot_rotation * input_rotation.inv()
        self.mapping_translation = self.current_robot_position - self.mapping_rotation.apply(input_position)

    def reset_mapping(self, input_position: np.ndarray, input_wxyz: np.ndarray) -> None:
        self.initial_input_position = np.array(input_position, dtype=float)
        self.initial_input_wxyz = np.array(input_wxyz, dtype=float)

        self._update_mapping_from_current_pose(self.initial_input_position, self.initial_input_wxyz)

    def map_pose(
        self,
        input_position: np.ndarray,
        input_wxyz: np.ndarray,
        *,
        precision_mode: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        input_position = np.array(input_position, dtype=float)
        input_wxyz = np.array(input_wxyz, dtype=float)

        if self.initial_input_position is None or self.initial_input_wxyz is None:
            self.open_session(input_position, input_wxyz)

        if precision_mode != self.last_precision_mode:
            self.reset_mapping(input_position, input_wxyz)
        self.last_precision_mode = precision_mode

        scale = (self.sensitivity_precision if precision_mode else self.sensitivity_normal) * self.mapping_gain
        translation_axis_scale = (
            self.translation_axis_scale_precision if precision_mode else self.translation_axis_scale_normal
        )

        delta = (input_position - self.initial_input_position) * scale
        delta *= translation_axis_scale
        delta = self._apply_translation_deadband(delta)

        initial_rotation = rotation_from_quat(self.initial_input_wxyz, scalar_first=True)
        current_input_rotation = rotation_from_quat(input_wxyz, scalar_first=True)
        relative_rotation = initial_rotation.inv() * current_input_rotation
        rotvec = relative_rotation.as_rotvec() * (self.rotation_sensitivity * self.mapping_gain)
        rotvec *= self.rotation_axis_scale
        rotvec = self._apply_rotation_deadband(rotvec)
        scaled_rotation = R.from_rotvec(rotvec)

        if self.mapping_rotation is None or self.mapping_translation is None:
            self._update_mapping_from_current_pose(self.initial_input_position, self.initial_input_wxyz)

        if self.incremental_mode:
            robot_rotation_base = rotation_from_quat(self.current_robot_wxyz, scalar_first=True)
            robot_rotation = robot_rotation_base * scaled_rotation
            robot_position = self.current_robot_position + self.mapping_rotation.apply(delta)
            self.current_robot_position = robot_position
            self.current_robot_wxyz = quat_as_scalar_first(robot_rotation)
            self.initial_input_position = input_position.copy()
            self.initial_input_wxyz = input_wxyz.copy()
            self._update_mapping_from_current_pose(self.initial_input_position, self.initial_input_wxyz)
            return robot_position, self.current_robot_wxyz

        scaled_position = self.initial_input_position + delta
        mapped_input_rotation = initial_rotation * scaled_rotation
        robot_rotation = self.mapping_rotation * mapped_input_rotation
        robot_position = self.mapping_rotation.apply(scaled_position) + self.mapping_translation

        self.current_robot_position = robot_position
        self.current_robot_wxyz = quat_as_scalar_first(robot_rotation)
        return robot_position, self.current_robot_wxyz
