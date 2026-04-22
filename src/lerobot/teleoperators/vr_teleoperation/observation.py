from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .models import ControlledArmObservation

ObservationDenormalizer = Callable[[Sequence[float]], list[float]]


class ControlledArmObservationSelector:
    def __init__(
        self,
        *,
        arm_side: str,
        joint_names: Sequence[str],
        observation_uses_degrees: bool,
        denormalize_observation: ObservationDenormalizer,
    ) -> None:
        self.arm_side = arm_side.lower()
        self.joint_names = tuple(joint_names)
        self.observation_uses_degrees = observation_uses_degrees
        self.denormalize_observation = denormalize_observation
        self.prefix = "left_" if self.arm_side == "left" else "right_"

    def extract(self, observation: Mapping[str, Any] | None) -> ControlledArmObservation | None:
        if observation is None:
            return None

        joint_keys = self._match_joint_keys(observation)
        if joint_keys is None:
            return None

        try:
            raw_values = [float(observation[key]) for key in joint_keys]
        except (TypeError, ValueError):
            return None

        joint_positions_deg = raw_values if self.observation_uses_degrees else self.denormalize_observation(raw_values)
        if len(joint_positions_deg) != len(joint_keys):
            return None

        return ControlledArmObservation(
            arm_side=self.arm_side,
            joint_names=self.joint_names,
            joint_keys=joint_keys,
            joint_positions_deg=list(joint_positions_deg),
        )

    def _match_joint_keys(self, observation: Mapping[str, Any]) -> tuple[str, ...] | None:
        preferred = tuple(f"{self.prefix}{joint_name}.pos" for joint_name in self.joint_names)
        if all(key in observation for key in preferred):
            return preferred

        fallback_keys: list[str] = []
        for joint_name in self.joint_names:
            suffix = f"{joint_name}.pos"
            matches = [obs_key for obs_key in observation.keys() if obs_key.endswith(suffix)]
            if len(matches) != 1:
                return None
            fallback_keys.append(matches[0])
        return tuple(fallback_keys)
