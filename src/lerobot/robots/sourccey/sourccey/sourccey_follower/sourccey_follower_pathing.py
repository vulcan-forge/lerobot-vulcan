import logging
from dataclasses import dataclass

from lerobot.robots.sourccey.sourccey.sourccey_follower.config_sourccey_follower import SourcceyFollowerConfig

logger = logging.getLogger(__name__)


@dataclass
class _RecoveryStage:
    name: str
    goal_pos: dict[str, float]
    hold_cycles: int


class SourcceyFollowerPathing:
    """Lightweight opt-in recovery planning for stalled follower motion."""

    POSTURE_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")

    def __init__(self, config: SourcceyFollowerConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._last_present_pos: dict[str, float] | None = None
        self._stall_cycles = 0
        self._last_stalled_signature: tuple[str, ...] = ()
        self._recovery_queue: list[_RecoveryStage] = []

    def apply_recovery_pathing(
        self,
        goal_pos: dict[str, float],
        present_pos: dict[str, float],
    ) -> dict[str, float]:
        """Return the requested goal or an intermediate recovery waypoint."""
        if not self.config.enable_recovery_pathing:
            return goal_pos

        requested_goal = goal_pos.copy()

        if self._recovery_queue:
            staged_goal = self._consume_recovery_stage()
            self._last_present_pos = present_pos.copy()
            return staged_goal

        stalled_joints = self._find_stalled_joints(requested_goal, present_pos)
        if stalled_joints:
            self._track_stall_signature(stalled_joints)
            if self._stall_cycles >= self.config.recovery_stall_window:
                self._recovery_queue = self._build_recovery_queue(requested_goal, present_pos, stalled_joints)
                self._reset_stall_tracking()
                if self._recovery_queue:
                    logger.warning(
                        "Recovery pathing activated after repeated stalled motion on %s.",
                        ", ".join(stalled_joints),
                    )
                    staged_goal = self._consume_recovery_stage()
                    self._last_present_pos = present_pos.copy()
                    return staged_goal
        else:
            self._reset_stall_tracking()

        self._last_present_pos = present_pos.copy()
        return requested_goal

    def _consume_recovery_stage(self) -> dict[str, float]:
        stage = self._recovery_queue[0]
        goal_pos = stage.goal_pos.copy()
        stage.hold_cycles -= 1
        if stage.hold_cycles <= 0:
            self._recovery_queue.pop(0)
            if not self._recovery_queue:
                self._reset_stall_tracking()
        return goal_pos

    def _find_stalled_joints(
        self,
        goal_pos: dict[str, float],
        present_pos: dict[str, float],
    ) -> list[str]:
        if self._last_present_pos is None:
            return []

        stalled_joints: list[str] = []
        for motor_name, goal_value in goal_pos.items():
            if motor_name == "gripper":
                continue

            current_value = present_pos.get(motor_name)
            last_value = self._last_present_pos.get(motor_name)
            if current_value is None or last_value is None:
                continue

            remaining_error = goal_value - current_value
            if abs(remaining_error) < self.config.recovery_min_remaining_error:
                continue

            progress = abs(current_value - last_value)
            if progress <= self.config.recovery_min_progress:
                stalled_joints.append(motor_name)

        return stalled_joints

    def _track_stall_signature(self, stalled_joints: list[str]) -> None:
        signature = tuple(sorted(stalled_joints))
        if signature == self._last_stalled_signature:
            self._stall_cycles += 1
            return

        self._last_stalled_signature = signature
        self._stall_cycles = 1

    def _reset_stall_tracking(self) -> None:
        self._stall_cycles = 0
        self._last_stalled_signature = ()

    def _build_recovery_queue(
        self,
        goal_pos: dict[str, float],
        present_pos: dict[str, float],
        stalled_joints: list[str],
    ) -> list[_RecoveryStage]:
        recovery_queue: list[_RecoveryStage] = []

        backoff_goal = goal_pos.copy()
        for motor_name in stalled_joints:
            if motor_name not in goal_pos or motor_name not in present_pos:
                continue

            remaining_error = goal_pos[motor_name] - present_pos[motor_name]
            direction = 1.0 if remaining_error > 0 else -1.0
            backoff_goal[motor_name] = present_pos[motor_name] - (direction * self.config.recovery_joint_backoff)

        tucked_goal = backoff_goal.copy()
        for motor_name in self.POSTURE_MOTORS:
            if motor_name not in tucked_goal:
                continue

            tucked_goal[motor_name] = self._move_towards(
                tucked_goal[motor_name],
                self.config.recovery_neutral_pose_value,
                self.config.recovery_posture_step,
            )

        if backoff_goal != goal_pos:
            recovery_queue.append(
                _RecoveryStage(
                    name="backoff",
                    goal_pos=backoff_goal,
                    hold_cycles=self.config.recovery_stage_hold_cycles,
                )
            )

        if tucked_goal != backoff_goal:
            recovery_queue.append(
                _RecoveryStage(
                    name="tuck",
                    goal_pos=tucked_goal,
                    hold_cycles=self.config.recovery_stage_hold_cycles,
                )
            )

        return recovery_queue

    @staticmethod
    def _move_towards(current_value: float, target_value: float, max_step: float) -> float:
        delta = target_value - current_value
        if abs(delta) <= max_step:
            return target_value

        direction = 1.0 if delta > 0 else -1.0
        return current_value + (direction * max_step)
