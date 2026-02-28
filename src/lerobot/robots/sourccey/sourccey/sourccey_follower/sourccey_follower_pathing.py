import json
import logging
from dataclasses import dataclass
from pathlib import Path

from lerobot.robots.sourccey.sourccey.sourccey_follower.config_sourccey_follower import SourcceyFollowerConfig

logger = logging.getLogger(__name__)


@dataclass
class _RecoveryStage:
    name: str
    goal_pos: dict[str, float]
    hold_cycles: int
    motors: tuple[str, ...]


class SourcceyFollowerPathing:
    """Deterministic staged recovery for stalled follower motion."""

    POSTURE_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")

    def __init__(self, config: SourcceyFollowerConfig):
        self.config = config
        self._default_recovery_pose = self._load_default_recovery_pose()
        self._active_recovery_pose = self._load_active_recovery_pose()
        self.reset()

    def reset(self) -> None:
        self._last_present_pos: dict[str, float] | None = None
        self._stall_cycles = 0
        self._last_stalled_signature: tuple[str, ...] = ()
        self._recovery_queue: list[_RecoveryStage] = []
        self._recovery_attempts: dict[tuple[str, ...], int] = {}

    def apply_recovery_pathing(
        self,
        goal_pos: dict[str, float],
        present_pos: dict[str, float],
        pause_motors: set[str] | None = None,
    ) -> dict[str, float]:
        """Return the requested goal or an intermediate recovery waypoint."""
        if not self.config.enable_recovery_pathing:
            return goal_pos

        requested_goal = goal_pos.copy()
        pause_motors = pause_motors or set()

        if self._recovery_queue:
            staged_goal = self._consume_recovery_stage(pause_motors)
            self._last_present_pos = present_pos.copy()
            return staged_goal

        stalled_joints = self._find_stalled_joints(requested_goal, present_pos)
        if stalled_joints:
            self._track_stall_signature(stalled_joints)
            if self._stall_cycles >= self.config.recovery_stall_window:
                stalled_signature = tuple(sorted(stalled_joints))
                attempt_number = self._recovery_attempts.get(stalled_signature, 0)
                strategy_name, self._recovery_queue = self._build_recovery_queue(requested_goal, present_pos, stalled_joints)
                self._reset_stall_tracking()
                if self._recovery_queue:
                    logger.warning(
                        "Recovery pathing activated after repeated stalled motion on %s using %s (attempt %d).",
                        ", ".join(stalled_joints),
                        strategy_name,
                        attempt_number + 1,
                    )
                    staged_goal = self._consume_recovery_stage(pause_motors)
                    self._last_present_pos = present_pos.copy()
                    return staged_goal
        else:
            self._reset_stall_tracking()

        self._last_present_pos = present_pos.copy()
        return requested_goal

    def _consume_recovery_stage(self, pause_motors: set[str]) -> dict[str, float]:
        stage = self._recovery_queue[0]
        goal_pos = stage.goal_pos.copy()
        if pause_motors.intersection(stage.motors):
            return goal_pos

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
    ) -> tuple[str, list[_RecoveryStage]]:
        stalled_signature = tuple(sorted(stalled_joints))
        attempt_number = self._recovery_attempts.get(stalled_signature, 0)
        self._recovery_attempts[stalled_signature] = attempt_number + 1
        return "table_escape_sequence", self._build_escape_sequence(goal_pos)

    def _build_escape_sequence(self, goal_pos: dict[str, float]) -> list[_RecoveryStage]:
        recovery_queue: list[_RecoveryStage] = []
        staged_goal = goal_pos.copy()

        sequence = (
            ("shoulder_lift_up", ("shoulder_lift",), self._active_recovery_pose),
            ("elbow_flex_up", ("elbow_flex",), self._active_recovery_pose),
            ("shoulder_pan_out", ("shoulder_pan",), self._active_recovery_pose),
            ("shoulder_elbow_down", ("shoulder_lift", "elbow_flex"), self._default_recovery_pose),
            ("shoulder_pan_in", ("shoulder_pan",), self._default_recovery_pose),
        )

        for stage_name, motors, target_pose in sequence:
            next_goal = staged_goal.copy()
            changed = False
            for motor_name in motors:
                if motor_name not in next_goal or motor_name not in target_pose:
                    continue

                target_value = target_pose[motor_name]
                if next_goal[motor_name] == target_value:
                    continue

                next_goal[motor_name] = target_value
                changed = True

            if not changed:
                continue

            recovery_queue.append(
                _RecoveryStage(
                    name=stage_name,
                    goal_pos=next_goal,
                    hold_cycles=self.config.recovery_stage_hold_cycles,
                    motors=motors,
                )
            )
            staged_goal = next_goal

        return recovery_queue

    def _load_default_recovery_pose(self) -> dict[str, float]:
        defaults_dir = Path(__file__).resolve().parents[1] / "sourccey" / "defaults"
        default_action_fpath = defaults_dir / f"{self.config.orientation}_arm_default_action.json"
        return self._load_recovery_pose(default_action_fpath)

    def _load_active_recovery_pose(self) -> dict[str, float]:
        defaults_dir = Path(__file__).resolve().parents[1] / "sourccey" / "defaults"
        active_action_fpath = defaults_dir / f"{self.config.orientation}_arm_default_active_action.json"
        return self._load_recovery_pose(active_action_fpath)

    def _load_recovery_pose(self, action_fpath: Path) -> dict[str, float]:
        try:
            with action_fpath.open() as file:
                payload = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to load recovery pose from %s: %s. Falling back to neutral recovery heuristics.",
                action_fpath,
                exc,
            )
            return {
                motor_name: self.config.recovery_neutral_pose_value
                for motor_name in self.POSTURE_MOTORS
            }

        normalized_pose: dict[str, float] = {}
        for key, value in payload.items():
            if not key.endswith(".pos"):
                continue

            motor_name = key.removesuffix(".pos")
            if motor_name.startswith("left_"):
                motor_name = motor_name.removeprefix("left_")
            elif motor_name.startswith("right_"):
                motor_name = motor_name.removeprefix("right_")

            normalized_pose[motor_name] = value

        return normalized_pose
