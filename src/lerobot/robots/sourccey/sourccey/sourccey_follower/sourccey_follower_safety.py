import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.robots.sourccey.sourccey.sourccey_follower.sourccey_follower import SourcceyFollower

logger = logging.getLogger(__name__)


@dataclass
class _ProtectedMotor:
    goal_pos: float
    safe_cycles: int = 0
    torque_disabled: bool = False


class SourcceyFollowerSafety:
    """Runtime current-based safety checks for follower motion."""

    CURRENT_SAFETY_POSITION_TOLERANCE = 0.25

    # These are intentionally lower than the hardware protection limits so software backs off
    # before the motors sit and fight an obstruction.
    DEFAULT_CURRENT_LIMITS = {
        "shoulder_pan": 900,
        "shoulder_lift": 1800,
        "elbow_flex": 1800,
        "wrist_flex": 900,
        "wrist_roll": 900,
        "gripper": 300,
    }

    def __init__(self, robot: "SourcceyFollower"):
        self.robot = robot
        self._last_goal_pos: dict[str, float] = {}
        self._protected_motors: dict[str, _ProtectedMotor] = {}

    def apply_current_safety(
        self,
        goal_pos: dict[str, float],
        present_pos: dict[str, float],
        currents: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Adjust goals if a motor is already over current and still being pushed deeper."""
        if currents is None:
            currents = self.robot.bus.sync_read("Present_Current")
        requested_goal = goal_pos.copy()
        resting_motors = self._check_rest_current_safety(currents)
        overcurrent_motors = self._check_current_safety(currents)
        goal_pos = self._handle_resting_motors(resting_motors, overcurrent_motors, goal_pos, present_pos)
        goal_pos = self._handle_overcurrent_motors(overcurrent_motors, goal_pos, present_pos)
        self._update_protected_motors(goal_pos, requested_goal, present_pos, currents)
        return self._apply_protected_motors(goal_pos, present_pos)

    def get_pathing_pause_motors(self, currents: dict[str, float] | None = None) -> set[str]:
        """Return motors that should pause recovery path progression until current settles."""
        if currents is None:
            currents = self.robot.bus.sync_read("Present_Current")

        pause_motors = set(self._protected_motors)
        pause_motors.update(self._check_rest_current_safety(currents))
        pause_motors.update(self._check_current_safety(currents))
        return pause_motors

    def remember_goal(self, goal_pos: dict[str, float]) -> None:
        """Store the most recent commanded goal so we can infer blocked direction next frame."""
        self._last_goal_pos = goal_pos.copy()

    def _check_rest_current_safety(self, currents: dict[str, float]) -> dict[str, float]:
        """Return motors currently above the early-rest threshold but below hard overcurrent."""
        rest_threshold_ratio = self.robot.config.current_rest_safety_threshold_ratio
        if rest_threshold_ratio is None or rest_threshold_ratio <= 0:
            return {}

        resting_motors: dict[str, float] = {}
        for motor, current in currents.items():
            if self._is_nan(current):
                continue

            hard_limit = self._get_motor_current_safety_limit(motor)
            rest_limit = hard_limit * rest_threshold_ratio
            absolute_current = abs(current)
            if rest_limit < absolute_current <= hard_limit:
                resting_motors[motor] = current

        return resting_motors

    def _check_current_safety(self, currents: dict[str, float]) -> dict[str, float]:
        """Return motors currently above their runtime current threshold."""
        overcurrent_motors: dict[str, float] = {}
        for motor, current in currents.items():
            if self._is_nan(current):
                continue

            current_limit = self._get_motor_current_safety_limit(motor)
            if abs(current) > current_limit:
                overcurrent_motors[motor] = current

        return overcurrent_motors

    def _get_motor_current_safety_limit(self, motor_name: str) -> int:
        """Return the runtime current limit for a motor."""
        if motor_name == "gripper" and self.robot.config.gripper_current_safety_threshold is not None:
            return self.robot.config.gripper_current_safety_threshold

        if motor_name in self.DEFAULT_CURRENT_LIMITS:
            return self.DEFAULT_CURRENT_LIMITS[motor_name]

        return self.robot.config.max_current_safety_threshold

    def _get_motor_current_safety_backoff(self, motor_name: str) -> float:
        """Return how far to retreat after an overcurrent event."""
        if motor_name == "gripper":
            return self.robot.config.gripper_current_safety_backoff

        return self.robot.config.current_safety_backoff

    def _get_motor_current_rest_backoff(self, motor_name: str) -> float:
        """Return how far to retreat when unloading before hard overcurrent."""
        if motor_name == "gripper":
            return self.robot.config.gripper_current_rest_backoff

        return self.robot.config.current_rest_backoff

    def _get_motor_current_release_limit(self, motor_name: str) -> float:
        """Return the current threshold below which a protected motor can rejoin the action stream."""
        release_ratio = self.robot.config.current_safe_release_threshold_ratio
        return self._get_motor_current_safety_limit(motor_name) * release_ratio

    def _get_stalled_direction(self, motor_name: str, present_pos: dict[str, float]) -> int:
        """Infer the blocked direction from the last target that failed to move."""
        if motor_name not in self._last_goal_pos or motor_name not in present_pos:
            return 0

        remaining_delta = self._last_goal_pos[motor_name] - present_pos[motor_name]
        if abs(remaining_delta) <= self.CURRENT_SAFETY_POSITION_TOLERANCE:
            return 0

        return 1 if remaining_delta > 0 else -1

    def _handle_resting_motors(
        self,
        resting_motors: dict[str, float],
        overcurrent_motors: dict[str, float],
        goal_pos: dict[str, float],
        present_pos: dict[str, float],
    ) -> dict[str, float]:
        """Preemptively unload motors that are drawing elevated current."""
        if not resting_motors:
            return goal_pos

        modified_goal_pos = goal_pos.copy()
        for motor_name, current in resting_motors.items():
            if motor_name in overcurrent_motors:
                continue

            if motor_name not in modified_goal_pos or motor_name not in present_pos:
                continue

            target_delta = modified_goal_pos[motor_name] - present_pos[motor_name]
            if abs(target_delta) <= self.CURRENT_SAFETY_POSITION_TOLERANCE:
                continue

            target_direction = 1 if target_delta > 0 else -1
            stalled_direction = self._get_stalled_direction(motor_name, present_pos)

            if stalled_direction != 0 and target_direction != stalled_direction:
                continue

            retreat_direction = stalled_direction if stalled_direction != 0 else target_direction
            backoff = self._get_motor_current_rest_backoff(motor_name)
            rest_goal = present_pos[motor_name] - (retreat_direction * backoff)
            modified_goal_pos[motor_name] = rest_goal
            logger.warning(
                f"Resting {motor_name} before overcurrent ({current}mA): "
                f"requested {goal_pos[motor_name]}, rest target {rest_goal}."
            )

        return modified_goal_pos

    def _handle_overcurrent_motors(
        self,
        overcurrent_motors: dict[str, float],
        goal_pos: dict[str, float],
        present_pos: dict[str, float],
    ) -> dict[str, float]:
        """Clamp or back off motors that are already pushing into an obstruction."""
        if not overcurrent_motors:
            return goal_pos

        modified_goal_pos = goal_pos.copy()
        for motor_name, current in overcurrent_motors.items():
            if motor_name not in modified_goal_pos or motor_name not in present_pos:
                continue

            if self.robot.config.disable_torque_on_hard_overcurrent:
                self._disable_motor_torque(motor_name, present_pos[motor_name], current)
                modified_goal_pos[motor_name] = present_pos[motor_name]
                continue

            target_delta = modified_goal_pos[motor_name] - present_pos[motor_name]
            if abs(target_delta) <= self.CURRENT_SAFETY_POSITION_TOLERANCE:
                continue

            target_direction = 1 if target_delta > 0 else -1
            stalled_direction = self._get_stalled_direction(motor_name, present_pos)

            if stalled_direction != 0 and target_direction != stalled_direction:
                continue

            if stalled_direction == 0:
                modified_goal_pos[motor_name] = present_pos[motor_name]
                logger.warning(
                    f"Holding {motor_name} at {present_pos[motor_name]} because it is already over current ({current}mA)."
                )
                continue

            backoff = self._get_motor_current_safety_backoff(motor_name)
            safe_goal = present_pos[motor_name] - (stalled_direction * backoff)
            modified_goal_pos[motor_name] = safe_goal
            logger.warning(
                f"Backing off {motor_name} from overcurrent ({current}mA): "
                f"requested {goal_pos[motor_name]}, safe target {safe_goal}."
            )

        return modified_goal_pos

    def _disable_motor_torque(self, motor_name: str, present_value: float, current: float) -> None:
        """Drop torque on a single motor after a hard overcurrent event."""
        protected_motor = self._protected_motors.get(motor_name)
        if protected_motor and protected_motor.torque_disabled:
            protected_motor.goal_pos = present_value
            protected_motor.safe_cycles = 0
            return

        self.robot.bus.disable_torque(motor_name)
        self._protected_motors[motor_name] = _ProtectedMotor(
            goal_pos=present_value,
            safe_cycles=0,
            torque_disabled=True,
        )
        logger.warning(
            f"Disabling torque on {motor_name} after hard overcurrent ({current}mA)."
        )

    def _update_protected_motors(
        self,
        safe_goal_pos: dict[str, float],
        requested_goal_pos: dict[str, float],
        present_pos: dict[str, float],
        currents: dict[str, float],
    ) -> None:
        """Latch safe retreat goals until the joint has settled at a low current."""
        newly_protected: set[str] = set()
        for motor_name, safe_goal in safe_goal_pos.items():
            if motor_name not in requested_goal_pos or motor_name not in present_pos:
                continue

            requested_goal = requested_goal_pos[motor_name]
            if abs(safe_goal - requested_goal) <= self.CURRENT_SAFETY_POSITION_TOLERANCE:
                continue

            protected_motor = self._protected_motors.get(motor_name)
            if protected_motor is None:
                self._protected_motors[motor_name] = _ProtectedMotor(goal_pos=safe_goal, safe_cycles=0)
            else:
                protected_motor.goal_pos = safe_goal
                protected_motor.safe_cycles = 0
            newly_protected.add(motor_name)

        protected_motor_names = list(self._protected_motors)
        for motor_name in protected_motor_names:
            if motor_name in newly_protected:
                continue

            if motor_name not in currents or motor_name not in present_pos:
                del self._protected_motors[motor_name]
                continue

            protected_motor = self._protected_motors[motor_name]
            current = currents[motor_name]
            if self._is_nan(current):
                protected_motor.safe_cycles = 0
                continue

            if (
                abs(current) <= self._get_motor_current_release_limit(motor_name)
                and abs(protected_motor.goal_pos - present_pos[motor_name]) <= self.CURRENT_SAFETY_POSITION_TOLERANCE
            ):
                protected_motor.safe_cycles += 1
            else:
                protected_motor.safe_cycles = 0

            if protected_motor.safe_cycles >= self.robot.config.current_safe_hold_cycles:
                if protected_motor.torque_disabled:
                    self.robot.bus.enable_torque(motor_name)
                    protected_motor.goal_pos = present_pos[motor_name]
                    protected_motor.safe_cycles = 0
                    protected_motor.torque_disabled = False
                    logger.warning(
                        f"Re-enabled torque on {motor_name} after current settled; holding position before resuming stream."
                    )
                    continue

                del self._protected_motors[motor_name]

    def _apply_protected_motors(
        self,
        goal_pos: dict[str, float],
        present_pos: dict[str, float],
    ) -> dict[str, float]:
        """Override incoming actions while a motor is still in a protected settle window."""
        if not self._protected_motors:
            return goal_pos

        modified_goal_pos = goal_pos.copy()
        for motor_name, protected_motor in self._protected_motors.items():
            if motor_name not in modified_goal_pos or motor_name not in present_pos:
                continue

            if protected_motor.torque_disabled:
                modified_goal_pos[motor_name] = present_pos[motor_name]
                continue

            if abs(protected_motor.goal_pos - present_pos[motor_name]) <= self.CURRENT_SAFETY_POSITION_TOLERANCE:
                modified_goal_pos[motor_name] = present_pos[motor_name]
                continue

            modified_goal_pos[motor_name] = protected_motor.goal_pos

        return modified_goal_pos

    @staticmethod
    def _is_nan(value: float) -> bool:
        try:
            return math.isnan(value)
        except TypeError:
            return False
