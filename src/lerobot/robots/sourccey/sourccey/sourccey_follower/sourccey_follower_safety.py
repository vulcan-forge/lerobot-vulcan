import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.robots.sourccey.sourccey.sourccey_follower.sourccey_follower import SourcceyFollower

logger = logging.getLogger(__name__)


class SourcceyFollowerSafety:
    STEP_SAFETY_STARTUP_WINDOW_S = 3.0
    STEP_SAFETY_DELTA_THRESHOLDS_DEGREES = {
        "shoulder_pan": 20.0,
        "shoulder_lift": 20.0,
        "elbow_flex": 20.0,
        "wrist_flex": 15.0,
        "wrist_roll": 20.0,
        "gripper": 15.0,
    }
    STEP_SAFETY_DELTA_THRESHOLDS_NORMALIZED = {
        "shoulder_pan": 15.0,
        "shoulder_lift": 15.0,
        "elbow_flex": 15.0,
        "wrist_flex": 10.0,
        "wrist_roll": 15.0,
        "gripper": 15.0,
    }
    DEFAULT_CURRENT_LIMITS = {
        "shoulder_pan": 37.5,
        "shoulder_lift": 75.0,
        "elbow_flex": 75.0,
        "wrist_flex": 37.5,
        "wrist_roll": 37.5,
        "gripper": 12.5,
    }

    def __init__(self, robot: "SourcceyFollower"):
        self.robot = robot
        self._last_goal_pos: dict[str, float] = {}
        self._action_stream_start_time: float | None = None
        self._step_safety_log_active = False
        self._last_overcurrent_log_time = 0.0
        self._overcurrent_log_interval_s = 0.25

    def remember_goal(self, goal_pos: dict[str, float]) -> None:
        """Store the most recent commanded goal so we can infer blocked direction next frame."""
        self._last_goal_pos = goal_pos.copy()

    def should_use_step_safety(
        self,
        goal_pos: dict[str, float],
        present_pos: dict[str, float],
    ) -> bool:
        """Detect large action jumps or startup transitions that should use slow stepping."""
        now = time.monotonic()
        if self._action_stream_start_time is None:
            self._action_stream_start_time = now

        startup_active = (now - self._action_stream_start_time) <= self.STEP_SAFETY_STARTUP_WINDOW_S
        thresholds = (
            self.STEP_SAFETY_DELTA_THRESHOLDS_DEGREES
            if self.robot.config.use_degrees
            else self.STEP_SAFETY_DELTA_THRESHOLDS_NORMALIZED
        )

        large_deltas: dict[str, dict[str, float]] = {}
        for motor_name, target_pos in goal_pos.items():
            if motor_name not in present_pos:
                continue

            threshold = thresholds.get(motor_name, 15.0)
            delta = abs(float(target_pos) - float(present_pos[motor_name]))
            if delta >= threshold:
                large_deltas[motor_name] = {
                    "delta": round(delta, 2),
                    "threshold": threshold,
                }

        should_use = startup_active or bool(large_deltas)
        if should_use and not self._step_safety_log_active:
            reasons: list[str] = []
            if startup_active:
                reasons.append(f"startup<{self.STEP_SAFETY_STARTUP_WINDOW_S}s")
            if large_deltas:
                reasons.append(f"large_delta={large_deltas}")

            logger.warning(
                "Step safety trigger for %s arm: %s",
                self.robot.config.orientation,
                ", ".join(reasons),
            )

        self._step_safety_log_active = should_use
        return should_use

    def detect_overcurrent_motors(self) -> dict[str, float]:
        """Return motors that are currently over their runtime current threshold."""
        overcurrent_motors: dict[str, float] = {}

        for motor_name in self.robot.bus.motors:
            current, overloaded = self._read_current_state(motor_name)
            if overloaded:
                overcurrent_motors[motor_name] = current
                continue

            if motor_name == "gripper" and self.robot.config.gripper_current_safety_threshold is not None:
                current_limit = self.robot.config.gripper_current_safety_threshold
            else:
                current_limit = self.DEFAULT_CURRENT_LIMITS.get(
                    motor_name,
                    self.robot.config.max_current_safety_threshold,
                )

            if abs(current) > current_limit:
                overcurrent_motors[motor_name] = current

        return overcurrent_motors

    def log_overcurrent_motors(self, overcurrent_motors: dict[str, float]) -> None:
        """Log overcurrent state periodically while active so current can be tested live."""
        if not overcurrent_motors:
            return

        now = time.monotonic()
        if now - self._last_overcurrent_log_time < self._overcurrent_log_interval_s:
            return

        formatted = {
            motor_name: ("overload" if current == float("inf") else round(current, 2))
            for motor_name, current in overcurrent_motors.items()
        }
        logger.warning(
            "Overcurrent trigger for %s arm: %s",
            self.robot.config.orientation,
            formatted,
        )
        self._last_overcurrent_log_time = now

    def _read_current_state(
        self,
        motor_name: str,
        max_retries: int = 1,
        base_delay: float = 0.02,
    ) -> tuple[float, bool]:
        """Read motor current and flag overload events using the calibrator's detection pattern."""
        for attempt in range(max_retries + 1):
            try:
                current = self.robot.bus.read("Present_Current", motor_name, normalize=False)
                return float(current), False
            except Exception as e:
                if "Overload error" in str(e):
                    return float("inf"), True

                if attempt == max_retries:
                    logger.error(
                        "Failed to read current for %s after %s attempts: %s",
                        motor_name,
                        max_retries + 1,
                        e,
                    )
                    return 0.0, False

                delay = base_delay * (2**attempt)
                logger.warning(
                    "Current read failed for %s on attempt %s. Retrying in %.3fs: %s",
                    motor_name,
                    attempt + 1,
                    delay,
                    e,
                )
                time.sleep(delay)

        return 0.0, False
