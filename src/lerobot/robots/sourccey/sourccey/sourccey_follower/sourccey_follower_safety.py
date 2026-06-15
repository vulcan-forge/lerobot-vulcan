import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class SourcceyFollowerSafety:
    STEP_SAFETY_STARTUP_WINDOW_S = 3.0
    STEP_CURRENT_TARGET_TOLERANCE = 2.0
    STEP_SAFETY_DELTA_THRESHOLDS = {
        "shoulder_pan": 45.0,
        "shoulder_lift": 45.0,
        "elbow_flex": 45.0,
        "wrist_flex": 45.0,
        "wrist_roll": 45.0,
        "gripper": 45.0,
    }
    STEP_SAFETY_MAX_STEPS = {
        "shoulder_pan": 5.0,
        "shoulder_lift": 5.0,
        "elbow_flex": 5.0,
        "wrist_flex": 4.0,
        "wrist_roll": 5.0,
        "gripper": 5.0,
    }
    DEFAULT_STEP_CURRENT_LIMITS = {
        "shoulder_pan": 50.0,
        "shoulder_lift": 120.0,
        "elbow_flex": 70.0,
        "wrist_flex": 30.0,
        "wrist_roll": 30.0,
        "gripper": 30.0,
    }
    DEFAULT_REVERSE_CURRENT_LIMITS = {
        "shoulder_pan": 80.0,
        "shoulder_lift": 192.0,
        "elbow_flex": 112.0,
        "wrist_flex": 48.0,
        "wrist_roll": 48.0,
        "gripper": 48.0,
    }

    def __init__(self, robot: Any):
        self.robot = robot
        self._last_goal_pos: dict[str, float] = {}
        self._action_stream_start_time: float | None = None
        self._step_safety_log_active = False
        self._last_overcurrent_log_time = 0.0
        self._overcurrent_log_interval_s = 0.5
        self._last_step_current_log_time = 0.0

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

        large_deltas: dict[str, dict[str, float]] = {}
        for motor_name, target_pos in goal_pos.items():
            if motor_name not in present_pos:
                continue

            threshold = self.STEP_SAFETY_DELTA_THRESHOLDS.get(motor_name, 15.0)
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

    def apply_step_safety(
        self,
        goal_pos: dict[str, float],
        present_pos: dict[str, float],
    ) -> dict[str, float]:
        """Move toward the target in small per-joint increments instead of jumping directly."""
        slowed_goal_pos: dict[str, float] = {}

        for motor_name, target_pos in goal_pos.items():
            if motor_name not in present_pos:
                slowed_goal_pos[motor_name] = target_pos
                continue

            max_step = self.STEP_SAFETY_MAX_STEPS.get(motor_name, 5.0)
            current_pos = float(present_pos[motor_name])
            delta = float(target_pos) - current_pos

            if delta > max_step:
                slowed_goal_pos[motor_name] = current_pos + max_step
            elif delta < -max_step:
                slowed_goal_pos[motor_name] = current_pos - max_step
            else:
                slowed_goal_pos[motor_name] = float(target_pos)

        return slowed_goal_pos

    def detect_step_current_motors(self) -> dict[str, float]:
        """Return motors that have crossed the lower threshold for slow-motion mode."""
        return self._detect_current_threshold_motors(self.DEFAULT_STEP_CURRENT_LIMITS)

    def detect_active_step_current_motors(
        self,
        goal_pos: dict[str, float],
        present_pos: dict[str, float],
    ) -> dict[str, float]:
        """Return low-threshold current events only for joints that are still meaningfully moving."""
        step_current_motors = self.detect_step_current_motors()
        active_step_current_motors: dict[str, float] = {}

        for motor_name, current in step_current_motors.items():
            if motor_name not in goal_pos or motor_name not in present_pos:
                continue

            remaining_delta = abs(float(goal_pos[motor_name]) - float(present_pos[motor_name]))
            if remaining_delta >= self.STEP_CURRENT_TARGET_TOLERANCE:
                active_step_current_motors[motor_name] = current

        return active_step_current_motors

    def detect_overcurrent_motors(self) -> dict[str, float]:
        """Return motors that have crossed the higher threshold for reverse/stop behavior."""
        return self._detect_current_threshold_motors(self.DEFAULT_REVERSE_CURRENT_LIMITS)

    def _detect_current_threshold_motors(self, default_limits: dict[str, float]) -> dict[str, float]:
        """Return motors that are currently over a supplied current threshold map."""
        overcurrent_motors: dict[str, float] = {}

        for motor_name in self.robot.bus.motors:
            current, overloaded = self._read_current_state(motor_name)
            if overloaded:
                overcurrent_motors[motor_name] = current
                continue

            if motor_name == "gripper" and self.robot.config.gripper_current_safety_threshold is not None:
                current_limit = self.robot.config.gripper_current_safety_threshold
            else:
                current_limit = default_limits.get(
                    motor_name,
                    self.robot.config.max_current_safety_threshold,
                )

            if abs(current) > current_limit:
                overcurrent_motors[motor_name] = current

        return overcurrent_motors

    def log_step_current_motors(self, step_current_motors: dict[str, float]) -> None:
        """Log the lower slow-motion current threshold periodically while active."""
        self._log_current_threshold_motors(
            step_current_motors,
            label="Step-current trigger",
            last_log_attr="_last_step_current_log_time",
        )

    def log_overcurrent_motors(self, overcurrent_motors: dict[str, float]) -> None:
        """Log the higher reverse/stop current threshold periodically while active."""
        self._log_current_threshold_motors(
            overcurrent_motors,
            label="Overcurrent trigger",
            last_log_attr="_last_overcurrent_log_time",
        )

    def _log_current_threshold_motors(
        self,
        motors: dict[str, float],
        *,
        label: str,
        last_log_attr: str,
    ) -> None:
        """Shared throttled logger for current-threshold events."""
        if not motors:
            return

        now = time.monotonic()
        last_log_time = float(getattr(self, last_log_attr))
        if now - last_log_time < self._overcurrent_log_interval_s:
            return

        formatted = {
            motor_name: ("overload" if current == float("inf") else round(current, 2))
            for motor_name, current in motors.items()
        }
        logger.warning("%s for %s arm: %s", label, self.robot.config.orientation, formatted)
        setattr(self, last_log_attr, now)

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
