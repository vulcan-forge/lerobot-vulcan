import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class SourcceyFollowerSafety:
    ###################################################################
    # Tunable Thresholds
    #
    # These constants define when we:
    # - enter slow-step mode because a move is large or the joint is loaded
    # - enter stronger overcurrent handling because torque is too high
    ###################################################################
    STEP_SAFETY_STARTUP_WINDOW_S = 3.0
    STEP_CURRENT_TARGET_TOLERANCE = 2.0
    OVERCURRENT_DIRECTION_TOLERANCE = 1.0
    MANUAL_PUSH_DIRECTION_TOLERANCE = 0.75
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
        "shoulder_pan": 60.0,
        "shoulder_lift": 120.0,
        "elbow_flex": 100.0,
        "wrist_flex": 60.0,
        "wrist_roll": 60.0,
        "gripper": 35.0,
    }
    DEFAULT_REVERSE_CURRENT_LIMITS = {
        "shoulder_pan": 96.0,
        "shoulder_lift": 192.0,
        "elbow_flex": 160.0,
        "wrist_flex": 96.0,
        "wrist_roll": 96.0,
        "gripper": 52.0,
    }

    ###################################################################
    # Lifecycle / State
    ###################################################################
    def __init__(self, robot: Any):
        self.robot = robot
        self._last_goal_pos: dict[str, float] = {}
        self._last_present_pos: dict[str, float] = {}
        self._action_stream_start_time: float | None = None
        self._step_safety_log_active = False
        self._last_overcurrent_log_time = 0.0
        self._overcurrent_log_interval_s = 5.0
        self._last_step_current_log_time = 0.0

    ###################################################################
    # Public API: Command Memory
    #
    # Used by:
    # - SourcceyFollower.send_action(...)
    #
    # Purpose:
    # - Remember the last commanded target so stronger safety behavior can
    #   infer whether a joint is still trying to push deeper into an obstacle.
    ###################################################################
    def remember_goal(self, goal_pos: dict[str, float]) -> None:
        """Store the most recent requested goal so we can infer blocked direction next frame."""
        self._last_goal_pos = goal_pos.copy()

    def remember_present(self, present_pos: dict[str, float]) -> None:
        """Store the latest measured joint positions so we can detect manual push direction next frame."""
        self._last_present_pos = present_pos.copy()

    ###################################################################
    # Public API: Step-Safety Triggering
    #
    # Used by:
    # - SourcceyFollower._apply_runtime_safety(...)
    #
    # Public functions in this section:
    # - should_use_step_safety(...)
    # - apply_step_safety(...)
    #
    # Purpose:
    # - Enter slow-step mode during startup or large target jumps
    # - Turn one large move into a sequence of smaller safe increments
    ###################################################################
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

    ###################################################################
    # Public API: Current Threshold Detection
    #
    # Used by:
    # - SourcceyFollower._apply_runtime_safety(...)
    # - SourcceyFollower.get_observation(...)
    #
    # Public functions in this section:
    # - detect_step_current_motors(...)
    # - detect_active_step_current_motors(...)
    # - detect_overcurrent_motors(...)
    #
    # Private helpers used here:
    # - _detect_current_threshold_motors(...)
    # - _read_current_state(...)
    #
    # Purpose:
    # - Split current monitoring into two levels:
    #   1. lower threshold for slow-step mode
    #   2. higher threshold for retreat / stop behavior
    ###################################################################
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

    ###################################################################
    # Public API: Overcurrent Response
    #
    # Used by:
    # - SourcceyFollower._apply_runtime_safety(...)
    #
    # Private helpers used here:
    # - _get_current_safety_backoff(...)
    # - _direction_from_delta(...)
    #
    # Purpose:
    # - Once a joint crosses the higher threshold, reject deeper motion for
    #   that joint and substitute a tiny retreat target instead.
    ###################################################################
    def apply_overcurrent_retreat(
        self,
        goal_pos: dict[str, float],
        present_pos: dict[str, float],
        overcurrent_motors: dict[str, float],
    ) -> dict[str, float]:
        """Replace unsafe overcurrent joint targets with a small retreat away from the loaded direction."""
        safe_goal_pos = goal_pos.copy()
        for motor_name in overcurrent_motors:
            if motor_name not in present_pos:
                continue

            current_pos = float(present_pos[motor_name])
            requested_delta = float(goal_pos.get(motor_name, current_pos)) - current_pos
            requested_direction = self._direction_from_delta(requested_delta)
            blocked_direction = self._get_blocked_direction(
                motor_name,
                current_pos,
                requested_direction,
            )

            if blocked_direction == 0:
                safe_goal_pos[motor_name] = current_pos
                continue

            # If the fresh command is already backing away from the blocked direction,
            # let it through immediately instead of forcing the previous retreat.
            if requested_direction != 0 and requested_direction != blocked_direction:
                continue

            relief_direction = self._get_overcurrent_relief_direction(
                motor_name,
                current_pos,
                blocked_direction,
            )

            if relief_direction == 0:
                safe_goal_pos[motor_name] = current_pos
                continue

            # Make the overcurrent retreat at least as strong as one normal slow-step
            # increment so the joint visibly backs away instead of just twitching.
            retreat = max(
                self._get_current_safety_backoff(motor_name),
                self.STEP_SAFETY_MAX_STEPS.get(motor_name, 0.0),
            )
            safe_goal_pos[motor_name] = current_pos + (relief_direction * retreat)

        return safe_goal_pos

    ###################################################################
    # Private Helpers: Current Detection Core
    #
    # Used by:
    # - detect_step_current_motors(...)
    # - detect_overcurrent_motors(...)
    #
    # Purpose:
    # - Shared threshold comparison logic so both current levels use the same
    #   overload-aware raw read path.
    ###################################################################
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

    ###################################################################
    # Public API: Logging
    #
    # Used by:
    # - SourcceyFollower._apply_runtime_safety(...)
    # - SourcceyFollower.get_observation(...)
    #
    # Public functions in this section:
    # - log_step_current_motors(...)
    # - log_overcurrent_motors(...)
    #
    # Private helpers used here:
    # - _log_current_threshold_motors(...)
    #
    # Purpose:
    # - Emit throttled logs for tuning without flooding the console.
    ###################################################################
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

    ###################################################################
    # Private Helpers: Logging
    #
    # Used by:
    # - log_step_current_motors(...)
    # - log_overcurrent_motors(...)
    ###################################################################
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

    ###################################################################
    # Private Helpers: Retreat Direction / Magnitude
    #
    # Used by:
    # - apply_overcurrent_retreat(...)
    ###################################################################
    def _get_overcurrent_relief_direction(
        self,
        motor_name: str,
        current_pos: float,
        blocked_direction: int,
    ) -> int:
        """Choose the safest direction to unload a joint during overcurrent.

        Priority:
        1. If the joint was physically moved since the last frame, follow that measured motion.
           This makes the arm yield in the direction a person is actively pushing it.
        2. Otherwise, move away from the blocked direction.
        """
        manual_push_direction = self._get_manual_push_direction(motor_name, current_pos)
        if manual_push_direction != 0 and (
            blocked_direction == 0 or manual_push_direction != blocked_direction
        ):
            return manual_push_direction

        if blocked_direction != 0:
            return -blocked_direction

        return 0

    def _get_blocked_direction(
        self,
        motor_name: str,
        current_pos: float,
        requested_direction: int,
    ) -> int:
        """Infer which direction is currently pushing the joint deeper into a jam."""
        if requested_direction != 0:
            return requested_direction

        blocked_delta = float(self._last_goal_pos.get(motor_name, current_pos)) - current_pos
        return self._direction_from_delta(blocked_delta)

    def _get_manual_push_direction(self, motor_name: str, current_pos: float) -> int:
        """Infer external push direction from recent measured position drift."""
        last_present_pos = self._last_present_pos.get(motor_name)
        if last_present_pos is None:
            return 0

        recent_motion = current_pos - float(last_present_pos)
        if recent_motion > self.MANUAL_PUSH_DIRECTION_TOLERANCE:
            return 1
        if recent_motion < -self.MANUAL_PUSH_DIRECTION_TOLERANCE:
            return -1

        return 0

    def _get_current_safety_backoff(self, motor_name: str) -> float:
        """Return the small retreat distance to use when reducing overcurrent on a joint."""
        if motor_name == "gripper":
            return self.robot.config.gripper_current_safety_backoff

        return self.robot.config.current_safety_backoff

    def _direction_from_delta(self, delta: float) -> int:
        """Collapse a delta into -1 / 0 / 1 using a small tolerance band."""
        if delta > self.OVERCURRENT_DIRECTION_TOLERANCE:
            return 1
        if delta < -self.OVERCURRENT_DIRECTION_TOLERANCE:
            return -1
        return 0

    ###################################################################
    # Private Helpers: Raw Motor Reads
    #
    # Used by:
    # - _detect_current_threshold_motors(...)
    #
    # Purpose:
    # - Read Present_Current with runtime-friendly retry behavior while still
    #   treating overload exceptions as real safety events.
    ###################################################################
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
