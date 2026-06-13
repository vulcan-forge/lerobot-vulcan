import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.robots.sourccey.sourccey.sourccey_follower.sourccey_follower import SourcceyFollower

logger = logging.getLogger(__name__)


class SourcceyFollowerSafety:

    def __init__(self, robot: "SourcceyFollower"):
        self.robot = robot
        self._last_goal_pos: dict[str, float] = {}

    def remember_goal(self, goal_pos: dict[str, float]) -> None:
        """Store the most recent commanded goal so we can infer blocked direction next frame."""
        self._last_goal_pos = goal_pos.copy()

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
                    logger.warning("Detected overload for %s while reading current.", motor_name)
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
