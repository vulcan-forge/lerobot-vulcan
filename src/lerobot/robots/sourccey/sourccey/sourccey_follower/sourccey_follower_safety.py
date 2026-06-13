import logging
import math
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
