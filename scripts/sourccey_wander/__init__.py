"""Compatibility package for legacy ``sourccey_wander`` script imports."""

from pathlib import Path

_SLAM_WANDER = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "lerobot"
    / "robots"
    / "sourccey"
    / "sourccey"
    / "sourccey"
    / "modules"
    / "slam"
    / "wander"
)

__path__ = [str(_SLAM_WANDER)]
