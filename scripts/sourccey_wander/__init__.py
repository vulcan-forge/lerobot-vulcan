"""Compatibility package for legacy ``sourccey_wander`` script imports.

The real implementation now lives under the Sourccey SLAM package.  Do not point
``__path__`` at that folder directly: Python would then load the files as
``sourccey_wander.*`` modules, breaking their package-relative imports like
``from ..lidar``.  Instead, alias the legacy module names to the real package
modules so old scripts keep working while the implementation stays organized.
"""

from __future__ import annotations

import importlib
import sys

_REAL_PACKAGE = "lerobot.robots.sourccey.sourccey.sourccey.modules.slam.wander"
_SUBMODULES = (
    "wander_types",
    "imu_heading",
    "lidar_feed",
    "stop_zone",
    "frontier",
    "boxed_in",
    "mapping",
    "localization",
    "calibration",
    "driving",
)

for _name in _SUBMODULES:
    _module = importlib.import_module(f"{_REAL_PACKAGE}.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _module

__all__ = list(_SUBMODULES)
