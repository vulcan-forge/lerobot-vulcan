"""Compatibility wrapper for the LiDAR direct snapshot stitch command."""

from lerobot.robots.sourccey.sourccey.sourccey.modules.slam.lidar import direct_snapshot_stitch as _impl
from lerobot.robots.sourccey.sourccey.sourccey.modules.slam.lidar.direct_snapshot_stitch import *  # noqa: F401,F403
from lerobot.robots.sourccey.sourccey.sourccey.modules.slam.lidar.direct_snapshot_stitch import main as _main

globals().update(
    {name: getattr(_impl, name) for name in dir(_impl) if name.startswith("_") and not name.startswith("__")}
)


if __name__ == "__main__":
    raise SystemExit(_main())
