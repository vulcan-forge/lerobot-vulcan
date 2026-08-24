#!/usr/bin/env python3
"""Live check of Sourccey's two arm (wrist) cameras.

Runs on your control PC and connects to the robot exactly like the teleop client
does — the host's observation stream on `tcp://<remote-ip>:5556` — so
`sourccey_host.py` must be running on the Pi. The connection is passive
(observations only, no command socket), so this cannot move the robot.

    # watch the feeds live
    uv run python scripts/sourccey_check_arm_cameras.py --remote-ip 192.168.1.237

    # unattended 15s pass/fail with saved frames, no viewer
    uv run python scripts/sourccey_check_arm_cameras.py --remote-ip 192.168.1.237 \
        --view none --seconds 15 --save-dir camera_health

Cameras checked:
  wrist_left   /dev/cameraWristLeft
  wrist_right  /dev/cameraWristRight

Move the arms by hand (or with the normal teleop tools) if you want to confirm
the view is live; the summary also flags frozen feeds on its own.

Exit code is 0 only when both cameras delivered live, non-blank, non-frozen frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sourccey_check_common import build_parser, run_camera_check  # noqa: E402

from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import (  # noqa: E402
    sourccey_cameras_config,
)

CAMERA_KEYS = ("wrist_left", "wrist_right")


def main() -> int:
    parser = build_parser("Live view + health check for the two arm/wrist cameras.")
    # Different MJPEG port than the eye check so both previews can run at once.
    parser.set_defaults(http_port=8092)
    args = parser.parse_args()

    configured = sourccey_cameras_config()
    expected_fps = {key: float(configured[key].fps) for key in CAMERA_KEYS if key in configured}
    return run_camera_check(CAMERA_KEYS, expected_fps, "Sourccey arm cameras", args)


if __name__ == "__main__":
    raise SystemExit(main())
