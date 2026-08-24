#!/usr/bin/env python3
"""Check the two arm (wrist) cameras directly - run this ON THE PI.

Opens `/dev/cameraWristLeft` and `/dev/cameraWristRight` itself, so it needs no
host and no network. That is the point: over the observation stream, a broken
camera and a camera the host never opened look the same, because the host
substitutes a black frame when a read fails. Here you see the device itself.

`sourccey_host.py` must be STOPPED - it holds the cameras exclusively, so a
running host shows up as a loud open failure rather than a misleading pass.

    # check both wrist cameras (5s, saves a frame from each)
    uv run python scripts/sourccey_check_arm_cameras_pi.py

    # watch them live from your PC while it runs
    uv run python scripts/sourccey_check_arm_cameras_pi.py --view mjpeg --seconds 0

This never commands the arms - it is a passive camera reader. Move them by hand
if you want to confirm the view is live; the check also flags frozen feeds itself.

For the same cameras as the ROBOT sees them (host running, from your control PC),
use `sourccey_check_arm_cameras.py --remote-ip <pi-ip>` instead.

Exit code is 0 only when both cameras opened and produced live, non-blank,
non-frozen frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sourccey_check_devices import build_parser, run_device_check, specs_for  # noqa: E402

ARM_KEYS = ("wrist_left", "wrist_right")


def main() -> int:
    parser = build_parser("Check the arm/wrist cameras directly on the Pi.")
    # Different MJPEG port than the eye check so both previews can run at once.
    parser.set_defaults(http_port=8095)
    args = parser.parse_args()

    return run_device_check(specs_for(ARM_KEYS, args), "Sourccey arm cameras (direct)", args)


if __name__ == "__main__":
    raise SystemExit(main())
