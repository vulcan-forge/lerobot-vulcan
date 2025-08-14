#!/usr/bin/env python

import argparse

from lerobot.teleoperators.sourccey_v3beta.sourccey_v3beta_leader.config_sourccey_v3beta_leader import (
    SourcceyV3BetaLeaderConfig,
)
from lerobot.teleoperators.sourccey_v3beta.sourccey_v3beta_leader.sourccey_v3beta_leader import (
    SourcceyV3BetaLeader,
)


"""
Calibrate Sourccey V3 Beta leader arms (manual calibration).

This script calibrates the leader arm(s) you select: left, right, or both.
It runs the device's built-in manual calibration flow:
  1) Moves arm to the middle of its range, press ENTER
  2) Move each joint through its full range, press ENTER to finish

Example commands:

# Calibrate LEFT only
python examples/sourccey/sourccey_v3beta/calibrate_leaders.py \
  --arms left \
  --left-port COM28 \
  --left-id sourccey_v3beta_leader_left

# Calibrate RIGHT only
python examples/sourccey/sourccey_v3beta/calibrate_leaders.py \
  --arms right \
  --right-port COM23 \
  --right-id sourccey_v3beta_leader_right

# Calibrate BOTH
python examples/sourccey/sourccey_v3beta/calibrate_leaders.py \
  --arms both \
  --left-port COM28 --left-id sourccey_v3beta_leader_left \
  --right-port COM23 --right-id sourccey_v3beta_leader_right

Notes:
- Leader arms are NOT geared down like followers, so calibration is more direct
- On Linux the ports will look like /dev/ttyUSB0; on Windows they look like COM23
- Calibration files are saved to ~/.cache/huggingface/lerobot/calibration/teleoperators/
"""


def _calibrate_leader_arm(side: str, port: str, device_id: str) -> None:
    orientation = "right" if side == "right" else "left"
    cfg = SourcceyV3BetaLeaderConfig(
        port=port,
        id=device_id,
        orientation=orientation,
    )
    arm = SourcceyV3BetaLeader(cfg)
    arm.connect(calibrate=False)
    print(f"\nStarting MANUAL calibration for {side.upper()} leader arm ({device_id}) on port {port}...")
    print("Follow on-screen prompts: center the joints, press ENTER; then move through full range, press ENTER.")
    arm.calibrate()
    arm.disconnect()
    print(f"Completed calibration for {side.upper()} leader arm ({device_id}).\n")


def main():
    parser = argparse.ArgumentParser(description="Manual calibration for Sourccey V3 Beta leader arms")
    parser.add_argument("--arms", choices=["left", "right", "both"], default="both")
    parser.add_argument("--left-port", type=str, default=None, help="Serial port for left leader arm (e.g., COM28 or /dev/ttyUSB0)")
    parser.add_argument("--right-port", type=str, default=None, help="Serial port for right leader arm (e.g., COM23 or /dev/ttyUSB1)")
    parser.add_argument("--left-id", type=str, default="sourccey_v3beta_leader_left", help="Calibration ID for left leader arm")
    parser.add_argument("--right-id", type=str, default="sourccey_v3beta_leader_right", help="Calibration ID for right leader arm")
    args = parser.parse_args()

    if args.arms in ("left", "both") and not args.left_port:
        raise SystemExit("--left-port is required when --arms is 'left' or 'both'")
    if args.arms in ("right", "both") and not args.right_port:
        raise SystemExit("--right-port is required when --arms is 'right' or 'both'")

    if args.arms in ("left", "both"):
        _calibrate_leader_arm("left", args.left_port, args.left_id)
    if args.arms in ("right", "both"):
        _calibrate_leader_arm("right", args.right_port, args.right_id)


if __name__ == "__main__":
    main()
