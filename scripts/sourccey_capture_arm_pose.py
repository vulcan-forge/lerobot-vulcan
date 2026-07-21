"""Capture the robot's arm pose and save it as the wander STOW pose.

It first RELAXES the arms (makes them limp) without ever driving them, waits for
you to position them by hand where they clear the LiDAR, then reads and saves the
joint positions. The wander/explore script loads this and holds it before spinning.

    uv run python scripts/sourccey_capture_arm_pose.py --remote-ip 192.168.1.237

Nothing is driven during capture — the arms are only ever relaxed and read.
"""
from __future__ import annotations

import argparse

from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient
from sourccey_arm_pose import ARM_JOINTS, DEFAULT_POSE_PATH, capture_pose, relax_arms, save_pose


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-ip", required=True, help="Robot host IP.")
    parser.add_argument("--robot-id", default="sourccey")
    parser.add_argument("--out", default=str(DEFAULT_POSE_PATH),
                        help="Where to write the stow pose JSON (defaults next to the scripts).")
    args = parser.parse_args()

    robot = SourcceyClient(SourcceyClientConfig(id=args.robot_id, remote_ip=args.remote_ip))
    robot.connect()
    try:
        print("Relaxing the arms (they will go limp — support them if needed) ...")
        if not relax_arms(robot):
            print("ERROR: the arms never came online (still reading zero). Is the host up?")
            return 1
        input("\nArms are LIMP. Position them out of the LiDAR's way by hand, then press Enter to capture...")
        pose = capture_pose(robot)
        if pose is None:
            print("ERROR: never received a complete arm reading from the robot.")
            return 1
        out = save_pose(pose, args.out)
        print(f"\nSaved stow pose -> {out}")
        for k in ARM_JOINTS:
            print(f"  {k:26s} {pose[k]:+9.3f}")
        print("\nWander/explore will move the arms to this pose (torqued) before spinning.")
    finally:
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
