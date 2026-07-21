"""Capture the arms' CURRENT calibrated positions and save them as the stow pose.

Never torques, never drives the arms. On a fresh-booted host the arms report
zeros until connected, so the script connects them with torque OFF (no targets
take effect — this sequence has run motion-free in the field). Then it reads the
positions, saves them, and closes WITHOUT the client's normal disconnect (which
would send a relax command) — nothing else ever leaves the command socket.

Position the arms physically (resting stably), run this, done. The explorer
drives the arms into the saved pose at startup only when launched with
--arm-stow (opt-in).

    uv run python scripts/sourccey_capture_arm_pose.py --remote-ip 192.168.1.237
"""
from __future__ import annotations

import argparse

from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient
from sourccey_arm_pose import (
    ARM_JOINTS,
    DEFAULT_POSE_PATH,
    bring_arms_online,
    read_pose_patient,
    save_pose,
)


def _close_without_sending(robot) -> None:
    """Tear the client down WITHOUT the usual disconnect (which sends a relax
    command). Nothing may leave the command socket, so close everything by hand."""
    for step in (
        robot._stop_slam_publish_thread,
        lambda: robot.zmq_observation_socket.close(0),
        lambda: robot.zmq_cmd_socket.close(0),
        lambda: robot.zmq_context.term(),
    ):
        try:
            step()
        except BaseException:
            pass
    robot._is_connected = False


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
        print("Reading arm positions ...")
        pose = read_pose_patient(robot, timeout_s=4.0)
        if pose is None:
            # Fresh-boot host: the arms report zeros until connected. Connect them
            # with torque OFF — no targets take effect, nothing is driven. (This
            # exact sequence ran motion-free in the field twice; the damage came
            # from the later torque-lock step, which no longer exists here.)
            print("Arms are offline (fresh host) — connecting them WITHOUT torque to read them.")
            print("NOTE: they stay unpowered/limp; if they are only balanced in place, make sure")
            print("they are resting stably.")
            bring_arms_online(robot)
            pose = read_pose_patient(robot, timeout_s=4.0)
            if pose is None:
                print("ERROR: arms still report all-zero after connecting. Check the host log.")
                return 1
        out = save_pose(pose, args.out)
        print(f"\nSaved stow pose -> {out}")
        for k in ARM_JOINTS:
            print(f"  {k:26s} {pose[k]:+9.3f}")
        print("\nThe explorer will move the arms into this pose at startup when run with")
        print("--arm-stow. First time, keep hands and objects clear of the arms.")
    finally:
        _close_without_sending(robot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
