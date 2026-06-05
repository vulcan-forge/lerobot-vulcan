import time
from dataclasses import dataclass

from lerobot.configs import parser
from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClient, SourcceyClientConfig
from lerobot.utils.robot_utils import precise_sleep

from .manual_drive_bridge import _connect_with_retry
from .nav_follow_bridge import _build_default_arm_pose_action


@dataclass
class ApplyDefaultArmPoseConfig:
    id: str = "sourccey"
    remote_ip: str = "127.0.0.1"
    repeats: int = 4
    settle_s: float = 0.35
    hold_s: float = -1.0


def _send_default_arm_pose_burst(
    robot: SourcceyClient,
    *,
    observation: dict[str, object],
    repeats: int,
    settle_s: float,
    hold_s: float,
) -> None:
    action = _build_default_arm_pose_action(observation)
    for attempt in range(max(int(repeats), 1)):
        robot.send_action(action)
        if attempt < max(int(repeats), 1) - 1:
            precise_sleep(0.03)
    precise_sleep(max(float(settle_s), 0.0))
    end_time = time.monotonic() + max(float(hold_s), 0.0)
    while time.monotonic() < end_time:
        robot.send_action(action)
        precise_sleep(0.05)


@parser.wrap()
def apply_default_arm_pose(cfg: ApplyDefaultArmPoseConfig):
    robot_config = SourcceyClientConfig(remote_ip=cfg.remote_ip, id=cfg.id)
    setattr(robot_config, "untorque_on_disconnect", False)
    robot = SourcceyClient(robot_config)
    _connect_with_retry(robot)

    try:
        observation = robot.get_observation()
        if not isinstance(observation, dict):
            observation = {}
        _send_default_arm_pose_burst(
            robot,
            observation=observation,
            repeats=cfg.repeats,
            settle_s=cfg.settle_s,
            hold_s=cfg.hold_s,
        )
        if float(cfg.hold_s) < 0.0:
            action = _build_default_arm_pose_action(observation)
            print("Applied default arm pose. Holding until Ctrl+C.")
            while True:
                robot.send_action(action)
                precise_sleep(0.05)
        print("Applied default arm pose.")
    except KeyboardInterrupt:
        print("Default arm pose hold interrupted, shutting down.")
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


def main():
    apply_default_arm_pose()


if __name__ == "__main__":
    main()
