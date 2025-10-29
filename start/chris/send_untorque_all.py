from dataclasses import dataclass

from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClientConfig, SourcceyClient
from lerobot.configs import parser


@dataclass
class SendUntorqueAllConfig:
    id: str = "sourccey"
    remote_ip: str = "192.168.1.225"


@parser.wrap()
def send_untorque_all(cfg: SendUntorqueAllConfig):
    robot_config = SourcceyClientConfig(remote_ip=cfg.remote_ip, id=cfg.id)
    robot = SourcceyClient(robot_config)

    robot.connect()
    try:
        print("CLIENT: Sending untorque_all command to host...")
        robot.send_action({"untorque_all": True})
        print("CLIENT: Command sent successfully")
    finally:
        robot.disconnect()
        print("CLIENT: Disconnected")


def main():
    send_untorque_all()


if __name__ == "__main__":
    main()

