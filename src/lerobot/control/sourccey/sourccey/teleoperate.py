import time
from dataclasses import dataclass

from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClientConfig, SourcceyClient
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.sourccey.sourccey.bi_sourccey_leader.bi_sourccey_leader import BiSourcceyLeader
from lerobot.teleoperators.sourccey.sourccey.bi_sourccey_leader.config_bi_sourccey_leader import BiSourcceyLeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.configs import parser

from pynput import keyboard


@dataclass
class SourcceyTeleoperateConfig:
    id: str = "sourccey"
    remote_ip: str = "192.168.1.243"
    left_arm_port: str = "COM4"
    right_arm_port: str = "COM3"
    keyboard_port: str = "keyboard"
    fps: int = 30
    reversed: bool = False

@parser.wrap()
def teleoperate(cfg: SourcceyTeleoperateConfig):
    # Create the robot and teleoperator configurations
    robot_config = SourcceyClientConfig(remote_ip=cfg.remote_ip, id=cfg.id, reversed=cfg.reversed)
    teleop_arm_config = BiSourcceyLeaderConfig(left_arm_port=cfg.left_arm_port, right_arm_port=cfg.right_arm_port, id=cfg.id)
    keyboard_config = KeyboardTeleopConfig(id=cfg.keyboard_port)

    robot = SourcceyClient(robot_config)
    leader_arm = BiSourcceyLeader(teleop_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    robot.connect()

    try:
        leader_arm.connect()
    except Exception as e:
        print(f"Teleoperating without leader arm")
        pass

    try:
        keyboard.connect()
    except Exception as e:
        print(f"Teleoperating without keyboard")
        pass

    start_speed_listener(robot)
    init_rerun(session_name="sourccey_teleop")

    print("Teleoperating Sourccey")
    while True:
        t0 = time.perf_counter()

        observation = robot.get_observation()
        arm_action = leader_arm.get_action()
        keyboard_keys = keyboard.get_action()

        base_action = robot._from_keyboard_to_base_action(keyboard_keys)

        log_rerun_data(observation, {**arm_action, **base_action})

        action = {**arm_action, **base_action}
        robot.send_action(action)

        precise_sleep(max(1.0 / cfg.fps - (time.perf_counter() - t0), 0.0))

def start_speed_listener(robot: SourcceyClient):
    def on_press(key):
        # Only normal keys have .char
        if hasattr(key, "char") and key.char:
            robot.on_key_down(key.char)

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener

def main():
    teleoperate()

if __name__ == "__main__":
    main()
