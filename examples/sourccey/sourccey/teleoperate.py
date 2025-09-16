import time

from lerobot.robots.sourccey.sourccey.sourccey import Sourccey, SourcceyClientConfig, SourcceyClient
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.sourccey.sourccey.bi_sourccey_leader.bi_sourccey_leader import BiSourcceyLeader
from lerobot.teleoperators.sourccey.sourccey.bi_sourccey_leader.config_bi_sourccey_leader import BiSourcceyLeaderConfig
from lerobot.teleoperators.sourccey.sourccey.sourccey_leader.config_sourccey_leader import SourcceyLeaderConfig
from lerobot.teleoperators.sourccey.sourccey.sourccey_leader.sourccey_leader import SourcceyLeader
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

FPS = 30

# Create the robot and teleoperator configurations
robot_config = SourcceyClientConfig(remote_ip="192.168.1.237", id="sourccey")
teleop_arm_config = BiSourcceyLeaderConfig(left_arm_port="COM6", right_arm_port="COM5", id="sourccey")
keyboard_config = KeyboardTeleopConfig(id="keyboard")

robot = SourcceyClient(robot_config)
leader_arm = BiSourcceyLeader(teleop_arm_config)
keyboard = KeyboardTeleop(keyboard_config)

robot.connect()
leader_arm.connect()
keyboard.connect()

_init_rerun(session_name="sourccey_teleop")

if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
    raise ValueError("Robot, leader arm of keyboard is not connected!")

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

    busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
