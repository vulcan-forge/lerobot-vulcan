import time

from lerobot.robots.sourccey.sourccey_v3beta.sourccey_v3beta import SourcceyV3Beta, SourcceyV3BetaClientConfig, SourcceyV3BetaClient
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.sourccey_v3beta.bi_sourccey_v3beta_leader.bi_sourccey_v3beta_leader import BiSourcceyV3BetaLeader
from lerobot.teleoperators.sourccey_v3beta.bi_sourccey_v3beta_leader.config_bi_sourccey_v3beta_leader import BiSourcceyV3BetaLeaderConfig
from lerobot.teleoperators.sourccey_v3beta.sourccey_v3beta_leader.config_sourccey_v3beta_leader import SourcceyV3BetaLeaderConfig
from lerobot.teleoperators.sourccey_v3beta.sourccey_v3beta_leader.sourccey_v3beta_leader import SourcceyV3BetaLeader
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

FPS = 30

# Create the robot and teleoperator configurations
robot_config = SourcceyV3BetaClientConfig(remote_ip="192.168.1.219", id="sourccey_v3beta")
teleop_arm_config = BiSourcceyV3BetaLeaderConfig(left_arm_port="COM41", right_arm_port="COM42", id="bi_sourccey_v3beta_leader")
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

robot = SourcceyV3BetaClient(robot_config)
leader_arm = BiSourcceyV3BetaLeader(teleop_arm_config)
keyboard = KeyboardTeleop(keyboard_config)

robot.connect()
leader_arm.connect()
keyboard.connect()

_init_rerun(session_name="sourccey_v3beta_teleop")

if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
    raise ValueError("Robot, leader arm of keyboard is not connected!")

print("Teleoperating Sourccey V3 Beta")
while True:
    t0 = time.perf_counter()

    observation = robot.get_observation()

    arm_action = leader_arm.get_action()

    keyboard_keys = keyboard.get_action()
    base_action = robot._from_keyboard_to_base_action(keyboard_keys)

    log_rerun_data(observation, {**arm_action, **base_action})

    action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

    robot.send_action(action)

    busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
