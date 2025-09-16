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

    # Debug: Show what we're sending
    if base_action["x.vel"] != 0.0 or base_action["y.vel"] != 0.0 or base_action["theta.vel"] != 0.0:
        print(f"DEBUG TELEOP: Arm action keys: {list(arm_action.keys())}")
        print(f"DEBUG TELEOP: Arm action sample values: {dict(list(arm_action.items())[:3])}")
        print(f"DEBUG TELEOP: Keyboard keys: {keyboard_keys}")
        print(f"DEBUG TELEOP: Base action: {base_action}")
        print(f"DEBUG TELEOP: Action value types: {[(k, type(v).__name__) for k, v in list({**arm_action, **base_action}.items())[:5]]}")

    log_rerun_data(observation, {**arm_action, **base_action})

    action = {**arm_action, **base_action}

    robot.send_action(action)

    busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
