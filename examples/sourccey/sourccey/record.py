from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.sourccey.sourccey.sourccey import Sourccey, SourcceyClientConfig, SourcceyClient
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.sourccey.sourccey.bi_sourccey_leader.bi_sourccey_leader import BiSourcceyLeader
from lerobot.teleoperators.sourccey.sourccey.bi_sourccey_leader.config_bi_sourccey_leader import BiSourcceyLeaderConfig
from lerobot.teleoperators.sourccey.sourccey.sourccey_leader.config_sourccey_leader import SourcceyLeaderConfig
from lerobot.teleoperators.sourccey.sourccey.sourccey_leader.sourccey_leader import SourcceyLeader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun
from lerobot.record import record_loop

NUM_EPISODES = 1
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 5
TASK_DESCRIPTION = "Drive around the room"

# Create the robot and teleoperator configurations
robot_config = SourcceyClientConfig(remote_ip="192.168.1.235", id="sourccey")
teleop_arm_config = BiSourcceyLeaderConfig(left_arm_port="COM8", right_arm_port="COM3", id="sourccey")
keyboard_config = KeyboardTeleopConfig(id="keyboard")

robot = SourcceyClient(robot_config)
leader_arm = BiSourcceyLeader(teleop_arm_config)
keyboard = KeyboardTeleop(keyboard_config)

robot.connect()
leader_arm.connect()
keyboard.connect()

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
from lerobot.constants import HF_LEROBOT_HOME

repo_id = "sourccey-001__drive_test_1"

dataset = LeRobotDataset.create(
    repo_id=repo_id,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    root=HF_LEROBOT_HOME / "local" / repo_id,
    use_videos=True,
    image_writer_threads=4,
)

_init_rerun(session_name="sourccey_record")

listener, events = init_keyboard_listener()

if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
    raise ValueError("Robot, leader arm of keyboard is not connected!")

recorded_episodes = 0
while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {recorded_episodes}")

    # Run the record loop
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        dataset=dataset,
        teleop=[leader_arm, keyboard],
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Logic for reset env
    if not events["stop_recording"] and (
        (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
    ):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=[leader_arm, keyboard],
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

    if events["rerecord_episode"]:
        log_say("Re-record episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    recorded_episodes += 1

# Upload to hub and clean up
# dataset.push_to_hub()

robot.disconnect()
leader_arm.disconnect()
keyboard.disconnect()
listener.stop()
