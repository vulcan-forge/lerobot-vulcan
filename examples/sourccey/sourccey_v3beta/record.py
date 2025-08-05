from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.sourccey.sourccey_v3beta.sourccey_v3beta import SourcceyV3Beta, SourcceyV3BetaClientConfig, SourcceyV3BetaClient
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.sourccey_v3beta.bi_sourccey_v3beta_leader.bi_sourccey_v3beta_leader import BiSourcceyV3BetaLeader
from lerobot.teleoperators.sourccey_v3beta.bi_sourccey_v3beta_leader.config_bi_sourccey_v3beta_leader import BiSourcceyV3BetaLeaderConfig
from lerobot.teleoperators.sourccey_v3beta.sourccey_v3beta_leader.config_sourccey_v3beta_leader import SourcceyV3BetaLeaderConfig
from lerobot.teleoperators.sourccey_v3beta.sourccey_v3beta_leader.sourccey_v3beta_leader import SourcceyV3BetaLeader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun
from lerobot.record import record_loop

NUM_EPISODES = 10
FPS = 30
EPISODE_TIME_SEC = 30
RESET_TIME_SEC = 5
TASK_DESCRIPTION = "Grab the tape and put it in the cup"

# Create the robot and teleoperator configurations
robot_config = SourcceyV3BetaClientConfig(remote_ip="192.168.1.219", id="sourccey_v3beta")
teleop_arm_config = BiSourcceyV3BetaLeaderConfig(left_arm_port="COM41", right_arm_port="COM42", id="bi_sourccey_v3beta_leader")
keyboard_config = KeyboardTeleopConfig()

robot = SourcceyV3BetaClient(robot_config)
leader_arm = BiSourcceyV3BetaLeader(teleop_arm_config)
keyboard = KeyboardTeleop(keyboard_config)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id="local/sourccey_v3beta-001__tape-a__set000__nickm",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# To connect you already should have this script running on Sourccey V2 Beta: `python -m lerobot.common.robots.sourccey_v2beta.sourccey_v2beta_host --robot.id=sourccey_v2beta`
robot.connect()
leader_arm.connect()
keyboard.connect()

_init_rerun(session_name="sourccey_v3beta_record")

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
