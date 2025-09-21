import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClientConfig, SourcceyClient
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

EPISODE_IDX = 0

# Intialize the robot config and the robot
robot_config = SourcceyClientConfig(remote_ip="192.168.1.237", id="sourccey")
robot = SourcceyClient(robot_config)

# Fetch the dataset to replay
dataset = LeRobotDataset("local/sourccey-001__tape-cup10", episodes=[EPISODE_IDX])

# Filter dataset to only include frames from the specified episode since episodes are chunked in dataset V3.0
episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == EPISODE_IDX)
actions = episode_frames.select_columns("action")

# Connect to the robot
robot.connect()

if not robot.is_connected:
    raise ValueError("Robot is not connected!")

print("Starting replay loop...")
log_say(f"Replaying episode {EPISODE_IDX}")
for idx in range(len(episode_frames)):
    t0 = time.perf_counter()

    action = {
        name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
    }

    robot.send_action(action)

    busy_wait(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))

robot.disconnect()
