import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.sourccey.sourccey.sourccey import Sourccey, SourcceyClientConfig, SourcceyClient
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

EPISODE_IDX = 0

robot_config = SourcceyClientConfig(remote_ip="192.168.1.237", id="sourccey")
robot = SourcceyClient(robot_config)

dataset = LeRobotDataset("local/sourccey-001__drive_test_1", episodes=[EPISODE_IDX])
actions = dataset.hf_dataset.select_columns("action")

robot.connect()

if not robot.is_connected:
    raise ValueError("Robot is not connected!")

log_say(f"Replaying episode {EPISODE_IDX}")
for idx in range(dataset.num_frames):
    t0 = time.perf_counter()

    action = {
        name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
    }

    robot.send_action(action)

    busy_wait(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))

robot.disconnect()
