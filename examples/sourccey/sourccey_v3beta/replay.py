import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.sourccey.sourccey_v3beta.sourccey_v3beta import SourcceyV3Beta, SourcceyV3BetaClientConfig, SourcceyV3BetaClient
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

EPISODE_IDX = 0

robot_config = SourcceyV3BetaClientConfig(remote_ip="192.168.1.219", id="sourccey_v3beta")
robot = SourcceyV3BetaClient(robot_config)

dataset = LeRobotDataset("local/sourccey_v3beta-001__towel-a__set000__nickm", episodes=[EPISODE_IDX])
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
