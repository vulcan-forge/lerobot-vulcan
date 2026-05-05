#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Calibrate the seven-joint new_arm follower with its mixed Feetech motor models.

Example:

```shell
lerobot-calibrate-new-arm-follower \
    --port=COM25 \
    --id=follower7
```
"""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import draccus

from lerobot.common.so_arm import SOJointConfig
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.utils.utils import init_logging


def make_new_arm_follower_joint_configs() -> dict[str, SOJointConfig]:
    return {
        "roll_1": SOJointConfig(id=1, model="sts3032", homing_position=2048),
        "pitch_1": SOJointConfig(id=2, model="sts3032", homing_position=1024),
        "roll_2": SOJointConfig(id=3, model="sts3250", homing_position=2048),
        "pitch_2": SOJointConfig(id=4, model="sts3250", homing_position=1024),
        "roll_3": SOJointConfig(id=5, model="sts3215", homing_position=2048),
        "pitch_3": SOJointConfig(id=6, model="sts3215", homing_position=1024),
        "gripper": SOJointConfig(
            id=7,
            model="sts3215",
            homing_position=3000,
            range_min=1800,
            range_max=3300,
            is_gripper=True,
        ),
    }


@dataclass
class NewArmFollowerCalibrateConfig:
    port: str
    id: str | None = "follower7"
    calibration_dir: Path | None = None
    disable_torque_on_disconnect: bool = True
    use_degrees: bool = True
    reverse: bool = False


@draccus.wrap()
def calibrate_new_arm_follower(cfg: NewArmFollowerCalibrateConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot_cfg = SOFollowerRobotConfig(
        port=cfg.port,
        id=cfg.id,
        calibration_dir=cfg.calibration_dir,
        disable_torque_on_disconnect=cfg.disable_torque_on_disconnect,
        use_degrees=cfg.use_degrees,
        reverse=cfg.reverse,
        motors=make_new_arm_follower_joint_configs(),
    )
    robot = SOFollower(robot_cfg)

    robot.connect(calibrate=False)
    try:
        robot.calibrate()
    finally:
        robot.disconnect()


def main() -> None:
    calibrate_new_arm_follower()


if __name__ == "__main__":
    main()
