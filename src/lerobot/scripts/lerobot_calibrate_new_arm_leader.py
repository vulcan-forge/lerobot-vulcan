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
Calibrate the seven-joint new_arm leader with its mixed Feetech motor models.

Example:

```shell
lerobot-calibrate-new-arm-leader \
    --port=COM26 \
    --id=leader7
```
"""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import draccus

from lerobot.teleoperators.new_arm_leader.config_new_arm_leader import NewBotLeaderTeleopConfig
from lerobot.teleoperators.new_arm_leader.new_arm_leader import NewBotLeader
from lerobot.utils.utils import init_logging


@dataclass
class NewArmLeaderCalibrateConfig:
    port: str
    id: str | None = "leader7"
    calibration_dir: Path | None = None
    use_degrees: bool = True
    reverse: bool = False


@draccus.wrap()
def calibrate_new_arm_leader(cfg: NewArmLeaderCalibrateConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    teleop_cfg = NewBotLeaderTeleopConfig(
        port=cfg.port,
        id=cfg.id,
        calibration_dir=cfg.calibration_dir,
        use_degrees=cfg.use_degrees,
    )
    teleop = NewBotLeader(teleop_cfg)

    teleop.connect(calibrate=False)
    try:
        teleop.calibrate()
    finally:
        teleop.disconnect()


def main() -> None:
    calibrate_new_arm_leader()


if __name__ == "__main__":
    main()
