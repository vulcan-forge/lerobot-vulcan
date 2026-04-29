#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Run policy rollout on Sourccey over remote ZMQ (no recording).

This is a Sourccey-focused helper around the rollout engine's base strategy.
It mirrors the LeKiwi remote rollout pattern while exposing a small set of
Sourccey defaults that are easy to override from the CLI.

Example:
    uv run python src/lerobot/control/sourccey/sourccey/rollout.py \
      --remote_ip=192.168.1.212 \
      --model_path=outputs/train/xvla_s_sourccey-shirt-fold-c-001/checkpoints/1000000/pretrained_model \
      --task="Fold the shirt" \
      --duration=300 \
      --fps=30
"""

from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig, parser
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClientConfig
from lerobot.rollout import (
    BaseStrategy,
    BaseStrategyConfig,
    RTCInferenceConfig,
    RolloutConfig,
    SyncInferenceConfig,
    build_rollout_context,
)
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun


@dataclass
class SourcceyRolloutRTCConfig:
    execution_horizon: int = 10
    max_guidance_weight: float = 10.0
    queue_threshold: int = 30


@dataclass
class SourcceyRolloutConfig:
    id: str = "sourccey"
    remote_ip: str = "192.168.1.243"
    model_path: str = (
        "outputs/train/xvla_s_sourccey-shirt-fold-c-001/checkpoints/1000000/pretrained_model"
    )
    task: str = "Fold the shirt"
    fps: float = 30.0
    duration: float = 300.0
    inference_type: str = "sync"  # sync | rtc
    rtc: SourcceyRolloutRTCConfig = field(default_factory=SourcceyRolloutRTCConfig)
    display_data: bool = True
    display_ip: str | None = None
    display_port: int | None = None
    use_torch_compile: bool = False
    compile_warmup_inferences: int = 2
    device: str | None = None
    return_to_initial_position: bool = True


@parser.wrap()
def rollout(cfg: SourcceyRolloutConfig):
    init_logging()

    robot_config = SourcceyClientConfig(remote_ip=cfg.remote_ip, id=cfg.id)

    policy_config = PreTrainedConfig.from_pretrained(cfg.model_path)
    policy_config.pretrained_path = cfg.model_path

    inference_type = cfg.inference_type.strip().lower()
    if inference_type == "rtc":
        inference_config = RTCInferenceConfig(
            rtc=RTCConfig(
                execution_horizon=cfg.rtc.execution_horizon,
                max_guidance_weight=cfg.rtc.max_guidance_weight,
            ),
            queue_threshold=cfg.rtc.queue_threshold,
        )
    elif inference_type == "sync":
        inference_config = SyncInferenceConfig()
    else:
        raise ValueError(
            f"Invalid inference_type '{cfg.inference_type}'. Expected 'sync' or 'rtc'."
        )

    rollout_cfg = RolloutConfig(
        robot=robot_config,
        policy=policy_config,
        strategy=BaseStrategyConfig(),
        inference=inference_config,
        fps=cfg.fps,
        duration=cfg.duration,
        device=cfg.device,
        task=cfg.task,
        display_data=cfg.display_data,
        display_ip=cfg.display_ip,
        display_port=cfg.display_port,
        return_to_initial_position=cfg.return_to_initial_position,
        use_torch_compile=cfg.use_torch_compile,
        compile_warmup_inferences=cfg.compile_warmup_inferences,
    )

    if rollout_cfg.display_data:
        init_rerun(session_name="sourccey_rollout", ip=rollout_cfg.display_ip, port=rollout_cfg.display_port)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    ctx = build_rollout_context(rollout_cfg, signal_handler.shutdown_event)

    strategy = BaseStrategy(rollout_cfg.strategy)
    try:
        strategy.setup(ctx)
        strategy.run(ctx)
    finally:
        strategy.teardown(ctx)


def main():
    rollout()


if __name__ == "__main__":
    main()
