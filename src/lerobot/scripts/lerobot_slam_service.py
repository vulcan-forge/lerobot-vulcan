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

"""Dedicated SLAM runtime service (input: stereo+odom, output: pose+health)."""

import logging
import signal
import time
from dataclasses import dataclass
from threading import Event

import zmq

from lerobot.configs import parser
from lerobot.slam.orchestrator import SlamOrchestrator
from lerobot.slam.orbslam3_adapter import ORBSLAM3Adapter
from lerobot.slam.transport import (
    DEFAULT_SLAM_INPUT_ENDPOINT,
    DEFAULT_SLAM_OUTPUT_ENDPOINT,
    parse_slam_input_packet,
    serialize_slam_output,
)

logger = logging.getLogger(__name__)


@dataclass
class SlamServiceConfig:
    """Config for `lerobot-slam-service`."""

    input_endpoint: str = DEFAULT_SLAM_INPUT_ENDPOINT
    output_endpoint: str = DEFAULT_SLAM_OUTPUT_ENDPOINT
    backend: str = "orbslam3"
    target_hz: float = 15.0
    healthy_timeout_s: float = 0.75
    stale_timeout_s: float = 2.0
    map_save_enabled: bool = False
    publish_heartbeat_hz: float = 2.0


@parser.wrap()
def run(cfg: SlamServiceConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    shutdown = Event()

    def _handle_signal(signum, _frame):
        logger.info("Received signal %s, shutting down SLAM service", signum)
        shutdown.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(cfg.input_endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"")

    pub = context.socket(zmq.PUB)
    pub.bind(cfg.output_endpoint)

    adapter = ORBSLAM3Adapter()
    orchestrator = SlamOrchestrator(
        adapter=adapter,
        target_hz=cfg.target_hz,
        healthy_timeout_s=cfg.healthy_timeout_s,
        stale_timeout_s=cfg.stale_timeout_s,
        map_save_enabled=cfg.map_save_enabled,
    )
    orchestrator.start()

    heartbeat_period_s = 1.0 / max(cfg.publish_heartbeat_hz, 1e-3)
    last_publish = 0.0
    logger.info(
        "SLAM service started (input=%s, output=%s, backend=%s, target_hz=%.2f)",
        cfg.input_endpoint,
        cfg.output_endpoint,
        cfg.backend,
        cfg.target_hz,
    )

    try:
        while not shutdown.is_set():
            got_input = False
            while True:
                try:
                    payload = sub.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                slam_input = parse_slam_input_packet(payload)
                orchestrator.submit(slam_input)
                got_input = True

            now = time.perf_counter()
            if got_input or (now - last_publish) >= heartbeat_period_s:
                output = orchestrator.get_latest()
                pub.send(serialize_slam_output(output))
                last_publish = now
            time.sleep(0.001)
    finally:
        if cfg.map_save_enabled:
            artifact = orchestrator.save_map()
            if artifact:
                logger.info("Saved map artifact: %s", artifact)
        orchestrator.stop()
        sub.close()
        pub.close()
        context.term()
        logger.info("SLAM service stopped")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
