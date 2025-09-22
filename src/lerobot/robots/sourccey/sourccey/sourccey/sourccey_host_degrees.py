#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import base64
import json
import logging
import time

import cv2
import zmq

from .config_sourccey import SourcceyConfig, SourcceyHostConfig
from .sourccey import Sourccey


class SourcceyHost:
    def __init__(self, config: SourcceyHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


def main():
    logging.info("Configuring Sourccey")
    robot_config = SourcceyConfig(id="sourccey", right_arm_use_degrees=True, left_arm_use_degrees=True)
    robot = Sourccey(robot_config)

    logging.info("Connecting Sourccey")
    robot.connect()

    logging.info("Starting HostAgent")
    host_config = SourcceyHostConfig()
    host = SourcceyHost(host_config)

    last_cmd_time = time.time()
    last_print_time = time.time()
    watchdog_active = False
    logging.info("Waiting for commands...")
    print("Waiting for commands...")
    try:
        # Business logic
        start = time.perf_counter()
        duration = 0

        observation = None
        previous_observation = None
        while duration < host.connection_time_s:
            loop_start_time = time.time()
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))

                # Filter out empty arm actions - only send commands for arms that have actions
                filtered_data = {}
                
                # Check for left arm actions
                left_actions = {k: v for k, v in data.items() if k.startswith("left_")}
                if left_actions:
                    filtered_data.update(left_actions)
                
                # Check for right arm actions  
                right_actions = {k: v for k, v in data.items() if k.startswith("right_")}
                if right_actions:
                    filtered_data.update(right_actions)
                
                # Always include base/wheel commands
                base_actions = {k: v for k, v in data.items() if k.endswith(".vel")}
                filtered_data.update(base_actions)
                
                # Only send action if we have something to send
                if filtered_data:
                    _action_sent = robot.send_action(filtered_data)
                    robot.update()

                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                if not watchdog_active:
                    # logging.warning("No command available")
                    pass
            except Exception as e:
                logging.error("Message fetching failed: %s", e)

            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.warning(
                    f"Command not received for more than {host.watchdog_timeout_ms} milliseconds. Stopping the base."
                )
                watchdog_active = True
                robot.stop_base()

            if observation is not None and observation != {}:
                previous_observation = observation
            observation = robot.get_observation()

            # Encode ndarrays to base64 strings
            for cam_key, _ in robot.cameras.items():
                if cam_key in observation:
                    ret, buffer = cv2.imencode(
                        ".jpg", observation[cam_key], [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                    )
                    if ret:
                        observation[cam_key] = base64.b64encode(buffer).decode("utf-8")
                    else:
                        observation[cam_key] = ""

            # Send the observation to the remote agent
            try:
                # Don't send an empty observation
                # Maybe we send the last observation instead? Nick - 7/29/2025
                if observation is None or observation == {}:
                    observation = previous_observation
                    logging.warning("No observation received. Sending previous observation.")
                host.zmq_observation_socket.send_string(json.dumps(observation), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.info("Dropping observation, no client connected")

            # Ensure a short sleep to avoid overloading the CPU.
            elapsed = time.time() - loop_start_time

            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
            duration = time.perf_counter() - start
        print("Cycle time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down Sourccey Host.")
        robot.disconnect()
        host.disconnect()

    logging.info("Finished Sourccey cleanly")


if __name__ == "__main__":
    main()
