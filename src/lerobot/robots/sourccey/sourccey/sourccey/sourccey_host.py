# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 Vulcan Robotics, Inc. All rights reserved.
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

import logging
import time

import zmq

from .config_sourccey import SourcceyConfig, SourcceyHostConfig
from .sourccey import Sourccey

# Import protobuf modules
from ..protobuf.generated import sourccey_pb2

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
    robot_config = SourcceyConfig(id="sourccey")
    robot = Sourccey(robot_config)

    logging.info("Connecting Sourccey")
    robot.connect()

    logging.info("Starting Host")
    host_config = SourcceyHostConfig()
    host = SourcceyHost(host_config)

    print("Waiting for commands...")

    last_cmd_time = time.time()
    watchdog_active = False
    untorque_left_prev = False
    untorque_right_prev = False

    try:
        # Business logic
        start = time.perf_counter()
        duration = 0

        observation = None
        previous_observation = None
        while duration < host.connection_time_s:
            loop_start_time = time.time()
            try:
                # Receive protobuf message instead of JSON
                msg_bytes = host.zmq_cmd_socket.recv(zmq.NOBLOCK)
                print(f"HOST: Received message from client ({len(msg_bytes)} bytes)")

                # Convert protobuf to action dictionary using existing method
                robot_action = sourccey_pb2.SourcceyRobotAction()
                robot_action.ParseFromString(msg_bytes)

                # Note: per-arm untorque flags are handled after conversion below

                data = robot.protobuf_converter.protobuf_to_action(robot_action)

                # If per-arm untorque flags are set, disable torque and block those arm's positions
                try:
                    left_flag = bool(data.get("untorque_left", False))
                    right_flag = bool(data.get("untorque_right", False))
                    print(f"HOST: Flags received - untorque_left={left_flag} untorque_right={right_flag}")

                    # Left arm handling
                    if left_flag:
                        if not untorque_left_prev:
                            print("HOST: Left falling->rising: disabling left torque")
                        print("HOST: untorque_left=True -> stripping left_* positions")
                        robot.left_arm.bus.disable_torque()
                        data = {k: v for k, v in data.items() if not k.startswith("left_")}
                    elif untorque_left_prev and not left_flag:
                        print("HOST: Left rising->falling: enabling left torque")
                        robot.left_arm.bus.enable_torque()

                    # Right arm handling
                    if right_flag:
                        if not untorque_right_prev:
                            print("HOST: Right falling->rising: disabling right torque")
                        print("HOST: untorque_right=True -> stripping right_* positions")
                        robot.right_arm.bus.disable_torque()
                        data = {k: v for k, v in data.items() if not k.startswith("right_")}
                    elif untorque_right_prev and not right_flag:
                        print("HOST: Right rising->falling: enabling right torque")
                        robot.right_arm.bus.enable_torque()

                    untorque_left_prev = left_flag
                    untorque_right_prev = right_flag

                    print(f"HOST: Action after strip contains keys: {list(data.keys())}")
                except Exception as e:
                    print(f"HOST: Error applying per-arm untorque flags: {e}")

                # Send action to robot
                _action_sent = robot.send_action(data)

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

            # Send the observation to the remote agent
            try:
                # Don't send an empty observation
                if observation is None or observation == {}:
                    observation = previous_observation
                    logging.warning("No observation received. Sending previous observation.")

                if observation is not None and observation != {}:
                    # Convert observation to protobuf using existing method
                    robot_state = robot.protobuf_converter.observation_to_protobuf(observation)

                    # Send protobuf message instead of JSON
                    host.zmq_observation_socket.send(robot_state.SerializeToString(), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.info("Dropping observation, no client connected")
            except Exception as e:
                logging.error(f"Failed to send observation: {e}")

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
