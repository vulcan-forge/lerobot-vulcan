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
import json
import socket
import signal
import threading
import time

import zmq

from .config_sourccey import SourcceyConfig, SourcceyHostConfig
from .modules.host import silence_camera_warnings_for_host
from .modules.imu import IMUReporter
from .modules.relay import poll_relay, start_relay, stop_relay
from .sourccey import Sourccey

from ..protobuf.generated import sourccey_pb2

DISCOVERY_MAGIC = "SOURCCEY_DISCOVER_V1"
DISCOVERY_READ_TIMEOUT_S = 0.25


class DiscoveryResponder:
    def __init__(self, config: SourcceyHostConfig):
        self.discovery_port = config.discovery_port
        self.payload = json.dumps(
            {
                "host": "",
                "robot_name": "Sourccey",
                "nickname": "sourccey",
                "robot_type": "sourccey",
                "port_zmq_cmd": config.port_zmq_cmd,
                "port_zmq_observations": config.port_zmq_observations,
            }
        ).encode("utf-8")
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", self.discovery_port))
        sock.settimeout(DISCOVERY_READ_TIMEOUT_S)
        self._socket = sock
        self._thread = threading.Thread(target=self._serve, name="sourccey-discovery", daemon=True)
        self._thread.start()
        logging.info("Discovery responder listening on UDP %s", self.discovery_port)

    def stop(self) -> None:
        self._shutdown.set()
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _serve(self) -> None:
        if self._socket is None:
            return

        while not self._shutdown.is_set():
            try:
                payload, address = self._socket.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                if not self._shutdown.is_set():
                    logging.exception("Discovery responder socket error")
                break

            if payload.decode("utf-8", errors="ignore").strip() != DISCOVERY_MAGIC:
                continue

            try:
                self._socket.sendto(self.payload, address)
            except OSError:
                if not self._shutdown.is_set():
                    logging.exception("Failed to reply to discovery request")


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
        self.discovery_responder = DiscoveryResponder(config)
        self.discovery_responder.start()

    def disconnect(self):
        self.discovery_responder.stop()
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


def main():
    def _handle_termination_signal(signum, _frame):
        logging.info(f"Received signal {signum}. Shutting down Sourccey Host.")
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_termination_signal)
    silence_camera_warnings_for_host()

    logging.info("Configuring Sourccey")
    robot_config = SourcceyConfig(id="sourccey")
    robot = Sourccey(robot_config)

    logging.info("Connecting Sourccey")
    robot.connect()

    logging.info("Starting Host")
    host_config = SourcceyHostConfig()
    host = SourcceyHost(host_config)
    imu_reporter = IMUReporter(host_config)
    imu_reporter.start()
    relay = start_relay(host_config)

    print("Waiting for commands...")

    last_cmd_time = time.time()
    watchdog_active = False

    try:
        # Business logic
        start = time.perf_counter()
        duration = 0

        observation = None
        previous_observation = None
        while duration < host.connection_time_s:
            loop_start_time = time.time()
            poll_relay(relay)
            try:
                # Receive protobuf message instead of JSON
                msg_bytes = host.zmq_cmd_socket.recv(zmq.NOBLOCK)

                # Convert protobuf to action dictionary using existing method
                robot_action = sourccey_pb2.SourcceyRobotAction()
                robot_action.ParseFromString(msg_bytes)

                data = robot.protobuf_converter.protobuf_to_action(robot_action)

                # Send action to robot
                _action_sent = robot.send_action(data)

                # Update the robot
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
                logging.debug(
                    f"Command not received for more than {host.watchdog_timeout_ms} milliseconds. "
                )
                watchdog_active = True

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
        stop_relay(relay)
        imu_reporter.stop()
        robot.disconnect()
        host.disconnect()

    logging.info("Finished Sourccey cleanly")


if __name__ == "__main__":
    main()
