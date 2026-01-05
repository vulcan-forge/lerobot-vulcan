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
import subprocess
import shutil

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

        # Text communication sockets
        self.zmq_text_in_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_text_in_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_text_in_socket.bind(f"tcp://*:{config.port_zmq_text_in}")

        self.zmq_text_out_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_text_out_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_text_out_socket.bind(f"tcp://*:{config.port_zmq_text_out}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

        # Text message callback (can be set externally)
        self.text_message_callback = None

        # Track a running TTS process so we can avoid stacking audio
        self._tts_process = None

    def set_text_message_callback(self, callback):
        """Set a callback function to handle incoming text messages.
        
        Args:
            callback: Function that takes a string message as argument
        """
        self.text_message_callback = callback

    def speak_text(self, message: str) -> bool:
        """Speak text on the host device (best-effort, non-blocking)."""
        msg = (message or "").strip()
        if not msg:
            return False

        # Prefer espeak-ng, fall back to espeak
        tts_cmd = shutil.which("espeak-ng") or shutil.which("espeak")
        if not tts_cmd:
            logging.warning("No TTS engine found (expected `espeak-ng` or `espeak`).")
            return False

        try:
            # Stop any previous speech to prevent overlaps
            if self._tts_process and self._tts_process.poll() is None:
                try:
                    self._tts_process.terminate()
                except Exception:
                    pass

            # High-pitched, clear, robot-y voice
            self._tts_process = subprocess.Popen(
                [
                    tts_cmd,
                    "-v", "en-us+f3",  # higher-pitched female variant
                    "-p", "75",        # pitch (50 = default, 70–80 = robot sweet spot)
                    "-s", "165",       # speed (slightly fast, responsive)
                    "-a", "190",       # amplitude (cuts through motors)
                    msg,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True

        except Exception as e:
            logging.error(f"Failed to speak text: {e}")
            return False

    def send_text(self, message: str) -> bool:
        """Send a text message to the client.
        
        Args:
            message: Text string to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            self.zmq_text_out_socket.send_string(message, flags=zmq.NOBLOCK)
            return True
        except zmq.Again:
            logging.debug("No client connected for text message")
            return False
        except Exception as e:
            logging.error(f"Failed to send text message: {e}")
            return False

    def _poll_text_messages(self):
        """Poll for incoming text messages and call the callback if set."""
        try:
            msg = self.zmq_text_in_socket.recv_string(zmq.NOBLOCK)
            # Always print received text messages
            print(f"[TEXT MESSAGE] {msg}")
            logging.info(f"Received text message: {msg}")

            # Default behavior: speak incoming text
            self.speak_text(msg)

            if self.text_message_callback:
                self.text_message_callback(msg)
        except zmq.Again:
            pass  # No message available
        except Exception as e:
            logging.error(f"Error receiving text message: {e}")

    def disconnect(self):
        self.zmq_text_out_socket.close()
        self.zmq_text_in_socket.close()
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


def main():
    _silence_camera_warnings_for_host()

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

            # Poll for text messages (non-blocking)
            host._poll_text_messages()

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


def _silence_camera_warnings_for_host() -> None:
    """
    Host-mode ergonomics: camera disconnects are expected sometimes; don't spam WARNING logs.
    """
    # Silence our OpenCV camera wrapper warnings
    logging.getLogger("lerobot.cameras.opencv.camera_opencv").setLevel(logging.ERROR)
    # Silence Sourccey camera fallback warnings (black frame fallback)
    logging.getLogger("lerobot.robots.sourccey.sourccey.sourccey.sourccey").setLevel(logging.ERROR)

    # Best-effort: silence OpenCV's own internal logging if available
    try:
        import cv2  # type: ignore

        # OpenCV 4.x often exposes cv2.utils.logging.setLogLevel
        if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging") and hasattr(cv2.utils.logging, "setLogLevel"):
            level = getattr(cv2.utils.logging, "LOG_LEVEL_ERROR", None)
            if level is None:
                level = getattr(cv2.utils.logging, "LOG_LEVEL_SILENT", None)
            if level is not None:
                cv2.utils.logging.setLogLevel(level)
            return

        # Some builds expose cv2.setLogLevel
        if hasattr(cv2, "setLogLevel"):
            level = getattr(cv2, "LOG_LEVEL_ERROR", None)
            if level is None:
                level = getattr(cv2, "LOG_LEVEL_SILENT", None)
            if level is not None:
                cv2.setLogLevel(level)
    except Exception:
        # Don't fail startup just because OpenCV logging APIs differ
        pass

if __name__ == "__main__":
    main()
