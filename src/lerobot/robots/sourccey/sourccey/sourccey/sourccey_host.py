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
import signal
import sys
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import zmq

from .config_sourccey import SourcceyConfig, SourcceyHostConfig
from .sourccey import Sourccey
from .modules.imu import IMUReporter
from .modules.relay_agent.bridge import RelayBridge
from .modules.relay_agent.config import RelayAgentConfig

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



class _RelayAgentProcessManager:
    def __init__(self, config: SourcceyHostConfig):
        self.config = config
        self._process: subprocess.Popen[Any] | None = None
        self._restart_count = 0
        self._next_restart_at = 0.0

    def start(self) -> None:
        if self._is_running():
            return
        self._spawn()

    def poll(self) -> None:
        if not self.config.relay_agent_autostart:
            return
        if self._is_running():
            return
        if self._process is None:
            return

        return_code = self._process.poll()
        if return_code is None:
            return

        logging.warning("Relay agent exited with code %s", return_code)
        self._process = None

        if not self.config.relay_agent_restart_on_exit:
            return
        if self._restart_count >= max(0, self.config.relay_agent_max_restarts):
            logging.warning("Relay agent restart budget exhausted.")
            return

        now = time.monotonic()
        if now < self._next_restart_at:
            return

        self._spawn()

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return

        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logging.warning("Relay agent did not exit on terminate; killing process.")
            process.kill()
            process.wait(timeout=3)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to stop relay agent cleanly: %s", exc)

    def _spawn(self) -> None:
        python_exec = self.config.relay_agent_python_executable or sys.executable
        command = [python_exec, "-m", self.config.relay_agent_module]
        try:
            self._process = subprocess.Popen(command)  # noqa: S603
            self._restart_count += 1
            self._next_restart_at = time.monotonic() + max(0.0, self.config.relay_agent_restart_backoff_s)
            logging.info("Started relay agent subprocess: %s", " ".join(command))
        except Exception as exc:  # noqa: BLE001
            self._process = None
            self._next_restart_at = time.monotonic() + max(0.0, self.config.relay_agent_restart_backoff_s)
            logging.warning("Failed to start relay agent subprocess: %s", exc)

    def _is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None


class _RelayAgentEmbeddedManager:
    def __init__(self, config: SourcceyHostConfig):
        self.config = config
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._thread_main,
            daemon=True,
            name="sourccey_relay_embedded",
        )
        self._thread.start()
        self._started = True

    def poll(self) -> None:
        if not self.config.relay_agent_autostart:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        if not self._started:
            return
        if not self.config.relay_agent_restart_on_exit:
            return
        self.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=3.0)

    def _thread_main(self) -> None:
        import asyncio

        def _utc_now() -> str:
            return datetime.now(timezone.utc).isoformat()

        def _redact_ws_url(ws_url: str) -> str:
            token_marker = "token="
            token_idx = ws_url.find(token_marker)
            if token_idx == -1:
                return ws_url
            token_start = token_idx + len(token_marker)
            token_end = ws_url.find("&", token_start)
            if token_end == -1:
                token_end = len(ws_url)
            visible = ws_url[token_start : min(token_start + 4, token_end)]
            return f"{ws_url[:token_start]}{visible}...redacted{ws_url[token_end:]}"

        try:
            cfg = RelayAgentConfig.from_env()
        except Exception as exc:  # noqa: BLE001
            if not self.config.relay_agent_silent_failures:
                logging.warning("Relay agent embedded start skipped: invalid relay env/config (%s)", exc)
            return

        if _is_localhost_ws_url(cfg.relay_ws_base_url):
            logging.warning(
                "[%s] relay_agent.warn using localhost relay base URL (%s). "
                "If relay runs on another host, set VULCAN_RELAY_WS_BASE_URL to that host/IP.",
                _utc_now(),
                cfg.relay_ws_base_url,
            )

        async def _runner() -> None:
            backoff_s = cfg.connect_retry_backoff_s
            max_backoff_s = max(cfg.connect_retry_backoff_s, cfg.connect_retry_max_backoff_s)
            mode = "full_bridge" if self.config.relay_agent_forward_observations else "commands_only"
            redacted_ws_url = _redact_ws_url(cfg.ws_url)

            while not self._stop_event.is_set():
                bridge = RelayBridge(
                    cfg,
                    forward_observations=self.config.relay_agent_forward_observations,
                    forward_commands=True,
                )
                try:
                    if not self.config.relay_agent_silent_failures:
                        logging.info(
                            "[%s] relay_agent.connecting mode=%s ws_url=%s",
                            _utc_now(),
                            mode,
                            redacted_ws_url,
                        )
                    await bridge.run_forever()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    if not self.config.relay_agent_silent_failures:
                        logging.warning(
                            "[%s] relay_agent.connect_failed retry_in_s=%.1f error_type=%s error=%r",
                            _utc_now(),
                            backoff_s,
                            type(exc).__name__,
                            exc,
                        )
                    if self._stop_event.is_set():
                        break
                    await asyncio.sleep(backoff_s)
                    backoff_s = min(max_backoff_s, backoff_s * 2.0)
                else:
                    if self._stop_event.is_set():
                        break
                    await asyncio.sleep(1.0)
                finally:
                    await bridge.close()

        try:
            asyncio.run(_runner())
        except Exception as exc:  # noqa: BLE001
            logging.warning("Embedded relay agent thread exited: %s", exc)


def _is_localhost_ws_url(ws_base_url: str) -> bool:
    try:
        host = (urlparse(ws_base_url).hostname or "").strip().lower()
    except Exception:
        return False
    return host in {"127.0.0.1", "localhost", "::1"}


def main():
    def _handle_termination_signal(signum, _frame):
        logging.info(f"Received signal {signum}. Shutting down Sourccey Host.")
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_termination_signal)

    _silence_camera_warnings_for_host()

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

    if host_config.relay_agent_embedded:
        relay_agent_manager: _RelayAgentEmbeddedManager | _RelayAgentProcessManager = (
            _RelayAgentEmbeddedManager(host_config)
        )
    else:
        relay_agent_manager = _RelayAgentProcessManager(host_config)

    if host_config.relay_agent_autostart:
        if host_config.relay_agent_embedded:
            mode = "full_bridge" if host_config.relay_agent_forward_observations else "commands_only"
            logging.info("Relay agent autostart enabled (embedded mode=%s).", mode)
        else:
            logging.info(
                "Relay agent autostart enabled (subprocess mode). "
                "Set VULCAN_RELAY_AGENT_AUTOSTART=false to run it manually."
            )
        relay_agent_manager.start()
    else:
        if host_config.relay_agent_embedded:
            logging.info("Relay agent autostart disabled.")
        else:
            logging.info("Relay agent autostart disabled. Run relay agent separately if needed.")

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
            relay_agent_manager.poll()
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
        relay_agent_manager.stop()
        imu_reporter.stop()
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
