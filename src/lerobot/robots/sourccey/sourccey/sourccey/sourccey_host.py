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
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

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


class _HostPerfReporter:
    """Collects lightweight host-loop timing stats and writes one fresh report per run."""

    def __init__(self, config: SourcceyHostConfig, stats_provider=None):
        self.enabled = config.perf_log_enabled
        self.path = Path(config.perf_log_path).expanduser()
        self.interval_s = max(float(config.perf_log_interval_s), 0.5)
        self.stats_provider = stats_provider
        self.start_monotonic = time.monotonic()
        self._last_flush_ts = self.start_monotonic

        self.loop_count = 0
        self.cmd_received = 0
        self.cmd_missed = 0
        self.observation_empty = 0
        self.observation_sent = 0
        self.observation_dropped_no_client = 0
        self.send_errors = 0

        self.loop_elapsed_total_s = 0.0
        self.loop_elapsed_max_s = 0.0
        self.robot_update_total_s = 0.0
        self.robot_update_max_s = 0.0
        self.get_observation_total_s = 0.0
        self.get_observation_max_s = 0.0
        self.protobuf_total_s = 0.0
        self.protobuf_max_s = 0.0
        self.send_total_s = 0.0
        self.send_max_s = 0.0

        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(self._render(status="starting"))

    def record_loop(
        self,
        *,
        loop_elapsed_s: float,
        cmd_received: bool,
        robot_update_s: float,
        get_observation_s: float,
        protobuf_s: float,
        send_s: float,
        observation_empty: bool,
        sent: bool,
        dropped_no_client: bool,
        send_error: bool,
    ) -> None:
        self.loop_count += 1
        self.loop_elapsed_total_s += loop_elapsed_s
        self.loop_elapsed_max_s = max(self.loop_elapsed_max_s, loop_elapsed_s)

        if cmd_received:
            self.cmd_received += 1
        else:
            self.cmd_missed += 1

        self.robot_update_total_s += robot_update_s
        self.robot_update_max_s = max(self.robot_update_max_s, robot_update_s)

        self.get_observation_total_s += get_observation_s
        self.get_observation_max_s = max(self.get_observation_max_s, get_observation_s)

        self.protobuf_total_s += protobuf_s
        self.protobuf_max_s = max(self.protobuf_max_s, protobuf_s)

        self.send_total_s += send_s
        self.send_max_s = max(self.send_max_s, send_s)

        if observation_empty:
            self.observation_empty += 1
        if sent:
            self.observation_sent += 1
        if dropped_no_client:
            self.observation_dropped_no_client += 1
        if send_error:
            self.send_errors += 1

        self._maybe_flush()

    def finalize(self, status: str) -> None:
        if not self.enabled:
            return
        self.path.write_text(self._render(status=status))

    def _maybe_flush(self) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last_flush_ts < self.interval_s:
            return
        self._last_flush_ts = now
        self.path.write_text(self._render(status="running"))

    def _render(self, *, status: str) -> str:
        uptime_s = time.monotonic() - self.start_monotonic
        loop_hz = self.loop_count / uptime_s if uptime_s > 0 else 0.0

        def avg_ms(total_s: float) -> float:
            return (total_s / self.loop_count * 1000.0) if self.loop_count else 0.0

        def max_ms(value_s: float) -> float:
            return value_s * 1000.0

        lines = [
            f"status: {status}",
            f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}",
            f"uptime_s: {uptime_s:.3f}",
            f"loop_count: {self.loop_count}",
            f"loop_rate_hz: {loop_hz:.2f}",
            f"cmd_received: {self.cmd_received}",
            f"cmd_missed: {self.cmd_missed}",
            f"observation_empty: {self.observation_empty}",
            f"observation_sent: {self.observation_sent}",
            f"observation_dropped_no_client: {self.observation_dropped_no_client}",
            f"send_errors: {self.send_errors}",
            f"loop_avg_ms: {avg_ms(self.loop_elapsed_total_s):.3f}",
            f"loop_max_ms: {max_ms(self.loop_elapsed_max_s):.3f}",
            f"robot_update_avg_ms: {avg_ms(self.robot_update_total_s):.3f}",
            f"robot_update_max_ms: {max_ms(self.robot_update_max_s):.3f}",
            f"get_observation_avg_ms: {avg_ms(self.get_observation_total_s):.3f}",
            f"get_observation_max_ms: {max_ms(self.get_observation_max_s):.3f}",
            f"protobuf_avg_ms: {avg_ms(self.protobuf_total_s):.3f}",
            f"protobuf_max_ms: {max_ms(self.protobuf_max_s):.3f}",
            f"send_avg_ms: {avg_ms(self.send_total_s):.3f}",
            f"send_max_ms: {max_ms(self.send_max_s):.3f}",
        ]
        if self.stats_provider is not None:
            try:
                extra_stats = self.stats_provider() or {}
            except Exception as exc:  # noqa: BLE001
                extra_stats = {"stats_provider_error": str(exc)}
            for key in sorted(extra_stats):
                value = extra_stats[key]
                if isinstance(value, float):
                    lines.append(f"{key}: {value:.3f}")
                else:
                    lines.append(f"{key}: {value}")
        return "\n".join(lines) + "\n"


class _IMUReporter:
    """Background logger that prints IMU telemetry at a fixed interval."""

    def __init__(self, config: SourcceyHostConfig):
        self.config = config
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._imu = None

    def start(self) -> None:
        if not self.config.imu_print_enabled:
            return
        if self.config.imu_print_interval_s <= 0:
            logging.warning("IMU reporter disabled: imu_print_interval_s must be > 0")
            return
        try:
            from lerobot.sensors.imu import AdafruitLSM6DSOXLIS3MDLIMU, IMUConfig
        except Exception as exc:  # noqa: BLE001
            logging.warning("IMU reporter unavailable (import failed): %s", exc)
            return

        imu_config = IMUConfig(
            bus_num=self.config.imu_bus_num,
            lsm6dsox_address=self.config.imu_lsm6dsox_address,
            lis3mdl_address=self.config.imu_lis3mdl_address,
        )
        self._imu = AdafruitLSM6DSOXLIS3MDLIMU(config=imu_config)
        try:
            self._imu.connect()
        except Exception as exc:  # noqa: BLE001
            logging.warning("IMU reporter disabled: failed to connect IMU (%s)", exc)
            self._imu = None
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="sourccey_imu_reporter")
        self._thread.start()
        print(f"IMU reporter started (interval={self.config.imu_print_interval_s:.2f}s)")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._imu is not None:
            try:
                self._imu.disconnect()
            except Exception:  # noqa: BLE001
                pass
        self._imu = None

    def _run(self) -> None:
        assert self._imu is not None
        interval_s = float(self.config.imu_print_interval_s)
        while not self._stop_event.is_set():
            try:
                sample = self._imu.read()
                if sample.valid:
                    stamp = datetime.now(timezone.utc).isoformat()
                    print(
                        f"[{stamp}] IMU accel={tuple(round(v, 4) for v in sample.accel_m_s2)} "
                        f"gyro={tuple(round(v, 4) for v in sample.gyro_rad_s)} "
                        f"mag={tuple(round(v, 2) for v in sample.mag_uT)} "
                        f"temp_c={None if sample.temperature_c is None else round(sample.temperature_c, 2)}"
                    )
                else:
                    logging.warning("IMU read invalid: %s", sample.error)
            except Exception as exc:  # noqa: BLE001
                logging.warning("IMU reporter read error: %s", exc)

            self._stop_event.wait(interval_s)


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
    imu_reporter = _IMUReporter(host_config)
    perf_reporter = _HostPerfReporter(host_config, stats_provider=robot.get_observation_perf_snapshot)
    imu_reporter.start()

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
            cmd_received = False
            robot_update_s = 0.0
            try:
                # Receive protobuf message instead of JSON
                msg_bytes = host.zmq_cmd_socket.recv(zmq.NOBLOCK)
                cmd_received = True

                # Convert protobuf to action dictionary using existing method
                robot_action = sourccey_pb2.SourcceyRobotAction()
                robot_action.ParseFromString(msg_bytes)

                data = robot.protobuf_converter.protobuf_to_action(robot_action)

                # Send action to robot
                _action_sent = robot.send_action(data)

                # Update the robot
                robot_update_start = time.perf_counter()
                robot.update()
                robot_update_s = time.perf_counter() - robot_update_start

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
            get_observation_start = time.perf_counter()
            observation = robot.get_observation()
            get_observation_s = time.perf_counter() - get_observation_start

            # Send the observation to the remote agent
            protobuf_s = 0.0
            send_s = 0.0
            observation_empty = False
            sent = False
            dropped_no_client = False
            send_error = False
            try:
                # Don't send an empty observation
                if observation is None or observation == {}:
                    observation = previous_observation
                    observation_empty = True
                    logging.warning("No observation received. Sending previous observation.")

                if observation is not None and observation != {}:
                    # Convert observation to protobuf using existing method
                    protobuf_start = time.perf_counter()
                    robot_state = robot.protobuf_converter.observation_to_protobuf(observation)
                    protobuf_s = time.perf_counter() - protobuf_start

                    # Send protobuf message instead of JSON
                    send_start = time.perf_counter()
                    host.zmq_observation_socket.send(robot_state.SerializeToString(), flags=zmq.NOBLOCK)
                    send_s = time.perf_counter() - send_start
                    sent = True
            except zmq.Again:
                logging.info("Dropping observation, no client connected")
                dropped_no_client = True
            except Exception as e:
                logging.error(f"Failed to send observation: {e}")
                send_error = True

            # Ensure a short sleep to avoid overloading the CPU.
            elapsed = time.time() - loop_start_time
            perf_reporter.record_loop(
                loop_elapsed_s=elapsed,
                cmd_received=cmd_received,
                robot_update_s=robot_update_s,
                get_observation_s=get_observation_s,
                protobuf_s=protobuf_s,
                send_s=send_s,
                observation_empty=observation_empty,
                sent=sent,
                dropped_no_client=dropped_no_client,
                send_error=send_error,
            )

            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
            duration = time.perf_counter() - start
        print("Cycle time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down Sourccey Host.")
        imu_reporter.stop()
        perf_reporter.finalize(status="stopped")
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
