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

import argparse
import logging
import signal
import threading
import time
from datetime import datetime, timezone

import zmq

from .config_sourccey import SourcceyConfig, SourcceyHostConfig
from .sourccey import Sourccey

# Import protobuf modules
from ..protobuf.generated import sourccey_pb2

HOST_FPS_LOG_INTERVAL_S = 5.0
HOST_FIX_OBSERVATION_FPS = 15.0


def _parse_bool_arg(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value '{value}'. Use true/false, 1/0, yes/no, or on/off."
    )


def _load_host_config_from_cli() -> SourcceyHostConfig:
    parser = argparse.ArgumentParser(
        description="Run Sourccey host.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--enable_host_fps_fix",
        "--enable-host-fps-fix",
        dest="enable_host_fps_fix",
        type=_parse_bool_arg,
        default=None,
        help=(
            "Enable host FPS fix. false keeps legacy behavior, "
            "true decouples observation capture rate from publish/control loop rate."
        ),
    )
    args, _unknown = parser.parse_known_args()

    cfg = SourcceyHostConfig()
    if args.enable_host_fps_fix is not None:
        cfg.enable_host_fps_fix = args.enable_host_fps_fix
    return cfg

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
        self.enable_host_fps_fix = bool(config.enable_host_fps_fix)

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


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
    host_config = _load_host_config_from_cli()
    host = SourcceyHost(host_config)
    imu_reporter = _IMUReporter(host_config)
    imu_reporter.start()

    print(
        "Host mode: "
        + ("patched (enable_host_fps_fix=true)" if host.enable_host_fps_fix else "legacy (enable_host_fps_fix=false)")
    )
    print("Waiting for commands...")

    last_cmd_time = time.time()
    watchdog_active = False
    fps_window_start = time.monotonic()
    fps_window_loops = 0
    fps_window_fresh_captures = 0
    fps_window_publishes = 0
    fps_window_encodes = 0
    fps_window_cmd_parse_s = 0.0
    fps_window_send_action_s = 0.0
    fps_window_robot_update_s = 0.0
    fps_window_observation_timing_sums: dict[str, float] = {}
    fps_window_publish_send_s = 0.0
    fps_window_encode_s = 0.0
    fps_window_sleep_s = 0.0
    fps_window_cmd_count = 0
    last_observation_capture_t = 0.0
    latest_observation_wire_bytes: bytes | None = None

    try:
        # Business logic
        start = time.perf_counter()
        duration = 0

        observation = None
        previous_observation = None
        while duration < host.connection_time_s:
            loop_start_time = time.time()
            fps_window_loops += 1
            try:
                # Receive protobuf message instead of JSON
                cmd_parse_start = time.perf_counter()
                msg_bytes = host.zmq_cmd_socket.recv(zmq.NOBLOCK)

                # Convert protobuf to action dictionary using existing method
                robot_action = sourccey_pb2.SourcceyRobotAction()
                robot_action.ParseFromString(msg_bytes)

                data = robot.protobuf_converter.protobuf_to_action(robot_action)
                fps_window_cmd_parse_s += time.perf_counter() - cmd_parse_start
                fps_window_cmd_count += 1

                # Send action to robot
                send_action_start = time.perf_counter()
                _action_sent = robot.send_action(data)
                fps_window_send_action_s += time.perf_counter() - send_action_start

                # Update the robot
                robot_update_start = time.perf_counter()
                robot.update()
                fps_window_robot_update_s += time.perf_counter() - robot_update_start

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

            observation_timing: dict[str, float] = {}
            if host.enable_host_fps_fix:
                min_capture_dt_s = 1.0 / HOST_FIX_OBSERVATION_FPS
                now_mono = time.monotonic()
                should_capture = (now_mono - last_observation_capture_t) >= min_capture_dt_s
                if should_capture:
                    if observation is not None and observation != {}:
                        previous_observation = observation
                    observation = robot.get_observation(parallel_camera_reads=True, timing=observation_timing)
                    fps_window_fresh_captures += 1
                    last_observation_capture_t = time.monotonic()
                    latest_observation_wire_bytes = None
            else:
                if observation is not None and observation != {}:
                    previous_observation = observation
                observation = robot.get_observation(timing=observation_timing)
                fps_window_fresh_captures += 1

            for key, value in observation_timing.items():
                fps_window_observation_timing_sums[key] = fps_window_observation_timing_sums.get(key, 0.0) + value

            # Send the observation to the remote agent
            try:
                # Don't send an empty observation
                if observation is None or observation == {}:
                    observation = previous_observation
                    logging.warning("No observation received. Sending previous observation.")

                if observation is not None and observation != {}:
                    if host.enable_host_fps_fix:
                        if latest_observation_wire_bytes is None:
                            # Encode only when a fresh observation is captured; reuse cached bytes otherwise.
                            encode_start = time.perf_counter()
                            robot_state = robot.protobuf_converter.observation_to_protobuf(observation)
                            latest_observation_wire_bytes = robot_state.SerializeToString()
                            fps_window_encode_s += time.perf_counter() - encode_start
                            fps_window_encodes += 1
                        publish_send_start = time.perf_counter()
                        host.zmq_observation_socket.send(latest_observation_wire_bytes, flags=zmq.NOBLOCK)
                        fps_window_publish_send_s += time.perf_counter() - publish_send_start
                    else:
                        # Legacy behavior: re-encode observation every publish.
                        encode_start = time.perf_counter()
                        robot_state = robot.protobuf_converter.observation_to_protobuf(observation)
                        encoded_bytes = robot_state.SerializeToString()
                        fps_window_encode_s += time.perf_counter() - encode_start
                        publish_send_start = time.perf_counter()
                        host.zmq_observation_socket.send(encoded_bytes, flags=zmq.NOBLOCK)
                        fps_window_publish_send_s += time.perf_counter() - publish_send_start
                        fps_window_encodes += 1
                    fps_window_publishes += 1
            except zmq.Again:
                logging.info("Dropping observation, no client connected")
            except Exception as e:
                logging.error(f"Failed to send observation: {e}")

            # Ensure a short sleep to avoid overloading the CPU.
            elapsed = time.time() - loop_start_time

            requested_sleep_s = max(1 / host.max_loop_freq_hz - elapsed, 0)
            sleep_start = time.perf_counter()
            time.sleep(requested_sleep_s)
            fps_window_sleep_s += time.perf_counter() - sleep_start
            now_mono = time.monotonic()
            fps_window_elapsed = now_mono - fps_window_start
            if fps_window_elapsed >= HOST_FPS_LOG_INTERVAL_S:
                host_loop_fps = fps_window_loops / fps_window_elapsed
                host_capture_fps = fps_window_fresh_captures / fps_window_elapsed
                host_publish_fps = fps_window_publishes / fps_window_elapsed
                host_encode_fps = fps_window_encodes / fps_window_elapsed
                fresh_publish_ratio = 0.0
                if host_publish_fps > 0:
                    fresh_publish_ratio = (host_capture_fps / host_publish_fps) * 100.0

                obs_count = max(fps_window_fresh_captures, 1)
                loop_count = max(fps_window_loops, 1)
                cmd_count = max(fps_window_cmd_count, 1)
                publish_count = max(fps_window_publishes, 1)
                encode_count = max(fps_window_encodes, 1)
                avg_obs_total_ms = (fps_window_observation_timing_sums.get("observation_total_s", 0.0) / obs_count) * 1000.0
                avg_left_ms = (fps_window_observation_timing_sums.get("left_arm_s", 0.0) / obs_count) * 1000.0
                avg_right_ms = (fps_window_observation_timing_sums.get("right_arm_s", 0.0) / obs_count) * 1000.0
                avg_base_ms = (fps_window_observation_timing_sums.get("base_s", 0.0) / obs_count) * 1000.0
                avg_z_ms = (fps_window_observation_timing_sums.get("z_s", 0.0) / obs_count) * 1000.0
                avg_cameras_ms = (fps_window_observation_timing_sums.get("cameras_s", 0.0) / obs_count) * 1000.0
                avg_cmd_parse_ms = (fps_window_cmd_parse_s / cmd_count) * 1000.0
                avg_send_action_ms = (fps_window_send_action_s / cmd_count) * 1000.0
                avg_robot_update_ms = (fps_window_robot_update_s / cmd_count) * 1000.0
                avg_encode_ms = (fps_window_encode_s / encode_count) * 1000.0
                avg_publish_send_ms = (fps_window_publish_send_s / publish_count) * 1000.0
                avg_sleep_ms = (fps_window_sleep_s / loop_count) * 1000.0
                camera_keys = sorted(
                    key for key in fps_window_observation_timing_sums if key.startswith("camera_") and key.endswith("_s")
                )
                camera_avg_parts = []
                for key in camera_keys:
                    cam_name = key.removeprefix("camera_").removesuffix("_s")
                    cam_avg_ms = (fps_window_observation_timing_sums[key] / obs_count) * 1000.0
                    camera_avg_parts.append(f"{cam_name}={cam_avg_ms:.1f}")
                camera_avg_str = ", ".join(camera_avg_parts) if camera_avg_parts else "none"
                print(
                    "Host FPS: "
                    f"loop={host_loop_fps:.2f} Hz, "
                    f"fresh_capture={host_capture_fps:.2f} Hz, "
                    f"publish={host_publish_fps:.2f} Hz, "
                    f"encode={host_encode_fps:.2f} Hz "
                    f"(target={float(host.max_loop_freq_hz):.2f} Hz, window={fps_window_elapsed:.2f}s)"
                )
                print(
                    "Host Timing: "
                    f"fresh/publish={fresh_publish_ratio:.1f}%, "
                    f"cmd_parse={avg_cmd_parse_ms:.2f}ms, send_action={avg_send_action_ms:.2f}ms, "
                    f"update={avg_robot_update_ms:.2f}ms, obs_total={avg_obs_total_ms:.2f}ms "
                    f"[left={avg_left_ms:.2f}, right={avg_right_ms:.2f}, base={avg_base_ms:.2f}, "
                    f"z={avg_z_ms:.2f}, cameras={avg_cameras_ms:.2f}, per_cam={camera_avg_str}], "
                    f"encode={avg_encode_ms:.2f}ms, publish_send={avg_publish_send_ms:.2f}ms, "
                    f"sleep={avg_sleep_ms:.2f}ms"
                )
                fps_window_start = now_mono
                fps_window_loops = 0
                fps_window_fresh_captures = 0
                fps_window_publishes = 0
                fps_window_encodes = 0
                fps_window_cmd_parse_s = 0.0
                fps_window_send_action_s = 0.0
                fps_window_robot_update_s = 0.0
                fps_window_observation_timing_sums = {}
                fps_window_publish_send_s = 0.0
                fps_window_encode_s = 0.0
                fps_window_sleep_s = 0.0
                fps_window_cmd_count = 0
            duration = time.perf_counter() - start
        print("Cycle time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down Sourccey Host.")
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
