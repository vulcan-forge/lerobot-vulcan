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
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zmq
from lerobot.configs import parser
from lerobot.sensors.imu.types import IMUSample as HostIMUSample

from .config_sourccey import (
    SourcceyConfig,
    SourcceyHostConfig,
    sourccey_cameras_config,
    sourccey_slam_eye_only_cameras_config,
)
from .modules.slam import SlamInputPublisher, close_slam_pub_socket, create_slam_pub_socket
from .sourccey import Sourccey

# Import protobuf modules
from ..protobuf.generated import sourccey_pb2


class _HostPanoramaFusion:
    """Adds the calibrated front-eye panorama to normal host observations."""

    def __init__(self, config: SourcceyHostConfig):
        self._camera_key = config.fused_vision_camera_key
        self._interval_s = 0.0 if config.fused_vision_publish_fps <= 0 else 1.0 / config.fused_vision_publish_fps
        self._last_publish_ts = 0.0
        self._virt = None
        self._refiner = None
        self._cal = None
        self._fuse_eyes = None

        if not config.fused_vision_enabled:
            return

        # The existing panorama script owns the calibration format and image
        # fusion math. Reuse it directly so ZMQ and the former HTTP feed render
        # the exact same fused view.
        try:
            from scripts.sourccey_eye_panorama import (
                OverlapRefiner,
                fuse_eyes,
                load_calibration_or_default,
            )

            self._cal, calibrated = load_calibration_or_default(Path(config.fused_vision_calibration_dir))
            self._refiner = OverlapRefiner()
            self._fuse_eyes = fuse_eyes
            print(
                "[HOST] Fused camera stream enabled: "
                f"{self._camera_key} at {config.fused_vision_publish_fps:.1f} FPS "
                f"(calibrated={calibrated})"
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("Fused camera stream disabled: unable to initialize panorama fusion (%s)", exc)

    def add_to_observation(self, observation: dict) -> None:
        if self._fuse_eyes is None:
            return
        now = time.monotonic()
        if self._interval_s > 0 and now - self._last_publish_ts < self._interval_s:
            return

        left = observation.get("front_left")
        right = observation.get("front_right")
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return
        try:
            fused, self._virt = self._fuse_eyes(
                left,
                right,
                self._cal,
                self._virt,
                refine=True,
                refiner=self._refiner,
            )
            observation[self._camera_key] = fused
            self._last_publish_ts = now
        except Exception as exc:  # noqa: BLE001
            logging.warning("Skipping fused camera frame: %s", exc)


class SourcceyHost:
    def __init__(self, config: SourcceyHostConfig, *, imu_provider=None):
        self.config = config
        self.imu_provider = imu_provider
        self.panorama_fusion = _HostPanoramaFusion(config)
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        # Keep the legacy PUSH channel for Python clients while exposing the
        # same live observation packets to Unity's SUB socket.
        self.zmq_observation_broadcast_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_observation_broadcast_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_broadcast_socket.bind(
            f"tcp://*:{config.port_zmq_observations_broadcast}"
        )
        print(
            "[HOST] Unity camera stream: PUB on "
            f"tcp://*:{config.port_zmq_observations_broadcast}"
        )

        self.zmq_slam_input_socket = None
        self.slam_input_publisher = _build_host_slam_input_publisher(config)
        if config.slam_input_enabled:
            self.zmq_slam_input_socket = create_slam_pub_socket(self.zmq_context, config.slam_input_endpoint)
            # Keep this visible even when the host logger is configured above INFO.
            # The panorama service consumes this exact stereo stream on port 5560.
            print(
                "[HOST] Fused-vision stereo stream: PUB on "
                f"{config.slam_input_endpoint} "
                f"({config.slam_stereo_left_key}, {config.slam_stereo_right_key})"
            )
            logging.info(
                "Host SLAM publisher enabled: endpoint=%s left=%s right=%s extra=%s jpeg=%d publish_fps=%.2f resize=%sx%s",
                config.slam_input_endpoint,
                config.slam_stereo_left_key,
                config.slam_stereo_right_key,
                ["bottom"] if config.bottom_camera_enabled else [],
                config.slam_jpeg_quality,
                config.slam_publish_fps,
                config.slam_resize_width,
                config.slam_resize_height,
            )
        self.zmq_slam_obstacle_socket = None
        self.slam_obstacle_publisher = _build_host_slam_obstacle_publisher(config)
        if config.slam_obstacle_input_enabled:
            self.zmq_slam_obstacle_socket = create_slam_pub_socket(
                self.zmq_context,
                config.slam_obstacle_input_endpoint,
            )
            logging.info(
                "Host SLAM obstacle publisher enabled: endpoint=%s left=%s right=%s jpeg=%d publish_fps=%.2f resize=%sx%s",
                config.slam_obstacle_input_endpoint,
                config.slam_obstacle_stereo_left_key,
                config.slam_obstacle_stereo_right_key,
                config.slam_obstacle_jpeg_quality,
                config.slam_obstacle_publish_fps,
                config.slam_obstacle_resize_width,
                config.slam_obstacle_resize_height,
            )

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = (
            max(config.max_loop_freq_hz, config.slam_eye_loop_freq_hz)
            if config.slam_eye_only_mode
            else config.max_loop_freq_hz
        )

    def disconnect(self):
        close_slam_pub_socket(self.zmq_slam_input_socket)
        close_slam_pub_socket(self.zmq_slam_obstacle_socket)
        self.zmq_observation_broadcast_socket.close()
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()

    def publish_slam_input(self, observation: dict) -> None:
        frames = {
            key: value
            for key, value in observation.items()
            if isinstance(value, np.ndarray)
        }
        imu_samples = None if self.imu_provider is None else self.imu_provider.recent_samples()
        if self.slam_input_publisher is not None:
            self.slam_input_publisher.publish(
                socket=self.zmq_slam_input_socket,
                observation=observation,
                frames=frames,
                imu_samples=imu_samples,
            )
        if self.slam_obstacle_publisher is not None:
            self.slam_obstacle_publisher.publish(
                socket=self.zmq_slam_obstacle_socket,
                observation=observation,
                frames=frames,
                imu_samples=imu_samples,
            )

    def add_fused_camera(self, observation: dict) -> None:
        self.panorama_fusion.add_to_observation(observation)


def _build_slam_eye_v4l2_controls(config: SourcceyHostConfig) -> dict[str, int]:
    controls: dict[str, int] = {
        "power_line_frequency": int(config.slam_eye_power_line_frequency),
        "exposure_dynamic_framerate": int(bool(config.slam_eye_exposure_dynamic_framerate)),
    }
    if config.slam_eye_auto_exposure is not None:
        controls["auto_exposure"] = int(config.slam_eye_auto_exposure)
    if config.slam_eye_exposure_time_absolute is not None:
        controls["exposure_time_absolute"] = int(config.slam_eye_exposure_time_absolute)
    if config.slam_eye_gain is not None:
        controls["gain"] = int(config.slam_eye_gain)
    if config.slam_eye_sharpness is not None:
        controls["sharpness"] = int(config.slam_eye_sharpness)
    if config.slam_eye_backlight_compensation is not None:
        controls["backlight_compensation"] = int(config.slam_eye_backlight_compensation)
    return controls


def _configure_front_slam_eye_camera_devices(
    robot_config: SourcceyConfig,
    host_config: SourcceyHostConfig,
) -> None:
    controls = _build_slam_eye_v4l2_controls(host_config)
    for cam_key in ("front_left", "front_right"):
        cam_cfg = robot_config.cameras.get(cam_key)
        if cam_cfg is None:
            continue
        device_path = Path(str(cam_cfg.index_or_path))
        if not str(device_path).startswith("/dev/"):
            logging.warning("Skipping V4L2 tuning for %s: unsupported path %s", cam_key, device_path)
            continue

        commands: list[list[str]] = []
        if host_config.slam_eye_fourcc:
            commands.append(
                [
                    "v4l2-ctl",
                    "--device",
                    str(device_path),
                    (
                        "--set-fmt-video="
                        f"width={host_config.slam_eye_width},height={host_config.slam_eye_height},"
                        f"pixelformat={host_config.slam_eye_fourcc}"
                    ),
                ]
            )
        commands.append(
            [
                "v4l2-ctl",
                "--device",
                str(device_path),
                f"--set-parm={host_config.slam_eye_camera_fps}",
            ]
        )
        if controls:
            control_arg = ",".join(f"{key}={value}" for key, value in controls.items())
            commands.append(
                [
                    "v4l2-ctl",
                    "--device",
                    str(device_path),
                    f"--set-ctrl={control_arg}",
                ]
            )

        for command in commands:
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
            except FileNotFoundError:
                logging.warning("v4l2-ctl not found; skipping V4L2 tuning for %s", cam_key)
                return
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.strip() if exc.stderr else str(exc)
                logging.warning("Failed V4L2 tuning for %s with %s: %s", cam_key, command[-1], stderr)
                break
        else:
            logging.info(
                "Configured %s at %sx%s %s %s FPS with controls %s",
                cam_key,
                host_config.slam_eye_width,
                host_config.slam_eye_height,
                host_config.slam_eye_fourcc,
                host_config.slam_eye_camera_fps,
                controls,
            )


def _handle_command_watchdog_timeout(robot: Sourccey, watchdog_timeout_ms: int) -> None:
    logging.warning(
        "Command not received for more than %d milliseconds. Stopping base motion.",
        watchdog_timeout_ms,
    )
    robot.watchdog_stop_motion()
    # Flush the zero-motion target out to the hardware immediately so a lost client
    # cannot leave the base latched on its previous wheel command.
    update_fn = getattr(robot, "update", None)
    if callable(update_fn):
        update_fn()


def _build_host_slam_input_publisher(config: SourcceyHostConfig) -> SlamInputPublisher | None:
    if not config.slam_input_enabled:
        return None
    extra_camera_keys: tuple[str, ...] = ("bottom",) if config.bottom_camera_enabled else ()
    return SlamInputPublisher(
        source_prefix="sourccey_host",
        source_id="sourccey",
        stereo_left_key=config.slam_stereo_left_key,
        stereo_right_key=config.slam_stereo_right_key,
        jpeg_quality=config.slam_jpeg_quality,
        eye_only_mode=config.slam_publish_eye_only_mode,
        publish_fps=config.slam_publish_fps,
        resize_width=config.slam_resize_width,
        resize_height=config.slam_resize_height,
        extra_camera_keys=extra_camera_keys,
    )


def _build_host_slam_obstacle_publisher(config: SourcceyHostConfig) -> SlamInputPublisher | None:
    if not config.slam_obstacle_input_enabled:
        return None
    return SlamInputPublisher(
        source_prefix="sourccey_host_obstacle",
        source_id="sourccey",
        stereo_left_key=config.slam_obstacle_stereo_left_key,
        stereo_right_key=config.slam_obstacle_stereo_right_key,
        jpeg_quality=config.slam_obstacle_jpeg_quality,
        eye_only_mode=config.slam_obstacle_publish_eye_only_mode,
        publish_fps=config.slam_obstacle_publish_fps,
        resize_width=config.slam_obstacle_resize_width,
        resize_height=config.slam_obstacle_resize_height,
    )

class _IMUReporter:
    """Background IMU collector with optional periodic console logging."""

    def __init__(self, config: SourcceyHostConfig):
        self.config = config
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._imu = None
        self._lock = threading.Lock()
        self._samples: deque[HostIMUSample] = deque(
            maxlen=max(int(config.slam_imu_max_samples_per_packet), 1) * 4
        )

    def start(self) -> None:
        if not self.config.imu_print_enabled and not self.config.slam_imu_enabled:
            return
        if self.config.imu_print_enabled and self.config.imu_print_interval_s <= 0:
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
            sample_rate_hz=max(float(self.config.slam_imu_sample_rate_hz), 1.0),
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
        print(
            "IMU reporter started "
            f"(slam_imu_enabled={self.config.slam_imu_enabled} "
            f"sample_rate_hz={float(self.config.slam_imu_sample_rate_hz):.2f} "
            f"print_interval_s={float(self.config.imu_print_interval_s):.2f})"
        )

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

    def recent_samples(self) -> list[HostIMUSample]:
        if not self.config.slam_imu_enabled:
            return []
        with self._lock:
            if not self._samples:
                return []
            max_samples = max(int(self.config.slam_imu_max_samples_per_packet), 1)
            return list(self._samples)[-max_samples:]

    def _run(self) -> None:
        assert self._imu is not None
        interval_s = 1.0 / max(float(self.config.slam_imu_sample_rate_hz), 1.0)
        next_print_ts = time.monotonic()
        while not self._stop_event.is_set():
            try:
                sample = self._imu.read()
                if sample.valid:
                    with self._lock:
                        self._samples.append(sample)
                    if self.config.imu_print_enabled and time.monotonic() >= next_print_ts:
                        stamp = datetime.now(timezone.utc).isoformat()
                        print(
                            f"[{stamp}] IMU accel={tuple(round(v, 4) for v in sample.accel_m_s2)} "
                            f"gyro={tuple(round(v, 4) for v in sample.gyro_rad_s)} "
                            f"mag={tuple(round(v, 2) for v in sample.mag_uT)} "
                            f"temp_c={None if sample.temperature_c is None else round(sample.temperature_c, 2)}"
                        )
                        next_print_ts = (
                            time.monotonic() + max(float(self.config.imu_print_interval_s), 0.001)
                        )
                else:
                    logging.warning("IMU read invalid: %s", sample.error)
            except Exception as exc:  # noqa: BLE001
                logging.warning("IMU reporter read error: %s", exc)

            self._stop_event.wait(interval_s)


@parser.wrap()
def main(host_config: SourcceyHostConfig):
    def _handle_termination_signal(signum, _frame):
        logging.info(f"Received signal {signum}. Shutting down Sourccey Host.")
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_termination_signal)

    _silence_camera_warnings_for_host(host_config)

    logging.info("Configuring Sourccey")
    robot_config = SourcceyConfig(id="sourccey")
    if host_config.slam_three_camera_front_priority_mode:
        if not host_config.slam_eye_only_mode:
            raise ValueError("slam_three_camera_front_priority_mode requires --slam_eye_only_mode=true.")
        if not host_config.bottom_camera_enabled:
            raise ValueError("slam_three_camera_front_priority_mode requires --bottom_camera_enabled=true.")
        if host_config.slam_obstacle_input_enabled:
            raise ValueError(
                "slam_three_camera_front_priority_mode is incompatible with --slam_obstacle_input_enabled=true "
                "because it disables the wrist cameras."
            )
        robot_config.cameras = sourccey_cameras_config(
            front_fps=host_config.slam_eye_camera_fps,
            front_width=host_config.slam_eye_width,
            front_height=host_config.slam_eye_height,
            front_fourcc=host_config.slam_eye_fourcc,
            include_wrist=False,
            bottom_fps=host_config.slam_bottom_camera_fps,
            bottom_width=host_config.slam_bottom_width,
            bottom_height=host_config.slam_bottom_height,
            bottom_fourcc=host_config.slam_bottom_fourcc,
            include_bottom=True,
            bottom_path=host_config.bottom_camera_path,
        )
        logging.info(
            "Sourccey Host front-priority 3-camera SLAM mode enabled: front=%dx%d@%d bottom=%dx%d@%d host_loop=%dHz",
            host_config.slam_eye_width,
            host_config.slam_eye_height,
            host_config.slam_eye_camera_fps,
            host_config.slam_bottom_width,
            host_config.slam_bottom_height,
            host_config.slam_bottom_camera_fps,
            max(host_config.max_loop_freq_hz, host_config.slam_eye_loop_freq_hz),
)
    elif host_config.slam_eye_only_mode and not host_config.slam_obstacle_input_enabled:
        robot_config.cameras = sourccey_slam_eye_only_cameras_config(
            front_fps=host_config.slam_eye_camera_fps,
            front_width=host_config.slam_eye_width,
            front_height=host_config.slam_eye_height,
            front_fourcc=host_config.slam_eye_fourcc,
            include_wrist=host_config.slam_obstacle_input_enabled,
            include_bottom=host_config.bottom_camera_enabled,
            bottom_path=host_config.bottom_camera_path,
        )
        logging.info(
            "Sourccey Host eye-only SLAM mode enabled: front stereo%s at %d FPS, host loop target %d Hz",
            " + bottom" if host_config.bottom_camera_enabled else "",
            host_config.slam_eye_camera_fps,
            max(host_config.max_loop_freq_hz, host_config.slam_eye_loop_freq_hz),
        )
    elif host_config.slam_eye_only_mode and host_config.slam_obstacle_input_enabled:
        robot_config.cameras = sourccey_cameras_config(
            front_fps=host_config.slam_eye_camera_fps,
            front_width=host_config.slam_eye_width,
            front_height=host_config.slam_eye_height,
            front_fourcc=host_config.slam_eye_fourcc,
            wrist_fps=host_config.slam_eye_camera_fps,
            wrist_width=host_config.slam_eye_width,
            wrist_height=host_config.slam_eye_height,
            wrist_fourcc=host_config.slam_eye_fourcc,
            include_wrist=True,
            bottom_fps=host_config.slam_eye_camera_fps,
            bottom_width=host_config.slam_eye_width,
            bottom_height=host_config.slam_eye_height,
            bottom_fourcc=host_config.slam_eye_fourcc,
            include_bottom=host_config.bottom_camera_enabled,
            bottom_path=host_config.bottom_camera_path,
        )
        logging.info(
            "Sourccey Host eye-only SLAM mode requested with obstacle publishing enabled; "
            "using the legacy full camera config so both front and wrist feeds stay on the known-good path."
        )
    elif host_config.bottom_camera_enabled:
        robot_config.cameras = sourccey_cameras_config(
            include_bottom=True,
            bottom_path=host_config.bottom_camera_path,
        )
    if host_config.slam_eye_only_mode:
        _configure_front_slam_eye_camera_devices(robot_config, host_config)
    robot = Sourccey(robot_config)

    if host_config.arm_connect_on_startup:
        logging.warning(
            "Sourccey Host ignores arm_connect_on_startup=true and always starts in passive-arm mode. "
            "Follower arms will only connect later if a client sends arm joint commands."
        )
    logging.info(
        "Connecting Sourccey in passive host mode "
        "(connect_arms=false, arm_calibrate_on_connect=%s, arm_relax_on_startup=%s)",
        host_config.arm_calibrate_on_connect,
        host_config.arm_relax_on_startup,
    )
    robot.connect(
        calibrate=host_config.arm_calibrate_on_connect,
        connect_arms=False,
    )
    logging.info("Sourccey Host started without connecting follower arms.")

    logging.info("Starting Host")
    imu_reporter = _IMUReporter(host_config)
    imu_reporter.start()
    host = SourcceyHost(host_config, imu_provider=imu_reporter)

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
                try:
                    _handle_command_watchdog_timeout(robot, host.watchdog_timeout_ms)
                except Exception as e:
                    logging.error("Failed to stop robot motion on watchdog timeout: %s", e)
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
                    host.add_fused_camera(observation)
                    host.publish_slam_input(observation)

                    # Convert observation to protobuf using existing method
                    # Convert observation to protobuf using existing method
                    robot_state = robot.protobuf_converter.observation_to_protobuf(observation)

                    payload = robot_state.SerializeToString()

                    # The recorder's legacy PULL stream is optional. A missing
                    # legacy client must not prevent Unity's PUB stream from
                    # receiving the same camera-bearing observation.
                    try:
                        host.zmq_observation_socket.send(payload, flags=zmq.NOBLOCK)
                    except zmq.Again:
                        pass

                    host.zmq_observation_broadcast_socket.send(payload, flags=zmq.NOBLOCK)
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
        imu_reporter.stop()
        robot.disconnect()
        host.disconnect()

    logging.info("Finished Sourccey cleanly")


def _silence_camera_warnings_for_host(config: SourcceyHostConfig) -> None:
    """
    Host-mode ergonomics: camera disconnects are expected sometimes; don't spam WARNING logs.
    """
    # Silence our OpenCV camera wrapper warnings
    logging.getLogger("lerobot.cameras.opencv.camera_opencv").setLevel(logging.ERROR)
    # Silence Sourccey camera fallback warnings (black frame fallback)
    if not (config.slam_eye_only_mode and config.slam_eye_log_camera_warnings):
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
