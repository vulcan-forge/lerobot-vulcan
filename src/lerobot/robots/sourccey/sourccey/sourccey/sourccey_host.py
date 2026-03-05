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
import os
import signal
import time
from typing import Any

import zmq

from .arm_debug_capture import ArmDebugCapture, extract_arm_positions
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


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _calibration_to_dict(calibration: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {
        motor: {
            "id": int(cal.id),
            "drive_mode": int(cal.drive_mode),
            "homing_offset": int(cal.homing_offset),
            "range_min": int(cal.range_min),
            "range_max": int(cal.range_max),
        }
        for motor, cal in calibration.items()
    }


def _safe_bus_read(bus: Any, data_name: str, motor: str) -> float | None:
    try:
        return float(bus.read(data_name, motor, normalize=False))
    except Exception:
        return None


def _read_shoulder_lift_diagnostics(robot: Sourccey) -> dict[str, dict[str, float | None]]:
    def _read_arm(arm: Any) -> dict[str, float | None]:
        bus = arm.bus
        return {
            "goal_position_raw": _safe_bus_read(bus, "Goal_Position", "shoulder_lift"),
            "present_position_raw": _safe_bus_read(bus, "Present_Position", "shoulder_lift"),
            "present_current_raw": _safe_bus_read(bus, "Present_Current", "shoulder_lift"),
            "torque_enable_raw": _safe_bus_read(bus, "Torque_Enable", "shoulder_lift"),
        }

    return {
        "left_shoulder_lift": _read_arm(robot.left_arm),
        "right_shoulder_lift": _read_arm(robot.right_arm),
    }


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

    host_arm_debug = ArmDebugCapture(
        enabled=_env_flag("SOURCCEY_HOST_ARM_DEBUG_CAPTURE", True),
        duration_s=_env_float("SOURCCEY_HOST_ARM_DEBUG_DURATION_S", 5.0),
        motion_threshold=_env_float("SOURCCEY_HOST_ARM_DEBUG_MOTION_THRESHOLD", 1.0),
        label="host",
        capture_path=os.getenv("SOURCCEY_HOST_ARM_DEBUG_PATH"),
    )
    if host_arm_debug.path:
        logging.warning("Host arm debug capture enabled. Writing to %s", host_arm_debug.path)
    try:
        host_arm_debug.record_session(
            "host_calibration_snapshot",
            {
                "robot_id": robot.id,
                "left_calibration_file": str(robot.left_arm.calibration_fpath),
                "right_calibration_file": str(robot.right_arm.calibration_fpath),
                "left_file_calibration": _calibration_to_dict(robot.left_arm.calibration),
                "right_file_calibration": _calibration_to_dict(robot.right_arm.calibration),
                "left_motor_calibration": _calibration_to_dict(robot.left_arm.bus.read_calibration()),
                "right_motor_calibration": _calibration_to_dict(robot.right_arm.bus.read_calibration()),
            },
        )
    except Exception as e:
        host_arm_debug.record_session(
            "host_calibration_snapshot_error",
            {"error": str(e)},
        )

    print("Waiting for commands...")

    last_cmd_time = time.time()
    watchdog_active = False

    try:
        # Business logic
        start = time.perf_counter()
        duration = 0

        try:
            observation = robot.get_observation()
        except Exception:
            observation = {}
        previous_observation = observation
        while duration < host.connection_time_s:
            loop_start_time = time.time()
            try:
                # Receive protobuf message instead of JSON
                msg_bytes = host.zmq_cmd_socket.recv(zmq.NOBLOCK)

                # Convert protobuf to action dictionary using existing method
                robot_action = sourccey_pb2.SourcceyRobotAction()
                robot_action.ParseFromString(msg_bytes)

                data = robot.protobuf_converter.protobuf_to_action(robot_action)

                host_arm_debug.maybe_start(action=data, observation=previous_observation)

                received_arm_target = extract_arm_positions(data)
                previous_obs_arm = extract_arm_positions(previous_observation)
                common_keys = [k for k in received_arm_target if k in previous_obs_arm]
                max_abs_delta_target_vs_prev_obs = (
                    max(abs(received_arm_target[k] - previous_obs_arm[k]) for k in common_keys)
                    if common_keys
                    else None
                )
                host_arm_debug.record(
                    "host_received_action",
                    {
                        "received_arm_target": received_arm_target,
                        "previous_observed_arm_position": previous_obs_arm,
                        "max_abs_delta_target_vs_prev_obs": max_abs_delta_target_vs_prev_obs,
                        "received_base_target": {
                            "x.vel": float(data.get("x.vel", 0.0)),
                            "y.vel": float(data.get("y.vel", 0.0)),
                            "theta.vel": float(data.get("theta.vel", 0.0)),
                            "z.pos": float(data.get("z.pos", 0.0)),
                        },
                    },
                )

                # Send action to robot
                _action_sent = robot.send_action(data)
                host_arm_debug.record(
                    "host_sent_action",
                    {
                        "sent_arm_target": extract_arm_positions(_action_sent),
                        "sent_base_target": {
                            "x.vel": float(_action_sent.get("x.vel", 0.0)),
                            "y.vel": float(_action_sent.get("y.vel", 0.0)),
                            "theta.vel": float(_action_sent.get("theta.vel", 0.0)),
                            "z.pos": float(_action_sent.get("z.pos", 0.0)),
                        },
                    },
                )

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
                    "Stopping base and releasing arm torque."
                )
                watchdog_active = True
                robot.watchdog_stop_and_relax()

            if observation is not None and observation != {}:
                previous_observation = observation
            observation = robot.get_observation()
            shoulder_lift_diag = (
                _read_shoulder_lift_diagnostics(robot) if host_arm_debug.is_active else None
            )
            host_arm_debug.record(
                "host_observation",
                {
                    "observed_arm_position": extract_arm_positions(observation),
                    "shoulder_lift_diagnostics": shoulder_lift_diag,
                },
            )

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
        host_arm_debug.close()

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
