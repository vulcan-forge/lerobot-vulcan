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

import base64
import json
import logging
from functools import cached_property
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import zmq

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.robots.robot import Robot
from .arm_debug_capture import ArmDebugCapture, extract_arm_positions
from .config_sourccey import SourcceyClientConfig

# Import protobuf modules
from ..protobuf.generated import sourccey_pb2
from ..protobuf.sourccey_protobuf import SourcceyProtobuf

_LEFT_SHOULDER_LIFT_KEY = "left_shoulder_lift.pos"
_RIGHT_SHOULDER_LIFT_KEY = "right_shoulder_lift.pos"
_STARTUP_SHOULDER_KEYS = (_LEFT_SHOULDER_LIFT_KEY, _RIGHT_SHOULDER_LIFT_KEY)

class SourcceyClient(Robot):
    config_class = SourcceyClientConfig
    name = "sourccey_client"

    def __init__(self, config: SourcceyClientConfig):
        super().__init__(config)
        self.config = config
        self.id = config.id
        self.robot_type = config.type

        self.remote_ip = config.remote_ip
        self.port_zmq_cmd = config.port_zmq_cmd
        self.port_zmq_observations = config.port_zmq_observations

        self.teleop_keys = config.teleop_keys

        self.polling_timeout_ms = config.polling_timeout_ms
        self.connect_timeout_s = config.connect_timeout_s

        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None

        self.last_frames = {}
        self.last_remote_state = {}

        # Define three speed levels and a current index
        self.speed_levels = [
            {"x": 0.8,  "y": 0.8,  "z": 1.0, "theta": 0.8},   # slow
            {"x": 0.9, "y": 0.9, "z": 1.0, "theta": 0.9},  # medium
            {"x": 1.0,  "y": 1.0,  "z": 1.0, "theta": 1.0},   # fast
        ]
        self.speed_index = 1  # Start at medium speed (0.9)
        self.reverse = config.reverse

        self._is_connected = False
        self.logs = {}
        self._startup_run_t0: float | None = None
        self._startup_seam_warned: bool = False
        self._startup_seam_intervention_count: int = 0
        self._startup_seam_filter_active: bool = bool(config.startup_shoulder_seam_filter_enabled)
        self._startup_seam_plausible_streak: int = 0
        self._startup_seam_clean_streak: int = 0
        self._startup_seam_last_shoulders: dict[str, float] | None = None
        self._startup_seam_last_t: float | None = None
        self._startup_seam_release_reason: str | None = None
        self._startup_seam_release_logged: bool = False
        self._startup_action_filter_intervention_count: int = 0
        self._startup_last_sent_shoulders: dict[str, float] | None = None
        self._startup_last_sent_t: float | None = None

        # Initialize protobuf converter
        self.protobuf_converter = SourcceyProtobuf()

        # Per-arm untorque toggle state and key edge detection
        self.untorque_left_active = False
        self.untorque_right_active = False
        self._prev_keys: set[str] = set()

        # Time of last command
        self._last_cmd_t = time.monotonic()

        # Base movement smoothing
        self._slew_time_s_levels = [0.25, 0.25, 1.0]
        self._x_deadbane = 0.02

        # max change in x.vel per second (tune this)
        self._x_accel_levels = [7.0, 5.0, 3.0]   # units: (x.vel units) / s
        self._x_decel_levels = [7.0, 5.0, 3.0]   # allow faster slowing down than speeding up (optional)
        self._x_cmd_smoothed = 0.0

        # Z Position Control
        # You measured ~5s for z to go from +100 to -100 units (200-unit travel).
        self._z_min = -100.0
        self._z_max = 100.0
        self._z_full_travel_s = 1.0
        self._z_units_per_s = (self._z_max - self._z_min) / self._z_full_travel_s

        # Stored target position (what we "expect" z to be at while holding keys).
        self._z_pos_cmd = 0.0

        # One-shot arm debug capture for first few seconds of motion.
        self._arm_debug_capture = ArmDebugCapture(
            enabled=bool(config.debug_capture_enabled),
            duration_s=config.debug_capture_duration_s,
            motion_threshold=config.debug_capture_motion_threshold,
            label="client",
            capture_path=config.debug_capture_path,
        )
        if self._arm_debug_capture.path:
            logging.warning("Client arm debug capture enabled. Writing to %s", self._arm_debug_capture.path)

    ###################################################################
    # Properties and Attributes
    ###################################################################
    @cached_property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                "left_shoulder_pan.pos",
                "left_shoulder_lift.pos",
                "left_elbow_flex.pos",
                "left_wrist_flex.pos",
                "left_wrist_roll.pos",
                "left_gripper.pos",
                "right_shoulder_pan.pos",
                "right_shoulder_lift.pos",
                "right_elbow_flex.pos",
                "right_wrist_flex.pos",
                "right_wrist_roll.pos",
                "right_gripper.pos",
                "z.pos",
                "x.vel",
                "y.vel",
                "theta.vel",
            ),
            float,
        )

    @cached_property
    def _state_order(self) -> tuple[str, ...]:
        return tuple(self._state_ft.keys())

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        pass

    ###################################################################
    # Event Management
    ###################################################################
    def on_key_down(self, key_char: str) -> None:
        if key_char == self.teleop_keys["speed_up"]:
            self.speed_index = min(self.speed_index + 1, len(self.speed_levels) - 1)
            print(f"Speed index: {self.speed_index}")
        elif key_char == self.teleop_keys["speed_down"]:
            self.speed_index = max(self.speed_index - 1, 0)
            print(f"Speed index: {self.speed_index}")

    ###################################################################
    # Connection Management
    ###################################################################
    def connect(self) -> None:
        """Establishes ZMQ sockets with the remote mobile robot"""

        if self._is_connected:
            raise DeviceAlreadyConnectedError(
                "SourcceyClient is already connected. Do not run `robot.connect()` twice."
            )

        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PUSH)
        zmq_cmd_locator = f"tcp://{self.remote_ip}:{self.port_zmq_cmd}"
        self.zmq_cmd_socket.connect(zmq_cmd_locator)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PULL)
        zmq_observations_locator = f"tcp://{self.remote_ip}:{self.port_zmq_observations}"
        self.zmq_observation_socket.connect(zmq_observations_locator)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)

        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)
        socks = dict(poller.poll(self.connect_timeout_s * 1000))
        if self.zmq_observation_socket not in socks or socks[self.zmq_observation_socket] != zmq.POLLIN:
            raise DeviceNotConnectedError("Timeout waiting for Sourccey Host to connect expired.")

        self._is_connected = True
        self._startup_run_t0 = time.monotonic()
        self._startup_seam_warned = False
        self._startup_seam_intervention_count = 0
        self._startup_seam_filter_active = bool(self.config.startup_shoulder_seam_filter_enabled)
        self._startup_seam_plausible_streak = 0
        self._startup_seam_clean_streak = 0
        self._startup_seam_last_shoulders = None
        self._startup_seam_last_t = None
        self._startup_seam_release_reason = None
        self._startup_seam_release_logged = False
        self._startup_action_filter_intervention_count = 0
        self._startup_last_sent_shoulders = None
        self._startup_last_sent_t = None

    def calibrate(self) -> None:
        pass

    def configure(self):
        pass

    def _send_relax_command(self) -> None:
        """
        Best-effort final command before disconnect: stop base and untorque both arms.
        """
        relax_action = {
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0,
            "z.pos": float(self._z_pos_cmd),
            "untorque_left": True,
            "untorque_right": True,
        }
        robot_action = self.protobuf_converter.action_to_protobuf(relax_action)
        self.zmq_cmd_socket.send(robot_action.SerializeToString(), flags=zmq.NOBLOCK)

    def disconnect(self):
        """Cleans ZMQ comms"""

        if not self._is_connected:
            raise DeviceNotConnectedError(
                "SourcceyClient is not connected. You need to run `robot.connect()` before disconnecting."
            )
        try:
            self._send_relax_command()
        except zmq.Again:
            logging.debug("Could not send final relax command before disconnect: socket not ready.")
        except Exception as e:
            logging.debug(f"Could not send final relax command before disconnect: {e}")
        self._arm_debug_capture.record_session(
            "startup_shoulder_seam_filter_summary",
            {
                "enabled": bool(self.config.startup_shoulder_seam_filter_enabled),
                "duration_s": float(self.config.startup_shoulder_seam_filter_duration_s),
                "abs_threshold": float(self.config.startup_shoulder_seam_abs_threshold),
                "intervention_count": int(self._startup_seam_intervention_count),
                "required_plausible_frames": int(self.config.startup_shoulder_seam_required_plausible_frames),
                "plausible_streak_at_disconnect": int(self._startup_seam_plausible_streak),
                "clean_streak_at_disconnect": int(self._startup_seam_clean_streak),
                "filter_active_at_disconnect": bool(self._startup_seam_filter_active),
                "release_reason": self._startup_seam_release_reason,
            },
        )
        self._arm_debug_capture.record_session(
            "startup_shoulder_action_filter_summary",
            {
                "enabled": bool(self.config.startup_shoulder_seam_filter_enabled),
                "intervention_count": int(self._startup_action_filter_intervention_count),
                "last_sent_shoulders": self._startup_last_sent_shoulders,
            },
        )
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()
        self._is_connected = False
        self._arm_debug_capture.close()

    ###################################################################
    # Data Management
    ###################################################################
    def get_observation(self) -> dict[str, Any]:
        """
        Capture observations from the remote robot: current follower arm positions,
        present wheel speeds (converted to body-frame velocities: x, y, theta),
        and a camera frame. Receives over ZMQ, translate to body-frame vel
        """
        if not self._is_connected:
            raise DeviceNotConnectedError("SourcceyClient is not connected. You need to run `robot.connect()`.")

        frames, obs_dict = self._get_data()

        # Loop over each configured camera
        for cam_name, frame in frames.items():
            if frame is None:
                logging.warning("Frame is None")
                frame = np.zeros((640, 480, 3), dtype=np.uint8)
            obs_dict[cam_name] = frame

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command sourccey to move to a target joint configuration. Translates to motor space + sends over ZMQ

        Args:
            action (np.ndarray): array containing the goal positions for the motors.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            np.ndarray: the action sent to the motors, potentially clipped.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        requested_action = dict(action)
        action_to_send = self._apply_startup_shoulder_action_canonicalization(requested_action)
        requested_arm = extract_arm_positions(requested_action)
        outgoing_arm = extract_arm_positions(action_to_send)
        startup_override_keys = [
            key
            for key in _STARTUP_SHOULDER_KEYS
            if abs(float(outgoing_arm.get(key, 0.0)) - float(requested_arm.get(key, 0.0))) > 1e-6
        ]

        self._arm_debug_capture.maybe_start(action=action_to_send, observation=self.last_remote_state)
        self._arm_debug_capture.record(
            "client_outgoing_action",
            {
                "requested_arm_target": requested_arm,
                "outgoing_arm_target": outgoing_arm,
                "latest_remote_arm_observation": extract_arm_positions(self.last_remote_state),
                "startup_shoulder_filter_active": bool(self._startup_seam_filter_active),
                "startup_shoulder_action_filter_override_keys": startup_override_keys,
                "outgoing_base": {
                    "x.vel": float(action_to_send.get("x.vel", 0.0)),
                    "y.vel": float(action_to_send.get("y.vel", 0.0)),
                    "theta.vel": float(action_to_send.get("theta.vel", 0.0)),
                    "z.pos": float(action_to_send.get("z.pos", 0.0)),
                },
            },
        )

        # Convert action to protobuf and send
        robot_action = self.protobuf_converter.action_to_protobuf(action_to_send)
        self.zmq_cmd_socket.send(robot_action.SerializeToString())

        # TODO(Steven): Remove the np conversion when it is possible to record a non-numpy array value
        actions = np.array([action_to_send.get(k, 0.0) for k in self._state_order], dtype=np.float32)

        action_sent = {key: actions[i] for i, key in enumerate(self._state_order)}
        action_sent["action"] = actions
        return action_sent

    def record_policy_pipeline_debug(
        self,
        *,
        raw_policy_action: Any,
        policy_robot_action: dict[str, Any] | None,
        robot_action_to_send: dict[str, Any] | None,
        observation: dict[str, Any] | None,
    ) -> None:
        """Record policy pipeline stages to identify where startup commands diverge."""
        if not self._is_connected:
            return

        policy_robot_action = policy_robot_action or {}
        robot_action_to_send = robot_action_to_send or {}

        self._arm_debug_capture.maybe_start(action=robot_action_to_send, observation=self.last_remote_state)
        self._arm_debug_capture.record(
            "client_policy_pipeline",
            {
                "raw_policy_action_summary": self._summarize_action_like(raw_policy_action),
                "policy_robot_arm_target": extract_arm_positions(policy_robot_action),
                "processor_outgoing_arm_target": extract_arm_positions(robot_action_to_send),
                "latest_remote_arm_observation": extract_arm_positions(self.last_remote_state),
                "loop_observation_arm": extract_arm_positions(observation),
                "max_abs_delta_policy_to_processor": self._max_abs_delta_between_actions(
                    policy_robot_action, robot_action_to_send
                ),
            },
        )

    ###################################################################
    # Private Data Management
    ###################################################################
    def _get_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Dict[str, Any]]:
        """
        Polls the video socket for the latest observation data.

        Attempts to retrieve and decode the latest message within a short timeout.
        If successful, updates and returns the new frames, speed, and arm state.
        If no new data arrives or decoding fails, returns the last known values.
        """

        # 1. Get the latest message bytes from the socket
        latest_message_bytes = self._poll_and_get_latest_message()

        # 2. If no message, return cached data
        if latest_message_bytes is None:
            return self.last_frames, self.last_remote_state

        # 3. Parse the protobuf message
        try:
            robot_state = sourccey_pb2.SourcceyRobotState()
            robot_state.ParseFromString(latest_message_bytes)
            observation = self.protobuf_converter.protobuf_to_observation(robot_state)
        except Exception as e:
            logging.error(f"Error parsing protobuf observation: {e}")
            return self.last_frames, self.last_remote_state

        # 4. If protobuf parsing failed, return cached data
        if observation is None:
            return self.last_frames, self.last_remote_state

        # 5. Process the valid observation data
        try:
            new_frames, new_state = self._remote_state_from_obs(observation)
        except Exception as e:
            logging.error(f"Error processing observation data, serving last observation: {e}")
            return self.last_frames, self.last_remote_state

        self.last_frames = new_frames
        self.last_remote_state = new_state

        return new_frames, new_state

    ###################################################################
    # Private Message and Parsing Functions
    ###################################################################
    def _poll_and_get_latest_message(self) -> Optional[bytes]:
        """Polls the ZMQ socket for a limited time and returns the latest message bytes."""
        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)

        try:
            socks = dict(poller.poll(self.polling_timeout_ms))
        except zmq.ZMQError as e:
            logging.error(f"ZMQ polling error: {e}")
            return None

        if self.zmq_observation_socket not in socks:
            logging.info("No new data available within timeout.")
            return None

        last_msg = None
        while True:
            try:
                msg = self.zmq_observation_socket.recv(zmq.NOBLOCK)
                last_msg = msg
            except zmq.Again:
                break

        if last_msg is None:
            logging.warning("Poller indicated data, but failed to retrieve message.")

        return last_msg

    def _parse_observation_json(self, obs_string: str) -> Optional[Dict[str, Any]]:
        """Parses the JSON observation string."""
        try:
            return json.loads(obs_string)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON observation: {e}")
            return None

    def _decode_image_from_b64(self, image_b64: str) -> Optional[np.ndarray]:
        """Decodes a base64 encoded image string to an OpenCV image."""
        if not image_b64:
            return None
        try:
            jpg_data = base64.b64decode(image_b64)
            np_arr = np.frombuffer(jpg_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                logging.warning("cv2.imdecode returned None for an image.")
            return frame
        except (TypeError, ValueError) as e:
            logging.error(f"Error decoding base64 image data: {e}")
            return None

    def _remote_state_from_obs(
        self, observation: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Extracts frames, and state from the parsed observation."""
        flat_state = {key: observation.get(key, 0.0) for key in self._state_order}
        pre_filter_shoulder = {
            "left_shoulder_lift.pos": float(flat_state.get("left_shoulder_lift.pos", 0.0)),
            "right_shoulder_lift.pos": float(flat_state.get("right_shoulder_lift.pos", 0.0)),
        }

        seam_filtered_obs_keys: list[str] = []
        if self._should_apply_startup_shoulder_seam_filter():
            flat_state, seam_filtered_obs_keys, seam_filter_debug = self._apply_startup_shoulder_sign_canonicalization(
                flat_state
            )
            if seam_filtered_obs_keys and not self._startup_seam_warned:
                logging.warning(
                    "Startup shoulder seam filter active; canonicalized observation keys: %s",
                    ", ".join(seam_filtered_obs_keys),
                )
                self._startup_seam_warned = True
            if seam_filtered_obs_keys:
                self._startup_seam_intervention_count += 1
                post_filter_shoulder = {
                    "left_shoulder_lift.pos": float(flat_state.get("left_shoulder_lift.pos", 0.0)),
                    "right_shoulder_lift.pos": float(flat_state.get("right_shoulder_lift.pos", 0.0)),
                }
                startup_elapsed_s = None
                if self._startup_run_t0 is not None:
                    startup_elapsed_s = time.monotonic() - self._startup_run_t0

                self._arm_debug_capture.record_session(
                    "startup_shoulder_seam_filter_intervention",
                    {
                        "keys": seam_filtered_obs_keys,
                        "raw_shoulder_observation": pre_filter_shoulder,
                        "filtered_shoulder_observation": post_filter_shoulder,
                        "abs_threshold": float(self.config.startup_shoulder_seam_abs_threshold),
                        "startup_elapsed_s": startup_elapsed_s,
                        "intervention_index": int(self._startup_seam_intervention_count),
                        "sample_plausible": bool(seam_filter_debug.get("sample_plausible", True)),
                        "plausible_streak": int(seam_filter_debug.get("plausible_streak", 0)),
                        "clean_streak": int(seam_filter_debug.get("clean_streak", 0)),
                        "max_delta_allowed": seam_filter_debug.get("max_delta_allowed"),
                    },
                )

        state_vec = np.array([flat_state[key] for key in self._state_order], dtype=np.float32)

        obs_dict: Dict[str, Any] = {**flat_state, "observation.state": state_vec}

        # Decode images
        current_frames: Dict[str, np.ndarray] = {}
        for cam_name, image_data in observation.items():
            if cam_name not in self._cameras_ft:
                continue

            # Handle both numpy arrays (from protobuf) and base64 strings (legacy)
            if isinstance(image_data, np.ndarray):
                current_frames[cam_name] = image_data
            elif isinstance(image_data, str):
                frame = self._decode_image_from_b64(image_data)
                if frame is not None:
                    current_frames[cam_name] = frame

        return current_frames, obs_dict

    def _apply_startup_shoulder_action_canonicalization(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._should_apply_startup_shoulder_seam_filter():
            return action
        if not action:
            return action

        out = dict(action)
        threshold = float(self.config.startup_shoulder_seam_abs_threshold)
        now_t = time.monotonic()
        max_delta_allowed: float | None = None
        if self._startup_last_sent_t is not None:
            dt_s = max(1e-6, now_t - self._startup_last_sent_t)
            max_delta_allowed = (
                float(self.config.startup_shoulder_seam_max_delta_per_s) * dt_s
                + float(self.config.startup_shoulder_seam_delta_margin)
            )

        changed_sign_keys: list[str] = []
        slew_limited_keys: list[str] = []
        requested_shoulders: dict[str, float] = {}
        outgoing_shoulders: dict[str, float] = {}
        observed_shoulders = {
            key: float(self.last_remote_state.get(key, 0.0)) for key in _STARTUP_SHOULDER_KEYS
        }

        for key in _STARTUP_SHOULDER_KEYS:
            if key not in out:
                continue

            try:
                requested = float(out[key])
            except (TypeError, ValueError):
                continue

            requested_shoulders[key] = requested
            candidate = requested
            observed = float(observed_shoulders.get(key, requested))

            # Near seam, keep outgoing action on the sign branch closest to observed state.
            if max(abs(requested), abs(observed)) >= threshold:
                opposite = -requested
                if abs(opposite - observed) + 1e-6 < abs(candidate - observed):
                    candidate = opposite
                    changed_sign_keys.append(key)

            prev_sent = None
            if self._startup_last_sent_shoulders is not None:
                prev_sent = self._startup_last_sent_shoulders.get(key)

            # During startup seam filtering only, rate-limit sudden shoulder jumps.
            if prev_sent is not None and max_delta_allowed is not None:
                delta = candidate - float(prev_sent)
                if abs(delta) > max_delta_allowed:
                    candidate = float(prev_sent + np.sign(delta) * max_delta_allowed)
                    slew_limited_keys.append(key)

            bounded = float(min(100.0, max(-100.0, candidate)))
            out[key] = bounded
            outgoing_shoulders[key] = bounded

        if outgoing_shoulders:
            self._startup_last_sent_shoulders = outgoing_shoulders
            self._startup_last_sent_t = now_t

        if changed_sign_keys or slew_limited_keys:
            self._startup_action_filter_intervention_count += 1
            self._arm_debug_capture.record_session(
                "startup_shoulder_action_filter_intervention",
                {
                    "keys_sign_aligned": changed_sign_keys,
                    "keys_slew_limited": slew_limited_keys,
                    "requested_shoulder_action": requested_shoulders,
                    "outgoing_shoulder_action": outgoing_shoulders,
                    "observed_shoulder_state": observed_shoulders,
                    "max_delta_allowed": max_delta_allowed,
                    "intervention_index": int(self._startup_action_filter_intervention_count),
                },
            )

        return out

    def _should_apply_startup_shoulder_seam_filter(self) -> bool:
        if not bool(self.config.startup_shoulder_seam_filter_enabled):
            return False
        if not bool(self._startup_seam_filter_active):
            return False

        # Deterministic upper bound: always release the startup seam filter after
        # the configured startup window, even if continuity heuristics still see
        # seam-aliasing samples.
        duration_s = float(self.config.startup_shoulder_seam_filter_duration_s)
        if duration_s > 0.0 and self._startup_run_t0 is not None:
            startup_elapsed_s = time.monotonic() - self._startup_run_t0
            if startup_elapsed_s >= duration_s:
                self._release_startup_shoulder_seam_filter("duration_elapsed")
                return False

        return True

    def _apply_startup_shoulder_sign_canonicalization(
        self, state_like: Dict[str, Any]
    ) -> tuple[Dict[str, Any], list[str], dict[str, Any]]:
        threshold = float(self.config.startup_shoulder_seam_abs_threshold)
        out = dict(state_like)
        changed: list[str] = []
        debug_info: dict[str, Any] = {}

        raw_values: dict[str, float] = {}
        corrected_values: dict[str, float] = {}

        for key in _STARTUP_SHOULDER_KEYS:
            if key not in out:
                continue

            try:
                val = float(out[key])
            except (TypeError, ValueError):
                continue

            raw_values[key] = val
            corrected_values[key] = val
            if abs(val) < threshold:
                continue

            # Near seam, choose the sign branch nearest to previous filtered sample.
            if self._startup_seam_last_shoulders is None or key not in self._startup_seam_last_shoulders:
                continue
            prev = float(self._startup_seam_last_shoulders[key])
            opposite = -val
            if abs(opposite - prev) + 1e-6 < abs(val - prev):
                corrected_values[key] = opposite

        now_t = time.monotonic()
        max_delta_allowed: float | None = None
        sample_plausible = True
        last_shoulders = self._startup_seam_last_shoulders
        if (
            last_shoulders is not None
            and self._startup_seam_last_t is not None
            and raw_values
        ):
            dt_s = max(1e-6, now_t - self._startup_seam_last_t)
            max_delta_allowed = (
                float(self.config.startup_shoulder_seam_max_delta_per_s) * dt_s
                + float(self.config.startup_shoulder_seam_delta_margin)
            )
            debug_info["dt_s"] = dt_s

        for key, raw_val in raw_values.items():
            chosen = raw_val
            corrected = corrected_values.get(key, raw_val)

            if last_shoulders is None or max_delta_allowed is None:
                # No continuity baseline yet: take corrected seam sign if detected.
                if corrected != raw_val:
                    chosen = corrected
            else:
                prev_val = float(last_shoulders.get(key, raw_val))
                raw_jump = abs(raw_val - prev_val)
                corrected_jump = abs(corrected - prev_val)
                if corrected != raw_val and raw_jump > max_delta_allowed and corrected_jump <= max_delta_allowed:
                    chosen = corrected
                elif corrected != raw_val and raw_jump > max_delta_allowed and corrected_jump < raw_jump:
                    chosen = corrected

            out[key] = chosen
            if chosen != raw_val:
                changed.append(key)

            if last_shoulders is not None and max_delta_allowed is not None:
                prev_val = float(last_shoulders.get(key, chosen))
                if abs(chosen - prev_val) > max_delta_allowed:
                    sample_plausible = False

        if raw_values:
            if last_shoulders is None or max_delta_allowed is None:
                self._startup_seam_plausible_streak = 1
            elif sample_plausible:
                self._startup_seam_plausible_streak += 1
            else:
                self._startup_seam_plausible_streak = 0

            # Release only after stable plausible samples that require no seam correction.
            if sample_plausible and not changed:
                self._startup_seam_clean_streak += 1
            else:
                self._startup_seam_clean_streak = 0

            self._startup_seam_last_shoulders = {
                key: float(out.get(key, raw_values[key])) for key in raw_values
            }
            self._startup_seam_last_t = now_t

        required_plausible = max(1, int(self.config.startup_shoulder_seam_required_plausible_frames))
        if self._startup_seam_filter_active and self._startup_seam_clean_streak >= required_plausible:
            self._release_startup_shoulder_seam_filter("clean_plausible_streak_reached")

        debug_info["sample_plausible"] = sample_plausible
        debug_info["plausible_streak"] = int(self._startup_seam_plausible_streak)
        debug_info["clean_streak"] = int(self._startup_seam_clean_streak)
        debug_info["max_delta_allowed"] = max_delta_allowed
        debug_info["filter_active"] = bool(self._startup_seam_filter_active)
        return out, changed, debug_info

    def _release_startup_shoulder_seam_filter(self, reason: str) -> None:
        if not self._startup_seam_filter_active:
            return

        self._startup_seam_filter_active = False
        self._startup_seam_release_reason = reason
        if self._startup_seam_release_logged:
            return

        startup_elapsed_s = None
        if self._startup_run_t0 is not None:
            startup_elapsed_s = time.monotonic() - self._startup_run_t0

        self._arm_debug_capture.record_session(
            "startup_shoulder_seam_filter_released",
            {
                "reason": reason,
                "startup_elapsed_s": startup_elapsed_s,
                "plausible_streak": int(self._startup_seam_plausible_streak),
                "clean_streak": int(self._startup_seam_clean_streak),
                "required_plausible_frames": int(self.config.startup_shoulder_seam_required_plausible_frames),
            },
        )
        logging.info(
            "Startup shoulder seam filter released (%s). clean plausible frames=%d.",
            reason,
            self._startup_seam_clean_streak,
        )
        self._startup_seam_release_logged = True

    ###################################################################
    # Private Control Functions
    ###################################################################
    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray, z_obs_pos: float | None = None):
        reverse = self.reverse
        speed_setting = self.speed_levels[self.speed_index]
        x_speed = speed_setting["x"]
        y_speed = speed_setting["y"]
        z_speed = speed_setting["z"]
        theta_speed = speed_setting["theta"]

        pressed = set(pressed_keys)

        x_cmd = 0.0
        y_cmd = 0.0
        theta_cmd = 0.0
        x_cmd_target = 0.0

        if self.teleop_keys["forward"] in pressed:
            if reverse:
                x_cmd_target -= x_speed
            else:
                x_cmd_target += x_speed
        if self.teleop_keys["backward"] in pressed:
            if reverse:
                x_cmd_target += x_speed
            else:
                x_cmd_target -= x_speed
        if self.teleop_keys["left"] in pressed:
            if reverse:
                y_cmd -= y_speed
            else:
                y_cmd += y_speed
        if self.teleop_keys["right"] in pressed:
            if reverse:
                y_cmd += y_speed
            else:
                y_cmd -= y_speed
        # Z: integrate held keys into a stored position command (z.pos)
        z_dir = 0.0
        if self.teleop_keys["up"] in pressed:
            if reverse:
                z_dir -= z_speed
            else:
                z_dir += z_speed
        if self.teleop_keys["down"] in pressed:
            if reverse:
                z_dir += z_speed
            else:
                z_dir -= z_speed
        if self.teleop_keys["rotate_left"] in pressed:
            if reverse:
                theta_cmd -= theta_speed
            else:
                theta_cmd += theta_speed
        if self.teleop_keys["rotate_right"] in pressed:
            if reverse:
                theta_cmd += theta_speed
            else:
                theta_cmd -= theta_speed

        slew_time_s = self._slew_time_s_levels[self.speed_index]
        x_accel = self._x_accel_levels[self.speed_index]
        x_decel = self._x_decel_levels[self.speed_index]

        now = time.monotonic()
        dt = now - self._last_cmd_t
        self._last_cmd_t = now
        dt = max(0.0, min(dt, slew_time_s))  # cap big jumps if the loop stalls

        # Z position target integration should reflect the *actual* loop timing.
        # If we cap dt to 1/30 while the loop runs slower (e.g., due to camera/network load),
        # z.pos changes become tiny and it can take ~seconds before the actuator deadband is exceeded.
        # We already cap dt above with `slew_time_s`, so using dt here is safe and makes Z feel immediate.
        z_rate = float(self._z_units_per_s)
        self._z_pos_cmd = float(np.clip(self._z_pos_cmd + (z_dir * z_rate * dt), self._z_min, self._z_max))

        if z_obs_pos is not None and z_dir == 0.0:
            self._z_pos_cmd = float(z_obs_pos)

        if abs(self._x_cmd_smoothed) >= self._x_deadbane:
            # already moving -> smooth changes
            self._x_cmd_smoothed = self._slew(
                current=self._x_cmd_smoothed,
                target=x_cmd_target,
                dt=dt,
                up_rate=x_accel,
                down_rate=x_decel,
            )
        else:
            # basically stopped -> jump immediately
            self._x_cmd_smoothed = x_cmd_target

        x_cmd = float(self._x_cmd_smoothed)
        action = {
            "x.vel": x_cmd,
            "y.vel": y_cmd,
            "theta.vel": theta_cmd,
            "z.pos": self._z_pos_cmd,
        }

        # Integrated keyboard controls: toggle per-arm untorque on key press (edge-triggered)
        try:
            left_key = self.teleop_keys.get("untorque_left")
            right_key = self.teleop_keys.get("untorque_right")

            if left_key and (left_key in pressed) and (left_key not in self._prev_keys):
                self.untorque_left_active = not self.untorque_left_active
            if right_key and (right_key in pressed) and (right_key not in self._prev_keys):
                self.untorque_right_active = not self.untorque_right_active

            # Always include current flags so host can enforce per-arm blocking
            action["untorque_left"] = self.untorque_left_active
            action["untorque_right"] = self.untorque_right_active

            self._prev_keys = pressed
        except Exception:
            pass

        return action

    def _from_analog_to_base_action(self, x: float, y: float, theta: float):
        """Map analog base inputs (in [-1,1]) through the same speed scaling used for keyboard.

        Ensures behavior is consistent with `_from_keyboard_to_base_action` speed levels.

        Note: since z is now position-controlled, we interpret analog z as an absolute position
        target in [-100, 100] (i.e. z=-1 -> -100, z=+1 -> +100).
        """
        # Clamp to [-1, 1]
        x_in = max(-1.0, min(1.0, float(x)))
        y_in = max(-1.0, min(1.0, float(y)))
        theta_in = max(-1.0, min(1.0, float(theta)))

        speed_setting = self.speed_levels[self.speed_index]
        x_speed = speed_setting["x"]
        y_speed = speed_setting["y"]
        theta_speed = speed_setting["theta"]

        return {
            "x.vel": float(x_in * x_speed),
            "y.vel": float(y_in * y_speed),
            "theta.vel": float(theta_in * theta_speed),
        }

    def _max_abs_delta_between_actions(
        self,
        action_a: dict[str, Any] | None,
        action_b: dict[str, Any] | None,
    ) -> float | None:
        arm_a = extract_arm_positions(action_a)
        arm_b = extract_arm_positions(action_b)
        common_keys = [k for k in arm_a if k in arm_b]
        if not common_keys:
            return None
        return max(abs(arm_a[k] - arm_b[k]) for k in common_keys)

    def _summarize_action_like(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {"type": "none"}

        if isinstance(value, dict):
            summary: dict[str, Any] = {
                "type": "dict",
                "key_count": len(value),
                "arm_targets": extract_arm_positions(value),
            }
            action_value = value.get("action")
            action_vec = self._as_numpy_array(action_value)
            if action_vec is not None:
                summary["action_vector"] = self._summarize_array(action_vec)
            return summary

        arr = self._as_numpy_array(value)
        if arr is not None:
            return {"type": "array_like", "action_vector": self._summarize_array(arr)}

        return {"type": type(value).__name__, "repr": repr(value)[:200]}

    def _as_numpy_array(self, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (list, tuple)):
            try:
                return np.asarray(value, dtype=np.float32)
            except Exception:
                return None
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            try:
                return value.detach().cpu().numpy()
            except Exception:
                return None
        return None

    def _summarize_array(self, value: np.ndarray) -> dict[str, Any]:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return {"size": 0, "shape": list(value.shape)}
        return {
            "shape": list(value.shape),
            "size": int(arr.size),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "first_values": [float(v) for v in arr[:12]],
        }

    def _slew(self, current: float, target: float, dt: float, up_rate: float, down_rate: float) -> float:
        delta = target - current
        # If delta would reduce |current| (i.e., braking toward 0), use down_rate, else up_rate.
        rate = down_rate if (current != 0.0 and (current * delta) < 0.0) else up_rate
        max_step = rate * dt
        if delta > max_step:
            return current + max_step
        if delta < -max_step:
            return current - max_step
        return target
