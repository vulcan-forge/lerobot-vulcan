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
from functools import cached_property
from typing import Any, Iterable
import numpy as np
import threading
import time

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.motors.dc_pwm.dc_pwm import PWMDCMotorsController

from lerobot.robots.robot import Robot
from lerobot.robots.sourccey.sourccey.protobuf.sourccey_protobuf import SourcceyProtobuf
from lerobot.robots.sourccey.sourccey.sourccey_follower.config_sourccey_follower import SourcceyFollowerConfig
from lerobot.robots.sourccey.sourccey.sourccey_follower.sourccey_follower import SourcceyFollower
from lerobot.robots.sourccey.sourccey.sourccey_z_actuator.sourccey_z_actuator import SourcceyZActuator, ZSensor
from .config_sourccey import SourcceyConfig

logger = logging.getLogger(__name__)

class Sourccey(Robot):
    """
    The robot includes a four mecanum wheel mobile base, 1 DC actuator, and 2 remote follower arms.
    The leader arm is connected locally (on the laptop) and its joint positions are recorded and then
    forwarded to the remote follower arm (after applying a safety clamp).
    In parallel, keyboard teleoperation is used to generate raw velocity commands for the wheels.
    """

    config_class = SourcceyConfig
    name = "sourccey"

    def __init__(self, config: SourcceyConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SourcceyFollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            motor_models=config.left_arm_motor_models,
            port=config.left_arm_port,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            use_degrees=config.left_arm_use_degrees,
            cameras={},
        )
        right_arm_config = SourcceyFollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            motor_models=config.right_arm_motor_models,
            port=config.right_arm_port,
            orientation="right",
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            use_degrees=config.right_arm_use_degrees,
            cameras={},
        )

        self.left_arm = SourcceyFollower(left_arm_config)
        self.right_arm = SourcceyFollower(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)
        # Cameras are treated as best-effort: startup should not fail if a camera fails to connect/read.
        self._connected_cameras: set[str] = set()
        self._camera_lock = threading.RLock()

        self.dc_motors_controller = PWMDCMotorsController(
            motors=self.config.dc_motors,
            config=self.config.dc_motors_config,
        )

         # Z Actuator Code
        self.z_sensor = ZSensor(adc_channel=1, vref=3.30, average_samples=50)
        self.z_actuator = SourcceyZActuator(
            sensor=self.z_sensor,
            driver=self.dc_motors_controller,
            motor="linear_actuator",
        )

        # Initialize protobuf converter
        self.protobuf_converter = SourcceyProtobuf()

        # Track per-arm untorque state for edge detection
        self.untorque_left_prev = False
        self.untorque_right_prev = False
        self._arms_connected = False
        self._arms_available = True
        self._arms_unavailable_reason: str | None = None
        self._connect_arms_on_startup = True
        self._arm_observation_enabled = True

    def __del__(self):
        # Destructors can run on partially initialized objects if __init__ raised.
        try:
            if hasattr(self, "left_arm") and hasattr(self, "right_arm") and self.is_connected:
                self.disconnect()
        except Exception:
            pass

    ###################################################################
    # Properties and Attributes
    ###################################################################
    @property
    def _state_ft(self) -> dict[str, type]:
        return {
            f"{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"{motor}.pos": float for motor in self.right_arm.bus.motors} | {
                "z.pos": float,
                "x.vel": float,
                "y.vel": float,
                "theta.vel": float,
            }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        left_arm_connected = self.left_arm is not None and self.left_arm.is_connected
        right_arm_connected = self.right_arm is not None and self.right_arm.is_connected
        arms_connected = (
            left_arm_connected and right_arm_connected
            if self._connect_arms_on_startup or self._arms_connected
            else True
        )

        # Only require cameras that successfully connected (failed cameras are ignored).
        cams_connected = all(self.cameras[k].is_connected for k in self._connected_cameras)
        base_connected = bool(self.dc_motors_controller.is_connected)
        z_connected = bool(self.z_actuator.is_connected)
        return arms_connected and cams_connected and base_connected and z_connected

    ###################################################################
    # Connection Management
    ###################################################################
    def connect(
        self,
        calibrate: bool = True,
        connect_arms: bool = True,
        camera_keys: Iterable[str] | None = None,
    ) -> None:
        self._connect_arms_on_startup = bool(connect_arms)
        self._arms_connected = False
        if connect_arms:
            self._connect_arms(calibrate=calibrate)
        self.dc_motors_controller.connect()
        self.z_actuator.connect()

        # Connect cameras in a stable order and avoid strict per-camera warmup deadlocks.
        with self._camera_lock:
            self._connected_cameras.clear()
        self.connect_cameras(self._camera_connect_order(camera_keys))

    def _camera_connect_order(self, camera_keys: Iterable[str] | None = None) -> list[str]:
        allowed = None if camera_keys is None else set(camera_keys)
        preferred = ["front_left", "front_right", "bottom"]
        ordered = [key for key in preferred if key in self.cameras and (allowed is None or key in allowed)]
        ordered.extend(
            key
            for key in self.cameras.keys()
            if key not in ordered and (allowed is None or key in allowed)
        )
        return ordered

    def connect_cameras(self, camera_keys: Iterable[str]) -> None:
        """Connect a selected set of camera devices without touching other hardware."""
        connect_order = self._camera_connect_order(camera_keys)
        with self._camera_lock:
            for index, cam_key in enumerate(connect_order):
                if cam_key in self._connected_cameras:
                    continue
                camera = self.cameras[cam_key]
                try:
                    self._connect_camera(camera, cam_key)
                    self._connected_cameras.add(cam_key)
                    if index < len(connect_order) - 1:
                        time.sleep(0.35)
                except Exception as e:
                    logger.warning(f"Camera '{cam_key}' failed to connect: {e}. Continuing without it.")
                    try:
                        if getattr(camera, "is_connected", False):
                            camera.disconnect()
                    except Exception:
                        pass

    def disconnect_cameras(self, camera_keys: Iterable[str] | None = None) -> None:
        """Disconnect selected cameras, or all connected cameras if no keys are provided."""
        with self._camera_lock:
            keys = list(self._connected_cameras if camera_keys is None else camera_keys)
            for cam_key in keys:
                if cam_key not in self._connected_cameras:
                    continue
                try:
                    self.cameras[cam_key].disconnect()
                except Exception:
                    logger.exception("Failed to disconnect camera '%s'", cam_key)
                self._connected_cameras.discard(cam_key)

    def set_connected_cameras(self, camera_keys: Iterable[str]) -> None:
        """Switch the active camera set to exactly the requested keys."""
        desired = {key for key in camera_keys if key in self.cameras}
        with self._camera_lock:
            current = set(self._connected_cameras)
        self.disconnect_cameras(current - desired)
        self.connect_cameras(self._camera_connect_order(desired - current))

    def connected_camera_keys(self) -> tuple[str, ...]:
        with self._camera_lock:
            return tuple(self._camera_connect_order(self._connected_cameras))

    def _connect_camera(self, camera: Any, cam_key: str) -> None:
        if isinstance(camera, OpenCVCamera):
            camera.connect(warmup=False)
            self._wait_for_camera_ready(camera, cam_key)
            return
        camera.connect()

    def _wait_for_camera_ready(self, camera: OpenCVCamera, cam_key: str) -> None:
        startup_deadline_s = 6.0 if cam_key.startswith("front_") else 4.0
        stale_threshold_ms = 1200 if cam_key.startswith("front_") else 1800
        poll_timeout_ms = 700 if cam_key.startswith("front_") else 900
        deadline = time.perf_counter() + startup_deadline_s
        last_error: Exception | None = None

        while time.perf_counter() < deadline:
            try:
                camera.read_latest(max_age_ms=stale_threshold_ms)
                logger.info("Camera '%s' connected and delivering frames.", cam_key)
                return
            except Exception as read_latest_error:
                last_error = read_latest_error
                try:
                    camera.async_read(timeout_ms=poll_timeout_ms)
                    logger.info("Camera '%s' connected and delivered first async frame.", cam_key)
                    return
                except Exception as async_error:
                    last_error = async_error
                    time.sleep(0.08)

        raise ConnectionError(
            f"{camera} failed to deliver a fresh frame within {startup_deadline_s:.1f}s"
            + (f" (last error: {last_error})" if last_error is not None else "")
        )

    def _connect_arms(self, *, calibrate: bool) -> None:
        if self._arms_connected:
            return
        if not self._arms_available:
            raise RuntimeError(
                self._arms_unavailable_reason or "follower arms are unavailable"
            )
        try:
            self.left_arm.connect(calibrate)
            self.right_arm.connect(calibrate)
        except Exception as exc:
            # A removed arm can leave the first follower partially connected
            # when discovery fails. Roll the lazy connection back atomically
            # and latch it unavailable so every subsequent base packet does not
            # retry the same broken serial handshake.
            for arm in (self.left_arm, self.right_arm):
                if getattr(arm, "is_connected", False):
                    try:
                        arm.disconnect()
                    except Exception:
                        logger.exception("Failed to roll back partial arm connection")
            self._arms_connected = False
            self._arms_available = False
            self._arms_unavailable_reason = str(exc)
            raise
        self._arms_connected = True

    def disable_arm_hardware(self, reason: str = "disabled by host configuration") -> None:
        """Permanently quarantine follower-arm I/O for this host process."""
        self._arms_connected = False
        self._arms_available = False
        self._arms_unavailable_reason = str(reason)

    def set_arm_observation_enabled(self, enabled: bool) -> None:
        """Enable/disable follower-arm readback in the host observation loop.

        SLAM mapping may briefly connect the arms to latch a safe stow pose, but
        continuous arm polling competes with the Pi's other USB/serial devices
        and is unnecessary for mapping. Teleop can re-enable readback when arm
        state is part of the operator-facing observation stream.
        """
        self._arm_observation_enabled = bool(enabled)

    def disconnect(self):
        if not self.is_connected:
            return

        logger.info(f"Disconnecting Sourccey")
        if self.left_arm.is_connected:
            self.left_arm.disconnect()
        if self.right_arm.is_connected:
            self.right_arm.disconnect()
        self._arms_connected = False

        self.stop_base()
        self.dc_motors_controller.disconnect()

        self.z_actuator.stop()
        self.z_actuator.disconnect()

        # Disconnect only those we connected
        self.disconnect_cameras()

        logger.info(f"Sourccey disconnected.")

    ###################################################################
    # Calibration and Configuration Management
    ###################################################################
    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def auto_calibrate(self, full_reset: bool = False, arm: str | None = None) -> None:
        """
        Auto-calibrate robot joints. If arm is None, calibrate Z first, then both arms in parallel.
        arm can be "left" or "right" to calibrate only that side.
        """
        if arm is None:
            if full_reset:
                self.disable_arm_torque()

            # Soft robot calibration should not physically move the Z actuator.
            # Only a full reset re-detects Z limits by movement.
            self.z_actuator.calibrator.auto_calibrate(full_reset=full_reset)

            calibration_errors: list[tuple[str, BaseException]] = []

            def run_arm_calibration(arm_name: str, arm_obj, **kwargs) -> None:
                try:
                    arm_obj.auto_calibrate(**kwargs)
                except BaseException as exc:
                    logger.exception("Auto-calibration failed for %s arm", arm_name)
                    calibration_errors.append((arm_name, exc))

            # Create threads for each arm
            left_thread = threading.Thread(
                target=run_arm_calibration,
                args=("left", self.left_arm),
                kwargs={"reverse": False, "full_reset": full_reset},
            )
            right_thread = threading.Thread(
                target=run_arm_calibration,
                args=("right", self.right_arm),
                kwargs={"reverse": True, "full_reset": full_reset},
            )

            # Start left arm immediately
            left_thread.start()

            # Wait 3 seconds before starting right arm
            time.sleep(3)
            right_thread.start()

            # Wait for both threads to complete
            left_thread.join()
            right_thread.join()

            if calibration_errors:
                error_messages = ", ".join(f"{arm_name}: {exc}" for arm_name, exc in calibration_errors)
                raise RuntimeError(f"Arm auto-calibration failed ({error_messages})") from calibration_errors[0][1]

        elif arm == "left":
            self.left_arm.auto_calibrate(reverse=False, full_reset=full_reset)
        elif arm == "right":
            self.right_arm.auto_calibrate(reverse=True, full_reset=full_reset)
        else:
            raise ValueError("arm must be one of: None, 'left', 'right'")

        print("Auto-calibration completed")
        return True

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    ###################################################################
    # Data Management
    ###################################################################

    def get_observation(self) -> dict[str, Any]:
        try:
            start = time.perf_counter()

            obs_dict = {}

            if self._arms_connected and self._arm_observation_enabled:
                left_obs = self.left_arm.get_observation()
                obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

                right_obs = self.right_arm.get_observation()
                obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})
            else:
                for motor in self.left_arm.bus.motors:
                    obs_dict[f"left_{motor}.pos"] = 0.0
                for motor in self.right_arm.bus.motors:
                    obs_dict[f"right_{motor}.pos"] = 0.0

            base_wheel_vel = self.dc_motors_controller.get_velocities()
            base_vel = self._wheel_normalized_to_body(base_wheel_vel)
            obs_dict.update(base_vel)

            # Z actuator position (best-effort; keep schema stable)
            try:
                if self.z_actuator is not None and self.z_actuator.is_connected and self.z_actuator.use_z_actuator:
                    obs_dict["z.pos"] = float(self.z_actuator.read_position())
                else:
                    obs_dict["z.pos"] = 100.0
            except Exception:
                obs_dict["z.pos"] = 100.0

            with self._camera_lock:
                active_camera_keys = list(self._connected_cameras)
            for cam_key in active_camera_keys:
                try:
                    start = time.perf_counter()
                    obs_dict[cam_key] = self.cameras[cam_key].read_latest()
                    dt_ms = (time.perf_counter() - start) * 1e3
                    logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
                except Exception as e:
                    # Keep active camera schemas stable even if a connected camera read fails.
                    h = int(self.config.cameras[cam_key].height)
                    w = int(self.config.cameras[cam_key].width)
                    obs_dict[cam_key] = np.zeros((h, w, 3), dtype=np.uint8)
                    logger.warning(f"Camera '{cam_key}' read failed: {e}. Using black frame.")

            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} get_observation took: {dt_ms:.1f}ms")

            return obs_dict
        except Exception as e:
            print(f"Error getting observation: {e}")
            return {}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        try:
            # Commit the real-time base command before touching either serial arm
            # bus. A slow or wedged follower servo must never starve wheel control.
            base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}
            wheel_action = self._body_to_wheel_normalized(
                base_goal_vel.get("x.vel", 0.0),
                base_goal_vel.get("y.vel", 0.0),
                base_goal_vel.get("theta.vel", 0.0),
            )
            self.dc_motors_controller.set_velocities(wheel_action)

            # Apply per-arm untorque flags automatically
            action = self.apply_untorque_flags(action)

            left_action = {key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")}
            right_action = {key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")}
            base_goal_pos = {k: v for k, v in action.items() if k.endswith(".pos")}
            left_connect_requested = bool(left_action) and not bool(action.get("untorque_left", False))
            right_connect_requested = bool(right_action) and not bool(action.get("untorque_right", False))

            # Proto3 cannot mark arm targets as ABSENT — every command arrives with
            # all 12 joints present, defaulting to exactly 0.0. Forwarding such an
            # all-zero block DRIVES the arms to the zero pose (the servos move on
            # any goal write; field 2026-07-21: repeated zero-pose slams during
            # connect, one cracked arm). No real command targets exactly 0.0 on
            # every joint of an arm, so an all-zero block means "no arm command".
            if left_action and not any(abs(float(v)) > 1e-9 for v in left_action.values()):
                left_action = {}
            if right_action and not any(abs(float(v)) > 1e-9 for v in right_action.values()):
                right_action = {}

            # Lazy connect: real arm targets OR an explicit untorque=False request
            # bring the arms online (a torque request with no targets must still
            # connect them so they start reporting — with nothing forwarded below,
            # connecting cannot move them).
            wants_arms = bool(left_action or right_action or left_connect_requested or right_connect_requested)
            if wants_arms and not self._arms_connected:
                if self._arms_available:
                    try:
                        self._connect_arms(calibrate=False)
                    except Exception as exc:
                        logger.error(
                            "Follower arms unavailable; quarantining all arm "
                            "targets while base control continues: %s",
                            exc,
                        )
                if not self._arms_connected:
                    left_action = {}
                    right_action = {}
                    wants_arms = False

            prefixed_send_action_left = {}
            prefixed_send_action_right = {}

            # Only send to followers if there are keys for that arm
            if left_action:
                try:
                    sent_left = self.left_arm.send_action(left_action)
                except Exception as exc:
                    # Arm communication and base actuation are independent
                    # subsystems. A transient follower-bus fault must not skip
                    # the wheel write below and immobilize autonomous recovery.
                    logger.error("Left arm command failed; base command will continue: %s", exc)
                    sent_left = {}
            else:
                sent_left = {}
            if right_action:
                try:
                    sent_right = self.right_arm.send_action(right_action)
                except Exception as exc:
                    logger.error("Right arm command failed; base command will continue: %s", exc)
                    sent_right = {}
            else:
                sent_right = {}

            prefixed_send_action_left = {f"left_{key}": value for key, value in sent_left.items()}
            prefixed_send_action_right = {f"right_{key}": value for key, value in sent_right.items()}

            # Z actuator is position-controlled; drive toward the latest z.pos target (non-blocking).
            if "z.pos" in base_goal_pos and self.z_actuator.use_z_actuator:
                try:
                    self.z_actuator.move_to_position(float(base_goal_pos.get("z.pos", 100.0)), hz=30.0, instant=True)
                except Exception as e:
                    logger.warning(f"Failed to command z actuator: {e}")

            sent_action = {**prefixed_send_action_left, **prefixed_send_action_right, **base_goal_pos, **base_goal_vel}
            return sent_action
        except Exception as e:
            print(f"Error sending action: {e}")
            return {}

    ###################################################################
    # Control Management
    ###################################################################
    def apply_untorque_flags(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Apply per-arm untorque flags: disable/enable torque and strip positions.
        Manages internal state for edge detection.

        Returns:
            dict: modified action with positions stripped if untorqued
        """
        left_flag = bool(action.get("untorque_left", False))
        right_flag = bool(action.get("untorque_right", False))

        if not self._arms_connected:
            self.untorque_left_prev = left_flag
            self.untorque_right_prev = right_flag
            return {
                k: v for k, v in action.items()
                if not (left_flag and k.startswith("left_")) and not (right_flag and k.startswith("right_"))
            }

        # Left arm handling
        if left_flag:
            if not self.untorque_left_prev:
                self.left_arm.bus.disable_torque()
            action = {k: v for k, v in action.items() if not k.startswith("left_")}
        elif self.untorque_left_prev and not left_flag:
            self.left_arm.bus.enable_torque()

        # Right arm handling
        if right_flag:
            if not self.untorque_right_prev:
                self.right_arm.bus.disable_torque()
            action = {k: v for k, v in action.items() if not k.startswith("right_")}
        elif self.untorque_right_prev and not right_flag:
            self.right_arm.bus.enable_torque()

        # Update state
        self.untorque_left_prev = left_flag
        self.untorque_right_prev = right_flag

        return action

    def update(self):
        # Can be used to update the robot every cycle. Such as potentially a motor
        self.dc_motors_controller.update_velocity(max_step=0.25)

    def get_base_velocity(self) -> dict[str, Any]:
        """Read wheel-derived body velocity without touching cameras or arms."""
        return self._wheel_normalized_to_body(
            self.dc_motors_controller.get_velocities()
        )

    # Base Functions
    def stop_base(self):
        self.dc_motors_controller.set_velocities({"front_left": 0, "front_right": 0, "rear_left": 0, "rear_right": 0})

    def disable_arm_torque(self) -> None:
        """
        Disable torque on both follower arms so they can be freely moved by hand.

        Also marks internal untorque edge state as active so the next streamed command
        with untorque flags set to `False` will cleanly re-enable torque.
        """
        if not self._arms_connected:
            self.untorque_left_prev = True
            self.untorque_right_prev = True
            return

        for arm_name, arm in (("left", self.left_arm), ("right", self.right_arm)):
            try:
                arm.bus.disable_torque()
            except Exception as e:
                logger.warning(f"Failed to disable torque on {arm_name} arm: {e}")

        self.untorque_left_prev = True
        self.untorque_right_prev = True

    def watchdog_stop_and_relax(self) -> None:
        """
        Host watchdog fail-safe: stop mobile + z motion and fully release arm torque.
        This is unused at the moment as we wont want the robot relaxing while its supposed to be active
        """
        self.stop_base()
        try:
            self.z_actuator.stop()
        except Exception as e:
            logger.warning(f"Failed to stop z actuator during watchdog: {e}")
        self.disable_arm_torque()

    def watchdog_stop_motion(self) -> None:
        """
        Host watchdog fail-safe for command-stream loss during active operation.

        Stops only the mobile base and z actuator motion without relaxing follower-arm torque.
        This is safer for autonomous/teleop command dropouts where we want motion to halt
        immediately but do not want the arms to suddenly go limp.
        """
        self.stop_base()
        try:
            self.z_actuator.stop()
        except Exception as e:
            logger.warning(f"Failed to stop z actuator during motion watchdog: {e}")

    ##################################################################################
    # Private Kinematic Functions
    ##################################################################################
    def _body_to_wheel_normalized(
        self,
        x: float,
        y: float,
        theta: float,
    ) -> dict:
        velocity_vector = np.array([x, y, theta], dtype=float)

        # Build the correct kinematic matrix for mecanum wheels
        # Columns correspond to [x, y (strafe), theta (turn)].
        # This mapping makes A/D (y.vel) strafe and Z/X (theta.vel) rotate.
        m = np.array([
            [ 1, -1,  1], # Front-left wheel
            [-1, -1,  1], # Front-right wheel
            [ 1,  1,  1], # Rear-left wheel
            [-1,  1,  1], # Rear-right wheel
        ])

        wheel_normalized = m.dot(velocity_vector)
        wheel_normalized = np.clip(wheel_normalized, -1.0, 1.0)
        wheel_dict = {
            "front_left": float(wheel_normalized[0]),
            "front_right": float(wheel_normalized[1]),
            "rear_left": float(wheel_normalized[2]),
            "rear_right": float(wheel_normalized[3]),
        }

        return wheel_dict

    def _wheel_normalized_to_body(
        self,
        wheel_normalized: dict[str, Any],
    ) -> dict[str, Any]:

        # Convert each normalized command back to an angular speed in deg/s.
        wheel_array = np.array([
            wheel_normalized["front_left"],
            wheel_normalized["front_right"],
            wheel_normalized["rear_left"],
            wheel_normalized["rear_right"],
        ])

        # Kinematic matrix for mecanum wheels (must match forward kinematics)
        m = np.array([
            [ 1, -1,  1], # Front-left wheel
            [-1, -1,  1], # Front-right wheel
            [ 1,  1,  1], # Rear-left wheel
            [-1,  1,  1], # Rear-right wheel
        ])

        # Solve the inverse kinematics: body_velocity = M⁺ · wheel_linear_speeds.
        m_pinv = np.linalg.pinv(m)
        velocity_vector = m_pinv.dot(wheel_array)
        x, y, theta = velocity_vector

        return {
            "x.vel": self.clean_value(x),
            "y.vel": self.clean_value(y),
            "theta.vel": self.clean_value(theta),
        }

    # Round to prevent floating-point precision issues and handle -0.0
    def clean_value(self, val):
        rounded = round(val, 8)

        # Convert -0.0 to 0.0 and very small values to 0.0
        return 0.0 if abs(rounded) < 1e-10 else rounded

    ##################################################################################
    # Motor Configuration Functions
    ##################################################################################
    def set_baud_rate(self, baud_rate: int) -> None:
        self.left_arm.bus.set_baudrate(baud_rate)
        self.right_arm.bus.set_baudrate(baud_rate)
