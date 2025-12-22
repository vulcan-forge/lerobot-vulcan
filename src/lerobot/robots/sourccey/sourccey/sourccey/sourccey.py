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
from typing import Any
import numpy as np
import threading
import time

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors.dc_pwm.dc_pwm import PWMDCMotorsController

from lerobot.robots.robot import Robot
from lerobot.robots.sourccey.sourccey.protobuf.sourccey_protobuf import SourcceyProtobuf
from lerobot.robots.sourccey.sourccey.sourccey_follower.config_sourccey_follower import SourcceyFollowerConfig
from lerobot.robots.sourccey.sourccey.sourccey_follower.sourccey_follower import SourcceyFollower
from .config_sourccey import SourcceyConfig

logger = logging.getLogger(__name__)

try:
    from gpiozero import MCP3008
except ImportError:
    MCP3008 = None

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

        self.dc_motors_controller = PWMDCMotorsController(
            motors=self.config.dc_motors,
            config=self.config.dc_motors_config,
        )

        # Initialize protobuf converter
        self.protobuf_converter = SourcceyProtobuf()

        # Track per-arm untorque state for edge detection
        self.untorque_left_prev = False
        self.untorque_right_prev = False

    def __del__(self):
        # Best-effort cleanup; avoid raising during interpreter shutdown.
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:
            # __del__ exceptions are ignored by Python but still printed;
            # swallow anything here to keep shutdown noise-free.
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
        arms_connected = self.left_arm.is_connected and self.right_arm.is_connected
        cams_connected = all(self.cameras[k].is_connected for k in self.cameras.keys())
        return arms_connected and cams_connected

    ###################################################################
    # Connection Management
    ###################################################################
    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        self.dc_motors_controller.connect()

        # Connect only target cameras
        for cam_key in self.cameras.keys():
            self.cameras[cam_key].connect()

    def disconnect(self):
        print("Disconnecting Sourccey")

        # Make per-subsystem disconnects idempotent to support multiple calls.
        if getattr(self.left_arm, "is_connected", False):
            self.left_arm.disconnect()
        if getattr(self.right_arm, "is_connected", False):
            self.right_arm.disconnect()

        self.stop_base()

        # PWMDCMotorsController may already be disconnected; guard if attribute exists.
        if getattr(self.dc_motors_controller, "is_connected", True):
            self.dc_motors_controller.disconnect()

        # Disconnect only those cameras that are still connected.
        for cam_key, cam in self.cameras.items():
            if getattr(cam, "is_connected", False):
                cam.disconnect()

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
        Auto-calibrate arms and (optionally) the linear actuator.

        - If ``arm`` is None, calibrate both arms in parallel.
        - If ``arm`` is "left" or "right", only that side is calibrated.
        - If ``full_reset`` is True, also run the linear actuator limit-finding
          routine and store its calibration in ``calibration_extras``.
        """
        # --- Arm calibration ---
        if arm is None:
            # Create threads for each arm
            left_thread = threading.Thread(
                target=self.left_arm.auto_calibrate,
                kwargs={"reversed": False, "full_reset": full_reset},
            )
            right_thread = threading.Thread(
                target=self.right_arm.auto_calibrate,
                kwargs={"reversed": True, "full_reset": full_reset},
            )

            # Start left arm immediately
            left_thread.start()

            # Wait 3 seconds before starting right arm
            time.sleep(3)
            right_thread.start()

            # Wait for both threads to complete
            left_thread.join()
            right_thread.join()
        else:
            if arm not in ("left", "right"):
                raise ValueError("arm must be one of: None, 'left', 'right'")

            if arm == "left":
                self.left_arm.auto_calibrate(reversed=False, full_reset=full_reset)
            else:
                self.right_arm.auto_calibrate(reversed=True, full_reset=full_reset)

        # --- Linear actuator calibration (only on full reset) ---
        if full_reset:
            try:
                limits = self.auto_calibrate_linear_actuator()
                mech_min_raw = limits["raw_at_mech_min"]
                mech_max_raw = limits["raw_at_mech_max"]
                direction_sign = limits.get("direction_sign", 1.0)

                self.calibration_extras["potentiometer"] = {
                    "adc_channel": 1,
                    "spi_port": 0,
                    "spi_device": 0,
                    "raw_at_mech_min": mech_min_raw,
                    "raw_at_mech_max": mech_max_raw,
                    "output_range": [-100.0, 100.0],
                    "direction_sign": direction_sign,
                }
                self._save_calibration()
                logger.info(
                    "Linear actuator calibrated. raw_at_mech_min=%s, raw_at_mech_max=%s",
                    mech_min_raw,
                    mech_max_raw,
                )
            except Exception as e:
                logger.error("Linear actuator auto-calibration failed: %s", e)

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
            obs_dict = {}

            left_obs = self.left_arm.get_observation()
            obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

            right_obs = self.right_arm.get_observation()
            obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

            base_wheel_vel = self.dc_motors_controller.get_velocities()
            base_vel = self._wheel_normalized_to_body(base_wheel_vel)
            obs_dict.update(base_vel)

            # Linear actuator position from calibrated potentiometer
            try:
                raw = self._read_actuator_raw()
                obs_dict["z.pos"] = self._raw_to_z(raw)
            except Exception as e:
                logger.warning(f"Failed to read linear actuator position: {e}")
                # Fall back to 0.0 if ADC is not available
                obs_dict.setdefault("z.pos", 0.0)

            for cam_key in self.cameras.keys():
                obs_dict[cam_key] = self.cameras[cam_key].async_read()

            return obs_dict
        except Exception as e:
            print(f"Error getting observation: {e}")
            return {}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        try:
            # Apply per-arm untorque flags automatically
            action = self.apply_untorque_flags(action)

            left_action = {key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")}
            right_action = {key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")}
            base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}
            base_goal_pos = {k: v for k, v in action.items() if k.endswith(".pos")}

            prefixed_send_action_left = {}
            prefixed_send_action_right = {}

            # Only send to followers if there are keys for that arm
            if left_action:
                sent_left = self.left_arm.send_action(left_action)
            else:
                sent_left = {}
            if right_action:
                sent_right = self.right_arm.send_action(right_action)
            else:
                sent_right = {}

            prefixed_send_action_left = {f"left_{key}": value for key, value in sent_left.items()}
            prefixed_send_action_right = {f"right_{key}": value for key, value in sent_right.items()}

            # Base velocity
            wheel_action = self._body_to_wheel_normalized(
                base_goal_vel.get("x.vel", 0.0),
                base_goal_vel.get("y.vel", 0.0),
                base_goal_vel.get("theta.vel", 0.0)
            )

            linear_actuator_action = self._body_to_linear_actuator_normalized(
                base_goal_pos.get("z.pos", 0.0)
            )

            dc_motors_action = {**wheel_action, **linear_actuator_action }
            self.dc_motors_controller.set_velocities(dc_motors_action)

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
        # self.dc_motors_controller.update_velocity(max_step=0.1)
        pass

    # Base Functions
    def stop_base(self):
        self.dc_motors_controller.set_velocities({"front_left": 0, "front_right": 0, "rear_left": 0, "rear_right": 0, "linear_actuator": 0})

    ##################################################################################
    # Private Kinematic Functions
    ##################################################################################
    def _body_to_wheel_normalized(
        self,
        x: float,
        y: float,
        theta: float,
    ) -> dict:
        velocity_vector = np.array([x, y, theta])

        # Build the correct kinematic matrix for mecanum wheels
        # Flip the sign of the lateral (y) column to correct strafing direction
        m = np.array([
            [ 1, -1, -1], # Front-left wheel
            [-1, -1, -1], # Front-right wheel
            [ 1,  1, -1], # Rear-left wheel
            [-1,  1, -1], # Rear-right wheel
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
            [ 1, -1, -1], # Front-left wheel
            [-1, -1, -1], # Front-right wheel
            [ 1,  1, -1], # Rear-left wheel
            [-1,  1, -1], # Rear-right wheel
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

    def _body_to_linear_actuator_normalized(
        self,
        z_pos: float,
    ) -> dict:
        """
        Map desired actuator position ``z_pos`` (calibrated units) to a normalized
        velocity command for the DC motor driving the linear actuator.

        If potentiometer calibration is available, we:
        - Read the current actuator position from the ADC
        - Compare it with the desired ``z_pos``
        - Drive the actuator at a fixed speed toward the target until it reaches
          a small deadband around the goal.

        If calibration/ADC is not available, we fall back to interpreting
        ``z_pos`` as a direct (clamped) velocity command in approximately
        the same units as ``output_range`` (default [-100, 100]).
        """
        # Fallback: if ADC or calibration is not usable, treat z_pos as velocity
        try:
            current_raw = self._read_actuator_raw()
            current_z = self._raw_to_z(current_raw)
            _, _, _, _, direction_sign = self._get_pot_calibration()
        except Exception:
            # Interpret z_pos as a direct velocity-like command
            # Scale from roughly [-100, 100] to [-1, 1]
            speed = max(-1.0, min(1.0, z_pos / 100.0))
            return {"linear_actuator": self.clean_value(speed)}

        # Simple bang-bang position control toward target
        Z_DEADBAND = 2.0   # units of calibrated z (e.g. within 2 out of 200 range)
        BASE_SPEED = 0.4   # normalized motor speed (0..1)

        error = z_pos - current_z
        if abs(error) <= Z_DEADBAND:
            speed = 0.0
        else:
            # direction_sign encodes which DC motor sign increases the ADC reading
            direction = 1.0 if error > 0 else -1.0
            speed = direction * direction_sign * BASE_SPEED

        speed = max(-1.0, min(1.0, speed))
        return {"linear_actuator": self.clean_value(speed)}

    def _linear_actuator_normalized_to_body(
        self,
        linear_actuator_normalized: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "z.pos": 100.0, # Deprecated placeholder; we now read z.pos from the potentiometer directly.
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

    # ------------------------------------------------------------------
    # Linear actuator + potentiometer helpers
    # ------------------------------------------------------------------
    def _get_actuator_adc(self):
        if MCP3008 is None:
            raise RuntimeError("gpiozero is not available; cannot read MCP3008 for actuator calibration.")
        if not hasattr(self, "_actuator_adc"):
            # You can move these into config later
            self._actuator_adc = MCP3008(channel=1, port=0, device=0)
        return self._actuator_adc

    def _read_actuator_raw(self, samples: int = 8) -> int:
        adc = self._get_actuator_adc()
        total = 0.0
        for _ in range(samples):
            total += float(adc.raw_value)  # 0..1023
        return int(round(total / samples))

    def _get_pot_calibration(self) -> tuple[float, float, float, float, float]:
        """
        Retrieve potentiometer calibration:
        (raw_min, raw_max, out_min, out_max, direction_sign)

        direction_sign encodes which sign of DC motor command increases the ADC reading.
        """
        extras = self.calibration_extras.get("potentiometer", {}) or {}
        raw_min = float(extras.get("raw_at_mech_min", 0.0))
        raw_max = float(extras.get("raw_at_mech_max", 1023.0))

        output_range = extras.get("output_range", [-100.0, 100.0])
        if not isinstance(output_range, (list, tuple)) or len(output_range) != 2:
            output_range = [-100.0, 100.0]
        out_min, out_max = float(output_range[0]), float(output_range[1])

        direction_sign = float(extras.get("direction_sign", 1.0))

        # Guard against degenerate calibration
        if raw_max == raw_min:
            raw_min, raw_max = 0.0, 1.0

        return raw_min, raw_max, out_min, out_max, direction_sign

    def _raw_to_z(self, raw: int) -> float:
        """Map raw ADC reading to calibrated z position."""
        raw_min, raw_max, out_min, out_max, _ = self._get_pot_calibration()
        # Normalize 0..1 within calibrated range
        t = (float(raw) - raw_min) / (raw_max - raw_min)
        t = max(0.0, min(1.0, t))
        z = out_min + t * (out_max - out_min)
        return self.clean_value(z)

    def _set_linear_actuator_speed(self, speed: float) -> None:
        # IMPORTANT: speed units depend on PWMDCMotorsController.
        # If it expects [-1..1], use 0.2/ -0.2 etc.
        # If it expects [-100..100], use 20/-20 etc.
        self.dc_motors_controller.set_velocities({"linear_actuator": speed})

    def _drive_until_stall_by_pot(
        self,
        speed: float,
        timeout_s: float = 8.0,
        sample_dt: float = 0.05,
        stable_eps_counts: int = 2,
        stable_time_s: float = 0.25,
    ) -> int:
        """
        Drive actuator and declare 'limit reached' when pot raw stops changing.
        Returns the raw ADC reading at the detected limit.
        """
        start_t = time.monotonic()
        last = self._read_actuator_raw()
        stable_start = None

        self._set_linear_actuator_speed(speed)
        try:
            while True:
                time.sleep(sample_dt)
                raw = self._read_actuator_raw()

                # If we're not moving (within eps), start/continue stability timer
                if abs(raw - last) <= stable_eps_counts:
                    if stable_start is None:
                        stable_start = time.monotonic()
                    elif (time.monotonic() - stable_start) >= stable_time_s:
                        return raw
                else:
                    stable_start = None

                last = raw

                if (time.monotonic() - start_t) >= timeout_s:
                    raise TimeoutError("Actuator limit detection timed out (pot never stabilized).")
        finally:
            self._set_linear_actuator_speed(0.0)

    def auto_calibrate_linear_actuator(self) -> dict[str, int]:
        """
        Fully-automatic endpoint detection for the linear actuator using the potentiometer.
        Returns {"raw_at_mech_min": ..., "raw_at_mech_max": ..., "direction_sign": ...}
        """
        # Use gentle speed during calibration to reduce stall current / brownout risk
        CAL_SPEED = -0.2

        # Find one end
        raw_a = self._drive_until_stall_by_pot(speed=CAL_SPEED)

        time.sleep(0.3)

        # Find the other end
        raw_b = self._drive_until_stall_by_pot(speed=-CAL_SPEED)

        # Normalize ordering and record DC direction that increases the ADC reading.
        mech_min_raw = min(raw_a, raw_b)
        mech_max_raw = max(raw_a, raw_b)

        # Second move used +|CAL_SPEED|; see whether that increased or decreased raw.
        if raw_b > raw_a:
            direction_sign = 1.0  # positive speed -> increasing ADC
        elif raw_b < raw_a:
            direction_sign = -1.0  # positive speed -> decreasing ADC
        else:
            direction_sign = 1.0

        return {
            "raw_at_mech_min": mech_min_raw,
            "raw_at_mech_max": mech_max_raw,
            "direction_sign": direction_sign,
        }
