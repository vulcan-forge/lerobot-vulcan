#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from functools import cached_property
from itertools import chain
from typing import Any, Optional

import numpy as np
import threading

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.constants import OBS_IMAGES, OBS_STATE
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dc_pwm.dc_pwm import PWMDCMotorsController, PWMProtocolHandler
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.motors.dc_motors_controller import DCMotor, MotorNormMode

from lerobot.robots.robot import Robot
from lerobot.robots.sourccey.sourccey_v3beta.sourccey_v3beta_follower.config_sourccey_v3beta_follower import SourcceyV3BetaFollowerConfig
from lerobot.robots.sourccey.sourccey_v3beta.sourccey_v3beta_follower.sourccey_v3beta_follower import SourcceyV3BetaFollower
from lerobot.robots.utils import ensure_safe_goal_position
from .config_sourccey_v3beta import SourcceyV3BetaConfig

logger = logging.getLogger(__name__)


class SourcceyV3Beta(Robot):
    """
    The robot includes a four mecanum wheel mobile base, 1 DC actuator, and 2 remote follower arms.
    The leader arm is connected locally (on the laptop) and its joint positions are recorded and then
    forwarded to the remote follower arm (after applying a safety clamp).
    In parallel, keyboard teleoperation is used to generate raw velocity commands for the wheels.
    """

    config_class = SourcceyV3BetaConfig
    name = "sourccey_v3beta"

    def __init__(self, config: SourcceyV3BetaConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SourcceyV3BetaFollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            use_degrees=config.left_arm_use_degrees,
            cameras={},
        )
        right_arm_config = SourcceyV3BetaFollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            orientation="right",
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            use_degrees=config.right_arm_use_degrees,
            cameras={},
        )

        self.left_arm = SourcceyV3BetaFollower(left_arm_config)
        self.right_arm = SourcceyV3BetaFollower(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)
        self.dc_motors_controller = PWMDCMotorsController(
            config=self.config.dc_motors
        )

    @property
    def _state_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"{motor}.pos": float for motor in self.right_arm.bus.motors} | {
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
        return (
            self.left_arm.is_connected and
            self.right_arm.is_connected and
            self.dc_motors_controller.is_connected and
            all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        # Connect DC motors
        self.dc_motors_controller.connect()

        for cam in self.cameras.values():
            cam.connect()

    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        self.stop_base()
        self.dc_motors_controller.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def auto_calibrate(self, full_reset: bool = False) -> None:
        """
        Auto-calibrate both arms simultaneously using threading.
        """
        # Create threads for each arm
        left_thread = threading.Thread(
            target=self.left_arm.auto_calibrate,
            kwargs={"reversed": False, "full_reset": full_reset}
        )
        right_thread = threading.Thread(
            target=self.right_arm.auto_calibrate,
            kwargs={"reversed": True, "full_reset": full_reset}
        )

        # Start both threads
        left_thread.start()
        right_thread.start()

        # Wait for both threads to complete
        left_thread.join()
        right_thread.join()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_observation(self) -> dict[str, Any]:
        try:
            obs_dict = {}

            left_obs = self.left_arm.get_observation()
            obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

            right_obs = self.right_arm.get_observation()
            obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

            for cam_key, cam in self.cameras.items():
                obs_dict[cam_key] = cam.async_read()

            return obs_dict
        except Exception as e:
            print(f"Error getting observation: {e}")
            return {}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        try:
            left_action = {
                key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
            }
            right_action = {
                key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
            }

            send_action_left = self.left_arm.send_action(left_action)
            send_action_right = self.right_arm.send_action(right_action)

            prefixed_send_action_left = {f"left_{key}": value for key, value in send_action_left.items()}
            prefixed_send_action_right = {f"right_{key}": value for key, value in send_action_right.items()}

            return {**prefixed_send_action_left, **prefixed_send_action_right}
        except Exception as e:
            print(f"Error sending action: {e}")
            return {}

    # Base Functions
    def stop_base(self):
        self.dc_motors_controller.set_velocity("base_front_left_wheel", 0)
        self.dc_motors_controller.set_velocity("base_front_right_wheel", 0)
        self.dc_motors_controller.set_velocity("base_rear_left_wheel", 0)
        self.dc_motors_controller.set_velocity("base_rear_right_wheel", 0)

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        magnitude = raw_speed
        degps = magnitude / steps_per_deg
        return degps

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        # Cap the value to fit within signed 16-bit range (-32768 to 32767)
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF  # 32767 -> maximum positive value
        elif speed_int < -0x8000:
            speed_int = -0x8000  # -32768 -> minimum negative value
        return speed_int

    def _body_to_wheel_raw(
        self,
        x: float,
        y: float,
        theta: float,
        wheel_radius: float = 0.05,
        wheelbase: float = 0.25,  # Distance between front and rear wheels
        track_width: float = 0.25,  # Distance between left and right wheels
        max_raw: int = 3000,
    ) -> dict:
        """
        Convert desired body-frame velocities into wheel raw commands for 4-wheel mechanum drive.

        Parameters:
          x_cmd      : Linear velocity in x (m/s).
          y_cmd      : Linear velocity in y (m/s).
          theta_cmd  : Rotational velocity (deg/s).
          wheel_radius: Radius of each wheel (meters).
          wheelbase  : Distance between front and rear wheels (meters).
          track_width: Distance between left and right wheels (meters).
          max_raw    : Maximum allowed raw command (ticks) per wheel.

        Returns:
          A dictionary with wheel raw commands:
             {"base_front_left_wheel": value, "base_front_right_wheel": value,
              "base_rear_left_wheel": value, "base_rear_right_wheel": value}.

        Notes:
          - Internally, the method converts theta_cmd to rad/s for the kinematics.
          - The raw command is computed from the wheels angular speed in deg/s
            using _degps_to_raw(). If any command exceeds max_raw, all commands
            are scaled down proportionally.
          - Mechanum wheels allow for omnidirectional movement including strafing.
        """
        # Convert rotational velocity from deg/s to rad/s.
        theta_rad = theta * (np.pi / 180.0)

        # Create the body velocity vector [x, y, theta_rad].
        velocity_vector = np.array([x, y, theta_rad])

        # Mechanum wheel kinematic matrix
        # For mechanum wheels, the kinematic relationship is:
        # [v_fl]   [1  1  -(wheelbase + track_width)/2] [v_x]
        # [v_fr] = [1 -1   (wheelbase + track_width)/2] [v_y]
        # [v_rl]   [1 -1  -(wheelbase + track_width)/2] [v_theta]
        # [v_rr]   [1  1   (wheelbase + track_width)/2]

        # Calculate the effective radius for rotation
        effective_radius = np.sqrt((wheelbase/2)**2 + (track_width/2)**2)

        # Build the kinematic matrix for mechanum wheels
        # Each row represents: [forward/backward, left/right, rotation]
        m = np.array([
            [1,  1, -effective_radius],  # Front-left wheel
            [1, -1,  effective_radius],  # Front-right wheel
            [1, -1, -effective_radius],  # Rear-left wheel
            [1,  1,  effective_radius],  # Rear-right wheel
        ])

        # Compute each wheel's linear speed (m/s) and then its angular speed (rad/s).
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius

        # Convert wheel angular speeds from rad/s to deg/s.
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        # Scaling to respect maximum raw command
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        # Convert each wheel's angular speed (deg/s) to a raw integer.
        wheel_raw = [self._degps_to_raw(deg) for deg in wheel_degps]

        return {
            "base_front_left_wheel": wheel_raw[0],
            "base_front_right_wheel": wheel_raw[1],
            "base_rear_left_wheel": wheel_raw[2],
            "base_rear_right_wheel": wheel_raw[3],
        }

    def _wheel_raw_to_body(
        self,
        front_left_wheel_speed,
        front_right_wheel_speed,
        rear_left_wheel_speed,
        rear_right_wheel_speed,
        wheel_radius: float = 0.05,
        wheelbase: float = 0.25,  # Distance between front and rear wheels
        track_width: float = 0.25,  # Distance between left and right wheels
    ) -> dict[str, Any]:
        """
        Convert wheel raw command feedback back into body-frame velocities for 4-wheel mechanum drive.

        Parameters:
          wheel_raw   : Vector with raw wheel commands ("base_front_left_wheel", "base_front_right_wheel",
                       "base_rear_left_wheel", "base_rear_right_wheel").
          wheel_radius: Radius of each wheel (meters).
          wheelbase   : Distance between front and rear wheels (meters).
          track_width : Distance between left and right wheels (meters).

        Returns:
          A dict (x.vel, y.vel, theta.vel) all in m/s and deg/s
        """

        # Convert each raw command back to an angular speed in deg/s.
        wheel_degps = np.array([
            self._raw_to_degps(front_left_wheel_speed),
            self._raw_to_degps(front_right_wheel_speed),
            self._raw_to_degps(rear_left_wheel_speed),
            self._raw_to_degps(rear_right_wheel_speed),
        ])

        # Convert from deg/s to rad/s.
        wheel_radps = wheel_degps * (np.pi / 180.0)
        # Compute each wheel's linear speed (m/s) from its angular speed.
        wheel_linear_speeds = wheel_radps * wheel_radius

        # Calculate the effective radius for rotation
        effective_radius = np.sqrt((wheelbase/2)**2 + (track_width/2)**2)

        # Build the kinematic matrix for mechanum wheels (same as forward kinematics)
        m = np.array([
            [1,  1, -effective_radius],  # Front-left wheel
            [1, -1,  effective_radius],  # Front-right wheel
            [1, -1, -effective_radius],  # Rear-left wheel
            [1,  1,  effective_radius],  # Rear-right wheel
        ])

        # Solve the inverse kinematics: body_velocity = M⁺ · wheel_linear_speeds.
        # Use pseudo-inverse since we have 4 equations and 3 unknowns
        m_pinv = np.linalg.pinv(m)
        velocity_vector = m_pinv.dot(wheel_linear_speeds)
        x, y, theta_rad = velocity_vector
        theta = theta_rad * (180.0 / np.pi)

        return {
            "x.vel": x,
            "y.vel": y,
            "theta.vel": theta,
        }  # m/s and deg/s



