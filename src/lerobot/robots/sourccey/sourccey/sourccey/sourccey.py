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
import random
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
from lerobot.robots.sourccey.sourccey.sourccey_follower.config_sourccey_follower import SourcceyFollowerConfig
from lerobot.robots.sourccey.sourccey.sourccey_follower.sourccey_follower import SourcceyFollower
from lerobot.robots.utils import ensure_safe_goal_position
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
        # Optional limiter set by CLI: None | "left" | "right"
        self.limit_arm: str | None = None

        left_arm_config = SourcceyFollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            use_degrees=config.left_arm_use_degrees,
            cameras={},
        )
        right_arm_config = SourcceyFollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
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


    def __del__(self):
        self.disconnect()

    @property
    def _state_ft(self) -> dict[str, type]:
        return {
            f"{motor}.pos": float for motor in self.left_arm.bus.motors} | {
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
        left_ok = self.left_arm.is_connected if (self.limit_arm is None or self.limit_arm == "left") else True
        right_ok = self.right_arm.is_connected if (self.limit_arm is None or self.limit_arm == "right") else True
        # Cameras: only require target cameras for selected arm(s)
        target_cam_keys = self._target_camera_keys()
        cams_ok = all(self.cameras[k].is_connected for k in target_cam_keys)
        return left_ok and right_ok and cams_ok

    def connect(self, calibrate: bool = True) -> None:
        # Connect only requested arm if limit set; else both
        if self.limit_arm == "left":
            self.left_arm.connect(calibrate)
        elif self.limit_arm == "right":
            self.right_arm.connect(calibrate)
        else:
            self.left_arm.connect(calibrate)
            self.right_arm.connect(calibrate)

        self.dc_motors_controller.connect()

        # Connect only target cameras
        for cam_key in self._target_camera_keys():
            self.cameras[cam_key].connect()

    def disconnect(self):
        if self.limit_arm == "left":
            self.left_arm.disconnect()
        elif self.limit_arm == "right":
            self.right_arm.disconnect()
        else:
            self.left_arm.disconnect()
            self.right_arm.disconnect()

        self.stop_base()
        self.dc_motors_controller.disconnect()

        # Disconnect only those we connected
        for cam_key in self._target_camera_keys():
            cam = self.cameras[cam_key]
            if cam.is_connected:
                cam.disconnect()

    @property
    def is_calibrated(self) -> bool:
        if self.limit_arm == "left":
            return self.left_arm.is_calibrated
        if self.limit_arm == "right":
            return self.right_arm.is_calibrated
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        if self.limit_arm == "left":
            self.left_arm.calibrate()
            return
        if self.limit_arm == "right":
            self.right_arm.calibrate()
            return
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def auto_calibrate(self, full_reset: bool = False, arm: str | None = None) -> None:
        """
        Auto-calibrate arms. If arm is None, calibrate both in parallel.
        arm can be "left" or "right" to calibrate only that side.
        """
        if arm is None:
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
            return

        if arm not in ("left", "right"):
            raise ValueError("arm must be one of: None, 'left', 'right'")

        if arm == "left":
            self.left_arm.auto_calibrate(reversed=False, full_reset=full_reset)
        else:
            self.right_arm.auto_calibrate(reversed=True, full_reset=full_reset)

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_observation(self) -> dict[str, Any]:
        try:
            obs_dict = {}
            if self.limit_arm is None or self.limit_arm == "left":
                left_obs = self.left_arm.get_observation()
                obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})
            if self.limit_arm is None or self.limit_arm == "right":
                right_obs = self.right_arm.get_observation()
                obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

            for cam_key in self._target_camera_keys():
                cam = self.cameras[cam_key]
                obs_dict[cam_key] = cam.async_read()

            return obs_dict
        except Exception as e:
            print(f"Error getting observation: {e}")
            return {}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        try:
            left_action = {key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")}
            right_action = {key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")}

            prefixed_send_action_left = {}
            prefixed_send_action_right = {}

            if self.limit_arm is None or self.limit_arm == "left":
                sent_left = self.left_arm.send_action(left_action)
                prefixed_send_action_left = {f"left_{key}": value for key, value in sent_left.items()}
            if self.limit_arm is None or self.limit_arm == "right":
                sent_right = self.right_arm.send_action(right_action)
                prefixed_send_action_right = {f"right_{key}": value for key, value in sent_right.items()}

            # Base velocity
            x_vel = action.get("x.vel", 0)
            y_vel = action.get("y.vel", 0)
            theta_vel = action.get("theta.vel", 0)
            action = self._body_to_wheel_normalized(x_vel, y_vel, theta_vel)
            self.dc_motors_controller.set_velocities(action)

            return {**prefixed_send_action_left, **prefixed_send_action_right, **action}
        except Exception as e:
            print(f"Error sending action: {e}")
            return {}

    # Base Functions
    def stop_base(self):
        self.dc_motors_controller.set_velocities({"front_left": 0, "front_right": 0, "rear_left": 0, "rear_right": 0})

    def update(self):
        self.dc_motors_controller.update_velocity()

    def _target_camera_keys(self) -> list[str]:
        """Return camera keys to connect/check based on limit_arm.
        We keep front cameras always. We exclude the wrist camera on the non-selected arm.
        """
        keys = list(self.cameras.keys())
        if self.limit_arm == "left":
            # Exclude right wrist camera if present
            return [k for k in keys if k != "wrist_right"]
        if self.limit_arm == "right":
            # Exclude left wrist camera if present
            return [k for k in keys if k != "wrist_left"]
        return keys

    @staticmethod
    def _normalized_to_degps(normalized_speed: float) -> float:
        """
        Convert normalized DC motor speed (-1 to 1) to degrees per second.

        Parameters:
          normalized_speed: DC motor speed from -1.0 to 1.0

        Returns:
          Angular velocity in degrees per second
        """
        # Define maximum angular velocity for DC motors (adjust based on your motor specs)
        max_degps = 360.0  # Adjust this value based on your DC motor's maximum speed

        # Clamp normalized speed to valid range
        normalized_speed = np.clip(normalized_speed, -1.0, 1.0)

        # Convert to degrees per second
        degps = normalized_speed * max_degps
        return degps

    @staticmethod
    def _degps_to_normalized(degps: float) -> float:
        """
        Convert degrees per second to normalized DC motor speed (-1 to 1).

        Parameters:
          degps: Angular velocity in degrees per second

        Returns:
          Normalized DC motor speed from -1.0 to 1.0
        """
        # Define maximum angular velocity for DC motors (adjust based on your motor specs)
        max_degps = 360.0  # Adjust this value based on your DC motor's maximum speed

        # Convert to normalized speed and clamp to valid range
        normalized_speed = degps / max_degps
        normalized_speed = np.clip(normalized_speed, -1.0, 1.0)

        return normalized_speed

    def _body_to_wheel_normalized(
        self,
        x: float,
        y: float,
        theta: float,
        wheel_radius: float = 0.05,
        wheelbase: float = 0.25,  # Distance between front and rear wheels
        track_width: float = 0.25,  # Distance between left and right wheels
        max_normalized: float = 1.0,
    ) -> dict:
        """
        Convert desired body-frame velocities into normalized wheel commands for 4-wheel mechanum drive.

        Parameters:
          x_cmd      : Linear velocity in x (m/s).
          y_cmd      : Linear velocity in y (m/s).
          theta_cmd  : Rotational velocity (deg/s).
          wheel_radius: Radius of each wheel (meters).
          wheelbase  : Distance between front and rear wheels (meters).
          track_width: Distance between left and right wheels (meters).
          max_normalized: Maximum allowed normalized command per wheel (typically 1.0).

        Returns:
          A dictionary with normalized wheel commands:
             {"front_left": value, "front_right": value,
              "rear_left": value, "rear_right": value}.

        Notes:
          - Internally, the method converts theta_cmd to rad/s for the kinematics.
          - The normalized command is computed from the wheels angular speed in deg/s
            using _degps_to_normalized(). If any command exceeds max_normalized, all commands
            are scaled down proportionally.
          - Mechanum wheels allow for omnidirectional movement including strafing.
          - Right wheels are inverted to match physical motor direction.
        """
        # Convert rotational velocity from deg/s to rad/s.
        theta_rad = theta * (np.pi / 180.0)

        # Create the body velocity vector [x, y, theta_rad].
        velocity_vector = np.array([x, y, theta_rad])

        # Calculate the effective radius for rotation
        effective_radius = np.sqrt((wheelbase/2)**2 + (track_width/2)**2)

        # Build the kinematic matrix for mechanum wheels
        # For mechanum wheels, turning requires all wheels to rotate in the same direction
        # Right wheels are inverted to match physical motor direction
        # Each row represents: [forward/backward, left/right, rotation]
        m = np.array([
            [1,  1, -effective_radius],  # Front-left wheel
            [-1, 1, -effective_radius],  # Front-right wheel (inverted for forward/backward, same for rotation)
            [1, -1, -effective_radius],  # Rear-left wheel
            [-1, -1, -effective_radius], # Rear-right wheel (inverted for forward/backward, same for rotation)
        ])

        # Compute each wheel's linear speed (m/s) and then its angular speed (rad/s).
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius

        # Convert wheel angular speeds from rad/s to deg/s.
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        # Convert to normalized speeds
        wheel_normalized = np.array([self._degps_to_normalized(deg) for deg in wheel_degps])

        # Scaling to respect maximum normalized command
        max_normalized_computed = np.max(np.abs(wheel_normalized))
        if max_normalized_computed > max_normalized:
            scale = max_normalized / max_normalized_computed
            wheel_normalized = wheel_normalized * scale

        return {
            "front_left": float(wheel_normalized[0]),
            "front_right": float(wheel_normalized[1]),
            "rear_left": float(wheel_normalized[2]),
            "rear_right": float(wheel_normalized[3]),
        }

    def _wheel_normalized_to_body(
        self,
        front_left,
        front_right,
        rear_left,
        rear_right,
        wheel_radius: float = 0.05,
        wheelbase: float = 0.25,  # Distance between front and rear wheels
        track_width: float = 0.25,  # Distance between left and right wheels
    ) -> dict[str, Any]:
        """
        Convert normalized wheel command feedback back into body-frame velocities for 4-wheel mechanum drive.

        Parameters:
          wheel_normalized: Vector with normalized wheel commands (front_left, front_right,
                           rear_left, rear_right) from -1.0 to 1.0.
          wheel_radius: Radius of each wheel (meters).
          wheelbase   : Distance between front and rear wheels (meters).
          track_width : Distance between left and right wheels (meters).

        Returns:
          A dict (x.vel, y.vel, theta.vel) all in m/s and deg/s
        """

        # Convert each normalized command back to an angular speed in deg/s.
        wheel_degps = np.array([
            self._normalized_to_degps(front_left),
            self._normalized_to_degps(front_right),
            self._normalized_to_degps(rear_left),
            self._normalized_to_degps(rear_right),
        ])

        # Convert from deg/s to rad/s.
        wheel_radps = wheel_degps * (np.pi / 180.0)
        # Compute each wheel's linear speed (m/s) from its angular speed.
        wheel_linear_speeds = wheel_radps * wheel_radius

        # Calculate the effective radius for rotation
        effective_radius = np.sqrt((wheelbase/2)**2 + (track_width/2)**2)

        # Build the kinematic matrix for mechanum wheels (same as forward kinematics)
        # Right wheels are inverted to match physical motor direction
        m = np.array([
            [1,  1, -effective_radius],  # Front-left wheel
            [-1, 1, -effective_radius],  # Front-right wheel (inverted for forward/backward, same for rotation)
            [1, -1, -effective_radius],  # Rear-left wheel
            [-1, -1, -effective_radius], # Rear-right wheel (inverted for forward/backward, same for rotation)
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



