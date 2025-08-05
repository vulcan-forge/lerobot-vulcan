from functools import cached_property
import json
import time
from typing import Any
from venv import logger

import numpy as np
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors.feetech.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode
from lerobot.robots.robot import Robot
from lerobot.robots.sourccey.sourccey_v3beta.sourccey_v3beta_follower.calibration_sourccey_v3beta_follower import SourcceyV3BetaCalibrator
from lerobot.robots.sourccey.sourccey_v3beta.sourccey_v3beta_follower.config_sourccey_v3beta_follower import SourcceyV3BetaFollowerConfig
from lerobot.robots.utils import ensure_safe_goal_position
import os
from pathlib import Path

class SourcceyV3BetaFollower(Robot):
    config_class = SourcceyV3BetaFollowerConfig
    name = "sourccey_v3beta_follower"

    def __init__(self, config: SourcceyV3BetaFollowerConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        motor_ids = [1, 2, 3, 4, 5, 6]
        if self.config.orientation == "right":
            motor_ids = [7, 8, 9, 10, 11, 12]

        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(motor_ids[0], "sts3215", norm_mode_body),
                "shoulder_lift": Motor(motor_ids[1], "sts3250", norm_mode_body),
                "elbow_flex": Motor(motor_ids[2], "sts3215", norm_mode_body),
                "wrist_flex": Motor(motor_ids[3], "sts3215", norm_mode_body),
                "wrist_roll": Motor(motor_ids[4], "sts3215", norm_mode_body),
                "gripper": Motor(motor_ids[5], "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

        # Initialize calibrator
        self.calibrator = SourcceyV3BetaCalibrator(
            robot=self
        )

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """Perform manual calibration."""
        self.calibration = self.calibrator.manual_calibrate()

    def auto_calibrate(self, reversed: bool = False, full_reset: bool = False) -> None:
        """Perform automatic calibration."""
        self.calibration = self.calibrator.auto_calibrate(reversed=reversed, full_reset=full_reset)

    def configure(self) -> None:
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.bus.write("P_Coefficient", motor, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.bus.write("I_Coefficient", motor, 0)
            self.bus.write("D_Coefficient", motor, 32)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Check for NaN values and skip sending actions if any are found
        present_pos = self.bus.sync_read("Present_Position")
        if any(np.isnan(v) for v in goal_pos.values()) or any(np.isnan(v) for v in present_pos.values()):
            logger.warning("NaN values detected in goal positions. Skipping action execution.")
            return {f"{motor}.pos": val for motor, val in present_pos.items()}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_present_pos = self._apply_minimum_action(goal_present_pos)
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)

        # Check safety after sending goals
        overcurrent_motors = self._check_current_safety()
        if overcurrent_motors and len(overcurrent_motors) > 0:
            logger.warning(f"Safety triggered: {overcurrent_motors} current > {self.config.max_current_safety_threshold}mA")
            return self._handle_overcurrent_motors(overcurrent_motors, goal_pos, present_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def _apply_minimum_action(self, goal_present_pos: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
        """Apply a minimum action to the robot's geared down motors.

        This function ensures that geared-down motors receive a minimum movement threshold
        to overcome friction and backlash. If the desired movement is below the threshold,
        it's amplified to the minimum threshold while preserving direction.
        """
        # Define geared down motors and their minimum action thresholds
        geared_down_motors = ["shoulder_lift"]

        adjusted_goal_present_pos = {}

        for key, (goal_pos, present_pos) in goal_present_pos.items():
            motor_name = key.replace(".pos", "")
            if motor_name in geared_down_motors:
                desired_movement = goal_pos - present_pos
                movement_magnitude = abs(desired_movement)

                # If movement is below threshold, apply minimum action
                if movement_magnitude > 0 and movement_magnitude < self.config.min_action_threshold:
                    direction = 1 if desired_movement > 0 else -1
                    adjusted_movement = direction * self.config.min_action_threshold
                    adjusted_goal_pos = present_pos + adjusted_movement
                    adjusted_goal_present_pos[key] = (adjusted_goal_pos, present_pos)
                else:
                    adjusted_goal_present_pos[key] = (goal_pos, present_pos)
            else:
                adjusted_goal_present_pos[key] = (goal_pos, present_pos)
        return adjusted_goal_present_pos

    def _check_current_safety(self) -> list[str]:
        """
        Check if any motor is over current limit and return safety status.

        Returns:
            tuple: (is_safe, overcurrent_motors)
            - is_safe: True if all motors are under current limit
            - overcurrent_motors: List of motor names that are over current
        """
        # Read current from all motors
        currents = self.bus.sync_read("Present_Current")
        overcurrent_motors = []
        for motor, current in currents.items():
            if current > self.config.max_current_safety_threshold:
                overcurrent_motors.append(motor)
                logger.warning(f"Safety triggered: {motor} current {current}mA > {self.config.max_current_safety_threshold}mA")
        return overcurrent_motors

    def _handle_overcurrent_motors(
        self,
        overcurrent_motors: list[str],
        goal_pos: dict[str, float],
        present_pos: dict[str, float],
    ) -> dict[str, float]:
        """
        Handle overcurrent motors by replacing their goal positions with present positions.

        Args:
            goal_pos: Dictionary of goal positions with keys like "shoulder_pan.pos"
            present_pos: Dictionary of present positions with keys like "shoulder_pan.pos"
            overcurrent_motors: List of motor names that are over current (e.g., ["shoulder_pan", "elbow_flex"])

        Returns:
            Dictionary of goal positions with overcurrent motors replaced by present positions
        """
        if not overcurrent_motors or len(overcurrent_motors) == 0:
            return goal_pos

        # Create copies of the goal positions to modify
        modified_goal_pos = goal_pos.copy()
        for motor_name in overcurrent_motors:
            goal_key = f"{motor_name}.pos"
            if goal_key in modified_goal_pos:
                modified_goal_pos[goal_key] = present_pos[motor_name]
                logger.warning(f"Replaced goal position for {motor_name} with present position: {present_pos[motor_name]}")

        # Sync write the modified goal positions
        goal_pos_raw = {k.replace(".pos", ""): v for k, v in modified_goal_pos.items()}
        self.bus.sync_write("Goal_Position", goal_pos_raw)
        return modified_goal_pos

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")


