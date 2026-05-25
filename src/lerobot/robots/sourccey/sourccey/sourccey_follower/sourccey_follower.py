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

import json
import logging
import os
import time
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors.feetech.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors.motors_bus import Motor, MotorNormMode
from lerobot.robots.robot import Robot
from lerobot.robots.sourccey.sourccey.sourccey_follower.config_sourccey_follower import SourcceyFollowerConfig
from lerobot.robots.sourccey.sourccey.sourccey_follower.sourccey_follower_calibrator import (
    SourcceyFollowerCalibrator,
)
from lerobot.robots.sourccey.sourccey.sourccey_follower.sourccey_follower_safety import SourcceyFollowerSafety
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)

class SourcceyFollower(Robot):
    config_class = SourcceyFollowerConfig
    name = "sourccey_follower"
    _STARTUP_TORQUE_ENABLE_RETRIES = 10
    _STARTUP_TORQUE_VERIFY_RETRIES = 3
    _STS_PHASE_ANGLE_FEEDBACK_BIT = 0x10
    _STARTUP_LOG_ENV = "LEROBOT_SOURCCEY_STARTUP_LOG_PATH"

    def __init__(self, config: SourcceyFollowerConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        motor_ids = [1, 2, 3, 4, 5, 6]
        if self.config.orientation == "right":
            motor_ids = [7, 8, 9, 10, 11, 12]

        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(motor_ids[0], config.motor_models["shoulder_pan"], norm_mode_body),
                "shoulder_lift": Motor(motor_ids[1], config.motor_models["shoulder_lift"], norm_mode_body),
                "elbow_flex": Motor(motor_ids[2], config.motor_models["elbow_flex"], norm_mode_body),
                "wrist_flex": Motor(motor_ids[3], config.motor_models["wrist_flex"], norm_mode_body),
                "wrist_roll": Motor(motor_ids[4], config.motor_models["wrist_roll"], norm_mode_body),
                "gripper": Motor(motor_ids[5], config.motor_models["gripper"], MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

        # Initialize calibrator
        self.calibrator = SourcceyFollowerCalibrator(
            robot=self
        )
        self.safety = SourcceyFollowerSafety(
            robot=self
        )

        # Track last warning time for throttling
        self._last_write_warning_time = 0.0
        self._write_warning_throttle_interval = 60.0  # seconds
        self._startup_safety_armed = False

    def __del__(self):
        # Destructors can run on partially initialized objects if __init__ raised.
        try:
            if hasattr(self, "bus") and self.is_connected:
                self.disconnect()
        except Exception:
            pass

    ###################################################################
    # Properties and Attributes
    ###################################################################
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

    ###################################################################
    # Connection Management
    ###################################################################
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

        self._startup_safety_armed = False
        self.bus.connect()
        if not self.is_calibrated:
            if self.calibration:
                logger.warning(
                    "Calibration mismatch detected at connect. Reapplying calibration file values to the motors."
                )
                self.bus.disable_torque()
                self.bus.write_calibration(self.calibration)
            elif calibrate:
                logger.info(
                    "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
                )
                self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    def disconnect(self) -> None:
         # Make disconnect idempotent: calling it twice should be harmless.
        if not self.is_connected:
            logger.info(f"{self} is not connected. Skipping disconnect.")
            return

        logger.info(f"Disconnecting Sourccey {self.config.orientation} Follower")
        self._startup_safety_armed = False

        self.bus.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    ###################################################################
    # Calibration and Configuration Management
    ###################################################################
    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """Perform manual calibration."""
        self.calibration = self.calibrator.manual_calibrate()

    def auto_calibrate(self, reverse: bool = False, full_reset: bool = False) -> None:
        """Perform automatic calibration."""
        if full_reset:
            self.calibration = self.calibrator.auto_calibrate(reverse=reverse)
        else:
            self.calibration = self.calibrator.default_calibrate(reverse=reverse)

    def configure(self) -> None:
        self._startup_safety_armed = False
        self.bus.disable_torque()
        try:
            # Normalize startup motor registers on every power-on.
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

                if motor == "gripper":
                    self.bus.write("P_Coefficient", motor, 24)
                    self.bus.write("I_Coefficient", motor, 0)
                    self.bus.write("D_Coefficient", motor, 48)
                    self.bus.write("Max_Torque_Limit", motor, 500)  # 50% of max torque to avoid burnout
                    self.bus.write("Protection_Current", motor, 400)  # 50% of max current to avoid burnout
                    self.bus.write("Overload_Torque", motor, 25)  # 25% torque when overloaded
                elif motor in {"shoulder_lift", "elbow_flex"}:
                    self.bus.write("P_Coefficient", motor, 12)
                    self.bus.write("I_Coefficient", motor, 0)
                    self.bus.write("D_Coefficient", motor, 48)  # Optimal damping (64 was too high)
                    self.bus.write("Max_Torque_Limit", motor, 2000)
                    self.bus.write("Protection_Current", motor, 4200)  # 4.2A for STS3250
                    self.bus.write("Overload_Torque", motor, 25)  # 25% torque when overloaded
                    self.bus.write("Minimum_Startup_Force", motor, 10)
                    self.bus.write("CW_Dead_Zone", motor, 2)
                    self.bus.write("CCW_Dead_Zone", motor, 2)
                    self.bus.write("Acceleration", motor, 180)
                else:
                    self.bus.write("P_Coefficient", motor, 16)
                    self.bus.write("I_Coefficient", motor, 2)
                    self.bus.write("D_Coefficient", motor, 32)
                    self.bus.write("Max_Torque_Limit", motor, 1500)  # 80% of max torque
                    self.bus.write("Protection_Current", motor, 1500)  # 80% of max current
                    self.bus.write("Overload_Torque", motor, 25)  # 25% torque when overloaded

            startup_raw_positions = self.bus.sync_read("Present_Position", normalize=False)
            startup_phase_faults = self._get_startup_phase_faults()
            startup_position_faults = self._get_startup_position_faults(startup_raw_positions)
            self._write_startup_diagnostic(
                status="precheck",
                startup_raw_positions=startup_raw_positions,
                startup_phase_faults=startup_phase_faults,
                startup_position_faults=startup_position_faults,
            )
            if startup_phase_faults or startup_position_faults:
                startup_error = self._build_startup_safety_error(
                    startup_phase_faults=startup_phase_faults,
                    startup_position_faults=startup_position_faults,
                )
                logger.error(startup_error)
                raise RuntimeError(startup_error)

            # Prime goal to current raw position before re-enabling torque to avoid startup jumps.
            # Keep startup physically static: prime to the literal raw position currently reported by the servo.
            # Reconciled values are only for software-side interpretation of wrap/branch ambiguity.
            self.bus.sync_write("Goal_Position", startup_raw_positions, normalize=False)

            self._enable_torque_with_verification()
            self._startup_safety_armed = True
            self._write_startup_diagnostic(
                status="armed",
                startup_raw_positions=startup_raw_positions,
                startup_phase_faults=startup_phase_faults,
                startup_position_faults=startup_position_faults,
            )
        except Exception:
            self._startup_safety_armed = False
            try:
                phase_faults = self._get_startup_phase_faults()
            except Exception:
                phase_faults = {}
            try:
                raw_positions = self.bus.sync_read("Present_Position", normalize=False)
            except Exception:
                raw_positions = {}
            try:
                position_faults = self._get_startup_position_faults(raw_positions)
            except Exception:
                position_faults = {}
            self._write_startup_diagnostic(
                status="failed",
                startup_raw_positions=raw_positions,
                startup_phase_faults=phase_faults,
                startup_position_faults=position_faults,
            )
            try:
                self.bus.disable_torque()
            except Exception as torque_error:
                logger.warning(f"Failed to disable torque after startup safety failure on {self}: {torque_error}")
            raise

    def _get_startup_phase_faults(self) -> dict[str, int]:
        phase_faults: dict[str, int] = {}
        for motor, motor_cfg in self.bus.motors.items():
            if not motor_cfg.model.startswith("sts"):
                continue
            phase = int(self.bus.read("Phase", motor, normalize=False))
            if phase & self._STS_PHASE_ANGLE_FEEDBACK_BIT:
                phase_faults[motor] = phase

        return phase_faults

    def _get_startup_position_faults(self, raw_positions: dict[str, int | float]) -> dict[str, tuple[float, int, int]]:
        position_faults: dict[str, tuple[float, int, int]] = {}
        for motor, raw_value in raw_positions.items():
            calibration = self.bus.calibration.get(motor)
            if calibration is None:
                continue

            low = calibration.range_min
            high = calibration.range_max
            raw = float(raw_value)
            resolution = self.bus.model_resolution_table[self.bus.motors[motor].model]
            candidates = (raw, raw + resolution, raw - resolution)
            if not any(low <= candidate <= high for candidate in candidates):
                position_faults[motor] = (raw, calibration.range_min, calibration.range_max)

        return position_faults

    def _read_torque_enable_snapshot(self) -> dict[str, int | None]:
        snapshot: dict[str, int | None] = {}
        for motor in self.bus.motors:
            try:
                snapshot[motor] = int(self.bus.read("Torque_Enable", motor, normalize=False))
            except Exception:
                snapshot[motor] = None
        return snapshot

    def _enable_torque_with_verification(self) -> None:
        last_error: Exception | None = None
        for attempt in range(self._STARTUP_TORQUE_VERIFY_RETRIES):
            try:
                self.bus.enable_torque(num_retry=self._STARTUP_TORQUE_ENABLE_RETRIES)
                return
            except ConnectionError as e:
                last_error = e
                torque_snapshot = self._read_torque_enable_snapshot()
                if all(value == 1 for value in torque_snapshot.values() if value is not None):
                    logger.warning(
                        f"{self} saw a transient torque-enable packet error, "
                        f"but torque readback is enabled: {torque_snapshot}"
                    )
                    return
                logger.warning(
                    f"{self} torque-enable attempt {attempt + 1}/{self._STARTUP_TORQUE_VERIFY_RETRIES} failed. "
                    f"Current torque snapshot={torque_snapshot}. Retrying."
                )

        # Preserve previous non-blocking behavior as much as possible:
        # if communication is noisy, allow startup to continue but report clearly.
        ping_snapshot = {}
        for motor, motor_cfg in self.bus.motors.items():
            try:
                ping_snapshot[motor] = self.bus.ping(motor_cfg.id)
            except Exception:
                ping_snapshot[motor] = None
        logger.error(
            f"{self} failed to confirm torque enable after retries. "
            f"Continuing startup to avoid blocking host. Ping snapshot={ping_snapshot}. "
            f"Last error={last_error}"
        )

    def _build_startup_safety_error(
        self,
        *,
        startup_phase_faults: dict[str, int],
        startup_position_faults: dict[str, tuple[float, int, int]],
    ) -> str:
        details = [
            f"{self} startup safety check failed. Refusing to enable torque.",
            (
                f"Action required: Recalibrate the {self.config.orientation} arm before the next run. "
                "Do not continue inference until recalibration is complete."
            ),
        ]

        if startup_phase_faults:
            details.append("STS phase faults (angle-feedback bit 0x10 still set):")
            for motor, phase in startup_phase_faults.items():
                details.append(f"  - {motor}: Phase=0x{phase:02x}")

        if startup_position_faults:
            details.append("Raw joint positions outside calibrated limits:")
            for motor, (raw, range_min, range_max) in startup_position_faults.items():
                details.append(f"  - {motor}: raw={raw:.0f}, range=[{range_min}, {range_max}]")

        return "\n".join(details)

    def _get_startup_diagnostic_path(self) -> Path:
        configured = os.environ.get(self._STARTUP_LOG_ENV, "").strip()
        if configured:
            return Path(configured).expanduser()
        return Path.home() / "sourccey_startup_diagnostics.jsonl"

    def _write_startup_diagnostic(
        self,
        *,
        status: str,
        startup_raw_positions: dict[str, int | float],
        startup_phase_faults: dict[str, int],
        startup_position_faults: dict[str, tuple[float, int, int]],
    ) -> None:
        calibration_snapshot = {}
        for motor, cal in self.bus.calibration.items():
            calibration_snapshot[motor] = {
                "id": cal.id,
                "range_min": cal.range_min,
                "range_max": cal.range_max,
                "homing_offset": cal.homing_offset,
                "drive_mode": cal.drive_mode,
            }

        raw_by_motor = {motor: float(val) for motor, val in startup_raw_positions.items()}
        phase_by_motor = {}
        for motor, motor_cfg in self.bus.motors.items():
            if not motor_cfg.model.startswith("sts"):
                continue
            try:
                phase_by_motor[motor] = int(self.bus.read("Phase", motor, normalize=False))
            except Exception:
                phase_by_motor[motor] = None

        diagnostic = {
            "ts_utc": datetime.now(UTC).isoformat(),
            "status": status,
            "orientation": self.config.orientation,
            "port": self.config.port,
            "is_calibrated": self.is_calibrated,
            "startup_safety_armed": self._startup_safety_armed,
            "raw_present_position": raw_by_motor,
            "phase_register": phase_by_motor,
            "phase_faults": startup_phase_faults,
            "position_faults": {
                motor: {"raw": raw, "range_min": range_min, "range_max": range_max}
                for motor, (raw, range_min, range_max) in startup_position_faults.items()
            },
            "calibration": calibration_snapshot,
        }

        try:
            log_path = self._get_startup_diagnostic_path()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(diagnostic, sort_keys=True) + "\n")
            logger.info(f"{self} startup diagnostics logged to {log_path} ({status=})")
        except Exception as e:
            logger.warning(f"Failed to write startup diagnostic log for {self}: {e}")

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    ###################################################################
    # Data Management
    ###################################################################
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
        if not self._startup_safety_armed:
            raise RuntimeError(f"{self} startup safety is not armed. Refusing to command motion.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        present_pos: dict[str, float] | None = None

        try:
            # Check for NaN values and skip sending actions if any are found
            present_pos = self.bus.sync_read("Present_Position")
            if any(np.isnan(v) for v in goal_pos.values()) or any(np.isnan(v) for v in present_pos.values()):
                logger.warning("NaN values detected in goal positions. Skipping action execution.")
                return {f"{motor}.pos": val for motor, val in present_pos.items()}

            # Cap goal position when too far away from present position.
            # /!\ Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
                goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

            # If a joint is already over current, avoid commanding it deeper into the obstruction.
            goal_pos = self.safety.apply_current_safety(goal_pos, present_pos)

            # Send goal position to the arm with error handling
            self.bus.sync_write("Goal_Position", goal_pos)
            self.safety.remember_goal(goal_pos)
            return {f"{motor}.pos": val for motor, val in goal_pos.items()}

        except ConnectionError as e:
            current_time = time.time()
            # Only log warning if enough time has passed since last warning
            if current_time - self._last_write_warning_time >= self._write_warning_throttle_interval:
                logger.warning(f"Status packet error during sync_read / sync_write in {self}: {e}. Returning present position.")
                self._last_write_warning_time = current_time
            # Return present position instead of goal position when write fails
            fallback_pos = present_pos if present_pos is not None else goal_pos
            return {f"{motor}.pos": val for motor, val in fallback_pos.items()}
