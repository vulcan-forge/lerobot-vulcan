#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.cameras import make_cameras_from_configs
from lerobot.common.so_arm import make_motor_bus_motors
from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_new_arm import NewBotRobotConfig

logger = logging.getLogger(__name__)


class NewBot(Robot):
    """Seven-DOF Feetech robot using the NewBot joint layout."""

    config_class = NewBotRobotConfig
    name = "new_bot"

    def __init__(self, config: NewBotRobotConfig):
        super().__init__(config)
        self.config = config
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors=make_motor_bus_motors(config.motors, use_degrees=config.use_degrees),
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        self._wrap_guard_motors = tuple(self.bus.motors)
        self._wrap_guard_state: dict[str, str | None] = {motor: None for motor in self._wrap_guard_motors}
        self._last_effective_raw: dict[str, int] = {}

    def _apply_wrap_guards(self, raw_positions: dict[str, int]) -> dict[str, int]:
        if not self.calibration:
            return raw_positions

        guarded_positions: dict[str, int] = {}
        for motor, value in raw_positions.items():
            calibration = self.calibration.get(motor)
            if calibration is None:
                guarded_positions[motor] = value
                continue

            min_ = calibration.range_min
            max_ = calibration.range_max
            bounded_value = min(max_, max(min_, value))

            state = self._wrap_guard_state.get(motor)
            last_value = self._last_effective_raw.get(motor)
            band = max(32, int((max_ - min_) * 0.08))
            high_band = max_ - band
            low_band = min_ + band

            if state == "high":
                if high_band <= value <= max_:
                    state = None
                    effective_value = value
                else:
                    effective_value = max_
            elif state == "low":
                if min_ <= value <= low_band:
                    state = None
                    effective_value = value
                else:
                    effective_value = min_
            elif last_value is not None and last_value >= high_band and value <= low_band:
                state = "high"
                effective_value = max_
            elif last_value is not None and last_value <= low_band and value >= high_band:
                state = "low"
                effective_value = min_
            else:
                effective_value = bounded_value

            self._wrap_guard_state[motor] = state
            guarded_positions[motor] = effective_value
            self._last_effective_raw[motor] = effective_value

        return guarded_positions

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

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
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
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        fixed_range_motors = {name for name, cfg in self.config.motors.items() if cfg.fixed_range}
        unknown_range_motors = [motor for motor in self.bus.motors if motor not in fixed_range_motors]
        if fixed_range_motors:
            fixed_range_text = "', '".join(sorted(fixed_range_motors))
            print(
                f"Move all joints except '{fixed_range_text}' sequentially through their "
                "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
            )
        else:
            print("Move all joints sequentially through their entire ranges of motion.\nRecording positions. Press ENTER to stop...")

        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        for motor in fixed_range_motors:
            range_mins[motor] = self.config.motors[motor].range_min
            range_maxes[motor] = self.config.motors[motor].range_max

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=self.config.motors[motor].drive_mode,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                self.bus.write("P_Coefficient", motor, 16)
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

                if self.config.motors[motor].is_gripper:
                    self.bus.write("Max_Torque_Limit", motor, 500)
                    self.bus.write("Protection_Current", motor, 250)
                    self.bus.write("Overload_Torque", motor, 25)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        raw_positions = self.bus.sync_read("Present_Position", normalize=False)
        guarded_positions = self._apply_wrap_guards(raw_positions)
        ids_values = {self.bus.motors[motor].id: int(value) for motor, value in guarded_positions.items()}
        normalized_positions = self.bus._normalize(ids_values)
        obs_dict = {f"{motor}.pos": normalized_positions[self.bus.motors[motor].id] for motor in guarded_positions}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self):
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")


NewArm = NewBot
