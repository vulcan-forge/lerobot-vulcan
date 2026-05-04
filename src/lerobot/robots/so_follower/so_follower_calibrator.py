from typing import Any, Dict

import logging

from lerobot.motors.feetech.feetech import OperatingMode
from lerobot.motors.motors_bus import MotorCalibration

logger = logging.getLogger(__name__)


class SOFollowerCalibrator:
    """Handles calibration operations for SO follower robots."""

    def __init__(self, robot):
        self.robot = robot

    def default_calibrate(self, reverse: bool = False) -> Dict[str, MotorCalibration]:
        """Perform default calibration."""

        homing_offsets = self._initialize_calibration(reverse)

        min_ranges = {}
        max_ranges = {}
        default_calibration = self._load_default_calibration(reverse)
        for motor, m in self.robot.bus.motors.items():
            min_ranges[motor] = default_calibration[motor]["range_min"]
            max_ranges[motor] = default_calibration[motor]["range_max"]

        self.robot.calibration = self._create_calibration_dict(homing_offsets, min_ranges, max_ranges)
        self.robot.bus.write_calibration(self.robot.calibration)
        self._save_calibration()
        logger.info(f"Default calibration completed and saved to {self.robot.calibration_fpath}")
        return self.robot.calibration

    def manual_calibrate(self) -> Dict[str, MotorCalibration]:
        """Perform manual calibration with user interaction."""
        if self.robot.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.robot.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.robot.id} to the motors")
                self.robot.bus.write_calibration(self.robot.calibration)
                return self.robot.calibration

        logger.info(f"\nRunning calibration of robot {self.robot.id}")
        self.robot.bus.disable_torque()
        for motor in self.robot.bus.motors:
            self.robot.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move robot to the middle of its range of motion and press ENTER....")
        homing_offsets = self._initialize_calibration()
        range_mins, range_maxes = self._record_or_fill_ranges()

        self.robot.calibration = self._create_calibration_dict(homing_offsets, range_mins, range_maxes)
        self.robot.bus.write_calibration(self.robot.calibration)
        self._save_calibration()
        print("Calibration saved to", self.robot.calibration_fpath)
        return self.robot.calibration

    def _create_calibration_dict(
        self,
        homing_offsets: Dict[str, int],
        range_mins: Dict[str, Any],
        range_maxes: Dict[str, int] | None = None,
    ) -> Dict[str, MotorCalibration]:
        calibration = {}
        for motor, m in self.robot.bus.motors.items():
            drive_mode = self.robot.config.motors[motor].drive_mode
            range_min = range_mins[motor]
            range_max = range_maxes[motor]
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=drive_mode,
                homing_offset=homing_offsets[motor],
                range_min=range_min,
                range_max=range_max,
            )
        return calibration

    def _initialize_calibration(self, reverse: bool = False) -> Dict[str, int]:
        """Initialize calibration using configured homing targets."""
        target_positions = {
            motor: cfg.homing_position
            for motor, cfg in self.robot.config.motors.items()
            if cfg.homing_position is not None
        }
        homing_offsets: Dict[str, int] = {}
        if target_positions:
            homing_offsets.update(self.robot.bus.set_position_homings(target_positions))

        remaining_motors = [motor for motor in self.robot.bus.motors if motor not in homing_offsets]
        if remaining_motors:
            homing_offsets.update(self.robot.bus.set_half_turn_homings(remaining_motors))

        return homing_offsets

    def _load_default_calibration(self, reverse: bool = False) -> Dict[str, Any]:
        """Load default calibration from the current config."""
        return self._create_default_calibration(reverse)

    def _create_default_calibration(self, reverse: bool = False) -> Dict[str, Any]:
        """Create default calibration data from the configured joint layout."""
        return {
            motor: {
                "id": cfg.id,
                "drive_mode": cfg.drive_mode,
                "homing_offset": 0,
                "range_min": cfg.range_min,
                "range_max": cfg.range_max,
            }
            for motor, cfg in self.robot.config.motors.items()
        }

    def _record_or_fill_ranges(self) -> tuple[dict[str, int], dict[str, int]]:
        record_range_motors = [motor for motor, cfg in self.robot.config.motors.items() if not cfg.fixed_range]

        if record_range_motors:
            print(
                "Move the following joints sequentially through their entire ranges of motion.\n"
                f"Recording positions for: {', '.join(record_range_motors)}.\n"
                "Press ENTER to stop..."
            )
            range_mins, range_maxes = self.robot.bus.record_ranges_of_motion(record_range_motors)
        else:
            range_mins, range_maxes = {}, {}

        for motor, cfg in self.robot.config.motors.items():
            if cfg.fixed_range:
                range_mins[motor] = cfg.range_min
                range_maxes[motor] = cfg.range_max

        return range_mins, range_maxes

    def _save_calibration(self) -> None:
        """Save calibration to file."""
        self.robot.bus.write_calibration(self.robot.calibration)
        self.robot._save_calibration()
