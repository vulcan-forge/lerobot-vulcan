import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple
from venv import logger

from lerobot.motors.feetech.feetech import OperatingMode
from lerobot.motors.motors_bus import MotorCalibration

class SOFollowerCalibrator:
    """Handles calibration operations for Sourccey robots."""

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
            # Calibration file exists, ask user whether to use it or run new calibration
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
        homing_offsets = self.robot.bus.set_half_turn_homings()

        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.robot.bus.motors if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.robot.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.robot.calibration = self._create_calibration_dict(homing_offsets, range_mins, range_maxes)
        self.robot.bus.write_calibration(self.robot.calibration)
        self._save_calibration()
        print("Calibration saved to", self.robot.calibration_fpath)
        return self.robot.calibration

    def _create_calibration_dict(self, homing_offsets: Dict[str, int],
                                range_mins: Dict[str, Any], range_maxes: Dict[str, int] = None) -> Dict[str, MotorCalibration]:
        calibration = {}
        for motor, m in self.robot.bus.motors.items():
            drive_mode = 0
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
        """Initialize the calibration of the robot."""
        # Set all motors to half turn homings except shoulder_lift
        homing_offsets = self.robot.bus.set_position_homings({
            "shoulder_pan": 2047 if reverse else 2048,
            "shoulder_lift": 3300 if reverse else 795,
            "elbow_flex": 1000 if reverse else 3095,
            "wrist_flex": 1200 if reverse else 2895,
            "wrist_roll": 1995 if reverse else 2100,
            "gripper": 1130 if reverse else 2965
        })
        return homing_offsets

    def _load_default_calibration(self, reverse: bool = False) -> Dict[str, Any]:
        """Load default calibration from file."""
        # Get the directory of the current file
        current_dir = Path(__file__).parent
        calibration_dir = current_dir.parent / "so100_follower"
        calibration_file = calibration_dir / "default_calibration.json"

        # Create the calibration directory if it doesn't exist
        calibration_dir.mkdir(parents=True, exist_ok=True)

        # If the calibration file doesn't exist, create it with default values
        if not calibration_file.exists():
            logger.info(f"Calibration file {calibration_file} not found. Creating default calibration...")
            default_calibration = self._create_default_calibration(reverse)
            with open(calibration_file, "w") as f:
                json.dump(default_calibration, f, indent=4)
            logger.info(f"Created default calibration file: {calibration_file}")

        with open(calibration_file, "r") as f:
            return json.load(f)

    def _create_default_calibration(self, reverse: bool = False) -> Dict[str, Any]:
        """Create default calibration data for the robot."""

        return {
            "shoulder_pan": {
                "id": 1,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 1000,
                "range_max": 3095
            },
            "shoulder_lift": {
                "id": 2,
                "drive_mode": 1,
                "homing_offset": 0,
                "range_min": 800,
                "range_max": 3295
            },
            "elbow_flex": {
                "id": 3,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 850,
                "range_max": 3345
            },
            "wrist_flex": {
                "id": 4,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 750,
                "range_max": 3245
            },
            "wrist_roll": {
                "id": 5,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 0,
                "range_max": 4095
            },
            "gripper": {
                "id": 6,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 2023,
                "range_max": 3500
            }
        }

    def _save_calibration(self) -> None:
        """Save calibration to file."""
        self.robot.bus.write_calibration(self.robot.calibration)
        self.robot._save_calibration()
