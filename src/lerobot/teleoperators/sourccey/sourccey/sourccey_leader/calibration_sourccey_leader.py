import json
import logging
from pathlib import Path
from typing import Any, Dict

from lerobot.motors.motors_bus import MotorCalibration


logger = logging.getLogger(__name__)


class SourcceyLeaderCalibrator:
    """Handles calibration operations for Sourccey leader teleoperators."""

    def __init__(self, teleop):
        self.teleop = teleop

    def auto_calibrate(self, reversed: bool = False, full_reset: bool = False, com_port_1: str = None, com_port_2: str = None) -> Dict[str, MotorCalibration]:
        """
        This function will calibrate the teleoperator arm.
        It will first set the homing offsets based on the starting position of the arm.
        Then it will record the range of motion of the arm.
        Then it will create a calibration dictionary and save it to the file.

        Args:
            reversed: If True, the arm will be calibrated in the reversed direction.
        """
        print(f"Starting automatic calibration of robot {self.teleop.id}")

        # Step 1: Adjust calibration so current positions become desired logical positions
        print("Adjusting calibration to align current positions with desired logical positions...")
        homing_offsets = self._initialize_calibration(reversed)

        detected_ranges = {}

        default_calibration = self._load_default_calibration(reversed)
        for motor, m in self.teleop.bus.motors.items():
            detected_ranges[motor] = {
                "min": default_calibration[motor]["range_min"],
                "max": default_calibration[motor]["range_max"],
            }

        # Step 4: Create calibration dictionary
        self.teleop.calibration = self._create_calibration_dict(homing_offsets, detected_ranges)

        # Step 5: Write calibration to motors and save
        self.teleop.bus.write_calibration(self.teleop.calibration)
        self._save_calibration()
        print(f"Automatic calibration completed and saved to {self.teleop.calibration_fpath}")
        return self.teleop.calibration

    def _initialize_calibration(self, reversed: bool = False) -> Dict[str, int]:
        """Initialize the calibration of the robot."""

        shoulder_pan_homing_offset = self.teleop.bus.set_position_homings(
            {"shoulder_pan": 2474 if reversed else 1554},
            {"shoulder_lift": 483 if reversed else 3667},
            {"elbow_flex": 3884 if reversed else 151},
            {"wrist_flex": 6086 if reversed else 2144},
            {"wrist_roll": 2078 if reversed else 2069},
            {"gripper": 5571 if reversed else 2725}
        )

        # Set the homing offsets for the motors
        homing_offsets = {
            "shoulder_pan": shoulder_pan_homing_offset["shoulder_pan"],
            "shoulder_lift": shoulder_pan_homing_offset["shoulder_lift"],
            "elbow_flex": shoulder_pan_homing_offset["elbow_flex"],
            "wrist_flex": shoulder_pan_homing_offset["wrist_flex"],
            "wrist_roll": shoulder_pan_homing_offset["wrist_roll"],
            "gripper": shoulder_pan_homing_offset["gripper"]
        }

        return homing_offsets

    def _create_calibration_dict(self, homing_offsets: Dict[str, int],
                                range_mins: Dict[str, Any], range_maxes: Dict[str, int] = None) -> Dict[str, MotorCalibration]:

        calibration = {}
        for motor, m in self.teleop.bus.motors.items():
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

    def _load_default_calibration(self, reversed: bool = False) -> Dict[str, Any]:
        """Load default calibration from file."""
        # Get the directory of the current file
        current_dir = Path(__file__).parent
        # Navigate to the sourccey directory where the calibration files are located
        calibration_dir = current_dir.parent / "sourccey_leader"

        print("Calibration directory: ", calibration_dir)

        if reversed:
            calibration_file = calibration_dir / "right_arm_default_calibration.json"
        else:
            calibration_file = calibration_dir / "left_arm_default_calibration.json"

        # Create the calibration directory if it doesn't exist
        calibration_dir.mkdir(parents=True, exist_ok=True)

        # If the calibration file doesn't exist, create it with default values
        if not calibration_file.exists():
            logger.info(f"Calibration file {calibration_file} not found. Creating default calibration...")
            default_calibration = self._create_default_calibration(reversed)
            with open(calibration_file, "w") as f:
                json.dump(default_calibration, f, indent=4)
            logger.info(f"Created default calibration file: {calibration_file}")

        with open(calibration_file, "r") as f:
            return json.load(f)

    def _create_default_calibration(self, reversed: bool = False) -> Dict[str, Any]:
        """Create default calibration data for the robot."""
        print("Creating default calibration data for the robot... CURRENTLY NOT CORRECT WILL BE FIXING LATER")
        if reversed:
            # Right arm calibration (IDs 7-12)
            return {
                "shoulder_pan": {
                    "id": 7,
                    "drive_mode": 0,
                    "homing_offset": -1849,
                    "range_min": 1021,
                    "range_max": 3058
                },
                "shoulder_lift": {
                    "id": 8,
                    "drive_mode": 0,
                    "homing_offset": 1520,
                    "range_min": 434,
                    "range_max": 3443
                },
                "elbow_flex": {
                    "id": 9,
                    "drive_mode": 0,
                    "homing_offset": -954,
                    "range_min": 223,
                    "range_max": 3898
                },
                "wrist_flex": {
                    "id": 10,
                    "drive_mode": 0,
                    "homing_offset": -2038,
                    "range_min": 1035,
                    "range_max": 3245
                },
                "wrist_roll": {
                    "id": 11,
                    "drive_mode": 0,
                    "homing_offset": -516,
                    "range_min": 0,
                    "range_max": 4095
                },
                "gripper": {
                    "id": 12,
                    "drive_mode": 0,
                    "homing_offset": -2009,
                    "range_min": 1471,
                    "range_max": 3019
                }
            }
        else:
            # Left arm calibration (IDs 1-6)
            return {
                "shoulder_pan": {
                    "id": 1,
                    "drive_mode": 0,
                    "homing_offset": -1253,
                    "range_min": 1028,
                    "range_max": 3056
                },
                "shoulder_lift": {
                    "id": 2,
                    "drive_mode": 0,
                    "homing_offset": -136,
                    "range_min": 378,
                    "range_max": 3578
                },
                "elbow_flex": {
                    "id": 3,
                    "drive_mode": 0,
                    "homing_offset": 1964,
                    "range_min": 109,
                    "range_max": 3787
                },
                "wrist_flex": {
                    "id": 4,
                    "drive_mode": 0,
                    "homing_offset": -1080,
                    "range_min": 1020,
                    "range_max": 3157
                },
                "wrist_roll": {
                    "id": 5,
                    "drive_mode": 0,
                    "homing_offset": -215,
                    "range_min": 0,
                    "range_max": 4095
                },
                "gripper": {
                    "id": 6,
                    "drive_mode": 0,
                    "homing_offset": -1660,
                    "range_min": 1215,
                    "range_max": 2730
                }
            }

    def _save_calibration(self) -> None:
        """Save calibration to file."""
        self.teleop.bus.write_calibration(self.teleop.calibration)
        self.teleop._save_calibration()
