import json
from pathlib import Path

from lerobot.robots.sourccey.sourccey.motor_layout import get_sourccey_arm_motor_ids


def test_sourccey_seven_motor_layout_matches_expected_ids():
    assert get_sourccey_arm_motor_ids("left") == {
        "shoulder_pan": 1,
        "shoulder_lift": 2,
        "elbow_twist": 3,
        "elbow_flex": 4,
        "wrist_flex": 5,
        "wrist_roll": 6,
        "gripper": 7,
    }
    assert get_sourccey_arm_motor_ids("right") == {
        "shoulder_pan": 8,
        "shoulder_lift": 9,
        "elbow_twist": 10,
        "elbow_flex": 11,
        "wrist_flex": 12,
        "wrist_roll": 13,
        "gripper": 14,
    }


def test_sourccey_default_calibration_files_use_seven_motor_layout():
    repo_root = Path(__file__).resolve().parents[3]
    calibration_files = [
        repo_root / "src/lerobot/teleoperators/sourccey/sourccey/sourccey_leader/defaults/left_arm_default_calibration.json",
        repo_root / "src/lerobot/teleoperators/sourccey/sourccey/sourccey_leader/defaults/right_arm_default_calibration.json",
        repo_root / "src/lerobot/robots/sourccey/sourccey/sourccey/defaults/left_arm_default_calibration.json",
        repo_root / "src/lerobot/robots/sourccey/sourccey/sourccey/defaults/right_arm_default_calibration.json",
    ]

    expected_ids = {
        "left_arm_default_calibration.json": get_sourccey_arm_motor_ids("left"),
        "right_arm_default_calibration.json": get_sourccey_arm_motor_ids("right"),
    }

    for calibration_file in calibration_files:
        data = json.loads(calibration_file.read_text())
        arm_expected_ids = expected_ids[calibration_file.name]
        assert {joint: values["id"] for joint, values in data.items()} == arm_expected_ids
