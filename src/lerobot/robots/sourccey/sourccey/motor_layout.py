"""Shared motor layout for Sourccey arms."""

from typing import Literal


ArmOrientation = Literal["left", "right"]

SOURCCEY_ARM_MOTOR_IDS: dict[ArmOrientation, dict[str, int]] = {
    "left": {
        "shoulder_pan": 1,
        "shoulder_lift": 2,
        "elbow_twist": 3,
        "elbow_flex": 4,
        "wrist_flex": 5,
        "wrist_roll": 6,
        "gripper": 7,
    },
    "right": {
        "shoulder_pan": 8,
        "shoulder_lift": 9,
        "elbow_twist": 10,
        "elbow_flex": 11,
        "wrist_flex": 12,
        "wrist_roll": 13,
        "gripper": 14,
    },
}


def get_sourccey_arm_motor_ids(orientation: ArmOrientation) -> dict[str, int]:
    return SOURCCEY_ARM_MOTOR_IDS[orientation].copy()
