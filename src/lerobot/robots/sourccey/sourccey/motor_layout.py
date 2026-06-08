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
        "gripper": 13,
    },
    "right": {
        "shoulder_pan": 7,
        "shoulder_lift": 8,
        "elbow_twist": 9,
        "elbow_flex": 10,
        "wrist_flex": 11,
        "wrist_roll": 12,
        "gripper": 14,
    },
}


def get_sourccey_arm_motor_ids(orientation: ArmOrientation) -> dict[str, int]:
    return SOURCCEY_ARM_MOTOR_IDS[orientation].copy()
