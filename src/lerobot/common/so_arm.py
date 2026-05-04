from dataclasses import dataclass

from lerobot.motors import Motor, MotorNormMode


@dataclass
class SOJointConfig:
    id: int
    model: str = "sts3215"
    drive_mode: int = 0
    homing_position: int | None = None
    range_min: int = 0
    range_max: int = 4095
    fixed_range: bool = False
    is_gripper: bool = False


def make_motor_bus_motors(motors: dict[str, SOJointConfig], *, use_degrees: bool) -> dict[str, Motor]:
    body_norm_mode = MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    return {
        name: Motor(
            cfg.id,
            cfg.model,
            MotorNormMode.RANGE_0_100 if cfg.is_gripper else body_norm_mode,
        )
        for name, cfg in motors.items()
    }


def make_so_follower_joint_configs() -> dict[str, SOJointConfig]:
    return {
        "shoulder_pan": SOJointConfig(id=1, homing_position=2048, range_min=1000, range_max=3095),
        "shoulder_lift": SOJointConfig(id=2, drive_mode=1, homing_position=795, range_min=800, range_max=3295),
        "elbow_flex": SOJointConfig(id=3, homing_position=3095, range_min=850, range_max=3345),
        "wrist_flex": SOJointConfig(id=4, homing_position=2895, range_min=750, range_max=3245),
        "wrist_roll": SOJointConfig(
            id=5,
            homing_position=2100,
            range_min=0,
            range_max=4095,
            fixed_range=True,
        ),
        "gripper": SOJointConfig(
            id=6,
            homing_position=2965,
            range_min=2023,
            range_max=3500,
            is_gripper=True,
        ),
    }


def make_so_leader_joint_configs() -> dict[str, SOJointConfig]:
    return {
        "shoulder_pan": SOJointConfig(id=1, homing_position=2048, range_min=1000, range_max=3095),
        "shoulder_lift": SOJointConfig(id=2, drive_mode=1, homing_position=770, range_min=800, range_max=3295),
        "elbow_flex": SOJointConfig(id=3, homing_position=3095, range_min=850, range_max=3345),
        "wrist_flex": SOJointConfig(id=4, homing_position=2760, range_min=750, range_max=3245),
        "wrist_roll": SOJointConfig(
            id=5,
            homing_position=2085,
            range_min=0,
            range_max=4095,
            fixed_range=True,
        ),
        "gripper": SOJointConfig(
            id=6,
            homing_position=2905,
            range_min=2023,
            range_max=3500,
            is_gripper=True,
        ),
    }


def make_so7_rprprpg_joint_configs(
    *,
    start_id: int = 1,
    model: str = "sts3215",
    homing_positions: dict[str, int | None] | None = None,
    range_mins: dict[str, int] | None = None,
    range_maxes: dict[str, int] | None = None,
    drive_modes: dict[str, int] | None = None,
    fixed_range_motors: set[str] | None = None,
) -> dict[str, SOJointConfig]:
    joint_names = (
        "roll_1",
        "pitch_1",
        "roll_2",
        "pitch_2",
        "roll_3",
        "pitch_3",
        "gripper",
    )
    homing_positions = homing_positions or {}
    range_mins = range_mins or {}
    range_maxes = range_maxes or {}
    drive_modes = drive_modes or {}
    fixed_range_motors = fixed_range_motors or set()

    motors: dict[str, SOJointConfig] = {}
    for index, joint_name in enumerate(joint_names):
        motors[joint_name] = SOJointConfig(
            id=start_id + index,
            model=model,
            drive_mode=drive_modes.get(joint_name, 0),
            homing_position=homing_positions.get(joint_name),
            range_min=range_mins.get(joint_name, 0),
            range_max=range_maxes.get(joint_name, 4095),
            fixed_range=joint_name in fixed_range_motors,
            is_gripper=joint_name == "gripper",
        )

    return motors
