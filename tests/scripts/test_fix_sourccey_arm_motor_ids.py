import re

import pytest

from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.robots.sourccey.sourccey.sourccey_follower.config_sourccey_follower import (
    sourccey_arm_motor_ids,
    sourccey_motor_models,
)
from lerobot.scripts.sourccey.configs.fix_arm_motor_ids import MOTOR_NAMES, plan_arm_repair


def _discovered(ids: tuple[int, ...]) -> dict[int, int]:
    models = sourccey_motor_models()
    return {
        motor_id: FeetechMotorsBus.model_number_table[models[motor]]
        for motor, motor_id in zip(MOTOR_NAMES, ids, strict=True)
    }


def test_canonical_sourccey_arm_motor_ids() -> None:
    assert sourccey_arm_motor_ids("left") == (7, 8, 9, 10, 11, 12)
    assert sourccey_arm_motor_ids("right") == (1, 2, 3, 4, 5, 6)


def test_plan_repairs_fully_swapped_left_arm() -> None:
    plan = plan_arm_repair("left", "/dev/left", _discovered((1, 2, 3, 4, 5, 6)))

    assert [(repair.current_id, repair.target_id) for repair in plan.changes] == [
        (1, 7),
        (2, 8),
        (3, 9),
        (4, 10),
        (5, 11),
        (6, 12),
    ]


def test_plan_repairs_fully_swapped_right_arm() -> None:
    plan = plan_arm_repair("right", "/dev/right", _discovered((7, 8, 9, 10, 11, 12)))

    assert [(repair.current_id, repair.target_id) for repair in plan.changes] == [
        (7, 1),
        (8, 2),
        (9, 3),
        (10, 4),
        (11, 5),
        (12, 6),
    ]


def test_plan_is_idempotent_for_correct_layout() -> None:
    plan = plan_arm_repair("left", "/dev/left", _discovered((7, 8, 9, 10, 11, 12)))

    assert plan.changes == ()


def test_plan_can_resume_partial_repair() -> None:
    plan = plan_arm_repair("left", "/dev/left", _discovered((7, 2, 9, 4, 11, 6)))

    assert [(repair.current_id, repair.target_id) for repair in plan.changes] == [(2, 8), (4, 10), (6, 12)]


@pytest.mark.parametrize(
    ("discovered", "message"),
    [
        ({1: 777, 7: 777}, "both old ID 1 and target ID 7 respond"),
        ({}, "neither old ID 1 nor target ID 7 responds"),
        ({13: 777}, "unexpected motor IDs present: [13]"),
    ],
)
def test_plan_rejects_ambiguous_or_incomplete_layout(discovered: dict[int, int], message: str) -> None:
    with pytest.raises(RuntimeError, match=re.escape(message)):
        plan_arm_repair("left", "/dev/left", discovered)


def test_plan_rejects_wrong_motor_model() -> None:
    discovered = _discovered((1, 2, 3, 4, 5, 6))
    discovered[1] = FeetechMotorsBus.model_number_table["sts3250"]

    with pytest.raises(RuntimeError, match=r"expected 777 \(sts3215\)"):
        plan_arm_repair("left", "/dev/left", discovered)


def test_unknown_orientation_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported Sourccey arm orientation"):
        sourccey_arm_motor_ids("center")
