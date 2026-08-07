# Copyright 2026 Vulcan Robotics, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Repair swapped Sourccey follower-arm motor IDs.

The canonical layout is left arm IDs 7-12 and right arm IDs 1-6. The command is
safe to rerun: it scans both the old and desired ID for every joint, refuses
ambiguous layouts, and only changes IDs that are still in the old range.

Run ``sourccey-fix-arm-motor-ids`` for a read-only preflight, then rerun it
with ``--apply`` to apply the displayed repair plan.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import get_address
from lerobot.robots.sourccey.sourccey.sourccey_follower.config_sourccey_follower import (
    sourccey_arm_motor_ids,
    sourccey_motor_models,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


DEFAULT_PORTS = {"left": "/dev/robotLeftArm", "right": "/dev/robotRightArm"}
MOTOR_NAMES = tuple(sourccey_motor_models())


@dataclass(frozen=True)
class MotorIdRepair:
    motor: str
    model: str
    current_id: int
    target_id: int

    @property
    def needs_change(self) -> bool:
        return self.current_id != self.target_id


@dataclass(frozen=True)
class ArmRepairPlan:
    arm: str
    port: str
    motors: tuple[MotorIdRepair, ...]

    @property
    def changes(self) -> tuple[MotorIdRepair, ...]:
        return tuple(motor for motor in self.motors if motor.needs_change)


def plan_arm_repair(
    arm: str,
    port: str,
    discovered: Mapping[int, int],
) -> ArmRepairPlan:
    """Validate a bus scan and return an idempotent old-to-canonical repair plan."""
    target_ids = sourccey_arm_motor_ids(arm)
    old_ids = sourccey_arm_motor_ids("right" if arm == "left" else "left")
    motor_models = sourccey_motor_models()
    expected_model_numbers = FeetechMotorsBus.model_number_table
    candidate_ids = set(target_ids) | set(old_ids)
    errors: list[str] = []
    unexpected_ids = sorted(set(discovered) - candidate_ids)
    if unexpected_ids:
        errors.append(f"unexpected motor IDs present: {unexpected_ids}")

    repairs: list[MotorIdRepair] = []
    for motor, old_id, target_id in zip(MOTOR_NAMES, old_ids, target_ids, strict=True):
        responding_ids = [motor_id for motor_id in (old_id, target_id) if motor_id in discovered]
        if len(responding_ids) != 1:
            if responding_ids:
                errors.append(f"{motor}: both old ID {old_id} and target ID {target_id} respond")
            else:
                errors.append(f"{motor}: neither old ID {old_id} nor target ID {target_id} responds")
            continue

        current_id = responding_ids[0]
        model = motor_models[motor]
        expected_model_number = expected_model_numbers[model]
        actual_model_number = discovered[current_id]
        if actual_model_number != expected_model_number:
            errors.append(
                f"{motor}: ID {current_id} has model number {actual_model_number}, "
                f"expected {expected_model_number} ({model})"
            )
            continue
        repairs.append(MotorIdRepair(motor, model, current_id, target_id))

    if errors:
        details = "\n  - ".join(errors)
        raise RuntimeError(f"Unsafe {arm} arm layout on {port}; no IDs were changed:\n  - {details}")

    return ArmRepairPlan(arm, port, tuple(repairs))


def discover_arm_motors(bus: FeetechMotorsBus) -> dict[int, int]:
    """Discover IDs 1-12, falling back when Feetech broadcast replies are unavailable."""
    discovered = bus.broadcast_ping(num_retry=0)
    if discovered is not None:
        return discovered

    print(f"Broadcast discovery returned no status packet on {bus.port}; trying individual IDs 1-12.")
    return {
        motor_id: model_number
        for motor_id in range(1, 13)
        if (model_number := bus.ping(motor_id, num_retry=0)) is not None
    }


def scan_arm(arm: str, port: str) -> tuple[FeetechMotorsBus, ArmRepairPlan]:
    bus = FeetechMotorsBus(port=port, motors={})
    bus.connect(handshake=False)
    try:
        discovered = discover_arm_motors(bus)
        return bus, plan_arm_repair(arm, port, discovered)
    except BaseException:
        bus.disconnect(disable_torque=False)
        raise


def print_plan(plan: ArmRepairPlan) -> None:
    print(f"\n{plan.arm.capitalize()} arm ({plan.port})")
    for motor in plan.motors:
        action = (
            f"change {motor.current_id} -> {motor.target_id}"
            if motor.needs_change
            else f"keep {motor.target_id}"
        )
        print(f"  {motor.motor:14} {action}")


def disable_plan_torque(bus: FeetechMotorsBus, plan: ArmRepairPlan) -> None:
    for motor in plan.motors:
        bus._disable_torque(motor.current_id, motor.model, num_retry=2)


def apply_plan(bus: FeetechMotorsBus, plan: ArmRepairPlan) -> None:
    for motor in plan.changes:
        address, length = get_address(bus.model_ctrl_table, motor.model, "ID")
        bus._write(
            address,
            length,
            motor.current_id,
            motor.target_id,
            num_retry=2,
            raise_on_error=True,
            err_msg=f"Failed to change {plan.arm} {motor.motor} from ID {motor.current_id} to {motor.target_id}.",
        )

        found_model = bus.ping(motor.target_id, num_retry=2)
        expected_model = bus.model_number_table[motor.model]
        if found_model != expected_model or bus.ping(motor.current_id, num_retry=1) is not None:
            raise RuntimeError(
                f"Could not verify {plan.arm} {motor.motor} ID change "
                f"{motor.current_id} -> {motor.target_id}. Rerun the command to safely inspect the partial state."
            )
        print(f"Updated {plan.arm} {motor.motor}: {motor.current_id} -> {motor.target_id}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--left-port", default=DEFAULT_PORTS["left"])
    parser.add_argument("--right-port", default=DEFAULT_PORTS["right"])
    parser.add_argument("--arm", choices=("left", "right", "both"), default="both")
    parser.add_argument(
        "--apply", action="store_true", help="Apply the repair. Without this flag, only scan."
    )
    parser.add_argument(
        "--yes", action="store_true", help="Skip the interactive confirmation used with --apply."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    selected_arms = ("left", "right") if args.arm == "both" else (args.arm,)
    ports = {"left": args.left_port, "right": args.right_port}
    if len(selected_arms) == 2 and ports["left"] == ports["right"]:
        raise ValueError("Left and right arm ports must be different.")

    open_buses: list[FeetechMotorsBus] = []
    plans: list[ArmRepairPlan] = []
    try:
        # Scan every selected arm before mutating either one.
        for arm in selected_arms:
            bus, plan = scan_arm(arm, ports[arm])
            open_buses.append(bus)
            plans.append(plan)
            print_plan(plan)

        total_changes = sum(len(plan.changes) for plan in plans)
        if not args.apply:
            print(
                f"\nPreflight passed: {total_changes} motor ID change(s) needed. Rerun with --apply to proceed."
            )
            return
        if total_changes == 0:
            print("\nMotor IDs already match the canonical Sourccey layout; nothing to change.")
            return
        if not args.yes:
            confirmation = input(f"\nType APPLY to change {total_changes} motor ID(s): ")
            if confirmation != "APPLY":
                print("Cancelled; no IDs were changed.")
                return

        # Stop every selected arm before changing any persistent IDs.
        for bus, plan in zip(open_buses, plans, strict=True):
            disable_plan_torque(bus, plan)
        for bus, plan in zip(open_buses, plans, strict=True):
            apply_plan(bus, plan)

        print("\nMotor ID repair complete. Both arms were left untorqued.")
    finally:
        for bus in open_buses:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
