#!/usr/bin/env python3
"""
Set Homing_Offset to 0 for motors:
- COM12: IDs 8, 9
- COM13: IDs 2, 3
"""

import time
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode


def set_offsets_zero():
    # COM12: IDs 7–12
    motors_com12 = {
        f"motor_{i}": Motor(i, "sts3215", MotorNormMode.RANGE_0_100)
        for i in range(7, 13)
    }
    # COM13: IDs 1–6
    motors_com13 = {
        f"motor_{i}": Motor(i, "sts3215", MotorNormMode.RANGE_0_100)
        for i in range(1, 7)
    }

    bus_com12 = FeetechMotorsBus(port="COM12", motors=motors_com12)
    bus_com13 = FeetechMotorsBus(port="COM13", motors=motors_com13)

    # Names for the specific motors we want to zero
    targets_com12 = ["motor_8", "motor_9"]
    targets_com13 = ["motor_2", "motor_3"]

    try:
        print("Connecting to COM12...")
        bus_com12.connect(handshake=False)
        print("Connecting to COM13...")
        bus_com13.connect(handshake=False)

        print("\nDisabling torque on target motors...")
        bus_com12.disable_torque(targets_com12)
        bus_com13.disable_torque(targets_com13)
        time.sleep(0.1)

        for bus, targets, label in [
            (bus_com12, targets_com12, "COM12"),
            (bus_com13, targets_com13, "COM13"),
        ]:
            print(f"\n{label}: Setting Homing_Offset = 0 for {targets}")
            for name in targets:
                try:
                    old_offset = bus.read("Homing_Offset", name, normalize=False, num_retry=5)
                except Exception:
                    old_offset = None
                try:
                    old_pos = bus.read("Present_Position", name, normalize=False, num_retry=5)
                except Exception:
                    old_pos = None

                print(f"  {name}: old Homing_Offset={old_offset}, old Present_Position={old_pos}")

                # Write raw 0 to Homing_Offset
                bus.write("Homing_Offset", name, 0, normalize=False)
                time.sleep(0.05)

                try:
                    new_offset = bus.read("Homing_Offset", name, normalize=False, num_retry=5)
                except Exception:
                    new_offset = None
                try:
                    new_pos = bus.read("Present_Position", name, normalize=False, num_retry=5)
                except Exception:
                    new_pos = None

                print(f"       new Homing_Offset={new_offset}, new Present_Position={new_pos}")

        print("\nRe-enabling torque on target motors...")
        bus_com12.enable_torque(targets_com12)
        bus_com13.enable_torque(targets_com13)

    finally:
        print("\nDisconnecting...")
        try:
            bus_com12.disconnect(disable_torque=False)
        except Exception as e:
            print(f"Warning: Error disconnecting COM12: {e}")
        try:
            bus_com13.disconnect(disable_torque=False)
        except Exception as e:
            print(f"Warning: Error disconnecting COM13: {e}")


if __name__ == "__main__":
    set_offsets_zero()
    print("\nDone!")
