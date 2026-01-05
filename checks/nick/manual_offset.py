#!/usr/bin/env python3
"""
Manually set Homing_Offset for a single Feetech motor.

- Prompts for:
    - COM port (e.g. COM12 or COM13)
    - Motor ID (1..12)
    - New homing offset (recommended range: -2047 .. 2047)

- Prints old/new Homing_Offset and Present_Position.
"""

import time

from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode


def set_single_motor_offset(port: str, motor_id: int, new_offset: int):
    # Sanity-check offset range for Feetech sign-magnitude (11-bit magnitude)
    if not -2047 <= new_offset <= 2047:
        print(f"WARNING: Offset {new_offset} is outside the recommended range [-2047, 2047].")
        print("The SDK may clamp or wrap this value based on sign-magnitude encoding.")

    motor_name = f"motor_{motor_id}"

    # Minimal bus: only the one motor we care about
    motors = {
        motor_name: Motor(motor_id, "sts3215", MotorNormMode.RANGE_0_100),
    }

    bus = FeetechMotorsBus(port=port, motors=motors)

    try:
        print(f"Connecting to {port}...")
        bus.connect(handshake=False)

        print(f"\nDisabling torque on {motor_name} (ID {motor_id})...")
        bus.disable_torque(motor_name)
        time.sleep(0.1)

        # Read current values
        try:
            old_offset = bus.read("Homing_Offset", motor_name, normalize=False, num_retry=5)
        except Exception as e:
            print(f"  Could not read old Homing_Offset: {e}")
            old_offset = None

        try:
            old_pos = bus.read("Present_Position", motor_name, normalize=False, num_retry=5)
        except Exception as e:
            print(f"  Could not read old Present_Position: {e}")
            old_pos = None

        print(f"\n{motor_name} (ID {motor_id}) BEFORE:")
        print(f"  Homing_Offset     = {old_offset}")
        print(f"  Present_Position  = {old_pos}")

        # Write new offset (raw units)
        print(f"\nWriting new Homing_Offset = {new_offset} ...")
        bus.write("Homing_Offset", motor_name, new_offset, normalize=False)
        time.sleep(0.2)

        # Read back new values
        try:
            readback_offset = bus.read("Homing_Offset", motor_name, normalize=False, num_retry=5)
        except Exception as e:
            print(f"  Could not read new Homing_Offset: {e}")
            readback_offset = None

        try:
            new_pos = bus.read("Present_Position", motor_name, normalize=False, num_retry=5)
        except Exception as e:
            print(f"  Could not read new Present_Position: {e}")
            new_pos = None

        print(f"\n{motor_name} (ID {motor_id}) AFTER:")
        print(f"  Homing_Offset     = {readback_offset}")
        print(f"  Present_Position  = {new_pos}")

        print("\nRe-enabling torque...")
        bus.enable_torque(motor_name)

    finally:
        print("\nDisconnecting...")
        try:
            bus.disconnect(disable_torque=False)
        except Exception as e:
            print(f"  Warning: error during disconnect: {e}")


if __name__ == "__main__":
    print("Manual Feetech Homing_Offset adjustment")

    # Simple interactive prompts
    port = input("Enter COM port (e.g. COM12 or COM13) [default: COM12]: ").strip() or "COM12"

    while True:
        try:
            motor_id_str = input("Enter motor ID (integer, e.g. 1..12): ").strip()
            motor_id = int(motor_id_str)
            break
        except ValueError:
            print("Please enter a valid integer for motor ID.")

    while True:
        try:
            offset_str = input("Enter new Homing_Offset (integer, recommended -2047..2047): ").strip()
            new_offset = int(offset_str)
            break
        except ValueError:
            print("Please enter a valid integer for Homing_Offset.")

    set_single_motor_offset(port, motor_id, new_offset)
    print("\nDone!")
