#!/usr/bin/env python3
"""
Script to get the present position of all motors on COM12 and COM13.
COM12 has motor IDs 1-6, COM13 has motor IDs 7-12.
"""

from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode


def get_motor_positions():
    """Get present positions from all motors on both COM ports."""

    # COM12: motor IDs 1-6
    # Using "sts3215" as the motor model - change if you have different motors
    motors_com12 = {
        f"motor_{i}": Motor(i, "sts3215", MotorNormMode.RANGE_0_100)
        for i in range(7, 13)
    }

    # COM13: motor IDs 7-12
    motors_com13 = {
        f"motor_{i}": Motor(i, "sts3215", MotorNormMode.RANGE_0_100)
        for i in range(1, 7)
    }

    # Create bus instances
    bus_com12 = FeetechMotorsBus(port="COM12", motors=motors_com12)
    bus_com13 = FeetechMotorsBus(port="COM13", motors=motors_com13)

    try:
        # Connect to both buses
        print("Connecting to COM12...")
        bus_com12.connect(handshake=False)  # Set to True if you want to verify motors exist

        print("Connecting to COM13...")
        bus_com13.connect(handshake=False)  # Set to True if you want to verify motors exist

        # Read positions from COM12
        print("\nReading positions from COM12 (motor IDs 1-6)...")
        positions_com12 = bus_com12.sync_read("Present_Position", normalize=False)

        # Read positions from COM13
        print("Reading positions from COM13 (motor IDs 7-12)...")
        positions_com13 = bus_com13.sync_read("Present_Position", normalize=False)

        # Print results
        print("\n" + "="*60)
        print("MOTOR POSITIONS (Raw Encoder Values)")
        print("="*60)
        print("\nCOM12 (Motor IDs 1-6):")
        for motor_name, position in sorted(positions_com12.items()):
            motor_id = motor_name.split("_")[1]
            print(f"  Motor ID {motor_id:2s}: {position:5d}")

        print("\nCOM13 (Motor IDs 7-12):")
        for motor_name, position in sorted(positions_com13.items()):
            motor_id = motor_name.split("_")[1]
            print(f"  Motor ID {motor_id:2s}: {position:5d}")

        print("\n" + "="*60)

        # Return combined results
        all_positions = {**positions_com12, **positions_com13}
        return all_positions

    except Exception as e:
        print(f"Error reading motor positions: {e}")
        raise
    finally:
        # Disconnect both buses
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
    positions = get_motor_positions()
    print("\nDone!")
