#!/usr/bin/env python3
"""
Script to safely remove torque from all motors on the bus.
Useful for emergency stops or when motors are stuck.
"""

import time
import sys
from pathlib import Path

from lerobot.common.motors.motors_bus import MotorCalibration

# Add the project root to the path so we can import lerobot
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lerobot.common.motors import Motor, MotorNormMode
from lerobot.common.motors.feetech import FeetechMotorsBus


def reset_all_motors():
    """Remove torque from all motors on the bus."""

    # Motor configuration - adjust these based on your setup
    motor_id = 1  # Change this to your motor's ID
    motor_model = "sts3215"  # Change this to your motor's model
    port = "COM13"  # Change this to your port

    # Create a single motor configuration (or expand for multiple motors)
    motors = {
        "test_motor": Motor(motor_id, motor_model, MotorNormMode.RANGE_M100_100),
    }

    print(f"Resetting all motors on {port}")
    print(f"Motor ID: {motor_id}")
    print(f"Motor Model: {motor_model}")
    print("-" * 50)

    bus = FeetechMotorsBus(port=port, motors=motors, calibration={"test_motor": MotorCalibration(id=motor_id, drive_mode=0, homing_offset=0, range_min=0, range_max=4095)})

    try:
        # Connect to the motor
        print("Connecting to motor...")
        bus.connect(handshake=False)
        print("✓ Connected successfully")

        # Test ping to verify motor is responding
        print(f"Testing ping on motor ID {motor_id}...")
        try:
            model_number = bus.ping(motor_id)
            print(f"✓ Motor responded with model number: {model_number}")
        except Exception as e:
            print(f"✗ Ping failed: {e}")
            return False

        # Read current status before reset
        print("\n=== CURRENT MOTOR STATUS ===")
        try:
            torque_enable = bus.read("Torque_Enable", "test_motor", normalize=False)
            current_pos = bus.read("Present_Position", "test_motor", normalize=False)
            current = bus.read("Present_Current", "test_motor", normalize=False)
            load = bus.read("Present_Load", "test_motor", normalize=False)

            print(f"Torque Enable: {torque_enable} (0=Disabled, 1=Enabled)")
            print(f"Current Position: {current_pos}")
            print(f"Current: {current}mA")
            print(f"Load: {load}")
        except Exception as e:
            print(f"Error reading motor status: {e}")

        # Remove torque from all motors
        print("\n=== REMOVING TORQUE FROM ALL MOTORS ===")
        try:
            print("Disabling torque...")
            bus.disable_torque()  # This disables torque on all motors in the bus
            print("✓ Torque disabled successfully")

            # Verify torque was disabled
            time.sleep(0.5)
            torque_enable = bus.read("Torque_Enable", "test_motor", normalize=False)
            print(f"Torque Enable after reset: {torque_enable} (should be 0)")

            if torque_enable == 0:
                print("✓ Torque successfully removed")
            else:
                print("⚠ Torque may not have been fully disabled")

        except Exception as e:
            print(f"Error disabling torque: {e}")
            return False

        print("\n=== MOTOR RESET COMPLETED ===")
        print("All motors should now be free to move manually")
        print("You can now safely move the motor shaft by hand")

        return True

    except Exception as e:
        print(f"✗ Error during motor reset: {e}")
        return False

    finally:
        # Clean up
        try:
            print("\nDisconnecting...")
            bus.disconnect()
            print("✓ Disconnected successfully")
        except:
            pass

if __name__ == "__main__":
    print("Motor Reset Tool")
    print("=" * 50)

    success = reset_all_motors()

    if success:
        print("\n🎉 Motor reset completed successfully!")
        print("Motors are now free to move manually")
        sys.exit(0)
    else:
        print("\n❌ Motor reset failed!")
        sys.exit(1)
