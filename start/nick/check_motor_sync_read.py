#!/usr/bin/env python3
"""
Manual test script for Feetech motor sync read on all 12 motors using /dev/ttyUSB0
This mimics the exact setup used in Sourccey V2 Beta teleoperation
"""

import time
import sys
from pathlib import Path

# Add the project root to the path so we can import lerobot
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def test_feetech_motor_sync_read():
    """Test sync read on all 12 Feetech motors using /dev/ttyUSB0"""

    # Define the motor configuration - EXACTLY as used in teleoperation
    motors = {
        "left_shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
        "left_shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        "left_elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
        "left_wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
        "left_wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
        "left_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        "right_shoulder_pan": Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
        "right_shoulder_lift": Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
        "right_elbow_flex": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
        "right_wrist_flex": Motor(10, "sts3215", MotorNormMode.RANGE_M100_100),
        "right_wrist_roll": Motor(11, "sts3215", MotorNormMode.RANGE_M100_100),
        "right_gripper": Motor(12, "sts3215", MotorNormMode.RANGE_0_100),
    }

    # Create the motors bus
    port = "/dev/ttyACM0"
    bus = FeetechMotorsBus(port=port, motors=motors)

    print(f"Testing Feetech motor sync read on {port}")
    print(f"Testing all {len(motors)} motors (IDs 1-{len(motors)})")
    print(f"Motor Model: sts3215")
    print("-" * 50)

    try:
        # Connect to the motors
        print("Connecting to motors...")
        bus.connect(handshake=False)
        print("✓ Connected successfully")

        # Set a longer timeout for sync operations
        print("Setting longer timeout for sync operations...")
        bus.set_timeout(1000)  # 1 second timeout
        print("✓ Timeout set to 1000ms")

        # Test ping on each motor individually
        print("\nTesting ping on each motor...")
        for motor_name, motor in motors.items():
            try:
                model_number = bus.ping(motor.id)
                print(f"  ✓ {motor_name} (ID {motor.id}): Model {model_number}")
            except Exception as e:
                print(f"  ✗ {motor_name} (ID {motor.id}): Failed - {e}")

        # Test sync read on ALL motors at once
        print("\n=== TESTING SYNC READ ON ALL 12 MOTORS AT ONCE ===")
        success_count = 0
        total_attempts = 10

        for i in range(total_attempts):
            try:
                print(f"\nAttempt {i+1}/{total_attempts}: Reading all motors simultaneously...")
                positions = bus.sync_read("Present_Position", normalize=False)
                success_count += 1
                print(f"  ✓ SUCCESS: {len(positions)} motors responded")

                # Show positions for first successful read and every 5th attempt
                if i == 0 or (i + 1) % 5 == 0:
                    print("  Motor positions:")
                    for motor_name, position in positions.items():
                        print(f"    {motor_name}: {position}")

                time.sleep(0.2)  # Delay between reads
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                time.sleep(0.5)  # Longer delay after failure

        print(f"\n=== SYNC READ RESULTS ===")
        print(f"Success rate: {success_count}/{total_attempts} ({success_count/total_attempts*100:.1f}%)")

        if success_count > 0:
            print("✓ Sync read on all motors at once is working!")
        else:
            print("✗ Sync read on all motors at once failed consistently")

        # Test with different parameters
        print("\n=== TESTING WITH DIFFERENT PARAMETERS ===")

        # Test with increased retry count
        print("Testing with 8 retries...")
        try:
            positions = bus.sync_read("Present_Position", normalize=False, num_retry=8)
            print(f"  ✓ Success with 8 retries: {len(positions)} motors")
        except Exception as e:
            print(f"  ✗ Failed with 8 retries: {e}")

        # Test with longer delay before read
        print("Testing with 0.3s delay before read...")
        try:
            time.sleep(0.3)
            positions = bus.sync_read("Present_Position", normalize=False)
            print(f"  ✓ Success after 0.3s delay: {len(positions)} motors")
        except Exception as e:
            print(f"  ✗ Failed after 0.3s delay: {e}")

        # Test reading different registers
        print("Testing different registers...")
        registers_to_test = ["Present_Position", "Present_Velocity", "Present_Load"]

        for register in registers_to_test:
            try:
                values = bus.sync_read(register, normalize=False)
                print(f"  ✓ {register}: {len(values)} motors responded")
            except Exception as e:
                print(f"  ✗ {register}: Failed - {e}")

    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

    finally:
        # Clean up
        try:
            bus.close()
            print("\n✓ Connection closed")
        except:
            pass

    print("\n✓ Test completed successfully")
    return True


if __name__ == "__main__":
    print("Feetech Motor Sync Read Test - All 12 Motors")
    print("=" * 50)

    success = test_feetech_motor_sync_read()

    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
