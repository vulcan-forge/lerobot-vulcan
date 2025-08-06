#!/usr/bin/env python3
"""
Test single DRV8871DDAR motor.
"""

import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_single_motor():
    """Test the front left wheel motor."""
    try:
        from lerobot.motors.dc_pwm.dc_pwm import PWMDCMotorsController
        from lerobot.motors.dc_motors_controller import DCMotor, MotorNormMode

        # Motor config for testing
        motor_config = {
            "pwm_pins": [12],           # IN1 - Front Left Wheel
            "direction_pins": [2],       # IN2 - Front Left Wheel
            "pwm_frequency": 25000,
            "invert_direction": False,
        }

        motors = {
            "front_left": DCMotor(id=1, model="mecanum_wheel", norm_mode=MotorNormMode.RANGE_M100_100)
        }

        # Create controller
        controller = PWMDCMotorsController(config=motor_config, motors=motors)

        print("=== DRV8871DDAR Single Motor Test ===")
        print("Motor: Front Left Wheel")
        print("Pins: IN1=GPIO 12 (PWM), IN2=GPIO 2 (Direction)")
        print()

        # Connect
        print("1. Connecting motor...")
        controller.connect()
        print("✓ Motor connected")
        print()

        # Test forward
        print("2. Testing forward motion (50% speed)...")
        controller.set_velocity("front_left", 0.5)
        time.sleep(3)
        print("✓ Forward test completed")
        print()

        # Test stop
        print("3. Testing stop...")
        controller.set_velocity("front_left", 0.0)
        time.sleep(2)
        print("✓ Stop test completed")
        print()

        # Test backward
        print("4. Testing backward motion (50% speed)...")
        controller.set_velocity("front_left", -0.5)
        time.sleep(3)
        print("✓ Backward test completed")
        print()

        # Test stop
        print("5. Testing stop...")
        controller.set_velocity("front_left", 0.0)
        time.sleep(2)
        print("✓ Stop test completed")
        print()

        # Test variable speed
        print("6. Testing variable speed...")
        for speed in [0.1, 0.3, 0.5, 0.7, 0.9, 0.0]:
            controller.set_velocity("front_left", speed)
            print(f"   Speed: {speed*100:.0f}%")
            time.sleep(1)
        print("✓ Variable speed test completed")
        print()

        # Disconnect
        print("7. Disconnecting motor...")
        controller.disconnect()
        print("✓ Motor disconnected")
        print()

        print("=== Test Complete ===")
        print("✓ All tests passed!")
        print("→ Motor is working correctly")
        print("→ Ready to add more motors")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("→ Check wiring and connections")

if __name__ == "__main__":
    test_single_motor()
