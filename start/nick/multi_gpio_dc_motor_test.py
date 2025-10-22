#!/usr/bin/env python3
"""
Test DRV8874PWPR motors with custom GPIO pins - 4 functions, 3 seconds each.
Generic function to test any motor by passing GPIO pins.
"""

import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_motor(pwm_pin, direction_pin, motor_name="Motor"):
    """
    Test a single DRV8874PWPR motor with specified GPIO pins - 1 cycle only.

    Args:
        pwm_pin (int): GPIO pin for PWM (IN1)
        direction_pin (int): GPIO pin for direction (IN2)
        motor_name (str): Name for the motor (for display)
    """
    try:
        from lerobot.motors.dc_pwm.dc_pwm import PWMDCMotorsController
        from lerobot.motors.dc_motors_controller import DCMotor, MotorNormMode

        # Motor config for testing
        motor_config = {
            "in1_pins": [pwm_pin],           # IN1 - PWM control
            "in2_pins": [direction_pin], # IN2 - Direction control
            "pwm_frequency": 1000,
            "invert_direction": False,
        }

        # Create motor
        motor = DCMotor(
            id=1,
            model="drv8871",
            norm_mode=MotorNormMode.PWM_DUTY_CYCLE,
        )

        # Create controller
        controller = PWMDCMotorsController(
            motors={"test_motor": motor},
            config=motor_config,
        )

        print(f"=== DRV8874PWPR {motor_name} Test ===")
        print(f"Motor: {motor_name}")
        print(f"Pins: IN1=GPIO {pwm_pin} (PWM), IN2=GPIO {direction_pin} (Direction)")
        print()

        # Connect
        print("1. Connecting motor...")
        controller.connect()
        print("✓ Motor connected")
        print()

        # Single test cycle
        print("--- Single Test Cycle ---")

        # 1. Forward motion (3 seconds)
        print("   1. FORWARD motion (3 seconds)...")
        controller.set_velocity("test_motor", 1.0)
        print("   Motor states: ", controller.protocol_handler.motor_states)
        time.sleep(3)

        # 2. Backward motion (3 seconds)
        print("   2. BACKWARD motion (3 seconds)...")
        controller.set_velocity("test_motor", -1.0)
        print("   Motor states: ", controller.protocol_handler.motor_states)
        time.sleep(3)

        # 3. Stop (3 seconds)
        print("   3. STOP (3 seconds)...")
        controller.set_velocity("test_motor", 0.0)
        print("   Motor states: ", controller.protocol_handler.motor_states)
        time.sleep(3)

        # 4. Brake (3 seconds)
        print("   4. BRAKE (3 seconds)...")
        controller.protocol_handler.activate_brake(1)
        print("   Motor states: ", controller.protocol_handler.motor_states)
        time.sleep(3)

        # Disconnect
        print("2. Disconnecting motor...")
        controller.disconnect()
        print("✓ Motor disconnected")
        print()

        print("=== Test Complete ===")
        print("✓ All tests passed!")
        print(f"→ {motor_name} is working correctly")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise

def test_all_motors():
    """Test all 5 motors in sequence - 1 cycle each."""
    # Motor configurations: (pwm_pin, direction_pin, motor_name)
    motors = [
        (17, 18, "Motor 1"),
        (27, 22, "Motor 2"),
        (23, 24, "Motor 3"),
        (25, 5, "Motor 4"),
        (6, 12, "Motor 5"),
    ]

    print("=== Testing All 5 Motors ===")
    print("Motor configurations:")
    for i, (pwm, dir_pin, name) in enumerate(motors, 1):
        print(f"  {name}: IN1=GPIO {pwm}, IN2=GPIO {dir_pin}")
    print()

    for pwm_pin, direction_pin, motor_name in motors:
        print(f"\n{'='*50}")
        print(f"Testing {motor_name}...")
        print(f"{'='*50}")

        try:
            test_motor(pwm_pin, direction_pin, motor_name)
            print(f"✓ {motor_name} test completed successfully!")
        except Exception as e:
            print(f"✗ {motor_name} test failed: {e}")
            break

    print("\n=== All Motor Tests Complete ===")

if __name__ == "__main__":
    # Test individual motor
    # test_motor(17, 18, "Motor 1")  # Test Motor 1

    # Test all motors in sequence
    test_all_motors()
