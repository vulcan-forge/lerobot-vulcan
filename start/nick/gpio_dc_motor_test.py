#!/usr/bin/env python3
"""
Test single DRV8871DDAR motor - Forward and Backward motion.
"""

import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_forward_backward():
    """Test forward and backward motion in a loop."""
    try:
        from lerobot.motors.dc_pwm.dc_pwm import PWMDCMotorsController
        from lerobot.motors.dc_motors_controller import DCMotor, MotorNormMode

        # Motor config for testing
        motor_config = {
            "pwm_pins": [12],           # IN1 - Front Left Wheel (PWM)
            "direction_pins": [22],      # IN2 - Front Left Wheel (Direction)
            "pwm_frequency": 1000,
            "invert_direction": False,
        }

        motors = {
            "front_left": DCMotor(id=1, model="mecanum_wheel", norm_mode=MotorNormMode.RANGE_M100_100)
        }

        # Create controller
        controller = PWMDCMotorsController(config=motor_config, motors=motors)

        print("=== DRV8871DDAR Forward/Backward Test ===")
        print("Motor: Front Left Wheel")
        print("Pins: IN1=GPIO 12 (PWM), IN2=GPIO 22 (Direction)")
        print("Pattern: 3s Forward → 3s Backward → Loop")
        print()

        # Connect
        print("1. Connecting motor...")
        controller.connect()
        print("✓ Motor connected")
        print()

        # Test loop
        print("2. Starting forward/backward test loop...")
        print("   Press Ctrl+C to stop the test")
        print()

        cycle = 1
        try:
            while True:
                print(f"--- Cycle {cycle} ---")

                # Forward motion (3 seconds)
                print("   FORWARD motion (3 seconds)...")
                controller.set_velocity("front_left", 1.0)
                print("   Motor states: ", controller.protocol_handler.motor_states)
                time.sleep(3)

                # Backward motion (3 seconds)
                print("   BACKWARD motion (3 seconds)...")
                controller.set_velocity("front_left", -1.0)
                print("   Motor states: ", controller.protocol_handler.motor_states)
                time.sleep(3)

                cycle += 1
                print()

        except KeyboardInterrupt:
            print()
            print("   Stopping motor...")
            controller.set_velocity("front_left", 0.0)
            print("✓ Motor stopped")
            print()

        # Disconnect
        print("3. Disconnecting motor...")
        controller.disconnect()
        print("✓ Motor disconnected")
        print()

        print("=== Test Complete ===")
        print("✓ Forward/backward motion test completed")
        print("→ Did the motor alternate between forward and backward?")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("→ Check wiring and connections")

if __name__ == "__main__":
    test_forward_backward()
