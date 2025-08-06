#!/usr/bin/env python3
"""
Test single DRV8871DDAR motor - Forward motion only.
"""

import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_forward_only():
    """Test full speed forward motion and hold it."""
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

        print("=== DRV8871DDAR Forward Motion Test ===")
        print("Motor: Front Left Wheel")
        print("Pins: IN1=GPIO 12 (PWM), IN2=GPIO 22 (Direction)")
        print()

        # Connect
        print("1. Connecting motor...")
        controller.connect()
        print("✓ Motor connected")
        print()

        # Test full speed forward
        print("2. Testing FULL SPEED FORWARD motion...")
        print("   Setting velocity to 1.0 (100% forward)")
        controller.set_velocity("front_left", 1.0)
        print("Motor states: ", controller.protocol_handler.motor_states)
        print("   Motor should be running at full speed forward")
        print("   Press Ctrl+C to stop the test")
        print()

        # Hold the motor running
        try:
            while True:
                time.sleep(1)
                print("   Motor still running... (Ctrl+C to stop)")
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
        print("✓ Forward motion test completed")
        print("→ Did the motor spin forward at full speed?")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("→ Check wiring and connections")

if __name__ == "__main__":
    test_forward_only()
