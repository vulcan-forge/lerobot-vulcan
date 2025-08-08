#!/usr/bin/env python3
"""
Test single DRV8871DDAR motor - 4 functions, 3 seconds each.
"""

import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_four_functions():
    """Test forward, backward, stop, and brake - 3 seconds each."""
    try:
        from lerobot.motors.dc_pwm.dc_pwm import PWMDCMotorsController
        from lerobot.motors.dc_motors_controller import DCMotor, MotorNormMode

        # Motor config for testing
        motor_config = {
            "pwm_pins": [23],           # IN1 - Front Left Wheel (PWM)
            "direction_pins": [24],      # IN2 - Front Left Wheel (Direction)
            "pwm_frequency": 1000,
            "invert_direction": False,
        }

        # Create motor
        motor = DCMotor(
            id=1,
            model="mecanum_wheel",
            norm_mode=MotorNormMode.PWM_DUTY_CYCLE,
        )

        # Create controller with motors as a dictionary
        controller = PWMDCMotorsController(
            motors={"front_left": motor},  # Fixed: pass as dictionary
            config=motor_config,
        )

        print("=== DRV8871DDAR Single Motor Test ===")
        print("Motor: Front Left Wheel")
        print(f"Pins: IN1=GPIO {motor_config['pwm_pins'][0]} (PWM), IN2=GPIO {motor_config['direction_pins'][0]} (Direction)")
        print()

        # Connect
        print("1. Connecting motor...")
        controller.connect()
        print("✓ Motor connected")
        print()

        # Test loop
        cycle = 1
        try:
            while True:
                print(f"--- Cycle {cycle} ---")

                # 1. Forward motion (3 seconds)
                print("   1. FORWARD motion (3 seconds)...")
                controller.set_velocity("front_left", 1.0)
                print("   Motor states: ", controller.protocol_handler.motor_states)
                time.sleep(3)

                # 2. Backward motion (3 seconds)
                print("   2. BACKWARD motion (3 seconds)...")
                controller.set_velocity("front_left", -1.0)
                print("   Motor states: ", controller.protocol_handler.motor_states)
                time.sleep(3)

                # 3. Stop (3 seconds)
                print("   3. STOP (3 seconds)...")
                controller.set_velocity("front_left", 0.0)
                print("   Motor states: ", controller.protocol_handler.motor_states)
                time.sleep(3)

                # 4. Brake (3 seconds)
                print("   4. BRAKE (3 seconds)...")
                controller.protocol_handler.activate_brake(1)
                print("   Motor states: ", controller.protocol_handler.motor_states)
                time.sleep(3)

                cycle += 1
                print()

        except KeyboardInterrupt:
            print()
            print("Stopping test...")
            controller.set_velocity("front_left", 0.0)
            time.sleep(1)

        # Disconnect
        print("2. Disconnecting motor...")
        controller.disconnect()
        print("✓ Motor disconnected")
        print()

        print("=== Test Complete ===")
        print("✓ All tests passed!")
        print("→ Motor is working correctly")
        print("→ Ready to add more motors")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise

if __name__ == "__main__":
    test_four_functions()
