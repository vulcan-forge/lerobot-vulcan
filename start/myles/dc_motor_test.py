#!/usr/bin/env python3
"""
Test single DC motor - turn for 5 seconds.
"""

import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_motor_turn():
    """Turn DC motor for 5 seconds."""
    try:
        from lerobot.motors.dc_pwm.dc_pwm import PWMDCMotorsController
        from lerobot.motors.dc_motors_controller import DCMotor, MotorNormMode

        # Motor config for testing
        motor_config = {
            "pwm_pins": [24, 22],           # motor1 PWM, motor2 PWM
            "direction_pins": [23, 27],      # motor1 direction, motor2 direction
            "pwm_frequency": 1000,
            "invert_direction": False,
        }

        # Create motors
        motor1 = DCMotor(
            id=1,
            model="mecanum_wheel",
            norm_mode=MotorNormMode.PWM_DUTY_CYCLE,
        )

        motor2 = DCMotor(
            id=2,
            model="mecanum_wheel",
            norm_mode=MotorNormMode.PWM_DUTY_CYCLE,
        )

        # Create controller with motors as a dictionary
        controller = PWMDCMotorsController(
            motors={"motor1": motor1, "motor2": motor2},  # Add both motors
            config=motor_config,
        )

        print("=== DC Motor Test ===")
        print("Motor: Single DC Motor")
        print(f"Pins: IN1=GPIO {motor_config['pwm_pins'][0]} (PWM), IN2=GPIO {motor_config['direction_pins'][0]} (Direction)")
        print()

        # Connect
        print("1. Connecting motor...")
        controller.connect()
        print("✓ Motor connected")
        print()

        try:
            # Turn motor 1 and 2 forward at full speed for 5 seconds
            print("2. Turning motor forward for 5 seconds...")
            controller.set_velocity("motor1", 1.0)
            controller.set_velocity("motor2", 1.0)
            print("   Motor states: ", controller.protocol_handler.motor_states)
            time.sleep(5)

            # Stop motors
            print("3. Stopping motor...")
            controller.set_velocity("motor1", 0.0)
            controller.set_velocity("motor2", 0.0)
            print("   Motor states: ", controller.protocol_handler.motor_states)
            time.sleep(1)

        except KeyboardInterrupt:
            print()
            print("Stopping motors...")
            controller.set_velocity("motor1", 0.0)
            controller.set_velocity("motor2", 0.0)
            time.sleep(1)

        # Disconnect
        print("4. Disconnecting motors...")
        controller.disconnect()
        print("✓ Motors disconnected")
        print()

        print("=== Test Complete ===")
        print("✓ Motors turned for 5 seconds successfully!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise

if __name__ == "__main__":
    test_motor_turn()
