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
            "in1_pins": [17,23,24,26,5], # Physical pins: [11, 16, 18, 37, 29]
            "in2_pins": [27,22,25,16,6], # Physical pins: [13, 15, 22, 36, 31]
            "pwm_frequency": 1000,
        }

        # Create motors
        front_left = DCMotor(
            id=1,
            model="mecanum_wheel",
            norm_mode=MotorNormMode.PWM_DUTY_CYCLE,
        )

        front_right = DCMotor(
            id=2,
            model="mecanum_wheel",
            norm_mode=MotorNormMode.PWM_DUTY_CYCLE,
        )

        rear_left = DCMotor(
            id=3,
            model="mecanum_wheel",
            norm_mode=MotorNormMode.PWM_DUTY_CYCLE,
        )

        rear_right = DCMotor(
            id=4,
            model="mecanum_wheel",
            norm_mode=MotorNormMode.PWM_DUTY_CYCLE,
        )

        linear_actuator = DCMotor(
            id=5,
            model="linear_actuator",
            norm_mode=MotorNormMode.PWM_DUTY_CYCLE,
        )

        # Create controller with motors as a dictionary
        controller = PWMDCMotorsController(
            motors={"front_left": front_left, "front_right": front_right, "rear_left": rear_left, "rear_right": rear_right, "linear_actuator": linear_actuator},
            config=motor_config,
        )

        # Connect
        print("1. Connecting motor...")
        controller.connect()
        print("✓ Motor connected")
        print()

        try:
            #Wheel Test
            print("2. Wheel Test...")
            controller.set_velocity("front_left", 0.12)
            controller.set_velocity("front_right", 0.12)
            controller.set_velocity("rear_left", 0.12)
            controller.set_velocity("rear_right", 0.12)
            time.sleep(15)

            # Stop motors
            print("3. Stopping motor...")
            controller.set_velocity("front_left", 0.0)
            controller.set_velocity("front_right", 0.0)
            controller.set_velocity("rear_left", 0.0)
            controller.set_velocity("rear_right", 0.0)
            controller.set_velocity("linear_actuator", 0.0)
            print("   Motor states: ", controller.protocol_handler.motor_states)
            time.sleep(1)

        except KeyboardInterrupt:
            print()
            print("Stopping motors...")
            controller.set_velocity("front_left", 0.0)
            controller.set_velocity("front_right", 0.0)
            controller.set_velocity("rear_left", 0.0)
            controller.set_velocity("rear_right", 0.0)
            controller.set_velocity("linear_actuator", 0.0)
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
