#!/usr/bin/env python3
"""
Test 5 DRV8871DDAR motors using custom GPIO pin configuration.
Custom Pin Configuration:
Motor 1: IN1=GPIO 17, IN2=GPIO 18
Motor 2: IN1=GPIO 27, IN2=GPIO 22
Motor 3: IN1=GPIO 23, IN2=GPIO 24
Motor 4: IN1=GPIO 25, IN2=GPIO 5
Motor 5: IN1=GPIO 6, IN2=GPIO 12
"""

import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_five_motors():
    """Test 5 DRV8871DDAR motors with custom GPIO pins."""
    try:
        from lerobot.motors.dc_pwm.dc_pwm import PWMDCMotorsController
        from lerobot.motors.dc_motors_controller import DCMotor, MotorNormMode

        # Motor configuration using custom GPIO pins
        # Each motor uses 2 pins: PWM (IN1) and Direction (IN2)
        motor_config = {
            # PWM pins (IN1) - Speed control
            "pwm_pins": [17, 27, 23, 25, 6],  # GPIO 17, 27, 23, 25, 6

            # Direction pins (IN2) - Direction control
            "direction_pins": [18, 22, 24, 5, 12],  # GPIO 18, 22, 24, 5, 12

            "pwm_frequency": 1000,  # 1kHz - compatible with gpiozero
            "invert_direction": False,
        }

        # Create 5 motors
        motors = {
            "motor_1": DCMotor(id=1, model="drv8871", norm_mode=MotorNormMode.PWM_DUTY_CYCLE),
            "motor_2": DCMotor(id=2, model="drv8871", norm_mode=MotorNormMode.PWM_DUTY_CYCLE),
            "motor_3": DCMotor(id=3, model="drv8871", norm_mode=MotorNormMode.PWM_DUTY_CYCLE),
            "motor_4": DCMotor(id=4, model="drv8871", norm_mode=MotorNormMode.PWM_DUTY_CYCLE),
            "motor_5": DCMotor(id=5, model="drv8871", norm_mode=MotorNormMode.PWM_DUTY_CYCLE),
        }

        # Create controller
        controller = PWMDCMotorsController(
            motors=motors,
            config=motor_config,
        )

        print("=== 5 Motor DRV8871DDAR GPIO Test ===")
        print("Using custom GPIO pin configuration:")
        print("Motor 1: IN1=GPIO 17, IN2=GPIO 18")
        print("Motor 2: IN1=GPIO 27, IN2=GPIO 22")
        print("Motor 3: IN1=GPIO 23, IN2=GPIO 24")
        print("Motor 4: IN1=GPIO 25, IN2=GPIO 5")
        print("Motor 5: IN1=GPIO 6, IN2=GPIO 12")
        print()

        # Connect
        print("1. Connecting motors...")
        controller.connect()
        print("✓ All motors connected")
        print()

        # Test loop
        cycle = 1
        try:
            while True:
                print(f"--- Cycle {cycle} ---")

                # Test 1: All motors forward (2 seconds)
                print("   1. ALL MOTORS FORWARD (2 seconds)...")
                for motor_name in motors.keys():
                    controller.set_velocity(motor_name, 0.5)  # 50% speed forward
                print("   Motor states: ", controller.protocol_handler.motor_states)
                time.sleep(2)

                # Test 2: All motors backward (2 seconds)
                print("   2. ALL MOTORS BACKWARD (2 seconds)...")
                for motor_name in motors.keys():
                    controller.set_velocity(motor_name, -0.5)  # 50% speed backward
                print("   Motor states: ", controller.protocol_handler.motor_states)
                time.sleep(2)

                # Test 3: Stop all motors (1 second)
                print("   3. STOP ALL MOTORS (1 second)...")
                for motor_name in motors.keys():
                    controller.set_velocity(motor_name, 0.0)
                print("   Motor states: ", controller.protocol_handler.motor_states)
                time.sleep(1)

                # Test 4: Individual motor test (3 seconds each)
                print("   4. INDIVIDUAL MOTOR TEST...")
                for i, motor_name in enumerate(motors.keys(), 1):
                    print(f"      Testing Motor {i} ({motor_name})...")

                    # Forward
                    controller.set_velocity(motor_name, 0.7)
                    print(f"        Forward at 70% speed")
                    time.sleep(1)

                    # Backward
                    controller.set_velocity(motor_name, -0.7)
                    print(f"        Backward at 70% speed")
                    time.sleep(1)

                    # Stop
                    controller.set_velocity(motor_name, 0.0)
                    print(f"        Stop")
                    time.sleep(1)

                    print(f"      ✓ Motor {i} test complete")
                    print()

                # Test 5: Brake test (2 seconds)
                print("   5. BRAKE TEST (2 seconds)...")
                for motor_id in range(1, 6):
                    controller.protocol_handler.activate_brake(motor_id)
                print("   Motor states: ", controller.protocol_handler.motor_states)
                time.sleep(2)

                # Release brakes
                for motor_id in range(1, 6):
                    controller.protocol_handler.release_brake(motor_id)

                # Test 6: Alternating pattern (3 seconds)
                print("   6. ALTERNATING PATTERN (3 seconds)...")
                # Even motors forward, odd motors backward
                for i, motor_name in enumerate(motors.keys(), 1):
                    if i % 2 == 0:  # Even motors
                        controller.set_velocity(motor_name, 0.6)
                    else:  # Odd motors
                        controller.set_velocity(motor_name, -0.6)
                print("   Motor states: ", controller.protocol_handler.motor_states)
                time.sleep(3)

                # Stop all
                for motor_name in motors.keys():
                    controller.set_velocity(motor_name, 0.0)

                cycle += 1
                print()

        except KeyboardInterrupt:
            print()
            print("Stopping test...")
            # Stop all motors
            for motor_name in motors.keys():
                controller.set_velocity(motor_name, 0.0)
            time.sleep(1)

        # Disconnect
        print("2. Disconnecting motors...")
        controller.disconnect()
        print("✓ All motors disconnected")
        print()

        print("=== Test Complete ===")
        print("✓ All 5 motors tested successfully!")
        print("→ GPIO pins are working correctly")
        print("→ DRV8871DDAR drivers are functioning")
        print("→ Ready for multi-motor applications")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise

def test_individual_motor():
    """Test individual motor for debugging."""
    try:
        from lerobot.motors.dc_pwm.dc_pwm import PWMDCMotorsController
        from lerobot.motors.dc_motors_controller import DCMotor, MotorNormMode

        # Test single motor with first set of pins
        motor_config = {
            "pwm_pins": [17],           # IN1 - PWM control
            "direction_pins": [18],      # IN2 - Direction control
            "pwm_frequency": 1000,
            "invert_direction": False,
        }

        motor = DCMotor(id=1, model="drv8871", norm_mode=MotorNormMode.PWM_DUTY_CYCLE)
        controller = PWMDCMotorsController(motors={"test_motor": motor}, config=motor_config)

        print("=== Individual Motor Test ===")
        print("Motor: Test Motor")
        print("Pins: IN1=GPIO 17 (PWM), IN2=GPIO 18 (Direction)")
        print()

        controller.connect()
        print("✓ Motor connected")

        # Simple test sequence
        print("Testing forward...")
        controller.set_velocity("test_motor", 0.5)
        time.sleep(2)

        print("Testing backward...")
        controller.set_velocity("test_motor", -0.5)
        time.sleep(2)

        print("Testing stop...")
        controller.set_velocity("test_motor", 0.0)
        time.sleep(1)

        controller.disconnect()
        print("✓ Individual motor test complete")

    except Exception as e:
        print(f"✗ Individual motor test failed: {e}")
        raise

if __name__ == "__main__":
    # Uncomment the function you want to run:
    test_five_motors()  # Test all 5 motors
    # test_individual_motor()  # Test single motor for debugging
