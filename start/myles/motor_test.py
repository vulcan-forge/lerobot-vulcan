#!/usr/bin/env python3

from lerobot.common.robots.sourccey_v2beta.config_sourccey_v2beta import SourcceyV2BetaConfig
from lerobot.common.robots.sourccey_v2beta.sourccey_v2beta import SourcceyV2Beta

def main():
    print("=== Right Arm Motor Test ===\n")
    
    config = SourcceyV2BetaConfig()
    robot = SourcceyV2Beta(config)
    
    print(f"Right arm port: {robot.right_arm_bus.port}")
    print(f"Left arm port: {robot.left_arm_bus.port}")
    
    # Test right arm motors
    print("\n1. Testing Right Arm Motors:")
    try:
        robot.right_arm_bus.connect()
        print("✓ Right arm bus connected")
        
        for motor_name in robot.right_arm_motors:
            motor_id = robot.right_arm_bus.motors[motor_name].id
            print(f"\n  Testing {motor_name} (ID: {motor_id})...")
            
            try:
                # Try to ping the motor
                ping_result = robot.right_arm_bus.ping(motor_id)
                print(f"    Ping result: {ping_result}")
                
                # Try to read present position
                pos = robot.right_arm_bus.read("Present_Position", motor_name)
                print(f"    Present Position: {pos}")
                
                # Try to read model number
                model = robot.right_arm_bus.read("Model_Number", motor_name)
                print(f"    Model Number: {model}")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
        
    except Exception as e:
        print(f"✗ Right arm bus connection failed: {e}")
    finally:
        try:
            robot.right_arm_bus.disconnect()
        except:
            pass
    
    # Test left arm motors
    print("\n2. Testing Left Arm Motors:")
    try:
        robot.left_arm_bus.connect()
        print("✓ Left arm bus connected")
        
        for motor_name in robot.left_arm_motors:
            motor_id = robot.left_arm_bus.motors[motor_name].id
            print(f"\n  Testing {motor_name} (ID: {motor_id})...")
            
            try:
                # Try to ping the motor
                ping_result = robot.left_arm_bus.ping(motor_id)
                print(f"    Ping result: {ping_result}")
                
                # Try to read present position
                pos = robot.left_arm_bus.read("Present_Position", motor_name)
                print(f"    Present Position: {pos}")
                
                # Try to read model number
                model = robot.left_arm_bus.read("Model_Number", motor_name)
                print(f"    Model Number: {model}")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
        
    except Exception as e:
        print(f"✗ Left arm bus connection failed: {e}")
    finally:
        try:
            robot.left_arm_bus.disconnect()
        except:
            pass

if __name__ == "__main__":
    main() 