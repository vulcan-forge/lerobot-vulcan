"""
Motor Diagnostics Script - Test individual motors to find communication issues.

This script tests motors 1-6 (left arm) and 7-12 (right arm) one by one to 
identify which motor is failing during sync read operations.
"""
import time
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode


def test_arm_motors(arm_name, port, motor_info):
    """Test each motor on a specific arm individually."""
    
    print("=" * 70)
    print(f"{arm_name} Motor Diagnostics - Testing motors {list(motor_info.keys())}")
    print(f"Port: {port}")
    print("=" * 70)
    print()
    
    results = {}
    
    # Test each motor individually
    for motor_id, motor_name in motor_info.items():
        print(f"Testing Motor {motor_id} ({motor_name})...")
        print("-" * 70)
        
        try:
            # Create a bus with only this motor
            norm_mode = MotorNormMode.RANGE_0_100 if motor_name == "gripper" else MotorNormMode.RANGE_M100_100
            
            bus = FeetechMotorsBus(
                port=port,
                motors={motor_name: Motor(motor_id, "sts3215", norm_mode)},
                protocol_version=0
            )
            
            # Connect to the bus
            bus.connect()
            
            # Try to read Present_Position multiple times
            success_count = 0
            fail_count = 0
            
            for i in range(5):
                try:
                    position = bus.read("Present_Position", motor_name, num_retry=3)
                    print(f"  Read {i+1}: Success - Position = {position}")
                    success_count += 1
                    time.sleep(0.1)
                except Exception as e:
                    print(f"  Read {i+1}: FAILED - {e}")
                    fail_count += 1
                    time.sleep(0.1)
            
            # Disconnect
            bus.disconnect()
            
            # Store results
            results[motor_id] = {
                "name": motor_name,
                "success": success_count,
                "failed": fail_count,
                "status": "OK" if success_count == 5 else "ISSUE" if success_count > 0 else "FAILED"
            }
            
            print(f"  Summary: {success_count}/5 successful reads")
            print(f"  Status: {results[motor_id]['status']}")
            
        except Exception as e:
            print(f"  ERROR: Could not initialize motor - {e}")
            results[motor_id] = {
                "name": motor_name,
                "success": 0,
                "failed": 5,
                "status": "FAILED",
                "error": str(e)
            }
        
        print()
        time.sleep(0.5)
    
    return results


def print_arm_summary(arm_name, results):
    """Print summary for a specific arm."""
    print("=" * 70)
    print(f"{arm_name} DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print()
    
    for motor_id, result in results.items():
        status_symbol = "✓" if result["status"] == "OK" else "⚠" if result["status"] == "ISSUE" else "✗"
        print(f"{status_symbol} Motor {motor_id:2d} ({result['name']:15s}): {result['status']:6s} - "
              f"{result['success']}/5 reads successful")
    
    print()
    
    # Identify problematic motors
    problematic = [motor_id for motor_id, result in results.items() if result["status"] != "OK"]
    
    if problematic:
        print("⚠ PROBLEMATIC MOTORS DETECTED:")
        for motor_id in problematic:
            result = results[motor_id]
            print(f"  - Motor {motor_id} ({result['name']}): {result['status']}")
            if "error" in result:
                print(f"    Error: {result['error']}")
    else:
        print("✓ All motors are communicating properly!")
    
    print()
    print("=" * 70)
    print()


def test_both_arms():
    """Test both left and right arms."""
    
    # Define motor configurations for both arms
    left_arm_motors = {
        1: "shoulder_pan",
        2: "shoulder_lift",
        3: "elbow_flex",
        4: "wrist_flex",
        5: "wrist_roll",
        6: "gripper"
    }
    
    right_arm_motors = {
        7: "shoulder_pan",
        8: "shoulder_lift",
        9: "elbow_flex",
        10: "wrist_flex",
        11: "wrist_roll",
        12: "gripper"
    }
    
    # Port symbolic links
    left_port = "/dev/robotLeftArm"
    right_port = "/dev/robotRightArm"
    
    all_results = {}
    
    # Test left arm
    print("\n")
    left_results = test_arm_motors("LEFT ARM", left_port, left_arm_motors)
    all_results['left'] = left_results
    print_arm_summary("LEFT ARM", left_results)
    
    # Test right arm
    print("\n")
    right_results = test_arm_motors("RIGHT ARM", right_port, right_arm_motors)
    all_results['right'] = right_results
    print_arm_summary("RIGHT ARM", right_results)
    
    # Overall summary
    print("=" * 70)
    print("OVERALL SYSTEM SUMMARY")
    print("=" * 70)
    print()
    
    left_issues = sum(1 for r in left_results.values() if r["status"] != "OK")
    right_issues = sum(1 for r in right_results.values() if r["status"] != "OK")
    
    print(f"Left Arm:  {6 - left_issues}/6 motors OK")
    print(f"Right Arm: {6 - right_issues}/6 motors OK")
    print()
    
    if left_issues == 0 and right_issues == 0:
        print("✓ ALL MOTORS ARE FUNCTIONING PROPERLY!")
    else:
        print("⚠ ISSUES DETECTED - See individual arm summaries above for details")
    
    print()
    print("=" * 70)
    
    return all_results

if __name__ == "__main__":
    try:
        results = test_both_arms()
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()