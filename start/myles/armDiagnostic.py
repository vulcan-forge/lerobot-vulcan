"""
Motor Diagnostics Script - Test individual motors to find communication issues.

This script tests motors 7-12 (right arm) one by one to identify which motor
is failing during sync read operations.
"""
import time
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode

def test_individual_motors():
    """Test each motor individually to identify communication issues."""
    
    # Define motor IDs and names for the right arm
    motor_info = {
        7: "shoulder_pan",
        8: "shoulder_lift",
        9: "elbow_flex",
        10: "wrist_flex",
        11: "wrist_roll",
        12: "gripper"
    }
    
    port = "COM6"  # Right arm port
    
    print("=" * 70)
    print("Right Arm Motor Diagnostics - Testing motors 7-12 individually")
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
    
    # Print summary
    print("=" * 70)
    print("DIAGNOSTIC SUMMARY")
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
    
    return results

if __name__ == "__main__":
    try:
        results = test_individual_motors()
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()