#!/usr/bin/env python3
"""
Script to set homing offsets for all motors so their present positions become 2047.
COM12 has motor IDs 7-12, COM13 has motor IDs 1-6.
"""

import time
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode, get_address
from lerobot.motors.encoding_utils import decode_sign_magnitude


def read_raw_register(bus, motor_name, data_name):
    """Read raw register value without sign-magnitude decoding."""
    motor_id = bus.motors[motor_name].id
    model = bus.motors[motor_name].model
    addr, length = get_address(bus.model_ctrl_table, model, data_name)

    # Read directly without decoding
    value, comm, error = bus._read(addr, length, motor_id, num_retry=3, raise_on_error=False)
    return value, comm, error


def fix_problematic_motor(bus, motor_name, target_position=2047):
    """Special handling for problematic motors with multiple strategies."""
    print(f"    Fixing {motor_name} with advanced method...")

    # Strategy 1: Reset everything and read raw values
    print(f"      Strategy 1: Reset and read raw...")

    # Reset homing offset to 0
    bus.write("Homing_Offset", motor_name, 0, normalize=False)
    time.sleep(0.3)

    # Read raw register multiple times
    raw_values = []
    decoded_values = []
    for _ in range(5):
        raw_val, comm, error = read_raw_register(bus, motor_name, "Present_Position")
        if comm == bus._comm_success:
            raw_values.append(raw_val)
            # Manually decode sign-magnitude (bit 15 is sign bit for Present_Position)
            decoded = decode_sign_magnitude(raw_val, 15)
            decoded_values.append(decoded)
        time.sleep(0.1)

    if not raw_values:
        print(f"      ERROR: Could not read raw register!")
        return None

    raw_values.sort()
    decoded_values.sort()
    raw_median = raw_values[len(raw_values) // 2]
    decoded_median = decoded_values[len(decoded_values) // 2]

    print(f"      Raw register (median): {raw_median} (0x{raw_median:04X})")
    print(f"      Manually decoded (median): {decoded_median}")

    # Normalize decoded value to valid range [0, 4095]
    max_res = 4095
    encoder_range = 4096

    # Handle wrap-around
    if decoded_median < 0:
        # Convert negative to positive equivalent
        decoded_median = decoded_median % encoder_range
    elif decoded_median > max_res:
        decoded_median = decoded_median % encoder_range

    print(f"      Normalized position: {decoded_median}")

    # Calculate offset
    mid = max_res // 2
    raw_offset = decoded_median - target_position

    # Try multiple offset calculations
    offsets_to_try = []

    # Standard calculation
    offset1 = raw_offset % encoder_range
    if offset1 > mid:
        offset1 -= encoder_range
    offsets_to_try.append(offset1)

    # Alternative: if offset is large, try the opposite direction
    if abs(offset1) > 1000:
        offset2 = raw_offset - encoder_range if raw_offset > 0 else raw_offset + encoder_range
        offset2 = offset2 % encoder_range
        if offset2 > mid:
            offset2 -= encoder_range
        offsets_to_try.append(offset2)

    # Try each offset and see which one works
    best_position = None
    best_offset = None

    for offset in offsets_to_try:
        print(f"      Trying offset: {offset}")
        bus.write("Homing_Offset", motor_name, offset, normalize=False)
        time.sleep(0.3)

        # Read position multiple times
        positions = []
        for _ in range(3):
            pos = bus.read("Present_Position", motor_name, normalize=False, num_retry=3)
            positions.append(pos)
            time.sleep(0.1)

        positions.sort()
        final_pos = positions[len(positions) // 2]
        print(f"        Result: {final_pos}")

        # Check if this is acceptable
        if 1500 <= final_pos <= 2500:
            print(f"        ✓ SUCCESS with offset {offset}!")
            return final_pos

        # Track the best result
        if best_position is None or abs(final_pos - target_position) < abs(best_position - target_position):
            best_position = final_pos
            best_offset = offset

    # If none worked perfectly, use the best one
    if best_offset is not None:
        print(f"      Best result: position {best_position} with offset {best_offset}")
        bus.write("Homing_Offset", motor_name, best_offset, normalize=False)
        time.sleep(0.2)
        return best_position

    return None


def set_motor_offsets_to_2047():
    """Set homing offsets so all motors read 2047 as their present position."""

    # COM12: motor IDs 7-12 (based on actual output)
    motors_com12 = {
        f"motor_{i}": Motor(i, "sts3215", MotorNormMode.RANGE_0_100)
        for i in range(7, 13)
    }

    # COM13: motor IDs 1-6 (based on actual output)
    motors_com13 = {
        f"motor_{i}": Motor(i, "sts3215", MotorNormMode.RANGE_0_100)
        for i in range(1, 7)
    }

    # Create bus instances
    bus_com12 = FeetechMotorsBus(port="COM12", motors=motors_com12)
    bus_com13 = FeetechMotorsBus(port="COM13", motors=motors_com13)

    try:
        # Connect to both buses
        print("Connecting to COM12...")
        bus_com12.connect(handshake=False)

        print("Connecting to COM13...")
        bus_com13.connect(handshake=False)

        # Disable torque on all motors (required to write to EEPROM)
        print("\nDisabling torque on all motors...")
        bus_com12.disable_torque()
        bus_com13.disable_torque()
        time.sleep(0.1)

        # Set target positions to 2047 for all motors
        target_position = 2047
        target_positions_com12 = {motor: target_position for motor in motors_com12.keys()}
        target_positions_com13 = {motor: target_position for motor in motors_com13.keys()}

        print(f"\nSetting homing offsets to make all motors read {target_position}...")
        print("\nCOM12 (Motor IDs 7-12):")
        homing_offsets_com12 = bus_com12.set_position_homings(target_positions_com12)
        for motor_name, offset in sorted(homing_offsets_com12.items()):
            motor_id = motor_name.split("_")[1]
            print(f"  Motor ID {motor_id:2s}: Homing_Offset = {offset:6d}")

        print("\nCOM13 (Motor IDs 1-6):")
        homing_offsets_com13 = bus_com13.set_position_homings(target_positions_com13)
        for motor_name, offset in sorted(homing_offsets_com13.items()):
            motor_id = motor_name.split("_")[1]
            print(f"  Motor ID {motor_id:2s}: Homing_Offset = {offset:6d}")

        # Re-enable torque
        print("\nRe-enabling torque on all motors...")
        bus_com12.enable_torque()
        bus_com13.enable_torque()
        time.sleep(0.2)

        # Acceptable range
        min_acceptable = 1000
        max_acceptable = 3000

        # Retry logic for motors that don't get into acceptable range
        max_retries = 2  # Reduced since we're doing more work per retry
        for retry in range(max_retries):
            time.sleep(0.1)

            print(f"\nVerifying positions (attempt {retry + 1}/{max_retries})...")
            positions_com12 = bus_com12.sync_read("Present_Position", normalize=False)
            positions_com13 = bus_com13.sync_read("Present_Position", normalize=False)

            # Check which motors need retry
            motors_to_retry_com12 = []
            motors_to_retry_com13 = []

            for motor_name, position in positions_com12.items():
                if not (min_acceptable <= position <= max_acceptable) or abs(position) > 5000:
                    motors_to_retry_com12.append(motor_name)

            for motor_name, position in positions_com13.items():
                if not (min_acceptable <= position <= max_acceptable) or abs(position) > 5000:
                    motors_to_retry_com13.append(motor_name)

            if not motors_to_retry_com12 and not motors_to_retry_com13:
                break

            if retry < max_retries - 1 and (motors_to_retry_com12 or motors_to_retry_com13):
                print(f"\nRetrying {len(motors_to_retry_com12) + len(motors_to_retry_com13)} motors...")

                if motors_to_retry_com12:
                    bus_com12.disable_torque(motors_to_retry_com12)
                    time.sleep(0.1)
                    for motor_name in motors_to_retry_com12:
                        fix_problematic_motor(bus_com12, motor_name, target_position)
                    time.sleep(0.1)
                    bus_com12.enable_torque(motors_to_retry_com12)
                    time.sleep(0.2)

                if motors_to_retry_com13:
                    bus_com13.disable_torque(motors_to_retry_com13)
                    time.sleep(0.1)
                    for motor_name in motors_to_retry_com13:
                        fix_problematic_motor(bus_com13, motor_name, target_position)
                    time.sleep(0.1)
                    bus_com13.enable_torque(motors_to_retry_com13)
                    time.sleep(0.2)

        # Final verification
        time.sleep(0.2)
        positions_com12 = bus_com12.sync_read("Present_Position", normalize=False)
        positions_com13 = bus_com13.sync_read("Present_Position", normalize=False)
        time.sleep(0.1)
        positions_com12_final = bus_com12.sync_read("Present_Position", normalize=False)
        positions_com13_final = bus_com13.sync_read("Present_Position", normalize=False)

        positions_com12 = positions_com12_final
        positions_com13 = positions_com13_final

        print("\n" + "="*60)
        print(f"FINAL VERIFICATION - MOTOR POSITIONS (acceptable range: {min_acceptable}-{max_acceptable})")
        print("="*60)
        print("\nCOM12 (Motor IDs 7-12):")
        all_correct = True
        for motor_name, position in sorted(positions_com12.items()):
            motor_id = motor_name.split("_")[1]
            is_acceptable = min_acceptable <= position <= max_acceptable
            status = "✓" if is_acceptable else "✗"
            print(f"  Motor ID {motor_id:2s}: {position:5d} {status}")
            if not is_acceptable:
                all_correct = False

        print("\nCOM13 (Motor IDs 1-6):")
        for motor_name, position in sorted(positions_com13.items()):
            motor_id = motor_name.split("_")[1]
            is_acceptable = min_acceptable <= position <= max_acceptable
            status = "✓" if is_acceptable else "✗"
            print(f"  Motor ID {motor_id:2s}: {position:5d} {status}")
            if not is_acceptable:
                all_correct = False

        print("\n" + "="*60)
        if all_correct:
            print(f"✓ SUCCESS: All motors are in acceptable range ({min_acceptable}-{max_acceptable})!")
        else:
            print(f"✗ WARNING: Some motors are outside acceptable range ({min_acceptable}-{max_acceptable})")
            print("  Motors with issues may have hardware problems or need manual calibration.")

        return all_correct

    except Exception as e:
        print(f"Error setting motor offsets: {e}")
        raise
    finally:
        print("\nDisconnecting...")
        try:
            bus_com12.disconnect(disable_torque=False)
        except Exception as e:
            print(f"Warning: Error disconnecting COM12: {e}")

        try:
            bus_com13.disconnect(disable_torque=False)
        except Exception as e:
            print(f"Warning: Error disconnecting COM13: {e}")


if __name__ == "__main__":
    success = set_motor_offsets_to_2047()
    print("\nDone!")
    exit(0 if success else 1)
