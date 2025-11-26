def diagnose_problematic_motors(bus, motor_names):
    """Diagnose what's wrong with problematic motors."""
    print("\n" + "="*60)
    print("DIAGNOSTIC INFORMATION FOR PROBLEMATIC MOTORS")
    print("="*60)

    for motor_name in motor_names:
        motor_id = bus.motors[motor_name].id
        print(f"\n{motor_name} (ID {motor_id}):")
        print("-" * 40)

        # Read homing offset
        try:
            offset = bus.read("Homing_Offset", motor_name, normalize=False, num_retry=3)
            print(f"  Homing_Offset: {offset}")
        except Exception as e:
            print(f"  Homing_Offset: ERROR - {e}")

        # Read present position (decoded)
        try:
            pos_decoded = bus.read("Present_Position", motor_name, normalize=False, num_retry=3)
            print(f"  Present_Position (decoded): {pos_decoded}")
        except Exception as e:
            print(f"  Present_Position (decoded): ERROR - {e}")

        # Read raw register
        try:
            raw_val, comm, error = read_raw_register(bus, motor_name, "Present_Position")
            print(f"  Present_Position (raw): {raw_val} (0x{raw_val:04X})")
            print(f"  Communication status: {comm}")
            print(f"  Error status: {error}")

            # Manually decode
            decoded_manual = decode_sign_magnitude(raw_val, 15)
            print(f"  Manually decoded: {decoded_manual}")

            # Check if raw value makes sense
            if raw_val > 0x7FFF:
                print(f"  WARNING: Raw value exceeds 15-bit range!")
            if decoded_manual > 4095 or decoded_manual < -2047:
                print(f"  WARNING: Decoded value is outside normal range!")
        except Exception as e:
            print(f"  Raw register read: ERROR - {e}")

        # Read other status registers
        try:
            voltage = bus.read("Present_Voltage", motor_name, normalize=False, num_retry=3)
            print(f"  Present_Voltage: {voltage} (should be ~120-140 for 12V)")
        except Exception as e:
            print(f"  Present_Voltage: ERROR - {e}")

        try:
            temp = bus.read("Present_Temperature", motor_name, normalize=False, num_retry=3)
            print(f"  Present_Temperature: {temp}°C")
        except Exception as e:
            print(f"  Present_Temperature: ERROR - {e}")

        try:
            status = bus.read("Status", motor_name, normalize=False, num_retry=3)
            print(f"  Status register: {status} (0x{status:02X})")
            if status != 0:
                print(f"  WARNING: Status register shows errors!")
        except Exception as e:
            print(f"  Status register: ERROR - {e}")
