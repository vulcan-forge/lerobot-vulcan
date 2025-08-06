#!/usr/bin/env python3
"""
Simple GPIO test using gpiozero (modern GPIO library for Raspberry Pi).
"""

import sys
import time

def test_gpiozero_import():
    """Test if gpiozero can be imported."""
    try:
        import gpiozero
        print("✓ gpiozero imported successfully")
        return gpiozero
    except ImportError as e:
        print(f"✗ gpiozero import failed: {e}")
        return None

def test_gpiozero_version(gpiozero):
    """Test gpiozero version."""
    if gpiozero is None:
        return False

    try:
        version = gpiozero.__version__
        print(f"✓ gpiozero version: {version}")
        return True
    except AttributeError:
        print("✗ Could not determine gpiozero version")
        return False

def test_gpiozero_setup(gpiozero):
    """Test basic gpiozero setup."""
    if gpiozero is None:
        return False

    try:
        # Test LED setup (using a safe pin like 18)
        test_pin = 18
        led = gpiozero.LED(test_pin)
        print(f"✓ gpiozero LED setup on pin {test_pin} successful")

        # Test PWM setup
        pwm_led = gpiozero.PWMLED(test_pin)
        print("✓ gpiozero PWM setup successful")

        # Test PWM control
        pwm_led.pulse(fade_in_time=0.1, fade_out_time=0.1, n=1)
        print("✓ gpiozero PWM control test successful")

        # Test basic LED control
        led.on()
        time.sleep(0.1)
        led.off()
        print("✓ gpiozero LED control test successful")

        # Cleanup
        led.close()
        pwm_led.close()
        print("✓ gpiozero cleanup successful")

        return True

    except Exception as e:
        print(f"✗ gpiozero setup failed: {e}")
        return False

def test_system_info():
    """Test system information."""
    import platform
    import os

    print(f"System: {platform.system()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Platform: {platform.platform()}")

    # Check if we're on a Raspberry Pi
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'Raspberry Pi' in cpuinfo:
                print("✓ Running on Raspberry Pi")
                return True
            else:
                print("✗ Not running on Raspberry Pi")
                return False
    except FileNotFoundError:
        print("✗ Cannot read /proc/cpuinfo")
        return False

def test_permissions():
    """Test GPIO permissions."""
    import os

    # Check if running as root
    if os.geteuid() == 0:
        print("✓ Running as root (good for GPIO)")
    else:
        print("⚠ Running as non-root user")
        print("  → gpiozero should work without root permissions")
        print("  → If it fails, try: sudo python3 gpio_test.py")

def main():
    """Run all GPIO tests."""
    print("=== GPIO Test Suite (gpiozero) ===")
    print()

    # System info
    print("1. System Information:")
    test_system_info()
    print()

    # Permissions
    print("2. Permission Check:")
    test_permissions()
    print()

    # GPIO import
    print("3. GPIO Import Test:")
    gpiozero = test_gpiozero_import()
    print()

    # GPIO version
    print("4. GPIO Version Test:")
    test_gpiozero_version(gpiozero)
    print()

    # GPIO setup
    print("5. GPIO Setup Test:")
    success = test_gpiozero_setup(gpiozero)
    print()

    # Summary
    print("=== Summary ===")
    if success:
        print("✓ All gpiozero tests passed!")
        print("  → GPIO should work with your robot code")
    else:
        print("✗ gpiozero tests failed!")
        print("  → Check gpiozero installation")
        print("  → Or run on actual Raspberry Pi hardware")

if __name__ == "__main__":
    main()
