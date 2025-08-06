#!/usr/bin/env python3
"""
Simple GPIO test to diagnose Raspberry Pi GPIO issues.
"""

import sys
import time

def test_gpio_import():
    """Test if RPi.GPIO can be imported."""
    try:
        import RPi.GPIO as GPIO # type: ignore
        print("✓ RPi.GPIO imported successfully")
        return GPIO
    except ImportError as e:
        print(f"✗ RPi.GPIO import failed: {e}")
        return None

def test_gpio_version(gpio):
    """Test RPi.GPIO version."""
    if gpio is None:
        return False

    try:
        version = gpio.VERSION
        print(f"✓ RPi.GPIO version: {version}")
        return True
    except AttributeError:
        print("✗ Could not determine RPi.GPIO version")
        return False

def test_gpio_setup(gpio):
    """Test basic GPIO setup."""
    if gpio is None:
        return False

    try:
        # Test BCM mode setup
        gpio.setmode(gpio.BCM)
        print("✓ GPIO BCM mode set successfully")

        # Test pin setup (using a safe pin like 18)
        test_pin = 18
        gpio.setup(test_pin, gpio.OUT)
        print(f"✓ GPIO pin {test_pin} setup successfully")

        # Test PWM setup
        pwm = gpio.PWM(test_pin, 1000)  # 1kHz frequency
        pwm.start(0)
        print("✓ PWM setup successful")

        # Test PWM control
        pwm.ChangeDutyCycle(50)  # 50% duty cycle
        time.sleep(0.1)
        pwm.ChangeDutyCycle(0)
        print("✓ PWM control test successful")

        # Cleanup
        pwm.stop()
        gpio.cleanup()
        print("✓ GPIO cleanup successful")

        return True

    except RuntimeError as e:
        print(f"✗ GPIO setup failed: {e}")
        if "Cannot determine SOC peripheral base address" in str(e):
            print("  → This indicates you're not running on a Raspberry Pi")
            print("  → Or GPIO permissions are insufficient")
        return False
    except Exception as e:
        print(f"✗ GPIO test failed: {e}")
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
        print("  → GPIO may require root permissions")
        print("  → Try: sudo python3 gpio_test.py")

def main():
    """Run all GPIO tests."""
    print("=== GPIO Test Suite ===")
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
    gpio = test_gpio_import()
    print()

    # GPIO version
    print("4. GPIO Version Test:")
    test_gpio_version(gpio)
    print()

    # GPIO setup
    print("5. GPIO Setup Test:")
    success = test_gpio_setup(gpio)
    print()

    # Summary
    print("=== Summary ===")
    if success:
        print("✓ All GPIO tests passed!")
        print("  → GPIO should work with your robot code")
    else:
        print("✗ GPIO tests failed!")
        print("  → Use MockPWMProtocolHandler for testing")
        print("  → Or run on actual Raspberry Pi hardware")

if __name__ == "__main__":
    main()
