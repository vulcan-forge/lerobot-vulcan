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
        # Try different ways to get version
        if hasattr(gpiozero, '__version__'):
            version = gpiozero.__version__
            print(f"✓ gpiozero version: {version}")
            return True
        elif hasattr(gpiozero, 'VERSION'):
            version = gpiozero.VERSION
            print(f"✓ gpiozero version: {version}")
            return True
        else:
            print("✓ gpiozero imported (version unknown)")
            return True
    except AttributeError:
        print("✓ gpiozero imported (version unknown)")
        return True

def test_gpiozero_backend():
    """Test which GPIO backend gpiozero is using."""
    try:
        import gpiozero
        from gpiozero.pins import Factory

        # Check which factory is being used
        factory = Factory()
        print(f"✓ gpiozero backend: {factory.__class__.__name__}")

        # Try to get more info about the backend
        if hasattr(factory, 'name'):
            print(f"✓ Backend name: {factory.name}")

        return True
    except Exception as e:
        print(f"✗ Could not determine gpiozero backend: {e}")
        return False

def test_gpiozero_setup(gpiozero):
    """Test basic gpiozero setup."""
    if gpiozero is None:
        return False

    try:
        # Test LED setup (using pin 18)
        led_pin = 18
        led = gpiozero.LED(led_pin)
        print(f"✓ gpiozero LED setup on pin {led_pin} successful")

        # Test PWM setup (using different pin 12)
        pwm_pin = 12
        pwm_led = gpiozero.PWMLED(pwm_pin)
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
        if "Cannot determine SOC peripheral base address" in str(e):
            print("  → This indicates gpiozero is falling back to RPi.GPIO")
            print("  → Install lgpio: sudo apt install python3-lgpio")
        elif "already in use" in str(e):
            print("  → Pin conflict detected")
            print("  → This might be due to previous test runs")
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

def test_lgpio_availability():
    """Test if lgpio is available."""
    try:
        import lgpio
        print("✓ lgpio module available")
        return True
    except ImportError:
        print("✗ lgpio module not available")
        print("  → Install with: sudo apt install python3-lgpio")
        return False

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

    # lgpio availability
    print("3. lgpio Availability:")
    test_lgpio_availability()
    print()

    # GPIO import
    print("4. GPIO Import Test:")
    gpiozero = test_gpiozero_import()
    print()

    # GPIO backend
    print("5. GPIO Backend Test:")
    test_gpiozero_backend()
    print()

    # GPIO version
    print("6. GPIO Version Test:")
    test_gpiozero_version(gpiozero)
    print()

    # GPIO setup
    print("7. GPIO Setup Test:")
    success = test_gpiozero_setup(gpiozero)
    print()

    # Summary
    print("=== Summary ===")
    if success:
        print("✓ All gpiozero tests passed!")
        print("  → GPIO should work with your robot code")
    else:
        print("✗ gpiozero tests failed!")
        print("  → Install lgpio: sudo apt install python3-lgpio")
        print("  → Or run: uv pip install lgpio")

if __name__ == "__main__":
    main()
