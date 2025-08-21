import logging
from typing import Dict, Optional, List

from lerobot.motors.dc_motors_controller import BaseDCMotorsController, DCMotor, ProtocolHandler

logger = logging.getLogger(__name__)


# Pi 5 Hardware PWM Configuration
PI5_HARDWARE_PWM_PINS = {
    "pwm0": [12, 18],  # PWM0 channels
    "pwm1": [13, 19],  # PWM1 channels
    "pwm2": [14, 20],  # PWM2 channels
    "pwm3": [15, 21],  # PWM3 channels
}

# Pi 5 All Available GPIO Pins (40-pin header)
PI5_ALL_GPIO_PINS = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41
]

# Pi 5 Optimal Settings for DRV8871DDAR
PI5_OPTIMAL_FREQUENCY = 1000  # 1kHz - more compatible with gpiozero
PI5_MAX_FREQUENCY = 50000      # 50kHz - Pi 5 can handle higher frequencies
PI5_RESOLUTION = 12            # 12-bit resolution


class PWMProtocolHandler(ProtocolHandler):
    """
    PWM protocol handler optimized for DRV8871DDAR H-bridge motor drivers.

    DRV8871DDAR Features:
    - IN1: PWM speed control (hardware PWM recommended)
    - IN2: Direction control (regular GPIO)
    - Built-in current limiting and thermal protection
    - 25kHz PWM frequency optimal

    Configuration:
    - pwm_pins: IN1 pins (PWM speed control)
    - direction_pins: IN2 pins (direction control)
    - pwm_frequency: 25000Hz (optimal for DRV8871DDAR)
    """

    def __init__(self, config: Dict, motors: Dict[str, DCMotor]):
        self.config = config
        self.pwm_pins = config.get("pwm_pins", [])
        self.direction_pins = config.get("direction_pins", [])
        self.enable_pins = config.get("enable_pins", [])
        self.brake_pins = config.get("brake_pins", [])
        self.pwm_frequency = config.get("pwm_frequency", PI5_OPTIMAL_FREQUENCY)
        self.invert_direction = config.get("invert_direction", False)
        self.invert_enable = config.get("invert_enable", False)
        self.invert_brake = config.get("invert_brake", False)

        # Motor configuration and state tracking
        self.motors: Dict[str, DCMotor] = motors
        self.motor_states: Dict[int, Dict] = {}  # Track motor state by ID
        self.pwm_channels = {}
        self.direction_channels = {}
        self.enable_channels = {}
        self.brake_channels = {}

        # Validate Pi 5 pins
        self._validate_pi5_pins()

        # Import gpiozero
        self._import_gpiozero()

    def _validate_pi5_pins(self):
        """Validate that pins are valid GPIO pins on Pi 5."""
        all_hardware_pwm = []
        for pwm_pins in PI5_HARDWARE_PWM_PINS.values():
            all_hardware_pwm.extend(pwm_pins)

        # Validate PWM pins (IN1 - should be hardware PWM for best performance)
        invalid_pwm_pins = [pin for pin in self.pwm_pins if pin not in all_hardware_pwm]
        if invalid_pwm_pins:
            logger.warning(
                f"PWM pins {invalid_pwm_pins} are not hardware PWM pins on Pi 5. "
                f"Hardware PWM pins: {all_hardware_pwm}"
            )

        # Validate direction pins (IN2 - can be any GPIO)
        invalid_dir_pins = [pin for pin in self.direction_pins if pin not in PI5_ALL_GPIO_PINS]
        if invalid_dir_pins:
            logger.warning(
                f"Direction pins {invalid_dir_pins} are not valid GPIO pins on Pi 5. "
                f"Valid GPIO pins: {PI5_ALL_GPIO_PINS}"
            )

        # Check for pin conflicts
        all_used_pins = set(self.pwm_pins + self.direction_pins + self.enable_pins + self.brake_pins)
        if len(all_used_pins) != len(self.pwm_pins + self.direction_pins + self.enable_pins + self.brake_pins):
            logger.warning("Duplicate pins detected in configuration")

        # Validate motor count
        motor_count = len(self.pwm_pins)
        logger.info(f"Configuring {motor_count} DRV8871DDAR motors with gpiozero")

    def _import_gpiozero(self):
        """Import gpiozero."""
        try:
            import gpiozero
            self.gpiozero = gpiozero
            logger.info("Using gpiozero for DRV8871DDAR motor control")

        except ImportError:
            raise ImportError(
                "gpiozero not available. Install with: uv pip install gpiozero>=2.0"
            )

    def connect(self) -> None:
        """Initialize gpiozero for DRV8871DDAR motor drivers."""
        try:
            # Initialize motor pins
            print()
            print()
            print()
            print()
            print()
            print()
            print()
            print()
            print("pwm_pins", self.pwm_pins)
            for i, pin in enumerate(self.pwm_pins):
                motor_id = i + 1
                print("motor_id", motor_id)

                # Initialize motor state
                self.motor_states[motor_id] = {
                    "position": 0.0,
                    "velocity": 0.0,
                    "pwm": 0.0,
                    "enabled": False,
                    "brake_active": False
                }

                print()
                print()
                print()
                print()
                print()
                print()
                print()
                print()

                try:
                    # IN1 pin - PWM for speed control (hardware PWM recommended)
                    # Use a more compatible frequency
                    pwm_led = self.gpiozero.PWMLED(pin, frequency=1000)
                    print("pwm_led", pwm_led)
                    pwm_led.off()  # Start with 0% duty cycle
                    print("pwm_led.off()", pwm_led)
                    self.pwm_channels[motor_id] = pwm_led
                    print("self.pwm_channels[motor_id]", self.pwm_channels[motor_id])
                    print()
                    print()
                    print()
                    print()
                    print()
                    print()
                    print()
                    print()
                    logger.debug(f"Motor {motor_id} IN1 (PWM) setup on pin {pin}")
                    print()
                    print()
                    print()
                    print()
                    print()
                    print()
                    print()
                    print()
                except Exception as e:
                    logger.warning(f"Could not setup IN1 (PWM) pin {pin}: {e}")
                    # Try with default frequency if custom frequency fails
                    try:
                        pwm_led = self.gpiozero.PWMLED(pin)  # Use default frequency
                        pwm_led.off()
                        self.pwm_channels[motor_id] = pwm_led
                        logger.debug(f"Motor {motor_id} IN1 (PWM) setup on pin {pin} with default frequency")
                    except Exception as e2:
                        logger.error(f"Failed to setup PWM pin {pin} even with default frequency: {e2}")

                # IN2 pin - Direction control (regular GPIO)
                if i < len(self.direction_pins):
                    try:
                        dir_pin = self.direction_pins[i]
                        direction_led = self.gpiozero.LED(dir_pin)
                        direction_led.off()  # Start with direction off
                        self.direction_channels[motor_id] = direction_led
                        logger.debug(f"Motor {motor_id} IN2 (direction) setup on pin {dir_pin}")
                    except Exception as e:
                        logger.warning(f"Could not setup IN2 (direction) pin {dir_pin}: {e}")

            total_pins = len(self.pwm_pins) + len(self.direction_pins)
            logger.info(f"DRV8871DDAR motor driver setup with {len(self.pwm_pins)} motors using {total_pins} GPIO pins")
            logger.info(f"PWM frequency: 1kHz (compatible with gpiozero)")

        except Exception as e:
            logger.error(f"gpiozero setup failed: {e}")
            raise RuntimeError(f"gpiozero hardware not available")

    def disconnect(self) -> None:
        """Clean up gpiozero channels."""
        # Close all PWM channels (IN1)
        for pwm_channel in self.pwm_channels.values():
            try:
                pwm_channel.close()
            except Exception as e:
                logger.warning(f"Error closing PWM channel: {e}")

        # Close all direction channels (IN2)
        for direction_channel in self.direction_channels.values():
            try:
                direction_channel.close()
            except Exception as e:
                logger.warning(f"Error closing direction channel: {e}")

        logger.info("DRV8871DDAR motor driver disconnected")

    # Position Functions
    def get_position(self, motor_id: int) -> Optional[float]:
        """Get current motor position if encoder available."""
        return self.motor_states.get(motor_id, {}).get("position", 0.0)

    def set_position(self, motor_id: int, position: float) -> None:
        """
        Set motor position (0 to 1).
        Note: This is a simplified implementation. For precise position control,
        you'd need encoders and PID control.
        """
        if position < 0:
            position = 0
        elif position > 1:
            position = 1

        self.motor_states[motor_id]["position"] = position

        # Convert position to PWM (simple linear mapping)
        pwm_duty = position
        self.set_pwm(motor_id, pwm_duty)

    # Velocity Functions
    def get_velocity(self, motor_id: int) -> float:
        """Get current motor velocity."""
        return self.motor_states.get(motor_id, {}).get("velocity", 0.0)

    def set_velocity(self, motor_id: int, velocity: float) -> None:
        """
        Set motor velocity (normalized -1 to 1) for DRV8871DDAR.
        Negative values = reverse, Positive values = forward.
        """
        # Clamp velocity to [-1, 1]
        velocity = max(-1.0, min(1.0, velocity))

        self.motor_states[motor_id]["velocity"] = velocity
        self.motor_states[motor_id]["pwm"] = abs(velocity)
        # Reset brake state when setting velocity
        self.motor_states[motor_id]["brake_active"] = False

        # Correct DRV8871DDAR Logic (from datasheet):
        print("pwm_channels", self.pwm_channels)
        print("direction_channels", self.direction_channels)
        if velocity > 0:  # Forward
            # IN1 = 1, IN2 = 0 (Forward)
            self.pwm_channels[motor_id].on()
            self.direction_channels[motor_id].off()
        elif velocity < 0:  # Backward
            # IN1 = 0, IN2 = 1 (Reverse)
            self.pwm_channels[motor_id].off()
            self.direction_channels[motor_id].on()
        else:  # Stop
            # IN1 = 0, IN2 = 0 (Coast/Stop)
            self.pwm_channels[motor_id].off()
            self.direction_channels[motor_id].off()

    # PWM Functions
    def get_pwm(self, motor_id: int) -> float:
        """Get current PWM duty cycle."""
        return self.motor_states.get(motor_id, {}).get("pwm", 0.0)

    def set_pwm(self, motor_id: int, duty_cycle: float) -> None:
        """
        Set PWM duty cycle (0 to 1) for DRV8871DDAR IN1 pin.
        """
        # Clamp duty cycle to [0, 1]
        duty_cycle = max(0.0, min(1.0, duty_cycle))

        self.motor_states[motor_id]["pwm"] = duty_cycle

        # Use gpiozero PWM for IN1 speed control
        if motor_id in self.pwm_channels:
            try:
                pwm_channel = self.pwm_channels[motor_id]
                pwm_channel.value = duty_cycle
                logger.debug(f"Motor {motor_id} IN1 PWM set to {duty_cycle:.3f}")
            except Exception as e:
                logger.warning(f"Error setting PWM for motor {motor_id}: {e}")
        else:
            logger.warning(f"Motor {motor_id} not found in PWM channels")

    # Enable/Disable Functions
    def enable_motor(self, motor_id: int) -> None:
        """Enable motor."""
        self.motor_states[motor_id]["enabled"] = True
        logger.debug(f"Motor {motor_id} enabled")

    def disable_motor(self, motor_id: int) -> None:
        """Disable motor by setting PWM to 0."""
        self.set_pwm(motor_id, 0.0)
        self.motor_states[motor_id]["enabled"] = False
        logger.debug(f"Motor {motor_id} disabled")

    # Helper methods for DRV8871DDAR-specific functionality
    def _get_direction(self, motor_id: int) -> bool:
        """Get motor direction."""
        if motor_id not in self.direction_channels:
            return False
        return self.direction_channels[motor_id].value == 1

    def _set_direction(self, motor_id: int, forward: bool) -> None:
        """
        Set motor direction for DRV8871DDAR IN2 pin.
        Forward: IN1=PWM, IN2=LOW
        Backward: IN1=LOW, IN2=PWM
        """
        if motor_id not in self.direction_channels:
            return

        # Apply direction inversion if configured
        if self.invert_direction:
            forward = not forward

        try:
            # Set IN2 for direction control
            self.direction_channels[motor_id].on() if forward else self.direction_channels[motor_id].off()
            logger.debug(f"Motor {motor_id} direction set to {'forward' if forward else 'backward'}")
        except Exception as e:
            logger.warning(f"Error setting direction for motor {motor_id}: {e}")

    # DRV8871DDAR-specific convenience methods
    def activate_brake(self, motor_id: int) -> None:
        """
        Activate motor brake for DRV8871DDAR.
        Brake: IN1=1, IN2=1 (both inputs high)
        """
        if motor_id in self.pwm_channels and motor_id in self.direction_channels:
            try:
                # Set both IN1 and IN2 high for brake
                self.pwm_channels[motor_id].on()
                self.direction_channels[motor_id].on()
                self.motor_states[motor_id]["brake_active"] = True
                logger.debug(f"Motor {motor_id} brake activated")
            except Exception as e:
                logger.warning(f"Error activating brake for motor {motor_id}: {e}")

    def release_brake(self, motor_id: int) -> None:
        """
        Release motor brake for DRV8871DDAR.
        Stop: IN1=0, IN2=0 (coast mode)
        """
        if motor_id in self.pwm_channels and motor_id in self.direction_channels:
            try:
                # Set both IN1 and IN2 low for stop
                self.pwm_channels[motor_id].off()
                self.direction_channels[motor_id].off()
                self.motor_states[motor_id]["brake_active"] = False
                logger.debug(f"Motor {motor_id} brake released")
            except Exception as e:
                logger.warning(f"Error releasing brake for motor {motor_id}: {e}")


class PWMDCMotorsController(BaseDCMotorsController):
    """PWM-based DC motor controller optimized for DRV8871DDAR H-bridge drivers."""

    def __init__(self, config: dict | None = None, motors: dict[str, DCMotor] | None = None, protocol: str = "pwm"):
        super().__init__(config, motors, protocol)

    def _create_protocol_handler(self) -> ProtocolHandler:
        return PWMProtocolHandler(self.config, self.motors)
