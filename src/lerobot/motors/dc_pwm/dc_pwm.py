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

# Pi 5 Digital Output Pins (excluding hardware PWM pins)
PI5_DIGITAL_OUTPUT_PINS = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41
]

# Pi 5 Optimal Settings
PI5_OPTIMAL_FREQUENCY = 25000  # 25kHz - optimal for most DC motors
PI5_MAX_FREQUENCY = 50000      # 50kHz - Pi 5 can handle higher frequencies
PI5_RESOLUTION = 12            # 12-bit resolution


class PWMProtocolHandler(ProtocolHandler):
    """
    PWM protocol handler optimized for Raspberry Pi 5.

    Pi 5 Features:
    - 4 hardware PWM channels (PWM0-PWM3)
    - 40 GPIO pins total (24+ available for motor control)
    - Higher PWM frequencies (up to 50kHz)
    - Better real-time performance
    - Lower latency

    For 5+ motors:
    - First 4 motors use hardware PWM (optimal performance)
    - Additional motors use software PWM (acceptable performance)
    - All motors can have direction, enable, and brake pins

    Configuration options:
    - pwm_pins: List of PWM pin numbers (use hardware PWM pins: 12,13,14,15,18,19,20,21)
    - direction_pins: List of direction pin numbers (any GPIO pin 2-41, excluding PWM pins)
    - enable_pins: List of enable pin numbers (optional, for motor drivers with enable)
    - brake_pins: List of brake pin numbers (optional, for motor drivers with brake)
    - pwm_frequency: PWM frequency in Hz (default: 25000 for Pi 5)
    - invert_direction: Whether to invert direction logic (default: False)
    - invert_enable: Whether to invert enable logic (default: False)
    - invert_brake: Whether to invert brake logic (default: False)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.pwm_pins = config.get("pwm_pins", [])
        self.direction_pins = config.get("direction_pins", [])
        self.enable_pins = config.get("enable_pins", [])
        self.brake_pins = config.get("brake_pins", [])
        self.pwm_frequency = config.get("pwm_frequency", PI5_OPTIMAL_FREQUENCY)
        self.invert_direction = config.get("invert_direction", False)
        self.invert_enable = config.get("invert_enable", False)
        self.invert_brake = config.get("invert_brake", False)

        # Motor state tracking
        self.motors: Dict[int, Dict] = config.get("motors", {})
        self.pwm_channels = {}
        self.direction_channels = {}
        self.enable_channels = {}
        self.brake_channels = {}
        self.software_pwm_channels = {}  # For motors beyond hardware PWM

        # Validate Pi 5 pins
        self._validate_pi5_pins()

        # Import RPi.GPIO for Pi 5
        self._import_rpi_gpio()

    def _validate_pi5_pins(self):
        """Validate that pins are valid GPIO pins on Pi 5."""
        all_hardware_pwm = []
        for pwm_pins in PI5_HARDWARE_PWM_PINS.values():
            all_hardware_pwm.extend(pwm_pins)

        # Validate PWM pins
        invalid_pwm_pins = [pin for pin in self.pwm_pins if pin not in all_hardware_pwm]
        if invalid_pwm_pins:
            logger.warning(
                f"PWM pins {invalid_pwm_pins} are not hardware PWM pins on Pi 5. "
                f"Hardware PWM pins: {all_hardware_pwm}"
            )

        # Validate direction pins
        invalid_dir_pins = [pin for pin in self.direction_pins if pin not in PI5_ALL_GPIO_PINS]
        if invalid_dir_pins:
            logger.warning(
                f"Direction pins {invalid_dir_pins} are not valid GPIO pins on Pi 5. "
                f"Valid GPIO pins: {PI5_ALL_GPIO_PINS}"
            )

        # Validate enable pins
        invalid_enable_pins = [pin for pin in self.enable_pins if pin not in PI5_ALL_GPIO_PINS]
        if invalid_enable_pins:
            logger.warning(
                f"Enable pins {invalid_enable_pins} are not valid GPIO pins on Pi 5. "
                f"Valid GPIO pins: {PI5_ALL_GPIO_PINS}"
            )

        # Validate brake pins
        invalid_brake_pins = [pin for pin in self.brake_pins if pin not in PI5_ALL_GPIO_PINS]
        if invalid_brake_pins:
            logger.warning(
                f"Brake pins {invalid_brake_pins} are not valid GPIO pins on Pi 5. "
                f"Valid GPIO pins: {PI5_ALL_GPIO_PINS}"
            )

        # Check for pin conflicts
        all_used_pins = set(self.pwm_pins + self.direction_pins + self.enable_pins + self.brake_pins)
        if len(all_used_pins) != len(self.pwm_pins + self.direction_pins + self.enable_pins + self.brake_pins):
            logger.warning("Duplicate pins detected in configuration")

        # Validate motor count vs available pins
        motor_count = len(self.pwm_pins)
        if motor_count > 4:
            logger.info(f"Configuring {motor_count} motors: {min(4, motor_count)} hardware PWM + {max(0, motor_count - 4)} software PWM")

    def _import_rpi_gpio(self):
        """Import RPi.GPIO with Pi 5 support."""
        try:
            import RPi.GPIO as GPIO # type: ignore
            self.GPIO = GPIO

            # Check RPi.GPIO version for Pi 5 support
            if hasattr(self.GPIO, 'VERSION'):
                version = self.GPIO.VERSION
                logger.info(f"RPi.GPIO version: {version}")

                if version < "0.7.1":
                    logger.warning(
                        "RPi.GPIO version < 0.7.1 detected. "
                        "Pi 5 support requires version 0.7.1 or later. "
                        "Consider upgrading: uv pip install -e '.[sourccey]'"
                    )

            logger.info("Using RPi.GPIO for Pi 5 PWM control")

        except ImportError:
            raise ImportError(
                "RPi.GPIO not available. Install with: pip install RPi.GPIO>=0.7.1"
            )

    def connect(self) -> None:
        """Initialize GPIO and PWM channels for Pi 5."""
        self.GPIO.setmode(self.GPIO.BCM)
        self.GPIO.setwarnings(False)

        # Initialize PWM pins
        for i, pin in enumerate(self.pwm_pins):
            motor_id = i + 1
            self.GPIO.setup(pin, self.GPIO.OUT)

            # Use hardware PWM for first 4 motors, software PWM for rest
            if i < 4:
                # Hardware PWM
                pwm_channel = self.GPIO.PWM(pin, self.pwm_frequency)
                pwm_channel.start(0)  # Start with 0% duty cycle
                self.pwm_channels[motor_id] = pwm_channel
                logger.debug(f"Motor {motor_id} using hardware PWM on pin {pin}")
            else:
                # Software PWM
                pwm_channel = self.GPIO.PWM(pin, self.pwm_frequency)
                pwm_channel.start(0)  # Start with 0% duty cycle
                self.software_pwm_channels[motor_id] = pwm_channel
                logger.debug(f"Motor {motor_id} using software PWM on pin {pin}")

            # Initialize direction pin if available
            if i < len(self.direction_pins):
                dir_pin = self.direction_pins[i]
                self.GPIO.setup(dir_pin, self.GPIO.OUT)
                self.direction_channels[motor_id] = dir_pin

            # Initialize enable pin if available
            if i < len(self.enable_pins):
                enable_pin = self.enable_pins[i]
                self.GPIO.setup(enable_pin, self.GPIO.OUT)
                self.enable_channels[motor_id] = enable_pin

            # Initialize brake pin if available
            if i < len(self.brake_pins):
                brake_pin = self.brake_pins[i]
                self.GPIO.setup(brake_pin, self.GPIO.OUT)
                self.brake_channels[motor_id] = brake_pin

            # Initialize motor state
            self.motors[motor_id] = {
                "position": 0.0,
                "velocity": 0.0,
                "pwm": 0.0,
                "enabled": False,
                "brake_active": False
            }

        total_pins = len(self.pwm_pins) + len(self.direction_pins) + len(self.enable_pins) + len(self.brake_pins)
        hw_pwm_count = min(4, len(self.pwm_pins))
        sw_pwm_count = max(0, len(self.pwm_pins) - 4)
        logger.info(f"Pi 5 PWM protocol handler connected with {len(self.pwm_pins)} motors using {total_pins} GPIO pins")
        logger.info(f"Hardware PWM: {hw_pwm_count} motors, Software PWM: {sw_pwm_count} motors at {self.pwm_frequency}Hz")

    def disconnect(self) -> None:
        """Clean up GPIO and PWM channels."""
        # Stop all PWM channels
        for pwm_channel in self.pwm_channels.values():
            pwm_channel.stop()

        for pwm_channel in self.software_pwm_channels.values():
            pwm_channel.stop()

        # Clean up GPIO
        self.GPIO.cleanup()

        logger.info("Pi 5 PWM protocol handler disconnected")

    # Position Functions
    def get_position(self, motor_id: int) -> Optional[float]:
        """Get current motor position if encoder available."""
        # For PWM-only control, we can only estimate position
        # In a real implementation, you'd read from an encoder
        return self.motors.get(motor_id, {}).get("position", 0.0)

    def set_position(self, motor_id: int, position: float) -> None:
        """
        Set motor position (0 to 1).
        Note: This is a simplified implementation. For precise position control,
        you'd need encoders and PID control.
        """
        # For PWM-only control, position is limited
        if position < 0:
            position = 0
        elif position > 1:
            position = 1

        self.motors[motor_id]["position"] = position

        # Convert position to PWM (simple linear mapping)
        pwm_duty = position
        self.set_pwm(motor_id, pwm_duty)

    # Velocity Functions
    def get_velocity(self, motor_id: int) -> float:
        """Get current motor velocity."""
        return self.motors.get(motor_id, {}).get("velocity", 0.0)

    def set_velocity(self, motor_id: int, velocity: float) -> None:
        """
        Set motor velocity (normalized -1 to 1).
        Negative values = reverse, Positive values = forward.
        """
        # Clamp velocity to [-1, 1]
        velocity = max(-1.0, min(1.0, velocity))

        self.motors[motor_id]["velocity"] = velocity

        # Convert velocity to PWM and direction
        abs_velocity = abs(velocity)
        direction = velocity >= 0

        # Set direction if direction pin is available
        if motor_id in self.direction_channels:
            self._set_direction(motor_id, direction)

        # Set PWM duty cycle
        self.set_pwm(motor_id, abs_velocity)

    # PWM Functions
    def get_pwm(self, motor_id: int) -> float:
        """Get current PWM duty cycle."""
        return self.motors.get(motor_id, {}).get("pwm", 0.0)

    def set_pwm(self, motor_id: int, duty_cycle: float) -> None:
        """
        Set PWM duty cycle (0 to 1).
        """
        # Clamp duty cycle to [0, 1]
        duty_cycle = max(0.0, min(1.0, duty_cycle))

        self.motors[motor_id]["pwm"] = duty_cycle

        # Use appropriate PWM channel (hardware or software)
        if motor_id in self.pwm_channels:
            # Hardware PWM
            self.pwm_channels[motor_id].ChangeDutyCycle(duty_cycle * 100)
        elif motor_id in self.software_pwm_channels:
            # Software PWM
            self.software_pwm_channels[motor_id].ChangeDutyCycle(duty_cycle * 100)
        else:
            logger.warning(f"Motor {motor_id} not found in PWM channels")

        logger.debug(f"Motor {motor_id} PWM set to {duty_cycle:.3f}")

    # Enable/Disable Functions
    def enable_motor(self, motor_id: int) -> None:
        """Enable motor."""
        self.motors[motor_id]["enabled"] = True
        self._set_enable(motor_id, True)
        logger.debug(f"Motor {motor_id} enabled")

    def disable_motor(self, motor_id: int) -> None:
        """Disable motor by setting PWM to 0 and disabling enable pin."""
        self.set_pwm(motor_id, 0.0)
        self.motors[motor_id]["enabled"] = False
        self._set_enable(motor_id, False)
        logger.debug(f"Motor {motor_id} disabled")

    # Helper methods for PWM-specific functionality
    def _get_direction(self, motor_id: int) -> bool:
        """Get motor direction."""
        if motor_id not in self.direction_channels:
            return False
        return self.GPIO.input(self.direction_channels[motor_id]) == self.GPIO.HIGH

    def _set_direction(self, motor_id: int, forward: bool) -> None:
        """Set motor direction."""
        if motor_id not in self.direction_channels:
            return

        # Apply direction inversion if configured
        if self.invert_direction:
            forward = not forward

        self.GPIO.output(self.direction_channels[motor_id], self.GPIO.HIGH if forward else self.GPIO.LOW)

    # Enable/Disable Functions
    def _set_enable(self, motor_id: int, enabled: bool) -> None:
        """Set motor enable state."""
        if motor_id not in self.enable_channels:
            return

        # Apply enable inversion if configured
        if self.invert_enable:
            enabled = not enabled

        self.GPIO.output(self.enable_channels[motor_id], self.GPIO.HIGH if enabled else self.GPIO.LOW)

    def _set_brake(self, motor_id: int, brake_active: bool) -> None:
        """Set motor brake state."""
        if motor_id not in self.brake_channels:
            return

        # Apply brake inversion if configured
        if self.invert_brake:
            brake_active = not brake_active

        self.GPIO.output(self.brake_channels[motor_id], self.GPIO.HIGH if brake_active else self.GPIO.LOW)
        self.motors[motor_id]["brake_active"] = brake_active

    # PWM-specific convenience methods (not part of base ProtocolHandler)
    def activate_brake(self, motor_id: int) -> None:
        """Activate motor brake."""
        self._set_brake(motor_id, True)
        logger.debug(f"Motor {motor_id} brake activated")

    def release_brake(self, motor_id: int) -> None:
        """Release motor brake."""
        self._set_brake(motor_id, False)
        logger.debug(f"Motor {motor_id} brake released")


class PWMDCMotorsController(BaseDCMotorsController):
    """PWM-based DC motor controller optimized for Raspberry Pi 5."""

    def __init__(self, motors: dict[str, DCMotor], config: dict):
        super().__init__(motors, config)

    def _create_protocol_handler(self) -> ProtocolHandler:
        return PWMProtocolHandler(self.config)
