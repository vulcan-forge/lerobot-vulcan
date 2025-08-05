import abc
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

NameOrID: TypeAlias = str | int

logger = logging.getLogger(__name__)


class MotorNormMode(str, Enum):
    RANGE_0_100 = "range_0_100"
    RANGE_M100_100 = "range_m100_100"
    DEGREES = "degrees"
    PWM_DUTY_CYCLE = "pwm_duty_cycle"  # 0 to 1 for PWM control


@dataclass
class DCMotor:
    id: int
    model: str
    norm_mode: MotorNormMode
    protocol: str = "pwm"  # pwm, i2c, can, serial


class ProtocolHandler(Protocol):
    """Protocol for different DC motor communication methods."""

    def connect(self) -> None:
        """Connect to the motor controller."""
        ...

    def disconnect(self) -> None:
        """Disconnect from the motor controller."""
        ...

    def set_position(self, motor_id: int, position: float) -> None:
        """Set motor position (0 to 1)."""
        ...

    def set_velocity(self, motor_id: int, velocity: float) -> None:
        """Set motor velocity (normalized -1 to 1)."""
        ...

    def set_pwm(self, motor_id: int, duty_cycle: float) -> None:
        """Set PWM duty cycle (0 to 1)."""
        ...

    def get_position(self, motor_id: int) -> float | None:
        """Get current motor position if encoder available."""
        ...

    def get_velocity(self, motor_id: int) -> float:
        """Get current motor velocity."""
        ...

    def get_pwm(self, motor_id: int) -> float:
        """Get current PWM duty cycle."""
        ...

    def enable_motor(self, motor_id: int) -> None:
        """Enable motor."""
        ...

    def disable_motor(self, motor_id: int) -> None:
        """Disable motor."""
        ...


class BaseDCMotorsController(abc.ABC):
    """
    Abstract base class for DC motor controllers.

    Concrete implementations should inherit from this class and implement
    the abstract methods for their specific protocol.
    """

    def __init__(
        self,
        motors: dict[str, DCMotor],
        protocol: str = "pwm",
        config: dict | None = None,
    ):
        self.motors = motors
        self.protocol = protocol
        self.config = config or {}

        self._id_to_name_dict = {m.id: motor for motor, m in self.motors.items()}
        self._name_to_id_dict = {motor: m.id for motor, m in self.motors.items()}

        self.protocol_handler: ProtocolHandler | None = None
        self._is_connected = False

        self._validate_motors()

    def __len__(self):
        return len(self.motors)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    Protocol: '{self.protocol}',\n"
            f"    Motors: {list(self.motors.keys())},\n"
            f"    Config: {self.config}\n"
            ")"
        )

    def _validate_motors(self) -> None:
        """Validate motor configuration."""
        if not self.motors:
            raise ValueError("At least one motor must be specified.")

        # Check for duplicate IDs
        ids = [m.id for m in self.motors.values()]
        if len(ids) != len(set(ids)):
            raise ValueError("Motor IDs must be unique.")

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def _get_motor_id(self, motor: NameOrID) -> int:
        """Get motor ID from name or ID."""
        if isinstance(motor, int):
            return motor
        elif isinstance(motor, str):
            if motor in self._name_to_id_dict:
                return self._name_to_id_dict[motor]
            else:
                raise ValueError(f"Motor '{motor}' not found.")
        else:
            raise TypeError(f"Motor must be string or int, got {type(motor)}")

    @abc.abstractmethod
    def _create_protocol_handler(self) -> ProtocolHandler:
        """Create the appropriate protocol handler based on configuration."""
        pass

    def connect(self) -> None:
        """Connect to the motor controller."""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        self.protocol_handler = self._create_protocol_handler()
        self.protocol_handler.connect()
        self._is_connected = True
        logger.info(f"{self} connected successfully.")

    def disconnect(self) -> None:
        """Disconnect from the motor controller."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.protocol_handler:
            self.protocol_handler.disconnect()

        self._is_connected = False
        logger.info(f"{self} disconnected.")

    # Position Functions
    def get_position(self, motor: NameOrID) -> float | None:
        """Get current motor position if encoder available."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        motor_id = self._get_motor_id(motor)
        return self.protocol_handler.get_position(motor_id)

    def set_position(self, motor: NameOrID, position: float) -> None:
        """Set motor position (0 to 1)."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        motor_id = self._get_motor_id(motor)
        self.protocol_handler.set_position(motor_id, position)

    # Velocity Functions
    def get_velocity(self, motor: NameOrID) -> float:
        """Get current motor velocity."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        motor_id = self._get_motor_id(motor)
        return self.protocol_handler.get_velocity(motor_id)

    def set_velocity(self, motor: NameOrID, velocity: float, normalize: bool = True) -> None:
        """
        Set motor velocity.

        Args:
            motor: Motor name or ID
            velocity: Target velocity (-1 to 1 if normalized, otherwise in RPM)
            normalize: Whether to normalize the velocity
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        motor_id = self._get_motor_id(motor)

        if normalize:
            # Clamp to [-1, 1]
            velocity = max(-1.0, min(1.0, velocity))

        self.protocol_handler.set_velocity(motor_id, velocity)
        logger.debug(f"Set motor {motor} velocity to {velocity}")

    # PWM Functions
    def get_pwm(self, motor: NameOrID) -> float:
        """Get current PWM duty cycle."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        motor_id = self._get_motor_id(motor)
        return self.protocol_handler.get_pwm(motor_id)

    def set_pwm(self, motor: NameOrID, duty_cycle: float) -> None:
        """
        Set PWM duty cycle.

        Args:
            motor: Motor name or ID
            duty_cycle: PWM duty cycle (0 to 1)
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        motor_id = self._get_motor_id(motor)

        # Clamp to [0, 1]
        duty_cycle = max(0.0, min(1.0, duty_cycle))

        self.protocol_handler.set_pwm(motor_id, duty_cycle)
        logger.debug(f"Set motor {motor} PWM to {duty_cycle}")

    # Enable/Disable Functions
    def enable_motor(self, motor: NameOrID | None = None) -> None:
        """Enable motor(s)."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if motor is None:
            # Enable all motors
            for motor_id in self._id_to_name_dict.keys():
                self.protocol_handler.enable_motor(motor_id)
        else:
            motor_id = self._get_motor_id(motor)
            self.protocol_handler.enable_motor(motor_id)

    def disable_motor(self, motor: NameOrID | None = None) -> None:
        """Disable motor(s)."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if motor is None:
            # Disable all motors
            for motor_id in self._id_to_name_dict.keys():
                self.protocol_handler.disable_motor(motor_id)
        else:
            motor_id = self._get_motor_id(motor)
            self.protocol_handler.disable_motor(motor_id)
