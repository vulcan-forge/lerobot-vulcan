import abc
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, TypeAlias
from typing import Callable

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

NameOrID: TypeAlias = str | int

logger = logging.getLogger(__name__)

class MotorNormMode(str, Enum):
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

    def set_position(self, motor_id: int, position: float, *args, **kwargs):
        """Set motor position / target. Implementations may support closed-loop position via extra args."""
        ...

    def set_velocity(self, motor_id: int, velocity: float, instant: bool = True) -> None:
        """Set motor velocity (normalized -1 to 1)."""
        ...

    def update_velocity(self, motor_id: int, max_step: float = 1.0) -> None:
        """Update motor velocity."""
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

    def set_pwm(self, motor_id: int, duty_cycle: float) -> None:
        """Set PWM duty cycle (0 to 1)."""
        ...

    def enable_motor(self, motor_id: int) -> None:
        """Enable motor."""
        ...

    def disable_motor(self, motor_id: int) -> None:
        """Disable motor."""
        ...


@dataclass(slots=True)
class SetPositionCmd:
    target_position: float
    get_position: Callable[[NameOrID], float]
    kp: float
    tolerance: float
    dt: float
    timeout_s: float
    max_velocity: float
    min_velocity: float
    settle_steps: int

    generation: int
    start_t: float
    in_tol_count: int = 0
    done: bool = False
    success: bool = False
    done_event: threading.Event = field(default_factory=threading.Event)


@dataclass(slots=True)
class SetPositionResult:
    generation: int
    success: bool
    finished_t: float


class BaseDCMotorsController(abc.ABC):
    """
    Abstract base class for DC motor controllers.

    Concrete implementations should inherit from this class and implement
    the abstract methods for their specific protocol.
    """

    def __init__(
        self,
        config: dict | None = None,
        motors: dict[str, DCMotor] | None = None,
        protocol: str = "pwm",
    ):
        self.config = config or {}
        self.motors = motors or {}
        self.protocol = protocol

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
            f"    Config: {self.config}\n"
            f"    Motors: {list(self.motors.keys())}\n"
            f"    Protocol: '{self.protocol}'\n"
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

    ##############################################################################################################################
    # Connection
    ##############################################################################################################################

    def connect(self) -> None:
        """Connect to the motor controller."""
        if self._is_connected:
            logger.info(f"{self} is already connected.")
            return

        self.protocol_handler = self._create_protocol_handler()
        self.protocol_handler.connect()
        self._is_connected = True
        logger.info(f"{self} connected successfully.")

    def disconnect(self) -> None:
        """Disconnect from the motor controller."""
        if not self._is_connected:
            logger.info(f"{self} is not connected.")
            return

        # Stop protocol-handler background "set position" workers (if present) so they don't write while IO is torn down.
        ph = self.protocol_handler
        if ph is not None and hasattr(ph, "move_cv"):
            try:
                with ph.move_cv:  # type: ignore[attr-defined]
                    ph.move_stop_all = True  # type: ignore[attr-defined]
                    for cmd in ph.move_cmds.values():  # type: ignore[attr-defined]
                        cmd.done = True
                        cmd.success = False
                        cmd.done_event.set()
                    ph.move_cv.notify_all()  # type: ignore[attr-defined]

                for t in list(ph.move_threads.values()):  # type: ignore[attr-defined]
                    t.join(timeout=1.0)

                with ph.move_cv:  # type: ignore[attr-defined]
                    ph.move_threads.clear()  # type: ignore[attr-defined]
                    ph.move_cmds.clear()  # type: ignore[attr-defined]
                    ph.move_stop_all = False  # type: ignore[attr-defined]
            except Exception:
                # Best-effort shutdown; protocol handler may not support these fields.
                pass

        if self.protocol_handler:
            self.protocol_handler.disconnect()

        self._is_connected = False
        logger.info(f"{self} disconnected.")

    ##############################################################################################################################
    # Position Functions
    ##############################################################################################################################

    def get_position(self, motor: NameOrID) -> float | None:
        """Get current motor position if encoder available."""
        if not self._is_connected:
            logger.info(f"{self} is not connected.")
            return None

        motor_id = self._get_motor_id(motor)
        return self.protocol_handler.get_position(motor_id)

    def set_position(
        self,
        motor: NameOrID,
        target_position: float,
        get_position: Callable[[NameOrID], float],
        *,
        kp: float = 2.0,
        tolerance: float = 0.01,
        dt: float = 0.02,
        timeout_s: float = 5.0,
        max_velocity: float = 1.0,
        min_velocity: float = 0.08,
        settle_steps: int = 5,
        blocking: bool = False,
    ) -> bool:
        """
        Set motor position.

        Args:
            motor: Motor name or ID
            target_position: Target position (-100 to 100)
            get_position: Function to get current motor position
            kp: Proportional gain
            tolerance: Tolerance for position error
            dt: Time step for position control
            timeout_s: Timeout for position control
            max_velocity: Maximum velocity
            min_velocity: Minimum velocity
            settle_steps: Number of steps to settle at the target position
            blocking: Whether to block until the position is reached
        """
        if not self._is_connected:
            logger.info(f"{self} is not connected.")
            return False

        motor_id = self._get_motor_id(motor)
        return self.protocol_handler.set_position(  # type: ignore[call-arg]
            motor_id, 
            target_position,
            get_position,
            kp=kp,
            tolerance=tolerance,
            dt=dt,
            timeout_s=timeout_s,
            max_velocity=max_velocity,
            min_velocity=min_velocity,
            settle_steps=settle_steps,
            blocking=blocking
        )

    def get_set_position_status(self, motor: NameOrID) -> dict[str, object]:
        """
        Get status for the last/current move_to_position command for a motor.

        Returns:
            dict with keys: active, generation, success (if finished), finished_t (if finished)
        """
        motor_id = self._get_motor_id(motor)
        ph = self.protocol_handler
        if ph is None or not hasattr(ph, "move_lock"):
            return {"active": False, "generation": None}
        with ph.move_lock:  # type: ignore[attr-defined]
            cmd = ph.move_cmds.get(motor_id)  # type: ignore[attr-defined]
            if cmd and not cmd.done:
                return {"active": True, "generation": cmd.generation}

            last = ph.move_last_result.get(motor_id)  # type: ignore[attr-defined]
            if last is None:
                return {"active": False, "generation": None}
            return {
                "active": False,
                "generation": last.generation,
                "success": last.success,
                "finished_t": last.finished_t,
            }

    def cancel_set_position(self, motor: NameOrID) -> None:
        """Cancel any active move_to_position for this motor (coast stop)."""
        motor_id = self._get_motor_id(motor)
        ph = self.protocol_handler
        if ph is None or not hasattr(ph, "move_cv"):
            return
        with ph.move_cv:  # type: ignore[attr-defined]
            cmd = ph.move_cmds.pop(motor_id, None)  # type: ignore[attr-defined]
            if cmd is not None:
                cmd.done = True
                cmd.success = False
                cmd.done_event.set()
                ph.move_last_result[motor_id] = SetPositionResult(  # type: ignore[attr-defined]
                    generation=cmd.generation,
                    success=False,
                    finished_t=time.monotonic(),
                )
            ph.move_cv.notify_all()  # type: ignore[attr-defined]

        self.set_velocity(motor, 0.0, normalize=True)

    ##############################################################################################################################
    # Velocity Functions
    ##############################################################################################################################

    def get_velocity(self, motor: NameOrID) -> float:
        """Get current motor velocity."""
        if not self._is_connected:
            logger.info(f"{self} is not connected.")
            return None

        motor_id = self._get_motor_id(motor)
        return self.protocol_handler.get_velocity(motor_id)

    def get_velocities(self) -> dict[NameOrID, float]:
        """Get current motor velocities."""
        if not self._is_connected:
            logger.info(f"{self} is not connected.")
            return { }

        return {motor: self.get_velocity(motor) for motor in self.motors.keys()}

    def set_velocity(self, motor: NameOrID, velocity: float, normalize: bool = True,) -> None:
        """
        Set motor velocity with ramp-up.

        Args:
            motor: Motor name or ID
            velocity: Target velocity (-1 to 1 if normalized, otherwise in RPM)
            normalize: Whether to normalize the velocity
        """
        if not self._is_connected:
            logger.info(f"{self} is not connected.")
            return

        motor_id = self._get_motor_id(motor)
        if normalize:
            velocity = max(-1.0, min(1.0, velocity)) # Clamp to [-1, 1]

        self.protocol_handler.set_velocity(motor_id, velocity)
        logger.debug(f"Set motor {motor} velocity to {velocity}")

    def set_velocities(self, motors: dict[NameOrID, float], normalize: bool = True) -> None:
        if not self._is_connected:
            return

        """
        Set motor velocities.

        Args:
            motors: Dictionary of motor names or IDs and target velocities
            normalize: Whether to normalize the velocity
        """
        for motor, velocity in motors.items():
            self.set_velocity(motor, velocity, normalize)

    ##############################################################################################################################
    # Update velocity
    ##############################################################################################################################

    def update_velocity(self, motor: NameOrID | None = None, max_step: float = 1.0) -> None:
        """Update motor velocity."""
        if not self._is_connected:
            logger.info(f"{self} is not connected.")
            return

        if motor is None:
            for motor_id in self._id_to_name_dict.keys():
                self.protocol_handler.update_velocity(motor_id, max_step)
        else:
            motor_id = self._get_motor_id(motor)
            self.protocol_handler.update_velocity(motor_id, max_step)

    ##############################################################################################################################
    # PWM Functions
    ##############################################################################################################################

    def get_pwm(self, motor: NameOrID) -> float:
        """Get current PWM duty cycle."""
        if not self._is_connected:
            logger.info(f"{self} is not connected.")
            return

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
            logger.info(f"{self} is not connected.")
            return

        motor_id = self._get_motor_id(motor)

        # Clamp to [0, 1]
        duty_cycle = max(0.0, min(1.0, duty_cycle))

        self.protocol_handler.set_pwm(motor_id, duty_cycle)
        logger.debug(f"Set motor {motor} PWM to {duty_cycle}")

    ##############################################################################################################################
    # Enable/Disable Functions
    ##############################################################################################################################

    def enable_motor(self, motor: NameOrID | None = None) -> None:
        """Enable motor(s)."""
        if not self._is_connected:
            logger.info(f"{self} is not connected.")
            return

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
            logger.info(f"{self} is not connected.")
            return

        if motor is None:
            # Disable all motors
            for motor_id in self._id_to_name_dict.keys():
                self.protocol_handler.disable_motor(motor_id)
        else:
            motor_id = self._get_motor_id(motor)
            self.protocol_handler.disable_motor(motor_id)
