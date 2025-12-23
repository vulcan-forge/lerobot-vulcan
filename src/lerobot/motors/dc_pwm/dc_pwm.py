import logging
import threading
from typing import Callable, Dict, Optional, List

from lerobot.motors.dc_motors_controller import BaseDCMotorsController, DCMotor, ProtocolHandler
from lerobot.motors.dc_motors_controller import SetPositionCmd, SetPositionResult
from lerobot.motors.dc_motors_controller import NameOrID
import time

logger = logging.getLogger(__name__)


# Pi 5 Hardware PWM Configuration
PI5_HARDWARE_PWM_PINS = {
    "pwm0": [12],  # PWM0 channels
    "pwm1": [13],  # PWM1 channels
    "pwm2": [18],  # PWM2 channels
    "pwm3": [19],  # PWM3 channels
}

# Pi 5 All Available GPIO Pins (40-pin header)
PI5_ALL_GPIO_PINS = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41
]

# Pi 5 Optimal Settings for DRV8874PWPR
PI5_OPTIMAL_FREQUENCY = 25000  # 25kHz - more compatible with gpiozero
PI5_MAX_FREQUENCY = 25000      # 25kHz - Pi 5 can handle higher frequencies
PI5_RESOLUTION = 12            # 12-bit resolution

class PWMDCMotorsController(BaseDCMotorsController):
    """PWM-based DC motor controller optimized for DRV8874PWPR H-bridge drivers."""

    def __init__(self, config: dict | None = None, motors: dict[str, DCMotor] | None = None, protocol: str = "pwm"):
        super().__init__(config, motors, protocol)

    def _create_protocol_handler(self) -> ProtocolHandler:
        return PWMProtocolHandler(self.config, self.motors)


class PWMProtocolHandler(ProtocolHandler):
    """
    PWM protocol handler optimized for DRV8874PWPR H-bridge motor drivers.

    DRV8874PWPR Features:
    - IN1: PWM speed control (hardware PWM recommended)
    - IN2: Direction control (regular GPIO)
    - Built-in current limiting and thermal protection
    - 25kHz PWM frequency optimal

    Configuration:
    - in1_pins: IN1 pins (PWM speed control)
    - in2_pins: IN2 pins (direction control)
    - pwm_frequency: 25000Hz (optimal for DRV8874PWPR)
    """

    ##############################################################################################################################
    # Configuration
    ##############################################################################################################################

    def __init__(self, config: Dict, motors: Dict[str, DCMotor]):
        self.config = config
        self.in1_pins = config.get("in1_pins", [])
        self.in2_pins = config.get("in2_pins", [])
        self.enable_pins = config.get("enable_pins", [])
        self.brake_pins = config.get("brake_pins", [])
        self.pwm_frequency = config.get("pwm_frequency", PI5_OPTIMAL_FREQUENCY)
        self.invert_direction = config.get("invert_direction", False)
        self.invert_enable = config.get("invert_enable", False)
        self.invert_brake = config.get("invert_brake", False)

        # Motor configuration and state tracking
        self.motors: Dict[str, DCMotor] = motors
        self._id_to_name: dict[int, str] = {m.id: name for name, m in motors.items()}
        self.motor_states: Dict[int, Dict] = {}
        self.in1_channels = {}
        self.in2_channels = {}
        self.enable_channels = {}
        self.brake_channels = {}

        # Background "set position" workers (one thread per motor_id).
        # Re-calling set_position() for the same motor updates the command that the worker reads next cycle.
        self.move_lock = threading.RLock()
        self.move_cv = threading.Condition(self.move_lock)
        self.move_stop_all = False
        self.move_threads: dict[int, threading.Thread] = {}
        self.move_cmds: dict[int, SetPositionCmd] = {}
        self.move_last_result: dict[int, SetPositionResult] = {}

        # Validate Pi 5 pins
        self._validate_pi5_pins()

        # Import gpiozero
        self._import_gpiozero()

    def _validate_pi5_pins(self):
        """Validate that pins are valid GPIO pins on Pi 5."""
        all_hardware_pwm = []
        for pwm_pins in PI5_HARDWARE_PWM_PINS.values():
            all_hardware_pwm.extend(pwm_pins)

        # Validate IN1 pins (should be hardware PWM for best performance)
        invalid_in1_pins = [pin for pin in self.in1_pins if pin not in all_hardware_pwm]
        if invalid_in1_pins:
            # logger.warning(
            #     f"IN1 pins {invalid_in1_pins} are not hardware PWM pins on Pi 5. "
            #     f"Hardware PWM pins: {all_hardware_pwm}"
            # )
            pass

        # Validate IN2 pins (can be any GPIO)
        invalid_in2_pins = [pin for pin in self.in2_pins if pin not in PI5_ALL_GPIO_PINS]
        if invalid_in2_pins:
            logger.warning(
                f"IN2 pins {invalid_in2_pins} are not valid GPIO pins on Pi 5. "
                f"Valid GPIO pins: {PI5_ALL_GPIO_PINS}"
            )

        # Check for pin conflicts
        all_used_pins = set(self.in1_pins + self.in2_pins + self.enable_pins + self.brake_pins)
        if len(all_used_pins) != len(self.in1_pins + self.in2_pins + self.enable_pins + self.brake_pins):
            logger.warning("Duplicate pins detected in configuration")

        # Validate motor count
        motor_count = len(self.in1_pins)
        logger.info(f"Configuring {motor_count} DRV8874PWPR motors with gpiozero")

    def _import_gpiozero(self):
        """Import gpiozero."""
        try:
            import gpiozero  # pyright: ignore[reportMissingImports]
            self.gpiozero = gpiozero
            logger.info("Using gpiozero for DRV8874PWPR motor control")

        except ImportError:
            raise ImportError(
                "gpiozero not available. Install with: uv pip install gpiozero>=2.0"
            )

    def _setup_pwmled(self, pin: int, label: str) -> 'gpiozero.PWMLED': # type: ignore
        """Safely set up a PWMLED on the given pin, with fallback to default frequency."""
        try:
            return self.gpiozero.PWMLED(pin, frequency=self.pwm_frequency)
        except Exception as e:
            logger.warning(f"{label}: Failed with frequency {self.pwm_frequency}, retrying with default. ({e})")
            try:
                return self.gpiozero.PWMLED(pin)
            except Exception as e2:
                logger.error(f"{label}: Failed to setup PWMLED on pin {pin}: {e2}")
                raise

    def _safe_close(self, channel: 'gpiozero.PWMLED', label: str) -> None: # type: ignore
        """Safely close a PWMLED channel."""
        try:
            channel.close()
        except Exception as e:
            logger.warning(f"Error closing {label}: {e}")

    ##############################################################################################################################
    # Connection
    ##############################################################################################################################

    def connect(self) -> None:
        """Initialize gpiozero for DRV8874PWPR motor drivers with symmetric PWM on IN1 and IN2."""
        try:
            for motor_id, (in1_pin, in2_pin) in enumerate(zip(self.in1_pins, self.in2_pins), start=1):
                self.motor_states[motor_id] = {
                    "position": 0.0,
                    "velocity": 0.0,
                    "pwm": 0.0,
                    "enabled": False,
                    "brake_active": False,
                    "direction": 1
                }

                in1 = self._setup_pwmled(in1_pin, f"Motor {motor_id} IN1")
                in1.off()
                self.in1_channels[motor_id] = in1
                logger.debug(f"Motor {motor_id} IN1 setup on pin {in1_pin}")

                in2 = self._setup_pwmled(in2_pin, f"Motor {motor_id} IN2")
                in2.off()
                self.in2_channels[motor_id] = in2
                logger.debug(f"Motor {motor_id} IN2 setup on pin {in2_pin}")

            total_pins = len(self.in1_pins) + len(self.in2_pins)
            logger.info(f"DRV8874PWPR setup complete: {len(self.in1_pins)} motors, {total_pins} GPIOs used")
            logger.info(f"PWM frequency: {self.pwm_frequency} Hz")
        except Exception as e:
            logger.error(f"Motor driver setup failed: {e}")
            raise RuntimeError("gpiozero hardware not available")

    def disconnect(self) -> None:
        """Stop background workers and clean up gpiozero PWMLED channels for IN1 and IN2."""
        # Stop any background position workers first.
        with self.move_cv:
            self.move_stop_all = True
            for cmd in self.move_cmds.values():
                cmd.done = True
                cmd.success = False
                cmd.done_event.set()
            self.move_cv.notify_all()

        for t in list(self.move_threads.values()):
            t.join(timeout=1.0)

        with self.move_cv:
            self.move_threads.clear()
            self.move_cmds.clear()
            self.move_stop_all = False

        for motor_id, channel in self.in1_channels.items():
            self._safe_close(channel, f"IN1 (motor {motor_id})")

        for motor_id, channel in self.in2_channels.items():
            self._safe_close(channel, f"IN2 (motor {motor_id})")

        logger.info("DRV8874PWPR motor driver disconnected")

    ##############################################################################################################################
    # Position Functions
    ##############################################################################################################################
    def get_position(self, motor_id: int) -> Optional[float]:
        """Get current motor position if encoder available."""
        return self.motor_states.get(motor_id, {}).get("position", 0.0)

    
    def set_position(
        self,
        motor_id: int,
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
        Closed-loop move for DC motors using an external position sensor.

        - Non-blocking by default (starts/updates a background worker thread for the motor).
        - Calling again for the same motor updates the existing worker command (no new thread).

        Returns:
            If blocking=False: True if scheduled, False if not connected.
            If blocking=True: True on success, False on timeout/cancel.
        """
        if not self._is_connected:
            logger.info(f"{self} is not connected.")
            return False

        with self.move_cv:
            prev = self.move_cmds.get(motor_id)
            generation = 1 if prev is None else prev.generation + 1

            # If there's an active command, mark it as cancelled so any blockers are released.
            if prev is not None and not prev.done:
                prev.done = True
                prev.success = False
                prev.done_event.set()
                self.move_last_result[motor_id] = SetPositionResult(
                    generation=prev.generation,
                    success=False,
                    finished_t=time.monotonic(),
                )

            cmd = SetPositionCmd(
                target_position=float(target_position),
                get_position=get_position,
                kp=float(kp),
                tolerance=float(tolerance),
                dt=float(dt),
                timeout_s=float(timeout_s),
                max_velocity=float(max_velocity),
                min_velocity=float(min_velocity),
                settle_steps=int(settle_steps),
                generation=generation,
                start_t=time.monotonic(),
            )
            self.move_cmds[motor_id] = cmd
            self._ensure_move_worker(motor_id)
            self.move_cv.notify_all()

        if not blocking:
            return True

        # Wait for completion of *this* generation
        cmd.done_event.wait(timeout=timeout_s + 0.5)
        with self.move_lock:
            last = self.move_last_result.get(motor_id)
            if last and last.generation == generation:
                return bool(last.success)
            return False

    def _ensure_move_worker(self, motor_id: int) -> None:
        with self.move_lock:
            t = self.move_threads.get(motor_id)
            if t and t.is_alive():
                return
            t = threading.Thread(target=self._move_worker, args=(motor_id,), daemon=True)
            self.move_threads[motor_id] = t
            t.start()

    def _move_worker(self, motor_id: int) -> None:
        def _stop() -> None:
            self.set_velocity(motor_id, 0.0, instant=True)

        while True:
            with self.move_cv:
                while not self.move_stop_all and motor_id not in self.move_cmds:
                    self.move_cv.wait()
                if self.move_stop_all:
                    _stop()
                    return
                cmd = self.move_cmds[motor_id]

            # Snapshot fields for this cycle
            gen = cmd.generation
            target = cmd.target_position
            get_pos = cmd.get_position
            kp = cmd.kp
            tol = cmd.tolerance
            dt = cmd.dt
            timeout_s = cmd.timeout_s
            vmax = cmd.max_velocity
            vmin = cmd.min_velocity
            settle_steps = cmd.settle_steps
            start_t = cmd.start_t

            if not self._is_connected:
                _stop()
                return

            if time.monotonic() - start_t > timeout_s:
                _stop()
                with self.move_lock:
                    cur = self.move_cmds.get(motor_id)
                    if cur and cur.generation == gen:
                        cur.done = True
                        cur.success = False
                        cur.done_event.set()
                        self.move_last_result[motor_id] = SetPositionResult(
                            generation=gen,
                            success=False,
                            finished_t=time.monotonic(),
                        )
                        self.move_cmds.pop(motor_id, None)
                time.sleep(dt)
                continue

            sensor_key: NameOrID = self._id_to_name.get(motor_id, motor_id)
            try:
                current = float(get_pos(sensor_key))
            except Exception:
                _stop()
                with self.move_lock:
                    cur = self.move_cmds.get(motor_id)
                    if cur and cur.generation == gen:
                        cur.done = True
                        cur.success = False
                        cur.done_event.set()
                        self.move_last_result[motor_id] = SetPositionResult(
                            generation=gen,
                            success=False,
                            finished_t=time.monotonic(),
                        )
                        self.move_cmds.pop(motor_id, None)
                time.sleep(dt)
                continue

            err = target - current

            if abs(err) <= tol:
                with self.move_lock:
                    cur = self.move_cmds.get(motor_id)
                    if not cur or cur.generation != gen:
                        time.sleep(dt)
                        continue
                    cur.in_tol_count += 1
                    if cur.in_tol_count >= settle_steps:
                        _stop()
                        cur.done = True
                        cur.success = True
                        cur.done_event.set()
                        self.move_last_result[motor_id] = SetPositionResult(
                            generation=gen,
                            success=True,
                            finished_t=time.monotonic(),
                        )
                        self.move_cmds.pop(motor_id, None)
                time.sleep(dt)
                continue
            else:
                with self.move_lock:
                    cur = self.move_cmds.get(motor_id)
                    if cur and cur.generation == gen:
                        cur.in_tol_count = 0

            v = kp * err
            v = max(-vmax, min(vmax, v))
            if 0.0 < abs(v) < vmin:
                v = vmin if v > 0 else -vmin

            self.set_velocity(motor_id, v, instant=True)
            time.sleep(dt)

    ##############################################################################################################################
    # Velocity Functions
    ##############################################################################################################################
    def get_velocity(self, motor_id: int) -> float:
        """Get current motor velocity."""
        return self.motor_states.get(motor_id, {}).get("velocity", 0.0)

    def set_velocity(self, motor_id: int, target_velocity: float, instant: bool = True) -> None:
        """
        Set the target velocity for the motor (-1.0 to 1.0).
        Actual velocity will be slewed toward this value in update_velocity().
        """
        target_velocity = max(-1.0, min(1.0, target_velocity))  # clamp
        self.motor_states[motor_id]["target_velocity"] = target_velocity

        if instant:
            self.update_velocity(motor_id, 1.0)

    def update_velocity(self, motor_id: int, max_step: float = 1.0) -> None:
        """
        Gradually update the motor velocity toward its target using a slew-rate limiter.
        Call this periodically (e.g., every 10–20 ms).
        """
        state = self.motor_states[motor_id]
        current = state.get("velocity", 0.0)
        target = state.get("target_velocity", 0.0)

        # Apply slew-rate limit
        if target > current:
            new_velocity = min(current + max_step, target)
        elif target < current:
            new_velocity = max(current - max_step, target)
        else:
            new_velocity = target

        # Save new velocity
        state["velocity"] = new_velocity

        # Convert to PWM duty cycle
        pwm_duty = self._velocity_to_pwm(new_velocity)
        state["pwm"] = pwm_duty
        state["brake_active"] = False

        in1 = self.in1_channels.get(motor_id)
        in2 = self.in2_channels.get(motor_id)

        if new_velocity > 0:
            if in1: in1.value = pwm_duty
            if in2: in2.off()
            state["direction"] = 1

        elif new_velocity < 0:
            if in1: in1.off()
            if in2: in2.value = pwm_duty
            state["direction"] = -1

        else:
            if in1: in1.off()
            if in2: in2.off()
            state["direction"] = 0

    ##############################################################################################################################
    # PWM Functsions
    ##############################################################################################################################

    def get_pwm(self, motor_id: int) -> float:
        """Get current PWM duty cycle."""
        return self.motor_states.get(motor_id, {}).get("pwm", 0.0)

    def set_pwm(self, motor_id: int, duty_cycle: float) -> None:
        """
        Set PWM duty cycle (0..0.98) respecting current direction.
        Uses symmetric PWM: IN1 for forward, IN2 for reverse.
        """

        # Cap your duty to 0.98 (to avoid DRV8871's fixed off-time weirdness at 100%)
        duty_cycle = max(0.0, min(0.98, duty_cycle))
        self.motor_states[motor_id]["pwm"] = duty_cycle

        direction = self.motor_states[motor_id].get("direction", 1)
        in1 = self.in1_channels.get(motor_id)
        in2 = self.in2_channels.get(motor_id)

        if not in1 or not in2:
            logger.warning(f"Motor {motor_id} missing IN1/IN2 channel(s)")
            return

        try:
            if direction > 0:   # forward
                in1.value = duty_cycle
                in2.off()
            elif direction < 0:   # reverse
                in1.off()
                in2.value = duty_cycle
            else:
                in1.off()
                in2.off()
            logger.debug(f"Motor {motor_id} PWM={duty_cycle:.3f} dir={'FWD' if direction>0 else 'REV' if direction<0 else 'STOP'}")
        except Exception as e:
            logger.warning(f"Error setting PWM for motor {motor_id}: {e}")

    ##############################################################################################################################
    # Enable/Disable Functions
    ##############################################################################################################################

    def enable_motor(self, motor_id: int) -> None:
        """Enable motor."""
        self.motor_states[motor_id]["enabled"] = True
        logger.debug(f"Motor {motor_id} enabled")

    def disable_motor(self, motor_id: int) -> None:
        """Disable motor by setting PWM to 0."""
        self.set_pwm(motor_id, 0.0)
        self.motor_states[motor_id]["enabled"] = False
        logger.debug(f"Motor {motor_id} disabled")

    ##############################################################################################################################
    # Helper methods for DRV8874PWPR-specific functionality
    ##############################################################################################################################

    def _get_direction(self, motor_id: int) -> bool:
        """Get motor direction."""
        if motor_id not in self.in2_channels:
            return False
        return self.in2_channels[motor_id].value == 1

    def _set_direction(self, motor_id: int, forward: bool) -> None:
        """
        Set motor direction for DRV8874PWPR.
        This method updates the direction state and applies appropriate PWM.
        """
        if motor_id not in self.in2_channels:
            return

        # Apply direction inversion if configured
        if self.invert_direction:
            forward = not forward

        # Update direction state
        self.motor_states[motor_id]["direction"] = 1 if forward else -1

        try:
            # Set IN2 for direction control
            self.in2_channels[motor_id].on() if not forward else self.in2_channels[motor_id].off()

            # Re-apply current PWM with new direction
            current_pwm = self.motor_states[motor_id].get("pwm", 0.0)
            if current_pwm > 0:
                self.set_pwm(motor_id, current_pwm)

            logger.debug(f"Motor {motor_id} direction set to {'forward' if forward else 'backward'}")
        except Exception as e:
            logger.warning(f"Error setting direction for motor {motor_id}: {e}")

    # DRV8874PWPR-specific convenience methods
    def activate_brake(self, motor_id: int) -> None:
        """
        Activate motor brake for DRV8874PWPR.
        Brake mode: IN1 = HIGH, IN2 = HIGH
        """
        in1 = self.in1_channels.get(motor_id)
        in2 = self.in2_channels.get(motor_id)

        if not in1 or not in2:
            logger.warning(f"Cannot activate brake: IN1 or IN2 not found for motor {motor_id}")
            return

        try:
            in1.on()
            in2.on()
            self.motor_states[motor_id]["brake_active"] = True
            logger.debug(f"Motor {motor_id} brake activated (IN1=1, IN2=1)")
        except Exception as e:
            logger.warning(f"Error activating brake for motor {motor_id}: {e}")

    def release_brake(self, motor_id: int) -> None:
        """
        Release motor brake for DRV8874PWPR.
        Coast mode: IN1 = LOW, IN2 = LOW
        """
        in1 = self.in1_channels.get(motor_id)
        in2 = self.in2_channels.get(motor_id)

        if not in1 or not in2:
            logger.warning(f"Cannot release brake: IN1 or IN2 not found for motor {motor_id}")
            return

        try:
            in1.off()
            in2.off()
            self.motor_states[motor_id]["brake_active"] = False
            logger.debug(f"Motor {motor_id} brake released (IN1=0, IN2=0)")
        except Exception as e:
            logger.warning(f"Error releasing brake for motor {motor_id}: {e}")

    ##############################################################################################################################
    # Helper methods for DRV8874PWPR-specific functionality
    ##############################################################################################################################

    def _velocity_to_pwm(self, velocity: float) -> float:
        """
        Convert normalized velocity (-1 to 1) into PWM duty cycle (0.0 to 1.0).
        Linear mapping: velocity maps directly to PWM duty cycle.
        """
        # This code works for the 30RPM 10kg.cm torque motor.
        v = abs(velocity)

        # Special case: stop = true 0 duty
        if v == 0:
            return 0.0

        # Linear mapping: velocity maps directly to PWM duty cycle
        pwm = v

        return pwm
