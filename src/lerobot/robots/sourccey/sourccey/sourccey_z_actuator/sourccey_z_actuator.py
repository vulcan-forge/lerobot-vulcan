from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Optional, Protocol

from lerobot.robots.sourccey.sourccey.sourccey_z_actuator.sourccey_z_calibrator import SourcceyZCalibrator
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from lerobot.utils.robot_utils import precise_sleep


try:
    from gpiozero import MCP3008  # type: ignore
except Exception:  # pragma: no cover
    MCP3008 = None  # type: ignore


@dataclass(frozen=True)
class ZActuatorReading:
    raw: int        # 0..1023 (native MCP3008)
    voltage: float  # volts at the MCP3008 pin


@dataclass(frozen=True)
class ZActuatorCalibration:
    raw_min: int
    raw_max: int
    invert: bool

class ZMotorDriver(Protocol):
    """Small protocol so we can inject Sourccey’s DC controller without importing it here."""
    def set_velocity(self, motor: str | int, velocity: float, normalize: bool = True, instant: bool = True) -> None: ...
    def set_pwm(self, motor: str | int, duty_cycle: float) -> None: ...


class ZSensor:
    """
    Potentiometer sensor reader via MCP3008.

    - Raw MCP3008 is 10-bit (0..1023).
    - We keep everything in native units for calibration and conversion.
    - Calibration maps raw -> position in [-100, 100].
    """

    def __init__(
        self,
        *,
        adc_channel: int = 1,
        vref: float = 3.30,
        average_samples: int = 50,
        invert: bool = True,  # invert published position convention (e.g. make "up" positive)
    ) -> None:
        self.adc_channel = int(adc_channel)
        self.vref = float(vref)
        self.average_samples = int(average_samples)

        # Calibration bounds in native MCP3008 units.
        self.raw_min = 0
        self.raw_max = 1023
        self.calibration_min = self.raw_min
        self.calibration_max = self.raw_max
        self.invert = bool(invert)

        self._adc: Optional["MCP3008"] = None

    @property
    def is_connected(self) -> bool:
        return self._adc is not None

    def connect(self) -> None:
        if MCP3008 is None:
            raise RuntimeError("gpiozero is not installed; MCP3008 is unavailable on this machine.")
        if self._adc is None:
            candidate = None
            try:
                candidate = MCP3008(channel=self.adc_channel)
                _ = candidate.raw_value  # probe once to confirm the ADC responds

                # Check if signal is floating (no sensor connected)
                if self._detect_floating_signal(candidate):
                    if candidate is not None:
                        try:
                            candidate.close()  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    print(
                        f"MCP3008 on channel {self.adc_channel} appears to have a floating signal "
                        f"(no sensor connected). This robot may not have a Z sensor installed."
                    )
                    return
            except RuntimeError:
                raise  # Re-raise our floating signal error
            except Exception as exc:
                if candidate is not None:
                    try:
                        candidate.close()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                print(f"Failed to initialize MCP3008 on channel {self.adc_channel}. {exc}")
                return

            self._adc = candidate

    def disconnect(self) -> None:
        if self._adc is not None:
            close = getattr(self._adc, "close", None)
            if callable(close):
                close()
            self._adc = None

    def set_calibration(self, *, raw_min: int, raw_max: int, invert: Optional[bool] = None) -> None:
        self.calibration_min = int(raw_min)
        self.calibration_max = int(raw_max)
        if invert is not None:
            self.invert = bool(invert)

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else hi if x > hi else x


    ############################################################
    # Read Functions
    ############################################################
    def read_raw(self) -> ZActuatorReading:
        """Read averaged raw (0..1023) and voltage."""
        if self._adc is None:
            self.connect()
        assert self._adc is not None

        total = 0.0
        for _ in range(self.average_samples):
            total += float(self._adc.raw_value)
        raw = int(round(total / self.average_samples))
        voltage = (raw / 1023.0) * self.vref
        return ZActuatorReading(raw=raw, voltage=voltage)

    def read_position_m100_100(self) -> float:
        return self.raw_to_pos_m100_100(self.read_raw().raw)

    def _detect_floating_signal(self, adc: "MCP3008", num_samples: int = 100) -> bool:
        """
        Detect if the ADC signal is floating (unconnected).

        A floating signal typically:
        - Has very high variance (unstable readings)
        - Or is stuck at extremes (0, 1023) or mid-range (512)
        - Or oscillates between a few values

        Returns True if signal appears to be floating.
        """
        samples = []
        for _ in range(num_samples):
            try:
                samples.append(float(adc.raw_value))
            except Exception:
                return True  # If we can't read, assume floating

        if len(samples) < num_samples:
            return True

        # Calculate statistics
        mean = sum(samples) / len(samples)
        variance = sum((x - mean) ** 2 for x in samples) / len(samples)
        std_dev = variance ** 0.5
        min_val = min(samples)
        max_val = max(samples)
        value_range = max_val - min_val

        # Check 1: Very high variance (unstable floating signal)
        # Lower threshold - floating signals often have std_dev > 50-100
        if std_dev > 50.0:
            return True

        # Check 2: Very low variance but stuck at common floating values
        # Check if most samples (80%+) are near common floating values
        stuck_threshold = 10  # Allow more noise
        near_zero = sum(1 for s in samples if abs(s - 0) < stuck_threshold)
        near_mid = sum(1 for s in samples if abs(s - 512) < stuck_threshold)
        near_max = sum(1 for s in samples if abs(s - 1023) < stuck_threshold)

        if near_zero / len(samples) > 0.8:
            return True  # 80%+ stuck near 0
        if near_mid / len(samples) > 0.8:
            return True  # 80%+ stuck near mid-range
        if near_max / len(samples) > 0.8:
            return True  # 80%+ stuck near max

        # Check 3: Very narrow range (stuck at a single value with small noise)
        # A real sensor should have a reasonable range when moved
        if value_range < 20 and std_dev < 5.0:
            return True  # Very stable, narrow range - likely floating

        # Check 4: Moderate variance but oscillating between few values
        # Count unique values - floating signals often have few distinct values
        unique_values = len(set(round(s) for s in samples))
        if unique_values < 10 and std_dev > 10.0:
            return True  # Few unique values with some variance - likely floating

        return False

    ############################################################
    # Conversion Functions
    ############################################################

    def raw_to_pos_m100_100(self, raw: int) -> float:
        """Convert native raw (0..1023) into position [-100, 100] using current calibration."""
        rmin, rmax = float(self.calibration_min), float(self.calibration_max)
        if rmax == rmin:
            return 0.0

        t = (float(raw) - rmin) / (rmax - rmin)  # 0..1 ideally
        t = self._clamp(t, 0.0, 1.0)
        pos = -100.0 + 200.0 * t  # -100..100

        # Sensor invert: ensures your published position follows your chosen convention (e.g. "up is +")
        return -pos if self.invert else pos

    def position_m100_100_to_raw(self, position: float) -> int:
        """Map [-100,100] -> [calibration_min, calibration_max] (inverse of raw_to_pos_m100_100)."""
        p = self._clamp(float(position), -100.0, 100.0)
        if self.invert:
            p = -p
        t = (p + 100.0) / 200.0
        rmin, rmax = float(self.calibration_min), float(self.calibration_max)
        return int(round(rmin + t * (rmax - rmin)))


class SourcceyZActuator:
    """
    Higher-level Z module:
    - reads position through a ZSensor
    - writes motor commands through an injected driver (e.g. Sourccey’s PWM DC controller)

    This keeps the ADC concerns (sensor) separate from actuation concerns (motor driver).
    """

    def __init__(
        self,
        *,
        sensor: ZSensor,
        driver: ZMotorDriver | None = None,
        motor: str | int = "linear_actuator"
    ) -> None:

        self.name = "sourccey_z_actuator"

        self.sensor = sensor
        self.driver = driver
        self.motor = motor
        self.use_z_actuator = False

        # Position target (public API is position-only; motor command is internal).
        self._target_pos_m100_100: float = 0.0
        self.invert = sensor.invert

        # Tunables (safe defaults; tune on hardware).
        self.kp: float = 0.05
        self.kd: float = 0.02 # Derivative gain: damps overshoot by "braking" when error is changing quickly. # (cmd = kp*err + kd*d(err)/dt)

        self.max_cmd: float = 1.0
        self.deadband: float = 1.0

        # Endpoint assist: when commanding near +/-100, we may need full power to overcome
        # stiction / deadzone near the ends.
        self.endpoint_target_threshold: float = 90.0
        # Hysteresis: enter assist when far, exit when close (prevents rapid toggling/oscillation).
        self.endpoint_enter_margin: float = 4.0
        self.endpoint_exit_margin: float = 1.0
        self._endpoint_assist_active: bool = False

        # Controller state for D term.
        self._prev_err: float = 0.0
        self._prev_err_valid: bool = False

        # Debugging
        self._debug_mode = False
        self._last_cmd_print_t = 0.0

        # Calibration
        self.calibration_dir = (
            HF_LEROBOT_CALIBRATION / ROBOTS / self.name
        )
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_fpath = self.calibration_dir / f"{self.name}.json"
        self.calibration: dict[str, ZActuatorCalibration] = {}
        if self.calibration_fpath.is_file():
            self._load_calibration()

        self.calibrator = SourcceyZCalibrator(self)

        # --- background "servo-like" position controller thread ---
        self._ctl_lock = threading.Lock()
        self._ctl_stop_event = threading.Event()
        self._ctl_thread: Optional[threading.Thread] = None
        self._ctl_hz: float = 30.0
        self._ctl_instant: bool = True

    @property
    def is_connected(self) -> bool:
        return self.sensor.is_connected

    def connect(self) -> None:
        self.sensor.connect()
        self.use_z_actuator = True if self.sensor.is_connected else False

    def disconnect(self) -> None:
        # Ensure no background thread is still calling update() while we disconnect the ADC.
        self.stop_position_controller()
        self.sensor.disconnect()

    def update(self, dt_s: float, *, instant: bool = True) -> None:
        if self.driver is None:
            raise RuntimeError("No driver provided. Pass `driver=...` (e.g. Sourccey.dc_motors_controller).")

        pos = float(self.read_position())
        target = float(self._target_pos_m100_100)
        err = target - pos

        # Keep the existing deadband behavior for normal targets, but when commanding extreme
        # endpoints, tighten the deadband so we actually reach +/-100 instead of stopping short.
        deadband = float(self.deadband)
        if abs(target) >= 99.0:
            deadband = 0.1

        # --- PD control (P + D damping) ---
        if abs(err) <= deadband:
            # Reset derivative state so we don't get a "kick" when restarting from a stop.
            self._prev_err_valid = False
            self.stop()
            return

        # Endpoint assist with hysteresis:
        # - uses enter/exit margins to avoid rapid toggling near the endpoint
        # - only pushes *toward* the endpoint, never away (prevents banging back and forth)
        near_endpoint_target = (abs(target) >= 99.0) or (abs(target) >= float(self.endpoint_target_threshold))
        if near_endpoint_target:
            if (not self._endpoint_assist_active) and (abs(err) >= float(self.endpoint_enter_margin)):
                self._endpoint_assist_active = True
            elif self._endpoint_assist_active and (abs(err) <= float(self.endpoint_exit_margin)):
                self._endpoint_assist_active = False

            if self._endpoint_assist_active:
                if (target > 0.0 and err > 0.0):
                    cmd = self.max_cmd
                elif (target < 0.0 and err < 0.0):
                    cmd = -self.max_cmd
                else:
                    cmd = 0.0  # past the endpoint; don't drive back with full power

                # Reset controller state to avoid a D "kick" when we hand back to PD.
                self._prev_err_valid = False
                if self.invert:
                    cmd = -cmd
                self.driver.set_velocity(self.motor, cmd, normalize=True, instant=instant)
                return

        dt = float(dt_s)
        if dt <= 1e-6:
            dt = 1e-3

        # Derivative of error (finite difference). This adds damping near the target.
        derr = 0.0
        if self._prev_err_valid:
            derr = (err - self._prev_err) / dt
        self._prev_err = err
        self._prev_err_valid = True

        cmd = (self.kp * err) + (self.kd * derr)
        cmd = max(-self.max_cmd, min(self.max_cmd, cmd))

        if self.invert:
            cmd = -cmd

        self.driver.set_velocity(self.motor, cmd, normalize=True, instant=instant)

        # --- debug: print once per second ---
        now = time.monotonic()
        if self._debug_mode and now - self._last_cmd_print_t >= 1.0:
            self._last_cmd_print_t = now
            print(
                {
                    "z_cmd": round(float(cmd), 3),
                    "z_err": round(float(err), 2),
                    "z_pos": round(float(pos), 2),
                    "z_target": round(float(self._target_pos_m100_100), 2),
                }
            )


    ############################################################
    # Control Functions
    ############################################################
    def _control_loop(self) -> None:
        last_t = time.monotonic()
        last_print = time.monotonic()
        it = 0
        while not self._ctl_stop_event.is_set():
            # If we're not ready to drive, just idle.
            if self.driver is None:
                precise_sleep(0.05)
                last_t = time.monotonic()
                continue

            now = time.monotonic()
            dt = now - last_t
            last_t = now

            with self._ctl_lock:
                hz = float(self._ctl_hz)
                instant = bool(self._ctl_instant)

            try:
                self.update(dt, instant=instant)
            except Exception as e:
                # Don't let the thread die on transient hardware/read errors.
                try:
                    self.stop()
                except Exception as e2:
                    pass
                precise_sleep(0.1)



            it += 1
            if self._debug_mode and now - last_print >= 1.0:
                last_print = now
                try:
                    raw = self.sensor.read_raw().raw
                except Exception as e:
                    raw = f"ERR:{type(e).__name__}"
                print({
                    "it": it,
                    "stop_event": self._ctl_stop_event.is_set(),
                    "hz": hz,
                    "instant": instant,
                    "target": round(float(self._target_pos_m100_100), 2),
                    "pos": round(float(self.read_position()), 2),
                    "raw": raw,
                })

            period = 1.0 / max(1.0, hz)
            precise_sleep(period)

        # best-effort stop on exit
        try:
            self.stop()
        except Exception as e3:
            pass

    def _ensure_controller_running(self, *, hz: float = 30.0, instant: bool = True) -> None:
        with self._ctl_lock:
            self._ctl_hz = float(hz)
            self._ctl_instant = bool(instant)

        if self._ctl_thread is not None and self._ctl_thread.is_alive():
            return

        self._ctl_stop_event.clear()
        self._ctl_thread = threading.Thread(
            target=self._control_loop,
            name="SourcceyZActuatorControl",
            daemon=True,
        )
        self._ctl_thread.start()

    def stop_position_controller(self, *, join_timeout_s: float = 1.0) -> None:
        print("Stopping Z actuator position controller")
        """Stop the background position controller (if running) and stop motor output."""
        self._ctl_stop_event.set()
        t = self._ctl_thread
        if t is not None and t.is_alive():
            t.join(timeout=float(join_timeout_s))
        self._ctl_thread = None
        self.stop()

    def stop(self) -> None:
        """Stop motor output and reset integrator."""
        if self.driver is not None:
            self.driver.set_velocity(self.motor, 0.0, normalize=True, instant=True)

    ############################################################
    # Calibration Functions
    ############################################################
    def _load_calibration(self, fpath: Path | None = None) -> bool:
        """
        Load Z sensor calibration from `<calibration_dir>/<filename>`.

        Expected JSON:
          "z_actuator": {
            "raw_min": 123,
            "raw_max": 987,
            "invert": true   # optional
          }

        Returns True if loaded, False if file doesn't exist.
        """
        fpath = self.calibration_fpath if fpath is None else fpath
        if not fpath.is_file():
            return False

        with open(fpath, "r") as f:
            data = json.load(f)

        raw_min = int(data["z_actuator"]["raw_min"])
        raw_max = int(data["z_actuator"]["raw_max"])
        invert = bool(data["z_actuator"]["invert"])

        self.sensor.set_calibration(raw_min=raw_min, raw_max=raw_max, invert=invert)

        # Keep actuator inversion consistent with sensor inversion (since update() uses self.invert).
        self.invert = bool(self.sensor.invert)
        return True

    def _save_calibration(self, fpath: Path | None = None) -> None:
        """
        Save Z sensor calibration to `<calibration_dir>/<filename>`.
        """
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, "w") as f:
            json.dump({
                "z_actuator": {
                    "raw_min": self.sensor.calibration_min,
                    "raw_max": self.sensor.calibration_max,
                    "invert": self.sensor.invert
                }
            }, f, indent=4)

    ############################################################
    # Read / Write Functions
    ############################################################

    # --- Reads (delegated to sensor) ---
    def read_position(self) -> float:
        return self.sensor.read_position_m100_100()

     # --- Write Functions ---
    def write_position(self, target_pos_m100_100: float) -> None:
        self._target_pos_m100_100 = max(-100.0, min(100.0, float(target_pos_m100_100)))

    ############################################################
    # Move Position Functions
    ############################################################
    def move_to_position(
        self,
        target_pos_m100_100: float,
        *,
        hz: float = 30.0,
        instant: bool = True,
    ) -> None:
        """
        Non-blocking "servo-like" position command.

        - Sets the target position immediately.
        - Starts (or reconfigures) a background loop that continuously drives toward the latest target.
        - Call again at any time with a new target; the background loop will move to the new value.
        """
        if self.driver is None:
            raise RuntimeError("No driver provided. Pass `driver=...` to move the actuator.")

        self.write_position(float(target_pos_m100_100))
        self._ensure_controller_running(hz=float(hz), instant=bool(instant))

    def move_to_position_blocking(
        self,
        target_pos_m100_100: float,
        *,
        timeout_s: float = 10.0,
        hz: float = 30.0,
        instant: bool = True,
    ) -> float:
        """
        Blocking move: set a target position and drive until within deadband (or timeout).
        Returns the final measured position.
        """
        if self.driver is None:
            raise RuntimeError("No driver provided. Pass `driver=...` to move the actuator.")

        # Avoid concurrent control from background thread.
        self.stop_position_controller()

        self.write_position(float(target_pos_m100_100))

        period = 1.0 / max(1.0, float(hz))
        t_end = time.monotonic() + float(timeout_s)
        last_t = time.monotonic()

        while True:
            now = time.monotonic()
            if now >= t_end:
                self.stop()
                raise TimeoutError(f"Timed out moving Z to {target_pos_m100_100}")

            dt = now - last_t
            last_t = now

            self.update(dt, instant=instant)
            pos = float(self.read_position())

            if abs(pos - float(target_pos_m100_100)) <= float(self.deadband):
                self.stop()
                return pos

            time.sleep(period)
