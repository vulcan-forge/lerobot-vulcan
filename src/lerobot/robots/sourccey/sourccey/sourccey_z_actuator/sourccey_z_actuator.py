from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Optional, Protocol

from lerobot.robots.sourccey.sourccey.sourccey_z_actuator.sourccey_z_calibrator import SourcceyZCalibrator
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS


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
            self._adc = MCP3008(channel=self.adc_channel)

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

        # Position target (public API is position-only; motor command is internal).
        self._target_pos_m100_100: float = 0.0
        self.invert = sensor.invert

        # Tunables (safe defaults; tune on hardware).
        self.kp: float = 0.02
        self.max_cmd: float = 1.0
        self.deadband: float = 2.5

        # Debugging
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
        self._ctl_hz: float = 5.0
        self._ctl_instant: bool = True

    @property
    def is_connected(self) -> bool:
        return self.sensor.is_connected

    def connect(self) -> None:
        self.sensor.connect()

    def disconnect(self) -> None:
        # Ensure no background thread is still calling update() while we disconnect the ADC.
        self.stop_position_controller()
        self.sensor.disconnect()

    def update(self, dt_s: float, *, instant: bool = True) -> None:
        if self.driver is None:
            raise RuntimeError("No driver provided. Pass `driver=...` (e.g. Sourccey.dc_motors_controller).")

        pos = float(self.read_position())
        err = float(self._target_pos_m100_100) - pos

        # Bang-bang control: full speed toward target
        # This is fine because the z actuator motor is slow and has a lot of torque.
        if err > float(self.deadband):
            cmd = 1.0
        elif err < -float(self.deadband):
            cmd = -1.0
        else:
            self.stop()
            return

        if self.invert:
            cmd = -cmd

        self.driver.set_velocity(self.motor, cmd, normalize=True, instant=instant)

         # --- debug: print once per second ---
        now = time.monotonic()
        if now - self._last_cmd_print_t >= 1.0:
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
        while not self._ctl_stop_event.is_set():
            # If we're not ready to drive, just idle.
            if self.driver is None:
                time.sleep(0.05)
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
            except Exception:
                # Don't let the thread die on transient hardware/read errors.
                try:
                    self.stop()
                except Exception:
                    pass
                time.sleep(0.1)

            period = 1.0 / max(1.0, hz)
            time.sleep(period)

        # best-effort stop on exit
        try:
            self.stop()
        except Exception:
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
