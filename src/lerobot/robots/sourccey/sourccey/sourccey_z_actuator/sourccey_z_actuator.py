from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, Protocol

from lerobot.robots.sourccey.sourccey.sourccey_z_actuator.sourccey_z_calibrator import SourcceyZCalibrator


try:
    from gpiozero import MCP3008  # type: ignore
except Exception:  # pragma: no cover
    MCP3008 = None  # type: ignore


@dataclass(frozen=True)
class ZActuatorReading:
    raw: int        # 0..1023 (native MCP3008)
    voltage: float  # volts at the MCP3008 pin


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


class ZActuator:
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
        motor: str | int = "linear_actuator",
        invert: bool = False,
    ) -> None:
        self.sensor = sensor
        self.driver = driver
        self.motor = motor
        self.invert = bool(invert)

        # Position target (public API is position-only; motor command is internal).
        self._target_pos_m100_100: float = 0.0

        # Tunables (safe defaults; tune on hardware).
        self.kp: float = 0.02
        self.max_cmd: float = 1.0
        self.deadband: float = 1.0

        # Debugging
        self._last_cmd_print_t = 0.0

        # Calibrator
        self.calibrator = SourcceyZCalibrator(self)

    @property
    def is_connected(self) -> bool:
        return self.sensor.is_connected

    def connect(self) -> None:
        self.sensor.connect()

    def disconnect(self) -> None:
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


    def stop(self) -> None:
        """Stop motor output and reset integrator."""
        if self.driver is not None:
            self.driver.set_velocity(self.motor, 0.0, normalize=True, instant=True)

    # --- Reads (delegated to sensor) ---
    def read_position(self) -> float:
        return self.sensor.read_position_m100_100()

     # --- Write Functions ---
    def write_position(self, target_pos_m100_100: float) -> None:
        self._target_pos_m100_100 = max(-100.0, min(100.0, float(target_pos_m100_100)))
