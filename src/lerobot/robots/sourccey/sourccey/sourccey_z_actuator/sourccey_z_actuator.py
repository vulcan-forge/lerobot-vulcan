from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


try:
    from gpiozero import MCP3008  # type: ignore
except Exception:  # pragma: no cover
    MCP3008 = None  # type: ignore


@dataclass(frozen=True)
class ZActuatorReading:
    raw_10bit: int     # 0..1023 (native MCP3008)
    raw: int           # 0..raw_scale_max (scaled; default 0..4095)
    voltage: float     # volts at the MCP3008 pin


class SourcceyZActuator:
    """
    Reads a linear actuator feedback potentiometer via MCP3008 and exposes a position in [-100, 100].

    Default calibration assumes you're using the *scaled* raw (0..4095):
      raw_min=1800 -> -100
      raw_max=2000 -> +100
    """

    def __init__(
        self,
        *,
        adc_channel: int = 1,
        vref: float = 3.30,
        average_samples: int = 50,
        raw_scale_max: int = 4095,
        raw_min: int = 1800,
        raw_max: int = 2000,
        invert: bool = False,
    ) -> None:
        self.adc_channel = adc_channel
        self.vref = vref
        self.average_samples = average_samples

        self.raw_scale_max = raw_scale_max
        self.raw_min = raw_min
        self.raw_max = raw_max
        self.invert = invert

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
        self.raw_min = int(raw_min)
        self.raw_max = int(raw_max)
        if invert is not None:
            self.invert = bool(invert)

    def read_raw_10bit(self) -> int:
        """Averaged native MCP3008 raw (0..1023)."""
        if self._adc is None:
            self.connect()
        assert self._adc is not None

        total = 0.0
        for _ in range(self.average_samples):
            total += float(self._adc.raw_value)
        return int(round(total / self.average_samples))

    def read(self) -> ZActuatorReading:
        """Read averaged raw_10bit, scaled raw, and voltage."""
        raw_10bit = self.read_raw_10bit()

        voltage = (raw_10bit / 1023.0) * self.vref

        # Scale 10-bit raw into e.g. 0..4095 so calibration values like 1800..2000 make sense.
        raw_scaled = int(round((raw_10bit / 1023.0) * float(self.raw_scale_max)))

        return ZActuatorReading(raw_10bit=raw_10bit, raw=raw_scaled, voltage=voltage)

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else hi if x > hi else x

    def raw_to_pos_m100_100(self, raw: int) -> float:
        """Convert scaled raw into position [-100, 100] using current calibration."""
        rmin, rmax = float(self.raw_min), float(self.raw_max)
        if rmax == rmin:
            return 0.0

        t = (float(raw) - rmin) / (rmax - rmin)  # 0..1 ideally
        t = self._clamp(t, 0.0, 1.0)
        pos = -100.0 + 200.0 * t  # -100..100

        return -pos if self.invert else pos

    def read_position_m100_100(self) -> float:
        """One-shot: read sensor and return position in [-100, 100]."""
        reading = self.read()
        return self.raw_to_pos_m100_100(reading.raw)
