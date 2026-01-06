from dataclasses import dataclass
import json
from typing import Optional
import time

# gpiozero is only available on target hardware (e.g., Raspberry Pi).
# Make it optional so this module can be imported on dev machines.
try:
    from gpiozero import MCP3008  # type: ignore
except Exception:  # pragma: no cover
    MCP3008 = None  # type: ignore

# ADC Configuration
ADC_CHANNEL = 1

# Measure this with a multimeter if you can for better accuracy
VREF = 3.30  # Pi 3.3V rail (adjust to your measured value)

# Potentiometer wiper is typically already 0..3.3V, so no divider:
VOLTAGE_DIVIDER_RATIO = 1.0

# Sampling configuration
AVERAGE_SAMPLES = 50

# Optional smoothing (battery.py-style). If your value feels "laggy", increase alpha.
FILTER_ALPHA = 0.2  # 0..1, higher = more responsive
_filtered_voltage: Optional[float] = None

# Global ADC instance (initialized on first use)
_adc: Optional["MCP3008"] = None


@dataclass
class LinearActuatorData:
    raw: int
    voltage_instant: float
    voltage_filtered: float


def _get_adc() -> "MCP3008":
    global _adc
    if MCP3008 is None:
        raise RuntimeError("gpiozero is not installed; MCP3008 is unavailable on this machine.")
    if _adc is None:
        _adc = MCP3008(channel=ADC_CHANNEL)
    return _adc


def get_linear_actuator_voltage() -> LinearActuatorData:
    """Return averaged + optionally filtered voltage on MCP3008 channel 1."""
    global _filtered_voltage
    adc = _get_adc()

    total_raw = 0.0
    total_v = 0.0

    for _ in range(AVERAGE_SAMPLES):
        raw = float(adc.raw_value)  # 0..1023
        v_adc = (raw / 1023.0) * VREF
        v_sig = v_adc / VOLTAGE_DIVIDER_RATIO
        total_raw += raw
        total_v += v_sig

    raw_avg = int(round(total_raw / AVERAGE_SAMPLES))
    instant_voltage = total_v / AVERAGE_SAMPLES

    if _filtered_voltage is None:
        _filtered_voltage = instant_voltage
    else:
        _filtered_voltage = (
            FILTER_ALPHA * instant_voltage
            + (1.0 - FILTER_ALPHA) * _filtered_voltage
        )

    return LinearActuatorData(
        raw=raw_avg,
        voltage_instant=instant_voltage,
        voltage_filtered=_filtered_voltage,
    )


if __name__ == "__main__":
    while True:
        d = get_linear_actuator_voltage()
        print(json.dumps({"raw": d.raw, "voltage_filtered": round(d.voltage_filtered, 4)}))
        time.sleep(0.2)
