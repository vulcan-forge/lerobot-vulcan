from __future__ import annotations

import json
from dataclasses import dataclass
import time
from typing import Optional

from gpiozero import MCP3008

# IMPORTANT:
# gpiozero MCP3008 channels are 0..7 (0-based).
# If you want "ADC Channel 1" on the chip, set this to 1.
ADC_CHANNEL = 1

# Measure your Pi 3.3V rail if you want better accuracy
VREF = 3.30

# Mechanical min and max values for the potentiometer
POT_MECHANICAL_MIN = 563
POT_MECHANICAL_MAX = 77

# Sampling / smoothing (copied style from your battery.py)
AVERAGE_SAMPLES = 8
FILTER_ALPHA = 0.4  # 0..1, lower = smoother
RAW_DEADBAND = 3
_filtered_value: Optional[float] = None


@dataclass
class PotentiometerData:
    raw: int          # 0..1023
    volts: float      # 0..VREF
    normalized: float # 0..1
    percent: int      # 0..100


_adc = MCP3008(channel=ADC_CHANNEL, port=0, device=0)  # CE0 -> /dev/spidev0.0

def _get_adc() -> MCP3008:
    global _adc
    if _adc is None:
        _adc = MCP3008(channel=ADC_CHANNEL)
    return _adc


def get_pot_raw_filtered() -> float:
    """Returns filtered raw reading as float (0..1023)."""
    global _filtered_value
    adc = _get_adc()

    total = 0.0
    for _ in range(AVERAGE_SAMPLES):
        total += float(adc.raw_value)

    instant = total / AVERAGE_SAMPLES

    # Deadband: if the new reading is very close to the current filtered value,
    # treat it as noise and don't move the filter.
    if _filtered_value is not None and abs(instant - _filtered_value) <= RAW_DEADBAND:
        instant = _filtered_value

    if _filtered_value is None:
        _filtered_value = instant
    else:
        _filtered_value = FILTER_ALPHA * instant + (1.0 - FILTER_ALPHA) * _filtered_value

    return _filtered_value

def normalize_between(raw: int, raw_min: int, raw_max: int) -> float:
    # Handles reversed wiring too (raw decreases as you turn "forward")
    if raw_min == raw_max:
        return 0.0

    lo, hi = (raw_min, raw_max) if raw_min < raw_max else (raw_max, raw_min)

    x = (raw - lo) / (hi - lo)  # 0..1 (unclamped)
    x = max(0.0, min(1.0, x))   # clamp

    # If user-defined "max" is numerically less than "min", invert
    if raw_max < raw_min:
        x = 1.0 - x

    return x * 200.0 - 100.0

def get_pot_data() -> PotentiometerData:
    raw_f = get_pot_raw_filtered()
    raw = int(round(max(0.0, min(1023.0, raw_f))))
    volts = (raw / 1023.0) * VREF
    normalized = normalize_between(raw, POT_MECHANICAL_MIN, POT_MECHANICAL_MAX)
    percent = int(round(normalized))
    return PotentiometerData(raw=raw, volts=volts, normalized=normalized, percent=percent)


if __name__ == "__main__":
    try:
        while True:
            d = get_pot_data()
            print(
                json.dumps(
                    {
                        "raw": d.raw,
                        "volts": round(d.volts, 3),
                        "normalized": round(d.normalized, 4),
                        "percent": d.percent,
                        "adc_channel": ADC_CHANNEL,
                    }
                )
            )
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    except Exception:
        print(json.dumps({"raw": -1, "volts": -1.0, "normalized": -1.0, "percent": -1, "adc_channel": ADC_CHANNEL}))
