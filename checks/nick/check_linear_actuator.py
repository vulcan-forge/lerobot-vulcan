import argparse
import json
import time
from dataclasses import dataclass
from typing import Optional

# gpiozero only exists on target hardware
try:
    from gpiozero import MCP3008  # type: ignore
except Exception:
    MCP3008 = None  # type: ignore

# MCP3008 / signal config
ACTUATOR_ADC_CHANNEL = 1
VREF = 3.30  # measure your Pi 3V3 rail for better accuracy
AVERAGE_SAMPLES = 50

# Calibration (fill these in after you observe raw at each end-stop)
RAW_MIN = 120  # fully retracted raw_value
RAW_MAX = 920  # fully extended raw_value
STROKE_MM = 100.0  # optional

_adc: Optional["MCP3008"] = None


@dataclass
class ActuatorData:
    raw: int
    voltage: float
    pos01: float
    pos_mm: float


def _get_adc() -> "MCP3008":
    global _adc
    if MCP3008 is None:
        raise RuntimeError("gpiozero is not installed; MCP3008 is unavailable on this machine.")
    if _adc is None:
        _adc = MCP3008(channel=ACTUATOR_ADC_CHANNEL)
    return _adc


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def read_actuator_data() -> ActuatorData:
    adc = _get_adc()

    # Average raw counts for stability
    total_raw = 0.0
    for _ in range(AVERAGE_SAMPLES):
        total_raw += float(adc.raw_value)  # 0..1023

    raw = int(round(total_raw / AVERAGE_SAMPLES))
    voltage = (raw / 1023.0) * VREF

    # Map raw -> 0..1 position
    if RAW_MAX == RAW_MIN:
        pos01 = 0.0
    else:
        pos01 = (raw - RAW_MIN) / float(RAW_MAX - RAW_MIN)
        pos01 = _clamp(pos01, 0.0, 1.0)

    pos_mm = pos01 * STROKE_MM

    return ActuatorData(raw=raw, voltage=voltage, pos01=pos01, pos_mm=pos_mm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="Print repeatedly instead of once")
    ap.add_argument("--period", type=float, default=0.2, help="Seconds between prints in --watch mode")
    args = ap.parse_args()

    def emit():
        d = read_actuator_data()
        print(
            json.dumps(
                {
                    "raw": d.raw,
                    "voltage": round(d.voltage, 4),
                    "pos01": round(d.pos01, 4),
                    "pos_mm": round(d.pos_mm, 2),
                }
            )
        )

    if args.watch:
        while True:
            emit()
            time.sleep(args.period)
    else:
        emit()


if __name__ == "__main__":
    main()
