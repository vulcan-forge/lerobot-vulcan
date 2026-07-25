"""Read Sourccey's Z-axis potentiometer without commanding the actuator."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from lerobot.robots.sourccey.sourccey.sourccey_z_actuator.sourccey_z_actuator import ZSensor
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

DEFAULT_CALIBRATION_PATH = (
    HF_LEROBOT_CALIBRATION
    / ROBOTS
    / "sourccey_z_actuator"
    / "sourccey_z_actuator.json"
)


def load_calibration(path: Path) -> tuple[int, int, bool]:
    """Load the same Z calibration used by the robot runtime."""
    with path.open(encoding="utf-8") as calibration_file:
        data = json.load(calibration_file)["z_actuator"]

    raw_min = int(data["raw_min"])
    raw_max = int(data["raw_max"])
    invert = bool(data["invert"])
    if raw_min == raw_max:
        raise ValueError(f"Invalid Z calibration in {path}: raw_min equals raw_max ({raw_min}).")
    return raw_min, raw_max, invert


def classify_reading(
    raw: int,
    position: float,
    previous_position: float | None,
    *,
    calibration_min: int = 0,
    calibration_max: int = 1023,
) -> str:
    """Flag electrical rail readings and physically implausible one-sample jumps."""
    warnings: list[str] = []
    if raw <= 2 or raw >= 1021:
        warnings.append("ADC_RAIL")
    calibrated_low = min(calibration_min, calibration_max)
    calibrated_high = max(calibration_min, calibration_max)
    if raw < calibrated_low:
        warnings.append("BELOW_CAL_MIN")
    elif raw > calibrated_high:
        warnings.append("ABOVE_CAL_MAX")
    if previous_position is not None and abs(position - previous_position) > 40.0:
        warnings.append("LARGE_JUMP")
    return ",".join(warnings) if warnings else "OK"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read the MCP3008 Z potentiometer directly. This diagnostic never drives the actuator."
        )
    )
    parser.add_argument("--channel", type=int, default=1, help="MCP3008 channel (default: 1).")
    parser.add_argument("--vref", type=float, default=3.30, help="ADC reference voltage (default: 3.30).")
    parser.add_argument("--samples", type=int, default=50, help="ADC samples averaged per reading (default: 50).")
    parser.add_argument("--interval", type=float, default=0.2, help="Seconds between readings (default: 0.2).")
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of readings; 0 watches continuously until Ctrl+C (default: 0).",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help=f"Calibration JSON (default: {DEFAULT_CALIBRATION_PATH}).",
    )
    parser.add_argument("--raw-min", type=int, help="Override calibrated raw minimum.")
    parser.add_argument("--raw-max", type=int, help="Override calibrated raw maximum.")
    parser.add_argument(
        "--invert",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override calibration direction with --invert or --no-invert.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be at least 1")
    if args.interval < 0:
        raise SystemExit("--interval cannot be negative")
    if args.count < 0:
        raise SystemExit("--count cannot be negative")

    if args.calibration.is_file():
        raw_min, raw_max, invert = load_calibration(args.calibration)
        calibration_source = str(args.calibration)
    else:
        raw_min, raw_max, invert = 0, 1023, True
        calibration_source = "full ADC range (calibration file not found)"

    raw_min = args.raw_min if args.raw_min is not None else raw_min
    raw_max = args.raw_max if args.raw_max is not None else raw_max
    invert = args.invert if args.invert is not None else invert
    if raw_min == raw_max:
        raise SystemExit("raw minimum and maximum must differ")

    sensor = ZSensor(
        adc_channel=args.channel,
        vref=args.vref,
        average_samples=args.samples,
        invert=invert,
    )
    sensor.set_calibration(raw_min=raw_min, raw_max=raw_max, invert=invert)

    raw_midpoint = (raw_min + raw_max) / 2.0
    min_position = sensor.raw_to_pos_m100_100(raw_min)
    max_position = sensor.raw_to_pos_m100_100(raw_max)

    print("Sourccey Z potentiometer monitor (read-only; actuator output is not touched)")
    print("CALIBRATION")
    print(f"  file:         {calibration_source}")
    print(f"  raw min:      {raw_min:7d} -> z.pos {min_position:+7.2f}")
    print(f"  raw midpoint: {raw_midpoint:7.1f} -> z.pos   +0.00")
    print(f"  raw max:      {raw_max:7d} -> z.pos {max_position:+7.2f}")
    print(f"  inverted:     {invert}")
    print(f"ADC SETTINGS: channel={args.channel} vref={args.vref:.2f}V samples={args.samples}")
    print("time                 raw  voltage    z.pos    delta  status")

    previous_position: float | None = None
    reading_count = 0
    try:
        sensor.connect()
        while args.count == 0 or reading_count < args.count:
            reading = sensor.read_raw()
            position = sensor.raw_to_pos_m100_100(reading.raw)
            delta = math.nan if previous_position is None else position - previous_position
            status = classify_reading(
                reading.raw,
                position,
                previous_position,
                calibration_min=raw_min,
                calibration_max=raw_max,
            )
            delta_text = "   ---" if math.isnan(delta) else f"{delta:+7.2f}"
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {reading.raw:4d}  "
                f"{reading.voltage:7.4f}  {position:+7.2f}  {delta_text}  {status}",
                flush=True,
            )
            previous_position = position
            reading_count += 1
            if args.count == 0 or reading_count < args.count:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sensor.disconnect()


if __name__ == "__main__":
    main()
