#!/usr/bin/env python3
"""Read and print Sourccey IMU data from LSM6DSOX + LIS3MDL."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from lerobot.sensors.imu import AdafruitLSM6DSOXLIS3MDLIMU, IMUConfig
from lerobot.sensors.imu.calibration import IMUCalibration


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Print IMU data from LSM6DSOX + LIS3MDL")
    p.add_argument("--interval-s", type=float, default=10.0, help="Print interval in seconds (default: 10)")
    p.add_argument("--bus", type=int, default=1, help="I2C bus number (default: 1)")
    p.add_argument("--lsm6dsox-address", type=lambda x: int(x, 0), default=0x6A, help="LSM6DSOX I2C address")
    p.add_argument("--lis3mdl-address", type=lambda x: int(x, 0), default=0x1C, help="LIS3MDL I2C address")
    p.add_argument("--calibration-json", type=Path, default=None, help="Optional path to calibration JSON")
    p.add_argument("--once", action="store_true", help="Print one sample and exit")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    return p


def _sample_to_payload(sample) -> dict[str, object]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_timestamp_ns": sample.timestamp_ns,
        "valid": sample.valid,
        "error": sample.error,
        "accel_m_s2": [round(v, 5) for v in sample.accel_m_s2],
        "gyro_rad_s": [round(v, 5) for v in sample.gyro_rad_s],
        "mag_uT": [round(v, 5) for v in sample.mag_uT],
        "temperature_c": None if sample.temperature_c is None else round(sample.temperature_c, 3),
    }


def _emit(payload: dict[str, object], *, pretty: bool) -> None:
    if pretty:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


def main() -> int:
    args = _build_parser().parse_args()
    if args.interval_s <= 0:
        raise SystemExit("--interval-s must be > 0")

    calibration = IMUCalibration()
    if args.calibration_json is not None:
        calibration = IMUCalibration.from_json_file(args.calibration_json)

    config = IMUConfig(
        bus_num=args.bus,
        lsm6dsox_address=args.lsm6dsox_address,
        lis3mdl_address=args.lis3mdl_address,
    )
    imu = AdafruitLSM6DSOXLIS3MDLIMU(config=config, calibration=calibration)
    imu.connect()

    try:
        if args.once:
            _emit(_sample_to_payload(imu.read()), pretty=bool(args.pretty))
            return 0

        while True:
            _emit(_sample_to_payload(imu.read()), pretty=bool(args.pretty))
            time.sleep(float(args.interval_s))
    except KeyboardInterrupt:
        print("Stopped IMU monitoring.")
    finally:
        imu.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

