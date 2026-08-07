#!/usr/bin/env python3
"""Print a single-line IMU sample for quick hardware sanity checks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from lerobot.sensors.imu import AdafruitLSM6DSOXLIS3MDLIMU, IMUConfig
from lerobot.sensors.imu.calibration import IMUCalibration


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="One-line IMU sanity check")
    p.add_argument("--bus", type=int, default=1, help="I2C bus number (default: 1)")
    p.add_argument("--lsm6dsox-address", type=lambda x: int(x, 0), default=0x6A, help="LSM6DSOX I2C address")
    p.add_argument("--lis3mdl-address", type=lambda x: int(x, 0), default=0x1C, help="LIS3MDL I2C address")
    p.add_argument("--calibration-json", type=Path, default=None, help="Optional calibration JSON path")
    p.add_argument("--precision", type=int, default=3, help="Decimal precision (default: 3)")
    return p


def _fmt_vec(v: tuple[float, float, float], precision: int) -> str:
    return f"({v[0]:.{precision}f},{v[1]:.{precision}f},{v[2]:.{precision}f})"


def main() -> int:
    args = _build_parser().parse_args()

    calibration = IMUCalibration()
    if args.calibration_json is not None:
        calibration = IMUCalibration.from_json_file(args.calibration_json)

    config = IMUConfig(
        bus_num=args.bus,
        lsm6dsox_address=args.lsm6dsox_address,
        lis3mdl_address=args.lis3mdl_address,
    )
    imu = AdafruitLSM6DSOXLIS3MDLIMU(config=config, calibration=calibration)

    variant = None
    try:
        imu.connect()
        variant = getattr(imu, "_imu6_variant", None)
        sample = imu.read()
    finally:
        imu.disconnect()

    stamp = datetime.now(timezone.utc).isoformat()
    if not sample.valid:
        print(f"{stamp} IMU_CHECK FAIL error={sample.error}")
        return 1

    temp_str = "None" if sample.temperature_c is None else f"{sample.temperature_c:.{args.precision}f}"
    print(
        f"{stamp} IMU_CHECK OK "
        f"imu6_variant={variant} "
        f"accel_m_s2={_fmt_vec(sample.accel_m_s2, args.precision)} "
        f"gyro_rad_s={_fmt_vec(sample.gyro_rad_s, args.precision)} "
        f"mag_uT={_fmt_vec(sample.mag_uT, args.precision)} "
        f"temp_c={temp_str}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
