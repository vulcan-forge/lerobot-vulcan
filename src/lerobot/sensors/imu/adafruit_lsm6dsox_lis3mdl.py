"""Adafruit/Blinka IMU implementation for LSM6DSOX + LIS3MDL."""

from __future__ import annotations

import time
from typing import Any

from .base import BaseIMU, IMUConfig
from .calibration import IMUCalibration
from .types import IMUSample


class AdafruitLSM6DSOXLIS3MDLIMU(BaseIMU):
    """Read 9-DoF data from LSM6DSOX + LIS3MDL on a shared I2C bus."""

    def __init__(self, config: IMUConfig, calibration: IMUCalibration | None = None) -> None:
        super().__init__(config)
        self.calibration = calibration if calibration is not None else IMUCalibration()
        self._i2c: Any = None
        self._imu6: Any = None
        self._mag: Any = None

    def connect(self) -> None:
        if self._connected:
            return

        try:
            import board
            from adafruit_lis3mdl import LIS3MDL
            from adafruit_lsm6ds.lsm6dsox import LSM6DSOX
        except ImportError as exc:
            raise RuntimeError(
                "Missing Adafruit IMU dependencies. Install with: "
                "uv sync --extra sourccey or pip install adafruit-blinka "
                "adafruit-circuitpython-lsm6ds adafruit-circuitpython-lis3mdl"
            ) from exc

        self._i2c = board.I2C()
        self._imu6 = LSM6DSOX(self._i2c, address=int(self.config.lsm6dsox_address))
        self._mag = LIS3MDL(self._i2c, address=int(self.config.lis3mdl_address))
        self._connected = True

    def disconnect(self) -> None:
        if not self._connected:
            return
        self._imu6 = None
        self._mag = None
        self._i2c = None
        self._connected = False

    def read(self) -> IMUSample:
        if not self._connected:
            raise RuntimeError("IMU is not connected. Call connect() first.")
        assert self._imu6 is not None
        assert self._mag is not None

        try:
            raw_accel = tuple(float(v) for v in self._imu6.acceleration)
            raw_gyro = tuple(float(v) for v in self._imu6.gyro)
            raw_mag = tuple(float(v) for v in self._mag.magnetic)
            temp = float(self._imu6.temperature)
        except Exception as exc:
            return IMUSample(
                timestamp_ns=time.time_ns(),
                accel_m_s2=(0.0, 0.0, 0.0),
                gyro_rad_s=(0.0, 0.0, 0.0),
                mag_uT=(0.0, 0.0, 0.0),
                temperature_c=None,
                valid=False,
                error=str(exc),
            )

        accel = self.calibration.apply_accel(raw_accel)
        gyro = self.calibration.apply_gyro(raw_gyro)
        mag = self.calibration.apply_mag(raw_mag)

        return IMUSample(
            timestamp_ns=time.time_ns(),
            accel_m_s2=accel,
            gyro_rad_s=gyro,
            mag_uT=mag,
            temperature_c=temp,
            valid=True,
            error=None,
        )

