from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class IMUReporterConfig:
    """Scaffold config for upcoming IMU module extraction from sourccey_host."""

    enabled: bool = False
    interval_s: float = 10.0
    bus_num: int = 1
    lsm6dsox_address: int = 0x6A
    lis3mdl_address: int = 0x1C
