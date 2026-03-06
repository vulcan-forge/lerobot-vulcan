#!/usr/bin/env python3
"""Read capacity and basic telemetry from a TI bq34z100 fuel gauge.

Hardware assumptions for your board:
- I2C connected to bq34z100 (default address 0x55)
- SRP/SRN shunt = 12.5 mOhm
- BAT and TS are wired, so voltage/temperature readings are available

Notes:
- Capacity/current values are only accurate after the gauge is configured for your
  LiFePO4 pack (design capacity + chemistry + calibration).
- If the gauge was configured for a different shunt value than the actual hardware,
  use --configured-rsense-mohm and --actual-rsense-mohm to compensate.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_I2C_ADDR = 0x55

# bq34z100 standard command addresses
REG_TEMPERATURE = 0x02  # 0.1 K
REG_VOLTAGE = 0x04  # mV
REG_AVERAGE_CURRENT = 0x10  # mA (signed)
REG_REMAINING_CAPACITY = 0x0C  # mAh
REG_FULL_CHARGE_CAPACITY = 0x0E  # mAh
REG_STATE_OF_CHARGE = 0x22  # %


@dataclass
class BatteryGaugeData:
    voltage_v: float
    current_a: float
    temperature_c: float
    state_of_charge_pct: int
    remaining_capacity_ah: float
    full_charge_capacity_ah: float
    shunt_correction_factor: float


class BQ34Z100:
    def __init__(self, bus: int = 1, address: int = DEFAULT_I2C_ADDR) -> None:
        self._bus_num = bus
        self._addr = address

    @staticmethod
    def _open_bus(bus_num: int) -> Any:
        try:
            from smbus2 import SMBus
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "smbus2 is required. Install with: pip install smbus2"
            ) from exc
        return SMBus(bus_num)

    def _read_u16(self, bus: Any, reg: int) -> int:
        data = bus.read_i2c_block_data(self._addr, reg, 2)
        return data[0] | (data[1] << 8)

    @staticmethod
    def _to_s16(raw: int) -> int:
        return raw - 0x10000 if raw & 0x8000 else raw

    def read(self, configured_rsense_mohm: float, actual_rsense_mohm: float) -> BatteryGaugeData:
        if actual_rsense_mohm <= 0.0 or configured_rsense_mohm <= 0.0:
            raise ValueError("Sense resistor values must be > 0")

        correction = configured_rsense_mohm / actual_rsense_mohm

        with self._open_bus(self._bus_num) as bus:
            temp_raw = self._read_u16(bus, REG_TEMPERATURE)
            voltage_mv = self._read_u16(bus, REG_VOLTAGE)
            current_raw = self._read_u16(bus, REG_AVERAGE_CURRENT)
            rem_cap_mah = self._read_u16(bus, REG_REMAINING_CAPACITY)
            fcc_mah = self._read_u16(bus, REG_FULL_CHARGE_CAPACITY)
            soc_pct = self._read_u16(bus, REG_STATE_OF_CHARGE)

        current_ma = self._to_s16(current_raw)

        return BatteryGaugeData(
            voltage_v=voltage_mv / 1000.0,
            current_a=(current_ma * correction) / 1000.0,
            temperature_c=(temp_raw / 10.0) - 273.15,
            state_of_charge_pct=max(0, min(100, soc_pct)),
            remaining_capacity_ah=(rem_cap_mah * correction) / 1000.0,
            full_charge_capacity_ah=(fcc_mah * correction) / 1000.0,
            shunt_correction_factor=correction,
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read battery capacity from bq34z100")
    p.add_argument("--bus", type=int, default=1, help="I2C bus number (default: 1)")
    p.add_argument("--address", type=lambda x: int(x, 0), default=DEFAULT_I2C_ADDR, help="I2C address, e.g. 0x55")
    p.add_argument(
        "--actual-rsense-mohm",
        type=float,
        default=12.5,
        help="Actual SRP/SRN shunt resistor in milliohms",
    )
    p.add_argument(
        "--configured-rsense-mohm",
        type=float,
        default=12.5,
        help="Shunt value that the gauge firmware is configured for (milliohms)",
    )
    p.add_argument("--json", action="store_true", help="Print JSON output")
    p.add_argument(
        "--watch",
        type=float,
        default=0.0,
        help="Read continuously every N seconds (0 = single read)",
    )
    return p


def _print_human(data: BatteryGaugeData) -> None:
    print(f"Voltage            : {data.voltage_v:.3f} V")
    print(f"Current            : {data.current_a:+.3f} A")
    print(f"Temperature        : {data.temperature_c:.2f} C")
    print(f"State of Charge    : {data.state_of_charge_pct:d} %")
    print(f"Remaining Capacity : {data.remaining_capacity_ah:.3f} Ah")
    print(f"Full Charge Cap.   : {data.full_charge_capacity_ah:.3f} Ah")
    print(f"Shunt Correction   : x{data.shunt_correction_factor:.4f}")


def main() -> int:
    args = _build_parser().parse_args()
    gauge = BQ34Z100(bus=args.bus, address=args.address)

    while True:
        data = gauge.read(
            configured_rsense_mohm=args.configured_rsense_mohm,
            actual_rsense_mohm=args.actual_rsense_mohm,
        )

        if args.json:
            print(json.dumps(asdict(data), separators=(",", ":")))
        else:
            _print_human(data)

        if args.watch <= 0.0:
            break
        time.sleep(args.watch)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
