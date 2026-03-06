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

# bq34z100 standard command addresses (TI TRM, Data Commands table)
REG_STATE_OF_CHARGE = 0x02  # 1 byte (%)
REG_MAX_ERROR = 0x03  # 1 byte (%)
REG_REMAINING_CAPACITY = 0x04  # 2 bytes (mAh)
REG_FULL_CHARGE_CAPACITY = 0x06  # 2 bytes (mAh)
REG_VOLTAGE = 0x08  # 2 bytes (mV)
REG_AVERAGE_CURRENT = 0x0A  # 2 bytes (mA, signed)
REG_TEMPERATURE = 0x0C  # 2 bytes (0.1 K)
REG_FLAGS = 0x0E  # 2 bytes


@dataclass
class BatteryGaugeData:
    bat_pin_voltage_v: float
    pack_voltage_est_v: float
    current_a: float
    temperature_c: float
    state_of_charge_pct: int
    max_error_pct: int
    remaining_capacity_ah: float
    full_charge_capacity_ah: float
    flags_hex: str
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

    @staticmethod
    def _swap_u16(raw: int) -> int:
        return ((raw & 0xFF) << 8) | ((raw >> 8) & 0xFF)

    def _read_u16(self, bus: Any, reg: int, swap_word_bytes: bool) -> int:
        # bq34z100 commands are 16-bit "read-word" transactions.
        raw = bus.read_word_data(self._addr, reg)
        return self._swap_u16(raw) if swap_word_bytes else raw

    def _read_u8(self, bus: Any, reg: int) -> int:
        return bus.read_byte_data(self._addr, reg)

    @staticmethod
    def _to_s16(raw: int) -> int:
        return raw - 0x10000 if raw & 0x8000 else raw

    def read(
        self,
        configured_rsense_mohm: float,
        actual_rsense_mohm: float,
        swap_word_bytes: bool,
        divider_top_kohm: float,
        divider_bottom_kohm: float,
    ) -> BatteryGaugeData:
        if actual_rsense_mohm <= 0.0 or configured_rsense_mohm <= 0.0:
            raise ValueError("Sense resistor values must be > 0")
        if divider_top_kohm < 0.0 or divider_bottom_kohm <= 0.0:
            raise ValueError("Divider values must be top >= 0 and bottom > 0")

        correction = configured_rsense_mohm / actual_rsense_mohm

        with self._open_bus(self._bus_num) as bus:
            soc_pct = self._read_u8(bus, REG_STATE_OF_CHARGE)
            max_error_pct = self._read_u8(bus, REG_MAX_ERROR)
            rem_cap_mah = self._read_u16(bus, REG_REMAINING_CAPACITY, swap_word_bytes)
            fcc_mah = self._read_u16(bus, REG_FULL_CHARGE_CAPACITY, swap_word_bytes)
            temp_raw = self._read_u16(bus, REG_TEMPERATURE, swap_word_bytes)
            voltage_mv = self._read_u16(bus, REG_VOLTAGE, swap_word_bytes)
            current_raw = self._read_u16(bus, REG_AVERAGE_CURRENT, swap_word_bytes)
            flags = self._read_u16(bus, REG_FLAGS, swap_word_bytes)

        current_ma = self._to_s16(current_raw)
        bat_pin_voltage_v = voltage_mv / 1000.0
        divider_gain = (divider_top_kohm + divider_bottom_kohm) / divider_bottom_kohm

        return BatteryGaugeData(
            bat_pin_voltage_v=bat_pin_voltage_v,
            pack_voltage_est_v=bat_pin_voltage_v * divider_gain,
            current_a=(current_ma * correction) / 1000.0,
            temperature_c=(temp_raw / 10.0) - 273.15,
            state_of_charge_pct=max(0, min(100, soc_pct)),
            max_error_pct=max_error_pct,
            remaining_capacity_ah=(rem_cap_mah * correction) / 1000.0,
            full_charge_capacity_ah=(fcc_mah * correction) / 1000.0,
            flags_hex=f"0x{flags:04X}",
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
        "--swap-word-bytes",
        action="store_true",
        help="Swap high/low bytes for 16-bit commands (use if adapter endianness is reversed)",
    )
    p.add_argument(
        "--divider-top-kohm",
        type=float,
        default=249.0,
        help="Voltage-divider top resistor from pack+ to BAT pin (kOhm)",
    )
    p.add_argument(
        "--divider-bottom-kohm",
        type=float,
        default=16.5,
        help="Voltage-divider bottom resistor from BAT pin to GND (kOhm)",
    )
    p.add_argument(
        "--watch",
        type=float,
        default=0.0,
        help="Read continuously every N seconds (0 = single read)",
    )
    return p


def _print_human(data: BatteryGaugeData) -> None:
    print(f"BAT Pin Voltage    : {data.bat_pin_voltage_v:.3f} V")
    print(f"Pack Voltage (est) : {data.pack_voltage_est_v:.3f} V")
    print(f"Current            : {data.current_a:+.3f} A")
    print(f"Temperature        : {data.temperature_c:.2f} C")
    print(f"State of Charge    : {data.state_of_charge_pct:d} %")
    print(f"Max Error          : {data.max_error_pct:d} %")
    print(f"Remaining Capacity : {data.remaining_capacity_ah:.3f} Ah")
    print(f"Full Charge Cap.   : {data.full_charge_capacity_ah:.3f} Ah")
    print(f"Flags              : {data.flags_hex}")
    print(f"Shunt Correction   : x{data.shunt_correction_factor:.4f}")


def main() -> int:
    args = _build_parser().parse_args()
    gauge = BQ34Z100(bus=args.bus, address=args.address)

    while True:
        data = gauge.read(
            configured_rsense_mohm=args.configured_rsense_mohm,
            actual_rsense_mohm=args.actual_rsense_mohm,
            swap_word_bytes=args.swap_word_bytes,
            divider_top_kohm=args.divider_top_kohm,
            divider_bottom_kohm=args.divider_bottom_kohm,
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
