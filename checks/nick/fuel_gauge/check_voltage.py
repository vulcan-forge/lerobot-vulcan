# Read BQ34Z100 voltage with 249k/16.5k divider and output pack voltage.
# Requires: pip install smbus2

from __future__ import annotations

import argparse
import time
from smbus2 import SMBus

I2C_BUS = 1
BQ_ADDR = 0x55
CMD_VOLTAGE = 0x08

R_TOP_OHMS = 249_000.0
R_BOTTOM_OHMS = 16_500.0
V_DIV_RATIO = R_BOTTOM_OHMS / (R_TOP_OHMS + R_BOTTOM_OHMS)


def _read_voltage_bytes(bus: SMBus) -> tuple[int, int]:
    # Write command pointer, short delay, read 2 bytes.
    bus.write_byte(BQ_ADDR, CMD_VOLTAGE)
    time.sleep(0.005)
    b = bus.read_i2c_block_data(BQ_ADDR, CMD_VOLTAGE, 2)
    return b[0], b[1]


def _decode(b0: int, b1: int, byteorder: str) -> int:
    if byteorder == "le":
        return b0 | (b1 << 8)
    if byteorder == "be":
        return (b0 << 8) | b1
    raise ValueError("byteorder must be 'le' or 'be'")


def _pack_from_raw(raw: int, mode: str) -> float:
    if mode == "pack":
        return raw / 1000.0
    if mode == "bat":
        return (raw / 1000.0) / V_DIV_RATIO
    raise ValueError("mode must be 'pack' or 'bat'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read BQ34Z100 pack voltage using divider.")
    args = parser.parse_args()

    with SMBus(I2C_BUS) as bus:
        try:
            b0, b1 = _read_voltage_bytes(bus)
        except OSError as exc:
            print(f"Read failed: {exc}")
            return

    print(f"Raw bytes: 0x{b0:02X} 0x{b1:02X}")
    if b0 == 0xFF and b1 == 0xFF:
        print("Read invalid (0xFFFF).")
        return

    raw_le = _decode(b0, b1, "le")
    raw_be = _decode(b0, b1, "be")

    if raw_le == 0xFFFF and raw_be == 0xFFFF:
        print("Read invalid (0xFFFF).")
        return

    print(f"Raw LE: 0x{raw_le:04X} ({raw_le} mV)")
    print(f"Raw BE: 0x{raw_be:04X} ({raw_be} mV)")

    for label, raw in (("LE", raw_le), ("BE", raw_be)):
        if raw == 0xFFFF:
            continue
        pack_v = _pack_from_raw(raw, "pack")
        bat_v = raw / 1000.0
        print(f"{label} as PACK: {pack_v:.2f} V")
        if 0.1 <= bat_v <= 2.0:
            pack_from_bat = _pack_from_raw(raw, "bat")
            print(f"{label} as BAT:  {bat_v:.3f} V  -> Pack {pack_from_bat:.2f} V")
        else:
            print(f"{label} as BAT:  {bat_v:.3f} V (out of BAT range)")


if __name__ == "__main__":
    main()
