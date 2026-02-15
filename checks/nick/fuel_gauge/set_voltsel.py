#!/usr/bin/env python3
# Set/clear VOLTSEL bit in Pack Config (subclass 64, offset 0).
# VOLTSEL=0 -> 0x08 reports BAT; VOLTSEL=1 -> 0x08 reports PACK.

from __future__ import annotations

import argparse
import time
from smbus2 import SMBus

I2C_BUS_DEFAULT = 1
BQ_ADDR_DEFAULT = 0x55

# Data Flash registers
REG_DF_CLASS = 0x3E
REG_DF_BLOCK = 0x3F
REG_BLOCKDATA_START = 0x40
REG_BLOCKDATA_CKSUM = 0x60
REG_BLOCKDATA_CTRL = 0x61

PACK_SUBCLASS = 64
PACK_BLOCK = 0
PACK_CFG_OFFSET = 0
SERIES_OFFSET_DEFAULT = 7

VOLTSEL_MASK = 0x0800

# Control() register and subcommands
CMD_CONTROL = 0x00
SUB_CAL_ENABLE = 0x002D
SUB_RESET = 0x0041


def _checksum(block: bytes) -> int:
    return 255 - (sum(block) % 256)


def _read_block(bus: SMBus, subclass: int, block_index: int) -> bytes:
    bus.write_byte_data(BQ_ADDR, REG_BLOCKDATA_CTRL, 0x00)
    bus.write_byte_data(BQ_ADDR, REG_DF_CLASS, subclass)
    bus.write_byte_data(BQ_ADDR, REG_DF_BLOCK, block_index & 0xFF)
    return bytes(bus.read_i2c_block_data(BQ_ADDR, REG_BLOCKDATA_START, 32))


def _write_block_diff(bus: SMBus, subclass: int, block_index: int, old_block: bytes, new_block: bytes) -> None:
    bus.write_byte_data(BQ_ADDR, REG_BLOCKDATA_CTRL, 0x00)
    bus.write_byte_data(BQ_ADDR, REG_DF_CLASS, subclass)
    bus.write_byte_data(BQ_ADDR, REG_DF_BLOCK, block_index & 0xFF)
    for i, (old_b, new_b) in enumerate(zip(old_block, new_block)):
        if old_b != new_b:
            bus.write_byte_data(BQ_ADDR, REG_BLOCKDATA_START + i, new_b)
    bus.write_byte_data(BQ_ADDR, REG_BLOCKDATA_CKSUM, _checksum(new_block))


def _u16_be(b: bytes, offset: int) -> int:
    return (b[offset] << 8) | b[offset + 1]


def _set_u16_be(b: bytearray, offset: int, value: int) -> None:
    b[offset] = (value >> 8) & 0xFF
    b[offset + 1] = value & 0xFF


def _write_control(bus: SMBus, subcmd: int) -> None:
    lo = subcmd & 0xFF
    hi = (subcmd >> 8) & 0xFF
    bus.write_i2c_block_data(BQ_ADDR, CMD_CONTROL, [lo, hi])


def main() -> None:
    ap = argparse.ArgumentParser(description="Set or clear VOLTSEL bit in Pack Config.")
    ap.add_argument("--bus", type=int, default=I2C_BUS_DEFAULT)
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=BQ_ADDR_DEFAULT)
    ap.add_argument("--value", type=int, choices=[0, 1], default=0, help="0=BAT, 1=PACK")
    ap.add_argument("--series-offset", type=int, default=SERIES_OFFSET_DEFAULT)
    ap.add_argument("--cfgupdate", action="store_true", help="Toggle CAL_ENABLE around the write.")
    ap.add_argument("--reset", action="store_true", help="Send RESET after write.")
    ap.add_argument("--write", action="store_true", help="Apply changes to the device.")
    args = ap.parse_args()

    global BQ_ADDR
    BQ_ADDR = args.addr

    with SMBus(args.bus) as bus:
        old_block = _read_block(bus, PACK_SUBCLASS, PACK_BLOCK)
        pack_cfg = _u16_be(old_block, PACK_CFG_OFFSET)
        series_cells = old_block[args.series_offset]
        voltsel = 1 if (pack_cfg & VOLTSEL_MASK) else 0

        print(f"Pack Config: 0x{pack_cfg:04X} (VOLTSEL={voltsel})")
        print(f"Series Cells: {series_cells} (offset {args.series_offset})")

        new_cfg = pack_cfg & ~VOLTSEL_MASK if args.value == 0 else pack_cfg | VOLTSEL_MASK
        if new_cfg == pack_cfg:
            print("No change needed.")
            return

        new_block = bytearray(old_block)
        _set_u16_be(new_block, PACK_CFG_OFFSET, new_cfg)

        print(f"Planned update: 0x{pack_cfg:04X} -> 0x{new_cfg:04X}")
        if not args.write:
            print("Dry-run only. Re-run with --write to apply.")
            return

        if args.cfgupdate:
            _write_control(bus, SUB_CAL_ENABLE)
            time.sleep(0.02)

        _write_block_diff(bus, PACK_SUBCLASS, PACK_BLOCK, old_block, bytes(new_block))
        time.sleep(0.05)

        if args.cfgupdate:
            _write_control(bus, SUB_CAL_ENABLE)
            time.sleep(0.02)
        if args.reset:
            _write_control(bus, SUB_RESET)
            time.sleep(0.1)

        verify_block = _read_block(bus, PACK_SUBCLASS, PACK_BLOCK)
        verify_cfg = _u16_be(verify_block, PACK_CFG_OFFSET)
        verify_voltsel = 1 if (verify_cfg & VOLTSEL_MASK) else 0
        print(f"Verify Pack Config: 0x{verify_cfg:04X} (VOLTSEL={verify_voltsel})")


if __name__ == "__main__":
    main()
