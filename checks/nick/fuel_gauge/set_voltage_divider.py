#!/usr/bin/env python3
# Set BQ34Z100 voltage divider ratio (ratio*1000) in data flash.
# For 249k/16.5k: ratio = (Rtop+Rbottom)/Rbottom = 16.091 -> 16091
#
# Default behavior is dry-run. Use --write to apply.

from __future__ import annotations

import argparse
import sys
from smbus2 import SMBus

I2C_BUS_DEFAULT = 1
BQ_ADDR_DEFAULT = 0x55

# Data Flash registers
REG_DF_CLASS = 0x3E
REG_DF_BLOCK = 0x3F
REG_BLOCKDATA_START = 0x40
REG_BLOCKDATA_CKSUM = 0x60
REG_BLOCKDATA_CTRL = 0x61

# Voltage Divider field location (subclass 104, block 0, offset 14)
DIV_SUBCLASS = 104
DIV_BLOCK = 0
DIV_OFFSET = 14  # 0x0E


def _checksum(block: bytes) -> int:
    return 255 - (sum(block) % 256)


def _read_block(bus: SMBus, subclass: int, block_index: int) -> tuple[bytes, int]:
    bus.write_byte_data(BQ_ADDR, REG_BLOCKDATA_CTRL, 0x00)
    bus.write_byte_data(BQ_ADDR, REG_DF_CLASS, subclass)
    bus.write_byte_data(BQ_ADDR, REG_DF_BLOCK, block_index & 0xFF)
    block = bytes(bus.read_i2c_block_data(BQ_ADDR, REG_BLOCKDATA_START, 32))
    cksum = bus.read_byte_data(BQ_ADDR, REG_BLOCKDATA_CKSUM)
    return block, cksum


def _write_block(bus: SMBus, subclass: int, block_index: int, new_block: bytes) -> None:
    bus.write_byte_data(BQ_ADDR, REG_BLOCKDATA_CTRL, 0x00)
    bus.write_byte_data(BQ_ADDR, REG_DF_CLASS, subclass)
    bus.write_byte_data(BQ_ADDR, REG_DF_BLOCK, block_index & 0xFF)
    bus.write_i2c_block_data(BQ_ADDR, REG_BLOCKDATA_START, list(new_block))
    bus.write_byte_data(BQ_ADDR, REG_BLOCKDATA_CKSUM, _checksum(new_block))


def _u16_le(b: bytes, offset: int) -> int:
    return b[offset] | (b[offset + 1] << 8)


def _set_u16_le(b: bytearray, offset: int, value: int) -> None:
    b[offset] = value & 0xFF
    b[offset + 1] = (value >> 8) & 0xFF


def main() -> None:
    ap = argparse.ArgumentParser(description="Set BQ34Z100 voltage divider ratio (ratio*1000).")
    ap.add_argument("--bus", type=int, default=I2C_BUS_DEFAULT)
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=BQ_ADDR_DEFAULT)
    ap.add_argument("--value", type=int, default=16091, help="Divider ratio * 1000 (default 16091).")
    ap.add_argument("--write", action="store_true", help="Apply changes to the device.")
    args = ap.parse_args()

    global BQ_ADDR
    BQ_ADDR = args.addr

    with SMBus(args.bus) as bus:
        block, cksum = _read_block(bus, DIV_SUBCLASS, DIV_BLOCK)
        cur = _u16_le(block, DIV_OFFSET)
        print(f"Current divider (ratio*1000): {cur}")
        print(f"Block checksum: 0x{cksum:02X}")

        if cur == args.value:
            print("No change needed.")
            return

        new_block = bytearray(block)
        _set_u16_le(new_block, DIV_OFFSET, args.value)

        print(f"Planned update: {cur} -> {args.value}")
        print(f"Bytes @ offset 0x{DIV_OFFSET:02X}: {block[DIV_OFFSET]:02X} {block[DIV_OFFSET+1]:02X} -> "
              f"{new_block[DIV_OFFSET]:02X} {new_block[DIV_OFFSET+1]:02X}")

        if not args.write:
            print("Dry-run only. Re-run with --write to apply.")
            return

        _write_block(bus, DIV_SUBCLASS, DIV_BLOCK, bytes(new_block))
        verify_block, _ = _read_block(bus, DIV_SUBCLASS, DIV_BLOCK)
        verify = _u16_le(verify_block, DIV_OFFSET)
        if verify != args.value:
            print(f"Verify failed: read back {verify}")
            sys.exit(1)
        print("Write complete and verified.")


if __name__ == "__main__":
    main()
