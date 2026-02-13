#!/usr/bin/env python3
"""
bq34z100_df_dump.py (READ ONLY)

Attempts to read BQ34Z100(-R2) Data Flash blocks over SMBus/I2C.

Notes:
- Data Flash access via 0x3E/0x3F is not available in SEALED mode. :contentReference[oaicite:2]{index=2}
- This script does NOT write Data Flash; it only reads and prints blocks.

Usage:
  python checks/nick/fuel_gauge/bq34z100_df_dump.py
  python checks/nick/fuel_gauge/bq34z100_df_dump.py --class 0x52 --block 0
"""

from __future__ import annotations

import argparse
import time
from smbus2 import SMBus

BUS_DEFAULT = 1
ADDR_DEFAULT = 0x55

# Extended commands
CMD_CONTROL = 0x00

CMD_DF_CLASS = 0x3E
CMD_DF_BLOCK = 0x3F
CMD_BLOCKDATA_START = 0x40  # 0x40..0x5F
CMD_BLOCKDATA_CKSUM = 0x60
CMD_BLOCKDATA_CTRL = 0x61

# Control subcommands commonly used
SUBCMD_CONTROL_STATUS = 0x0000

def parse_int_auto(s: str) -> int:
    return int(s, 0)

def sleep_short():
    time.sleep(0.002)

def read_word_le(bus: SMBus, addr: int, cmd: int) -> int:
    bus.write_byte(addr, cmd)
    sleep_short()
    b0, b1 = bus.read_i2c_block_data(addr, cmd, 2)
    return b0 | (b1 << 8)

def write_word_le(bus: SMBus, addr: int, cmd: int, value: int) -> None:
    # Write command + 2 bytes little-endian
    b0 = value & 0xFF
    b1 = (value >> 8) & 0xFF
    bus.write_i2c_block_data(addr, cmd, [b0, b1])

def write_byte(bus: SMBus, addr: int, cmd: int, value: int) -> None:
    bus.write_i2c_block_data(addr, cmd, [value & 0xFF])

def read_block32(bus: SMBus, addr: int) -> list[int]:
    # Read 32 bytes from 0x40..0x5F
    # Some SMBus stacks limit block sizes; smbus2 handles it fine on Pi.
    return bus.read_i2c_block_data(addr, CMD_BLOCKDATA_START, 32)

def fmt_bytes(bs: list[int]) -> str:
    return " ".join(f"{b:02X}" for b in bs)

def get_control_status(bus: SMBus, addr: int) -> int:
    # Write subcommand to Control() then read it back from Control()
    write_word_le(bus, addr, CMD_CONTROL, SUBCMD_CONTROL_STATUS)
    sleep_short()
    return read_word_le(bus, addr, CMD_CONTROL)

def try_read_df_block(bus: SMBus, addr: int, df_class: int, df_block: int) -> tuple[bool, list[int] | None, str | None]:
    try:
        # Enable block data control
        write_byte(bus, addr, CMD_BLOCKDATA_CTRL, 0x00)
        sleep_short()

        # Select class + block
        write_byte(bus, addr, CMD_DF_CLASS, df_class)
        sleep_short()
        write_byte(bus, addr, CMD_DF_BLOCK, df_block)
        sleep_short()

        # Read 32 bytes window
        data = read_block32(bus, addr)
        return True, data, None
    except OSError as e:
        return False, None, str(e)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bus", type=int, default=BUS_DEFAULT)
    ap.add_argument("--addr", type=parse_int_auto, default=ADDR_DEFAULT)
    ap.add_argument("--class", dest="df_class", type=parse_int_auto, default=None, help="DataFlashClass (e.g. 0x52)")
    ap.add_argument("--block", dest="df_block", type=int, default=0, help="DataFlashBlock number (0..)")
    args = ap.parse_args()

    with SMBus(args.bus) as bus:
        cs = get_control_status(bus, args.addr)
        print(f"CONTROL_STATUS: 0x{cs:04X}")
        print("Note: If sealed, DF reads via 0x3E/0x3F will fail (that's expected).")

        if args.df_class is not None:
            ok, data, err = try_read_df_block(bus, args.addr, args.df_class, args.df_block)
            if ok and data is not None:
                print(f"DF Class 0x{args.df_class:02X} Block {args.df_block}:")
                print(fmt_bytes(data))
            else:
                print(f"DF Class 0x{args.df_class:02X} Block {args.df_block}: ERROR {err}")
            return

        # If no class specified, try a small set of likely classes.
        # We will identify the correct one from your output.
        candidates = [
            0x40,  # often Pack Config (varies by variant) :contentReference[oaicite:3]{index=3}
            0x52,  # calibration-ish candidates (device/variant-dependent)
            0x53,
            0x54,
            0x55,
        ]

        for c in candidates:
            ok, data, err = try_read_df_block(bus, args.addr, c, 0)
            if ok and data is not None:
                print(f"DF Class 0x{c:02X} Block 0: {fmt_bytes(data)}")
            else:
                print(f"DF Class 0x{c:02X} Block 0: ERROR {err}")

if __name__ == "__main__":
    main()
