#!/usr/bin/env python3
# Per-chip voltage calibration for BQ34Z100 by updating Voltage Divider.
# If the voltage register returns BAT (mV), compute directly:
#   newDivider = (measuredPack / measuredBat) * 1000
# If the voltage register returns PACK (mV), use:
#   newDivider = oldDivider * (measuredPack / rawPack)
#
# Dry-run by default. Use --write to apply.

from __future__ import annotations

import argparse
import sys
import time
from smbus2 import SMBus, i2c_msg

I2C_BUS_DEFAULT = 1
BQ_ADDR_DEFAULT = 0x55
CMD_VOLTAGE = 0x08

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


def _read_voltage_bytes(bus: SMBus) -> tuple[int, int]:
    write = i2c_msg.write(BQ_ADDR, [CMD_VOLTAGE])
    read = i2c_msg.read(BQ_ADDR, 2)
    bus.i2c_rdwr(write, read)
    b = list(read)
    return b[0], b[1]


def _u16_le(b: bytes, offset: int) -> int:
    return b[offset] | (b[offset + 1] << 8)


def _set_u16_le(b: bytearray, offset: int, value: int) -> None:
    b[offset] = value & 0xFF
    b[offset + 1] = (value >> 8) & 0xFF


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-chip voltage calibration by updating Voltage Divider."
    )
    ap.add_argument("--bus", type=int, default=I2C_BUS_DEFAULT)
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=BQ_ADDR_DEFAULT)
    ap.add_argument(
        "--measured-pack",
        type=float,
        required=True,
        help="Measured pack voltage in volts (from a meter).",
    )
    ap.add_argument("--samples", type=int, default=8, help="Number of samples to average.")
    ap.add_argument("--delay", type=float, default=0.1, help="Delay between samples (s).")
    ap.add_argument(
        "--mode",
        choices=["auto", "bat", "pack"],
        default="auto",
        help="Interpret raw voltage as BAT or PACK. 'auto' uses raw<2V as BAT.",
    )
    ap.add_argument("--write", action="store_true", help="Apply changes to the device.")
    args = ap.parse_args()

    global BQ_ADDR
    BQ_ADDR = args.addr

    with SMBus(args.bus) as bus:
        block, cksum = _read_block(bus, DIV_SUBCLASS, DIV_BLOCK)
        cur = _u16_le(block, DIV_OFFSET)
        print(f"Current divider (ratio*1000): {cur}")
        print(f"Block checksum: 0x{cksum:02X}")

        vals = []
        for _ in range(args.samples):
            b0, b1 = _read_voltage_bytes(bus)
            if b0 == 0xFF and b1 == 0xFF:
                time.sleep(args.delay)
                continue
            raw = _u16_le(bytes([b0, b1]), 0)
            vals.append(raw)
            time.sleep(args.delay)

        if not vals:
            print("No valid voltage samples (all 0xFFFF). Aborting.")
            return

        raw_mv = sum(vals) / len(vals)
        raw_v = raw_mv / 1000.0
        print(f"Raw voltage (avg): {raw_mv:.0f} mV")

        # Interpret raw voltage
        if args.mode == "auto":
            mode = "bat" if raw_v < 2.0 else "pack"
        else:
            mode = args.mode

        if mode == "bat":
            if raw_v <= 0.0 or raw_v > 2.0:
                print(f"Raw BAT voltage out of expected range: {raw_v:.3f} V")
                return
            raw_bat = raw_v
            new_div = int(round((args.measured_pack / raw_bat) * 1000.0))
            print(f"Interpreting raw as BAT: {raw_bat:.3f} V -> new divider {new_div}")
        else:
            raw_pack = raw_v
            if raw_pack <= 0.0:
                print("Invalid raw pack voltage.")
                return
            new_div = int(round(cur * (args.measured_pack / raw_pack)))
            print(f"Interpreting raw as PACK: {raw_pack:.3f} V -> new divider {new_div}")

        if new_div < 1 or new_div > 65535:
            print(f"Computed divider out of range: {new_div}")
            return

        new_block = bytearray(block)
        _set_u16_le(new_block, DIV_OFFSET, new_div)

        print(f"Planned update: {cur} -> {new_div}")
        print(
            f"Bytes @ offset 0x{DIV_OFFSET:02X}: {block[DIV_OFFSET]:02X} {block[DIV_OFFSET+1]:02X} -> "
            f"{new_block[DIV_OFFSET]:02X} {new_block[DIV_OFFSET+1]:02X}"
        )

        if not args.write:
            print("Dry-run only. Re-run with --write to apply.")
            return

        _write_block(bus, DIV_SUBCLASS, DIV_BLOCK, bytes(new_block))
        # Read back with a short delay to avoid transient 0s
        time.sleep(0.05)
        verify_block, _ = _read_block(bus, DIV_SUBCLASS, DIV_BLOCK)
        verify = _u16_le(verify_block, DIV_OFFSET)
        if verify != new_div:
            print(f"Verify failed: read back {verify}")
            sys.exit(1)
        print("Write complete and verified.")


if __name__ == "__main__":
    main()
