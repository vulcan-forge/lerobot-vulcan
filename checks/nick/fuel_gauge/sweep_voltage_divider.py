#!/usr/bin/env python3
# Sweep the BQ34Z100 Voltage Divider value to match a target BAT or PACK voltage.
# WARNING: This writes Data Flash repeatedly; use sparingly.

from __future__ import annotations

import argparse
import statistics
import time
from smbus2 import SMBus, i2c_msg

I2C_BUS_DEFAULT = 1
BQ_ADDR_DEFAULT = 0x55

# Standard commands
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
DIV_OFFSET = 14

# Pack config to detect VOLTSEL
PACK_SUBCLASS = 64
PACK_BLOCK = 0
PACK_CFG_OFFSET = 0
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


def _u16_le(b: bytes, offset: int) -> int:
    return b[offset] | (b[offset + 1] << 8)


def _set_u16_le(b: bytearray, offset: int, value: int) -> None:
    b[offset] = value & 0xFF
    b[offset + 1] = (value >> 8) & 0xFF


def _read_voltage_mv(bus: SMBus) -> int | None:
    write = i2c_msg.write(BQ_ADDR, [CMD_VOLTAGE])
    read = i2c_msg.read(BQ_ADDR, 2)
    bus.i2c_rdwr(write, read)
    b = list(read)
    raw = b[0] | (b[1] << 8)
    if raw == 0xFFFF:
        return None
    return raw


def _read_voltage_median(bus: SMBus, samples: int, delay_s: float) -> int | None:
    vals: list[int] = []
    for _ in range(samples):
        v = _read_voltage_mv(bus)
        if v is not None:
            vals.append(v)
        time.sleep(delay_s)
    if not vals:
        return None
    return int(statistics.median(vals))


def _read_pack_cfg(bus: SMBus) -> int:
    block = _read_block(bus, PACK_SUBCLASS, PACK_BLOCK)
    return (block[PACK_CFG_OFFSET] << 8) | block[PACK_CFG_OFFSET + 1]


def _write_control(bus: SMBus, subcmd: int) -> None:
    lo = subcmd & 0xFF
    hi = (subcmd >> 8) & 0xFF
    bus.write_i2c_block_data(BQ_ADDR, CMD_CONTROL, [lo, hi])


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep Voltage Divider to match target BAT voltage.")
    ap.add_argument("--bus", type=int, default=I2C_BUS_DEFAULT)
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=BQ_ADDR_DEFAULT)
    ap.add_argument("--start", type=int, default=None, help="Starting divider value (default: current).")
    ap.add_argument("--step", type=int, default=1, help="Step size per iteration.")
    ap.add_argument("--min", dest="min_value", type=int, default=0, help="Minimum divider value.")
    ap.add_argument("--max-steps", type=int, default=500, help="Maximum steps to try.")
    ap.add_argument("--target-bat", type=float, default=None, help="Target BAT voltage in volts (e.g. 0.808).")
    ap.add_argument("--target-pack", type=float, default=None, help="Target PACK voltage in volts (e.g. 13.0).")
    ap.add_argument("--tolerance-mv", type=int, default=5, help="Acceptable BAT error in mV.")
    ap.add_argument("--samples", type=int, default=3, help="Samples per step (median).")
    ap.add_argument("--sample-delay", type=float, default=0.02, help="Delay between samples in seconds.")
    ap.add_argument("--settle", type=float, default=0.05, help="Delay after write before sampling.")
    ap.add_argument("--direction", choices=["down", "up"], default="down", help="Sweep direction.")
    ap.add_argument("--cfgupdate", action="store_true", help="Toggle CAL_ENABLE around writes.")
    ap.add_argument("--reset", action="store_true", help="Send RESET after sweep.")
    ap.add_argument("--force", action="store_true", help="Proceed even if VOLTSEL is set.")
    ap.add_argument(
        "--clear-voltsel",
        action="store_true",
        help="Temporarily clear VOLTSEL (PACK->BAT) before sweep.",
    )
    ap.add_argument(
        "--restore-voltsel",
        action="store_true",
        help="Restore original VOLTSEL state after sweep (only if --clear-voltsel).",
    )
    ap.add_argument("--write", action="store_true", help="Actually write divider values.")
    args = ap.parse_args()

    if (args.target_bat is None) == (args.target_pack is None):
        ap.error("Specify exactly one of --target-bat or --target-pack.")

    global BQ_ADDR
    BQ_ADDR = args.addr

    if args.target_bat is not None:
        target_mv = int(round(args.target_bat * 1000.0))
        target_label = "BAT"
    else:
        target_mv = int(round(args.target_pack * 1000.0))
        target_label = "PACK"

    if not args.write:
        print("Dry-run only. Re-run with --write to apply sweep.")
        return

    with SMBus(args.bus) as bus:
        pack_block = bytearray(_read_block(bus, PACK_SUBCLASS, PACK_BLOCK))
        pack_cfg = (pack_block[PACK_CFG_OFFSET] << 8) | pack_block[PACK_CFG_OFFSET + 1]
        original_pack_cfg = pack_cfg

        if args.clear_voltsel and (pack_cfg & VOLTSEL_MASK):
            if args.cfgupdate:
                _write_control(bus, SUB_CAL_ENABLE)
            pack_cfg &= ~VOLTSEL_MASK
            pack_block[PACK_CFG_OFFSET] = (pack_cfg >> 8) & 0xFF
            pack_block[PACK_CFG_OFFSET + 1] = pack_cfg & 0xFF
            _write_block_diff(bus, PACK_SUBCLASS, PACK_BLOCK, _read_block(bus, PACK_SUBCLASS, PACK_BLOCK), bytes(pack_block))
            time.sleep(0.05)
            print(f"Cleared VOLTSEL: Pack Config now 0x{pack_cfg:04X}")

        if (pack_cfg & VOLTSEL_MASK):
            if target_label == "BAT" and not args.force:
                print(f"Pack Config: 0x{pack_cfg:04X} (VOLTSEL set). 0x08 reports PACK, not BAT.")
                print("Use --clear-voltsel or re-run with --force if you really want to sweep anyway.")
                return
        else:
            if target_label == "PACK" and not args.force:
                print(f"Pack Config: 0x{pack_cfg:04X} (VOLTSEL clear). 0x08 reports BAT, not PACK.")
                print("Set VOLTSEL or re-run with --force if you really want to sweep anyway.")
                return

        block = _read_block(bus, DIV_SUBCLASS, DIV_BLOCK)
        current = _u16_le(block, DIV_OFFSET)
        start = current if args.start is None else args.start

        step = abs(args.step)
        direction = -1 if args.direction == "down" else 1

        print(f"Starting divider: {start} (current {current})")
        print(f"Target {target_label}: {target_mv} mV  tolerance: ±{args.tolerance_mv} mV")

        if args.cfgupdate:
            _write_control(bus, SUB_CAL_ENABLE)

        value = start
        for i in range(args.max_steps):
            # Read current block fresh each iteration to avoid stale data.
            old_block = _read_block(bus, DIV_SUBCLASS, DIV_BLOCK)
            new_block = bytearray(old_block)
            _set_u16_le(new_block, DIV_OFFSET, value)
            _write_block_diff(bus, DIV_SUBCLASS, DIV_BLOCK, old_block, bytes(new_block))

            time.sleep(args.settle)
            mv = _read_voltage_median(bus, args.samples, args.sample_delay)
            if mv is None:
                print(f"[{i}] divider={value} -> read invalid")
            else:
                err = mv - target_mv
                print(f"[{i}] divider={value} -> {target_label}={mv} mV (err {err:+d} mV)")
                if abs(err) <= args.tolerance_mv:
                    print("Target reached.")
                    break

            value += direction * step
            if value < args.min_value:
                print("Reached minimum value.")
                break

        if args.cfgupdate:
            _write_control(bus, SUB_CAL_ENABLE)
        if args.clear_voltsel and args.restore_voltsel and (original_pack_cfg != pack_cfg):
            pack_block = bytearray(_read_block(bus, PACK_SUBCLASS, PACK_BLOCK))
            pack_block[PACK_CFG_OFFSET] = (original_pack_cfg >> 8) & 0xFF
            pack_block[PACK_CFG_OFFSET + 1] = original_pack_cfg & 0xFF
            _write_block_diff(bus, PACK_SUBCLASS, PACK_BLOCK, _read_block(bus, PACK_SUBCLASS, PACK_BLOCK), bytes(pack_block))
            time.sleep(0.05)
            print(f"Restored VOLTSEL: Pack Config back to 0x{original_pack_cfg:04X}")
        if args.reset:
            _write_control(bus, SUB_RESET)
            time.sleep(0.1)


if __name__ == "__main__":
    main()
