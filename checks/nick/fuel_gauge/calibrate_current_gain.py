#!/usr/bin/env python3
# Calibrate BQ34Z100 current gain by updating CC Gain/CC Delta in DF.
# Procedure based on TI guidance:
#   - Enable calibration (CAL_ENABLE), ENTER_CAL
#   - Read CC Offset (I2) and Board Offset (I1) from subclass 104
#   - Read AverageCurrent (0x0A) as signed raw
#   - Compute:
#       cc_gain = measured_current_mA / (avg_raw - (cc_offset + board_offset)/16)
#       cc_delta = cc_gain * 1193046.0
#   - Write CC Gain (F4) at offset 0, CC Delta (F4) at offset 4
#   - EXIT_CAL and CAL_ENABLE
#
# Dry-run by default. Use --write to apply.

from __future__ import annotations

import argparse
import sys
import time
import math
from smbus2 import SMBus, i2c_msg

I2C_BUS_DEFAULT = 1
BQ_ADDR_DEFAULT = 0x55

# Control() register and subcommands
CMD_CONTROL = 0x00
SUB_CAL_ENABLE = 0x002D
SUB_ENTER_CAL = 0x0081
SUB_EXIT_CAL = 0x0080

# Data Flash registers
REG_DF_CLASS = 0x3E
REG_DF_BLOCK = 0x3F
REG_BLOCKDATA_START = 0x40
REG_BLOCKDATA_CKSUM = 0x60
REG_BLOCKDATA_CTRL = 0x61

# Calibration Data subclass (CC Gain/Delta/Offsets)
CAL_SUBCLASS = 104
CAL_BLOCK = 0
CC_GAIN_OFFSET = 0   # F4
CC_DELTA_OFFSET = 4  # F4
CC_OFFSET_OFFSET = 8  # I2
BOARD_OFFSET_OFFSET = 10  # I1

CMD_AVG_CURRENT = 0x0A


def _checksum(block: bytes) -> int:
    return 255 - (sum(block) % 256)


def _write_control(bus: SMBus, subcmd: int) -> None:
    lo = subcmd & 0xFF
    hi = (subcmd >> 8) & 0xFF
    write = i2c_msg.write(BQ_ADDR, [CMD_CONTROL, lo, hi])
    bus.i2c_rdwr(write)


def _read_avg_current_raw(bus: SMBus) -> int:
    write = i2c_msg.write(BQ_ADDR, [CMD_AVG_CURRENT])
    read = i2c_msg.read(BQ_ADDR, 2)
    bus.i2c_rdwr(write, read)
    b = list(read)
    raw = b[0] | (b[1] << 8)
    return raw - 0x10000 if raw & 0x8000 else raw


def _df_read_block(bus: SMBus, subclass: int, block_index: int) -> bytes:
    bus.write_byte_data(BQ_ADDR, REG_BLOCKDATA_CTRL, 0x00)
    bus.write_byte_data(BQ_ADDR, REG_DF_CLASS, subclass)
    bus.write_byte_data(BQ_ADDR, REG_DF_BLOCK, block_index & 0xFF)
    return bytes(bus.read_i2c_block_data(BQ_ADDR, REG_BLOCKDATA_START, 32))


def _df_write_block(bus: SMBus, subclass: int, block_index: int, new_block: bytes) -> None:
    bus.write_byte_data(BQ_ADDR, REG_BLOCKDATA_CTRL, 0x00)
    bus.write_byte_data(BQ_ADDR, REG_DF_CLASS, subclass)
    bus.write_byte_data(BQ_ADDR, REG_DF_BLOCK, block_index & 0xFF)
    bus.write_i2c_block_data(BQ_ADDR, REG_BLOCKDATA_START, list(new_block))
    bus.write_byte_data(BQ_ADDR, REG_BLOCKDATA_CKSUM, _checksum(new_block))


def _s16_be(b: bytes) -> int:
    v = (b[0] << 8) | b[1]
    return v - 0x10000 if v & 0x8000 else v


def _s8(b: int) -> int:
    return b - 256 if b & 0x80 else b


def _ti_f4_encode(value: float) -> bytes:
    # TI Host Calibration F4 format (from TI app note SLUA640B)
    if value == 0.0:
        return bytes([0x80, 0x00, 0x00, 0x00])

    val = float(value)
    mod_val = -val if val < 0 else val

    exp = 0
    tmp = mod_val
    tmp = tmp * (1.0 + (2.0 ** -25))

    if tmp < 0.5:
        while tmp < 0.5:
            tmp *= 2.0
            exp -= 1
    else:
        while tmp >= 1.0:
            tmp /= 2.0
            exp += 1

    exp = max(-128, min(127, exp))
    tmp = (2.0 ** (8 - exp)) * mod_val - 128.0
    byte2 = int(math.floor(tmp))
    tmp = (2.0 ** 8) * (tmp - float(byte2))
    byte1 = int(math.floor(tmp))
    tmp = (2.0 ** 8) * (tmp - float(byte1))
    byte0 = int(math.floor(tmp))

    r0 = (exp + 128) & 0xFF
    r1 = byte2 & 0x7F
    if val < 0:
        r1 |= 0x80
    return bytes([r0, r1, byte1 & 0xFF, byte0 & 0xFF])


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate BQ34Z100 current gain (CC Gain/Delta).")
    ap.add_argument("--bus", type=int, default=I2C_BUS_DEFAULT)
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=BQ_ADDR_DEFAULT)
    ap.add_argument("--measured-ma", type=float, required=True, help="Measured current in mA (signed).")
    ap.add_argument("--samples", type=int, default=10, help="Average current samples.")
    ap.add_argument("--delay", type=float, default=0.1, help="Delay between samples (s).")
    ap.add_argument("--write", action="store_true", help="Apply changes to the device.")
    ap.add_argument("--no-verify", action="store_true", help="Skip read-back verification.")
    args = ap.parse_args()

    global BQ_ADDR
    BQ_ADDR = args.addr

    with SMBus(args.bus) as bus:
        # Enter calibration mode
        _write_control(bus, SUB_CAL_ENABLE)
        _write_control(bus, SUB_ENTER_CAL)

        block = _df_read_block(bus, CAL_SUBCLASS, CAL_BLOCK)
        cc_offset = _s16_be(block[CC_OFFSET_OFFSET:CC_OFFSET_OFFSET + 2])
        board_offset = _s8(block[BOARD_OFFSET_OFFSET])

        # Average raw current
        raws = []
        for _ in range(args.samples):
            raws.append(_read_avg_current_raw(bus))
            time.sleep(args.delay)
        avg_raw = sum(raws) / len(raws)

        denom = avg_raw - (cc_offset + board_offset) / 16.0
        if denom == 0:
            print("Invalid denominator (zero). Aborting.")
            return

        cc_gain = args.measured_ma / denom
        cc_delta = cc_gain * 1193046.0

        print(f"CC Offset: {cc_offset}  Board Offset: {board_offset}")
        print(f"Avg Raw Current: {avg_raw:.2f}")
        print(f"Computed CC Gain: {cc_gain:.6g}")
        print(f"Computed CC Delta: {cc_delta:.6g}")

        new_block = bytearray(block)
        new_block[CC_GAIN_OFFSET:CC_GAIN_OFFSET + 4] = _ti_f4_encode(cc_gain)
        new_block[CC_DELTA_OFFSET:CC_DELTA_OFFSET + 4] = _ti_f4_encode(cc_delta)

        print(
            f"CC Gain bytes: {block[CC_GAIN_OFFSET:CC_GAIN_OFFSET+4].hex()} -> "
            f"{new_block[CC_GAIN_OFFSET:CC_GAIN_OFFSET+4].hex()}"
        )
        print(
            f"CC Delta bytes: {block[CC_DELTA_OFFSET:CC_DELTA_OFFSET+4].hex()} -> "
            f"{new_block[CC_DELTA_OFFSET:CC_DELTA_OFFSET+4].hex()}"
        )

        if not args.write:
            print("Dry-run only. Re-run with --write to apply.")
            return

        _df_write_block(bus, CAL_SUBCLASS, CAL_BLOCK, bytes(new_block))
        if not args.no_verify:
            time.sleep(0.05)
            verify = _df_read_block(bus, CAL_SUBCLASS, CAL_BLOCK)
            if verify[CC_GAIN_OFFSET:CC_GAIN_OFFSET + 4] != new_block[CC_GAIN_OFFSET:CC_GAIN_OFFSET + 4]:
                print("Verify failed: CC Gain bytes differ.")
                sys.exit(1)
            if verify[CC_DELTA_OFFSET:CC_DELTA_OFFSET + 4] != new_block[CC_DELTA_OFFSET:CC_DELTA_OFFSET + 4]:
                print("Verify failed: CC Delta bytes differ.")
                sys.exit(1)

        # Exit calibration mode
        _write_control(bus, SUB_EXIT_CAL)
        _write_control(bus, SUB_CAL_ENABLE)

    print("Current gain calibration complete.")


if __name__ == "__main__":
    main()
