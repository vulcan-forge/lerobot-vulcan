#!/usr/bin/env python3
# Manual tweak of BQ34Z100 current gain by scaling CC Gain/CC Delta.
# This is for fine-tuning after offset calibration.
# Dry-run by default. Use --write to apply.

from __future__ import annotations

import argparse
import sys
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

# Calibration Data subclass (CC Gain/Delta)
CAL_SUBCLASS = 104
CAL_BLOCK = 0
CC_GAIN_OFFSET = 0   # F4
CC_DELTA_OFFSET = 4  # F4


def _checksum(block: bytes) -> int:
    return 255 - (sum(block) % 256)


def _read_block(bus: SMBus, subclass: int, block_index: int) -> bytes:
    bus.write_byte_data(BQ_ADDR, REG_BLOCKDATA_CTRL, 0x00)
    bus.write_byte_data(BQ_ADDR, REG_DF_CLASS, subclass)
    bus.write_byte_data(BQ_ADDR, REG_DF_BLOCK, block_index & 0xFF)
    return bytes(bus.read_i2c_block_data(BQ_ADDR, REG_BLOCKDATA_START, 32))


def _write_block(bus: SMBus, subclass: int, block_index: int, new_block: bytes) -> None:
    bus.write_byte_data(BQ_ADDR, REG_BLOCKDATA_CTRL, 0x00)
    bus.write_byte_data(BQ_ADDR, REG_DF_CLASS, subclass)
    bus.write_byte_data(BQ_ADDR, REG_DF_BLOCK, block_index & 0xFF)
    bus.write_i2c_block_data(BQ_ADDR, REG_BLOCKDATA_START, list(new_block))
    bus.write_byte_data(BQ_ADDR, REG_BLOCKDATA_CKSUM, _checksum(new_block))


def _ti_f4_decode(raw4: bytes) -> float:
    r0, r1, r2, r3 = raw4
    exp = int(r0) - 128
    neg = (r1 & 0x80) != 0
    byte2 = r1 & 0x7F
    frac = float(byte2) + (float(r2) / 256.0) + (float(r3) / 65536.0)
    p = 8 - exp
    mag = (frac + 128.0) / (2.0 ** p)
    return -mag if neg else mag


def _ti_f4_encode(value: float) -> bytes:
    if value == 0.0:
        return bytes([0x80, 0x00, 0x00, 0x00])
    val = float(value)
    mod_val = -val if val < 0 else val
    exp = 0
    tmp = mod_val * (1.0 + (2.0 ** -25))
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
    byte2 = int(tmp)
    tmp = (2.0 ** 8) * (tmp - float(byte2))
    byte1 = int(tmp)
    tmp = (2.0 ** 8) * (tmp - float(byte1))
    byte0 = int(tmp)
    r0 = (exp + 128) & 0xFF
    r1 = byte2 & 0x7F
    if val < 0:
        r1 |= 0x80
    return bytes([r0, r1, byte1 & 0xFF, byte0 & 0xFF])


def main() -> None:
    ap = argparse.ArgumentParser(description="Tweak CC Gain/Delta by a scaling factor.")
    ap.add_argument("--bus", type=int, default=I2C_BUS_DEFAULT)
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=BQ_ADDR_DEFAULT)
    ap.add_argument("--factor", type=float, help="Multiply CC Gain/Delta by this factor.")
    ap.add_argument("--measured-ma", type=float, help="Measured current in mA.")
    ap.add_argument("--reported-ma", type=float, help="Gauge-reported current in mA.")
    ap.add_argument("--write", action="store_true", help="Apply changes to the device.")
    args = ap.parse_args()

    if args.factor is None:
        if args.measured_ma is None or args.reported_ma is None or args.reported_ma == 0:
            ap.error("Provide --factor OR both --measured-ma and --reported-ma.")
        args.factor = args.measured_ma / args.reported_ma

    global BQ_ADDR
    BQ_ADDR = args.addr

    with SMBus(args.bus) as bus:
        block = _read_block(bus, CAL_SUBCLASS, CAL_BLOCK)
        cc_gain_raw = block[CC_GAIN_OFFSET:CC_GAIN_OFFSET + 4]
        cc_delta_raw = block[CC_DELTA_OFFSET:CC_DELTA_OFFSET + 4]

        cc_gain = _ti_f4_decode(cc_gain_raw)
        cc_delta = _ti_f4_decode(cc_delta_raw)

        new_gain = cc_gain * args.factor
        new_delta = cc_delta * args.factor

        print(f"Current CC Gain: {cc_gain:.6g}")
        print(f"Current CC Delta: {cc_delta:.6g}")
        print(f"Scale factor: {args.factor:.6g}")
        print(f"New CC Gain: {new_gain:.6g}")
        print(f"New CC Delta: {new_delta:.6g}")

        new_block = bytearray(block)
        new_block[CC_GAIN_OFFSET:CC_GAIN_OFFSET + 4] = _ti_f4_encode(new_gain)
        new_block[CC_DELTA_OFFSET:CC_DELTA_OFFSET + 4] = _ti_f4_encode(new_delta)

        print(
            f"CC Gain bytes: {cc_gain_raw.hex()} -> "
            f"{new_block[CC_GAIN_OFFSET:CC_GAIN_OFFSET+4].hex()}"
        )
        print(
            f"CC Delta bytes: {cc_delta_raw.hex()} -> "
            f"{new_block[CC_DELTA_OFFSET:CC_DELTA_OFFSET+4].hex()}"
        )

        if not args.write:
            print("Dry-run only. Re-run with --write to apply.")
            return

        _write_block(bus, CAL_SUBCLASS, CAL_BLOCK, bytes(new_block))
        time.sleep(0.05)
        verify = _read_block(bus, CAL_SUBCLASS, CAL_BLOCK)
        if verify[CC_GAIN_OFFSET:CC_GAIN_OFFSET + 4] != new_block[CC_GAIN_OFFSET:CC_GAIN_OFFSET + 4]:
            print("Verify failed: CC Gain bytes differ.")
            sys.exit(1)
        if verify[CC_DELTA_OFFSET:CC_DELTA_OFFSET + 4] != new_block[CC_DELTA_OFFSET:CC_DELTA_OFFSET + 4]:
            print("Verify failed: CC Delta bytes differ.")
            sys.exit(1)

    print("Tweak complete.")


if __name__ == "__main__":
    main()
