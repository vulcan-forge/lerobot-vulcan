#!/usr/bin/env python3
# Read charge-related metrics from BQ34Z100 over I2C.
# Uses repeated-start reads to avoid byte-swap issues.

from __future__ import annotations

import argparse
from smbus2 import SMBus, i2c_msg

I2C_BUS_DEFAULT = 1
BQ_ADDR_DEFAULT = 0x55

# Divider (board values)
R_TOP_OHMS = 249_000.0
R_BOTTOM_OHMS = 16_500.0
V_DIV_RATIO = R_BOTTOM_OHMS / (R_TOP_OHMS + R_BOTTOM_OHMS)

# Data Flash registers (for optional calibration dump)
REG_DF_CLASS = 0x3E
REG_DF_BLOCK = 0x3F
REG_BLOCKDATA_START = 0x40
REG_BLOCKDATA_CKSUM = 0x60
REG_BLOCKDATA_CTRL = 0x61

# Calibration Data subclass (TI datasheet)
CAL_SUBCLASS = 104

# Design Data subclass (contains Design Capacity)
DESIGN_SUBCLASS = 48

# Standard commands (SBS command map for BQ34Z100)
CMD_TEMPERATURE = 0x08
CMD_VOLTAGE = 0x09
CMD_CURRENT = 0x0A
CMD_AVG_CURRENT = 0x0B
CMD_SOC = 0x0D
CMD_REMAINING = 0x0F
CMD_FULL = 0x10
CMD_FLAGS = 0x16


def _read_word(bus: SMBus, cmd: int) -> int:
    write = i2c_msg.write(BQ_ADDR, [cmd])
    read = i2c_msg.read(BQ_ADDR, 2)
    bus.i2c_rdwr(write, read)
    b = list(read)
    return b[0] | (b[1] << 8)


def _s16(x: int) -> int:
    return x - 0x10000 if x & 0x8000 else x


def _df_read_block(bus: SMBus, subclass: int, block_index: int) -> bytes:
    bus.write_byte_data(BQ_ADDR, REG_BLOCKDATA_CTRL, 0x00)
    bus.write_byte_data(BQ_ADDR, REG_DF_CLASS, subclass)
    bus.write_byte_data(BQ_ADDR, REG_DF_BLOCK, block_index & 0xFF)
    return bytes(bus.read_i2c_block_data(BQ_ADDR, REG_BLOCKDATA_START, 32))


def _df_read_bytes(bus: SMBus, subclass: int, offset: int, length: int) -> bytes:
    block_index = offset // 32
    block = _df_read_block(bus, subclass, block_index)
    start = offset % 32
    return block[start:start + length]


def _ti_f4_decode(raw4: bytes) -> float:
    # TI "F4" format used in Calibration Data (see TI docs).
    r0, r1, r2, r3 = raw4
    exp = int(r0) - 128
    neg = (r1 & 0x80) != 0
    byte2 = r1 & 0x7F
    frac = float(byte2) + (float(r2) / 256.0) + (float(r3) / 65536.0)
    p = 8 - exp
    mag = (frac + 128.0) / (2.0 ** p)
    return -mag if neg else mag


def main() -> None:
    ap = argparse.ArgumentParser(description="Read charge metrics from BQ34Z100.")
    ap.add_argument("--bus", type=int, default=I2C_BUS_DEFAULT)
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=BQ_ADDR_DEFAULT)
    ap.add_argument("--no-temp", action="store_true", help="Skip temperature read/output.")
    ap.add_argument("--cal", action="store_true", help="Dump calibration fields (subclass 104).")
    args = ap.parse_args()

    global BQ_ADDR
    BQ_ADDR = args.addr

    with SMBus(args.bus) as bus:
        voltage_mv = _read_word(bus, CMD_VOLTAGE)
        temp_dK = None if args.no_temp else _read_word(bus, CMD_TEMPERATURE)
        curr_ma = _s16(_read_word(bus, CMD_CURRENT))
        avg_ma = _s16(_read_word(bus, CMD_AVG_CURRENT))
        soc = _read_word(bus, CMD_SOC)
        rem_mah = _read_word(bus, CMD_REMAINING)
        full_mah = _read_word(bus, CMD_FULL)
        try:
            flags = _read_word(bus, CMD_FLAGS)
        except OSError:
            flags = None

        design_cap_raw = _df_read_bytes(bus, DESIGN_SUBCLASS, 11, 2)
        design_cap = int.from_bytes(design_cap_raw, "big", signed=False)

        if args.cal:
            cc_gain_raw = _df_read_bytes(bus, CAL_SUBCLASS, 0, 4)
            cc_delta_raw = _df_read_bytes(bus, CAL_SUBCLASS, 4, 4)
            cc_offset_raw = _df_read_bytes(bus, CAL_SUBCLASS, 8, 2)
            board_offset_raw = _df_read_bytes(bus, CAL_SUBCLASS, 10, 1)
            vdiv_raw = _df_read_bytes(bus, CAL_SUBCLASS, 14, 2)

            cc_gain = _ti_f4_decode(cc_gain_raw)
            cc_delta = _ti_f4_decode(cc_delta_raw)
            cc_offset = int.from_bytes(cc_offset_raw, "big", signed=True)
            board_offset = int.from_bytes(board_offset_raw, "big", signed=True)
            vdiv = int.from_bytes(vdiv_raw, "little", signed=False)

    # If the voltage looks like BAT (< 2 V), estimate pack using divider.
    if voltage_mv < 2000:
        pack_v = (voltage_mv / 1000.0) / V_DIV_RATIO
        print(f"Voltage: {voltage_mv} mV (BAT) -> Pack ~ {pack_v:.2f} V")
    else:
        print(f"Voltage: {voltage_mv} mV (PACK)")
    if temp_dK is not None:
        print(f"Temperature: {temp_dK/10.0 - 273.15:.1f} C ({temp_dK} in 0.1K)")
    print(f"Current: {curr_ma} mA")
    print(f"Avg Current: {avg_ma} mA")
    print(f"SOC: {soc} %")
    print(f"Remaining Capacity: {rem_mah} mAh")
    print(f"Full Charge Capacity: {full_mah} mAh")
    print(f"Design Capacity (DF): {design_cap} mAh")
    if flags is None:
        print("Flags: <read failed>")
    else:
        print(f"Flags: 0x{flags:04X}")

    if args.cal:
        print("Calibration (DF subclass 104):")
        print(f"  CC Gain (F4): {cc_gain:.6g}")
        print(f"  CC Delta (F4): {cc_delta:.6g}")
        print(f"  CC Offset (I2): {cc_offset}")
        print(f"  Board Offset (I1): {board_offset}")
        print(f"  Voltage Divider (U2): {vdiv}")


if __name__ == "__main__":
    main()
