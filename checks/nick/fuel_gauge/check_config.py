#!/usr/bin/env python3
# Read key configuration parameters from BQ34Z100 data flash.

from __future__ import annotations

import argparse
from smbus2 import SMBus

I2C_BUS_DEFAULT = 1
BQ_ADDR_DEFAULT = 0x55

# Data Flash registers
REG_DF_CLASS = 0x3E
REG_DF_BLOCK = 0x3F
REG_BLOCKDATA_START = 0x40
REG_BLOCKDATA_CKSUM = 0x60
REG_BLOCKDATA_CTRL = 0x61

# Subclasses
DESIGN_SUBCLASS = 48
PACK_SUBCLASS = 64
CAL_SUBCLASS = 104


def _read_block(bus: SMBus, subclass: int, block_index: int) -> bytes:
    bus.write_byte_data(BQ_ADDR, REG_BLOCKDATA_CTRL, 0x00)
    bus.write_byte_data(BQ_ADDR, REG_DF_CLASS, subclass)
    bus.write_byte_data(BQ_ADDR, REG_DF_BLOCK, block_index & 0xFF)
    return bytes(bus.read_i2c_block_data(BQ_ADDR, REG_BLOCKDATA_START, 32))


def _df_read_bytes(bus: SMBus, subclass: int, offset: int, length: int) -> bytes:
    block_index = offset // 32
    block = _read_block(bus, subclass, block_index)
    start = offset % 32
    return block[start:start + length]


def _u16_be(b: bytes) -> int:
    return (b[0] << 8) | b[1]


def _u16_le(b: bytes) -> int:
    return b[0] | (b[1] << 8)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check BQ34Z100 key config values.")
    ap.add_argument("--bus", type=int, default=I2C_BUS_DEFAULT)
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=BQ_ADDR_DEFAULT)
    args = ap.parse_args()

    global BQ_ADDR
    BQ_ADDR = args.addr

    with SMBus(args.bus) as bus:
        design_cap = _u16_be(_df_read_bytes(bus, DESIGN_SUBCLASS, 11, 2))
        design_energy = _u16_be(_df_read_bytes(bus, DESIGN_SUBCLASS, 13, 2))
        chg_v_t1 = _u16_be(_df_read_bytes(bus, DESIGN_SUBCLASS, 16, 2))
        chg_v_t2 = _u16_be(_df_read_bytes(bus, DESIGN_SUBCLASS, 18, 2))
        chg_v_t3 = _u16_be(_df_read_bytes(bus, DESIGN_SUBCLASS, 20, 2))

        series_cells = _df_read_bytes(bus, PACK_SUBCLASS, 7, 1)[0]
        pack_cfg = _u16_be(_df_read_bytes(bus, PACK_SUBCLASS, 0, 2))

        vdiv = _u16_le(_df_read_bytes(bus, CAL_SUBCLASS, 14, 2))

    print("Design Data (subclass 48):")
    print(f"  Design Capacity: {design_cap} mAh")
    print(f"  Design Energy: {design_energy} cWh")
    print(f"  Cell Charge Voltage T1-T2: {chg_v_t1} mV")
    print(f"  Cell Charge Voltage T2-T3: {chg_v_t2} mV")
    print(f"  Cell Charge Voltage T3-T4: {chg_v_t3} mV")
    print("Pack Data (subclass 64):")
    print(f"  Series Cells: {series_cells}")
    print(f"  Pack Config: 0x{pack_cfg:04X}")
    print("Calibration Data (subclass 104):")
    print(f"  Voltage Divider (ratio*1000): {vdiv}")


if __name__ == "__main__":
    main()
