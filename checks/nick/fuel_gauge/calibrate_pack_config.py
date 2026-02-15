#!/usr/bin/env python3
# Set Design Energy, Cell Charge Voltage(s), and Series Cells for BQ34Z100.
# Design Data subclass 48 (big-endian U2):
#   Design Energy: offset 13
#   Cell Chg V T1-T2: offset 16
#   Cell Chg V T2-T3: offset 18
#   Cell Chg V T3-T4: offset 20
# Pack Data subclass 64 (U1):
#   Series Cells: offset 7
#
# Dry-run by default. Use --write to apply.

from __future__ import annotations

import argparse
import sys
from smbus2 import SMBus

I2C_BUS_DEFAULT = 1
BQ_ADDR_DEFAULT = 0x55

REG_DF_CLASS = 0x3E
REG_DF_BLOCK = 0x3F
REG_BLOCKDATA_START = 0x40
REG_BLOCKDATA_CKSUM = 0x60
REG_BLOCKDATA_CTRL = 0x61

DESIGN_SUBCLASS = 48
PACK_SUBCLASS = 64


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


def _u16_be(b: bytes, offset: int) -> int:
    return (b[offset] << 8) | b[offset + 1]


def _set_u16_be(b: bytearray, offset: int, value: int) -> None:
    b[offset] = (value >> 8) & 0xFF
    b[offset + 1] = value & 0xFF


def main() -> None:
    ap = argparse.ArgumentParser(description="Set design energy, cell charge voltage, and series cells.")
    ap.add_argument("--bus", type=int, default=I2C_BUS_DEFAULT)
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=BQ_ADDR_DEFAULT)
    ap.add_argument("--design-energy", type=int, default=12800, help="Design energy (cWh).")
    ap.add_argument("--cell-charge-mv", type=int, default=3600, help="Cell charge voltage (mV).")
    ap.add_argument("--series-cells", type=int, default=4, help="Number of series cells.")
    ap.add_argument("--write", action="store_true", help="Apply changes to the device.")
    args = ap.parse_args()

    global BQ_ADDR
    BQ_ADDR = args.addr

    with SMBus(args.bus) as bus:
        design_block = bytearray(_read_block(bus, DESIGN_SUBCLASS, 0))
        pack_block = bytearray(_read_block(bus, PACK_SUBCLASS, 0))

        cur_energy = _u16_be(design_block, 13)
        cur_chg1 = _u16_be(design_block, 16)
        cur_chg2 = _u16_be(design_block, 18)
        cur_chg3 = _u16_be(design_block, 20)
        cur_pack_cfg = _u16_be(pack_block, 0)
        cur_series = pack_block[7]

        print("Current values:")
        print(f"  Design Energy: {cur_energy} cWh")
        print(f"  Cell Chg V T1-T2: {cur_chg1} mV")
        print(f"  Cell Chg V T2-T3: {cur_chg2} mV")
        print(f"  Cell Chg V T3-T4: {cur_chg3} mV")
        print(f"  Series Cells: {cur_series}")
        print(f"  Pack Config: 0x{cur_pack_cfg:04X}")

        _set_u16_be(design_block, 13, args.design_energy)
        _set_u16_be(design_block, 16, args.cell_charge_mv)
        _set_u16_be(design_block, 18, args.cell_charge_mv)
        _set_u16_be(design_block, 20, args.cell_charge_mv)
        pack_block[7] = args.series_cells & 0xFF
        # Set VOLTSEL bit (0x0800) so series cells are used.
        pack_cfg = cur_pack_cfg | 0x0800
        _set_u16_be(pack_block, 0, pack_cfg)

        print("Planned updates:")
        print(f"  Design Energy: {cur_energy} -> {args.design_energy} cWh")
        print(f"  Cell Charge Voltage: {cur_chg1}/{cur_chg2}/{cur_chg3} -> {args.cell_charge_mv} mV")
        print(f"  Series Cells: {cur_series} -> {args.series_cells}")
        print(f"  Pack Config: 0x{cur_pack_cfg:04X} -> 0x{pack_cfg:04X}")

        if not args.write:
            print("Dry-run only. Re-run with --write to apply.")
            return

        _write_block(bus, DESIGN_SUBCLASS, 0, bytes(design_block))
        _write_block(bus, PACK_SUBCLASS, 0, bytes(pack_block))

        # Verify
        v_design = _read_block(bus, DESIGN_SUBCLASS, 0)
        v_pack = _read_block(bus, PACK_SUBCLASS, 0)
        v_energy = _u16_be(v_design, 13)
        v_chg1 = _u16_be(v_design, 16)
        v_series = v_pack[7]
        v_pack_cfg = _u16_be(v_pack, 0)
        if (
            v_energy != args.design_energy
            or v_chg1 != args.cell_charge_mv
            or v_series != args.series_cells
            or (v_pack_cfg & 0x0800) == 0
        ):
            print("Verify failed: values did not match.")
            sys.exit(1)
        print("Write complete and verified.")


if __name__ == "__main__":
    main()
