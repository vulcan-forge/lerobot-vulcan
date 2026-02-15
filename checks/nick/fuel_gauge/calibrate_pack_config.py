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
POWER_SUBCLASS = 68

# Control() register and subcommand
CMD_CONTROL = 0x00
SUB_CAL_ENABLE = 0x002D
SUB_CONTROL_STATUS = 0x0000


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


def _write_control(bus: SMBus, subcmd: int) -> None:
    lo = subcmd & 0xFF
    hi = (subcmd >> 8) & 0xFF
    bus.write_i2c_block_data(BQ_ADDR, CMD_CONTROL, [lo, hi])


def _read_control_status(bus: SMBus) -> int:
    # Issue CONTROL_STATUS (0x0000) then read 2 bytes
    bus.write_i2c_block_data(BQ_ADDR, CMD_CONTROL, [0x00, 0x00])
    data = bus.read_i2c_block_data(BQ_ADDR, CMD_CONTROL, 2)
    return data[0] | (data[1] << 8)


def _is_sealed(status: int) -> bool:
    # CONTROL_STATUS high byte bit 5 (0x2000) is SS per TRM
    return (status & 0x2000) != 0


def _unseal(bus: SMBus, key0: int, key1: int) -> None:
    # Keys are written LSB first (per TRM note)
    bus.write_i2c_block_data(BQ_ADDR, CMD_CONTROL, [key0 & 0xFF, (key0 >> 8) & 0xFF])
    bus.write_i2c_block_data(BQ_ADDR, CMD_CONTROL, [key1 & 0xFF, (key1 >> 8) & 0xFF])


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
    ap.add_argument(
        "--series-offset",
        type=int,
        default=7,
        help="Byte offset for Series Cells in subclass 64 block 0 (default 7).",
    )
    ap.add_argument(
        "--cal-enable",
        action="store_true",
        help="Send CAL_ENABLE (0x002D) before writing to bypass Flash Update OK voltage.",
    )
    ap.add_argument(
        "--flash-update-ok-mv",
        type=int,
        default=None,
        help="Optional Flash Update OK Cell Voltage (mV) to write (subclass 68 offset 0).",
    )
    ap.add_argument("--unseal-key0", type=lambda x: int(x, 0), default=None, help="Unseal key word 0 (e.g. 0x0414).")
    ap.add_argument("--unseal-key1", type=lambda x: int(x, 0), default=None, help="Unseal key word 1 (e.g. 0x3672).")
    ap.add_argument("--write", action="store_true", help="Apply changes to the device.")
    ap.add_argument("--no-verify", action="store_true", help="Skip read-back verification.")
    args = ap.parse_args()

    global BQ_ADDR
    BQ_ADDR = args.addr

    with SMBus(args.bus) as bus:
        status = _read_control_status(bus)
        if _is_sealed(status):
            print(f"CONTROL_STATUS: 0x{status:04X} (SEALED)")
            if args.unseal_key0 is None or args.unseal_key1 is None:
                print("Device is sealed. Provide --unseal-key0 and --unseal-key1 to proceed.")
                return
            _unseal(bus, args.unseal_key0, args.unseal_key1)
            status = _read_control_status(bus)
            print(f"CONTROL_STATUS after unseal: 0x{status:04X}")
            if _is_sealed(status):
                print("Unseal failed; still sealed.")
                return
        else:
            print(f"CONTROL_STATUS: 0x{status:04X} (UNSEALED)")

        design_block = bytearray(_read_block(bus, DESIGN_SUBCLASS, 0))
        pack_block = bytearray(_read_block(bus, PACK_SUBCLASS, 0))
        power_block = bytearray(_read_block(bus, POWER_SUBCLASS, 0))

        cur_energy = _u16_be(design_block, 13)
        cur_chg1 = _u16_be(design_block, 16)
        cur_chg2 = _u16_be(design_block, 18)
        cur_chg3 = _u16_be(design_block, 20)
        cur_pack_cfg = _u16_be(pack_block, 0)
        cur_series = pack_block[args.series_offset]
        cur_update_ok = _u16_be(power_block, 0)

        print("Current values:")
        print(f"  Design Energy: {cur_energy} cWh")
        print(f"  Cell Chg V T1-T2: {cur_chg1} mV")
        print(f"  Cell Chg V T2-T3: {cur_chg2} mV")
        print(f"  Cell Chg V T3-T4: {cur_chg3} mV")
        print(f"  Series Cells: {cur_series} (offset {args.series_offset})")
        print(f"  Pack Config: 0x{cur_pack_cfg:04X}")
        print(f"  Pack block[0:16]: {pack_block[:16].hex()}")
        print(f"  Flash Update OK Cell Volt: {cur_update_ok} mV")

        _set_u16_be(design_block, 13, args.design_energy)
        _set_u16_be(design_block, 16, args.cell_charge_mv)
        _set_u16_be(design_block, 18, args.cell_charge_mv)
        _set_u16_be(design_block, 20, args.cell_charge_mv)
        pack_block[args.series_offset] = args.series_cells & 0xFF
        # Set CAL_EN (0x4000) and VOLTSEL (0x0800) so series cells are used.
        pack_cfg = cur_pack_cfg | 0x4000 | 0x0800
        _set_u16_be(pack_block, 0, pack_cfg)
        if args.flash_update_ok_mv is not None:
            _set_u16_be(power_block, 0, args.flash_update_ok_mv)

        print("Planned updates:")
        print(f"  Design Energy: {cur_energy} -> {args.design_energy} cWh")
        print(f"  Cell Charge Voltage: {cur_chg1}/{cur_chg2}/{cur_chg3} -> {args.cell_charge_mv} mV")
        print(f"  Series Cells: {cur_series} -> {args.series_cells} (offset {args.series_offset})")
        print(f"  Pack Config: 0x{cur_pack_cfg:04X} -> 0x{pack_cfg:04X}")
        if args.flash_update_ok_mv is not None:
            print(f"  Flash Update OK Cell Volt: {cur_update_ok} -> {args.flash_update_ok_mv} mV")

        if not args.write:
            print("Dry-run only. Re-run with --write to apply.")
            return

        if args.cal_enable:
            _write_control(bus, SUB_CAL_ENABLE)

        _write_block(bus, DESIGN_SUBCLASS, 0, bytes(design_block))
        _write_block(bus, PACK_SUBCLASS, 0, bytes(pack_block))
        if args.flash_update_ok_mv is not None:
            _write_block(bus, POWER_SUBCLASS, 0, bytes(power_block))

        if not args.no_verify:
            # Verify
            v_design = _read_block(bus, DESIGN_SUBCLASS, 0)
            v_pack = _read_block(bus, PACK_SUBCLASS, 0)
            v_energy = _u16_be(v_design, 13)
            v_chg1 = _u16_be(v_design, 16)
            v_series = v_pack[args.series_offset]
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
        else:
            print("Write complete (verification skipped).")


if __name__ == "__main__":
    main()
