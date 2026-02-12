# Configure BQ34Z100-R2 data flash over I2C.
# Requires: pip install smbus2
#
# SAFETY:
# - Default is dry-run (no writes).
# - Use --write to actually update flash.
# - This script assumes default UNSEAL key (0x36720414).
#
# References:
# - BQ34Z100-R2 Technical Reference Manual (data flash access and checksum)
# - Data Flash Summary tables for offsets/subclasses

from __future__ import annotations

import argparse
from dataclasses import dataclass
from smbus2 import SMBus

I2C_BUS = 1
BQ_ADDR = 0x55

# Standard commands
CMD_CONTROL = 0x00
CMD_BLOCK_DATA_CONTROL = 0x61
CMD_DATA_FLASH_CLASS = 0x3E
CMD_DATA_FLASH_BLOCK = 0x3F
CMD_BLOCK_DATA = 0x40  # 0x40..0x5F (32 bytes)
CMD_BLOCK_DATA_CHECKSUM = 0x60

# Control subcommands
SUBCMD_CONTROL_STATUS = 0x0000
SUBCMD_RESET = 0x0041
SUBCMD_SEAL = 0x0020

# Default unseal key (LSW then MSW)
UNSEAL_KEY = 0x36720414

# Data flash locations (subclass, offset)
# Configuration Data (class): subclass 48
DF_DESIGN_VOLTAGE = (48, 0, 2)  # mV per cell
DF_DESIGN_CAPACITY = (48, 11, 2)  # mAh
DF_DESIGN_ENERGY = (48, 13, 2)  # cWh (pack)
DF_CELL_CHG_V_T1T2 = (48, 17, 2)  # mV per cell
DF_CELL_CHG_V_T2T3 = (48, 19, 2)  # mV per cell
DF_CELL_CHG_V_T3T4 = (48, 21, 2)  # mV per cell

# Configuration Registers: subclass 64
DF_PACK_CONFIG = (64, 0, 2)  # Pack Configuration (H2)
DF_NUM_SERIES_CELLS = (64, 7, 1)  # Number of Series Cells (U1)

# Calibration Data: subclass 104
DF_VOLTAGE_DIVIDER = (104, 14, 2)  # Voltage Divider (mV)


@dataclass
class PackConfig:
    series_cells: int
    design_capacity_mAh: int
    design_voltage_mV: int
    design_energy_cWh: int
    cell_charge_mV: int | None
    voltage_divider_value: int
    set_voltsel: bool


def _write_control_word(bus: SMBus, subcmd: int) -> None:
    bus.write_i2c_block_data(BQ_ADDR, CMD_CONTROL, [subcmd & 0xFF, (subcmd >> 8) & 0xFF])


def _read_control_word(bus: SMBus, subcmd: int) -> int:
    _write_control_word(bus, subcmd)
    data = bus.read_i2c_block_data(BQ_ADDR, CMD_CONTROL, 2)
    return data[0] | (data[1] << 8)


def _unseal(bus: SMBus) -> None:
    key = UNSEAL_KEY
    lsw = key & 0xFFFF
    msw = (key >> 16) & 0xFFFF
    _write_control_word(bus, lsw)
    _write_control_word(bus, msw)


def _seal(bus: SMBus) -> None:
    _write_control_word(bus, SUBCMD_SEAL)


def _set_dataflash_class_block(bus: SMBus, subclass: int, block: int) -> None:
    bus.write_byte_data(BQ_ADDR, CMD_BLOCK_DATA_CONTROL, 0x00)
    bus.write_byte_data(BQ_ADDR, CMD_DATA_FLASH_CLASS, subclass)
    bus.write_byte_data(BQ_ADDR, CMD_DATA_FLASH_BLOCK, block)


def _read_block(bus: SMBus, subclass: int, block: int) -> list[int]:
    _set_dataflash_class_block(bus, subclass, block)
    data = bus.read_i2c_block_data(BQ_ADDR, CMD_BLOCK_DATA, 32)
    return list(data)


def _write_block(bus: SMBus, subclass: int, block: int, data: list[int]) -> None:
    _set_dataflash_class_block(bus, subclass, block)
    for i, b in enumerate(data):
        bus.write_byte_data(BQ_ADDR, CMD_BLOCK_DATA + i, b & 0xFF)
    checksum = (255 - (sum(data) & 0xFF)) & 0xFF
    bus.write_byte_data(BQ_ADDR, CMD_BLOCK_DATA_CHECKSUM, checksum)


def _get_block_offset(offset: int) -> tuple[int, int]:
    block = offset // 32
    in_block = offset % 32
    return block, in_block


def _set_u1(data: list[int], offset: int, value: int) -> None:
    data[offset] = value & 0xFF


def _set_u2(data: list[int], offset: int, value: int) -> None:
    data[offset] = value & 0xFF
    data[offset + 1] = (value >> 8) & 0xFF


def _get_u2(data: list[int], offset: int) -> int:
    return data[offset] | (data[offset + 1] << 8)


def _apply_pack_config(cfg: PackConfig, bus: SMBus, dry_run: bool) -> None:
    updates: list[tuple[str, int, int, int]] = []
    print("Target values:")
    print(f"  Series Cells: {cfg.series_cells}")
    print(f"  Design Capacity (mAh): {cfg.design_capacity_mAh}")
    print(f"  Design Voltage (mV/cell): {cfg.design_voltage_mV}")
    print(f"  Design Energy (cWh): {cfg.design_energy_cWh}")
    print(f"  Voltage Divider (ratio*1000): {cfg.voltage_divider_value}")
    print(f"  Set VOLTSEL: {cfg.set_voltsel}")
    if cfg.cell_charge_mV is not None:
        print(f"  Cell Charge Voltage (mV): {cfg.cell_charge_mV}")

    # Configuration Data (subclass 48)
    for name, (subclass, offset, size), value in [
        ("Design Voltage (mV/cell)", DF_DESIGN_VOLTAGE, cfg.design_voltage_mV),
        ("Design Capacity (mAh)", DF_DESIGN_CAPACITY, cfg.design_capacity_mAh),
        ("Design Energy (cWh)", DF_DESIGN_ENERGY, cfg.design_energy_cWh),
    ]:
        block, in_block = _get_block_offset(offset)
        data = _read_block(bus, subclass, block)
        old = _get_u2(data, in_block)
        if old != value:
            updates.append((name, old, value, subclass))
            if not dry_run:
                _set_u2(data, in_block, value)
                _write_block(bus, subclass, block, data)

    if cfg.cell_charge_mV is not None:
        for name, (subclass, offset, size), value in [
            ("Cell Chg V T1-T2 (mV)", DF_CELL_CHG_V_T1T2, cfg.cell_charge_mV),
            ("Cell Chg V T2-T3 (mV)", DF_CELL_CHG_V_T2T3, cfg.cell_charge_mV),
            ("Cell Chg V T3-T4 (mV)", DF_CELL_CHG_V_T3T4, cfg.cell_charge_mV),
        ]:
            block, in_block = _get_block_offset(offset)
            data = _read_block(bus, subclass, block)
            old = _get_u2(data, in_block)
            if old != value:
                updates.append((name, old, value, subclass))
                if not dry_run:
                    _set_u2(data, in_block, value)
                    _write_block(bus, subclass, block, data)

    # Configuration Registers (subclass 64)
    # Pack Configuration: set VOLTSEL bit (bit 3 of MSB)
    subclass, offset, _size = DF_PACK_CONFIG
    block, in_block = _get_block_offset(offset)
    data = _read_block(bus, subclass, block)
    old_pack_cfg = _get_u2(data, in_block)
    new_pack_cfg = old_pack_cfg
    if cfg.set_voltsel:
        new_pack_cfg = old_pack_cfg | (1 << 11)  # bit 3 of MSB
    if new_pack_cfg != old_pack_cfg:
        updates.append(("Pack Config (VOLTSEL)", old_pack_cfg, new_pack_cfg, subclass))
        if not dry_run:
            _set_u2(data, in_block, new_pack_cfg)
            _write_block(bus, subclass, block, data)

    # Number of series cells
    subclass, offset, _size = DF_NUM_SERIES_CELLS
    block, in_block = _get_block_offset(offset)
    data = _read_block(bus, subclass, block)
    old_cells = data[in_block]
    if old_cells != cfg.series_cells:
        updates.append(("Series Cells", old_cells, cfg.series_cells, subclass))
        if not dry_run:
            _set_u1(data, in_block, cfg.series_cells)
            _write_block(bus, subclass, block, data)

    # Calibration Data (subclass 104): Voltage Divider
    subclass, offset, _size = DF_VOLTAGE_DIVIDER
    block, in_block = _get_block_offset(offset)
    data = _read_block(bus, subclass, block)
    old_div = _get_u2(data, in_block)
    if old_div != cfg.voltage_divider_value:
        updates.append(("Voltage Divider", old_div, cfg.voltage_divider_value, subclass))
        if not dry_run:
            _set_u2(data, in_block, cfg.voltage_divider_value)
            _write_block(bus, subclass, block, data)

    print("Planned changes:")
    if not updates:
        print("  (no changes needed)")
    for name, old, new, subclass in updates:
        print(f"  - {name}: {old} -> {new} (subclass {subclass})")


def _dump_block(bus: SMBus, subclass: int, block: int) -> None:
    data = _read_block(bus, subclass, block)
    hex_bytes = " ".join(f"{b:02X}" for b in data)
    print(f"Subclass {subclass} Block {block} (offset {block * 32:03d}):")
    print(f"  {hex_bytes}")


def _dump_known_fields(bus: SMBus) -> None:
    print("Dumping raw data blocks for verification.")
    for subclass in [48, 64, 104]:
        _dump_block(bus, subclass, 0)
        _dump_block(bus, subclass, 1)
    print("Note: offsets are firmware-dependent. We will validate mapping before writes.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure BQ34Z100-R2 data flash over I2C.")
    parser.add_argument("--write", action="store_true", help="Perform writes (default is dry-run).")
    parser.add_argument("--dump", action="store_true", help="Dump raw data flash blocks (no writes).")
    parser.add_argument("--seal", action="store_true", help="Seal after writing.")
    parser.add_argument("--series", type=int, default=4, help="Number of series cells (e.g., 4).")
    parser.add_argument("--capacity-mAh", type=int, default=10000, help="Design capacity in mAh.")
    parser.add_argument("--design-voltage-mV", type=int, default=3200, help="Design voltage per cell (mV).")
    parser.add_argument(
        "--cell-charge-mV",
        type=int,
        default=None,
        help="Set cell charge voltage (mV) for all temperature ranges.",
    )
    parser.add_argument("--divider-top-ohms", type=float, default=249_000.0)
    parser.add_argument("--divider-bottom-ohms", type=float, default=16_500.0)
    parser.add_argument("--set-voltsel", action="store_true", help="Set VOLTSEL bit for external divider.")
    args = parser.parse_args()

    # Compute divider value: ratio * 1000 (e.g., 5:1 internal -> 5000)
    ratio = (args.divider_top_ohms + args.divider_bottom_ohms) / args.divider_bottom_ohms
    divider_value = int(round(ratio * 1000))

    # Design energy (cWh) = mAh * mV_total / 10000
    design_voltage_pack = args.design_voltage_mV * args.series
    design_energy_cWh = int(round(args.capacity_mAh * design_voltage_pack / 10000.0))

    cfg = PackConfig(
        series_cells=args.series,
        design_capacity_mAh=args.capacity_mAh,
        design_voltage_mV=args.design_voltage_mV,
        design_energy_cWh=design_energy_cWh,
        cell_charge_mV=args.cell_charge_mV,
        voltage_divider_value=divider_value,
        set_voltsel=args.set_voltsel,
    )

    with SMBus(I2C_BUS) as bus:
        _read_control_word(bus, SUBCMD_CONTROL_STATUS)
        if args.dump:
            _dump_known_fields(bus)
            return
        if args.write:
            _unseal(bus)
        _apply_pack_config(cfg, bus, dry_run=not args.write)
        if args.write:
            _write_control_word(bus, SUBCMD_RESET)
            if args.seal:
                _seal(bus)


if __name__ == "__main__":
    main()
