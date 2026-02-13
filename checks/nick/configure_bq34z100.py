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
import time
import json
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
    time.sleep(0.02)


def _seal(bus: SMBus) -> None:
    _write_control_word(bus, SUBCMD_SEAL)


def _set_dataflash_class_block(bus: SMBus, subclass: int, block: int) -> None:
    bus.write_byte_data(BQ_ADDR, CMD_BLOCK_DATA_CONTROL, 0x00)
    bus.write_byte_data(BQ_ADDR, CMD_DATA_FLASH_CLASS, subclass)
    bus.write_byte_data(BQ_ADDR, CMD_DATA_FLASH_BLOCK, block)
    time.sleep(0.01)


def _read_block(bus: SMBus, subclass: int, block: int) -> list[int]:
    _set_dataflash_class_block(bus, subclass, block)
    data = bus.read_i2c_block_data(BQ_ADDR, CMD_BLOCK_DATA, 32)
    return list(data)


def _read_block_retry(bus: SMBus, subclass: int, block: int, retries: int = 3) -> list[int]:
    last_err = None
    for _ in range(retries):
        try:
            return _read_block(bus, subclass, block)
        except OSError as err:
            last_err = err
            time.sleep(0.05)
    if last_err:
        raise last_err
    return []


def _write_block(bus: SMBus, subclass: int, block: int, data: list[int]) -> None:
    _set_dataflash_class_block(bus, subclass, block)
    for i, b in enumerate(data):
        bus.write_byte_data(BQ_ADDR, CMD_BLOCK_DATA + i, b & 0xFF)
    checksum = (255 - (sum(data) & 0xFF)) & 0xFF
    bus.write_byte_data(BQ_ADDR, CMD_BLOCK_DATA_CHECKSUM, checksum)
    time.sleep(0.01)


def _write_block_retry(bus: SMBus, subclass: int, block: int, data: list[int], retries: int = 3) -> None:
    last_err = None
    for _ in range(retries):
        try:
            _write_block(bus, subclass, block, data)
            return
        except OSError as err:
            last_err = err
            time.sleep(0.05)
    if last_err:
        raise last_err


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


def _read_u1_field(bus: SMBus, subclass: int, offset: int) -> int:
    block, in_block = _get_block_offset(offset)
    data = _read_block_retry(bus, subclass, block)
    return data[in_block]


def _read_u2_field(bus: SMBus, subclass: int, offset: int) -> int:
    block, in_block = _get_block_offset(offset)
    data = _read_block_retry(bus, subclass, block)
    return _get_u2(data, in_block)


def _print_current_values(bus: SMBus) -> None:
    print("Current values:")
    print(f"  Design Voltage (mV/cell): {_read_u2_field(bus, 48, 0)}")
    print(f"  Design Capacity (mAh): {_read_u2_field(bus, 48, 11)}")
    print(f"  Design Energy (cWh): {_read_u2_field(bus, 48, 13)}")
    print(f"  Cell Chg V T1-T2 (mV): {_read_u2_field(bus, 48, 17)}")
    print(f"  Cell Chg V T2-T3 (mV): {_read_u2_field(bus, 48, 19)}")
    print(f"  Cell Chg V T3-T4 (mV): {_read_u2_field(bus, 48, 21)}")
    print(f"  Pack Config (VOLTSEL bit): {bool(_read_u2_field(bus, 64, 0) & (1 << 11))}")
    print(f"  Series Cells: {_read_u1_field(bus, 64, 7)}")
    print(f"  Voltage Divider (ratio*1000): {_read_u2_field(bus, 104, 14)}")


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

    # Collect per-block edits to avoid overwriting earlier changes in the same block.
    block_edits: dict[tuple[int, int], list[tuple[str, int, int, str]]] = {}

    def add_u2(name: str, subclass: int, offset: int, value: int) -> None:
        block, in_block = _get_block_offset(offset)
        data = _read_block_retry(bus, subclass, block)
        old = _get_u2(data, in_block)
        if old != value:
            updates.append((name, old, value, subclass))
            block_edits.setdefault((subclass, block), [])
            block_edits[(subclass, block)].append(("u2", in_block, value, name))

    def add_u1(name: str, subclass: int, offset: int, value: int) -> None:
        block, in_block = _get_block_offset(offset)
        data = _read_block_retry(bus, subclass, block)
        old = data[in_block]
        if old != value:
            updates.append((name, old, value, subclass))
            block_edits.setdefault((subclass, block), [])
            block_edits[(subclass, block)].append(("u1", in_block, value, name))

    # Configuration Data (subclass 48)
    add_u2("Design Voltage (mV/cell)", 48, 0, cfg.design_voltage_mV)
    add_u2("Design Capacity (mAh)", 48, 11, cfg.design_capacity_mAh)
    add_u2("Design Energy (cWh)", 48, 13, cfg.design_energy_cWh)

    if cfg.cell_charge_mV is not None:
        add_u2("Cell Chg V T1-T2 (mV)", 48, 17, cfg.cell_charge_mV)
        add_u2("Cell Chg V T2-T3 (mV)", 48, 19, cfg.cell_charge_mV)
        add_u2("Cell Chg V T3-T4 (mV)", 48, 21, cfg.cell_charge_mV)

    # Configuration Registers (subclass 64)
    # Pack Configuration: set VOLTSEL bit (bit 3 of MSB)
    subclass, offset, _size = DF_PACK_CONFIG
    block, in_block = _get_block_offset(offset)
    data = _read_block_retry(bus, subclass, block)
    old_pack_cfg = _get_u2(data, in_block)
    new_pack_cfg = old_pack_cfg
    if cfg.set_voltsel:
        new_pack_cfg = old_pack_cfg | (1 << 11)  # bit 3 of MSB
    if new_pack_cfg != old_pack_cfg:
        updates.append(("Pack Config (VOLTSEL)", old_pack_cfg, new_pack_cfg, subclass))
        block_edits.setdefault((subclass, block), [])
        block_edits[(subclass, block)].append(("u2", in_block, new_pack_cfg, "Pack Config (VOLTSEL)"))

    # Number of series cells
    add_u1("Series Cells", 64, 7, cfg.series_cells)

    # Calibration Data (subclass 104): Voltage Divider
    add_u2("Voltage Divider", 104, 14, cfg.voltage_divider_value)

    print("Planned changes:")
    if not updates:
        print("  (no changes needed)")
    for name, old, new, subclass in updates:
        print(f"  - {name}: {old} -> {new} (subclass {subclass})")

    if dry_run or not updates:
        return

    for (subclass, block), edits in block_edits.items():
        data = _read_block_retry(bus, subclass, block)
        for kind, in_block, value, _name in edits:
            if kind == "u2":
                _set_u2(data, in_block, value)
            else:
                _set_u1(data, in_block, value)
        _write_block_retry(bus, subclass, block, data)


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


def _backup_to_file(bus: SMBus, path: str) -> None:
    payload = {"subclasses": {}}
    for subclass in [48, 64, 104]:
        payload["subclasses"][str(subclass)] = {}
        for block in [0, 1]:
            data = _read_block_retry(bus, subclass, block)
            payload["subclasses"][str(subclass)][str(block)] = data
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Backup saved to {path}")


def _restore_from_file(bus: SMBus, path: str, dry_run: bool) -> None:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    subclasses = payload.get("subclasses", {})
    for subclass_str, blocks in subclasses.items():
        subclass = int(subclass_str)
        for block_str, data in blocks.items():
            block = int(block_str)
            if len(data) != 32:
                raise ValueError(f"Invalid block length for subclass {subclass} block {block}")
            if dry_run:
                print(f"Would restore subclass {subclass} block {block}")
            else:
                _write_block_retry(bus, subclass, block, list(data))
                print(f"Restored subclass {subclass} block {block}")


def _preset_default() -> dict[int, dict[int, list[int]]]:
    # Captured from initial dump before modifications.
    return {
        48: {
            0: [
                0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x03,
                0x84, 0x64, 0x03, 0xE8, 0x15, 0x18, 0xFE, 0x70,
                0x10, 0x68, 0x10, 0x68, 0x10, 0x04, 0x0A, 0x32,
                0x1E, 0x00, 0x0A, 0x2D, 0x37, 0x01, 0x01, 0xA0,
            ],
            1: [
                0x0B, 0x62, 0x71, 0x33, 0x34, 0x7A, 0x31, 0x30,
                0x30, 0x2D, 0x47, 0x31, 0x0B, 0x54, 0x65, 0x78,
                0x61, 0x73, 0x20, 0x49, 0x6E, 0x73, 0x74, 0x2E,
                0x04, 0x4C, 0x49, 0x4F, 0x4E, 0x00, 0x00, 0xD7,
            ],
        },
        64: {
            0: [
                0xD9, 0xAF, 0x37, 0x00, 0x00, 0x00, 0x01, 0x00,
                0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xEA,
            ],
            1: [
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF,
            ],
        },
        104: {
            0: [
                0x71, 0x20, 0x5C, 0x94, 0x08, 0x98, 0xC0, 0xFB,
                0x50, 0x00, 0x00, 0x00, 0x00, 0x13, 0x88, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xB9,
            ],
            1: [
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF,
            ],
        },
    }


def _preset_custom(cfg: PackConfig) -> dict[int, dict[int, list[int]]]:
    # Start from the captured default blocks and apply only the known custom fields.
    preset = _preset_default()
    data48 = preset[48][0][:]
    # Design Voltage (offset 0, U2)
    data48[0] = cfg.design_voltage_mV & 0xFF
    data48[1] = (cfg.design_voltage_mV >> 8) & 0xFF
    # Design Capacity (offset 11, U2)
    data48[11] = cfg.design_capacity_mAh & 0xFF
    data48[12] = (cfg.design_capacity_mAh >> 8) & 0xFF
    # Design Energy (offset 13, U2)
    data48[13] = cfg.design_energy_cWh & 0xFF
    data48[14] = (cfg.design_energy_cWh >> 8) & 0xFF
    # Cell charge voltages (offsets 17, 19, 21)
    if cfg.cell_charge_mV is not None:
        for off in (17, 19, 21):
            data48[off] = cfg.cell_charge_mV & 0xFF
            data48[off + 1] = (cfg.cell_charge_mV >> 8) & 0xFF
    preset[48][0] = data48

    # Pack Config (offset 0, U2) and Series Cells (offset 7)
    data64 = preset[64][0][:]
    pack_cfg = data64[0] | (data64[1] << 8)
    if cfg.set_voltsel:
        pack_cfg |= (1 << 11)
    data64[0] = pack_cfg & 0xFF
    data64[1] = (pack_cfg >> 8) & 0xFF
    data64[7] = cfg.series_cells & 0xFF
    preset[64][0] = data64

    # Voltage Divider (offset 14, U2)
    data104 = preset[104][0][:]
    data104[14] = cfg.voltage_divider_value & 0xFF
    data104[15] = (cfg.voltage_divider_value >> 8) & 0xFF
    preset[104][0] = data104

    return preset


def _apply_preset(bus: SMBus, preset: dict[int, dict[int, list[int]]], dry_run: bool) -> None:
    for subclass, blocks in preset.items():
        for block, data in blocks.items():
            if len(data) != 32:
                raise ValueError(f"Invalid block length for subclass {subclass} block {block}")
            if dry_run:
                print(f"Would restore subclass {subclass} block {block}")
            else:
                _write_block_retry(bus, subclass, block, list(data))
                print(f"Restored subclass {subclass} block {block}")


def _apply_preset_subset(
    bus: SMBus,
    preset: dict[int, dict[int, list[int]]],
    subclasses: set[int],
    dry_run: bool,
) -> None:
    for subclass in subclasses:
        blocks = preset.get(subclass, {})
        for block, data in blocks.items():
            if len(data) != 32:
                raise ValueError(f"Invalid block length for subclass {subclass} block {block}")
            if dry_run:
                print(f"Would restore subclass {subclass} block {block}")
            else:
                _write_block_retry(bus, subclass, block, list(data))
                print(f"Restored subclass {subclass} block {block}")


def _preset_trial_capacity() -> dict[int, dict[int, list[int]]]:
    # Start from the "default" preset and change only Design Capacity to 10000 mAh.
    preset = _preset_default()
    data = preset[48][0][:]
    # Design Capacity offset 11 (U2), little-endian.
    data[11] = 0x10
    data[12] = 0x27
    preset[48][0] = data
    return preset


def _preset_trial_voltage() -> dict[int, dict[int, list[int]]]:
    # Start from the "default" preset and change only voltage divider + VOLTSEL.
    preset = _preset_default()
    # Set VOLTSEL bit in Pack Config (class 64, block 0, offset 0, U2).
    data64 = preset[64][0][:]
    pack_cfg = data64[0] | (data64[1] << 8)
    pack_cfg |= (1 << 11)
    data64[0] = pack_cfg & 0xFF
    data64[1] = (pack_cfg >> 8) & 0xFF
    preset[64][0] = data64
    # Voltage Divider in class 104, block 0, offset 14 (U2) = 16091 (0x3EDB).
    data104 = preset[104][0][:]
    data104[14] = 0xDB
    data104[15] = 0x3E
    preset[104][0] = data104
    return preset


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure BQ34Z100-R2 data flash over I2C.")
    parser.add_argument("--write", action="store_true", help="Perform writes (default is dry-run).")
    parser.add_argument("--dump", action="store_true", help="Dump raw data flash blocks (no writes).")
    parser.add_argument("--backup", type=str, help="Backup data flash blocks to a JSON file.")
    parser.add_argument("--restore", type=str, help="Restore data flash blocks from a JSON file.")
    parser.add_argument(
        "--repair-custom",
        action="store_true",
        help="Restore baseline config (subclasses 64/104) then apply custom values.",
    )
    parser.add_argument(
        "--preset",
        choices=["default", "custom", "trial-capacity", "trial-voltage"],
        help="Restore a built-in preset (default or custom).",
    )
    parser.add_argument(
        "--apply",
        choices=["voltage"],
        help="Apply a single in-place fix without restoring full presets.",
    )
    parser.add_argument("--seal", action="store_true", help="Seal after writing.")
    parser.add_argument("--series", type=int, default=4, help="Number of series cells (e.g., 4).")
    parser.add_argument("--capacity-mAh", type=int, default=10000, help="Design capacity in mAh.")
    parser.add_argument("--design-voltage-mV", type=int, default=3200, help="Design voltage per cell (mV).")
    parser.add_argument(
        "--cell-charge-mV",
        type=int,
        default=3600,
        help="Set cell charge voltage (mV) for all temperature ranges.",
    )
    parser.add_argument("--divider-top-ohms", type=float, default=249_000.0)
    parser.add_argument("--divider-bottom-ohms", type=float, default=16_500.0)
    parser.add_argument("--set-voltsel", action="store_true", help="Set VOLTSEL bit for external divider.")
    parser.set_defaults(set_voltsel=True)
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
        status = _read_control_word(bus, SUBCMD_CONTROL_STATUS)
        sealed = bool(status & (1 << 13))
        if sealed:
            print("Gauge appears SEALED. Attempting unseal.")
        _print_current_values(bus)
        if args.dump:
            _dump_known_fields(bus)
            return
        if args.backup:
            _backup_to_file(bus, args.backup)
            return
        if args.restore:
            _unseal(bus)
            _restore_from_file(bus, args.restore, dry_run=not args.write)
            if args.write:
                _write_control_word(bus, SUBCMD_RESET)
                time.sleep(0.2)
            return
        if args.repair_custom:
            _unseal(bus)
            if not args.write:
                print("Would restore full custom preset from baseline defaults.")
                _apply_preset(bus, _preset_custom(cfg), dry_run=True)
                return
            _apply_preset(bus, _preset_custom(cfg), dry_run=False)
            _write_control_word(bus, SUBCMD_RESET)
            time.sleep(0.2)
            _print_current_values(bus)
            return
        if args.preset:
            _unseal(bus)
            if args.preset == "default":
                _apply_preset(bus, _preset_default(), dry_run=not args.write)
            elif args.preset == "trial-capacity":
                _apply_preset(bus, _preset_trial_capacity(), dry_run=not args.write)
            elif args.preset == "trial-voltage":
                _apply_preset(bus, _preset_trial_voltage(), dry_run=not args.write)
            else:
                _apply_preset(bus, _preset_custom(cfg), dry_run=not args.write)
                if args.write:
                    _write_control_word(bus, SUBCMD_RESET)
                    time.sleep(0.2)
            return
        if args.apply == "voltage":
            _unseal(bus)
            if not args.write:
                print("Would set VOLTSEL and Voltage Divider in-place.")
                return
            # Set VOLTSEL bit in Pack Config (class 64, offset 0)
            subclass, offset, _size = DF_PACK_CONFIG
            block, in_block = _get_block_offset(offset)
            data = _read_block_retry(bus, subclass, block)
            pack_cfg = _get_u2(data, in_block) | (1 << 11)
            _set_u2(data, in_block, pack_cfg)
            _write_block_retry(bus, subclass, block, data)
            # Set Voltage Divider (class 104, offset 14)
            subclass, offset, _size = DF_VOLTAGE_DIVIDER
            block, in_block = _get_block_offset(offset)
            data = _read_block_retry(bus, subclass, block)
            _set_u2(data, in_block, 16091)
            _write_block_retry(bus, subclass, block, data)
            _write_control_word(bus, SUBCMD_RESET)
            time.sleep(0.2)
            return
        if args.write:
            _unseal(bus)
            status_after = _read_control_word(bus, SUBCMD_CONTROL_STATUS)
            sealed_after = bool(status_after & (1 << 13))
            if sealed_after:
                raise RuntimeError("Unseal failed. Check unseal key or security settings.")
        _apply_pack_config(cfg, bus, dry_run=not args.write)
        if args.write:
            _write_control_word(bus, SUBCMD_RESET)
            time.sleep(0.2)
            _print_current_values(bus)
            if args.seal:
                _seal(bus)


if __name__ == "__main__":
    main()
