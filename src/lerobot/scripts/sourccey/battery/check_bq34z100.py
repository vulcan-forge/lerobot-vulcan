#!/usr/bin/env python3
"""Dump bq34z100 diagnostics as JSON.

This script is read-only.

Default output is compact and operator-focused:
- voltage/current/SOC/capacity
- pack config + detected series-cell count
- key gauge health/config values
- FC-learning thresholds + live gate snapshot

Use ``--full`` for a detailed dump with raw registers/fields/blocks.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from lerobot.scripts.sourccey.battery.battery import (
    DEFAULT_I2C_ADDR,
    REG_AVERAGE_CURRENT,
    REG_FULL_CHARGE_CAPACITY,
    REG_MAX_ERROR,
    REG_REMAINING_CAPACITY,
    REG_STATE_OF_CHARGE,
    REG_VOLTAGE,
    get_battery_data,
)
from lerobot.scripts.sourccey.battery.configure_bq34z100 import (
    CTRL_CHEM_ID,
    CTRL_CONTROL_STATUS,
    CTRL_DEVICE_TYPE,
    CTRL_FW_VERSION,
    FIELDS,
    BQ34Z100R2,
)


def _word_views(value: int) -> dict[str, Any]:
    v = int(value) & 0xFFFF
    return {
        "int": v,
        "hex": f"0x{v:04X}",
        "bin": f"0b{v:016b}",
    }


def _swap_u16(raw: int) -> int:
    return ((raw & 0xFF) << 8) | ((raw >> 8) & 0xFF)


def _to_s16(raw: int) -> int:
    return raw - 0x10000 if raw & 0x8000 else raw


def _calc_df_block_checksum(block: list[int]) -> int:
    return (0xFF - (sum(block) & 0xFF)) & 0xFF


def _read_raw_standard_registers(
    gauge: BQ34Z100R2,
    *,
    swap_word_bytes: bool,
) -> dict[str, Any]:
    try:
        from smbus2 import SMBus
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("smbus2 is required. Install with: pip install smbus2") from exc

    out: dict[str, Any] = {}
    with SMBus(gauge.bus_num) as bus:
        out["state_of_charge"] = {
            "reg": "0x02",
            "raw_u8": int(bus.read_byte_data(gauge.address, REG_STATE_OF_CHARGE)),
        }
        out["max_error"] = {
            "reg": "0x03",
            "raw_u8": int(bus.read_byte_data(gauge.address, REG_MAX_ERROR)),
        }

        word_regs = {
            "remaining_capacity_mah_raw": REG_REMAINING_CAPACITY,
            "full_charge_capacity_mah_raw": REG_FULL_CHARGE_CAPACITY,
            "voltage_mv_raw": REG_VOLTAGE,
            "average_current_ma_raw": REG_AVERAGE_CURRENT,
        }
        for name, reg in word_regs.items():
            raw = int(bus.read_word_data(gauge.address, reg))
            normalized = _swap_u16(raw) if swap_word_bytes else raw
            out[name] = {
                "reg": f"0x{reg:02X}",
                "raw_word": _word_views(raw),
                "normalized_word": _word_views(normalized),
            }

        cur_norm = out["average_current_ma_raw"]["normalized_word"]["int"]
        out["average_current_ma_raw"]["normalized_signed"] = _to_s16(cur_norm)

    return out


def _read_control_info(gauge: BQ34Z100R2) -> dict[str, Any]:
    device_type = gauge.read_control_subcmd(CTRL_DEVICE_TYPE)
    fw_version = gauge.read_control_subcmd(CTRL_FW_VERSION)
    chem_id = gauge.read_control_subcmd(CTRL_CHEM_ID)
    control_status = gauge.read_control_subcmd(CTRL_CONTROL_STATUS)
    return {
        "device_type": _word_views(device_type),
        "fw_version": _word_views(fw_version),
        "chem_id": _word_views(chem_id),
        "control_status": _word_views(control_status),
    }


def _read_data_flash_fields(gauge: BQ34Z100R2) -> dict[str, Any]:
    fields_out: dict[str, Any] = {}
    for key in sorted(FIELDS):
        spec = FIELDS[key]
        value = gauge.read_field(spec)
        raw_bytes = gauge.read_df_bytes(spec.subclass, spec.offset, spec.size)
        entry: dict[str, Any] = {
            "name": spec.name,
            "description": spec.description,
            "subclass": spec.subclass,
            "offset": spec.offset,
            "size": spec.size,
            "type": spec.dtype,
            "value_int": int(value),
            "value_hex": f"0x{(int(value) & ((1 << (spec.size * 8)) - 1)):0{spec.size * 2}X}",
            "value_bin": f"0b{(int(value) & ((1 << (spec.size * 8)) - 1)):0{spec.size * 8}b}",
            "raw_bytes_hex": raw_bytes.hex().upper(),
        }
        fields_out[key] = entry
    return fields_out


def _read_data_flash_blocks(gauge: BQ34Z100R2) -> list[dict[str, Any]]:
    used_blocks = sorted({(spec.subclass, spec.offset // 32) for spec in FIELDS.values()})
    blocks: list[dict[str, Any]] = []
    for subclass, block_idx in used_blocks:
        data = gauge.read_df_block(subclass, block_idx)
        checksum = _calc_df_block_checksum(data)
        blocks.append(
            {
                "subclass": subclass,
                "block_index": block_idx,
                "raw_bytes_hex": [f"{b:02X}" for b in data],
                "calculated_checksum": f"0x{checksum:02X}",
            }
        )
    return blocks


def _read_learning_thresholds(gauge: BQ34Z100R2) -> dict[str, int]:
    # Gas Gauging Current Thresholds live in subclass 81:
    # 0: Dsg Current Threshold, 2: Chg Current Threshold, 4: Quit Current.
    def _read_i16(subclass: int, offset: int) -> int:
        raw = gauge.read_df_bytes(subclass, offset, 2)
        return int.from_bytes(raw, byteorder="big", signed=True)

    return {
        "dsg_current_threshold_ma": _read_i16(81, 0),
        "chg_current_threshold_ma": _read_i16(81, 2),
        "quit_current_ma": _read_i16(81, 4),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump detailed bq34z100 diagnostics as JSON")
    p.add_argument("--bus", type=int, default=1, help="I2C bus number (default: 1)")
    p.add_argument("--address", type=lambda x: int(x, 0), default=DEFAULT_I2C_ADDR, help="I2C address (default: 0x55)")
    p.add_argument(
        "--actual-rsense-mohm",
        type=float,
        default=12.5,
        help="Actual SRP/SRN shunt resistor in milliohms",
    )
    p.add_argument(
        "--configured-rsense-mohm",
        type=float,
        default=12.5,
        help="Shunt value configured in gauge firmware (milliohms)",
    )
    p.add_argument(
        "--swap-word-bytes",
        action="store_true",
        help="Swap high/low bytes for 16-bit standard command reads",
    )
    p.add_argument(
        "--include-block-dump",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include raw 32-byte data-flash blocks containing known fields (default: enabled)",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Output full diagnostic payload (default: compact summary)",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON (default is compact JSON)",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    base_payload: dict[str, Any] = {
        "chip": "bq34z100",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "i2c": {
            "bus": args.bus,
            "address": f"0x{args.address:02X}",
        },
        "read_options": {
            "configured_rsense_mohm": args.configured_rsense_mohm,
            "actual_rsense_mohm": args.actual_rsense_mohm,
            "swap_word_bytes": bool(args.swap_word_bytes),
            "include_block_dump": bool(args.include_block_dump),
        },
    }

    gauge = BQ34Z100R2(bus_num=args.bus, address=args.address)
    section_errors: dict[str, str] = {}

    telemetry_out: dict[str, Any] | None = None
    control_info: dict[str, Any] | None = None
    data_flash_fields: dict[str, Any] | None = None
    raw_standard_registers: dict[str, Any] | None = None
    data_flash_blocks: list[dict[str, Any]] | None = None
    learning_thresholds: dict[str, int] | None = None

    try:
        telemetry = get_battery_data(
            bus=args.bus,
            address=args.address,
            configured_rsense_mohm=args.configured_rsense_mohm,
            actual_rsense_mohm=args.actual_rsense_mohm,
            swap_word_bytes=args.swap_word_bytes,
        )
        telemetry_out = {
            "voltage": round(float(telemetry.voltage), 3),
            "current_a": round(float(telemetry.current_a), 3),
            "remaining_capacity_ah": round(float(telemetry.remaining_capacity_ah), 3),
            "max_capacity_ah": round(float(telemetry.max_capacity_ah), 3),
            "state_of_charge": int(telemetry.state_of_charge),
            "max_error": int(telemetry.max_error),
        }
    except Exception as exc:  # noqa: BLE001 - script should keep going
        section_errors["telemetry"] = str(exc)

    try:
        raw_standard_registers = _read_raw_standard_registers(
            gauge,
            swap_word_bytes=args.swap_word_bytes,
        )
    except Exception as exc:  # noqa: BLE001
        section_errors["raw_standard_registers"] = str(exc)

    try:
        control_info = _read_control_info(gauge)
    except Exception as exc:  # noqa: BLE001
        section_errors["control_info"] = str(exc)

    try:
        data_flash_fields = _read_data_flash_fields(gauge)
    except Exception as exc:  # noqa: BLE001
        section_errors["data_flash_fields"] = str(exc)

    try:
        learning_thresholds = _read_learning_thresholds(gauge)
    except Exception as exc:  # noqa: BLE001
        section_errors["learning_thresholds"] = str(exc)

    if args.full and args.include_block_dump:
        try:
            data_flash_blocks = _read_data_flash_blocks(gauge)
        except Exception as exc:  # noqa: BLE001
            section_errors["data_flash_blocks"] = str(exc)

    if args.full:
        payload: dict[str, Any] = dict(base_payload)
        if telemetry_out is not None:
            payload["telemetry"] = telemetry_out
        if raw_standard_registers is not None:
            payload["raw_standard_registers"] = raw_standard_registers
        if control_info is not None:
            payload["control_info"] = control_info
        if data_flash_fields is not None:
            payload["data_flash_fields"] = data_flash_fields
        if learning_thresholds is not None:
            payload["learning_thresholds"] = learning_thresholds
        if data_flash_blocks is not None:
            payload["data_flash_blocks"] = data_flash_blocks
    else:
        summary: dict[str, Any] = dict(base_payload)
        if telemetry_out is not None:
            summary["telemetry"] = telemetry_out

        pack_summary: dict[str, Any] = {}
        if data_flash_fields is not None:
            def _field_int(name: str) -> int | None:
                item = data_flash_fields.get(name)
                return None if item is None else int(item["value_int"])

            pack_cfg = _field_int("pack_configuration")
            if pack_cfg is not None:
                msb = (pack_cfg >> 8) & 0xFF
                pack_summary["pack_configuration"] = f"0x{pack_cfg:04X}"
                pack_summary["voltsel_enabled"] = bool(msb & (1 << 3))
            series_cells = _field_int("number_of_series_cells")
            if series_cells is not None:
                pack_summary["series_cells"] = series_cells
            voltage_divider = _field_int("voltage_divider")
            if voltage_divider is not None:
                pack_summary["voltage_divider"] = voltage_divider
            update_status = _field_int("update_status")
            if update_status is not None:
                pack_summary["update_status"] = f"0x{update_status:02X}"
            design_capacity = _field_int("design_capacity_mah")
            if design_capacity is not None:
                pack_summary["design_capacity_mah"] = design_capacity
            qmax = _field_int("qmax_cell0_mah")
            if qmax is not None:
                pack_summary["qmax_cell0_mah"] = qmax

        if control_info is not None:
            chip_info = {
                "device_type": control_info["device_type"]["hex"],
                "fw_version": control_info["fw_version"]["hex"],
                "chem_id": control_info["chem_id"]["hex"],
                "control_status": control_info["control_status"]["hex"],
            }
            summary["chip_info"] = chip_info

        if pack_summary:
            summary["pack"] = pack_summary

        if raw_standard_registers is not None:
            summary["raw_current_ma_signed"] = raw_standard_registers["average_current_ma_raw"][
                "normalized_signed"
            ]

        learning_summary: dict[str, Any] = {}
        if data_flash_fields is not None:
            def _field_int(name: str) -> int | None:
                item = data_flash_fields.get(name)
                return None if item is None else int(item["value_int"])

            series_cells = _field_int("number_of_series_cells")
            taper_current_ma = _field_int("taper_current_ma")
            charge_cell_mv = _field_int("cell_charge_voltage_t2_t3_mv")

            if taper_current_ma is not None:
                learning_summary["taper_current_ma"] = taper_current_ma
            if series_cells is not None:
                learning_summary["series_cells"] = series_cells
            if charge_cell_mv is not None:
                learning_summary["charge_voltage_per_cell_mv"] = charge_cell_mv
            if series_cells is not None and charge_cell_mv is not None:
                charge_target_mv = series_cells * charge_cell_mv
                learning_summary["charge_voltage_target_mv"] = charge_target_mv
                learning_summary["charge_voltage_window_mv"] = [charge_target_mv - 100, charge_target_mv + 100]

        if learning_thresholds is not None:
            learning_summary.update(learning_thresholds)

        current_now_ma: int | None = None
        voltage_now_mv: int | None = None
        if raw_standard_registers is not None:
            current_now_ma = int(raw_standard_registers["average_current_ma_raw"]["normalized_signed"])
            learning_summary["current_now_ma"] = current_now_ma
        if telemetry_out is not None:
            voltage_now_mv = int(round(float(telemetry_out["voltage"]) * 1000.0))
            learning_summary["voltage_now_mv"] = voltage_now_mv

        charge_target_mv = learning_summary.get("charge_voltage_target_mv")
        taper_current_ma = learning_summary.get("taper_current_ma")
        quit_current_ma = learning_summary.get("quit_current_ma")

        if charge_target_mv is not None and voltage_now_mv is not None:
            learning_summary["voltage_within_100mv"] = abs(int(voltage_now_mv) - int(charge_target_mv)) <= 100
        if taper_current_ma is not None and current_now_ma is not None:
            learning_summary["current_below_taper"] = int(current_now_ma) < int(taper_current_ma)
        if quit_current_ma is not None and current_now_ma is not None:
            learning_summary["current_above_quit"] = int(current_now_ma) > int(quit_current_ma)

        if all(k in learning_summary for k in ("voltage_within_100mv", "current_below_taper", "current_above_quit")):
            learning_summary["fc_conditions_now"] = bool(
                learning_summary["voltage_within_100mv"]
                and learning_summary["current_below_taper"]
                and learning_summary["current_above_quit"]
            )

        if learning_summary:
            summary["learning"] = learning_summary

        payload = summary

    if section_errors:
        payload["errors"] = section_errors

    if args.pretty:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
