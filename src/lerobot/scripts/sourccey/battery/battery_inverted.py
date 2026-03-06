#!/usr/bin/env python3
"""Read bq34z100 telemetry with software correction for inverted current polarity.

This is a wrapper around checks/nick/battery.py that:
- Inverts the reported current sign by default.
- Maintains a software SOC/remaining-Ah estimate by integrating corrected current.

Important:
- This does NOT fix the gauge's internal SOC/FCC/RM learning.
- It gives a practical software-side estimate when hardware polarity cannot be changed.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import battery_standard as base_battery


DEFAULT_STATE_FILE = Path(__file__).resolve().parent / "battery_inverted_state.json"


@dataclass
class SoftwareSocState:
    remaining_ah: float
    capacity_ah: float
    last_timestamp_s: float


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _load_state(path: Path, capacity_ah: float) -> SoftwareSocState | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None

    try:
        remaining_ah = float(raw["remaining_ah"])
        last_timestamp_s = float(raw["last_timestamp_s"])
        saved_capacity = float(raw.get("capacity_ah", capacity_ah))
    except (KeyError, TypeError, ValueError):
        return None

    # If capacity changed, keep percentage and remap remaining Ah to new capacity.
    if saved_capacity > 0.0 and capacity_ah > 0.0 and saved_capacity != capacity_ah:
        pct = _clamp(remaining_ah / saved_capacity, 0.0, 1.0)
        remaining_ah = pct * capacity_ah

    return SoftwareSocState(
        remaining_ah=_clamp(remaining_ah, 0.0, capacity_ah),
        capacity_ah=capacity_ah,
        last_timestamp_s=last_timestamp_s,
    )


def _save_state(path: Path, state: SoftwareSocState) -> None:
    payload = {
        "remaining_ah": state.remaining_ah,
        "capacity_ah": state.capacity_ah,
        "last_timestamp_s": state.last_timestamp_s,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = base_battery._build_parser()
    p.description = "Read battery telemetry for inverted-polarity current sense"
    p.add_argument(
        "--no-invert-current",
        action="store_true",
        help="Do not invert current sign (debug mode)",
    )
    p.add_argument(
        "--software-capacity-ah",
        type=float,
        default=10.0,
        help="Software SOC nominal pack capacity (Ah)",
    )
    p.add_argument(
        "--software-initial-soc-pct",
        type=float,
        default=100.0,
        help="Initial software SOC percentage used when no state exists",
    )
    p.add_argument(
        "--software-state-file",
        type=Path,
        default=DEFAULT_STATE_FILE,
        help=f"State file path (default: {DEFAULT_STATE_FILE})",
    )
    p.add_argument(
        "--software-max-integrate-gap-s",
        type=float,
        default=10.0,
        help="Max time gap to integrate over; larger gaps are ignored",
    )
    p.add_argument(
        "--software-reset-state",
        action="store_true",
        help="Reset software SOC state before reading",
    )
    p.add_argument(
        "--software-disable",
        action="store_true",
        help="Disable software SOC integration/output",
    )
    return p


def _run_raw_mode(args: argparse.Namespace, gauge: base_battery.BQ34Z100) -> bool:
    raw_mode_count = int(args.raw_byte is not None) + int(args.raw_word is not None) + int(args.dump_range is not None)
    if raw_mode_count > 1:
        raise SystemExit("Use only one of: --raw-byte, --raw-word, or --dump-range")

    if args.raw_byte is not None:
        if args.raw_byte < 0x00 or args.raw_byte > 0xFF:
            raise SystemExit("--raw-byte register must be in 0x00..0xFF")
        value = gauge.read_byte_raw(args.raw_byte)
        print(f"reg {args.raw_byte:#04x} byte = 0x{value:02X} ({value})")
        return True

    if args.raw_word is not None:
        if args.raw_word < 0x00 or args.raw_word > 0xFE:
            raise SystemExit("--raw-word register must be in 0x00..0xFE")
        value = gauge.read_word_raw(args.raw_word, args.swap_word_bytes)
        signed = base_battery.BQ34Z100._to_s16(value)
        print(f"reg {args.raw_word:#04x} word = 0x{value:04X} (u16={value}, s16={signed})")
        return True

    if args.dump_range is not None:
        start = int(args.dump_range[0], 0)
        end = int(args.dump_range[1], 0)
        rows = gauge.dump_range_raw(start, end, args.swap_word_bytes)
        for row in rows:
            word = row.get("word", "----")
            print(f"{row['reg']}: byte={row['byte']} word={word}")
        return True

    return False


def _init_or_reset_state(args: argparse.Namespace) -> SoftwareSocState:
    if args.software_capacity_ah <= 0.0:
        raise SystemExit("--software-capacity-ah must be > 0")

    initial_pct = _clamp(args.software_initial_soc_pct, 0.0, 100.0)
    state = SoftwareSocState(
        remaining_ah=(args.software_capacity_ah * initial_pct) / 100.0,
        capacity_ah=args.software_capacity_ah,
        last_timestamp_s=time.time(),
    )
    _save_state(args.software_state_file, state)
    return state


def _get_state(args: argparse.Namespace) -> SoftwareSocState:
    if args.software_reset_state:
        return _init_or_reset_state(args)

    loaded = _load_state(args.software_state_file, args.software_capacity_ah)
    if loaded is not None:
        return loaded
    return _init_or_reset_state(args)


def _update_software_soc(
    state: SoftwareSocState,
    corrected_current_a: float,
    now_s: float,
    max_gap_s: float,
) -> tuple[int, float, float]:
    if max_gap_s < 0.0:
        max_gap_s = 0.0

    dt_s = max(0.0, now_s - state.last_timestamp_s)
    if dt_s <= max_gap_s:
        state.remaining_ah = _clamp(
            state.remaining_ah + (corrected_current_a * (dt_s / 3600.0)),
            0.0,
            state.capacity_ah,
        )

    state.last_timestamp_s = now_s
    soc_pct = int(round((state.remaining_ah / state.capacity_ah) * 100.0))
    return soc_pct, state.remaining_ah, dt_s


def _print_human(
    data: base_battery.BatteryGaugeData,
    gauge_current_a: float,
    corrected_current_a: float,
    software_soc_pct: int | None,
    software_remaining_ah: float | None,
    software_capacity_ah: float | None,
) -> None:
    print(f"Voltage Cmd            : {data.voltage_cmd_v:.3f} V")
    print(f"VoltScale              : {data.volt_scale:d}")
    print(f"Voltage*VoltScale      : {data.voltage_times_scale_v:.3f} V")
    print(f"Voltage Mode Used      : {data.voltage_mode_used}")
    print(f"BAT Pin Voltage        : {data.bat_pin_voltage_v:.3f} V")
    print(f"Pack Voltage (est)     : {data.pack_voltage_est_v:.3f} V")
    print(f"Current (Gauge Raw)    : {gauge_current_a:+.3f} A")
    print(f"Current (Corrected)    : {corrected_current_a:+.3f} A")
    print(f"Temperature            : {data.temperature_c:.2f} C")
    print(f"SoC (Gauge)            : {data.state_of_charge_pct:d} %")
    print(f"Max Error              : {data.max_error_pct:d} %")
    print(f"Remaining Cap. (Gauge) : {data.remaining_capacity_ah:.3f} Ah")
    print(f"Full Charge Cap. (Gauge): {data.full_charge_capacity_ah:.3f} Ah")
    if software_soc_pct is not None and software_remaining_ah is not None and software_capacity_ah is not None:
        print(f"SoC (Software)         : {software_soc_pct:d} %")
        print(f"Remaining Cap. (Soft)  : {software_remaining_ah:.3f} Ah")
        print(f"Full Charge Cap. (Soft): {software_capacity_ah:.3f} Ah")
    print(f"Flags                  : {data.flags_hex}")
    print(f"Shunt Correction       : x{data.shunt_correction_factor:.4f}")


def main() -> int:
    args = _build_parser().parse_args()
    gauge = base_battery.BQ34Z100(bus=args.bus, address=args.address)

    if _run_raw_mode(args, gauge):
        return 0

    state: SoftwareSocState | None = None
    if not args.software_disable:
        state = _get_state(args)

    while True:
        data = gauge.read(
            configured_rsense_mohm=args.configured_rsense_mohm,
            actual_rsense_mohm=args.actual_rsense_mohm,
            swap_word_bytes=args.swap_word_bytes,
            divider_top_kohm=args.divider_top_kohm,
            divider_bottom_kohm=args.divider_bottom_kohm,
            voltage_mode=args.voltage_mode,
        )

        gauge_current_a = data.current_a
        corrected_current_a = gauge_current_a if args.no_invert_current else -gauge_current_a
        data.current_a = corrected_current_a

        software_soc_pct: int | None = None
        software_remaining_ah: float | None = None
        integrate_dt_s: float | None = None
        if state is not None:
            now_s = time.time()
            software_soc_pct, software_remaining_ah, integrate_dt_s = _update_software_soc(
                state=state,
                corrected_current_a=corrected_current_a,
                now_s=now_s,
                max_gap_s=args.software_max_integrate_gap_s,
            )
            _save_state(args.software_state_file, state)

        if args.json:
            payload: dict[str, Any] = asdict(data)
            payload["gauge_current_a"] = gauge_current_a
            payload["corrected_current_a"] = corrected_current_a
            payload["software_soc_enabled"] = state is not None
            if state is not None and software_soc_pct is not None and software_remaining_ah is not None:
                payload["software_soc_pct"] = software_soc_pct
                payload["software_remaining_capacity_ah"] = software_remaining_ah
                payload["software_full_charge_capacity_ah"] = state.capacity_ah
                if integrate_dt_s is not None:
                    payload["software_integrated_dt_s"] = integrate_dt_s
            print(json.dumps(payload, separators=(",", ":")))
        else:
            _print_human(
                data=data,
                gauge_current_a=gauge_current_a,
                corrected_current_a=corrected_current_a,
                software_soc_pct=software_soc_pct,
                software_remaining_ah=software_remaining_ah,
                software_capacity_ah=(state.capacity_ah if state is not None else None),
            )

        if args.watch <= 0.0:
            break
        time.sleep(args.watch)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
