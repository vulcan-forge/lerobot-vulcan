# Read BQ34Z100 voltage with 249k/16.5k divider and output pack voltage.
# Requires: pip install smbus2

from __future__ import annotations

import argparse
import statistics
import time
from smbus2 import SMBus

I2C_BUS = 1
BQ_ADDR = 0x55
CMD_VOLTAGE = 0x08

R_TOP_OHMS = 249_000.0
R_BOTTOM_OHMS = 16_500.0
V_DIV_RATIO = R_BOTTOM_OHMS / (R_TOP_OHMS + R_BOTTOM_OHMS)


def _read_voltage_bytes(bus: SMBus) -> tuple[int, int]:
    # Write command pointer, short delay, read 2 bytes.
    bus.write_byte(BQ_ADDR, CMD_VOLTAGE)
    time.sleep(0.005)
    b = bus.read_i2c_block_data(BQ_ADDR, CMD_VOLTAGE, 2)
    return b[0], b[1]


def _decode(b0: int, b1: int, byteorder: str) -> int:
    if byteorder == "le":
        return b0 | (b1 << 8)
    if byteorder == "be":
        return (b0 << 8) | b1
    raise ValueError("byteorder must be 'le' or 'be'")


def _score_pack(pack_v: float, expected_v: float) -> float:
    # Prefer plausible range and closeness to expected.
    if not (6.0 <= pack_v <= 20.0):
        return 0.0
    # Higher score for closer to expected.
    return 1.0 / (1.0 + abs(pack_v - expected_v))


def _score_bat(bat_v: float, expected_bat: float) -> float:
    # BAT pin should be small (sub‑2V). Prefer closeness to expected BAT.
    if not (0.1 <= bat_v <= 2.0):
        return 0.0
    return 1.0 / (1.0 + abs(bat_v - expected_bat))


def _pick_pack_voltage(
    b0: int,
    b1: int,
    expected_v: float,
    byteorder: str | None,
    assume: str | None,
) -> tuple[float, str]:
    # Try LE and BE, and for each treat raw as pack mV or BAT mV.
    candidates = []
    expected_bat = expected_v * V_DIV_RATIO
    orders = [byteorder] if byteorder else ["le", "be"]
    for bo in orders:
        raw = _decode(b0, b1, bo)
        if raw == 0xFFFF:
            continue
        raw_v = raw / 1000.0
        if assume in (None, "pack"):
            pack_v = raw_v
            score = _score_pack(pack_v, expected_v)
            candidates.append((pack_v, f"{bo}-pack", score))
        if assume in (None, "bat"):
            bat_v = raw_v
            pack_v = bat_v / V_DIV_RATIO
            score = _score_bat(bat_v, expected_bat)
            candidates.append((pack_v, f"{bo}-bat", score))

    if not candidates:
        return float("nan"), "invalid"

    best = max(candidates, key=lambda c: c[2])
    return best[0], best[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Read BQ34Z100 pack voltage using divider.")
    parser.add_argument("--samples", type=int, default=15, help="Number of samples to take.")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between samples (s).")
    parser.add_argument("--expected-v", type=float, default=12.8, help="Expected pack voltage (V).")
    parser.add_argument("--byteorder", choices=["le", "be"], default=None, help="Force byteorder.")
    parser.add_argument("--assume", choices=["pack", "bat"], default=None, help="Assume raw is pack or bat mV.")
    args = parser.parse_args()

    pack_vals = []
    modes = {}
    bad = 0

    with SMBus(I2C_BUS) as bus:
        for _ in range(args.samples):
            try:
                b0, b1 = _read_voltage_bytes(bus)
                if b0 == 0xFF and b1 == 0xFF:
                    bad += 1
                else:
                    pack_v, mode = _pick_pack_voltage(
                        b0,
                        b1,
                        args.expected_v,
                        args.byteorder,
                        args.assume,
                    )
                    if pack_v == pack_v:  # not NaN
                        pack_vals.append(pack_v)
                        modes[mode] = modes.get(mode, 0) + 1
                    else:
                        bad += 1
            except OSError:
                bad += 1
            time.sleep(args.delay)

    print(f"Samples: {args.samples}, bad: {bad}")
    if pack_vals:
        med = statistics.median(pack_vals)
        print(f"Pack voltage (median): {med:.2f} V")
        print(f"BAT estimate: {med * V_DIV_RATIO * 1000:.0f} mV")
        print(f"Mode counts: {modes}")
    else:
        print("No valid samples.")


if __name__ == "__main__":
    main()
