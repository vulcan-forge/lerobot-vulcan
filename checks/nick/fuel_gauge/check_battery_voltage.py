# Read battery voltage from BQ34Z100-R2 and scale using divider.
# Requires: pip install smbus2

from __future__ import annotations

import argparse
import statistics
import time
from smbus2 import SMBus

I2C_BUS = 1
BQ_ADDR = 0x55
CMD_VOLTAGE = 0x08

# Divider values
R_TOP_OHMS = 249_000.0
R_BOTTOM_OHMS = 16_500.0
V_DIV_RATIO = R_BOTTOM_OHMS / (R_TOP_OHMS + R_BOTTOM_OHMS)


def _read_voltage_bytes(bus: SMBus) -> tuple[int, int]:
    # Write command pointer, then read 2 bytes.
    bus.write_byte(BQ_ADDR, CMD_VOLTAGE)
    time.sleep(0.005)
    b = bus.read_i2c_block_data(BQ_ADDR, CMD_VOLTAGE, 2)
    return b[0], b[1]


def _decode_voltage_mV(b0: int, b1: int, byteorder: str) -> int:
    if byteorder == "le":
        return b0 | (b1 << 8)
    if byteorder == "be":
        return (b0 << 8) | b1
    raise ValueError("byteorder must be 'le' or 'be'")


def _score_mV(v: int) -> int:
    # Prefer plausible pack mV (2V–20V), then BAT pin mV (0.2V–2V).
    if 2000 <= v <= 20000:
        return 2
    if 200 <= v <= 2000:
        return 1
    return 0


def read_voltage(samples: int, delay_s: float, byteorder: str | None) -> None:
    pack_vals = []
    bat_vals = []
    bad = 0

    with SMBus(I2C_BUS) as bus:
        for _ in range(samples):
            try:
                b0, b1 = _read_voltage_bytes(bus)
                # Reject obvious bad reads
                if b0 == 0xFF and b1 == 0xFF:
                    bad += 1
                    time.sleep(delay_s)
                    continue

                if byteorder:
                    mv = _decode_voltage_mV(b0, b1, byteorder)
                else:
                    mv_le = _decode_voltage_mV(b0, b1, "le")
                    mv_be = _decode_voltage_mV(b0, b1, "be")
                    mv = mv_be if _score_mV(mv_be) > _score_mV(mv_le) else mv_le

                # Classify as pack or bat for reporting
                if 2000 <= mv <= 20000:
                    pack_vals.append(mv / 1000.0)
                else:
                    bat_vals.append(mv / 1000.0)
            except OSError:
                bad += 1
            time.sleep(delay_s)

    print(f"Samples: {samples}, bad: {bad}")
    if pack_vals:
        med_pack = statistics.median(pack_vals)
        print(f"Pack median: {med_pack:.2f} V (n={len(pack_vals)})")
        print(f"BAT est: {med_pack * V_DIV_RATIO * 1000:.0f} mV")
    if bat_vals:
        med_bat = statistics.median(bat_vals)
        print(f"BAT median: {med_bat * 1000:.0f} mV (n={len(bat_vals)})")
        print(f"Pack est: {med_bat / V_DIV_RATIO:.2f} V")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read BQ34Z100 voltage and scale with divider.")
    parser.add_argument("--samples", type=int, default=15, help="Number of samples to take.")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between samples in seconds.")
    parser.add_argument(
        "--byteorder",
        choices=["le", "be"],
        default=None,
        help="Force byte order (le/be). Default: auto.",
    )
    args = parser.parse_args()

    read_voltage(samples=args.samples, delay_s=args.delay, byteorder=args.byteorder)


if __name__ == "__main__":
    main()
