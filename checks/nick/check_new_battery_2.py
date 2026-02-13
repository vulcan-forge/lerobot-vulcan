#!/usr/bin/env python3
"""
bq34z100_calibrated_voltage.py

Reads BQ34Z100-R2 at I2C addr 0x55, command 0x08 (your working voltage-like register),
then converts the returned word (raw_mV) into a *calibrated pack voltage* using a
linear fit derived from your measured data.

Usage:
  python bq34z100_calibrated_voltage.py
  python bq34z100_calibrated_voltage.py --samples 25 --delay 0.2
  python bq34z100_calibrated_voltage.py --once
  python bq34z100_calibrated_voltage.py --cmd 0x08
"""

from __future__ import annotations

import argparse
import statistics
import time
from smbus2 import SMBus

# --------- I2C config ----------
DEFAULT_BUS = 1
DEFAULT_ADDR = 0x55
DEFAULT_CMD = 0x08  # confirmed by your i2cget returning 0x02b6 etc.

# --------- Divider (optional reporting) ----------
# You said: 249k top, 16.5k bottom
R_TOP_OHMS = 249_000.0
R_BOTTOM_OHMS = 16_500.0
DIV_RATIO = R_BOTTOM_OHMS / (R_TOP_OHMS + R_BOTTOM_OHMS)  # ~0.062147...

# --------- Calibration derived from your table ----------
# Model: V_pack = M * raw_mV + C
# where raw_mV is the word read from CMD 0x08 (e.g. 0x030C -> 780)
CAL_M = 0.018520164994728302
CAL_C = 0.14820830276163136


def parse_int_auto(s: str) -> int:
    """Parse ints like '8', '0x08', '55', '0x55'."""
    return int(s, 0)


def read_word_le(bus: SMBus, addr: int, cmd: int, delay_s: float = 0.005) -> int:
    """
    Read 16-bit word as LSB then MSB (little-endian on the wire), which matched your data:
      bytes B6 02 -> raw 0x02B6 -> 694 mV.
    """
    bus.write_byte(addr, cmd)
    time.sleep(delay_s)
    b0, b1 = bus.read_i2c_block_data(addr, cmd, 2)
    return b0 | (b1 << 8)


def pack_voltage_from_raw(raw_mV: int) -> float:
    """Convert raw word (mV-ish) -> calibrated pack volts."""
    return (CAL_M * float(raw_mV)) + CAL_C


def bat_voltage_est_from_pack(pack_v: float) -> float:
    """Estimate BAT pin voltage (Volts) using the divider ratio."""
    return pack_v * DIV_RATIO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bus", type=int, default=DEFAULT_BUS)
    ap.add_argument("--addr", type=parse_int_auto, default=DEFAULT_ADDR)
    ap.add_argument("--cmd", type=parse_int_auto, default=DEFAULT_CMD)
    ap.add_argument("--samples", type=int, default=1, help="number of samples (default 1)")
    ap.add_argument("--delay", type=float, default=0.2, help="delay between samples seconds")
    ap.add_argument("--once", action="store_true", help="same as --samples 1 (prints one reading)")
    args = ap.parse_args()

    samples = 1 if args.once else max(1, args.samples)

    raw_list: list[int] = []
    pack_list: list[float] = []

    with SMBus(args.bus) as bus:
        for i in range(samples):
            raw = read_word_le(bus, args.addr, args.cmd)
            pack_v = pack_voltage_from_raw(raw)

            raw_list.append(raw)
            pack_list.append(pack_v)

            if samples == 1:
                bat_v_est = bat_voltage_est_from_pack(pack_v)
                print(f"Addr: 0x{args.addr:02X}  CMD: 0x{args.cmd:02X}")
                print(f"Raw: 0x{raw:04X} ({raw} mV)")
                print(f"Pack (cal): {pack_v:.2f} V")
                print(f"BAT  (est): {bat_v_est:.3f} V  (ratio={DIV_RATIO:.6f})")
                return

            if i < samples - 1:
                time.sleep(args.delay)

    # Summary for multi-sample mode
    raw_med = int(statistics.median(raw_list))
    pack_med = statistics.median(pack_list)
    pack_min = min(pack_list)
    pack_max = max(pack_list)

    print(f"Addr: 0x{args.addr:02X}  CMD: 0x{args.cmd:02X}")
    print(f"Samples: {samples}")
    print(f"Raw median: 0x{raw_med:04X} ({raw_med} mV)")
    print(f"Pack median (cal): {pack_med:.2f} V (min {pack_min:.2f}, max {pack_max:.2f})")
    print(f"BAT  est (median): {bat_voltage_est_from_pack(pack_med):.3f} V  (ratio={DIV_RATIO:.6f})")
    print("")
    print("Calibration used:")
    print(f"  Pack_V = {CAL_M:.15f} * raw_mV + {CAL_C:.15f}")


if __name__ == "__main__":
    main()
