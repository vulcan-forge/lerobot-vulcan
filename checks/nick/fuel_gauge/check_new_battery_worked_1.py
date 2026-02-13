#!/usr/bin/env python3
"""
Read BAT voltage from a BQ34Z100-R2 over I2C.

Example:
  python bq34z100_read_voltage.py
  python bq34z100_read_voltage.py --cmd 0x08
  python bq34z100_read_voltage.py --cmd 0x08 --samples 25 --delay 0.2
"""

from __future__ import annotations

import argparse
import statistics
import time
from smbus2 import SMBus

I2C_BUS_DEFAULT = 1
BQ_ADDR_DEFAULT = 0x55

# Your divider (used only for optional pack estimate)
R_TOP_OHMS_DEFAULT = 249_000.0
R_BOTTOM_OHMS_DEFAULT = 16_500.0


def parse_int_auto(s: str) -> int:
    """Parse ints like '8', '0x08', '55', '0x55'."""
    return int(s, 0)


def read_word_via_i2cget_style(bus: SMBus, addr: int, cmd: int) -> int:
    """
    Robustly read a 16-bit word from devices like TI gauges.

    Many TI fuel gauges return word data as LSB then MSB.
    But some SMBus stacks / helper commands can swap.
    We read two raw bytes directly and reconstruct LSB-first.
    """
    # Issue command pointer then read 2 bytes
    bus.write_byte(addr, cmd)
    time.sleep(0.005)
    b0, b1 = bus.read_i2c_block_data(addr, cmd, 2)

    # TI gauges typically return LSB first => little-endian word:
    raw_le = b0 | (b1 << 8)

    return raw_le


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bus", type=int, default=I2C_BUS_DEFAULT, help="I2C bus number (default: 1)")
    ap.add_argument("--addr", type=parse_int_auto, default=BQ_ADDR_DEFAULT, help="7-bit I2C address (default: 0x55)")
    ap.add_argument("--cmd", type=parse_int_auto, default=0x08, help="command/register to read (default: 0x08)")
    ap.add_argument("--samples", type=int, default=1, help="number of samples to take (default: 1)")
    ap.add_argument("--delay", type=float, default=0.2, help="delay between samples seconds (default: 0.2)")

    ap.add_argument("--rtop", type=float, default=R_TOP_OHMS_DEFAULT, help="top divider resistor in ohms")
    ap.add_argument("--rbottom", type=float, default=R_BOTTOM_OHMS_DEFAULT, help="bottom divider resistor in ohms")
    ap.add_argument("--no-pack-est", action="store_true", help="only print BAT voltage")

    args = ap.parse_args()

    v_div_ratio = args.rbottom / (args.rtop + args.rbottom)

    bat_volts = []
    raw_words = []

    with SMBus(args.bus) as bus:
        for i in range(args.samples):
            raw = read_word_via_i2cget_style(bus, args.addr, args.cmd)

            # In your observed case: i2cget 0x08 w => 0x02B6 => 694 mV
            # So interpret the word as millivolts directly:
            bat_v = raw / 1000.0

            raw_words.append(raw)
            bat_volts.append(bat_v)

            if args.samples > 1 and i < args.samples - 1:
                time.sleep(args.delay)

    # Print single sample or summary
    if args.samples == 1:
        raw = raw_words[0]
        bat_v = bat_volts[0]
        print(f"Addr: 0x{args.addr:02X}  CMD: 0x{args.cmd:02X}")
        print(f"Raw word: 0x{raw:04X} ({raw} mV)")
        print(f"BAT: {bat_v:.3f} V")
        if not args.no_pack_est:
            pack_est = bat_v / v_div_ratio
            print(f"Pack est (divider): {pack_est:.2f} V  (ratio={v_div_ratio:.6f})")
    else:
        med_bat = statistics.median(bat_volts)
        min_bat = min(bat_volts)
        max_bat = max(bat_volts)

        print(f"Addr: 0x{args.addr:02X}  CMD: 0x{args.cmd:02X}")
        print(f"Samples: {args.samples}")
        print(f"BAT median: {med_bat:.3f} V  (min {min_bat:.3f}, max {max_bat:.3f})")

        if not args.no_pack_est:
            pack_est = med_bat / v_div_ratio
            print(f"Pack est (divider, median): {pack_est:.2f} V  (ratio={v_div_ratio:.6f})")


if __name__ == "__main__":
    main()
