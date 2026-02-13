#!/usr/bin/env python3
"""
bq34z100_current.py

Step 1: Probe a few likely current registers and show which one responds to load.
Step 2: Once identified, run in "read" mode to show current continuously.

Discharge is typically NEGATIVE, charge is POSITIVE (we will confirm by testing).

Usage:
  # Probe likely registers (motors off, then on)
  python bq34z100_current.py --probe --loop --delay 0.5

  # Read a specific register continuously (after you identify it)
  python bq34z100_current.py --reg 0x0A --loop --delay 0.5

  # One-shot read
  python bq34z100_current.py --reg 0x0A
"""

from __future__ import annotations
import argparse
import time
from smbus2 import SMBus

DEFAULT_BUS = 1
DEFAULT_ADDR = 0x55

# Common SBS-ish candidates for current/avg current on many TI gauges.
# Your device may differ, so we probe a small set.
PROBE_REGS = [0x0A, 0x0B, 0x14, 0x15, 0x18]  # includes "Current", "AverageCurrent"-like slots on many gauges


def parse_int_auto(s: str) -> int:
    return int(s, 0)


def read_word_le(bus: SMBus, addr: int, cmd: int, delay_s: float = 0.002) -> int:
    """
    Read 16-bit word where device returns LSB then MSB (little-endian on the wire).
    This matched your voltage register behavior (0x08 -> 0x02B6 etc).
    """
    bus.write_byte(addr, cmd)
    time.sleep(delay_s)
    b0, b1 = bus.read_i2c_block_data(addr, cmd, 2)
    return b0 | (b1 << 8)


def to_signed16(u: int) -> int:
    u &= 0xFFFF
    return u - 0x10000 if u & 0x8000 else u


def read_current_mA(bus: SMBus, addr: int, reg: int) -> int:
    """
    Interpret the register as signed 16-bit milliamps.
    Many TI gauges report current in mA as a signed int16.
    """
    raw = read_word_le(bus, addr, reg)
    return to_signed16(raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bus", type=int, default=DEFAULT_BUS)
    ap.add_argument("--addr", type=parse_int_auto, default=DEFAULT_ADDR)

    ap.add_argument("--probe", action="store_true", help="probe likely current registers")
    ap.add_argument("--reg", type=parse_int_auto, default=None, help="read this register as current (e.g. 0x0A)")

    ap.add_argument("--loop", action="store_true", help="loop output")
    ap.add_argument("--delay", type=float, default=0.5)

    args = ap.parse_args()

    if not args.probe and args.reg is None:
        raise SystemExit("Use --probe or specify --reg 0x..")

    regs = PROBE_REGS if args.probe else [args.reg]

    def print_line() -> None:
        with SMBus(args.bus) as bus:
            parts = []
            for r in regs:
                try:
                    mA = read_current_mA(bus, args.addr, r)
                    parts.append(f"reg 0x{r:02X}: {mA:6d} mA")
                except OSError as e:
                    parts.append(f"reg 0x{r:02X}: ERROR ({e})")
        print(" | ".join(parts))

    if not args.loop:
        print_line()
        return

    while True:
        print_line()
        time.sleep(args.delay)


if __name__ == "__main__":
    main()
