#!/usr/bin/env python3
"""
bq34z100_current_only.py

Reads BQ34Z100 current from register 0x0A (based on your probe results).
Negative = discharging, Positive = charging (verify by plugging charger / PSU behavior).

Usage:
  python checks/nick/bq34z100_current_only.py
  python checks/nick/bq34z100_current_only.py --loop --delay 0.2
  python checks/nick/bq34z100_current_only.py --loop --delay 0.2 --window 25
"""

from __future__ import annotations
import argparse
import time
from collections import deque
from smbus2 import SMBus

DEFAULT_BUS = 1
DEFAULT_ADDR = 0x55
CURRENT_REG = 0x0A  # chosen from your probe: ~ -795 mA stable


def parse_int_auto(s: str) -> int:
    return int(s, 0)


def to_signed16(u: int) -> int:
    u &= 0xFFFF
    return u - 0x10000 if u & 0x8000 else u


def read_word_le(bus: SMBus, addr: int, cmd: int, delay_s: float = 0.002) -> int:
    """
    Read 16-bit word (LSB then MSB). Matches your BQ behavior.
    """
    bus.write_byte(addr, cmd)
    time.sleep(delay_s)
    b0, b1 = bus.read_i2c_block_data(addr, cmd, 2)
    return b0 | (b1 << 8)


def read_current_mA(bus: SMBus, addr: int) -> int:
    raw = read_word_le(bus, addr, CURRENT_REG)
    return to_signed16(raw)


def status_from_mA(mA: float, deadband_mA: float) -> str:
    if mA > deadband_mA:
        return "CHARGING"
    if mA < -deadband_mA:
        return "DISCHARGING"
    return "IDLE"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bus", type=int, default=DEFAULT_BUS)
    ap.add_argument("--addr", type=parse_int_auto, default=DEFAULT_ADDR)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=15, help="moving average window size")
    ap.add_argument("--deadband", type=float, default=50.0, help="mA deadband for IDLE")
    args = ap.parse_args()

    hist = deque(maxlen=max(1, args.window))

    def print_once(bus: SMBus) -> None:
        mA = read_current_mA(bus, args.addr)
        hist.append(mA)
        avg_mA = sum(hist) / len(hist)

        st = status_from_mA(avg_mA, args.deadband)

        # Nice human units
        inst_A = mA / 1000.0
        avg_A = avg_mA / 1000.0

        print(
            f"I = {mA:7d} mA ({inst_A:+.3f} A) | "
            f"avg({len(hist):02d}) = {avg_mA:8.1f} mA ({avg_A:+.3f} A) | "
            f"{st}"
        )

    if not args.loop:
        with SMBus(args.bus) as bus:
            print_once(bus)
        return

    with SMBus(args.bus) as bus:
        while True:
            try:
                print_once(bus)
            except OSError as e:
                # Keep running if a transient I2C hiccup happens
                print(f"I2C error: {e}")
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
