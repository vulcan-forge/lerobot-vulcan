#!/usr/bin/env python3
"""
bq34z100_dump.py

Attempts to read:
- Standard SBS-like runtime metrics
- Control/status word(s)
- If possible, a small set of ManufacturerAccess reads
- If possible, tries to read Data Flash blocks (may fail if sealed)

This script is safe: it only READS. No writes.

Usage:
  python checks/nick/bq34z100_dump.py
  python checks/nick/bq34z100_dump.py --loop --delay 1.0
"""

from __future__ import annotations
import argparse
import time
from smbus2 import SMBus

BUS_DEFAULT = 1
ADDR_DEFAULT = 0x55

def parse_int_auto(s: str) -> int:
    return int(s, 0)

def to_signed16(u: int) -> int:
    u &= 0xFFFF
    return u - 0x10000 if u & 0x8000 else u

def read_word_le(bus: SMBus, addr: int, cmd: int, delay_s: float = 0.002) -> int:
    """
    Read 2 bytes after setting cmd pointer; interpret LSB-first.
    Matches what worked for you (0x08 voltage, 0x0A current).
    """
    bus.write_byte(addr, cmd)
    time.sleep(delay_s)
    b0, b1 = bus.read_i2c_block_data(addr, cmd, 2)
    return b0 | (b1 << 8)

def read_bytes(bus: SMBus, addr: int, cmd: int, n: int, delay_s: float = 0.002) -> list[int]:
    bus.write_byte(addr, cmd)
    time.sleep(delay_s)
    return bus.read_i2c_block_data(addr, cmd, n)

def safe_read_word(bus: SMBus, addr: int, cmd: int) -> tuple[bool, int | None, str | None]:
    try:
        return True, read_word_le(bus, addr, cmd), None
    except OSError as e:
        return False, None, str(e)

def safe_read_bytes(bus: SMBus, addr: int, cmd: int, n: int) -> tuple[bool, list[int] | None, str | None]:
    try:
        return True, read_bytes(bus, addr, cmd, n), None
    except OSError as e:
        return False, None, str(e)

def fmt_hex_bytes(bs: list[int]) -> str:
    return " ".join(f"{b:02X}" for b in bs)

def dump_once(bus: SMBus, addr: int) -> None:
    # Known-working on your setup
    VOLT_CMD = 0x08
    CURR_CMD = 0x0A

    ok_v, v_raw, v_err = safe_read_word(bus, addr, VOLT_CMD)
    ok_i, i_raw, i_err = safe_read_word(bus, addr, CURR_CMD)

    print(f"Addr 0x{addr:02X}")

    if ok_v and v_raw is not None:
        print(f"  0x08 (voltage-like): 0x{v_raw:04X} = {v_raw} (raw mV units on your setup)")
    else:
        print(f"  0x08 (voltage-like): ERROR {v_err}")

    if ok_i and i_raw is not None:
        i_ma = to_signed16(i_raw)
        print(f"  0x0A (current-like): 0x{i_raw:04X} = {i_ma} mA (signed)")
    else:
        print(f"  0x0A (current-like): ERROR {i_err}")

    # Probe a few common SBS commands (may or may not map on your device)
    candidates = {
        "Control/MA (0x00)": 0x00,
        "Temperature (0x02)": 0x02,
        "Voltage (0x04)": 0x04,
        "Flags/Status (0x06)": 0x06,
        "AvgCurrent? (0x0B)": 0x0B,
        "SOC? (0x0D)": 0x0D,
        "RemCap? (0x0F)": 0x0F,
        "FullCap? (0x10)": 0x10,
    }

    for name, cmd in candidates.items():
        ok, w, err = safe_read_word(bus, addr, cmd)
        if ok and w is not None:
            s = to_signed16(w)
            print(f"  {name:<18} cmd 0x{cmd:02X}: 0x{w:04X}  unsigned={w:5d}  signed={s:6d}")
        else:
            print(f"  {name:<18} cmd 0x{cmd:02X}: ERROR {err}")

    # Try reading a few bytes from 0x00 (sometimes returns a control word / status)
    okb, bs, errb = safe_read_bytes(bus, addr, 0x00, 2)
    if okb and bs is not None:
        print(f"  0x00 bytes: {fmt_hex_bytes(bs)}")

    # NOTE: True Data Flash reads usually require ManufacturerAccess/DataFlash commands and possibly unsealing.
    # We don't guess those writes here (risk). We only show what is readable safely.
    print("")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bus", type=int, default=BUS_DEFAULT)
    ap.add_argument("--addr", type=parse_int_auto, default=ADDR_DEFAULT)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()

    with SMBus(args.bus) as bus:
        if not args.loop:
            dump_once(bus, args.addr)
            return
        while True:
            dump_once(bus, args.addr)
            time.sleep(args.delay)

if __name__ == "__main__":
    main()
