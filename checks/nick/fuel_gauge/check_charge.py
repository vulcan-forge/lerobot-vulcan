#!/usr/bin/env python3
# Read charge-related metrics from BQ34Z100 over I2C.
# Uses repeated-start reads to avoid byte-swap issues.

from __future__ import annotations

import argparse
from smbus2 import SMBus, i2c_msg

I2C_BUS_DEFAULT = 1
BQ_ADDR_DEFAULT = 0x55

# Commands (based on your prior reads + SBS defaults)
CMD_VOLTAGE = 0x08
CMD_TEMPERATURE = 0x0C
CMD_CURRENT = 0x0A
CMD_AVG_CURRENT = 0x0B
CMD_SOC_ALT = 0x02   # you previously read SOC here
CMD_SOC = 0x0D       # SBS Relative State of Charge
CMD_REMAINING = 0x0F
CMD_FULL = 0x10


def _read_word(bus: SMBus, cmd: int) -> int:
    write = i2c_msg.write(BQ_ADDR, [cmd])
    read = i2c_msg.read(BQ_ADDR, 2)
    bus.i2c_rdwr(write, read)
    b = list(read)
    return b[0] | (b[1] << 8)


def _s16(x: int) -> int:
    return x - 0x10000 if x & 0x8000 else x


def main() -> None:
    ap = argparse.ArgumentParser(description="Read charge metrics from BQ34Z100.")
    ap.add_argument("--bus", type=int, default=I2C_BUS_DEFAULT)
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=BQ_ADDR_DEFAULT)
    args = ap.parse_args()

    global BQ_ADDR
    BQ_ADDR = args.addr

    with SMBus(args.bus) as bus:
        voltage_mv = _read_word(bus, CMD_VOLTAGE)
        temp_dK = _read_word(bus, CMD_TEMPERATURE)
        curr_ma = _s16(_read_word(bus, CMD_CURRENT))
        avg_ma = _s16(_read_word(bus, CMD_AVG_CURRENT))
        soc_alt = _read_word(bus, CMD_SOC_ALT)
        soc = _read_word(bus, CMD_SOC)
        rem_mah = _read_word(bus, CMD_REMAINING)
        full_mah = _read_word(bus, CMD_FULL)

    print(f"Voltage: {voltage_mv} mV")
    print(f"Temperature: {temp_dK/10.0 - 273.15:.1f} C ({temp_dK} in 0.1K)")
    print(f"Current: {curr_ma} mA")
    print(f"Avg Current: {avg_ma} mA")
    print(f"SOC (0x02): {soc_alt}")
    print(f"SOC (0x0D): {soc}")
    print(f"Remaining Capacity: {rem_mah} mAh")
    print(f"Full Charge Capacity: {full_mah} mAh")


if __name__ == "__main__":
    main()
