#!/usr/bin/env python3
# Calibrate current offsets for BQ34Z100 (CC Offset / Board Offset).
# Follows TI procedure using Control() subcommands:
#   CAL_ENABLE (0x002D), ENTER_CAL (0x0081), CC_OFFSET (0x000A),
#   CC_OFFSET_SAVE (0x000B), EXIT_CAL (0x0080), CAL_ENABLE (0x002D).
# Poll CONTROL_STATUS (0x0000) for CCA/BCA bits to clear.
#
# IMPORTANT: Run with zero current flowing through the shunt.
# Dry-run by default. Use --write to execute.

from __future__ import annotations

import argparse
import time
from smbus2 import SMBus, i2c_msg

I2C_BUS_DEFAULT = 1
BQ_ADDR_DEFAULT = 0x55

# Control() register and subcommands
CMD_CONTROL = 0x00
SUB_CAL_ENABLE = 0x002D
SUB_ENTER_CAL = 0x0081
SUB_EXIT_CAL = 0x0080
SUB_CC_OFFSET = 0x000A
SUB_CC_OFFSET_SAVE = 0x000B

# CONTROL_STATUS bits (high byte)
CCA_BIT = 0x0800  # high byte bit 3
BCA_BIT = 0x0400  # high byte bit 2


def _write_control(bus: SMBus, subcmd: int) -> None:
    # Write subcommand LSB first to Control() (0x00)
    lo = subcmd & 0xFF
    hi = (subcmd >> 8) & 0xFF
    write = i2c_msg.write(BQ_ADDR, [CMD_CONTROL, lo, hi])
    bus.i2c_rdwr(write)


def _read_control_status(bus: SMBus) -> int:
    # Read Control() status word (0x0000)
    write = i2c_msg.write(BQ_ADDR, [CMD_CONTROL, 0x00, 0x00])
    read = i2c_msg.read(BQ_ADDR, 2)
    bus.i2c_rdwr(write, read)
    b = list(read)
    return b[0] | (b[1] << 8)


def _poll_clear(bus: SMBus, mask: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = _read_control_status(bus)
        if (status & mask) == 0:
            return
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for bits 0x{mask:04X} to clear (status=0x{status:04X})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate BQ34Z100 current offsets (CC/Board).")
    ap.add_argument("--bus", type=int, default=I2C_BUS_DEFAULT)
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=BQ_ADDR_DEFAULT)
    ap.add_argument("--cc-offset", action="store_true", help="Run CC offset calibration.")
    ap.add_argument("--board-offset", action="store_true", help="Run board offset calibration.")
    ap.add_argument("--timeout", type=float, default=60.0, help="Timeout seconds per step.")
    ap.add_argument("--write", action="store_true", help="Execute calibration (otherwise dry-run).")
    args = ap.parse_args()

    if not (args.cc_offset or args.board_offset):
        args.cc_offset = True
        args.board_offset = True

    global BQ_ADDR
    BQ_ADDR = args.addr

    print("Calibration steps:")
    print(f"  CC Offset: {args.cc_offset}")
    print(f"  Board Offset: {args.board_offset}")
    print("  NOTE: Ensure ZERO current through the shunt.")

    if not args.write:
        print("Dry-run only. Re-run with --write to execute.")
        return

    with SMBus(args.bus) as bus:
        # Enable calibration and enter calibration mode
        _write_control(bus, SUB_CAL_ENABLE)
        _write_control(bus, SUB_ENTER_CAL)

        # CC offset calibration
        if args.cc_offset:
            _write_control(bus, SUB_CC_OFFSET)
            _poll_clear(bus, CCA_BIT, args.timeout)
            _write_control(bus, SUB_CC_OFFSET_SAVE)

        # Board offset calibration (poll both CCA and BCA)
        if args.board_offset:
            _write_control(bus, SUB_CC_OFFSET)
            _poll_clear(bus, CCA_BIT | BCA_BIT, args.timeout)
            _write_control(bus, SUB_CC_OFFSET_SAVE)

        # Exit calibration mode
        _write_control(bus, SUB_EXIT_CAL)
        _write_control(bus, SUB_CAL_ENABLE)

    print("Calibration complete.")


if __name__ == "__main__":
    main()
