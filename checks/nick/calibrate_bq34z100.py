# Software calibration helper for BQ34Z100-R2.
# Requires: pip install smbus2
#
# This script automates:
# - CC Offset calibration
# - Board Offset calibration
# - Voltage Divider calibration (proportional adjustment)
#
# References (primary):
# - BQ34Z100-R2 TRM (SLUUCO5A) for CAL_ENABLE/ENTER_CAL/EXIT_CAL, CC_OFFSET,
#   BOARD_OFFSET, CC_OFFSET_SAVE, ControlStatus bits, and Calibration Data fields.
# - bq34z100EVM User's Guide (SLUU904B) for calibration workflow steps.

from __future__ import annotations

import argparse
import time
from smbus2 import SMBus

I2C_BUS = 1
BQ_ADDR = 0x55

# Standard commands
CMD_CONTROL = 0x00
CMD_VOLTAGE = 0x08

# Data flash access
CMD_BLOCK_DATA_CONTROL = 0x61
CMD_DATA_FLASH_CLASS = 0x3E
CMD_DATA_FLASH_BLOCK = 0x3F
CMD_BLOCK_DATA = 0x40
CMD_BLOCK_DATA_CHECKSUM = 0x60

# Control subcommands (TRM)
SUBCMD_CONTROL_STATUS = 0x0000
SUBCMD_BOARD_OFFSET = 0x0009
SUBCMD_CC_OFFSET = 0x000A
SUBCMD_CC_OFFSET_SAVE = 0x000B
SUBCMD_CAL_ENABLE = 0x002D
SUBCMD_RESET = 0x0041
SUBCMD_EXIT_CAL = 0x0080
SUBCMD_ENTER_CAL = 0x0081

# Data flash fields (TRM, Calibration Data class 104)
DF_VOLTAGE_DIVIDER = (104, 14, 2)  # U2, mV


def _write_control_word(bus: SMBus, subcmd: int) -> None:
    bus.write_i2c_block_data(BQ_ADDR, CMD_CONTROL, [subcmd & 0xFF, (subcmd >> 8) & 0xFF])


def _read_control_word(bus: SMBus, subcmd: int) -> int:
    _write_control_word(bus, subcmd)
    data = bus.read_i2c_block_data(BQ_ADDR, CMD_CONTROL, 2)
    return data[0] | (data[1] << 8)


def _set_dataflash_class_block(bus: SMBus, subclass: int, block: int) -> None:
    bus.write_byte_data(BQ_ADDR, CMD_BLOCK_DATA_CONTROL, 0x00)
    bus.write_byte_data(BQ_ADDR, CMD_DATA_FLASH_CLASS, subclass)
    bus.write_byte_data(BQ_ADDR, CMD_DATA_FLASH_BLOCK, block)
    time.sleep(0.01)


def _read_block(bus: SMBus, subclass: int, block: int) -> list[int]:
    _set_dataflash_class_block(bus, subclass, block)
    data = bus.read_i2c_block_data(BQ_ADDR, CMD_BLOCK_DATA, 32)
    return list(data)


def _write_block(bus: SMBus, subclass: int, block: int, data: list[int]) -> None:
    _set_dataflash_class_block(bus, subclass, block)
    for i, b in enumerate(data):
        bus.write_byte_data(BQ_ADDR, CMD_BLOCK_DATA + i, b & 0xFF)
    checksum = (255 - (sum(data) & 0xFF)) & 0xFF
    bus.write_byte_data(BQ_ADDR, CMD_BLOCK_DATA_CHECKSUM, checksum)
    # TRM recommends delay after flash writes.
    time.sleep(0.25)


def _get_block_offset(offset: int) -> tuple[int, int]:
    block = offset // 32
    in_block = offset % 32
    return block, in_block


def _get_u2(data: list[int], offset: int) -> int:
    return data[offset] | (data[offset + 1] << 8)


def _set_u2(data: list[int], offset: int, value: int) -> None:
    data[offset] = value & 0xFF
    data[offset + 1] = (value >> 8) & 0xFF


def _read_voltage_mV(bus: SMBus) -> int:
    b = bus.read_i2c_block_data(BQ_ADDR, CMD_VOLTAGE, 2)
    return b[0] | (b[1] << 8)


def _enter_cal(bus: SMBus) -> None:
    _write_control_word(bus, SUBCMD_CAL_ENABLE)
    _write_control_word(bus, SUBCMD_ENTER_CAL)


def _exit_cal(bus: SMBus) -> None:
    _write_control_word(bus, SUBCMD_EXIT_CAL)
    _write_control_word(bus, SUBCMD_CAL_ENABLE)


def _wait_for_bits_clear(bus: SMBus, mask: int, timeout_s: float = 90.0) -> None:
    start = time.time()
    while True:
        status = _read_control_word(bus, SUBCMD_CONTROL_STATUS)
        high = (status >> 8) & 0xFF
        if (high & mask) == 0:
            return
        if time.time() - start > timeout_s:
            raise TimeoutError("Timeout waiting for calibration bits to clear")
        time.sleep(0.2)


def calibrate_cc_offset(bus: SMBus) -> None:
    _enter_cal(bus)
    _write_control_word(bus, SUBCMD_CC_OFFSET)
    _wait_for_bits_clear(bus, mask=0x08)  # CCA bit (bit 3)
    _write_control_word(bus, SUBCMD_CC_OFFSET_SAVE)
    _exit_cal(bus)


def calibrate_board_offset(bus: SMBus) -> None:
    _enter_cal(bus)
    _write_control_word(bus, SUBCMD_BOARD_OFFSET)
    _wait_for_bits_clear(bus, mask=0x04)  # BCA bit (bit 2)
    _exit_cal(bus)


def calibrate_voltage_divider(bus: SMBus, applied_mV: int) -> None:
    subclass, offset, _size = DF_VOLTAGE_DIVIDER
    block, in_block = _get_block_offset(offset)
    data = _read_block(bus, subclass, block)
    old_div = _get_u2(data, in_block)
    measured_mV = _read_voltage_mV(bus)
    if measured_mV <= 0:
        raise ValueError("Measured voltage is invalid (<=0).")
    # Proportional adjustment (inference): scale divider so measured matches applied.
    new_div = int(round(old_div * (applied_mV / measured_mV)))
    _set_u2(data, in_block, new_div)
    _write_block(bus, subclass, block, data)
    print(f"Voltage Divider updated: {old_div} -> {new_div} (applied={applied_mV} mV, measured={measured_mV} mV)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Software calibration for BQ34Z100-R2.")
    parser.add_argument("--cc-offset", action="store_true", help="Run CC offset calibration (no load).")
    parser.add_argument("--board-offset", action="store_true", help="Run board offset calibration (no load).")
    parser.add_argument(
        "--voltage-cal-mV",
        type=int,
        help="Calibrate voltage divider using applied pack voltage (mV).",
    )
    parser.add_argument("--reset", action="store_true", help="Reset gauge after calibration.")
    args = parser.parse_args()

    if not (args.cc_offset or args.board_offset or args.voltage_cal_mV):
        raise SystemExit("No calibration action specified.")

    with SMBus(I2C_BUS) as bus:
        if args.cc_offset:
            print("Running CC offset calibration (no load).")
            calibrate_cc_offset(bus)
        if args.board_offset:
            print("Running board offset calibration (no load).")
            calibrate_board_offset(bus)
        if args.voltage_cal_mV:
            print(f"Running voltage calibration using applied {args.voltage_cal_mV} mV.")
            calibrate_voltage_divider(bus, args.voltage_cal_mV)
        if args.reset:
            _write_control_word(bus, SUBCMD_RESET)


if __name__ == "__main__":
    main()
