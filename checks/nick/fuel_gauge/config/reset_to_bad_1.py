# Reset BQ34Z100-R2 data flash to CUSTOM_1 preset (dry-run by default).
# Requires: pip install smbus2

from __future__ import annotations

import argparse
import time
from smbus2 import SMBus

CUSTOM_1_PRESET = {
    0x40: {
        0: [
            0x41, 0xD9, 0xAF, 0x37, 0x00, 0x00, 0x00, 0x01,
            0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ],
    },
    0x52: {
        0: [
            0x03, 0xE8, 0x00, 0x00, 0x00, 0x10, 0x68, 0xFE,
            0xD5, 0xFB, 0x95, 0x00, 0x02, 0x00, 0x14, 0x03,
            0xE8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ],
    },
    0x53: {
        0: [
            0x01, 0x07, 0x10, 0x63, 0x10, 0x48, 0x10, 0x2D,
            0x10, 0x15, 0x0F, 0xFC, 0x0F, 0xE6, 0x0F, 0xD0,
            0x0F, 0xBC, 0x0F, 0xA8, 0x0F, 0x96, 0x0F, 0x84,
            0x0F, 0x74, 0x0F, 0x65, 0x0F, 0x56, 0x0F, 0x45,
        ],
    },
    0x54: {
        0: [
            0xFF, 0x2B, 0xFF, 0x41, 0xFF, 0x56, 0xFF, 0x61,
            0xFF, 0x67, 0xFF, 0x3B, 0xFF, 0x16, 0xFF, 0x21,
            0xFF, 0x21, 0xFE, 0xEB, 0xFE, 0xB5, 0xFE, 0x7F,
            0xFE, 0x5E, 0xFE, 0x7F, 0xFE, 0xE8, 0x00, 0x02,
        ],
    },
    0x55: {
        0: [
            0xFF, 0x65, 0xFF, 0xAC, 0xFF, 0x98, 0xFF, 0x75,
            0xFF, 0xBB, 0xFF, 0x82, 0xFF, 0x93, 0xFF, 0xBB,
            0xFF, 0xD5, 0xFF, 0xE4, 0xFF, 0xCE, 0xFF, 0xAD,
            0x00, 0x80, 0xFF, 0x73, 0x00, 0x00, 0x00, 0x00,
        ],
    },
}

I2C_BUS = 1
BQ_ADDR = 0x55

CMD_CONTROL = 0x00
CMD_BLOCK_DATA_CONTROL = 0x61
CMD_DATA_FLASH_CLASS = 0x3E
CMD_DATA_FLASH_BLOCK = 0x3F
CMD_BLOCK_DATA = 0x40
CMD_BLOCK_DATA_CHECKSUM = 0x60

SUBCMD_CONTROL_STATUS = 0x0000
SUBCMD_RESET = 0x0041
SUBCMD_SEAL = 0x0020

UNSEAL_KEY = 0x36720414


def _write_control_word(bus: SMBus, subcmd: int) -> None:
    bus.write_i2c_block_data(BQ_ADDR, CMD_CONTROL, [subcmd & 0xFF, (subcmd >> 8) & 0xFF])


def _read_control_word(bus: SMBus, subcmd: int) -> int:
    _write_control_word(bus, subcmd)
    data = bus.read_i2c_block_data(BQ_ADDR, CMD_CONTROL, 2)
    return data[0] | (data[1] << 8)


def _unseal(bus: SMBus) -> None:
    lsw = UNSEAL_KEY & 0xFFFF
    msw = (UNSEAL_KEY >> 16) & 0xFFFF
    _write_control_word(bus, lsw)
    _write_control_word(bus, msw)
    time.sleep(0.02)


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
    time.sleep(0.25)


def _diff_block(current: list[int], target: list[int]) -> list[int]:
    return [i for i, (c, t) in enumerate(zip(current, target)) if c != t]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset BQ34Z100-R2 data flash to CUSTOM_1_PRESET (dry-run by default)."
    )
    parser.add_argument("--write", action="store_true", help="Apply changes to the gauge.")
    parser.add_argument("--seal", action="store_true", help="Seal after writing.")
    args = parser.parse_args()

    with SMBus(I2C_BUS) as bus:
        status = _read_control_word(bus, SUBCMD_CONTROL_STATUS)
        sealed = bool(status & (1 << 13))
        if sealed:
            print("Gauge appears SEALED. Attempting unseal.")

        if args.write:
            _unseal(bus)

        changes = 0
        for subclass, blocks in CUSTOM_1_PRESET.items():
            for block, target in blocks.items():
                current = _read_block(bus, subclass, block)
                diff_idx = _diff_block(current, target)
                if diff_idx:
                    changes += 1
                    print(f"Subclass {subclass:#04x} Block {block} will change at offsets: {diff_idx}")
                    if args.write:
                        _write_block(bus, subclass, block, target)
                        print(f"Applied subclass {subclass:#04x} block {block}")
                else:
                    print(f"Subclass {subclass:#04x} Block {block}: no changes needed")

        if not changes:
            print("No changes required.")

        if args.write:
            _write_control_word(bus, SUBCMD_RESET)
            time.sleep(0.2)
            if args.seal:
                _write_control_word(bus, SUBCMD_SEAL)


if __name__ == "__main__":
    main()
