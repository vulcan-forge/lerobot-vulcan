# checks/nick/check_new_battery.py
# Requires: pip install smbus2

from dataclasses import dataclass
from smbus2 import SMBus

I2C_BUS = 1
BQ_ADDR = 0x55  # 7-bit address
DEBUG = True

# Command codes (standard + extended)
CMD_SOC = 0x02
CMD_REMAINING_CAP = 0x04
CMD_FULL_CAP = 0x06
CMD_VOLTAGE = 0x08
CMD_AVG_CURRENT = 0x0A
CMD_TEMPERATURE = 0x0C

CMD_VOLT_SCALE = 0x20
CMD_CURR_SCALE = 0x21


def _read_word_raw(bus: SMBus, cmd: int) -> tuple[int, int]:
    b = bus.read_i2c_block_data(BQ_ADDR, cmd, 2)
    return b[0], b[1]


def _word_from_bytes(b0: int, b1: int, byteorder: str) -> int:
    if byteorder == "little":
        return b0 | (b1 << 8)
    if byteorder == "big":
        return (b0 << 8) | b1
    raise ValueError(f"unsupported byteorder: {byteorder}")


def _read_word(bus: SMBus, cmd: int, byteorder: str) -> int:
    b0, b1 = _read_word_raw(bus, cmd)
    return _word_from_bytes(b0, b1, byteorder)


def _read_sword(bus: SMBus, cmd: int, byteorder: str) -> int:
    val = _read_word(bus, cmd, byteorder)
    return val - 0x10000 if val & 0x8000 else val


def _read_byte(bus: SMBus, cmd: int) -> int:
    return bus.read_byte_data(BQ_ADDR, cmd)


@dataclass
class BatteryStatus:
    soc_percent: float
    remaining_mAh: float
    full_mAh: float
    voltage_mV: float
    avg_current_mA: float
    temperature_C: float


def _choose_byteorder(soc_b0: int, soc_b1: int, volt_b0: int, volt_b1: int) -> str:
    soc_le = _word_from_bytes(soc_b0, soc_b1, "little")
    soc_be = _word_from_bytes(soc_b0, soc_b1, "big")
    volt_le = _word_from_bytes(volt_b0, volt_b1, "little")
    volt_be = _word_from_bytes(volt_b0, volt_b1, "big")

    score_le = 0
    score_be = 0
    if 0 <= soc_le <= 100:
        score_le += 1
    if 0 <= soc_be <= 100:
        score_be += 1
    if 1000 <= volt_le <= 20000:
        score_le += 1
    if 1000 <= volt_be <= 20000:
        score_be += 1

    return "big" if score_be > score_le else "little"


def read_battery() -> BatteryStatus:
    with SMBus(I2C_BUS) as bus:
        soc_b0, soc_b1 = _read_word_raw(bus, CMD_SOC)
        volt_b0, volt_b1 = _read_word_raw(bus, CMD_VOLTAGE)
        byteorder = _choose_byteorder(soc_b0, soc_b1, volt_b0, volt_b1)

        volt_scale = _read_byte(bus, CMD_VOLT_SCALE) or 1
        curr_scale = _read_byte(bus, CMD_CURR_SCALE) or 1

        soc = _read_word(bus, CMD_SOC, byteorder)
        remaining = _read_word(bus, CMD_REMAINING_CAP, byteorder) * curr_scale
        full = _read_word(bus, CMD_FULL_CAP, byteorder) * curr_scale
        voltage = _read_word(bus, CMD_VOLTAGE, byteorder) * volt_scale
        avg_current = _read_sword(bus, CMD_AVG_CURRENT, byteorder) * curr_scale
        temp_dK = _read_word(bus, CMD_TEMPERATURE, byteorder)  # 0.1 K
        temp_c = (temp_dK / 10.0) - 273.15

        if DEBUG:
            temp_b0, temp_b1 = _read_word_raw(bus, CMD_TEMPERATURE)
            curr_b0, curr_b1 = _read_word_raw(bus, CMD_AVG_CURRENT)
            print("DEBUG raw bytes:")
            print(f"  SOC: 0x{soc_b0:02X} 0x{soc_b1:02X}")
            print(f"  VOLT: 0x{volt_b0:02X} 0x{volt_b1:02X}")
            print(f"  TEMP: 0x{temp_b0:02X} 0x{temp_b1:02X}")
            print(f"  CURR: 0x{curr_b0:02X} 0x{curr_b1:02X}")
            print("DEBUG endian check (SOC/VOLT):")
            print(f"  SOC LE/BE: {soc_le} / {soc_be}")
            print(f"  VOLT LE/BE: {volt_le} / {volt_be}")
            print(f"  Byteorder chosen: {byteorder}")

        return BatteryStatus(
            soc_percent=float(soc),
            remaining_mAh=float(remaining),
            full_mAh=float(full),
            voltage_mV=float(voltage),
            avg_current_mA=float(avg_current),
            temperature_C=float(temp_c),
        )


if __name__ == "__main__":
    status = read_battery()
    print(f"SOC: {status.soc_percent:.1f}%")
    print(f"Remaining: {status.remaining_mAh:.0f} mAh")
    print(f"Full: {status.full_mAh:.0f} mAh")
    print(f"Voltage: {status.voltage_mV:.0f} mV")
    print(f"Avg Current: {status.avg_current_mA:.0f} mA")
    print(f"Temp: {status.temperature_C:.1f} C")
