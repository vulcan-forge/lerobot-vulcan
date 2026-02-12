# checks/nick/check_new_battery.py
# Requires: pip install smbus2

from dataclasses import dataclass
import time
from smbus2 import SMBus

I2C_BUS = 1
BQ_ADDR = 0x55  # 7-bit address
DEBUG = True

# Voltage divider: top to pack+, bottom to GND, BAT pin at the midpoint.
R_TOP_OHMS = 249_000.0
R_BOTTOM_OHMS = 16_500.0
V_DIV_RATIO = R_BOTTOM_OHMS / (R_TOP_OHMS + R_BOTTOM_OHMS)

# Pack / gauge expectations (for warnings only; does not configure the gauge)
CHEMISTRY = "LiFePO4 4S"
EXPECTED_CAPACITY_MAH = 10000
SHUNT_MILLIOHMS = 12.5

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
    # Use SMBus "read word" style: write command, short delay, then read 2 bytes.
    bus.write_byte(BQ_ADDR, cmd)
    time.sleep(0.005)
    b = bus.read_i2c_block_data(BQ_ADDR, cmd, 2)
    return b[0], b[1]


def _read_word_raw_retry(bus: SMBus, cmd: int, retries: int = 6) -> tuple[int, int]:
    last_err = None
    for _ in range(retries):
        try:
            b0, b1 = _read_word_raw(bus, cmd)
            if b0 == 0xFF and b1 == 0xFF:
                raise OSError("read returned 0xFFFF")
            # read twice and require a consistent value
            b0_2, b1_2 = _read_word_raw(bus, cmd)
            if (b0_2, b1_2) != (b0, b1):
                raise OSError("read mismatch")
            return b0, b1
        except OSError as err:
            last_err = err
            time.sleep(0.02)
    if last_err:
        raise last_err
    return 0, 0


def _word_from_bytes(b0: int, b1: int, byteorder: str) -> int:
    if byteorder == "little":
        return b0 | (b1 << 8)
    if byteorder == "big":
        return (b0 << 8) | b1
    raise ValueError(f"unsupported byteorder: {byteorder}")


def _read_word(bus: SMBus, cmd: int, byteorder: str) -> int:
    b0, b1 = _read_word_raw_retry(bus, cmd)
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
    pack_voltage_V: float
    avg_current_mA: float
    temperature_C: float


def read_battery() -> BatteryStatus:
    with SMBus(I2C_BUS) as bus:
        soc_b0, soc_b1 = _read_word_raw_retry(bus, CMD_SOC)
        volt_b0, volt_b1 = _read_word_raw_retry(bus, CMD_VOLTAGE)
        byteorder = "little"

        volt_scale = _read_byte(bus, CMD_VOLT_SCALE) or 1
        curr_scale = _read_byte(bus, CMD_CURR_SCALE) or 1

        soc_raw = _read_word(bus, CMD_SOC, byteorder)
        soc = soc_raw / 256.0 if soc_raw > 1000 else float(soc_raw)
        remaining = _read_word(bus, CMD_REMAINING_CAP, byteorder) * curr_scale
        full = _read_word(bus, CMD_FULL_CAP, byteorder) * curr_scale
        bat_mV = _read_word(bus, CMD_VOLTAGE, byteorder) * volt_scale
        pack_voltage_V = (bat_mV / 1000.0) / V_DIV_RATIO
        avg_current = _read_sword(bus, CMD_AVG_CURRENT, byteorder) * curr_scale
        temp_dK = _read_word(bus, CMD_TEMPERATURE, byteorder)  # 0.1 K
        temp_c = (temp_dK / 10.0) - 273.15

        if DEBUG:
            temp_b0, temp_b1 = _read_word_raw_retry(bus, CMD_TEMPERATURE)
            curr_b0, curr_b1 = _read_word_raw_retry(bus, CMD_AVG_CURRENT)
            soc_le = _word_from_bytes(soc_b0, soc_b1, "little")
            soc_be = _word_from_bytes(soc_b0, soc_b1, "big")
            volt_le = _word_from_bytes(volt_b0, volt_b1, "little")
            volt_be = _word_from_bytes(volt_b0, volt_b1, "big")
            temp_le = _word_from_bytes(temp_b0, temp_b1, "little")
            temp_be = _word_from_bytes(temp_b0, temp_b1, "big")
            curr_le = _word_from_bytes(curr_b0, curr_b1, "little")
            curr_be = _word_from_bytes(curr_b0, curr_b1, "big")
            print("DEBUG raw bytes:")
            print(f"  SOC: 0x{soc_b0:02X} 0x{soc_b1:02X}")
            print(f"  VOLT: 0x{volt_b0:02X} 0x{volt_b1:02X}")
            print(f"  TEMP: 0x{temp_b0:02X} 0x{temp_b1:02X}")
            print(f"  CURR: 0x{curr_b0:02X} 0x{curr_b1:02X}")
            print("DEBUG endian check (SOC/VOLT):")
            print(f"  SOC LE/BE: {soc_le} / {soc_be}")
            print(f"  VOLT LE/BE: {volt_le} / {volt_be}")
            print(f"  TEMP LE/BE: {temp_le} / {temp_be}")
            print(f"  CURR LE/BE: {curr_le} / {curr_be}")
            print(f"  Byteorder chosen: {byteorder}")
            print(f"  VoltScale: {volt_scale}")
            print(f"  CurrScale: {curr_scale}")

        return BatteryStatus(
            soc_percent=float(soc),
            remaining_mAh=float(remaining),
            full_mAh=float(full),
            voltage_mV=float(bat_mV),
            pack_voltage_V=float(pack_voltage_V),
            avg_current_mA=float(avg_current),
            temperature_C=float(temp_c),
        )


if __name__ == "__main__":
    status = read_battery()
    print(f"SOC: {status.soc_percent:.1f}%")
    print(f"Remaining: {status.remaining_mAh:.0f} mAh")
    print(f"Full: {status.full_mAh:.0f} mAh")
    print(f"BAT pin: {status.voltage_mV:.0f} mV")
    print(f"Pack: {status.pack_voltage_V:.2f} V")
    direction = "charging" if status.avg_current_mA > 0 else "discharging"
    print(f"Avg Current: {status.avg_current_mA:.0f} mA ({direction})")
    print(f"Temp: {status.temperature_C:.1f} C")
    if EXPECTED_CAPACITY_MAH:
        if abs(status.full_mAh - EXPECTED_CAPACITY_MAH) > (0.2 * EXPECTED_CAPACITY_MAH):
            print(
                "WARN: Full capacity differs from expected. Gauge likely needs configuration "
                f"(expected ~{EXPECTED_CAPACITY_MAH} mAh)."
            )
