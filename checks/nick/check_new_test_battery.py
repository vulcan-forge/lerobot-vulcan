# bq34z100_bat_voltage.py
# pip install smbus2

from smbus2 import SMBus
import time

I2C_BUS = 1
BQ_ADDR = 0x55

# Try 0x04 first (common "Voltage" on many TI gauges).
# If your setup truly uses 0x08 for voltage, switch back to 0x08.
CMD_VOLTAGE = 0x04

# Divider (if you want to estimate pack voltage from BAT pin reading)
R_TOP_OHMS = 249_000.0
R_BOTTOM_OHMS = 16_500.0
V_DIV_RATIO = R_BOTTOM_OHMS / (R_TOP_OHMS + R_BOTTOM_OHMS)  # ~0.06217


def read_word_le(bus: SMBus, addr: int, cmd: int, delay_s: float = 0.005) -> int:
    """
    Read a 16-bit word where device returns LSB then MSB (little-endian on the wire),
    which is typical for SMBus word data.
    """
    # Many gauges support the "command then read 2 bytes" pattern.
    bus.write_byte(addr, cmd)
    time.sleep(delay_s)
    b0, b1 = bus.read_i2c_block_data(addr, cmd, 2)
    return b0 | (b1 << 8)


def main() -> None:
    with SMBus(I2C_BUS) as bus:
        raw = read_word_le(bus, BQ_ADDR, CMD_VOLTAGE)

    # The simplest interpretation (matches your observed 0x02B6 -> 694 mV)
    bat_mv = raw
    bat_v = bat_mv / 1000.0

    # Optional: infer pack voltage if BAT is truly a divided representation
    pack_v_est = bat_v / V_DIV_RATIO

    print(f"CMD: 0x{CMD_VOLTAGE:02X}")
    print(f"Raw word (LE): 0x{raw:04X} ({raw} mV)")
    print(f"BAT pin: {bat_v:.3f} V")

    # Comment this out if you ONLY want BAT
    print(f"Pack est (from divider): {pack_v_est:.2f} V")


if __name__ == "__main__":
    main()
