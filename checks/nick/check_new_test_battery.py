# Minimal voltage reader for BQ34Z100-R2
# Requires: pip install smbus2

from smbus2 import SMBus
import time

I2C_BUS = 1
BQ_ADDR = 0x55
CMD_VOLTAGE = 0x08

# If you want pack voltage, set divider values here.
R_TOP_OHMS = 249_000.0
R_BOTTOM_OHMS = 16_500.0
V_DIV_RATIO = R_BOTTOM_OHMS / (R_TOP_OHMS + R_BOTTOM_OHMS)


def read_voltage_raw(retries: int = 5) -> int:
    with SMBus(I2C_BUS) as bus:
        last_err = None
        for _ in range(retries):
            try:
                # Write command, short delay, read 2 bytes.
                bus.write_byte(BQ_ADDR, CMD_VOLTAGE)
                time.sleep(0.005)
                b = bus.read_i2c_block_data(BQ_ADDR, CMD_VOLTAGE, 2)
                raw = b[0] | (b[1] << 8)
                if raw == 0xFFFF:
                    raise OSError("read returned 0xFFFF")
                return raw
            except OSError as err:
                last_err = err
                time.sleep(0.02)
        raise last_err if last_err else RuntimeError("voltage read failed")


def main() -> None:
    raw = read_voltage_raw()
    # Voltage() returns mV; interpret as BAT pin by default.
    bat_mV = raw
    pack_V = (bat_mV / 1000.0) / V_DIV_RATIO
    print(f"Raw Voltage: 0x{raw:04X} ({raw} mV)")
    print(f"BAT pin: {bat_mV:.0f} mV")
    print(f"Pack: {pack_V:.2f} V")


if __name__ == "__main__":
    main()
