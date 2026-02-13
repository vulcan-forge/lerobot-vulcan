# Minimal voltage reader for BQ34Z100-R2
# Requires: pip install smbus2

from smbus2 import SMBus
import time
import statistics

I2C_BUS = 1
BQ_ADDR = 0x55
CMD_VOLTAGE = 0x08

# If you want pack voltage, set divider values here.
R_TOP_OHMS = 249_000.0
R_BOTTOM_OHMS = 16_500.0
V_DIV_RATIO = R_BOTTOM_OHMS / (R_TOP_OHMS + R_BOTTOM_OHMS)


def read_voltage_bytes(retries: int = 5) -> tuple[int, int]:
    with SMBus(I2C_BUS) as bus:
        last_err = None
        for _ in range(retries):
            try:
                # Write command, short delay, read 2 bytes.
                bus.write_byte(BQ_ADDR, CMD_VOLTAGE)
                time.sleep(0.005)
                b = bus.read_i2c_block_data(BQ_ADDR, CMD_VOLTAGE, 2)
                raw_le = b[0] | (b[1] << 8)
                raw_be = (b[0] << 8) | b[1]
                if raw_le == 0xFFFF or raw_be == 0xFFFF:
                    raise OSError("read returned 0xFFFF")
                return b[0], b[1]
            except OSError as err:
                last_err = err
                time.sleep(0.02)
        raise last_err if last_err else RuntimeError("voltage read failed")


def main() -> None:
    b0, b1 = read_voltage_bytes()
    raw_le = b0 | (b1 << 8)
    raw_be = (b0 << 8) | b1

    def _score(voltage_mV: int) -> int:
        # Prefer values that look like a pack voltage (2V–20V), then BAT pin (0.2V–2V).
        if 2000 <= voltage_mV <= 20000:
            return 2
        if 200 <= voltage_mV <= 2000:
            return 1
        return 0

    # Force big-endian for voltage to match observed behavior on this setup.
    voltage_mV = raw_be
    mode = "pack" if 2000 <= voltage_mV <= 20000 else "bat"

    if mode == "pack":
        pack_V = voltage_mV / 1000.0
        bat_mV = voltage_mV * V_DIV_RATIO
    else:
        bat_mV = float(voltage_mV)
        pack_V = (bat_mV / 1000.0) / V_DIV_RATIO

    print(f"Raw bytes: 0x{b0:02X} 0x{b1:02X}")
    print(f"Raw LE: 0x{raw_le:04X} ({raw_le} mV)")
    print(f"Raw BE: 0x{raw_be:04X} ({raw_be} mV)")
    print(f"Mode: {mode}")
    print(f"BAT pin: {bat_mV:.0f} mV")
    print(f"Pack: {pack_V:.2f} V")


def sample_voltage(samples: int = 25, delay_s: float = 0.2) -> None:
    good_pack = []
    good_bat = []
    bad = 0
    for _ in range(samples):
        try:
            b0, b1 = read_voltage_bytes()
            raw_be = (b0 << 8) | b1
            if raw_be == 0xFFFF or raw_be >= 60000:
                bad += 1
            else:
                if 2000 <= raw_be <= 20000:
                    good_pack.append(raw_be / 1000.0)
                else:
                    good_bat.append(raw_be / 1000.0)
        except OSError:
            bad += 1
        time.sleep(delay_s)

    print(f"Samples: {samples}, bad: {bad}")
    if good_pack:
        med = statistics.median(good_pack)
        print(f"Pack median: {med:.2f} V (n={len(good_pack)})")
        print(f"BAT est: {med * V_DIV_RATIO * 1000:.0f} mV")
    if good_bat:
        med = statistics.median(good_bat)
        print(f"BAT median: {med * 1000:.0f} mV (n={len(good_bat)})")
        print(f"Pack est: {med / V_DIV_RATIO:.2f} V")


if __name__ == "__main__":
    main()
    print("")
    sample_voltage()
