from smbus2 import SMBus
import time

BUS = 1
ADDR = 0x55

def read_word(bus, cmd):
    bus.write_byte(ADDR, cmd)
    time.sleep(0.002)
    b0, b1 = bus.read_i2c_block_data(ADDR, cmd, 2)
    return b0 | (b1 << 8)

with SMBus(BUS) as bus:
    ctrl = read_word(bus, 0x00)
    flags = read_word(bus, 0x06)

    print(f"Control (0x00): 0x{ctrl:04X}")
    print(f"Flags   (0x06): 0x{flags:04X}")
