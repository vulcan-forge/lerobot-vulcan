import time
from gpiozero import DigitalInputDevice, DigitalOutputDevice

CLK  = DigitalOutputDevice(11)   # SCLK (idle low)
MISO = DigitalInputDevice(9)     # DOUT (no pull-up)
MOSI = DigitalOutputDevice(10)   # DIN
CS   = DigitalOutputDevice(8)    # CS (idle high)

T_CS   = 5e-6    # CS setup/hold
T_BIT  = 3e-6    # settle after rising edge
T_IDLE = 2e-6    # idle between bits

def _rise_sample():
    CLK.on()
    time.sleep(T_BIT)
    bit = 1 if MISO.value else 0
    CLK.off()
    time.sleep(T_IDLE)
    return bit

def read_adc(ch: int) -> int:
    if not 0 <= ch <= 7:
        raise ValueError("Channel 0–7")

    # Idle bus, then select
    CLK.off(); MOSI.off(); CS.on()
    time.sleep(T_CS)
    CS.off()
    time.sleep(T_CS)

    # Control word: Start(1), SGL(1), D2,D1,D0  (5 bits, MSB first)
    cmd = (0b11 << 6) | ((ch & 7) << 3)
    for _ in range(5):
        MOSI.value = (cmd & 0x80) != 0   # stable while CLK low
        _ = _rise_sample()               # clock it in on rising edge
        cmd <<= 1

    MOSI.off()

    # One null bit (discard), plus one extra guard bit
    _ = _rise_sample()
    _ = _rise_sample()

    # Read the 10 data bits MSB..LSB
    val = 0
    for _ in range(10):
        val = (val << 1) | _rise_sample()

    CS.on()
    time.sleep(T_CS)
    return val
