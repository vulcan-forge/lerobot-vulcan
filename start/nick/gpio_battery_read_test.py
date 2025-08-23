# Robust bit-banged MCP3008 read (gpiozero, SPI mode 0)
import time
from gpiozero import DigitalInputDevice, DigitalOutputDevice

CLK  = DigitalOutputDevice(11)   # SCLK (idle low)
MISO = DigitalInputDevice(9)     # DOUT
MOSI = DigitalOutputDevice(10)   # DIN
CS   = DigitalOutputDevice(8)    # CS (idle high)

T_CS   = 2e-6    # CS setup/hold
T_BIT  = 1e-6    # data settle after rising edge
T_IDLE = 1e-6    # inter-bit idle

def _rise_sample():
    CLK.on()
    time.sleep(T_BIT)            # let DOUT settle
    bit = 1 if MISO.value else 0
    CLK.off()
    time.sleep(T_IDLE)
    return bit

def read_adc(ch: int) -> int:
    if not 0 <= ch <= 7:
        raise ValueError("Channel 0–7")

    # idle bus
    CLK.off(); MOSI.off(); CS.on()
    time.sleep(T_CS)

    # select chip
    CS.off()
    time.sleep(T_CS)

    # Control: Start(1), SGL(1), D2,D1,D0   (5 bits total)
    cmd = (0b11 << 6) | ((ch & 7) << 3)
    for _ in range(5):
        MOSI.value = (cmd & 0x80) != 0     # present while CLK low
        _ = _rise_sample()                 # clock into ADC on rising edge
        cmd <<= 1

    MOSI.off()

    # --- Alignment guard: clock 2 extra times, discard both ---
    _ = _rise_sample()     # nominal "null" bit
    _ = _rise_sample()     # guard bit (so the next is guaranteed MSB)

    # Now read exact 10 data bits (MSB..LSB)
    val = 0
    for _ in range(10):
        val = (val << 1) | _rise_sample()

    # deselect
    CS.on()
    time.sleep(T_CS)

    return val

if __name__ == "__main__":
    VREF = 3.3
    while True:
        raw = read_adc(0)
        v = raw * VREF / 1023.0
        print(f"Raw: {raw:4d}   {v:0.3f} V")
        time.sleep(0.25)
