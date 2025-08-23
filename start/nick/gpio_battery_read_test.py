import time
from gpiozero import DigitalInputDevice, DigitalOutputDevice

CLK  = DigitalOutputDevice(11)  # SCLK (idle low)
MISO = DigitalInputDevice(9)    # DOUT
MOSI = DigitalOutputDevice(10)  # DIN
CS   = DigitalOutputDevice(8)   # CS/SHDN

def _clk_pulse():
    CLK.on()
    # tiny hold so the ADC can present data (Python is slow anyway, but keep it explicit)
    # time.sleep(0.000001)  # 1 µs (optional)
    bit = MISO.value
    CLK.off()
    return bit

def read_adc(channel: int) -> int:
    if not 0 <= channel <= 7:
        raise ValueError("Channel must be 0–7")

    CS.on()        # idle high
    CLK.off()      # idle low
    MOSI.off()
    CS.off()       # select

    # Send Start(1), SGL/DIFF(1=single-ended), D2,D1,D0 (5 control bits total)
    command = 0b11 << 6
    command |= (channel & 0x07) << 3

    for i in range(5):
        MOSI.value = (command >> (7 - i)) & 1
        _ = _clk_pulse()  # clock the control bit into the ADC

    # One “null” clock before data
    MOSI.off()
    _ = _clk_pulse()

    # Read 10 data bits, MSB first. Read MISO right after raising CLK.
    result = 0
    for _ in range(10):
        CLK.on()
        bit = MISO.value
        result = (result << 1) | bit
        CLK.off()

    CS.on()
    return result

read_adc(0)
