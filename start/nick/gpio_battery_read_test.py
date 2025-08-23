# mcp3008_bitbang_gpiozero.py
import time
from gpiozero import DigitalInputDevice, DigitalOutputDevice

# BCM numbering (gpiozero default)
CLK  = DigitalOutputDevice(11)   # SCLK (idle low)
MISO = DigitalInputDevice(9)     # DOUT  (no pull-up)
MOSI = DigitalOutputDevice(10)   # DIN
CS   = DigitalOutputDevice(8)    # CS/SHDN (idle high)

def _clk_rise_read():
    CLK.on()                     # rising edge: ADC outputs next bit
    bit = int(MISO.value)        # sample MISO *while clock is high*
    CLK.off()                    # falling edge: ADC latches DIN
    return bit

def read_adc(ch: int) -> int:
    if not 0 <= ch <= 7:
        raise ValueError("Channel must be 0–7")

    # bus idle
    CS.on(); CLK.off(); MOSI.off()
    time.sleep(2e-6)

    # select chip
    CS.off()
    time.sleep(2e-6)             # tCSS

    # Send control word: Start(1), SGL(1), D2,D1,D0  => 5 bits total
    cmd = (0b11 << 6) | ((ch & 0x07) << 3)
    for _ in range(5):
        MOSI.value = (cmd & 0x80) != 0     # present bit while CLK low
        _ = _clk_rise_read()               # clock it in
        cmd <<= 1

    MOSI.off()

    # One null bit (discard)
    _ = _clk_rise_read()

    # Read 10 data bits MSB..LSB, sample on rising edge
    result = 0
    for _ in range(10):
        result = (result << 1) | _clk_rise_read()

    # deselect
    CS.on()
    return result

if __name__ == "__main__":
    print("Reading MCP3008 CH0… (Ctrl+C to stop)")
    while True:
        raw = read_adc(0)
        v = raw * 3.3 / 1023.0   # Vref = 3.3 V
        print(f"Raw: {raw:4d}   {v:0.3f} V")
        time.sleep(0.25)
