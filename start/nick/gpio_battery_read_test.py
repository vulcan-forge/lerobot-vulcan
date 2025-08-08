import time
from gpiozero import DigitalInputDevice, DigitalOutputDevice

# GPIO pin setup for bit-banged SPI
CLK  = DigitalOutputDevice(11)  # GPIO 11 (Pin 23) - SCLK
MISO = DigitalInputDevice(9)    # GPIO 9 (Pin 21) - MISO
MOSI = DigitalOutputDevice(10)  # GPIO 10 (Pin 19) - MOSI
CS   = DigitalOutputDevice(8)   # GPIO 8 (Pin 24) - CS (CE0)

def read_adc(channel):
    if not 0 <= channel <= 7:
        raise ValueError("Channel must be 0–7")

    CS.on()
    CLK.off()
    CS.off()

    # Start + single-ended + channel (5-bit command)
    command = 0b11 << 6                 # Start bit + single-ended
    command |= (channel & 0x07) << 3    # Channel number

    # Send command (5 bits)
    for i in range(5):
        MOSI.value = (command >> (7 - i)) & 0x01
        CLK.on()
        CLK.off()

    # Read null bit
    CLK.on()
    CLK.off()

    # Read 10 data bits
    result = 0
    for _ in range(10):
        CLK.on()
        CLK.off()
        result = (result << 1) | MISO.value

    CS.on()
    return result

# Main loop to read CH0
try:
    while True:
        raw = read_adc(0)
        voltage = (raw / 1023.0) * 3.3  # MCP3008 uses VREF = 3.3V
        print(f"Raw ADC: {raw} | Voltage: {voltage:.3f} V")
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopped.")
