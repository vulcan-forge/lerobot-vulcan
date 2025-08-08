import time
from gpiozero import DigitalInputDevice, DigitalOutputDevice

# Bit-banged SPI pins (not using hardware SPI)
CLK  = DigitalOutputDevice(13)  # GPIO13
MISO = DigitalInputDevice(19)   # GPIO19
MOSI = DigitalOutputDevice(16)  # GPIO16
CS   = DigitalOutputDevice(26)  # GPIO26

def read_adc(channel):
    if not 0 <= channel <= 7:
        raise ValueError("Channel must be 0–7")

    CS.on()
    CLK.off()
    CS.off()

    # Start bit + single-ended mode + channel number (5 bits)
    command = 0b11 << 6  # Start + single-ended
    command |= (channel & 0x07) << 3

    for i in range(5):
        bit = (command >> (7 - i)) & 0x01
        MOSI.value = bit
        CLK.on()
        CLK.off()

    # Read 1 null bit + 10-bit result
    result = 0
    for _ in range(12):
        CLK.on()
        CLK.off()
        result = (result << 1) | MISO.value

    CS.on()
    return (result >> 1) & 0x3FF  # Strip null bit

# 🧪 Read CH0 in a loop
try:
    while True:
        raw = read_adc(0)
        voltage = (raw / 1023.0) * 3.3  # assuming 3.3V reference
        print(f"Raw: {raw}, Voltage: {voltage:.3f} V")
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopped.")
