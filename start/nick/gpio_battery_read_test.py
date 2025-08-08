import time
from gpiozero import DigitalOutputDevice, DigitalInputDevice

# GPIO pin setup
CLK  = DigitalOutputDevice(13)  # GPIO13 → CLK
MISO = DigitalInputDevice(19)   # GPIO19 → DOUT
MOSI = DigitalOutputDevice(16)  # GPIO16 → DIN
CS   = DigitalOutputDevice(26)  # GPIO26 → CS/SHDN

def read_adc(channel):
    if channel < 0 or channel > 7:
        raise ValueError("Channel must be 0-7")

    CS.on()
    CLK.off()
    CS.off()

    # Start bit + single-ended + channel (3 bits)
    command = channel
    command |= 0b00011000  # Start bit + single-ended
    for i in range(5):
        MOSI.value = (command >> (4 - i)) & 1
        CLK.on()
        CLK.off()

    # Read 10 bits for MCP3008 (not 12)
    result = 0
    for _ in range(10):  # Changed from 12 to 10
        CLK.on()
        CLK.off()
        result <<= 1
        if MISO.value:
            result |= 1

    CS.on()
    return result  # No need to shift right for MCP3008

def read_adc_with_debug(channel):
    if channel < 0 or channel > 7:
        raise ValueError("Channel must be 0-7")

    print("Starting ADC read...")
    CS.on()
    CLK.off()
    CS.off()
    print(f"CS activated - CLK: {CLK.value} | MISO: {MISO.value} | MOSI: {MOSI.value} | CS: {CS.value}")

    # Start bit + single-ended + channel (3 bits)
    command = channel
    command |= 0b00011000  # Start bit + single-ended
    print(f"Command: {bin(command)}")

    for i in range(5):
        MOSI.value = (command >> (4 - i)) & 1
        print(f"Bit {i}: MOSI={MOSI.value}")
        CLK.on()
        CLK.off()

    # Read 10 bits for MCP3008
    result = 0
    print("Reading 10 bits...")
    for bit in range(10):
        CLK.on()
        CLK.off()
        result <<= 1
        if MISO.value:
            result |= 1
        print(f"Bit {bit}: MISO={MISO.value}, Result so far: {bin(result)}")

    CS.on()
    print(f"CS deactivated - CLK: {CLK.value} | MISO: {MISO.value} | MOSI: {MOSI.value} | CS: {CS.value}")
    return result

if __name__ == "__main__":
    try:
        while True:
            adc_val = read_adc_with_debug(0)
            print(f"Final ADC Value: {adc_val}")
            print("=" * 60)
            time.sleep(2)
    except KeyboardInterrupt:
        print("Exiting.")
