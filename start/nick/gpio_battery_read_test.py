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

    # Read 12 bits total for MCP3008 (2 null bits + 10 data bits)
    result = 0
    for _ in range(12):  # Changed back to 12 bits
        CLK.on()
        CLK.off()
        result <<= 1
        if MISO.value:
            result |= 1

    CS.on()
    # Extract only the 10 data bits (bits 2-11)
    return (result >> 2) & 0x3FF  # Mask to get only 10 bits

def calculate_voltage(adc_value):
    v_ref = 3.3
    return adc_value * (v_ref / 1023.0)

if __name__ == "__main__":
    try:
        while True:
            adc_val = read_adc(0)
            voltage_at_adc = calculate_voltage(adc_val)

            # Calculate what ADC should read for 2.514V
            expected_adc = 2.514 / 3.3 * 1023

            print(f"Raw: {adc_val}, Expected: {expected_adc:.0f}, ADC Input: {voltage_at_adc:.3f}V | CLK: {CLK.value} | MISO: {MISO.value} | MOSI: {MOSI.value} | CS: {CS.value}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting.")
