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

def calculate_battery_voltage(adc_value):
    v_ref = 3.3
    v_out = adc_value * (v_ref / 1023.0)
    # Use the actual measured ratio: 13.15V / 2.514V = 5.23
    voltage_divider_ratio = 13.15 / 2.514
    v_in = v_out * voltage_divider_ratio
    return v_in

if __name__ == "__main__":
    try:
        while True:
            adc_val = read_adc(0)
            voltage = calculate_battery_voltage(adc_val)

            # Debug: Show intermediate calculations
            v_out = adc_val * (3.3 / 1023.0)
            voltage_divider_ratio = 13.15 / 2.514

            print(f"Raw: {adc_val}, V_out: {v_out:.3f}V, Ratio: {voltage_divider_ratio:.2f}, Voltage: {voltage:.3f} V | CLK: {CLK.value} | MISO: {MISO.value} | MOSI: {MOSI.value} | CS: {CS.value}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting.")
