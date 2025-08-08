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

    # Read 12 bits (first bit is null)
    result = 0
    for _ in range(12):
        CLK.on()
        CLK.off()
        result <<= 1
        if MISO.value:
            result |= 1

    CS.on()
    return result >> 1  # drop null bit

def calculate_battery_voltage(adc_value, r1, r2):
    v_ref = 3.3
    v_out = adc_value * (v_ref / 1023.0)
    v_in = v_out * ((r1 + r2) / r2)
    return v_in

if __name__ == "__main__":
    R1 = 390_000  # Ohms
    R2 = 100_000  # Ohms

    try:
        while True:
            adc_val = read_adc(0)
            voltage = calculate_battery_voltage(adc_val, R1, R2)
            percent = min(100, max(0, int((voltage - 3.0) / (4.2 - 3.0) * 100)))
            print(f"ADC: {adc_val} | Battery Voltage: {voltage:.2f} V | Battery: {percent}%")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting.")
