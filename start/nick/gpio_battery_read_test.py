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
    # Fix: Use 4095 for 12-bit ADC instead of 1023 for 10-bit
    v_out = adc_value * (v_ref / 4095.0)
    v_in = v_out * ((r1 + r2) / r2)
    return v_in

def calculate_lifepo4_percentage(voltage):
    """
    Calculate battery percentage for LiFePO4 battery
    LiFePO4 voltage range: 2.0V (0%) to 3.65V (100%)
    """
    # LiFePO4 voltage characteristics
    min_voltage = 2.0    # Cutoff voltage (0%)
    max_voltage = 3.65   # Maximum voltage (100%)
    nominal_voltage = 3.2 # Nominal voltage

    if voltage <= min_voltage:
        return 0
    elif voltage >= max_voltage:
        return 100
    else:
        # Linear interpolation between min and max voltage
        percentage = ((voltage - min_voltage) / (max_voltage - min_voltage)) * 100
        return max(0, min(100, percentage))

def get_battery_status(voltage):
    """
    Get battery status based on voltage for LiFePO4
    """
    if voltage >= 3.4:
        return "FULL"
    elif voltage >= 3.2:
        return "GOOD"
    elif voltage >= 2.8:
        return "LOW"
    elif voltage >= 2.5:
        return "CRITICAL"
    else:
        return "EMPTY"

if __name__ == "__main__":
    R1 = 390_000  # Ohms
    R2 = 100_000  # Ohms

    try:
        while True:
            adc_val = read_adc(0)
            voltage = calculate_battery_voltage(adc_val, R1, R2)
            percent = calculate_lifepo4_percentage(voltage)
            status = get_battery_status(voltage)

            print(f"ADC: {adc_val} | Battery Voltage: {voltage:.2f} V | Battery: {percent:.1f}% | Status: {status}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting.")
