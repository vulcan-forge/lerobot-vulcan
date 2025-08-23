import time
from gpiozero import DigitalInputDevice, DigitalOutputDevice

# GPIO pin setup for bit-banged SPI
CLK  = DigitalOutputDevice(11)  # GPIO 11 (Pin 23) - SCLK
MISO = DigitalInputDevice(9)    # GPIO 9 (Pin 21) - MISO
MOSI = DigitalOutputDevice(10)  # GPIO 10 (Pin 19) - MOSI
CS   = DigitalOutputDevice(8)   # GPIO 8 (Pin 24) - CS (CE0)

T_CS   = 5e-6    # CS setup/hold
T_BIT  = 3e-6    # settle after rising edge
T_IDLE = 2e-6    # idle between bits

def _rise_sample():
    CLK.on()
    time.sleep(T_BIT)
    bit = 1 if MISO.value else 0
    CLK.off()
    time.sleep(T_IDLE)
    return bit

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

def main():
    """Main function to continuously read ADC values"""
    try:
        print("Starting ADC reading... Press Ctrl+C to stop")
        while True:
            raw = read_adc(0)
            voltage = (raw / 1023.0) * 3.3  # MCP3008 uses VREF = 3.3V
            print(f"Raw ADC: {raw} | Voltage: {voltage:.3f} V")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        # Clean up GPIO resources
        CLK.close()
        MISO.close()
        MOSI.close()
        CS.close()
        print("GPIO resources cleaned up.")

if __name__ == "__main__":
    main()
