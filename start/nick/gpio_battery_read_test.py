from gpiozero import MCP3008
import time

adc = MCP3008(channel=0)
vref = 3.3
sample_interval = 0.2  # seconds between ADC reads
average_window = 50    # number of samples to average

# Voltage divider ratio: R2/(R1+R2) = 750/(3000+750) = 0.2
voltage_divider_ratio = 0.2

while True:
    total = 0
    for _ in range(average_window):
        raw = adc.raw_value
        adc_voltage = raw / 1023 * vref
        battery_voltage = adc_voltage / voltage_divider_ratio
        total += battery_voltage
        print(f"Raw={raw:<4}  ADC={adc_voltage:.3f}V  Battery={battery_voltage:.3f}V")
        time.sleep(sample_interval)

    avg_battery_voltage = total / average_window
    print()
    print(f"Averaged Battery Voltage = {avg_battery_voltage:.3f} V")
    print()

    time.sleep(5)  # wait remainder of the 10s cycle
