# # # import time
# # # import numpy as np
# # # from gpiozero import MCP3008

# # # # Create ADC channel object (using manual GPIO pin SPI)
# # # pot = MCP3008(channel=0, clock_pin=11, mosi_pin=10, miso_pin=9, select_pin=8)

# # # # Parameters
# # # tsample = 0.02  # Sampling interval in seconds
# # # tstop = 10      # Total runtime in seconds
# # # vref = 3.3      # Reference voltage for MCP3008

# # # # Low-pass filter setup
# # # fc = 2                    # Cutoff frequency in Hz
# # # wc = 2 * np.pi * fc       # Angular frequency
# # # tau = 1 / wc              # Time constant
# # # c0 = tsample / (tsample + tau)
# # # c1 = tau / (tsample + tau)

# # # # Initialize filter
# # # valueprev = pot.value
# # # time.sleep(tsample)
# # # tprev = 0
# # # tcurr = 0
# # # tstart = time.perf_counter()

# # # # Start loop
# # # print(f"Reading MCP3008 CH0 for {tstop} seconds...\n")
# # # while tcurr <= tstop:
# # #     tcurr = time.perf_counter() - tstart

# # #     if (np.floor(tcurr / tsample) - np.floor(tprev / tsample)) == 1:
# # #         valuecurr = pot.value  # Normalized (0–1)
# # #         valuefilt = c0 * valuecurr + c1 * valueprev

# # #         voltage_raw = valuecurr * vref
# # #         voltage_filt = valuefilt * vref

# # #         print(f"t={tcurr:.2f}s | Raw={voltage_raw:.3f} V | Filtered={voltage_filt:.3f} V")

# # #         valueprev = valuefilt

# # #     tprev = tcurr

# # # # Cleanup
# # # pot.close()
# # # print("Done.")

# # from gpiozero import MCP3008
# # import time

# # # adc = MCP3008(channel=0, clock_pin=11, mosi_pin=10, miso_pin=9, select_pin=8)
# # # adc = MCP3008(channel=0)
# # # vref = 3.3

# # # while True:
# # #     raw = adc.raw_value        # 0–1023
# # #     voltage = raw / 1023 * vref
# # #     print(f"Raw={raw}  Voltage={voltage:.3f} V")
# # #     time.sleep(0.2)

# # # alpha = 0.1  # Lower = more smoothing, but slower
# # # smoothed = 0

# # # while True:
# # #     raw = adc.raw_value
# # #     voltage = raw / 1023 * vref
# # #     smoothed = alpha * voltage + (1 - alpha) * smoothed
# # #     print(f"Raw={raw}  Voltage={voltage:.3f} V  Filtered={smoothed:.3f} V")
# # #     time.sleep(0.2)

# # adc = MCP3008(channel=0)
# # vref = 3.3
# # alpha = 0.01  # Very stable, slow changes
# # filtered = vref  # Start at max Vref

# # print("Reading MCP3008... Press Ctrl+C to stop.\n")
# # try:
# #     while True:
# #         raw = adc.raw_value            # 0–1023
# #         voltage = raw / 1023 * vref    # Convert to volts
# #         filtered = alpha * voltage + (1 - alpha) * filtered
# #         print(f"Raw={raw:<4} Voltage={voltage:.3f} V  Filtered={filtered:.3f} V")
# #         time.sleep(0.1)
# # except KeyboardInterrupt:
# #     print("Done.")

# import spidev
# import time

# # Setup SPI
# spi = spidev.SpiDev()
# spi.open(0, 0)  # Bus 0, Device 0 (use 0,1 if using CE1)
# spi.max_speed_hz = 1350000

# def read_channel(channel):
#     # MCP3008 protocol: [start=1, SGL/DIFF + channel<<4, 0]
#     adc = spi.xfer2([1, (8 + channel) << 4, 0])
#     data = ((adc[1] & 3) << 8) + adc[2]
#     return data

# vref = 3.3
# alpha = 0.05
# filtered = vref

# while True:
#     raw = read_channel(0)
#     voltage = raw / 1023.0 * vref
#     filtered = alpha * voltage + (1 - alpha) * filtered
#     print(f"Raw={raw:<4}  Voltage={voltage:.3f} V  Filtered={filtered:.3f} V")
#     time.sleep(0.1)

from gpiozero import MCP3008
import time

adc = MCP3008(channel=0)
vref = 3.3
sample_interval = 0.1  # seconds between ADC reads
average_window = 50    # number of samples to average (e.g. 50 * 0.1s = 5s)

while True:
    total = 0
    for _ in range(average_window):
        raw = adc.raw_value
        voltage = raw / 1023 * vref
        total += voltage
        time.sleep(sample_interval)

    avg_voltage = total / average_window
    print(f"Averaged Voltage = {avg_voltage:.3f} V")

    time.sleep(5)  # wait remainder of the 10s cycle
