# import time
# import numpy as np
# from gpiozero import MCP3008

# # Create ADC channel object (using manual GPIO pin SPI)
# pot = MCP3008(channel=0, clock_pin=11, mosi_pin=10, miso_pin=9, select_pin=8)

# # Parameters
# tsample = 0.02  # Sampling interval in seconds
# tstop = 10      # Total runtime in seconds
# vref = 3.3      # Reference voltage for MCP3008

# # Low-pass filter setup
# fc = 2                    # Cutoff frequency in Hz
# wc = 2 * np.pi * fc       # Angular frequency
# tau = 1 / wc              # Time constant
# c0 = tsample / (tsample + tau)
# c1 = tau / (tsample + tau)

# # Initialize filter
# valueprev = pot.value
# time.sleep(tsample)
# tprev = 0
# tcurr = 0
# tstart = time.perf_counter()

# # Start loop
# print(f"Reading MCP3008 CH0 for {tstop} seconds...\n")
# while tcurr <= tstop:
#     tcurr = time.perf_counter() - tstart

#     if (np.floor(tcurr / tsample) - np.floor(tprev / tsample)) == 1:
#         valuecurr = pot.value  # Normalized (0–1)
#         valuefilt = c0 * valuecurr + c1 * valueprev

#         voltage_raw = valuecurr * vref
#         voltage_filt = valuefilt * vref

#         print(f"t={tcurr:.2f}s | Raw={voltage_raw:.3f} V | Filtered={voltage_filt:.3f} V")

#         valueprev = valuefilt

#     tprev = tcurr

# # Cleanup
# pot.close()
# print("Done.")

from gpiozero import MCP3008
import time

adc = MCP3008(channel=0, clock_pin=11, mosi_pin=10, miso_pin=9, select_pin=8)
vref = 3.3

while True:
    raw = adc.raw_value        # 0–1023
    voltage = raw / 1023 * vref
    print(f"Raw={raw}  Voltage={voltage:.3f} V")
    time.sleep(0.2)
