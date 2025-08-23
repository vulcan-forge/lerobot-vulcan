#!/usr/bin/env python3
# Robust MCP3008 bit-banged SPI with gpiozero (Raspberry Pi, BCM pins)

import time
from statistics import median
from collections import deque
from gpiozero import DigitalInputDevice, DigitalOutputDevice

# -----------------------
# CONFIG
# -----------------------
CHANNEL        = 0         # MCP3008 channel (0..7)
VREF           = 3.3       # MCP3008 reference voltage
R_TOP          = 30000.0   # Divider top resistor (to battery)
R_BOT          = 7500.0    # Divider bottom resistor (to GND)
PRINT_EVERY_S  = 0.25      # Print cadence

# Bit-bang timing (microsecond-ish guards; Python is slow, so be generous)
T_CS   = 5e-6   # CS setup/hold
T_BIT  = 3e-6   # settle time after rising edge before sampling MISO
T_IDLE = 2e-6   # idle between bits

# Hardened reading options
USE_WINDOWED_READ = True   # Read extra bits and keep last 10 (very robust)
USE_MEDIAN_N      = 5      # Median-of-N reads (1 = disabled)

# Debug: set True to print one raw bitstream then continue normal reads
PRINT_ONE_DEBUG_BITSTREAM = False

# -----------------------
# GPIO (BCM numbering)
# -----------------------
CLK  = DigitalOutputDevice(11)  # SCLK (phys 23)  idle low
MISO = DigitalInputDevice(9)    # DOUT (phys 21)  high-Z when CS high
MOSI = DigitalOutputDevice(10)  # DIN  (phys 19)
CS   = DigitalOutputDevice(8)   # CS   (phys 24)  idle high

def _rise_sample() -> int:
    """Clock rising edge, then sample MISO while CLK is high (mode-0)."""
    CLK.on()
    time.sleep(T_BIT)                 # allow DOUT to settle
    bit = 1 if MISO.value else 0
    CLK.off()
    time.sleep(T_IDLE)
    return bit

def _send_control_and_prep(ch: int):
    """Assert CS, send Start+SGL+D2..D0 (5 bits), MOSI stable while CLK low."""
    if not 0 <= ch <= 7:
        raise ValueError("Channel must be 0–7")

    # Idle bus
    CLK.off(); MOSI.off(); CS.on()
    time.sleep(T_CS)

    # Select
    CS.off()
    time.sleep(T_CS)

    # Control word: Start(1), SGL(1), D2,D1,D0 (MSB-first)
    cmd = (0b11 << 6) | ((ch & 7) << 3)
    for _ in range(5):
        MOSI.value = (cmd & 0x80) != 0   # present while CLK is low
        _ = _rise_sample()               # clock it into ADC
        cmd <<= 1

    MOSI.off()

def read_adc_strict(ch: int) -> int:
    """Exactly 5 control + 1 null + 10 data bits."""
    _send_control_and_prep(ch)
    _ = _rise_sample()  # discard the null bit
    val = 0
    for _ in range(10):
        val = (val << 1) | _rise_sample()
    CS.on(); time.sleep(T_CS)
    return val

def read_adc_windowed(ch: int) -> int:
    """Read a long window and keep the last 10 bits (guards against misalignment)."""
    _send_control_and_prep(ch)
    # Read generous window (null + data + a few extras)
    bits = [ _rise_sample() for _ in range(16) ]
    CS.on(); time.sleep(T_CS)

    val = 0
    for b in bits[-10:]:               # keep the last 10 bits
        val = (val << 1) | b
    return val

def read_bits_debug(ch: int) -> str:
    """Return a string of raw bits after control to inspect alignment."""
    _send_control_and_prep(ch)
    bits = [ _rise_sample() for _ in range(16) ]
    CS.on(); time.sleep(T_CS)
    return "".join("1" if b else "0" for b in bits)

def read_adc_filtered(ch: int) -> int:
    """Median-of-N wrapper around the selected read method."""
    reader = read_adc_windowed if USE_WINDOWED_READ else read_adc_strict
    if USE_MEDIAN_N <= 1:
        return reader(ch)
    return median(reader(ch) for _ in range(USE_MEDIAN_N))

def code_to_volts(raw: int) -> float:
    return raw * VREF / 1023.0

def ch0_to_battery(v_ch0: float) -> float:
    # Divider back-calculation
    return v_ch0 * (R_TOP + R_BOT) / R_BOT

def main():
    try:
        if PRINT_ONE_DEBUG_BITSTREAM:
            bits = read_bits_debug(CHANNEL)
            print(f"[debug] raw bitstream (after control): {bits}  (keeping last 10)")
            # fall through to normal loop

        # Optional small moving-average for display smoothness (cheap + readable)
        window = deque(maxlen=4)

        print("Reading MCP3008…  (Ctrl+C to stop)")
        print(f"Mode: {'windowed' if USE_WINDOWED_READ else 'strict'}, "
              f"median N={USE_MEDIAN_N}, Vref={VREF} V, divider={int(R_TOP)}/{int(R_BOT)} Ω")
        while True:
            raw = int(round(read_adc_filtered(CHANNEL)))
            v_ch0 = code_to_volts(raw)
            v_bat = ch0_to_battery(v_ch0)
            window.append((raw, v_ch0, v_bat))

            # display last sample (or uncomment to average the window)
            r, vc, vb = window[-1]
            # r = round(sum(x[0] for x in window)/len(window))
            # vc = sum(x[1] for x in window)/len(window)
            # vb = sum(x[2] for x in window)/len(window)

            print(f"Raw: {r:4d}  CH0: {vc:0.3f} V  Battery≈ {vb:0.2f} V")
            time.sleep(PRINT_EVERY_S)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        # Clean up GPIO
        CLK.close(); MISO.close(); MOSI.close(); CS.close()
        print("GPIO cleaned up.")

if __name__ == "__main__":
    main()
