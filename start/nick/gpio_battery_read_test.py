#!/usr/bin/env python3
# MCP3008 robust bit-bang (gpiozero) — windowed capture + guard delays

import time
from statistics import median
from gpiozero import DigitalInputDevice, DigitalOutputDevice

# ------------ USER CONFIG ------------
CHANNEL   = 0            # MCP3008 channel 0..7
VREF      = 3.3
R_TOP     = 30000.0      # divider top (to battery)
R_BOT     = 7500.0       # divider bottom (to GND)
PRINT_DT  = 0.25         # print cadence (s)

# If you can: use CS on BCM22 (physical pin 15). If not, set CS_PIN = 8.
#CS_PIN = 22  # 22 (phys 15) recommended; 8 (phys 24) if you can't rewire

# Timing guards (us-scale)
T_CS   = 5e-6   # CS setup/hold
T_BIT  = 3e-6   # settle after rising edge before sampling MISO
T_IDLE = 2e-6   # idle between bits
CS_IDLE_GAP = 50e-6  # small idle gap between whole transactions

# Robustness knobs
WINDOW_BITS = 24   # read this many bits after control; keep last 10
MEDIAN_N    = 5    # 1 = no median; higher kills occasional hiccups

# ------------ PINS (BCM numbering) ------------
CLK  = DigitalOutputDevice(11)         # SCLK (phys 23), idle low
MISO = DigitalInputDevice(9)           # DOUT (phys 21)
MOSI = DigitalOutputDevice(10)         # DIN  (phys 19)
CS   = DigitalOutputDevice(22)     # CS   (phys 15 if 22; phys 24 if 8), idle high

def _rise_sample() -> int:
    """SPI mode-0: raise CLK, wait a hair, sample while high, then drop."""
    CLK.on()
    time.sleep(T_BIT)
    bit = 1 if MISO.value else 0
    CLK.off()
    time.sleep(T_IDLE)
    return bit

def _send_control(ch: int):
    """Assert CS, send Start+SGL+D2..D0 (5 bits MSB-first)."""
    if not 0 <= ch <= 7:
        raise ValueError("Channel must be 0–7")
    # bus idle -> select
    CLK.off(); MOSI.off(); CS.on(); time.sleep(T_CS)
    CS.off(); time.sleep(T_CS)

    # Control word: Start(1), SGL(1), D2, D1, D0
    cmd = (0b11 << 6) | ((ch & 7) << 3)
    for _ in range(5):
        MOSI.value = (cmd & 0x80) != 0        # present bit while CLK low
        _ = _rise_sample()                    # latch into ADC on rising edge
        cmd <<= 1
    MOSI.off()

def _read_windowed(ch: int) -> int:
    """Read a generous bit window; keep only the last 10 data bits."""
    _send_control(ch)
    bits = [_rise_sample() for _ in range(WINDOW_BITS)]
    CS.on(); time.sleep(T_CS); time.sleep(CS_IDLE_GAP)

    val = 0
    for b in bits[-10:]:                      # drop null + any misalignment
        val = (val << 1) | b
    return val

def read_filtered(ch: int) -> int:
    if MEDIAN_N <= 1:
        return _read_windowed(ch)
    return median(_read_windowed(ch) for _ in range(MEDIAN_N))

def _code_to_v(raw: int) -> float:
    return raw * VREF / 1023.0

def _batt_from_node(v_node: float) -> float:
    return v_node * (R_TOP + R_BOT) / R_BOT

def main():
    try:
        print("MCP3008 windowed reader. Ctrl+C to stop.")
        print(f"CS=BCM{22}  Vref={VREF} V  Divider={int(R_TOP)}/{int(R_BOT)} Ω")
        while True:
            raw = int(round(read_filtered(CHANNEL)))
            vch = _code_to_v(raw)
            vb  = _batt_from_node(vch)
            print(f"Raw: {raw:4d}  CH0: {vch:0.3f} V  Battery≈ {vb:0.2f} V")
            time.sleep(PRINT_DT)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        CLK.close(); MISO.close(); MOSI.close(); CS.close()

if __name__ == "__main__":
    main()
