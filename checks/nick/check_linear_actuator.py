import time

# Optional import (same pattern as battery.py)
try:
    from gpiozero import MCP3008  # type: ignore
except Exception:
    MCP3008 = None  # type: ignore

VREF = 3.30  # adjust to measured 3V3 rail if you want accuracy
ACTUATOR_ADC_CHANNEL = 1

# If your actuator voltage goes through a resistor divider before the MCP3008,
# set this to (V_adc / V_actuator). Example: same as battery.py divider => 0.2
ACTUATOR_VOLTAGE_DIVIDER_RATIO = 1.0  # <-- change this if you have a divider

AVERAGE_SAMPLES = 50

def read_channel_voltage(channel: int, *, average_samples: int = AVERAGE_SAMPLES) -> float:
    if MCP3008 is None:
        raise RuntimeError("gpiozero is not installed; MCP3008 is unavailable on this machine.")

    adc = MCP3008(channel=channel)

    total = 0.0
    for _ in range(average_samples):
        raw = adc.raw_value                 # 0..1023
        v_adc = (raw / 1023.0) * VREF       # volts at the MCP3008 pin
        total += v_adc

    return total / average_samples

def read_actuator_voltage() -> float:
    v_adc = read_channel_voltage(ACTUATOR_ADC_CHANNEL)
    return v_adc / ACTUATOR_VOLTAGE_DIVIDER_RATIO

if __name__ == "__main__":
    while True:
        v = read_actuator_voltage()
        print(f"Actuator voltage (ch {ACTUATOR_ADC_CHANNEL}): {v:.3f} V")
        time.sleep(0.2)
