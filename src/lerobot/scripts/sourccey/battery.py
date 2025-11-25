from gpiozero import MCP3008
from dataclasses import dataclass
from typing import Optional, List, Tuple

# ADC Configuration
ADC_CHANNEL = 0

# Measure this with a multimeter if you can for better accuracy
VREF = 3.30  # Pi 3.3V rail (adjust to your measured value)

# Your divider: R1 = 3000, R2 = 750 -> 0.2
VOLTAGE_DIVIDER_RATIO = 0.2  # V_adc = V_batt * 0.2

# Sampling configuration
AVERAGE_SAMPLES = 10  # Number of samples to average for stability

# Battery voltage range for clamping
BATTERY_VOLTAGE_MIN = 11.0  # "Empty" (under load) – adjust based on your pack/BMS cutoff
BATTERY_VOLTAGE_MAX_OCV = 13.6  # "Full" resting open-circuit voltage
BATTERY_VOLTAGE_MAX_CHARGE = 14.6  # Charging voltage – treat as 100%
FILTER_ALPHA = 0.1  # 0..1, lower = more smoothing
_filtered_voltage: Optional[float] = None

# Approximate voltage vs SoC curve for a 4S LiFePO4 pack (resting voltage)
# (voltage, percent)
LIFEPO4_CURVE: List[Tuple[float, int]] = [
    (11.0,   0),
    (12.0,  10),
    (12.4,  20),
    (12.8,  40),
    (13.0,  55),
    (13.1,  65),
    (13.2,  75),
    (13.3,  85),
    (13.4,  93),
    (13.6, 100),
]

@dataclass
class BatteryData:
    """Battery data container"""
    voltage: float
    percent: int

# Global ADC instance (initialized on first use)
_adc: Optional[MCP3008] = None


def _get_adc() -> MCP3008:
    """Get or initialize the ADC instance"""
    global _adc
    if _adc is None:
        _adc = MCP3008(channel=ADC_CHANNEL)
    return _adc


def get_battery_voltage() -> float:
    global _filtered_voltage
    adc = _get_adc()
    total = 0.0

    for _ in range(AVERAGE_SAMPLES):
        raw = adc.raw_value
        adc_voltage = (raw / 1023.0) * VREF
        battery_voltage = adc_voltage / VOLTAGE_DIVIDER_RATIO
        total += battery_voltage

    instant_voltage = total / AVERAGE_SAMPLES

    if _filtered_voltage is None:
        _filtered_voltage = instant_voltage
    else:
        _filtered_voltage = (
            FILTER_ALPHA * instant_voltage +
            (1.0 - FILTER_ALPHA) * _filtered_voltage
        )

    return _filtered_voltage


def _voltage_to_percent_curve(voltage: float) -> int:
    """
    Map battery voltage to percentage using a LiFePO4 curve
    with linear interpolation between points.
    """

    # Treat above open-circuit "full" voltage as 100%
    if voltage >= BATTERY_VOLTAGE_MAX_OCV:
        return 100

    # Clamp below minimum
    if voltage <= BATTERY_VOLTAGE_MIN:
        return 0

    # Find two curve points voltage is between
    for i in range(len(LIFEPO4_CURVE) - 1):
        v1, p1 = LIFEPO4_CURVE[i]
        v2, p2 = LIFEPO4_CURVE[i + 1]

        if v1 <= voltage <= v2:
            # Linear interpolation between (v1, p1) and (v2, p2)
            t = (voltage - v1) / (v2 - v1)
            percent = p1 + t * (p2 - p1)
            return int(round(percent))

    # Fallback (should not hit because of clamping)
    return 0


def get_battery_percent() -> int:
    """
    Get battery percentage based on voltage and the LiFePO4 curve.

    Returns:
        Battery percentage (0-100)
    """
    voltage = get_battery_voltage()

    # If we're clearly in "charging" range, just say 100%
    if voltage >= BATTERY_VOLTAGE_MAX_CHARGE:
        return 100

    percent = _voltage_to_percent_curve(voltage)

    # Final clamp
    return max(0, min(100, percent))


def get_battery_data() -> BatteryData:
    """
    Get both battery voltage and percentage.

    Returns:
        BatteryData containing voltage and percent
    """
    voltage = get_battery_voltage()
    percent = get_battery_percent()
    return BatteryData(voltage=voltage, percent=percent)


if __name__ == "__main__":
    # When run as a script, output JSON (for Rust integration)
    import json
    try:
        battery_data = get_battery_data()
        result = {
            "voltage": round(battery_data.voltage, 2),
            "percent": battery_data.percent
        }
        print(json.dumps(result))
    except Exception as e:
        # Output error JSON so Rust knows battery reading failed
        print(json.dumps({"voltage": -1.0, "percent": -1}))
