from dataclasses import dataclass
from typing import Optional, List, Tuple
import time
import os

# gpiozero is only available on target hardware (e.g., Raspberry Pi).
# Make it optional so this module can be imported on dev machines.
try:
    from gpiozero import MCP3008  # type: ignore
except Exception:  # pragma: no cover
    MCP3008 = None  # type: ignore

# ADC Configuration
ADC_CHANNEL = 0

# Measure this with a multimeter if you can for better accuracy
VREF = 3.30  # Pi 3.3V rail (adjust to your measured value)

# Your divider: R1 = 3000, R2 = 750 -> 0.2
VOLTAGE_DIVIDER_RATIO = 0.2  # V_adc = V_batt * 0.2

# Sampling configuration
AVERAGE_SAMPLES = 25  # Number of samples to average for stability

# Battery voltage range for clamping
BATTERY_VOLTAGE_MIN = 11.0  # "Empty" (under load) – adjust based on your pack/BMS cutoff
BATTERY_VOLTAGE_MAX_OCV = 13.6  # "Full" resting open-circuit voltage
BATTERY_VOLTAGE_MAX_CHARGE = 14.6  # Charging voltage – treat as 100%
FILTER_ALPHA = 0.05  # 0..1, lower = more smoothing (lower = smoother)
_filtered_voltage: Optional[float] = None

# Smoothed percentage filter state
_last_percent: Optional[float] = None
PERCENT_ALPHA = 0.2  # 0..1, higher = more responsive, lower = smoother

# Charging detection (voltage-only heuristic)
# Notes:
# - This is best-effort with only voltage sensing. For a definitive signal, measure charge current
#   or a charger-present GPIO.
# - Tune thresholds based on your pack chemistry and charger behavior.
CHARGING_VOLTAGE_THRESHOLD = 13.9   # V; above this is very likely charging for a 4S LiFePO4 system
CHARGING_DVDT_THRESHOLD = 0.015     # V/s; sustained voltage rise suggests charging
CHARGING_DEBOUNCE_S = 2.0           # require charging-like condition for this long

_last_charge_eval_t: Optional[float] = None
_last_charge_eval_v: Optional[float] = None
_charging_since: Optional[float] = None
_is_charging: bool = False

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
    charging: bool

# Global ADC instance (initialized on first use)
_adc: Optional["MCP3008"] = None


def _get_adc() -> "MCP3008":
    """Get or initialize the ADC instance"""
    global _adc
    if MCP3008 is None:
        raise RuntimeError("gpiozero is not installed; battery ADC is unavailable on this machine.")
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
    Get a smoothed battery percentage based on voltage and the LiFePO4 curve.

    Returns:
        Battery percentage (0-100)
    """
    global _last_percent

    voltage = get_battery_voltage()

    # If we're clearly in "charging" range, just say 100%
    if voltage >= BATTERY_VOLTAGE_MAX_CHARGE:
        raw_percent = 100
    else:
        raw_percent = _voltage_to_percent_curve(voltage)

    # First sample: initialize the filter directly
    if _last_percent is None:
        _last_percent = float(raw_percent)
    else:
        # Exponential smoothing on the percentage itself
        _last_percent = (
            PERCENT_ALPHA * float(raw_percent)
            + (1.0 - PERCENT_ALPHA) * _last_percent
        )

    # Final clamp
    return max(0, min(100, int(round(_last_percent))))


def is_battery_charging() -> bool:
    """
    Best-effort charging detection based on filtered voltage trend.

    Returns:
        True if battery likely charging, else False.
    """
    print("Charging detection")
    global _last_charge_eval_t, _last_charge_eval_v, _charging_since, _is_charging

    v = get_battery_voltage()
    print("here 1")
    now = time.monotonic()
    print("here 2")
    print(f"last_charge_eval_t: {_last_charge_eval_t}")
    print(f"last_charge_eval_v: {_last_charge_eval_v}")
    # Initialize trend state on first call
    if _last_charge_eval_t is None or _last_charge_eval_v is None:
        _last_charge_eval_t = now
        _last_charge_eval_v = v
        _charging_since = None
        _is_charging = False
        print("here 3")
        return _is_charging
    print("here 4")

    dt = max(1e-3, now - _last_charge_eval_t)
    dvdt = (v - _last_charge_eval_v) / dt

    _last_charge_eval_t = now
    _last_charge_eval_v = v

    looks_like_charging = (v >= CHARGING_VOLTAGE_THRESHOLD) or (dvdt >= CHARGING_DVDT_THRESHOLD)
    print(
            f"[battery] V={v:.3f}V dt={dt:.3f}s dvdt={dvdt:.5f}V/s "
            f"looks_like_charging={looks_like_charging} charging={_is_charging}"
        )

    if looks_like_charging:
        if _charging_since is None:
            _charging_since = now
        if (now - _charging_since) >= CHARGING_DEBOUNCE_S:
            _is_charging = True
    else:
        _charging_since = None
        _is_charging = False

    return _is_charging


def get_battery_data() -> BatteryData:
    """
    Get both battery voltage and percentage.

    Returns:
        BatteryData containing voltage and percent
    """
    voltage = get_battery_voltage()
    percent = get_battery_percent()
    charging = is_battery_charging()
    return BatteryData(voltage=voltage, percent=percent, charging=charging)


if __name__ == "__main__":
    # When run as a script, output JSON (for Rust integration)
    import json
    try:
        battery_data = get_battery_data()
        result = {
            "voltage": round(battery_data.voltage, 2),
            "percent": battery_data.percent,
            "charging": battery_data.charging,
        }
        print(json.dumps(result))
    except Exception as e:
        # Output error JSON so Rust knows battery reading failed
        print(json.dumps({"voltage": -1.0, "percent": -1, "charging": False}))
