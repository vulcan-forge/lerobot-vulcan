from dataclasses import dataclass
from typing import Optional, List, Tuple
import time
from collections import deque

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

# Charging detection (voltage-only heuristic; windowed slope)
# Notes:
# - With only voltage, this will never be perfect under load. For production-grade detection,
#   use a charger-present signal and/or charge current sensing.
# - This is a "just works" baseline: it samples voltage once per loop and estimates slope over
#   a rolling window to reduce noise.
# Note: the script’s bounded-sampling mode uses `_slope_v_per_s(samples)` directly, but these
# thresholds are also used for the final decision and for any codepaths that rely on the
# rolling window helper.
CHARGING_WINDOW_S = 5             # seconds of history to use for slope
CHARGING_MIN_POINTS = 4             # minimum samples required before using slope
CHARGING_SLOPE_THRESHOLD = 0.00005  # V/s (tune for your system; filtered voltage slopes are small)
CHARGING_VOLTAGE_THRESHOLD = 13.8   # V (tune; should reflect your charger/pack behavior)
# Script execution (when run as __main__)
# Use bounded sampling so the script returns quickly with a slope-based decision.
SCRIPT_WINDOW_S = 5               # total sampling time
SCRIPT_SAMPLE_PERIOD_S = 0.25       # time between samples
SCRIPT_INCLUDE_SLOPE = True         # include slope in JSON for validation/testing

_samples: "deque[tuple[float, float]]" = deque()  # (t, filtered_voltage)

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
    return get_battery_percent_from_voltage(voltage)


def get_battery_percent_from_voltage(voltage: float) -> int:
    """Compute smoothed battery percentage from a provided voltage reading."""
    global _last_percent

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


def _slope_v_per_s(samples: list[tuple[float, float]]) -> float:
    """Least-squares slope of voltage vs time over samples. Returns V/s."""
    n = len(samples)
    if n < 2:
        return 0.0

    t0 = samples[0][0]
    ts = [t - t0 for t, _ in samples]
    vs = [v for _, v in samples]

    t_mean = sum(ts) / n
    v_mean = sum(vs) / n
    num = sum((t - t_mean) * (v - v_mean) for t, v in zip(ts, vs))
    den = sum((t - t_mean) ** 2 for t in ts)
    return (num / den) if den > 0 else 0.0


def is_battery_charging_from_voltage(voltage: float, now: float) -> tuple[bool, float]:
    """
    Voltage-only charging detection using a rolling-window slope.

    Returns:
        (charging, slope_v_per_s)
    """
    _samples.append((now, voltage))

    cutoff = now - CHARGING_WINDOW_S
    while _samples and _samples[0][0] < cutoff:
        _samples.popleft()

    slope = _slope_v_per_s(list(_samples)) if len(_samples) >= 2 else 0.0

    charging = (voltage >= CHARGING_VOLTAGE_THRESHOLD) or (
        len(_samples) >= CHARGING_MIN_POINTS and slope >= CHARGING_SLOPE_THRESHOLD
    )

    return charging, slope


def get_battery_data() -> BatteryData:
    """
    Get both battery voltage and percentage.

    Returns:
        BatteryData containing voltage and percent
    """
    # Read voltage ONCE for consistency (percent + charging slope use the same reading)
    now = time.monotonic()
    voltage = get_battery_voltage()
    percent = get_battery_percent_from_voltage(voltage)
    charging, _slope = is_battery_charging_from_voltage(voltage, now)
    return BatteryData(voltage=voltage, percent=percent, charging=charging)


if __name__ == "__main__":
    # When run as a script, output JSON (for Rust integration)
    import json
    try:
        end_t = time.monotonic() + SCRIPT_WINDOW_S
        samples: list[tuple[float, float]] = []

        while time.monotonic() < end_t:
            t = time.monotonic()
            v = get_battery_voltage()
            samples.append((t, v))
            time.sleep(SCRIPT_SAMPLE_PERIOD_S)

        slope = _slope_v_per_s(samples)
        voltage = samples[-1][1] if samples else -1.0
        charging = (voltage >= CHARGING_VOLTAGE_THRESHOLD) or (
            len(samples) >= CHARGING_MIN_POINTS and slope >= CHARGING_SLOPE_THRESHOLD
        )

        result = {
            "voltage": round(voltage, 2),
            "percent": get_battery_percent_from_voltage(voltage),
            "charging": charging,
        }
        if SCRIPT_INCLUDE_SLOPE:
            result["slope_v_per_s"] = slope
            result["samples"] = len(samples)

        print(json.dumps(result))
    except Exception as e:
        # Output error JSON so Rust knows battery reading failed
        print(json.dumps({"voltage": -1.0, "percent": -1, "charging": False, "error": str(e)}))
