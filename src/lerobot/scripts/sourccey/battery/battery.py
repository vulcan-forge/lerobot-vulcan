from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BatteryData:
    voltage: float
    percent: int
    charging: bool
    current_a: float | None = None
    remaining_capacity_ah: float | None = None
    full_charge_capacity_ah: float | None = None


def _mode_is_inverted() -> bool:
    return os.getenv("SOURCCEY_BATTERY_MODE", "standard").strip().lower() in {"inverted", "invert"}


def _run_standard_backend() -> dict[str, Any]:
    script = Path(__file__).resolve().parent / "battery_standard.py"
    cmd = [sys.executable, str(script), "--json", "--voltage-mode", "pack"]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"{script.name} exited with code {proc.returncode}")

    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError(f"{script.name} produced no output")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{script.name} returned non-JSON output: {stdout}") from exc


def _from_payload(payload: dict[str, Any], inverted: bool) -> BatteryData:
    voltage = float(payload.get("pack_voltage_est_v", payload.get("voltage_cmd_v", -1.0)))
    raw_percent = int(payload.get("state_of_charge_pct", -1))
    current = float(payload.get("current_a", 0.0))
    remaining = float(payload.get("remaining_capacity_ah", 0.0))
    full = float(payload.get("full_charge_capacity_ah", 0.0))

    if inverted:
        if raw_percent >= 0:
            raw_percent = max(0, min(100, 100 - raw_percent))
        current = -current
        if full > 0.0:
            remaining = max(0.0, full - remaining)

    return BatteryData(
        voltage=voltage,
        percent=raw_percent,
        charging=current > 0.05,
        current_a=current,
        remaining_capacity_ah=remaining,
        full_charge_capacity_ah=full,
    )


def get_battery_data() -> BatteryData:
    payload = _run_standard_backend()
    return _from_payload(payload, inverted=_mode_is_inverted())


if __name__ == "__main__":
    try:
        battery_data = get_battery_data()
        result = {
            "voltage": round(battery_data.voltage, 3),
            "percent": battery_data.percent,
            "charging": battery_data.charging,
        }
        if battery_data.current_a is not None:
            result["current_a"] = round(battery_data.current_a, 3)
        if battery_data.remaining_capacity_ah is not None:
            result["remaining_capacity_ah"] = round(battery_data.remaining_capacity_ah, 3)
        if battery_data.full_charge_capacity_ah is not None:
            result["full_charge_capacity_ah"] = round(battery_data.full_charge_capacity_ah, 3)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"voltage": -1.0, "percent": -1, "charging": False, "error": str(e)}))
