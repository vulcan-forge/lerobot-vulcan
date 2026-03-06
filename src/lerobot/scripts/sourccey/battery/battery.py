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
    current_sign: str | None = None
    remaining_capacity_ah: float | None = None
    full_charge_capacity_ah: float | None = None


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve_mode() -> str:
    # Default to standard when env var is absent.
    return os.getenv("SOURCCEY_BATTERY_MODE", "standard").strip().lower()


def _resolve_script_path(mode: str) -> Path:
    if mode in {"inverted", "invert"}:
        return _script_dir() / "battery_inverted.py"
    return _script_dir() / "battery_standard.py"


def _run_backend(script_path: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(script_path),
        "--json",
        "--voltage-mode",
        "pack",
    ]
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise RuntimeError(stderr or f"{script_path.name} exited with code {proc.returncode}")

    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError(f"{script_path.name} produced no output")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{script_path.name} returned non-JSON output: {stdout}") from exc


def _map_standard(payload: dict[str, Any]) -> BatteryData:
    current = float(payload.get("current_a", 0.0))
    return BatteryData(
        voltage=float(payload.get("pack_voltage_est_v", payload.get("voltage_cmd_v", -1.0))),
        percent=int(payload.get("state_of_charge_pct", -1)),
        charging=current > 0.05,
        current_a=current,
        current_sign="+" if current >= 0.0 else "-",
        remaining_capacity_ah=float(payload.get("remaining_capacity_ah", 0.0)),
        full_charge_capacity_ah=float(payload.get("full_charge_capacity_ah", 0.0)),
    )


def _map_inverted(payload: dict[str, Any]) -> BatteryData:
    current = float(payload.get("corrected_current_a", payload.get("current_a", 0.0)))
    percent = int(payload.get("software_soc_pct", payload.get("state_of_charge_pct", -1)))
    remaining = float(payload.get("software_remaining_capacity_ah", payload.get("remaining_capacity_ah", 0.0)))
    full = float(payload.get("software_full_charge_capacity_ah", payload.get("full_charge_capacity_ah", 0.0)))
    return BatteryData(
        voltage=float(payload.get("pack_voltage_est_v", payload.get("voltage_cmd_v", -1.0))),
        percent=percent,
        charging=current > 0.05,
        current_a=current,
        current_sign="+" if current >= 0.0 else "-",
        remaining_capacity_ah=remaining,
        full_charge_capacity_ah=full,
    )


def get_battery_data() -> BatteryData:
    mode = _resolve_mode()
    script_path = _resolve_script_path(mode)
    payload = _run_backend(script_path)
    if mode in {"inverted", "invert"}:
        return _map_inverted(payload)
    return _map_standard(payload)


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
            result["current_sign"] = battery_data.current_sign
        if battery_data.remaining_capacity_ah is not None:
            result["remaining_capacity_ah"] = round(battery_data.remaining_capacity_ah, 3)
        if battery_data.full_charge_capacity_ah is not None:
            result["full_charge_capacity_ah"] = round(battery_data.full_charge_capacity_ah, 3)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"voltage": -1.0, "percent": -1, "charging": False, "error": str(e)}))
