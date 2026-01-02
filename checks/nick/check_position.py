from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    # checks/nick/check_position.py -> checks/nick -> checks -> REPO_ROOT
    return Path(__file__).resolve().parents[2]


def defaults_dir() -> Path:
    return repo_root() / "src" / "lerobot" / "teleoperators" / "sourccey" / "sourccey" / "sourccey_leader" / "defaults"


def arm_json_path(arm: str, *, active: bool) -> Path:
    if arm not in {"left", "right"}:
        raise ValueError("arm must be 'left' or 'right'")
    suffix = "default_active_action" if active else "default_action"
    return defaults_dir() / f"{arm}_arm_{suffix}.json"


def read_arm_positions(arm: str, *, active: bool = True) -> dict[str, float]:
    p = arm_json_path(arm, active=active)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {p}, got {type(data).__name__}")
    # Ensure float values
    return {str(k): float(v) for k, v in data.items()}


def write_action_json(path: Path, data: dict[str, Any], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    path.write_text(json.dumps(data, indent=4, sort_keys=True) + "\n", encoding="utf-8")


def copy_active_to_default(arm: str, *, overwrite: bool = False) -> Path:
    src = arm_json_path(arm, active=True)
    dst = arm_json_path(arm, active=False)
    data = read_arm_positions(arm, active=True)
    write_action_json(dst, data, overwrite=overwrite)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true", help="Print left/right action positions")
    parser.add_argument("--copy-right", action="store_true", help="Copy right_arm_default_action.json -> right_arm_default_action.json")
    parser.add_argument("--copy-left", action="store_true", help="Copy left_arm_default_action.json -> left_arm_default_action.json")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting destination default_action.json files")
    args = parser.parse_args()

    if args.print:
        left = read_arm_positions("left", active=True)
        right = read_arm_positions("right", active=True)
        print("Left arm (active):")
        for k, v in left.items():
            print(f"  {k}: {v}")
        print("\nRight arm (active):")
        for k, v in right.items():
            print(f"  {k}: {v}")

    if args.copy_right:
        dst = copy_active_to_default("right", overwrite=args.overwrite)
        print(f"Wrote: {dst}")

    if args.copy_left:
        dst = copy_active_to_default("left", overwrite=args.overwrite)
        print(f"Wrote: {dst}")


if __name__ == "__main__":
    main()
