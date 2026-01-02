from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    # checks/nick/check_position.py -> checks/nick -> checks -> REPO_ROOT
    return Path(__file__).resolve().parents[2]


def defaults_dir() -> Path:
    return (
        repo_root()
        / "src"
        / "lerobot"
        / "teleoperators"
        / "sourccey"
        / "sourccey"
        / "sourccey_leader"
        / "defaults"
    )


def arm_default_action_json_path(arm: str) -> Path:
    if arm not in {"left", "right"}:
        raise ValueError("arm must be 'left' or 'right'")
    return defaults_dir() / f"{arm}_arm_default_action.json"


def write_action_json(path: Path, data: dict[str, Any], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_src_on_path() -> None:
    """Allow importing from repo `src/` without requiring an editable install."""
    src_dir = repo_root() / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def read_live_arm_action(
    *,
    arm: str,
    port: str,
    base_id: str,
    calibration_dir: Path | None,
) -> dict[str, float]:
    """
    Connect to the real Feetech motors, read Present_Position (normalized), return a dict like:
      { "shoulder_pan.pos": ..., "gripper.pos": ..., ... }
    """
    if arm not in {"left", "right"}:
        raise ValueError("arm must be 'left' or 'right'")

    _ensure_src_on_path()
    from lerobot.robots.sourccey.sourccey.sourccey_follower.config_sourccey_follower import (  # noqa: E402
        SourcceyFollowerConfig,
    )
    from lerobot.robots.sourccey.sourccey.sourccey_follower.sourccey_follower import (  # noqa: E402
        SourcceyFollower,
    )

    cfg = SourcceyFollowerConfig(
        id=f"{base_id}_{arm}",
        calibration_dir=calibration_dir,
        port=port,
        orientation=arm,
        cameras={},  # don't try to connect cameras for this check script
    )
    robot = SourcceyFollower(cfg)

    # IMPORTANT: don't auto-calibrate here (follower.calibrate() is manual/interactive).
    if not robot.calibration:
        raise RuntimeError(
            f"No calibration loaded for id='{cfg.id}'.\n"
            f"Expected calibration file at: {robot.calibration_fpath}\n"
            "Create/choose the correct --calibration-dir and --id, or calibrate the arm first."
        )

    robot.connect(calibrate=False)
    try:
        obs = robot.get_observation()
        return {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
    finally:
        if robot.is_connected:
            robot.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="sourccey", help="Base id used to load calibration (<id>_left / <id>_right)")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="Directory containing calibration json files. If omitted, uses the default HF calibration location.",
    )

    parser.add_argument("--left-port", type=str, default=None, help="Serial port for LEFT follower (e.g. COM5)")
    parser.add_argument("--right-port", type=str, default=None, help="Serial port for RIGHT follower (e.g. COM6)")

    parser.add_argument("--print", action="store_true", default=True, help="Print live left/right actions read from motors")
    parser.add_argument("--write-left", action="store_true", help="Write live LEFT action to left_arm_default_action.json")
    parser.add_argument(
        "--write-right", action="store_true", help="Write live RIGHT action to right_arm_default_action.json"
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting destination default_action.json files")
    args = parser.parse_args()

    need_left = args.print or args.write_left
    need_right = args.print or args.write_right

    left_action: dict[str, float] | None = None
    right_action: dict[str, float] | None = None

    if need_left:
        if not args.left_port:
            raise SystemExit("--left-port is required for --print and/or --write-left")
        left_action = read_live_arm_action(
            arm="left",
            port=args.left_port,
            base_id=args.id,
            calibration_dir=args.calibration_dir,
        )

    if need_right:
        if not args.right_port:
            raise SystemExit("--right-port is required for --print and/or --write-right")
        right_action = read_live_arm_action(
            arm="right",
            port=args.right_port,
            base_id=args.id,
            calibration_dir=args.calibration_dir,
        )

    if args.print:
        if left_action is not None:
            print("Left arm (live from motors):")
            for k, v in left_action.items():
                print(f"  {k}: {v}")
        if right_action is not None:
            print("\nRight arm (live from motors):")
            for k, v in right_action.items():
                print(f"  {k}: {v}")

    if args.write_left and left_action is not None:
        dst = arm_default_action_json_path("left")
        write_action_json(dst, left_action, overwrite=args.overwrite)
        print(f"Wrote: {dst}")

    if args.write_right and right_action is not None:
        dst = arm_default_action_json_path("right")
        write_action_json(dst, right_action, overwrite=args.overwrite)
        print(f"Wrote: {dst}")


if __name__ == "__main__":
    main()
