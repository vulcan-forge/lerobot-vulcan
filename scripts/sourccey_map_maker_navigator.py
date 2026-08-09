"""Build one fresh 360deg Sourccey LiDAR map, then navigate on it.

This is the short, practical operator path:

1. Run the existing explorer in spin-only handoff mode.
2. Save that initial 360deg anchor map to a timestamped NPZ.
3. Launch the saved-map navigator on that exact map.
4. When the navigator exits, copy its writable working map to a new grown-map NPZ.

The mapping and navigation logic intentionally stays in the proven existing
scripts. This file is glue, not a second SLAM implementation.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_STOW_POSE = Path("scripts") / "sourccey_arm_stow_pose_sourccey_current_bot.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _default_map_path() -> Path:
    return _script_dir() / f"sourccey_saved_map_{_timestamp()}_mapmaker.npz"


def _run(command: list[str], *, cwd: Path, label: str) -> None:
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(f"\n[map-maker] {label}")
    print(f"[map-maker] {printable}")
    completed = subprocess.run(command, cwd=str(cwd))
    if completed.returncode != 0:
        raise SystemExit(f"[map-maker] {label} failed with exit code {completed.returncode}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-ip", required=True, help="Robot/Pi host IP address.")
    parser.add_argument("--robot-id", default="sourccey")
    parser.add_argument("--lidar-port", type=int, default=8765)
    parser.add_argument("--imu-yaw-port", type=int, default=8770)
    parser.add_argument("--imu-yaw-sign", type=float, default=1.0)
    parser.add_argument(
        "--map-output",
        default=None,
        help="Initial map output NPZ. Defaults to scripts/sourccey_saved_map_<timestamp>_mapmaker.npz.",
    )
    parser.add_argument(
        "--final-map-output",
        default=None,
        help="Final grown map copy written after navigator exits. Defaults beside --map-output.",
    )
    parser.add_argument(
        "--anchor-passes",
        type=int,
        choices=(1, 2),
        default=1,
        help="Initial mapping passes. 1 is the requested single 360deg sweep; 2 does reverse verification.",
    )
    parser.add_argument("--spin-speed", type=float, default=0.80)
    parser.add_argument("--max-spin-seconds", type=float, default=75.0)
    parser.add_argument("--matcher-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--rerun-mode", choices=("web", "local"), default="web")
    parser.add_argument("--slam-input-endpoint", default="")
    parser.add_argument("--bottom-odometry", choices=("on", "off", "required"), default="on")
    parser.add_argument("--drive-command-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--auto-correct-drive-command-sign", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reverse-progress-trigger-m", type=float, default=0.08)
    parser.add_argument(
        "--arm-stow",
        action="store_true",
        help="Move arms to --arm-stow-pose during the initial 360deg map build.",
    )
    parser.add_argument(
        "--arm-stow-pose",
        default=str(DEFAULT_STOW_POSE),
        help="Saved arm stow pose JSON used with --arm-stow.",
    )
    parser.add_argument(
        "--skip-initial-sweep",
        action="store_true",
        help="Debug/operator escape hatch: skip map building and open the navigator on --map-output.",
    )
    parser.add_argument(
        "--reset-working-map",
        action="store_true",
        help="Tell the navigator to rebuild its writable working map from the fresh source map.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo = _repo_root()
    scripts = _script_dir()
    map_output = Path(args.map_output) if args.map_output else _default_map_path()
    if not map_output.is_absolute():
        map_output = repo / map_output
    map_output.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_initial_sweep:
        explore_cmd = [
            sys.executable,
            str(scripts / "sourccey_explore.py"),
            "--remote-ip",
            str(args.remote_ip),
            "--robot-id",
            str(args.robot_id),
            "--lidar-port",
            str(args.lidar_port),
            "--imu-yaw-port",
            str(args.imu_yaw_port),
            "--imu-yaw-sign",
            str(args.imu_yaw_sign),
            "--spin-only",
            "--exit-after-spin-only",
            "--anchor-passes",
            str(args.anchor_passes),
            "--spin-speed",
            str(args.spin_speed),
            "--max-spin-seconds",
            str(args.max_spin_seconds),
            "--matcher-device",
            str(args.matcher_device),
            "--rerun-mode",
            str(args.rerun_mode),
            "--saved-map",
            str(map_output),
        ]
        if args.slam_input_endpoint:
            explore_cmd.extend(["--slam-input-endpoint", str(args.slam_input_endpoint)])
        if args.arm_stow:
            explore_cmd.extend(["--arm-stow", "--arm-stow-pose", str(args.arm_stow_pose)])
        _run(explore_cmd, cwd=repo, label="building initial 360deg map")
    elif not map_output.exists():
        raise SystemExit(f"[map-maker] --skip-initial-sweep requires an existing --map-output ({map_output}).")

    if not map_output.exists():
        raise SystemExit(f"[map-maker] Initial map was not created: {map_output}")

    navigator_cmd = [
        sys.executable,
        str(scripts / "sourccey_saved_map_navigator.py"),
        "--remote-ip",
        str(args.remote_ip),
        "--robot-id",
        str(args.robot_id),
        "--map",
        str(map_output),
        "--lidar-port",
        str(args.lidar_port),
        "--imu-yaw-port",
        str(args.imu_yaw_port),
        "--imu-yaw-sign",
        str(args.imu_yaw_sign),
        "--bottom-odometry",
        str(args.bottom_odometry),
        "--drive-command-sign",
        str(args.drive_command_sign),
        "--reverse-progress-trigger-m",
        str(args.reverse_progress_trigger_m),
        "--trust-map-start-pose",
    ]
    if args.slam_input_endpoint:
        navigator_cmd.extend(["--slam-input-endpoint", str(args.slam_input_endpoint)])
    if not bool(args.auto_correct_drive_command_sign):
        navigator_cmd.append("--no-auto-correct-drive-command-sign")
    if args.reset_working_map:
        navigator_cmd.append("--reset-working-map")

    try:
        _run(navigator_cmd, cwd=repo, label="opening navigator on the fresh map")
    finally:
        working_map = map_output.with_name(f"{map_output.stem}_navigator_working.npz")
        source_for_final = working_map if working_map.exists() else map_output
        if args.final_map_output:
            final_map = Path(args.final_map_output)
            if not final_map.is_absolute():
                final_map = repo / final_map
        else:
            final_map = map_output.with_name(f"{map_output.stem}_grown_{_timestamp()}.npz")
        final_map.parent.mkdir(parents=True, exist_ok=True)
        if source_for_final.exists():
            shutil.copy2(source_for_final, final_map)
            print(f"[map-maker] final grown map saved: {final_map}")
        else:
            print(f"[map-maker] WARNING: no final map source found at {source_for_final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

