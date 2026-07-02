from __future__ import annotations

import argparse
from pathlib import Path
import signal
import subprocess
import sys
import time


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _spawn_process(command: list[str], *, cwd: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=None,
        stderr=None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch Sourccey host + headless LiDAR TCP streamer for client-side mapping."
    )
    parser.add_argument("--lidar-port", default="/dev/ttyUSB0", help="Serial port for the LiDAR.")
    parser.add_argument("--lidar-baud", type=int, default=230400, help="Serial baud rate for the LiDAR.")
    parser.add_argument("--lidar-bind-host", default="0.0.0.0", help="TCP bind host for the LiDAR stream.")
    parser.add_argument("--lidar-bind-port", type=int, default=8765, help="TCP bind port for the LiDAR stream.")
    parser.add_argument(
        "--slam-publish-fps",
        type=float,
        default=15.0,
        help="Front stereo packet publish rate from Sourccey host.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = _repo_root()
    python_exe = sys.executable

    lidar_script = repo_root / "scripts" / "ldlidar_stream_host.py"
    if not lidar_script.exists():
        raise FileNotFoundError(f"LiDAR host script not found: {lidar_script}")

    host_cmd = [
        python_exe,
        "-m",
        "lerobot.robots.sourccey.sourccey.sourccey.sourccey_host",
        "--slam_input_enabled",
        "true",
        "--slam_imu_enabled",
        "true",
        "--slam_publish_fps",
        str(float(args.slam_publish_fps)),
    ]
    lidar_cmd = [
        python_exe,
        str(lidar_script),
        "--port",
        str(args.lidar_port),
        "--baud",
        str(int(args.lidar_baud)),
        "--bind-host",
        str(args.lidar_bind_host),
        "--bind-port",
        str(int(args.lidar_bind_port)),
        "--headless",
    ]

    print("Starting Sourccey offboard host stack")
    print("Host:", " ".join(host_cmd))
    print("LiDAR:", " ".join(lidar_cmd))

    children: list[subprocess.Popen[bytes]] = [
        _spawn_process(host_cmd, cwd=repo_root),
        _spawn_process(lidar_cmd, cwd=repo_root),
    ]

    stopping = False

    def _terminate_children() -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        for proc in children:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        deadline = time.time() + 5.0
        for proc in children:
            while proc.poll() is None and time.time() < deadline:
                time.sleep(0.05)
        for proc in children:
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _handle_signal(signum: int, _frame: object) -> None:
        print(f"Received signal {signum}, shutting down offboard host stack.")
        _terminate_children()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while True:
            for proc in children:
                code = proc.poll()
                if code is not None:
                    print(f"Child exited with code {code}, stopping offboard host stack.")
                    _terminate_children()
                    return int(code)
            time.sleep(0.2)
    finally:
        _terminate_children()


if __name__ == "__main__":
    raise SystemExit(main())
