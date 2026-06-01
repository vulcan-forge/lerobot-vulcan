from __future__ import annotations

import time
from dataclasses import dataclass

from lerobot.configs import parser
from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClient, SourcceyClientConfig


@dataclass
class SourcceySlamBridgeConfig:
    remote_ip: str = "192.168.1.214"
    robot_id: str = "sourccey"
    slam_input_endpoint: str = "tcp://127.0.0.1:5560"
    fps: int = 30
    log_interval_s: float = 2.0


@parser.wrap()
def main(cfg: SourcceySlamBridgeConfig) -> None:
    robot = SourcceyClient(
        SourcceyClientConfig(
            remote_ip=cfg.remote_ip,
            id=cfg.robot_id,
            slam_input_enabled=True,
            slam_input_endpoint=cfg.slam_input_endpoint,
        )
    )

    robot.connect()
    print(
        "SLAM bridge connected. Publishing slam_input.v1 on "
        f"{cfg.slam_input_endpoint}"
    )

    last_log = time.time()
    frame_count = 0

    try:
        while True:
            observation = robot.get_observation()
            frame_count += 1
            now = time.time()
            if now - last_log >= max(float(cfg.log_interval_s), 0.1):
                cameras = [
                    key
                    for key, value in observation.items()
                    if hasattr(value, "shape") and key != "observation.state"
                ]
                print(f"alive frames={frame_count} cameras={cameras}")
                last_log = now
            time.sleep(1 / max(int(cfg.fps), 1))
    except KeyboardInterrupt:
        print("Stopping SLAM bridge...")
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
