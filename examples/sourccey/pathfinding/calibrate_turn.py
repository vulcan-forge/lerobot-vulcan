import argparse
import json
import time
from pathlib import Path

from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClientConfig, SourcceyClient
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


def _load_default_arm_actions() -> dict[str, float]:
    """Load default arm actions from teleop JSONs (arms at side)."""
    try:
        from lerobot.teleoperators.sourccey.sourccey.sourccey_leader import sourccey_leader

        base_dir = Path(sourccey_leader.__file__).parent / "defaults"
        left_path = base_dir / "left_arm_default_action.json"
        right_path = base_dir / "right_arm_default_action.json"

        left = json.loads(left_path.read_text(encoding="utf-8")) if left_path.exists() else {}
        right = json.loads(right_path.read_text(encoding="utf-8")) if right_path.exists() else {}

        action = {f"left_{k}": float(v) for k, v in left.items()}
        action.update({f"right_{k}": float(v) for k, v in right.items()})
        return action
    except Exception:
        return {}


def _parse_args():
    p = argparse.ArgumentParser(description="Calibrate in-place turn timing")
    p.add_argument("--remote_ip", required=True, help="Raspberry Pi IP running sourccey_host")
    p.add_argument("--duration_s", type=float, default=0.6, help="Seconds to rotate in place")
    p.add_argument("--theta", type=float, default=0.7, help="theta.vel command to apply")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--reverse", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    init_rerun(session_name="sourccey_turn_calibration")

    robot_config = SourcceyClientConfig(remote_ip=args.remote_ip, reverse=args.reverse, id="sourccey")
    robot = SourcceyClient(robot_config)
    robot.connect()
    arm_action = _load_default_arm_actions()

    try:
        start = time.perf_counter()
        while True:
            action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": args.theta}
            robot.send_action({**arm_action, **action})

            try:
                obs = robot.get_observation()
            except Exception:
                obs = None
            log_rerun_data(obs, action, compress_images=True)

            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - start), 0.0))
            if time.perf_counter() - start >= args.duration_s:
                break
    finally:
        robot.send_action({**arm_action, **{"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}})
        robot.disconnect()

    print("Turn complete. Adjust --duration_s or --theta until ~90 degrees.")


if __name__ == "__main__":
    main()
