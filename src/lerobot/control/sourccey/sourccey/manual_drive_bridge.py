import json
import socket
import time
from dataclasses import dataclass

from lerobot.configs import parser
from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClient, SourcceyClientConfig
from lerobot.utils.robot_utils import precise_sleep

ALLOWED_KEYS = {"w", "a", "s", "d", "z", "x", "q", "e", "r", "f", "n", "m"}
ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass
class ManualDriveBridgeConfig:
    id: str = "sourccey"
    remote_ip: str = "127.0.0.1"
    udp_port: int = 5561
    fps: int = 30
    stale_timeout_ms: int = 250


def _connect_with_retry(robot: SourcceyClient, delay_s: float = 0.25) -> None:
    attempt = 0
    while True:
        attempt += 1
        try:
            robot.connect()
            print(f"Manual drive bridge connected after {attempt} attempt(s).")
            return
        except Exception as exc:
            print(f"Manual drive bridge connect attempt {attempt} failed: {exc}")
            time.sleep(delay_s)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _build_arm_hold_action(observation: dict[str, object]) -> dict[str, float]:
    action: dict[str, float] = {}
    for arm in ("left", "right"):
        for joint in ARM_JOINTS:
            key = f"{arm}_{joint}.pos"
            action[key] = _safe_float(observation.get(key, 0.0), 0.0)
    return action


def _sanitize_pressed_keys(raw_keys: list[object]) -> set[str]:
    sanitized: set[str] = set()
    for key in raw_keys:
        normalized = str(key).strip().lower()
        if normalized in ALLOWED_KEYS:
            sanitized.add(normalized)
    return sanitized


@parser.wrap()
def manual_drive_bridge(cfg: ManualDriveBridgeConfig):
    robot_config = SourcceyClientConfig(remote_ip=cfg.remote_ip, id=cfg.id)
    robot = SourcceyClient(robot_config)
    _connect_with_retry(robot)

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", cfg.udp_port))
    udp.setblocking(False)

    pressed_keys: set[str] = set()
    prev_effective_keys: set[str] = set()
    last_packet_time = time.monotonic()
    stale_timeout_s = max(float(cfg.stale_timeout_ms) / 1000.0, 0.05)
    last_observation: dict[str, object] = {}

    print(f"Manual drive bridge started on udp://127.0.0.1:{cfg.udp_port}")

    try:
        while True:
                try:
                    loop_started = time.perf_counter()

                    while True:
                        try:
                            payload_bytes, _ = udp.recvfrom(4096)
                        except BlockingIOError:
                            break

                        try:
                            payload = json.loads(payload_bytes.decode("utf-8"))
                        except Exception:
                            continue

                        packet_nickname = str(payload.get("nickname", "")).strip()
                        if packet_nickname and packet_nickname != cfg.id:
                            continue

                        packet_keys = payload.get("pressed_keys", [])
                        if isinstance(packet_keys, list):
                            pressed_keys = _sanitize_pressed_keys(packet_keys)
                            last_packet_time = time.monotonic()

                    try:
                        observation = robot.get_observation()
                        if isinstance(observation, dict) and observation:
                            last_observation = observation
                    except Exception:
                        observation = last_observation

                    if time.monotonic() - last_packet_time > stale_timeout_s:
                        effective_keys: set[str] = set()
                    else:
                        effective_keys = pressed_keys

                    # Speed keys are handled as key-down edges in SourcceyClient.
                    key_down_edges = effective_keys - prev_effective_keys
                    for key in sorted(key_down_edges):
                        try:
                            robot.on_key_down(key)
                        except Exception:
                            pass
                    prev_effective_keys = set(effective_keys)

                    z_obs_pos = _safe_float(last_observation.get("z.pos", 0.0), 0.0)
                    base_action = robot._from_keyboard_to_base_action(effective_keys, z_obs_pos=z_obs_pos)
                    arm_hold_action = _build_arm_hold_action(last_observation)

                    action = {**arm_hold_action, **base_action}
                    robot.send_action(action)

                    precise_sleep(max(1.0 / max(cfg.fps, 1) - (time.perf_counter() - loop_started), 0.0))
                except Exception as exc:
                    print(f"Manual drive bridge loop error: {exc}")
                    try:
                        robot.disconnect()
                    except Exception:
                        pass
                    _connect_with_retry(robot)
                    pressed_keys = set()
                    prev_effective_keys = set()
                    last_packet_time = time.monotonic()
    except KeyboardInterrupt:
        print("Manual drive bridge interrupted, shutting down.")
    finally:
        try:
            z_obs_pos = _safe_float(last_observation.get("z.pos", 0.0), 0.0)
            final_base = robot._from_keyboard_to_base_action(set(), z_obs_pos=z_obs_pos)
            final_base["x.vel"] = 0.0
            final_base["y.vel"] = 0.0
            final_base["theta.vel"] = 0.0
            final_action = {**_build_arm_hold_action(last_observation), **final_base}
            robot.send_action(final_action)
        except Exception:
            pass
        try:
            udp.close()
        except Exception:
            pass
        robot.disconnect()


def main():
    manual_drive_bridge()


if __name__ == "__main__":
    main()
