import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.configs import parser
from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClient, SourcceyClientConfig
from lerobot.utils.robot_utils import precise_sleep

from .manual_drive_bridge import _connect_with_retry, _safe_float

ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

NAV_FOLLOW_STATUS_SCHEMA = "sourccey.nav_follow_live_status.v1"


@dataclass
class NavFollowBridgeConfig:
    id: str = "sourccey"
    remote_ip: str = "127.0.0.1"
    status_path: str = "artifacts/navigation/nav_follow_live_status.json"
    fps: int = 15
    stale_timeout_ms: int = 400
    status_log_interval_s: float = 1.0
    max_x_vel_m_s: float = 0.10
    max_theta_vel_rad_s: float = 0.35
    x_vel_scale: float = 1.0
    theta_vel_scale: float = 1.0
    exit_on_completed: bool = True


@dataclass
class NavFollowBridgeStatus:
    path: Path
    schema: str
    mtime_ns: int
    recommended_kind: str
    recommended_x_vel_m_s: float
    recommended_theta_vel_rad_s: float
    completed: bool
    detail: str
    target_waypoint_index: int
    progress_ratio: float
    distance_to_target_m: float | None
    heading_error_deg: float | None


def _build_arm_hold_action(observation: dict[str, object]) -> dict[str, float]:
    action: dict[str, float] = {}
    for arm in ("left", "right"):
        for joint in ARM_JOINTS:
            key = f"{arm}_{joint}.pos"
            action[key] = _safe_float(observation.get(key, 0.0), 0.0)
    return action


def _load_nav_follow_status(path: str | Path) -> NavFollowBridgeStatus:
    status_path = Path(path)
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    schema = str(payload.get("schema", ""))
    if schema != NAV_FOLLOW_STATUS_SCHEMA:
        raise ValueError(f"Unsupported nav follow status schema: {schema}")
    stat = status_path.stat()
    return NavFollowBridgeStatus(
        path=status_path,
        schema=schema,
        mtime_ns=stat.st_mtime_ns,
        recommended_kind=str(payload.get("recommended_kind", "stop")),
        recommended_x_vel_m_s=float(payload.get("recommended_x_vel_m_s", 0.0)),
        recommended_theta_vel_rad_s=float(payload.get("recommended_theta_vel_rad_s", 0.0)),
        completed=bool(payload.get("completed", False)),
        detail=str(payload.get("detail", "")),
        target_waypoint_index=int(payload.get("target_waypoint_index", 0)),
        progress_ratio=float(payload.get("progress_ratio", 0.0)),
        distance_to_target_m=(
            None
            if payload.get("distance_to_target_m") is None
            else float(payload.get("distance_to_target_m"))
        ),
        heading_error_deg=(
            None if payload.get("heading_error_deg") is None else float(payload.get("heading_error_deg"))
        ),
    )


def _is_status_stale(status: NavFollowBridgeStatus, stale_timeout_s: float) -> bool:
    age_s = (time.time_ns() - status.mtime_ns) / 1_000_000_000.0
    return age_s > stale_timeout_s


def _build_base_action_from_status(
    status: NavFollowBridgeStatus,
    cfg: NavFollowBridgeConfig,
) -> dict[str, float]:
    if status.completed or status.recommended_kind == "stop":
        return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

    x_vel = status.recommended_x_vel_m_s * float(cfg.x_vel_scale)
    theta_vel = status.recommended_theta_vel_rad_s * float(cfg.theta_vel_scale)
    x_vel = max(-float(cfg.max_x_vel_m_s), min(float(cfg.max_x_vel_m_s), x_vel))
    theta_vel = max(
        -float(cfg.max_theta_vel_rad_s),
        min(float(cfg.max_theta_vel_rad_s), theta_vel),
    )
    if status.recommended_kind == "turn":
        x_vel = 0.0
    elif status.recommended_kind != "drive":
        x_vel = 0.0
        theta_vel = 0.0
    return {"x.vel": float(x_vel), "y.vel": 0.0, "theta.vel": float(theta_vel)}


@parser.wrap()
def nav_follow_bridge(cfg: NavFollowBridgeConfig):
    robot_config = SourcceyClientConfig(remote_ip=cfg.remote_ip, id=cfg.id)
    robot = SourcceyClient(robot_config)
    _connect_with_retry(robot)

    status_path = Path(cfg.status_path)
    stale_timeout_s = max(float(cfg.stale_timeout_ms) / 1000.0, 0.05)
    log_interval_s = max(float(cfg.status_log_interval_s), 0.0)
    last_observation: dict[str, object] = {}
    last_log_time = 0.0
    last_summary: str | None = None

    print(
        "Nav follow bridge started "
        f"status={status_path} remote_ip={cfg.remote_ip} fps={cfg.fps}"
    )

    try:
        while True:
            loop_started = time.perf_counter()
            try:
                observation = robot.get_observation()
                if isinstance(observation, dict) and observation:
                    last_observation = observation
            except Exception:
                observation = last_observation

            z_hold = _safe_float(last_observation.get("z.pos", 0.0), 0.0)
            arm_hold = _build_arm_hold_action(last_observation)

            status: NavFollowBridgeStatus | None = None
            reason = "tracking_unavailable"
            try:
                status = _load_nav_follow_status(status_path)
            except Exception as exc:
                reason = f"status_error:{exc}"

            if status is not None and _is_status_stale(status, stale_timeout_s):
                reason = "status_stale"
                status = None

            if status is None:
                base_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
            else:
                reason = status.detail or "tracking"
                base_action = _build_base_action_from_status(status, cfg)

            action = {**arm_hold, **base_action, "z.pos": z_hold}
            robot.send_action(action)

            summary = (
                "stopping"
                if status is None
                else (
                    f"kind={status.recommended_kind} "
                    f"x={base_action['x.vel']:.2f} theta={base_action['theta.vel']:.2f} "
                    f"target_wp={status.target_waypoint_index} "
                    f"progress={status.progress_ratio * 100.0:.1f}% "
                    f"dist={0.0 if status.distance_to_target_m is None else status.distance_to_target_m:.2f}m "
                    f"heading_err={0.0 if status.heading_error_deg is None else status.heading_error_deg:.1f}deg "
                    f"detail={reason}"
                )
            )
            now = time.monotonic()
            if summary != last_summary or (log_interval_s > 0.0 and now - last_log_time >= log_interval_s):
                print(f"Nav follow bridge: {summary}")
                last_summary = summary
                last_log_time = now

            if status is not None and status.completed and cfg.exit_on_completed:
                print("Nav follow bridge: goal reached, stopping bridge.")
                break

            precise_sleep(max(1.0 / max(cfg.fps, 1) - (time.perf_counter() - loop_started), 0.0))
    except KeyboardInterrupt:
        print("Nav follow bridge interrupted, shutting down.")
    finally:
        try:
            z_hold = _safe_float(last_observation.get("z.pos", 0.0), 0.0)
            final_action = {
                **_build_arm_hold_action(last_observation),
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": 0.0,
                "z.pos": z_hold,
            }
            robot.send_action(final_action)
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass


def main():
    nav_follow_bridge()


if __name__ == "__main__":
    main()
