import json
import time
from pathlib import Path

from lerobot.control.sourccey.sourccey.apply_default_arm_pose import (
    _send_default_arm_pose_burst,
)
from lerobot.control.sourccey.sourccey.nav_follow_bridge import (
    NAV_FOLLOW_STATUS_SCHEMA,
    NavFollowBridgeConfig,
    _apply_min_effective_magnitude,
    _build_base_action_from_status,
    _build_default_arm_pose_action,
    _is_status_stale,
    _load_nav_follow_status,
    _resolve_motion_mode,
)


def _write_status(path: Path, **overrides):
    payload = {
        "schema": NAV_FOLLOW_STATUS_SCHEMA,
        "recommended_kind": "drive",
        "recommended_x_vel_m_s": 0.18,
        "recommended_theta_vel_rad_s": 0.22,
        "completed": False,
        "detail": "drive_to_target",
        "target_waypoint_index": 4,
        "progress_ratio": 0.55,
        "distance_to_target_m": 0.31,
        "heading_error_deg": 12.0,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_nav_follow_status_reads_expected_fields(tmp_path: Path):
    path = tmp_path / "nav_follow_live_status.json"
    _write_status(path, recommended_kind="turn", completed=True)

    status = _load_nav_follow_status(path)

    assert status.schema == NAV_FOLLOW_STATUS_SCHEMA
    assert status.recommended_kind == "turn"
    assert status.completed is True
    assert status.target_waypoint_index == 4
    assert status.progress_ratio == 0.55


def test_build_base_action_from_status_clamps_drive_command(tmp_path: Path):
    path = tmp_path / "nav_follow_live_status.json"
    _write_status(
        path,
        recommended_x_vel_m_s=0.42,
        recommended_theta_vel_rad_s=1.2,
        heading_error_deg=24.0,
    )
    status = _load_nav_follow_status(path)
    cfg = NavFollowBridgeConfig(
        max_x_vel_m_s=0.10,
        max_theta_vel_rad_s=0.35,
        drive_theta_mix=1.0,
        drive_theta_heading_deadband_deg=0.0,
    )

    action = _build_base_action_from_status(status, cfg, mode="drive")

    assert action == {"x.vel": 0.10, "y.vel": 0.0, "theta.vel": 0.35}


def test_build_base_action_from_status_turn_zeroes_forward_speed(tmp_path: Path):
    path = tmp_path / "nav_follow_live_status.json"
    _write_status(path, recommended_kind="turn", recommended_x_vel_m_s=0.18)
    status = _load_nav_follow_status(path)

    action = _build_base_action_from_status(status, NavFollowBridgeConfig(), mode="turn")

    assert action["x.vel"] == 0.0
    assert action["theta.vel"] == 0.22


def test_build_base_action_from_status_drive_suppresses_small_theta(tmp_path: Path):
    path = tmp_path / "nav_follow_live_status.json"
    _write_status(
        path,
        recommended_kind="drive",
        recommended_x_vel_m_s=0.12,
        recommended_theta_vel_rad_s=0.18,
        heading_error_deg=6.0,
    )
    status = _load_nav_follow_status(path)
    cfg = NavFollowBridgeConfig(
        x_vel_scale=3.0,
        max_x_vel_m_s=0.60,
        theta_vel_scale=2.0,
        max_theta_vel_rad_s=0.90,
        min_effective_x_vel_m_s=0.25,
        min_effective_theta_vel_rad_s=0.20,
        drive_theta_heading_deadband_deg=8.0,
        drive_theta_mix=0.35,
    )

    action = _build_base_action_from_status(status, cfg, mode="drive")

    assert action == {"x.vel": 0.36, "y.vel": 0.0, "theta.vel": 0.0}


def test_build_base_action_from_status_drive_keeps_large_theta_with_floor(tmp_path: Path):
    path = tmp_path / "nav_follow_live_status.json"
    _write_status(
        path,
        recommended_kind="drive",
        recommended_x_vel_m_s=0.06,
        recommended_theta_vel_rad_s=0.12,
        heading_error_deg=18.0,
    )
    status = _load_nav_follow_status(path)
    cfg = NavFollowBridgeConfig(
        x_vel_scale=4.0,
        max_x_vel_m_s=0.60,
        theta_vel_scale=2.0,
        max_theta_vel_rad_s=0.90,
        min_effective_x_vel_m_s=0.25,
        min_effective_theta_vel_rad_s=0.20,
        drive_theta_heading_deadband_deg=8.0,
        drive_theta_mix=0.35,
    )

    action = _build_base_action_from_status(status, cfg, mode="drive")

    assert action == {"x.vel": 0.25, "y.vel": 0.0, "theta.vel": 0.20}


def test_resolve_motion_mode_prefers_turn_until_exit_threshold(tmp_path: Path):
    path = tmp_path / "nav_follow_live_status.json"
    _write_status(path, recommended_kind="drive", heading_error_deg=18.0)
    status = _load_nav_follow_status(path)
    cfg = NavFollowBridgeConfig(turn_only_heading_enter_deg=14.0, turn_only_heading_exit_deg=7.0)

    assert _resolve_motion_mode(status, cfg, previous_mode="drive") == "turn"

    _write_status(path, recommended_kind="drive", heading_error_deg=9.0)
    status = _load_nav_follow_status(path)
    assert _resolve_motion_mode(status, cfg, previous_mode="turn") == "turn"

    _write_status(path, recommended_kind="drive", heading_error_deg=5.0)
    status = _load_nav_follow_status(path)
    assert _resolve_motion_mode(status, cfg, previous_mode="turn") == "drive"


def test_apply_min_effective_magnitude_preserves_zero():
    assert _apply_min_effective_magnitude(0.0, minimum_abs=0.25) == 0.0
    assert _apply_min_effective_magnitude(0.10, minimum_abs=0.25) == 0.25
    assert _apply_min_effective_magnitude(-0.10, minimum_abs=0.25) == -0.25


def test_is_status_stale_uses_file_mtime(tmp_path: Path):
    path = tmp_path / "nav_follow_live_status.json"
    _write_status(path)
    status = _load_nav_follow_status(path)
    assert _is_status_stale(status, stale_timeout_s=5.0) is False

    old_mtime = time.time() - 10.0
    Path(path).touch()
    import os

    os.utime(path, (old_mtime, old_mtime))
    stale_status = _load_nav_follow_status(path)
    assert _is_status_stale(stale_status, stale_timeout_s=0.5) is True


def test_build_default_arm_pose_action_uses_default_files_and_zero_base_motion():
    action = _build_default_arm_pose_action({"z.pos": 0.42})

    assert action["x.vel"] == 0.0
    assert action["y.vel"] == 0.0
    assert action["theta.vel"] == 0.0
    assert action["z.pos"] == 0.42
    assert "left_shoulder_pan.pos" in action
    assert "right_shoulder_pan.pos" in action


def test_send_default_arm_pose_burst_repeats_commands():
    class _FakeRobot:
        def __init__(self) -> None:
            self.actions: list[dict[str, float]] = []

        def send_action(self, action: dict[str, float]) -> None:
            self.actions.append(action)

    robot = _FakeRobot()

    _send_default_arm_pose_burst(
        robot,
        observation={"z.pos": 0.25},
        repeats=3,
        settle_s=0.0,
        hold_s=0.0,
    )

    assert len(robot.actions) == 3
    assert all(action["x.vel"] == 0.0 for action in robot.actions)
    assert all(action["z.pos"] == 0.25 for action in robot.actions)
