import json
import time
from pathlib import Path

from lerobot.control.sourccey.sourccey.nav_follow_bridge import (
    NAV_FOLLOW_STATUS_SCHEMA,
    NavFollowBridgeConfig,
    _build_base_action_from_status,
    _is_status_stale,
    _load_nav_follow_status,
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
    _write_status(path, recommended_x_vel_m_s=0.42, recommended_theta_vel_rad_s=1.2)
    status = _load_nav_follow_status(path)
    cfg = NavFollowBridgeConfig(max_x_vel_m_s=0.10, max_theta_vel_rad_s=0.35)

    action = _build_base_action_from_status(status, cfg)

    assert action == {"x.vel": 0.10, "y.vel": 0.0, "theta.vel": 0.35}


def test_build_base_action_from_status_turn_zeroes_forward_speed(tmp_path: Path):
    path = tmp_path / "nav_follow_live_status.json"
    _write_status(path, recommended_kind="turn", recommended_x_vel_m_s=0.18)
    status = _load_nav_follow_status(path)

    action = _build_base_action_from_status(status, NavFollowBridgeConfig())

    assert action["x.vel"] == 0.0
    assert action["theta.vel"] == 0.22


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
