from __future__ import annotations

import json

import pytest

from lerobot.robots.sourccey.sourccey.sourccey.modules.websocket_relay.config import (
    NoActiveRobotSessionError,
    WebsocketRelayConfig,
)


def test_from_env_uses_self_contained_cloud_credentials(monkeypatch, tmp_path) -> None:
    credentials_path = tmp_path / "cloud_device_credentials.json"
    credentials_path.write_text(
        json.dumps(
            {
                "relay_ws_base_url": "wss://relay.example.com",
                "device_auth_token": "device-token-1234",
                "active_session_id": "session-5678",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("VULCAN_DEVICE_CREDENTIALS_PATH", str(credentials_path))
    monkeypatch.delenv("VULCAN_WEBSOCKET_RELAY_WS_BASE_URL", raising=False)
    monkeypatch.delenv("VULCAN_WEBSOCKET_RELAY_SESSION_ID", raising=False)
    monkeypatch.delenv("VULCAN_WEBSOCKET_RELAY_ROBOT_TOKEN", raising=False)
    monkeypatch.delenv("VULCAN_RELAY_WS_BASE_URL", raising=False)
    monkeypatch.delenv("VULCAN_RELAY_SESSION_ID", raising=False)
    monkeypatch.delenv("VULCAN_RELAY_ROBOT_TOKEN", raising=False)

    cfg = WebsocketRelayConfig.from_env()

    assert cfg.websocket_relay_ws_base_url == "wss://relay.example.com"
    assert cfg.websocket_relay_session_id == "session-5678"
    assert cfg.websocket_relay_robot_token == "device-token-1234"


def test_from_env_preserves_explicit_env_over_cloud_defaults(monkeypatch, tmp_path) -> None:
    credentials_path = tmp_path / "cloud_device_credentials.json"
    credentials_path.write_text(
        json.dumps(
            {
                "relay_ws_base_url": "wss://relay.example.com",
                "device_auth_token": "device-token-1234",
                "active_session_id": "session-5678",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("VULCAN_DEVICE_CREDENTIALS_PATH", str(credentials_path))
    monkeypatch.setenv("VULCAN_WEBSOCKET_RELAY_WS_BASE_URL", "wss://override.example.com")
    monkeypatch.setenv("VULCAN_WEBSOCKET_RELAY_SESSION_ID", "env-session")
    monkeypatch.setenv("VULCAN_WEBSOCKET_RELAY_ROBOT_TOKEN", "env-token")

    cfg = WebsocketRelayConfig.from_env()

    assert cfg.websocket_relay_ws_base_url == "wss://override.example.com"
    assert cfg.websocket_relay_session_id == "env-session"
    assert cfg.websocket_relay_robot_token == "env-token"


def test_from_env_requires_session_id_when_pairing_state_missing(monkeypatch, tmp_path) -> None:
    credentials_path = tmp_path / "cloud_device_credentials.json"
    credentials_path.write_text(
        json.dumps(
            {
                "relay_ws_base_url": "wss://relay.example.com",
                "device_auth_token": "device-token-1234",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("VULCAN_DEVICE_CREDENTIALS_PATH", str(credentials_path))
    monkeypatch.delenv("VULCAN_WEBSOCKET_RELAY_SESSION_ID", raising=False)
    monkeypatch.delenv("VULCAN_RELAY_SESSION_ID", raising=False)

    with pytest.raises(RuntimeError, match="SESSION_ID"):
        WebsocketRelayConfig.from_env()


def test_from_env_falls_back_to_saved_credentials_when_active_session_is_idle(
    monkeypatch, tmp_path
) -> None:
    credentials_path = tmp_path / "cloud_device_credentials.json"
    credentials_path.write_text(
        json.dumps(
            {
                "relay_http_base_url": "https://relay.example.com",
                "relay_ws_base_url": "wss://relay.example.com",
                "device_auth_token": "device-token-1234",
                "active_session_id": "session-5678",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("VULCAN_DEVICE_CREDENTIALS_PATH", str(credentials_path))
    monkeypatch.delenv("VULCAN_WEBSOCKET_RELAY_WS_BASE_URL", raising=False)
    monkeypatch.delenv("VULCAN_WEBSOCKET_RELAY_SESSION_ID", raising=False)
    monkeypatch.delenv("VULCAN_WEBSOCKET_RELAY_ROBOT_TOKEN", raising=False)
    monkeypatch.delenv("VULCAN_RELAY_WS_BASE_URL", raising=False)
    monkeypatch.delenv("VULCAN_RELAY_SESSION_ID", raising=False)
    monkeypatch.delenv("VULCAN_RELAY_ROBOT_TOKEN", raising=False)
    monkeypatch.setattr(
        "lerobot.robots.sourccey.sourccey.sourccey.modules.websocket_relay.config._fetch_active_robot_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            NoActiveRobotSessionError("paired robot has no active cloud session")
        ),
    )

    cfg = WebsocketRelayConfig.from_env()

    assert cfg.websocket_relay_ws_base_url == "wss://relay.example.com"
    assert cfg.websocket_relay_session_id == "session-5678"
    assert cfg.websocket_relay_robot_token == "device-token-1234"


def test_from_env_raises_no_active_session_without_saved_relay_session(
    monkeypatch, tmp_path
) -> None:
    credentials_path = tmp_path / "cloud_device_credentials.json"
    credentials_path.write_text(
        json.dumps(
            {
                "relay_http_base_url": "https://relay.example.com",
                "device_auth_token": "device-token-1234",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("VULCAN_DEVICE_CREDENTIALS_PATH", str(credentials_path))
    monkeypatch.delenv("VULCAN_WEBSOCKET_RELAY_WS_BASE_URL", raising=False)
    monkeypatch.delenv("VULCAN_WEBSOCKET_RELAY_SESSION_ID", raising=False)
    monkeypatch.delenv("VULCAN_WEBSOCKET_RELAY_ROBOT_TOKEN", raising=False)
    monkeypatch.delenv("VULCAN_RELAY_WS_BASE_URL", raising=False)
    monkeypatch.delenv("VULCAN_RELAY_SESSION_ID", raising=False)
    monkeypatch.delenv("VULCAN_RELAY_ROBOT_TOKEN", raising=False)
    monkeypatch.setattr(
        "lerobot.robots.sourccey.sourccey.sourccey.modules.websocket_relay.config._fetch_active_robot_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            NoActiveRobotSessionError("paired robot has no active cloud session")
        ),
    )

    with pytest.raises(NoActiveRobotSessionError, match="no active cloud session"):
        WebsocketRelayConfig.from_env()
