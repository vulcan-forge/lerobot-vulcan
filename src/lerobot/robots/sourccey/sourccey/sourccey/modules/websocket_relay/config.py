from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


_CLOUD_PAIRING_STATE_FILE_NAME = "cloud_pairing_state.json"


def _env_str_alias(names: tuple[str, ...], default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        value = value.strip()
        if value:
            return value
    return default


def _env_float_alias(names: tuple[str, ...], default: float) -> float:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _env_bool_alias(names: tuple[str, ...], default: bool) -> bool:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _env_path_alias(names: tuple[str, ...]) -> Path | None:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        value = value.strip()
        if value:
            return Path(value)
    return None


def _json_str(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _load_json_dict(path: Path) -> dict[str, object]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed, dict):
        return {}

    return parsed


def _load_cloud_relay_defaults() -> dict[str, str]:
    credentials_path = _env_path_alias(
        ("VULCAN_DEVICE_CREDENTIALS_PATH", "SOURCCEY_CLOUD_CREDENTIALS_PATH")
    )
    if credentials_path is None:
        return {}

    credentials = _load_json_dict(credentials_path)
    if not credentials:
        return {}

    pairing_state = _load_json_dict(
        credentials_path.with_name(_CLOUD_PAIRING_STATE_FILE_NAME)
    )

    return {
        "websocket_relay_ws_base_url": _json_str(credentials.get("relay_ws_base_url")),
        # Device auth tokens are the long-lived robot credentials persisted after claim.
        "websocket_relay_robot_token": _json_str(credentials.get("device_auth_token")),
        "websocket_relay_session_id": _json_str(
            credentials.get("active_session_id")
            or pairing_state.get("active_session_id")
            or credentials.get("session_id")
        ),
    }


@dataclass(slots=True)
class WebsocketRelayConfig:
    websocket_relay_ws_base_url: str
    websocket_relay_session_id: str
    websocket_relay_robot_token: str
    robot_id: str
    zmq_cmd_endpoint: str
    zmq_obs_endpoint: str
    heartbeat_seconds: int
    connect_retry_backoff_s: float
    connect_retry_max_backoff_s: float
    websocket_ping_interval_s: float
    websocket_ping_timeout_s: float
    log_actions: bool
    log_actions_interval_s: float

    @classmethod
    def from_env(cls) -> "WebsocketRelayConfig":
        cloud_defaults = _load_cloud_relay_defaults()
        cfg = cls(
            websocket_relay_ws_base_url=(
                _env_str_alias(
                    ("VULCAN_WEBSOCKET_RELAY_WS_BASE_URL", "VULCAN_RELAY_WS_BASE_URL"),
                    "",
                )
                or cloud_defaults.get("websocket_relay_ws_base_url", "")
                or "ws://127.0.0.1:5100"
            ),
            websocket_relay_session_id=_env_str_alias(
                ("VULCAN_WEBSOCKET_RELAY_SESSION_ID", "VULCAN_RELAY_SESSION_ID"),
                "",
            ),
            websocket_relay_robot_token=(
                _env_str_alias(
                    ("VULCAN_WEBSOCKET_RELAY_ROBOT_TOKEN", "VULCAN_RELAY_ROBOT_TOKEN"),
                    "",
                )
                or cloud_defaults.get("websocket_relay_robot_token", "")
            ),
            robot_id=os.getenv("VULCAN_ROBOT_ID", "sourccey").strip(),
            zmq_cmd_endpoint=os.getenv("VULCAN_ZMQ_CMD_ENDPOINT", "tcp://127.0.0.1:5555").strip(),
            zmq_obs_endpoint=os.getenv("VULCAN_ZMQ_OBS_ENDPOINT", "tcp://127.0.0.1:5556").strip(),
            heartbeat_seconds=max(1, int(os.getenv("VULCAN_HEARTBEAT_SECONDS", "5"))),
            connect_retry_backoff_s=max(
                0.5,
                _env_float_alias(
                    (
                        "VULCAN_WEBSOCKET_RELAY_CONNECT_RETRY_BACKOFF_S",
                        "VULCAN_RELAY_CONNECT_RETRY_BACKOFF_S",
                    ),
                    2.0,
                ),
            ),
            connect_retry_max_backoff_s=max(
                1.0,
                _env_float_alias(
                    (
                        "VULCAN_WEBSOCKET_RELAY_CONNECT_RETRY_MAX_BACKOFF_S",
                        "VULCAN_RELAY_CONNECT_RETRY_MAX_BACKOFF_S",
                    ),
                    30.0,
                ),
            ),
            websocket_ping_interval_s=_env_float_alias(
                ("VULCAN_WEBSOCKET_RELAY_WS_PING_INTERVAL_S", "VULCAN_RELAY_WS_PING_INTERVAL_S"),
                0.0,
            ),
            websocket_ping_timeout_s=_env_float_alias(
                ("VULCAN_WEBSOCKET_RELAY_WS_PING_TIMEOUT_S", "VULCAN_RELAY_WS_PING_TIMEOUT_S"),
                0.0,
            ),
            log_actions=_env_bool_alias(
                ("VULCAN_WEBSOCKET_RELAY_LOG_ACTIONS", "VULCAN_RELAY_LOG_ACTIONS"), True
            ),
            log_actions_interval_s=max(
                1.0,
                _env_float_alias(
                    (
                        "VULCAN_WEBSOCKET_RELAY_LOG_ACTIONS_INTERVAL_S",
                        "VULCAN_RELAY_LOG_ACTIONS_INTERVAL_S",
                    ),
                    30.0,
                ),
            ),
        )
        if not cfg.websocket_relay_session_id:
            cfg.websocket_relay_session_id = cloud_defaults.get("websocket_relay_session_id", "")
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.websocket_relay_session_id:
            raise RuntimeError(
                "VULCAN_WEBSOCKET_RELAY_SESSION_ID (or legacy VULCAN_RELAY_SESSION_ID) is required."
            )
        if not self.websocket_relay_robot_token:
            raise RuntimeError(
                "VULCAN_WEBSOCKET_RELAY_ROBOT_TOKEN (or legacy VULCAN_RELAY_ROBOT_TOKEN) is required."
            )

    @property
    def ws_url(self) -> str:
        return (
            f"{self.websocket_relay_ws_base_url.rstrip('/')}/ws/robot"
            f"?session_id={quote(self.websocket_relay_session_id)}"
            f"&token={quote(self.websocket_relay_robot_token)}"
        )
