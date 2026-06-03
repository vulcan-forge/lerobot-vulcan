from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote


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
        cfg = cls(
            websocket_relay_ws_base_url=_env_str_alias(
                ("VULCAN_WEBSOCKET_RELAY_WS_BASE_URL", "VULCAN_RELAY_WS_BASE_URL"),
                "ws://127.0.0.1:5100",
            ),
            websocket_relay_session_id=_env_str_alias(
                ("VULCAN_WEBSOCKET_RELAY_SESSION_ID", "VULCAN_RELAY_SESSION_ID"), ""
            ),
            websocket_relay_robot_token=_env_str_alias(
                ("VULCAN_WEBSOCKET_RELAY_ROBOT_TOKEN", "VULCAN_RELAY_ROBOT_TOKEN"), ""
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
