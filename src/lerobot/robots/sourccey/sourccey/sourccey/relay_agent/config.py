from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote


@dataclass(slots=True)
class RelayAgentConfig:
    relay_ws_base_url: str
    relay_session_id: str
    relay_robot_token: str
    robot_id: str
    zmq_cmd_endpoint: str
    zmq_obs_endpoint: str
    heartbeat_seconds: int
    connect_retry_backoff_s: float
    connect_retry_max_backoff_s: float
    websocket_ping_interval_s: float
    websocket_ping_timeout_s: float
    log_actions: bool

    @classmethod
    def from_env(cls) -> "RelayAgentConfig":
        cfg = cls(
            relay_ws_base_url=os.getenv("VULCAN_RELAY_WS_BASE_URL", "ws://127.0.0.1:5100").strip(),
            relay_session_id=os.getenv("VULCAN_RELAY_SESSION_ID", "").strip(),
            relay_robot_token=os.getenv("VULCAN_RELAY_ROBOT_TOKEN", "").strip(),
            robot_id=os.getenv("VULCAN_ROBOT_ID", "sourccey").strip(),
            zmq_cmd_endpoint=os.getenv("VULCAN_ZMQ_CMD_ENDPOINT", "tcp://127.0.0.1:5555").strip(),
            zmq_obs_endpoint=os.getenv("VULCAN_ZMQ_OBS_ENDPOINT", "tcp://127.0.0.1:5556").strip(),
            heartbeat_seconds=max(1, int(os.getenv("VULCAN_HEARTBEAT_SECONDS", "5"))),
            connect_retry_backoff_s=max(0.5, float(os.getenv("VULCAN_RELAY_CONNECT_RETRY_BACKOFF_S", "2.0"))),
            connect_retry_max_backoff_s=max(1.0, float(os.getenv("VULCAN_RELAY_CONNECT_RETRY_MAX_BACKOFF_S", "30.0"))),
            websocket_ping_interval_s=float(os.getenv("VULCAN_RELAY_WS_PING_INTERVAL_S", "0")),
            websocket_ping_timeout_s=float(os.getenv("VULCAN_RELAY_WS_PING_TIMEOUT_S", "0")),
            log_actions=os.getenv("VULCAN_RELAY_LOG_ACTIONS", "true").strip().lower() in {"1", "true", "yes", "on"},
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.relay_session_id:
            raise RuntimeError("VULCAN_RELAY_SESSION_ID is required.")
        if not self.relay_robot_token:
            raise RuntimeError("VULCAN_RELAY_ROBOT_TOKEN is required.")

    @property
    def ws_url(self) -> str:
        return (
            f"{self.relay_ws_base_url.rstrip('/')}/ws/robot"
            f"?session_id={quote(self.relay_session_id)}"
            f"&token={quote(self.relay_robot_token)}"
        )
