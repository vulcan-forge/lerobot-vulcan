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

    @classmethod
    def from_env(cls) -> "RelayAgentConfig":
        cfg = cls(
            relay_ws_base_url=os.getenv("VULCAN_RELAY_WS_BASE_URL", "ws://127.0.0.1:5180").strip(),
            relay_session_id=os.getenv("VULCAN_RELAY_SESSION_ID", "").strip(),
            relay_robot_token=os.getenv("VULCAN_RELAY_ROBOT_TOKEN", "").strip(),
            robot_id=os.getenv("VULCAN_ROBOT_ID", "sourccey").strip(),
            zmq_cmd_endpoint=os.getenv("VULCAN_ZMQ_CMD_ENDPOINT", "tcp://127.0.0.1:5555").strip(),
            zmq_obs_endpoint=os.getenv("VULCAN_ZMQ_OBS_ENDPOINT", "tcp://127.0.0.1:5556").strip(),
            heartbeat_seconds=max(1, int(os.getenv("VULCAN_HEARTBEAT_SECONDS", "5"))),
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
