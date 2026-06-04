from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from ...config_sourccey import SourcceyHostConfig

if TYPE_CHECKING:
    from ..websocket_relay.manager import WebsocketRelayManager


def _redact_env_value(value: str | None) -> str:
    if value is None:
        return "<unset>"
    value = value.strip()
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def _relay_env_snapshot() -> str:
    ws_base_url = os.getenv("VULCAN_WEBSOCKET_RELAY_WS_BASE_URL") or os.getenv(
        "VULCAN_RELAY_WS_BASE_URL"
    )
    session_id = os.getenv("VULCAN_WEBSOCKET_RELAY_SESSION_ID") or os.getenv(
        "VULCAN_RELAY_SESSION_ID"
    )
    robot_token = os.getenv("VULCAN_WEBSOCKET_RELAY_ROBOT_TOKEN") or os.getenv(
        "VULCAN_RELAY_ROBOT_TOKEN"
    )
    autostart = os.getenv("VULCAN_WEBSOCKET_RELAY_AUTOSTART") or os.getenv(
        "VULCAN_RELAY_AGENT_AUTOSTART"
    )
    credentials_path = os.getenv("VULCAN_DEVICE_CREDENTIALS_PATH") or os.getenv(
        "SOURCCEY_CLOUD_CREDENTIALS_PATH"
    )
    return (
        "Relay env snapshot: "
        f"ws_base_url={_redact_env_value(ws_base_url)}, "
        f"session_id={_redact_env_value(session_id)}, "
        f"robot_token={_redact_env_value(robot_token)}, "
        f"autostart={_redact_env_value(autostart)}, "
        f"credentials_path={_redact_env_value(credentials_path)}"
    )


def _relay_effective_snapshot() -> str:
    credentials_path = os.getenv("VULCAN_DEVICE_CREDENTIALS_PATH") or os.getenv(
        "SOURCCEY_CLOUD_CREDENTIALS_PATH"
    )
    try:
        from ..websocket_relay.config import WebsocketRelayConfig

        cfg = WebsocketRelayConfig.from_env()
        return (
            "Relay resolved config: "
            f"ws_base_url={cfg.websocket_relay_ws_base_url}, "
            f"session_id={cfg.websocket_relay_session_id}, "
            f"robot_token={cfg.websocket_relay_robot_token}, "
            f"robot_id={cfg.robot_id}, "
            f"ws_url={cfg.ws_url}, "
            f"credentials_path={credentials_path or '<unset>'}"
        )
    except Exception as exc:  # noqa: BLE001
        return (
            "Relay resolved config unavailable: "
            f"error={exc}, "
            f"credentials_path={credentials_path or '<unset>'}"
        )


def start_relay(host_config: SourcceyHostConfig) -> WebsocketRelayManager | None:
    print(_relay_env_snapshot())
    print(_relay_effective_snapshot())
    try:
        from ..websocket_relay.manager import WebsocketRelayManager

        relay = WebsocketRelayManager(host_config)
        if relay.start_if_configured():
            print("Relay started.")
        else:
            print("Relay disabled.")
        return relay
    except Exception as exc:  # noqa: BLE001
        logging.info("Relay unavailable: %s", exc)
        print(f"Relay unavailable: {exc}")
        print("Relay disabled.")
        return None


def poll_relay(relay: WebsocketRelayManager | None) -> None:
    if relay is not None:
        relay.poll()


def stop_relay(relay: WebsocketRelayManager | None) -> None:
    if relay is not None:
        relay.stop()
