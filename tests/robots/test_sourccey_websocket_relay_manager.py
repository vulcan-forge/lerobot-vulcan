from __future__ import annotations

import threading
from dataclasses import dataclass

from lerobot.robots.sourccey.sourccey.sourccey.modules.websocket_relay.manager import (
    WebsocketRelayManager,
)


@dataclass
class _HostConfig:
    websocket_relay_autostart: bool = True
    websocket_relay_forward_observations: bool = False


@dataclass
class _RelayConfig:
    websocket_relay_ws_base_url: str = "wss://relay.example.com"
    websocket_relay_session_id: str = "session-1234"
    websocket_relay_robot_token: str = "robot-token-1234"
    robot_id: str = "sourccey"
    zmq_cmd_endpoint: str = "tcp://127.0.0.1:5555"
    zmq_obs_endpoint: str = "tcp://127.0.0.1:5556"
    heartbeat_seconds: int = 5
    connect_retry_backoff_s: float = 0.01
    connect_retry_max_backoff_s: float = 0.01
    websocket_ping_interval_s: float = 0.0
    websocket_ping_timeout_s: float = 0.0
    log_actions: bool = True
    log_actions_interval_s: float = 30.0

    @property
    def ws_url(self) -> str:
        return (
            f"{self.websocket_relay_ws_base_url.rstrip('/')}/ws/robot"
            f"?session_id={self.websocket_relay_session_id}"
            f"&token={self.websocket_relay_robot_token}"
        )


def test_websocket_relay_manager_attempts_bridge_run_when_configured(monkeypatch) -> None:
    run_event = threading.Event()
    close_event = threading.Event()
    manager = WebsocketRelayManager(_HostConfig())

    monkeypatch.setattr(
        "lerobot.robots.sourccey.sourccey.sourccey.modules.websocket_relay.manager.WebsocketRelayConfig.from_env",
        lambda: _RelayConfig(),
    )

    class _FakeBridge:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def run_forever(self) -> None:
            run_event.set()
            manager._stop_event.set()

        async def close(self) -> None:
            close_event.set()

    monkeypatch.setattr(
        "lerobot.robots.sourccey.sourccey.sourccey.modules.websocket_relay.manager.WebsocketRelayBridge",
        _FakeBridge,
    )

    manager._thread_main()

    assert run_event.is_set()
    assert close_event.is_set()


def test_websocket_relay_manager_logs_connecting_once_across_retries(monkeypatch) -> None:
    manager = WebsocketRelayManager(_HostConfig())
    emitted_messages: list[str] = []
    attempts = {"count": 0}

    monkeypatch.setattr(
        "lerobot.robots.sourccey.sourccey.sourccey.modules.websocket_relay.manager.WebsocketRelayConfig.from_env",
        lambda: _RelayConfig(),
    )
    monkeypatch.setattr(
        "lerobot.robots.sourccey.sourccey.sourccey.modules.websocket_relay.manager.is_localhost_ws_url",
        lambda _ws_url: False,
    )

    async def _sleep_stub(_delay: float) -> None:
        return None

    class _FakeBridge:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def run_forever(self) -> None:
            attempts["count"] += 1
            if attempts["count"] >= 3:
                manager._stop_event.set()
                return
            raise TimeoutError("timed out during opening handshake")

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        "lerobot.robots.sourccey.sourccey.sourccey.modules.websocket_relay.manager.WebsocketRelayBridge",
        _FakeBridge,
    )
    monkeypatch.setattr(
        "lerobot.robots.sourccey.sourccey.sourccey.modules.websocket_relay.manager.asyncio.sleep",
        _sleep_stub,
    )
    monkeypatch.setattr(
        "lerobot.robots.sourccey.sourccey.sourccey.modules.websocket_relay.manager.print",
        lambda message: emitted_messages.append(message),
    )

    manager._thread_main()

    connecting_messages = [
        message for message in emitted_messages if "websocket_relay.connecting" in message
    ]
    connect_failed_messages = [
        message for message in emitted_messages if "websocket_relay.connect_failed" in message
    ]

    assert attempts["count"] == 3
    assert len(connecting_messages) == 1
    assert not connect_failed_messages
