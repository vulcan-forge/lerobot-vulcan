from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ...config_sourccey import SourcceyHostConfig

if TYPE_CHECKING:
    from ..websocket_relay.manager import WebsocketRelayManager

def start_relay(host_config: SourcceyHostConfig) -> WebsocketRelayManager | None:
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
