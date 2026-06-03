import logging
from typing import TYPE_CHECKING

from ...config_sourccey import SourcceyHostConfig

if TYPE_CHECKING:
    from .bridge import RelayRobotBridge


def start_relay_bridge(host_config: SourcceyHostConfig) -> "RelayRobotBridge | None":
    try:
        from .bridge import RelayRobotBridge

        relay_bridge = RelayRobotBridge.from_environment(host_config)
        if relay_bridge is not None:
            relay_bridge.start()
            print("Relay robot bridge started.")
        else:
            print("Relay robot bridge disabled.")
        return relay_bridge
    except Exception as exc:
        logging.info("Relay robot bridge unavailable: %s", exc)
        print("Relay robot bridge disabled.")
        return None


def stop_relay_bridge(relay_bridge: "RelayRobotBridge | None") -> None:
    if relay_bridge is not None:
        relay_bridge.stop()
