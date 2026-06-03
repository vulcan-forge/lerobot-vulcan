from .runtime import start_relay_bridge, stop_relay_bridge


def __getattr__(name: str):
    if name in {"DEFAULT_CREDENTIALS_PATH", "RelayRobotBridge", "RelayRobotCredentials"}:
        from .bridge import DEFAULT_CREDENTIALS_PATH, RelayRobotBridge, RelayRobotCredentials

        exports = {
            "DEFAULT_CREDENTIALS_PATH": DEFAULT_CREDENTIALS_PATH,
            "RelayRobotBridge": RelayRobotBridge,
            "RelayRobotCredentials": RelayRobotCredentials,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "start_relay_bridge",
    "stop_relay_bridge",
    "DEFAULT_CREDENTIALS_PATH",
    "RelayRobotBridge",
    "RelayRobotCredentials",
]
