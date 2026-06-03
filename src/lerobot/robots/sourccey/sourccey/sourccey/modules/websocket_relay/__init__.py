from __future__ import annotations

from .manager import WebsocketRelayManager


def main() -> None:
    from .main import main as _main

    _main()


__all__ = ["main", "WebsocketRelayManager"]
