from __future__ import annotations

import argparse
import asyncio

from .bridge import RelayBridge
from .config import RelayAgentConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sourccey relay agent.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Validate config and print websocket URL, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = RelayAgentConfig.from_env()

    if args.once:
        print(f"relay_agent.ready=true ws_url={cfg.ws_url}")
        return

    bridge = RelayBridge(cfg)

    async def _runner() -> None:
        try:
            await bridge.run_forever()
        finally:
            await bridge.close()

    asyncio.run(_runner())


if __name__ == "__main__":
    main()
