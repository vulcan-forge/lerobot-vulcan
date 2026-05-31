from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from urllib.parse import urlparse

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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _is_localhost_ws_url(ws_base_url: str) -> bool:
    try:
        host = (urlparse(ws_base_url).hostname or "").strip().lower()
    except Exception:
        return False
    return host in {"127.0.0.1", "localhost", "::1"}


def main() -> None:
    args = _parse_args()
    cfg = RelayAgentConfig.from_env()

    if args.once:
        print(f"relay_agent.ready=true ws_url={cfg.ws_url}")
        return

    if _is_localhost_ws_url(cfg.relay_ws_base_url):
        print(
            f"[{_utc_now()}] relay_agent.warn using localhost relay base URL "
            f"({cfg.relay_ws_base_url}). If relay runs on another host, set "
            "VULCAN_RELAY_WS_BASE_URL to that host/IP."
        )

    async def _runner() -> None:
        backoff_s = cfg.connect_retry_backoff_s
        max_backoff_s = max(cfg.connect_retry_backoff_s, cfg.connect_retry_max_backoff_s)
        while True:
            bridge = RelayBridge(cfg)
            try:
                print(f"[{_utc_now()}] relay_agent.connecting ws_url={cfg.ws_url}")
                await bridge.run_forever()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[{_utc_now()}] relay_agent.connect_failed "
                    f"retry_in_s={backoff_s:.1f} error={exc}"
                )
                await asyncio.sleep(backoff_s)
                backoff_s = min(max_backoff_s, backoff_s * 2.0)
            else:
                # Unexpected normal exit; reconnect defensively after short delay.
                await asyncio.sleep(1.0)
            finally:
                await bridge.close()

    asyncio.run(_runner())


if __name__ == "__main__":
    main()
