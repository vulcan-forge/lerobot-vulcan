from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Protocol
from urllib.parse import urlparse

from .bridge import WebsocketRelayBridge
from .config import WebsocketRelayConfig


class HostWebsocketRelayConfig(Protocol):
    websocket_relay_autostart: bool
    websocket_relay_forward_observations: bool


class WebsocketRelayManager:
    def __init__(self, config: HostWebsocketRelayConfig):
        self.config = config
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._thread_main,
            daemon=True,
            name="sourccey_websocket_relay",
        )
        self._thread.start()
        self._started = True

    def poll(self) -> None:
        if not self.config.websocket_relay_autostart:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        if not self._started:
            return
        self.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=3.0)

    def _thread_main(self) -> None:
        import asyncio

        def _utc_now() -> str:
            return datetime.now(timezone.utc).isoformat()

        def _redact_ws_url(ws_url: str) -> str:
            token_marker = "token="
            token_idx = ws_url.find(token_marker)
            if token_idx == -1:
                return ws_url
            token_start = token_idx + len(token_marker)
            token_end = ws_url.find("&", token_start)
            if token_end == -1:
                token_end = len(ws_url)
            visible = ws_url[token_start : min(token_start + 4, token_end)]
            return f"{ws_url[:token_start]}{visible}...redacted{ws_url[token_end:]}"

        try:
            cfg = WebsocketRelayConfig.from_env()
        except Exception as exc:  # noqa: BLE001
            logging.debug("Websocket relay start skipped: invalid websocket relay env/config (%s)", exc)
            return

        if is_localhost_ws_url(cfg.websocket_relay_ws_base_url):
            logging.warning(
                "[%s] websocket_relay.warn using localhost relay base URL (%s). "
                "If relay runs on another host, set VULCAN_WEBSOCKET_RELAY_WS_BASE_URL "
                "(or legacy VULCAN_RELAY_WS_BASE_URL) to that host/IP.",
                _utc_now(),
                cfg.websocket_relay_ws_base_url,
            )

        async def _runner() -> None:
            backoff_s = cfg.connect_retry_backoff_s
            max_backoff_s = max(cfg.connect_retry_backoff_s, cfg.connect_retry_max_backoff_s)
            mode = "full_bridge" if self.config.websocket_relay_forward_observations else "commands_only"
            redacted_ws_url = _redact_ws_url(cfg.ws_url)

            while not self._stop_event.is_set():
                bridge = WebsocketRelayBridge(
                    cfg,
                    forward_observations=self.config.websocket_relay_forward_observations,
                    forward_commands=True,
                )
                try:
                    logging.info(
                        "[%s] websocket_relay.connecting mode=%s ws_url=%s",
                        _utc_now(),
                        mode,
                        redacted_ws_url,
                    )
                    await bridge.run_forever()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logging.debug(
                        "[%s] websocket_relay.connect_failed retry_in_s=%.1f error_type=%s error=%r",
                        _utc_now(),
                        backoff_s,
                        type(exc).__name__,
                        exc,
                    )
                    if self._stop_event.is_set():
                        break
                    await asyncio.sleep(backoff_s)
                    backoff_s = min(max_backoff_s, backoff_s * 2.0)
                else:
                    if self._stop_event.is_set():
                        break
                    await asyncio.sleep(1.0)
                finally:
                    await bridge.close()

        try:
            asyncio.run(_runner())
        except Exception as exc:  # noqa: BLE001
            logging.warning("Websocket relay thread exited: %s", exc)


def is_localhost_ws_url(ws_base_url: str) -> bool:
    try:
        host = (urlparse(ws_base_url).hostname or "").strip().lower()
    except Exception:
        return False
    return host in {"127.0.0.1", "localhost", "::1"}
