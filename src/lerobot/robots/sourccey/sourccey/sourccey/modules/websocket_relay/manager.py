from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Protocol
from urllib.parse import urlparse

from .bridge import WebsocketRelayBridge
from .config import NoActiveRobotSessionError, WebsocketRelayConfig


class HostWebsocketRelayConfig(Protocol):
    websocket_relay_autostart: bool
    websocket_relay_forward_observations: bool


class WebsocketRelayManager:
    def __init__(self, config: HostWebsocketRelayConfig):
        self.config = config
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False

    def start_if_configured(self) -> bool:
        if not self.config.websocket_relay_autostart:
            logging.info("Websocket relay autostart disabled.")
            return False

        mode = "full_bridge" if self.config.websocket_relay_forward_observations else "commands_only"
        logging.info("Websocket relay autostart enabled (mode=%s).", mode)
        try:
            self.start()
            return True
        except Exception as exc:  # noqa: BLE001
            logging.warning("Websocket relay failed to start (continuing without relay): %s", exc)
            return False

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
        try:
            if not self.config.websocket_relay_autostart:
                return
            if self._thread is not None and self._thread.is_alive():
                return
            if not self._started:
                return
            self.start()
        except Exception as exc:  # noqa: BLE001
            logging.warning("Websocket relay poll failed (continuing host loop): %s", exc)

    def stop(self) -> None:
        try:
            self._stop_event.set()
            thread = self._thread
            self._thread = None
            if thread is not None:
                thread.join(timeout=3.0)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Websocket relay stop failed: %s", exc)

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

        async def _runner() -> None:
            mode = "full_bridge" if self.config.websocket_relay_forward_observations else "commands_only"
            last_status: str | None = None

            while not self._stop_event.is_set():
                try:
                    cfg = WebsocketRelayConfig.from_env()
                except NoActiveRobotSessionError:
                    if last_status != "idle":
                        logging.info(
                            "[%s] websocket_relay.session_state status=idle waiting_for_active_session",
                            _utc_now(),
                        )
                        last_status = "idle"
                    await asyncio.sleep(2.0)
                    continue
                except Exception as exc:  # noqa: BLE001
                    logging.warning(
                        "[%s] websocket_relay.config_failed error_type=%s error=%r",
                        _utc_now(),
                        type(exc).__name__,
                        exc,
                    )
                    await asyncio.sleep(2.0)
                    continue

                if last_status != "ready":
                    logging.info(
                        "[%s] websocket_relay.session_state status=ready session_id=%s robot_id=%s",
                        _utc_now(),
                        cfg.websocket_relay_session_id,
                        cfg.robot_id,
                    )
                    last_status = "ready"

                if is_localhost_ws_url(cfg.websocket_relay_ws_base_url):
                    logging.warning(
                        "[%s] websocket_relay.warn using localhost relay base URL (%s). "
                        "If relay runs on another host, set VULCAN_WEBSOCKET_RELAY_WS_BASE_URL "
                        "(or legacy VULCAN_RELAY_WS_BASE_URL) to that host/IP.",
                        _utc_now(),
                        cfg.websocket_relay_ws_base_url,
                    )

                backoff_s = cfg.connect_retry_backoff_s
                max_backoff_s = max(cfg.connect_retry_backoff_s, cfg.connect_retry_max_backoff_s)
                redacted_ws_url = _redact_ws_url(cfg.ws_url)
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
                    logging.warning(
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
                    logging.warning(
                        "[%s] websocket_relay.disconnected session_id=%s robot_id=%s reconnect_in_s=1.0",
                        _utc_now(),
                        cfg.websocket_relay_session_id,
                        cfg.robot_id,
                    )
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
