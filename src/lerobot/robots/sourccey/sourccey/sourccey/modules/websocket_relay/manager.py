from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Protocol
from urllib.parse import urlparse

import websockets

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
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] websocket_relay.autostart_enabled "
            f"mode={mode}"
        )
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

        def _emit(message: str) -> None:
            print(message)
            logging.info(message)

        _emit(f"[{_utc_now()}] websocket_relay.thread_started")

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
            waiting_for_session_logged = False
            localhost_warning_logged = False
            connecting_logged = False

            while not self._stop_event.is_set():
                try:
                    cfg = WebsocketRelayConfig.from_env()
                    waiting_for_session_logged = False
                except NoActiveRobotSessionError as exc:
                    if not waiting_for_session_logged:
                        _emit(
                            f"[{_utc_now()}] websocket_relay.waiting_for_active_session "
                            f"reason={exc}"
                        )
                        waiting_for_session_logged = True
                    await asyncio.sleep(2.0)
                    continue
                except Exception as exc:  # noqa: BLE001
                    waiting_for_session_logged = False
                    _emit(
                        f"[{_utc_now()}] websocket_relay.config_failed "
                        f"error_type={type(exc).__name__} error={exc!r}"
                    )
                    await asyncio.sleep(2.0)
                    continue

                if is_localhost_ws_url(cfg.websocket_relay_ws_base_url) and not localhost_warning_logged:
                    _emit(
                        f"[{_utc_now()}] websocket_relay.warn using localhost relay base URL "
                        f"({cfg.websocket_relay_ws_base_url}). If relay runs on another host, "
                        "set VULCAN_WEBSOCKET_RELAY_WS_BASE_URL (or legacy "
                        "VULCAN_RELAY_WS_BASE_URL) to that host/IP."
                    )
                    localhost_warning_logged = True

                backoff_s = cfg.connect_retry_backoff_s
                max_backoff_s = max(cfg.connect_retry_backoff_s, cfg.connect_retry_max_backoff_s)
                redacted_ws_url = _redact_ws_url(cfg.ws_url)
                bridge = WebsocketRelayBridge(
                    cfg,
                    forward_observations=self.config.websocket_relay_forward_observations,
                    forward_commands=True,
                )
                try:
                    if not connecting_logged:
                        _emit(
                            f"[{_utc_now()}] websocket_relay.connecting "
                            f"mode={mode} ws_url={redacted_ws_url}"
                        )
                        connecting_logged = True
                    await bridge.run_forever()
                except websockets.ConnectionClosed as exc:
                    _emit(
                        f"[{_utc_now()}] websocket_relay.disconnected "
                        f"session_id={cfg.websocket_relay_session_id} "
                        f"robot_id={cfg.robot_id} "
                        f"code={exc.code} reason={exc.reason or 'none'} reconnect_in_s=1.0"
                    )
                    connecting_logged = False
                    if self._stop_event.is_set():
                        break
                    await asyncio.sleep(1.0)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    if self._stop_event.is_set():
                        break
                    _emit(
                        f"[{_utc_now()}] websocket_relay.connect_failed "
                        f"retry_in_s={backoff_s:.1f} "
                        f"error_type={type(exc).__name__} error={exc!r}"
                    )
                    await asyncio.sleep(backoff_s)
                    backoff_s = min(max_backoff_s, backoff_s * 2.0)
                finally:
                    await bridge.close()

        try:
            asyncio.run(_runner())
        except Exception as exc:  # noqa: BLE001
            _emit(
                f"[{_utc_now()}] websocket_relay.thread_failed "
                f"error_type={type(exc).__name__} error={exc!r}"
            )
            logging.warning("Websocket relay thread exited: %s", exc)


def is_localhost_ws_url(ws_base_url: str) -> bool:
    try:
        host = (urlparse(ws_base_url).hostname or "").strip().lower()
    except Exception:
        return False
    return host in {"127.0.0.1", "localhost", "::1"}
