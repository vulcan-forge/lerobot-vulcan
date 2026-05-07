from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PendingToggleRequest:
    request_id: int
    peer: str
    created_at: float = field(default_factory=time.time)
    completed: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None


class RemoteRecordingControlServer:
    def __init__(self, *, host: str, port: int, response_timeout_s: float = 1.5) -> None:
        self.host = host
        self.port = port
        self.response_timeout_s = response_timeout_s

        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._pending_toggles: deque[PendingToggleRequest] = deque()
        self._next_request_id = 1
        self._phase = "starting"
        self._recording = False

    def start(self) -> None:
        if self._httpd is not None:
            return

        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                logger.debug("Remote recording HTTP: " + format, *args)

            def _write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/recording/status":
                    self._write_json(server.get_status())
                    return
                if self.path == "/recording/ping":
                    self._write_json({"ok": True, "message": "pong"})
                    return
                self._write_json(
                    {"ok": False, "message": f"Unknown endpoint: {self.path}"},
                    status=HTTPStatus.NOT_FOUND,
                )

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/recording/toggle":
                    self._write_json(
                        {"ok": False, "message": f"Unknown endpoint: {self.path}"},
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return

                peer = self.client_address[0] if self.client_address else "unknown"
                logger.info("HTTP POST /recording/toggle from %s", peer)
                response = server.enqueue_toggle(peer=peer)
                status = HTTPStatus.OK if response.get("ok", False) else HTTPStatus.SERVICE_UNAVAILABLE
                self._write_json(response, status=status)

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="remote-recording-control",
            daemon=True,
        )
        self._thread.start()
        logger.info("Remote recording control server listening on http://%s:%s", self.host, self.port)

    def stop(self) -> None:
        if self._httpd is None:
            return

        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._httpd = None
        self._thread = None

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ok": True,
                "phase": self._phase,
                "recording": self._recording,
            }

    def set_phase(self, phase: str, *, recording: bool | None = None) -> None:
        with self._lock:
            self._phase = phase
            if recording is not None:
                self._recording = bool(recording)

    def enqueue_toggle(self, *, peer: str) -> dict[str, Any]:
        with self._lock:
            request = PendingToggleRequest(request_id=self._next_request_id, peer=peer)
            self._next_request_id += 1
            self._pending_toggles.append(request)
            logger.info("Queued remote recording toggle request id=%s", request.request_id)

        completed = request.completed.wait(timeout=self.response_timeout_s)
        if not completed:
            logger.warning("Remote recording toggle timed out id=%s", request.request_id)
            status = self.get_status()
            return {
                "ok": False,
                "message": "No recording signal returned.",
                "request_id": request.request_id,
                **status,
            }

        assert request.response is not None
        return request.response

    def pop_pending_toggle(self) -> PendingToggleRequest | None:
        with self._lock:
            request = self._pending_toggles.popleft() if self._pending_toggles else None
            phase = self._phase

        if request is not None:
            logger.info(
                "Remote recording toggle received while phase=%s (id=%s)",
                phase,
                request.request_id,
            )
        return request

    def complete_toggle(
        self,
        request: PendingToggleRequest,
        *,
        ok: bool,
        recording: bool,
        phase: str | None = None,
        message: str,
    ) -> None:
        if phase is not None:
            self.set_phase(phase, recording=recording)
        else:
            self.set_phase(self.get_status()["phase"], recording=recording)

        response = {
            "ok": ok,
            "message": message,
            "request_id": request.request_id,
            **self.get_status(),
        }
        request.response = response
        request.completed.set()
        logger.info(
            "Completed remote recording toggle id=%s -> recording=%s phase=%s",
            request.request_id,
            response["recording"],
            response["phase"],
        )
