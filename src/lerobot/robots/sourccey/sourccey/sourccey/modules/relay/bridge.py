import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import websocket
import zmq

from ...config_sourccey import SourcceyHostConfig
from ...protobuf.generated import sourccey_pb2
from ...protobuf.sourccey_protobuf import SourcceyProtobuf

logger = logging.getLogger(__name__)

DEFAULT_CREDENTIALS_PATH = (
    Path.home() / ".cache" / "huggingface" / "lerobot" / "pairing" / "cloud_device_credentials.json"
)


@dataclass
class RelayRobotCredentials:
    device_id: str
    owned_robot_id: str
    relay_http_base_url: str
    relay_ws_base_url: str
    device_auth_token: str


class RelayRobotBridge:
    def __init__(self, host_config: SourcceyHostConfig):
        self.host_config = host_config
        self.protobuf_converter = SourcceyProtobuf()
        self.stop_event = threading.Event()
        self.ws_app: websocket.WebSocketApp | None = None
        self.ws_thread: threading.Thread | None = None
        self.heartbeat_thread: threading.Thread | None = None
        self.observation_thread: threading.Thread | None = None
        self.active_session_id: str | None = None
        self.credentials_reported = False
        self.last_polled_status: str | None = None
        self._ws_lock = threading.Lock()

        self.zmq_context = zmq.Context.instance()
        self.cmd_socket = self.zmq_context.socket(zmq.PUSH)
        self.cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.cmd_socket.connect(f"tcp://127.0.0.1:{host_config.port_zmq_cmd}")

        self.observation_socket = self.zmq_context.socket(zmq.PULL)
        self.observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.observation_socket.connect(f"tcp://127.0.0.1:{host_config.port_zmq_observations}")
        self.observation_poller = zmq.Poller()
        self.observation_poller.register(self.observation_socket, zmq.POLLIN)

    @classmethod
    def from_environment(cls, host_config: SourcceyHostConfig) -> "RelayRobotBridge | None":
        credentials_path = os.getenv("VULCAN_DEVICE_CREDENTIALS_PATH") or os.getenv(
            "SOURCCEY_CLOUD_CREDENTIALS_PATH"
        )
        path = Path(credentials_path).expanduser() if credentials_path else DEFAULT_CREDENTIALS_PATH
        if not path.exists():
            _emit_status(f"Relay bridge disabled. Credentials file not found at {path}")
            return None

        try:
            cls._load_credentials(path)
        except Exception as exc:
            _emit_status(f"Relay bridge disabled. Failed to load credentials from {path}: {exc}")
            return None

        bridge = cls(host_config)
        bridge.credentials_path = path
        return bridge

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self._run, name="sourccey-relay-bridge", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        self.stop_event.set()
        self._close_websocket()
        self._send_safe_stop()
        try:
            self.observation_socket.close(linger=0)
        except Exception:
            pass
        try:
            self.cmd_socket.close(linger=0)
        except Exception:
            pass

    def _run(self) -> None:
        _emit_status(f"Relay bridge starting with credentials at {self.credentials_path}")
        while not self.stop_event.is_set():
            try:
                credentials = self._load_credentials(self.credentials_path)
                if not self.credentials_reported:
                    _emit_status(
                        f"Relay bridge credentials loaded for device {credentials.device_id} "
                        f"and robot {credentials.owned_robot_id}."
                    )
                    self.credentials_reported = True
                active_session = self._fetch_active_session(credentials)
                status = str(active_session.get("status") or "").strip().lower()
                if status != self.last_polled_status:
                    _emit_status(f"Relay active session status changed to {status or 'unknown'}.")
                    self.last_polled_status = status
                if status != "ready":
                    if self.active_session_id is not None:
                        _emit_status("Relay session is no longer ready; disconnecting robot websocket.")
                    self.active_session_id = None
                    self._close_websocket()
                    self._sleep_with_stop(3.0)
                    continue

                relay = active_session.get("relay") if isinstance(active_session.get("relay"), dict) else {}
                robot_ws_url = str(relay.get("robot_ws_url") or "").strip()
                session_id = str(active_session.get("session_id") or "").strip()
                if not robot_ws_url or not session_id:
                    _emit_status("Relay active session response is missing robot websocket details.")
                    self._sleep_with_stop(3.0)
                    continue

                if self.ws_thread is not None and self.ws_thread.is_alive() and self.active_session_id == session_id:
                    self._sleep_with_stop(2.0)
                    continue

                self.active_session_id = session_id
                self._connect_session(robot_ws_url)
            except Exception as exc:
                _emit_status(f"Relay bridge loop failed: {exc}")
                self._close_websocket()
                self._sleep_with_stop(5.0)

    def _connect_session(self, robot_ws_url: str) -> None:
        self._close_websocket()
        _emit_status(f"Connecting robot websocket to {robot_ws_url}")
        self.ws_app = websocket.WebSocketApp(
            robot_ws_url,
            on_open=self._on_ws_open,
            on_message=self._on_ws_message,
            on_error=self._on_ws_error,
            on_close=self._on_ws_close,
        )
        self.ws_thread = threading.Thread(
            target=self.ws_app.run_forever,
            kwargs={"ping_interval": None},
            name="sourccey-relay-ws",
            daemon=True,
        )
        self.ws_thread.start()

    def _on_ws_open(self, ws: websocket.WebSocketApp) -> None:
        _emit_status("Robot websocket connected.")
        self._send_json({"type": "heartbeat"})

        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="sourccey-relay-heartbeat",
            daemon=True,
        )
        self.heartbeat_thread.start()

        self.observation_thread = threading.Thread(
            target=self._observation_loop,
            name="sourccey-relay-observations",
            daemon=True,
        )
        self.observation_thread.start()

    def _on_ws_message(self, ws: websocket.WebSocketApp, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError as exc:
            _emit_status(f"Relay websocket delivered invalid JSON: {exc}")
            return

        message_type = str(message.get("type") or "").strip()
        if message_type == "session.state":
            return

        if message_type == "robot.command.v1":
            self._handle_robot_command(message)
            return

        if message_type in {"webrtc.offer", "webrtc.answer", "webrtc.ice"}:
            logger.debug("Ignoring WebRTC relay message in host mode: %s", message_type)
            return

        logger.debug("Ignoring unsupported relay message type: %s", message_type)

    def _on_ws_error(self, ws: websocket.WebSocketApp, error: Any) -> None:
        _emit_status(f"Robot websocket error: {error}")

    def _on_ws_close(
        self,
        ws: websocket.WebSocketApp,
        close_status_code: int | None,
        close_msg: str | None,
    ) -> None:
        _emit_status(f"Robot websocket closed: code={close_status_code} message={close_msg}")
        self._send_safe_stop()

    def _handle_robot_command(self, message: dict[str, Any]) -> None:
        command = message.get("command")
        if not isinstance(command, dict):
            return

        action = command.get("action")
        if not isinstance(action, dict):
            return

        try:
            robot_action = self.protobuf_converter.action_to_protobuf(action)
            self.cmd_socket.send(robot_action.SerializeToString(), flags=zmq.NOBLOCK)
            self._send_action_ack(command)
        except zmq.Again:
            logger.debug("Dropping relay command because local ZMQ command socket is busy.")
        except Exception as exc:
            _emit_status(f"Failed to forward relay command to host ZMQ: {exc}")
            self._send_json(
                {
                    "type": "robot.error.v1",
                    "error": "command_forward_failed",
                    "message": str(exc),
                    "sent_at_utc": self._utc_now_iso(),
                }
            )

    def _send_action_ack(self, command: dict[str, Any]) -> None:
        ack: dict[str, Any] = {
            "type": "robot.action_ack.v1",
            "status": "forwarded_to_host",
            "received_at_utc": self._utc_now_iso(),
        }
        if "seq" in command:
            ack["seq"] = command.get("seq")
        if "sent_at_utc" in command:
            ack["command_sent_at_utc"] = command.get("sent_at_utc")
        self._send_json(ack)

    def _heartbeat_loop(self) -> None:
        while not self.stop_event.is_set() and self._is_ws_open():
            self._send_json({"type": "heartbeat"})
            self._sleep_with_stop(5.0)

    def _observation_loop(self) -> None:
        while not self.stop_event.is_set() and self._is_ws_open():
            try:
                sockets = dict(self.observation_poller.poll(timeout=500))
            except zmq.ZMQError:
                return

            if self.observation_socket not in sockets:
                continue

            try:
                msg_bytes = self.observation_socket.recv(zmq.NOBLOCK)
                robot_state = sourccey_pb2.SourcceyRobotState()
                robot_state.ParseFromString(msg_bytes)
                observation = self.protobuf_converter.protobuf_to_observation(robot_state)
                payload = {
                    "type": "robot.observation.v1",
                    "sent_at_utc": self._utc_now_iso(),
                    "observation": self._to_json_safe_observation(observation),
                }
                self._send_json(payload)
            except zmq.Again:
                continue
            except Exception as exc:
                logger.debug("Skipping observation relay update: %s", exc)

    def _fetch_active_session(self, credentials: RelayRobotCredentials) -> dict[str, Any]:
        url = f"{credentials.relay_http_base_url.rstrip('/')}/api/v1/robot/session/active"
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {credentials.device_auth_token}"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Relay active-session response was not a JSON object.")
        return payload

    def _close_websocket(self) -> None:
        with self._ws_lock:
            ws_app = self.ws_app
            self.ws_app = None

        if ws_app is not None:
            try:
                ws_app.close()
            except Exception:
                pass

    def _is_ws_open(self) -> bool:
        ws_app = self.ws_app
        if ws_app is None or ws_app.sock is None:
            return False
        return bool(ws_app.sock and ws_app.sock.connected)

    def _send_json(self, payload: dict[str, Any]) -> None:
        ws_app = self.ws_app
        if ws_app is None or not self._is_ws_open():
            return
        try:
            ws_app.send(json.dumps(payload))
        except Exception as exc:
            logger.debug("Failed to send relay websocket payload: %s", exc)

    def _send_safe_stop(self) -> None:
        safe_stop_action = {
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0,
            "z.pos": 0.0,
            "untorque_left": False,
            "untorque_right": False,
        }
        try:
            robot_action = self.protobuf_converter.action_to_protobuf(safe_stop_action)
            self.cmd_socket.send(robot_action.SerializeToString(), flags=zmq.NOBLOCK)
        except Exception:
            pass

    @staticmethod
    def _load_credentials(path: Path) -> RelayRobotCredentials:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("Credentials file must contain a JSON object.")

        device_id = str(payload.get("device_id") or "").strip()
        owned_robot_id = str(payload.get("owned_robot_id") or "").strip()
        relay_http_base_url = str(payload.get("relay_http_base_url") or "").strip().rstrip("/")
        relay_ws_base_url = str(payload.get("relay_ws_base_url") or "").strip().rstrip("/")
        device_auth_token = str(payload.get("device_auth_token") or "").strip()

        if not device_id or not owned_robot_id or not relay_http_base_url or not device_auth_token:
            raise RuntimeError("Credentials file is missing required relay device fields.")

        if not relay_ws_base_url:
            if relay_http_base_url.startswith("https://"):
                relay_ws_base_url = f"wss://{relay_http_base_url[len('https://') :]}"
            elif relay_http_base_url.startswith("http://"):
                relay_ws_base_url = f"ws://{relay_http_base_url[len('http://') :]}"
            else:
                relay_ws_base_url = relay_http_base_url

        return RelayRobotCredentials(
            device_id=device_id,
            owned_robot_id=owned_robot_id,
            relay_http_base_url=relay_http_base_url,
            relay_ws_base_url=relay_ws_base_url,
            device_auth_token=device_auth_token,
        )

    @staticmethod
    def _to_json_safe_observation(observation: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        camera_names: list[str] = []
        for key, value in observation.items():
            if isinstance(value, (int, float, bool, str)) or value is None:
                safe[key] = value
                continue

            camera_names.append(key)

        if camera_names:
            safe["camera_streams"] = camera_names
        return safe

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _sleep_with_stop(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while not self.stop_event.is_set() and time.time() < deadline:
            time.sleep(0.1)


def _emit_status(message: str) -> None:
    print(message, flush=True)
    logger.info(message)
