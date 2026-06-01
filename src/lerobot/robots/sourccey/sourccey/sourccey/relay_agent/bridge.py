from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime
from typing import Any

import websockets
import zmq
import zmq.asyncio

from .codec import RelayCodec
from .config import RelayAgentConfig


class RelayBridge:
    def __init__(self, config: RelayAgentConfig) -> None:
        self._config = config
        self._codec = RelayCodec()
        self._context = zmq.asyncio.Context.instance()
        self._cmd_socket = self._context.socket(zmq.PUSH)
        self._obs_socket = self._context.socket(zmq.PULL)
        self._ws: Any = None
        self._tasks: list[asyncio.Task[None]] = []
        self._sequence = 0

    async def run_forever(self) -> None:
        self._cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self._obs_socket.setsockopt(zmq.CONFLATE, 1)
        self._cmd_socket.connect(self._config.zmq_cmd_endpoint)
        self._obs_socket.connect(self._config.zmq_obs_endpoint)

        self._ws = await websockets.connect(self._config.ws_url, max_size=2**22)
        print(
            f"[{datetime.now(UTC).isoformat()}] relay_agent.connected "
            f"session_id={self._config.relay_session_id} "
            f"robot_id={self._config.robot_id}"
        )
        self._tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._forward_observations_loop()),
            asyncio.create_task(self._receive_commands_loop()),
        ]
        await asyncio.gather(*self._tasks)

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()

        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None

        with contextlib.suppress(Exception):
            self._cmd_socket.close(0)
        with contextlib.suppress(Exception):
            self._obs_socket.close(0)

    async def _heartbeat_loop(self) -> None:
        assert self._ws is not None
        while True:
            await asyncio.sleep(self._config.heartbeat_seconds)
            await self._ws.send('{"type":"heartbeat"}')

    async def _forward_observations_loop(self) -> None:
        assert self._ws is not None
        while True:
            raw_payload = await self._obs_socket.recv()
            self._sequence += 1
            try:
                payload = self._codec.decode_observation(raw_payload)
            except Exception as exc:
                await self._send_error("observation_decode_failed", str(exc))
                continue

            message = {
                "type": "robot.observation.v1",
                "session_id": self._config.relay_session_id,
                "robot_id": self._config.robot_id,
                "observation_seq": self._sequence,
                "sent_at_utc": datetime.now(UTC).isoformat(),
                "payload": payload,
            }
            await self._ws.send(json.dumps(message, separators=(",", ":")))

    async def _receive_commands_loop(self) -> None:
        assert self._ws is not None
        async for raw_message in self._ws:
            if isinstance(raw_message, bytes):
                try:
                    raw_message = raw_message.decode("utf-8")
                except UnicodeDecodeError:
                    continue

            message = _parse_json(raw_message)
            if not message:
                continue

            message_type = str(message.get("type", "")).strip()
            if message_type in {"session.state", "webrtc.offer", "webrtc.answer", "webrtc.ice"}:
                continue
            if message_type != "robot.command.v1":
                continue

            command = message.get("command")
            if not isinstance(command, dict):
                await self._send_error("invalid_command_payload", "command field is required")
                continue

            encoded, error_code = self._codec.encode_action(command)
            if encoded is None:
                await self._send_error(error_code or "encode_failed", "failed to encode action")
                continue

            try:
                await self._cmd_socket.send(encoded)
            except Exception as exc:
                await self._send_error("zmq_send_failed", str(exc))
                continue

            ack = {
                "type": "robot.action_ack.v1",
                "session_id": self._config.relay_session_id,
                "robot_id": self._config.robot_id,
                "ack_at_utc": datetime.now(UTC).isoformat(),
            }
            await self._ws.send(json.dumps(ack, separators=(",", ":")))

    async def _send_error(self, code: str, detail: str) -> None:
        if self._ws is None:
            return
        error = {
            "type": "robot.error.v1",
            "session_id": self._config.relay_session_id,
            "robot_id": self._config.robot_id,
            "error": code,
            "message": detail,
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
        await self._ws.send(json.dumps(error, separators=(",", ":")))


def _parse_json(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed
