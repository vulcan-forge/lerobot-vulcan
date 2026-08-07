from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass

import numpy as np
import pytest
import zmq

from lerobot.robots.sourccey.sourccey.protobuf.generated import sourccey_pb2
from lerobot.robots.sourccey.sourccey.protobuf.sourccey_protobuf import (
    SourcceyProtobuf,
)
from lerobot.robots.sourccey.sourccey.sourccey.modules.websocket_relay.bridge import (
    WebsocketRelayBridge,
)


@dataclass
class _RelayConfig:
    websocket_relay_ws_base_url: str = "wss://relay.example.com"
    websocket_relay_session_id: str = "session-1234"
    websocket_relay_robot_token: str = "robot-token-1234"
    robot_id: str = "sourccey"
    zmq_cmd_endpoint: str = "tcp://127.0.0.1:5555"
    zmq_obs_endpoint: str = "tcp://127.0.0.1:5556"
    heartbeat_seconds: int = 5
    connect_retry_backoff_s: float = 0.01
    connect_retry_max_backoff_s: float = 0.01
    websocket_ping_interval_s: float = 0.0
    websocket_ping_timeout_s: float = 0.0
    log_actions: bool = True
    log_actions_interval_s: float = 30.0

    @property
    def ws_url(self) -> str:
        return (
            f"{self.websocket_relay_ws_base_url.rstrip('/')}/ws/robot"
            f"?session_id={self.websocket_relay_session_id}"
            f"&token={self.websocket_relay_robot_token}"
        )


class _FakeWebSocket:
    def __init__(self, incoming: list[str | bytes] | None = None) -> None:
        self.sent: list[str | bytes] = []
        self._incoming = list(incoming or [])
        self.sent_event = asyncio.Event()

    async def send(self, payload: str | bytes) -> None:
        self.sent.append(payload)
        self.sent_event.set()

    async def close(self) -> None:
        return None

    def __aiter__(self) -> "_FakeWebSocket":
        return self

    async def __anext__(self) -> str | bytes:
        if not self._incoming:
            raise StopAsyncIteration
        return self._incoming.pop(0)


class _FakeObservationSocket:
    def __init__(
        self,
        blocking_payloads: list[bytes],
        *,
        nonblocking_payloads: list[bytes] | None = None,
    ) -> None:
        self._blocking_payloads = list(blocking_payloads)
        self._nonblocking_payloads = list(nonblocking_payloads or [])
        self._pending = asyncio.Future()

    async def recv(self, flags: int = 0) -> bytes:
        if flags == zmq.NOBLOCK:
            if self._nonblocking_payloads:
                return self._nonblocking_payloads.pop(0)
            raise zmq.Again()

        if self._blocking_payloads:
            return self._blocking_payloads.pop(0)

        if self._pending.done():
            self._pending = asyncio.Future()
        return await self._pending

    def close(self, _linger: int = 0) -> None:
        if not self._pending.done():
            self._pending.cancel()


class _FakeCommandSocket:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    async def send(self, payload: bytes) -> None:
        self.sent.append(payload)

    def close(self, _linger: int = 0) -> None:
        return None


async def _run_observation_loop_once(bridge: WebsocketRelayBridge) -> None:
    task = asyncio.create_task(bridge._forward_observations_loop())
    await asyncio.wait_for(bridge._ws.sent_event.wait(), timeout=1.0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def test_forward_observations_sends_raw_binary_payload() -> None:
    async def _test() -> None:
        bridge = WebsocketRelayBridge(_RelayConfig())
        bridge._ws = _FakeWebSocket()
        bridge._obs_socket = _FakeObservationSocket([b"\x01robot-observation\xff"])

        await _run_observation_loop_once(bridge)

        assert bridge._ws.sent == [b"\x01robot-observation\xff"]

    asyncio.run(_test())


def test_forward_observations_drains_to_latest_payload() -> None:
    async def _test() -> None:
        bridge = WebsocketRelayBridge(_RelayConfig())
        bridge._ws = _FakeWebSocket()
        bridge._obs_socket = _FakeObservationSocket(
            [b"stale"],
            nonblocking_payloads=[b"newer", b"newest"],
        )

        await _run_observation_loop_once(bridge)

        assert bridge._ws.sent == [b"newest"]

    asyncio.run(_test())


def test_binary_observations_and_text_commands_coexist() -> None:
    async def _test() -> None:
        command_message = json.dumps(
            {
                "type": "robot.command.v1",
                "command": {
                    "action": {
                        "x.vel": 0.1,
                        "y.vel": -0.2,
                        "theta.vel": 0.3,
                        "z.pos": 0.4,
                    }
                },
            }
        )
        raw_observation = b"\x10\x20\x30\x40"

        bridge = WebsocketRelayBridge(_RelayConfig())
        bridge._ws = _FakeWebSocket(incoming=[command_message])
        bridge._obs_socket = _FakeObservationSocket([raw_observation])
        bridge._cmd_socket = _FakeCommandSocket()

        obs_task = asyncio.create_task(bridge._forward_observations_loop())
        cmd_task = asyncio.create_task(bridge._receive_commands_loop())

        await asyncio.wait_for(cmd_task, timeout=1.0)
        for _ in range(20):
            if len(bridge._ws.sent) >= 2:
                break
            await asyncio.sleep(0.01)

        obs_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await obs_task

        assert raw_observation in bridge._ws.sent
        ack_messages = [
            json.loads(payload)
            for payload in bridge._ws.sent
            if isinstance(payload, str)
        ]
        assert any(message["type"] == "robot.action_ack.v1" for message in ack_messages)

        assert len(bridge._cmd_socket.sent) == 1
        action_msg = sourccey_pb2.SourcceyRobotAction()
        action_msg.ParseFromString(bridge._cmd_socket.sent[0])
        action = SourcceyProtobuf().protobuf_to_action(action_msg)
        assert action["x.vel"] == pytest.approx(0.1)
        assert action["y.vel"] == pytest.approx(-0.2)
        assert action["theta.vel"] == pytest.approx(0.3)
        assert action["z.pos"] == pytest.approx(0.4)

    asyncio.run(_test())


def test_observation_protobuf_round_trip_preserves_state_and_cameras() -> None:
    converter = SourcceyProtobuf()
    observation = {
        "left_shoulder_pan.pos": 1.0,
        "left_shoulder_lift.pos": 2.0,
        "left_elbow_flex.pos": 3.0,
        "left_wrist_flex.pos": 4.0,
        "left_wrist_roll.pos": 5.0,
        "left_gripper.pos": 6.0,
        "right_shoulder_pan.pos": 7.0,
        "right_shoulder_lift.pos": 8.0,
        "right_elbow_flex.pos": 9.0,
        "right_wrist_flex.pos": 10.0,
        "right_wrist_roll.pos": 11.0,
        "right_gripper.pos": 12.0,
        "x.vel": 0.1,
        "y.vel": -0.2,
        "theta.vel": 0.3,
        "z.pos": 42.0,
        "front_left": np.full((8, 8, 3), 90, dtype=np.uint8),
        "front_right": np.full((8, 8, 3), 180, dtype=np.uint8),
    }

    proto = converter.observation_to_protobuf(observation)

    assert len(proto.cameras) == 2
    assert all(camera.image_data for camera in proto.cameras)

    serialized = proto.SerializeToString()
    parsed = sourccey_pb2.SourcceyRobotState()
    parsed.ParseFromString(serialized)

    decoded = converter.protobuf_to_observation(parsed)

    assert decoded["left_shoulder_pan.pos"] == pytest.approx(1.0)
    assert decoded["right_gripper.pos"] == pytest.approx(12.0)
    assert decoded["z.pos"] == pytest.approx(42.0)
    assert decoded["front_left"].shape == (8, 8, 3)
    assert decoded["front_right"].shape == (8, 8, 3)
