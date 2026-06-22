# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


def test_sourccey_client_get_data_raises_on_decode_failure_when_waiting_for_fresh_observation(monkeypatch):
    from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
    from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient

    client = SourcceyClient(SourcceyClientConfig(remote_ip="127.0.0.1"))
    client._poll_and_get_latest_message = MagicMock(return_value=b"bad-packet")

    class DummyRobotState:
        def ParseFromString(self, _payload: bytes) -> None:
            raise ValueError("decode failed")

    monkeypatch.setattr(
        "lerobot.robots.sourccey.sourccey.sourccey.sourccey_client.sourccey_pb2.SourcceyRobotState",
        DummyRobotState,
    )

    with pytest.raises(TimeoutError, match="could not be decoded"):
        client._get_data()


def test_sourccey_client_rejects_stale_packet_sequence():
    from lerobot.robots.sourccey.sourccey.protobuf.generated import sourccey_pb2
    from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
    from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient

    client = SourcceyClient(SourcceyClientConfig(remote_ip="127.0.0.1"))
    client._last_observation_packet_seq = 10

    robot_state = sourccey_pb2.SourcceyRobotState()
    robot_state.packet_seq = 10
    robot_state.packet_time_ns = 123
    for index, camera_key in enumerate(client.required_fresh_camera_keys, start=1):
        camera = robot_state.cameras.add()
        camera.name = camera_key
        camera.capture_time_ns = index

    with pytest.raises(TimeoutError, match="stale or out-of-order"):
        client._validate_observation_packet_freshness(robot_state)


def test_sourccey_client_rejects_packet_when_required_camera_does_not_advance():
    from lerobot.robots.sourccey.sourccey.protobuf.generated import sourccey_pb2
    from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
    from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient

    client = SourcceyClient(SourcceyClientConfig(remote_ip="127.0.0.1"))
    client._last_observation_packet_seq = 10
    client._last_camera_capture_time_ns = {
        camera_key: 1000 + index for index, camera_key in enumerate(client.required_fresh_camera_keys)
    }

    robot_state = sourccey_pb2.SourcceyRobotState()
    robot_state.packet_seq = 11
    robot_state.packet_time_ns = 456
    for camera_key, timestamp_ns in client._last_camera_capture_time_ns.items():
        camera = robot_state.cameras.add()
        camera.name = camera_key
        camera.capture_time_ns = timestamp_ns

    with pytest.raises(TimeoutError, match="reused stale camera frames"):
        client._validate_observation_packet_freshness(robot_state)


def test_sourccey_protobuf_preserves_packet_freshness_metadata():
    from lerobot.robots.sourccey.sourccey.protobuf.sourccey_protobuf import SourcceyProtobuf

    converter = SourcceyProtobuf()
    observation = {
        "front": np.zeros((4, 4, 3), dtype=np.uint8),
        "left_shoulder_pan.pos": 1.0,
        "right_shoulder_pan.pos": 2.0,
        "z.pos": 3.0,
        "x.vel": 4.0,
        "y.vel": 5.0,
        "theta.vel": 6.0,
    }
    packet = converter.observation_to_protobuf(
        observation,
        packet_seq=42,
        packet_time_ns=123456789,
        camera_timestamps_ns={"front": 987654321},
    )

    metadata = converter.protobuf_to_observation_metadata(packet)

    assert packet.packet_seq == 42
    assert packet.packet_time_ns == 123456789
    assert metadata["packet_seq"] == 42
    assert metadata["packet_time_ns"] == 123456789
    assert metadata["camera_capture_time_ns"]["front"] == 987654321
