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
