from __future__ import annotations

from lerobot.robots.sourccey.sourccey.sourccey.modules.websocket_relay.codec import (
    RelayCodec,
)


def test_websocket_relay_codec_imports() -> None:
    codec = RelayCodec()
    assert codec is not None
