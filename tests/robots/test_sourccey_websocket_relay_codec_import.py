from __future__ import annotations

from lerobot.robots.sourccey.sourccey.sourccey.modules.websocket_relay.codec import (
    RelayCodec,
    _action_from_command,
)


def test_websocket_relay_codec_imports() -> None:
    codec = RelayCodec()
    assert codec is not None


def test_operator_axes_without_z_preserve_hold_semantics() -> None:
    action = _action_from_command({"input": {"axes": {"x": 0.5}}})

    assert action is not None
    assert "z.pos" not in action


def test_short_action_chunk_preserves_z_hold_semantics() -> None:
    action = _action_from_command({"action_chunk": [[0.1, 0.2, 0.3]]})

    assert action is not None
    assert "z.pos" not in action
