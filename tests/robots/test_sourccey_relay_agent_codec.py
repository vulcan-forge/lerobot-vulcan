from __future__ import annotations

from lerobot.robots.sourccey.sourccey.sourccey.relay_agent.codec import RelayCodec


def test_encode_action_from_action_dict() -> None:
    codec = RelayCodec()
    payload, error = codec.encode_action(
        {
            "action": {
                "x.vel": 0.1,
                "y.vel": -0.2,
                "theta.vel": 0.3,
                "z.pos": 0.0,
            }
        }
    )
    assert error is None
    assert payload is not None
    assert len(payload) > 0


def test_encode_action_from_operator_axes() -> None:
    codec = RelayCodec()
    payload, error = codec.encode_action(
        {
            "input": {
                "axes": {
                    "x": 1.0,
                    "y": -1.0,
                    "theta": 0.5,
                    "z": 0.25,
                }
            }
        }
    )
    assert error is None
    assert payload is not None
    assert len(payload) > 0


def test_encode_action_from_action_chunk() -> None:
    codec = RelayCodec()
    payload, error = codec.encode_action(
        {
            "action_chunk": [[0.4, -0.4, 0.2, 0.1]]
        }
    )
    assert error is None
    assert payload is not None
    assert len(payload) > 0


def test_encode_action_invalid_payload() -> None:
    codec = RelayCodec()
    payload, error = codec.encode_action({"unexpected": "shape"})
    assert payload is None
    assert error == "invalid_command_payload"
