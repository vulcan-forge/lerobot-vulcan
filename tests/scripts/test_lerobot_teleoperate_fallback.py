from __future__ import annotations

import pytest

from lerobot.scripts.lerobot_teleoperate import _connect_teleop_with_optional_default_fallback


class _DummyTeleopWithFallback:
    def __init__(self) -> None:
        self.connected = False
        self.disconnect_calls = 0

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self) -> None:
        raise RuntimeError("serial port not found")

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    def get_action(self) -> dict[str, float]:
        return {"joint.pos": 0.0}


class _DummyTeleopNoFallback(_DummyTeleopWithFallback):
    def get_action(self) -> dict[str, float]:
        raise RuntimeError("not connected")


def test_connect_teleop_falls_back_when_default_action_is_available() -> None:
    teleop = _DummyTeleopWithFallback()
    connected = _connect_teleop_with_optional_default_fallback(
        teleop, allow_default_fallback=True
    )

    assert connected is False
    assert teleop.disconnect_calls == 1


def test_connect_teleop_raises_when_no_fallback_action_available() -> None:
    teleop = _DummyTeleopNoFallback()
    with pytest.raises(RuntimeError, match="no disconnected default action fallback"):
        _connect_teleop_with_optional_default_fallback(teleop, allow_default_fallback=True)


def test_connect_teleop_raises_original_error_when_fallback_disabled() -> None:
    teleop = _DummyTeleopWithFallback()
    with pytest.raises(RuntimeError, match="serial port not found"):
        _connect_teleop_with_optional_default_fallback(teleop, allow_default_fallback=False)
