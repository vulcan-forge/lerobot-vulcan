from __future__ import annotations

import logging

from lerobot.scripts.lerobot_teleoperate import connect_teleop


class _DummyTeleop:
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


class _ConnectedDummyTeleop(_DummyTeleop):
    def connect(self) -> None:
        self.connected = True


def test_connect_teleop_returns_true_when_connection_succeeds() -> None:
    teleop = _ConnectedDummyTeleop()

    assert connect_teleop(teleop) is True
    assert teleop.is_connected is True


def test_connect_teleop_falls_back_after_connection_error(caplog) -> None:
    teleop = _DummyTeleop()

    with caplog.at_level(logging.WARNING):
        assert connect_teleop(teleop) is False

    assert "serial port not found" in caplog.text
    assert "Continuing with disconnected default actions" in caplog.text
