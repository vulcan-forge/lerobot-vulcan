import logging
from types import SimpleNamespace

from lerobot.teleoperators.sourccey.sourccey.sourccey_leader.sourccey_leader import SourcceyLeader


def _make_disconnected_leader(*, using_arm: bool = True) -> SourcceyLeader:
    leader = SourcceyLeader.__new__(SourcceyLeader)
    leader.bus = SimpleNamespace(is_connected=False)
    leader.config = SimpleNamespace(orientation="left", port="COM7")
    leader.leader_connected = True
    leader.using_arm = using_arm
    leader._default_action = {"shoulder_pan.pos": 0.0}
    leader._default_active_action = {"shoulder_pan.pos": 1.0}
    leader._last_disconnected_warning_time = 0.0
    leader._disconnected_warning_throttle_interval = 5.0
    return leader


def test_disconnected_active_leader_logs_what_is_disconnected_every_five_seconds(
    caplog, monkeypatch
) -> None:
    leader = _make_disconnected_leader()
    now = 10.0
    monkeypatch.setattr(
        "lerobot.teleoperators.sourccey.sourccey.sourccey_leader.sourccey_leader.time.monotonic",
        lambda: now,
    )

    with caplog.at_level(logging.WARNING):
        assert leader.get_action() == leader._default_active_action
        assert leader.get_action() == leader._default_active_action
        now = 15.0
        assert leader.get_action() == leader._default_active_action

    warnings = [record.message for record in caplog.records]
    assert len(warnings) == 2
    assert all("Left Sourccey leader" in message for message in warnings)
    assert all("motor bus on port 'COM7'" in message for message in warnings)


def test_disconnected_unused_leader_does_not_log(caplog) -> None:
    leader = _make_disconnected_leader(using_arm=False)

    with caplog.at_level(logging.WARNING):
        assert leader.get_action() == leader._default_action

    assert not caplog.records
