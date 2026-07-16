from types import SimpleNamespace
from unittest.mock import MagicMock

from lerobot.robots.sourccey.sourccey.sourccey.modules.torque import untorque


def _make_robot():
    left_bus = MagicMock()
    right_bus = MagicMock()
    robot = SimpleNamespace(
        connect=MagicMock(),
        disconnect=MagicMock(),
        left_arm=SimpleNamespace(bus=left_bus),
        right_arm=SimpleNamespace(bus=right_bus),
    )
    return robot, left_bus, right_bus


def test_untorque_left_arm_uses_bus_connections_only(monkeypatch):
    robot, left_bus, right_bus = _make_robot()
    config = SimpleNamespace(left_arm_port=None, right_arm_port=None)

    monkeypatch.setattr(untorque, "SourcceyConfig", MagicMock(return_value=config))
    monkeypatch.setattr(untorque, "Sourccey", MagicMock(return_value=robot))
    monkeypatch.setattr(
        "sys.argv",
        ["untorque.py", "--id=test-bot", "--left"],
    )

    untorque.main()

    robot.connect.assert_not_called()
    robot.disconnect.assert_not_called()
    left_bus.connect.assert_called_once_with()
    left_bus.disable_torque.assert_called_once_with()
    left_bus.disconnect.assert_called_once_with(disable_torque=True)
    right_bus.connect.assert_not_called()


def test_enable_preserves_torque_on_disconnect(monkeypatch):
    robot, left_bus, right_bus = _make_robot()
    config = SimpleNamespace(left_arm_port=None, right_arm_port=None)

    monkeypatch.setattr(untorque, "SourcceyConfig", MagicMock(return_value=config))
    monkeypatch.setattr(untorque, "Sourccey", MagicMock(return_value=robot))
    monkeypatch.setattr(
        "sys.argv",
        ["untorque.py", "--enable"],
    )

    untorque.main()

    left_bus.enable_torque.assert_called_once_with()
    right_bus.enable_torque.assert_called_once_with()
    left_bus.disconnect.assert_called_once_with(disable_torque=False)
    right_bus.disconnect.assert_called_once_with(disable_torque=False)
