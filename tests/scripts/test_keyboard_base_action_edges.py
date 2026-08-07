from lerobot.scripts.lerobot_record import _get_keyboard_base_action as get_record_keyboard_base_action
from lerobot.scripts.lerobot_teleoperate import _get_keyboard_base_action as get_teleop_keyboard_base_action


class _DummyKeyboard:
    def __init__(self, pressed: dict[str, None], key_down_edges: list[str]) -> None:
        self.is_connected = True
        self._pressed = pressed
        self._key_down_edges = key_down_edges

    def pop_key_down_edges(self) -> list[str]:
        return list(self._key_down_edges)

    def get_action(self) -> dict[str, None]:
        return dict(self._pressed)


class _DummyRobot:
    def __init__(self) -> None:
        self.key_down_calls: list[str] = []
        self.base_action_calls: list[tuple[dict[str, None], float | None]] = []

    def on_key_down(self, key_char: str) -> None:
        self.key_down_calls.append(key_char)

    def _from_keyboard_to_base_action(
        self, keyboard_action: dict[str, None], z_pos: float | None = None
    ) -> dict[str, object]:
        self.base_action_calls.append((keyboard_action, z_pos))
        return {"keyboard_action": keyboard_action, "z_pos": z_pos}


def test_teleoperate_keyboard_base_action_forwards_key_down_edges() -> None:
    robot = _DummyRobot()
    keyboard = _DummyKeyboard(pressed={"w": None, "r": None}, key_down_edges=["r"])

    action = get_teleop_keyboard_base_action(robot, {"z.pos": 42.0}, keyboard)

    assert robot.key_down_calls == ["r"]
    assert robot.base_action_calls == [({"w": None, "r": None}, 42.0)]
    assert action == {"keyboard_action": {"w": None, "r": None}, "z_pos": 42.0}


def test_record_keyboard_base_action_forwards_key_down_edges() -> None:
    robot = _DummyRobot()
    keyboard = _DummyKeyboard(pressed={"f": None}, key_down_edges=["f"])

    action = get_record_keyboard_base_action(robot, {"z.pos": -5.0}, keyboard)

    assert robot.key_down_calls == ["f"]
    assert robot.base_action_calls == [({"f": None}, -5.0)]
    assert action == {"keyboard_action": {"f": None}, "z_pos": -5.0}
