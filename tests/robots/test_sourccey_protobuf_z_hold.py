import pytest

from lerobot.robots.sourccey.sourccey.protobuf.generated import sourccey_pb2
from lerobot.robots.sourccey.sourccey.protobuf.sourccey_protobuf import SourcceyProtobuf


def test_missing_z_action_is_omitted_from_protobuf() -> None:
    encoded = SourcceyProtobuf().action_to_protobuf({"x.vel": 0.25})

    assert encoded.HasField("base_target_position") is False


def test_missing_z_protobuf_field_decodes_as_hold() -> None:
    message = sourccey_pb2.SourcceyRobotAction()
    message.base_target_velocity.x_vel = 0.25

    action = SourcceyProtobuf().protobuf_to_action(message)

    assert action["x.vel"] == pytest.approx(0.25)
    assert "z.pos" not in action


@pytest.mark.parametrize("position", [-100.0, 0.0, 42.5, 100.0])
def test_explicit_z_position_preserves_value_and_presence(position: float) -> None:
    converter = SourcceyProtobuf()

    encoded = converter.action_to_protobuf({"z.pos": position})
    decoded = converter.protobuf_to_action(encoded)

    assert encoded.HasField("base_target_position") is True
    assert decoded["z.pos"] == pytest.approx(position)
