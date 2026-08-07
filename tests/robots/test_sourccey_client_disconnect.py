from unittest.mock import MagicMock

from lerobot.robots.sourccey.sourccey.protobuf.generated import sourccey_pb2
from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClient, SourcceyClientConfig


def _arm_action(client: SourcceyClient) -> dict[str, float]:
    arm_keys = tuple(
        key for key in client._state_order if key.startswith(("left_", "right_")) and key.endswith(".pos")
    )
    return {key: float(index + 1) for index, key in enumerate(arm_keys)}


def test_disconnect_sends_best_effort_stop_command() -> None:
    client = SourcceyClient(SourcceyClientConfig(id="test-client", remote_ip="127.0.0.1"))
    client._last_sent_action = _arm_action(client)
    client._last_sent_action["untorque_left"] = True
    client._last_sent_action["untorque_right"] = False
    command_socket = MagicMock()
    client.zmq_cmd_socket = command_socket
    client.zmq_observation_socket = MagicMock()
    client.zmq_context = MagicMock()
    client._is_connected = True

    client.disconnect()

    payload = command_socket.send.call_args.args[0]
    command = sourccey_pb2.SourcceyRobotAction()
    command.ParseFromString(payload)
    assert command.base_target_velocity.x_vel == 0.0
    assert command.base_target_velocity.y_vel == 0.0
    assert command.base_target_velocity.theta_vel == 0.0
    assert command.HasField("base_target_position") is False
    assert command.left_arm_target_joints.shoulder_pan == 1.0
    assert command.left_arm_target_joints.gripper == 6.0
    assert command.right_arm_target_joints.shoulder_pan == 7.0
    assert command.right_arm_target_joints.gripper == 12.0
    assert command.untorque_left is True
    assert command.untorque_right is False


def test_disconnect_skips_stop_packet_without_complete_arm_state() -> None:
    client = SourcceyClient(SourcceyClientConfig(id="test-client", remote_ip="127.0.0.1"))
    command_socket = MagicMock()
    client.zmq_cmd_socket = command_socket
    client.zmq_observation_socket = MagicMock()
    client.zmq_context = MagicMock()
    client._is_connected = True

    client.disconnect()

    command_socket.send.assert_not_called()


def test_send_action_caches_successfully_sent_ai_arm_targets() -> None:
    client = SourcceyClient(SourcceyClientConfig(id="test-client", remote_ip="127.0.0.1"))
    client.zmq_cmd_socket = MagicMock()
    client._is_connected = True
    action = {**_arm_action(client), "z.pos": 42.0, "x.vel": 0.25}

    client.send_action(action)

    assert client._last_sent_action["left_shoulder_pan.pos"] == 1.0
    assert client._last_sent_action["right_gripper.pos"] == 12.0
    assert client._last_sent_action["z.pos"] == 42.0
    assert client._last_sent_action["x.vel"] == 0.25
    assert client._last_sent_action["y.vel"] == 0.0
    assert client._last_sent_action["theta.vel"] == 0.0
