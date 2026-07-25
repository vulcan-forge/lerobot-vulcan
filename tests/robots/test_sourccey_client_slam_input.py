#!/usr/bin/env python

import base64
import json
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

import lerobot.robots.sourccey.sourccey.sourccey.sourccey_client as sourccey_client_module
from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClient, SourcceyClientConfig
from lerobot.robots.sourccey.sourccey.sourccey.modules.slam.config import SlamInputConfig


def _make_client() -> SourcceyClient:
    config = SourcceyClientConfig(
        id="test-client",
        remote_ip="127.0.0.1",
        slam=SlamInputConfig(
            input_enabled=True,
            input_endpoint="tcp://127.0.0.1:5560",
            stereo_left_key="front_left",
            stereo_right_key="front_right",
            jpeg_quality=80,
        ),
    )
    return SourcceyClient(config)


def test_z_command_initializes_from_observation_instead_of_endpoint() -> None:
    client = _make_client()

    action = client._from_keyboard_to_base_action(np.array([], dtype=str), z_obs_pos=37.5)

    assert client._z_pos_cmd_initialized is True
    assert client._z_pos_cmd == 37.5
    assert action["z.pos"] == 37.5


def test_z_teleop_uses_precise_bounded_target_steps(monkeypatch) -> None:
    client = _make_client()
    client._last_cmd_t = 10.0
    monkeypatch.setattr(sourccey_client_module.time, "monotonic", lambda: 10.25)

    action = client._from_keyboard_to_base_action(np.array(["q"]), z_obs_pos=0.0)

    # A delayed teleop frame is capped rather than producing a large target jump.
    assert action["z.pos"] == pytest.approx(2.0)


def test_z_teleop_uses_fast_rate_at_every_base_speed() -> None:
    client = _make_client()

    assert [level["z"] for level in client.speed_levels] == [1.5, 1.5, 1.5]


def test_z_teleop_compensates_stale_observation_once_on_release(monkeypatch) -> None:
    client = _make_client()
    times = iter((10.05, 10.10, 10.15))
    client._last_cmd_t = 10.0
    monkeypatch.setattr(sourccey_client_module.time, "monotonic", lambda: next(times))

    moving = client._from_keyboard_to_base_action(np.array(["q"]), z_obs_pos=0.0)
    released = client._from_keyboard_to_base_action(np.array([]), z_obs_pos=1.25)
    idle = client._from_keyboard_to_base_action(np.array([]), z_obs_pos=0.5)

    assert moving["z.pos"] == pytest.approx(2.0)
    assert released["z.pos"] == pytest.approx(2.0)
    assert idle["z.pos"] == pytest.approx(2.0)


def test_z_release_compensation_is_bounded(monkeypatch) -> None:
    client = _make_client()
    client._z_pos_cmd_initialized = True
    client._z_pos_cmd = 20.0
    client._z_last_direction = 1.0
    client._last_cmd_t = 10.0
    monkeypatch.setattr(sourccey_client_module.time, "monotonic", lambda: 10.05)

    action = client._from_keyboard_to_base_action(np.array([]), z_obs_pos=5.0)

    assert action["z.pos"] == pytest.approx(6.0)


def test_legacy_flat_slam_config_fields_still_work() -> None:
    config = SourcceyClientConfig(
        id="test-client",
        remote_ip="127.0.0.1",
        slam_input_enabled=True,
        slam_input_endpoint="tcp://127.0.0.1:5561",
        slam_stereo_left_key="left_cam",
        slam_stereo_right_key="right_cam",
        slam_jpeg_quality=72,
    )
    assert config.slam.input_enabled is True
    assert config.slam.input_endpoint == "tcp://127.0.0.1:5561"
    assert config.slam.stereo_left_key == "left_cam"
    assert config.slam.stereo_right_key == "right_cam"
    assert config.slam.jpeg_quality == 72


def _make_frames() -> dict[str, np.ndarray]:
    left = np.full((24, 24, 3), 80, dtype=np.uint8)
    right = np.full((24, 24, 3), 160, dtype=np.uint8)
    return {"front_left": left, "front_right": right}


def test_build_slam_input_packet_contains_required_fields() -> None:
    client = _make_client()
    frames = _make_frames()
    observation = {"x.vel": 0.2, "y.vel": -0.1, "theta.vel": 0.3}

    payload = client._build_slam_input_packet(observation=observation, frames=frames)
    assert payload is not None
    data = json.loads(payload.decode("utf-8"))

    assert data["schema"] == "slam_input.v1"
    assert data["stereo_left"] == "front_left"
    assert data["stereo_right"] == "front_right"
    assert data["imu_samples"] == []
    assert data["base_velocity"] == {"x.vel": 0.2, "y.vel": -0.1, "theta.vel": 0.3}
    assert "host_monotonic_ns" in data
    assert "source" in data

    for cam_name in ("front_left", "front_right"):
        camera_payload = data["cameras"][cam_name]
        assert camera_payload["frame_id"] == 1
        encoded = base64.b64decode(camera_payload["jpeg_b64"])
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape == (24, 24, 3)


def test_build_slam_input_packet_increments_frame_ids_per_camera() -> None:
    client = _make_client()
    frames = _make_frames()
    observation = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

    payload_1 = client._build_slam_input_packet(observation=observation, frames=frames)
    payload_2 = client._build_slam_input_packet(observation=observation, frames=frames)
    assert payload_1 is not None and payload_2 is not None

    packet_1 = json.loads(payload_1.decode("utf-8"))
    packet_2 = json.loads(payload_2.decode("utf-8"))
    assert packet_1["cameras"]["front_left"]["frame_id"] == 1
    assert packet_2["cameras"]["front_left"]["frame_id"] == 2
    assert packet_1["cameras"]["front_right"]["frame_id"] == 1
    assert packet_2["cameras"]["front_right"]["frame_id"] == 2


def test_build_slam_input_packet_returns_none_when_required_stereo_missing() -> None:
    client = _make_client()
    frames = {"front_left": np.zeros((24, 24, 3), dtype=np.uint8)}
    observation = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

    payload = client._build_slam_input_packet(observation=observation, frames=frames)
    assert payload is None


def test_publish_slam_input_sends_packet_when_build_succeeds() -> None:
    client = _make_client()
    client.zmq_slam_input_socket = MagicMock()
    frames = _make_frames()
    observation = {"x.vel": 0.1, "y.vel": 0.2, "theta.vel": 0.3}

    client._publish_slam_input(observation=observation, frames=frames)

    assert client.zmq_slam_input_socket.send.call_count == 1
    sent_payload = client.zmq_slam_input_socket.send.call_args.args[0]
    sent_data = json.loads(sent_payload.decode("utf-8"))
    assert sent_data["schema"] == "slam_input.v1"


def test_publish_slam_input_does_not_send_when_required_stereo_missing() -> None:
    client = _make_client()
    client.zmq_slam_input_socket = MagicMock()
    frames = {"front_left": np.zeros((24, 24, 3), dtype=np.uint8)}
    observation = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

    client._publish_slam_input(observation=observation, frames=frames)

    client.zmq_slam_input_socket.send.assert_not_called()


def test_get_observation_only_publishes_slam_for_fresh_packets() -> None:
    client = _make_client()
    client._is_connected = True
    client._publish_slam_input = MagicMock()
    frames = _make_frames()
    state = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

    client._get_data = MagicMock(return_value=(frames, state, False))
    _ = client.get_observation()
    client._publish_slam_input.assert_not_called()

    client._get_data = MagicMock(return_value=(frames, state, True))
    _ = client.get_observation()
    assert client._publish_slam_input.call_count == 1
