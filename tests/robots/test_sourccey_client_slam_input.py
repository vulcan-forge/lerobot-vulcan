#!/usr/bin/env python

import base64
import json
from unittest.mock import MagicMock

import cv2
import numpy as np

from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClient, SourcceyClientConfig
from lerobot.robots.sourccey.sourccey.sourccey.modules.slam import SlamInputConfig


def _make_client(*, eye_only_mode: bool = False) -> SourcceyClient:
    config = SourcceyClientConfig(
        id="test-client",
        remote_ip="127.0.0.1",
        slam=SlamInputConfig(
            input_enabled=True,
            input_endpoint="tcp://127.0.0.1:5560",
            stereo_left_key="front_left",
            stereo_right_key="front_right",
            jpeg_quality=80,
            eye_only_mode=eye_only_mode,
        ),
    )
    return SourcceyClient(config)


def test_legacy_flat_slam_config_fields_still_work() -> None:
    config = SourcceyClientConfig(
        id="test-client",
        remote_ip="127.0.0.1",
        slam_input_enabled=True,
        slam_input_endpoint="tcp://127.0.0.1:5561",
        slam_stereo_left_key="left_cam",
        slam_stereo_right_key="right_cam",
        slam_jpeg_quality=72,
        slam_eye_only_mode=True,
        slam_publish_fps=5.0,
        slam_resize_width=160,
        slam_resize_height=120,
    )
    assert config.slam.input_enabled is True
    assert config.slam.input_endpoint == "tcp://127.0.0.1:5561"
    assert config.slam.stereo_left_key == "left_cam"
    assert config.slam.stereo_right_key == "right_cam"
    assert config.slam.jpeg_quality == 72
    assert config.slam.eye_only_mode is True
    assert config.slam.publish_fps == 5.0
    assert config.slam.resize_width == 160
    assert config.slam.resize_height == 120


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


def test_build_slam_input_packet_resizes_frames_when_requested() -> None:
    config = SourcceyClientConfig(
        id="test-client",
        remote_ip="127.0.0.1",
        slam=SlamInputConfig(
            input_enabled=True,
            input_endpoint="tcp://127.0.0.1:5560",
            stereo_left_key="front_left",
            stereo_right_key="front_right",
            jpeg_quality=80,
            resize_width=12,
            resize_height=10,
        ),
    )
    client = SourcceyClient(config)
    frames = _make_frames()
    observation = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

    payload = client._build_slam_input_packet(observation=observation, frames=frames)
    assert payload is not None
    data = json.loads(payload.decode("utf-8"))

    encoded = base64.b64decode(data["cameras"]["front_left"]["jpeg_b64"])
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (10, 12, 3)


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


def test_build_slam_input_packet_eye_only_mode_omits_wrist_cameras() -> None:
    client = _make_client(eye_only_mode=True)
    frames = {
        **_make_frames(),
        "wrist_left": np.full((24, 24, 3), 220, dtype=np.uint8),
    }
    observation = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

    payload = client._build_slam_input_packet(observation=observation, frames=frames)
    assert payload is not None

    packet = json.loads(payload.decode("utf-8"))
    assert set(packet["cameras"]) == {"front_left", "front_right"}


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


def test_publish_slam_input_respects_publish_fps_throttle() -> None:
    config = SourcceyClientConfig(
        id="test-client",
        remote_ip="127.0.0.1",
        slam=SlamInputConfig(
            input_enabled=True,
            input_endpoint="tcp://127.0.0.1:5560",
            stereo_left_key="front_left",
            stereo_right_key="front_right",
            jpeg_quality=80,
            publish_fps=1.0,
        ),
    )
    client = SourcceyClient(config)
    client.zmq_slam_input_socket = MagicMock()
    frames = _make_frames()
    observation = {"x.vel": 0.1, "y.vel": 0.2, "theta.vel": 0.3}

    client._publish_slam_input(observation=observation, frames=frames)
    client._publish_slam_input(observation=observation, frames=frames)

    assert client.zmq_slam_input_socket.send.call_count == 1


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
    client._enqueue_slam_input = MagicMock()
    frames = _make_frames()
    state = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

    client._get_data = MagicMock(return_value=(frames, state, False))
    _ = client.get_observation()
    client._enqueue_slam_input.assert_not_called()

    client._get_data = MagicMock(return_value=(frames, state, True))
    _ = client.get_observation()
    assert client._enqueue_slam_input.call_count == 1


def test_enqueue_slam_input_keeps_only_latest_pending_job() -> None:
    client = _make_client()
    observation_1 = {"x.vel": 0.1}
    observation_2 = {"x.vel": 0.2}
    frames_1 = _make_frames()
    frames_2 = {
        "front_left": np.full((24, 24, 3), 10, dtype=np.uint8),
        "front_right": np.full((24, 24, 3), 20, dtype=np.uint8),
    }

    client._enqueue_slam_input(observation=observation_1, frames=frames_1)
    client._enqueue_slam_input(observation=observation_2, frames=frames_2)

    pending = client._slam_publish_pending
    assert pending is not None
    observation, frames = pending
    assert observation["x.vel"] == 0.2
    assert np.array_equal(frames["front_left"], frames_2["front_left"])
