#!/usr/bin/env python

from __future__ import annotations

import time

import numpy as np
import zmq

from lerobot.slam.integrations.rollout import (
    RolloutSlamConfig,
    SlamAwareDatasetProxy,
    SlamAwareRobotProxy,
    add_rollout_slam_dataset_features,
    build_rollout_slam_session,
    require_rollout_slam_backend,
)
from lerobot.slam.transport import parse_slam_input_packet


class _FakeSlamSession:
    def __init__(self) -> None:
        self.observed: list[dict] = []
        self.stopped = False

    def observe(self, observation: dict) -> None:
        self.observed.append(observation)

    def stop(self) -> None:
        self.stopped = True

    def frame_fields(self) -> dict:
        return {
            "slam.pose.xyz": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "slam.pose.quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "slam.health.status": np.array([0], dtype=np.int64),
            "slam.health.latency_ms": np.array([5.0], dtype=np.float32),
        }


class _FakeRobot:
    def __init__(self) -> None:
        self.disconnected = False
        self._obs = {"front_left": np.zeros((2, 2, 3), dtype=np.uint8)}

    def get_observation(self) -> dict:
        return dict(self._obs)

    def send_action(self, action: dict) -> dict:
        return dict(action)

    def disconnect(self) -> None:
        self.disconnected = True


class _FakeDataset:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)


def test_dataset_proxy_appends_slam_fields() -> None:
    dataset = _FakeDataset()
    session = _FakeSlamSession()
    proxy = SlamAwareDatasetProxy(dataset, session)

    proxy.add_frame({"task": "demo"})
    assert len(dataset.frames) == 1
    frame = dataset.frames[0]
    assert frame["task"] == "demo"
    assert "slam.pose.xyz" in frame
    assert frame["slam.pose.xyz"].shape == (3,)


def test_robot_proxy_observes_and_stops_session() -> None:
    robot = _FakeRobot()
    session = _FakeSlamSession()
    proxy = SlamAwareRobotProxy(robot, session)

    obs = proxy.get_observation()
    assert obs["front_left"].shape == (2, 2, 3)
    assert len(session.observed) == 1

    proxy.disconnect()
    assert session.stopped
    assert robot.disconnected


def test_add_rollout_slam_dataset_features() -> None:
    features = {"observation.images.front_left": {"dtype": "video"}}
    add_rollout_slam_dataset_features(features)

    assert "slam.pose.xyz" in features
    assert "slam.pose.quat" in features
    assert "slam.health.status" in features
    assert "slam.health.latency_ms" in features


def test_require_rollout_slam_backend_raises_when_backend_unavailable() -> None:
    class _Adapter:
        backend_ready = False
        backend_error = "No module named 'orbslam3'"

    class _Orchestrator:
        adapter = _Adapter()

    class _Runtime:
        orchestrator = _Orchestrator()

    class _Session:
        runtime = _Runtime()

    try:
        require_rollout_slam_backend(_Session(), "orbslam3")
    except RuntimeError as exc:
        message = str(exc)
        assert "backend 'orbslam3' is unavailable" in message
        assert "No module named 'orbslam3'" in message
    else:
        raise AssertionError("Expected require_rollout_slam_backend to raise RuntimeError.")


def test_remote_rollout_session_publishes_client_observations() -> None:
    input_endpoint = "inproc://rollout-slam-input-bridge"
    output_endpoint = "inproc://rollout-slam-output-bridge"
    cfg = RolloutSlamConfig(
        enabled=True,
        source_mode="remote_endpoint",
        input_endpoint=input_endpoint,
        remote_endpoint=output_endpoint,
        stereo_left_key="front_left",
        stereo_right_key="front_right",
    )
    session = build_rollout_slam_session(cfg)

    context = zmq.Context.instance()
    input_sub = context.socket(zmq.SUB)
    input_sub.connect(input_endpoint)
    input_sub.setsockopt(zmq.SUBSCRIBE, b"")
    try:
        # Allow PUB/SUB handshake before first publish.
        time.sleep(0.03)
        session.observe(
            {
                "front_left": np.full((8, 8, 3), 10, dtype=np.uint8),
                "front_right": np.full((8, 8, 3), 20, dtype=np.uint8),
                "x.vel": 0.25,
                "y.vel": 0.0,
                "theta.vel": 0.5,
            }
        )
        time.sleep(0.03)
        payload = input_sub.recv(flags=zmq.NOBLOCK)
        slam_input = parse_slam_input_packet(payload)
        assert slam_input.left.name == "front_left"
        assert slam_input.right.name == "front_right"
        assert slam_input.base_velocity["x.vel"] == 0.25
    finally:
        session.stop()
        input_sub.close()
