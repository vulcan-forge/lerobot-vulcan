#!/usr/bin/env python

from __future__ import annotations

import numpy as np

from lerobot.slam.integrations.rollout import (
    SlamAwareDatasetProxy,
    SlamAwareRobotProxy,
    add_rollout_slam_dataset_features,
)


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
