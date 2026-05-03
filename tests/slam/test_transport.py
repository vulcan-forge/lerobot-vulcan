#!/usr/bin/env python

from __future__ import annotations

import time

import numpy as np
import zmq

from lerobot.slam.transport import (
    RolloutSlamSubscriber,
    build_slam_input_packet,
    parse_slam_input_packet,
    parse_slam_output,
    serialize_slam_output,
)
from lerobot.slam.types import SlamHealth, SlamOutput, SlamPose, SlamStatus


def test_slam_input_packet_roundtrip() -> None:
    left = np.full((32, 32, 3), 127, dtype=np.uint8)
    right = np.full((32, 32, 3), 64, dtype=np.uint8)
    packet = build_slam_input_packet(
        source="unit-test",
        host_monotonic_ns=123456789,
        base_velocity={"x.vel": 1.0, "y.vel": 0.5, "theta.vel": 0.1},
        frame_ids={"front_left": 10, "front_right": 11},
        capture_monotonic_ns={"front_left": 111, "front_right": 222},
        camera_frames={"front_left": left, "front_right": right},
        jpeg_quality=80,
    )

    slam_input = parse_slam_input_packet(packet)
    assert slam_input.source == "unit-test"
    assert slam_input.host_monotonic_ns == 123456789
    assert slam_input.left.frame_id == 10
    assert slam_input.right.frame_id == 11
    assert slam_input.left.capture_monotonic_ns == 111
    assert slam_input.right.capture_monotonic_ns == 222
    assert slam_input.left.image.shape == left.shape
    assert slam_input.right.image.shape == right.shape
    assert slam_input.base_velocity["x.vel"] == 1.0


def test_slam_output_roundtrip() -> None:
    output = SlamOutput(
        timestamp_ns=999,
        pose=SlamPose(x=1.2, y=3.4, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        covariance=[0.1] * 36,
        health=SlamHealth(status=SlamStatus.HEALTHY, detail="ok", latency_ms=3.2),
        backend="orbslam3",
    )
    encoded = serialize_slam_output(output)
    decoded = parse_slam_output(encoded)
    assert decoded.timestamp_ns == 999
    assert decoded.pose.x == 1.2
    assert decoded.pose.y == 3.4
    assert decoded.health.status == SlamStatus.HEALTHY
    assert decoded.backend == "orbslam3"


def test_rollout_slam_subscriber_receives_latest() -> None:
    endpoint = "inproc://slam-output-test"
    context = zmq.Context.instance()
    pub = context.socket(zmq.PUB)
    pub.bind(endpoint)

    sub = RolloutSlamSubscriber(endpoint)
    sub.start()

    # Allow PUB/SUB subscription handshake.
    time.sleep(0.03)
    pub.send(
        serialize_slam_output(
            SlamOutput(
                timestamp_ns=123,
                pose=SlamPose(x=7.0, y=8.0, z=0.0),
                health=SlamHealth(status=SlamStatus.HEALTHY, detail="ok"),
                backend="orbslam3",
            )
        )
    )
    time.sleep(0.03)
    latest = sub.poll_latest()
    assert latest is not None
    assert latest.pose.x == 7.0
    assert latest.pose.y == 8.0
    assert latest.health.status == SlamStatus.HEALTHY

    sub.stop()
    pub.close()

