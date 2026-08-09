from __future__ import annotations

import zmq


def create_slam_pub_socket(zmq_context: zmq.Context, endpoint: str) -> zmq.Socket:
    socket = zmq_context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.bind(endpoint)
    except zmq.ZMQError as e:
        socket.close(0)
        raise RuntimeError(f"Failed to bind SLAM input publisher at {endpoint}: {e}") from e
    return socket


def close_slam_pub_socket(socket: zmq.Socket | None) -> None:
    if socket is not None:
        socket.close(0)
