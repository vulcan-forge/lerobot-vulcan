from __future__ import annotations

from typing import Any, Mapping

from .models import VRTeleopSample


class GrpcPoseStream:
    def __init__(self, *, port: int) -> None:
        self.port = port
        self.server = None
        self.pose_service = None

    def start(self) -> None:
        from lerobot.transport.phone_teleop_grpc.pos_grpc_server import start_grpc_server

        self.server, self.pose_service = start_grpc_server(port=self.port)

    def stop(self) -> None:
        if self.server is not None:
            self.server.stop(0)
        self.server = None
        self.pose_service = None

    def wait_for_pose(self, timeout: float) -> VRTeleopSample | None:
        return self._read(block=True, timeout=timeout)

    def read_fresh(self, timeout: float) -> VRTeleopSample | None:
        return self._read(block=True, timeout=timeout)

    def read_latest(self) -> VRTeleopSample | None:
        return self._read(block=False, timeout=None)

    def _read(self, *, block: bool, timeout: float | None) -> VRTeleopSample | None:
        if self.pose_service is None:
            raise RuntimeError("Pose stream has not been started")
        payload = self.pose_service.get_latest_pose(block=block, timeout=timeout)
        return self.coerce_payload(payload)

    @staticmethod
    def coerce_payload(payload: Mapping[str, Any] | None) -> VRTeleopSample | None:
        return VRTeleopSample.from_payload(payload)
