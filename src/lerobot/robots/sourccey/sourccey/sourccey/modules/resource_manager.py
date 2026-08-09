"""Runtime hardware mode management for the Sourccey host.

The host can keep the base process alive while switching task-specific hardware
profiles.  For example, mapping can release the front/wrist cameras and keep
only bottom odometry active, while teleop can bring the normal camera set back.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zmq

logger = logging.getLogger(__name__)

HOST_MODE_SCHEMA = "sourccey.host_mode.v1"

ARM_JOINTS: tuple[str, ...] = (
    "left_shoulder_pan.pos",
    "left_shoulder_lift.pos",
    "left_elbow_flex.pos",
    "left_wrist_flex.pos",
    "left_wrist_roll.pos",
    "left_gripper.pos",
    "right_shoulder_pan.pos",
    "right_shoulder_lift.pos",
    "right_elbow_flex.pos",
    "right_wrist_flex.pos",
    "right_wrist_roll.pos",
    "right_gripper.pos",
)


@dataclass(frozen=True)
class HostModeStatus:
    mode: str
    active_camera_keys: tuple[str, ...]
    slam_input_active: bool
    arm_stow_applied: bool
    message: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "schema": HOST_MODE_SCHEMA,
            "mode": self.mode,
            "active_camera_keys": list(self.active_camera_keys),
            "slam_input_active": bool(self.slam_input_active),
            "arm_stow_applied": bool(self.arm_stow_applied),
            "message": self.message,
        }


class HostResourceManager:
    """Switch Sourccey's hardware devices by task-oriented mode."""

    VALID_MODES = {
        "idle",
        "base_only",
        "slam_mapping",
        "slam_front",
        "teleop_full",
        "safety_only",
    }

    def __init__(self, robot, config):
        self.robot = robot
        self.config = config
        self._lock = threading.RLock()
        self._mode = "idle"
        self._slam_input_active = False
        self._arm_stow_applied = False
        self._message = "initialized"

    def status(self) -> HostModeStatus:
        with self._lock:
            return HostModeStatus(
                mode=self._mode,
                active_camera_keys=self.robot.connected_camera_keys(),
                slam_input_active=self._slam_input_active,
                arm_stow_applied=self._arm_stow_applied,
                message=self._message,
            )

    def slam_input_active(self) -> bool:
        with self._lock:
            return self._slam_input_active

    def fused_camera_active(self) -> bool:
        with self._lock:
            return self._mode in {"slam_front", "teleop_full"}

    def resolve_initial_mode(self, requested: str) -> str:
        requested = str(requested or "auto").strip().lower()
        if requested != "auto":
            return self._normalize_mode(requested)
        if bool(getattr(self.config, "lidar_mapping_bottom_only_mode", False)):
            return "slam_mapping"
        if bool(getattr(self.config, "lidar_mapping_control_only_mode", False)):
            return "base_only"
        if bool(getattr(self.config, "slam_eye_only_mode", False)):
            return "slam_front"
        if (
            bool(getattr(self.config, "slam_input_enabled", False))
            and str(getattr(self.config, "host_slam_camera_profile", "")).strip().lower() == "bottom"
        ):
            return "slam_mapping"
        return "teleop_full"

    def apply_mode(
        self,
        mode: str,
        *,
        stow_arms: bool | None = None,
        stow_pose_path: str | None = None,
        reason: str = "",
    ) -> HostModeStatus:
        mode = self._normalize_mode(mode)
        should_stow = self._should_stow(mode, stow_arms)
        pose_path = stow_pose_path or str(getattr(self.config, "host_slam_stow_pose_path", ""))

        with self._lock:
            logger.info("Applying Sourccey host mode '%s' (%s)", mode, reason or "no reason provided")
            self.robot.watchdog_stop_motion()
            self.robot.set_connected_cameras(self._camera_keys_for_mode(mode))
            self._slam_input_active = mode in {"slam_mapping", "slam_front", "safety_only"}
            self._mode = mode
            self._message = reason or f"mode switched to {mode}"

            if should_stow:
                pose = self._load_stow_pose(pose_path)
                self._apply_stow_pose(
                    pose,
                    settle_s=float(getattr(self.config, "host_slam_stow_settle_s", 3.0)),
                )
                self._arm_stow_applied = True
                self._message = f"{self._message}; arms stowed from {pose_path}"
            return self.status()

    def _normalize_mode(self, mode: str) -> str:
        normalized = str(mode or "").strip().lower().replace("-", "_")
        aliases = {
            "slam": "slam_mapping",
            "mapping": "slam_mapping",
            "lidar_mapping": "slam_mapping",
            "teleop": "teleop_full",
            "full": "teleop_full",
            "base": "base_only",
            "safety": "safety_only",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in self.VALID_MODES:
            raise ValueError(
                f"unknown host mode '{mode}'. Expected one of: {', '.join(sorted(self.VALID_MODES))}"
            )
        return normalized

    def _camera_keys_for_mode(self, mode: str) -> tuple[str, ...]:
        available = set(self.robot.cameras.keys())
        if mode in {"idle", "base_only"}:
            return ()
        if mode in {"slam_mapping", "safety_only"}:
            return tuple(key for key in ("bottom",) if key in available)
        if mode == "slam_front":
            keys = ["front_left", "front_right"]
            if bool(getattr(self.config, "bottom_camera_enabled", True)):
                keys.append("bottom")
            return tuple(key for key in keys if key in available)
        if mode == "teleop_full":
            preferred = ["front_left", "front_right", "wrist_left", "wrist_right", "bottom"]
            ordered = [key for key in preferred if key in available]
            ordered.extend(key for key in self.robot.cameras.keys() if key not in ordered)
            return tuple(ordered)
        return ()

    def _should_stow(self, mode: str, requested: bool | None) -> bool:
        if requested is not None:
            return bool(requested)
        return mode.startswith("slam") and bool(getattr(self.config, "host_slam_stow_arms", False))

    def _load_stow_pose(self, path: str) -> dict[str, float]:
        pose_path = Path(path)
        if not pose_path.is_absolute():
            pose_path = Path.cwd() / pose_path
        try:
            data = json.loads(pose_path.read_text())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"unable to load arm stow pose at {pose_path}") from exc
        if not all(key in data for key in ARM_JOINTS):
            raise RuntimeError(f"arm stow pose at {pose_path} is missing one or more joints")
        pose = {key: float(data[key]) for key in ARM_JOINTS}
        if all(abs(value) < 1e-6 for value in pose.values()):
            raise RuntimeError(f"arm stow pose at {pose_path} is all zeros; refusing to command it")
        return pose

    def _apply_stow_pose(self, pose: dict[str, float], *, settle_s: float) -> None:
        base = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0, "z.pos": 100.0}
        posed_on = {**base, **pose, "untorque_left": False, "untorque_right": False}
        relax = {**base, "untorque_left": True, "untorque_right": True}
        self._stream_action(posed_on, 0.5)
        self._stream_action(relax, 0.5)
        self._stream_action(posed_on, settle_s)

    def _stream_action(self, action: dict[str, Any], duration_s: float, rate_hz: float = 20.0) -> None:
        deadline = time.monotonic() + max(0.0, float(duration_s))
        interval_s = 1.0 / max(float(rate_hz), 1.0)
        while time.monotonic() < deadline:
            self.robot.send_action(dict(action))
            time.sleep(interval_s)


class HostModeControlService:
    """Token-gated JSON/REQ endpoint for explicit client mode requests."""

    def __init__(self, context: zmq.Context, config, manager: HostResourceManager):
        self.context = context
        self.config = config
        self.manager = manager
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket = None

    def start(self) -> None:
        if self._thread is not None:
            return
        token = self._control_token()
        if not token:
            raise RuntimeError("host mode control requires non-empty host_mode_control_token")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="sourccey-host-mode-control", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None

    def _run(self) -> None:
        socket = self.context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        self._socket = socket
        endpoint = f"tcp://*:{int(self.config.port_zmq_host_mode)}"
        socket.bind(endpoint)
        print(f"[HOST] Mode control: REP on {endpoint} (token required)")
        while not self._stop.is_set():
            if not socket.poll(100, zmq.POLLIN):
                continue
            try:
                request = socket.recv_json(flags=zmq.NOBLOCK)
                response = self._handle_request(request)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Host mode request failed")
                response = {
                    "ok": False,
                    "error": str(exc),
                    **self.manager.status().as_json(),
                }
            try:
                socket.send_json(response, flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        token = self._control_token()
        if str(request.get("token", "")) != token:
            raise PermissionError("invalid host mode control token")
        action = str(request.get("action", "set_mode")).strip().lower()
        if action == "status":
            return {"ok": True, **self.manager.status().as_json()}
        if action != "set_mode":
            raise ValueError(f"unknown host mode action '{action}'")
        requested_stow = request.get("stow_arms")
        if bool(requested_stow) and not bool(getattr(self.config, "host_mode_allow_remote_arm_stow", False)):
            raise PermissionError("remote arm stow is disabled on this host")
        status = self.manager.apply_mode(
            str(request.get("mode", "")),
            stow_arms=requested_stow,
            stow_pose_path=request.get("stow_pose_path"),
            reason=str(request.get("reason", "client request")),
        )
        return {"ok": True, **status.as_json()}

    def _control_token(self) -> str:
        return str(
            getattr(self.config, "host_mode_control_token", "")
            or os.environ.get("SOURCCEY_HOST_MODE_TOKEN", "")
            or ""
        )
