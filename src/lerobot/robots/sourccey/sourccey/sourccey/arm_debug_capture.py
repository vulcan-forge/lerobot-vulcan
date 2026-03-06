import json
import os
import time
from pathlib import Path
from typing import Any


ARM_POSITION_KEYS = (
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


def extract_arm_positions(data: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(data, dict):
        return {}
    arm_data: dict[str, float] = {}
    for key in ARM_POSITION_KEYS:
        if key in data:
            try:
                arm_data[key] = float(data[key])
            except (TypeError, ValueError):
                continue
    return arm_data


def _default_capture_dir() -> Path:
    configured_dir = os.getenv("SOURCCEY_ARM_DEBUG_DIR")
    if configured_dir:
        return Path(configured_dir).expanduser()
    return Path.home() / "Desktop" / "calibrations" / "run-logs"


class ArmDebugCapture:
    def __init__(
        self,
        *,
        enabled: bool,
        duration_s: float,
        motion_threshold: float,
        label: str,
        capture_path: str | Path | None = None,
    ) -> None:
        self.enabled = enabled
        self.duration_s = max(float(duration_s), 0.0)
        self.motion_threshold = max(float(motion_threshold), 0.0)
        self.label = label

        self.capture_started_at: float | None = None
        self.capture_start_reason: str | None = None
        self.capture_start_delta: float | None = None
        self.closed = False

        self.fpath: Path | None = None
        self._fh = None
        if not self.enabled:
            return

        if capture_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.fpath = _default_capture_dir() / f"sourccey_{self.label}_arm_debug_{timestamp}.jsonl"
        else:
            self.fpath = Path(capture_path).expanduser()

        self.fpath.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.fpath.open("w", encoding="utf-8", buffering=1)
        self._write(
            "session_start",
            {
                "label": self.label,
                "duration_s": self.duration_s,
                "motion_threshold": self.motion_threshold,
                "path": str(self.fpath),
            },
            include_dt=False,
            durable=True,
        )

    @property
    def path(self) -> str | None:
        return str(self.fpath) if self.fpath else None

    @property
    def is_active(self) -> bool:
        return self.capture_started_at is not None and not self.closed

    def maybe_start(self, *, action: dict[str, Any] | None, observation: dict[str, Any] | None) -> None:
        if not self.enabled or self.closed or self.capture_started_at is not None:
            return

        action_arm = extract_arm_positions(action)
        if not action_arm:
            return

        obs_arm = extract_arm_positions(observation)
        common_keys = [k for k in action_arm if k in obs_arm]
        max_abs_delta = 0.0
        if common_keys:
            max_abs_delta = max(abs(action_arm[k] - obs_arm[k]) for k in common_keys)
            if max_abs_delta < self.motion_threshold:
                return
            reason = "action_observation_delta"
        else:
            reason = "first_arm_action_no_observation_baseline"

        self.capture_started_at = time.monotonic()
        self.capture_start_reason = reason
        self.capture_start_delta = max_abs_delta
        self._write(
            "capture_start",
            {
                "reason": reason,
                "max_abs_delta": max_abs_delta,
                "arm_action": action_arm,
                "arm_observation": obs_arm,
            },
            include_dt=False,
            durable=True,
        )

    def record(self, event: str, payload: dict[str, Any]) -> None:
        if not self.enabled or self.closed or self.capture_started_at is None:
            return

        elapsed = time.monotonic() - self.capture_started_at
        if elapsed > self.duration_s:
            self._write(
                "capture_end",
                {
                    "elapsed_s": elapsed,
                    "reason": "duration_elapsed",
                },
                durable=True,
            )
            self.close()
            return

        self._write(event, payload)

    def record_session(self, event: str, payload: dict[str, Any]) -> None:
        if not self.enabled or self.closed:
            return
        self._write(event, payload, include_dt=False)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._fh is not None:
            try:
                self._flush_to_disk()
                self._fh.close()
            except Exception:
                pass
        self._fh = None

    def _write(
        self,
        event: str,
        payload: dict[str, Any],
        *,
        include_dt: bool = True,
        durable: bool = False,
    ) -> None:
        if self._fh is None:
            return

        row: dict[str, Any] = {
            "event": event,
            "wall_time_s": time.time(),
        }
        if include_dt:
            row["capture_dt_s"] = (
                time.monotonic() - self.capture_started_at if self.capture_started_at is not None else None
            )
        row.update(payload)
        self._fh.write(json.dumps(row, ensure_ascii=True) + "\n")
        if durable:
            self._flush_to_disk()

    def _flush_to_disk(self) -> None:
        if self._fh is None:
            return
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        except Exception:
            # Best-effort durability. Logging must not crash control loop.
            pass
