from __future__ import annotations

import base64
import math
from typing import Any

from ....protobuf.generated import sourccey_pb2
from ....protobuf.sourccey_protobuf import SourcceyProtobuf


class RelayCodec:
    def __init__(self) -> None:
        self._converter = SourcceyProtobuf()

    def decode_observation(self, raw_payload: bytes) -> dict[str, Any]:
        state_msg = sourccey_pb2.SourcceyRobotState()
        state_msg.ParseFromString(raw_payload)
        observation = self._converter.protobuf_to_observation(state_msg)
        if isinstance(observation, dict):
            return _json_safe(observation)
        return {"observation": str(observation)}

    def encode_action(self, command: dict[str, Any]) -> tuple[bytes | None, str | None]:
        raw_b64 = command.get("raw_protobuf_b64")
        if isinstance(raw_b64, str) and raw_b64:
            try:
                return base64.b64decode(raw_b64), None
            except Exception:
                return None, "invalid_raw_protobuf_b64"

        action = _action_from_command(command)
        if action is None:
            return None, "invalid_command_payload"

        try:
            proto = self._converter.action_to_protobuf(action)
            return proto.SerializeToString(), None
        except Exception:
            return None, "protobuf_encode_failed"


def _action_from_command(command: dict[str, Any]) -> dict[str, Any] | None:
    if "action" in command and isinstance(command["action"], dict):
        return dict(command["action"])

    if "input" in command and isinstance(command["input"], dict):
        input_payload = command["input"]
        axes = input_payload.get("axes") if isinstance(input_payload, dict) else {}
        if not isinstance(axes, dict):
            axes = {}
        return {
            "x.vel": float(axes.get("x", 0.0)),
            "y.vel": float(axes.get("y", 0.0)),
            "theta.vel": float(axes.get("theta", 0.0)),
            "z.pos": float(axes.get("z", 0.0)),
            "untorque_left": False,
            "untorque_right": False,
        }

    chunk = command.get("action_chunk")
    if isinstance(chunk, list) and chunk:
        first = chunk[0]
        if isinstance(first, list):
            values = first
            return {
                "x.vel": float(values[0]) if len(values) > 0 else 0.0,
                "y.vel": float(values[1]) if len(values) > 1 else 0.0,
                "theta.vel": float(values[2]) if len(values) > 2 else 0.0,
                "z.pos": float(values[3]) if len(values) > 3 else 0.0,
                "untorque_left": False,
                "untorque_right": False,
            }

    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        # JSON doesn't support NaN/Inf. Return None to keep payload valid and avoid server-side parse failures.
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return {"omitted": "camera_frame"}
        return [_json_safe(item) for item in value]
    # Fast-path ndarray-like camera frames without materializing huge nested Python lists.
    ndim = getattr(value, "ndim", None)
    if isinstance(ndim, int) and ndim >= 2:
        return {"omitted": "camera_frame"}
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list) and converted and isinstance(converted[0], list):
            return {"omitted": "camera_frame"}
        return converted
    return str(value)
