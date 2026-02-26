# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 Vulcan Robotics, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Any, Mapping

CAMERA_CONTROL_PREFIX = b"SCY_CAMCFG_V1:"

CAMERA_PROFILE_ACTION_KEY = "camera.profile"
CAMERA_NAME_ACTION_KEY = "camera.name"
CAMERA_DEVICE_ACTION_KEY = "camera.device"
CAMERA_WIDTH_ACTION_KEY = "camera.width"
CAMERA_HEIGHT_ACTION_KEY = "camera.height"
CAMERA_FPS_ACTION_KEY = "camera.fps"

CAMERA_CONTROL_ACTION_KEYS = {
    CAMERA_PROFILE_ACTION_KEY,
    CAMERA_NAME_ACTION_KEY,
    CAMERA_DEVICE_ACTION_KEY,
    CAMERA_WIDTH_ACTION_KEY,
    CAMERA_HEIGHT_ACTION_KEY,
    CAMERA_FPS_ACTION_KEY,
}


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _optional_non_empty_str(value: Any) -> str | None:
    if value is None:
        return None
    parsed = str(value).strip()
    return parsed or None


def _sanitize_camera_control_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    profile = _optional_non_empty_str(payload.get("profile", payload.get(CAMERA_PROFILE_ACTION_KEY)))
    if not profile:
        return None

    control: dict[str, Any] = {"profile": profile}

    camera_name = _optional_non_empty_str(payload.get("camera_name", payload.get(CAMERA_NAME_ACTION_KEY)))
    if camera_name is not None:
        control["camera_name"] = camera_name

    device = _optional_non_empty_str(payload.get("device", payload.get(CAMERA_DEVICE_ACTION_KEY)))
    if device is not None:
        control["device"] = device

    width = _optional_positive_int(payload.get("width", payload.get(CAMERA_WIDTH_ACTION_KEY)))
    if width is not None:
        control["width"] = width

    height = _optional_positive_int(payload.get("height", payload.get(CAMERA_HEIGHT_ACTION_KEY)))
    if height is not None:
        control["height"] = height

    fps = _optional_positive_int(payload.get("fps", payload.get(CAMERA_FPS_ACTION_KEY)))
    if fps is not None:
        control["fps"] = fps

    return control


def extract_camera_control_from_action(action: Mapping[str, Any]) -> dict[str, Any] | None:
    return _sanitize_camera_control_payload(action)


def strip_camera_control_keys(action: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in action.items() if key not in CAMERA_CONTROL_ACTION_KEYS}


def encode_camera_control_message(control: Mapping[str, Any]) -> bytes:
    payload = _sanitize_camera_control_payload(control)
    if payload is None:
        raise ValueError("Camera control payload must include a non-empty profile.")
    return CAMERA_CONTROL_PREFIX + json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_camera_control_message(raw: bytes) -> dict[str, Any] | None:
    if not raw.startswith(CAMERA_CONTROL_PREFIX):
        return None

    payload_bytes = raw[len(CAMERA_CONTROL_PREFIX):]
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    return _sanitize_camera_control_payload(payload)
