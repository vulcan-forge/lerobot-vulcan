from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SlamInputConfig:
    """Configuration for Sourccey -> SLAM sidecar publishing."""

    input_enabled: bool = False
    input_endpoint: str = "tcp://127.0.0.1:5560"
    stereo_left_key: str = "front_left"
    stereo_right_key: str = "front_right"
    jpeg_quality: int = 80
    eye_only_mode: bool = False
    publish_fps: float = 0.0
    resize_width: int | None = None
    resize_height: int | None = None
    extra_camera_keys: tuple[str, ...] = ()
