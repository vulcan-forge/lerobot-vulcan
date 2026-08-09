"""Texture-conditioned motion calibration for Sourccey's LiDAR SLAM.

The bottom camera is deliberately *not* used here as a second global
localization sensor.  It recognizes the floor surface currently under the
robot (for example, a particular beige carpet or wood floor) and selects a
motion calibration learned on that surface.  LiDAR-confirmed displacement is
the only source permitted to teach or correct that calibration.

This gives the pose predictor a more honest starting point between LiDAR
scans: a command that normally travels 0.60 m on wood may travel only 0.45 m
on carpet.  The predictor remains bounded and LiDAR remains authoritative.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import cv2
import numpy as np


SURFACE_MOTION_VERSION = 1


def ground_texture_signature(image_bgr: np.ndarray) -> list[float]:
    """Return a compact, lighting-tolerant descriptor of the nearby floor.

    Only the lower-middle image is used.  It sees the floor under the robot,
    avoids the bright horizon, and avoids the tiny bottom strip that is often
    shadowed by the chassis.  Colour histograms distinguish rug/wood/etc.; a
    normalized gradient histogram distinguishes similarly coloured textures.
    The output is intentionally a descriptor, not a semantic AI label.
    """
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] < 3 or image.shape[0] < 32 or image.shape[1] < 32:
        return []
    height, width = image.shape[:2]
    crop = image[int(0.52 * height) : int(0.94 * height), int(0.05 * width) : int(0.95 * width), :3]
    if crop.size == 0:
        return []
    crop = cv2.resize(crop, (96, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Hue is meaningful only where saturation exists.  That avoids treating a
    # gray carpet's compression noise as a unique colour identity.
    saturation = hsv[:, :, 1]
    hue_hist = cv2.calcHist([hsv], [0], saturation, [8], [0, 180]).reshape(-1)
    sat_hist = cv2.calcHist([hsv], [1], None, [4], [0, 256]).reshape(-1)
    value_hist = cv2.calcHist([hsv], [2], None, [4], [0, 256]).reshape(-1)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    # Texture orientation is less lighting-dependent than raw brightness.
    orientation_hist, _ = np.histogram(angle, bins=8, range=(0.0, 360.0), weights=magnitude)
    texture_energy = np.asarray([float(np.mean(magnitude)) / 255.0], dtype=np.float32)

    descriptor = np.concatenate((hue_hist, sat_hist, value_hist, orientation_hist, texture_energy)).astype(
        np.float64
    )
    norm = float(np.linalg.norm(descriptor))
    if not math.isfinite(norm) or norm <= 1e-9:
        return []
    return (descriptor / norm).astype(np.float32).tolist()


def texture_similarity(first: list[float] | np.ndarray, second: list[float] | np.ndarray) -> float:
    """Compare two normalized texture descriptors on a stable 0..1 scale."""
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    if len(a) == 0 or a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9 or not math.isfinite(denom):
        return 0.0
    return float(np.clip((float(a @ b) / denom + 1.0) * 0.5, 0.0, 1.0))


@dataclass
class SurfaceMotionModel:
    """One learned relationship between a floor texture and base movement."""

    surface_id: int
    signature: list[float]
    observations: int = 0
    forward_scale: float = 1.0
    lateral_scale: float = 1.0
    forward_abs_error_ratio: float = 0.25
    lateral_abs_error_ratio: float = 0.35
    turn_translation_m_per_deg: float = 0.0


@dataclass
class SurfaceMotionAtlas:
    """Persistent texture-to-motion models and trusted map texture landmarks.

    A model is updated only after the caller has accepted a small LiDAR-local
    displacement.  It therefore cannot learn from a global alias, a motor
    command that did not move the base, or an uncertain scan.
    """

    models: list[SurfaceMotionModel] = field(default_factory=list)
    landmarks: list[dict[str, Any]] = field(default_factory=list)
    current_surface_id: int | None = None
    _last_landmark_surface_id: int | None = field(default=None, repr=False)
    _last_landmark_xy: np.ndarray | None = field(default=None, repr=False)

    similarity_threshold: float = 0.88
    maximum_models: int = 24

    @classmethod
    def from_metadata(cls, payload: object) -> "SurfaceMotionAtlas":
        """Restore an atlas from map JSON, ignoring malformed older data."""
        atlas = cls()
        if not isinstance(payload, dict) or int(payload.get("version", -1)) != SURFACE_MOTION_VERSION:
            return atlas
        for raw in payload.get("models", []):
            if not isinstance(raw, dict):
                continue
            try:
                signature = [float(value) for value in raw["signature"]]
                if not signature:
                    continue
                atlas.models.append(
                    SurfaceMotionModel(
                        surface_id=int(raw["surface_id"]),
                        signature=signature,
                        observations=max(0, int(raw.get("observations", 0))),
                        forward_scale=float(np.clip(raw.get("forward_scale", 1.0), 0.25, 2.5)),
                        lateral_scale=float(np.clip(raw.get("lateral_scale", 1.0), 0.25, 2.5)),
                        forward_abs_error_ratio=float(
                            np.clip(raw.get("forward_abs_error_ratio", 0.25), 0.03, 1.5)
                        ),
                        lateral_abs_error_ratio=float(
                            np.clip(raw.get("lateral_abs_error_ratio", 0.35), 0.03, 1.5)
                        ),
                        turn_translation_m_per_deg=float(
                            np.clip(raw.get("turn_translation_m_per_deg", 0.0), 0.0, 0.03)
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        atlas.models.sort(key=lambda model: model.surface_id)
        atlas.landmarks = [
            dict(item)
            for item in payload.get("landmarks", [])
            if isinstance(item, dict) and isinstance(item.get("pose"), (list, tuple))
        ][-512:]
        return atlas

    def to_metadata(self) -> dict[str, object]:
        """Serialize only durable calibration state; no camera frames are saved."""
        return {
            "version": SURFACE_MOTION_VERSION,
            "models": [asdict(model) for model in self.models],
            "landmarks": list(self.landmarks[-512:]),
        }

    def observe_texture(self, signature: list[float] | np.ndarray) -> tuple[int | None, bool, float]:
        """Classify one current floor view and create a model for a new texture.

        Returns ``(surface_id, changed, similarity)``.  A descriptor is only
        considered new after it is sufficiently distinct from every prior
        model.  The caller records the map boundary only after a trusted LiDAR
        pose is available.
        """
        values = [float(value) for value in np.asarray(signature, dtype=np.float32).reshape(-1)]
        if not values:
            return None, False, 0.0
        scores = [texture_similarity(values, model.signature) for model in self.models]
        if scores:
            best_index = int(np.argmax(scores))
            best_score = float(scores[best_index])
            if best_score >= float(self.similarity_threshold):
                surface_id = self.models[best_index].surface_id
                changed = surface_id != self.current_surface_id
                self.current_surface_id = surface_id
                return surface_id, changed, best_score
        if len(self.models) >= int(self.maximum_models):
            # At capacity, use the closest class rather than emitting endless
            # labels from transient exposure changes.
            if scores:
                best_index = int(np.argmax(scores))
                surface_id = self.models[best_index].surface_id
                changed = surface_id != self.current_surface_id
                self.current_surface_id = surface_id
                return surface_id, changed, float(scores[best_index])
            return None, False, 0.0
        surface_id = max((model.surface_id for model in self.models), default=-1) + 1
        self.models.append(SurfaceMotionModel(surface_id=surface_id, signature=values))
        changed = surface_id != self.current_surface_id
        self.current_surface_id = surface_id
        return surface_id, changed, 1.0

    def model(self, surface_id: int | None = None) -> SurfaceMotionModel | None:
        """Return the requested or current surface calibration model."""
        wanted = self.current_surface_id if surface_id is None else surface_id
        return next((item for item in self.models if item.surface_id == wanted), None)

    def predict_body_translation(
        self,
        commanded_forward_m: float,
        commanded_left_m: float,
        *,
        surface_id: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """Scale an intended body translation and return a conservative error.

        The intended distance is a control-model prediction, never an odometry
        observation.  The returned uncertainty expands the following LiDAR
        search bubble when little calibration evidence exists.
        """
        model = self.model(surface_id)
        if model is None:
            return np.asarray([commanded_forward_m, commanded_left_m], dtype=np.float64), 0.20
        prediction = np.asarray(
            [
                float(commanded_forward_m) * float(model.forward_scale),
                float(commanded_left_m) * float(model.lateral_scale),
            ],
            dtype=np.float64,
        )
        uncertainty = max(
            0.03,
            abs(float(commanded_forward_m)) * float(model.forward_abs_error_ratio),
            abs(float(commanded_left_m)) * float(model.lateral_abs_error_ratio),
        )
        return prediction, float(min(0.45, uncertainty))

    def update_from_lidar(
        self,
        commanded_forward_m: float,
        commanded_left_m: float,
        lidar_forward_m: float,
        lidar_left_m: float,
        *,
        turn_delta_deg: float = 0.0,
        surface_id: int | None = None,
    ) -> bool:
        """Learn a surface scale from a LiDAR-confirmed short motion segment.

        Outliers are rejected before touching a model.  In particular, a pose
        correction after an alias cannot become a fake carpet calibration.
        """
        model = self.model(surface_id)
        if model is None:
            return False
        command = np.asarray([commanded_forward_m, commanded_left_m], dtype=np.float64)
        actual = np.asarray([lidar_forward_m, lidar_left_m], dtype=np.float64)
        if not np.all(np.isfinite(command)) or not np.all(np.isfinite(actual)):
            return False
        if float(np.hypot(*command)) < 0.04 or float(np.hypot(*actual)) > 0.55:
            return False
        updated = False
        # The baseline command model can be imperfect, but it cannot claim a
        # reversed movement or a tenfold scale from one LiDAR correction.
        for axis, scale_name, error_name in (
            (0, "forward_scale", "forward_abs_error_ratio"),
            (1, "lateral_scale", "lateral_abs_error_ratio"),
        ):
            if abs(float(command[axis])) < 0.04:
                continue
            ratio = float(actual[axis] / command[axis])
            if not math.isfinite(ratio) or ratio < 0.25 or ratio > 2.5:
                continue
            previous = float(getattr(model, scale_name))
            alpha = 0.22 if model.observations < 8 else 0.08
            scale = float(np.clip((1.0 - alpha) * previous + alpha * ratio, 0.25, 2.5))
            setattr(model, scale_name, scale)
            relative_error = abs(float(actual[axis]) - float(command[axis]) * scale) / max(
                0.04, abs(float(command[axis]))
            )
            previous_error = float(getattr(model, error_name))
            setattr(model, error_name, float(np.clip(0.85 * previous_error + 0.15 * relative_error, 0.03, 1.5)))
            updated = True
        if abs(float(turn_delta_deg)) >= 4.0:
            turn_drift = float(np.hypot(*actual)) / abs(float(turn_delta_deg))
            if math.isfinite(turn_drift) and turn_drift <= 0.03:
                model.turn_translation_m_per_deg = 0.85 * float(model.turn_translation_m_per_deg) + 0.15 * turn_drift
        if updated:
            model.observations += 1
        return updated

    def record_trusted_landmark(
        self,
        pose_xyz: tuple[float, float, float],
        *,
        minimum_spacing_m: float = 0.20,
    ) -> bool:
        """Record a texture boundary/region sample at an accepted LiDAR pose."""
        model = self.model()
        if model is None:
            return False
        xy = np.asarray(pose_xyz[:2], dtype=np.float64)
        changed = model.surface_id != self._last_landmark_surface_id
        moved = (
            self._last_landmark_xy is None
            or float(np.hypot(*(xy - self._last_landmark_xy))) >= float(minimum_spacing_m)
        )
        if not changed and not moved:
            return False
        self.landmarks.append(
            {
                "surface_id": int(model.surface_id),
                "pose": [float(pose_xyz[0]), float(pose_xyz[1]), float(pose_xyz[2])],
                "signature": list(model.signature),
                "observations": int(model.observations),
            }
        )
        self.landmarks = self.landmarks[-512:]
        self._last_landmark_surface_id = model.surface_id
        self._last_landmark_xy = xy
        return True
