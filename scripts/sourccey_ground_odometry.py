"""Short-baseline ground odometry from Sourccey's bottom camera.

This module deliberately estimates *relative translation only*.  IMU yaw is
the heading authority and LiDAR scan matching remains the pose authority.  The
camera measurement is a motion prior which helps the LiDAR matcher stay in the
correct local basin when the base slips or crosses a doorway.

No commanded motor value is consumed here: Sourccey's DC PWM base currently
has no wheel encoders, so a requested velocity is not an odometry measurement.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from sourccey_camera_geometry import CameraModel, default_bottom


class CameraFrameSource(Protocol):
    def latest(self, cam_name: str) -> tuple[np.ndarray | None, float | None]: ...


class YawSource(Protocol):
    def deg(self) -> float | None: ...


@dataclass(frozen=True)
class GroundOdometryDelta:
    """Accumulated robot-centre translation in the continuous IMU frame."""

    forward_imu_m: float
    left_imu_m: float
    samples: int
    support: int
    residual_m: float


@dataclass(frozen=True)
class GroundFlowEstimate:
    """One frame-pair translation expressed in the previous robot frame."""

    forward_m: float
    left_m: float
    support: int
    tracked: int
    residual_m: float


def _floor_points_robot_frame(
    pixels_xy: np.ndarray,
    image_shape: tuple[int, ...],
    model: CameraModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project image pixels onto the floor in robot-centre coordinates."""
    pixels = np.asarray(pixels_xy, dtype=np.float64).reshape((-1, 2))
    height, width = int(image_shape[0]), int(image_shape[1])
    x_ratio = pixels[:, 0] / max(1.0, float(width - 1))
    y_ratio = pixels[:, 1] / max(1.0, float(height - 1))
    depression_deg = float(model.pitch_down_deg) + (
        y_ratio - 0.5
    ) * float(model.vfov_deg)
    valid = depression_deg > 1.0
    distance = np.full(len(pixels), np.nan, dtype=np.float64)
    distance[valid] = float(model.height_m) / np.tan(
        np.radians(depression_deg[valid])
    )
    valid &= np.isfinite(distance) & (distance >= 0.06) & (distance <= 2.5)
    bearing_deg = float(model.bearing_sign) * (
        float(model.yaw_deg) - (x_ratio - 0.5) * float(model.hfov_deg)
    )
    bearing = np.radians(bearing_deg)
    points = np.column_stack(
        (
            float(model.forward_offset_m) + distance * np.cos(bearing),
            float(model.lateral_offset_m) + distance * np.sin(bearing),
        )
    )
    return points, valid


def estimate_ground_translation_from_tracks(
    previous_pixels_xy: np.ndarray,
    current_pixels_xy: np.ndarray,
    image_shape: tuple[int, ...],
    model: CameraModel,
    yaw_delta_deg: float,
    *,
    minimum_support: int = 10,
    maximum_residual_m: float = 0.055,
) -> GroundFlowEstimate | None:
    """Estimate centre translation from matched floor features.

    For a static floor point, ``p_previous = translation + R(dyaw) p_current``.
    Each feature therefore votes for the same robot-centre translation.  A
    median/MAD consensus rejects moving objects and mistracked pixels.
    """
    previous = np.asarray(previous_pixels_xy, dtype=np.float64).reshape((-1, 2))
    current = np.asarray(current_pixels_xy, dtype=np.float64).reshape((-1, 2))
    if len(previous) != len(current) or len(previous) < int(minimum_support):
        return None
    p_previous, valid_previous = _floor_points_robot_frame(
        previous, image_shape, model
    )
    p_current, valid_current = _floor_points_robot_frame(
        current, image_shape, model
    )
    valid = valid_previous & valid_current
    if int(np.count_nonzero(valid)) < int(minimum_support):
        return None
    p_previous = p_previous[valid]
    p_current = p_current[valid]
    angle = math.radians(float(yaw_delta_deg))
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    votes = p_previous - p_current @ rotation.T
    median = np.median(votes, axis=0)
    residuals = np.hypot(*(votes - median).T)
    median_residual = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median_residual)))
    threshold = max(0.018, min(float(maximum_residual_m), median_residual + 3.5 * mad))
    inliers = residuals <= threshold
    support = int(np.count_nonzero(inliers))
    if support < int(minimum_support):
        return None
    translation = np.median(votes[inliers], axis=0)
    final_residual = float(
        np.median(np.hypot(*(votes[inliers] - translation).T))
    )
    if not np.all(np.isfinite(translation)) or final_residual > float(maximum_residual_m):
        return None
    return GroundFlowEstimate(
        forward_m=float(translation[0]),
        left_m=float(translation[1]),
        support=support,
        tracked=len(votes),
        residual_m=final_residual,
    )


def estimate_ground_flow(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    model: CameraModel,
    yaw_delta_deg: float,
    *,
    minimum_support: int = 10,
) -> GroundFlowEstimate | None:
    """Track floor texture and estimate one short-baseline body translation."""
    previous = np.asarray(previous_gray)
    current = np.asarray(current_gray)
    if previous.shape != current.shape or previous.ndim != 2 or previous.size == 0:
        return None
    height, width = previous.shape
    mask = np.zeros_like(previous, dtype=np.uint8)
    horizon = int(
        np.clip((model.horizon_y_ratio() + 0.035) * height, 0, height - 1)
    )
    mask[horizon : max(horizon + 1, int(0.96 * height)), :] = 255
    features = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=180,
        qualityLevel=0.012,
        minDistance=6,
        mask=mask,
        blockSize=5,
    )
    if features is None or len(features) < int(minimum_support):
        return None
    forward, status_forward, _error = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        features,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if forward is None or status_forward is None:
        return None
    backward, status_backward, _error = cv2.calcOpticalFlowPyrLK(
        current,
        previous,
        forward,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if backward is None or status_backward is None:
        return None
    p0 = features.reshape((-1, 2))
    p1 = forward.reshape((-1, 2))
    p0_back = backward.reshape((-1, 2))
    fb_error = np.hypot(*(p0_back - p0).T)
    good = (
        status_forward.reshape(-1).astype(bool)
        & status_backward.reshape(-1).astype(bool)
        & np.isfinite(p1).all(axis=1)
        & (fb_error <= 1.25)
        & (p1[:, 0] >= 0)
        & (p1[:, 0] < width)
        & (p1[:, 1] >= horizon)
        & (p1[:, 1] < 0.98 * height)
    )
    if int(np.count_nonzero(good)) < int(minimum_support):
        return None
    return estimate_ground_translation_from_tracks(
        p0[good],
        p1[good],
        previous.shape,
        model,
        yaw_delta_deg,
        minimum_support=minimum_support,
    )


class BottomCameraGroundOdometry:
    """Continuously accumulate validated bottom-camera translation samples."""

    def __init__(
        self,
        camera_source: CameraFrameSource,
        yaw_source: YawSource,
        *,
        camera_key: str = "bottom",
        model: CameraModel | None = None,
        rate_hz: float = 20.0,
        maximum_frame_age_s: float = 0.60,
    ) -> None:
        self.camera_source = camera_source
        self.yaw_source = yaw_source
        self.camera_key = str(camera_key)
        self.model = model or default_bottom()
        self.period_s = 1.0 / max(1.0, float(rate_hz))
        self.maximum_frame_age_s = float(maximum_frame_age_s)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._previous_gray: np.ndarray | None = None
        self._previous_frame_identity: object | None = None
        self._previous_yaw: float | None = None
        self._previous_at: float | None = None
        self._pending = np.zeros(2, dtype=np.float64)
        self._pending_samples = 0
        self._pending_support = 0
        self._pending_residual_sum = 0.0
        self._accepted_total = 0
        self._rejected_total = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="bottom-camera-ground-odometry",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _reseed(
        self,
        frame: np.ndarray,
        frame_identity: object,
        yaw: float,
        now: float,
    ) -> None:
        self._previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._previous_frame_identity = frame_identity
        self._previous_yaw = float(yaw)
        self._previous_at = float(now)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample_fn = getattr(self.camera_source, "latest_sample", None)
            if callable(sample_fn):
                frame, age_s, sequence = sample_fn(self.camera_key)
                frame_identity: object | None = sequence
            else:
                frame, age_s = self.camera_source.latest(self.camera_key)
                frame_identity = id(frame) if frame is not None else None
            yaw = self.yaw_source.deg()
            now = time.monotonic()
            if (
                frame is None
                or age_s is None
                or frame_identity is None
                or float(age_s) > self.maximum_frame_age_s
                or yaw is None
            ):
                time.sleep(self.period_s)
                continue
            if frame_identity == self._previous_frame_identity:
                time.sleep(self.period_s)
                continue
            if self._previous_gray is None or self._previous_yaw is None or self._previous_at is None:
                self._reseed(frame, frame_identity, float(yaw), now)
                time.sleep(self.period_s)
                continue
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dt = max(1e-3, now - self._previous_at)
            yaw_delta = float(yaw) - float(self._previous_yaw)
            estimate = estimate_ground_flow(
                self._previous_gray,
                current_gray,
                self.model,
                yaw_delta,
            )
            previous_yaw = float(self._previous_yaw)
            self._previous_gray = current_gray
            self._previous_frame_identity = frame_identity
            self._previous_yaw = float(yaw)
            self._previous_at = now
            if estimate is None:
                with self._lock:
                    self._rejected_total += 1
                time.sleep(self.period_s)
                continue
            translation = np.asarray(
                [estimate.forward_m, estimate.left_m], dtype=np.float64
            )
            # A camera frame pair is a prior, never permission to teleport.
            maximum_step = min(0.22, 0.04 + 1.2 * dt)
            if float(np.hypot(*translation)) > maximum_step:
                with self._lock:
                    self._rejected_total += 1
                time.sleep(self.period_s)
                continue
            angle = math.radians(previous_yaw)
            rotation = np.asarray(
                [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
                dtype=np.float64,
            )
            delta_imu = rotation @ translation
            with self._lock:
                self._pending += delta_imu
                self._pending_samples += 1
                self._pending_support += int(estimate.support)
                self._pending_residual_sum += float(estimate.residual_m)
                self._accepted_total += 1
            time.sleep(self.period_s)

    def take_delta(self) -> GroundOdometryDelta | None:
        with self._lock:
            if self._pending_samples <= 0:
                return None
            result = GroundOdometryDelta(
                forward_imu_m=float(self._pending[0]),
                left_imu_m=float(self._pending[1]),
                samples=int(self._pending_samples),
                support=int(self._pending_support),
                residual_m=float(
                    self._pending_residual_sum / max(1, self._pending_samples)
                ),
            )
            self._pending[:] = 0.0
            self._pending_samples = 0
            self._pending_support = 0
            self._pending_residual_sum = 0.0
        return result

    def discard(self) -> None:
        """Discard translation accumulated during a pivot or stationary reset."""
        with self._lock:
            self._pending[:] = 0.0
            self._pending_samples = 0
            self._pending_support = 0
            self._pending_residual_sum = 0.0

    @property
    def healthy(self) -> bool:
        with self._lock:
            return self._accepted_total > 0

    @property
    def totals(self) -> tuple[int, int]:
        with self._lock:
            return int(self._accepted_total), int(self._rejected_total)
