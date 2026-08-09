"""All image interpretation performed with Sourccey's bottom camera.

This is the single human-readable home for bottom-camera perception.  The host
still owns camera capture and transport, while navigation owns final decisions;
everything that *interprets* a bottom-camera image lives here.

PUBLIC FUNCTION AND CLASS INDEX
===============================

``default_bottom_camera_model``
    Describes the physical mounting and field-calibrated optics of the camera.

``floor_points_robot_frame``
    Converts image pixels into metric floor points relative to robot centre.

``estimate_ground_translation_from_tracks``
    Converts already-matched image features into one robust translation sample.

``estimate_ground_flow``
    Finds and tracks floor texture between two frames, then calls the metric
    translation estimator above.

``BottomCameraGroundOdometry``
    Runs optical flow in a background thread and accumulates validated short
    translations for the LiDAR/IMU localization stack.

``wait_for_live_bottom_camera``
    Proves that the host is sending fresh, changing, non-placeholder bottom
    frames before bottom-camera odometry is allowed to call itself enabled.

``detect_bottom_floor_obstacles`` / ``BottomFloorDetector``
    Finds the floor contact of low obstacles that sit below the 2-D LiDAR scan
    plane.  The class is a small state-free adapter used by the safety monitor.

``bottom_obstacle_explains_floor_point``
    Answers whether a bottom-camera obstacle corroborates an eye-camera floor
    projection, preventing a low object from being mislabeled as an overhang.

``BottomGroundSafetyGate``
    Applies corridor geometry and temporal hysteresis to obstacle observations,
    producing the bottom camera's final low-obstacle safety decision.

``render_bottom_overlay``
    Draws the horizon, obstacle candidates, and current ground-stop decision for
    diagnostics.  It never changes a safety decision.

The odometry path deliberately estimates *relative translation only*.  IMU yaw
is the heading authority and LiDAR scan matching remains the final pose
authority.  No commanded motor value is consumed here: Sourccey's DC PWM base
has no wheel encoders, so a requested velocity is not odometry.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from .geometry import CameraModel


class BottomSafetySettings(Protocol):
    """Configuration fields required by bottom-camera obstacle processing.

    A protocol is used instead of importing ``ElevatedSafetyConfig``.  This
    keeps the bottom-camera module independent of the larger safety monitor and
    prevents a circular import, while documenting exactly which settings the
    detector and diagnostic renderer consume.
    """

    bottom_model: CameraModel
    ground_column_bands: int
    ground_edge_density: float
    ground_max_distance_m: float
    ground_min_obstacle_height_m: float
    ground_gate_enabled: bool
    ground_stop_distance_m: float
    ground_corridor_half_width_m: float
    ground_trip_frames: int
    ground_clear_frames: int
    bottom_bearing_tolerance_deg: float
    bottom_distance_tolerance_m: float


class BottomHazardState(Protocol):
    """Small read-only view of the safety state needed by the overlay.

    The renderer does not need or own the complete three-sensor hazard state.
    Depending only on these fields makes that boundary explicit and keeps the
    diagnostic drawing function reusable in isolation.
    """

    ground_active: bool
    ground_side: str
    ground_distance_m: float | None


class CameraFrameSource(Protocol):
    """Interface supplied by the shared camera-stream subscriber."""

    def latest(self, cam_name: str) -> tuple[np.ndarray | None, float | None]:
        """Return the latest decoded frame and its current age in seconds."""

        ...


@dataclass(frozen=True)
class BottomCameraStreamHealth:
    """Result of the bounded bottom-camera startup health probe.

    Merely decoding one image is not proof of a live camera: a disconnected
    host can retain or repeatedly publish a cached placeholder.  These counts
    make the startup decision and its failure message directly auditable.
    """

    ready: bool
    reason: str
    fresh_frames: int
    usable_frames: int
    changing_pairs: int


def _bottom_frame_has_information(frame: np.ndarray) -> tuple[bool, str]:
    """Check that a decoded frame contains texture usable by optical flow.

    This intentionally does not estimate motion.  The robot may be stationary
    during startup, but a real floor image must still have a valid shape,
    brightness range, and enough trackable corners.  Black/solid placeholders
    therefore cannot satisfy ``--bottom-odometry required``.
    """

    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
        return False, "invalid image array"
    if frame.shape[0] < 48 or frame.shape[1] < 64:
        return False, f"image is only {frame.shape[1]}x{frame.shape[0]}"
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.asarray(gray, dtype=np.uint8)
    low, high = np.percentile(gray, (5.0, 95.0))
    if float(high - low) < 8.0 or float(np.std(gray)) < 3.0:
        return False, "frame is blank or has insufficient contrast"
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=40,
        qualityLevel=0.01,
        minDistance=6.0,
        blockSize=5,
    )
    corner_count = 0 if corners is None else int(len(corners))
    if corner_count < 8:
        return False, f"only {corner_count} trackable floor features"
    return True, f"{corner_count} trackable features"


def wait_for_live_bottom_camera(
    camera_source: CameraFrameSource,
    *,
    timeout_s: float = 5.0,
    camera_key: str = "bottom",
    minimum_fresh_frames: int = 3,
) -> BottomCameraStreamHealth:
    """Wait for a genuinely live and odometry-usable bottom-camera stream.

    The probe requires several fresh subscriber sequence advances, useful image
    content, and at least one small content change.  This distinguishes a real
    stationary camera (whose sensor/JPEG frames still vary slightly) from one
    cached image repeatedly carried in otherwise healthy SLAM packets.  It is
    bounded by ``timeout_s`` and never starts a background worker of its own.
    """

    deadline = time.monotonic() + max(0.0, float(timeout_s))
    required = max(2, int(minimum_fresh_frames))
    seen_sequences: set[int] = set()
    usable_frames = 0
    changing_pairs = 0
    previous_gray: np.ndarray | None = None
    last_problem = "no decoded frame arrived"
    synthetic_sequence = 0
    while time.monotonic() < deadline:
        latest_sample = getattr(camera_source, "latest_sample", None)
        if callable(latest_sample):
            frame, age_s, sequence = latest_sample(camera_key)
        else:
            frame, age_s = camera_source.latest(camera_key)
            synthetic_sequence += 1
            sequence = synthetic_sequence if frame is not None else None
        if frame is None or age_s is None or sequence is None:
            time.sleep(0.05)
            continue
        if float(age_s) > 0.75:
            last_problem = f"latest frame is stale ({float(age_s):.2f}s old)"
            time.sleep(0.05)
            continue
        sequence = int(sequence)
        if sequence in seen_sequences:
            time.sleep(0.05)
            continue
        seen_sequences.add(sequence)
        usable, detail = _bottom_frame_has_information(frame)
        if usable:
            usable_frames += 1
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (80, 60), interpolation=cv2.INTER_AREA)
            if previous_gray is not None:
                mean_change = float(
                    np.mean(np.abs(gray.astype(np.float32) - previous_gray.astype(np.float32)))
                )
                if mean_change >= 0.10:
                    changing_pairs += 1
            previous_gray = gray
        else:
            last_problem = detail
        if len(seen_sequences) >= required and usable_frames >= required and changing_pairs >= 1:
            return BottomCameraStreamHealth(
                True,
                (
                    f"{len(seen_sequences)} fresh frames, {usable_frames} usable, "
                    f"{changing_pairs} changing pair(s)"
                ),
                len(seen_sequences),
                usable_frames,
                changing_pairs,
            )
        time.sleep(0.05)
    if len(seen_sequences) < required:
        last_problem = f"only {len(seen_sequences)}/{required} fresh frame sequence(s) arrived"
    elif usable_frames < required:
        last_problem = f"only {usable_frames}/{required} fresh frames were usable ({last_problem})"
    elif changing_pairs < 1:
        last_problem = "fresh sequence IDs carried an unchanged cached image"
    return BottomCameraStreamHealth(
        False,
        last_problem,
        len(seen_sequences),
        usable_frames,
        changing_pairs,
    )


class YawSource(Protocol):
    """Interface supplied by the continuous IMU yaw reader."""

    def deg(self) -> float | None:
        """Return continuous yaw in degrees, or ``None`` before IMU lock."""

        ...


def default_bottom_camera_model() -> CameraModel:
    """Return the calibrated physical model of Sourccey's floor camera.

    The camera is about 12.7 cm above the floor and 20 cm forward of robot
    centre.  Its small downward pitch was measured in the existing two-distance
    wall calibration.  Keeping this factory beside the algorithms prevents the
    detector, optical-flow estimator, and visualization from silently using
    different camera geometry.
    """

    return CameraModel(
        name="bottom",
        height_m=0.127,
        pitch_down_deg=3.6,
        yaw_deg=0.0,
        forward_offset_m=0.20,
        lateral_offset_m=0.0,
        hfov_deg=62.0,
        vfov_deg=48.0,
    )


@dataclass(frozen=True)
class GroundOdometryDelta:
    """Accumulated robot-centre translation in the continuous IMU frame.

    Navigation consumes this immutable value atomically.  ``samples`` and the
    feature statistics are carried with the displacement so logs can distinguish
    a strong visual prior from a small or weak one without reaching into the
    estimator's internal state.
    """

    forward_imu_m: float
    left_imu_m: float
    samples: int
    support: int
    residual_m: float


@dataclass(frozen=True)
class GroundFlowEstimate:
    """One frame-pair translation expressed in the previous robot frame.

    This is the local result before it is rotated into the continuous IMU frame.
    Keeping the intermediate representation explicit makes the yaw/translation
    fusion auditable and unit-testable.
    """

    forward_m: float
    left_m: float
    support: int
    tracked: int
    residual_m: float


def floor_points_robot_frame(
    pixels_xy: np.ndarray,
    image_shape: tuple[int, ...],
    model: CameraModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project image pixels onto the floor in robot-centre coordinates.

    Each pixel is converted to a vertical depression angle and horizontal
    bearing using the calibrated camera model.  A ray/ground-plane intersection
    then gives ``(forward, left)`` metres from robot centre.  Pixels at or above
    the horizon cannot see the floor, and implausibly near or distant
    intersections are marked invalid instead of being allowed to create a
    translation or obstacle measurement.
    """
    pixels = np.asarray(pixels_xy, dtype=np.float64).reshape((-1, 2))
    height, width = int(image_shape[0]), int(image_shape[1])
    x_ratio = pixels[:, 0] / max(1.0, float(width - 1))
    y_ratio = pixels[:, 1] / max(1.0, float(height - 1))
    depression_deg = float(model.pitch_down_deg) + (y_ratio - 0.5) * float(model.vfov_deg)
    valid = depression_deg > 1.0
    distance = np.full(len(pixels), np.nan, dtype=np.float64)
    distance[valid] = float(model.height_m) / np.tan(np.radians(depression_deg[valid]))
    valid &= np.isfinite(distance) & (distance >= 0.06) & (distance <= 2.5)
    bearing_deg = float(model.bearing_sign) * (float(model.yaw_deg) - (x_ratio - 0.5) * float(model.hfov_deg))
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
    Each feature therefore votes for the same robot-centre translation.  The
    camera's offset from robot centre is already included by the projection, so
    pure rotation does not masquerade as translation.  A median/MAD consensus
    rejects moving objects, carpet shimmer, and mistracked pixels.  The function
    returns ``None`` instead of guessing whenever support or residual quality is
    insufficient; LiDAR and IMU continue without a camera prior in that case.
    """
    previous = np.asarray(previous_pixels_xy, dtype=np.float64).reshape((-1, 2))
    current = np.asarray(current_pixels_xy, dtype=np.float64).reshape((-1, 2))
    if len(previous) != len(current) or len(previous) < int(minimum_support):
        return None
    p_previous, valid_previous = floor_points_robot_frame(previous, image_shape, model)
    p_current, valid_current = floor_points_robot_frame(current, image_shape, model)
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
    final_residual = float(np.median(np.hypot(*(votes[inliers] - translation).T)))
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
    """Track floor texture and estimate one short-baseline body translation.

    Shi-Tomasi features are selected only in the floor-visible part of the
    image.  Pyramidal Lucas-Kanade flow tracks them forward and backward; tracks
    that do not return to their starting pixel are rejected as mismatches.  The
    remaining correspondences are converted into metres by
    ``estimate_ground_translation_from_tracks``.  A failed estimate returns
    ``None`` and therefore contributes no motion to localization.
    """
    previous = np.asarray(previous_gray)
    current = np.asarray(current_gray)
    if previous.shape != current.shape or previous.ndim != 2 or previous.size == 0:
        return None
    height, width = previous.shape
    mask = np.zeros_like(previous, dtype=np.uint8)
    horizon = int(np.clip((model.horizon_y_ratio() + 0.035) * height, 0, height - 1))
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
    """Continuously accumulate validated bottom-camera translation samples.

    A daemon thread consumes each new bottom frame exactly once, obtains the
    matching IMU yaw, and estimates camera motion over the short frame-to-frame
    baseline.  Accepted body-frame translations are rotated into the continuous
    IMU frame and accumulated until navigation calls ``take_delta``.  This is a
    prediction for the next LiDAR match, never an independent global pose and
    never permission to move the robot.
    """

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
        """Configure the frame/yaw sources and conservative acceptance limits.

        Construction performs no I/O and starts no thread.  A caller can thus
        assemble the complete sensor stack, verify the bottom stream, and only
        then call ``start``.  The optional model supports calibration tests while
        production defaults to the one canonical Sourccey bottom-camera model.
        """

        self.camera_source = camera_source
        self.yaw_source = yaw_source
        self.camera_key = str(camera_key)
        self.model = model or default_bottom_camera_model()
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
        """Start the non-blocking optical-flow worker exactly once.

        Camera work is intentionally kept off the navigation/control thread so
        frame decoding and feature tracking cannot delay a safety stop or base
        command.  Repeated calls are harmless.
        """

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
        """Request worker shutdown and wait briefly for an orderly exit.

        Joining the daemon prevents it from continuing to read camera frames
        after the SLAM application has begun closing its transports.
        """

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
        """Store a fresh reference frame after startup or a discontinuity.

        Reseeding deliberately creates no motion estimate: optical flow needs a
        real before/after pair, and treating the first frame as displacement
        would inject arbitrary motion into the LiDAR seed.
        """

        self._previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._previous_frame_identity = frame_identity
        self._previous_yaw = float(yaw)
        self._previous_at = float(now)

    def _run(self) -> None:
        """Consume unique frames, validate flow, and accumulate IMU-frame motion.

        Stale frames, repeated sequence numbers, missing yaw, weak optical flow,
        and physically implausible jumps are skipped.  Every accepted translation
        is rotated by the yaw at the beginning of its frame interval, producing a
        continuous-frame displacement that can span turns without using wheel
        commands as fake odometry.
        """

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
            translation = np.asarray([estimate.forward_m, estimate.left_m], dtype=np.float64)
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
        """Atomically return and clear all validated motion accumulated so far.

        The take-and-clear contract ensures a camera displacement is applied to
        exactly one LiDAR prediction.  If no pair was accepted, ``None`` tells the
        caller to keep its existing pose seed.
        """

        with self._lock:
            if self._pending_samples <= 0:
                return None
            result = GroundOdometryDelta(
                forward_imu_m=float(self._pending[0]),
                left_imu_m=float(self._pending[1]),
                samples=int(self._pending_samples),
                support=int(self._pending_support),
                residual_m=float(self._pending_residual_sum / max(1, self._pending_samples)),
            )
            self._pending[:] = 0.0
            self._pending_samples = 0
            self._pending_support = 0
            self._pending_residual_sum = 0.0
        return result

    def discard(self) -> None:
        """Discard translation accumulated across a known pose discontinuity.

        Navigation calls this after pivots, stationary relocalization, and global
        pose resets.  Those events establish a new pose authority, so replaying
        older camera translation afterward would double-count motion.
        """
        with self._lock:
            self._pending[:] = 0.0
            self._pending_samples = 0
            self._pending_support = 0
            self._pending_residual_sum = 0.0

    @property
    def healthy(self) -> bool:
        """Report whether at least one frame pair has passed every quality gate."""

        with self._lock:
            return self._accepted_total > 0

    @property
    def totals(self) -> tuple[int, int]:
        """Return lifetime accepted/rejected pair counts for diagnostics."""

        with self._lock:
            return int(self._accepted_total), int(self._rejected_total)


# ---------------------------------------------------------------------------
# Low-obstacle detection and visualization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloorObstacle:
    """One bottom-camera observation of a possible floor obstacle.

    ``distance_m`` and ``bearing_deg`` locate the object's floor contact in the
    robot frame.  ``crosses_horizon`` is the conservative height classifier used
    by the safety gate: an edge that continues above the camera-height horizon
    cannot be a flat cable or carpet seam and is tall enough to matter to the
    chassis.
    """

    bearing_deg: float
    distance_m: float
    x_ratio: float
    y_ratio: float
    height_m: float = 0.0
    crosses_horizon: bool = False


def detect_bottom_floor_obstacles(
    frame_bgr: np.ndarray,
    model: CameraModel,
    settings: BottomSafetySettings,
) -> list[FloorObstacle]:
    """Find low obstacle bases in one bottom-camera frame.

    The frame is blurred and edge-detected, then divided into vertical bands.
    Within each band the lowest strong edge is interpreted as the nearest floor
    contact, whose image row gives metric distance through the camera model.  A
    contiguous edge run upward estimates height and checks whether the silhouette
    crosses the camera-height horizon.  This deliberately ignores isolated floor
    texture, cables, and carpet seams while still reporting table legs, wheels,
    walls, and other objects beneath the LiDAR plane.

    The function is stateless and returns an empty list for malformed frames or
    when no candidate survives the geometric and height checks.  Hysteresis and
    stop decisions remain the responsibility of the safety monitor.
    """

    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[0] < 8 or frame.shape[1] < 8:
        return []
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 130)
    horizon = float(np.clip(model.horizon_y_ratio(), 0.05, 0.9))
    y_start = int(height * min(horizon + 0.04, 0.95))
    if y_start >= height - 2:
        return []

    obstacles: list[FloorObstacle] = []
    bands = max(int(settings.ground_column_bands), 4)
    band_width = max(width // bands, 4)
    minimum_edges = max(
        int(band_width * float(settings.ground_edge_density)),
        2,
    )
    minimum_height_m = float(settings.ground_min_obstacle_height_m)
    for band in range(bands):
        x0 = band * band_width
        x1 = min(x0 + band_width, width)
        column = edges[y_start:height, x0:x1]
        row_counts = (column > 0).sum(axis=1)
        strong_rows = np.nonzero(row_counts >= minimum_edges)[0]
        if len(strong_rows) == 0:
            continue

        base_index = int(strong_rows[-1])
        base_row = y_start + base_index
        y_ratio = base_row / max(height - 1, 1)
        distance = model.floor_distance_for_row(y_ratio)
        if distance is None or distance > float(settings.ground_max_distance_m):
            continue

        top_index = base_index
        gap = 0
        scan = base_index - 1
        while scan >= 0 and gap <= 4:
            if row_counts[scan] >= 1:
                top_index = scan
                gap = 0
            else:
                gap += 1
            scan -= 1
        top_row = y_start + top_index
        top_ratio = top_row / max(height - 1, 1)
        top_depression_deg = model.depression_deg_for_row(top_ratio)
        obstacle_height_m = float(model.height_m) - float(distance) * math.tan(
            math.radians(top_depression_deg)
        )
        if obstacle_height_m < minimum_height_m:
            continue

        crosses_horizon = False
        if top_index <= 1:
            above_y0 = max(0, int(horizon * height) - int(0.08 * height))
            above_band = edges[
                above_y0 : max(above_y0 + 1, y_start - 2),
                x0:x1,
            ]
            crosses_horizon = bool(above_band.size > 0 and (above_band > 0).any(axis=1).sum() >= 2)

        x_ratio = ((x0 + x1) * 0.5) / max(width - 1, 1)
        obstacles.append(
            FloorObstacle(
                bearing_deg=float(model.bearing_deg_for_column(x_ratio)),
                distance_m=float(distance),
                x_ratio=float(x_ratio),
                y_ratio=float(y_ratio),
                height_m=float(obstacle_height_m),
                crosses_horizon=crosses_horizon,
            )
        )
    return obstacles


class BottomFloorDetector:
    """Compatibility adapter used by the multi-sensor safety monitor.

    The detector owns no temporal state.  It simply remembers the shared camera
    model and settings, then delegates each frame to the public functional API.
    Keeping this tiny adapter preserves the monitor's existing call site while
    leaving all bottom-camera interpretation in this module.
    """

    def __init__(
        self,
        model: CameraModel,
        settings: BottomSafetySettings,
    ) -> None:
        """Bind one camera model and safety configuration for repeated frames."""

        self.model = model
        self.settings = settings

    def detect(self, frame_bgr: np.ndarray) -> list[FloorObstacle]:
        """Analyze one frame and return metric floor-obstacle observations."""

        return detect_bottom_floor_obstacles(frame_bgr, self.model, self.settings)


def bottom_obstacle_explains_floor_point(
    obstacles: list[FloorObstacle],
    bearing_deg: float,
    distance_m: float,
    settings: BottomSafetySettings,
) -> bool:
    """Return whether bottom-camera evidence supports a proposed floor point.

    The elevated-eye pipeline sometimes sees the top of a low object.  Before
    treating that edge as a floating tabletop, it projects the edge onto the
    floor and asks this function whether a bottom-camera candidate occupies the
    same bearing/range neighborhood.  Angular wraparound is handled explicitly.
    A positive result classifies the eye observation as a low floor object; it
    does not itself stop or move the robot.
    """

    for obstacle in obstacles:
        bearing_delta = abs(((obstacle.bearing_deg - float(bearing_deg)) + 180.0) % 360.0 - 180.0)
        if bearing_delta > float(settings.bottom_bearing_tolerance_deg):
            continue
        if abs(obstacle.distance_m - float(distance_m)) <= float(settings.bottom_distance_tolerance_m):
            return True
    return False


@dataclass(frozen=True)
class BottomGroundGateDecision:
    """The complete low-obstacle decision exported to the safety monitor.

    Separating this result from the larger three-sensor ``HazardState`` makes it
    obvious which fields are authored by the bottom camera and allows the gate
    to be tested without constructing the eye/depth pipeline.
    """

    active: bool = False
    distance_m: float | None = None
    side: str = "none"


class BottomGroundSafetyGate:
    """Convert frame-by-frame floor candidates into a stable safety decision.

    Only tall candidates inside the forward body corridor and stop range count
    as hits.  Several consecutive hit frames are required to activate the gate,
    and several consecutive clear frames are required to release it.  This
    hysteresis prevents one noisy image from stopping the base and prevents a
    flickering edge from repeatedly releasing it.  If the camera becomes stale
    or the gate is disabled, any old bottom-camera stop is cleared rather than
    latched forever.
    """

    def __init__(self, settings: BottomSafetySettings) -> None:
        """Bind the safety thresholds and initialize an inactive gate."""

        self.settings = settings
        self._hit_frames = 0
        self._clear_frames = 0
        self._decision = BottomGroundGateDecision()

    def update(
        self,
        obstacles: list[FloorObstacle],
        *,
        camera_available: bool,
    ) -> BottomGroundGateDecision:
        """Advance the hysteresis state with observations from one camera frame.

        The nearest qualifying obstacle determines range and side once the trip
        count is reached.  A currently active decision remains active through a
        short detector dropout and clears only after the configured clean-frame
        count.  The returned immutable object is the sole bottom-camera safety
        state consumed by the multi-sensor monitor.
        """

        if not camera_available or not bool(self.settings.ground_gate_enabled):
            self._hit_frames = 0
            self._clear_frames = 0
            self._decision = BottomGroundGateDecision()
            return self._decision

        in_corridor = [
            obstacle
            for obstacle in obstacles
            if obstacle.crosses_horizon
            and obstacle.distance_m <= float(self.settings.ground_stop_distance_m)
            and abs(obstacle.distance_m * math.sin(math.radians(obstacle.bearing_deg)))
            <= float(self.settings.ground_corridor_half_width_m)
        ]
        if in_corridor:
            self._hit_frames += 1
            self._clear_frames = 0
        else:
            self._hit_frames = 0

        if self._hit_frames >= max(int(self.settings.ground_trip_frames), 1):
            nearest = min(in_corridor, key=lambda obstacle: obstacle.distance_m)
            side = (
                "front"
                if abs(nearest.bearing_deg) <= 8.0
                else ("left" if nearest.bearing_deg > 0 else "right")
            )
            self._decision = BottomGroundGateDecision(
                active=True,
                distance_m=float(nearest.distance_m),
                side=side,
            )
        elif self._decision.active and not in_corridor:
            self._clear_frames += 1
            if self._clear_frames >= max(int(self.settings.ground_clear_frames), 1):
                self._decision = BottomGroundGateDecision()
                self._clear_frames = 0
        elif self._decision.active:
            self._clear_frames = 0
        return self._decision


def render_bottom_overlay(
    frame_bgr: np.ndarray,
    obstacles: list[FloorObstacle],
    state: BottomHazardState,
    settings: BottomSafetySettings,
) -> np.ndarray:
    """Draw a diagnostic explanation of bottom-camera perception.

    The calibrated horizon is drawn in cyan; each detected floor contact is
    labelled with metric range and whether its silhouette is tall enough to
    cross the horizon.  Red denotes a tall candidate inside the stop distance,
    yellow denotes a tall but farther candidate, and gray denotes a flat feature.
    The current hysteretic ground-gate decision is written at the top.  This
    function only copies and annotates pixels—it cannot activate or clear safety.
    """

    canvas = frame_bgr.copy()
    height, width = canvas.shape[:2]
    horizon_row = int(np.clip(settings.bottom_model.horizon_y_ratio(), 0.02, 0.98) * height)
    cv2.line(canvas, (0, horizon_row), (width, horizon_row), (255, 200, 0), 1)
    for obstacle in obstacles:
        x = int(obstacle.x_ratio * (width - 1))
        y = int(obstacle.y_ratio * (height - 1))
        near = obstacle.crosses_horizon and obstacle.distance_m <= float(settings.ground_stop_distance_m)
        color = (0, 0, 255) if near else ((0, 220, 220) if obstacle.crosses_horizon else (160, 160, 160))
        cv2.circle(canvas, (x, y), 4, color, -1)
        tall_tag = "X" if obstacle.crosses_horizon else "flat"
        cv2.putText(
            canvas,
            f"{obstacle.distance_m:.2f} {tall_tag}",
            (max(x - 24, 0), max(y - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    label = (
        f"bottom ground_stop {state.ground_side} {state.ground_distance_m:.2f}m"
        if state.ground_active and state.ground_distance_m is not None
        else "bottom clear"
    )
    color = (0, 0, 255) if state.ground_active else (0, 200, 0)
    cv2.putText(
        canvas,
        label,
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        label,
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )
    return canvas
