"""Two-eye panorama: calibrate Sourccey's angled eye cameras against each
other and fuse them into one virtual forward-facing view.

The eyes are yawed ~15deg outward with an 82deg hfov, so their views share a
~52deg binocular cone dead ahead. The lenses sit only 0.038m apart, so beyond
~0.5m the two views differ by (almost) pure rotation: each eye maps onto a
virtual forward camera with a homography — no depth needed. What was never
measured is each eye's ROLL and the true hfov; this tool solves them
photometrically: warp both eyes onto the virtual camera and search the
calibration parameters until the overlap band agrees.

Workflow (all offline data lives in artifacts/eye_panorama/):

1. CAPTURE synchronized eye pairs (aim at scenes >= 1.5m away — closer scenes
   violate the rotation-only assumption). By default the robot rotates itself
   ~12deg in place between pairs for scene variety (open loop, base only,
   never the arms); pass --rotate-deg 0 to keep it passive and turn by hand:
       uv run python scripts/sourccey_eye_panorama.py --mode capture \
           --remote-ip 192.168.1.237 --pairs 10 --interval 2.5

2. CALIBRATE (coarse grid over hfov + rolls incl. a yaw-sign hypothesis
   check, then coordinate-descent refinement of 7 parameters; writes
   calibration.json and a stitched preview per pair):
       uv run python scripts/sourccey_eye_panorama.py --mode calibrate

3. PREVIEW the live fused feed. Works before calibration too (streams with
   default geometry). On headless OpenCV builds it auto-serves a
   self-refreshing browser page instead of a GUI window:
       uv run python scripts/sourccey_eye_panorama.py --mode preview \
           --remote-ip 192.168.1.237
   Then open the printed live_panorama.html URL. Force the sink with
   --sink browser (headless) or --sink window (GUI OpenCV).

4. SERVE the fused view to an external consumer (e.g. a VR teleop rig that
   should show the fused forward view where it used to show the right eye):
       uv run python scripts/sourccey_eye_panorama.py --mode serve \
           --remote-ip 192.168.1.237 --serve-port 8091
   Point the right-eye source at  http://<robot-host>:8091/  (MJPEG). Or,
   from Python, use PanoramaStreamer for a pull API shaped like a camera:
       s = PanoramaStreamer("192.168.1.237").start()
       jpeg = s.latest_jpeg()   # bytes; swap in wherever you grab front_right
   or the generator  fused_vision_frames("192.168.1.237")  ->  (bgr, stamp).

Pure math (intrinsics, rotations, warp grids, scoring, solver) is torch-free
and unit-testable. NEVER touches the robot base or arms: passive listener.

Conventions (self-contained, standard CV): camera frame x=right, y=down,
z=forward. yaw is positive LOOKING LEFT (ccw from above), pitch positive
looking down, roll positive rotating the image ccw. front_left is expected
to look left (+yaw); if the mounting is actually crossed the calibration's
yaw-sign hypothesis pass detects it and says so.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Physical base values: SOURCCEY_PHYSICAL_SPECS.md / sourccey_camera_geometry.
BASE_PITCH_DOWN_DEG = 13.9  # field-calibrated 2026-07-09
BASE_YAW_OUT_DEG = 15.0  # operator-measured
BASE_HFOV_DEG = 82.0  # ASSUMED (4:3 aspect from calibrated vfov=66) — solved here
EYE_BASELINE_M = 0.038  # lens separation; parallax floor for the warp model

DEFAULT_OUTPUT_DIR = Path("artifacts") / "eye_panorama"


# ---------------------------------------------------------------------------
# Pure math
# ---------------------------------------------------------------------------
def intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Pinhole K for a centered principal point; fy = fx (square pixels —
    the calibrated vfov 66 with 4:3 240px agrees with hfov 82 over 320px to
    within a percent, so one focal serves both axes and hfov is the single
    field-of-view parameter the solver adjusts)."""
    fx = (width / 2.0) / math.tan(math.radians(float(hfov_deg)) / 2.0)
    return np.array(
        [[fx, 0.0, (width - 1) / 2.0], [0.0, fx, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotation_cam_to_ref(yaw_deg: float, pitch_down_deg: float, roll_deg: float) -> np.ndarray:
    """Camera->reference rotation. Reference frame: x right, y down, z level
    forward. Applied yaw (about ref up = -y), then pitch down, then roll
    about the camera's own forward axis."""
    y = math.radians(float(yaw_deg))
    p = math.radians(float(pitch_down_deg))
    r = math.radians(float(roll_deg))
    # yaw: positive = looking left => forward z maps to [-sin(y), 0, cos(y)]
    r_yaw = np.array(
        [[math.cos(y), 0.0, -math.sin(y)], [0.0, 1.0, 0.0], [math.sin(y), 0.0, math.cos(y)]],
        dtype=np.float64,
    )
    # pitch down: forward z maps to [0, sin(p), cos(p)] (y is down)
    r_pitch = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(p), -math.sin(p)], [0.0, math.sin(p), math.cos(p)]],
        dtype=np.float64,
    )
    # roll about camera z
    r_roll = np.array(
        [[math.cos(r), -math.sin(r), 0.0], [math.sin(r), math.cos(r), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return r_yaw @ r_pitch @ r_roll


@dataclass
class EyeCalibration:
    """Solved panorama calibration. Angles in degrees."""

    hfov_deg: float = BASE_HFOV_DEG
    yaw_left_deg: float = BASE_YAW_OUT_DEG
    yaw_right_deg: float = -BASE_YAW_OUT_DEG
    pitch_left_deg: float = BASE_PITCH_DOWN_DEG
    pitch_right_deg: float = BASE_PITCH_DOWN_DEG
    roll_left_deg: float = 0.0
    roll_right_deg: float = 0.0
    score: float = 0.0
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "schema": "sourccey.eye_panorama_calibration.v1",
            "hfov_deg": self.hfov_deg,
            "yaw_left_deg": self.yaw_left_deg,
            "yaw_right_deg": self.yaw_right_deg,
            "pitch_left_deg": self.pitch_left_deg,
            "pitch_right_deg": self.pitch_right_deg,
            "roll_left_deg": self.roll_left_deg,
            "roll_right_deg": self.roll_right_deg,
            "score": self.score,
            "meta": self.meta,
        }

    @staticmethod
    def from_json(data: dict) -> "EyeCalibration":
        return EyeCalibration(
            hfov_deg=float(data["hfov_deg"]),
            yaw_left_deg=float(data["yaw_left_deg"]),
            yaw_right_deg=float(data["yaw_right_deg"]),
            pitch_left_deg=float(data["pitch_left_deg"]),
            pitch_right_deg=float(data["pitch_right_deg"]),
            roll_left_deg=float(data["roll_left_deg"]),
            roll_right_deg=float(data["roll_right_deg"]),
            score=float(data.get("score", 0.0)),
            meta=dict(data.get("meta", {})),
        )


@dataclass(frozen=True)
class VirtualCamera:
    """The fused forward view: yaw 0, the eyes' shared pitch, roll 0."""

    width: int
    height: int
    hfov_deg: float
    pitch_down_deg: float = BASE_PITCH_DOWN_DEG

    def k(self) -> np.ndarray:
        return intrinsics(self.width, self.height, self.hfov_deg)

    def rotation(self) -> np.ndarray:
        return rotation_cam_to_ref(0.0, self.pitch_down_deg, 0.0)


def default_virtual_camera(
    eye_width: int,
    eye_height: int,
    hfov_deg: float,
    pitch_down_deg: float = BASE_PITCH_DOWN_DEG,
    yaw_out_deg: float = BASE_YAW_OUT_DEG,
) -> VirtualCamera:
    """Panorama sized to cover both eyes at native angular resolution.
    Coverage is +-(yaw + hfov/2); a pinhole is fine below 90deg half-angle
    but pixels stretch as tan(), so cap the half-angle at 62deg — beyond that
    the outer wings stretch too hard to be useful."""
    half_cover_deg = min(float(yaw_out_deg) + float(hfov_deg) / 2.0, 62.0)
    fx = (eye_width / 2.0) / math.tan(math.radians(float(hfov_deg)) / 2.0)
    width = int(round(2.0 * fx * math.tan(math.radians(half_cover_deg)))) & ~1
    return VirtualCamera(
        width=width,
        height=int(eye_height * 1.3) & ~1,  # roll/pitch slack at the top/bottom
        hfov_deg=2.0 * half_cover_deg,
        pitch_down_deg=float(pitch_down_deg),
    )


def warp_grid(
    virt: VirtualCamera,
    eye_k: np.ndarray,
    eye_rotation: np.ndarray,
    eye_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(map_x, map_y, valid_mask) mapping each virtual pixel to eye-image
    coordinates (cv2.remap convention). Rotation-only model — exact for
    distant scenes given the 0.038m baseline."""
    h, w = int(virt.height), int(virt.width)
    kv_inv = np.linalg.inv(virt.k())
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    rays = np.stack([u.ravel(), v.ravel(), np.ones(h * w)], axis=0)
    rays = kv_inv @ rays
    rays = virt.rotation() @ rays  # virtual cam -> reference
    rays = eye_rotation.T @ rays  # reference -> eye cam
    z = rays[2]
    valid = z > 1e-6
    z_safe = np.where(valid, z, 1.0)
    px = eye_k @ (rays / z_safe)
    map_x = px[0].reshape(h, w).astype(np.float32)
    map_y = px[1].reshape(h, w).astype(np.float32)
    eh, ew = int(eye_shape[0]), int(eye_shape[1])
    # Half-pixel tolerance: exact-identity mappings land at -1e-7 on the
    # border rows from float rounding and must not be masked out.
    inside = (
        valid.reshape(h, w)
        & (map_x >= -0.5)
        & (map_x <= ew - 0.5)
        & (map_y >= -0.5)
        & (map_y <= eh - 0.5)
    )
    return map_x, map_y, inside


def _eye_rotations(cal: EyeCalibration) -> tuple[np.ndarray, np.ndarray]:
    left = rotation_cam_to_ref(cal.yaw_left_deg, cal.pitch_left_deg, cal.roll_left_deg)
    right = rotation_cam_to_ref(cal.yaw_right_deg, cal.pitch_right_deg, cal.roll_right_deg)
    return left, right


def warp_pair(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    cal: EyeCalibration,
    virt: VirtualCamera | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, VirtualCamera]:
    """Warp both eyes onto the virtual camera. Returns (left_warp,
    right_warp, left_mask, right_mask, virt)."""
    eh, ew = left_bgr.shape[:2]
    if virt is None:
        # The virtual camera shares the eyes' common pitch so the panorama is
        # framed naturally (no spurious vertical warp); only relative eye
        # rotation drives the seam, so this choice is cosmetic but nicer.
        common_pitch = 0.5 * (float(cal.pitch_left_deg) + float(cal.pitch_right_deg))
        yaw_out = 0.5 * (abs(float(cal.yaw_left_deg)) + abs(float(cal.yaw_right_deg)))
        virt = default_virtual_camera(
            ew, eh, cal.hfov_deg, pitch_down_deg=common_pitch, yaw_out_deg=yaw_out
        )
    eye_k = intrinsics(ew, eh, cal.hfov_deg)
    rot_l, rot_r = _eye_rotations(cal)
    out = []
    for frame, rot in ((left_bgr, rot_l), (right_bgr, rot_r)):
        map_x, map_y, mask = warp_grid(virt, eye_k, rot, (eh, ew))
        warped = cv2.remap(
            frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=0
        )
        out.append((warped, mask))
    return out[0][0], out[1][0], out[0][1], out[1][1], virt


def overlap_score(
    left_warp: np.ndarray,
    right_warp: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> float:
    """Zero-mean NCC of gradient magnitudes over the shared overlap. Gradient
    space is robust to per-camera exposure/white-balance differences; NCC in
    [-1, 1], higher = better registration."""
    both = left_mask & right_mask
    if np.count_nonzero(both) < 500:
        return -1.0
    feats = []
    for frame in (left_warp, right_warp):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        feats.append(np.hypot(gx, gy)[both])
    a, b = feats
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return -1.0
    return float(np.dot(a, b) / denom)


def score_calibration(pairs: list[tuple[np.ndarray, np.ndarray]], cal: EyeCalibration) -> float:
    """Mean overlap NCC across all captured pairs."""
    scores = []
    for left, right in pairs:
        lw, rw, lm, rm, _ = warp_pair(left, right, cal)
        scores.append(overlap_score(lw, rw, lm, rm))
    return float(np.mean(scores)) if scores else -1.0


def _residual_similarity(
    left_warp: np.ndarray, right_warp: np.ndarray, both: np.ndarray
) -> np.ndarray | None:
    """Feature-based residual rigid alignment of the overlap: the 2x3 affine
    (rotation + uniform scale + translation) that maps the right warp onto
    the left. Corrects the leftover GLOBAL rotation/shift the calibration
    couldn't remove — the kink where the table ledge steps across the seam.
    Returns None if too few matches or the estimate is implausibly large
    (bad match) so the caller can skip it."""
    band = (both.astype(np.uint8)) * 255
    orb = cv2.ORB_create(1500)
    gl = cv2.cvtColor(left_warp, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(right_warp, cv2.COLOR_BGR2GRAY)
    kl, dl = orb.detectAndCompute(gl, band)
    kr, dr = orb.detectAndCompute(gr, band)
    if dl is None or dr is None or len(kl) < 10 or len(kr) < 10:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(dr, dl)
    if len(matches) < 12:
        return None
    src = np.float32([kr[m.queryIdx].pt for m in matches])  # right
    dst = np.float32([kl[m.trainIdx].pt for m in matches])  # left
    m, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if m is None or inliers is None or int(inliers.sum()) < 10:
        return None
    # Sanity: this is a small RESIDUAL, not a full re-registration. Reject a
    # wild solution from spurious matches (rotation > 12deg, scale off > 12%,
    # translation > 60px) — better to fall back to flow-only than to smear.
    scale = float(np.hypot(m[0, 0], m[0, 1]))
    angle = abs(math.degrees(math.atan2(m[1, 0], m[0, 0])))
    shift = float(np.hypot(m[0, 2], m[1, 2]))
    if angle > 12.0 or abs(scale - 1.0) > 0.12 or shift > 60.0:
        return None
    return m


def refine_overlap(
    left_warp: np.ndarray,
    right_warp: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    *,
    max_flow_px: float = 14.0,
    feather_px: float = 28.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame alignment of the overlap so the shared content stops
    ghosting AND stops stepping across the seam. Two stages, left is anchor:

      1. GLOBAL residual — a feature-based rigid transform (rotation + scale
         + shift) applied to the WHOLE right warp. This removes the leftover
         rotation the calibration missed (the table-ledge kink); because it
         is global, the right wing rotates consistently with the overlap, so
         there is no boundary discontinuity.
      2. LOCAL parallax — dense optical flow, feathered to zero at the
         overlap boundary, for the residual per-pixel disagreement the 0.038m
         baseline and lens distortion still leave. Flow is clipped so a
         low-texture mismatch cannot tear the image.

    Returns (corrected_right_warp, corrected_right_mask) — the mask follows
    the global warp so blending stays correct. The left warp is unchanged."""
    both = left_mask & right_mask
    if int(np.count_nonzero(both)) < 800:
        return right_warp, right_mask
    h, w = right_mask.shape

    # -- stage 1: global residual rigid alignment -----------------------------
    m = _residual_similarity(left_warp, right_warp, both)
    if m is not None:
        right_warp = cv2.warpAffine(right_warp, m, (w, h), flags=cv2.INTER_LINEAR)
        right_mask = (
            cv2.warpAffine(right_mask.astype(np.uint8), m, (w, h), flags=cv2.INTER_NEAREST) > 0
        )
        both = left_mask & right_mask
        if int(np.count_nonzero(both)) < 800:
            return right_warp, right_mask

    # -- stage 2: local parallax flow -----------------------------------------
    gl = cv2.cvtColor(left_warp, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(right_warp, cv2.COLOR_BGR2GRAY)
    # Flow[y,x] = displacement d s.t. right[y,x] matches left[y+dy, x+dx].
    flow = cv2.calcOpticalFlowFarneback(gr, gl, None, 0.5, 4, 31, 5, 7, 1.5, 0)
    mag = np.linalg.norm(flow, axis=2)
    clip = np.minimum(1.0, float(max_flow_px) / np.maximum(mag, 1e-6))
    flow *= clip[..., None]
    # Feather weight: 0 outside the overlap and at its boundary, ramping to 1
    # over feather_px inward (distance transform of the overlap mask).
    dist = cv2.distanceTransform(both.astype(np.uint8), cv2.DIST_L2, 5)
    weight = np.clip(dist / max(float(feather_px), 1.0), 0.0, 1.0) * both.astype(np.float32)
    weight = cv2.GaussianBlur(weight, (0, 0), 3.0)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xs - weight * flow[..., 0]
    map_y = ys - weight * flow[..., 1]
    right_warp = cv2.remap(
        right_warp, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return right_warp, right_mask


class OverlapRefiner:
    """Temporally-STABLE overlap refinement for the LIVE stream.

    refine_overlap() re-estimates the global transform every call, so a live
    feed's solid right wing squiggles (the global transform moves the whole
    right image). The key fix: the global transform corrects the RELATIVE EYE
    GEOMETRY, which is FIXED HARDWARE — it is a constant, so re-solving it per
    frame is both unnecessary and the entire source of the wobble. Instead:

      - GLOBAL transform: averaged over a short warmup, then FROZEN. Once
        frozen the right wing is a fixed remap and cannot move again — the
        camera parts are locked. (Frames whose features fail during warmup
        are skipped; the freeze waits for enough good ones.)
      - LOCAL flow: the only genuinely per-frame effect is scene parallax in
        the overlap. It stays live but is feathered to the overlap, EMA-
        smoothed, AND soft-deadbanded so a STATIC scene produces ZERO
        correction (sub-pixel estimator noise never reaches the image). So the
        middle is rock-steady when nothing moves and only bends for real
        parallax.

    transform_smooth / flow_smooth in [0,1): higher = steadier but laggier.
    freeze_after: good warmup frames before the global transform locks."""

    def __init__(
        self,
        *,
        transform_smooth: float = 0.9,
        flow_smooth: float = 0.8,
        freeze_after: int = 40,
        flow_deadband_px: float = 1.0,
    ) -> None:
        self._t_smooth = float(transform_smooth)
        self._f_smooth = float(flow_smooth)
        self._freeze_after = int(freeze_after)
        self._deadband = float(flow_deadband_px)
        self._m_ema: np.ndarray | None = None  # [a, b, tx, ty], a=s*cos, b=s*sin
        self._m_frozen: np.ndarray | None = None
        self._good_count = 0
        self._flow_ema: np.ndarray | None = None

    @property
    def locked(self) -> bool:
        """True once the global eye-alignment transform has frozen."""
        return self._m_frozen is not None

    def refine(
        self,
        left_warp: np.ndarray,
        right_warp: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
        *,
        max_flow_px: float = 12.0,
        feather_px: float = 28.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        both = left_mask & right_mask
        if int(np.count_nonzero(both)) < 800:
            return right_warp, right_mask
        h, w = right_mask.shape

        # -- stage 1: global rigid transform — warm up, then FREEZE -----------
        if self._m_frozen is not None:
            m_use = self._m_frozen
        else:
            m = _residual_similarity(left_warp, right_warp, both)
            if m is not None:
                params = np.array([m[0, 0], m[1, 0], m[0, 2], m[1, 2]], dtype=np.float64)
                if self._m_ema is None:
                    self._m_ema = params
                else:
                    self._m_ema = self._t_smooth * self._m_ema + (1.0 - self._t_smooth) * params
                self._good_count += 1
                if self._good_count >= self._freeze_after:
                    # The relative eye geometry is constant; lock it in.
                    self._m_frozen = self._m_ema.copy()
            m_use = self._m_ema
        if m_use is not None:
            a, b, tx, ty = m_use
            m_s = np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64)
            right_warp = cv2.warpAffine(right_warp, m_s, (w, h), flags=cv2.INTER_LINEAR)
            right_mask = (
                cv2.warpAffine(right_mask.astype(np.uint8), m_s, (w, h), flags=cv2.INTER_NEAREST)
                > 0
            )
            both = left_mask & right_mask
            if int(np.count_nonzero(both)) < 800:
                return right_warp, right_mask

        # -- stage 2: feathered + deadbanded + EMA-smoothed local flow --------
        gl = cv2.cvtColor(left_warp, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(right_warp, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gr, gl, None, 0.5, 4, 31, 5, 7, 1.5, 0)
        mag = np.linalg.norm(flow, axis=2)
        # Soft deadband: shrink every displacement by `deadband` px (floored at
        # 0). Sub-pixel estimator noise on a static scene becomes exactly zero;
        # real parallax passes through, minus a small constant.
        shrunk = np.maximum(0.0, mag - self._deadband)
        scale = np.where(mag > 1e-6, np.minimum(shrunk, float(max_flow_px)) / np.maximum(mag, 1e-6), 0.0)
        flow *= scale[..., None]
        dist = cv2.distanceTransform(both.astype(np.uint8), cv2.DIST_L2, 5)
        weight = np.clip(dist / max(float(feather_px), 1.0), 0.0, 1.0) * both.astype(np.float32)
        weight = cv2.GaussianBlur(weight, (0, 0), 3.0)
        feathered = flow * weight[..., None]
        if self._flow_ema is None or self._flow_ema.shape != feathered.shape:
            self._flow_ema = feathered
        else:
            self._flow_ema = self._f_smooth * self._flow_ema + (1.0 - self._f_smooth) * feathered
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = xs - self._flow_ema[..., 0]
        map_y = ys - self._flow_ema[..., 1]
        right_warp = cv2.remap(
            right_warp, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        return right_warp, right_mask


def blend_panorama(
    left_warp: np.ndarray,
    right_warp: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> np.ndarray:
    """Feathered blend: each eye's weight fades linearly with distance from
    its own side, so the seam crosses the overlap smoothly."""
    h, w = left_mask.shape
    ramp = np.linspace(1.0, 0.0, w, dtype=np.float32)[None, :]
    w_l = np.where(left_mask, ramp, 0.0)
    w_r = np.where(right_mask, ramp[:, ::-1], 0.0)
    total = w_l + w_r
    total[total < 1e-6] = 1.0
    w_l, w_r = (w_l / total)[..., None], (w_r / total)[..., None]
    fused = left_warp.astype(np.float32) * w_l + right_warp.astype(np.float32) * w_r
    return fused.astype(np.uint8)


class PerceptionMosaic:
    """GEOMETRIC eye fusion for the safety/perception stack (not for humans).

    The display panorama (fuse_eyes) feather-blends and flow-bends the
    overlap — pretty, but blended ghosts and non-rigid bending are poison for
    a metric depth model. Perception instead gets a HARD-CUT mosaic: the left
    eye owns every column left of the panorama center (bearing 0), the right
    eye owns the rest, warped through the CALIBRATED rotations only. Every
    pixel therefore has an exact, analytic ray through the virtual forward
    camera — including the per-eye roll that the legacy per-eye CameraModels
    cannot even represent (which is why their bearings were off by up to
    ~6-8deg and edge placement by ~15cm at 1.5m).

    The single seam column can show a small step (residual ~1-3deg
    calibration error) and blended-black slivers at uncovered corners —
    `seam_cols` and `coverage()` let the perception layer forgive exactly
    those regions instead of hallucinating obstacles from them.

    Known approximation, documented on purpose: the stitch calibration solved
    under square pixels (fy=fx from cal.hfov) while the true vertical scale
    is the field-calibrated vfov=66deg. Both eyes share the error, so seam
    REGISTRATION is unaffected; metrically we interpret panorama rows with
    the TRUE vfov (66deg over the eye row count). With ±6deg rolls this
    leaves up to ~2-3deg of row error at the far frame edges — comparable to
    or better than the old per-eye stack, whose roll was not modeled at all.
    """

    def __init__(self, cal: EyeCalibration, eye_width: int = 320, eye_height: int = 240) -> None:
        from sourccey_camera_geometry import CameraModel

        self.cal = cal
        yaw_out = 0.5 * (abs(float(cal.yaw_left_deg)) + abs(float(cal.yaw_right_deg)))
        half_cover_deg = min(yaw_out + float(cal.hfov_deg) / 2.0, 62.0)
        fx = (eye_width / 2.0) / math.tan(math.radians(float(cal.hfov_deg)) / 2.0)
        width = int(round(2.0 * fx * math.tan(math.radians(half_cover_deg)))) & ~1
        # Same height as the eyes: taller framings sample rows the eyes never
        # covered (black), and the metric row scale below assumes eye rows.
        common_pitch = 0.5 * (float(cal.pitch_left_deg) + float(cal.pitch_right_deg))
        self.virt = VirtualCamera(
            width=width,
            height=int(eye_height),
            hfov_deg=2.0 * half_cover_deg,
            pitch_down_deg=common_pitch,
        )
        eye_k = intrinsics(eye_width, eye_height, cal.hfov_deg)
        rot_l, rot_r = _eye_rotations(cal)
        grids = []
        for rot in (rot_l, rot_r):
            grids.append(warp_grid(self.virt, eye_k, rot, (eye_height, eye_width)))
        (self._lx, self._ly, lm), (self._rx, self._ry, rm) = grids
        self.seam_col = width // 2
        half_band = 14
        self.seam_cols = (self.seam_col - half_band, self.seam_col + half_band)
        own_left = np.zeros((self.virt.height, width), dtype=bool)
        own_left[:, : self.seam_col] = True
        self._coverage = (lm & own_left) | (rm & ~own_left)
        self._own_left = own_left
        # The metric camera model for analyze_depth. Pitch is the FIELD-
        # CALIBRATED 13.9 (the stitch calibration's common pitch is gauge —
        # it orients the panorama, not the world); vfov is the TRUE vertical
        # scale (see class docstring); hfov matches the warp's horizontal
        # focal exactly. Position: eye height, midpoint between the lenses.
        self.model = CameraModel(
            name="panorama",
            height_m=0.914,
            pitch_down_deg=13.9,
            yaw_deg=0.0,
            forward_offset_m=0.203,
            lateral_offset_m=0.0,
            hfov_deg=self.virt.hfov_deg,
            vfov_deg=66.0,
        )

    def coverage(self) -> np.ndarray:
        """Boolean mask of panorama pixels actually covered by an eye."""
        return self._coverage

    def compose(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        """Two raw eye frames -> hard-cut perception mosaic (uncovered
        pixels are black; exclude them via coverage())."""
        lw = cv2.remap(left_bgr, self._lx, self._ly, interpolation=cv2.INTER_LINEAR, borderValue=0)
        rw = cv2.remap(right_bgr, self._rx, self._ry, interpolation=cv2.INTER_LINEAR, borderValue=0)
        mosaic = np.where(self._own_left[..., None], lw, rw)
        mosaic[~self._coverage] = 0
        return mosaic


def load_perception_mosaic(output_dir: Path | None = None) -> PerceptionMosaic:
    """The wander stack's entry point: build the calibrated perception
    mosaic, or fail LOUD if the eye calibration has not been run (project
    rule: no silent fallbacks)."""
    out = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    cal, is_calibrated = load_calibration_or_default(out)
    if not is_calibrated:
        raise RuntimeError(
            f"eye panorama calibration not found at {out / 'calibration.json'} — run "
            "scripts/sourccey_eye_panorama.py --mode capture then --mode calibrate "
            "(or pass --eye-fusion per-eye to use the legacy per-eye models)"
        )
    return PerceptionMosaic(cal)


def fuse_eyes(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    cal: EyeCalibration,
    virt: VirtualCamera | None = None,
    *,
    refine: bool = True,
    refiner: "OverlapRefiner | None" = None,
) -> tuple[np.ndarray, VirtualCamera]:
    """One call: two raw eye frames -> a single fused forward panorama.

    The reusable building block — warp both eyes onto the virtual forward
    camera with `cal`, optionally refine the overlap (align + de-ghost), then
    feather-blend. Pass the returned `virt` back in on the next frame to skip
    re-deriving the output size. Any consumer (a live viewer, or a future
    perception pass that wants one straight-ahead image instead of two angled
    ones) needs only this.

    refine=False gives the raw global-warp blend. For a LIVE feed, pass a
    persistent `refiner` (OverlapRefiner) so the correction is temporally
    smoothed — the solid wings stop squiggling; without it each frame is
    refined independently (fine for stills, jittery for video)."""
    lw, rw, lm, rm, virt = warp_pair(left_bgr, right_bgr, cal, virt)
    if refine:
        if refiner is not None:
            rw, rm = refiner.refine(lw, rw, lm, rm)
        else:
            rw, rm = refine_overlap(lw, rw, lm, rm)
    return blend_panorama(lw, rw, lm, rm), virt


def load_calibration_or_default(output_dir: Path) -> tuple[EyeCalibration, bool]:
    """Return (calibration, is_calibrated). Uses the saved solve when present,
    else the physical-spec defaults (roll 0, hfov 82) so the fused view is
    still viewable before any calibration has been run — the eyes are just
    not yet seam-matched."""
    cal_path = output_dir / "calibration.json"
    if cal_path.exists():
        return EyeCalibration.from_json(json.loads(cal_path.read_text(encoding="utf-8"))), True
    return EyeCalibration(), False


# ---------------------------------------------------------------------------
# Solver — FEATURE-BASED (ORB correspondences -> ray-coincidence fit)
# ---------------------------------------------------------------------------
# The photometric NCC search this replaced could not find the true geometry:
# the eyes turned out ~22deg-out (not the 15deg estimate) with ~20deg of
# relative roll, and the gradient-NCC landscape near the physical defaults was
# flat-zero, so the search never left the wrong basin (field 2026-07-15). ORB
# matches in the overlap give hard correspondences; a distant matched point
# projects to the SAME world ray from both eyes, so the calibration is the
# rotation set (+hfov) that makes matched rays coincide.


def _extract_correspondences(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    max_total: int = 3000,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """ORB-match every pair, RANSAC-homography filter to inliers, and stack
    the surviving (front_left_px, front_right_px) correspondences. Returns
    (ptsL, ptsR, n_pairs_used)."""
    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    left_pts: list[np.ndarray] = []
    right_pts: list[np.ndarray] = []
    used = 0
    for left, right in pairs:
        kl, dl = orb.detectAndCompute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), None)
        kr, dr = orb.detectAndCompute(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), None)
        if dl is None or dr is None:
            continue
        matches = bf.match(dl, dr)
        if len(matches) < 8:
            continue
        pl = np.float32([kl[m.queryIdx].pt for m in matches])
        pr = np.float32([kr[m.trainIdx].pt for m in matches])
        _h, inliers = cv2.findHomography(pl, pr, cv2.RANSAC, 3.0)
        if inliers is None:
            continue
        mask = inliers.ravel().astype(bool)
        if int(mask.sum()) < 6:
            continue
        left_pts.append(pl[mask])
        right_pts.append(pr[mask])
        used += 1
    if not left_pts:
        return None, None, 0
    all_l = np.vstack(left_pts)
    all_r = np.vstack(right_pts)
    if len(all_l) > max_total:
        idx = np.linspace(0, len(all_l) - 1, max_total).astype(int)
        all_l, all_r = all_l[idx], all_r[idx]
    return all_l, all_r, used


def _pixel_rays(k: np.ndarray, rotation: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Unit world-frame rays for pixel points, through intrinsics K and a
    camera->world rotation."""
    homog = np.hstack([pts, np.ones((len(pts), 1))])
    rays = (rotation @ np.linalg.inv(k) @ homog.T).T
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _correspondence_residual_deg(
    all_l: np.ndarray,
    all_r: np.ndarray,
    frame_shape: tuple[int, int],
    half_yaw: float,
    pitch: float,
    half_roll: float,
    hfov: float,
) -> np.ndarray:
    """Per-correspondence angle (deg) between the two eyes' rays under a
    SYMMETRIC calibration (yaw ±half_yaw, common pitch, roll ±half_roll).
    Zero when the matched features project to the same world ray."""
    eh, ew = int(frame_shape[0]), int(frame_shape[1])
    k = intrinsics(ew, eh, hfov)
    rot_l = rotation_cam_to_ref(+half_yaw, pitch, +half_roll)
    rot_r = rotation_cam_to_ref(-half_yaw, pitch, -half_roll)
    a = _pixel_rays(k, rot_l, all_l)
    b = _pixel_rays(k, rot_r, all_r)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1.0, 1.0)))


def solve_panorama_calibration(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    verbose: bool = True,
) -> EyeCalibration:
    """Feature-based calibration. Solves the SYMMETRIC rotation set
    (yaw ±Y, common pitch P, roll ±R) plus hfov that makes ORB
    correspondences' world rays coincide, by multi-start coordinate descent
    on the MEDIAN angular residual (robust to remaining match outliers).

    Symmetric because only the RELATIVE eye rotation affects the seam; any
    common yaw/pitch/roll just orients the whole panorama and is gauge. Falls
    back to physical-spec defaults (with a warning in .meta) if too few
    correspondences are found."""

    def log(msg: str) -> None:
        if verbose:
            print(f"[calibrate] {msg}")

    all_l, all_r, used = _extract_correspondences(pairs)
    if all_l is None or len(all_l) < 40:
        log(
            f"WARNING: only {0 if all_l is None else len(all_l)} correspondences "
            f"from {used} pairs — too few to calibrate. Keeping physical-spec "
            "defaults. Recapture with more texture/overlap in view."
        )
        cal = EyeCalibration()
        cal.meta = {"warning": "insufficient_correspondences", "correspondences": 0}
        return cal
    log(f"{len(all_l)} correspondences from {used} pairs")
    frame_shape = pairs[0][0].shape[:2]

    def median_resid(y: float, p: float, r: float, h: float) -> float:
        return float(
            np.median(_correspondence_residual_deg(all_l, all_r, frame_shape, y, p, r, h))
        )

    # Multi-start (the yaw/hfov trade-off has shallow local minima), then
    # coordinate descent with shrinking steps. Params: half_yaw, pitch,
    # half_roll, hfov.
    best_params = None
    best_cost = None
    for h0 in (70.0, 80.0, 90.0):
        for y0 in (12.0, 18.0, 24.0):
            params = [y0, BASE_PITCH_DOWN_DEG, 0.0, h0]
            steps = [4.0, 4.0, 4.0, 4.0]
            cost = median_resid(*params)
            for _ in range(80):
                improved = False
                for i in range(4):
                    for direction in (+1.0, -1.0):
                        trial = list(params)
                        trial[i] += direction * steps[i]
                        # Keep parameters physically sane.
                        trial[0] = float(np.clip(trial[0], 3.0, 45.0))  # half_yaw
                        trial[1] = float(np.clip(trial[1], -5.0, 35.0))  # pitch
                        trial[2] = float(np.clip(trial[2], -25.0, 25.0))  # half_roll
                        trial[3] = float(np.clip(trial[3], 55.0, 95.0))  # hfov
                        c = median_resid(*trial)
                        if c < cost - 1e-4:
                            params, cost = trial, c
                            improved = True
                if not improved:
                    steps = [s * 0.5 for s in steps]
                    if max(steps) < 0.05:
                        break
            if best_cost is None or cost < best_cost:
                best_cost, best_params = cost, params

    y, p, r, h = best_params
    resid = _correspondence_residual_deg(all_l, all_r, frame_shape, y, p, r, h)
    cal = EyeCalibration(
        hfov_deg=float(h),
        yaw_left_deg=float(y),
        yaw_right_deg=float(-y),
        pitch_left_deg=float(p),
        pitch_right_deg=float(p),
        roll_left_deg=float(r),
        roll_right_deg=float(-r),
    )
    # `score` stays the overlap NCC so the preview quality readout and older
    # tooling keep a single 0..1 number; the residual is the real solve metric.
    cal.score = score_calibration(pairs, cal)
    cal.meta = {
        "method": "feature_ray_coincidence",
        "correspondences": int(len(all_l)),
        "pairs_used": int(used),
        "median_residual_deg": round(float(np.median(resid)), 3),
        "p90_residual_deg": round(float(np.percentile(resid, 90)), 3),
    }
    log(
        f"solved (median ray residual {np.median(resid):.2f}deg): "
        f"half_yaw={y:.2f} pitch={p:.2f} half_roll={r:.2f} hfov={h:.2f}"
    )
    return cal


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def _load_pairs(capture_dir: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for left_path in sorted(capture_dir.glob("pair_*_front_left.png")):
        right_path = Path(str(left_path).replace("front_left", "front_right"))
        if not right_path.exists():
            continue
        left = cv2.imread(str(left_path))
        right = cv2.imread(str(right_path))
        if left is not None and right is not None and left.shape == right.shape:
            pairs.append((left, right))
    return pairs


# Open-loop rotation is imprecise, but the calibration only wants scene
# VARIETY between pairs, not tracked degrees. This matches the wander loop's
# observed cadence (~7.5deg per 0.24s burst at turn_speed 0.82) closely
# enough to convert a requested nudge into a burst count.
_DEG_PER_TURN_BURST = 7.5
_TURN_BURST_S = 0.24
_TURN_SETTLE_S = 0.25


def _connect_robot_for_rotation(args: argparse.Namespace):
    """Connect the base client ONLY to nudge the heading between pairs. Sends
    base velocity + the existing z.pos lift command exactly like the wander
    loop — never any arm command (hard project rule)."""
    from ldlidar_auto_snapshot_stitch import _send_stop
    from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
    from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient

    robot = SourcceyClient(SourcceyClientConfig(id=args.robot_id, remote_ip=args.remote_ip))
    robot.connect()
    _send_stop(robot)
    return robot


def _nudge_heading(robot, args: argparse.Namespace) -> None:
    """Rotate a small, approximate amount in place (open loop) so the next
    pair sees a different scene."""
    from ldlidar_auto_snapshot_stitch import _execute_turn_burst

    bursts = max(1, int(round(abs(float(args.rotate_deg)) / _DEG_PER_TURN_BURST)))
    direction = 1.0 if float(args.rotate_deg) >= 0.0 else -1.0
    print(
        f"[capture] rotating ~{args.rotate_deg:+.0f}deg between pairs "
        f"({bursts} burst(s) at speed {args.turn_speed:.2f})"
    )
    for _ in range(bursts):
        _execute_turn_burst(
            robot=robot,
            direction_sign=direction,
            turn_speed=float(args.turn_speed),
            turn_burst_s=_TURN_BURST_S,
            turn_settle_s=_TURN_SETTLE_S,
        )


def run_capture(args: argparse.Namespace, output_dir: Path) -> int:
    from sourccey_elevated_safety import SlamCameraSubscriber, endpoint_from_remote_ip

    capture_dir = output_dir / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    endpoint = endpoint_from_remote_ip(args.remote_ip)
    subscriber = SlamCameraSubscriber(endpoint=endpoint, camera_keys=("front_left", "front_right"))
    subscriber.start()
    robot = None
    rotate = abs(float(args.rotate_deg)) > 1e-3
    try:
        print(f"[capture] waiting for both eyes on {endpoint} ...")
        if not subscriber.wait_for_frames(10.0, required=("front_left", "front_right")):
            print("[capture] ERROR: no eye frames within 10s — is the host streaming?")
            return 1
        if rotate:
            print("[capture] connecting base client to rotate between pairs ...")
            robot = _connect_robot_for_rotation(args)
        variety_note = (
            f"the robot rotates ~{args.rotate_deg:+.0f}deg between pairs for scene variety"
            if rotate
            else "rotation disabled (--rotate-deg 0); vary the heading by hand between shots"
        )
        print(
            f"[capture] saving {args.pairs} pairs to {capture_dir}\n"
            "[capture] point the eyes at structure >= 1.5m away (closer scenes break the\n"
            f"[capture] rotation-only stitch model); {variety_note}"
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = 0
        while saved < int(args.pairs):
            time.sleep(max(0.2, float(args.interval)))
            left, age_l = subscriber.latest("front_left")
            right, age_r = subscriber.latest("front_right")
            if left is None or right is None:
                continue
            if (age_l or 0.0) > 1.0 or (age_r or 0.0) > 1.0:
                print("[capture] stale frames, retrying ...")
                continue
            saved += 1
            for name, frame in (("front_left", left), ("front_right", right)):
                cv2.imwrite(str(capture_dir / f"pair_{stamp}_{saved:03d}_{name}.png"), frame)
            print(f"[capture] pair {saved}/{args.pairs} saved (ages {age_l:.2f}/{age_r:.2f}s)")
            # Nudge AFTER the shot (not before the last one): the eyes must be
            # settled and static when a pair is grabbed, and there is no point
            # rotating past the final capture.
            if robot is not None and saved < int(args.pairs):
                _nudge_heading(robot, args)
        print(f"[capture] done -> {capture_dir}")
        return 0
    except KeyboardInterrupt:
        print("\n[capture] interrupted")
        return 0
    finally:
        subscriber.stop()
        if robot is not None:
            try:
                from ldlidar_auto_snapshot_stitch import _send_stop

                _send_stop(robot)
                robot.disconnect()
            except Exception as exc:
                print(f"[capture] warning: robot cleanup failed: {exc}")


def run_calibrate(args: argparse.Namespace, output_dir: Path) -> int:
    capture_dir = output_dir / "captures"
    pairs = _load_pairs(capture_dir)
    if len(pairs) < 3:
        print(
            f"[calibrate] ERROR: only {len(pairs)} usable pairs in {capture_dir} — "
            "run --mode capture first (>= 5 pairs recommended)"
        )
        return 1
    print(f"[calibrate] solving from {len(pairs)} pairs ...")
    started = time.monotonic()
    cal = solve_panorama_calibration(pairs)
    cal.meta = {
        **cal.meta,  # keep the solver's residual / correspondence diagnostics
        "solved_at": datetime.now().isoformat(timespec="seconds"),
        "pairs": len(pairs),
        "frame_shape": list(pairs[0][0].shape[:2]),
        "solve_seconds": round(time.monotonic() - started, 1),
    }
    cal_path = output_dir / "calibration.json"
    cal_path.write_text(json.dumps(cal.to_json(), indent=2), encoding="utf-8")
    resid = cal.meta.get("median_residual_deg")
    quality = (
        "no residual (fell back to defaults)"
        if resid is None
        else f"median ray residual {resid:.2f}deg "
        "(< 1.5 = crisp seam, > 3 = recapture with more overlap/texture)"
    )
    print(
        f"[calibrate] DONE in {cal.meta['solve_seconds']}s: {quality}; overlap NCC={cal.score:.3f}\n"
        f"[calibrate] hfov={cal.hfov_deg:.2f}deg "
        f"yaw=({cal.yaw_left_deg:+.2f},{cal.yaw_right_deg:+.2f}) "
        f"pitch=({cal.pitch_left_deg:.2f},{cal.pitch_right_deg:.2f}) "
        f"roll=({cal.roll_left_deg:+.2f},{cal.roll_right_deg:+.2f})\n"
        f"[calibrate] saved -> {cal_path}"
    )
    preview_dir = output_dir / "calibration_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for index, (left, right) in enumerate(pairs, start=1):
        lw, rw, lm, rm, _ = warp_pair(left, right, cal)
        rw_ref, rm_ref = refine_overlap(lw, rw, lm, rm)
        # The refined panorama is the deliverable; keep raw + refined overlap
        # ghosts side by side so the alignment gain is visible (crisp =
        # registered). The refined ghost should show a single, un-kinked
        # table ledge instead of the stepped double edge.
        cv2.imwrite(
            str(preview_dir / f"pair_{index:03d}_panorama.png"),
            blend_panorama(lw, rw_ref, lm, rm_ref),
        )
        cv2.imwrite(
            str(preview_dir / f"pair_{index:03d}_overlap_ghost_raw.png"),
            cv2.addWeighted(lw, 0.5, rw, 0.5, 0),
        )
        cv2.imwrite(
            str(preview_dir / f"pair_{index:03d}_overlap_ghost_refined.png"),
            cv2.addWeighted(lw, 0.5, rw_ref, 0.5, 0),
        )
    print(f"[calibrate] per-pair panoramas + raw/refined overlap ghosts -> {preview_dir}")
    return 0


_LIVE_HTML = """<!doctype html><meta charset=utf-8>
<title>Sourccey panorama (live)</title>
<style>body{{margin:0;background:#111;color:#ccc;font:14px sans-serif;text-align:center}}
img{{max-width:100vw;image-rendering:auto}}p{{margin:6px}}</style>
<p>Sourccey fused forward panorama — {label}. Auto-refreshing ~10fps.</p>
<img id=v src="live_panorama.jpg">
<script>setInterval(()=>{{document.getElementById('v').src='live_panorama.jpg?'+Date.now()}},100)</script>
"""


class PanoramaStreamer:
    """Live fused central-vision source for EXTERNAL consumers — e.g. a VR
    teleop rig that today reads the right eye (front_right) and should instead
    show the single fused forward view.

    Shaped like a camera source: a background thread keeps the newest fused
    frame ready; pull it whenever your pipeline currently grabs the right eye:

        streamer = PanoramaStreamer("192.168.1.237")
        streamer.start()
        ...
        jpeg = streamer.latest_jpeg()        # bytes, or None until warm
        # or: bgr = streamer.latest_bgr()
        ...
        streamer.stop()

    This serves the DISPLAY fusion (feather-blended, flow-de-ghosted, and
    temporally frozen via OverlapRefiner) — the human-facing view you tuned,
    NOT the hard-cut perception mosaic. Uncalibrated is allowed (streams with
    default geometry + a warning) since a teleop feed tolerates a soft seam;
    it does not fail loud like the safety path.

    Robustness for a live human feed: a momentary missing/stale eye HOLDS the
    last fused frame instead of blanking. `frame_age_s` lets the consumer
    decide when a hold has gone stale.
    """

    def __init__(
        self,
        remote_ip: str,
        *,
        output_dir: Path | None = None,
        calibration: EyeCalibration | None = None,
        refine: bool = True,
        output_size: tuple[int, int] | None = None,
        max_input_age_s: float = 1.0,
    ) -> None:
        self._remote_ip = str(remote_ip)
        out = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
        if calibration is not None:
            self._cal, self._is_cal = calibration, True
        else:
            self._cal, self._is_cal = load_calibration_or_default(out)
        self._refine = bool(refine)
        self._output_size = output_size
        self._max_input_age = float(max_input_age_s)
        self._refiner = OverlapRefiner() if refine else None
        self._virt: VirtualCamera | None = None
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_mono: float | None = None
        self._recent_stamps: deque = deque(maxlen=30)
        self._subscriber = None
        self._first_frame_announced = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_calibrated(self) -> bool:
        return self._is_cal

    @property
    def locked(self) -> bool:
        """True once the fusion's eye-alignment transform has frozen (stable)."""
        return self._refiner is not None and self._refiner.locked

    @property
    def frame_age_s(self) -> float | None:
        with self._lock:
            if self._frame_mono is None:
                return None
            return max(0.0, time.monotonic() - self._frame_mono)

    @property
    def fps(self) -> float:
        with self._lock:
            stamps = list(self._recent_stamps)
        if len(stamps) < 2:
            return 0.0
        span = stamps[-1] - stamps[0]
        return (len(stamps) - 1) / span if span > 1e-6 else 0.0

    def start(self) -> "PanoramaStreamer":
        if self._thread is not None:
            return self
        from sourccey_elevated_safety import SlamCameraSubscriber, endpoint_from_remote_ip

        if not self._is_cal:
            print(
                "[panorama-stream] WARNING: no calibration.json — streaming fused "
                "view with DEFAULT geometry (soft seam). Run --mode calibrate for a "
                "matched seam."
            )
        endpoint = endpoint_from_remote_ip(self._remote_ip)
        self._subscriber = SlamCameraSubscriber(
            endpoint=endpoint, camera_keys=("front_left", "front_right")
        )
        self._subscriber.start()
        self._thread = threading.Thread(
            target=self._run, name="panorama-stream", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._subscriber is not None:
            self._subscriber.stop()
            self._subscriber = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            left, age_l = self._subscriber.latest("front_left")
            right, age_r = self._subscriber.latest("front_right")
            fresh = (
                left is not None
                and right is not None
                and age_l is not None
                and age_r is not None
                and max(age_l, age_r) <= self._max_input_age
                and abs(float(age_l) - float(age_r)) <= 0.35
            )
            if not fresh:
                time.sleep(0.02)
                continue
            fused, self._virt = fuse_eyes(
                left, right, self._cal, self._virt, refine=self._refine, refiner=self._refiner
            )
            if self._output_size is not None:
                fused = cv2.resize(fused, self._output_size)
            stamp = time.monotonic() - float(max(age_l, age_r))
            with self._lock:
                self._frame = fused
                self._frame_mono = stamp
                self._recent_stamps.append(time.monotonic())
            if not self._first_frame_announced:
                self._first_frame_announced = True
                print("[panorama-stream] First fused stereo frame received; HTTP snapshot feed is live.")

    def latest_bgr(self, max_age_s: float | None = None) -> np.ndarray | None:
        """Newest fused BGR frame (a copy), or None if none yet / too stale."""
        with self._lock:
            if self._frame is None:
                return None
            if max_age_s is not None and self._frame_mono is not None:
                if time.monotonic() - self._frame_mono > float(max_age_s):
                    return None
            return self._frame.copy()

    def latest_jpeg(self, quality: int = 85, max_age_s: float | None = None) -> bytes | None:
        """Newest fused frame JPEG-encoded — the drop-in for a right-eye feed
        that ships encoded frames. None if none yet / too stale."""
        frame = self.latest_bgr(max_age_s=max_age_s)
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes() if ok else None


def fused_vision_frames(remote_ip: str, **kwargs):
    """Convenience generator: yield (fused_bgr, monotonic_stamp) forever.

    Wraps PanoramaStreamer for code that prefers a `for frame in ...:` loop.
    kwargs pass through to PanoramaStreamer. Stops cleanly on GeneratorExit."""
    streamer = PanoramaStreamer(remote_ip, **kwargs).start()
    last_stamp = None
    try:
        while True:
            with streamer._lock:
                frame = None if streamer._frame is None else streamer._frame.copy()
                stamp = streamer._frame_mono
            if frame is not None and stamp != last_stamp:
                last_stamp = stamp
                yield frame, stamp
            else:
                time.sleep(0.01)
    finally:
        streamer.stop()


def serve_mjpeg(
    remote_ip: str,
    *,
    host: str = "0.0.0.0",
    port: int = 8091,
    output_dir: Path | None = None,
    quality: int = 85,
    refine: bool = True,
) -> int:
    """Serve the fused central vision as an MJPEG-over-HTTP stream — the
    lowest-friction drop-in for a VR/browser teleop consumer that reads a URL.
    Point the right-eye source at  http://<robot-host-ip>:<port>/  and it
    receives the fused forward view instead. Stdlib only (no extra deps)."""
    import socketserver
    from http.server import BaseHTTPRequestHandler

    streamer = PanoramaStreamer(remote_ip, output_dir=output_dir, refine=refine).start()

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):  # quiet
            pass

        def do_GET(self):
            if self.path == "/snapshot.jpg":
                jpeg = streamer.latest_jpeg(quality=quality)
                if jpeg is None:
                    self.send_error(503, "Fused vision is warming up")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.end_headers()
                self.wfile.write(jpeg)
                return

            if self.path not in ("/", "/stream", "/stream.mjpg"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            try:
                while True:
                    jpeg = streamer.latest_jpeg(quality=quality)
                    if jpeg is None:
                        time.sleep(0.03)
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                pass  # client (VR headset) disconnected

    class _Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = _Server((host, port), _Handler)
    print(
        f"[panorama-serve] fused central vision (MJPEG) at "
        f"http://{host}:{port}/  (calibrated={streamer.is_calibrated}); Ctrl+C to stop"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[panorama-serve] stopped")
    finally:
        server.shutdown()
        streamer.stop()
    return 0


def _gui_available() -> bool:
    """cv2.imshow only works when OpenCV was built with a GUI backend; the
    headless wheel (common under uv/WSL) raises at show time. Probe once."""
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("__probe__")
        return True
    except cv2.error:
        return False


def stream_panorama(
    remote_ip: str,
    output_dir: Path,
    *,
    sink: str = "auto",
    on_frame=None,
    refine: bool = True,
) -> int:
    """Stream the live fused forward panorama from both eyes.

    Uses the saved calibration when one exists, else the physical-spec
    defaults (so it streams the CURRENT view even before calibrating — the
    seam just is not matched yet). `on_frame(fused_bgr)` is called with each
    fused frame if given (headless callback use). `sink`:
      - "browser": write a self-refreshing HTML page + rolling JPEG (works
        everywhere, no GUI needed — open the printed URL).
      - "window": OpenCV window ('q' quits, 's' saves); needs GUI OpenCV.
      - "auto" (default): window if OpenCV has a GUI backend, else browser.
    Returns a process exit code."""
    from sourccey_elevated_safety import SlamCameraSubscriber, endpoint_from_remote_ip

    cal, is_calibrated = load_calibration_or_default(output_dir)
    label = (
        f"calibrated (score {cal.score:.2f}, hfov {cal.hfov_deg:.1f})"
        if is_calibrated
        else "UNCALIBRATED defaults (roll 0, hfov 82) — run --mode calibrate"
    )
    print(f"[stream] {label}")

    if sink == "auto":
        sink = "window" if _gui_available() else "browser"
    if sink == "window" and not _gui_available():
        print("[stream] OpenCV has no GUI backend here; falling back to browser sink")
        sink = "browser"

    live_jpg = output_dir / "live_panorama.jpg"
    if sink == "browser":
        live_html = output_dir / "live_panorama.html"
        live_html.write_text(_LIVE_HTML.format(label=label), encoding="utf-8")
        print(
            f"[stream] live — open this in a browser:\n"
            f"[stream]   {live_html.resolve().as_uri()}\n"
            f"[stream] (updates ~10fps; Ctrl+C here to stop)"
        )

    endpoint = endpoint_from_remote_ip(remote_ip)
    subscriber = SlamCameraSubscriber(endpoint=endpoint, camera_keys=("front_left", "front_right"))
    subscriber.start()
    virt: VirtualCamera | None = None
    # Persistent refiner: warms up then FREEZES the eye-alignment transform so
    # the solid regions lock and stop squiggling; only overlap parallax stays
    # live.
    refiner = OverlapRefiner() if refine else None
    announced_lock = False
    try:
        if not subscriber.wait_for_frames(10.0, required=("front_left", "front_right")):
            print("[stream] ERROR: no eye frames within 10s — is the host streaming?")
            return 1
        if sink == "window":
            print("[stream] live window — 'q' quits, 's' saves a still")
        if refiner is not None:
            print("[stream] warming up eye alignment (hold the view steady a few seconds)...")
        while True:
            left, _ = subscriber.latest("front_left")
            right, _ = subscriber.latest("front_right")
            if left is None or right is None:
                time.sleep(0.05)
                continue
            fused, virt = fuse_eyes(left, right, cal, virt, refine=refine, refiner=refiner)
            if refiner is not None and refiner.locked and not announced_lock:
                announced_lock = True
                print("[stream] eye alignment LOCKED — wings are now fixed; only the middle adapts")
            if on_frame is not None:
                on_frame(fused)
            if sink == "browser":
                # Atomic-ish swap so the browser never reads a half-written
                # file: encode to a temp path (kept a .jpg so OpenCV picks the
                # JPEG encoder), then replace.
                tmp = live_jpg.with_name("live_panorama.tmp.jpg")
                cv2.imwrite(str(tmp), fused, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                tmp.replace(live_jpg)
                time.sleep(0.08)
                continue
            if sink == "callback":
                time.sleep(0.03)
                continue
            cv2.imshow("sourccey panorama (q quits, s saves)", fused)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                return 0
            if key == ord("s"):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = output_dir / f"preview_{stamp}.png"
                cv2.imwrite(str(path), fused)
                print(f"[stream] saved {path}")
    except KeyboardInterrupt:
        print("\n[stream] stopped")
        return 0
    finally:
        subscriber.stop()
        if sink == "window":
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


def run_preview(args: argparse.Namespace, output_dir: Path) -> int:
    return stream_panorama(
        args.remote_ip, output_dir, sink=args.sink, refine=not args.no_refine
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate and fuse Sourccey's two eye cameras into one forward panorama."
    )
    parser.add_argument(
        "--mode", choices=("capture", "calibrate", "preview", "serve"), required=True
    )
    parser.add_argument("--remote-ip", default="192.168.1.237")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--interval", type=float, default=2.5)
    parser.add_argument(
        "--rotate-deg",
        type=float,
        default=12.0,
        help="capture mode: approximate in-place rotation (open loop) between "
        "pairs for scene variety. 0 disables rotation (capture stays passive).",
    )
    parser.add_argument(
        "--turn-speed",
        type=float,
        default=0.82,
        help="capture mode: base theta velocity for the between-pair rotation "
        "(matches the wander loop default).",
    )
    parser.add_argument("--robot-id", default="sourccey")
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="preview mode: disable the per-frame local overlap refinement "
        "(show the raw global-warp blend for A/B comparison).",
    )
    parser.add_argument(
        "--sink",
        choices=("auto", "browser", "window"),
        default="auto",
        help="preview output: 'browser' (self-refreshing HTML, works headless), "
        "'window' (OpenCV GUI), 'auto' (window if available else browser).",
    )
    parser.add_argument(
        "--serve-host",
        default="0.0.0.0",
        help="serve mode: interface to bind the MJPEG server (default all).",
    )
    parser.add_argument(
        "--serve-port",
        type=int,
        default=8091,
        help="serve mode: MJPEG HTTP port (point your VR right-eye source at "
        "http://<robot-host>:<port>/).",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "capture":
        return run_capture(args, output_dir)
    if args.mode == "calibrate":
        return run_calibrate(args, output_dir)
    if args.mode == "serve":
        return serve_mjpeg(
            args.remote_ip,
            host=args.serve_host,
            port=args.serve_port,
            output_dir=output_dir,
            refine=not args.no_refine,
        )
    return run_preview(args, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
