"""Monocular-depth perception for Sourccey's eye cameras.

Replaces the Hough edge detector as the ELEVATED-obstacle candidate source:
Depth-Anything-V2 (metric, indoor) turns each eye frame into per-pixel depth;
the field-calibrated CameraModel turns depth into robot-frame 3D; anything
0.30..1.10m above the floor is a surface the base can hit ("floating edge"),
with its distance, bearing and extent measured directly — no line heuristics,
no parallax, no vocabulary.

Scale anchoring: monocular metric depth can be globally off by a factor.
When lidar ranges are available, a 1-D search finds the scale that best
aligns depth pixels at the lidar plane height with the lidar's measured
ranges (the lidar is ground truth). Fallback: assume the bottom-center of
the frame is floor. Both are clamped to a sane band.

Pure math is torch-free (unit-testable); the model loads lazily inside
DepthEstimator, and DepthWorker degrades gracefully (failure -> the monitor
keeps using the classic detector).

NEVER touches the robot: passive perception only.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from sourccey_camera_geometry import CameraModel

DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

ELEVATED_MIN_M = 0.30  # just above the lidar scan plane (0.28m)
ELEVATED_MAX_M = 1.10
REPORT_RANGE_M = 2.50
CORRIDOR_HALF_WIDTH_M = 0.40
FLOOR_BAND_M = 0.12
SCALE_MIN = 0.45
SCALE_MAX = 2.20


@dataclass(frozen=True)
class DepthEyeResult:
    eye: str
    frame_monotonic: float  # when the analyzed frame was received
    scale: float
    scale_source: str  # "lidar" | "floor" | "raw"
    nearest_gate_m: float | None  # nearest elevated point inside the corridor
    nearest_any_m: float | None  # nearest elevated point at any bearing
    edge_height_m: float | None
    bearing_deg: float | None  # robot-frame bearing of the nearest point
    segment_p1: tuple[float, float] | None  # robot-frame (forward, lateral)
    segment_p2: tuple[float, float] | None
    # Floor-projected FOOTPRINT of the elevated region (robot-frame
    # (forward, lateral) points, wall columns excluded, decimated): the map
    # stamps THIS, like lidar points — an "edge line" fitted to a whole
    # visible tabletop is the wrong abstraction for a dense depth sensor.
    region_points_xy: tuple[tuple[float, float], ...]
    elevated_ratio: float
    # What the LIDAR says is nearest in the same corridor (robot-forward
    # meters), for on-overlay bias diagnosis of the depth ranges. None when
    # no lidar beams landed in the corridor.
    lidar_corridor_min_m: float | None
    overlay_bgr: np.ndarray | None


def _pixel_fields(model: CameraModel, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Per-pixel unit fields for a depth map of this shape: horizontal
    distance / vertical drop per meter of optical depth, plus the robot-frame
    bearing per column (scale-invariant)."""
    h, w = int(shape[0]), int(shape[1])
    fx = (w / 2.0) / math.tan(math.radians(float(model.hfov_deg) / 2.0))
    fy = (h / 2.0) / math.tan(math.radians(float(model.vfov_deg) / 2.0))
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    u = np.arange(w, dtype=np.float32)[None, :]
    v = np.arange(h, dtype=np.float32)[:, None]
    x_unit = (u - cx) / fx  # right, per meter of optical depth
    y_unit = (v - cy) / fy  # image-down, per meter of optical depth
    p = math.radians(float(model.pitch_down_deg))
    fwd_unit = math.cos(p) - y_unit * math.sin(p)  # level-forward per depth meter
    up_unit = -math.sin(p) - y_unit * math.cos(p)  # level-up per depth meter
    fwd_unit = np.broadcast_to(fwd_unit, (h, w))
    up_unit = np.broadcast_to(up_unit, (h, w))
    x_unit = np.broadcast_to(x_unit, (h, w))
    x_ratio = np.broadcast_to(u / max(w - 1, 1), (h, w))
    bearing_deg = float(model.bearing_sign) * (
        float(model.yaw_deg) - (x_ratio - 0.5) * float(model.hfov_deg)
    )
    return {
        "fwd_unit": fwd_unit.astype(np.float32),
        "up_unit": up_unit.astype(np.float32),
        "x_unit": x_unit.astype(np.float32),
        "bearing_deg": bearing_deg.astype(np.float32),
    }


def _floor_scale(model: CameraModel, depth_m: np.ndarray, fields: dict[str, np.ndarray]) -> float:
    """Scale that puts the bottom-center patch (assumed floor) at floor
    height: median over patch of (-cam_height / up_unit) / depth."""
    h, w = depth_m.shape
    rows = slice(int(h * 0.86), h)
    cols = slice(int(w * 0.30), int(w * 0.70))
    up = fields["up_unit"][rows, cols]
    patch_depth = depth_m[rows, cols]
    usable = (up < -1e-3) & (patch_depth > 1e-3)
    if not np.any(usable):
        return 1.0
    implied = (-float(model.height_m) / up[usable]) / patch_depth[usable]
    return float(np.median(implied))


def _lidar_scale(
    model: CameraModel,
    depth_m: np.ndarray,
    fields: dict[str, np.ndarray],
    lidar_bearings_deg: np.ndarray,
    lidar_ranges_m: np.ndarray,
    lidar_plane_height_m: float = 0.28,
) -> float | None:
    """1-D search for the depth scale that best aligns pixels at the lidar
    plane height with the lidar's measured ranges. The lidar is metric
    ground truth; monocular scale is the free variable."""
    half_cone = float(model.hfov_deg) / 2.0 - 3.0
    in_cone = (
        np.abs(((lidar_bearings_deg - float(model.yaw_deg) * float(model.bearing_sign)) + 180.0) % 360.0 - 180.0)
        <= half_cone
    )
    beams_b = np.asarray(lidar_bearings_deg, dtype=np.float32)[in_cone]
    beams_r = np.asarray(lidar_ranges_m, dtype=np.float32)[in_cone]
    good = (beams_r > 0.15) & (beams_r < 6.0)
    beams_b, beams_r = beams_b[good], beams_r[good]
    if len(beams_b) < 4:
        return None
    # Sample up to 12 beams evenly.
    idx = np.linspace(0, len(beams_b) - 1, min(12, len(beams_b))).astype(int)
    beams_b, beams_r = beams_b[idx], beams_r[idx]

    up = fields["up_unit"]
    fwd = fields["fwd_unit"]
    bearing = fields["bearing_deg"]
    cam_h = float(model.height_m)
    # The lidar sits ~2-3cm from the eye cameras: treat its ranges as slant
    # ranges from the camera. Depth pixels give the forward COMPONENT along
    # the camera's level axis; convert to slant via the off-axis angle.
    center_bearing = float(model.bearing_sign) * float(model.yaw_deg)
    # Fronto-parallel degeneracy: a mis-scaled tabletop range profile can
    # impersonate the wall's (both are ~r0/cos(b)). The floor anchor is an
    # independent scale witness — a soft prior toward it breaks the tie
    # without overriding an unambiguous lidar fit.
    floor_prior = _floor_scale(model, depth_m, fields)
    best_scale, best_err = None, None
    for s in np.linspace(SCALE_MIN, SCALE_MAX, 71):
        height = cam_h + s * up * depth_m
        plane = np.abs(height - float(lidar_plane_height_m)) <= 0.08
        if not np.any(plane):
            continue
        errs = []
        for b, r in zip(beams_b, beams_r):
            beam_pixels = plane & (np.abs(bearing - b) <= 3.0)
            if np.count_nonzero(beam_pixels) < 6:
                continue
            off_axis = math.cos(math.radians(float(b) - center_bearing))
            if off_axis < 0.3:
                continue
            depth_fwd = float(np.median((s * fwd * depth_m)[beam_pixels]))
            depth_slant = depth_fwd / off_axis
            errs.append(abs(depth_slant - float(r)))
        if len(errs) < 3:
            continue
        err = float(np.median(errs)) + 0.10 * abs(float(s) - float(floor_prior))
        if best_err is None or err < best_err:
            best_err, best_scale = err, float(s)
    if best_scale is None or best_err is None or best_err > 0.60:
        return None
    # The lidar REFINES the floor anchor; it may not overrule it. When
    # furniture occludes the wall at lidar height, every candidate scale
    # with plane-height pixels is an impostor (a mis-scaled tabletop
    # imitating the wall) — a fit far from the floor witness is exactly
    # that, so reject it and let the caller use the floor anchor.
    if abs(best_scale - float(floor_prior)) > 0.40 * max(float(floor_prior), 0.2):
        return None
    return best_scale


def analyze_depth(
    model: CameraModel,
    depth_m: np.ndarray,
    *,
    eye: str,
    frame_monotonic: float,
    frame_bgr: np.ndarray | None = None,
    lidar_bearings_deg: np.ndarray | None = None,
    lidar_ranges_m: np.ndarray | None = None,
) -> DepthEyeResult:
    """Turn one metric depth map into a robot-frame elevated-obstacle report."""
    depth_m = np.asarray(depth_m, dtype=np.float32)
    fields = _pixel_fields(model, depth_m.shape)

    scale_source = "raw"
    scale = 1.0
    if lidar_bearings_deg is not None and lidar_ranges_m is not None and len(lidar_bearings_deg):
        lidar_s = _lidar_scale(model, depth_m, fields, np.asarray(lidar_bearings_deg), np.asarray(lidar_ranges_m))
        if lidar_s is not None:
            scale, scale_source = lidar_s, "lidar"
    if scale_source == "raw":
        scale, scale_source = _floor_scale(model, depth_m, fields), "floor"
    scale = float(np.clip(scale, SCALE_MIN, SCALE_MAX))

    d = depth_m * scale
    height = float(model.height_m) + fields["up_unit"] * d
    # Exact Cartesian camera->robot transform (a flat table must stay flat:
    # projecting via horizontal*cos(linear_bearing) bowed surfaces ~15% at
    # the cone edges). Camera-level frame: forward along the camera axis,
    # lateral positive toward +bearing; then rotate by the eye yaw and
    # translate by its forward offset.
    fwd_cam = fields["fwd_unit"] * d
    lat_cam = -float(model.bearing_sign) * (fields["x_unit"] * d)
    yaw_rad = math.radians(float(model.bearing_sign) * float(model.yaw_deg))
    robot_forward = (
        float(model.forward_offset_m) + fwd_cam * math.cos(yaw_rad) - lat_cam * math.sin(yaw_rad)
    )
    robot_lateral = fwd_cam * math.sin(yaw_rad) + lat_cam * math.cos(yaw_rad)

    elevated = (
        (height >= ELEVATED_MIN_M)
        & (height <= ELEVATED_MAX_M)
        & (robot_forward > 0.05)
        & (robot_forward <= REPORT_RANGE_M)
        & (fwd_cam > 0.05)
    )
    elevated_ratio = float(np.mean(elevated))

    nearest_gate_m: float | None = None
    nearest_any_m: float | None = None
    edge_height_m: float | None = None
    bearing_deg: float | None = None
    seg_p1 = seg_p2 = None
    region_points: tuple[tuple[float, float], ...] = ()
    if np.any(elevated):
        # Wall columns — DEPTH-LOCAL test: exclude a column only when the
        # above-band surface sits at the SAME distance as the footprint
        # there (a wall rising from it). Testing "anything above 1.1m in
        # the column" excluded nearly every table column, because the
        # background wall/shelving BEHIND the table is above the band too
        # (field 2026-07-12: footprints gutted, segment fallback kept
        # stamping misplaced lines).
        wall_above = (
            (height > ELEVATED_MAX_M)
            & (height <= ELEVATED_MAX_M + 0.6)
            & (fwd_cam > 0.05)
        )
        # Footprint limited to 1.8m: monocular depth error grows with range
        # (~10-15%), and stops only need the nearby furniture placed well.
        fp_candidate = elevated & (robot_forward <= 1.8)
        big = np.float32(1e6)
        fp_fwd_col = np.where(fp_candidate, robot_forward, big).min(axis=0)
        above_fwd_col = np.where(wall_above, robot_forward, big).min(axis=0)
        # 0.30m: a true wall rises at ~zero forward offset from the band
        # surface; upper cabinets over a countertop are typically recessed
        # ~0.3m+ and must NOT suppress the counter's footprint.
        wall_cols = (
            (fp_fwd_col < big)
            & (above_fwd_col < big)
            & (np.abs(above_fwd_col - fp_fwd_col) < 0.30)
        )
        footprint = fp_candidate & ~wall_cols[None, :]
        if np.any(footprint):
            fp_fwd = robot_forward[footprint]
            fp_lat = robot_lateral[footprint]
            # Decimate to a 6cm grid, cap the count: the map rasterizes at
            # 4cm anyway and the queue must stay light.
            cells = np.unique(
                np.stack(
                    [np.round(fp_fwd / 0.06).astype(np.int32), np.round(fp_lat / 0.06).astype(np.int32)],
                    axis=1,
                ),
                axis=0,
            )
            if len(cells) > 240:
                keep = np.linspace(0, len(cells) - 1, 240).astype(int)
                cells = cells[keep]
            region_points = tuple((float(c[0]) * 0.06, float(c[1]) * 0.06) for c in cells)
        masked_fwd = np.where(elevated, robot_forward, np.inf)
        any_iy, any_ix = np.unravel_index(int(np.argmin(masked_fwd)), masked_fwd.shape)
        nearest_any_m = float(robot_forward[any_iy, any_ix])
        bearing_deg = float(fields["bearing_deg"][any_iy, any_ix])
        corridor = elevated & (np.abs(robot_lateral) <= CORRIDOR_HALF_WIDTH_M)
        ref_iy, ref_ix = any_iy, any_ix
        if np.any(corridor):
            corridor_fwd = np.where(corridor, robot_forward, np.inf)
            ref_iy, ref_ix = np.unravel_index(int(np.argmin(corridor_fwd)), corridor_fwd.shape)
            nearest_gate_m = float(robot_forward[ref_iy, ref_ix])
        # Near-boundary segment for the map: the depth band just behind the
        # nearest point, restricted to the CONNECTED surface that contains
        # it. Taking the global lateral extremes bridged two separate
        # objects at similar depth into a phantom wall across the free gap
        # between them (field 2026-07-12: red diagonals across open floor).
        near_ref = nearest_gate_m if nearest_gate_m is not None else nearest_any_m
        band = elevated & (robot_forward <= near_ref + 0.15)
        if np.count_nonzero(band) >= 8 and band[ref_iy, ref_ix]:
            _n_labels, band_labels = cv2.connectedComponents(
                band.astype(np.uint8), connectivity=8
            )
            component = band_labels == band_labels[ref_iy, ref_ix]
            if np.count_nonzero(component) >= 8:
                # WALL suppression: a wall passes THROUGH the furniture band
                # and keeps going above it, while furniture has open space
                # over its top. If the component's columns also show nearby
                # surface above the band, this is a wall — the lidar owns
                # walls, and publishing them painted phantom red lines
                # (field 2026-07-12: 0.89m-"edges" along the walls).
                above_band = (
                    (height > ELEVATED_MAX_M)
                    & (height <= ELEVATED_MAX_M + 0.6)
                    & (fwd_cam > 0.05)
                    & (robot_forward <= near_ref + 0.6)
                )
                comp_cols = np.unique(np.where(component)[1])
                above_cols = np.where(above_band.any(axis=0))[0]
                col_overlap = np.intersect1d(comp_cols, above_cols)
                is_wall = len(comp_cols) > 0 and (len(col_overlap) / len(comp_cols)) > 0.4
                if not is_wall:
                    comp_lat = robot_lateral[component].astype(np.float64)
                    comp_fwd = robot_forward[component].astype(np.float64)
                    # Least-squares line over the WHOLE component, endpoints
                    # at its (percentile-trimmed) lateral extent. The two raw
                    # extreme pixels are noise-prone: one bad corner pixel
                    # used to rotate the entire mapped segment into a
                    # diagonal.
                    lat_lo = float(np.percentile(comp_lat, 2.0))
                    lat_hi = float(np.percentile(comp_lat, 98.0))
                    if lat_hi - lat_lo >= 0.05:
                        slope, intercept = np.polyfit(comp_lat, comp_fwd, 1)
                        seg_p1 = (float(intercept + slope * lat_lo), lat_lo)
                        seg_p2 = (float(intercept + slope * lat_hi), lat_hi)
                        edge_height_m = float(np.median(height[component]))

    # Lidar cross-check: nearest lidar return in the same corridor (robot-
    # forward meters). Shown on the overlay next to the depth estimate so
    # any systematic range bias is measurable from the panels/bundles.
    lidar_corridor_min_m: float | None = None
    if lidar_bearings_deg is not None and lidar_ranges_m is not None and len(lidar_bearings_deg):
        lb = np.asarray(lidar_bearings_deg, dtype=np.float32)
        lr = np.asarray(lidar_ranges_m, dtype=np.float32)
        lidar_fwd = lr * np.cos(np.radians(lb)) + 0.229  # lidar sits 0.229m ahead of center
        lidar_lat = lr * np.sin(np.radians(lb))
        in_corr = (
            (np.abs(lidar_lat) <= CORRIDOR_HALF_WIDTH_M)
            & (lidar_fwd > 0.20)
            & (lidar_fwd <= REPORT_RANGE_M)
        )
        if np.any(in_corr):
            lidar_corridor_min_m = float(np.min(lidar_fwd[in_corr]))

    overlay = None
    if frame_bgr is not None:
        overlay = frame_bgr.copy()
        if overlay.shape[:2] == depth_m.shape:
            floorish = np.abs(height) <= FLOOR_BAND_M
            overlay[elevated] = (0.35 * overlay[elevated] + 0.65 * np.array([0, 0, 220])).astype(np.uint8)
            overlay[floorish] = (0.7 * overlay[floorish] + 0.3 * np.array([0, 160, 0])).astype(np.uint8)
            text = "clear" if nearest_gate_m is None else f"elevated {nearest_gate_m:.2f}m"
            if lidar_corridor_min_m is not None:
                text += f" | lidar {lidar_corridor_min_m:.2f}m"
            cv2.putText(
                overlay,
                f"depth[{scale_source} x{scale:.2f}] {text}",
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

    return DepthEyeResult(
        eye=str(eye),
        frame_monotonic=float(frame_monotonic),
        scale=scale,
        scale_source=scale_source,
        nearest_gate_m=nearest_gate_m,
        nearest_any_m=nearest_any_m,
        edge_height_m=edge_height_m,
        bearing_deg=bearing_deg,
        segment_p1=seg_p1,
        segment_p2=seg_p2,
        region_points_xy=region_points,
        elevated_ratio=elevated_ratio,
        lidar_corridor_min_m=lidar_corridor_min_m,
        overlay_bgr=overlay,
    )


class DepthEstimator:
    """Lazy wrapper around the Depth-Anything-V2 metric pipeline."""

    def __init__(self, model_name: str = DEFAULT_DEPTH_MODEL) -> None:
        self.model_name = str(model_name)
        self._pipe = None
        self.device = "cpu"

    def load(self) -> None:
        import torch
        from transformers import pipeline as hf_pipeline

        device = 0 if torch.cuda.is_available() else -1
        self.device = "cuda" if device == 0 else "cpu"
        self._pipe = hf_pipeline("depth-estimation", model=self.model_name, device=device)

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR frame -> metric depth map (meters), same HxW as the frame."""
        from PIL import Image

        if self._pipe is None:
            self.load()
        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result = self._pipe(pil)
        depth = result["predicted_depth"].squeeze().float().cpu().numpy()
        if depth.shape != frame_bgr.shape[:2]:
            depth = cv2.resize(
                depth, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        return depth.astype(np.float32)


class DepthWorker:
    """Daemon thread: alternately runs depth on each eye's freshest frame and
    publishes DepthEyeResults. Model load failures set .failure and the
    thread exits — consumers fall back to the classic detector."""

    def __init__(
        self,
        subscriber,
        eye_models: dict[str, CameraModel],
        *,
        lidar_ranges_fn=None,
        model_name: str = DEFAULT_DEPTH_MODEL,
        max_frame_age_s: float = 1.0,
    ) -> None:
        self._subscriber = subscriber
        self._eye_models = dict(eye_models)
        self._lidar_ranges_fn = lidar_ranges_fn
        self._estimator = DepthEstimator(model_name)
        self._max_frame_age_s = float(max_frame_age_s)
        self._lock = threading.Lock()
        self._results: dict[str, DepthEyeResult] = {}
        self.ready = False
        self.failure: str | None = None
        self.inference_s: float = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="depth-perception", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def latest(self, eye: str, max_age_s: float) -> DepthEyeResult | None:
        with self._lock:
            result = self._results.get(str(eye))
        if result is None:
            return None
        if time.monotonic() - result.frame_monotonic > float(max_age_s):
            return None
        return result

    def _run(self) -> None:
        try:
            self._estimator.load()
        except Exception as exc:  # missing deps / download failure: fail soft
            self.failure = f"{type(exc).__name__}: {exc}"
            print(
                f"[safety] WARNING: depth perception failed to load ({self.failure}); "
                "the classic edge detector remains active"
            )
            return
        self.ready = True
        print(
            f"[safety] depth perception READY on {self._estimator.device} "
            f"({self._estimator.model_name}); eyes now classified from metric 3D"
        )
        eyes = list(self._eye_models.keys())
        eye_index = 0
        while not self._stop_event.is_set():
            eye = eyes[eye_index % len(eyes)]
            eye_index += 1
            frame, age_s = self._subscriber.latest(eye)
            if frame is None or age_s is None or age_s > self._max_frame_age_s:
                time.sleep(0.05)
                continue
            frame_monotonic = time.monotonic() - float(age_s)
            lidar_b = lidar_r = None
            if self._lidar_ranges_fn is not None:
                try:
                    ranges = self._lidar_ranges_fn()
                    if ranges is not None:
                        lidar_b, lidar_r = ranges
                except Exception:
                    lidar_b = lidar_r = None
            started = time.monotonic()
            try:
                depth = self._estimator.infer(frame)
                result = analyze_depth(
                    self._eye_models[eye],
                    depth,
                    eye=eye,
                    frame_monotonic=frame_monotonic,
                    frame_bgr=frame,
                    lidar_bearings_deg=lidar_b,
                    lidar_ranges_m=lidar_r,
                )
            except Exception as exc:
                self.failure = f"{type(exc).__name__}: {exc}"
                return
            self.inference_s = time.monotonic() - started
            with self._lock:
                self._results[eye] = result
