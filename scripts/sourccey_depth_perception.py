"""Monocular-depth perception for Sourccey's eye cameras.

Uses the Hough edge detector as the ELEVATED-obstacle proposal source, then
uses Depth-Anything-V2 (metric, indoor) only inside that verified edge band;
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
from lerobot.control.sourccey.sourccey.elevated_edge_scan_live import (
    ElevatedEdgeDetector,
    ElevatedEdgeScanConfig,
)

DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

ELEVATED_MIN_M = 0.30  # just above the lidar scan plane (0.28m)
ELEVATED_MAX_M = 1.10
REPORT_RANGE_M = 2.50
CORRIDOR_HALF_WIDTH_M = 0.40
# Squeeze tier: the robot's half-width is 0.28m. When the WIDE corridor is
# obstructed but this narrow one is clear, the gap is physically passable —
# the gate demands creep speed instead of stopping, so the robot approaches,
# keeps refining the edge with close-range footprints, and squeezes through
# doorway-sized gaps beside furniture instead of striking the frontier out.
CORRIDOR_NARROW_HALF_WIDTH_M = 0.30
# Map only the leading band of an elevated component.  A metric-depth image
# sees an entire tabletop face, but filling that surface in the 2-D map turns
# a table into a large solid red slab and can seal a nearby doorway.  For
# navigation, its nearest boundary is the occupied geometry that matters.
FOOTPRINT_BOUNDARY_DEPTH_M = 0.10
FLOOR_BAND_M = 0.12
SCALE_MIN = 0.45
SCALE_MAX = 2.20
# Outer columns of the frame are excluded from map evidence (footprints and
# floor clearing): lens distortion and calibration error are worst at the
# horizontal FOV edges, and smeared edge-column footprints painted phantom
# blockage beside real furniture (field 2026-07-12: doorway walled off).
EDGE_COL_TRIM_RATIO = 0.08
# The eye lenses are mounted above and slightly ahead of the chassis. Their
# steep lower/outboard rays can still intersect the robot even after the FOV
# edge trim (last-run evidence: a -49 deg, 0.11 m "elevated" return).  This
# rectangle is entirely inside the robot's swept body envelope, so it cannot
# contain a forward obstacle the eyes should make the gate stop for.
SELF_BODY_FORWARD_M = 0.30
SELF_BODY_HALF_WIDTH_M = 0.45
# The 8% image trim still retains rays around +/-49 degrees.  On the current
# eye mount, near returns in that remaining outboard band are the chassis/lens
# rim, not geometry in the 18-inch robot's forward swept path.  A 0.50 m ray
# at 46 degrees is already 0.52 m off centre, well outside that path.  Keep
# real centre/forward obstacles intact while rejecting this proven artifact
# before it can trigger near-freeze or paint the map.
OUTER_NEAR_BEARING_DEG = 46.0
OUTER_NEAR_FORWARD_M = 0.50
# Occlusion-bridge ("membrane") rejection: monocular depth interpolates
# smoothly across open gaps between a near object and the far background,
# fabricating a surface where there is only air — field 2026-07-12: the open
# doorway read as part of the table, walling off the exit. Real furniture is
# axis-aligned: vertical faces have ~zero forward-gradient, horizontal tops
# have ~zero height-gradient, so min(|grad h|, |grad fwd|) is ~0 on every
# real surface interior but 0.035..0.08 m/deg on bridges (measured on exact
# ray-cast scenes). Pixels above this are fabrications, not obstacles.
MEMBRANE_MIN_GRAD_M_PER_DEG = 0.025
# Ghost-wall (lidar-transparency) veto: textureless white walls/doors give
# the model nothing to anchor depth on, so it hallucinates them nearer than
# they are (field 2026-07-12: a white wall beside the exit kept re-painting
# a phantom blob over the doorway corridor faster than floor evidence could
# clear it, and gated the exit shut). A claimed surface that CONTINUES DOWN
# to the floor is floor-connected — wall, cabinet, box — and the lidar MUST
# see it at about its claimed range; if every nearby beam passes well
# beyond the claim, the surface is fabricated. Genuine floating overhangs
# (tabletops, counters, pedestal tables) are not floor-connected and are
# NEVER tested against the lidar — the standing constraint holds: the lidar
# never clamps depth on overhangs.
GHOST_FLOOR_CONNECT_GAP_M = 0.30
GHOST_LIDAR_MARGIN_M = 0.45
GHOST_BEAM_WINDOW_DEG = 2.5
# Floor evidence covers the SAME range as footprints: the pitched-down eyes
# first see floor ~1.05m out, so a shorter cap left almost no clearing band
# and stale cells beyond it were unerasable. Occupied and free evidence must
# cover the same region or the map ratchets toward blocked.
FLOOR_EVIDENCE_RANGE_M = 1.8
# A vertical door, wall, or cabinet face reaches the floor at essentially the
# same range.  Those are already measured by the lidar at bumper height; if
# the eye model also treats their upper pixels as an elevated overhang, it
# joins them to a nearby table and paints the doorway red.  Keep camera depth
# for genuinely suspended geometry (tabletops/counters), and delegate
# floor-connected surfaces to lidar.
GROUND_CONNECTED_MAX_HEIGHT_M = 0.28
GROUND_CONNECTED_RANGE_TOLERANCE_M = 0.18
GROUND_CONNECTED_MIN_COLUMN_FRACTION = 0.55
GROUND_CONNECTED_MIN_COMPONENT_PIXELS = 12


@dataclass(frozen=True)
class DepthEyeResult:
    eye: str
    frame_monotonic: float  # when the analyzed frame was received
    scale: float
    scale_source: str  # "lidar" | "floor" | "raw"
    # A conventional contrast/line detector proposed this image region.
    # Depth may create an elevated candidate only inside this proposal.
    proposal_detected: bool
    proposal_score: float
    proposal_line_xy: tuple[tuple[int, int], tuple[int, int]] | None
    nearest_gate_m: float | None  # nearest elevated point inside the corridor
    # Nearest elevated point inside the NARROW (body-width) corridor: clear
    # here while the wide corridor is blocked = a squeezable gap.
    nearest_gate_narrow_m: float | None
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
    # Floor-band pixels' robot-frame (forward, lateral) positions (close
    # range, FOV-edge trimmed, decimated): POSITIVE evidence of traversable
    # floor, used by the map to CLEAR stale elevated cells it contradicts.
    # Occupancy mapping must be symmetric — without free-space evidence a
    # single overshot footprint blocks a doorway forever.
    floor_points_xy: tuple[tuple[float, float], ...]
    elevated_ratio: float
    # What the LIDAR says is nearest in the same corridor (robot-forward
    # meters), for on-overlay bias diagnosis of the depth ranges. None when
    # no lidar beams landed in the corridor.
    lidar_corridor_min_m: float | None
    overlay_bgr: np.ndarray | None


def _ground_connected_components(
    elevated: np.ndarray,
    height: np.ndarray,
    robot_forward: np.ndarray,
) -> np.ndarray:
    """Return elevated pixels belonging to floor-connected vertical faces.

    This is deliberately a component/column test rather than a color or
    semantic test. A door beside a table can be visually indistinguishable
    from the table, but in metric geometry the door continues below the
    elevated band at the same forward range in most of its columns. A table
    top has floor *behind* it, not directly below it at that range.
    """
    ground_connected = np.zeros_like(elevated, dtype=bool)
    count, labels = cv2.connectedComponents(elevated.astype(np.uint8), connectivity=8)
    row_indices = np.arange(elevated.shape[0])[:, None]
    lower_band = (height >= 0.02) & (height <= GROUND_CONNECTED_MAX_HEIGHT_M)
    for label in range(1, count):
        component = labels == label
        if np.count_nonzero(component) < GROUND_CONNECTED_MIN_COMPONENT_PIXELS:
            continue
        component_cols = np.unique(np.where(component)[1])
        supported_cols = 0
        for col in component_cols:
            component_rows = np.where(component[:, col])[0]
            if not len(component_rows):
                continue
            reference_forward = float(np.median(robot_forward[component_rows, col]))
            below_component = row_indices[:, 0] > int(component_rows.max())
            same_range_lower = (
                lower_band[:, col]
                & below_component
                & (np.abs(robot_forward[:, col] - reference_forward)
                   <= GROUND_CONNECTED_RANGE_TOLERANCE_M)
            )
            if np.any(same_range_lower):
                supported_cols += 1
        if (
            supported_cols / max(len(component_cols), 1)
            >= GROUND_CONNECTED_MIN_COLUMN_FRACTION
        ):
            ground_connected |= component
    return ground_connected


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
    if float(model.hfov_deg) > 90.0:
        # Wide pinhole (the fused panorama): the linear column->bearing
        # approximation breaks at tan-stretched edges (>10deg error at the
        # wings of a 104deg frame). Use the exact per-column ray angle.
        # Per-eye models keep the legacy linear form bit-for-bit — their
        # referee windows were field-tuned against it.
        bearing_deg = float(model.bearing_sign) * (
            float(model.yaw_deg) - np.degrees(np.arctan(x_unit))
        )
    else:
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


def _decimate_xy(fwd: np.ndarray, lat: np.ndarray, cell_m: float = 0.06, cap: int = 240) -> tuple[tuple[float, float], ...]:
    """Snap (forward, lateral) samples to a grid, dedupe, cap the count."""
    cells = np.unique(
        np.stack(
            [np.round(fwd / cell_m).astype(np.int32), np.round(lat / cell_m).astype(np.int32)],
            axis=1,
        ),
        axis=0,
    )
    if len(cells) > cap:
        keep = np.linspace(0, len(cells) - 1, cap).astype(int)
        cells = cells[keep]
    return tuple((float(c[0]) * cell_m, float(c[1]) * cell_m) for c in cells)


def analyze_depth(
    model: CameraModel,
    depth_m: np.ndarray,
    *,
    eye: str,
    frame_monotonic: float,
    frame_bgr: np.ndarray | None = None,
    lidar_bearings_deg: np.ndarray | None = None,
    lidar_ranges_m: np.ndarray | None = None,
    proposal_line_xy: tuple[tuple[int, int], tuple[int, int]] | None = None,
    proposal_score: float = 0.0,
    seam_cols: tuple[int, int] | None = None,
    coverage_mask: np.ndarray | None = None,
) -> DepthEyeResult:
    """Turn one metric depth map into a robot-frame elevated-obstacle report.

    seam_cols / coverage_mask (fused-panorama mode): the mosaic's hard seam
    sits dead ahead (bearing 0 — the most safety-critical column), where the
    two eyes' content can step by the residual calibration error and edges
    look chopped. The seam band is FORGIVEN, never trusted alone: evidence
    living only inside it cannot gate or map, proposals bridge across it, and
    floor evidence from it never erases the map. Pixels outside coverage
    (black, no eye saw them) are excluded from everything."""
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

    # Membrane rejection (see MEMBRANE_MIN_GRAD_M_PER_DEG): angular-
    # normalized gradients of height and robot-forward, on lightly smoothed
    # maps so per-pixel model noise cannot fake a slope. A pixel where BOTH
    # change fast lies on a diagonal ramp in the forward/height plane —
    # something the physical world almost never builds but depth bridging
    # always does. Silhouette rims (real depth jumps) also trip it; losing
    # that 1-2px halo is free (it never was surface).
    h_px, w_px = depth_m.shape
    row_deg = float(model.vfov_deg) / max(h_px - 1, 1)
    col_deg = float(model.hfov_deg) / max(w_px - 1, 1)
    height_sm = cv2.GaussianBlur(height, (5, 5), 1.0)
    forward_sm = cv2.GaussianBlur(robot_forward, (5, 5), 1.0)
    gh_r, gh_c = np.gradient(height_sm)
    gf_r, gf_c = np.gradient(forward_sm)
    grad_h = np.hypot(gh_r / row_deg, gh_c / col_deg)
    grad_f = np.hypot(gf_r / row_deg, gf_c / col_deg)
    membrane = (grad_h > MEMBRANE_MIN_GRAD_M_PER_DEG) & (grad_f > MEMBRANE_MIN_GRAD_M_PER_DEG)

    # The outer FOV columns are excluded from ALL eye evidence — the GATE
    # included, not just the map. Field 2026-07-13: the outermost columns
    # (bearing ~±56° = yaw + hfov/2) reported "elevated at 0.05m" — the
    # robot seeing its own chassis / lens-edge garbage — which held the
    # near-freeze tier on and phantom-stopped every exit approach.
    trim = int(round(w_px * EDGE_COL_TRIM_RATIO))
    col_ok = np.zeros(w_px, dtype=bool)
    col_ok[trim : w_px - trim] = True
    col_ok = col_ok[None, :]

    # Exclude the camera's view of Sourccey's own shell from every elevated
    # output, including nearest_any_m.  It must happen before the near-freeze
    # calculation: trimming only the outer 8% left the observed -49 deg body
    # return inside the retained columns, where it defeated the clear narrow
    # corridor and prevented squeeze_creep.
    self_body = (
        (robot_forward < SELF_BODY_FORWARD_M)
        & (np.abs(robot_lateral) < SELF_BODY_HALF_WIDTH_M)
    )
    outer_near_self_sighting = (
        (np.abs(fields["bearing_deg"]) >= OUTER_NEAR_BEARING_DEG)
        & (robot_forward < OUTER_NEAR_FORWARD_M)
    )

    # The depth model supplies metric geometry, not reliable object
    # boundaries. The classic detector must first find a long, continuous
    # high-contrast edge with a supported top and underside. Restrict the
    # depth result to a narrow image band around that exact line so texture,
    # lighting, and broad flat regions cannot invent a collision obstacle.
    proposal_mask_u8 = np.zeros((h_px, w_px), dtype=np.uint8)
    if proposal_line_xy is not None:
        (x1, y1), (x2, y2) = proposal_line_xy
        cv2.line(
            proposal_mask_u8,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            255,
            thickness=max(int(round(h_px * 0.075)), 18),
            lineType=cv2.LINE_AA,
        )
    proposal_mask = proposal_mask_u8.astype(bool)
    seam_band = None
    if seam_cols is not None:
        c0 = max(0, int(seam_cols[0]))
        c1 = min(w_px, int(seam_cols[1]))
        if c1 > c0:
            seam_band = np.zeros((h_px, w_px), dtype=bool)
            seam_band[:, c0:c1] = True
            # Proposal bridging: an edge crossing the seam is often broken or
            # stepped there, so the Hough line stops at the band. Any row
            # where the proposal reaches within 25px of either band edge gets
            # the band filled — the edge is allowed to continue across.
            near_l = proposal_mask[:, max(0, c0 - 25) : c0].any(axis=1)
            near_r = proposal_mask[:, c1 : min(w_px, c1 + 25)].any(axis=1)
            bridge_rows = near_l | near_r
            if np.any(bridge_rows):
                grow = cv2.dilate(
                    bridge_rows.astype(np.uint8)[:, None], np.ones((13, 1), np.uint8)
                ).ravel().astype(bool)
                proposal_mask[grow, c0:c1] = True

    elevated = (
        (height >= ELEVATED_MIN_M)
        & (height <= ELEVATED_MAX_M)
        & (robot_forward > 0.05)
        & (robot_forward <= REPORT_RANGE_M)
        & (fwd_cam > 0.05)
        & ~membrane
        & col_ok
        & ~self_body
        & ~outer_near_self_sighting
        & proposal_mask
    )
    if coverage_mask is not None and coverage_mask.shape == elevated.shape:
        elevated &= coverage_mask

    # Ghost-wall veto (see GHOST_* constants), applied per CONNECTED
    # COMPONENT — floor-connectivity is a property of the object, not of a
    # single column (a blob's lateral edge columns clip only its mid-height
    # and look "floating" column-locally). A component is vetoed only when
    # ALL THREE hold: it is floor-connected somewhere, NO column of it is
    # lidar-corroborated, and several columns have beams passing well
    # beyond the claim. One corroborated column spares the whole component,
    # so a table whose thin legs the lidar clips in only one column keeps
    # everything.
    if lidar_bearings_deg is not None and lidar_ranges_m is not None and len(lidar_bearings_deg):
        lb_all = np.asarray(lidar_bearings_deg, dtype=np.float32)
        lr_all = np.asarray(lidar_ranges_m, dtype=np.float32)
        good_l = (lr_all > 0.10) & (lr_all < 8.0)
        l_fwd = lr_all[good_l] * np.cos(np.radians(lb_all[good_l])) + 0.229
        l_lat = lr_all[good_l] * np.sin(np.radians(lb_all[good_l]))
        l_bear = np.degrees(np.arctan2(l_lat, l_fwd))
        l_rng = np.hypot(l_fwd, l_lat)
        near_elev = elevated & (robot_forward <= 2.0)
        if len(l_rng) and np.any(near_elev):
            below_band = (height > 0.03) & (height < ELEVATED_MIN_M) & (fwd_cam > 0.05)
            _n_ghost, ghost_labels = cv2.connectedComponents(
                near_elev.astype(np.uint8), connectivity=8
            )
            for lab in range(1, _n_ghost):
                comp = ghost_labels == lab
                if np.count_nonzero(comp) < 12:
                    continue
                floor_connected = False
                corroborated = 0
                passthrough = 0
                for col in np.unique(np.where(comp)[1]):
                    col_fwd = robot_forward[:, col]
                    rows = np.where(comp[:, col])[0]
                    iy = rows[np.argmin(col_fwd[rows])]
                    d_col = float(col_fwd[iy])
                    rows_below = below_band[:, col]
                    if (
                        not floor_connected
                        and np.any(rows_below)
                        and float(np.min(np.abs(col_fwd[rows_below] - d_col)))
                        <= GHOST_FLOOR_CONNECT_GAP_M
                    ):
                        floor_connected = True
                    claim_bear = math.degrees(
                        math.atan2(float(robot_lateral[iy, col]), d_col)
                    )
                    claim_rng = math.hypot(d_col, float(robot_lateral[iy, col]))
                    near = (
                        np.abs(((l_bear - claim_bear) + 180.0) % 360.0 - 180.0)
                        <= GHOST_BEAM_WINDOW_DEG
                    )
                    if not np.any(near):
                        continue
                    if float(np.min(np.abs(l_rng[near] - claim_rng))) <= GHOST_LIDAR_MARGIN_M:
                        corroborated += 1
                    elif float(np.min(l_rng[near])) > claim_rng + GHOST_LIDAR_MARGIN_M:
                        passthrough += 1
                if floor_connected and corroborated == 0 and passthrough >= 3:
                    elevated &= ~comp

    # Do not let a floor-connected vertical surface merge into the map/gate
    # component for a suspended edge beside it. The lidar, unlike monocular
    # depth, measures those surfaces directly at bumper height.
    ground_connected = _ground_connected_components(elevated, height, robot_forward)
    elevated = elevated & ~ground_connected

    # Seam forgiveness: a component living (almost) entirely inside the seam
    # band is a stitch artifact — the band sits DEAD AHEAD, so trusting it
    # would phantom-stop straight-line driving. A real obstacle ahead is
    # wider than the ~28px band and survives via its out-of-band pixels.
    if seam_band is not None and np.any(elevated & seam_band):
        _n_seam, seam_labels = cv2.connectedComponents(
            elevated.astype(np.uint8), connectivity=8
        )
        for lab in range(1, _n_seam):
            comp = seam_labels == lab
            n_comp = int(np.count_nonzero(comp))
            if n_comp and np.count_nonzero(comp & seam_band) / n_comp >= 0.85:
                elevated &= ~comp
    elevated_ratio = float(np.mean(elevated))

    floor_mask = (
        (np.abs(height) <= FLOOR_BAND_M)
        & (fwd_cam > 0.05)
        & (robot_forward > 0.05)
        & (robot_forward <= FLOOR_EVIDENCE_RANGE_M)
        & col_ok
        & ~membrane  # a bridge ramping down to the far floor is not floor
    )
    if coverage_mask is not None and coverage_mask.shape == floor_mask.shape:
        floor_mask &= coverage_mask
    if seam_band is not None:
        # Seam mush must never ERASE mapped obstacles either.
        floor_mask &= ~seam_band
    floor_points: tuple[tuple[float, float], ...] = ()
    if np.any(floor_mask):
        floor_points = _decimate_xy(robot_forward[floor_mask], robot_lateral[floor_mask])
        # Floor is genuinely visible UNDER furniture (the gap below a
        # tabletop is under the elevated band) — such points must never
        # erase the furniture's own footprint. Drop floor points within
        # ~0.12m of any elevated projection seen in THIS frame; phantom
        # cells have floor evidence with no co-located elevated surface.
        # RAW band pixels (no column trim / membrane / ghost filtering):
        # suppression of CLEARING must stay conservative — a surface too
        # dubious to stamp can still be real enough that erasing it is wrong.
        near_elev = (
            (height >= ELEVATED_MIN_M)
            & (height <= ELEVATED_MAX_M)
            & (fwd_cam > 0.05)
            & (robot_forward > 0.05)
            & (robot_forward <= FLOOR_EVIDENCE_RANGE_M + 0.3)
        )
        if floor_points and np.any(near_elev):
            elev_cells = {
                (int(ci), int(cj))
                for ci, cj in zip(
                    np.round(robot_forward[near_elev] / 0.06).astype(np.int32),
                    np.round(robot_lateral[near_elev] / 0.06).astype(np.int32),
                )
            }
            kept = []
            for pf, pl in floor_points:
                ci, cj = int(round(pf / 0.06)), int(round(pl / 0.06))
                if any(
                    (ci + di, cj + dj) in elev_cells
                    for di in (-2, -1, 0, 1, 2)
                    for dj in (-2, -1, 0, 1, 2)
                ):
                    continue
                kept.append((pf, pl))
            floor_points = tuple(kept)

    nearest_gate_m: float | None = None
    nearest_gate_narrow_m: float | None = None
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
        fp_candidate = elevated & (robot_forward <= 1.8) & col_ok
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
            narrow = corridor & (np.abs(robot_lateral) <= CORRIDOR_NARROW_HALF_WIDTH_M)
            if np.any(narrow):
                nearest_gate_narrow_m = float(np.min(np.where(narrow, robot_forward, np.inf)))
        # Map only the near boundary of the closest connected elevated
        # component.  Taking every pixel in `footprint` stamps a whole
        # tabletop/cabinet face as occupancy; a 2-D planner needs its leading
        # edge, not its visible surface area.  The connected-component guard
        # also prevents two nearby objects from being joined across open floor.
        near_ref = nearest_gate_m if nearest_gate_m is not None else nearest_any_m
        band = footprint & (robot_forward <= near_ref + FOOTPRINT_BOUNDARY_DEPTH_M)
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
                    # Decimate this 10cm leading band, not the entire
                    # elevated surface.  It produces a thin obstacle boundary
                    # in the world map and leaves the adjacent free floor
                    # available for path planning.
                    region_points = _decimate_xy(
                        robot_forward[component], robot_lateral[component]
                    )
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
            # Red: camera-owned suspended collision geometry. Orange:
            # floor-connected geometry intentionally delegated to lidar. The
            # distinct colors make a doorway/table separation visible during
            # a live run instead of looking like one giant red obstacle.
            overlay[ground_connected] = (
                0.45 * overlay[ground_connected] + 0.55 * np.array([0, 150, 255])
            ).astype(np.uint8)
            overlay[elevated] = (0.35 * overlay[elevated] + 0.65 * np.array([0, 0, 220])).astype(np.uint8)
            overlay[floorish] = (0.7 * overlay[floorish] + 0.3 * np.array([0, 160, 0])).astype(np.uint8)
            text = (
                "no hough edge"
                if proposal_line_xy is None
                else "clear" if nearest_gate_m is None else f"elevated {nearest_gate_m:.2f}m"
            )
            if nearest_gate_m is not None:
                text += (
                    " nw=-"
                    if nearest_gate_narrow_m is None
                    else f" nw={nearest_gate_narrow_m:.2f}"
                )
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
            if proposal_line_xy is not None:
                cv2.line(
                    overlay,
                    proposal_line_xy[0],
                    proposal_line_xy[1],
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

    return DepthEyeResult(
        eye=str(eye),
        frame_monotonic=float(frame_monotonic),
        scale=scale,
        scale_source=scale_source,
        proposal_detected=proposal_line_xy is not None,
        proposal_score=float(proposal_score),
        proposal_line_xy=proposal_line_xy,
        nearest_gate_m=nearest_gate_m,
        nearest_gate_narrow_m=nearest_gate_narrow_m,
        nearest_any_m=nearest_any_m,
        edge_height_m=edge_height_m,
        bearing_deg=bearing_deg,
        segment_p1=seg_p1,
        segment_p2=seg_p2,
        region_points_xy=region_points,
        floor_points_xy=floor_points,
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
    """Daemon thread: runs depth perception and publishes DepthEyeResults.

    Two modes:
      - per-eye (mosaic=None): alternately analyzes each eye's freshest frame
        through its own CameraModel (legacy geometry: yaw +-15, no roll).
      - panorama (mosaic=PerceptionMosaic): each cycle grabs BOTH eyes'
        freshest near-simultaneous frames, hard-cut fuses them through the
        CALIBRATED virtual forward camera, and runs ONE inference on the
        single central view — the result is published under the "panorama"
        key. One inference instead of two alternating ones roughly HALVES the
        gate's result staleness, and the narrow squeeze corridor is measured
        dead-center where the fused view is most accurate.

    Model load failures set .failure and the thread exits — consumers fail
    safe (deny forward), never silently fall back."""

    def __init__(
        self,
        subscriber,
        eye_models: dict[str, CameraModel],
        *,
        lidar_ranges_fn=None,
        edge_detector_config: ElevatedEdgeScanConfig | None = None,
        model_name: str = DEFAULT_DEPTH_MODEL,
        max_frame_age_s: float = 1.0,
        mosaic=None,
    ) -> None:
        self._subscriber = subscriber
        self._eye_models = dict(eye_models)
        self._lidar_ranges_fn = lidar_ranges_fn
        self._proposal_detector = ElevatedEdgeDetector(
            edge_detector_config or ElevatedEdgeScanConfig()
        )
        self._estimator = DepthEstimator(model_name)
        self._max_frame_age_s = float(max_frame_age_s)
        # PerceptionMosaic (sourccey_eye_panorama) or None for per-eye mode.
        self._mosaic = mosaic
        if mosaic is not None:
            self._eye_models = {"panorama": mosaic.model}
        self._lock = threading.Lock()
        self._results: dict[str, DepthEyeResult] = {}
        self.ready = False
        self.failure: str | None = None
        self.inference_s: float = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def eye_keys(self) -> tuple[str, ...]:
        """The result keys this worker publishes ("panorama",) or the eyes."""
        return tuple(self._eye_models.keys())

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
            if self._mosaic is not None:
                eye = "panorama"
                left, age_l = self._subscriber.latest("front_left")
                right, age_r = self._subscriber.latest("front_right")
                if (
                    left is None
                    or right is None
                    or age_l is None
                    or age_r is None
                    or max(age_l, age_r) > self._max_frame_age_s
                    or abs(float(age_l) - float(age_r)) > 0.35
                ):
                    # Both eyes must be fresh AND near-simultaneous: fusing
                    # frames from different moments puts the two halves of
                    # the world at different times.
                    time.sleep(0.05)
                    continue
                frame = self._mosaic.compose(left, right)
                frame_monotonic = time.monotonic() - float(max(age_l, age_r))
                seam_cols = self._mosaic.seam_cols
                coverage = self._mosaic.coverage()
            else:
                eye = eyes[eye_index % len(eyes)]
                eye_index += 1
                frame, age_s = self._subscriber.latest(eye)
                if frame is None or age_s is None or age_s > self._max_frame_age_s:
                    time.sleep(0.05)
                    continue
                frame_monotonic = time.monotonic() - float(age_s)
                seam_cols = None
                coverage = None
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
                if self._mosaic is not None:
                    # The Hough proposal detector is threshold-tuned for the
                    # 320px-wide per-eye frames: run it on a 320-wide copy of
                    # the mosaic (same calibrated regime) and scale the line
                    # back up to mosaic pixels.
                    mh, mw = frame.shape[:2]
                    det_w = 320
                    det_h = max(1, int(round(mh * det_w / mw)))
                    det_frame = cv2.resize(frame, (det_w, det_h))
                    proposal = self._proposal_detector.detect(det_frame)
                    proposal_line = None
                    if proposal.detected and proposal.line_xy is not None:
                        sx, sy = mw / det_w, mh / det_h
                        (px1, py1), (px2, py2) = proposal.line_xy
                        proposal_line = (
                            (int(round(px1 * sx)), int(round(py1 * sy))),
                            (int(round(px2 * sx)), int(round(py2 * sy))),
                        )
                else:
                    proposal = self._proposal_detector.detect(frame)
                    proposal_line = proposal.line_xy if proposal.detected else None
                depth = self._estimator.infer(frame)
                result = analyze_depth(
                    self._eye_models[eye],
                    depth,
                    eye=eye,
                    frame_monotonic=frame_monotonic,
                    frame_bgr=frame,
                    lidar_bearings_deg=lidar_b,
                    lidar_ranges_m=lidar_r,
                    proposal_line_xy=proposal_line,
                    proposal_score=float(proposal.score),
                    seam_cols=seam_cols,
                    coverage_mask=coverage,
                )
            except Exception as exc:
                self.failure = f"{type(exc).__name__}: {exc}"
                return
            self.inference_s = time.monotonic() - started
            with self._lock:
                self._results[eye] = result
