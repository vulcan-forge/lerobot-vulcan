"""Projective camera geometry for Sourccey's collision cameras.

Pure math (numpy only, no lerobot imports) so it is unit-testable anywhere.

Frames and conventions:
  - Robot frame: +x forward, bearing in degrees, positive toward +y
    (whether +y is physically left or right depends on the lidar decode; the
    `bearing_sign` field flips the camera bearings to match the lidar's local
    frame — verify once with the calibration script).
  - Image: x_ratio in [0,1] left->right, y_ratio in [0,1] top->bottom.
  - Depression angle: downward tilt of a viewing ray below horizontal;
    positive = looking down at the floor.

Measured hardware (2026-07-09, from the operator):
  - Eye cameras: 36 in (0.914 m) above floor, pitched 20 deg down, yawed
    15 deg outward, 1.5 in apart, lenses ~8 in (0.203 m) ahead of robot
    center. 320x240.
  - Bottom camera: 5 in (0.127 m) above floor, pointing straight ahead.
  - FOVs are NOT measured — defaults are typical UVC values; calibrate with
    scripts/sourccey_camera_calibration.py against a lidar-known wall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CameraModel:
    name: str
    height_m: float
    pitch_down_deg: float  # positive = tilted toward the floor
    yaw_deg: float  # positive toward +bearing (see bearing_sign)
    forward_offset_m: float  # camera position ahead of robot center
    lateral_offset_m: float = 0.0
    hfov_deg: float = 62.0
    vfov_deg: float = 48.0
    # Flip if the lidar local frame's +bearing is the camera's image-right.
    bearing_sign: float = 1.0

    # -- rows <-> rays ------------------------------------------------------
    def depression_deg_for_row(self, y_ratio: float) -> float:
        return float(self.pitch_down_deg) + (float(y_ratio) - 0.5) * float(self.vfov_deg)

    def row_for_depression_deg(self, depression_deg: float) -> float:
        return 0.5 + (float(depression_deg) - float(self.pitch_down_deg)) / float(self.vfov_deg)

    def horizon_y_ratio(self) -> float:
        """Image row that looks exactly horizontal (the camera-height plane)."""
        return self.row_for_depression_deg(0.0)

    # -- floor projection ----------------------------------------------------
    def floor_distance_for_row(self, y_ratio: float) -> float | None:
        """Horizontal distance (from the camera) to the floor point seen at
        this row, IF the object sits on the floor. None at/above horizon."""
        depression = self.depression_deg_for_row(y_ratio)
        if depression <= 0.5:  # at/above horizon: the floor is not visible here
            return None
        return float(self.height_m) / math.tan(math.radians(depression))

    def row_for_floor_distance(self, distance_m: float) -> float | None:
        if distance_m <= 0.0:
            return None
        depression = math.degrees(math.atan2(float(self.height_m), float(distance_m)))
        return self.row_for_depression_deg(depression)

    def elevated_distance_for_row(self, y_ratio: float, edge_height_m: float) -> float | None:
        """Horizontal distance to an edge of the given height seen at this
        row. None if the ray cannot intersect that height (looking above it)."""
        drop_m = float(self.height_m) - float(edge_height_m)
        if drop_m <= 0.0:
            return None
        depression = self.depression_deg_for_row(y_ratio)
        if depression <= 0.5:
            return None
        return drop_m / math.tan(math.radians(depression))

    # -- columns <-> bearings -------------------------------------------------
    def bearing_deg_for_column(self, x_ratio: float) -> float:
        """Bearing of the viewing ray in the robot's local frame. Image-left
        is toward +yaw for a normal (non-mirrored) camera."""
        return float(self.bearing_sign) * (
            float(self.yaw_deg) - (float(x_ratio) - 0.5) * float(self.hfov_deg)
        )

    # -- calibration -----------------------------------------------------------
    def vfov_from_floor_observation(self, y_ratio: float, known_distance_m: float) -> float | None:
        """Solve vfov given ONE observed floor point at a known camera
        distance (e.g. the wall-floor boundary at a lidar-measured range)."""
        if known_distance_m <= 0.0 or abs(float(y_ratio) - 0.5) < 1e-6:
            return None
        depression = math.degrees(math.atan2(float(self.height_m), float(known_distance_m)))
        vfov = (depression - float(self.pitch_down_deg)) / (float(y_ratio) - 0.5)
        return float(vfov) if vfov > 5.0 else None


def edge_point_robot_frame(
    model: CameraModel, x_ratio: float, y_ratio: float, edge_height_m: float
) -> tuple[float, float] | None:
    """Project one image point of an edge (of known height) to robot-frame
    floor coordinates (forward_m, lateral_m). Lateral sign follows the
    bearing convention (+lateral toward +bearing)."""
    distance = model.elevated_distance_for_row(y_ratio, edge_height_m)
    if distance is None:
        return None
    bearing_rad = math.radians(model.bearing_deg_for_column(x_ratio))
    return (
        float(model.forward_offset_m) + distance * math.cos(bearing_rad),
        distance * math.sin(bearing_rad),
    )


def edge_segment_hazard_distances(
    model: CameraModel,
    *,
    p1_image: tuple[float, float],
    p2_image: tuple[float, float],
    edge_height_m: float,
    corridor_half_width_m: float,
    extend_p1: bool = False,
    extend_p2: bool = False,
) -> tuple[float | None, float | None]:
    """Collision geometry for a detected edge SEGMENT (image-ratio endpoints).

    Returns (corridor_distance_m, nearest_distance_m):
      - corridor_distance_m: forward distance at which the segment enters the
        corridor the robot's body sweeps while driving straight
        (|lateral| <= corridor_half_width). None if the segment never enters
        it — the robot drives PAST this edge. Exact for diagonal approaches:
        the near END of the edge governs, not its center.
      - nearest_distance_m: euclidean distance from robot center to the
        segment (for rotation/near-tier safety), None if unprojectable.
    extend_p1/p2: an endpoint at the image border means the edge continues
    out of view — extend that end far along the line, conservatively.
    """
    a = edge_point_robot_frame(model, p1_image[0], p1_image[1], edge_height_m)
    b = edge_point_robot_frame(model, p2_image[0], p2_image[1], edge_height_m)
    if a is None and b is None:
        return None, None
    if a is None:
        a = b
    if b is None:
        b = a
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length > 1e-6:
        ux, uy = dx / length, dy / length
        if extend_p1:
            ax, ay = ax - ux * 3.0, ay - uy * 3.0
        if extend_p2:
            bx, by = bx + ux * 3.0, by + uy * 3.0
        dx, dy = bx - ax, by - ay

    # Nearest euclidean distance from the origin (robot center) to segment.
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        nearest = math.hypot(ax, ay)
    else:
        t_near = max(0.0, min(1.0, -(ax * dx + ay * dy) / seg_len_sq))
        nearest = math.hypot(ax + t_near * dx, ay + t_near * dy)

    # Corridor intersection: p(t) = a + t*(b-a), t in [0,1], require
    # |lateral(t)| <= w and forward(t) >= 0; forward is linear in t so its
    # minimum over the feasible interval is at an interval endpoint.
    w = float(corridor_half_width_m)
    t0, t1 = 0.0, 1.0
    if abs(dy) < 1e-9:
        if abs(ay) > w:
            return None, nearest
    else:
        ta = (-w - ay) / dy
        tb = (w - ay) / dy
        lo, hi = min(ta, tb), max(ta, tb)
        t0, t1 = max(t0, lo), min(t1, hi)
        if t0 > t1:
            return None, nearest
    x_at_t0 = ax + t0 * dx
    x_at_t1 = ax + t1 * dx
    candidates_x = [x for x in (x_at_t0, x_at_t1) if x >= 0.0]
    if x_at_t0 < 0.0 <= x_at_t1 or x_at_t1 < 0.0 <= x_at_t0:
        candidates_x.append(0.0)
    if not candidates_x:
        return None, nearest  # entirely behind the robot
    return float(min(candidates_x)), nearest


def solve_edge_by_parallax(
    model: CameraModel,
    *,
    first_y_ratio: float,
    last_y_ratio: float,
    forward_travel_m: float,
    min_row_delta: float = 0.02,
) -> tuple[float, float] | None:
    """Range a horizontal edge from two sightings separated by known forward
    travel. Returns (distance_from_current_position_m, edge_height_m).

    Geometry: the camera keeps a fixed height; an edge with height drop
    h = cam_height - edge_height satisfies h = d1*tan(a1) = (d1 - b)*tan(a2),
    so d1 = b*tan(a2) / (tan(a2) - tan(a1)). Approaching means the edge's
    image row moves DOWN (a2 > a1); if it doesn't, there is no solution.
    Works for floor objects too (edge_height ~ 0), which is exactly how
    floor-vs-elevated is discriminated without a second camera.
    """
    if forward_travel_m <= 0.03:
        return None
    if (float(last_y_ratio) - float(first_y_ratio)) < float(min_row_delta):
        return None
    a1 = math.radians(model.depression_deg_for_row(first_y_ratio))
    a2 = math.radians(model.depression_deg_for_row(last_y_ratio))
    if a1 <= 0.0 or a2 <= a1:
        return None
    t1 = math.tan(a1)
    t2 = math.tan(a2)
    d1 = float(forward_travel_m) * t2 / (t2 - t1)
    d_now = d1 - float(forward_travel_m)
    drop_m = d1 * t1
    edge_height_m = float(model.height_m) - drop_m
    if d_now <= 0.0 or d_now > 8.0:
        return None
    return float(d_now), float(edge_height_m)


# ---------------------------------------------------------------------------
# Default Sourccey camera models.
# Heights/yaws: operator-measured 2026-07-09.
# Eye pitch + vfov: FIELD-CALIBRATED 2026-07-09 via the two-distance wall
# protocol (sourccey_camera_calibration.py): junction rows 0.812@1.328m and
# 0.698@1.793m solve to vfov=66.0deg, pitch=13.9deg (the "20deg" mounting
# estimate was ~6deg off). Predicted rows match measurements to <0.5%.
# Eye hfov: derived from vfov assuming a 4:3 sensor (~82deg) — refine if
# bearing-based lidar matching looks systematically shifted.
# Bottom pitch: field-calibrated (+3.34/+3.94 across the two distances).
# Bottom vfov is weakly constrained by far walls (rows stay near center);
# 48deg assumed — verify ground-gate distances against a tape measure.
# ---------------------------------------------------------------------------
def default_eye_left() -> CameraModel:
    return CameraModel(
        name="front_left",
        height_m=0.914,
        pitch_down_deg=13.9,
        yaw_deg=15.0,
        forward_offset_m=0.203,
        lateral_offset_m=0.019,
        hfov_deg=82.0,
        vfov_deg=66.0,
    )


def default_eye_right() -> CameraModel:
    return CameraModel(
        name="front_right",
        height_m=0.914,
        pitch_down_deg=13.9,
        yaw_deg=-15.0,
        forward_offset_m=0.203,
        lateral_offset_m=-0.019,
        hfov_deg=82.0,
        vfov_deg=66.0,
    )


def default_bottom() -> CameraModel:
    return CameraModel(
        name="bottom",
        height_m=0.127,
        pitch_down_deg=3.6,
        yaw_deg=0.0,
        forward_offset_m=0.20,
        lateral_offset_m=0.0,
    )
