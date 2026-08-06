"""2D pose graph and Manhattan-axis machinery for the Sourccey explorer.

This is the professional-SLAM core the explorer was missing: stationary
keyframe poses become VARIABLES connected by measured constraints instead of
irreversible facts, and the occupancy grid becomes a rendering of the
optimized scan list.  Three constraint types, deliberately no more:

* BETWEEN factors — odometry carried between consecutive stops (bottom-camera
  flow + gyro), expressed in the earlier node's frame.
* POSE factors — a stationary scan-match consensus against the map.  Position
  information is ANISOTROPIC, derived from the scan's wall-direction
  distribution: a corridor's parallel walls constrain the lateral coordinate
  tightly while leaving the along-corridor coordinate to odometry.  This is
  exactly the observability split whose absence caused the 2026-08-05
  30-46cm discard/partial-accept oscillation.
* AXIS priors — the Manhattan-world heading compass.  Buildings have two
  dominant wall directions 90 degrees apart; a stationary scan whose wall
  histogram peaks within a small tolerance of a building axis yields an
  absolute heading measurement that, unlike the gyro, cannot drift.

The optimizer is plain dense Gauss-Newton over SE(2) (Grisetti et al.,
"A Tutorial on Graph-Based SLAM").  Stationary keyframes number ~100 per
mission, so a dense 3Nx3N solve is microseconds; no sparse machinery, no
external solver dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "PoseGraph2D",
    "axis_snap_delta_deg",
    "dominant_axis_deg",
    "wall_direction_masses",
    "wrap_deg",
    "wrap_deg_90",
]


def wrap_deg(angle: float) -> float:
    """Wrap to (-180, 180]."""
    return -((-float(angle) + 180.0) % 360.0 - 180.0)


def wrap_deg_90(angle: float) -> float:
    """Wrap an axis-family angle to (-45, 45] (axes repeat every 90 deg)."""
    return -((-float(angle) + 45.0) % 90.0 - 45.0)


# ---------------------------------------------------------------------------
# Wall-direction statistics.
#
# LiDAR points arrive angularly ordered, so consecutive returns on one wall
# are physical neighbours: the orientation of each short consecutive-pair
# segment votes for a wall direction, weighted by its length.  This gives a
# robust orientation histogram without any neighbour search.
# ---------------------------------------------------------------------------


def _segment_votes(
    points_xy: np.ndarray,
    *,
    baseline: int = 5,
    max_seg_m: float = 0.45,
    min_seg_m: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Orientations (deg, mod 180) and length weights of chord segments.

    Chords span ``baseline`` consecutive returns rather than one: adjacent
    lidar points are a few centimetres apart with ~0.5-1cm range noise, so a
    single-step segment's orientation is noise (±10-15 deg), while a ~15-30cm
    chord is accurate to a couple of degrees.  The length gates drop chords
    that jump across wall boundaries or gaps.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if len(pts) < baseline + 2:
        return np.zeros(0), np.zeros(0)
    delta = pts[baseline:] - pts[:-baseline]
    length = np.hypot(delta[:, 0], delta[:, 1])
    keep = (length >= min_seg_m) & (length <= max_seg_m)
    if not np.any(keep):
        return np.zeros(0), np.zeros(0)
    orientation = np.degrees(np.arctan2(delta[keep, 1], delta[keep, 0])) % 180.0
    return orientation, length[keep]


def _axis_histogram(
    orientation_deg: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """Length-weighted, circularly smoothed 1-degree histogram mod 90."""
    bins = np.zeros(90, dtype=np.float64)
    idx = np.floor(orientation_deg % 90.0).astype(np.int64) % 90
    np.add.at(bins, idx, weight)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    padded = np.concatenate([bins[-2:], bins, bins[:2]])
    return np.convolve(padded, kernel, mode="valid")


def dominant_axis_deg(
    points_xy: np.ndarray,
    *,
    min_votes: int = 40,
    min_mass_fraction: float = 0.45,
    mass_half_width_deg: float = 4.0,
) -> tuple[float, float] | None:
    """Dominant wall axis of a point set, mod 90.

    Returns ``(axis_deg, mass_fraction)`` where ``mass_fraction`` is the
    share of total wall length lying within ``mass_half_width_deg`` of the
    peak (folded mod 90).  Returns None when the environment does not have a
    confident rectilinear structure — the caller must then leave headings to
    the gyro machinery rather than force a Manhattan assumption onto a
    non-Manhattan room.
    """
    orientation, weight = _segment_votes(points_xy)
    if len(orientation) < int(min_votes) or float(weight.sum()) <= 0.0:
        return None
    smoothed = _axis_histogram(orientation, weight)
    peak = int(np.argmax(smoothed))
    folded = (orientation - peak) % 90.0
    folded = np.where(folded > 45.0, folded - 90.0, folded)
    near = np.abs(folded) <= float(mass_half_width_deg)
    mass = float(weight[near].sum()) / float(weight.sum())
    if mass < float(min_mass_fraction):
        return None
    # Refine with the circular mean of the near-peak votes.
    refined = float(peak) + float(
        np.average(folded[near], weights=weight[near])
    )
    return refined % 90.0, mass


def axis_snap_delta_deg(
    points_xy: np.ndarray,
    building_axis_deg: float,
    *,
    tolerance_deg: float = 7.0,
    min_votes: int = 40,
    min_mass_fraction: float = 0.45,
) -> float | None:
    """Rotation (deg) that aligns a scan's dominant wall axis to the building.

    Positive result means: rotate the scan/pose by this much.  None when the
    scan has no confident axis or the required correction exceeds
    ``tolerance_deg`` — a large disagreement is real geometry (angled wall,
    furniture pile), never something to force straight.  The tolerance is
    deliberately drift-sized: at 7 degrees only errors are corrected and
    genuinely diagonal structure is left alone.
    """
    result = dominant_axis_deg(
        points_xy,
        min_votes=min_votes,
        min_mass_fraction=min_mass_fraction,
    )
    if result is None:
        return None
    scan_axis, _mass = result
    delta = wrap_deg_90(float(building_axis_deg) - scan_axis)
    if abs(delta) > float(tolerance_deg):
        return None
    return float(delta)


def wall_direction_masses(
    points_xy: np.ndarray, building_axis_deg: float
) -> tuple[float, float] | None:
    """Wall-length mass parallel to each building axis (axis0, axis0+90).

    A wall PARALLEL to an axis constrains translation PERPENDICULAR to it,
    so the caller maps these to position information accordingly.  Returns
    None when there are too few wall segments to say anything.
    """
    orientation, weight = _segment_votes(points_xy)
    if len(orientation) < 20 or float(weight.sum()) <= 0.0:
        return None
    fold = (orientation - float(building_axis_deg)) % 180.0
    along_axis0 = (fold <= 20.0) | (fold >= 160.0)
    along_axis1 = np.abs(fold - 90.0) <= 20.0
    total = float(weight.sum())
    return (
        float(weight[along_axis0].sum()) / total,
        float(weight[along_axis1].sum()) / total,
    )


# ---------------------------------------------------------------------------
# SE(2) pose graph.
# ---------------------------------------------------------------------------


def _rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _wrap_rad(a: float) -> float:
    return -((-a + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass(slots=True)
class _Between:
    i: int
    j: int
    z: np.ndarray  # (dx, dy, dtheta_rad) in frame of node i
    info: np.ndarray  # 3x3


@dataclass(slots=True)
class _PosePrior:
    i: int
    z: np.ndarray  # (x, y, theta_rad)
    info: np.ndarray  # 3x3 (position block may be anisotropic, world frame)


@dataclass(slots=True)
class PoseGraph2D:
    """Dense Gauss-Newton SE(2) pose graph, degrees at the API surface."""

    nodes: list[np.ndarray] = field(default_factory=list)
    fixed: list[bool] = field(default_factory=list)
    _betweens: list[_Between] = field(default_factory=list)
    _priors: list[_PosePrior] = field(default_factory=list)

    def add_node(self, x: float, y: float, theta_deg: float, *, fixed: bool = False) -> int:
        self.nodes.append(
            np.array([float(x), float(y), math.radians(float(theta_deg))])
        )
        self.fixed.append(bool(fixed))
        return len(self.nodes) - 1

    def node_pose_deg(self, i: int) -> tuple[float, float, float]:
        x, y, th = self.nodes[i]
        return float(x), float(y), math.degrees(float(th))

    def add_between(
        self,
        i: int,
        j: int,
        dx: float,
        dy: float,
        dtheta_deg: float,
        *,
        sigma_xy_m: float,
        sigma_theta_deg: float,
    ) -> None:
        """Odometry: node j observed from node i (delta in node i's frame)."""
        info = np.diag(
            [
                1.0 / max(1e-4, float(sigma_xy_m)) ** 2,
                1.0 / max(1e-4, float(sigma_xy_m)) ** 2,
                1.0 / math.radians(max(1e-3, float(sigma_theta_deg))) ** 2,
            ]
        )
        self._betweens.append(
            _Between(
                int(i),
                int(j),
                np.array([float(dx), float(dy), math.radians(float(dtheta_deg))]),
                info,
            )
        )

    def add_pose_prior(
        self,
        i: int,
        x: float,
        y: float,
        theta_deg: float,
        *,
        position_info: np.ndarray,
        sigma_theta_deg: float | None,
    ) -> None:
        """Absolute fix (scan-match consensus).  ``position_info`` is the 2x2
        world-frame information matrix — anisotropic for corridors.  Pass
        ``sigma_theta_deg=None`` to leave heading unconstrained here (an axis
        prior owns it instead)."""
        info = np.zeros((3, 3))
        info[:2, :2] = np.asarray(position_info, dtype=np.float64)
        if sigma_theta_deg is not None:
            info[2, 2] = 1.0 / math.radians(max(1e-3, float(sigma_theta_deg))) ** 2
        self._priors.append(
            _PosePrior(
                int(i),
                np.array([float(x), float(y), math.radians(float(theta_deg))]),
                info,
            )
        )

    def add_heading_prior(self, i: int, theta_deg: float, *, sigma_deg: float) -> None:
        """Absolute heading (Manhattan axis compass)."""
        info = np.zeros((3, 3))
        info[2, 2] = 1.0 / math.radians(max(1e-3, float(sigma_deg))) ** 2
        self._priors.append(
            _PosePrior(
                int(i),
                np.array([0.0, 0.0, math.radians(float(theta_deg))]),
                info,
            )
        )

    # -- solver ------------------------------------------------------------

    def optimize(self, *, max_iterations: int = 12, damping: float = 1e-6) -> float:
        """Gauss-Newton; returns the final total chi-square."""
        n = len(self.nodes)
        if n == 0:
            return 0.0
        free = [k for k in range(n) if not self.fixed[k]]
        if not free:
            return self._chi2()
        col = {node: 3 * slot for slot, node in enumerate(free)}
        chi2 = 0.0
        for _ in range(int(max_iterations)):
            dim = 3 * len(free)
            h_mat = np.zeros((dim, dim))
            b_vec = np.zeros(dim)
            chi2 = 0.0
            for factor in self._betweens:
                e, a_jac, b_jac = self._between_error(factor)
                chi2 += float(e @ factor.info @ e)
                blocks = []
                if factor.i in col:
                    blocks.append((col[factor.i], a_jac))
                if factor.j in col:
                    blocks.append((col[factor.j], b_jac))
                for c1, j1 in blocks:
                    b_vec[c1 : c1 + 3] += j1.T @ factor.info @ e
                    for c2, j2 in blocks:
                        h_mat[c1 : c1 + 3, c2 : c2 + 3] += j1.T @ factor.info @ j2
            for prior in self._priors:
                e = self.nodes[prior.i] - prior.z
                e[2] = _wrap_rad(e[2])
                chi2 += float(e @ prior.info @ e)
                if prior.i in col:
                    c1 = col[prior.i]
                    b_vec[c1 : c1 + 3] += prior.info @ e
                    h_mat[c1 : c1 + 3, c1 : c1 + 3] += prior.info
            h_mat += float(damping) * np.eye(dim)
            try:
                dx = np.linalg.solve(h_mat, -b_vec)
            except np.linalg.LinAlgError:
                break
            for node, c in col.items():
                self.nodes[node] += dx[c : c + 3]
                self.nodes[node][2] = _wrap_rad(self.nodes[node][2])
            if float(np.max(np.abs(dx))) < 1e-6:
                break
        return self._chi2()

    def _chi2(self) -> float:
        total = 0.0
        for factor in self._betweens:
            e, _a, _b = self._between_error(factor)
            total += float(e @ factor.info @ e)
        for prior in self._priors:
            e = self.nodes[prior.i] - prior.z
            e[2] = _wrap_rad(e[2])
            total += float(e @ prior.info @ e)
        return total

    def _between_error(
        self, factor: _Between
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xi = self.nodes[factor.i]
        xj = self.nodes[factor.j]
        ri = _rotation(xi[2])
        rz = _rotation(factor.z[2])
        dt = xj[:2] - xi[:2]
        e = np.empty(3)
        e[:2] = rz.T @ (ri.T @ dt - factor.z[:2])
        e[2] = _wrap_rad(xj[2] - xi[2] - factor.z[2])
        dri_t = np.array(
            [
                [-math.sin(xi[2]), math.cos(xi[2])],
                [-math.cos(xi[2]), -math.sin(xi[2])],
            ]
        )
        a_jac = np.zeros((3, 3))
        a_jac[:2, :2] = -rz.T @ ri.T
        a_jac[:2, 2] = rz.T @ dri_t @ dt
        a_jac[2, 2] = -1.0
        b_jac = np.zeros((3, 3))
        b_jac[:2, :2] = rz.T @ ri.T
        b_jac[2, 2] = 1.0
        return e, a_jac, b_jac
