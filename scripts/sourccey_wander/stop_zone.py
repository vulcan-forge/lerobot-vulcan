"""The collision "stop box": the wedge of space directly around the robot where
a LiDAR return means "do not drive."

``StopZoneConfig`` holds the box geometry (how far forward it reaches, its width
and thickness, how many points trip it) plus the NARROW "squeeze" variant used
when threading a doorway, and a startup-measured baseline of the robot's own
residual self-hits. ``_point_in_stop_zone`` answers "is this one polar scan
return inside the box"; ``_blocked_points_for_frame`` counts how many returns in
a frame are inside it, which the drive loop compares against the trigger count to
decide whether to halt. Pure geometry — it decides nothing about WHERE to go,
only whether the immediate path is fouled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sourccey_wander.wander_types import _normalize_angle_deg

@dataclass(slots=True)
class StopZoneConfig:
    forward_angle_deg: float
    min_distance_m: float
    tripwire_distance_m: float
    tripwire_half_width_m: float
    tripwire_thickness_m: float
    min_points_to_trigger: int
    # Points the lidar permanently sees inside the box (robot's own shell /
    # fixtures), measured at startup. Blocking triggers only on points ABOVE
    # this baseline — otherwise a self-seeing lidar reports "blocked" at every
    # heading and the robot never drives.
    baseline_points: int = 0
    # SQUEEZE/EXIT-RUN mode: passing a doorway inherently clips the door-frame
    # posts into the box's lateral EDGES — field 2026-07-18: exit probes were
    # vetoed at baseline+6..20 points that were all frame edges the body would
    # clear, while a REAL frontal wall reads +30..60 in the box CENTER. Instead
    # of raising the count threshold (which erodes real collision response),
    # squeeze mode NARROWS the box laterally to just past the body's own
    # half-width: frame posts the robot physically clears leave the count
    # entirely, anything actually in the body's path still triggers at full
    # sensitivity. The narrow band needs its own self-hit baseline, measured at
    # startup alongside the full-box one.
    squeeze_active: bool = False
    squeeze_half_width_m: float = 0.24
    squeeze_baseline_points: int = 0

    def effective_half_width_m(self) -> float:
        """The box's current lateral half-width: the NARROW squeeze value when
        squeeze/exit mode is active, otherwise the normal tripwire width. This is
        the one knob that switches the box between "full body-guard" and "thread
        the doorway" behavior."""
        return float(
            self.squeeze_half_width_m if self.squeeze_active else self.tripwire_half_width_m
        )

    def blocked_trigger_count(self) -> int:
        """How many in-box points mean "stop" right now. It is the relevant
        self-hit baseline (squeeze baseline when squeezing, else the full-box
        one) PLUS the minimum genuinely-new points required to trip. Comparing a
        frame's count against this is the actual collision decision."""
        base = self.squeeze_baseline_points if self.squeeze_active else self.baseline_points
        return int(base) + int(self.min_points_to_trigger)




def _point_in_stop_zone(angle_deg: float, distance_m: float, cfg: StopZoneConfig) -> bool:
    """Is one polar LiDAR return (bearing + range) inside the stop box?

    Rotate the point's bearing so "straight ahead" is 0deg, then split its range
    into a forward component and a lateral component (project onto the robot's
    facing axis). The point counts as "in the box" when it is: far enough forward
    to clear the near edge (``min_distance_m``), not past the far edge
    (``tripwire_distance_m`` plus half the box thickness), and within the current
    lateral half-width. All three must hold — a rectangle of danger directly in
    front of the robot.
    """
    delta_deg = _normalize_angle_deg(float(angle_deg) - float(cfg.forward_angle_deg))
    theta = math.radians(delta_deg)
    forward_m = float(distance_m) * math.cos(theta)
    lateral_m = float(distance_m) * math.sin(theta)
    near_edge_m = max(0.0, float(cfg.min_distance_m))
    far_edge_m = max(near_edge_m, float(cfg.tripwire_distance_m) + float(cfg.tripwire_thickness_m) / 2.0)
    return (
        forward_m >= near_edge_m
        and forward_m <= far_edge_m
        and abs(lateral_m) <= cfg.effective_half_width_m()
    )


def _blocked_points_for_frame(frame, zone_cfg: StopZoneConfig) -> int:
    """Count how many returns in a whole LiDAR frame land inside the stop box.

    Runs ``_point_in_stop_zone`` over every point in the revolution and tallies
    the hits. The drive loop compares this tally against
    ``zone_cfg.blocked_trigger_count()`` to decide whether the path ahead is
    fouled and it must halt.
    """
    return sum(
        1
        for angle_deg, distance_m, _confidence in frame.points
        if _point_in_stop_zone(float(angle_deg), float(distance_m), zone_cfg)
    )


