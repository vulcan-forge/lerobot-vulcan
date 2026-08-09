"""Motion calibration: turn a COMMANDED motion into a calibrated dead-reckoned
ESTIMATE of the real motion.

The wander loop is closed-loop — it re-measures every burst with the LiDAR and
trusts that. This file matters only for the moments the LiDAR CANNOT measure
(scan-match lock lost mid-turn, a held drive burst, a blind maneuver at the
doorway). In those moments the old behavior was "assume the robot didn't move,"
which strands the pose and is what makes it thrash at the threshold. With the
factors measured by ``scripts/sourccey_wander_calibration.py`` (LiDAR ground
truth: e.g. a commanded 60deg turn really turns ~37deg), a blind maneuver gets an
HONEST estimate instead of a zero, so the next relocalization is seeded at the
right place and can re-lock.

The LiDAR stays the source of truth whenever it can see; this is strictly a
fallback for when it cannot. The IMU is not involved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class MotionCalibration:
    """Commanded->real scale factors. 1.0 means "no correction" (uncalibrated)."""

    translation_scale: float = 1.0   # real forward distance / commanded distance
    rotation_scale: float = 1.0      # real turn angle / commanded turn angle
    # Systematic mecanum drift of a "straight" drive, per metre of real forward
    # travel (LiDAR-measured). 0.0 = no measured drift. The wander loop's IMU
    # yaw-hold corrects drift in a CLOSED loop and does not require these; they are
    # a measured record and an optional feed-forward term.
    forward_yaw_drift_deg_per_m: float = 0.0
    forward_lateral_drift_m_per_m: float = 0.0
    source: str = "identity (uncalibrated)"

    @classmethod
    def load(cls, path: str | Path | None) -> "MotionCalibration":
        """Load the calibration JSON. Returns the identity (1.0/1.0) — never
        raises — if the path is missing or unreadable, so the wander loop always
        runs; a clear line is printed either way. A scale is only adopted when it
        is a sane positive number, so a malformed file degrades to uncalibrated
        rather than injecting nonsense."""
        if not path:
            print("[calib] no motion-calibration file given; motion dead-reckon is UNCALIBRATED (1.0x). "
                  "Run scripts/sourccey_wander_calibration.py to measure it.")
            return cls()
        p = Path(path)
        if not p.exists():
            print(f"[calib] motion-calibration file not found ({p}); dead-reckon UNCALIBRATED (1.0x). "
                  "Run scripts/sourccey_wander_calibration.py to create it.")
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[calib] could not read motion-calibration file ({p}: {exc}); dead-reckon UNCALIBRATED (1.0x).")
            return cls()

        def _sane(value, lo: float = 0.05, hi: float = 3.0) -> float | None:
            try:
                f = float(value)
            except (TypeError, ValueError):
                return None
            return f if (lo <= f <= hi) else None

        def _num(value, lo: float, hi: float) -> float | None:
            try:
                f = float(value)
            except (TypeError, ValueError):
                return None
            return f if (lo <= f <= hi) else None

        trans = _sane(data.get("translation_scale"))
        rot = _sane(data.get("rotation_scale_wheel"))
        yaw_drift = _num(data.get("forward_yaw_drift_deg_per_m"), -180.0, 180.0)
        lat_drift = _num(data.get("forward_lateral_drift_m_per_m"), -5.0, 5.0)
        cal = cls(
            translation_scale=trans if trans is not None else 1.0,
            rotation_scale=rot if rot is not None else 1.0,
            forward_yaw_drift_deg_per_m=yaw_drift if yaw_drift is not None else 0.0,
            forward_lateral_drift_m_per_m=lat_drift if lat_drift is not None else 0.0,
            source=str(p),
        )
        print(
            f"[calib] motion calibration loaded from {p.name}: "
            f"translation x{cal.translation_scale:.2f}, rotation x{cal.rotation_scale:.2f} "
            "(used ONLY to dead-reckon motion the LiDAR could not measure)."
        )
        return cal

    def deadreckon_turn_deg(self, commanded_deg: float) -> float:
        """The real rotation to EXPECT from a commanded turn the LiDAR couldn't track."""
        return float(commanded_deg) * float(self.rotation_scale)

    def deadreckon_forward_m(self, commanded_m: float) -> float:
        """The real distance to EXPECT from a commanded straight move the LiDAR couldn't track."""
        return float(commanded_m) * float(self.translation_scale)
