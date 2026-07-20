"""``SelfMaskedLidarFeed`` — a LiDAR feed that filters the robot's OWN returns.

The passive arms/grippers/shell sit inside the LiDAR's spin, so a raw scan
includes a fixed spray of very-close points that are the robot itself, not the
world. Those self-hits ate the stop-box trigger headroom, cast shadows that
halved some captures, and smeared noise into the stitched map. This subclass
calibrates ONCE at startup while stationary: it histograms which angular bins
persistently return closer than a threshold (those are the arms), and from then
on drops any point in a masked bin at/under the observed distance. Anything seen
BEYOND the arm on the same bearing still passes through. It overrides only
``latest()`` — the single choke point every consumer (mapping, stop box, frontier
detection, turn tracking) reads through — so one mask covers the whole system.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ldlidar_direct_snapshot_client import DirectLidarFeed

class SelfMaskedLidarFeed(DirectLidarFeed):
    """LiDAR feed that filters the robot's OWN returns (arms/grippers/shell in
    the beam) out of every frame, at the single choke point all consumers
    share (``wait_for_frame_after`` routes through ``latest``).

    Field 2026-07-18 (user: "the arms are a bit in the way of the lidar"): the
    arms put ~25 permanent points in the stop box (eating its trigger
    headroom), cast shadows that halved some captures, and smeared noise blobs
    into the stitched map. The mask is calibrated ONCE at startup while the
    robot is stationary: angular bins that persistently return closer than
    ``max_self_range_m`` are self-hits, and future points in those bins at or
    below the observed distance (+margin) are dropped. Anything seen BEYOND
    the arm in the same bearing still passes through.
    """

    MASK_BIN_WIDTH_DEG = 2.0

    def __init__(self, host: str, port: int) -> None:
        """Wrap the normal LiDAR feed and start with NO mask (passthrough).

        Calls the base feed's constructor, then sets up empty mask state:
        ``_mask_cutoffs`` (a per-bearing-bin drop distance, filled in later by
        ``calibrate_self_mask``; NaN means that bin is unmasked) and a one-frame
        cache so re-masking the same revolution twice is free. Until calibration
        runs, every frame passes through untouched.
        """
        super().__init__(host, port)
        self._mask_cutoffs: np.ndarray | None = None  # per-bin drop distance; NaN=open
        self._mask_cache_id: int | None = None
        self._mask_cache_frame = None

    def calibrate_self_mask(
        self,
        sample_frames,
        *,
        max_self_range_m: float = 0.50,
        margin_m: float = 0.20,
    ) -> tuple[int, float]:
        """Build the mask from stationary frames. Returns (masked_bins, masked_deg)."""
        n_bins = int(round(360.0 / self.MASK_BIN_WIDTH_DEG))
        per_bin: list[list[float]] = [[] for _ in range(n_bins)]
        for frame in sample_frames:
            for angle_deg, distance_m, _conf in frame.points:
                d = float(distance_m)
                if 0.0 < d < float(max_self_range_m):
                    b = int((float(angle_deg) % 360.0) / self.MASK_BIN_WIDTH_DEG) % n_bins
                    per_bin[b].append(d)
        # Persistent = seen in at least half the sample frames; transient specks
        # must not blind a bearing.
        min_hits = max(2, len(sample_frames) // 2)
        cutoffs = np.full(n_bins, np.nan, dtype=np.float64)
        for b, vals in enumerate(per_bin):
            if len(vals) >= min_hits:
                cutoffs[b] = min(0.60, max(vals) + float(margin_m))
        # Expand THREE bins (6deg) each side: the mask is calibrated stationary,
        # but the passive arms SWING while driving (field 2026-07-18 run 11:
        # calibrated mask left 6-14 jitter points leaking back per frame during
        # motion, and with the baseline at 0 that blocked EVERY drive burst at
        # 0.05m — the creep-and-turn doom loop). Stationary bearings +-6deg and
        # +0.20m of range slack cover the swing envelope.
        expanded = cutoffs.copy()
        for b in range(n_bins):
            if np.isnan(cutoffs[b]):
                neighbors = [
                    cutoffs[(b + off) % n_bins]
                    for off in (-3, -2, -1, 1, 2, 3)
                ]
                neighbors = [v for v in neighbors if not np.isnan(v)]
                if neighbors:
                    expanded[b] = max(neighbors)
        self._mask_cutoffs = expanded
        masked_bins = int(np.count_nonzero(~np.isnan(expanded)))
        return masked_bins, masked_bins * self.MASK_BIN_WIDTH_DEG

    def _apply_mask(self, frame_id: int, frame):
        """Drop this frame's self-hit points, reusing the cache within a frame.

        If no mask is calibrated yet (or the frame is None) it returns unchanged.
        Otherwise, for every point it looks up the cutoff distance for that
        point's bearing bin and discards the point if it is at/under the cutoff
        (that's the arm); points beyond the cutoff on the same bearing survive
        (that's the world behind the arm). If nothing was dropped it returns the
        original frame object; otherwise a copy with the trimmed point list. The
        result is cached by ``frame_id`` so repeated ``latest()`` calls on the
        same revolution don't re-filter.
        """
        if frame is None or self._mask_cutoffs is None:
            return frame
        if self._mask_cache_id == frame_id:
            return self._mask_cache_frame
        n_bins = len(self._mask_cutoffs)
        kept = []
        for point in frame.points:
            cutoff = self._mask_cutoffs[
                int((float(point[0]) % 360.0) / self.MASK_BIN_WIDTH_DEG) % n_bins
            ]
            if not np.isnan(cutoff) and float(point[1]) <= cutoff:
                continue
            kept.append(point)
        masked = (
            frame
            if len(kept) == len(frame.points)
            else dataclasses.replace(frame, points=kept)
        )
        self._mask_cache_id = frame_id
        self._mask_cache_frame = masked
        return masked

    def latest(self):
        """The one override that makes the mask universal.

        Every consumer in the system (mapping, stop box, frontier detection, turn
        tracking) reads frames through ``latest()``/``wait_for_frame_after``, and
        both funnel here. By masking at this single choke point, the arms are
        filtered out for the whole system with no per-caller changes: grab the
        base feed's newest frame, run it through ``_apply_mask``, hand back the
        cleaned version.
        """
        frame_id, frame = super().latest()
        return frame_id, self._apply_mask(frame_id, frame)


