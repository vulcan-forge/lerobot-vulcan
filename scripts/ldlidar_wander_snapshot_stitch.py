from __future__ import annotations

import argparse
import dataclasses
import json
import math
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ldlidar_auto_snapshot_stitch import (
    _ensure_clean_directory,
    _execute_turn_burst,
    _init_rerun,
    _log_rerun_state,
    _send_stop,
    _wait_for_initial_frame,
)
from ldlidar_defaults import (
    DEFAULT_LIDAR_FORWARD_ANGLE_DEG,
    DEFAULT_LIDAR_STOP_BOX_DISTANCE_M,
    DEFAULT_LIDAR_STOP_BOX_HALF_WIDTH_M,
    DEFAULT_LIDAR_STOP_BOX_MIN_DISTANCE_M,
    DEFAULT_LIDAR_STOP_BOX_THICKNESS_M,
)
from ldlidar_direct_snapshot_client import DirectLidarFeed, _save_snapshot, _scan_to_local_points
from ldlidar_direct_snapshot_stitch import (
    HTML_TEMPLATE,
    MotionHint,
    Pose2D,
    Snapshot,
    _advance_pose,
    _append_stitch_snapshot,
    _load_snapshots,
    _build_exploration_grid,
    _build_score_grids,
    _plan_frontier_path,
    _pose_dict,
    _prior_weights_for_hint,
    _score_candidate,
    _search_pose,
    _solve_arc_pose_on_grids,
    _solve_turn_arc_pose,
    _stitch_snapshots,
    _transform_points,
)
from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient


# Trusted heading slack (deg) applied to the dead-reckoned anchor while the IMU
# is supplying yaw. Tight (the gyro is accurate over the few seconds of a turn),
# so the wide-theta recovery tiebreaker can reject a room-symmetric wrong mode
# that a lidar-only, drifted anchor would wave through.
IMU_DEAD_RECK_SLACK_DEG = 20.0


class ImuYawClient:
    """Subscribes to the host's integrated-yaw PUB socket and exposes the latest
    heading in degrees (sign-corrected to the SLAM CCW convention).

    Wholly optional and non-fatal: if zmq is unavailable, the socket never
    connects, or samples stop arriving, ``deg()`` returns ``None`` and every
    caller falls back to the prior lidar-only behavior. Only yaw DELTAS are ever
    consumed downstream, so unbounded gyro drift in the absolute value is fine.
    """

    def __init__(self, endpoint: str, *, sign: float = 1.0, stale_after_s: float = 4.0) -> None:
        # stale_after_s is generous on purpose: the samples arrive on a background
        # thread, but the main loop holds the GIL through the heavy stitch/solve
        # compute (rebuilds up to ~1.5s), starving that thread. Crucially, that
        # compute runs while the robot is STATIONARY, so a yaw sample from just
        # before it is still valid at the next turn's start (heading unchanged).
        # A truly dead feed is still caught — no messages at all -> stale ->
        # clean lidar fallback.
        self._endpoint = str(endpoint)
        self._sign = float(sign)
        self._stale_after_s = float(stale_after_s)
        self._lock = threading.Lock()
        self._yaw_deg: float | None = None
        self._last_rx_monotonic: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock = None
        # Sign self-check: compare IMU deltas against well-tracked lidar turns.
        self._sign_agree = 0
        self._sign_disagree = 0
        self._sign_warned = False

    def start(self) -> None:
        try:
            import zmq

            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.SUB)
            sock.setsockopt(zmq.SUBSCRIBE, b"")
            sock.setsockopt(zmq.CONFLATE, 1)
            sock.setsockopt(zmq.RCVTIMEO, 200)
            sock.connect(self._endpoint)
            self._sock = sock
        except Exception as exc:  # noqa: BLE001
            print(
                f"[wander] IMU yaw client disabled (connect failed: {exc}); "
                "heading is lidar-only"
            )
            self._sock = None
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="wander_imu_yaw")
        self._thread.start()
        print(
            f"[wander] IMU yaw heading prior ENABLED (endpoint={self._endpoint}, "
            f"sign={self._sign:+.0f}); waiting for first sample"
        )

    def _run(self) -> None:
        import zmq

        first = True
        ever_received = False
        # RCVTIMEO is 200ms, so 25 empty polls ~= 5s. Warn when nothing is
        # arriving, but THROTTLED (at ~5s, ~35s, then every ~5min): the IMU is
        # advisory-only (lidar owns mapping), so a dead feed degrades the
        # recovery veto but must not flood the log for the whole session
        # (field 2026-07-18: the every-5s version drowned an otherwise clean
        # 12-capture run). Almost always: the robot host is running old code
        # (no 'IMU yaw publisher bound' at startup) or its IMU failed to
        # connect ('IMU reporter disabled' warning) — the host console says.
        empty_polls = 0
        next_warn_at = 25
        while not self._stop.is_set():
            try:
                msg = self._sock.recv()
            except zmq.Again:
                if not ever_received:
                    empty_polls += 1
                    if empty_polls >= next_warn_at:
                        next_warn_at = empty_polls + (150 if next_warn_at == 25 else 1500)
                        print(
                            f"[wander] WARNING: no IMU yaw samples on {self._endpoint} after "
                            f"~{empty_polls // 5}s — the advisory heading check is inactive "
                            "(mapping is lidar-only regardless). Check the Pi robot host "
                            "console: it must print 'IMU yaw publisher bound' at startup; "
                            "'IMU reporter disabled' means the IMU wiring/connection failed."
                        )
                continue
            except Exception:  # noqa: BLE001
                continue
            try:
                data = json.loads(msg.decode("utf-8"))
                yaw_deg = float(data["yaw_deg"]) * self._sign
            except Exception:  # noqa: BLE001
                continue
            ever_received = True
            with self._lock:
                self._yaw_deg = yaw_deg
                self._last_rx_monotonic = time.monotonic()
            if first:
                first = False
                print("[wander] IMU yaw feed live (heading prior active)")

    def deg(self) -> float | None:
        """Latest sign-corrected yaw in degrees, or None if absent/stale."""
        with self._lock:
            if self._yaw_deg is None or self._last_rx_monotonic is None:
                return None
            if (time.monotonic() - self._last_rx_monotonic) > self._stale_after_s:
                return None
            return float(self._yaw_deg)

    def deg_fresh(self, *, wait_up_to_s: float = 0.4, max_age_s: float = 0.3) -> float | None:
        """Yaw from a sample received within ``max_age_s``, waiting up to
        ``wait_up_to_s`` for one. Use this to BRACKET a maneuver: a sample that
        is merely "not too stale" (``deg()``'s 4s window) but was captured
        mid-turn under-reports the rotation. Call it at a moment the robot is
        stationary (just before a turn / after it settles) — the short sleep
        yields the GIL so the background receiver catches up if heavy main-loop
        compute starved it. Returns None only if the feed delivers nothing fresh
        in time (genuinely dead)."""
        deadline = time.monotonic() + max(0.0, float(wait_up_to_s))
        while True:
            with self._lock:
                yaw = self._yaw_deg
                rx = self._last_rx_monotonic
            now = time.monotonic()
            if yaw is not None and rx is not None and (now - rx) <= float(max_age_s):
                return float(yaw)
            if now >= deadline:
                return None
            time.sleep(0.02)

    def delta_since(self, yaw_before: float | None) -> float | None:
        """RAW yaw change since ``yaw_before`` (both sign-corrected), or None.

        Raw, never normalized: the published yaw integrates continuously and is
        never wrapped, so the plain difference is the exact rotation — including
        multi-revolution accumulations that ±180 normalization would corrupt."""
        if yaw_before is None:
            return None
        now = self.deg()
        if now is None:
            return None
        return float(now) - float(yaw_before)

    def note_tracked_turn(self, tracked_deg: float, imu_delta_deg: float | None) -> None:
        """Cross-check IMU sign against a confidently lidar-tracked turn."""
        if imu_delta_deg is None or abs(tracked_deg) < 8.0 or abs(imu_delta_deg) < 4.0:
            return
        if (tracked_deg >= 0.0) == (imu_delta_deg >= 0.0):
            self._sign_agree += 1
        else:
            self._sign_disagree += 1
        if (
            not self._sign_warned
            and self._sign_disagree >= 3
            and self._sign_disagree > self._sign_agree * 2
        ):
            self._sign_warned = True
            print(
                "[wander] WARNING: IMU yaw sign looks INVERTED versus the lidar-tracked turns "
                f"(agree={self._sign_agree}, disagree={self._sign_disagree}). Re-run with "
                "--imu-yaw-sign -1 (or flip the host imu_yaw_gyro_sign). Until then the IMU "
                "heading prior is IGNORED so it cannot fight the solver."
            )

    def sign_trustworthy(self) -> bool:
        """False once the self-check is confident the configured sign is wrong."""
        return not (self._sign_disagree >= 3 and self._sign_disagree > self._sign_agree * 2)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._sock is not None:
            try:
                self._sock.close(0)
            except Exception:  # noqa: BLE001
                pass
        self._sock = None


def _imu_anchor(imu: "ImuYawClient | None", theta_deg: float) -> tuple[float, float] | None:
    """Snapshot (gyro_yaw_now, dead_reck_theta) so the heading can later be
    re-derived from the gyro. Set at stationary relocalization-accept moments, so
    a fresh read is both safe and accurate. Returns None if the IMU is dead."""
    if imu is None:
        return None
    yaw = imu.deg_fresh()
    if yaw is None:
        return None
    return (float(yaw), float(theta_deg))


def _imu_resolved_theta(imu: "ImuYawClient | None", anchor: tuple[float, float] | None) -> float | None:
    """Dead-reckoned heading propagated from ``anchor`` by the gyro delta since
    it was set. Uses a FRESH read (the recovery tiebreaker fires right after a
    possibly-blind turn, when a stale sample would still show the pre-turn
    heading). None if the IMU is dead.

    The delta is RAW, never ±180-normalized: the published yaw is continuous, and
    the cumulative rotation across a long lost episode routinely exceeds 180deg —
    wrapping it silently corrupts the recovered heading. Pose theta in this
    module runs unbounded too; consumers normalize final comparisons."""
    if imu is None or anchor is None:
        return None
    now = imu.deg_fresh()
    if now is None:
        return None
    return float(anchor[1]) + (float(now) - float(anchor[0]))


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
        frame_id, frame = super().latest()
        return frame_id, self._apply_mask(frame_id, frame)


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
        return float(
            self.squeeze_half_width_m if self.squeeze_active else self.tripwire_half_width_m
        )

    def blocked_trigger_count(self) -> int:
        base = self.squeeze_baseline_points if self.squeeze_active else self.baseline_points
        return int(base) + int(self.min_points_to_trigger)


@dataclass(slots=True)
class FrontierChoice:
    delta_deg: float
    abs_angle_deg: float
    mean_distance_m: float
    width_deg: float
    score: float
    source: str = "live"


@dataclass(slots=True)
class ExploreTarget:
    world_x_m: float
    world_y_m: float
    source: str
    seeded_capture_index: int
    coarse_turns_used: int = 0


@dataclass(slots=True)
class DriveSequenceMeta:
    elapsed_s: float
    iterations: int
    bursts_completed: int
    stopped_by_block: bool
    blocked_points: int
    commanded_forward_speed: float
    steer_theta_vel: float
    host_feedback_updates: int
    host_forward_distance_estimate_m: float
    host_x_vel_peak: float
    host_theta_vel_peak: float
    host_x_vel_last: float
    host_theta_vel_last: float


@dataclass(slots=True)
class DriveBurstMeta:
    elapsed_s: float
    iterations: int
    host_feedback_updates: int
    host_forward_distance_estimate_m: float
    host_x_vel_peak: float
    host_theta_vel_peak: float
    host_x_vel_last: float
    host_theta_vel_last: float
    stopped_by_hazard: bool = False
    hazard_reason: str = ""


def _normalize_angle_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _turn_lever_arm_local_delta(
    dtheta_deg: float,
    *,
    lidar_offset_forward_m: float,
) -> tuple[float, float]:
    """Expected LiDAR translation (in the pre-turn sensor frame) for an in-place
    robot turn of `dtheta_deg`, given the sensor is mounted `lidar_offset_forward_m`
    forward of the robot's rotation center. The sensor traces an arc around the
    center, so pure robot rotation still translates the sensor by
    2*r*sin(|dtheta|/2) — ~0.32 m for a 90deg turn at r=0.23 m."""
    r = float(lidar_offset_forward_m)
    dtheta_rad = math.radians(float(dtheta_deg))
    return r * (math.cos(dtheta_rad) - 1.0), r * math.sin(dtheta_rad)


def _compose_motion_hints(first: MotionHint, second: MotionHint) -> MotionHint:
    """SE(2)-compose two consecutive expected motions into one hint (used when a
    capture is discarded and its motion must carry into the next capture)."""
    theta_rad = math.radians(float(first.expected_dtheta_deg))
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    dx = float(first.expected_dx_local_m) + c * float(second.expected_dx_local_m) - s * float(second.expected_dy_local_m)
    dy = float(first.expected_dy_local_m) + s * float(second.expected_dx_local_m) + c * float(second.expected_dy_local_m)
    # Two chained turns are still a pure in-place turn (same rotation center);
    # any other mix loses that guarantee and must use the generic solver.
    composed_kind = str(second.kind) if str(first.kind) == str(second.kind) else "mixed"
    return MotionHint(
        kind=composed_kind,
        expected_dx_local_m=float(dx),
        expected_dy_local_m=float(dy),
        expected_dtheta_deg=float(first.expected_dtheta_deg) + float(second.expected_dtheta_deg),
        search_xy_m=max(float(first.search_xy_m), float(second.search_xy_m)) + 0.10,
        search_theta_window_deg=max(float(first.search_theta_window_deg), float(second.search_theta_window_deg)) + 10.0,
        label=f"{first.label}+{second.label}",
    )


def _apply_min_effective_magnitude(value: float, *, minimum_abs: float) -> float:
    minimum_abs = max(float(minimum_abs), 0.0)
    value = float(value)
    if abs(value) <= 1e-9 or minimum_abs <= 1e-9:
        return value
    if abs(value) < minimum_abs:
        return float(minimum_abs if value > 0.0 else -minimum_abs)
    return value


def _poll_remote_base_state(robot: SourcceyClient) -> tuple[dict[str, float], bool]:
    try:
        _frames, remote_state, is_fresh = robot._get_data()
    except Exception:
        remote_state = getattr(robot, "last_remote_state", {}) or {}
        return {
            "x.vel": float(remote_state.get("x.vel", 0.0) or 0.0),
            "y.vel": float(remote_state.get("y.vel", 0.0) or 0.0),
            "theta.vel": float(remote_state.get("theta.vel", 0.0) or 0.0),
        }, False

    remote_state = remote_state or {}
    return {
        "x.vel": float(remote_state.get("x.vel", 0.0) or 0.0),
        "y.vel": float(remote_state.get("y.vel", 0.0) or 0.0),
        "theta.vel": float(remote_state.get("theta.vel", 0.0) or 0.0),
    }, bool(is_fresh)


def _heading_bin_index(theta_deg: float, *, bin_count: int) -> int:
    if int(bin_count) <= 0:
        return 0
    wrapped = float(theta_deg) % 360.0
    bin_width_deg = 360.0 / float(bin_count)
    return int(math.floor(wrapped / bin_width_deg)) % int(bin_count)


def _point_in_stop_zone(angle_deg: float, distance_m: float, cfg: StopZoneConfig) -> bool:
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
    return sum(
        1
        for angle_deg, distance_m, _confidence in frame.points
        if _point_in_stop_zone(float(angle_deg), float(distance_m), zone_cfg)
    )


def _select_frontier_choice(
    frame,
    *,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    max_distance_m: float,
    min_range_m: float,
    min_confidence: int,
    frontier_min_distance_m: float,
    frontier_bin_deg: float,
    min_gap_width_m: float = 0.0,
    corridor_veto=None,
) -> FrontierChoice | None:
    half_width = max(10.0, float(valid_angle_half_width_deg))
    bin_deg = max(1.0, float(frontier_bin_deg))
    bin_count = max(9, int(math.ceil((half_width * 2.0) / bin_deg)) + 1)
    bin_angles = np.linspace(-half_width, half_width, bin_count, dtype=np.float64)
    # Openness of a direction is its NEAREST return, not its farthest: a bin
    # with a chair leg at 0.3m and a wall at 1.4m is NOT 1.4m of open space.
    # Using the max here made the planner steer the robot nose-first into
    # near obstacles it could plainly see.
    nearest_ranges = np.zeros(bin_count, dtype=np.float64)
    hit_counts = np.zeros(bin_count, dtype=np.int32)

    for angle_deg, distance_m, confidence in frame.points:
        if int(confidence) < int(min_confidence):
            continue
        distance_m = float(distance_m)
        if not (float(min_range_m) <= distance_m <= float(max_distance_m)):
            continue
        delta_deg = _normalize_angle_deg(float(angle_deg) - float(forward_angle_deg))
        if abs(delta_deg) > half_width:
            continue
        idx = int(round((delta_deg + half_width) / bin_deg))
        idx = max(0, min(bin_count - 1, idx))
        if hit_counts[idx] == 0 or distance_m < nearest_ranges[idx]:
            nearest_ranges[idx] = distance_m
        hit_counts[idx] += 1

    if not np.any(hit_counts):
        return None

    # Smooth the range profile so a single noisy return does not dominate heading choice.
    kernel = np.array([0.2, 0.6, 0.2], dtype=np.float64)
    smoothed_ranges = np.convolve(nearest_ranges, kernel, mode="same")
    open_mask = smoothed_ranges >= float(frontier_min_distance_m)
    if not np.any(open_mask):
        if float(min_gap_width_m) > 0.0:
            # Caller wants drivable openings only. The least-bad-bin fallback
            # below is a single-bin (~6deg) pointer — following it is how the
            # robot noses into corner slots it cannot fit through.
            return None
        best_idx = int(np.argmax(smoothed_ranges))
        best_delta = float(bin_angles[best_idx])
        best_range = float(smoothed_ranges[best_idx])
        if best_range <= 0.0:
            return None
        return FrontierChoice(
            delta_deg=best_delta,
            abs_angle_deg=float(forward_angle_deg) + best_delta,
            mean_distance_m=best_range,
            width_deg=float(bin_deg),
            score=best_range,
            source="live",
        )

    segments: list[tuple[int, int]] = []
    start_idx: int | None = None
    for idx, is_open in enumerate(open_mask):
        if is_open and start_idx is None:
            start_idx = idx
        elif not is_open and start_idx is not None:
            segments.append((start_idx, idx - 1))
            start_idx = None
    if start_idx is not None:
        segments.append((start_idx, bin_count - 1))

    best_choice: FrontierChoice | None = None
    for start_idx, end_idx in segments:
        seg_ranges = smoothed_ranges[start_idx : end_idx + 1]
        seg_angles = bin_angles[start_idx : end_idx + 1]
        seg_mean = float(np.mean(seg_ranges))
        seg_width = float((end_idx - start_idx + 1) * bin_deg)
        # A narrow angular slot is not a drivable opening: a 12deg-wide gap at
        # 1.4m is ~0.3m across — the robot's body cannot pass. Chasing such
        # slots is how the robot wedges itself into clutter fields.
        gap_width_m = 2.0 * seg_mean * math.sin(math.radians(min(seg_width, 178.0)) / 2.0)
        if float(min_gap_width_m) > 0.0 and gap_width_m < float(min_gap_width_m):
            continue
        seg_center = int((start_idx + end_idx) // 2)
        seg_delta = float(bin_angles[seg_center])
        # A 2D scan at ~20cm height sees UNDER beds and tables: a wide, deep
        # "opening" whose corridor the eye cameras have mapped as elevated
        # furniture is an under-furniture tunnel, not a frontier. Field
        # 2026-07-17: the robot pirouetted in the bed/dresser corner for a
        # whole run committing probes into exactly these tunnels while every
        # safety gate (correctly) refused the drive.
        if corridor_veto is not None and corridor_veto(seg_delta, seg_mean):
            continue
        # Favor wide, far openings, but mildly penalize large steering angles.
        seg_score = seg_mean + 0.025 * seg_width - 0.003 * abs(seg_delta)
        candidate = FrontierChoice(
            delta_deg=seg_delta,
            abs_angle_deg=float(forward_angle_deg) + seg_delta,
            mean_distance_m=seg_mean,
            width_deg=seg_width,
            score=seg_score,
            source="live",
        )
        if best_choice is None or candidate.score > best_choice.score:
            best_choice = candidate
    return best_choice


def _build_world_occupancy(
    transformed_sets: list[np.ndarray],
    *,
    resolution_m: float,
    padding_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    non_empty_sets = [points for points in transformed_sets if len(points)]
    if not non_empty_sets:
        return None
    world_points_xy = np.concatenate(non_empty_sets, axis=0).astype(np.float32, copy=False)
    mins = np.min(world_points_xy, axis=0) - float(padding_m)
    maxs = np.max(world_points_xy, axis=0) + float(padding_m)
    width = int(math.ceil((maxs[0] - mins[0]) / float(resolution_m))) + 1
    height = int(math.ceil((maxs[1] - mins[1]) / float(resolution_m))) + 1
    if width <= 1 or height <= 1:
        return None

    grid = np.zeros((height, width), dtype=bool)
    ij = np.round((world_points_xy - mins) / float(resolution_m)).astype(np.int32)
    ij[:, 0] = np.clip(ij[:, 0], 0, width - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, height - 1)
    grid[ij[:, 1], ij[:, 0]] = True
    return grid, mins.astype(np.float32), world_points_xy


def _dilate_bool_grid(grid: np.ndarray, radius_cells: int) -> np.ndarray:
    if int(radius_cells) <= 0:
        return grid
    dilated = grid.copy()
    ys, xs = np.nonzero(grid)
    for y, x in zip(ys, xs, strict=False):
        y0 = max(0, int(y) - int(radius_cells))
        y1 = min(grid.shape[0], int(y) + int(radius_cells) + 1)
        x0 = max(0, int(x) - int(radius_cells))
        x1 = min(grid.shape[1], int(x) + int(radius_cells) + 1)
        dilated[y0:y1, x0:x1] = True
    return dilated


def _select_map_frontier_choice(
    *,
    transformed_sets: list[np.ndarray],
    current_pose,
    max_distance_m: float,
    frontier_min_distance_m: float,
    frontier_bin_deg: float,
    resolution_m: float,
    robot_radius_m: float = 0.30,
) -> FrontierChoice | None:
    occupancy = _build_world_occupancy(
        transformed_sets,
        resolution_m=max(0.03, float(resolution_m)),
        padding_m=max(0.35, float(max_distance_m) * 0.15),
    )
    if occupancy is None:
        return None

    occupied_grid, grid_origin_xy, world_points_xy = occupancy
    # Inflate obstacles by the ROBOT's radius, not a token margin: a gap the
    # map shows as open but narrower than the robot must read as blocked, or
    # the planner keeps steering into corners the robot cannot fit through.
    dilated_grid = _dilate_bool_grid(
        occupied_grid,
        radius_cells=max(1, int(round(max(0.08, float(robot_radius_m)) / max(0.03, float(resolution_m))))),
    )
    pose_xy = np.asarray([float(current_pose.x), float(current_pose.y)], dtype=np.float32)
    theta_rad = math.radians(-float(current_pose.theta_deg))
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    local_rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    local_world_points = (world_points_xy - pose_xy) @ local_rot.T
    local_angles_deg = np.degrees(np.arctan2(local_world_points[:, 1], local_world_points[:, 0]))
    local_ranges_m = np.linalg.norm(local_world_points, axis=1)

    bin_deg = max(4.0, float(frontier_bin_deg))
    step_m = max(0.05, float(resolution_m) * 0.9)
    candidate_deltas = np.arange(-180.0, 180.0 + 1e-6, bin_deg, dtype=np.float32)
    best_choice: FrontierChoice | None = None

    for delta_deg in candidate_deltas:
        heading_world_deg = float(current_pose.theta_deg) + float(delta_deg)
        ray_theta_rad = math.radians(heading_world_deg)
        direction = np.asarray([math.cos(ray_theta_rad), math.sin(ray_theta_rad)], dtype=np.float32)

        clearance_m = 0.0
        exited_known_map = False
        hit_obstacle = False
        distance_m = step_m
        # The robot's own footprint may sit inside the inflated region (its
        # center can legitimately be one robot-radius from a wall). Within
        # that footprint only EXACT wall cells block; inflation applies beyond.
        inflation_starts_m = float(robot_radius_m) + step_m
        while distance_m <= float(max_distance_m):
            sample_xy = pose_xy + direction * float(distance_m)
            cell_xy = np.round((sample_xy - grid_origin_xy) / float(resolution_m)).astype(np.int32)
            cell_x = int(cell_xy[0])
            cell_y = int(cell_xy[1])
            if (
                cell_x < 0
                or cell_x >= int(dilated_grid.shape[1])
                or cell_y < 0
                or cell_y >= int(dilated_grid.shape[0])
            ):
                exited_known_map = True
                clearance_m = float(distance_m)
                break
            blocking = (
                bool(dilated_grid[cell_y, cell_x])
                if float(distance_m) > inflation_starts_m
                else bool(occupied_grid[cell_y, cell_x])
            )
            if blocking:
                hit_obstacle = True
                clearance_m = float(distance_m)
                break
            clearance_m = float(distance_m)
            distance_m += step_m

        if not exited_known_map and not hit_obstacle:
            clearance_m = float(max_distance_m)

        support_mask = (
            np.abs((local_angles_deg - float(delta_deg) + 180.0) % 360.0 - 180.0)
            <= max(bin_deg * 0.75, 5.0)
        ) & (local_ranges_m <= float(max_distance_m) * 1.25)
        support_count = int(np.count_nonzero(support_mask))

        score = float(clearance_m)
        if exited_known_map:
            score += 1.75
        elif not hit_obstacle:
            score += 0.60
        if support_count <= 1:
            score += 0.25
        score -= 0.0015 * abs(float(delta_deg))
        if clearance_m < float(frontier_min_distance_m) * 0.75:
            score -= 2.5

        candidate = FrontierChoice(
            delta_deg=float(delta_deg),
            abs_angle_deg=float(heading_world_deg),
            mean_distance_m=float(clearance_m),
            width_deg=float(bin_deg),
            score=float(score),
            source="map",
        )
        if best_choice is None or candidate.score > best_choice.score:
            best_choice = candidate

    if best_choice is None:
        return None
    if best_choice.mean_distance_m <= max(0.25, float(frontier_min_distance_m) * 0.40):
        return None
    return best_choice


def _select_survey_target(
    *,
    transformed_sets: list[np.ndarray],
    current_pose: Pose2D,
    capture_poses: list[Pose2D],
    resolution_m: float,
    robot_radius_m: float,
    max_distance_m: float,
    min_pose_spacing_m: float = 1.0,
) -> tuple[float, float] | None:
    """When nothing nearby is worth mapping, pick a vantage the robot has NOT
    camped at yet: ray-cast through the robot-inflated map and choose the
    reachable point that is most open and farthest from every previous capture
    pose. This is what sends the robot across the room to photograph the other
    corner instead of milling around the current one."""
    occupancy = _build_world_occupancy(
        transformed_sets,
        resolution_m=max(0.03, float(resolution_m)),
        padding_m=0.5,
    )
    if occupancy is None:
        return None
    occupied_grid, grid_origin_xy, _world_points = occupancy
    inflated_grid = _dilate_bool_grid(
        occupied_grid,
        radius_cells=max(1, int(round(float(robot_radius_m) / max(0.03, float(resolution_m))))),
    )
    pose_xy = np.asarray([float(current_pose.x), float(current_pose.y)], dtype=np.float32)
    step_m = 0.06
    inflation_starts_m = float(robot_radius_m) + step_m
    best_score = 0.0
    best_target: tuple[float, float] | None = None
    for heading_deg in np.arange(-180.0, 180.0, 10.0, dtype=np.float32):
        heading_rad = math.radians(float(heading_deg))
        direction = np.asarray([math.cos(heading_rad), math.sin(heading_rad)], dtype=np.float32)
        clearance_m = 0.0
        distance_m = step_m
        while distance_m <= float(max_distance_m):
            sample_xy = pose_xy + direction * float(distance_m)
            cell_xy = np.round((sample_xy - grid_origin_xy) / max(0.03, float(resolution_m))).astype(np.int32)
            cell_x = int(cell_xy[0])
            cell_y = int(cell_xy[1])
            if (
                cell_x < 0
                or cell_x >= int(inflated_grid.shape[1])
                or cell_y < 0
                or cell_y >= int(inflated_grid.shape[0])
            ):
                break
            blocking = (
                bool(inflated_grid[cell_y, cell_x])
                if float(distance_m) > inflation_starts_m
                else bool(occupied_grid[cell_y, cell_x])
            )
            if blocking:
                break
            clearance_m = float(distance_m)
            distance_m += step_m
        candidate_distance_m = min(clearance_m * 0.7, clearance_m - 0.35)
        if candidate_distance_m < 0.60:
            continue
        candidate_xy = pose_xy + direction * float(candidate_distance_m)
        pose_spacing_m = min(
            (
                math.hypot(float(candidate_xy[0]) - float(p.x), float(candidate_xy[1]) - float(p.y))
                for p in capture_poses
            ),
            default=1e9,
        )
        if pose_spacing_m < float(min_pose_spacing_m):
            continue
        score = float(pose_spacing_m) + 0.3 * float(clearance_m)
        if score > best_score:
            best_score = score
            best_target = (float(candidate_xy[0]), float(candidate_xy[1]))
    return best_target


def _compute_drive_steer_theta_vel(
    frontier_choice: FrontierChoice | None,
    *,
    steer_gain: float,
    steer_max: float,
    steer_deadband_deg: float,
) -> float:
    if frontier_choice is None:
        return 0.0
    delta_deg = float(frontier_choice.delta_deg)
    if abs(delta_deg) <= float(steer_deadband_deg):
        return 0.0
    steer_theta_vel = float(delta_deg) * float(steer_gain)
    steer_limit = max(0.0, float(steer_max))
    return float(max(-steer_limit, min(steer_limit, steer_theta_vel)))


def _target_choice_from_world_point(
    *,
    target_x_m: float,
    target_y_m: float,
    current_pose: Pose2D,
    source: str,
) -> FrontierChoice | None:
    dx_world = float(target_x_m) - float(current_pose.x)
    dy_world = float(target_y_m) - float(current_pose.y)
    distance_m = math.hypot(dx_world, dy_world)
    if distance_m <= 1e-6:
        return None
    abs_angle_deg = math.degrees(math.atan2(dy_world, dx_world))
    delta_deg = _normalize_angle_deg(abs_angle_deg - float(current_pose.theta_deg))
    return FrontierChoice(
        delta_deg=float(delta_deg),
        abs_angle_deg=float(abs_angle_deg),
        mean_distance_m=float(distance_m),
        width_deg=0.0,
        score=float(distance_m),
        source=str(source),
    )


def _seed_explore_target_from_frontier(
    *,
    frontier_choice: FrontierChoice,
    current_pose: Pose2D,
    capture_index: int,
    step_distance_m: float,
) -> ExploreTarget:
    heading_rad = math.radians(float(frontier_choice.abs_angle_deg))
    travel_m = max(0.50, min(float(step_distance_m), float(frontier_choice.mean_distance_m)))
    return ExploreTarget(
        world_x_m=float(current_pose.x) + math.cos(heading_rad) * travel_m,
        world_y_m=float(current_pose.y) + math.sin(heading_rad) * travel_m,
        source=str(frontier_choice.source),
        seeded_capture_index=int(capture_index),
    )


def _target_distance_m(target: ExploreTarget, pose: Pose2D) -> float:
    return float(math.hypot(float(target.world_x_m) - float(pose.x), float(target.world_y_m) - float(pose.y)))


def _escape_boxed_in(
    *,
    robot: SourcceyClient,
    feed: DirectLidarFeed,
    args,
    after_frame_id: int,
    min_usable: int,
) -> int:
    """The founding scan is BOXED IN — a full revolution arrives but almost
    every return is within min_range of the lidar. Rotating in place cannot
    fix this (a 360deg scan is heading-invariant: the same obstacles stay at
    the same distances, just relabeled). The only escape is to TRANSLATE
    toward open space. Face the most-open direction, drive a short gentle
    burst toward it (gated on a LIVE clearance check, independent of the
    saturated stop box), and rescan — repeat until a healthy revolution
    arrives. Returns the frame_id of the last processed revolution. Raises
    only if the robot is genuinely surrounded (no direction with enough
    clearance) after the attempt budget — a physical situation the operator
    must fix. User directive 2026-07-18: 'if it finds itself boxed in, just
    turn around and keep scanning.'"""
    ESCAPE_CLEAR_M = 0.45  # an opening must be at least this deep to drive into
    FWD_BLOCKED_M = 0.35   # nearest return straight ahead below this = blocked
    ESCAPE_BURST_S = 0.22  # ~0.16m at the min effective speed
    turn_speed = float(args.turn_speed)
    turn_burst_s = float(args.turn_burst_s)
    deg_per_burst = max(4.0, math.degrees(turn_speed * turn_burst_s))
    last_id = int(after_frame_id)

    def _fresh_frame():
        nonlocal last_id
        fid, fr = feed.wait_for_frame_after(
            after_frame_id=last_id,
            timeout_s=float(args.fresh_frame_timeout_s),
            min_frame_advances=1,
            armed_wall_ts=time.time(),
        )
        if fr is not None:
            last_id = int(fid)
        return fr

    def _usable(fr) -> int:
        return len(
            _scan_to_local_points(
                points=fr.points,
                forward_angle_deg=float(args.forward_angle_deg),
                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                invert_lateral_axis=bool(args.invert_lateral_axis),
                max_distance_m=float(args.max_distance_m),
                min_confidence=int(args.min_confidence),
                min_range_m=float(args.min_range_m),
            )
        )

    def _forward_clear_m(fr) -> float:
        # Nearest return within a narrow (+-20deg) cone straight ahead.
        best = float(args.max_distance_m)
        for angle_deg, distance_m, confidence in fr.points:
            if int(confidence) < int(args.min_confidence):
                continue
            d = float(distance_m)
            if d <= 0.05:
                continue
            if abs(_normalize_angle_deg(float(angle_deg) - float(args.forward_angle_deg))) <= 20.0:
                best = min(best, d)
        return best

    def _best_open(fr):
        return _select_frontier_choice(
            fr,
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            max_distance_m=float(args.max_distance_m),
            min_range_m=float(args.min_range_m),
            min_confidence=int(args.min_confidence),
            frontier_min_distance_m=ESCAPE_CLEAR_M,
            frontier_bin_deg=float(args.frontier_bin_deg),
        )

    def _turn(delta_deg: float) -> None:
        if abs(delta_deg) < 8.0:
            return
        sign = 1.0 if delta_deg >= 0.0 else -1.0
        for _ in range(min(10, max(1, int(round(abs(delta_deg) / deg_per_burst))))):
            _execute_turn_burst(
                robot=robot, direction_sign=sign,
                turn_speed=turn_speed, turn_burst_s=turn_burst_s, turn_settle_s=0.08,
            )

    def _drive_forward() -> None:
        deadline = time.monotonic() + ESCAPE_BURST_S
        while time.monotonic() < deadline:
            robot.send_action(
                {
                    "x.vel": max(float(args.min_effective_move_speed), 0.75),
                    "y.vel": 0.0,
                    "theta.vel": 0.0,
                    "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
                    "untorque_left": True,
                    "untorque_right": True,
                }
            )
            time.sleep(0.05)
        _send_stop(robot)
        time.sleep(0.25)

    def _full_rotation_survey() -> float:
        # Physically complete a ~360deg rotation (per user: never conclude
        # boxed without a full look-around), returning the best open clearance
        # seen anywhere. The lidar already sees 360deg, so this is confirmation
        # + a deliberate, visible survey — not information the single scan
        # lacked.
        print("[wander] BOXED IN — no opening ahead; doing a FULL rotation to survey all directions")
        best = 0.0
        bursts_per_step = 2
        steps = max(6, int(round(360.0 / max(deg_per_burst * bursts_per_step, 24.0))))
        for _ in range(steps):
            for _ in range(bursts_per_step):
                _execute_turn_burst(
                    robot=robot, direction_sign=1.0,
                    turn_speed=turn_speed, turn_burst_s=turn_burst_s, turn_settle_s=0.05,
                )
            fr = _fresh_frame()
            if fr is None:
                continue
            if _usable(fr) >= int(min_usable):
                return float(args.max_distance_m)  # a healthy view appeared mid-survey
            c = _best_open(fr)
            if c is not None:
                best = max(best, float(c.mean_distance_m))
        print(f"[wander] full-rotation survey: best clearance found = {best:.2f}m")
        return best

    # COMMIT-AND-DRIVE: pick an opening, face it ONCE, then drive FORWARD
    # through it in repeated bursts (re-picking a new lateral bearing every
    # cycle made it oscillate between opposite openings and never translate —
    # field 2026-07-18: drove -84deg then +84deg then gave up). Re-survey only
    # when the committed direction becomes blocked ahead.
    for direction_attempt in range(6):
        fr = _fresh_frame()
        if fr is None:
            print(f"[wander] boxed-in escape: no fresh revolution (direction {direction_attempt + 1}/6)")
            continue
        if _usable(fr) >= int(min_usable):
            print(f"[wander] escaped the box (usable={_usable(fr)}); founding the map here")
            _send_stop(robot)
            return last_id
        choice = _best_open(fr)
        if choice is None or float(choice.mean_distance_m) < ESCAPE_CLEAR_M:
            best = _full_rotation_survey()
            if best >= float(args.max_distance_m):
                _send_stop(robot)
                return last_id  # healthy view appeared during the survey
            if best < ESCAPE_CLEAR_M:
                _send_stop(robot)
                raise RuntimeError(
                    "BOXED IN with no escape route: after a FULL rotation, every direction "
                    f"still has an obstacle within {ESCAPE_CLEAR_M:.2f}m of the lidar. The robot "
                    "is genuinely surrounded (draped material all around, or wedged in a tight "
                    "nook). Physically reposition it with clear space around the lidar — this is "
                    "NOT a host/feed problem."
                )
            continue  # an opening exists somewhere; re-pick and drive next loop
        # Face the opening, then commit to driving FORWARD through it.
        print(
            "[wander] BOXED IN — committing to the open direction "
            f"(bearing {float(choice.delta_deg):+.0f}deg, {float(choice.mean_distance_m):.2f}m clear) "
            "and driving through it"
        )
        _turn(float(choice.delta_deg))
        for _ in range(6):
            fr = _fresh_frame()
            if fr is None:
                break
            if _usable(fr) >= int(min_usable):
                print(f"[wander] escaped the box (usable={_usable(fr)}); founding the map here")
                _send_stop(robot)
                return last_id
            if _forward_clear_m(fr) < FWD_BLOCKED_M:
                break  # committed direction is now blocked — re-survey next loop
            _drive_forward()
    _send_stop(robot)
    raise RuntimeError(
        "Could not escape the boxed-in area after driving through 6 openings — the robot is "
        "likely surrounded or wedged. Physically reposition it with clear space around the lidar."
    )


def _capture_snapshot(
    *,
    feed: DirectLidarFeed,
    output_dir: Path,
    request_index: int,
    after_frame_id: int,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    invert_lateral_axis: bool,
    max_distance_m: float,
    min_range_m: float,
    min_confidence: int,
    fresh_frame_timeout_s: float,
    fresh_frame_advances: int,
    capture_config_extra: dict[str, object] | None = None,
) -> tuple[int, object, np.ndarray]:
    print(
        "[capture] waiting for fresh revolution "
        f"(after frame_id={after_frame_id}, min advances={fresh_frame_advances})"
    )
    frame = None
    frame_id = int(after_frame_id)
    local_points_xy = None
    # Two failure modes produce a capture too sparse to stitch, and they are
    # NOT the same problem:
    #   (1) DEGRADED FEED — the host serves partial revolutions (raw far
    #       below the ~240 norm), e.g. USB stutter / spin-up. A host issue.
    #   (2) BOXED IN — a FULL revolution arrives (raw ~240) but almost all
    #       returns fall within min_range_m (self-hits / walls right against
    #       the lidar), so few survive filtering. The robot is physically
    #       wedged / something is draped within ~min_range of the lidar. NOT
    #       a host issue (field 2026-07-18: baseline 168 self-hits, raw 241,
    #       usable 49 — misreported as a feed fault).
    # Either way the capture is refused (it would found a garbage map), but
    # the diagnosis must be accurate so the operator fixes the right thing.
    MIN_HEALTHY_LOCAL_POINTS = 60
    MIN_HEALTHY_RAW_POINTS = 150
    last_raw = 0
    last_usable = 0
    for attempt_index in range(6):
        armed_wall_ts = time.time()
        frame_id, frame = feed.wait_for_frame_after(
            after_frame_id=after_frame_id,
            timeout_s=float(fresh_frame_timeout_s),
            min_frame_advances=int(fresh_frame_advances),
            armed_wall_ts=armed_wall_ts,
        )
        if frame is None:
            # Transient feed dropouts self-heal (the client auto-reconnects);
            # give it a few chances before treating this as fatal.
            print(
                f"[capture] no fresh revolution yet (attempt {attempt_index + 1}/6); "
                "waiting for feed to recover"
            )
            continue
        local_points_xy = _scan_to_local_points(
            points=frame.points,
            forward_angle_deg=float(forward_angle_deg),
            valid_angle_half_width_deg=float(valid_angle_half_width_deg),
            invert_lateral_axis=bool(invert_lateral_axis),
            max_distance_m=float(max_distance_m),
            min_confidence=int(min_confidence),
            min_range_m=float(min_range_m),
        )
        last_raw = len(frame.points)
        last_usable = len(local_points_xy)
        if len(local_points_xy) >= MIN_HEALTHY_LOCAL_POINTS:
            break
        if last_raw < MIN_HEALTHY_RAW_POINTS:
            print(
                f"[capture] revolution {frame_id} DEGRADED FEED (raw={last_raw} < "
                f"{MIN_HEALTHY_RAW_POINTS} — host serving partial revolutions); waiting "
                f"(attempt {attempt_index + 1}/6)"
            )
        else:
            print(
                f"[capture] revolution {frame_id}: FULL revolution (raw={last_raw}) but only "
                f"{last_usable} points beyond {float(min_range_m):.2f}m — the robot is BOXED IN "
                f"(surrounded by returns within {float(min_range_m):.2f}m); waiting "
                f"(attempt {attempt_index + 1}/6)"
            )
        after_frame_id = int(frame_id)
        frame = None
        local_points_xy = None
        time.sleep(0.4)
    if frame is None or local_points_xy is None:
        if last_raw >= MIN_HEALTHY_RAW_POINTS:
            raise RuntimeError(
                f"The lidar feed is HEALTHY (~{last_raw} points/revolution) but only ~{last_usable} "
                f"survive the {float(min_range_m):.2f}m near-cutoff — the rest are self-hits / near "
                "returns. The robot is physically BOXED IN, or something (e.g. draped material) is "
                f"within ~{float(min_range_m):.2f}m of the lidar. Reposition it with clear space "
                "around the lidar. This is NOT a host/feed problem."
            )
        raise RuntimeError(
            f"LiDAR feed is delivering PARTIAL revolutions (raw ~{last_raw}, far below the ~240 "
            "norm) and did not recover within 6 attempts. Check the lidar stream host on the Pi "
            "(USB contention / power) before rerunning."
        )
    capture_config = {
        "forward_angle_deg": float(forward_angle_deg),
        "valid_angle_half_width_deg": float(valid_angle_half_width_deg),
        "invert_lateral_axis": bool(invert_lateral_axis),
        "max_distance_m": float(max_distance_m),
        "min_range_m": float(min_range_m),
        "min_confidence": int(min_confidence),
        "capture_mode": "wander_snapshot_stitch",
    }
    if capture_config_extra:
        capture_config.update(capture_config_extra)
    _save_snapshot(
        output_dir=output_dir,
        request_index=int(request_index),
        frame_id=int(frame_id),
        frame=frame,
        local_points_xy=local_points_xy,
        capture_config=capture_config,
    )
    snapshot_json_path = output_dir / f"snapshot_{int(request_index):03d}.json"
    snapshot_metadata = json.loads(snapshot_json_path.read_text(encoding="utf-8"))
    snapshot = Snapshot(
        name=f"snapshot_{int(request_index):03d}",
        points_xy=local_points_xy.astype(np.float32, copy=False),
        metadata=snapshot_metadata,
    )
    return int(frame_id), frame, local_points_xy, snapshot


def _save_motion_hints(output_dir: Path, motion_hints: list[MotionHint]) -> Path:
    payload = {
        "schema": "sourccey.wander_snapshot_hints.v1",
        "motion_hints": [
            {
                "kind": hint.kind,
                "label": hint.label,
                "expected_dx_local_m": round(float(hint.expected_dx_local_m), 6),
                "expected_dy_local_m": round(float(hint.expected_dy_local_m), 6),
                "expected_dtheta_deg": round(float(hint.expected_dtheta_deg), 6),
                "search_xy_m": round(float(hint.search_xy_m), 6),
                "search_theta_window_deg": round(float(hint.search_theta_window_deg), 6),
            }
            for hint in motion_hints
        ],
    }
    path = output_dir / "motion_hints.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_stitch_report(
    *,
    stitch_dir: Path,
    snapshot_dir: Path,
    transformed_sets: list[np.ndarray],
    poses,
    solve_log,
    motion_hints: list[MotionHint],
) -> tuple[Path, Path]:
    from ldlidar_direct_snapshot_stitch import _generate_svg

    svg = _generate_svg(transformed_sets, poses)
    report = {
        "schema": "sourccey.wander_snapshot_stitch.v1",
        "snapshot_dir": str(snapshot_dir.resolve()),
        "output_dir": str(stitch_dir.resolve()),
        "motion_hints": [
            {
                "kind": hint.kind,
                "label": hint.label,
                "expected_dx_local_m": round(float(hint.expected_dx_local_m), 6),
                "expected_dy_local_m": round(float(hint.expected_dy_local_m), 6),
                "expected_dtheta_deg": round(float(hint.expected_dtheta_deg), 6),
                "search_xy_m": round(float(hint.search_xy_m), 6),
                "search_theta_window_deg": round(float(hint.search_theta_window_deg), 6),
            }
            for hint in motion_hints
        ],
        "solve_log": solve_log,
        "final_pose": _pose_dict(poses[-1]),
    }
    html = HTML_TEMPLATE.format(svg=svg, report=json.dumps(report, indent=2))
    html_path = stitch_dir / "latest_stitched_overlay.html"
    report_path = stitch_dir / "latest_stitched_overlay.json"
    html_path.write_text(html, encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return html_path, report_path


def _rebuild_stitch(
    *,
    snapshot_dir: Path,
    stitch_dir: Path,
    motion_hints: list[MotionHint],
    resolution_m: float,
    fallback_search_xy_m: float,
) -> dict[str, object]:
    snapshots = _load_snapshots(snapshot_dir)
    poses, transformed_sets, solve_log = _stitch_snapshots(
        snapshots=snapshots,
        motion_hints=motion_hints,
        resolution_m=float(resolution_m),
        fallback_search_xy_m=float(fallback_search_xy_m),
    )
    html_path, report_path = _write_stitch_report(
        stitch_dir=stitch_dir,
        snapshot_dir=snapshot_dir,
        transformed_sets=transformed_sets,
        poses=poses,
        solve_log=solve_log,
        motion_hints=motion_hints,
    )
    _save_motion_hints(stitch_dir, motion_hints)
    return {
        "snapshots": snapshots,
        "poses": poses,
        "transformed_sets": transformed_sets,
        "solve_log": solve_log,
        "html_path": html_path,
        "report_path": report_path,
    }


def _append_stitch(
    *,
    stitch_dir: Path,
    snapshot_dir: Path,
    stitch_state: dict[str, object],
    motion_hints: list[MotionHint],
    new_snapshot: Snapshot,
    resolution_m: float,
    solved_pose_override: Pose2D | None = None,
    solve_meta_override: dict[str, object] | None = None,
) -> dict[str, object]:
    append_started = time.monotonic()
    if solved_pose_override is None:
        snapshots, poses, transformed_sets, solve_log = _append_stitch_snapshot(
            snapshots=list(stitch_state["snapshots"]),
            poses=list(stitch_state["poses"]),
            transformed_sets=list(stitch_state["transformed_sets"]),
            solve_log=list(stitch_state["solve_log"]),
            new_snapshot=new_snapshot,
            motion_hint=motion_hints[-1],
            resolution_m=float(resolution_m),
        )
    else:
        snapshots = [*list(stitch_state["snapshots"]), new_snapshot]
        poses = [*list(stitch_state["poses"]), solved_pose_override]
        transformed = _transform_points(new_snapshot.points_xy, solved_pose_override)
        transformed_sets = [*list(stitch_state["transformed_sets"]), transformed]
        solve_log = [
            *list(stitch_state["solve_log"]),
            {
                "snapshot": new_snapshot.name,
                "pose": _pose_dict(solved_pose_override),
                "initial_pose": _pose_dict(solved_pose_override),
                "score": None if solve_meta_override is None else solve_meta_override.get("score"),
                "solve_source": "live_relocalize_override"
                if solve_meta_override is None
                else solve_meta_override.get("source", "live_relocalize_override"),
                "local_score": None if solve_meta_override is None else solve_meta_override.get("local_score"),
                "whole_map_score": None if solve_meta_override is None else solve_meta_override.get("whole_map_score"),
                "whole_map_searched": None
                if solve_meta_override is None
                else solve_meta_override.get("whole_map_searched"),
                "timing_s": None if solve_meta_override is None else solve_meta_override.get("timing_s"),
                "motion_hint": {
                    "kind": motion_hints[-1].kind,
                    "label": motion_hints[-1].label,
                    "expected_dx_local_m": round(float(motion_hints[-1].expected_dx_local_m), 4),
                    "expected_dy_local_m": round(float(motion_hints[-1].expected_dy_local_m), 4),
                    "expected_dtheta_deg": round(float(motion_hints[-1].expected_dtheta_deg), 4),
                    "search_xy_m": round(float(motion_hints[-1].search_xy_m), 4),
                    "search_theta_window_deg": round(float(motion_hints[-1].search_theta_window_deg), 4),
                },
                "host_revolution_index": new_snapshot.metadata.get("host_revolution_index"),
                "host_point_digest": new_snapshot.metadata.get("host_point_digest"),
            },
        ]
    solve_elapsed_s = time.monotonic() - append_started
    write_started = time.monotonic()
    html_path, report_path = _write_stitch_report(
        stitch_dir=stitch_dir,
        snapshot_dir=snapshot_dir,
        transformed_sets=transformed_sets,
        poses=poses,
        solve_log=solve_log,
        motion_hints=motion_hints,
    )
    _save_motion_hints(stitch_dir, motion_hints)
    write_elapsed_s = time.monotonic() - write_started
    return {
        "snapshots": snapshots,
        "poses": poses,
        "transformed_sets": transformed_sets,
        "solve_log": solve_log,
        "html_path": html_path,
        "report_path": report_path,
        "timing": {
            "append_total_s": round(time.monotonic() - append_started, 4),
            "solve_phase_s": round(solve_elapsed_s, 4),
            "write_phase_s": round(write_elapsed_s, 4),
        },
    }


def _pose_delta_metrics(reference_pose: Pose2D, candidate_pose: Pose2D) -> tuple[float, float]:
    translation_m = math.hypot(float(candidate_pose.x) - float(reference_pose.x), float(candidate_pose.y) - float(reference_pose.y))
    theta_error_deg = abs(_normalize_angle_deg(float(candidate_pose.theta_deg) - float(reference_pose.theta_deg)))
    return float(translation_m), float(theta_error_deg)


def _estimate_pose_against_stitched_map(
    *,
    points_xy: np.ndarray,
    transformed_sets: list[np.ndarray],
    initial_pose: Pose2D,
    resolution_m: float,
    search_xy_m: float,
    theta_window_deg: float,
    max_translation_from_initial_m: float | None = None,
    prior_translation_weight: float = 0.12,
    prior_theta_weight: float = 0.04,
) -> tuple[Pose2D, dict[str, object]] | None:
    non_empty_sets = [points for points in transformed_sets if len(points)]
    if len(points_xy) < 12 or not non_empty_sets:
        return None

    global_points_xy = np.concatenate(non_empty_sets, axis=0)
    solved_pose, score_meta = _search_pose(
        snapshot_points_xy=points_xy,
        global_points_xy=global_points_xy,
        initial_pose=initial_pose,
        resolution_m=float(resolution_m),
        search_xy_m=max(0.45, float(search_xy_m)),
        coarse_angle_step_deg=4.0,
        fine_angle_step_deg=0.5,
        theta_window_deg=max(24.0, float(theta_window_deg)),
        whole_map_theta_center_deg=float(initial_pose.theta_deg),
        whole_map_theta_window_deg=max(90.0, float(theta_window_deg) * 2.0),
        max_translation_from_initial_m=max_translation_from_initial_m,
        prior_pose=initial_pose,
        prior_translation_weight=float(prior_translation_weight),
        prior_theta_weight=float(prior_theta_weight),
    )
    return solved_pose, score_meta


def _accept_relocalized_pose(
    *,
    label: str,
    candidate_pose: Pose2D,
    score_meta: dict[str, object],
    expected_pose: Pose2D,
    motion_hint: MotionHint | None,
    max_translation_error_m: float,
    max_theta_error_deg: float,
    min_score: float,
) -> bool:
    score = float(score_meta.get("score") or -1e9)
    local_score = float(score_meta.get("local_score") or -1e9)
    whole_map_score = float(score_meta.get("whole_map_score") or -1e9)
    source = str(score_meta.get("source") or "unknown")
    translation_error_m, theta_error_deg = _pose_delta_metrics(expected_pose, candidate_pose)

    rejection_reasons: list[str] = []
    if score < float(min_score):
        rejection_reasons.append(f"score={score:.3f}<min={float(min_score):.3f}")
    if translation_error_m > float(max_translation_error_m):
        rejection_reasons.append(
            f"translation_error={translation_error_m:.3f}m>max={float(max_translation_error_m):.3f}m"
        )
    if theta_error_deg > float(max_theta_error_deg):
        rejection_reasons.append(
            f"theta_error={theta_error_deg:.1f}deg>max={float(max_theta_error_deg):.1f}deg"
        )
    if source == "whole_map" and whole_map_score < (local_score + 0.35):
        rejection_reasons.append(
            f"whole_map_margin_too_small whole={whole_map_score:.3f} local={local_score:.3f}"
        )

    if rejection_reasons:
        hint_label = "none" if motion_hint is None else motion_hint.label
        hint_kind = "none" if motion_hint is None else motion_hint.kind
        print(
            "[wander] relocalization rejected "
            f"(label={label}, source={source}, hint={hint_kind}:{hint_label}, "
            f"candidate=({float(candidate_pose.x):.3f}, {float(candidate_pose.y):.3f}, "
            f"{float(candidate_pose.theta_deg):.1f}deg), expected=({float(expected_pose.x):.3f}, "
            f"{float(expected_pose.y):.3f}, {float(expected_pose.theta_deg):.1f}deg), "
            f"score={score:.3f}, local_score={local_score:.3f}, whole_map_score={whole_map_score:.3f}, "
            f"reasons={'; '.join(rejection_reasons)})"
        )
        return False

    print(
        "[wander] relocalization accepted "
        f"(label={label}, source={source}, score={score:.3f}, local_score={local_score:.3f}, "
        f"whole_map_score={whole_map_score:.3f}, translation_error={translation_error_m:.3f}m, "
        f"theta_error={theta_error_deg:.1f}deg)"
    )
    return True


def _log_live_pose_state(
    rr,
    *,
    capture_index: int,
    pose: Pose2D,
    points_xy: np.ndarray,
    cone_half_width_deg: float,
    body_radius_m: float = 0.29,
) -> None:
    rr.set_time("capture_index", sequence=int(capture_index))
    origin = np.asarray([[pose.x, pose.y, 0.0]], dtype=np.float32)
    vector = np.asarray(
        [[0.30 * math.cos(math.radians(pose.theta_deg)), 0.30 * math.sin(math.radians(pose.theta_deg)), 0.0]],
        dtype=np.float32,
    )
    rr.log("world/live_pose/origin", rr.Points3D(origin, colors=[[0, 255, 255]], radii=0.06))
    rr.log("world/live_pose/heading", rr.Arrows3D(origins=origin, vectors=vector, colors=[[0, 255, 255]]))

    def _circle(radius_m: float) -> np.ndarray:
        return np.asarray(
            [
                [pose.x + radius_m * math.cos(a), pose.y + radius_m * math.sin(a), 0.0]
                for a in np.linspace(0.0, 2.0 * math.pi, 48)
            ],
            dtype=np.float32,
        )

    # Body-footprint gauge: bright cyan = the PLANNING collision radius (what
    # the planner treats the robot as — a gap must exceed this DIAMETER to be
    # "reachable"); dim teal = the SQUEEZE radius it will actually attempt to
    # thread (radius - 5cm). Compare either circle against the exit gap.
    squeeze_radius_m = max(0.22, float(body_radius_m) - 0.05)
    rr.log(
        "world/live_pose/body_radius",
        rr.LineStrips3D([_circle(float(body_radius_m))], colors=[[0, 255, 255]], radii=0.006),
    )
    rr.log(
        "world/live_pose/squeeze_radius",
        rr.LineStrips3D([_circle(squeeze_radius_m)], colors=[[0, 140, 150]], radii=0.004),
    )
    if len(points_xy):
        world_points_xy = _transform_points(points_xy, pose)
        world_points_xyz = np.column_stack(
            [world_points_xy[:, 0], world_points_xy[:, 1], np.zeros((len(world_points_xy),), dtype=np.float32)]
        )
        rr.log("world/live_scan", rr.Points3D(world_points_xyz, colors=[[130, 200, 255]], radii=0.01))


def _drive_forward_burst(
    *,
    robot: SourcceyClient,
    feed: DirectLidarFeed,
    zone_cfg: StopZoneConfig,
    forward_speed: float,
    burst_s: float,
    steer_theta_vel: float,
    min_effective_move_speed: float,
    hazard_monitor=None,
) -> DriveBurstMeta:
    start = time.monotonic()
    iterations = 0
    host_feedback_updates = 0
    host_forward_distance_estimate_m = 0.0
    host_x_vel_peak = 0.0
    host_theta_vel_peak = 0.0
    host_x_vel_last = 0.0
    host_theta_vel_last = 0.0
    stopped_by_hazard = False
    hazard_reason = ""
    squeeze_notified = False
    effective_forward_speed = _apply_min_effective_magnitude(
        float(forward_speed),
        minimum_abs=float(min_effective_move_speed),
    )
    while (time.monotonic() - start) < float(burst_s):
        frame_id, frame = feed.latest()
        if frame is not None:
            blocked_points = _blocked_points_for_frame(frame, zone_cfg)
            if blocked_points >= zone_cfg.blocked_trigger_count():
                print(
                    "[drive] stop box became occupied during forward burst "
                    f"(frame_id={frame_id}, blocked_points={blocked_points}, baseline={zone_cfg.baseline_points})"
                )
                break
        commanded_forward_speed = effective_forward_speed
        if hazard_monitor is not None:
            hazard_ok, hazard_reason_now = hazard_monitor.forward_allowed()
            if not hazard_ok:
                stopped_by_hazard = True
                hazard_reason = str(hazard_reason_now)
                print(
                    "[safety] elevated hazard denied forward motion during burst "
                    f"({hazard_reason}); halting"
                )
                break
            if str(hazard_reason_now) == "squeeze_creep":
                # Passable gap beside furniture: advance at the minimum
                # effective speed so the approach keeps refining the edge
                # and the narrow-corridor gate can veto in time.
                commanded_forward_speed = min(
                    effective_forward_speed, float(min_effective_move_speed)
                )
                if not squeeze_notified:
                    squeeze_notified = True
                    measured_width = getattr(
                        hazard_monitor.state(), "clear_width_m", None
                    )
                    width_note = (
                        f"measured gap {measured_width:.2f}m wide"
                        if measured_width is not None
                        else "gap width unmeasured"
                    )
                    print(
                        "[safety] passable gap beside furniture: creeping through "
                        f"at minimum speed (squeeze; {width_note})"
                    )
        robot.send_action(
            {
                "x.vel": float(commanded_forward_speed),
                "y.vel": 0.0,
                "theta.vel": float(steer_theta_vel),
                "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
                "untorque_left": True,
                "untorque_right": True,
            }
        )
        iterations += 1
        remote_base_state, is_fresh_state = _poll_remote_base_state(robot)
        host_x_vel_last = float(remote_base_state.get("x.vel", 0.0))
        host_theta_vel_last = float(remote_base_state.get("theta.vel", 0.0))
        host_x_vel_peak = max(host_x_vel_peak, abs(host_x_vel_last))
        host_theta_vel_peak = max(host_theta_vel_peak, abs(host_theta_vel_last))
        if is_fresh_state:
            host_feedback_updates += 1
            host_forward_distance_estimate_m += float(host_x_vel_last) * 0.05
        time.sleep(0.05)
    _send_stop(robot)
    return DriveBurstMeta(
        elapsed_s=float(time.monotonic() - start),
        iterations=int(iterations),
        host_feedback_updates=int(host_feedback_updates),
        host_forward_distance_estimate_m=float(host_forward_distance_estimate_m),
        host_x_vel_peak=float(host_x_vel_peak),
        host_theta_vel_peak=float(host_theta_vel_peak),
        host_x_vel_last=float(host_x_vel_last),
        host_theta_vel_last=float(host_theta_vel_last),
        stopped_by_hazard=bool(stopped_by_hazard),
        hazard_reason=str(hazard_reason),
    )


def _run_drive_sequence(
    *,
    robot: SourcceyClient,
    feed: DirectLidarFeed,
    zone_cfg: StopZoneConfig,
    forward_speed: float,
    burst_s: float,
    burst_count: int,
    inter_burst_pause_s: float,
    steer_theta_vel: float,
    min_effective_move_speed: float,
) -> DriveSequenceMeta:
    total_elapsed_s = 0.0
    total_iterations = 0
    bursts_completed = 0
    stopped_by_block = False
    last_blocked_points = 0
    host_feedback_updates = 0
    host_forward_distance_estimate_m = 0.0
    host_x_vel_peak = 0.0
    host_theta_vel_peak = 0.0
    host_x_vel_last = 0.0
    host_theta_vel_last = 0.0
    effective_forward_speed = _apply_min_effective_magnitude(
        float(forward_speed),
        minimum_abs=float(min_effective_move_speed),
    )

    for burst_index in range(max(1, int(burst_count))):
        burst_meta = _drive_forward_burst(
            robot=robot,
            feed=feed,
            zone_cfg=zone_cfg,
            forward_speed=float(forward_speed),
            burst_s=float(burst_s),
            steer_theta_vel=float(steer_theta_vel),
            min_effective_move_speed=float(min_effective_move_speed),
        )
        total_elapsed_s += float(burst_meta.elapsed_s)
        total_iterations += int(burst_meta.iterations)
        bursts_completed += 1
        host_feedback_updates += int(burst_meta.host_feedback_updates)
        host_forward_distance_estimate_m += float(burst_meta.host_forward_distance_estimate_m)
        host_x_vel_peak = max(host_x_vel_peak, float(burst_meta.host_x_vel_peak))
        host_theta_vel_peak = max(host_theta_vel_peak, float(burst_meta.host_theta_vel_peak))
        host_x_vel_last = float(burst_meta.host_x_vel_last)
        host_theta_vel_last = float(burst_meta.host_theta_vel_last)

        frame_id, frame = feed.latest()
        if frame is not None:
            last_blocked_points = _blocked_points_for_frame(frame, zone_cfg)
            print(
                "[drive] burst checkpoint "
                f"(index={burst_index + 1}/{int(burst_count)}, frame_id={frame_id}, blocked_points={last_blocked_points}, "
                f"cmd_x={effective_forward_speed:.3f}, cmd_theta={float(steer_theta_vel):.3f}, "
                f"host_dx_est={burst_meta.host_forward_distance_estimate_m:.3f}, "
                f"host_updates={burst_meta.host_feedback_updates}, host_x_peak={burst_meta.host_x_vel_peak:.3f}, "
                f"host_theta_peak={burst_meta.host_theta_vel_peak:.3f}, host_x_last={burst_meta.host_x_vel_last:.3f}, "
                f"host_theta_last={burst_meta.host_theta_vel_last:.3f})"
            )
            if last_blocked_points >= zone_cfg.blocked_trigger_count():
                stopped_by_block = True
                break
        if burst_index < int(burst_count) - 1 and float(inter_burst_pause_s) > 0.0:
            time.sleep(float(inter_burst_pause_s))

    return DriveSequenceMeta(
        elapsed_s=float(total_elapsed_s),
        iterations=int(total_iterations),
        bursts_completed=int(bursts_completed),
        stopped_by_block=bool(stopped_by_block),
        blocked_points=int(last_blocked_points),
        commanded_forward_speed=float(effective_forward_speed),
        steer_theta_vel=float(steer_theta_vel),
        host_feedback_updates=int(host_feedback_updates),
        host_forward_distance_estimate_m=float(host_forward_distance_estimate_m),
        host_x_vel_peak=float(host_x_vel_peak),
        host_theta_vel_peak=float(host_theta_vel_peak),
        host_x_vel_last=float(host_x_vel_last),
        host_theta_vel_last=float(host_theta_vel_last),
    )


def _turn_with_arc_tracking(
    *,
    robot: SourcceyClient,
    feed: DirectLidarFeed,
    transformed_sets: list[np.ndarray],
    start_pose: Pose2D,
    lidar_offset_forward_m: float,
    resolution_m: float,
    target_turn_deg: float,
    direction_sign: float,
    turn_speed: float,
    turn_burst_s: float,
    turn_settle_s: float,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    invert_lateral_axis: bool,
    max_distance_m: float,
    min_range_m: float,
    min_confidence: int,
    stop_tolerance_deg: float,
    max_bursts: int,
    min_track_score: float = 5.0,
    track_theta_half_window_deg: float = 32.0,
    max_burst_delta_deg: float = 50.0,
    stall_bursts_before_boost: int = 3,
    stall_speed_boost: float = 1.3,
    max_speed_scale: float = 2.0,
    max_consecutive_held: int = 3,
    hazard_monitor=None,
) -> dict[str, float]:
    """Rotate in place while tracking the pose CONTINUOUSLY with the lidar.

    After every burst (robot momentarily at rest) the fresh revolution is
    arc-solved against the stitched map inside a small theta window around the
    last known heading. Between bursts the robot turns at most a few tens of
    degrees, so the solve is unambiguous — the ~90deg symmetry modes of a
    square room are simply outside the window. The turn stops when the
    lidar-tracked rotation reaches the target, so the amount turned is
    measured geometry, not commanded wheel motion."""
    non_empty_sets = [points for points in transformed_sets if len(points)]
    global_points_xy = np.concatenate(non_empty_sets, axis=0) if non_empty_sets else np.zeros((0, 2), np.float32)
    grids = _build_score_grids(global_points_xy, resolution_m=float(resolution_m), padding_m=0.9)
    r = float(lidar_offset_forward_m)
    start_theta_rad = math.radians(float(start_pose.theta_deg))
    arc_center_xy = (
        float(start_pose.x) - r * math.cos(start_theta_rad),
        float(start_pose.y) - r * math.sin(start_theta_rad),
    )

    current_theta_deg = float(start_pose.theta_deg)
    turned_deg = 0.0
    missed_updates = 0
    consecutive_held = 0
    lock_lost = False
    consecutive_timeouts = 0
    stall_count = 0
    speed_scale = 1.0
    last_frame_id = -1
    frame_wait_timeout_s = max(2.0, float(turn_burst_s) + float(turn_settle_s) + 1.5)

    hazard_frozen = False
    for burst_index in range(1, int(max_bursts) + 1):
        if hazard_monitor is not None:
            turn_ok, turn_deny_reason = hazard_monitor.turn_allowed()
            if not turn_ok:
                # Near-tier elevated hazard: rotation is frozen by policy
                # (only backing away is allowed). End the turn with the
                # tracked progress so the motion hint stays truthful.
                hazard_frozen = True
                print(
                    f"[safety] {turn_deny_reason}: rotation frozen near an elevated "
                    f"hazard; ending turn at tracked={abs(turned_deg):.1f}deg"
                )
                break
        _execute_turn_burst(
            robot=robot,
            direction_sign=float(direction_sign),
            turn_speed=float(turn_speed) * float(speed_scale),
            turn_burst_s=float(turn_burst_s),
            turn_settle_s=float(turn_settle_s),
        )
        frame_id, frame = feed.wait_for_frame_after(
            after_frame_id=int(last_frame_id),
            timeout_s=float(frame_wait_timeout_s),
            min_frame_advances=1,
        )
        if frame is None or int(frame_id) == int(last_frame_id):
            consecutive_timeouts += 1
            print(f"[turn] burst={burst_index:02d} timed out waiting for a fresh revolution")
            if consecutive_timeouts >= 3:
                lock_lost = True
                missed_updates += consecutive_timeouts
                print(
                    f"[turn] burst={burst_index:02d} lidar feed stalled during turn; "
                    f"ending turn with best tracked progress={abs(turned_deg):.1f}deg "
                    "and continuing to capture/relocalize instead of crashing"
                )
                break
            continue
        consecutive_timeouts = 0
        last_frame_id = int(frame_id)

        points_xy = _scan_to_local_points(
            points=frame.points,
            forward_angle_deg=float(forward_angle_deg),
            valid_angle_half_width_deg=float(valid_angle_half_width_deg),
            invert_lateral_axis=bool(invert_lateral_axis),
            max_distance_m=float(max_distance_m),
            min_confidence=int(min_confidence),
            min_range_m=float(min_range_m),
        )
        if len(points_xy) < 12:
            missed_updates += 1
            consecutive_held += 1
            print(f"[turn] burst={burst_index:02d} too few valid points to track; holding")
            if consecutive_held >= max(1, int(max_consecutive_held)):
                lock_lost = True
                print(
                    f"[turn] burst={burst_index:02d} tracking lock lost; "
                    "stopping the turn instead of rotating blind"
                )
                break
        else:
            # Window slightly leads in the commanded direction; per-burst
            # rotation cannot reach the next symmetry mode from here.
            expected_theta_deg = current_theta_deg + float(direction_sign) * 8.0
            solved_pose, solved_score = _solve_arc_pose_on_grids(
                snapshot_points_xy=points_xy,
                grids=grids,
                arc_center_xy=arc_center_xy,
                lidar_offset_forward_m=r,
                expected_theta_deg=float(expected_theta_deg),
                theta_half_window_deg=float(track_theta_half_window_deg),
                theta_step_deg=1.5,
                center_slack_m=0.08,
                slack_step_m=0.04,
                theta_prior_weight_per_deg=0.01,
                refine=False,
            )
            delta_deg = _normalize_angle_deg(float(solved_pose.theta_deg) - current_theta_deg)
            delta_along_deg = float(delta_deg) * float(direction_sign)
            if (
                float(solved_score) >= float(min_track_score)
                and -8.0 <= delta_along_deg <= float(max_burst_delta_deg)
            ):
                current_theta_deg = float(solved_pose.theta_deg)
                turned_deg += float(delta_deg)
                consecutive_held = 0
                if abs(delta_deg) >= 1.5:
                    stall_count = 0
                    speed_scale = 1.0
                else:
                    # Locked but not moving: genuine wheel stall, safe to boost.
                    stall_count += 1
                print(
                    f"[turn] burst={burst_index:02d} tracked={abs(turned_deg):6.1f}deg "
                    f"(delta={delta_along_deg:+5.1f}deg) target={float(target_turn_deg):5.1f}deg "
                    f"score={float(solved_score):0.3f}"
                )
            else:
                # Low-confidence solve: the robot may still be rotating but we
                # cannot see how far. NEVER boost here, and stop the turn after
                # a few held bursts — continuing means rotating blind, which is
                # how the map got corrupted before.
                missed_updates += 1
                consecutive_held += 1
                print(
                    f"[turn] burst={burst_index:02d} tracked={abs(turned_deg):6.1f}deg "
                    f"(solve held: delta={delta_along_deg:+5.1f}deg score={float(solved_score):0.3f}) "
                    f"target={float(target_turn_deg):5.1f}deg"
                )
                if consecutive_held >= max(1, int(max_consecutive_held)):
                    lock_lost = True
                    print(
                        f"[turn] burst={burst_index:02d} tracking lock lost; "
                        "stopping the turn instead of rotating blind"
                    )
                    break

        if stall_count >= max(1, int(stall_bursts_before_boost)) and speed_scale < float(max_speed_scale):
            speed_scale = min(float(max_speed_scale), speed_scale * float(stall_speed_boost))
            stall_count = 0
            print(f"[turn] burst={burst_index:02d} no rotation progress; boosting turn speed (scale={speed_scale:.2f})")

        if abs(turned_deg) >= (float(target_turn_deg) - float(stop_tolerance_deg)):
            break

    _send_stop(robot)
    if hazard_monitor is not None and abs(turned_deg) > 0.5:
        # Tracked rotation releases reverse-only holds once the nose points
        # well away from the stopped-at hazard (wedged-pocket escape).
        report_rotation = getattr(hazard_monitor, "report_rotation", None)
        if callable(report_rotation):
            report_rotation(float(turned_deg))
    reached = abs(turned_deg) >= (float(target_turn_deg) - float(stop_tolerance_deg))
    if not reached:
        print(
            "[turn] warning: arc-tracked turn ended before reaching its target "
            f"(tracked={abs(turned_deg):.1f}deg of {float(target_turn_deg):.1f}deg, "
            f"missed_updates={missed_updates}, lock_lost={lock_lost})"
        )
    return {
        "turned_deg": float(turned_deg),
        "final_theta_deg": float(current_theta_deg),
        "missed_updates": float(missed_updates),
        "lock_lost": 1.0 if lock_lost else 0.0,
        "completed": 1.0 if reached else 0.0,
        "hazard_frozen": 1.0 if hazard_frozen else 0.0,
    }


def _solve_drive_step_pose(
    *,
    points_xy: np.ndarray,
    grids: dict[str, object],
    current_pose: Pose2D,
    max_step_m: float = 0.60,
    min_step_m: float = -0.05,
    lateral_slack_m: float = 0.15,
    theta_half_window_deg: float = 12.0,
) -> tuple[Pose2D, float]:
    """Solve one drive-burst pose update: the robot moved at most one burst
    forward from `current_pose`, so search a short strip ahead (with a little
    lateral slack and a small heading window). Small windows keep the solve
    unambiguous, the same principle as the arc-tracked turn."""
    heading_rad = math.radians(float(current_pose.theta_deg))
    c = math.cos(heading_rad)
    s = math.sin(heading_rad)
    best_pose = current_pose
    best_score = -1e9
    for dtheta_deg in np.arange(-theta_half_window_deg, theta_half_window_deg + 1e-6, 2.0, dtype=np.float32):
        for forward_m in np.arange(min_step_m, max_step_m + 1e-6, 0.05, dtype=np.float32):
            for lateral_m in np.arange(-lateral_slack_m, lateral_slack_m + 1e-6, 0.05, dtype=np.float32):
                pose = Pose2D(
                    x=float(current_pose.x) + c * float(forward_m) - s * float(lateral_m),
                    y=float(current_pose.y) + s * float(forward_m) + c * float(lateral_m),
                    theta_deg=float(current_pose.theta_deg) + float(dtheta_deg),
                )
                score = _score_candidate(
                    _transform_points(points_xy, pose),
                    use_nearest_penalty=False,
                    **grids,
                )
                if score > best_score:
                    best_score = float(score)
                    best_pose = pose
    return best_pose, float(best_score)


def _drive_with_tracking(
    *,
    robot: SourcceyClient,
    feed: DirectLidarFeed,
    zone_cfg: StopZoneConfig,
    transformed_sets: list[np.ndarray],
    start_pose: Pose2D,
    resolution_m: float,
    forward_speed: float,
    min_effective_move_speed: float,
    burst_s: float,
    burst_count: int,
    inter_burst_pause_s: float,
    steer_theta_vel: float,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    invert_lateral_axis: bool,
    max_distance_m: float,
    min_range_m: float,
    min_confidence: int,
    min_track_score: float = 5.0,
    max_consecutive_held: int = 2,
    hazard_monitor=None,
    map_side_guard=None,
) -> tuple[Pose2D, dict[str, object]]:
    """Drive forward burst-by-burst while tracking the pose with the lidar
    after every burst. The drive stops when blocked, when the plan completes,
    or when tracking loses lock — the robot never travels more than one burst
    beyond its last confirmed pose, so there is no big-jump solve afterwards.

    map_side_guard: optional callable(pose) -> str | None. A virtual side
    bumper from the ELEVATED MAP: tabletop edges BESIDE the robot are
    invisible to every live sensor (panorama covers ±52° forward, the lidar
    passes under tabletops, the bottom camera watches the floor), but the
    map remembers where they are. A non-None reason halts the drive like a
    hazard stop."""
    non_empty_sets = [points for points in transformed_sets if len(points)]
    global_points_xy = np.concatenate(non_empty_sets, axis=0) if non_empty_sets else np.zeros((0, 2), np.float32)
    grids = _build_score_grids(global_points_xy, resolution_m=float(resolution_m), padding_m=1.4)

    current_pose = start_pose
    bursts_completed = 0
    stopped_by_block = False
    lock_lost = False
    stopped_by_hazard = False
    hazard_reason = ""
    consecutive_held = 0
    missed_updates = 0
    last_blocked_points = 0
    last_frame_id = -1
    elapsed_total_s = 0.0
    frame_wait_timeout_s = max(2.0, float(burst_s) + 1.5)

    for burst_index in range(max(1, int(burst_count))):
        if map_side_guard is not None:
            guard_reason = map_side_guard(current_pose)
            if guard_reason:
                stopped_by_hazard = True
                hazard_reason = str(guard_reason)
                print(f"[drive] {guard_reason}; halting before burst {burst_index + 1:02d}")
                break
        burst_meta = _drive_forward_burst(
            robot=robot,
            feed=feed,
            zone_cfg=zone_cfg,
            forward_speed=float(forward_speed),
            burst_s=float(burst_s),
            steer_theta_vel=float(steer_theta_vel),
            min_effective_move_speed=float(min_effective_move_speed),
            hazard_monitor=hazard_monitor,
        )
        bursts_completed += 1
        elapsed_total_s += float(burst_meta.elapsed_s)
        if burst_meta.stopped_by_hazard:
            stopped_by_hazard = True
            hazard_reason = str(burst_meta.hazard_reason)

        frame_id, frame = feed.wait_for_frame_after(
            after_frame_id=int(last_frame_id),
            timeout_s=float(frame_wait_timeout_s),
            min_frame_advances=1,
        )
        if frame is None:
            missed_updates += 1
            consecutive_held += 1
            print(f"[drive] burst={burst_index + 1:02d} no fresh revolution to track against")
        else:
            last_frame_id = int(frame_id)
            points_xy = _scan_to_local_points(
                points=frame.points,
                forward_angle_deg=float(forward_angle_deg),
                valid_angle_half_width_deg=float(valid_angle_half_width_deg),
                invert_lateral_axis=bool(invert_lateral_axis),
                max_distance_m=float(max_distance_m),
                min_confidence=int(min_confidence),
                min_range_m=float(min_range_m),
            )
            solved_pose, solved_score = _solve_drive_step_pose(
                points_xy=points_xy,
                grids=grids,
                current_pose=current_pose,
            )
            step_m = math.hypot(solved_pose.x - current_pose.x, solved_pose.y - current_pose.y)
            if solved_score >= float(min_track_score):
                current_pose = solved_pose
                consecutive_held = 0
                print(
                    f"[drive] burst={burst_index + 1:02d} tracked pose=({current_pose.x:.3f}, "
                    f"{current_pose.y:.3f}, {current_pose.theta_deg:.1f}deg) step={step_m:.3f}m "
                    f"score={solved_score:.3f}"
                )
            else:
                missed_updates += 1
                consecutive_held += 1
                print(
                    f"[drive] burst={burst_index + 1:02d} solve held "
                    f"(score={solved_score:.3f}); not updating pose"
                )
            last_blocked_points = _blocked_points_for_frame(frame, zone_cfg)
            if last_blocked_points >= zone_cfg.blocked_trigger_count():
                stopped_by_block = True
                print(
                    f"[drive] burst={burst_index + 1:02d} stop box occupied "
                    f"(blocked_points={last_blocked_points}, baseline={zone_cfg.baseline_points})"
                )
                break

        if stopped_by_hazard:
            # Pose above is already tracked for the partial burst; stop the
            # drive here — the elevated hazard is invisible to the lidar, so
            # continuing would ram it.
            break
        if consecutive_held >= max(1, int(max_consecutive_held)):
            lock_lost = True
            print(
                f"[drive] burst={burst_index + 1:02d} tracking lock lost; "
                "stopping the drive instead of moving blind"
            )
            break
        if burst_index < int(burst_count) - 1 and float(inter_burst_pause_s) > 0.0:
            time.sleep(float(inter_burst_pause_s))

    _send_stop(robot)
    return current_pose, {
        "bursts_completed": int(bursts_completed),
        "elapsed_s": float(elapsed_total_s),
        "stopped_by_block": bool(stopped_by_block),
        "blocked_points": int(last_blocked_points),
        "lock_lost": bool(lock_lost),
        "stopped_by_hazard": bool(stopped_by_hazard),
        "hazard_reason": str(hazard_reason),
        "missed_updates": int(missed_updates),
    }


def _map_clearance_behind_m(
    *,
    transformed_sets: list[np.ndarray],
    pose: Pose2D,
    resolution_m: float,
    robot_radius_m: float,
    lidar_offset_forward_m: float,
    max_check_m: float = 0.80,
) -> float:
    """How far straight backwards the MAP says the robot can move. The lidar
    cannot see behind the robot, but the space behind was traversed to get
    here, so the stitched map covers it. Marches from the ROBOT CENTER (the
    sensor sits lidar_offset forward of it) and ignores everything inside the
    robot's own footprint — any map point there is stitching noise, not an
    obstacle the robot could collide with."""
    occupancy = _build_world_occupancy(
        transformed_sets,
        resolution_m=max(0.03, float(resolution_m)),
        padding_m=0.5,
    )
    if occupancy is None:
        return 0.0
    occupied_grid, grid_origin_xy, _world_points = occupancy
    inflated_grid = _dilate_bool_grid(
        occupied_grid,
        radius_cells=max(1, int(round(float(robot_radius_m) / max(0.03, float(resolution_m))))),
    )
    heading_rad = math.radians(float(pose.theta_deg))
    back_dir = np.asarray([-math.cos(heading_rad), -math.sin(heading_rad)], dtype=np.float32)
    center_xy = np.asarray(
        [
            float(pose.x) - float(lidar_offset_forward_m) * math.cos(heading_rad),
            float(pose.y) - float(lidar_offset_forward_m) * math.sin(heading_rad),
        ],
        dtype=np.float32,
    )
    step_m = 0.05
    footprint_m = float(robot_radius_m)
    clearance_m = 0.0
    distance_m = step_m
    while distance_m <= footprint_m + float(max_check_m):
        sample_xy = center_xy + back_dir * float(distance_m)
        cell_xy = np.round((sample_xy - grid_origin_xy) / max(0.03, float(resolution_m))).astype(np.int32)
        cell_x = int(cell_xy[0])
        cell_y = int(cell_xy[1])
        if (
            cell_x < 0
            or cell_x >= int(inflated_grid.shape[1])
            or cell_y < 0
            or cell_y >= int(inflated_grid.shape[0])
        ):
            break
        if float(distance_m) > footprint_m:
            if bool(inflated_grid[cell_y, cell_x]):
                break
            clearance_m = float(distance_m) - footprint_m
        distance_m += step_m
    return clearance_m


def _reverse_escape(
    *,
    robot: SourcceyClient,
    feed: DirectLidarFeed,
    transformed_sets: list[np.ndarray],
    start_pose: Pose2D,
    resolution_m: float,
    robot_radius_m: float,
    lidar_offset_forward_m: float,
    reverse_speed: float,
    burst_s: float,
    bursts: int,
    forward_angle_deg: float,
    valid_angle_half_width_deg: float,
    invert_lateral_axis: bool,
    max_distance_m: float,
    min_range_m: float,
    min_confidence: int,
    min_track_score: float = 5.0,
    allow_blind: bool = False,
) -> tuple[Pose2D, dict[str, object]] | None:
    """Back straight out of a pocket the robot is wedged in. Rotating in place
    cannot free a robot whose stop box is occupied at every heading; reversing
    along the path it came in on can. Clearance is checked against the map
    (the lidar cannot see behind), and the resulting pose is re-solved from
    the lidar afterwards."""
    clearance_m = _map_clearance_behind_m(
        transformed_sets=transformed_sets,
        pose=start_pose,
        resolution_m=float(resolution_m),
        robot_radius_m=float(robot_radius_m),
        lidar_offset_forward_m=float(lidar_offset_forward_m),
    )
    # LIVE rear check: the map clearance is cast from the TRACKED pose, and
    # reverse escapes fire mostly during relocalization crises when that
    # pose is a stale fallback (field 2026-07-13: pose error >1m after a
    # doorway transit — "0.76m clear behind" was measured somewhere else and
    # the robot backed into the open door leaf). The live scan is robot-
    # relative, immune to pose error; ~270deg of it is usable, so much of
    # the rear quarter IS visible. Ranges below 0.65m are the robot's own
    # body (rear corner sits ~0.58m from the forward-offset lidar) and are
    # skipped; a fully occluded rear yields no points and falls back to the
    # map check unchanged.
    live_rear_m: float | None = None
    _live_id, live_frame = feed.latest()
    if live_frame is not None:
        all_points = _scan_to_local_points(
            points=live_frame.points,
            forward_angle_deg=float(forward_angle_deg),
            valid_angle_half_width_deg=180.0,
            invert_lateral_axis=bool(invert_lateral_axis),
            max_distance_m=3.0,
            min_confidence=int(min_confidence),
            min_range_m=float(lidar_offset_forward_m) + float(robot_radius_m) + 0.14,
        )
        if len(all_points):
            rear_bumper_x = -(float(lidar_offset_forward_m) + float(robot_radius_m))
            behind = (all_points[:, 0] < rear_bumper_x) & (
                np.abs(all_points[:, 1]) <= float(robot_radius_m) + 0.08
            )
            if np.any(behind):
                live_rear_m = float(np.min(-all_points[behind, 0]) + rear_bumper_x)
    # rear_visible: the live scan actually returned points in the rear
    # corridor, so we CAN see what is behind (not occluded). live_rear_m is
    # None either when the rear is genuinely open (no points) or when an
    # obstacle sits closer than the scan's near cutoff (~0.14m behind the
    # bumper) — those two are disambiguated by the map clearance below.
    rear_visible = live_rear_m is not None
    if live_rear_m is not None and live_rear_m < clearance_m:
        print(
            f"[wander] live lidar shows only {live_rear_m:.2f}m clear behind "
            f"(map claimed {clearance_m:.2f}m); trusting the live scan"
        )
        clearance_m = live_rear_m
    if clearance_m >= 0.30:
        target_reverse_m = min(0.45, clearance_m - 0.10)
        print(
            f"[wander] reversing out of pocket (map shows {clearance_m:.2f}m clear behind, "
            f"target={target_reverse_m:.2f}m)"
        )
    elif rear_visible:
        # The live lidar SEES the rear and it is tight — a real wall, not a
        # blind spot. Never drive into a wall we can see (field 2026-07-18:
        # the old blind reverse backed the robot into the wall behind it
        # repeatedly). Take only the genuinely clear margin; if there is
        # none, refuse and let the recovery ladder ROTATE instead.
        safe_reverse_m = clearance_m - 0.12
        if safe_reverse_m >= 0.08:
            target_reverse_m = safe_reverse_m
            reverse_speed = min(abs(float(reverse_speed)), 0.8)
            bursts = 1
            print(
                f"[wander] reversing the safe margin only ({target_reverse_m:.2f}m); live "
                f"lidar sees a wall {clearance_m:.2f}m behind"
            )
        else:
            print(
                f"[wander] cannot reverse-escape: live lidar sees a wall {clearance_m:.2f}m "
                "behind — refusing to back into it (recovery will rotate instead)"
            )
            return None
    elif allow_blind and clearance_m >= 0.15:
        # Rear OCCLUDED (no live points — too close to the scan cutoff or
        # genuinely empty) AND the map still credits some room: the robot is
        # wedged despite that, most likely pose drift. A short slow blind
        # nudge is the last resort; the stop box protects the front. NOT
        # taken when the map itself reports a wall right there (< 0.15m):
        # a confident map wall is a real wall, don't gamble into it.
        target_reverse_m = 0.18
        reverse_speed = min(abs(float(reverse_speed)), 0.8)
        bursts = 1
        print(
            f"[wander] WARNING: map shows {clearance_m:.2f}m clear behind but robot is "
            f"wedged and the rear is occluded; attempting short blind reverse "
            f"({target_reverse_m:.2f}m, slow)"
        )
    else:
        blocked_by = "map+live wall" if clearance_m < 0.15 else "rear occluded"
        print(
            f"[wander] cannot reverse-escape: only {clearance_m:.2f}m clear behind "
            f"({blocked_by}) — recovery will rotate instead"
        )
        return None
    for _burst_index in range(max(1, int(bursts))):
        burst_deadline = time.monotonic() + float(burst_s)
        while time.monotonic() < burst_deadline:
            robot.send_action(
                {
                    "x.vel": -abs(float(reverse_speed)),
                    "y.vel": 0.0,
                    "theta.vel": 0.0,
                    "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
                    "untorque_left": True,
                    "untorque_right": True,
                }
            )
            time.sleep(0.05)
        _send_stop(robot)
        time.sleep(0.15)

    latest_frame_id, _ = feed.latest()
    frame_id, frame = feed.wait_for_frame_after(
        after_frame_id=int(latest_frame_id),
        timeout_s=3.0,
        min_frame_advances=1,
    )
    solved_pose = start_pose
    solved_score = -1e9
    if frame is not None:
        points_xy = _scan_to_local_points(
            points=frame.points,
            forward_angle_deg=float(forward_angle_deg),
            valid_angle_half_width_deg=float(valid_angle_half_width_deg),
            invert_lateral_axis=bool(invert_lateral_axis),
            max_distance_m=float(max_distance_m),
            min_confidence=int(min_confidence),
            min_range_m=float(min_range_m),
        )
        if len(points_xy) >= 12:
            non_empty_sets = [points for points in transformed_sets if len(points)]
            grids = _build_score_grids(
                np.concatenate(non_empty_sets, axis=0), resolution_m=float(resolution_m), padding_m=1.2
            )
            solved_pose, solved_score = _solve_drive_step_pose(
                points_xy=points_xy,
                grids=grids,
                current_pose=start_pose,
                max_step_m=0.10,
                min_step_m=-(target_reverse_m + 0.25),
                lateral_slack_m=0.12,
                theta_half_window_deg=10.0,
            )
    locked = float(solved_score) >= float(min_track_score)
    if not locked:
        # Dead-reckon the reverse; the wide capture search will refine it.
        heading_rad = math.radians(float(start_pose.theta_deg))
        solved_pose = Pose2D(
            x=float(start_pose.x) - math.cos(heading_rad) * target_reverse_m * 0.8,
            y=float(start_pose.y) - math.sin(heading_rad) * target_reverse_m * 0.8,
            theta_deg=float(start_pose.theta_deg),
        )
    print(
        "[wander] reverse escape "
        f"({'tracked' if locked else 'dead-reckoned'} pose=({solved_pose.x:.3f}, {solved_pose.y:.3f}, "
        f"{solved_pose.theta_deg:.1f}deg), score={float(solved_score):.3f})"
    )
    return solved_pose, {"locked": bool(locked), "score": float(solved_score)}


def _port_is_free(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("0.0.0.0", int(port)))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def _pick_free_rerun_ports(grpc_port: int, web_port: int) -> tuple[int, int]:
    """Step past ports held by zombie rerun servers (crashed runs can leak an
    orphaned listener; rerun then 'starts' but every log message vanishes and
    the viewer sits on the welcome screen forever)."""
    for bump in range(0, 20, 2):
        candidate_grpc = int(grpc_port) + bump
        candidate_web = int(web_port) + bump
        if _port_is_free(candidate_grpc) and _port_is_free(candidate_web):
            if bump:
                print(
                    f"[rerun] ports {grpc_port}/{web_port} are in use (stale server from a "
                    f"previous run?); using {candidate_grpc}/{candidate_web} instead"
                )
            return candidate_grpc, candidate_web
    print(
        f"[rerun] WARNING: no free port pair found near {grpc_port}/{web_port}; "
        "trying the defaults anyway (the viewer may not receive data)"
    )
    return int(grpc_port), int(web_port)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatic wandering LiDAR capture + offline-style stitcher with live Rerun output."
    )
    parser.add_argument("--remote-ip", default="192.168.1.237", help="Sourccey host IP.")
    parser.add_argument("--robot-id", default="sourccey")
    parser.add_argument("--lidar-host", default="192.168.1.237", help="Pi LiDAR stream host.")
    parser.add_argument("--lidar-port", type=int, default=8765)
    parser.add_argument("--output-dir", default="artifacts/wander_snapshot_stitch")
    parser.add_argument(
        "--max-captures",
        type=int,
        default=0,
        help="Maximum snapshots to collect including the initial one. Use 0 for no limit.",
    )
    parser.add_argument("--forward-angle-deg", type=float, default=DEFAULT_LIDAR_FORWARD_ANGLE_DEG)
    parser.add_argument("--valid-angle-half-width-deg", type=float, default=180.0)
    parser.add_argument("--invert-lateral-axis", action="store_true", default=True)
    parser.add_argument("--no-invert-lateral-axis", dest="invert_lateral_axis", action="store_false")
    parser.add_argument("--max-distance-m", type=float, default=8.0)
    parser.add_argument(
        "--min-range-m",
        type=float,
        default=0.30,
        help="Returns closer than this are dropped from captures/tracking/frontiers. Must exceed "
        "the radius at which the lidar sees the robot's own shell (~0.25m), or every capture "
        "stitches phantom self-hit points into the map at the robot's location.",
    )
    parser.add_argument("--min-confidence", type=int, default=0)
    parser.add_argument("--fresh-frame-timeout-s", type=float, default=3.0)
    parser.add_argument("--fresh-frame-advances", type=int, default=1)
    parser.add_argument("--capture-settle-s", type=float, default=0.90)
    parser.add_argument("--tripwire-distance-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_DISTANCE_M)
    parser.add_argument("--tripwire-half-width-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_HALF_WIDTH_M)
    parser.add_argument("--tripwire-thickness-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_THICKNESS_M)
    parser.add_argument("--min-distance-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_MIN_DISTANCE_M)
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument("--move-speed", type=float, default=0.85)
    parser.add_argument(
        "--min-effective-move-speed",
        type=float,
        default=0.75,
        help="Minimum absolute x.vel command to use once a forward burst is requested so wheel stiction is overcome.",
    )
    parser.add_argument("--move-burst-s", type=float, default=1.40)
    parser.add_argument(
        "--drive-bursts-per-capture",
        type=int,
        default=4,
        help="How many forward bursts to chain together before stopping for the next stitched capture.",
    )
    parser.add_argument(
        "--max-drive-bursts-per-capture",
        type=int,
        default=6,
        help="Upper cap on chained forward bursts when the frontier ahead is especially open.",
    )
    parser.add_argument(
        "--inter-burst-pause-s",
        type=float,
        default=0.02,
        help="Short pause between chained forward bursts.",
    )
    parser.add_argument("--move-settle-s", type=float, default=0.70)
    parser.add_argument(
        "--frontier-drive-steer-gain",
        type=float,
        default=0.0035,
        help="Converts small residual frontier angle error in degrees into forward-drive yaw rate after heading alignment.",
    )
    parser.add_argument(
        "--max-drive-steer-theta-vel",
        type=float,
        default=0.10,
        help="Maximum residual yaw rate applied while driving toward a frontier.",
    )
    parser.add_argument(
        "--drive-steer-deadband-deg",
        type=float,
        default=10.0,
        help="If the chosen frontier is within this heading error, drive straight with zero yaw correction.",
    )
    parser.add_argument("--drive-hint-mps", type=float, default=0.75, help="Used only as the initial translation guess for stitching.")
    parser.add_argument("--drive-search-xy-m", type=float, default=0.35)
    parser.add_argument("--drive-theta-window-deg", type=float, default=12.0)
    parser.add_argument("--turn-direction", choices=("ccw", "cw"), default="ccw")
    parser.add_argument("--turn-deg", type=float, default=90.0)
    parser.add_argument("--turn-speed", type=float, default=0.82)
    parser.add_argument("--turn-burst-s", type=float, default=0.24)
    parser.add_argument("--turn-settle-s", type=float, default=0.18)
    parser.add_argument("--rotation-signature-min-distance-m", type=float, default=0.10)
    parser.add_argument("--rotation-signature-bin-deg", type=float, default=3.0)
    parser.add_argument("--stop-tolerance-deg", type=float, default=12.0)
    parser.add_argument("--min-turn-overlap-ratio", type=float, default=0.18)
    parser.add_argument("--max-turn-bursts", type=int, default=32)
    parser.add_argument("--turn-search-xy-m", type=float, default=0.30)
    parser.add_argument("--turn-theta-window-deg", type=float, default=54.0)
    parser.add_argument("--frontier-min-distance-m", type=float, default=1.20)
    parser.add_argument("--frontier-bin-deg", type=float, default=6.0)
    parser.add_argument(
        "--frontier-goal-step-m",
        type=float,
        default=1.35,
        help="World-space distance to step toward a stitched-map frontier before reseeding another exploration goal.",
    )
    parser.add_argument(
        "--frontier-goal-reached-m",
        type=float,
        default=0.45,
        help="Distance threshold for considering a stitched-map exploration target reached.",
    )
    parser.add_argument(
        "--frontier-align-threshold-deg",
        type=float,
        default=45.0,
        help="Only used for blocked/periodic scan turns; smart mode now prefers driving forward when the stop box is clear.",
    )
    parser.add_argument(
        "--wander-mode",
        choices=("smart", "scan_turn_every_capture", "turn_only"),
        default="smart",
        help=(
            "smart: drive when clear and rotate when blocked, with occasional scan turns; "
            "scan_turn_every_capture: drive burst then rotate before every capture; "
            "turn_only: never drive, just rotate/capture."
        ),
    )
    parser.add_argument(
        "--turn-every-capture",
        action="store_true",
        default=None,
        help="Legacy alias for scan_turn_every_capture behavior.",
    )
    parser.add_argument(
        "--no-turn-every-capture",
        dest="turn_every_capture",
        action="store_false",
        default=None,
        help="Legacy alias for smart behavior.",
    )
    parser.add_argument(
        "--scan-turn-interval",
        type=int,
        default=6,
        help="In smart mode, insert a scan turn after this many clear forward captures.",
    )
    parser.add_argument(
        "--bootstrap-turn-captures",
        type=int,
        default=4,
        help="In smart mode, spend the first N captures rotating in place to build an initial stitched room outline before driving.",
    )
    parser.add_argument("--stitch-resolution-m", type=float, default=0.03)
    parser.add_argument(
        "--robot-radius-m",
        type=float,
        default=0.29,
        help="Planning collision radius: 18in / 0.229m body half-width plus a 0.06m "
        "clearance margin. Map frontiers behind narrower gaps are unreachable.",
    )
    parser.add_argument(
        "--lidar-offset-forward-m",
        type=float,
        default=0.2286,
        help="How far the LiDAR sits forward of the robot's rotation center (9in default). "
        "Used to predict the sensor translation caused by in-place turns.",
    )
    parser.add_argument(
        "--min-append-score",
        type=float,
        default=5.5,
        help="Minimum stitch-solver score required to append a capture to the map. "
        "Captures scoring below this are discarded and retried instead of corrupting the map.",
    )
    parser.add_argument(
        "--on-complete",
        choices=("idle", "stop"),
        default="idle",
        help="When the map is complete: 'idle' holds position and watches the live scan, "
        "resuming exploration if new reachable space appears (e.g. a door opens); "
        "'stop' exits immediately.",
    )
    parser.add_argument(
        "--max-consecutive-append-discards",
        type=int,
        default=2,
        help="After this many consecutive discarded captures, the best available pose is "
        "accepted anyway (with a warning) so the run cannot stall forever.",
    )
    parser.add_argument("--rerun-mode", choices=("web", "local"), default="web")
    parser.add_argument("--rerun-grpc-port", type=int, default=9877)
    parser.add_argument("--rerun-web-port", type=int, default=9878)
    parser.add_argument(
        "--camera-feedback-hz",
        type=float,
        default=5.0,
        help="Live Rerun camera-panel update rate (0 disables). Shows annotated front eyes and bottom camera.",
    )
    parser.add_argument(
        "--elevated-safety",
        choices=("off", "on"),
        default="on",
        help="Camera-based elevated obstacle gate (table/counter/shelf edges the "
        "lidar cannot see). ON by default: without it, nothing can stop the base "
        "from driving under a table edge. 'off' only for bench runs with no "
        "elevated hazards.",
    )
    parser.add_argument(
        "--slam-input-endpoint",
        type=str,
        default="",
        help="slam_input.v1 camera stream endpoint (default tcp://<remote-ip>:5560).",
    )
    parser.add_argument(
        "--eye-perception",
        choices=("depth", "hough"),
        default="depth",
        help="Eye-camera hazard perception: 'depth' = Depth-Anything-V2 metric "
        "depth (3D obstacle test, lidar-anchored scale; falls back to 'hough' "
        "until the model is ready or if it fails to load); 'hough' = the "
        "classic line detector + parallax pipeline.",
    )
    parser.add_argument(
        "--eye-fusion",
        choices=("panorama", "per-eye"),
        default="panorama",
        help="How depth perception sees: 'panorama' (default) fuses both eyes "
        "into ONE calibrated central forward view (requires the eye panorama "
        "calibration — run scripts/sourccey_eye_panorama.py --mode capture "
        "then --mode calibrate; aborts loudly if missing); 'per-eye' uses the "
        "legacy per-eye camera models (yaw/roll known to be off).",
    )
    parser.add_argument(
        "--edge-detection",
        choices=("on", "off"),
        default="on",
        help="Camera-based ELEVATED edge detection (the eye/depth pipeline for "
        "tabletop/counter/shelf lips the lidar cannot see). 'off' disables it "
        "entirely — no depth model is loaded, and the eyes produce no elevated "
        "stops, holds, or edge-map cells; the bottom-camera FLOOR gate and the "
        "lidar stop box still protect. Use when the elevated edges are at lidar "
        "height (e.g. draped solid) so the lidar itself maps them.",
    )
    parser.add_argument(
        "--use-imu-heading",
        choices=("on", "off"),
        default="on",
        help="Use the host's integrated-gyro yaw as a heading PRIOR to disambiguate "
        "room-symmetric scan matches. The host must be publishing yaw (imu_yaw_pub_enabled, "
        "default on). Only yaw DELTAS between captures are used, so gyro drift is irrelevant. "
        "When 'off' or the yaw feed is silent, the loop falls back to lidar-only heading "
        "(exactly the prior behavior). This is what keeps the robot from getting permanently "
        "lost after a lock-lost turn into an open doorway.",
    )
    parser.add_argument(
        "--imu-host",
        default=None,
        help="Host serving the integrated-yaw ZMQ PUB socket. Defaults to --remote-ip.",
    )
    parser.add_argument(
        "--imu-yaw-port",
        type=int,
        default=8770,
        help="Port for the host's integrated-yaw PUB socket (host imu_yaw_pub_endpoint).",
    )
    parser.add_argument(
        "--imu-yaw-sign",
        type=float,
        default=1.0,
        help="Sign applied to the received yaw so +yaw matches the robot's CCW (SLAM theta) "
        "convention. If the run warns that the IMU sign looks inverted, pass -1.",
    )
    parser.add_argument(
        "--edge-mapping",
        choices=("on", "off"),
        default="on",
        help="When an elevated edge stops the wanderer (requires --elevated-safety on): "
        "back off, re-approach slowly to MEASURE it by parallax, back off again, and "
        "sweep the heading to map its full extent — then plan around the mapped "
        "boundary for the rest of the run. 'off' keeps only the plain safety stops.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    snapshot_dir = output_dir / "snapshots"
    stitch_dir = output_dir / "offline_stitch"
    _ensure_clean_directory(output_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stitch_dir.mkdir(parents=True, exist_ok=True)

    rerun_grpc_port, rerun_web_port = _pick_free_rerun_ports(
        int(args.rerun_grpc_port), int(args.rerun_web_port)
    )
    rr, viewer_url = _init_rerun(
        session_name="ldlidar_wander_snapshot_stitch",
        mode=args.rerun_mode,
        grpc_port=rerun_grpc_port,
        web_port=rerun_web_port,
    )

    zone_cfg = StopZoneConfig(
        forward_angle_deg=float(args.forward_angle_deg),
        min_distance_m=float(args.min_distance_m),
        tripwire_distance_m=float(args.tripwire_distance_m),
        tripwire_half_width_m=float(args.tripwire_half_width_m),
        tripwire_thickness_m=float(args.tripwire_thickness_m),
        min_points_to_trigger=max(int(args.min_points), 1),
        # Narrow squeeze band: 4cm inside the full box, floored just past the
        # body's half-width so anything it counts is a genuine collision course.
        squeeze_half_width_m=max(0.24, float(args.tripwire_half_width_m) - 0.04),
    )

    feed = SelfMaskedLidarFeed(args.lidar_host, int(args.lidar_port))
    feed.start()

    robot = SourcceyClient(SourcceyClientConfig(id=args.robot_id, remote_ip=args.remote_ip))
    robot.connect()
    _send_stop(robot)

    # Integrated-gyro yaw heading prior. Decoupled from the lidar feed and the
    # camera/observation stream: a dedicated PUB socket on the host. Optional and
    # non-fatal — if it never delivers, the loop stays lidar-only.
    imu_yaw: ImuYawClient | None = None
    if str(args.use_imu_heading) == "on":
        imu_host = args.imu_host or args.remote_ip
        imu_yaw = ImuYawClient(
            f"tcp://{imu_host}:{int(args.imu_yaw_port)}",
            sign=float(args.imu_yaw_sign),
        )
        imu_yaw.start()
    else:
        print("[wander] IMU heading prior OFF (--use-imu-heading off); heading is lidar-only")

    # Optional camera-based elevated obstacle gate. The lidar map stays the
    # map owner; the eye cameras only VETO unsafe motion (forward on any
    # active hazard, rotation too in the near tier; reverse always allowed).
    hazard_monitor = None
    hazard_subscriber = None
    if str(args.elevated_safety) == "on":
        from sourccey_elevated_safety import (
            ElevatedHazardMonitor,
            ElevatedSafetyConfig,
            SlamCameraSubscriber,
            endpoint_from_remote_ip,
        )

        safety_endpoint = str(args.slam_input_endpoint).strip() or endpoint_from_remote_ip(
            args.remote_ip
        )
        edge_detection_on = str(args.edge_detection) == "on"
        safety_config = ElevatedSafetyConfig(
            slam_input_endpoint=safety_endpoint,
            elevated_eye_enabled=edge_detection_on,
        )
        if not edge_detection_on:
            print(
                "[safety] elevated EDGE DETECTION DISABLED (--edge-detection off): no depth "
                "model, no elevated eye stops/holds/edge cells. Floor gate + lidar stop box "
                "still active. Elevated obstacles are protected ONLY if the lidar sees them."
            )
        hazard_subscriber = SlamCameraSubscriber(
            endpoint=safety_endpoint,
            camera_keys=(
                safety_config.left_key,
                safety_config.right_key,
                safety_config.bottom_key,
            ),
        )
        hazard_subscriber.start()
        if hazard_subscriber.wait_for_frames(
            timeout_s=5.0, required=(safety_config.left_key, safety_config.right_key)
        ):
            # The lidar referees eye candidates: a candidate whose if-on-floor
            # position matches a lidar return is a floor-standing object the
            # stop box already owns, not an elevated hazard.
            def _safety_lidar_ranges():
                _, safety_frame = feed.latest()
                if safety_frame is None:
                    return None
                safety_points = _scan_to_local_points(
                    points=safety_frame.points,
                    forward_angle_deg=float(args.forward_angle_deg),
                    valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                    invert_lateral_axis=bool(args.invert_lateral_axis),
                    max_distance_m=float(args.max_distance_m),
                    min_confidence=int(args.min_confidence),
                    min_range_m=float(args.min_range_m),
                )
                if len(safety_points) == 0:
                    return None
                return (
                    np.degrees(np.arctan2(safety_points[:, 1], safety_points[:, 0])),
                    np.hypot(safety_points[:, 0], safety_points[:, 1]),
                )

            depth_worker = None
            if edge_detection_on and str(args.eye_perception) == "depth":
                # NO FALLBACK: depth requested = depth required. If it cannot
                # load, the run aborts with the reason — never a silent switch
                # to the old detector (use --eye-perception hough explicitly
                # for that pipeline).
                from sourccey_depth_perception import DepthWorker

                eye_mosaic = None
                if str(args.eye_fusion) == "panorama":
                    # Fused central vision: both eyes hard-cut onto the
                    # CALIBRATED virtual forward camera; one inference on one
                    # straight-ahead view (the legacy per-eye models carried
                    # yaw ~5deg off and unmodeled ~6-10deg roll — every
                    # bearing and edge placement inherited that error).
                    # NO FALLBACK: panorama requested = calibration required.
                    from sourccey_eye_panorama import load_perception_mosaic

                    eye_mosaic = load_perception_mosaic()
                    print(
                        "[safety] eye fusion: calibrated panorama "
                        f"({eye_mosaic.virt.width}x{eye_mosaic.virt.height}, "
                        f"hfov={eye_mosaic.model.hfov_deg:.1f}deg, seam band "
                        f"cols {eye_mosaic.seam_cols[0]}..{eye_mosaic.seam_cols[1]} forgiven)"
                    )
                depth_worker = DepthWorker(
                    hazard_subscriber,
                    {
                        safety_config.left_key: safety_config.eye_left_model,
                        safety_config.right_key: safety_config.eye_right_model,
                    },
                    lidar_ranges_fn=_safety_lidar_ranges,
                    edge_detector_config=safety_config.detector_config(),
                    mosaic=eye_mosaic,
                )
                depth_worker.start()
                print(
                    "[safety] loading depth perception (Depth-Anything-V2 metric indoor; "
                    "first run downloads the weights) ..."
                )
                load_started = time.monotonic()
                while not depth_worker.ready and depth_worker.failure is None:
                    if time.monotonic() - load_started > 300.0:
                        raise SystemExit(
                            "[safety] FATAL: depth perception did not load within 300s"
                        )
                    time.sleep(0.5)
                if depth_worker.failure is not None:
                    raise SystemExit(
                        f"[safety] FATAL: depth perception failed to load "
                        f"({depth_worker.failure}). Fix the environment (is "
                        f"'--with transformers' in the run command?) or run with "
                        f"--eye-perception hough explicitly."
                    )
                # Model loaded — now require a first result on every key the
                # worker serves (the fused panorama, or both eyes) so the
                # mission starts with live depth, not a stale-stop.
                first_result_deadline = time.monotonic() + 60.0
                while time.monotonic() < first_result_deadline:
                    if all(
                        depth_worker.latest(eye, max_age_s=10.0) is not None
                        for eye in depth_worker.eye_keys
                    ):
                        break
                    if depth_worker.failure is not None:
                        raise SystemExit(
                            f"[safety] FATAL: depth inference failed on the first "
                            f"frames ({depth_worker.failure})"
                        )
                    time.sleep(0.25)
                else:
                    raise SystemExit(
                        "[safety] FATAL: depth perception produced no results "
                        "within 60s of loading"
                    )
                print(
                    f"[safety] depth perception live on {'/'.join(depth_worker.eye_keys)} "
                    f"(inference ~{max(depth_worker.inference_s, 0.01):.2f}s/frame)"
                )

            hazard_monitor = ElevatedHazardMonitor(
                safety_config,
                hazard_subscriber,
                lidar_ranges_fn=_safety_lidar_ranges,
                depth_worker=depth_worker,
            )
            hazard_monitor.start()
            bottom_present = hazard_subscriber.latest(safety_config.bottom_key)[0] is not None
            bottom_label = "present" if bottom_present else "ABSENT (low floor objects unprotected)"
            if not edge_detection_on:
                eye_perception_label = "DISABLED (--edge-detection off)"
            elif depth_worker is not None:
                eye_perception_label = "depth" + (
                    "+panorama" if "panorama" in depth_worker.eye_keys else ""
                )
            else:
                eye_perception_label = "hough"
            print(
                f"[safety] anti-collision gate ARMED (cameras via {safety_endpoint}, "
                f"bottom_camera={bottom_label}, lidar_referee=on, "
                f"eye_perception={eye_perception_label})"
            )
        else:
            hazard_subscriber.stop()
            hazard_subscriber = None
            print(
                "[safety] WARNING: no camera frames from the slam_input stream within 5s; "
                f"running WITHOUT the elevated obstacle gate (endpoint={safety_endpoint})"
            )
    else:
        print(
            "[safety] WARNING: elevated safety is OFF — table/counter/shelf edges "
            "above the lidar plane are INVISIBLE and the base WILL drive into them"
        )

    # Camera-confirmed elevated geometry, stamped into the PLANNING map
    # (world frame) so the wanderer never routes into a table the lidar
    # cannot see. Session-scoped: the map frame is rebuilt each run.
    # Depth mode stamps floor-projected FOOTPRINT POINTS (the region's true
    # shape); hough mode stamps fitted segments.
    elevated_segments_world: list[dict] = []
    # Rendered red points keyed by their 4cm map cell so floor (free-space)
    # evidence can remove them along with the planning cell.
    # Rendered red points keyed by their 4cm map cell so floor (free-space)
    # evidence can remove them along with the planning cell.
    elevated_points_world: dict[tuple[int, int], tuple[float, float, float]] = {}
    # Two-frame persistence gate for FAR footprint cells: beyond 0.9m a cell
    # becomes a planning obstacle only after being observed elevated in two
    # frames separated by >= 0.75s (a different frame, not this moment's
    # other eye). Monocular depth's worst failures are SINGLE-FRAME —
    # textureless-wall bulges and scale wobble (field 2026-07-13:
    # x1.08..x1.77 across frames) splatted one-off cells over the exit
    # corridor, and the add-only map kept them forever. Real furniture
    # re-observes on the same cells; noise does not. cell -> (mono, hits)
    elevated_pending_world: dict[tuple[int, int], tuple[float, int]] = {}
    # Decay metadata for DRIVE-BY eye cells: cell -> [stamp_bearing_deg,
    # miss_count]. A cell the fused view re-inspects (similar bearing, good
    # range) WITHOUT re-detecting counts a miss; 3 misses remove it.
    # Cells measured by an investigation (_stamp_world_segment) have no
    # entry — approach-verified geometry is permanent. Detection needs two
    # agreeing frames to add a cell; absence symmetrically takes it back
    # (field 2026-07-17: edge-bleed cells at a doorway were permanent
    # unfalsifiable walls that sealed a passable corridor).
    elevated_cell_meta: dict[tuple[int, int], list] = {}
    elevated_occupied_world: set[tuple[int, int]] = set()
    # Single-vantage promotion budget: a wedged/held robot re-observes its
    # own systematic depth artifacts from the SAME pose every cycle, so the
    # two-frame persistence gate confirms them forever (field 2026-07-17:
    # ~100 red cells piled up around one stuck pose and boxed the planner
    # in). Each ~0.25m/20deg pose bucket may promote at most 60 cells;
    # moving to a new vantage earns a fresh budget, so normal exploration
    # mapping is unaffected.
    elevated_vantage_promotions: dict[tuple[int, int, int], int] = {}
    investigated_spots_world: list[tuple[float, float]] = []
    sidestep_spots_world: list[tuple[float, float]] = []

    # Frontier goals that repeatedly failed (path blocked / hazard-held on
    # approach). Two strikes blacklist the frontier for a while so the
    # planner moves on to the rest of the room instead of grinding the same
    # unreachable scrap; entries expire so a transient block cannot
    # permanently hide real space.
    frontier_strike_counts: dict[tuple[int, int], int] = {}
    frontier_blacklist: list[dict[str, float]] = []
    # Anti-fixation: if the SAME frontier is targeted across many consecutive
    # planning cycles while the robot stays pose-lost (never localizing a capture
    # there to strike it the normal way), abandon it and force exploration
    # elsewhere. Without this the robot re-drives the same phantom corner forever
    # — the rest of the room stays "already mapped" in its frozen belief.
    stuck_frontier_face: tuple[float, float] | None = None
    stuck_frontier_lost_cycles = 0
    # Goal commitment: the frontier face chosen last cycle keeps priority in
    # the planner until consumed or blacklisted, so the target cannot flip
    # sides every capture (turn-thrash).
    committed_frontier_face: tuple[float, float] | None = None

    direction_sign = 1.0 if args.turn_direction == "ccw" else -1.0
    signed_turn_deg = float(args.turn_deg) * float(direction_sign)
    wander_mode = str(args.wander_mode)
    if args.turn_every_capture is True:
        wander_mode = "scan_turn_every_capture"
    elif args.turn_every_capture is False and str(args.wander_mode) == "scan_turn_every_capture":
        wander_mode = "smart"

    print(
        "[wander] startup "
        f"mode={wander_mode} "
        f"move_speed={float(args.move_speed):.2f} "
        f"min_effective_move_speed={float(args.min_effective_move_speed):.2f} "
        f"move_burst_s={float(args.move_burst_s):.2f} "
        f"drive_bursts_per_capture={int(args.drive_bursts_per_capture)} "
        f"drive_steer_gain={float(args.frontier_drive_steer_gain):.3f} "
        f"drive_steer_max={float(args.max_drive_steer_theta_vel):.2f} "
        f"turn_deg={float(args.turn_deg):.1f} "
        f"turn_direction={args.turn_direction} "
        f"bootstrap_turn_captures={int(args.bootstrap_turn_captures)} "
        f"scan_turn_interval={int(args.scan_turn_interval)} "
        f"lidar_offset_forward={float(args.lidar_offset_forward_m):.3f}m "
        f"min_append_score={float(args.min_append_score):.2f} "
        f"frontier=(min_distance={float(args.frontier_min_distance_m):.2f}m, "
        f"bin={float(args.frontier_bin_deg):.1f}deg, "
        f"align_threshold={float(args.frontier_align_threshold_deg):.1f}deg, "
        f"goal_step={float(args.frontier_goal_step_m):.2f}m, "
        f"goal_reached={float(args.frontier_goal_reached_m):.2f}m) "
        f"stop_box=(forward={float(args.forward_angle_deg):.1f}deg, "
        f"min={float(args.min_distance_m):.2f}m, "
        f"depth={float(args.tripwire_distance_m):.2f}m, "
        f"half_width={float(args.tripwire_half_width_m):.2f}m, "
        f"thickness={float(args.tripwire_thickness_m):.2f}m, "
        f"min_points={int(args.min_points)})"
    )
    motion_hints: list[MotionHint] = [
        MotionHint(
            kind="start",
            expected_dx_local_m=0.0,
            expected_dy_local_m=0.0,
            expected_dtheta_deg=0.0,
            search_xy_m=float(args.drive_search_xy_m),
            search_theta_window_deg=float(args.turn_theta_window_deg),
            label="initial_capture",
        )
    ]
    drive_checkpoints_since_scan = 0
    pending_turn_reason: str | None = None
    pending_turn_direction_sign = float(direction_sign)
    pending_turn_deg = float(args.turn_deg)
    force_drive_after_turn = False
    # One committed world-frame probe bearing during no_frontier episodes.
    # Field 2026-07-16: re-picking the widest live gap after every align
    # turn oscillated between openings on opposite sides (54 -> -42 -> 60
    # -> ...) and the robot pirouetted ~15 cycles without a single probe
    # drive. Commit once, finish turning to THAT bearing, then drive.
    probe_commit_world_deg: float | None = None
    probe_commit_align_turns = 0
    rotation_coverage_bin_count = 4
    rotation_coverage_bins_seen: set[int] = set()
    rotation_coverage_complete = False
    consecutive_turn_captures = 0
    active_explore_target: ExploreTarget | None = None
    active_explore_target_stall_count = 0
    active_explore_target_last_distance_m: float | None = None
    active_explore_target_blocked_count = 0
    force_live_frontier_cycles = 0
    pending_motion_hint: MotionHint | None = None
    # A failed turn capture is recovered by continuing around the room, not by
    # reversing into the same weak-reference view. Two 85deg sectors create
    # fresh overlap before the planner retries the deferred frontier.
    orbit_recovery_turns_remaining = 0
    orbit_recovery_direction_sign: float | None = None
    # At most one recovery turn may follow an unlocalized turn. If it also
    # loses tracking, further rotations are unobservable and only amplify the
    # heading error; force a fresh capture/drive cycle instead of spinning.
    orbit_recovery_turn_attempts = 0
    # Consecutive stationary "scan the new area" captures from one spot: a
    # second identical scan cannot add information, so >= 1 forces a turn.
    stationary_scan_streak = 0
    consecutive_append_discards = 0
    consecutive_blocked_cycles = 0
    failed_reverse_escapes = 0
    # ---- pose-integrity latch -------------------------------------------
    # THE map-corruption invariant (field 2026-07-17, twice): every ghost
    # stitch happened while the system ALREADY had loud evidence it was
    # lost — lock-lost turns, discarded captures, live relocalization
    # rejected at 2.7-3.2 — and appended anyway on a marginal solve
    # (wrong 90deg symmetry modes score up to ~11.9; healthy appends run
    # 13-16.5). When ANY lostness signal fires, the map becomes
    # READ-ONLY: every append path demands an absolute >=12.5 match and
    # elevated-cell stamping stops. The latch clears only when a
    # relocalization RE-PROVES the pose at >=12.0 — including a
    # wide-theta whole-map recovery search that ignores the (meaningless)
    # pose expectation, which is the exit this failure mode lacked.
    pose_lost = False
    POSE_LOST_APPEND_GATE = 12.5
    POSE_RECOVERY_MIN_SCORE = 12.0
    # The first appends after recovery re-anchor everything that follows —
    # keep them strict too (+2.0 on the gate for the next two appends).
    post_recovery_strict_appends = 0
    # Dead-reckoned heading bound for recovery (field 2026-07-17 run 9): in
    # a sparse near-symmetric room the WRONG 90/180deg mode scores 12-15 —
    # as high as truth — so score alone cannot pick the recovery basin. But
    # physics can: a turn that TRACKED cleanly bounds the true heading to
    # ~±30deg, and a recovery candidate 178deg away is impossible no matter
    # its score (observed: trusted at 31deg, tracked turn to ~-46deg, then
    # a 14.17-score recovery "restored" 131.5deg and later appends went in
    # under two different basins). The anchor theta advances by each turn's
    # TRACKED delta; slack grows by each turn's untracked remainder, so
    # after truly-blind rotation chains (run 7) the bound widens to
    # useless and recovery falls back to score-only — exactly right.
    dead_reck_theta_deg: float | None = None
    dead_reck_slack_deg = 15.0
    # Gyro anchor for the dead-reckoned heading: (imu_yaw_at_set, theta_at_set).
    # Re-derived from the gyro at the recovery tiebreaker so the heading stays
    # accurate through ANY rotation path (align turns, recovery spins, blind
    # bursts) — not just the main arc-tracked turn — which is what lets the
    # robot re-anchor instead of staying lost after a lock-lost turn.
    imu_dead_reck_anchor: tuple[float, float] | None = None
    # Consecutive redundant-vantage capture skips (see the redundant-vantage
    # gate at the checkpoint): capped so endless skipping inside covered
    # space cannot starve the planner of fresh captures forever.
    redundant_skip_streak = 0
    # EDGE-SURVEY bookkeeping: per-cluster (0.3m bucket) attempt counts and
    # last-adoption times. Each yellow cluster gets up to 3 deliberate
    # second-angle observation transits before it is left to the
    # squeeze/verification ladder.
    edge_survey_attempts: dict[tuple[int, int], int] = {}
    edge_survey_last_adopt: dict[tuple[int, int], float] = {}
    # Consecutive failed lost-recoveries. Against a ONE-snapshot map,
    # recovery can be mathematically unable to reach the 12.0 bar (field
    # 2026-07-17 run 11: first bootstrap turn lost lock, recoveries capped
    # at 8.6-10.3, robot deadlocked until Ctrl+C). A founding snapshot
    # holds no investment — after 2 failures the map is RE-FOUNDED from
    # the current position instead of waiting forever.
    lost_recovery_failures = 0
    # Rotate-away attempts that lost tracking lock in the current stuck
    # episode. A point-starved corner (wedged start) makes rotation
    # untrackable: each attempt gets discarded and re-anchored BACK — an
    # oscillation. Allow one lock-lost rotate-away per episode; after that
    # the ladder escalates straight to the slow blind reverse.
    rotate_away_lock_losses = 0
    # Keep recovery turns in one direction until a real forward drive
    # succeeds. Per-frame left/right hazard flips must not make the robot
    # rotate back and forth in place.
    recovery_turn_sign: float | None = None
    wedged_events = 0
    no_frontier_cycles = 0
    survey_targets_used = 0
    active_survey_xy: tuple[float, float] | None = None
    unreachable_targets_world: list[tuple[float, float]] = []
    # EXIT RUN: once the room interior is mapped, tiny frontier slivers must not
    # hold the robot hostage (field 2026-07-18: a 9-cell phantom beyond the left
    # wall was re-targeted for the entire back half of a run — endless 85-150deg
    # align spins at a room it had already finished, while the real doorway sat
    # visible in the live scan). After enough consecutive sliver/no-frontier
    # plans the room is declared effectively mapped and the planner is forced
    # onto the live-opening probe path — commit to the deepest opening the lidar
    # actually sees and DRIVE THROUGH IT. Crossing into the next room makes real
    # frontiers appear, which clears the mode and resumes normal mapping there.
    sliver_frontier_cycles = 0
    exit_mode = False
    exit_scan_turns = 0
    # Vantage positions snapshotted when the exit run latches: completion is
    # POSITIONAL (robot must physically leave this coverage), never inferred
    # from frontier cells alone — cells open by merely seeing through the door.
    exit_latch_poses: list[tuple[float, float]] = []
    # True while the adopted plan is a squeeze-width traversal; switches the
    # stop box to its narrow band (StopZoneConfig.squeeze_active, see there).
    squeeze_traversal_active = False

    def _near_unreachable_target(x_m: float, y_m: float, radius_m: float = 0.60) -> bool:
        return any(
            math.hypot(float(x_m) - ux, float(y_m) - uy) <= float(radius_m)
            for ux, uy in unreachable_targets_world
        )

    def _blacklist_target(target: ExploreTarget, reason: str) -> None:
        unreachable_targets_world.append((float(target.world_x_m), float(target.world_y_m)))
        del unreachable_targets_world[:-12]
        print(
            "[wander] marking exploration target unreachable "
            f"(world=({float(target.world_x_m):.3f}, {float(target.world_y_m):.3f}), "
            f"reason={reason}, blacklist_size={len(unreachable_targets_world)})"
        )

    def _attempt_reverse_escape_hint(start_pose: Pose2D, allow_blind: bool) -> MotionHint | None:
        reverse_result = _reverse_escape(
            robot=robot,
            feed=feed,
            transformed_sets=stitch_state["transformed_sets"],
            start_pose=start_pose,
            resolution_m=float(args.stitch_resolution_m),
            robot_radius_m=float(args.robot_radius_m),
            lidar_offset_forward_m=float(args.lidar_offset_forward_m),
            reverse_speed=max(float(args.min_effective_move_speed), float(args.move_speed) * 0.9),
            burst_s=min(1.0, float(args.move_burst_s)),
            bursts=2,
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            invert_lateral_axis=bool(args.invert_lateral_axis),
            max_distance_m=float(args.max_distance_m),
            min_range_m=float(args.min_range_m),
            min_confidence=int(args.min_confidence),
            allow_blind=bool(allow_blind),
        )
        if reverse_result is None:
            return None
        reverse_pose, reverse_meta = reverse_result
        ddx_world = float(reverse_pose.x) - float(start_pose.x)
        ddy_world = float(reverse_pose.y) - float(start_pose.y)
        theta0_rad = math.radians(float(start_pose.theta_deg))
        cos0 = math.cos(theta0_rad)
        sin0 = math.sin(theta0_rad)
        return MotionHint(
            kind="drive",
            expected_dx_local_m=cos0 * ddx_world + sin0 * ddy_world,
            expected_dy_local_m=-sin0 * ddx_world + cos0 * ddy_world,
            expected_dtheta_deg=_normalize_angle_deg(
                float(reverse_pose.theta_deg) - float(start_pose.theta_deg)
            ),
            search_xy_m=0.45 if bool(reverse_meta.get("locked")) else 0.90,
            search_theta_window_deg=18.0 if bool(reverse_meta.get("locked")) else 30.0,
            label=f"reverse_escape_{capture_index:02d}",
        )

    def _active_frontier_blacklist() -> list[tuple[float, float]]:
        return [
            (float(entry["x"]), float(entry["y"]))
            for entry in frontier_blacklist
            if float(capture_index) - float(entry["added_at"]) <= 30.0
        ]

    def _strike_frontier(face_xy, reason: str, force: bool = False) -> None:
        """A planned frontier failed (path blocked / hazard-held). Two strikes
        blacklist it for ~30 captures: the planner explores the REST of the
        room instead of re-planning the same doomed goal every cycle.
        force=True blacklists immediately (provably-failing actions, e.g. a
        lock-lost face-turn, must never be retried at all)."""
        nonlocal committed_frontier_face
        strike_key = (
            int(round(float(face_xy[0]) / 0.4)),
            int(round(float(face_xy[1]) / 0.4)),
        )
        if force:
            frontier_strike_counts[strike_key] = max(
                2, frontier_strike_counts.get(strike_key, 0) + 1
            )
        else:
            frontier_strike_counts[strike_key] = frontier_strike_counts.get(strike_key, 0) + 1
        if committed_frontier_face is not None and (
            math.hypot(
                float(face_xy[0]) - committed_frontier_face[0],
                float(face_xy[1]) - committed_frontier_face[1],
            )
            <= 0.6
        ):
            # A failed attempt breaks the commitment so the planner is free
            # to pick a different frontier next cycle.
            committed_frontier_face = None
        already_listed = any(
            math.hypot(float(face_xy[0]) - ax, float(face_xy[1]) - ay) <= 0.4
            for ax, ay in _active_frontier_blacklist()
        )
        if frontier_strike_counts[strike_key] >= 2 and not already_listed:
            frontier_blacklist.append(
                {
                    "x": float(face_xy[0]),
                    "y": float(face_xy[1]),
                    "added_at": float(capture_index),
                }
            )
            del frontier_blacklist[:-24]
            print(
                f"[wander] frontier at ({float(face_xy[0]):.2f}, {float(face_xy[1]):.2f}) "
                f"blacklisted after repeated failures ({reason}); exploring elsewhere first"
            )

    def _strike_frontier_if_reached(face_xy, stop_pose: Pose2D, reason: str) -> None:
        """Strike a frontier only when its own face was actually reached.

        A camera stop well before a distant frontier says that a *local*
        obstacle needs mapping and a path replan.  Counting it against the
        distant target blacklisted open exits after two shelf/table stops.
        """
        face_distance_m = math.hypot(
            float(face_xy[0]) - float(stop_pose.x),
            float(face_xy[1]) - float(stop_pose.y),
        )
        if face_distance_m > 0.50:
            print(
                "[wander] local hazard stop "
                f"{face_distance_m:.2f}m before frontier face; mapping/replanning "
                "without striking the distant frontier"
            )
            return
        _strike_frontier(face_xy, reason)

    def _plan_drives_into_blocked_front() -> bool:
        """True when the current plan's FIRST move is forward through the
        occupied stop box — the only case where front occupancy vetoes the
        plan. A waypoint off to the side means the transit starts with an
        in-place align turn, which the stop box does not forbid; the drive
        bursts re-check the box continuously once actually moving."""
        if plan_status not in ("ok", "survey", "observe"):
            return True  # no plan that could redeem the blockage
        wp_dx = float(frontier_plan["waypoint_xy"][0]) - float(current_live_pose.x)
        wp_dy = float(frontier_plan["waypoint_xy"][1]) - float(current_live_pose.y)
        if math.hypot(wp_dx, wp_dy) <= 0.05:
            return True
        wp_delta_deg = _normalize_angle_deg(
            math.degrees(math.atan2(wp_dy, wp_dx)) - float(current_live_pose.theta_deg)
        )
        return abs(wp_delta_deg) <= 40.0

    def _rotate_away_hint(
        start_pose: Pose2D, hazard_side: str, preferred_delta_deg: float | None = None
    ) -> MotionHint | None:
        """Wedged-pocket escape of last resort (field deadlock 2026-07-11):
        forward was held, reverse had no map clearance, and the hold could
        only be released by reversing — the wanderer looped forever. Rotate
        the nose well off the hazard instead: the tracked rotation releases
        reverse-only holds in the monitor, and forward motion afterwards
        moves AWAY from the hazard under the live gates."""
        if hazard_monitor is not None:
            turn_ok, turn_denial = hazard_monitor.turn_allowed()
            if not turn_ok:
                print(f"[safety] cannot rotate away: rotation frozen ({turn_denial})")
                return None
        nonlocal recovery_turn_sign
        if recovery_turn_sign is not None:
            rotate_sign = float(recovery_turn_sign)
        elif preferred_delta_deg is not None and abs(float(preferred_delta_deg)) >= 20.0:
            # Favor the live lidar's open direction over an arbitrary
            # side-based recovery turn.
            rotate_sign = 1.0 if float(preferred_delta_deg) >= 0.0 else -1.0
        else:
            rotate_sign = -1.0 if str(hazard_side) == "left" else 1.0
        recovery_turn_sign = float(rotate_sign)
        print(
            "[safety] rotating away from the hazard "
            f"(side={hazard_side}, target=85.0deg {'cw' if rotate_sign < 0 else 'ccw'})"
        )
        rotate_meta = _turn_with_arc_tracking(
            robot=robot,
            feed=feed,
            transformed_sets=stitch_state["transformed_sets"],
            start_pose=start_pose,
            lidar_offset_forward_m=float(args.lidar_offset_forward_m),
            resolution_m=float(args.stitch_resolution_m),
            target_turn_deg=85.0,
            direction_sign=float(rotate_sign),
            turn_speed=float(args.turn_speed),
            turn_burst_s=float(args.turn_burst_s),
            turn_settle_s=float(args.turn_settle_s),
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            invert_lateral_axis=bool(args.invert_lateral_axis),
            max_distance_m=float(args.max_distance_m),
            min_range_m=float(args.min_range_m),
            min_confidence=int(args.min_confidence),
            stop_tolerance_deg=min(8.0, float(args.stop_tolerance_deg)),
            max_bursts=int(args.max_turn_bursts),
            hazard_monitor=hazard_monitor,
        )
        nonlocal rotate_away_lock_losses
        rotated_deg = float(rotate_meta["turned_deg"])
        rotate_lock_lost = bool(rotate_meta.get("lock_lost"))
        if rotate_lock_lost:
            rotate_away_lock_losses += 1
        if abs(rotated_deg) < 5.0 and not rotate_lock_lost:
            print("[safety] rotate-away made no progress; holding")
            return None
        if rotate_lock_lost:
            # The wheels turned but tracking could not follow: the robot
            # rotated an UNKNOWN amount beyond the tracked degrees. Never
            # swallow that motion (field 2026-07-11: ~70deg untracked at
            # capture 1 wrecked the map) — hand the capture solve a wide
            # theta window so it can find the true heading.
            print(
                "[safety] rotate-away lost tracking "
                f"(tracked={rotated_deg:.1f}deg of an unknown physical rotation); "
                "widening the capture search window"
            )
        lever_dx_m, lever_dy_m = _turn_lever_arm_local_delta(
            rotated_deg, lidar_offset_forward_m=float(args.lidar_offset_forward_m)
        )
        return MotionHint(
            kind="turn",
            expected_dx_local_m=float(lever_dx_m),
            expected_dy_local_m=float(lever_dy_m),
            expected_dtheta_deg=rotated_deg,
            search_xy_m=0.60 if rotate_lock_lost else float(args.turn_search_xy_m),
            search_theta_window_deg=(
                85.0 if rotate_lock_lost else float(args.turn_theta_window_deg)
            ),
            label=f"hazard_rotate_away_{capture_index:02d}",
        )

    def _sidestep_waypoint_hint(
        start_pose: Pose2D,
        hazard_side: str,
        preferred_delta_deg: float | None = None,
    ) -> MotionHint | None:
        """Repair a locally bad waypoint with a guarded lateral offset.

        The base can strafe, but the established lidar stop box and eye gate
        guard forward motion.  Use them unchanged: turn 90 degrees away from
        the close side edge, advance a short amount, then turn back before
        replanning the same frontier from the offset pose.
        """
        if hazard_monitor is not None and not hazard_monitor.turn_allowed()[0]:
            return None
        # A left-side edge means translate right (clockwise turn first), and
        # vice versa. A front edge is the normal table-at-exit case: take the
        # side that points toward the current frontier, then restore heading.
        # This is the bounded 90deg -> forward -> -90deg bypass rather than
        # repeatedly turning in place at the same blocked waypoint.
        if str(hazard_side) == "left":
            turn_sign = -1.0
        elif str(hazard_side) == "right":
            turn_sign = 1.0
        elif preferred_delta_deg is not None and abs(float(preferred_delta_deg)) >= 10.0:
            turn_sign = 1.0 if float(preferred_delta_deg) >= 0.0 else -1.0
        else:
            return None
        print(
            "[wander] repairing blocked waypoint with guarded sidestep "
            f"(edge={hazard_side}, turn=90deg {'cw' if turn_sign < 0 else 'ccw'})"
        )

        first = _turn_with_arc_tracking(
            robot=robot, feed=feed, transformed_sets=stitch_state["transformed_sets"],
            start_pose=start_pose, lidar_offset_forward_m=float(args.lidar_offset_forward_m),
            resolution_m=float(args.stitch_resolution_m), target_turn_deg=90.0,
            direction_sign=turn_sign, turn_speed=float(args.turn_speed),
            turn_burst_s=float(args.turn_burst_s), turn_settle_s=float(args.turn_settle_s),
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            invert_lateral_axis=bool(args.invert_lateral_axis), max_distance_m=float(args.max_distance_m),
            min_range_m=float(args.min_range_m), min_confidence=int(args.min_confidence),
            stop_tolerance_deg=min(8.0, float(args.stop_tolerance_deg)),
            max_bursts=int(args.max_turn_bursts), hazard_monitor=hazard_monitor,
        )
        if bool(first.get("lock_lost")) or abs(float(first["turned_deg"])) < 70.0:
            print("[wander] sidestep cancelled: could not complete the first safe quarter-turn")
            return None
        d1x, d1y = _turn_lever_arm_local_delta(
            float(first["turned_deg"]), lidar_offset_forward_m=float(args.lidar_offset_forward_m)
        )
        t0 = math.radians(float(start_pose.theta_deg))
        side_pose = Pose2D(
            x=float(start_pose.x) + math.cos(t0) * d1x - math.sin(t0) * d1y,
            y=float(start_pose.y) + math.sin(t0) * d1x + math.cos(t0) * d1y,
            theta_deg=float(first["final_theta_deg"]),
        )
        side_pose, drive_meta = _drive_with_tracking(
            robot=robot, feed=feed, zone_cfg=zone_cfg,
            transformed_sets=stitch_state["transformed_sets"], start_pose=side_pose,
            resolution_m=float(args.stitch_resolution_m), forward_speed=float(args.move_speed),
            min_effective_move_speed=float(args.min_effective_move_speed), burst_s=0.25,
            burst_count=1, inter_burst_pause_s=0.0, steer_theta_vel=0.0,
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            invert_lateral_axis=bool(args.invert_lateral_axis), max_distance_m=float(args.max_distance_m),
            min_range_m=float(args.min_range_m), min_confidence=int(args.min_confidence),
            hazard_monitor=hazard_monitor,
            map_side_guard=_map_side_guard,
        )
        shifted_m = math.hypot(float(side_pose.x) - float(start_pose.x), float(side_pose.y) - float(start_pose.y))
        if (
            bool(drive_meta.get("stopped_by_hazard"))
            or bool(drive_meta.get("stopped_by_block"))
            or bool(drive_meta.get("lock_lost"))
            or shifted_m < 0.08
        ):
            print("[wander] sidestep stopped before a useful offset; keeping the turned safety pose")
            final_pose = side_pose
        else:
            second = _turn_with_arc_tracking(
                robot=robot, feed=feed, transformed_sets=stitch_state["transformed_sets"],
                start_pose=side_pose, lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                resolution_m=float(args.stitch_resolution_m), target_turn_deg=90.0,
                direction_sign=-turn_sign, turn_speed=float(args.turn_speed),
                turn_burst_s=float(args.turn_burst_s), turn_settle_s=float(args.turn_settle_s),
                forward_angle_deg=float(args.forward_angle_deg),
                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                invert_lateral_axis=bool(args.invert_lateral_axis), max_distance_m=float(args.max_distance_m),
                min_range_m=float(args.min_range_m), min_confidence=int(args.min_confidence),
                stop_tolerance_deg=min(8.0, float(args.stop_tolerance_deg)),
                max_bursts=int(args.max_turn_bursts), hazard_monitor=hazard_monitor,
            )
            d2x, d2y = _turn_lever_arm_local_delta(
                float(second["turned_deg"]), lidar_offset_forward_m=float(args.lidar_offset_forward_m)
            )
            t1 = math.radians(float(side_pose.theta_deg))
            final_pose = Pose2D(
                x=float(side_pose.x) + math.cos(t1) * d2x - math.sin(t1) * d2y,
                y=float(side_pose.y) + math.sin(t1) * d2x + math.cos(t1) * d2y,
                theta_deg=float(second["final_theta_deg"]),
            )
        t_start = math.radians(float(start_pose.theta_deg))
        dx = float(final_pose.x) - float(start_pose.x)
        dy = float(final_pose.y) - float(start_pose.y)
        print(f"[wander] sidestep offset={math.hypot(dx, dy):.2f}m; replanning the same frontier")
        return MotionHint(
            kind="mixed",
            expected_dx_local_m=math.cos(t_start) * dx + math.sin(t_start) * dy,
            expected_dy_local_m=-math.sin(t_start) * dx + math.cos(t_start) * dy,
            expected_dtheta_deg=_normalize_angle_deg(float(final_pose.theta_deg) - float(start_pose.theta_deg)),
            search_xy_m=0.55,
            search_theta_window_deg=30.0,
            label=f"waypoint_sidestep_{capture_index:02d}",
        )

    def _live_frontier_probe_hint(start_pose: Pose2D, choice: FrontierChoice) -> MotionHint:
        """Cautiously extend a map that ends before a live lidar opening.

        The stitched grid can have no reachable unknown cells immediately
        after a doorway transit, while the current lidar still sees open
        space.  One normal safety-gated forward burst creates the next map
        evidence; completion is never the right action in that state.
        """
        # Scale the probe to the measured opening (leaving a 0.8m stop
        # margin) — a single 0.15m nudge costs a full 5-12s stitch cycle
        # and barely adds map evidence; every burst is still safety-gated.
        probe_bursts = max(1, min(3, int((float(choice.mean_distance_m) - 0.8) / 0.25)))
        print(
            "[wander] map frontier exhausted but live lidar is open; probing forward "
            f"(delta={choice.delta_deg:.1f}deg, range={choice.mean_distance_m:.2f}m, "
            f"bursts={probe_bursts})"
        )
        end_pose, probe_meta = _drive_with_tracking(
            robot=robot, feed=feed, zone_cfg=zone_cfg,
            transformed_sets=stitch_state["transformed_sets"], start_pose=start_pose,
            resolution_m=float(args.stitch_resolution_m), forward_speed=float(args.move_speed),
            min_effective_move_speed=float(args.min_effective_move_speed),
            burst_s=min(0.80, float(args.move_burst_s)), burst_count=probe_bursts,
            inter_burst_pause_s=0.0, steer_theta_vel=0.0,
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            invert_lateral_axis=bool(args.invert_lateral_axis), max_distance_m=float(args.max_distance_m),
            min_range_m=float(args.min_range_m), min_confidence=int(args.min_confidence),
            hazard_monitor=hazard_monitor,
            map_side_guard=_map_side_guard,
        )
        t0 = math.radians(float(start_pose.theta_deg))
        dx = float(end_pose.x) - float(start_pose.x)
        dy = float(end_pose.y) - float(start_pose.y)
        if bool(probe_meta.get("stopped_by_hazard")):
            print("[wander] live-frontier probe stopped by a safety gate; replanning from the tracked stop pose")
        return MotionHint(
            kind="drive",
            expected_dx_local_m=math.cos(t0) * dx + math.sin(t0) * dy,
            expected_dy_local_m=-math.sin(t0) * dx + math.cos(t0) * dy,
            expected_dtheta_deg=_normalize_angle_deg(float(end_pose.theta_deg) - float(start_pose.theta_deg)),
            search_xy_m=0.45,
            search_theta_window_deg=18.0,
            label=f"live_frontier_probe_{capture_index:02d}",
        )

    def _elevated_hazard_engaged() -> bool:
        """True when an elevated-edge stop is in effect — including a
        post-stop HOLD left behind by one. Field run 2026-07-09: the hold
        (reverse-only, vanished_near) denied every drive burst while the
        state read neither active nor blind, so the investigation trigger
        never fired and the wanderer looped plan->denied->capture forever."""
        if hazard_monitor is None:
            return False
        hs = hazard_monitor.state()
        if hs.active or hs.blind_zone:
            return True
        return bool(
            hs.hold
            and str(hs.hold_reason)
            in ("vanished_near", "line_edge", "elevated_parallax", "depth_elevated", "hazard")
        )

    def _stamp_world_segment(w1: tuple[float, float], w2: tuple[float, float], height_m: float) -> int:
        """Rasterize one world-frame segment into the planning-map layer and
        the deep-red rerun overlay. Returns how many NEW cells it added."""
        if pose_lost:
            # World coordinates computed from a lost pose are fiction; this
            # run's fallback-pose stamps helped seal the planning grid shut.
            return 0
        elevated_segments_world.append({"p1": w1, "p2": w2, "height_m": float(height_m)})
        seg_len = math.hypot(w2[0] - w1[0], w2[1] - w1[1])
        steps = max(int(seg_len / 0.04), 1)
        segment_new_cells = 0
        for step in range(steps + 1):
            t = step / steps
            cell = (
                int(round((w1[0] + t * (w2[0] - w1[0])) / 0.04)),
                int(round((w1[1] + t * (w2[1] - w1[1])) / 0.04)),
            )
            if cell not in elevated_occupied_world and len(elevated_occupied_world) < 20000:
                elevated_occupied_world.add(cell)
                segment_new_cells += 1
            # Investigation-measured geometry is approach-verified: permanent
            # (no decay metadata).
            elevated_cell_meta.pop(cell, None)
        if segment_new_cells:
            # static=True: the edge layer is session-cumulative and must be
            # visible at EVERY timeline position — logged temporally, the red
            # lines disappear whenever the viewer's capture_index playhead
            # sits before the tick the edge was stamped at.
            rr.log(
                "world/elevated_edges",
                rr.LineStrips3D(
                    [
                        [
                            [seg["p1"][0], seg["p1"][1], seg["height_m"]],
                            [seg["p2"][0], seg["p2"][1], seg["height_m"]],
                        ]
                        for seg in elevated_segments_world
                    ],
                    colors=[[200, 0, 0]] * len(elevated_segments_world),
                    radii=0.015,
                ),
                static=True,
            )
        return segment_new_cells

    stop_debug_dir = output_dir / "stop_debug"
    stop_debug_dir.mkdir(parents=True, exist_ok=True)
    stop_debug_counter = {"n": 0}
    # Rate limiter for planning-denial bundles (same reason at most every 20s).
    planning_denial_last: dict[str, object] = {"reason": "", "mono": -1e9}

    def _dump_stop_debug(stop_pose: Pose2D, note: str) -> None:
        """Field-diagnosis bundle per hazard stop: what the robot SAW (eye
        overlays) and BELIEVED (pose, gate state) at the moment of the stop,
        so misplaced map geometry can be traced to mask vs projection vs
        pose instead of inferred from logs."""
        if hazard_monitor is None:
            return
        import cv2 as _cv2

        stop_debug_counter["n"] += 1
        bundle_index = stop_debug_counter["n"]
        hs = hazard_monitor.state()
        for cam in ("front_left", "front_right", "panorama", "bottom"):
            annotated = hazard_monitor.annotated(cam)
            if annotated is not None:
                _cv2.imwrite(
                    str(stop_debug_dir / f"stop{bundle_index:03d}_{cam}.png"), annotated
                )
        meta = {
            "note": str(note),
            "pose_xy_theta": [
                round(float(stop_pose.x), 4),
                round(float(stop_pose.y), 4),
                round(float(stop_pose.theta_deg), 2),
            ],
            "reason": hs.reason,
            "side": hs.side,
            "est_distance_m": hs.est_distance_m,
            "classification": hs.classification,
            "detail": hs.detail,
            # Ground-gate + hold state: planning-time denials are mostly
            # ground_stop_* / post_stop_hold and were undiagnosable without
            # these (field 2026-07-13 night: ~8 ground denials, zero bundles).
            "ground_active": bool(getattr(hs, "ground_active", False)),
            "ground_distance_m": getattr(hs, "ground_distance_m", None),
            "ground_side": getattr(hs, "ground_side", "none"),
            "hold": bool(getattr(hs, "hold", False)),
            "hold_reason": getattr(hs, "hold_reason", ""),
            "map_cells_total": len(elevated_occupied_world),
        }
        (stop_debug_dir / f"stop{bundle_index:03d}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"[debug] stop bundle #{bundle_index} saved -> {stop_debug_dir}")

    def _plan_edge_survey(grid, robot_pose: Pose2D) -> dict | None:
        """EDGE-SURVEY phase: pick the nearest POTENTIAL (yellow) cluster
        with attempts remaining and plan a transit to a vantage whose
        observation bearing differs >=30deg from the cluster's first
        sighting. Bleed lies along the original viewing ray, so the second
        angle either re-detects the cells at the same world position
        (promote to red — real edge, dimensions solid) or cleanly misses
        (decay — phantom, corridor opens). Runs when normal frontier
        exploration is dry, BEFORE the squeeze/verification ladder: solidify
        the edges first, then push through what remains."""
        if grid is None or not elevated_cell_meta:
            return None
        unconfirmed_cells = [
            c
            for c, m in elevated_cell_meta.items()
            if len(m) >= 3 and not m[2] and c in elevated_occupied_world
        ]
        if not unconfirmed_cells:
            return None
        rx, ry = float(robot_pose.x), float(robot_pose.y)
        unconfirmed_cells.sort(
            key=lambda c: (c[0] * 0.04 - rx) ** 2 + (c[1] * 0.04 - ry) ** 2
        )
        now_mono = time.monotonic()
        tried_buckets: set[tuple[int, int]] = set()
        for seed_cell in unconfirmed_cells[:24]:
            seed_x, seed_y = seed_cell[0] * 0.04, seed_cell[1] * 0.04
            cluster = [
                c
                for c in unconfirmed_cells
                if abs(c[0] * 0.04 - seed_x) <= 0.5 and abs(c[1] * 0.04 - seed_y) <= 0.5
            ]
            centroid_x = sum(c[0] for c in cluster) / len(cluster) * 0.04
            centroid_y = sum(c[1] for c in cluster) / len(cluster) * 0.04
            bucket = (int(round(centroid_x / 0.3)), int(round(centroid_y / 0.3)))
            if bucket in tried_buckets:
                continue
            tried_buckets.add(bucket)
            if edge_survey_attempts.get(bucket, 0) >= 3:
                continue
            first_bearings = [float(elevated_cell_meta[c][0]) for c in cluster]
            base_bearing = math.degrees(
                math.atan2(
                    sum(math.sin(math.radians(b)) for b in first_bearings),
                    sum(math.cos(math.radians(b)) for b in first_bearings),
                )
            )
            # Candidate vantages: side-offset bearings (kept <=95deg — past
            # ~100deg the object may self-occlude its own near boundary).
            for bearing_offset in (55.0, -55.0, 75.0, -75.0, 40.0, -40.0, 90.0, -90.0):
                observe_bearing = base_bearing + bearing_offset
                for standoff_m in (1.0, 1.3, 0.8):
                    vantage_x = centroid_x - standoff_m * math.cos(
                        math.radians(observe_bearing)
                    )
                    vantage_y = centroid_y - standoff_m * math.sin(
                        math.radians(observe_bearing)
                    )
                    candidate = _plan_frontier_path(
                        grid=grid,
                        robot_xy=(rx, ry),
                        robot_radius_m=float(args.robot_radius_m),
                        target_xy=(vantage_x, vantage_y),
                        robot_theta_deg=float(robot_pose.theta_deg),
                    )
                    if str(candidate.get("status")) != "ok":
                        continue
                    goal_x, goal_y = candidate["goal_xy"]
                    if math.hypot(goal_x - vantage_x, goal_y - vantage_y) > 0.35:
                        continue
                    achieved_bearing = math.degrees(
                        math.atan2(centroid_y - goal_y, centroid_x - goal_x)
                    )
                    if abs(_normalize_angle_deg(achieved_bearing - base_bearing)) < 30.0:
                        continue
                    if (
                        math.hypot(goal_x - rx, goal_y - ry) < 0.35
                        and abs(
                            _normalize_angle_deg(
                                math.degrees(
                                    math.atan2(centroid_y - ry, centroid_x - rx)
                                )
                                - float(robot_pose.theta_deg)
                            )
                        )
                        < 35.0
                    ):
                        # Already standing at this vantage FACING the
                        # cluster: the look is happening — re-adopting the
                        # same spot just spins in place (field run 21:
                        # "attempt 1/3" re-adopted forever). Try the next
                        # bearing offset for a genuinely different angle.
                        continue
                    # Attempt accounting: the adoption clock only advances
                    # when an attempt is charged — every-adoption timestamp
                    # updates made the 30s debounce unreachable and the
                    # counter stuck at 1 forever (run 21).
                    if now_mono - edge_survey_last_adopt.get(bucket, -1e9) > 30.0:
                        edge_survey_attempts[bucket] = (
                            edge_survey_attempts.get(bucket, 0) + 1
                        )
                        edge_survey_last_adopt[bucket] = now_mono
                    candidate["face_xy"] = (centroid_x, centroid_y)
                    print(
                        "[wander] EDGE SURVEY: observing yellow cluster at "
                        f"({centroid_x:.2f}, {centroid_y:.2f}) from a new angle "
                        f"(first sighting {base_bearing:.0f}deg -> new "
                        f"{achieved_bearing:.0f}deg, vantage=({goal_x:.2f}, {goal_y:.2f}), "
                        f"attempt {edge_survey_attempts.get(bucket, 0)}/3, "
                        f"path={float(candidate.get('path_length_m', 0.0) or 0.0):.2f}m)"
                    )
                    return candidate
        return None

    def _nearest_mapped_elevated_m(pose: Pose2D) -> float | None:
        """Distance from the body center to the nearest MAPPED elevated cell
        within 1.0m, or None. The map is the only sensor that covers the
        robot's SIDES at tabletop height (panorama ±52° forward, lidar under
        tabletops, bottom camera on the floor) — field 2026-07-17: the robot
        ground its shoulder along a mapped table edge while turning beside
        it ("no rotation progress")."""
        if not elevated_occupied_world:
            return None
        px, py = float(pose.x), float(pose.y)
        best: float | None = None
        for cell_x, cell_y in elevated_occupied_world:
            dist = math.hypot(cell_x * 0.04 - px, cell_y * 0.04 - py)
            if dist <= 1.0 and (best is None or dist < best):
                best = dist
        return best

    map_side_guard_state = {"pose": None, "streak": 0}

    def _map_side_guard(pose: Pose2D) -> str | None:
        """Drive-burst veto for the virtual side bumper: halt FORWARD motion
        only for mapped cells in the forward collision corridor — ahead of
        the body center and within body width. Cells beside/behind cannot
        be struck by driving forward; the original direction-blind radius
        check deadlocked the planner for 130+ zero-motion cycles (field
        2026-07-17 run 18: a cell 0.26m from center, beside/behind the
        robot after a reverse, vetoed every forward burst forever).
        Threshold radius + 2cm laterally so legitimate squeeze passes
        (planned at radius - 5cm inflation) are not blocked."""
        if not elevated_occupied_world:
            return None
        guard_r = float(args.robot_radius_m) + 0.02
        px, py = float(pose.x), float(pose.y)
        theta_rad = math.radians(float(pose.theta_deg))
        cos_g, sin_g = math.cos(theta_rad), math.sin(theta_rad)
        for cell_x, cell_y in elevated_occupied_world:
            dxc = cell_x * 0.04 - px
            dyc = cell_y * 0.04 - py
            if dxc * dxc + dyc * dyc > 1.0:
                continue
            fwd = cos_g * dxc + sin_g * dyc
            lat = -sin_g * dxc + cos_g * dyc
            if -0.02 <= fwd <= guard_r + 0.10 and abs(lat) <= guard_r:
                prev_pose = map_side_guard_state["pose"]
                if (
                    prev_pose is not None
                    and math.hypot(px - prev_pose[0], py - prev_pose[1]) < 0.05
                ):
                    map_side_guard_state["streak"] += 1
                else:
                    map_side_guard_state["streak"] = 1
                map_side_guard_state["pose"] = (px, py)
                return (
                    f"mapped elevated cell {math.hypot(dxc, dyc):.2f}m ahead in the "
                    "body corridor (side-collision guard)"
                )
        return None

    # Synthetic blockers REMOVED (user directive 2026-07-17, final): a stop
    # with no measurable geometry stamps NOTHING. Field run: a fabricated
    # 0.6m "blocker" vastly overestimated a desk's length and walled off
    # the room entrance. Stops still teach through frontier strikes and
    # blacklists (non-fabricating); only MEASURED edges reach the map.

    def _stamp_confirmed_edges(stamp_pose: Pose2D, newest_only: bool = False) -> int:
        """Transform freshly measured (robot-frame) edge segments into world
        coordinates at the given tracked pose and stamp them into the
        planning map. Only fresh measurements are used — the pose must match
        the measurement moment. newest_only: the robot was MOVING until the
        stop, so only the final confirmation tick (the one that tripped the
        gate, within ~0.3s of it) matches the stop pose; earlier ones were
        measured metres back and would stamp the edge beyond its true spot."""
        if hazard_monitor is None:
            return 0
        if pose_lost:
            # A lost pose puts every stamped cell somewhere fictional. Drain
            # and DROP the pending measurements (they are stamped against
            # the wrong pose forever) — the map stays clean and the eyes
            # keep measuring fresh ones after recovery.
            hazard_monitor.drain_confirmed_edges()
            if hasattr(hazard_monitor, "drain_floor_evidence"):
                hazard_monitor.drain_floor_evidence()
            if hasattr(hazard_monitor, "drain_view_reports"):
                hazard_monitor.drain_view_reports()
            return 0
        drained_edges = hazard_monitor.drain_confirmed_edges()
        if newest_only and drained_edges:
            newest_mono = max(float(edge.monotonic) for edge in drained_edges)
            drained_edges = [
                edge for edge in drained_edges if newest_mono - float(edge.monotonic) <= 0.3
            ]
        now_mono = time.monotonic()
        theta_rad = math.radians(float(stamp_pose.theta_deg))
        cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
        new_cells = 0
        new_points = 0
        segment_added = False
        # FREE-SPACE pass first: depth-observed floor CLEARS any elevated
        # cell it lands on. The map is otherwise add-only, so one overshot
        # footprint would wall off a doorway for the rest of the run even
        # while the eyes stare straight down its traversable floor (field
        # 2026-07-12). Clear-then-stamp: this drain's own footprints re-add
        # anything genuinely occupied. 3x3 neighborhood: floor evidence is
        # 6cm-decimated while cells are 4cm — exact-cell clearing would
        # leave stripes.
        cleared_cells = 0
        decayed_cells_total = 0
        promoted_cells = 0
        floor_batches = (
            hazard_monitor.drain_floor_evidence()
            if hasattr(hazard_monitor, "drain_floor_evidence")
            else []
        )
        if newest_only and floor_batches:
            newest_floor = max(float(ev.monotonic) for ev in floor_batches)
            floor_batches = [
                ev for ev in floor_batches if newest_floor - float(ev.monotonic) <= 0.3
            ]
        newest_floor_mono: dict[str, float] = {}
        for ev in floor_batches:
            if float(ev.monotonic) > newest_floor_mono.get(str(ev.eye), -1.0):
                newest_floor_mono[str(ev.eye)] = float(ev.monotonic)
        for ev in floor_batches:
            if now_mono - float(ev.monotonic) > 1.0:
                continue
            if float(ev.monotonic) < newest_floor_mono.get(str(ev.eye), -1.0):
                continue
            for pf, pl in ev.points_robot_xy:
                wx = float(stamp_pose.x) + float(pf) * cos_t - float(pl) * sin_t
                wy = float(stamp_pose.y) + float(pf) * sin_t + float(pl) * cos_t
                cx, cy = int(round(wx / 0.04)), int(round(wy / 0.04))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        cell = (cx + dx, cy + dy)
                        elevated_pending_world.pop(cell, None)
                        if cell in elevated_occupied_world:
                            elevated_occupied_world.discard(cell)
                            elevated_points_world.pop(cell, None)
                            elevated_cell_meta.pop(cell, None)
                            cleared_cells += 1
        # Footprints: the NEWEST frame per eye is the stamping frame; the
        # frame before it (>=0.4s older, within a 3.0s window — the robot is
        # settled/travel-compensated over that span) is CONFIRMATION-ONLY.
        # Far cells promote on the INTERSECTION of the two frames: an AND
        # across frames is noise-resistant where stamping the union of the
        # whole queue smeared an 800-point blob (field 2026-07-12). A
        # newest-only + cross-checkpoint persistence design mapped ZERO
        # cells in the field (2026-07-13 evening run): checkpoints are taken
        # at different poses/headings, so far cells never re-observed and
        # pended forever.
        footprint_monos: dict[str, list[float]] = {}
        for edge in drained_edges:
            if getattr(edge, "points_robot_xy", ()) or ():
                eye_key = str(edge.eye)
                mono = float(edge.monotonic)
                if mono not in footprint_monos.setdefault(eye_key, []):
                    footprint_monos[eye_key].append(mono)
        stamp_frame_mono: dict[str, float] = {}
        confirm_frame_mono: dict[str, float] = {}
        for eye_key, monos in footprint_monos.items():
            monos.sort(reverse=True)
            fresh = [m for m in monos if now_mono - m <= 3.0]
            if not fresh:
                continue
            stamp_frame_mono[eye_key] = fresh[0]
            for older in fresh[1:]:
                if fresh[0] - older >= 0.4:
                    confirm_frame_mono[eye_key] = older
                    break
        for edge in drained_edges:
            footprint = getattr(edge, "points_robot_xy", ()) or ()
            if footprint:
                eye_key = str(edge.eye)
                obs_mono = float(edge.monotonic)
                if obs_mono == confirm_frame_mono.get(eye_key):
                    # Confirmation-only frame: its points arm pending cells;
                    # they never stamp directly (weaker pose match).
                    for pf, pl in footprint:
                        wx = float(stamp_pose.x) + float(pf) * cos_t - float(pl) * sin_t
                        wy = float(stamp_pose.y) + float(pf) * sin_t + float(pl) * cos_t
                        cell = (int(round(wx / 0.04)), int(round(wy / 0.04)))
                        if (
                            cell not in elevated_occupied_world
                            and cell not in elevated_pending_world
                            and len(elevated_pending_world) < 40000
                        ):
                            elevated_pending_world[cell] = (obs_mono, 1)
                    continue
                if obs_mono != stamp_frame_mono.get(eye_key):
                    continue
                # Stamping frame. DISTANCE-SCALED TRUST: points within 0.9m
                # stamp immediately (close range is monocular depth's good
                # regime, and a stop must teach the planner NOW — directive:
                # every stop teaches something); farther points only become
                # obstacles when a pending observation >= 0.4s older agrees
                # (this drain's confirmation frame or a previous drain). The
                # exit-corridor splats were all 1.3..1.9m one-offs.
                # Promotions are BUDGETED per vantage (see
                # elevated_vantage_promotions) so a wedged robot cannot
                # confirm its own artifacts indefinitely from one pose.
                vantage_bucket = (
                    int(round(float(stamp_pose.x) / 0.25)),
                    int(round(float(stamp_pose.y) / 0.25)),
                    int(round((float(stamp_pose.theta_deg) % 360.0) / 20.0)),
                )
                vantage_used = elevated_vantage_promotions.get(vantage_bucket, 0)
                vantage_capped = False
                for pf, pl in footprint:
                    wx = float(stamp_pose.x) + float(pf) * cos_t - float(pl) * sin_t
                    wy = float(stamp_pose.y) + float(pf) * sin_t + float(pl) * cos_t
                    cell = (int(round(wx / 0.04)), int(round(wy / 0.04)))
                    if cell in elevated_occupied_world:
                        cell_meta = elevated_cell_meta.get(cell)
                        if cell_meta is not None:
                            # Re-detected: DECREMENT misses, don't reset.
                            # Systematic artifacts (edge-bleed re-detected
                            # from the same vantage every pass) must still
                            # decay when clean looks outnumber dirty ones.
                            cell_meta[1] = max(0, int(cell_meta[1]) - 1)
                            # CROSS-ANGLE CONFIRMATION: bleed lies along the
                            # viewing ray, so a re-detection at the SAME
                            # world position from a bearing >=28deg away is
                            # near-proof of a real edge — promote POTENTIAL
                            # (yellow) to CONFIRMED (red).
                            if len(cell_meta) >= 3 and not cell_meta[2]:
                                obs_bearing_deg = math.degrees(
                                    math.atan2(
                                        wy - float(stamp_pose.y),
                                        wx - float(stamp_pose.x),
                                    )
                                )
                                if (
                                    abs(
                                        _normalize_angle_deg(
                                            obs_bearing_deg - float(cell_meta[0])
                                        )
                                    )
                                    >= 28.0
                                ):
                                    cell_meta[2] = 1
                                    promoted_cells += 1
                        continue
                    if float(pf) > 0.9:
                        pending = elevated_pending_world.get(cell)
                        if pending is None:
                            if len(elevated_pending_world) < 40000:
                                elevated_pending_world[cell] = (obs_mono, 1)
                            continue
                        if obs_mono - pending[0] < 0.4:
                            continue
                    if vantage_used >= 60:
                        vantage_capped = True
                        continue
                    if len(elevated_occupied_world) < 20000:
                        elevated_pending_world.pop(cell, None)
                        elevated_occupied_world.add(cell)
                        # [first_bearing_deg, misses, confirmed] — new cells
                        # start POTENTIAL (yellow) until re-detected from a
                        # bearing >=28deg away (cross-angle confirmation).
                        elevated_cell_meta[cell] = [
                            math.degrees(
                                math.atan2(wy - float(stamp_pose.y), wx - float(stamp_pose.x))
                            ),
                            0,
                            0,
                        ]
                        vantage_used += 1
                        new_cells += 1
                        if len(elevated_points_world) < 6000:
                            elevated_points_world[cell] = (wx, wy, float(edge.height_m))
                            new_points += 1
                elevated_vantage_promotions[vantage_bucket] = vantage_used
                if vantage_capped:
                    print(
                        "[edge-map] vantage promotion budget reached at this pose; "
                        "further cells need a new viewpoint"
                    )
                # Pending entries that never re-confirm are noise; sweep the
                # stale ones so the dict cannot grow without bound.
                if len(elevated_pending_world) > 20000:
                    stale_before = now_mono - 30.0
                    for stale_cell in [
                        c for c, (m, _) in elevated_pending_world.items() if m < stale_before
                    ]:
                        del elevated_pending_world[stale_cell]
                continue
            if now_mono - float(edge.monotonic) > 1.0:
                continue
            w1 = (
                float(stamp_pose.x) + edge.p1_robot_xy[0] * cos_t - edge.p1_robot_xy[1] * sin_t,
                float(stamp_pose.y) + edge.p1_robot_xy[0] * sin_t + edge.p1_robot_xy[1] * cos_t,
            )
            w2 = (
                float(stamp_pose.x) + edge.p2_robot_xy[0] * cos_t - edge.p2_robot_xy[1] * sin_t,
                float(stamp_pose.y) + edge.p2_robot_xy[0] * sin_t + edge.p2_robot_xy[1] * cos_t,
            )
            seg_cells = _stamp_world_segment(w1, w2, float(edge.height_m))
            new_cells += seg_cells
            segment_added = segment_added or seg_cells > 0
        # NEGATIVE-EVIDENCE DECAY: fused frames the monitor analyzed report
        # their (possibly empty) footprint. A decayable cell the view
        # re-inspected — good range, in-view, from a bearing similar to the
        # one it was stamped from — without re-detecting counts a miss;
        # 3 misses remove it. Pose association uses the same freshness
        # windows as edge stamping (the robot is settled at drains).
        view_reports = (
            hazard_monitor.drain_view_reports()
            if hasattr(hazard_monitor, "drain_view_reports")
            else []
        )
        if view_reports and elevated_cell_meta:
            newest_report_mono = max(float(r.monotonic) for r in view_reports)
            if now_mono - newest_report_mono <= 3.0:
                report_window_s = 0.3 if newest_only else 1.5
                selected_reports = [
                    r
                    for r in view_reports
                    if newest_report_mono - float(r.monotonic) <= report_window_s
                ][-2:]
                decayed_cells = 0
                for report in selected_reports:
                    report_pts_world = [
                        (
                            float(stamp_pose.x) + float(pf) * cos_t - float(pl) * sin_t,
                            float(stamp_pose.y) + float(pf) * sin_t + float(pl) * cos_t,
                        )
                        for pf, pl in (report.points_robot_xy or ())
                    ]
                    for cell in list(elevated_cell_meta.keys()):
                        if cell not in elevated_occupied_world:
                            elevated_cell_meta.pop(cell, None)
                            continue
                        cell_meta = elevated_cell_meta[cell]
                        cell_wx, cell_wy = cell[0] * 0.04, cell[1] * 0.04
                        dxc = cell_wx - float(stamp_pose.x)
                        dyc = cell_wy - float(stamp_pose.y)
                        fwd = cos_t * dxc + sin_t * dyc
                        lat = -sin_t * dxc + cos_t * dyc
                        rng = math.hypot(dxc, dyc)
                        if not (
                            0.35 <= rng <= 1.55
                            and fwd > 0.0
                            and abs(math.degrees(math.atan2(lat, fwd))) <= 42.0
                        ):
                            continue
                        world_bearing = math.degrees(math.atan2(dyc, dxc))
                        bearing_diff = abs(
                            _normalize_angle_deg(world_bearing - float(cell_meta[0]))
                        )
                        cell_confirmed = len(cell_meta) >= 3 and bool(cell_meta[2])
                        # Verdict windows: CONFIRMED (red) cells only accept
                        # verdicts from near the original side (<=55deg — a
                        # far-side view legitimately sees a different leading
                        # boundary). POTENTIAL (yellow) cells accept up to
                        # 100deg: a clean look from a genuinely different
                        # angle is exactly the cross-check that disproves
                        # bleed; beyond ~100deg the object may self-occlude
                        # its own near boundary, so no verdict there either.
                        if bearing_diff > (55.0 if cell_confirmed else 100.0):
                            continue
                        near_hit = any(
                            (px - cell_wx) ** 2 + (py - cell_wy) ** 2 <= 0.15 * 0.15
                            for px, py in report_pts_world
                        )
                        if near_hit:
                            cell_meta[1] = max(0, int(cell_meta[1]) - 1)
                            if not cell_confirmed and bearing_diff >= 28.0:
                                cell_meta[2] = 1
                                promoted_cells += 1
                        else:
                            cell_meta[1] = int(cell_meta[1]) + 1
                            if cell_meta[1] >= 3:
                                elevated_occupied_world.discard(cell)
                                elevated_pending_world.pop(cell, None)
                                elevated_points_world.pop(cell, None)
                                elevated_cell_meta.pop(cell, None)
                                decayed_cells += 1
                if decayed_cells:
                    # Counted separately from floor clearing — merging the
                    # counts made the "floor evidence cleared N" line lie
                    # (run 19 logs: identical N on both lines).
                    decayed_cells_total += decayed_cells
                    print(
                        f"[edge-map] negative evidence decayed {decayed_cells} unconfirmed cells "
                        f"(map_cells={len(elevated_occupied_world)})"
                    )
        if new_points or cleared_cells or decayed_cells_total or promoted_cells:
            confirmed_pts: list[list[float]] = []
            potential_pts: list[list[float]] = []
            for point_cell, (px, py, ph) in elevated_points_world.items():
                point_meta = elevated_cell_meta.get(point_cell)
                if point_meta is None or (len(point_meta) >= 3 and point_meta[2]):
                    confirmed_pts.append([px, py, ph])
                else:
                    potential_pts.append([px, py, ph])
            rr.log(
                "world/elevated_regions",
                rr.Points3D(
                    confirmed_pts,
                    colors=[[200, 0, 0]] * len(confirmed_pts),
                    radii=0.02,
                ),
                static=True,
            )
            rr.log(
                "world/elevated_potential",
                rr.Points3D(
                    potential_pts,
                    colors=[[240, 205, 30]] * len(potential_pts),
                    radii=0.02,
                ),
                static=True,
            )
        if promoted_cells:
            print(
                f"[edge-map] cross-angle confirmed {promoted_cells} cells (yellow -> red, "
                f"map_cells={len(elevated_occupied_world)})"
            )
        if cleared_cells:
            print(
                "[edge-map] floor evidence cleared "
                f"{cleared_cells} stale cells (map_cells={len(elevated_occupied_world)})"
            )
        if new_points:
            print(
                "[edge-map] elevated footprint mapped "
                f"(+{new_points} points, map_cells={len(elevated_occupied_world)}, "
                f"pending={len(elevated_pending_world)})"
            )
        if segment_added and elevated_segments_world:
            latest = elevated_segments_world[-1]
            latest_width_m = math.hypot(
                latest["p2"][0] - latest["p1"][0], latest["p2"][1] - latest["p1"][1]
            )
            print(
                "[edge-map] elevated edge mapped "
                f"(height={latest['height_m']:.2f}m, width={latest_width_m:.2f}m, "
                f"segments={len(elevated_segments_world)}, map_cells={len(elevated_occupied_world)})"
            )
        return new_cells

    def _settle_and_stamp(stamp_pose: Pose2D) -> None:
        """Flush measurements taken mid-motion (their pose is unknowable),
        let the monitor re-measure at rest, then stamp with the settled pose."""
        if hazard_monitor is None:
            return
        hazard_monitor.drain_confirmed_edges()
        if hasattr(hazard_monitor, "drain_floor_evidence"):
            hazard_monitor.drain_floor_evidence()
        # Two fresh frames per eye (~0.56s cadence each) must accumulate so
        # the map's two-frame persistence gate can confirm far cells from
        # this settled pose, not just stamp the near band.
        time.sleep(1.3)
        _stamp_confirmed_edges(stamp_pose)

    def _investigate_edge_hint(start_pose: Pose2D) -> MotionHint | None:
        """Active edge survey (--edge-mapping on): back off for a parallax
        runway, re-approach slowly so the edge gets MEASURED, back off to a
        stand-off, then sweep the heading so the edge's full extent crosses
        both eyes. Segments are stamped at each settled tracked pose; the
        next planning cycle routes around the mapped boundary."""
        print("[edge-map] investigating elevated edge (back off -> measure -> sweep)")
        pose = start_pose
        cells_at_start = len(elevated_occupied_world)
        escape_hint = _attempt_reverse_escape_hint(pose, False)
        if escape_hint is None:
            print("[edge-map] no clearance to back off; skipping investigation")
            return None
        pose = _advance_pose(pose, escape_hint)
        _settle_and_stamp(pose)

        # Measurement pass: the slow re-approach IS the parallax baseline;
        # the safety gate stops it before contact.
        pose, approach_meta = _drive_with_tracking(
            robot=robot,
            feed=feed,
            zone_cfg=zone_cfg,
            transformed_sets=stitch_state["transformed_sets"],
            start_pose=pose,
            resolution_m=float(args.stitch_resolution_m),
            forward_speed=float(args.move_speed),
            min_effective_move_speed=float(args.min_effective_move_speed),
            burst_s=min(0.9, float(args.move_burst_s)),
            burst_count=2,
            inter_burst_pause_s=0.15,
            steer_theta_vel=0.0,
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            invert_lateral_axis=bool(args.invert_lateral_axis),
            max_distance_m=float(args.max_distance_m),
            min_range_m=float(args.min_range_m),
            min_confidence=int(args.min_confidence),
            hazard_monitor=hazard_monitor,
        )
        _settle_and_stamp(pose)
        if len(elevated_occupied_world) == cells_at_start:
            if bool(approach_meta.get("stopped_by_hazard")):
                # The re-approach got gate-stopped again but produced no
                # publishable geometry. NO fabricated scar (synthetic
                # blockers removed) — the frontier strike below is the
                # teaching mechanism.
                print(
                    "[edge-map] hazard stop with no measurable geometry; nothing mapped "
                    "(strikes handle repeat offenders)"
                )
            else:
                # The re-approach drove its full budget WITHOUT a stop: the
                # earlier detection did not reproduce. Treat it as a false
                # positive and map nothing (field 2026-07-11: stamping here
                # anyway planted a phantom wall in the middle of the room).
                print(
                    "[edge-map] re-approach passed cleanly; earlier stop looks like a "
                    "false positive - nothing mapped"
                )

        # Back to a safe survey stand-off (also releases any post-stop hold).
        escape_hint = _attempt_reverse_escape_hint(pose, False)
        if escape_hint is not None:
            pose = _advance_pose(pose, escape_hint)
            _settle_and_stamp(pose)

        # The old 35deg -> 70deg -> 35deg eye sweep ended at the original
        # heading but spent three turns oscillating in place. The leading
        # boundary already suffices for the 2-D planner; preserve this
        # stand-off heading and let the next cycle attempt a forward route.
        print("[edge-map] boundary measured; skipping rotational sweep and replanning forward")

        # One composite motion hint for the checkpoint capture: net tracked
        # motion from where the investigation began to where it ended.
        net_dx_world = float(pose.x) - float(start_pose.x)
        net_dy_world = float(pose.y) - float(start_pose.y)
        theta0_rad = math.radians(float(start_pose.theta_deg))
        cos0, sin0 = math.cos(theta0_rad), math.sin(theta0_rad)
        print(
            "[edge-map] investigation complete; capturing checkpoint and replanning "
            f"around {len(elevated_occupied_world)} mapped edge cells"
        )
        return MotionHint(
            kind="mixed",
            expected_dx_local_m=cos0 * net_dx_world + sin0 * net_dy_world,
            expected_dy_local_m=-sin0 * net_dx_world + cos0 * net_dy_world,
            expected_dtheta_deg=_normalize_angle_deg(
                float(pose.theta_deg) - float(start_pose.theta_deg)
            ),
            search_xy_m=0.50,
            search_theta_window_deg=28.0,
            label=f"edge_investigation_{capture_index:02d}",
        )

    try:
        last_frame_id, latest_frame = _wait_for_initial_frame(feed, timeout_s=5.0)
        print(
            "[wander] LiDAR feed ready "
            f"(frame_id={last_frame_id}, host_rev={latest_frame.revolution_index}, viewer={viewer_url or 'local'})"
        )

        # SELF-MASK calibration: while still stationary, learn which bearings
        # the lidar sees the robot's OWN arms/shell in and drop those returns
        # from every future frame (user 2026-07-18: "the arms are a bit in the
        # way of the lidar"). Must run BEFORE the stop-box baseline so the
        # baseline measures a clean view and keeps its full trigger headroom.
        mask_sample_frames = []
        mask_frame_id = int(last_frame_id)
        for _mask_index in range(20):
            mask_frame_id_new, mask_frame = feed.wait_for_frame_after(
                after_frame_id=mask_frame_id,
                timeout_s=1.0,
                min_frame_advances=1,
            )
            if mask_frame is None:
                break
            mask_frame_id = int(mask_frame_id_new)
            if len(mask_frame.points) < 150:
                continue
            mask_sample_frames.append(mask_frame)
            if len(mask_sample_frames) >= 8:
                break
        if len(mask_sample_frames) >= 4:
            masked_bins, masked_deg = feed.calibrate_self_mask(mask_sample_frames)
            last_frame_id = mask_frame_id
            if masked_bins > 0:
                print(
                    f"[wander] lidar self-mask ARMED: {masked_deg:.0f}deg of bearings show "
                    "persistent close returns (robot arms/shell) — those returns are now "
                    "filtered from mapping, stop box, and frontier detection"
                )
                # The mask zeroes the stop-box baseline, but arm JITTER during
                # motion still leaks a handful of points past it (run 11: 6-14
                # per frame — with trigger at baseline+6=6 that blocked every
                # single drive burst at 0.05m). Floor the trigger so residual
                # self-jitter cannot block; a thin real obstacle (~12+ pts in
                # the box) and any wall (30-60) still stop the robot.
                if zone_cfg.min_points_to_trigger < 12:
                    zone_cfg.min_points_to_trigger = 12
                    print(
                        "[wander] stop-box trigger floored at 12 points while the self-mask "
                        "is armed (residual arm jitter must not veto drives)"
                    )
            else:
                print("[wander] lidar self-mask: no persistent self-returns found (clean view)")
            if masked_deg > 150.0:
                print(
                    f"[wander] WARNING: self-mask covers {masked_deg:.0f}deg — if the robot is "
                    "parked close to a wall, that wall just got masked and collision "
                    "detection is blinded on those bearings. Reposition with clear space "
                    "and restart if this is not the arms."
                )
        else:
            print(
                "[wander] lidar self-mask skipped (not enough healthy stationary revolutions); "
                "arm returns will pollute the stop box and map"
            )

        # Measure how many points the stationary lidar ALWAYS reports inside the
        # stop box (robot shell / mounts). Blocking then triggers only on points
        # above this baseline; otherwise the robot believes it is permanently
        # blocked and spends the whole run rotating in place.
        baseline_samples: list[int] = []
        squeeze_baseline_samples: list[int] = []
        baseline_frame_id = int(last_frame_id)
        for _sample_index in range(24):
            sample_frame_id, sample_frame = feed.wait_for_frame_after(
                after_frame_id=baseline_frame_id,
                timeout_s=1.0,
                min_frame_advances=1,
            )
            if sample_frame is None:
                break
            baseline_frame_id = int(sample_frame_id)
            if len(sample_frame.points) < 150:
                # Degraded revolution (spin-up / USB stutter): a baseline
                # measured from 7-11-point frames (field 2026-07-17:
                # samples=[0,19,...] baseline=9 vs the true ~40) makes the
                # stop box simultaneously hair-triggered and blind.
                print(
                    f"[wander] skipping degraded revolution in baseline sampling "
                    f"(points={len(sample_frame.points)})"
                )
                continue
            baseline_samples.append(_blocked_points_for_frame(sample_frame, zone_cfg))
            # Same frame, narrow squeeze band: its own self-hit baseline.
            zone_cfg.squeeze_active = True
            squeeze_baseline_samples.append(_blocked_points_for_frame(sample_frame, zone_cfg))
            zone_cfg.squeeze_active = False
            if len(baseline_samples) >= 10:
                break
        if len(baseline_samples) < 5:
            raise RuntimeError(
                "Could not collect 5 healthy lidar revolutions for the stop-box baseline "
                f"(got {len(baseline_samples)}). The lidar feed is degraded — check the "
                "stream host on the Pi before rerunning."
            )
        if baseline_samples:
            zone_cfg.baseline_points = int(np.median(np.asarray(baseline_samples, dtype=np.int32)))
            zone_cfg.squeeze_baseline_points = int(
                np.median(np.asarray(squeeze_baseline_samples, dtype=np.int32))
            )
            last_frame_id = baseline_frame_id
        print(
            "[wander] stop box self-hit baseline "
            f"(samples={baseline_samples}, baseline={zone_cfg.baseline_points}, "
            f"trigger_at={zone_cfg.blocked_trigger_count()} points; squeeze band "
            f"{zone_cfg.squeeze_half_width_m:.2f}m baseline={zone_cfg.squeeze_baseline_points})"
        )
        if zone_cfg.baseline_points >= 60:
            print(
                f"[wander] WARNING: the stop box is nearly saturated at rest "
                f"(baseline={zone_cfg.baseline_points} points in a 0.14m box) — the robot is "
                "almost certainly WEDGED against an obstacle. Blocked detection is degraded and "
                "escape maneuvers will struggle; reposition the robot with clear space ahead "
                "before mapping."
            )
        elif zone_cfg.baseline_points > 0:
            print(
                "[wander] note: the lidar permanently sees part of the robot inside the stop box; "
                "consider recalibrating the stop box geometry"
            )

        # BOXED-IN ESCAPE (user directive 2026-07-18): if the founding scan is
        # surrounded by near returns, translate toward open space and keep
        # scanning until healthy — instead of failing. Rotating in place would
        # not help (a 360deg scan is heading-invariant).
        _probe_id, _probe_frame = feed.latest()
        if _probe_frame is not None:
            _probe_usable = _scan_to_local_points(
                points=_probe_frame.points,
                forward_angle_deg=float(args.forward_angle_deg),
                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                invert_lateral_axis=bool(args.invert_lateral_axis),
                max_distance_m=float(args.max_distance_m),
                min_confidence=int(args.min_confidence),
                min_range_m=float(args.min_range_m),
            )
            if len(_probe_usable) < 60:
                print(
                    f"[wander] founding scan is boxed in (usable={len(_probe_usable)} < 60); "
                    "translating toward open space before founding the map"
                )
                last_frame_id = _escape_boxed_in(
                    robot=robot,
                    feed=feed,
                    args=args,
                    after_frame_id=last_frame_id,
                    min_usable=60,
                )

        frame_id, captured_frame, _, initial_snapshot = _capture_snapshot(
            feed=feed,
            output_dir=snapshot_dir,
            request_index=1,
            after_frame_id=last_frame_id,
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            invert_lateral_axis=bool(args.invert_lateral_axis),
            max_distance_m=float(args.max_distance_m),
            min_range_m=float(args.min_range_m),
            min_confidence=int(args.min_confidence),
            fresh_frame_timeout_s=float(args.fresh_frame_timeout_s),
            fresh_frame_advances=int(args.fresh_frame_advances),
            capture_config_extra={"motion_hint": motion_hints[-1].kind},
        )
        last_frame_id = int(frame_id)
        last_captured_frame = captured_frame

        stitch_state = _append_stitch(
            stitch_dir=stitch_dir,
            snapshot_dir=snapshot_dir,
            stitch_state={"snapshots": [], "poses": [], "transformed_sets": [], "solve_log": []},
            motion_hints=motion_hints,
            new_snapshot=initial_snapshot,
            resolution_m=float(args.stitch_resolution_m),
        )
        _log_rerun_state(
            rr,
            capture_index=1,
            transformed_sets=stitch_state["transformed_sets"],
            poses=stitch_state["poses"],
            solve_log=stitch_state["solve_log"],
            cone_half_width_deg=float(args.valid_angle_half_width_deg),
            body_radius_m=float(args.robot_radius_m),
        )
        initial_pose = stitch_state["poses"][-1]
        initial_heading_bin = _heading_bin_index(
            float(initial_pose.theta_deg),
            bin_count=rotation_coverage_bin_count,
        )
        rotation_coverage_bins_seen.add(initial_heading_bin)
        print(
            "[wander] rotation coverage update "
            f"(capture=1, heading={float(initial_pose.theta_deg):.1f}deg, "
            f"bins={sorted(rotation_coverage_bins_seen)}/{rotation_coverage_bin_count})"
        )
        print(f"[wander] initial stitched map ready: {stitch_state['html_path']}")

        capture_index = 2
        last_camera_feedback_monotonic = 0.0
        # Panorama is a bounded phase, not a condition that can repeat after
        # discarded captures. Count commanded turns independently of appended
        # snapshots so a weak scan cannot trap the robot spinning in place.
        bootstrap_turn_commands = 0
        # Feed circuit breaker: if the lidar host stops delivering fresh
        # revolutions, ALL motion must stop. Field 2026-07-11: the host died
        # mid-run and the loop kept executing turns and reverses BLIND on a
        # frozen frame, appending the identical snapshot 13 times while the
        # robot physically moved with zero sensing.
        feed_watch_frame_id = -1
        feed_watch_advance_monotonic = time.monotonic()
        feed_stall_announced = False
        while True:
            if int(args.max_captures) > 0 and capture_index > int(args.max_captures):
                break
            live_frame_id, live_frame = feed.latest()
            if live_frame is None:
                raise RuntimeError("LiDAR feed disappeared during wander loop.")
            if int(live_frame_id) != feed_watch_frame_id:
                feed_watch_frame_id = int(live_frame_id)
                feed_watch_advance_monotonic = time.monotonic()
                if feed_stall_announced:
                    print("[wander] lidar feed recovered; resuming exploration")
                    feed_stall_announced = False
            elif time.monotonic() - feed_watch_advance_monotonic > 5.0:
                _send_stop(robot)
                if not feed_stall_announced:
                    print(
                        "[wander] WARNING: lidar feed STALLED (no new revolution for >5s; "
                        "host down or network lost) — ALL motion halted; waiting for the "
                        "feed to recover. Restart the lidar host on the Pi if this persists."
                    )
                    feed_stall_announced = True
                time.sleep(2.0)
                continue
            current_solved_pose = stitch_state["poses"][-1]
            # The stitched pose LAGS any motion whose capture was skipped as
            # redundant or discarded — the robot has physically moved but the
            # map anchor hasn't. Judge the live scan against the pending-motion-
            # advanced expectation, or a correctly-executed maneuver reads as
            # pose error (field 2026-07-18: a clean -36deg turn before a
            # skipped-as-redundant capture scored theta_error=36deg>24 and
            # latched pose LOST on a CORRECT 12.6-score match — one cycle after
            # the robot had finally driven out of the room).
            live_expected_pose = (
                _advance_pose(current_solved_pose, pending_motion_hint)
                if pending_motion_hint is not None
                else current_solved_pose
            )
            live_points_xy = _scan_to_local_points(
                points=live_frame.points,
                forward_angle_deg=float(args.forward_angle_deg),
                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                invert_lateral_axis=bool(args.invert_lateral_axis),
                max_distance_m=float(args.max_distance_m),
                min_confidence=int(args.min_confidence),
                min_range_m=float(args.min_range_m),
            )
            live_pose_result = _estimate_pose_against_stitched_map(
                points_xy=live_points_xy,
                transformed_sets=stitch_state["transformed_sets"],
                initial_pose=live_expected_pose,
                resolution_m=float(args.stitch_resolution_m),
                search_xy_m=max(float(args.drive_search_xy_m), 0.55),
                theta_window_deg=max(float(args.drive_theta_window_deg), 36.0),
                max_translation_from_initial_m=max(0.45, float(args.drive_search_xy_m) + 0.20),
                prior_translation_weight=0.20,
                prior_theta_weight=0.08,
            )
            live_pose_accepted = False
            if live_pose_result is not None:
                candidate_live_pose, live_pose_meta = live_pose_result
                if _accept_relocalized_pose(
                    label=f"live_frame_{live_frame_id}",
                    candidate_pose=candidate_live_pose,
                    score_meta=live_pose_meta,
                    expected_pose=live_expected_pose,
                    motion_hint=None,
                    max_translation_error_m=max(0.45, float(args.drive_search_xy_m) + 0.20),
                    max_theta_error_deg=max(24.0, float(args.drive_theta_window_deg) * 0.85),
                    min_score=7.0,
                ):
                    current_live_pose = candidate_live_pose
                    live_pose_accepted = True
                    dead_reck_theta_deg = float(candidate_live_pose.theta_deg)
                    dead_reck_slack_deg = 15.0
                    imu_dead_reck_anchor = _imu_anchor(imu_yaw, dead_reck_theta_deg)
                    if pose_lost and float(live_pose_meta.get("score") or 0.0) >= POSE_RECOVERY_MIN_SCORE:
                        pose_lost = False
                        post_recovery_strict_appends = 2
                        lost_recovery_failures = 0
                        print(
                            "[wander] pose integrity RESTORED "
                            f"(live score={float(live_pose_meta.get('score') or 0.0):.2f}); "
                            "map writes re-enabled (strict gates for the next 2 appends)"
                        )
                else:
                    current_live_pose = current_solved_pose
                    live_pose_meta = {
                        **live_pose_meta,
                        "source": "rejected_live_fallback",
                    }
                    if not pose_lost:
                        pose_lost = True
                        print(
                            "[wander] pose integrity LOST (live relocalization rejected); "
                            "map is READ-ONLY until a strong relocalization re-proves the pose"
                        )
                    print(
                        "[wander] live relocalization fallback "
                        f"(frame_id={live_frame_id}, using last stitched pose=({float(current_live_pose.x):.3f}, "
                        f"{float(current_live_pose.y):.3f}, {float(current_live_pose.theta_deg):.1f}deg))"
                    )
                    # Re-derive the dead-reckoned heading from the gyro BEFORE
                    # searching. The gyro integrated through every blind align/
                    # recovery turn since the anchor was set, so this stays
                    # accurate where the lidar-only anchor had gone stale.
                    imu_theta = _imu_resolved_theta(imu_yaw, imu_dead_reck_anchor)
                    if imu_theta is not None:
                        dead_reck_theta_deg = imu_theta
                        dead_reck_slack_deg = IMU_DEAD_RECK_SLACK_DEG
                    # Recovery search: when lost, the pose EXPECTATION is
                    # meaningless, so the normal accept gates (which compare
                    # against it — this failure rejected the correct
                    # whole-map candidate for a 58deg "theta error") cannot
                    # ever re-anchor a badly rotated robot. Search wide in
                    # theta and judge on ABSOLUTE match quality alone.
                    #
                    # LIDAR-ONLY mapping: an earlier "IMU-guided" variant searched
                    # centered on the gyro heading; it manufactured marginal
                    # just-past-the-gate candidates (12.2 from a wedged degenerate
                    # viewpoint), un-froze the map, and the next append ghosted
                    # the walls. The gyro's only role here is the VETO below —
                    # rejecting never writes, so it can't corrupt the map.
                    recovery_result = _estimate_pose_against_stitched_map(
                        points_xy=live_points_xy,
                        transformed_sets=stitch_state["transformed_sets"],
                        initial_pose=live_expected_pose,
                        resolution_m=float(args.stitch_resolution_m),
                        search_xy_m=1.10,
                        theta_window_deg=180.0,
                        max_translation_from_initial_m=1.10,
                        prior_translation_weight=0.03,
                        prior_theta_weight=0.0,
                    )
                    if recovery_result is not None:
                        recovery_pose, recovery_meta = recovery_result
                        recovery_score = float(recovery_meta.get("score") or -1e9)
                        dead_reck_err_deg: float | None = None
                        if dead_reck_theta_deg is not None:
                            dead_reck_err_deg = abs(
                                _normalize_angle_deg(
                                    float(recovery_pose.theta_deg) - float(dead_reck_theta_deg)
                                )
                            )
                        if recovery_score >= POSE_RECOVERY_MIN_SCORE and (
                            dead_reck_err_deg is None or dead_reck_err_deg <= dead_reck_slack_deg
                        ):
                            current_live_pose = recovery_pose
                            live_pose_meta = {**recovery_meta, "source": "lost_recovery"}
                            live_pose_accepted = True
                            pose_lost = False
                            post_recovery_strict_appends = 2
                            lost_recovery_failures = 0
                            dead_reck_theta_deg = float(recovery_pose.theta_deg)
                            dead_reck_slack_deg = 15.0
                            imu_dead_reck_anchor = _imu_anchor(imu_yaw, dead_reck_theta_deg)
                            print(
                                "[wander] pose integrity RESTORED via wide-theta recovery "
                                f"(pose=({recovery_pose.x:.3f}, {recovery_pose.y:.3f}, "
                                f"{recovery_pose.theta_deg:.1f}deg), score={recovery_score:.2f})"
                            )
                        elif recovery_score >= POSE_RECOVERY_MIN_SCORE:
                            lost_recovery_failures += 1
                            print(
                                "[wander] wide-theta recovery REJECTED by dead reckoning "
                                f"(candidate theta={recovery_pose.theta_deg:.1f}deg is "
                                f"{dead_reck_err_deg:.0f}deg from the tracked heading, bound=±"
                                f"{dead_reck_slack_deg:.0f}deg, score={recovery_score:.2f}); a "
                                "symmetric wrong mode scores this high too — staying lost"
                            )
                        else:
                            lost_recovery_failures += 1
                            print(
                                "[wander] wide-theta recovery inconclusive "
                                f"(best score={recovery_score:.2f} < {POSE_RECOVERY_MIN_SCORE:.1f}); staying lost"
                            )
                    if (
                        pose_lost
                        and len(stitch_state["snapshots"]) > 1
                        and lost_recovery_failures >= 2
                        and lost_recovery_failures % 2 == 0
                    ):
                        # LOST-RETREAT: the robot got lost by driving its
                        # view degenerate (field run 13: nose into a pocket,
                        # usable points 200->137, every solve <= 5, recovery
                        # capped at 7-8 forever — paralyzed until Ctrl+C).
                        # Rotating in place cannot fix a degenerate VIEW;
                        # backing out along the path it came in on can. The
                        # reverse helper is live-rear-checked
                        # (robot-relative, immune to the pose error we
                        # necessarily have while lost).
                        print(
                            "[wander] lost with recovery failing; backing out of the "
                            "degenerate viewpoint toward mapped territory"
                        )
                        retreat_hint = _attempt_reverse_escape_hint(
                            current_live_pose, lost_recovery_failures >= 6
                        )
                        if retreat_hint is not None:
                            pending_motion_hint = retreat_hint
                            continue
                        # Rear blocked (map+live agree): translate toward the
                        # deepest LIVE opening instead — robot-relative, so
                        # immune to the pose error we necessarily carry while
                        # lost, and TRANSLATION (not rotation) is what
                        # un-degenerates a view (same physics as the boxed-in
                        # escape). Field 2026-07-18 run 12: lost at the new
                        # room's sparse edge with 0.00m behind, the
                        # rotation-only fallback thrashed lock-lost turns
                        # until Ctrl+C. Only when the opening is roughly
                        # ahead — a big blind turn while lost IS the thrash
                        # this replaces.
                        retreat_choice = _select_frontier_choice(
                            live_frame,
                            forward_angle_deg=float(args.forward_angle_deg),
                            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                            max_distance_m=float(args.max_distance_m),
                            min_range_m=float(args.min_range_m),
                            min_confidence=int(args.min_confidence),
                            frontier_min_distance_m=1.0,
                            frontier_bin_deg=float(args.frontier_bin_deg),
                            min_gap_width_m=2.0 * (float(args.robot_radius_m) + 0.05),
                            corridor_veto=None,
                        )
                        if (
                            retreat_choice is not None
                            and abs(float(retreat_choice.delta_deg)) <= 50.0
                            and float(retreat_choice.mean_distance_m) >= 1.0
                        ):
                            print(
                                "[wander] rear blocked while lost — advancing toward the live "
                                f"opening (delta={retreat_choice.delta_deg:.1f}deg, "
                                f"range={retreat_choice.mean_distance_m:.2f}m) to change the view"
                            )
                            pending_motion_hint = _live_frontier_probe_hint(
                                current_live_pose, retreat_choice
                            )
                            continue
                    if (
                        pose_lost
                        and len(stitch_state["snapshots"]) <= 1
                        and lost_recovery_failures >= 2
                    ):
                        # RE-FOUND: the map is only its founding snapshot — it
                        # holds no investment worth a deadlock. Wipe it and
                        # start mapping fresh from wherever the robot stands.
                        print(
                            "[wander] map has only its founding snapshot and the pose cannot be "
                            "recovered — RE-FOUNDING the map from the current position"
                        )
                        _send_stop(robot)
                        motion_hints[:] = [
                            MotionHint(
                                kind="start",
                                expected_dx_local_m=0.0,
                                expected_dy_local_m=0.0,
                                expected_dtheta_deg=0.0,
                                search_xy_m=float(args.drive_search_xy_m),
                                search_theta_window_deg=float(args.turn_theta_window_deg),
                                label=f"refound_capture_{capture_index:02d}",
                            )
                        ]
                        refound_frame_id, refound_frame, _, refound_snapshot = _capture_snapshot(
                            feed=feed,
                            output_dir=snapshot_dir,
                            request_index=capture_index,
                            after_frame_id=int(live_frame_id),
                            forward_angle_deg=float(args.forward_angle_deg),
                            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                            invert_lateral_axis=bool(args.invert_lateral_axis),
                            max_distance_m=float(args.max_distance_m),
                            min_range_m=float(args.min_range_m),
                            min_confidence=int(args.min_confidence),
                            fresh_frame_timeout_s=float(args.fresh_frame_timeout_s),
                            fresh_frame_advances=int(args.fresh_frame_advances),
                            capture_config_extra={"motion_hint": "start"},
                        )
                        last_frame_id = int(refound_frame_id)
                        last_captured_frame = refound_frame
                        stitch_state = _append_stitch(
                            stitch_dir=stitch_dir,
                            snapshot_dir=snapshot_dir,
                            stitch_state={
                                "snapshots": [],
                                "poses": [],
                                "transformed_sets": [],
                                "solve_log": [],
                            },
                            motion_hints=motion_hints,
                            new_snapshot=refound_snapshot,
                            resolution_m=float(args.stitch_resolution_m),
                        )
                        capture_index += 1
                        # Everything expressed in the OLD world frame is now
                        # fiction — wipe it all.
                        elevated_occupied_world.clear()
                        elevated_pending_world.clear()
                        elevated_vantage_promotions.clear()
                        elevated_segments_world.clear()
                        frontier_strike_counts.clear()
                        frontier_blacklist.clear()
                        unreachable_targets_world.clear()
                        active_explore_target = None
                        active_survey_xy = None
                        pending_motion_hint = None
                        consecutive_append_discards = 0
                        pose_lost = False
                        post_recovery_strict_appends = 0
                        lost_recovery_failures = 0
                        dead_reck_theta_deg = float(stitch_state["poses"][-1].theta_deg)
                        dead_reck_slack_deg = 15.0
                        imu_dead_reck_anchor = _imu_anchor(imu_yaw, dead_reck_theta_deg)
                        rotation_coverage_bins_seen.clear()
                        rotation_coverage_bins_seen.add(
                            _heading_bin_index(
                                float(stitch_state["poses"][-1].theta_deg),
                                bin_count=rotation_coverage_bin_count,
                            )
                        )
                        rotation_coverage_complete = False
                        bootstrap_turn_commands = 0
                        consecutive_turn_captures = 0
                        print(
                            "[wander] map re-founded "
                            f"(new founding snapshot has {len(refound_snapshot.points_xy)} points); "
                            "bootstrap panorama restarts"
                        )
                        continue
            if live_pose_result is None:
                current_live_pose = current_solved_pose
                print(
                    "[wander] live relocalization unavailable; using last stitched pose "
                    f"(frame_id={live_frame_id}, pose=({float(current_live_pose.x):.3f}, "
                    f"{float(current_live_pose.y):.3f}, {float(current_live_pose.theta_deg):.1f}deg))"
                )
            _log_live_pose_state(
                rr,
                capture_index=max(1, capture_index - 1),
                pose=current_live_pose,
                points_xy=live_points_xy,
                cone_half_width_deg=float(args.valid_angle_half_width_deg),
                body_radius_m=float(args.robot_radius_m),
            )
            # BODY-OVERLAP EVICTION: a decayable eye cell strictly inside the
            # robot's own body radius at a TRUSTED pose is physically
            # disproven — the robot occupies that space and no gate reports
            # contact. Such cells sit below the decay pass's minimum
            # inspection range (0.35m) and would otherwise be unfalsifiable
            # at point-blank (field 2026-07-17 run 18).
            if live_pose_accepted and not pose_lost and elevated_cell_meta:
                overlap_r_sq = (float(args.robot_radius_m) - 0.03) ** 2
                overlap_cells = [
                    c
                    for c in elevated_cell_meta
                    if (c[0] * 0.04 - float(current_live_pose.x)) ** 2
                    + (c[1] * 0.04 - float(current_live_pose.y)) ** 2
                    < overlap_r_sq
                ]
                for cell in overlap_cells:
                    elevated_occupied_world.discard(cell)
                    elevated_pending_world.pop(cell, None)
                    elevated_points_world.pop(cell, None)
                    elevated_cell_meta.pop(cell, None)
                if overlap_cells:
                    print(
                        f"[edge-map] evicted {len(overlap_cells)} cells inside the robot's own "
                        "body footprint (physically disproven)"
                    )
            # PINNED-BY-GUARD ESCAPE: the side guard halting every forward
            # burst at the same pose is a livelock, not protection (run 18:
            # 130+ identical replan/halt cycles). Back away from the cell.
            if map_side_guard_state["streak"] >= 3:
                print(
                    "[wander] side-collision guard has pinned the robot "
                    f"({map_side_guard_state['streak']} zero-motion halts); backing away "
                    "from the mapped cell"
                )
                map_side_guard_state["streak"] = 0
                map_side_guard_state["pose"] = None
                pinned_retreat = _attempt_reverse_escape_hint(current_live_pose, True)
                if pinned_retreat is not None:
                    pending_motion_hint = pinned_retreat
                    continue
            # Keep the Rerun camera panels live between captures. These are
            # the exact annotated frames used by the safety monitor, not a
            # second video path; viewing them cannot affect control.
            feedback_hz = max(0.0, float(args.camera_feedback_hz))
            if (
                hazard_monitor is not None
                and feedback_hz > 0.0
                and time.monotonic() - last_camera_feedback_monotonic >= 1.0 / feedback_hz
            ):
                last_camera_feedback_monotonic = time.monotonic()
                live_state = hazard_monitor.state()
                for safety_cam in ("front_left", "front_right", "panorama", "bottom"):
                    safety_frame = hazard_monitor.annotated(safety_cam)
                    if safety_frame is not None:
                        rr.log(f"cameras/live_{safety_cam}", rr.Image(safety_frame[:, :, ::-1]))
                rr.log(
                    "safety/live_decision",
                    rr.TextLog(
                        f"{live_state.decision_label()} side={live_state.side} "
                        f"distance={live_state.est_distance_m} detail={live_state.detail}"
                    ),
                )

            blocked_points = _blocked_points_for_frame(live_frame, zone_cfg)
            blocked = blocked_points >= zone_cfg.blocked_trigger_count()
            drive_hint_m = 0.0
            should_turn = False
            turn_reason = "none"
            chosen_direction_sign = float(direction_sign)
            chosen_turn_deg = float(args.turn_deg)
            motion_hint: MotionHint | None = None
            # Set when a hazard stop happened after so little motion that a
            # checkpoint capture would re-anchor nothing: skip the capture
            # and carry the tracked motion into the next one instead.
            checkpoint_skippable = False
            settle_s = float(args.move_settle_s)
            turn_capture_theta_window_deg = 80.0
            bootstrap_scan_active = (
                wander_mode == "smart"
                and not rotation_coverage_complete
                and bootstrap_turn_commands < max(1, int(args.bootstrap_turn_captures))
            )
            # Legacy heuristic, superseded by the frontier planner in smart mode.
            forced_drive_due_to_turn_streak = (
                wander_mode == "smart-legacy"
                and not blocked
                and consecutive_turn_captures >= max(2, min(4, int(args.bootstrap_turn_captures)))
            )
            if forced_drive_due_to_turn_streak:
                force_drive_after_turn = True
                print(
                    "[wander] forcing forward exploration after repeated turn captures "
                    f"(turn_streak={consecutive_turn_captures}, coverage_complete={rotation_coverage_complete})"
                )

            if pending_turn_reason is not None:
                turn_reason = str(pending_turn_reason)
                chosen_direction_sign = float(pending_turn_direction_sign)
                chosen_turn_deg = float(pending_turn_deg)
                print(
                    "[wander] executing queued turn "
                    f"(reason={turn_reason}, target={chosen_turn_deg:.1f}deg "
                    f"{'ccw' if chosen_direction_sign >= 0.0 else 'cw'})"
                )
                should_turn = True
                pending_turn_reason = None
                pending_turn_direction_sign = float(direction_sign)
                pending_turn_deg = float(args.turn_deg)

            def _elevated_corridor_blocked(delta_deg: float, distance_m: float) -> bool:
                """True when the straight corridor toward a live-lidar opening
                crosses eye-mapped elevated cells. The scan sees under
                furniture; the eyes mapped that furniture — believe the eyes
                when choosing where to drive."""
                if not elevated_occupied_world:
                    return False
                bearing = math.radians(
                    float(current_live_pose.theta_deg) + float(delta_deg)
                )
                cos_b, sin_b = math.cos(bearing), math.sin(bearing)
                corridor_len = min(float(distance_m), 1.5)
                hits = 0
                for cx, cy in elevated_occupied_world:
                    dxc = cx * 0.04 - float(current_live_pose.x)
                    dyc = cy * 0.04 - float(current_live_pose.y)
                    along = dxc * cos_b + dyc * sin_b
                    if not (0.10 <= along <= corridor_len):
                        continue
                    if abs(-dxc * sin_b + dyc * cos_b) <= 0.33:
                        hits += 1
                        if hits >= 2:
                            return True
                return False

            live_frontier_choice = _select_frontier_choice(
                live_frame,
                forward_angle_deg=float(args.forward_angle_deg),
                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                max_distance_m=float(args.max_distance_m),
                min_range_m=float(args.min_range_m),
                min_confidence=int(args.min_confidence),
                frontier_min_distance_m=float(args.frontier_min_distance_m),
                frontier_bin_deg=float(args.frontier_bin_deg),
                min_gap_width_m=2.0 * (float(args.robot_radius_m) + 0.05),
                corridor_veto=_elevated_corridor_blocked,
            )
            # EXIT-RUN HARD VETO: an already-mapped open room is ALSO a "deep
            # opening" — from the doorway, looking back across the room clears
            # every depth/width bar, so the per-cycle deepest-opening pick
            # ping-ponged between the exit and the room behind it and the robot
            # kept driving AWAY from the door it had just reached (field
            # 2026-07-18 run 15; user: "HARDSTOP IT FROM EVER GOING BACK").
            # While exit_mode is latched, an opening is acceptable ONLY if its
            # far end lands OUTSIDE the pre-exit coverage — the same 0.80m
            # test as exit completion. Openings leading back into coverage are
            # dead, unconditionally.
            if exit_mode and exit_latch_poses and live_frontier_choice is not None:
                _vetoed_deltas: list[float] = []
                for _exit_veto_round in range(24):
                    _end_rad = math.radians(
                        float(current_live_pose.theta_deg)
                        + float(live_frontier_choice.delta_deg)
                    )
                    _end_reach = 0.85 * float(live_frontier_choice.mean_distance_m)
                    _end_x = float(current_live_pose.x) + _end_reach * math.cos(_end_rad)
                    _end_y = float(current_live_pose.y) + _end_reach * math.sin(_end_rad)
                    _end_clear = min(
                        math.hypot(_end_x - lx, _end_y - ly) for lx, ly in exit_latch_poses
                    )
                    if _end_clear >= 0.80:
                        break
                    print(
                        "[wander] EXIT RUN: opening at delta="
                        f"{live_frontier_choice.delta_deg:.1f}deg leads BACK into mapped "
                        f"coverage (far end {_end_clear:.2f}m from a pre-exit vantage) — "
                        "vetoed; going back is forbidden"
                    )
                    _vetoed_deltas.append(float(live_frontier_choice.delta_deg))
                    live_frontier_choice = _select_frontier_choice(
                        live_frame,
                        forward_angle_deg=float(args.forward_angle_deg),
                        valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                        max_distance_m=float(args.max_distance_m),
                        min_range_m=float(args.min_range_m),
                        min_confidence=int(args.min_confidence),
                        frontier_min_distance_m=float(args.frontier_min_distance_m),
                        frontier_bin_deg=float(args.frontier_bin_deg),
                        min_gap_width_m=2.0 * (float(args.robot_radius_m) + 0.05),
                        corridor_veto=(
                            lambda seg_delta, seg_mean, _blocked=_elevated_corridor_blocked, _bad=tuple(_vetoed_deltas): (
                                _blocked(seg_delta, seg_mean)
                                or any(
                                    abs(_normalize_angle_deg(seg_delta - b)) < 10.0
                                    for b in _bad
                                )
                            )
                        ),
                    )
                    if live_frontier_choice is None:
                        break
            if live_frontier_choice is not None:
                print(
                    "[wander] live frontier candidate "
                    f"(frame_id={live_frame_id}, delta={live_frontier_choice.delta_deg:.1f}deg, "
                    f"distance={live_frontier_choice.mean_distance_m:.2f}m, "
                    f"width={live_frontier_choice.width_deg:.1f}deg, score={live_frontier_choice.score:.2f})"
                )
            else:
                print(f"[wander] no live frontier candidate found (frame_id={live_frame_id})")

            map_frontier_choice: FrontierChoice | None = None
            # Legacy ray-based map frontier, superseded by the exploration-grid
            # planner in smart mode.
            if wander_mode != "smart" and len(stitch_state["snapshots"]) >= 2:
                map_frontier_choice = _select_map_frontier_choice(
                    transformed_sets=stitch_state["transformed_sets"],
                    current_pose=current_live_pose,
                    max_distance_m=float(args.max_distance_m),
                    frontier_min_distance_m=float(args.frontier_min_distance_m),
                    frontier_bin_deg=float(args.frontier_bin_deg),
                    resolution_m=float(args.stitch_resolution_m),
                    robot_radius_m=float(args.robot_radius_m),
                )
            if map_frontier_choice is not None:
                print(
                    "[wander] map frontier candidate "
                    f"(delta={map_frontier_choice.delta_deg:.1f}deg, "
                    f"distance={map_frontier_choice.mean_distance_m:.2f}m, "
                    f"width={map_frontier_choice.width_deg:.1f}deg, score={map_frontier_choice.score:.2f}, "
                    f"pose=({float(current_live_pose.x):.3f}, {float(current_live_pose.y):.3f}, {float(current_live_pose.theta_deg):.1f}deg))"
                )
            elif wander_mode != "smart":
                print(
                    "[wander] no stitched-map frontier candidate "
                    f"(captures={len(stitch_state['snapshots'])}, pose=({float(current_live_pose.x):.3f}, "
                    f"{float(current_live_pose.y):.3f}, {float(current_live_pose.theta_deg):.1f}deg))"
                )
            if map_frontier_choice is not None and float(map_frontier_choice.score) < 2.0:
                # Real frontiers score well above this (clearance + exit
                # bonuses). A weak/negative score is just the least-bad ray —
                # chasing one is how the robot walked into a dead-end pocket.
                print(
                    "[wander] ignoring weak map frontier "
                    f"(score={float(map_frontier_choice.score):.2f} < 2.0)"
                )
                map_frontier_choice = None

            # Nothing worth mapping from HERE: no reachable map frontier and no
            # active target. Before concluding the whole job is done, deliberately
            # relocate to a vantage the robot has not captured from yet (e.g. the
            # opposite corner) — milling around a finished corner wastes captures.
            # Legacy idle/survey/completion, superseded by the frontier
            # planner's own completion check in smart mode.
            idle_here = (
                wander_mode != "smart"
                and rotation_coverage_complete
                and len(stitch_state["snapshots"]) >= 8
                and map_frontier_choice is None
                and active_explore_target is None
            )
            if idle_here and survey_targets_used < 3:
                survey_target_xy = _select_survey_target(
                    transformed_sets=stitch_state["transformed_sets"],
                    current_pose=current_live_pose,
                    capture_poses=list(stitch_state["poses"]),
                    resolution_m=float(args.stitch_resolution_m),
                    robot_radius_m=float(args.robot_radius_m),
                    max_distance_m=float(args.max_distance_m),
                )
                if survey_target_xy is not None and not _near_unreachable_target(*survey_target_xy):
                    survey_targets_used += 1
                    active_explore_target = ExploreTarget(
                        world_x_m=float(survey_target_xy[0]),
                        world_y_m=float(survey_target_xy[1]),
                        source="survey",
                        seeded_capture_index=int(capture_index),
                    )
                    active_explore_target_stall_count = 0
                    active_explore_target_last_distance_m = None
                    active_explore_target_blocked_count = 0
                    force_live_frontier_cycles = 0
                    idle_here = False
                    print(
                        "[wander] nothing left to map from here; relocating to survey vantage "
                        f"({survey_target_xy[0]:.3f}, {survey_target_xy[1]:.3f}) "
                        f"(survey {survey_targets_used}/3)"
                    )
            # Mission completion (legacy modes only): in smart mode the frontier
            # planner owns no_frontier_cycles — resetting it here every cycle
            # made the smart completion threshold unreachable (infinite
            # verification scans).
            if wander_mode != "smart":
                if idle_here:
                    no_frontier_cycles += 1
                else:
                    no_frontier_cycles = 0
            if wander_mode != "smart" and no_frontier_cycles >= 6:
                print(
                    "[wander] mapping complete: rotation coverage done and no reachable "
                    f"unmapped frontiers remain after {no_frontier_cycles} consecutive checks; stopping"
                )
                break

            target_frontier_choice: FrontierChoice | None = None
            if active_explore_target is not None:
                current_target_distance_m = _target_distance_m(active_explore_target, current_live_pose)
                if current_target_distance_m <= float(args.frontier_goal_reached_m):
                    print(
                        "[wander] stitched-map exploration target reached "
                        f"(distance={current_target_distance_m:.2f}m, source={active_explore_target.source}, "
                        f"seed_capture={active_explore_target.seeded_capture_index})"
                    )
                    active_explore_target = None
                    active_explore_target_stall_count = 0
                    active_explore_target_last_distance_m = None
                    active_explore_target_blocked_count = 0
                    force_live_frontier_cycles = 0
                else:
                    target_frontier_choice = _target_choice_from_world_point(
                        target_x_m=float(active_explore_target.world_x_m),
                        target_y_m=float(active_explore_target.world_y_m),
                        current_pose=current_live_pose,
                        source="target",
                    )
                    if target_frontier_choice is not None:
                        print(
                            "[wander] active stitched-map target "
                            f"(delta={target_frontier_choice.delta_deg:.1f}deg, "
                            f"distance={target_frontier_choice.mean_distance_m:.2f}m, "
                            f"world=({float(active_explore_target.world_x_m):.3f}, "
                            f"{float(active_explore_target.world_y_m):.3f}))"
                        )

            if (
                active_explore_target is None
                and wander_mode == "smart"
                and rotation_coverage_complete
                and map_frontier_choice is not None
            ):
                candidate_explore_target = _seed_explore_target_from_frontier(
                    frontier_choice=map_frontier_choice,
                    current_pose=current_live_pose,
                    capture_index=capture_index,
                    step_distance_m=float(args.frontier_goal_step_m),
                )
                if _near_unreachable_target(
                    candidate_explore_target.world_x_m, candidate_explore_target.world_y_m
                ):
                    print(
                        "[wander] skipping frontier near a known-unreachable target "
                        f"(world=({candidate_explore_target.world_x_m:.3f}, "
                        f"{candidate_explore_target.world_y_m:.3f}))"
                    )
                    candidate_explore_target = None
                active_explore_target = candidate_explore_target
                explore_target_just_seeded = candidate_explore_target is not None
            else:
                explore_target_just_seeded = False
            if explore_target_just_seeded:
                active_explore_target_stall_count = 0
                active_explore_target_last_distance_m = None
                active_explore_target_blocked_count = 0
                force_live_frontier_cycles = 0
                target_frontier_choice = _target_choice_from_world_point(
                    target_x_m=float(active_explore_target.world_x_m),
                    target_y_m=float(active_explore_target.world_y_m),
                    current_pose=current_live_pose,
                    source="target",
                )
                print(
                    "[wander] seeded stitched-map exploration target "
                    f"(source={map_frontier_choice.source}, delta={map_frontier_choice.delta_deg:.1f}deg, "
                    f"distance={map_frontier_choice.mean_distance_m:.2f}m, target_world=("
                    f"{float(active_explore_target.world_x_m):.3f}, {float(active_explore_target.world_y_m):.3f}))"
                )

            escape_frontier_active = (
                wander_mode == "smart"
                and force_live_frontier_cycles > 0
                and live_frontier_choice is not None
            )
            if escape_frontier_active:
                planning_frontier_choice = live_frontier_choice or map_frontier_choice or target_frontier_choice
                planning_frontier_source = "live_escape"
                print(
                    "[wander] escape frontier override active "
                    f"(cycles_left={force_live_frontier_cycles}, "
                    f"live_delta={live_frontier_choice.delta_deg:.1f}deg, "
                    f"live_distance={live_frontier_choice.mean_distance_m:.2f}m)"
                )
            else:
                planning_frontier_choice = target_frontier_choice or map_frontier_choice or live_frontier_choice
                planning_frontier_source = (
                    planning_frontier_choice.source if planning_frontier_choice is not None else "none"
                )
            if planning_frontier_choice is not None:
                print(
                    "[wander] planning frontier "
                    f"(source={planning_frontier_source}, delta={planning_frontier_choice.delta_deg:.1f}deg, "
                    f"distance={planning_frontier_choice.mean_distance_m:.2f}m, score={planning_frontier_choice.score:.2f})"
                )

            if hazard_monitor is not None:
                _stamp_confirmed_edges(current_live_pose)

            if (
                wander_mode == "smart"
                and pending_motion_hint is not None
                and not live_pose_accepted
                and abs(float(pending_motion_hint.expected_dtheta_deg)) > 20.0
            ):
                # The last capture was DISCARDED after a turn: the robot has
                # physically rotated away from the map, but the map never
                # absorbed that rotation. Planning a fresh frontier turn from
                # the stale pose rotates even FURTHER from the only overlap we
                # have, scores collapse toward zero, and a garbage pose ends up
                # force-accepted (observed: score 0.053 poisoned a whole run).
                # Recover by turning BACK toward the last stitched heading,
                # where overlap — and therefore a confident solve — is
                # guaranteed, then capture to re-anchor before exploring on.
                if orbit_recovery_turns_remaining <= 0:
                    if orbit_recovery_turn_attempts >= 1:
                        # The prior recovery turn also produced no trusted
                        # pose. Do not issue another blind 85deg command: it
                        # is the source of the endless cw/ccw-looking loop in
                        # the field run. Leave the base stopped, discard the
                        # untrusted turn hint, and let the next cycle capture
                        # a fresh view before the normal safety-gated drive.
                        print(
                            "[wander] unlocalized-turn recovery budget exhausted; "
                            "stopping rotation and forcing a new-view drive cycle"
                        )
                        pending_motion_hint = None
                        orbit_recovery_direction_sign = None
                        force_drive_after_turn = True
                        continue
                    orbit_recovery_direction_sign = (
                        1.0 if float(pending_motion_hint.expected_dtheta_deg) >= 0.0 else -1.0
                    )
                    orbit_recovery_turns_remaining = 1
                    orbit_recovery_turn_attempts += 1
                orbit_recovery_turns_remaining -= 1
                should_turn = True
                pending_turn_reason = None
                bootstrap_turn_commands = max(
                    bootstrap_turn_commands, max(1, int(args.bootstrap_turn_captures))
                )
                rotation_coverage_complete = True
                bootstrap_scan_active = False
                chosen_direction_sign = float(orbit_recovery_direction_sign)
                chosen_turn_deg = 85.0
                turn_reason = "orbit_recovery"
                print(
                    "[wander] turn capture could not be localized; continuing around the room "
                    f"({orbit_recovery_turns_remaining + 1}/1, 85deg "
                    f"{'ccw' if chosen_direction_sign >= 0.0 else 'cw'}) before retrying that frontier"
                )
                if orbit_recovery_turns_remaining == 0:
                    force_drive_after_turn = True
                if False and committed_frontier_face is not None:
                    # A face-turn toward this frontier just lost tracking and
                    # got discarded: facing it from here PROVABLY fails. Never
                    # retry the identical turn — blacklist the frontier
                    # immediately (this also breaks the goal commitment that
                    # would otherwise re-pick it) and explore elsewhere.
                    _strike_frontier(committed_frontier_face, "turn_lock_lost", force=True)
            elif wander_mode == "smart":
                # ================= frontier exploration =================
                # One rule, no special cases: model what has been observed
                # (free / occupied / unknown), find the nearest reachable
                # boundary to unknown space, travel the traversable path to
                # it, face it, scan. Bootstrap spin, corner-leaving, and
                # completion all emerge from this same computation.
                should_turn = False
                pending_turn_reason = None
                turn_reason = "none"
                elevated_extra_xy = (
                    np.asarray(
                        [[c[0] * 0.04, c[1] * 0.04] for c in elevated_occupied_world],
                        dtype=np.float32,
                    )
                    if elevated_occupied_world
                    else None
                )
                exploration_grid = _build_exploration_grid(
                    poses=list(stitch_state["poses"]),
                    transformed_sets=list(stitch_state["transformed_sets"]),
                    resolution_m=max(0.05, float(args.stitch_resolution_m)),
                    robot_clear_radius_m=float(args.robot_radius_m) + 0.06,
                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                    extra_occupied_xy=elevated_extra_xy,
                )
                frontier_plan: dict[str, object] = {"status": "no_frontier"}
                if exploration_grid is not None:
                    frontier_plan = _plan_frontier_path(
                        grid=exploration_grid,
                        robot_xy=(float(current_live_pose.x), float(current_live_pose.y)),
                        robot_radius_m=float(args.robot_radius_m),
                        target_xy=active_survey_xy,
                        observed_from_xy=[
                            (float(p.x), float(p.y)) for p in stitch_state["poses"]
                        ],
                        robot_theta_deg=float(current_live_pose.theta_deg),
                        avoid_face_xy=_active_frontier_blacklist(),
                        prefer_face_xy=committed_frontier_face,
                    )
                if (
                    str(frontier_plan.get("status")) == "no_frontier"
                    and active_survey_xy is None
                    and not pose_lost
                ):
                    # EDGE-SURVEY phase (user strategy 2026-07-18): the
                    # low-hanging exploration is done — before squeezing or
                    # verification-driving through YELLOW (single-view)
                    # cells, deliberately arc to a second angle on each
                    # yellow cluster. Real edges re-detect at the same
                    # world position and turn red with solid dimensions;
                    # bleed fails to reappear and decays. Only after the
                    # yellows are resolved does the ladder push into new
                    # areas through what remains.
                    edge_survey_plan = _plan_edge_survey(exploration_grid, current_live_pose)
                    if edge_survey_plan is not None:
                        frontier_plan = edge_survey_plan
                if (
                    str(frontier_plan.get("status")) == "observe"
                    and int(frontier_plan.get("frontier_cells", 0) or 0) >= 12
                    and active_survey_xy is None
                    and not pose_lost
                    and exploration_grid is not None
                ):
                    # OBSERVE -> SQUEEZE-THROUGH: the best plan is to peer at a
                    # SUBSTANTIAL unreachable region from afar (the classic
                    # doorway pinched shut at comfortable margins). The user's
                    # read is correct — the robot "thinks it's too fat" at the
                    # 0.29m planning radius (0.229m body + 0.06m margin) and
                    # sits re-observing the exit instead of going through it.
                    # Before settling for a look-from-afar, try to actually
                    # REACH the region at squeeze margins (radius - 5cm ~=
                    # 0.24m, ~1cm/side clearance over the true body). Only
                    # drive through if it becomes REACHABLE (status ok); if it
                    # is still merely observable, it is a real wall — keep the
                    # observe. The live measured-gap gate owns the traversal.
                    squeeze_grid = _build_exploration_grid(
                        poses=list(stitch_state["poses"]),
                        transformed_sets=list(stitch_state["transformed_sets"]),
                        resolution_m=max(0.05, float(args.stitch_resolution_m)),
                        robot_clear_radius_m=float(args.robot_radius_m),
                        lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                        extra_occupied_xy=elevated_extra_xy,
                    )
                    if squeeze_grid is not None:
                        squeeze_plan = _plan_frontier_path(
                            grid=squeeze_grid,
                            robot_xy=(float(current_live_pose.x), float(current_live_pose.y)),
                            robot_radius_m=max(0.22, float(args.robot_radius_m) - 0.05),
                            observed_from_xy=[
                                (float(p.x), float(p.y)) for p in stitch_state["poses"]
                            ],
                            robot_theta_deg=float(current_live_pose.theta_deg),
                            avoid_face_xy=_active_frontier_blacklist(),
                            prefer_face_xy=committed_frontier_face,
                        )
                        if str(squeeze_plan.get("status")) == "ok":
                            print(
                                "[wander] OBSERVE->SQUEEZE: the unmapped region is only observable "
                                "at comfortable margins but REACHABLE at squeeze width — driving "
                                "through the gap instead of staring "
                                f"(path={float(squeeze_plan.get('path_length_m', 0.0) or 0.0):.2f}m); "
                                "the measured-gap gate owns the traversal"
                            )
                            squeeze_plan["squeeze"] = True
                            frontier_plan = squeeze_plan
                if (
                    str(frontier_plan.get("status")) == "no_frontier"
                    and int(frontier_plan.get("unreachable_frontier_cells", 0) or 0) >= 24
                    and active_survey_xy is None
                ):
                    # SQUEEZE REPLAN: unknown space exists but every path to
                    # it is pinched shut at COMFORTABLE margins (radius +
                    # 6cm grid carve + radius inflation). The live safety
                    # stack is better informed than this grid — the
                    # measured-gap squeeze creeps through anything the body
                    # actually fits (field run 15: a physically passable
                    # doorway corridor, robot photographed fitting with
                    # margin, was sealed by a table edge-bleed blob +
                    # inflation and the mission stalled at "no_frontier").
                    # Retry the plan at squeeze margins; the gates own the
                    # actual traversal.
                    tight_grid = _build_exploration_grid(
                        poses=list(stitch_state["poses"]),
                        transformed_sets=list(stitch_state["transformed_sets"]),
                        resolution_m=max(0.05, float(args.stitch_resolution_m)),
                        robot_clear_radius_m=float(args.robot_radius_m),
                        lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                        extra_occupied_xy=elevated_extra_xy,
                    )
                    if tight_grid is not None:
                        tight_plan = _plan_frontier_path(
                            grid=tight_grid,
                            robot_xy=(float(current_live_pose.x), float(current_live_pose.y)),
                            robot_radius_m=max(0.22, float(args.robot_radius_m) - 0.05),
                            observed_from_xy=[
                                (float(p.x), float(p.y)) for p in stitch_state["poses"]
                            ],
                            robot_theta_deg=float(current_live_pose.theta_deg),
                            avoid_face_xy=_active_frontier_blacklist(),
                            prefer_face_xy=committed_frontier_face,
                        )
                        if str(tight_plan.get("status")) in ("ok", "observe"):
                            print(
                                "[wander] SQUEEZE replan: frontier unreachable at comfortable "
                                "margins but passable at squeeze width "
                                f"(status={tight_plan.get('status')}, "
                                f"path={float(tight_plan.get('path_length_m', 0.0) or 0.0):.2f}m); "
                                "proceeding — the measured-gap gate owns the traversal"
                            )
                            tight_plan["squeeze"] = True
                            frontier_plan = tight_plan
                        elif elevated_occupied_world:
                            # VERIFICATION replan: even squeeze margins can't
                            # thread the pinch — eye cells sit IN the
                            # corridor. Eye cells are claims, not measured
                            # lidar geometry; plan WITHOUT them (real walls
                            # still enforced) and approach under full gate
                            # protection. Real furniture stops the robot and
                            # re-confirms the cells; phantom cells get
                            # re-inspected, miss, and decay away. Either way
                            # the ambiguity resolves instead of declaring
                            # completion behind a possibly-false wall.
                            verify_grid = _build_exploration_grid(
                                poses=list(stitch_state["poses"]),
                                transformed_sets=list(stitch_state["transformed_sets"]),
                                resolution_m=max(0.05, float(args.stitch_resolution_m)),
                                robot_clear_radius_m=float(args.robot_radius_m) + 0.06,
                                lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                                extra_occupied_xy=None,
                            )
                            if verify_grid is not None:
                                verify_plan = _plan_frontier_path(
                                    grid=verify_grid,
                                    robot_xy=(
                                        float(current_live_pose.x),
                                        float(current_live_pose.y),
                                    ),
                                    robot_radius_m=float(args.robot_radius_m),
                                    observed_from_xy=[
                                        (float(p.x), float(p.y)) for p in stitch_state["poses"]
                                    ],
                                    robot_theta_deg=float(current_live_pose.theta_deg),
                                    avoid_face_xy=_active_frontier_blacklist(),
                                    prefer_face_xy=committed_frontier_face,
                                )
                                if str(verify_plan.get("status")) in ("ok", "observe"):
                                    print(
                                        "[wander] VERIFICATION replan: the only route to unmapped "
                                        "space crosses EYE-mapped cells (lidar geometry allows it); "
                                        "approaching under full gates to confirm or decay them "
                                        f"(status={verify_plan.get('status')}, "
                                        f"path={float(verify_plan.get('path_length_m', 0.0) or 0.0):.2f}m)"
                                    )
                                    frontier_plan = verify_plan
                if (
                    active_survey_xy is None
                    and str(frontier_plan.get("status")) == "no_frontier"
                    and survey_targets_used < 2
                    and exploration_grid is not None
                ):
                    # No information left to gain from HERE. Before declaring
                    # the map finished, relocate to an unvisited vantage and
                    # re-check — captures from across the room fill occlusion
                    # shadows that are unreachable-unknown from this spot. The
                    # vantage comes from the SAME BFS grid used for navigation,
                    # so it is pathable by construction.
                    survey_probe = _plan_frontier_path(
                        grid=exploration_grid,
                        robot_xy=(float(current_live_pose.x), float(current_live_pose.y)),
                        robot_radius_m=float(args.robot_radius_m),
                        survey_from_xy=[
                            (float(p.x), float(p.y)) for p in stitch_state["poses"]
                        ],
                        robot_theta_deg=float(current_live_pose.theta_deg),
                    )
                    if str(survey_probe.get("status")) == "survey":
                        active_survey_xy = (
                            float(survey_probe["goal_xy"][0]),
                            float(survey_probe["goal_xy"][1]),
                        )
                        survey_targets_used += 1
                        print(
                            "[wander] no frontier from this vantage; relocating to survey vantage "
                            f"({active_survey_xy[0]:.2f}, {active_survey_xy[1]:.2f}) "
                            f"(spacing={float(survey_probe.get('survey_spacing_m', 0.0)):.2f}m from "
                            f"previous captures, survey {survey_targets_used}/2)"
                        )
                        frontier_plan = survey_probe
                    else:
                        print(
                            "[wander] no unvisited survey vantage either "
                            f"(best spacing {float(survey_probe.get('best_survey_spacing_m', 0.0)):.2f}m "
                            "< 0.80m from previous captures)"
                        )
                if str(frontier_plan.get("status")) == "target_unreachable":
                    print("[wander] survey vantage unreachable; abandoning it")
                    active_survey_xy = None
                    frontier_plan = {"status": "no_frontier"}
                plan_status = str(frontier_plan.get("status"))
                squeeze_traversal_active = bool(frontier_plan.get("squeeze", False))
                # ---- EXIT RUN ---------------------------------------------
                # Sliver fatigue: count consecutive plans that offer no REAL new
                # area (tiny cell counts are grid noise/phantoms, not rooms).
                plan_cells = int(frontier_plan.get("frontier_cells", 0) or 0)
                if not bootstrap_scan_active and len(stitch_state["poses"]) >= 5:
                    if plan_status == "no_frontier" or (
                        plan_status in ("ok", "observe", "survey") and plan_cells <= 10
                    ):
                        sliver_frontier_cycles += 1
                    else:
                        sliver_frontier_cycles = 0
                        if exit_mode and plan_cells >= 30:
                            # Completion is POSITIONAL, not visual. New frontier
                            # cells appear the moment the robot merely LOOKS
                            # through the doorway (field 2026-07-18 run 13: 33
                            # cells opened, "complete" declared, robot still
                            # standing INSIDE the room — the exit was never
                            # traversed and the run failed its mission). The
                            # robot is through only when its own position has
                            # left the coverage it had when the exit run
                            # latched.
                            exit_clearance_m = (
                                min(
                                    math.hypot(
                                        float(current_live_pose.x) - lx,
                                        float(current_live_pose.y) - ly,
                                    )
                                    for lx, ly in exit_latch_poses
                                )
                                if exit_latch_poses
                                else float("inf")
                            )
                            if exit_clearance_m >= 0.80:
                                exit_mode = False
                                print(
                                    "[wander] EXIT RUN complete: robot is "
                                    f"{exit_clearance_m:.2f}m beyond its pre-exit coverage "
                                    f"with {plan_cells} new frontier cells — THROUGH the "
                                    "exit; resuming normal mapping in the new space"
                                )
                            else:
                                print(
                                    "[wander] EXIT RUN: new area visible through the opening "
                                    f"({plan_cells} cells) but the robot is still INSIDE its "
                                    f"mapped coverage ({exit_clearance_m:.2f}m from a pre-exit "
                                    "vantage < 0.80m) — NOT through yet; keep driving"
                                )
                if not exit_mode and sliver_frontier_cycles >= 4:
                    exit_mode = True
                    exit_latch_poses = [
                        (float(p.x), float(p.y)) for p in stitch_state["poses"]
                    ]
                    print(
                        "[wander] ROOM MAPPED (4 straight cycles with only sliver/no "
                        "frontiers) — EXIT RUN: interior goals are DONE; committing to the "
                        "deepest opening the live lidar sees and driving THROUGH it "
                        f"(completion requires leaving all {len(exit_latch_poses)} current "
                        "vantages by 0.80m)"
                    )
                if exit_mode and live_frontier_choice is not None:
                    # An opening is in view again: the search-rotation budget refills.
                    exit_scan_turns = 0
                if (
                    exit_mode
                    and live_frontier_choice is not None
                    and plan_status in ("ok", "observe", "survey")
                    and plan_cells <= 10
                ):
                    # Interior sliver goals are dead: force the live-opening
                    # probe path (commit to one world bearing, align at most
                    # twice, then guarded forward probes). The stop box still
                    # owns safety; crossing into the next room surfaces real
                    # frontiers, which clears the mode above.
                    print(
                        f"[wander] EXIT RUN: ignoring the {plan_cells}-cell interior "
                        "frontier; heading for the live exit opening "
                        f"(delta={float(live_frontier_choice.delta_deg):.1f}deg, "
                        f"distance={float(live_frontier_choice.mean_distance_m):.2f}m, "
                        f"width={float(live_frontier_choice.width_deg):.1f}deg)"
                    )
                    committed_frontier_face = None
                    frontier_plan = {"status": "no_frontier"}
                    plan_status = "no_frontier"
                # -----------------------------------------------------------
                # Squeeze/exit traversals get the NARROW stop-box band for the
                # drives executed this cycle (doorway frame posts the body
                # clears must not veto the pass); everything else runs the full
                # box. The value persists on zone_cfg into the next cycle's
                # early blocked check, which is correct for multi-cycle passes.
                zone_cfg.squeeze_active = bool(exit_mode or squeeze_traversal_active)
                if plan_status != "no_frontier":
                    # A real plan supersedes any half-finished probe episode.
                    probe_commit_world_deg = None
                    probe_commit_align_turns = 0
                if plan_status in ("ok", "survey", "observe"):
                    committed_frontier_face = (
                        float(frontier_plan["face_xy"][0]),
                        float(frontier_plan["face_xy"][1]),
                    )
                    # Anti-fixation: count consecutive pose-lost cycles that keep
                    # re-targeting this same face. A healthy cycle (localized)
                    # resets it; a run of lost cycles on one phantom frontier
                    # blacklists it so the planner is forced to the rest of the room.
                    if pose_lost:
                        if stuck_frontier_face is not None and math.hypot(
                            committed_frontier_face[0] - stuck_frontier_face[0],
                            committed_frontier_face[1] - stuck_frontier_face[1],
                        ) <= 0.4:
                            stuck_frontier_lost_cycles += 1
                        else:
                            stuck_frontier_face = committed_frontier_face
                            stuck_frontier_lost_cycles = 1
                    else:
                        stuck_frontier_face = None
                        stuck_frontier_lost_cycles = 0
                    print(
                        "[wander] frontier plan "
                        f"(status={plan_status}, "
                        f"goal=({frontier_plan['goal_xy'][0]:.2f}, {frontier_plan['goal_xy'][1]:.2f}), "
                        f"waypoint=({frontier_plan['waypoint_xy'][0]:.2f}, {frontier_plan['waypoint_xy'][1]:.2f}), "
                        f"face=({frontier_plan['face_xy'][0]:.2f}, {frontier_plan['face_xy'][1]:.2f}), "
                        f"path={float(frontier_plan['path_length_m']):.2f}m, "
                        f"frontier_cells={int(frontier_plan['frontier_cells'])})"
                    )
                    if stuck_frontier_lost_cycles >= 3:
                        print(
                            "[wander] ABANDONING frontier "
                            f"({committed_frontier_face[0]:.2f}, {committed_frontier_face[1]:.2f}): "
                            f"targeted it for {stuck_frontier_lost_cycles} straight cycles while "
                            "pose-lost without ever localizing there — blacklisting it and "
                            "exploring the rest of the room instead"
                        )
                        _strike_frontier(
                            committed_frontier_face,
                            "repeated pose-loss targeting this frontier",
                            force=True,
                        )
                        stuck_frontier_face = None
                        stuck_frontier_lost_cycles = 0
                else:
                    committed_frontier_face = None
                    if plan_status == "no_frontier" and "unknown_cells" in frontier_plan:
                        unreachable_centroid = frontier_plan.get("unreachable_centroid_xy")
                        centroid_txt = (
                            f", unreachable_centroid=({unreachable_centroid[0]:.2f}, "
                            f"{unreachable_centroid[1]:.2f})"
                            if unreachable_centroid
                            else ""
                        )
                        print(
                            "[wander] frontier plan (status=no_frontier, "
                            f"unknown={int(frontier_plan['unknown_cells'])}, "
                            f"reachable_frontier={int(frontier_plan['reachable_frontier_cells'])}, "
                            f"unreachable_frontier={int(frontier_plan['unreachable_frontier_cells'])}"
                            f"{centroid_txt})"
                        )
                    else:
                        print(f"[wander] frontier plan (status={plan_status})")

                # Survey arrival is a STATIONARY action (scan in place), so it
                # must be decided before blocked handling — a vantage near
                # clutter can trip the stop box while the robot stands exactly
                # on the goal, and the blocked branch would spin it forever
                # without ever clearing the survey target.
                survey_arrived = False
                if active_survey_xy is not None and plan_status in ("ok", "survey"):
                    survey_goal_distance_m = math.hypot(
                        float(frontier_plan["goal_xy"][0]) - float(current_live_pose.x),
                        float(frontier_plan["goal_xy"][1]) - float(current_live_pose.y),
                    )
                    survey_arrived = survey_goal_distance_m <= max(
                        0.45, float(args.frontier_goal_reached_m)
                    )
                if bootstrap_scan_active:
                    # Panorama is deliberately finite and monotonic: four
                    # same-direction views, then translation. Do not chase a
                    # missing heading bin forever when turn tracking is weak.
                    bootstrap_turn_commands += 1
                    chosen_direction_sign = float(direction_sign)
                    chosen_turn_deg = min(85.0, float(args.turn_deg))
                    should_turn = True
                    turn_reason = "bootstrap_scan"
                    print(
                        "[wander] bootstrap panorama turn "
                        f"({bootstrap_turn_commands}/{max(1, int(args.bootstrap_turn_captures))}, "
                        f"turn={chosen_turn_deg:.1f}deg {'ccw' if chosen_direction_sign >= 0.0 else 'cw'})"
                    )
                    if bootstrap_turn_commands >= max(1, int(args.bootstrap_turn_captures)):
                        rotation_coverage_complete = True
                        force_drive_after_turn = True
                        print(
                            "[wander] bootstrap panorama budget reached; next cycle must translate "
                            "to a new viewpoint"
                        )
                elif survey_arrived:
                    print(
                        "[wander] survey vantage reached; scanning from the new viewpoint "
                        f"({active_survey_xy[0]:.2f}, {active_survey_xy[1]:.2f})"
                    )
                    active_survey_xy = None
                    no_frontier_cycles = 0
                    motion_hint = MotionHint(
                        kind="drive",
                        expected_dx_local_m=0.0,
                        expected_dy_local_m=0.0,
                        expected_dtheta_deg=0.0,
                        search_xy_m=0.30,
                        search_theta_window_deg=12.0,
                        label=f"survey_scan_{capture_index:02d}",
                    )
                    settle_s = float(args.capture_settle_s)
                elif (
                    hazard_monitor is not None
                    and not hazard_monitor.forward_allowed()[0]
                    and not bootstrap_scan_active
                ):
                    # A camera gate (or the hold it left behind) denies forward
                    # motion. The hazard is invisible to the lidar map, so
                    # replanning alone can never route around it — recover
                    # actively. Ladder: (1) investigate the edge (measure +
                    # map it), (2) back away, (3) rotate the nose off it so
                    # the reverse-only hold releases and forward becomes
                    # retreat, (4) hold and rescan. Field deadlock 2026-07-11:
                    # without (3), a robot wedged against furniture behind
                    # looped plan->denied->capture forever.
                    # NOT during bootstrap: a 1-capture map cannot track
                    # recovery maneuvers (field 2026-07-11: a lock-lost
                    # rotate-away at capture 1 left ~70deg of untracked
                    # rotation and wrecked the map) — bootstrap scans are
                    # turns anyway, and turns release holds by rotation.
                    hazard_state_now = hazard_monitor.state()
                    _forward_ok, forward_denial = hazard_monitor.forward_allowed()
                    no_frontier_cycles = 0
                    # The recovery below rotates/reverses; any committed probe
                    # bearing is stale (and may point INTO the hazard) after it.
                    probe_commit_world_deg = None
                    probe_commit_align_turns = 0
                    print(
                        f"[safety] forward denied while planning ({forward_denial}, "
                        f"side={hazard_state_now.side}); recovering"
                    )
                    # Bundle these denials too: they dominate runs (ground
                    # gate / holds) and were undiagnosable from logs alone.
                    # Rate-limited so a hold that denies every cycle does not
                    # flood the directory with identical bundles.
                    now_denial_mono = time.monotonic()
                    if forward_denial != planning_denial_last["reason"] or (
                        now_denial_mono - planning_denial_last["mono"] >= 20.0
                    ):
                        planning_denial_last["reason"] = forward_denial
                        planning_denial_last["mono"] = now_denial_mono
                        _dump_stop_debug(current_live_pose, f"planning_denial:{forward_denial}")
                    if (
                        _elevated_hazard_engaged()
                        and not hazard_state_now.frames_stale
                        and not hazard_state_now.blind_zone
                        and not hazard_state_now.ground_active
                        and str(hazard_state_now.side) in ("left", "right", "front")
                        and not any(
                            math.hypot(
                                float(current_live_pose.x) - sx,
                                float(current_live_pose.y) - sy,
                            )
                            <= 0.45
                            for sx, sy in sidestep_spots_world
                        )
                    ):
                        sidestep_hint = _sidestep_waypoint_hint(
                            current_live_pose,
                            str(hazard_state_now.side),
                            (
                                float(planning_frontier_choice.delta_deg)
                                if planning_frontier_choice is not None
                                else None
                            ),
                        )
                        if sidestep_hint is not None:
                            sidestep_spots_world.append(
                                (float(current_live_pose.x), float(current_live_pose.y))
                            )
                            del sidestep_spots_world[:-16]
                            motion_hint = sidestep_hint
                            settle_s = float(args.move_settle_s)
                    if (
                        motion_hint is None
                        and str(args.edge_mapping) == "on"
                        and _elevated_hazard_engaged()
                        and not hazard_state_now.frames_stale
                        and not any(
                            math.hypot(
                                float(current_live_pose.x) - ix, float(current_live_pose.y) - iy
                            )
                            <= 0.7
                            for ix, iy in investigated_spots_world
                        )
                    ):
                        investigation_hint = _investigate_edge_hint(current_live_pose)
                        if investigation_hint is not None:
                            # Cooldown only on a COMPLETED survey — a skipped
                            # one (no runway) must not suppress later attempts
                            # near here.
                            investigated_spots_world.append(
                                (float(current_live_pose.x), float(current_live_pose.y))
                            )
                            del investigated_spots_world[:-16]
                            motion_hint = investigation_hint
                            settle_s = float(args.move_settle_s)
                    if motion_hint is None:
                        # A prior rotate-away losing tracking lock counts as a
                        # failed escape: in a point-starved corner rotation is
                        # untrackable (discard -> re-anchor -> oscillation), so
                        # escalate to the slow blind reverse instead.
                        escape_hint = _attempt_reverse_escape_hint(
                            current_live_pose,
                            failed_reverse_escapes >= 1 or rotate_away_lock_losses >= 1,
                        )
                        if escape_hint is not None:
                            failed_reverse_escapes = 0
                            rotate_away_lock_losses = 0
                            motion_hint = escape_hint
                            settle_s = float(args.move_settle_s)
                    # One recovery turn is followed by a forward-planning
                    # attempt. Do not spin again just because the next depth
                    # frame calls the hazard the other side.
                    if (
                        motion_hint is None
                        and rotate_away_lock_losses == 0
                        and recovery_turn_sign is None
                    ):
                        rotate_hint = _rotate_away_hint(
                            current_live_pose,
                            hazard_state_now.side,
                            (
                                float(live_frontier_choice.delta_deg)
                                if live_frontier_choice is not None
                                else None
                            ),
                        )
                        if rotate_hint is not None:
                            if rotate_away_lock_losses == 0:
                                failed_reverse_escapes = 0
                            motion_hint = rotate_hint
                            settle_s = float(args.capture_settle_s)
                            force_live_frontier_cycles = max(force_live_frontier_cycles, 2)
                    if motion_hint is None:
                        failed_reverse_escapes += 1
                        motion_hint = MotionHint(
                            kind="drive",
                            expected_dx_local_m=0.0,
                            expected_dy_local_m=0.0,
                            expected_dtheta_deg=0.0,
                            search_xy_m=0.30,
                            search_theta_window_deg=12.0,
                            label=f"hazard_hold_{capture_index:02d}",
                        )
                        settle_s = float(args.capture_settle_s)
                elif (blocked and _plan_drives_into_blocked_front()) or plan_status == "stuck":
                    # The stop box vetoes a plan ONLY when the plan actually
                    # wants to drive through the occupied front. A robot parked
                    # nose-to-wall with a valid path pointing 60deg away must
                    # simply run the transit (align turn, then gated drive) —
                    # field 2026-07-11: this branch used to fire on ANY front
                    # occupancy and spun toward alternating live-frontier
                    # bearings, pirouetting cw/ccw against the wall forever.
                    no_frontier_cycles = 0
                    consecutive_blocked_cycles += 1
                    print(
                        "[wander] path blocked "
                        f"(blocked_points={blocked_points}, status={plan_status}); the next capture adds "
                        "the blocking geometry to the map, then the planner routes around it"
                    )
                    if plan_status in ("ok", "survey", "observe"):
                        _strike_frontier(frontier_plan["face_xy"], "path_blocked")
                    if consecutive_blocked_cycles >= 3:
                        consecutive_blocked_cycles = 0
                        escape_hint = _attempt_reverse_escape_hint(
                            current_live_pose, failed_reverse_escapes >= 1
                        )
                        if escape_hint is not None:
                            failed_reverse_escapes = 0
                            motion_hint = escape_hint
                            settle_s = float(args.move_settle_s)
                        else:
                            failed_reverse_escapes += 1
                    if motion_hint is None:
                        if live_frontier_choice is not None:
                            escape_delta_deg = float(live_frontier_choice.delta_deg)
                        elif plan_status in ("ok", "survey", "observe"):
                            escape_delta_deg = _normalize_angle_deg(
                                math.degrees(
                                    math.atan2(
                                        float(frontier_plan["waypoint_xy"][1]) - float(current_live_pose.y),
                                        float(frontier_plan["waypoint_xy"][0]) - float(current_live_pose.x),
                                    )
                                )
                                - float(current_live_pose.theta_deg)
                            )
                        else:
                            escape_delta_deg = 55.0 * float(direction_sign)
                        chosen_direction_sign = 1.0 if escape_delta_deg >= 0.0 else -1.0
                        chosen_turn_deg = max(15.0, min(85.0, abs(float(escape_delta_deg))))
                        should_turn = True
                        turn_reason = "blocked_replan"
                elif plan_status == "no_frontier":
                    # A live lidar opening is newer than the stitched grid and
                    # may be beyond its current boundary (especially just
                    # after crossing a doorway). Do not declare completion
                    # while it exists: align to it, then take one guarded
                    # probe drive to extend the map.
                    if live_frontier_choice is not None:
                        no_frontier_cycles = 0
                        # Commit to ONE bearing for the whole probe episode.
                        # The residual is measured against the committed
                        # world bearing, never the per-cycle widest gap —
                        # the widest gap changes with heading, which is what
                        # caused the pirouette loop.
                        if probe_commit_world_deg is None:
                            probe_commit_world_deg = _normalize_angle_deg(
                                float(current_live_pose.theta_deg)
                                + float(live_frontier_choice.delta_deg)
                            )
                            probe_commit_align_turns = 0
                        probe_residual_deg = _normalize_angle_deg(
                            probe_commit_world_deg - float(current_live_pose.theta_deg)
                        )
                        if abs(probe_residual_deg) > 20.0 and probe_commit_align_turns < 2:
                            probe_commit_align_turns += 1
                            chosen_direction_sign = 1.0 if probe_residual_deg >= 0.0 else -1.0
                            chosen_turn_deg = max(15.0, min(85.0, abs(probe_residual_deg)))
                            should_turn = True
                            turn_reason = "live_frontier_probe_align"
                            force_drive_after_turn = True
                            print(
                                "[wander] map has no frontier but live lidar is open; aligning "
                                f"{chosen_turn_deg:.1f}deg toward the committed probe bearing "
                                f"({probe_commit_align_turns}/2)"
                            )
                        else:
                            # Aligned (or align budget spent — the drive is
                            # safety-gated either way): take the probe now
                            # and release the commitment.
                            probe_commit_world_deg = None
                            probe_commit_align_turns = 0
                            motion_hint = _live_frontier_probe_hint(
                                current_live_pose, live_frontier_choice
                            )
                            settle_s = float(args.move_settle_s)
                    else:
                        # The opening vanished from the live scan — the
                        # commitment no longer refers to anything real.
                        probe_commit_world_deg = None
                        probe_commit_align_turns = 0
                        no_frontier_cycles += 1
                    if live_frontier_choice is None:
                        if motion_hint is None and no_frontier_cycles >= 2 and pose_lost:
                            # A lost robot cannot certify anything about the
                            # map — least of all that it is finished (field
                            # 2026-07-17: "mapping complete" declared inside
                            # a ghost-sealed grid with 14833 unknown cells).
                            print(
                                "[wander] map frontier exhausted but the pose is LOST — refusing to "
                                "declare completion; continuing recovery"
                            )
                            no_frontier_cycles = 0
                        elif motion_hint is None and no_frontier_cycles >= 2 and exit_mode and exit_scan_turns < 7:
                            # EXIT RUN with no opening visible from this heading:
                            # do NOT declare completion — rotate and look for the
                            # doorway. Bounded at 7 turns (~a full revolution);
                            # only then may the normal completion path conclude.
                            exit_scan_turns += 1
                            no_frontier_cycles = 0
                            chosen_direction_sign = float(direction_sign)
                            chosen_turn_deg = 55.0
                            should_turn = True
                            turn_reason = "exit_scan"
                            print(
                                "[wander] EXIT RUN: no opening visible from this heading; "
                                f"rotating to search for the doorway ({exit_scan_turns}/7)"
                            )
                        elif motion_hint is None and no_frontier_cycles >= 2:
                            unreachable_now = int(frontier_plan.get("unreachable_frontier_cells", 0) or 0)
                            if unreachable_now >= 24:
                                print(
                                    "[wander] WARNING: declaring completion with "
                                    f"{unreachable_now} UNREACHABLE frontier cells "
                                    f"(unknown={int(frontier_plan.get('unknown_cells', 0) or 0)}) — "
                                    "if that area should be reachable, the map is suspect "
                                    "(ghost walls or clutter seal); inspect the overlay before trusting this map"
                                )
                            print(
                                "[wander] mapping complete: no reachable unmapped space remains and all "
                                "survey vantages have been visited "
                                f"(captures={len(stitch_state['snapshots'])}, "
                                f"html={stitch_state['html_path']})"
                            )
                            if str(args.on_complete) == "stop":
                                break
                            print(
                                "[wander] entering idle watch: holding position; exploration resumes "
                                "automatically if new reachable space appears (Ctrl+C to exit)"
                            )
                            _send_stop(robot)
                            idle_resume = False
                            while not idle_resume:
                                time.sleep(3.0)
                                # Heartbeat: a zero-velocity action every tick keeps
                            # the radio link busy — an idle link lets the Pi's
                                # WiFi power-save kick in and the lidar stream then
                                # stalls into reconnect loops.
                                try:
                                    robot.send_action(
                                        {
                                            "x.vel": 0.0,
                                            "y.vel": 0.0,
                                            "theta.vel": 0.0,
                                            "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
                                            "untorque_left": True,
                                            "untorque_right": True,
                                        }
                                    )
                                except Exception:
                                    pass
                                _idle_frame_id, idle_frame = feed.latest()
                                if idle_frame is None:
                                    continue
                                idle_points_xy = _scan_to_local_points(
                                    points=idle_frame.points,
                                    forward_angle_deg=float(args.forward_angle_deg),
                                    valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                                    invert_lateral_axis=bool(args.invert_lateral_axis),
                                    max_distance_m=float(args.max_distance_m),
                                    min_confidence=int(args.min_confidence),
                                    min_range_m=float(args.min_range_m),
                                )
                                if len(idle_points_xy) < 12:
                                    continue
                                # Treat the live scan as a virtual capture: if the
                                # world changed (a door opened in view), the grid
                                # gains new free space and a frontier appears.
                                idle_world_xy = _transform_points(idle_points_xy, current_live_pose)
                                idle_grid = _build_exploration_grid(
                                    poses=[*list(stitch_state["poses"]), current_live_pose],
                                    transformed_sets=[*list(stitch_state["transformed_sets"]), idle_world_xy],
                                    resolution_m=max(0.05, float(args.stitch_resolution_m)),
                                    robot_clear_radius_m=float(args.robot_radius_m) + 0.06,
                                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                                    extra_occupied_xy=(
                                        np.asarray(
                                            [[c[0] * 0.04, c[1] * 0.04] for c in elevated_occupied_world],
                                            dtype=np.float32,
                                        )
                                        if elevated_occupied_world
                                        else None
                                    ),
                                )
                                if idle_grid is None:
                                    continue
                                idle_plan = _plan_frontier_path(
                                    grid=idle_grid,
                                    robot_xy=(float(current_live_pose.x), float(current_live_pose.y)),
                                    robot_radius_m=float(args.robot_radius_m),
                                    robot_theta_deg=float(current_live_pose.theta_deg),
                                    avoid_face_xy=_active_frontier_blacklist(),
                                )
                                if str(idle_plan.get("status")) == "ok":
                                    print(
                                        "[wander] new reachable space detected during idle watch; "
                                        "resuming exploration"
                                    )
                                    no_frontier_cycles = 0
                                    idle_resume = True
                            continue
                        should_turn = True
                        chosen_direction_sign = float(direction_sign)
                        chosen_turn_deg = 85.0
                        turn_reason = "completion_verification"
                        print("[wander] no frontier and no unvisited vantage; one verification scan before stopping")
                else:
                    no_frontier_cycles = 0
                    goal_dx_m = float(frontier_plan["goal_xy"][0]) - float(current_live_pose.x)
                    goal_dy_m = float(frontier_plan["goal_xy"][1]) - float(current_live_pose.y)
                    goal_distance_m = math.hypot(goal_dx_m, goal_dy_m)
                    if goal_distance_m <= max(0.40, float(args.frontier_goal_reached_m)):
                        face_dx_m = float(frontier_plan["face_xy"][0]) - float(current_live_pose.x)
                        face_dy_m = float(frontier_plan["face_xy"][1]) - float(current_live_pose.y)
                        face_distance_m = math.hypot(face_dx_m, face_dy_m)
                        if face_distance_m < 0.30:
                            # A face point at (or under) the robot has no
                            # meaningful bearing: atan2 over centimeters made
                            # "facing it" trivially true and the robot scanned
                            # in place for 4 identical captures (field
                            # 2026-07-13: "It didn't move"). The live frontier
                            # candidate knows the real direction — use it.
                            if live_frontier_choice is not None:
                                face_delta_deg = float(live_frontier_choice.delta_deg)
                            else:
                                face_delta_deg = 85.0
                            print(
                                "[wander] frontier face point degenerate "
                                f"({face_distance_m:.2f}m away); using live candidate "
                                f"bearing (delta={face_delta_deg:.1f}deg)"
                            )
                        else:
                            face_bearing_deg = math.degrees(math.atan2(face_dy_m, face_dx_m))
                            face_delta_deg = _normalize_angle_deg(
                                face_bearing_deg - float(current_live_pose.theta_deg)
                            )
                        if abs(face_delta_deg) <= 30.0 and stationary_scan_streak >= 1:
                            # Already scanned from this exact pose and the plan
                            # came back unchanged: a second identical scan can
                            # teach nothing. Rotate instead — motion is the
                            # only way out of a stale plan.
                            face_delta_deg = (
                                float(live_frontier_choice.delta_deg)
                                if live_frontier_choice is not None
                                and abs(float(live_frontier_choice.delta_deg)) > 30.0
                                else 85.0
                            )
                            print(
                                "[wander] stationary scan already taken here; forcing a "
                                f"turn (delta={face_delta_deg:.1f}deg) instead of re-scanning"
                            )
                        if abs(face_delta_deg) > 30.0:
                            stationary_scan_streak = 0
                            chosen_direction_sign = 1.0 if face_delta_deg >= 0.0 else -1.0
                            chosen_turn_deg = max(15.0, min(85.0, abs(face_delta_deg)))
                            should_turn = True
                            turn_reason = "face_frontier"
                            print(
                                "[wander] at the frontier; turning to face the unmapped area "
                                f"(delta={face_delta_deg:.1f}deg)"
                            )
                        else:
                            stationary_scan_streak += 1
                            motion_hint = MotionHint(
                                kind="drive",
                                expected_dx_local_m=0.0,
                                expected_dy_local_m=0.0,
                                expected_dtheta_deg=0.0,
                                search_xy_m=0.30,
                                search_theta_window_deg=12.0,
                                label=f"frontier_scan_{capture_index:02d}",
                            )
                            settle_s = float(args.capture_settle_s)
                            print("[wander] at the frontier and facing it; scanning the new area")
                    else:
                        # Multi-leg transit: traveling through already-mapped
                        # space, chain several lidar-tracked turn+drive legs
                        # and commit ONE capture at the end — captures en route
                        # add no map information and slow the journey down.
                        stationary_scan_streak = 0
                        transit_pose = current_live_pose
                        transit_plan: dict[str, object] = frontier_plan
                        transit_hint: MotionHint | None = None
                        transit_traveled_m = 0.0
                        transit_legs = 0
                        transit_stop_reason = "leg_limit"
                        while transit_legs < 3:
                            waypoint_dx_m = float(transit_plan["waypoint_xy"][0]) - float(transit_pose.x)
                            waypoint_dy_m = float(transit_plan["waypoint_xy"][1]) - float(transit_pose.y)
                            waypoint_distance_m = math.hypot(waypoint_dx_m, waypoint_dy_m)
                            waypoint_delta_deg = _normalize_angle_deg(
                                math.degrees(math.atan2(waypoint_dy_m, waypoint_dx_m))
                                - float(transit_pose.theta_deg)
                            )
                            leg_pre_hint: MotionHint | None = None
                            if abs(waypoint_delta_deg) > 32.0:
                                align_turn_deg = max(15.0, min(85.0, abs(waypoint_delta_deg)))
                                align_sign = 1.0 if waypoint_delta_deg >= 0.0 else -1.0
                                print(
                                    "[wander] turning to face the path toward the goal "
                                    f"(delta={waypoint_delta_deg:.1f}deg, turning {align_turn_deg:.1f}deg "
                                    f"{'ccw' if align_sign >= 0.0 else 'cw'})"
                                )
                                align_meta = _turn_with_arc_tracking(
                                    robot=robot,
                                    feed=feed,
                                    transformed_sets=stitch_state["transformed_sets"],
                                    start_pose=transit_pose,
                                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                                    resolution_m=float(args.stitch_resolution_m),
                                    target_turn_deg=float(align_turn_deg),
                                    direction_sign=float(align_sign),
                                    turn_speed=float(args.turn_speed),
                                    turn_burst_s=float(args.turn_burst_s),
                                    turn_settle_s=float(args.turn_settle_s),
                                    forward_angle_deg=float(args.forward_angle_deg),
                                    valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                                    invert_lateral_axis=bool(args.invert_lateral_axis),
                                    max_distance_m=float(args.max_distance_m),
                                    min_range_m=float(args.min_range_m),
                                    min_confidence=int(args.min_confidence),
                                    stop_tolerance_deg=min(8.0, float(args.stop_tolerance_deg)),
                                    max_bursts=int(args.max_turn_bursts),
                    hazard_monitor=hazard_monitor,
                                )
                                align_turned_deg = float(align_meta["turned_deg"])
                                align_lever_dx, align_lever_dy = _turn_lever_arm_local_delta(
                                    align_turned_deg,
                                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                                )
                                leg_pre_hint = MotionHint(
                                    kind="turn",
                                    expected_dx_local_m=float(align_lever_dx),
                                    expected_dy_local_m=float(align_lever_dy),
                                    expected_dtheta_deg=float(align_turned_deg),
                                    search_xy_m=float(args.turn_search_xy_m),
                                    search_theta_window_deg=min(
                                        80.0, 25.0 + 8.0 * float(align_meta["missed_updates"])
                                    ),
                                    label=f"align_turn_{capture_index:02d}",
                                )
                                transit_pose = _advance_pose(transit_pose, leg_pre_hint)
                                transit_hint = (
                                    _compose_motion_hints(transit_hint, leg_pre_hint)
                                    if transit_hint is not None
                                    else leg_pre_hint
                                )
                                waypoint_dx_m = float(transit_plan["waypoint_xy"][0]) - float(transit_pose.x)
                                waypoint_dy_m = float(transit_plan["waypoint_xy"][1]) - float(transit_pose.y)
                                waypoint_distance_m = math.hypot(waypoint_dx_m, waypoint_dy_m)
                                waypoint_delta_deg = _normalize_angle_deg(
                                    math.degrees(math.atan2(waypoint_dy_m, waypoint_dx_m))
                                    - float(transit_pose.theta_deg)
                                )
                                if (
                                    bool(align_meta.get("lock_lost"))
                                    or abs(waypoint_delta_deg) > 40.0
                                    or waypoint_distance_m < 0.25
                                ):
                                    transit_stop_reason = "align_residual"
                                    break
                            planned_bursts = max(1, min(4, int(math.ceil(waypoint_distance_m / 0.30))))
                            steer_theta_vel = max(
                                -float(args.max_drive_steer_theta_vel),
                                min(
                                    float(args.max_drive_steer_theta_vel),
                                    float(waypoint_delta_deg) * float(args.frontier_drive_steer_gain),
                                ),
                            )
                            print(
                                "[wander] driving the planned path "
                                f"(leg={transit_legs + 1}, waypoint_distance={waypoint_distance_m:.2f}m, "
                                f"bursts={planned_bursts}, steer={steer_theta_vel:.3f})"
                            )
                            drive_tracked_pose, drive_track_meta = _drive_with_tracking(
                                robot=robot,
                                feed=feed,
                                zone_cfg=zone_cfg,
                                transformed_sets=stitch_state["transformed_sets"],
                                start_pose=transit_pose,
                                resolution_m=float(args.stitch_resolution_m),
                                forward_speed=float(args.move_speed),
                                min_effective_move_speed=float(args.min_effective_move_speed),
                                burst_s=float(args.move_burst_s),
                                burst_count=int(planned_bursts),
                                inter_burst_pause_s=float(args.inter_burst_pause_s),
                                steer_theta_vel=float(steer_theta_vel),
                                forward_angle_deg=float(args.forward_angle_deg),
                                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                                invert_lateral_axis=bool(args.invert_lateral_axis),
                                max_distance_m=float(args.max_distance_m),
                                min_range_m=float(args.min_range_m),
                                min_confidence=int(args.min_confidence),
                                hazard_monitor=hazard_monitor,
                                map_side_guard=_map_side_guard,
                            )
                            drive_stopped_by_block = (
                                bool(drive_track_meta["stopped_by_block"])
                                or bool(drive_track_meta["lock_lost"])
                                or bool(drive_track_meta.get("stopped_by_hazard"))
                            )
                            drive_ddx_world = float(drive_tracked_pose.x) - float(transit_pose.x)
                            drive_ddy_world = float(drive_tracked_pose.y) - float(transit_pose.y)
                            drive_theta0_rad = math.radians(float(transit_pose.theta_deg))
                            drive_cos = math.cos(drive_theta0_rad)
                            drive_sin = math.sin(drive_theta0_rad)
                            drive_hint_dx_local_m = drive_cos * drive_ddx_world + drive_sin * drive_ddy_world
                            drive_hint_dy_local_m = -drive_sin * drive_ddx_world + drive_cos * drive_ddy_world
                            drive_hint_dtheta_deg = _normalize_angle_deg(
                                float(drive_tracked_pose.theta_deg) - float(transit_pose.theta_deg)
                            )
                            if drive_track_meta["lock_lost"] or int(drive_track_meta["missed_updates"]) > 0:
                                drive_search_xy_m = 0.80
                                drive_theta_window_deg = 30.0
                            else:
                                drive_search_xy_m = 0.40
                                drive_theta_window_deg = 16.0
                            print(
                                "[wander] path leg complete "
                                f"(bursts={int(drive_track_meta['bursts_completed'])}, "
                                f"stopped_by_block={bool(drive_track_meta['stopped_by_block'])}, "
                                f"lock_lost={bool(drive_track_meta['lock_lost'])}, "
                                f"tracked_move=({drive_hint_dx_local_m:.3f}m, {drive_hint_dy_local_m:.3f}m, "
                                f"{drive_hint_dtheta_deg:.1f}deg))"
                            )
                            if not drive_stopped_by_block:
                                consecutive_blocked_cycles = 0
                                failed_reverse_escapes = 0
                                rotate_away_lock_losses = 0
                                recovery_turn_sign = None
                                wedged_events = 0
                            drive_leg_hint = MotionHint(
                                kind="drive",
                                expected_dx_local_m=float(drive_hint_dx_local_m),
                                expected_dy_local_m=float(drive_hint_dy_local_m),
                                expected_dtheta_deg=float(drive_hint_dtheta_deg),
                                search_xy_m=float(drive_search_xy_m),
                                search_theta_window_deg=float(drive_theta_window_deg),
                                label=f"drive_capture_{capture_index:02d}",
                            )
                            transit_hint = (
                                _compose_motion_hints(transit_hint, drive_leg_hint)
                                if transit_hint is not None
                                else drive_leg_hint
                            )
                            transit_traveled_m += math.hypot(drive_ddx_world, drive_ddy_world)
                            transit_pose = drive_tracked_pose
                            transit_legs += 1
                            if bool(drive_track_meta.get("stopped_by_hazard")):
                                # A confirmed edge stopped this leg. Its measured
                                # geometry is seconds-fresh and the tracked stop
                                # pose is where it was measured — stamp it into
                                # the planning map NOW. Without this, only full
                                # investigations mapped edges, and stops near a
                                # cooled-down spot taught the planner nothing:
                                # it re-planned the same goal and oscillated
                                # approach/stop/reverse at the same table.
                                _dump_stop_debug(transit_pose, "transit_hazard_stop")
                                _stamp_confirmed_edges(transit_pose, newest_only=True)
                                # A real approach toward this frontier failed on
                                # a camera hazard: strike it (2 strikes
                                # blacklist). Only genuine attempts count — a
                                # hold lingering across planning cycles is one
                                # event, not repeated evidence.
                                _strike_frontier_if_reached(
                                    transit_plan["face_xy"], transit_pose, "hazard_stop"
                                )
                            if drive_stopped_by_block:
                                transit_stop_reason = "blocked_or_lock"
                                break
                            if transit_traveled_m >= 2.6:
                                transit_stop_reason = "distance_budget"
                                break
                            goal_distance_now_m = math.hypot(
                                float(transit_plan["goal_xy"][0]) - float(transit_pose.x),
                                float(transit_plan["goal_xy"][1]) - float(transit_pose.y),
                            )
                            if goal_distance_now_m <= max(0.40, float(args.frontier_goal_reached_m)):
                                transit_stop_reason = "arrived"
                                break
                            # Replan the next leg from the new pose against the
                            # SAME grid (no new captures yet): the waypoint
                            # advances along the path.
                            transit_plan = _plan_frontier_path(
                                grid=exploration_grid,
                                robot_xy=(float(transit_pose.x), float(transit_pose.y)),
                                robot_radius_m=float(args.robot_radius_m),
                                target_xy=active_survey_xy,
                                observed_from_xy=[
                                    (float(p.x), float(p.y)) for p in stitch_state["poses"]
                                ],
                                robot_theta_deg=float(transit_pose.theta_deg),
                                avoid_face_xy=_active_frontier_blacklist(),
                                prefer_face_xy=committed_frontier_face,
                            )
                            if str(transit_plan.get("status")) not in ("ok", "survey", "observe"):
                                transit_stop_reason = "replan_" + str(transit_plan.get("status"))
                                break
                        if transit_stop_reason == "arrived":
                            # Fold the face-the-target turn into the arrival so
                            # the checkpoint capture is already aimed at the
                            # unknown area — no separate face-then-capture cycle.
                            face_dx_m = float(transit_plan["face_xy"][0]) - float(transit_pose.x)
                            face_dy_m = float(transit_plan["face_xy"][1]) - float(transit_pose.y)
                            face_delta_deg = _normalize_angle_deg(
                                math.degrees(math.atan2(face_dy_m, face_dx_m))
                                - float(transit_pose.theta_deg)
                            )
                            if math.hypot(face_dx_m, face_dy_m) > 0.35 and abs(face_delta_deg) > 35.0:
                                face_turn_deg = max(15.0, min(85.0, abs(face_delta_deg)))
                                face_sign = 1.0 if face_delta_deg >= 0.0 else -1.0
                                print(
                                    "[wander] arrived; facing the target area before the capture "
                                    f"(delta={face_delta_deg:.1f}deg)"
                                )
                                face_meta = _turn_with_arc_tracking(
                                    robot=robot,
                                    feed=feed,
                                    transformed_sets=stitch_state["transformed_sets"],
                                    start_pose=transit_pose,
                                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                                    resolution_m=float(args.stitch_resolution_m),
                                    target_turn_deg=float(face_turn_deg),
                                    direction_sign=float(face_sign),
                                    turn_speed=float(args.turn_speed),
                                    turn_burst_s=float(args.turn_burst_s),
                                    turn_settle_s=float(args.turn_settle_s),
                                    forward_angle_deg=float(args.forward_angle_deg),
                                    valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                                    invert_lateral_axis=bool(args.invert_lateral_axis),
                                    max_distance_m=float(args.max_distance_m),
                                    min_range_m=float(args.min_range_m),
                                    min_confidence=int(args.min_confidence),
                                    stop_tolerance_deg=min(8.0, float(args.stop_tolerance_deg)),
                                    max_bursts=int(args.max_turn_bursts),
                    hazard_monitor=hazard_monitor,
                                )
                                face_turned_deg = float(face_meta["turned_deg"])
                                face_lever_dx, face_lever_dy = _turn_lever_arm_local_delta(
                                    face_turned_deg,
                                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                                )
                                face_turn_hint = MotionHint(
                                    kind="turn",
                                    expected_dx_local_m=float(face_lever_dx),
                                    expected_dy_local_m=float(face_lever_dy),
                                    expected_dtheta_deg=float(face_turned_deg),
                                    search_xy_m=float(args.turn_search_xy_m),
                                    search_theta_window_deg=min(
                                        80.0, 25.0 + 8.0 * float(face_meta["missed_updates"])
                                    ),
                                    label=f"face_turn_{capture_index:02d}",
                                )
                                transit_pose = _advance_pose(transit_pose, face_turn_hint)
                                transit_hint = (
                                    _compose_motion_hints(transit_hint, face_turn_hint)
                                    if transit_hint is not None
                                    else face_turn_hint
                                )
                        print(
                            "[wander] transit complete "
                            f"(legs={transit_legs}, traveled={transit_traveled_m:.2f}m, "
                            f"reason={transit_stop_reason}); capturing checkpoint"
                        )
                        motion_hint = transit_hint
                        if (
                            transit_stop_reason == "blocked_or_lock"
                            and transit_hint is not None
                            and transit_traveled_m < 0.15
                            and math.hypot(
                                float(transit_hint.expected_dx_local_m),
                                float(transit_hint.expected_dy_local_m),
                            )
                            < 0.15
                            and abs(float(transit_hint.expected_dtheta_deg)) < 10.0
                        ):
                            # A hazard stop after barely any motion: a full
                            # checkpoint capture would re-anchor nothing and
                            # just stacks rings at the same spot.
                            checkpoint_skippable = True
                        settle_s = float(args.move_settle_s)
                        if motion_hint is None:
                            # Degenerate: nothing moved (immediate align abort);
                            # commit a scan-in-place so the cycle stays sound.
                            motion_hint = MotionHint(
                                kind="drive",
                                expected_dx_local_m=0.0,
                                expected_dy_local_m=0.0,
                                expected_dtheta_deg=0.0,
                                search_xy_m=0.30,
                                search_theta_window_deg=12.0,
                                label=f"transit_hold_{capture_index:02d}",
                            )
            elif should_turn:
                pass
            elif bootstrap_scan_active:
                # Aim each bootstrap turn at the CLOSEST heading bin the map has
                # not covered yet, instead of blindly stepping 90deg the same
                # way — stiction makes actual turn sizes erratic, so blind steps
                # revisit the same headings while leaving one bin unseen.
                unseen_bins = [
                    b for b in range(rotation_coverage_bin_count) if b not in rotation_coverage_bins_seen
                ]
                if unseen_bins:
                    bin_width_deg = 360.0 / float(rotation_coverage_bin_count)
                    current_heading_deg = float(current_live_pose.theta_deg)
                    bin_deltas = [
                        (_normalize_angle_deg((b + 0.5) * bin_width_deg - current_heading_deg), b)
                        for b in unseen_bins
                    ]
                    target_delta_deg, target_bin = min(bin_deltas, key=lambda item: abs(item[0]))
                    chosen_direction_sign = 1.0 if target_delta_deg >= 0.0 else -1.0
                    chosen_turn_deg = max(25.0, min(55.0, abs(float(target_delta_deg))))
                    print(
                        "[wander] bootstrap scan capture targeting unseen heading bin "
                        f"(capture={capture_index}, bin={target_bin}, heading={current_heading_deg:.1f}deg, "
                        f"turn={chosen_turn_deg:.1f}deg {'ccw' if chosen_direction_sign >= 0.0 else 'cw'}, "
                        f"unseen={sorted(unseen_bins)})"
                    )
                else:
                    print(
                        "[wander] bootstrap scan capture "
                        f"(capture={capture_index}, target={float(args.turn_deg):.1f}deg {args.turn_direction})"
                    )
                should_turn = True
                turn_reason = "bootstrap_scan"
            elif wander_mode == "turn_only":
                print(
                    "[wander] turn-only mode active; rotating for capture "
                    f"(frame_id={live_frame_id}, blocked_points={blocked_points})"
                )
                should_turn = True
                turn_reason = "turn_only"
            elif blocked:
                print(
                    "[wander] stop box occupied; rotating in place "
                    f"(frame_id={live_frame_id}, blocked_points={blocked_points})"
                )
                consecutive_blocked_cycles += 1
                reverse_escape_done = False
                if consecutive_blocked_cycles >= 3:
                    # Blocked at every heading we try: the robot is wedged in a
                    # pocket. Rotating cannot free it — back out along the path
                    # it came in on, and stop chasing whatever lured it here.
                    wedged_events += 1
                    if active_explore_target is not None and active_explore_target.source != "retreat":
                        _blacklist_target(active_explore_target, "wedged_pocket")
                        active_explore_target = None
                        active_explore_target_stall_count = 0
                        active_explore_target_last_distance_m = None
                        active_explore_target_blocked_count = 0
                    force_live_frontier_cycles = max(force_live_frontier_cycles, 3)
                    consecutive_blocked_cycles = 0
                    reverse_result = _reverse_escape(
                        robot=robot,
                        feed=feed,
                        transformed_sets=stitch_state["transformed_sets"],
                        start_pose=current_live_pose,
                        resolution_m=float(args.stitch_resolution_m),
                        robot_radius_m=float(args.robot_radius_m),
                        lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                        reverse_speed=max(float(args.min_effective_move_speed), float(args.move_speed) * 0.9),
                        burst_s=min(1.0, float(args.move_burst_s)),
                        bursts=2,
                        forward_angle_deg=float(args.forward_angle_deg),
                        valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                        invert_lateral_axis=bool(args.invert_lateral_axis),
                        max_distance_m=float(args.max_distance_m),
                        min_range_m=float(args.min_range_m),
                        min_confidence=int(args.min_confidence),
                        allow_blind=failed_reverse_escapes >= 1,
                    )
                    if reverse_result is None:
                        failed_reverse_escapes += 1
                    else:
                        failed_reverse_escapes = 0
                    if reverse_result is not None:
                        reverse_pose, reverse_meta = reverse_result
                        rev_ddx_world = float(reverse_pose.x) - float(current_live_pose.x)
                        rev_ddy_world = float(reverse_pose.y) - float(current_live_pose.y)
                        rev_theta0_rad = math.radians(float(current_live_pose.theta_deg))
                        rev_cos = math.cos(rev_theta0_rad)
                        rev_sin = math.sin(rev_theta0_rad)
                        motion_hint = MotionHint(
                            kind="drive",
                            expected_dx_local_m=rev_cos * rev_ddx_world + rev_sin * rev_ddy_world,
                            expected_dy_local_m=-rev_sin * rev_ddx_world + rev_cos * rev_ddy_world,
                            expected_dtheta_deg=_normalize_angle_deg(
                                float(reverse_pose.theta_deg) - float(current_live_pose.theta_deg)
                            ),
                            search_xy_m=0.45 if bool(reverse_meta.get("locked")) else 0.90,
                            search_theta_window_deg=18.0 if bool(reverse_meta.get("locked")) else 30.0,
                            label=f"reverse_escape_{capture_index:02d}",
                        )
                        should_turn = False
                        reverse_escape_done = True
                        settle_s = float(args.move_settle_s)
                    else:
                        print(
                            "[wander] no room to reverse; escaping toward the widest live opening "
                            f"(force_live_frontier_cycles={force_live_frontier_cycles})"
                        )
                    if wedged_events >= 2:
                        # Local greedy escapes aren't working: this whole area
                        # is a clutter pocket. Retreat along the robot's own
                        # pose trail — that path was driven once, so it is
                        # traversable by construction — and resume exploring
                        # from open space instead of thrashing here.
                        retreat_pose = next(
                            (
                                pose
                                for pose in reversed(list(stitch_state["poses"])[:-1])
                                if math.hypot(
                                    float(pose.x) - float(current_live_pose.x),
                                    float(pose.y) - float(current_live_pose.y),
                                )
                                >= 0.90
                            ),
                            None,
                        )
                        if retreat_pose is not None:
                            active_explore_target = ExploreTarget(
                                world_x_m=float(retreat_pose.x),
                                world_y_m=float(retreat_pose.y),
                                source="retreat",
                                seeded_capture_index=int(capture_index),
                            )
                            active_explore_target_stall_count = 0
                            active_explore_target_last_distance_m = None
                            active_explore_target_blocked_count = 0
                            force_live_frontier_cycles = 0
                            wedged_events = 0
                            print(
                                "[wander] wedged repeatedly; retreating along own trail to "
                                f"({retreat_pose.x:.3f}, {retreat_pose.y:.3f})"
                            )
                if not reverse_escape_done:
                    should_turn = True
                    blocked_frontier = live_frontier_choice or planning_frontier_choice
                    if blocked_frontier is not None:
                        chosen_direction_sign = 1.0 if blocked_frontier.delta_deg >= 0.0 else -1.0
                        chosen_turn_deg = max(12.0, min(float(args.turn_deg), abs(float(blocked_frontier.delta_deg))))
                        turn_reason = "blocked_frontier"
                    else:
                        turn_reason = "blocked"
            elif (
                wander_mode == "smart"
                and rotation_coverage_complete
                and not force_drive_after_turn
                and (
                    active_explore_target is None
                    or int(active_explore_target.coarse_turns_used) <= 0
                )
                and planning_frontier_choice is not None
                and abs(float(planning_frontier_choice.delta_deg)) >= 115.0
            ):
                chosen_direction_sign = 1.0 if planning_frontier_choice.delta_deg >= 0.0 else -1.0
                chosen_turn_deg = max(22.0, min(55.0, abs(float(planning_frontier_choice.delta_deg)) - 32.0))
                should_turn = True
                turn_reason = "frontier_seek"
                print(
                    "[wander] rotation coverage is complete; performing one coarse frontier seek turn "
                    f"(frame_id={live_frame_id}, source={planning_frontier_source}, "
                    f"delta={planning_frontier_choice.delta_deg:.1f}deg, "
                    f"distance={planning_frontier_choice.mean_distance_m:.2f}m, "
                    f"width={planning_frontier_choice.width_deg:.1f}deg)"
                )
            elif (
                wander_mode == "smart"
                and not force_drive_after_turn
                and not rotation_coverage_complete
                and planning_frontier_choice is not None
                and abs(float(planning_frontier_choice.delta_deg)) >= float(args.frontier_align_threshold_deg)
            ):
                chosen_direction_sign = 1.0 if planning_frontier_choice.delta_deg >= 0.0 else -1.0
                chosen_turn_deg = max(15.0, min(float(args.turn_deg), abs(float(planning_frontier_choice.delta_deg))))
                should_turn = True
                turn_reason = "frontier_align"
                print(
                    "[wander] frontier is far off heading; aligning before drive "
                    f"(frame_id={live_frame_id}, source={planning_frontier_source}, "
                    f"delta={planning_frontier_choice.delta_deg:.1f}deg, "
                    f"distance={planning_frontier_choice.mean_distance_m:.2f}m, "
                    f"width={planning_frontier_choice.width_deg:.1f}deg)"
                )
            else:
                print(
                    "[wander] forward path is clear; driving sequence "
                    f"(frame_id={live_frame_id}, blocked_points={blocked_points})"
                )
                steer_theta_vel = _compute_drive_steer_theta_vel(
                    planning_frontier_choice,
                    steer_gain=float(args.frontier_drive_steer_gain),
                    steer_max=float(args.max_drive_steer_theta_vel),
                    steer_deadband_deg=float(args.drive_steer_deadband_deg),
                )
                planned_bursts = max(1, int(args.drive_bursts_per_capture))
                if planning_frontier_choice is not None:
                    if planning_frontier_choice.mean_distance_m >= 2.4:
                        planned_bursts += 1
                    if planning_frontier_choice.mean_distance_m >= 3.4 and planning_frontier_choice.width_deg >= 24.0:
                        planned_bursts += 1
                planned_bursts = min(planned_bursts, max(1, int(args.max_drive_bursts_per_capture)))
                print(
                    "[wander] drive plan "
                    f"(planned_bursts={planned_bursts}, frontier_delta="
                    f"{None if planning_frontier_choice is None else round(float(planning_frontier_choice.delta_deg), 1)}, "
                    f"frontier_distance={None if planning_frontier_choice is None else round(float(planning_frontier_choice.mean_distance_m), 2)}, "
                    f"frontier_source={planning_frontier_source}, "
                    f"drive_mode={'frontier_follow' if planning_frontier_choice is not None else 'straight_burst'}, "
                    f"steer_theta_vel={steer_theta_vel:.3f})"
                )
                drive_start_pose = current_live_pose
                drive_tracked_pose, drive_track_meta = _drive_with_tracking(
                    robot=robot,
                    feed=feed,
                    zone_cfg=zone_cfg,
                    transformed_sets=stitch_state["transformed_sets"],
                    start_pose=drive_start_pose,
                    resolution_m=float(args.stitch_resolution_m),
                    forward_speed=float(args.move_speed),
                    min_effective_move_speed=float(args.min_effective_move_speed),
                    burst_s=float(args.move_burst_s),
                    burst_count=int(planned_bursts),
                    inter_burst_pause_s=float(args.inter_burst_pause_s),
                    steer_theta_vel=float(steer_theta_vel),
                    forward_angle_deg=float(args.forward_angle_deg),
                    valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                    invert_lateral_axis=bool(args.invert_lateral_axis),
                    max_distance_m=float(args.max_distance_m),
                    min_range_m=float(args.min_range_m),
                    min_confidence=int(args.min_confidence),
                    hazard_monitor=hazard_monitor,
                    map_side_guard=_map_side_guard,
                )
                drive_stopped_by_block = (
                    bool(drive_track_meta["stopped_by_block"])
                    or bool(drive_track_meta["lock_lost"])
                    or bool(drive_track_meta.get("stopped_by_hazard"))
                )
                if bool(drive_track_meta.get("stopped_by_hazard")):
                    # Stamp the freshly measured geometry at the tracked
                    # stop pose (see transit path): every hazard stop teaches
                    # the planner the boundary, not just investigations.
                    _dump_stop_debug(drive_tracked_pose, "forward_hazard_stop")
                    _stamp_confirmed_edges(drive_tracked_pose, newest_only=True)
                    if plan_status in ("ok", "survey", "observe"):
                        _strike_frontier_if_reached(
                            frontier_plan["face_xy"], drive_tracked_pose, "hazard_stop"
                        )
                # Motion hint straight from the tracked poses — no commanded-
                # speed or wheel-feedback guessing.
                drive_ddx_world = float(drive_tracked_pose.x) - float(drive_start_pose.x)
                drive_ddy_world = float(drive_tracked_pose.y) - float(drive_start_pose.y)
                drive_theta0_rad = math.radians(float(drive_start_pose.theta_deg))
                drive_cos = math.cos(drive_theta0_rad)
                drive_sin = math.sin(drive_theta0_rad)
                drive_hint_dx_local_m = drive_cos * drive_ddx_world + drive_sin * drive_ddy_world
                drive_hint_dy_local_m = -drive_sin * drive_ddx_world + drive_cos * drive_ddy_world
                drive_hint_dtheta_deg = _normalize_angle_deg(
                    float(drive_tracked_pose.theta_deg) - float(drive_start_pose.theta_deg)
                )
                # LIDAR-ONLY: the tracked heading delta is the hint even after
                # lock loss (the wider search window below owns that case). The
                # gyro never writes into motion hints — advisory only.
                drive_lock_lost = bool(drive_track_meta["lock_lost"]) or int(
                    drive_track_meta["missed_updates"]
                ) > 0
                if drive_lock_lost:
                    drive_search_xy_m = 0.80
                    drive_theta_window_deg = 30.0
                else:
                    drive_search_xy_m = 0.40
                    drive_theta_window_deg = 16.0
                print(
                    "[wander] forward sequence complete "
                    f"(elapsed={float(drive_track_meta['elapsed_s']):.2f}s "
                    f"bursts={int(drive_track_meta['bursts_completed'])} "
                    f"stopped_by_block={bool(drive_track_meta['stopped_by_block'])} "
                    f"lock_lost={bool(drive_track_meta['lock_lost'])} "
                    f"blocked_points={int(drive_track_meta['blocked_points'])} "
                    f"tracked_move=({drive_hint_dx_local_m:.3f}m, {drive_hint_dy_local_m:.3f}m, "
                    f"{drive_hint_dtheta_deg:.1f}deg))"
                )
                force_drive_after_turn = False
                if dead_reck_theta_deg is not None:
                    dead_reck_theta_deg += float(drive_hint_dtheta_deg)
                    dead_reck_slack_deg += 5.0
                    if dead_reck_slack_deg > 100.0:
                        dead_reck_theta_deg = None
                drive_checkpoints_since_scan += 1
                if (
                    bool(drive_track_meta.get("stopped_by_hazard"))
                    and math.hypot(drive_hint_dx_local_m, drive_hint_dy_local_m) < 0.15
                    and abs(drive_hint_dtheta_deg) < 10.0
                ):
                    checkpoint_skippable = True
                if not drive_stopped_by_block:
                    # Only a drive that actually got somewhere disarms the
                    # wedge detection. A one-burst drive straight into the stop
                    # box must keep counting toward "we are stuck here", or the
                    # blocked-turn-drive dance resets the counter forever and
                    # the reverse/retreat escapes never fire.
                    consecutive_blocked_cycles = 0
                    failed_reverse_escapes = 0
                    rotate_away_lock_losses = 0
                    wedged_events = 0
                if wander_mode == "scan_turn_every_capture":
                    pending_turn_reason = "post_drive_scan"
                    pending_turn_direction_sign = float(direction_sign)
                    pending_turn_deg = float(args.turn_deg)
                    should_turn = False
                    turn_reason = "drive_only"
                elif drive_stopped_by_block:
                    should_turn = False
                    if active_explore_target is not None:
                        active_explore_target_blocked_count += 1
                        print(
                            "[wander] stitched-map target blocked during drive "
                            f"(source={active_explore_target.source}, "
                            f"seed_capture={active_explore_target.seeded_capture_index}, "
                            f"blocked_count={active_explore_target_blocked_count})"
                        )
                        force_live_frontier_cycles = max(force_live_frontier_cycles, 2)
                        if active_explore_target_blocked_count >= 3:
                            print(
                                "[wander] abandoning stitched-map target after repeated blocked drives "
                                f"(source={active_explore_target.source}, "
                                f"seed_capture={active_explore_target.seeded_capture_index})"
                            )
                            if active_explore_target.source != "retreat":
                                _blacklist_target(active_explore_target, "blocked_drives")
                            active_explore_target = None
                            active_explore_target_stall_count = 0
                            active_explore_target_last_distance_m = None
                            active_explore_target_blocked_count = 0
                    recovery_frontier = live_frontier_choice or planning_frontier_choice
                    if recovery_frontier is not None:
                        pending_turn_direction_sign = 1.0 if recovery_frontier.delta_deg >= 0.0 else -1.0
                        pending_turn_deg = max(18.0, min(float(args.turn_deg), abs(float(recovery_frontier.delta_deg))))
                        pending_turn_reason = "drive_blocked_frontier"
                    else:
                        pending_turn_direction_sign = float(direction_sign)
                        pending_turn_deg = float(args.turn_deg)
                        pending_turn_reason = "drive_blocked"
                    turn_reason = "drive_only"
                    drive_checkpoints_since_scan = 0
                elif (
                    wander_mode == "smart"
                    and not rotation_coverage_complete
                    and int(args.scan_turn_interval) > 0
                    and drive_checkpoints_since_scan >= int(args.scan_turn_interval)
                ):
                    should_turn = False
                    if planning_frontier_choice is not None and abs(planning_frontier_choice.delta_deg) >= float(args.frontier_align_threshold_deg):
                        pending_turn_direction_sign = 1.0 if planning_frontier_choice.delta_deg >= 0.0 else -1.0
                        pending_turn_deg = max(15.0, min(float(args.turn_deg), abs(float(planning_frontier_choice.delta_deg))))
                        pending_turn_reason = "periodic_frontier_scan"
                    else:
                        pending_turn_direction_sign = float(direction_sign)
                        pending_turn_deg = float(args.turn_deg)
                        pending_turn_reason = "periodic_scan"
                    turn_reason = "drive_only"
                    drive_checkpoints_since_scan = 0
                else:
                    should_turn = False
                    turn_reason = "drive_only"
                    if force_live_frontier_cycles > 0:
                        force_live_frontier_cycles -= 1
                motion_hint = MotionHint(
                    kind="drive",
                    expected_dx_local_m=float(drive_hint_dx_local_m),
                    expected_dy_local_m=float(drive_hint_dy_local_m),
                    expected_dtheta_deg=float(drive_hint_dtheta_deg),
                    search_xy_m=float(drive_search_xy_m),
                    search_theta_window_deg=float(drive_theta_window_deg),
                    label=f"drive_capture_{capture_index:02d}",
                )
                print(
                    "[wander] drive capture hint "
                    f"(pose_source=lidar_tracked, search_xy_m={drive_search_xy_m:.3f}, "
                    f"expected_dx_local_m={drive_hint_dx_local_m:.3f}, "
                    f"expected_dy_local_m={drive_hint_dy_local_m:.3f}, "
                    f"expected_dtheta_deg={drive_hint_dtheta_deg:.1f})"
                )
                settle_s = float(args.move_settle_s)

            if should_turn:
                # SIDE-COLLISION GUARD for rotation: the collision that
                # motivated this (field 2026-07-17) was a commanded turn
                # BESIDE a mapped table edge — "no rotation progress" while
                # the shoulder ground along the tabletop. No live sensor
                # covers the sides at tabletop height; the map does. Too
                # close to a mapped cell -> don't rotate here, back off
                # first and let the planner re-approach with clearance.
                turn_guard_prox = _nearest_mapped_elevated_m(current_live_pose)
                if (
                    turn_guard_prox is not None
                    and turn_guard_prox < float(args.robot_radius_m) + 0.04
                ):
                    print(
                        f"[wander] mapped elevated cell {turn_guard_prox:.2f}m from the body — "
                        "skipping rotation beside it and backing off first (side-collision guard)"
                    )
                    should_turn = False
                    turn_guard_retreat = _attempt_reverse_escape_hint(current_live_pose, True)
                    if turn_guard_retreat is not None:
                        motion_hint = turn_guard_retreat
                        settle_s = float(args.move_settle_s)
            if should_turn:
                turn_cap_deg = 85.0
                if (
                    len(stitch_state["poses"]) < 4
                    or len(rotation_coverage_bins_seen) < rotation_coverage_bin_count
                ) and turn_reason != "reanchor_turn_back":
                    # Until the map covers the full rotation (all 4 coverage
                    # BINS — not the rotation_coverage_complete flag, which
                    # the bootstrap budget force-sets with bins at [0,1]/4;
                    # field run 12: that lie disarmed this cap and every
                    # 85deg face_frontier turn into unmapped bearings lost
                    # lock), an 85deg turn can rotate the view past the
                    # map's angular edge and overlap collapses. 55deg keeps
                    # every solve anchored in mapped bearings; larger goals
                    # just take an extra capture.
                    turn_cap_deg = 55.0
                    if len(stitch_state["poses"]) < 2:
                        # Against the founding snapshot alone, even 55deg
                        # loses lock in sparse spots (run 11: lock lost at
                        # 46.5deg on the very first turn). 40deg keeps
                        # ~140deg of the founding view in frame.
                        turn_cap_deg = 40.0
                if float(chosen_turn_deg) > turn_cap_deg:
                    # A capture must land before the view rotates into mostly
                    # unmapped territory: with a ~180deg FOV, 55deg per capture
                    # keeps >=125deg of the previous view in frame, which keeps
                    # solve scores strong. Larger goals just take two captures.
                    print(
                        f"[wander] capping turn at {turn_cap_deg:.0f}deg per capture to keep map overlap strong "
                        f"(requested={float(chosen_turn_deg):.1f}deg)"
                    )
                    chosen_turn_deg = turn_cap_deg
                print(
                    "[wander] rotating before stitched capture "
                    f"(reason={turn_reason}, target={float(chosen_turn_deg):.1f}deg "
                    f"{'ccw' if chosen_direction_sign >= 0.0 else 'cw'})"
                )
                # Anchor tracking at the dead-reckoned pose INCLUDING any
                # pending (discarded-capture) motion — after a lock-lost
                # discard the robot is physically far from the last stitched
                # pose, and anchoring there makes every retry solve miss.
                # (If live relocalization already re-anchored this cycle, the
                # pending motion is baked into current_live_pose and will be
                # dropped at composition time — don't double-count it here.)
                turn_start_pose = (
                    _advance_pose(current_live_pose, pending_motion_hint)
                    if (pending_motion_hint is not None and not live_pose_accepted)
                    else current_live_pose
                )
                imu_yaw_before_turn = imu_yaw.deg_fresh() if imu_yaw is not None else None
                turn_meta = _turn_with_arc_tracking(
                    robot=robot,
                    feed=feed,
                    transformed_sets=stitch_state["transformed_sets"],
                    start_pose=turn_start_pose,
                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                    resolution_m=float(args.stitch_resolution_m),
                    target_turn_deg=float(chosen_turn_deg),
                    direction_sign=float(chosen_direction_sign),
                    turn_speed=float(args.turn_speed),
                    turn_burst_s=float(args.turn_burst_s),
                    turn_settle_s=float(args.turn_settle_s),
                    forward_angle_deg=float(args.forward_angle_deg),
                    valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                    invert_lateral_axis=bool(args.invert_lateral_axis),
                    max_distance_m=float(args.max_distance_m),
                    min_range_m=float(args.min_range_m),
                    min_confidence=int(args.min_confidence),
                    stop_tolerance_deg=min(8.0, float(args.stop_tolerance_deg)),
                    max_bursts=int(args.max_turn_bursts),
                    hazard_monitor=hazard_monitor,
                )
                tracked_turn_deg = float(turn_meta["turned_deg"])
                turn_lock_lost = int(turn_meta["completed"]) != 1
                # Gyro-measured rotation across the whole turn, bracketed by FRESH
                # samples (a mid-turn stale sample under-reports — it once read
                # -8deg for a real -55deg turn). Unlike the lidar arc tracker, the
                # IMU still measures the rotation when scan-match lock is lost
                # mid-turn (the exact case that used to strand the dead-reckoned
                # heading and let a room-symmetric wrong mode win).
                imu_turn_deg: float | None = None
                if imu_yaw is not None and imu_yaw_before_turn is not None:
                    imu_after = imu_yaw.deg_fresh()
                    if imu_after is not None:
                        # Raw difference: the published yaw is continuous, so this
                        # is exact even for rotations beyond +-180deg.
                        imu_turn_deg = float(imu_after) - float(imu_yaw_before_turn)
                    if not turn_lock_lost:
                        imu_yaw.note_tracked_turn(tracked_turn_deg, imu_turn_deg)
                # LIDAR-ONLY mapping (field 2026-07-18, run 4): the IMU is
                # ADVISORY, never an author. An earlier version substituted the
                # gyro delta into the motion hint on lock-lost turns; those
                # values got composed into append chains and (via marginal
                # guided restores) ghosted the map. The lidar's tracked value —
                # even when it under-counts after lock loss — is what the append
                # gates were tuned around; the gyro is only shown in the log and
                # keeps the recovery tiebreaker's general-idea heading fresh.
                measured_signed_turn_deg = tracked_turn_deg
                print(
                    "[wander] turn complete "
                    f"(tracked={tracked_turn_deg:+.1f}deg, "
                    f"imu={'n/a' if imu_turn_deg is None else f'{imu_turn_deg:+.1f}deg'}, "
                    f"final_theta={turn_meta['final_theta_deg']:.1f}deg, "
                    f"missed_updates={int(turn_meta['missed_updates'])}, completed={int(turn_meta['completed'])})"
                )
                if dead_reck_theta_deg is not None:
                    # Advance the general-idea anchor by the TRACKED turn; the
                    # untracked remainder (plus a per-turn margin) widens the
                    # slack. A chain of blind turns widens it past usefulness and
                    # recovery falls back to score-only. (When the gyro feed is
                    # live, the recovery judge re-derives this heading from the
                    # gyro anchor anyway — this path is the lidar-only fallback.)
                    dead_reck_theta_deg += measured_signed_turn_deg
                    dead_reck_slack_deg += 6.0 + max(
                        0.0, float(chosen_turn_deg) - abs(measured_signed_turn_deg)
                    )
                    if dead_reck_slack_deg > 100.0:
                        dead_reck_theta_deg = None
                if int(turn_meta["completed"]) != 1:
                    # Lock-lost turn: the robot physically rotated an UNKNOWN
                    # amount (bursts fire whether or not the solve tracks
                    # them). The measured hint under-counts, so every solve
                    # anchored on it is suspect — latch lost.
                    if not pose_lost:
                        pose_lost = True
                        print(
                            "[wander] pose integrity LOST (turn ended with tracking lock lost); "
                            "map is READ-ONLY until a strong relocalization re-proves the pose"
                        )
                # The capture solve only needs to confirm/refine the tracked
                # heading; widen its window a bit for each burst the tracker
                # could not update on.
                turn_capture_theta_window_deg = min(
                    80.0, 25.0 + 8.0 * float(turn_meta["missed_updates"])
                )
                print(
                    "[wander] turn motion hint "
                    f"(requested_dtheta_deg={float(chosen_turn_deg) * float(chosen_direction_sign):.1f}, "
                    f"measured_dtheta_deg={measured_signed_turn_deg:.1f})"
                )
                if turn_reason in {
                    "blocked",
                    "blocked_frontier",
                    "drive_blocked",
                    "drive_blocked_frontier",
                    "bootstrap_scan",
                    "turn_only",
                    "frontier_seek",
                    "frontier_align",
                    "periodic_frontier_scan",
                    "periodic_scan",
                }:
                    drive_checkpoints_since_scan = 0
                if turn_reason in {"frontier_align", "periodic_frontier_scan", "frontier_seek"}:
                    force_drive_after_turn = True
                    if active_explore_target is not None and turn_reason == "frontier_seek":
                        active_explore_target.coarse_turns_used += 1
                lever_dx_local_m, lever_dy_local_m = _turn_lever_arm_local_delta(
                    float(measured_signed_turn_deg),
                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                )
                motion_hint = MotionHint(
                    kind="turn",
                    expected_dx_local_m=float(lever_dx_local_m),
                    expected_dy_local_m=float(lever_dy_local_m),
                    expected_dtheta_deg=float(measured_signed_turn_deg),
                    search_xy_m=float(args.turn_search_xy_m),
                    search_theta_window_deg=float(args.turn_theta_window_deg),
                    label=f"turn_capture_{capture_index:02d}",
                )
                print(
                    "[wander] turn lever-arm hint "
                    f"(expected_dx_local={lever_dx_local_m:.3f}m, expected_dy_local={lever_dy_local_m:.3f}m, "
                    f"offset={float(args.lidar_offset_forward_m):.3f}m)"
                )
                settle_s = float(args.capture_settle_s)

            if (
                pending_motion_hint is None
                and str(motion_hint.label or "").startswith(("hazard_hold", "edge_hold"))
                and abs(float(motion_hint.expected_dx_local_m)) < 0.02
                and abs(float(motion_hint.expected_dy_local_m)) < 0.02
                and abs(float(motion_hint.expected_dtheta_deg)) < 2.0
            ):
                # Nothing moved since the last capture (a hold denied every
                # recovery action): an identical capture from an identical
                # pose teaches the map NOTHING. Wait briefly and replan
                # instead of stacking duplicate captures at one spot.
                print(
                    "[wander] nothing moved since the last capture (hold cycle); "
                    "skipping the duplicate capture and replanning"
                )
                time.sleep(max(0.5, float(settle_s)))
                continue
            if checkpoint_skippable and pending_motion_hint is None:
                # Hazard stop after <0.15m of motion: skip the checkpoint
                # capture (it would re-anchor nothing) and carry the tracked
                # motion into the NEXT capture's expectation instead.
                print(
                    "[wander] short hazard stop; skipping the checkpoint capture "
                    "and carrying the tracked motion forward"
                )
                pending_motion_hint = motion_hint
                time.sleep(0.3)
                continue

            # REDUNDANT-VANTAGE GATE: once the map is mature, a checkpoint
            # capture is only worth taking from a genuinely NEW viewpoint.
            # Field 2026-07-18 run 20: 22 captures piled up inside a ~0.5m
            # patch — every micro-transit (0.05m stop-box interruptions) and
            # every 55deg face-turn step stopped to append another snapshot
            # of a viewpoint the map already owned. Skip the capture, chain
            # the tracked motion, and keep moving toward the target vantage:
            # drive there, shoot once, plan the next vantage. Appends resume
            # automatically the moment the robot reaches unclaimed ground
            # (or whenever pose trust degrades — recovery needs anchors).
            if (
                not pose_lost
                and rotation_coverage_complete
                and consecutive_append_discards == 0
                and len(stitch_state["poses"]) >= 6
                and redundant_skip_streak < 6
            ):
                redundancy_hint = (
                    motion_hint
                    if pending_motion_hint is None
                    else _compose_motion_hints(pending_motion_hint, motion_hint)
                )
                chain_translation_m = math.hypot(
                    float(redundancy_hint.expected_dx_local_m),
                    float(redundancy_hint.expected_dy_local_m),
                )
                if chain_translation_m <= 1.2 and abs(
                    float(redundancy_hint.expected_dtheta_deg)
                ) <= 120.0:
                    redundancy_pose = _advance_pose(
                        stitch_state["poses"][-1], redundancy_hint
                    )
                    covered_by = None
                    for prior_index, prior_pose in enumerate(stitch_state["poses"]):
                        if (
                            math.hypot(
                                float(prior_pose.x) - float(redundancy_pose.x),
                                float(prior_pose.y) - float(redundancy_pose.y),
                            )
                            < 0.40
                            and abs(
                                _normalize_angle_deg(
                                    float(prior_pose.theta_deg)
                                    - float(redundancy_pose.theta_deg)
                                )
                            )
                            < 60.0
                        ):
                            covered_by = prior_index
                            break
                    if covered_by is not None:
                        redundant_skip_streak += 1
                        print(
                            "[wander] vantage already covered by capture "
                            f"#{covered_by + 1} (within 0.40m/60deg); skipping the "
                            "redundant capture and continuing toward a new viewpoint "
                            f"({redundant_skip_streak}/6 before a forced refresh)"
                        )
                        if redundant_skip_streak == 3 and committed_frontier_face is not None:
                            # PIROUETTE BREAKER (field 2026-07-18: "face the
                            # unmapped area" and "forcing a turn instead of
                            # re-scanning" alternated +-85deg for six straight
                            # skipped captures at one frontier). Three redundant
                            # skips while committed to one face means this
                            # frontier CANNOT be resolved from here — observing
                            # it again teaches nothing. Strike it so the planner
                            # moves on instead of turning in place forever.
                            print(
                                "[wander] frontier at "
                                f"({committed_frontier_face[0]:.2f}, {committed_frontier_face[1]:.2f}) "
                                "produced 3 straight redundant captures from this vantage — "
                                "striking it and moving on (pirouette breaker)"
                            )
                            _strike_frontier(
                                committed_frontier_face,
                                "unresolvable from this vantage (redundant captures)",
                                force=True,
                            )
                        pending_motion_hint = redundancy_hint
                        continue
            redundant_skip_streak = 0

            print(f"[wander] settling for {settle_s:.2f}s before capture")
            time.sleep(settle_s)

            frame_id, captured_frame, local_points_xy, captured_snapshot = _capture_snapshot(
                feed=feed,
                output_dir=snapshot_dir,
                request_index=capture_index,
                after_frame_id=last_frame_id,
                forward_angle_deg=float(args.forward_angle_deg),
                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                invert_lateral_axis=bool(args.invert_lateral_axis),
                max_distance_m=float(args.max_distance_m),
                min_range_m=float(args.min_range_m),
                min_confidence=int(args.min_confidence),
                fresh_frame_timeout_s=float(args.fresh_frame_timeout_s),
                fresh_frame_advances=int(args.fresh_frame_advances),
                capture_config_extra={
                    "motion_hint": motion_hint.kind,
                    "motion_hint_label": motion_hint.label,
                    "expected_dx_local_m": motion_hint.expected_dx_local_m,
                    "expected_dtheta_deg": motion_hint.expected_dtheta_deg,
                },
            )
            if len(local_points_xy) == 0:
                print(f"[wander] warning: snapshot {capture_index} contains zero local points after filtering")
            if int(frame_id) == int(last_frame_id):
                # The feed produced NO new revolution: this "capture" is the
                # previous frame again. Appending it would stitch a duplicate
                # of a frozen world (field 2026-07-11: a dead host got the
                # identical frame appended 13 times). Let the feed breaker
                # at the loop top hold the robot until data flows again.
                print(
                    f"[wander] capture returned the SAME frame (frame_id={frame_id}); "
                    "feed is stalled — discarding it and holding position"
                )
                _send_stop(robot)
                pending_motion_hint = (
                    motion_hint
                    if pending_motion_hint is None
                    else _compose_motion_hints(pending_motion_hint, motion_hint)
                )
                continue

            if pending_motion_hint is not None:
                if live_pose_accepted:
                    print(
                        "[wander] dropping pending discarded-capture motion; "
                        "live relocalization already re-anchored the pose"
                    )
                else:
                    motion_hint = _compose_motion_hints(pending_motion_hint, motion_hint)
                    print(
                        "[wander] composed pending motion from discarded capture into current hint "
                        f"(expected_dx={motion_hint.expected_dx_local_m:.3f}m, "
                        f"expected_dy={motion_hint.expected_dy_local_m:.3f}m, "
                        f"expected_dtheta={motion_hint.expected_dtheta_deg:.1f}deg)"
                    )
                pending_motion_hint = None

            motion_hints.append(motion_hint)
            rebuild_started = time.monotonic()
            print("[wander] appending stitched capture " f"(capture={capture_index}, snapshots={len(motion_hints)})")
            capture_expected_pose = _advance_pose(current_live_pose, motion_hint)
            capture_live_pose = None
            capture_live_meta = None
            append_solver_pose = None
            append_solver_meta = None
            if motion_hint.kind == "turn":
                # In-place turn: the robot center is pinned, so solve on the
                # lever-arm arc. The measured turn angle only centers a wide
                # search window — it is too unreliable to act as a hard prior.
                arc_theta_half_window_deg = min(
                    110.0,
                    float(turn_capture_theta_window_deg)
                    + (30.0 if "+" in str(motion_hint.label or "") else 0.0),
                )
                arc_result = _solve_turn_arc_pose(
                    snapshot_points_xy=captured_snapshot.points_xy,
                    transformed_sets=stitch_state["transformed_sets"],
                    previous_pose=current_live_pose,
                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                    resolution_m=float(args.stitch_resolution_m),
                    expected_theta_deg=float(capture_expected_pose.theta_deg),
                    theta_half_window_deg=float(arc_theta_half_window_deg),
                )
                if arc_result is not None:
                    arc_solved_pose, arc_meta = arc_result
                    append_solver_pose = arc_solved_pose
                    append_solver_meta = arc_meta
                    arc_score = float(arc_meta.get("score") or -1e9)
                    solved_turn_deg = _normalize_angle_deg(
                        float(arc_solved_pose.theta_deg) - float(current_live_pose.theta_deg)
                    )
                    print(
                        "[wander] turn arc solve "
                        f"(capture={capture_index}, pose=({arc_solved_pose.x:.3f}, {arc_solved_pose.y:.3f}, "
                        f"{arc_solved_pose.theta_deg:.1f}deg), turned={solved_turn_deg:.1f}deg, "
                        f"hinted={float(motion_hint.expected_dtheta_deg):.1f}deg, score={arc_score:.3f}, "
                        f"window=+-{arc_theta_half_window_deg:.0f}deg)"
                    )
                    hint_discrepancy_deg = abs(
                        _normalize_angle_deg(solved_turn_deg - float(motion_hint.expected_dtheta_deg))
                    )
                    arc_accept_gate = float(args.min_append_score)
                    if consecutive_append_discards > 0:
                        # Previous capture was discarded: the hint anchoring
                        # this solve is suspect — demand a strong match (see
                        # the append-gate escalation below).
                        arc_accept_gate += 2.0
                    if post_recovery_strict_appends > 0:
                        arc_accept_gate += 2.0
                    if pose_lost:
                        # Map is read-only while lost: only an absolute-trust
                        # match may append (field 2026-07-17: an 11.86 arc
                        # solve with a USELESS hint — hinted=0 after lock
                        # loss — passed the raised 9.0 gate at a wrong 90deg
                        # symmetry mode and ghosted the whole map).
                        arc_accept_gate = max(arc_accept_gate, POSE_LOST_APPEND_GATE)
                        print(
                            f"[wander] pose lost: arc append requires score>={arc_accept_gate:.1f}"
                        )
                    if hint_discrepancy_deg > 30.0:
                        # The solver is overriding the tracked turn by a lot. In a
                        # square room the wrong 90-degree rotation mode scores
                        # respectably, so a big override needs strong evidence,
                        # not the standard gate. Discarding here is safe: the
                        # motion hint carries forward and the next capture solves
                        # with a wider window and more map context.
                        arc_accept_gate += 1.5
                        print(
                            "[wander] warning: arc solve disagrees strongly with the measured turn "
                            f"(discrepancy={hint_discrepancy_deg:.1f}deg); possible room-symmetry mode, "
                            f"raising gate to {arc_accept_gate:.2f}"
                        )
                    if arc_score >= arc_accept_gate:
                        capture_live_pose = arc_solved_pose
                        capture_live_meta = arc_meta
                    elif hint_discrepancy_deg <= 8.0:
                        # Rotation is mutually confirmed by geometry and tracking,
                        # so a poor absolute fit usually means the robot CENTER has
                        # drifted (skid accumulates over consecutive in-place turns
                        # and the arc constraint only allows +-10cm). Refine xy
                        # around the arc pose and re-gate at full strength instead
                        # of painting a laterally-offset scan into the map.
                        refine_result = _estimate_pose_against_stitched_map(
                            points_xy=captured_snapshot.points_xy,
                            transformed_sets=stitch_state["transformed_sets"],
                            initial_pose=arc_solved_pose,
                            resolution_m=float(args.stitch_resolution_m),
                            search_xy_m=0.35,
                            theta_window_deg=8.0,
                            max_translation_from_initial_m=0.35,
                            prior_translation_weight=1.2,
                            prior_theta_weight=0.10,
                        )
                        refined_ok = False
                        if refine_result is not None:
                            refined_pose, refined_meta = refine_result
                            refined_score = float(refined_meta.get("score") or -1e9)
                            print(
                                "[wander] turn arc xy-refine "
                                f"(capture={capture_index}, pose=({refined_pose.x:.3f}, {refined_pose.y:.3f}, "
                                f"{refined_pose.theta_deg:.1f}deg), score={refined_score:.3f}, "
                                f"arc_score={arc_score:.3f})"
                            )
                            refine_gate = float(args.min_append_score) + (
                                2.0 if consecutive_append_discards > 0 else 0.0
                            )
                            if post_recovery_strict_appends > 0:
                                refine_gate += 2.0
                            if pose_lost:
                                refine_gate = max(refine_gate, POSE_LOST_APPEND_GATE)
                            if refined_score >= refine_gate:
                                capture_live_pose = refined_pose
                                capture_live_meta = {**refined_meta, "source": "turn_arc_refined"}
                                refined_ok = True
                        if not refined_ok:
                            print(
                                "[wander] turn arc solve below gate even after xy-refine "
                                f"(capture={capture_index}, arc_score={arc_score:.3f}, "
                                f"min={float(args.min_append_score):.2f})"
                            )
                    else:
                        print(
                            "[wander] turn arc solve below gate "
                            f"(capture={capture_index}, score={arc_score:.3f}, min={arc_accept_gate:.2f})"
                        )
                else:
                    print(f"[wander] turn arc solve unavailable (capture={capture_index})")
            else:
                capture_pose_result = _estimate_pose_against_stitched_map(
                    points_xy=captured_snapshot.points_xy,
                    transformed_sets=stitch_state["transformed_sets"],
                    initial_pose=capture_expected_pose,
                    resolution_m=float(args.stitch_resolution_m),
                    search_xy_m=max(float(motion_hint.search_xy_m), 0.45),
                    theta_window_deg=max(float(motion_hint.search_theta_window_deg), 24.0),
                    # The pose is burst-tracked to ~5cm: the solve REFINES the
                    # tracked expectation, it does not search for it. A loose
                    # bound plus a token prior let captures slide ~0.5m along
                    # featureless straight walls at full score (phantom
                    # duplicate-wall lines in the map).
                    max_translation_from_initial_m=max(0.45, float(motion_hint.search_xy_m) + 0.10),
                    prior_translation_weight=2.5,
                    prior_theta_weight=0.12,
                )
                if capture_pose_result is not None:
                    candidate_capture_pose, candidate_capture_meta = capture_pose_result
                    if _accept_relocalized_pose(
                        label=f"capture_{capture_index}",
                        candidate_pose=candidate_capture_pose,
                        score_meta=candidate_capture_meta,
                        expected_pose=capture_expected_pose,
                        motion_hint=motion_hint,
                        max_translation_error_m=max(0.30, 0.65 * float(motion_hint.search_xy_m)),
                        max_theta_error_deg=max(18.0, float(motion_hint.search_theta_window_deg) * 1.10),
                        min_score=POSE_LOST_APPEND_GATE if pose_lost else 7.5,
                    ):
                        capture_live_pose = candidate_capture_pose
                        capture_live_meta = candidate_capture_meta
                    else:
                        print(
                            "[wander] capture relocalization fallback "
                            f"(capture={capture_index}, using strict append solver around prior stitched map)"
                        )
                if capture_pose_result is None:
                    print(f"[wander] capture relocalization unavailable (capture={capture_index})")

            if capture_live_pose is None and motion_hint.kind != "turn":
                # Run the strict append solver here (instead of inside _append_stitch)
                # so its score can be gated before the capture is committed to the map.
                is_turn_like_hint = motion_hint.kind in {"turn", "bootstrap_turn"}
                append_global_points_xy = np.concatenate(
                    [points for points in stitch_state["transformed_sets"] if len(points)], axis=0
                )
                append_prior_pose, append_prior_tw, append_prior_thw = _prior_weights_for_hint(
                    capture_expected_pose, motion_hint
                )
                append_solver_pose, append_solver_meta = _search_pose(
                    snapshot_points_xy=captured_snapshot.points_xy,
                    global_points_xy=append_global_points_xy,
                    initial_pose=capture_expected_pose,
                    resolution_m=float(args.stitch_resolution_m),
                    search_xy_m=float(motion_hint.search_xy_m),
                    coarse_angle_step_deg=4.0,
                    fine_angle_step_deg=0.5,
                    theta_window_deg=float(motion_hint.search_theta_window_deg),
                    whole_map_theta_center_deg=float(capture_expected_pose.theta_deg),
                    whole_map_theta_window_deg=(
                        float(max(motion_hint.search_theta_window_deg, 36.0))
                        if is_turn_like_hint
                        else float(max(motion_hint.search_theta_window_deg, 45.0))
                    ),
                    # Motion is lidar-tracked burst-by-burst now, so even the
                    # fallback search stays bounded — an unbounded whole-map
                    # drive search is how phantom room-sized jumps got in.
                    max_translation_from_initial_m=(
                        float(max(motion_hint.search_xy_m + 0.20, 0.55))
                        if is_turn_like_hint
                        else float(max(motion_hint.search_xy_m + 0.35, 0.90))
                    ),
                    prior_pose=append_prior_pose,
                    prior_translation_weight=float(append_prior_tw),
                    prior_theta_weight=float(append_prior_thw),
                )
                append_solver_meta = {
                    **append_solver_meta,
                    "source": f"append_{append_solver_meta.get('source', 'unknown')}",
                }
                append_solver_score = float(append_solver_meta.get("score") or -1e9)
                append_translation_err_m, append_theta_err_deg = _pose_delta_metrics(
                    capture_expected_pose, append_solver_pose
                )
                append_gate = float(args.min_append_score)
                if consecutive_append_discards > 0:
                    # The PREVIOUS capture was discarded: the pose expectation
                    # this solve is anchored to is already suspect, which is
                    # exactly when a barely-above-gate score is most likely a
                    # wrong mode (square-room symmetry). Field 2026-07-11: a
                    # 6.08-score append after a discard chain stitched a
                    # visibly rotated scan into the map. Demand a STRONG match
                    # or discard again and re-anchor.
                    append_gate = max(append_gate, float(args.min_append_score) + 2.0)
                elif append_theta_err_deg > 20.0:
                    # Solver heading disagrees hard with burst-by-burst lidar
                    # tracking, which rarely errs by 20deg+. A barely-above-
                    # gate score at a rotated mode is the ghost-room
                    # signature (field 2026-07-17: 6.92 at 30deg-off stitched
                    # a rotated copy of the room INSIDE the map and walled
                    # off both the table and the exit). Strong match or
                    # discard.
                    append_gate = max(append_gate, float(args.min_append_score) + 2.0)
                elif append_translation_err_m <= 0.30 and append_theta_err_deg <= 10.0:
                    # Solver landed where burst-by-burst tracking said we are;
                    # mutual confirmation earns a relaxed absolute gate. (Not
                    # after a discard — a shaky expectation confirms nothing.)
                    append_gate = min(append_gate, 3.5)
                if post_recovery_strict_appends > 0:
                    append_gate = max(append_gate, float(args.min_append_score) + 2.0)
                if pose_lost:
                    # Map is read-only while lost: absolute-trust matches only.
                    append_gate = max(append_gate, POSE_LOST_APPEND_GATE)
                    print(
                        f"[wander] pose lost: append requires score>={append_gate:.1f}"
                    )
                # The fallback used to gate on SCORE alone. Near a featureless
                # wall a slid pose scores as well as the true one, so a capture
                # 0.64m from its fully-locked tracked expectation got in and
                # painted a ghost wall. Motion is lidar-tracked: bound the
                # fallback by the expectation just like the strict solve.
                append_translation_gate_m = max(0.40, 0.75 * float(motion_hint.search_xy_m))
                append_theta_gate_deg = max(18.0, float(motion_hint.search_theta_window_deg) * 1.10)
                append_pose_ok = (
                    append_translation_err_m <= append_translation_gate_m
                    and append_theta_err_deg <= append_theta_gate_deg
                )
                if append_solver_score >= append_gate and append_pose_ok:
                    capture_live_pose = append_solver_pose
                    capture_live_meta = append_solver_meta
                else:
                    print(
                        "[wander] append solver rejected "
                        f"(capture={capture_index}, score={append_solver_score:.3f}, min={append_gate:.2f}, "
                        f"translation_error={append_translation_err_m:.3f}m<= {append_translation_gate_m:.2f}m, "
                        f"theta_error={append_theta_err_deg:.1f}deg<= {append_theta_gate_deg:.0f}deg); "
                        "attempting wide-theta rescue relocalization"
                    )
                    # Rescue exists for one failure mode: the HEADING went blind
                    # (lock-lost turn bursts), so theta may be far off while the
                    # position stays bounded by tracked driving — lock loss stops
                    # all translation. Search wide in theta, tight in xy. The old
                    # flat 1.40m/85deg gates predate tracked motion and accepted
                    # a pose 0.97m/43deg out, skewing the whole map.
                    rescue_result = _estimate_pose_against_stitched_map(
                        points_xy=captured_snapshot.points_xy,
                        transformed_sets=stitch_state["transformed_sets"],
                        initial_pose=capture_expected_pose,
                        resolution_m=float(args.stitch_resolution_m),
                        search_xy_m=append_translation_gate_m + 0.10,
                        theta_window_deg=80.0,
                        max_translation_from_initial_m=append_translation_gate_m,
                        prior_translation_weight=0.05,
                        prior_theta_weight=0.02,
                    )
                    if rescue_result is not None:
                        rescue_pose, rescue_meta = rescue_result
                        if _accept_relocalized_pose(
                            label=f"capture_{capture_index}_rescue",
                            candidate_pose=rescue_pose,
                            score_meta=rescue_meta,
                            expected_pose=capture_expected_pose,
                            motion_hint=motion_hint,
                            max_translation_error_m=append_translation_gate_m,
                            max_theta_error_deg=max(
                                35.0, float(motion_hint.search_theta_window_deg) + 10.0
                            ),
                            min_score=POSE_LOST_APPEND_GATE if pose_lost else 7.5,
                        ):
                            capture_live_pose = rescue_pose
                            capture_live_meta = {**rescue_meta, "source": f"rescue_{rescue_meta.get('source', 'unknown')}"}

            if capture_live_pose is None:
                consecutive_append_discards += 1
                if not pose_lost:
                    pose_lost = True
                    print(
                        "[wander] pose integrity LOST (capture solve discarded); "
                        "map is READ-ONLY until a strong relocalization re-proves the pose"
                    )
                if consecutive_append_discards <= max(0, int(args.max_consecutive_append_discards)):
                    print(
                        "[wander] discarding capture to protect the map "
                        f"(capture={capture_index}, best_score="
                        f"{float((append_solver_meta or {}).get('score') or -1e9):.3f}, "
                        f"consecutive_discards={consecutive_append_discards}); "
                        "its expected motion will carry into the next capture"
                    )
                    motion_hints.pop()
                    pending_motion_hint = motion_hint
                    last_frame_id = int(frame_id)
                    last_captured_frame = captured_frame
                    continue
                # Discard limit reached. NEVER force-accept into the map —
                # a below-gate pose stitches the room's own walls back in
                # rotated (field 2026-07-17: a force-accepted 6.63 painted a
                # ghost room INSIDE the room, walling off the table and the
                # exit). The MAP is sacred; the POSE is recoverable: drop
                # this capture AND the accumulated motion chain (the chain
                # is precisely what is untrustworthy after repeated solve
                # failures) and let live relocalization — which keeps
                # scoring 12-16 against the trusted map during these
                # episodes — re-anchor on the next planning cycle.
                force_accept_score = float((append_solver_meta or {}).get("score") or -1e9)
                print(
                    "[wander] discard limit reached; dropping the capture AND its "
                    f"motion chain instead of force-accepting (capture={capture_index}, "
                    f"score={force_accept_score:.3f}); live relocalization re-anchors"
                )
                consecutive_append_discards = 0
                motion_hints.pop()
                pending_motion_hint = None
                last_frame_id = int(frame_id)
                last_captured_frame = captured_frame
                continue
            consecutive_append_discards = 0
            orbit_recovery_turn_attempts = 0
            orbit_recovery_turns_remaining = 0
            orbit_recovery_direction_sign = None
            dead_reck_theta_deg = float(capture_live_pose.theta_deg)
            dead_reck_slack_deg = 15.0
            imu_dead_reck_anchor = _imu_anchor(imu_yaw, dead_reck_theta_deg)
            if pose_lost:
                # This append cleared the absolute-trust lost-gate (>=12.5):
                # the pose is re-proven by construction.
                pose_lost = False
                post_recovery_strict_appends = 2
                lost_recovery_failures = 0
                print(
                    "[wander] pose integrity RESTORED (append cleared the absolute-trust gate); "
                    "map writes re-enabled (strict gates for the next 2 appends)"
                )
            elif post_recovery_strict_appends > 0:
                post_recovery_strict_appends -= 1

            stitch_state = _append_stitch(
                stitch_dir=stitch_dir,
                snapshot_dir=snapshot_dir,
                stitch_state=stitch_state,
                motion_hints=motion_hints,
                new_snapshot=captured_snapshot,
                resolution_m=float(args.stitch_resolution_m),
                solved_pose_override=capture_live_pose,
                solve_meta_override=capture_live_meta,
            )
            rebuild_elapsed = time.monotonic() - rebuild_started
            _log_rerun_state(
                rr,
                capture_index=capture_index,
                transformed_sets=stitch_state["transformed_sets"],
                poses=stitch_state["poses"],
                solve_log=stitch_state["solve_log"],
                cone_half_width_deg=float(args.valid_angle_half_width_deg),
                body_radius_m=float(args.robot_radius_m),
            )
            if hazard_monitor is not None:
                # Eye-camera panels + safety decision alongside the map, so
                # the user sees what the robot sees while it wanders.
                hazard_state_now = hazard_monitor.state()
                for safety_cam in ("front_left", "front_right", "panorama", "bottom"):
                    safety_frame = hazard_monitor.annotated(safety_cam)
                    if safety_frame is not None:
                        rr.log(f"cameras/{safety_cam}", rr.Image(safety_frame[:, :, ::-1]))
                rr.log(
                    "safety/decision",
                    rr.TextLog(
                        f"{hazard_state_now.decision_label()} side={hazard_state_now.side} "
                        f"confidence={hazard_state_now.confidence:.2f}"
                    ),
                )
            capture_index += 1
            _log_live_pose_state(
                rr,
                capture_index=capture_index,
                pose=stitch_state["poses"][-1],
                points_xy=captured_snapshot.points_xy,
                cone_half_width_deg=float(args.valid_angle_half_width_deg),
                body_radius_m=float(args.robot_radius_m),
            )
            last_frame_id = int(frame_id)
            last_captured_frame = captured_frame

            final_pose = stitch_state["poses"][-1]
            previous_pose = stitch_state["poses"][-2] if len(stitch_state["poses"]) >= 2 else None
            solved_dx = 0.0 if previous_pose is None else float(final_pose.x - previous_pose.x)
            solved_dy = 0.0 if previous_pose is None else float(final_pose.y - previous_pose.y)
            solved_dtheta = 0.0 if previous_pose is None else _normalize_angle_deg(float(final_pose.theta_deg - previous_pose.theta_deg))
            solve_meta = stitch_state["solve_log"][-1] if stitch_state["solve_log"] else {}
            search_timing = solve_meta.get("timing_s") if isinstance(solve_meta, dict) else None
            append_timing = stitch_state.get("timing", {})
            print(
                "[wander] stitched capture complete "
                f"(capture={capture_index}, rebuild={rebuild_elapsed:.2f}s, pose=({final_pose.x:.3f}, {final_pose.y:.3f}, {final_pose.theta_deg:.1f}deg), "
                f"html={stitch_state['html_path']})"
            )
            if previous_pose is not None:
                print(
                    "[wander] solved motion delta "
                    f"(capture={capture_index}, dx={solved_dx:.3f}m, dy={solved_dy:.3f}m, dtheta={solved_dtheta:.1f}deg, "
                    f"source={solve_meta.get('solve_source')}, score={solve_meta.get('score')})"
                )
            if isinstance(search_timing, dict):
                print(
                    "[wander] stitch timing "
                    f"(capture={capture_index}, build={float(search_timing.get('build_occupancy', 0.0)):.2f}s, "
                    f"local={float(search_timing.get('local_search', 0.0)):.2f}s, "
                    f"whole_map={float(search_timing.get('whole_map_search', 0.0)):.2f}s, "
                    f"search_total={float(search_timing.get('total_search', 0.0)):.2f}s, "
                    f"write={float(append_timing.get('write_phase_s', 0.0)):.2f}s)"
                )
            if active_explore_target is not None:
                target_distance_after_capture_m = _target_distance_m(active_explore_target, final_pose)
                previous_target_distance_m = active_explore_target_last_distance_m
                if previous_target_distance_m is not None:
                    target_progress_m = float(previous_target_distance_m) - float(target_distance_after_capture_m)
                    if target_progress_m < 0.08:
                        active_explore_target_stall_count += 1
                        print(
                            "[wander] stitched-map target progress stalled "
                            f"(capture={capture_index}, progress={target_progress_m:.3f}m, "
                            f"distance={target_distance_after_capture_m:.3f}m, "
                            f"stall_count={active_explore_target_stall_count})"
                        )
                    else:
                        active_explore_target_stall_count = 0
                        active_explore_target_blocked_count = 0
                        force_live_frontier_cycles = 0
                        print(
                            "[wander] stitched-map target progress improved "
                            f"(capture={capture_index}, progress={target_progress_m:.3f}m, "
                            f"distance={target_distance_after_capture_m:.3f}m)"
                        )
                active_explore_target_last_distance_m = float(target_distance_after_capture_m)
                if active_explore_target_stall_count >= 2:
                    print(
                        "[wander] dropping stitched-map target after repeated low-progress captures "
                        f"(distance={target_distance_after_capture_m:.3f}m, "
                        f"source={active_explore_target.source}, "
                        f"seed_capture={active_explore_target.seeded_capture_index})"
                    )
                    if active_explore_target.source != "retreat":
                        _blacklist_target(active_explore_target, "stalled_progress")
                    active_explore_target = None
                    active_explore_target_stall_count = 0
                    active_explore_target_last_distance_m = None
                    active_explore_target_blocked_count = 0
                    force_live_frontier_cycles = max(force_live_frontier_cycles, 2)
            if motion_hint.kind == "turn":
                consecutive_turn_captures += 1
                solved_heading_bin = _heading_bin_index(
                    float(final_pose.theta_deg),
                    bin_count=rotation_coverage_bin_count,
                )
                previous_bin_count = len(rotation_coverage_bins_seen)
                rotation_coverage_bins_seen.add(solved_heading_bin)
                if len(rotation_coverage_bins_seen) != previous_bin_count or capture_index <= 4:
                    print(
                        "[wander] rotation coverage update "
                        f"(capture={capture_index}, heading={float(final_pose.theta_deg):.1f}deg, "
                        f"bins={sorted(rotation_coverage_bins_seen)}/{rotation_coverage_bin_count})"
                    )
                if not rotation_coverage_complete and len(rotation_coverage_bins_seen) >= rotation_coverage_bin_count:
                    rotation_coverage_complete = True
                    force_drive_after_turn = True
                    drive_checkpoints_since_scan = 0
                    print(
                        "[wander] rotational coverage complete; forcing forward exploration "
                        f"(capture={capture_index}, bins={sorted(rotation_coverage_bins_seen)})"
                    )
            else:
                consecutive_turn_captures = 0

        completed_captures = capture_index - 1 if "capture_index" in locals() else 1
        print(f"[wander] complete. captures={completed_captures} viewer={viewer_url or 'local rerun'}")
        print(f"[wander] stitched html: {stitch_state['html_path']}")
        print(f"[wander] stitched report: {stitch_state['report_path']}")
        return 0
    finally:
        try:
            _send_stop(robot)
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        feed.stop()
        if imu_yaw is not None:
            imu_yaw.stop()
        if hazard_monitor is not None:
            hazard_monitor.stop()
        if hazard_subscriber is not None:
            hazard_subscriber.stop()


if __name__ == "__main__":
    raise SystemExit(main())
