"""Three-sensor anti-collision layer for the Sourccey base.

Height ladder — each sensor owns the band the others cannot see:
  - bottom camera (0.127 m, level):   floor band 0 - 0.28 m (dumbbells, cables)
  - 2D lidar     (0.28 m scan plane): legs, walls, furniture bodies
  - eye cameras  (0.914 m, 20 deg down): floating overhangs (table tops,
    counters, shelves) that neither of the above can see

The eye-camera edge detector is only a CANDIDATE GENERATOR. Every candidate
is interrogated with projective geometry before it may become a hazard:

  1. Compute the candidate's bearing and its distance IF it lay on the floor.
  2. Lidar shows a return there              -> tall floor-standing object;
                                                the lidar stop box owns it.
  3. Bottom camera shows a floor base there  -> low floor object; the bottom
                                                camera's own ground gate owns it.
  4. Neither sensor explains it              -> genuinely floating: an
     ELEVATED hazard. Motion parallax (image row vs lidar/velocity-tracked
     forward travel) then solves its true height and distance, which drive
     the two-tier gate: forward blocked in range, rotation also frozen when
     near. Reverse is ALWAYS allowed.

A blind-zone vanish latch backstops the close range where the edge detector
goes blind: a tracked near candidate that disappears WITHOUT receding
latches a stop until the edge is re-seen farther or the base measurably
backs away (integrated from the stream's base velocity).

This module must never command the arms.
"""

from __future__ import annotations

import base64
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from lerobot.control.sourccey.sourccey.elevated_edge_scan_live import (
    EdgeObservation,
    ElevatedEdgeDetector,
    ElevatedEdgeScanConfig,
)
from sourccey_camera_geometry import (
    CameraModel,
    default_bottom,
    default_eye_left,
    default_eye_right,
    edge_point_robot_frame,
    edge_segment_hazard_distances,
    solve_edge_by_parallax,
)


@dataclass(frozen=True)
class ConfirmedEdge:
    """A parallax-confirmed elevated edge, measured in the ROBOT frame:
    'this edge is X away, this wide, this tall'. Consumers (the wander
    mapper) transform it into world coordinates with the tracked pose and
    stamp it into the planning map as untouchable geometry."""

    eye: str
    p1_robot_xy: tuple[float, float]  # (forward_m, lateral_m)
    p2_robot_xy: tuple[float, float]
    height_m: float
    nearest_m: float
    monotonic: float
    # Depth mode: floor-projected footprint of the whole elevated region
    # (robot-frame points). When present the mapper stamps THESE like lidar
    # points instead of the p1/p2 line — the region's true shape and
    # placement, no fitted-line abstraction.
    points_robot_xy: tuple[tuple[float, float], ...] = ()


@dataclass
class ElevatedSafetyConfig:
    """All anti-collision tunables in one place."""

    slam_input_endpoint: str = "tcp://192.168.1.237:5560"
    left_key: str = "front_left"
    right_key: str = "front_right"
    bottom_key: str = "bottom"
    monitor_rate_hz: float = 10.0
    # Hysteresis: consecutive qualifying ticks to trip, clean ticks to clear.
    trip_frames: int = 3
    clear_frames: int = 5
    # Two-tier gate on the GEOMETRIC distance to a confirmed elevated edge
    # (camera-frame meters).
    forward_block_distance_m: float = 0.55
    near_freeze_distance_m: float = 0.35
    near_release_margin_m: float = 0.15
    # The forward gate intersects the detected edge SEGMENT with the corridor
    # the robot's body sweeps: exact for diagonal approaches (the near END of
    # the edge governs, not its center), and edges fully outside the corridor
    # no longer false-stop drive-bys. Half-width = robot half-width + margin.
    elevated_corridor_half_width_m: float = 0.40
    # Frames older than this while armed = driving blind: deny forward.
    stale_timeout_s: float = 1.5
    # Depth-perception results older than this fall back to the classic
    # detector for the tick (CPU inference runs ~1-2s/eye; the gate
    # compensates for the robot's travel since the analyzed frame).
    depth_stale_timeout_s: float = 3.5
    # Eye detector ROI band and score gates (candidate generation only).
    roi_min_y_ratio: float = 0.46
    roi_max_y_ratio: float = 0.95
    min_line_score: float = 860.0
    single_eye_min_line_score: float = 1040.0
    min_shadow_contrast: float = 16.0
    # Legacy detector distance-model endpoints — kept ONLY because the
    # underlying detector computes an internal estimate; gating no longer
    # uses it (geometry does).
    near_distance_m: float = 0.20
    far_distance_m: float = 1.55
    pair_center_y_tolerance_ratio: float = 0.10
    pair_distance_tolerance_m: float = 0.40
    # ---- geometry / classification -----------------------------------------
    eye_left_model: CameraModel = field(default_factory=default_eye_left)
    eye_right_model: CameraModel = field(default_factory=default_eye_right)
    bottom_model: CameraModel = field(default_factory=default_bottom)
    # Referee tolerances: a candidate is "explained" as a floor object when a
    # lidar return / bottom-camera base sits this close to its if-on-floor
    # position.
    lidar_bearing_tolerance_deg: float = 7.0
    lidar_distance_tolerance_m: float = 0.40
    bottom_bearing_tolerance_deg: float = 9.0
    bottom_distance_tolerance_m: float = 0.40
    # Parallax ranging of unexplained (elevated) candidates.
    parallax_min_travel_m: float = 0.15
    parallax_assoc_bearing_deg: float = 12.0
    parallax_floor_height_max_m: float = 0.28  # at/below the lidar plane = floor-ish
    parallax_max_edge_height_m: float = 1.05  # above this cannot hit the base
    # The edge detector flickers (score 1400 <-> 0 on curved/cluttered
    # edges); the parallax tracker must survive short gaps or its baseline
    # resets every blink and the edge never gets confirmed (field bug).
    parallax_miss_tolerance_ticks: int = 5
    # Distance assumed for an elevated candidate BEFORE parallax resolves it
    # (mid-range table height). Used only for display/vanish arming, not for
    # tripping the gate.
    assumed_edge_height_m: float = 0.60
    # ---- bottom-camera ground gate -------------------------------------------
    ground_gate_enabled: bool = True
    ground_stop_distance_m: float = 0.55
    ground_corridor_half_width_m: float = 0.35
    ground_trip_frames: int = 3
    ground_clear_frames: int = 5
    ground_column_bands: int = 12
    ground_edge_density: float = 0.30  # fraction of band columns that must be edges
    ground_max_distance_m: float = 2.5
    # Ignore floor features shorter than this: cables, carpet seams, and
    # thresholds are traversable and must not stop the base (field run
    # 2026-07-09: a floor cable ground-stopped the robot a foot early).
    # NOTE: the row-height estimate is confounded by depth-extended floor
    # features (cable loops measured 5-10cm), so the GATE additionally
    # requires the silhouette to cross the camera-height horizon — i.e. only
    # objects taller than the bottom camera (~13cm) forcibly stop the base.
    # Shorter obstacles are still reported for the eye-candidate referee.
    ground_min_obstacle_height_m: float = 0.04
    # ---- blind-zone vanish latch ----------------------------------------------
    # Arm when an unexplained candidate sits deep in the frame; latch when it
    # disappears without receding. Release by re-sighting higher in the frame
    # or by measured reverse travel.
    vanish_arm_row_ratio: float = 0.60
    vanish_recede_row_ratio: float = 0.55
    vanish_arm_frames: int = 3
    vanish_miss_frames: int = 3
    vanish_release_reverse_m: float = 0.50
    # Rotating well away also releases the latch: the blind zone sits under
    # the robot's NOSE, and after this much tracked rotation the nose points
    # somewhere the cameras can actually see (field deadlock 2026-07-11: a
    # robot wedged in a pocket could neither reverse — furniture behind — nor
    # go forward, because release demanded the one motion the map forbade).
    vanish_release_rotation_deg: float = 80.0
    # ---- post-stop HOLD (anti-ratchet) ------------------------------------------
    # Field failure 2026-07-09: every gate flickers, and instant resume after
    # each flicker ratcheted the robot INTO the table stop by stop. Once ANY
    # forcible stop occurs, forward stays denied until the robot measurably
    # BACKS AWAY or everything stays continuously clear for seconds (not
    # ticks). Repeated stops without retreat escalate to a permanent hold
    # that ONLY reversing releases.
    hold_release_reverse_m: float = 0.30
    hold_release_clear_s: float = 3.0
    hold_escalate_count: int = 3
    hold_escalate_window_s: float = 30.0
    # Reverse is the preferred release, but a wedged robot (furniture behind)
    # may not have it. Tracked rotation past this angle points "forward" away
    # from the stopped-at hazard, so forward motion becomes retreat — release
    # the hold and let the live gates own the new heading.
    hold_release_rotation_deg: float = 80.0

    def detector_config(self) -> ElevatedEdgeScanConfig:
        return ElevatedEdgeScanConfig(
            left_key=self.left_key,
            right_key=self.right_key,
            min_line_score=float(self.min_line_score),
            single_eye_min_line_score=float(self.single_eye_min_line_score),
            min_shadow_contrast=float(self.min_shadow_contrast),
            min_center_y_ratio=float(self.roi_min_y_ratio),
            max_center_y_ratio=float(self.roi_max_y_ratio),
            near_distance_m=float(self.near_distance_m),
            far_distance_m=float(self.far_distance_m),
            pair_center_y_tolerance_ratio=float(self.pair_center_y_tolerance_ratio),
            pair_distance_tolerance_m=float(self.pair_distance_tolerance_m),
        )


@dataclass(frozen=True)
class HazardState:
    """Thread-safe snapshot of the anti-collision decision."""

    enabled: bool = False
    active: bool = False  # elevated hazard (confirmed by geometry)
    near_freeze: bool = False
    side: str = "none"  # left | right | front | none
    confidence: float = 0.0
    # line_edge | elevated_parallax | vanished_near | ground_obstacle |
    # stale_frame | clear | disabled
    reason: str = "clear"
    est_distance_m: float | None = None
    edge_height_m: float | None = None
    frames_stale: bool = False
    blind_zone: bool = False
    # Bottom-camera ground gate (floor band the lidar cannot see).
    ground_active: bool = False
    ground_distance_m: float | None = None
    ground_side: str = "none"
    bottom_available: bool = False
    # Classification of the strongest current eye candidate (diagnostics):
    # none | floor_lidar | floor_low | floor_parallax | elevated_unresolved |
    # elevated_confirmed
    classification: str = "none"
    # Post-stop hold: forward stays denied after ANY forcible stop until the
    # base measurably reverses or everything stays clear for a sustained
    # period. permanent_hold releases ONLY by reversing.
    hold: bool = False
    hold_reason: str = ""
    permanent_hold: bool = False
    # Per-eye diagnostics: "conf g=0.42 n=0.38 h=0.51" style.
    detail: str = ""
    left_score: float = 0.0
    right_score: float = 0.0
    frame_age_s: float | None = None
    updated_monotonic: float = 0.0

    def decision_label(self) -> str:
        if not self.enabled:
            return "disabled"
        if self.frames_stale:
            return "camera_stale_stop"
        if self.blind_zone:
            return f"elevated_stop_{self.side}" if self.side != "none" else "elevated_stop_front"
        if self.active:
            return f"elevated_stop_{self.side}" if self.side != "none" else "elevated_stop_front"
        if self.ground_active:
            return f"ground_stop_{self.ground_side}" if self.ground_side != "none" else "ground_stop_front"
        return "clear"


def gate_forward_allowed(state: HazardState) -> tuple[bool, str]:
    """Forward motion (x.vel > 0) denied for any active hazard or stale feed."""
    if not state.enabled:
        return True, "gate_disabled"
    if state.frames_stale:
        return False, "camera_stale_stop"
    if state.active or state.blind_zone or state.ground_active:
        return False, state.decision_label()
    if state.hold:
        return False, f"post_stop_hold({state.hold_reason})"
    return True, "clear"


def gate_turn_allowed(state: HazardState) -> tuple[bool, str]:
    """Rotation denied only in the near tier / blind-zone latch."""
    if not state.enabled:
        return True, "gate_disabled"
    if (state.active or state.blind_zone) and state.near_freeze:
        return False, f"{state.decision_label()}_near_freeze"
    return True, "clear"


class SlamCameraSubscriber:
    """Subscribes to the host `slam_input.v1` stream: decodes camera JPEG
    frames (eyes + bottom when the host publishes it) and tracks the base
    velocity for retreat/parallax integration."""

    def __init__(
        self,
        *,
        endpoint: str,
        camera_keys: tuple[str, ...] = ("front_left", "front_right", "bottom"),
    ) -> None:
        self._endpoint = str(endpoint)
        self._camera_keys = tuple(camera_keys)
        self._lock = threading.Lock()
        self._frames: dict[str, np.ndarray] = {}
        self._frame_received_monotonic: dict[str, float] = {}
        self._base_x_vel: float | None = None
        self._base_vel_monotonic: float | None = None
        self._packets_received = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="slam-camera-subscriber", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        import zmq

        context = zmq.Context.instance()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.LINGER, 0)
        # A safety layer must never fall behind a queue of stale frames.
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.RCVTIMEO, 500)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.connect(self._endpoint)
        try:
            while not self._stop_event.is_set():
                try:
                    payload = socket.recv()
                except zmq.Again:
                    continue
                except zmq.ZMQError:
                    break
                self._ingest(payload)
        finally:
            socket.close(0)

    def _ingest(self, payload: bytes) -> None:
        try:
            packet = json.loads(payload)
        except Exception:
            return
        if not isinstance(packet, dict) or packet.get("schema") != "slam_input.v1":
            return
        base_velocity = packet.get("base_velocity")
        if isinstance(base_velocity, dict):
            try:
                x_vel = float(base_velocity.get("x.vel", 0.0))
                with self._lock:
                    self._base_x_vel = x_vel
                    self._base_vel_monotonic = time.monotonic()
            except Exception:
                pass
        cameras = packet.get("cameras")
        if not isinstance(cameras, dict):
            return
        now = time.monotonic()
        for cam_name in self._camera_keys:
            entry = cameras.get(cam_name)
            if not isinstance(entry, dict):
                continue
            jpeg_b64 = entry.get("jpeg_b64")
            if not isinstance(jpeg_b64, str) or not jpeg_b64:
                continue
            try:
                buffer = np.frombuffer(base64.b64decode(jpeg_b64), dtype=np.uint8)
                frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            except Exception:
                continue
            if frame is None:
                continue
            self._store_frame(cam_name, frame, received_monotonic=now)
        with self._lock:
            self._packets_received += 1

    def _store_frame(
        self, cam_name: str, frame: np.ndarray, *, received_monotonic: float | None = None
    ) -> None:
        stamp = time.monotonic() if received_monotonic is None else float(received_monotonic)
        with self._lock:
            self._frames[cam_name] = frame
            self._frame_received_monotonic[cam_name] = stamp

    def latest(self, cam_name: str) -> tuple[np.ndarray | None, float | None]:
        with self._lock:
            frame = self._frames.get(cam_name)
            stamp = self._frame_received_monotonic.get(cam_name)
        if frame is None or stamp is None:
            return None, None
        return frame, max(0.0, time.monotonic() - stamp)

    def latest_base_velocity(self) -> tuple[float | None, float | None]:
        with self._lock:
            x_vel = self._base_x_vel
            stamp = self._base_vel_monotonic
        if x_vel is None or stamp is None:
            return None, None
        return float(x_vel), max(0.0, time.monotonic() - stamp)

    @property
    def packets_received(self) -> int:
        with self._lock:
            return self._packets_received

    def wait_for_frames(self, timeout_s: float, required: tuple[str, ...] | None = None) -> bool:
        cams = tuple(required) if required is not None else self._camera_keys
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            if all(self.latest(cam)[0] is not None for cam in cams):
                return True
            time.sleep(0.1)
        return all(self.latest(cam)[0] is not None for cam in cams)


@dataclass
class FloorObstacle:
    bearing_deg: float
    distance_m: float
    x_ratio: float
    y_ratio: float
    height_m: float = 0.0
    # True when the contiguous silhouette crosses the camera-height horizon
    # row. For a level camera NOTHING lying on the floor can appear above the
    # horizon, so this is a depth-extent-proof "taller than the camera"
    # classifier — the row-height estimate is confounded by floor features
    # that extend away in depth (field bug: cable loops measured 5-10cm).
    crosses_horizon: bool = False


class BottomFloorDetector:
    """Finds floor-obstacle bases in the level bottom camera.

    Everything on the floor appears BELOW the camera's horizon row (the
    5-inch height plane); the image row of an object's floor contact gives
    its exact distance. Splits the frame into column bands and reports the
    lowest strong edge row per band as a base candidate."""

    def __init__(self, model: CameraModel, config: ElevatedSafetyConfig) -> None:
        self.model = model
        self.config = config

    def detect(self, frame_bgr: np.ndarray) -> list[FloorObstacle]:
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[0] < 8 or frame.shape[1] < 8:
            return []
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 130)
        horizon = float(np.clip(self.model.horizon_y_ratio(), 0.05, 0.9))
        y_start = int(height * min(horizon + 0.04, 0.95))
        if y_start >= height - 2:
            return []
        obstacles: list[FloorObstacle] = []
        bands = max(int(self.config.ground_column_bands), 4)
        band_w = max(width // bands, 4)
        min_edges = max(int(band_w * float(self.config.ground_edge_density)), 2)
        min_height_m = float(self.config.ground_min_obstacle_height_m)
        for band in range(bands):
            x0 = band * band_w
            x1 = min(x0 + band_w, width)
            column = edges[y_start:height, x0:x1]
            row_counts = (column > 0).sum(axis=1)
            strong_rows = np.nonzero(row_counts >= min_edges)[0]
            if len(strong_rows) == 0:
                continue
            base_row = y_start + int(strong_rows[-1])  # lowest strong edge = nearest base
            y_ratio = base_row / max(height - 1, 1)
            distance = self.model.floor_distance_for_row(y_ratio)
            if distance is None or distance > float(self.config.ground_max_distance_m):
                continue
            # Object height from its silhouette top: cables/carpet seams are a
            # few pixels tall and must not stop the base; dumbbells, table
            # legs, and walls tower above their base row. The silhouette must
            # be CONTIGUOUS with the base (small gaps only) — scattered
            # carpet-texture edges above a cable are not part of the object
            # (field bug: cables measured 0.10m tall from carpet noise).
            base_index = int(strong_rows[-1])
            top_index = base_index
            gap = 0
            scan = base_index - 1
            while scan >= 0 and gap <= 4:
                # A vertical silhouette edge is ~1px wide after Canny, so a
                # single edge pixel keeps the run alive; the gap limit is what
                # rejects scattered floor-texture noise.
                if row_counts[scan] >= 1:
                    top_index = scan
                    gap = 0
                else:
                    gap += 1
                scan -= 1
            top_row = y_start + top_index
            top_ratio = top_row / max(height - 1, 1)
            top_depression_deg = self.model.depression_deg_for_row(top_ratio)
            obstacle_height_m = float(self.model.height_m) - float(distance) * math.tan(
                math.radians(top_depression_deg)
            )
            if obstacle_height_m < min_height_m:
                continue
            # The scan region starts just below the horizon, so a contiguous
            # run reaching the very top of it means the silhouette continues
            # ACROSS the horizon: taller than the camera. Verify by checking
            # for edge content in the band just above the horizon row.
            crosses = False
            if top_index <= 1:
                above_y0 = max(0, int(horizon * height) - int(0.08 * height))
                above_band = edges[above_y0 : max(above_y0 + 1, y_start - 2), x0:x1]
                crosses = bool(above_band.size > 0 and (above_band > 0).any(axis=1).sum() >= 2)
            x_ratio = ((x0 + x1) * 0.5) / max(width - 1, 1)
            obstacles.append(
                FloorObstacle(
                    bearing_deg=float(self.model.bearing_deg_for_column(x_ratio)),
                    distance_m=float(distance),
                    x_ratio=float(x_ratio),
                    y_ratio=float(y_ratio),
                    height_m=float(obstacle_height_m),
                    crosses_horizon=crosses,
                )
            )
        return obstacles


@dataclass
class _EdgeTracker:
    """Parallax tracker for one eye's strongest unexplained candidate."""

    bearing_deg: float = 0.0
    first_y_ratio: float = 0.0
    travel_at_first_m: float = 0.0
    last_y_ratio: float = 0.0
    solved_height_m: float | None = None
    # Baseline the current height solve used. Parallax accuracy grows with
    # baseline, so the solve keeps REFINING as the robot approaches instead
    # of freezing at the first minimum-baseline estimate (field 2026-07-11:
    # a 0.15m first solve pinned a 0.74m table at 0.36m, throwing every
    # distance off ~2x and misplacing the mapped edge).
    solved_baseline_m: float = 0.0
    active: bool = False
    miss_ticks: int = 0

    def reset(self, *, bearing_deg: float, y_ratio: float, travel_m: float) -> None:
        self.bearing_deg = float(bearing_deg)
        self.first_y_ratio = float(y_ratio)
        self.travel_at_first_m = float(travel_m)
        self.last_y_ratio = float(y_ratio)
        self.solved_height_m = None
        self.solved_baseline_m = 0.0
        self.active = True
        self.miss_ticks = 0


class ElevatedHazardMonitor:
    """Classification pipeline + hysteresis + vanish latch + ground gate.

    `frame_source` needs `latest(cam) -> (frame|None, age_s|None)` and
    optionally `latest_base_velocity() -> (x_vel|None, age_s|None)`.
    `lidar_ranges_fn`, when given, returns (bearings_deg, distances_m) numpy
    arrays of the newest scan in the robot's local frame, or None.
    """

    def __init__(
        self, config: ElevatedSafetyConfig, frame_source, lidar_ranges_fn=None, depth_worker=None
    ) -> None:
        self.config = config
        self._source = frame_source
        self._lidar_ranges_fn = lidar_ranges_fn
        # Optional monocular-depth perception (sourccey_depth_perception.
        # DepthWorker). When it has FRESH results the eyes are classified from
        # metric 3D instead of the Hough detector; when absent/stale/failed,
        # the classic detector path below runs unchanged (fail-soft).
        self._depth_worker = depth_worker
        self._detector = ElevatedEdgeDetector(config.detector_config())
        self._bottom_detector = BottomFloorDetector(config.bottom_model, config)
        self._lock = threading.Lock()
        self._state = HazardState(enabled=True)
        self._annotated: dict[str, np.ndarray] = {}
        # Elevated-hazard hysteresis.
        self._hit_ticks = 0
        self._clean_ticks = 0
        # Ground-gate hysteresis.
        self._ground_hits = 0
        self._ground_clean = 0
        # Motion integration.
        self._last_tick_monotonic: float | None = None
        self._forward_travel_m = 0.0
        # (monotonic, cumulative forward travel) history so slow depth
        # results can be latency-compensated: distance NOW = distance at the
        # analyzed frame minus how far the robot has driven since.
        self._travel_log: deque[tuple[float, float]] = deque(maxlen=120)
        # Parallax trackers per eye.
        self._trackers = {config.left_key: _EdgeTracker(), config.right_key: _EdgeTracker()}
        # Blind-zone vanish latch.
        self._vanish_hits = 0
        self._vanish_misses = 0
        self._vanish_armed = False
        self._vanish_side = "front"
        self._vanish_row = 0.0
        self._blind_latched = False
        self._retreat_m = 0.0
        # Tracked rotation reported by the client since the last stop/latch
        # armed; releases holds/latches when reversing is unavailable.
        self._rotation_since_stop_deg = 0.0
        # When the client last finished rotating: depth results whose source
        # frame predates this were captured at a DIFFERENT heading and must
        # never be stamped into the map (travel compensation handles forward
        # motion, not rotation — field 2026-07-12: table segments stamped
        # rotated ~30deg because the depth frame was from mid-align-turn).
        self._last_rotation_monotonic = 0.0
        # Parallax-confirmed edge measurements awaiting the map consumer.
        self._confirmed_edge_queue: list[ConfirmedEdge] = []
        # Post-stop hold (anti-ratchet).
        self._hold_active = False
        self._hold_reason = ""
        self._hold_permanent = False
        self._hold_reverse_only = False
        self._hold_clear_since: float | None = None
        self._hold_retreat_m = 0.0
        self._hold_event_times: list[float] = []
        self._external_stop_reason: str | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="elevated-hazard-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        interval_s = 1.0 / max(float(self.config.monitor_rate_hz), 1.0)
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                self.tick()
            except Exception as exc:  # a crashed safety monitor fails LOUD and SAFE
                with self._lock:
                    self._state = replace(
                        self._state,
                        active=True,
                        frames_stale=True,
                        reason=f"monitor_error:{type(exc).__name__}",
                        updated_monotonic=time.monotonic(),
                    )
            self._stop_event.wait(max(0.0, interval_s - (time.monotonic() - started)))

    # -- state access ------------------------------------------------------------
    def state(self) -> HazardState:
        with self._lock:
            return self._state

    def forward_allowed(self) -> tuple[bool, str]:
        return gate_forward_allowed(self.state())

    def turn_allowed(self) -> tuple[bool, str]:
        return gate_turn_allowed(self.state())

    def drain_confirmed_edges(self) -> list[ConfirmedEdge]:
        """Measured, parallax-confirmed edge segments since the last drain.
        Robot-frame coordinates at the moment of measurement — pair each with
        the pose from that moment when stamping them into the world map."""
        with self._lock:
            edges = list(self._confirmed_edge_queue)
            self._confirmed_edge_queue.clear()
        return edges

    def report_external_stop(self, reason: str) -> None:
        """External gates (e.g. the lidar stop box) report their stops here so
        the post-stop hold covers them too — no gate may ratchet."""
        with self._lock:
            self._external_stop_reason = str(reason)

    def report_rotation(self, delta_deg: float) -> None:
        """The client reports its tracked in-place rotation here. Enough
        accumulated rotation releases reverse-only holds and the blind-zone
        latch: with the nose pointed well away from the stopped-at hazard,
        forward motion IS retreat, and the live gates own the new heading.
        Without this, a robot wedged against furniture behind deadlocks —
        release demands reverse, the map forbids reverse, forward is held."""
        with self._lock:
            self._rotation_since_stop_deg += abs(float(delta_deg))
            self._last_rotation_monotonic = time.monotonic()

    def _update_hold(
        self, stopping_now: bool, stop_reason: str, reverse_only: bool = False
    ) -> tuple[bool, str, bool]:
        """Post-stop hold state machine. Returns (hold, reason, reverse_required).

        reverse_only: the stop came from a BLIND-ZONE latch, where a "clear"
        reading means the eyes can't see the obstacle — not that it's gone.
        Such holds never time out; only measured reverse releases them.
        """
        cfg = self.config
        now = time.monotonic()
        if stopping_now:
            if not self._hold_active:
                self._hold_active = True
                self._hold_retreat_m = 0.0
                self._rotation_since_stop_deg = 0.0
                self._hold_event_times.append(now)
                window = float(cfg.hold_escalate_window_s)
                self._hold_event_times = [t for t in self._hold_event_times if now - t <= window]
                if len(self._hold_event_times) >= max(int(cfg.hold_escalate_count), 1):
                    # Repeated stops without retreat: the sensors cannot
                    # resolve this boundary. Never creep again — only
                    # reversing releases.
                    self._hold_permanent = True
            self._hold_reverse_only = self._hold_reverse_only or bool(reverse_only)
            self._hold_reason = str(stop_reason)
            self._hold_clear_since = None
        elif self._hold_active:
            if self._hold_retreat_m >= float(cfg.hold_release_reverse_m) or (
                self._rotation_since_stop_deg >= float(cfg.hold_release_rotation_deg)
            ):
                # Backing away — or rotating the nose well off the hazard so
                # forward motion IS backing away — genuinely improves the
                # situation: full reset. (Rotation matters for wedged robots
                # that have no reverse clearance.)
                self._hold_active = False
                self._hold_permanent = False
                self._hold_reverse_only = False
                self._hold_reason = ""
                self._hold_clear_since = None
                self._hold_event_times = []
            elif not self._hold_permanent and not self._hold_reverse_only:
                if self._hold_clear_since is None:
                    self._hold_clear_since = now
                if now - self._hold_clear_since >= float(cfg.hold_release_clear_s):
                    self._hold_active = False
                    self._hold_reason = ""
                    self._hold_clear_since = None
        return (
            self._hold_active,
            self._hold_reason,
            self._hold_permanent or self._hold_reverse_only,
        )

    def annotated(self, cam_name: str) -> np.ndarray | None:
        with self._lock:
            frame = self._annotated.get(cam_name)
        return None if frame is None else frame.copy()

    # -- helpers --------------------------------------------------------------
    def _integrate_motion(self, dt_s: float) -> None:
        velocity_fn = getattr(self._source, "latest_base_velocity", None)
        if not callable(velocity_fn):
            return
        x_vel, vel_age = velocity_fn()
        if x_vel is None or vel_age is None or vel_age > 1.0:
            return
        if float(x_vel) > 0.02:
            self._forward_travel_m += float(x_vel) * dt_s
        elif float(x_vel) < -0.02:
            if self._blind_latched:
                self._retreat_m += -float(x_vel) * dt_s
            if self._hold_active:
                self._hold_retreat_m += -float(x_vel) * dt_s
        self._travel_log.append((time.monotonic(), self._forward_travel_m))

    def _travel_at(self, monotonic_t: float) -> float:
        """Cumulative forward travel at (or just before) the given time."""
        best = None
        for t, travel in self._travel_log:
            if t <= float(monotonic_t):
                best = travel
            else:
                break
        if best is not None:
            return float(best)
        if self._travel_log:
            return float(self._travel_log[0][1])
        return float(self._forward_travel_m)

    def _candidates_from_depth(
        self, depth_results: dict[str, object]
    ) -> dict[str, tuple[str, float | None, float | None, float | None, float]]:
        """DepthEyeResults -> the classify ladder's candidate format.

        Distances are latency-compensated: the depth frame is old (CPU
        inference), so subtract however far the robot has driven since it
        was captured. Near segments are published as ConfirmedEdges so the
        map layer stamps REAL measured geometry instead of synthetic
        blockers."""
        cfg = self.config
        candidates: dict[str, tuple[str, float | None, float | None, float | None, float]] = {}
        now = time.monotonic()
        for eye_key, result in depth_results.items():
            advance = max(
                0.0, self._forward_travel_m - self._travel_at(float(result.frame_monotonic))
            )
            gate = (
                None
                if result.nearest_gate_m is None
                else max(0.05, float(result.nearest_gate_m) - advance)
            )
            nearest = (
                None
                if result.nearest_any_m is None
                else max(0.05, float(result.nearest_any_m) - advance)
            )
            if gate is None and nearest is None:
                continue
            bearing = 0.0 if result.bearing_deg is None else float(result.bearing_deg)
            candidates[eye_key] = (
                "elevated_confirmed",
                gate,
                nearest,
                None if result.edge_height_m is None else float(result.edge_height_m),
                bearing,
            )
            # Publish measured geometry for the map ONLY at stop range in
            # the CORRIDOR (the map layer records what stops the robot, not
            # everything the eyes sweep past) AND only from frames captured
            # AFTER the last rotation ended (a mid-turn frame's robot frame
            # is rotated relative to the stamp pose; travel compensation
            # cannot fix heading).
            footprint = tuple(
                (float(px) - advance, float(py))
                for px, py in getattr(result, "region_points_xy", ()) or ()
            )
            # Depth mode publishes FOOTPRINTS ONLY, and only from APPROACH-
            # range observations (corridor gate 0.30..1.5m, nearest >= 0.35m)
            # where monocular depth is proven good. Frame-filling close-ups
            # at the stop moment are physics-degenerate — no floor, no wall,
            # no scale cues; field 2026-07-12: a 0.19m close-up read
            # "elevated 1.01m" and smeared an 800-point blob across the map.
            # The stop itself is covered by the synthetic blocker at the
            # exactly-known stop pose.
            if (
                gate is not None
                and 0.30 <= gate <= 1.50
                and nearest is not None
                and nearest >= 0.35
                and float(result.frame_monotonic) >= self._last_rotation_monotonic + 0.2
                and footprint
            ):
                if (
                    result.segment_p1 is not None
                    and result.segment_p2 is not None
                    and result.edge_height_m is not None
                ):
                    seg_p1 = (
                        float(result.segment_p1[0]) - advance,
                        float(result.segment_p1[1]),
                    )
                    seg_p2 = (
                        float(result.segment_p2[0]) - advance,
                        float(result.segment_p2[1]),
                    )
                    seg_height = float(result.edge_height_m)
                else:
                    seg_p1 = seg_p2 = (float(gate), 0.0)
                    seg_height = 0.5
                with self._lock:
                    self._confirmed_edge_queue.append(
                        ConfirmedEdge(
                            eye=str(eye_key),
                            p1_robot_xy=seg_p1,
                            p2_robot_xy=seg_p2,
                            height_m=seg_height,
                            nearest_m=float(nearest if nearest is not None else gate),
                            monotonic=now,
                            points_robot_xy=footprint,
                        )
                    )
                    del self._confirmed_edge_queue[:-60]
        return candidates

    def _lidar_explains(self, bearing_deg: float, distance_m: float) -> bool:
        if self._lidar_ranges_fn is None:
            return False
        try:
            ranges = self._lidar_ranges_fn()
        except Exception:
            return False
        if ranges is None:
            return False
        bearings, distances = ranges
        if bearings is None or len(bearings) == 0:
            return False
        bearings = np.asarray(bearings, dtype=np.float32)
        distances = np.asarray(distances, dtype=np.float32)
        bearing_delta = np.abs(((bearings - float(bearing_deg)) + 180.0) % 360.0 - 180.0)
        mask = bearing_delta <= float(self.config.lidar_bearing_tolerance_deg)
        if not np.any(mask):
            return False
        return bool(
            np.any(
                np.abs(distances[mask] - float(distance_m))
                <= float(self.config.lidar_distance_tolerance_m)
            )
        )

    def _bottom_explains(
        self, obstacles: list[FloorObstacle], bearing_deg: float, distance_m: float
    ) -> bool:
        for obstacle in obstacles:
            bearing_delta = abs(
                ((obstacle.bearing_deg - float(bearing_deg)) + 180.0) % 360.0 - 180.0
            )
            if bearing_delta > float(self.config.bottom_bearing_tolerance_deg):
                continue
            if abs(obstacle.distance_m - float(distance_m)) <= float(
                self.config.bottom_distance_tolerance_m
            ):
                return True
        return False

    def _segment_distances(
        self,
        model: CameraModel,
        obs: EdgeObservation,
        edge_height_m: float,
        frame_shape: tuple[int, int],
    ) -> tuple[float | None, float | None]:
        """Corridor + nearest distances for the detected edge SEGMENT (exact
        for diagonal approaches). Falls back to the center-ray when the line
        endpoints are unavailable."""
        if obs.line_xy is None:
            center = (
                model.elevated_distance_for_row(
                    float(obs.center_y_ratio) if obs.center_y_ratio is not None else 0.7,
                    edge_height_m,
                )
            )
            return center, center
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        (px1, py1), (px2, py2) = obs.line_xy
        p1 = (px1 / max(frame_w - 1, 1), py1 / max(frame_h - 1, 1))
        p2 = (px2 / max(frame_w - 1, 1), py2 / max(frame_h - 1, 1))
        return edge_segment_hazard_distances(
            model,
            p1_image=p1,
            p2_image=p2,
            edge_height_m=float(edge_height_m),
            corridor_half_width_m=float(self.config.elevated_corridor_half_width_m),
            # An endpoint at the image border means the edge continues out of
            # view: extend that end conservatively.
            extend_p1=p1[0] < 0.03 or p1[0] > 0.97,
            extend_p2=p2[0] < 0.03 or p2[0] > 0.97,
        )

    def _classify_candidate(
        self,
        eye_key: str,
        model: CameraModel,
        obs: EdgeObservation,
        bottom_obstacles: list[FloorObstacle],
        bottom_available: bool,
        frame_shape: tuple[int, int],
    ) -> tuple[str, float | None, float | None, float | None, float]:
        """Returns (classification, corridor_gate_distance_m,
        nearest_distance_m, edge_height_m, bearing_deg)."""
        y_ratio = float(obs.center_y_ratio) if obs.center_y_ratio is not None else 0.7
        x_ratio = float(obs.center_x_ratio) if obs.center_x_ratio is not None else 0.5
        bearing = model.bearing_deg_for_column(x_ratio)
        floor_distance = model.floor_distance_for_row(y_ratio)

        # Track EVERY candidate for parallax — including referee-explained
        # ones. A 0.74m edge's if-on-floor projection lands ~5x its true
        # distance, so lidar returns in the background can "explain" it by
        # coincidence (observed in the field). Motion is the ground truth:
        # parallax outranks the referees in both directions.
        tracker = self._trackers[eye_key]
        bearing_delta = abs(((bearing - tracker.bearing_deg) + 180.0) % 360.0 - 180.0)
        if not tracker.active or bearing_delta > float(self.config.parallax_assoc_bearing_deg):
            tracker.reset(bearing_deg=bearing, y_ratio=y_ratio, travel_m=self._forward_travel_m)
        else:
            tracker.bearing_deg = bearing
            tracker.last_y_ratio = y_ratio

        baseline = self._forward_travel_m - tracker.travel_at_first_m
        if tracker.solved_height_m is None or baseline > tracker.solved_baseline_m + 0.05:
            # Solve — and keep RE-solving as the baseline grows. Parallax
            # accuracy scales with baseline; the first minimum-baseline
            # estimate is coarse and must not be frozen (a bad height skews
            # every downstream distance and the mapped segment's placement).
            solved = solve_edge_by_parallax(
                model,
                first_y_ratio=tracker.first_y_ratio,
                last_y_ratio=y_ratio,
                forward_travel_m=baseline,
            )
            if solved is not None and baseline >= float(self.config.parallax_min_travel_m):
                distance_now, edge_height = solved
                if edge_height <= float(self.config.parallax_floor_height_max_m):
                    # It moves like a floor object: not an elevated hazard.
                    tracker.active = False
                    return "floor_parallax", distance_now, distance_now, edge_height, bearing
                tracker.solved_height_m = min(
                    float(edge_height), float(self.config.parallax_max_edge_height_m)
                )
                tracker.solved_baseline_m = float(baseline)

        if tracker.solved_height_m is not None:
            # Confirmed edge: collision geometry from the full SEGMENT, not
            # its center — exact for diagonal approaches, and edges outside
            # the swept corridor stop gating drive-bys.
            gate_distance, nearest_distance = self._segment_distances(
                model, obs, tracker.solved_height_m, frame_shape
            )
            # Publish the measured segment (robot frame) for the map layer:
            # "this edge is X away, this wide, this tall".
            if obs.line_xy is not None and nearest_distance is not None:
                frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
                (lpx1, lpy1), (lpx2, lpy2) = obs.line_xy
                # ALWAYS project both endpoints at the line's mean image row:
                # per-endpoint depth is hypersensitive to image tilt (a few
                # degrees of camera roll = ~17px tilt = ~30deg of fake world
                # diagonal), so the published segment is the straight chord
                # at the center distance. Gating still uses the exact
                # unflattened geometry; a genuinely oblique edge gets traced
                # by chords from successive stop/sweep headings instead.
                mean_row = 0.5 * (float(lpy1) + float(lpy2))
                lpy1 = mean_row
                lpy2 = mean_row
                p1 = edge_point_robot_frame(
                    model,
                    lpx1 / max(frame_w - 1, 1),
                    lpy1 / max(frame_h - 1, 1),
                    tracker.solved_height_m,
                )
                p2 = edge_point_robot_frame(
                    model,
                    lpx2 / max(frame_w - 1, 1),
                    lpy2 / max(frame_h - 1, 1),
                    tracker.solved_height_m,
                )
                if p1 is not None and p2 is not None:
                    with self._lock:
                        self._confirmed_edge_queue.append(
                            ConfirmedEdge(
                                eye=eye_key,
                                p1_robot_xy=(float(p1[0]), float(p1[1])),
                                p2_robot_xy=(float(p2[0]), float(p2[1])),
                                height_m=float(tracker.solved_height_m),
                                nearest_m=float(nearest_distance),
                                monotonic=time.monotonic(),
                            )
                        )
                        del self._confirmed_edge_queue[:-60]
            return (
                "elevated_confirmed",
                gate_distance,
                nearest_distance,
                tracker.solved_height_m,
                bearing,
            )

        # Parallax unresolved: defer to the referees.
        if floor_distance is not None:
            if self._lidar_explains(bearing, floor_distance):
                return "floor_lidar", floor_distance, floor_distance, None, bearing
            if bottom_available and self._bottom_explains(bottom_obstacles, bearing, floor_distance):
                return "floor_low", floor_distance, floor_distance, None, bearing

        assumed = model.elevated_distance_for_row(y_ratio, float(self.config.assumed_edge_height_m))
        return "elevated_unresolved", assumed, assumed, None, bearing

    # -- core logic ---------------------------------------------------------------
    def tick(self) -> HazardState:
        """One full pipeline cycle. Public so tests can drive it synchronously."""
        cfg = self.config
        left_frame, left_age = self._source.latest(cfg.left_key)
        right_frame, right_age = self._source.latest(cfg.right_key)
        bottom_frame, bottom_age = self._source.latest(cfg.bottom_key)

        now_monotonic = time.monotonic()
        tick_dt_s = 0.0
        if self._last_tick_monotonic is not None:
            tick_dt_s = min(max(now_monotonic - self._last_tick_monotonic, 0.0), 0.5)
        self._last_tick_monotonic = now_monotonic
        self._integrate_motion(tick_dt_s)

        ages = [age for age in (left_age, right_age) if age is not None]
        worst_age = max(ages) if len(ages) == 2 else None
        stale = worst_age is None or worst_age > float(cfg.stale_timeout_s)
        if stale:
            hold, hold_reason, hold_permanent = self._update_hold(True, "camera_stale_stop")
            with self._lock:
                previous = self._state
                self._state = replace(
                    previous,
                    enabled=True,
                    active=True,
                    near_freeze=False,
                    side=previous.side if previous.active else "none",
                    reason="stale_frame",
                    frames_stale=True,
                    hold=hold,
                    hold_reason=hold_reason,
                    permanent_hold=hold_permanent,
                    frame_age_s=worst_age,
                    updated_monotonic=time.monotonic(),
                )
                return self._state

        bottom_available = (
            bottom_frame is not None
            and bottom_age is not None
            and bottom_age <= float(cfg.stale_timeout_s)
        )
        bottom_obstacles: list[FloorObstacle] = []
        if bottom_available:
            bottom_obstacles = self._bottom_detector.detect(bottom_frame)

        # ---- eye perception ------------------------------------------------------
        # With a depth worker attached, depth OWNS the eyes — no fallback.
        # Failed or stale depth is a blind sensor: fail SAFE (deny forward,
        # loud reason) exactly like stale camera frames, and stay that way
        # until depth recovers or the operator fixes it.
        used_depth = False
        depth_results: dict[str, object] = {}
        left_obs = right_obs = None
        candidates: dict[str, tuple[str, float | None, float | None, float | None, float]] = {}
        if self._depth_worker is not None:
            depth_failure = getattr(self._depth_worker, "failure", None)
            if depth_failure is None and getattr(self._depth_worker, "ready", False):
                for eye_key in (cfg.left_key, cfg.right_key):
                    depth_result = self._depth_worker.latest(
                        eye_key, max_age_s=float(cfg.depth_stale_timeout_s)
                    )
                    if depth_result is not None:
                        depth_results[eye_key] = depth_result
            if not depth_results:
                fail_reason = "depth_failed" if depth_failure is not None else "depth_stale"
                hold, hold_reason, hold_permanent = self._update_hold(True, fail_reason)
                with self._lock:
                    self._state = replace(
                        self._state,
                        enabled=True,
                        active=True,
                        near_freeze=False,
                        side="none",
                        reason=fail_reason,
                        frames_stale=True,
                        hold=hold,
                        hold_reason=hold_reason,
                        permanent_hold=hold_permanent,
                        frame_age_s=worst_age,
                        updated_monotonic=time.monotonic(),
                    )
                return self._state
            used_depth = True
            candidates = self._candidates_from_depth(depth_results)
        if not used_depth:
            left_obs = self._detector.detect(left_frame)
            right_obs = self._detector.detect(right_frame)

            # ---- classify eye candidates ---------------------------------------
            for eye_key, model, obs in (
                (cfg.left_key, cfg.eye_left_model, left_obs),
                (cfg.right_key, cfg.eye_right_model, right_obs),
            ):
                if obs.detected:
                    self._trackers[eye_key].miss_ticks = 0
                    candidates[eye_key] = self._classify_candidate(
                        eye_key,
                        model,
                        obs,
                        bottom_obstacles,
                        bottom_available,
                        left_frame.shape[:2] if eye_key == cfg.left_key else right_frame.shape[:2],
                    )
                else:
                    # Detection flicker tolerance: keep the parallax baseline
                    # alive across short gaps or it can never accumulate travel.
                    tracker = self._trackers[eye_key]
                    tracker.miss_ticks += 1
                    if tracker.miss_ticks > max(int(cfg.parallax_miss_tolerance_ticks), 0):
                        tracker.active = False

        elevated: list[tuple[str, str, float | None, float | None, float | None, float]] = []
        for eye_key, (klass, gate_dist, nearest_dist, height, bearing) in candidates.items():
            if klass in ("elevated_confirmed", "elevated_unresolved"):
                elevated.append((eye_key, klass, gate_dist, nearest_dist, height, bearing))

        # Strongest classification for diagnostics.
        classification = "none"
        if candidates:
            order = {
                "elevated_confirmed": 5,
                "elevated_unresolved": 4,
                "floor_parallax": 3,
                "floor_low": 2,
                "floor_lidar": 1,
            }
            classification = max((c[0] for c in candidates.values()), key=lambda k: order.get(k, 0))

        # ---- elevated hazard hysteresis (confirmed-in-range only) ---------------
        # A confirmed edge is a hazard when its SEGMENT enters the swept
        # corridor within the forward gate, OR when any part of it is inside
        # the near tier (a diagonal edge beside the shoulder must still stop
        # the robot even if its corridor entry point is farther out).
        confirmed_in_range = [
            (eye_key, gate_dist if gate_dist is not None else nearest_dist, nearest_dist, height)
            for eye_key, klass, gate_dist, nearest_dist, height, _ in elevated
            if klass == "elevated_confirmed"
            and (
                (gate_dist is not None and gate_dist <= float(cfg.forward_block_distance_m))
                or (nearest_dist is not None and nearest_dist <= float(cfg.near_freeze_distance_m))
            )
        ]
        if confirmed_in_range:
            self._hit_ticks += 1
            self._clean_ticks = 0
        else:
            self._hit_ticks = 0

        previous = self.state()
        previous_active = previous.active and not previous.frames_stale and not previous.blind_zone
        active = previous_active
        if self._hit_ticks >= max(int(cfg.trip_frames), 1):
            active = True
        elif previous_active:
            if confirmed_in_range:
                self._clean_ticks = 0
            else:
                self._clean_ticks += 1
                if self._clean_ticks >= max(int(cfg.clear_frames), 1):
                    active = False
                    self._clean_ticks = 0

        side = "none"
        est_distance: float | None = None
        edge_height: float | None = None
        confidence = 0.0
        nearest_confirmed: float | None = None
        if confirmed_in_range:
            eyes = {eye for eye, _, _, _ in confirmed_in_range}
            if cfg.left_key in eyes and cfg.right_key in eyes:
                side = "front"
            elif cfg.left_key in eyes:
                side = "left"
            else:
                side = "right"
            est_distance = min(d for _, d, _, _ in confirmed_in_range if d is not None)
            nearest_values = [n for _, _, n, _ in confirmed_in_range if n is not None]
            nearest_confirmed = min(nearest_values) if nearest_values else None
            heights = [h for _, _, _, h in confirmed_in_range if h is not None]
            edge_height = min(heights) if heights else None
            confidence = 0.9
        elif active:
            side = previous.side
            est_distance = previous.est_distance_m
            edge_height = previous.edge_height_m
            confidence = previous.confidence

        # The near tier (rotation freeze) uses the NEAREST point of the edge
        # segment — a diagonal edge beside the shoulder freezes rotation even
        # when its corridor entry point is farther ahead.
        near_metric = nearest_confirmed if nearest_confirmed is not None else est_distance
        near_freeze = False
        if active and near_metric is not None:
            if previous.near_freeze:
                near_freeze = near_metric <= (
                    float(cfg.near_freeze_distance_m) + float(cfg.near_release_margin_m)
                )
            else:
                near_freeze = near_metric <= float(cfg.near_freeze_distance_m)
        elif active and previous.near_freeze:
            near_freeze = True

        # ---- blind-zone vanish latch --------------------------------------------
        if used_depth:
            # Depth perception does not go blind close-in — a near surface is
            # a LARGE mask, not a vanished line — so the Hough-era latch is
            # unnecessary and its row bookkeeping has no inputs here. Release
            # any latch left over from detector-mode ticks: the depth result
            # now owns the near field.
            self._blind_latched = False
            self._vanish_armed = False
            self._vanish_hits = 0
            self._vanish_misses = 0
            deep_rows = []
        else:
            deep_rows = [
                float(obs.center_y_ratio)
                for eye_key, obs in ((cfg.left_key, left_obs), (cfg.right_key, right_obs))
                if obs.detected
                and obs.center_y_ratio is not None
                and candidates.get(eye_key, ("",))[0]
                in ("elevated_confirmed", "elevated_unresolved")
            ]
        deepest_row = max(deep_rows) if deep_rows else None
        # "Blind" must mean NO deep detection AT ALL — classification flapping
        # (elevated <-> floor_*) previously latched "EDGE LOST" while both
        # eyes were scoring 1800 on the edge (field bug).
        any_rows = [
            float(obs.center_y_ratio)
            for obs in (left_obs, right_obs)
            if obs is not None and obs.detected and obs.center_y_ratio is not None
        ]
        deepest_any = max(any_rows) if any_rows else None
        # Visible at/below the recede row (any classification) = we can still
        # see the edge region: never a "miss". A miss requires seeing NOTHING
        # down there.
        edge_region_visible = deepest_any is not None and deepest_any >= float(
            cfg.vanish_recede_row_ratio
        )
        if self._blind_latched:
            # The latch means "the eyes are BLIND down where the edge was".
            # Re-sighting anything in the edge region (whatever its
            # classification) means they are not: release; the distance gate
            # governs from there.
            released_by_sight = edge_region_visible
            released_by_retreat = self._retreat_m >= float(cfg.vanish_release_reverse_m)
            released_by_rotation = self._rotation_since_stop_deg >= float(
                cfg.vanish_release_rotation_deg
            )
            if released_by_sight or released_by_retreat or released_by_rotation:
                self._blind_latched = False
                self._retreat_m = 0.0
                self._vanish_armed = False
                self._vanish_hits = 0
                self._vanish_misses = 0
        if deepest_row is not None and deepest_row >= float(cfg.vanish_arm_row_ratio):
            self._vanish_hits += 1
            self._vanish_misses = 0
            if self._vanish_hits >= max(int(cfg.vanish_arm_frames), 1):
                self._vanish_armed = True
                self._vanish_row = deepest_row
                self._vanish_side = side if side != "none" else (
                    "left" if (left_obs is not None and left_obs.detected) else "right"
                )
        elif deepest_row is not None and deepest_row < float(cfg.vanish_recede_row_ratio):
            # Candidate receded (moved up in the frame): disarm.
            self._vanish_hits = 0
            self._vanish_misses = 0
            self._vanish_armed = False
        elif edge_region_visible:
            # Structure still visible in the edge region, just not classified
            # elevated (or in the recede..arm band): hold the armed state
            # without counting a miss.
            self._vanish_hits = 0
            self._vanish_misses = 0
        else:
            self._vanish_hits = 0
            if self._vanish_armed and not self._blind_latched:
                self._vanish_misses += 1
                if self._vanish_misses >= max(int(cfg.vanish_miss_frames), 1):
                    # A deep candidate vanished WITHOUT receding: blind zone.
                    self._blind_latched = True
                    self._retreat_m = 0.0
                    self._rotation_since_stop_deg = 0.0
                    self._vanish_armed = False
                    self._vanish_misses = 0

        reason = "clear"
        blind_zone = False
        if active:
            if used_depth:
                reason = "depth_elevated"
            else:
                reason = "elevated_parallax" if edge_height is not None else "line_edge"
        if self._blind_latched:
            active = True
            near_freeze = True
            blind_zone = True
            side = self._vanish_side
            confidence = max(confidence, 0.5)
            reason = "vanished_near"

        # ---- bottom-camera ground gate -------------------------------------------
        ground_active = previous.ground_active
        ground_distance: float | None = previous.ground_distance_m
        ground_side = previous.ground_side
        if bool(cfg.ground_gate_enabled) and bottom_available:
            in_corridor = [
                obstacle
                for obstacle in bottom_obstacles
                if obstacle.crosses_horizon
                and obstacle.distance_m <= float(cfg.ground_stop_distance_m)
                and abs(obstacle.distance_m * math.sin(math.radians(obstacle.bearing_deg)))
                <= float(cfg.ground_corridor_half_width_m)
            ]
            if in_corridor:
                self._ground_hits += 1
                self._ground_clean = 0
            else:
                self._ground_hits = 0
            if self._ground_hits >= max(int(cfg.ground_trip_frames), 1):
                ground_active = True
                nearest = min(in_corridor, key=lambda o: o.distance_m)
                ground_distance = float(nearest.distance_m)
                ground_side = (
                    "front"
                    if abs(nearest.bearing_deg) <= 8.0
                    else ("left" if nearest.bearing_deg > 0 else "right")
                )
            elif ground_active and not in_corridor:
                self._ground_clean += 1
                if self._ground_clean >= max(int(cfg.ground_clear_frames), 1):
                    ground_active = False
                    ground_distance = None
                    ground_side = "none"
                    self._ground_clean = 0
            elif ground_active:
                self._ground_clean = 0
        elif not bottom_available:
            # No referee, no ground gate — never latch a stale ground stop.
            ground_active = False
            ground_distance = None
            ground_side = "none"
        if ground_active and reason == "clear":
            reason = "ground_obstacle"

        # ---- post-stop hold (anti-ratchet) ----------------------------------------
        with self._lock:
            external_reason = self._external_stop_reason
            self._external_stop_reason = None
        stopping_now = bool(active or blind_zone or ground_active or external_reason)
        stop_reason = (
            external_reason
            if external_reason
            else ("vanished_near" if blind_zone else reason if reason != "clear" else "hazard")
        )
        hold, hold_reason, hold_permanent = self._update_hold(
            stopping_now, stop_reason, reverse_only=blind_zone
        )

        # Per-eye diagnostics for the log ("what does it actually see").
        def _cand_text(eye_key: str) -> str:
            entry = candidates.get(eye_key)
            if entry is None:
                return "-"
            klass, gate_dist, nearest_dist, height, bearing = entry
            g = "-" if gate_dist is None else f"{gate_dist:.2f}"
            n = "-" if nearest_dist is None else f"{nearest_dist:.2f}"
            h = "-" if height is None else f"{height:.2f}"
            return f"{klass} g={g} n={n} h={h} b={bearing:.0f}"

        detail = f"L[{_cand_text(cfg.left_key)}] R[{_cand_text(cfg.right_key)}]"

        state = HazardState(
            enabled=True,
            active=active,
            near_freeze=near_freeze,
            side=side if active else "none",
            confidence=confidence if active else 0.0,
            reason=reason,
            est_distance_m=est_distance,
            edge_height_m=edge_height,
            frames_stale=False,
            blind_zone=blind_zone,
            ground_active=bool(ground_active),
            ground_distance_m=ground_distance,
            ground_side=ground_side,
            bottom_available=bool(bottom_available),
            classification=classification,
            hold=hold,
            hold_reason=hold_reason,
            permanent_hold=hold_permanent,
            detail=("depth " if used_depth else "") + detail,
            left_score=float(left_obs.score) if left_obs is not None else 0.0,
            right_score=float(right_obs.score) if right_obs is not None else 0.0,
            frame_age_s=worst_age,
            updated_monotonic=time.monotonic(),
        )
        if used_depth:
            left_result = depth_results.get(cfg.left_key)
            right_result = depth_results.get(cfg.right_key)
            annotated_left = (
                left_result.overlay_bgr
                if left_result is not None and left_result.overlay_bgr is not None
                else left_frame
            )
            annotated_right = (
                right_result.overlay_bgr
                if right_result is not None and right_result.overlay_bgr is not None
                else right_frame
            )
        else:
            annotated_left = render_overlay(
                left_frame, cfg.left_key, left_obs, state, cfg,
                classification=candidates.get(cfg.left_key, ("none", None, None, None, 0.0)),
            )
            annotated_right = render_overlay(
                right_frame, cfg.right_key, right_obs, state, cfg,
                classification=candidates.get(cfg.right_key, ("none", None, None, None, 0.0)),
            )
        with self._lock:
            self._state = state
            self._annotated[cfg.left_key] = annotated_left
            self._annotated[cfg.right_key] = annotated_right
            if bottom_available:
                self._annotated[cfg.bottom_key] = render_bottom_overlay(
                    bottom_frame, bottom_obstacles, state, cfg
                )
        return state


def render_overlay(
    frame_bgr: np.ndarray,
    cam_name: str,
    obs: EdgeObservation,
    state: HazardState,
    config: ElevatedSafetyConfig,
    classification: tuple[str, float | None, float | None, float | None, float] = ("none", None, None, None, 0.0),
) -> np.ndarray:
    """ROI band + detection + geometric classification, for viewers/Rerun."""
    canvas = frame_bgr.copy()
    height, width = canvas.shape[:2]
    roi_y0 = int(height * float(config.roi_min_y_ratio))
    roi_y1 = int(height * float(config.roi_max_y_ratio))
    roi_color = (0, 0, 255) if (state.active or state.blind_zone) else (0, 200, 0)
    cv2.rectangle(canvas, (2, roi_y0), (width - 3, roi_y1), roi_color, 1)
    klass, distance, nearest_distance, edge_height, _bearing = classification
    if obs.bbox is not None:
        box_color = {
            "elevated_confirmed": (0, 0, 255),
            "elevated_unresolved": (0, 140, 255),
            "floor_lidar": (180, 180, 180),
            "floor_low": (180, 180, 180),
            "floor_parallax": (180, 180, 180),
        }.get(klass, (0, 140, 255))
        x0, y0, x1, y1 = obs.bbox
        cv2.rectangle(canvas, (x0, y0), (x1, y1), box_color, 2)
    if obs.line_xy is not None:
        cv2.line(canvas, obs.line_xy[0], obs.line_xy[1], (0, 220, 220), 2)
    dist_text = "-" if distance is None else f"{distance:.2f}m"
    height_text = "-" if edge_height is None else f"{edge_height:.2f}m"
    cy_text = "-" if obs.center_y_ratio is None else f"{obs.center_y_ratio:.2f}"
    lines = [
        f"{cam_name} {state.decision_label()}",
        f"{klass} dist={dist_text} h={height_text}",
        f"score={obs.score:.0f} cy={cy_text} {obs.reason}",
    ]
    y = 18
    for text in lines:
        cv2.putText(canvas, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18
    return canvas


def render_bottom_overlay(
    frame_bgr: np.ndarray,
    obstacles: list[FloorObstacle],
    state: HazardState,
    config: ElevatedSafetyConfig,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    height, width = canvas.shape[:2]
    horizon_row = int(np.clip(config.bottom_model.horizon_y_ratio(), 0.02, 0.98) * height)
    cv2.line(canvas, (0, horizon_row), (width, horizon_row), (255, 200, 0), 1)
    for obstacle in obstacles:
        x = int(obstacle.x_ratio * (width - 1))
        y = int(obstacle.y_ratio * (height - 1))
        near = obstacle.crosses_horizon and obstacle.distance_m <= float(
            config.ground_stop_distance_m
        )
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
    cv2.putText(canvas, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def format_hazard_log(state: HazardState) -> str:
    dist = "-" if state.est_distance_m is None else f"{state.est_distance_m:.2f}m"
    edge_h = "-" if state.edge_height_m is None else f"{state.edge_height_m:.2f}m"
    age = "-" if state.frame_age_s is None else f"{state.frame_age_s:.2f}s"
    ground = (
        f" ground={state.ground_side}@{state.ground_distance_m:.2f}m"
        if state.ground_active and state.ground_distance_m is not None
        else ""
    )
    hold_text = ""
    if state.hold:
        hold_text = f" HOLD({state.hold_reason}{', PERMANENT' if state.permanent_hold else ''})"
    return (
        f"decision={state.decision_label()} side={state.side} class={state.classification} "
        f"confidence={state.confidence:.2f} reason={state.reason} dist={dist} edge_h={edge_h} "
        f"near_freeze={state.near_freeze} bottom={'ok' if state.bottom_available else 'absent'}"
        f"{ground}{hold_text} frame_age={age} scores=({state.left_score:.0f}, {state.right_score:.0f}) "
        f"{state.detail}"
    )


def endpoint_from_remote_ip(remote_ip: str, port: int = 5560) -> str:
    return f"tcp://{str(remote_ip).strip()}:{int(port)}"
