"""SEMANTIC edge/obstacle detector — the second perception leg the user asked
for (2026-07-20). The metric-depth model (Depth-Anything) gives DISTANCE but is
unreliable on WHITE, flat, textureless surfaces (a folding-table lip reads as
"far"/floor), so the robot rammed those repeatedly. A monocular depth model
fails there for sensor-physics reasons, not a bad checkpoint. This detector adds
an INDEPENDENT signal: an open-vocabulary object detector (YOLO-World) that
recognizes a table/counter/shelf edge SEMANTICALLY — it can flag "that's a
table" even when the depth map cannot range it. The safety gate stops forward
motion if EITHER the depth gate OR this detector fires.

Design:
  - Runs in its OWN daemon thread at its OWN (fast) rate, decoupled from the slow
    depth cycle — YOLO-World is ~30-50ms/frame on GPU, fast enough to actually
    gate a moving robot (Florence-2/OWLv2 at ~0.5-2s would just repeat the depth
    latency problem the user is angry about).
  - Consumes the SAME fused panorama the depth leg uses (one calibrated forward
    view with a known hfov), so a detection's horizontal centre maps to a real
    world bearing.
  - "In my path and close" from a 2D box (no depth needed): the box's horizontal
    centre must fall inside the forward corridor, and its BOTTOM edge must sit
    low enough in the frame that the object is near the floor in front of the
    robot (a distant tabletop projects high; an imminent one projects low). This
    is a coarse proximity proxy, deliberately conservative — it triggers a STOP,
    never a drive.
  - OPT-IN and fail-soft: ultralytics is imported lazily inside load(); a missing
    dependency or download failure sets .failure and the thread exits without
    ever blocking (the depth gate + lidar box still protect). Off unless the
    caller enables it, so it cannot destabilize the working pipeline while it is
    being tuned on hardware.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

# Open-vocabulary classes to treat as "elevated furniture edge the lidar drives
# under". YOLO-World takes these as free-text prompts; tune on hardware.
#
# DELIBERATELY NARROW (field 2026-07-20): the semantic leg exists for ONE thing
# the depth model + lidar cannot see — the FLAT, WHITE, floating tabletop/counter
# LIP the base drives its nose under and rams. Bulky solid furniture (couch, sofa,
# bed, dresser, cabinet, shelf, chair) is fully visible to the lidar stop box and
# the depth gate already, so listing it here adds NO protection and turns the
# detector into a room-wide false-stop: with those classes the robot froze in a
# furnished room, vetoing forward for a sofa against the far wall (RUN 45) — the
# exact "stopping for a couch that isn't in the way" failure from the start of
# this project. Keep ONLY the flat-topped edge family.
DEFAULT_EDGE_CLASSES: tuple[str, ...] = (
    "table",
    "table edge",
    "desk",
    "coffee table",
    "counter",
    "countertop",
    "kitchen island",
    "workbench",
)

DEFAULT_YOLO_WORLD_WEIGHTS = "yolov8s-worldv2.pt"


@dataclass
class SemanticEdgeConfig:
    """Tunables for the semantic edge detector."""

    enabled: bool = False
    weights: str = DEFAULT_YOLO_WORLD_WEIGHTS
    classes: tuple[str, ...] = DEFAULT_EDGE_CLASSES
    # Detection confidence floor. Raised from 0.10 (field 2026-07-20): 0.10
    # accepted near-noise boxes and the detector vetoed every cycle. A real
    # imminent table edge scores well above this.
    min_confidence: float = 0.35
    # Forward corridor: a detection only blocks if its box centre bearing is
    # within this of straight-ahead (the robot only cares about what it drives
    # INTO, not furniture off to the side it will pass). Narrowed from 32deg
    # (field 2026-07-20) so furniture off to the sides no longer trips it.
    corridor_half_width_deg: float = 22.0
    # Panorama horizontal field of view (deg) — maps a box column to a bearing.
    # Must match the fused panorama the depth leg uses.
    panorama_hfov_deg: float = 104.5
    # PROXIMITY-BY-PROJECTION: the box's BOTTOM row, as a fraction of image
    # height, must be at or below this line to count as "close/in front". A
    # distant tabletop's near edge projects high in the frame; an imminent one
    # projects low. 0.0 = top, 1.0 = bottom. Conservative (bigger = only very
    # close). Raised from 0.55 (field 2026-07-20): 0.55 = the lower HALF of the
    # frame, which furniture across a small room easily projects into, so it
    # blocked constantly. 0.80 fires only when the edge's bottom is in the
    # lowest fifth of the view = genuinely about to be under the nose.
    min_bottom_row_ratio: float = 0.80
    # Ignore tiny detections (noise): box must cover at least this fraction of
    # the frame width. Raised from 0.10 (field 2026-07-20) — an imminent edge
    # fills the view.
    min_box_width_ratio: float = 0.16
    # Consecutive detections to trip / clean frames to clear (hysteresis, like
    # the depth gate). Keeps a single flicker from stopping the robot. Raised
    # from 2 (field 2026-07-20): a real imminent edge persists across frames.
    trip_frames: int = 3
    clear_frames: int = 3
    max_frame_age_s: float = 1.0


@dataclass
class SemanticEdgeResult:
    """What the detector believes right now."""

    blocking: bool = False
    # Bearing (deg, +left/-right per the panorama convention) to the nearest
    # blocking furniture edge, or None.
    bearing_deg: float | None = None
    label: str = ""
    confidence: float = 0.0
    monotonic: float = 0.0
    detections: list = field(default_factory=list)


class SemanticEdgeWorker:
    """Daemon thread: runs the open-vocab detector on the fused panorama and
    publishes a SemanticEdgeResult. Fail-soft: never raises into the caller;
    a load failure just leaves .failure set and .latest() returning a
    non-blocking result."""

    def __init__(
        self,
        subscriber,
        mosaic,
        config: SemanticEdgeConfig,
    ) -> None:
        self._subscriber = subscriber
        self._mosaic = mosaic
        self._cfg = config
        self._model = None
        self._lock = threading.Lock()
        self._result = SemanticEdgeResult()
        self._hit_frames = 0
        self._clean_frames = 0
        self.ready = False
        self.failure: str | None = None
        self.inference_s: float = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None or not self._cfg.enabled:
            return
        self._thread = threading.Thread(
            target=self._run, name="semantic-edge", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def latest(self) -> SemanticEdgeResult:
        """The freshest result, or a stale-safe non-blocking one. Never None."""
        with self._lock:
            res = self._result
        if not self.ready:
            return SemanticEdgeResult()
        if time.monotonic() - res.monotonic > float(self._cfg.max_frame_age_s):
            # Stale: do not assert a block on old data (the depth gate + stale
            # timeout own the "blind" case); report clear.
            return SemanticEdgeResult(monotonic=res.monotonic)
        return res

    def _load(self) -> None:
        from ultralytics import YOLOWorld  # lazy: optional dependency

        model = YOLOWorld(self._cfg.weights)
        model.set_classes(list(self._cfg.classes))
        self._model = model

    def _run(self) -> None:
        try:
            self._load()
        except Exception as exc:  # missing dep / download failure: fail soft
            self.failure = f"{type(exc).__name__}: {exc}"
            print(
                f"[safety] WARNING: semantic edge detector failed to load "
                f"({self.failure}); depth gate + lidar box still protect. "
                "Install it with `--with ultralytics` if you want this leg."
            )
            return
        self.ready = True
        print(
            "[safety] semantic edge detector READY (YOLO-World open-vocab) — "
            "an INDEPENDENT stop on table/counter/shelf edges the depth model "
            "misses on white surfaces"
        )
        while not self._stop_event.is_set():
            left, age_l = self._subscriber.latest("front_left")
            right, age_r = self._subscriber.latest("front_right")
            if (
                left is None
                or right is None
                or age_l is None
                or age_r is None
                or max(age_l, age_r) > float(self._cfg.max_frame_age_s)
                or abs(float(age_l) - float(age_r)) > 0.35
            ):
                time.sleep(0.03)
                continue
            frame = self._mosaic.compose(left, right)
            self._process(frame)

    def _process(self, frame: np.ndarray) -> None:
        t0 = time.monotonic()
        try:
            preds = self._model.predict(
                frame, conf=float(self._cfg.min_confidence), verbose=False
            )
        except Exception as exc:
            # A single inference hiccup must not kill the thread.
            print(f"[safety] semantic edge inference error ({type(exc).__name__}: {exc})")
            time.sleep(0.05)
            return
        self.inference_s = time.monotonic() - t0
        h, w = int(frame.shape[0]), int(frame.shape[1])
        cfg = self._cfg
        names = getattr(self._model, "names", {}) or {}

        best: dict | None = None
        detections: list = []
        for pred in preds:
            boxes = getattr(pred, "boxes", None)
            if boxes is None:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i].item())
                cls_id = int(boxes.cls[i].item())
                label = str(names.get(cls_id, cls_id))
                x1, y1, x2, y2 = xyxy
                box_w = (x2 - x1) / max(1.0, w)
                bottom_ratio = y2 / max(1.0, h)
                cx_ratio = ((x1 + x2) / 2.0) / max(1.0, w)
                # Column -> bearing across the panorama hfov (centre = 0deg,
                # +left). cx_ratio 0..1 left->right, so left is +bearing.
                bearing_deg = (0.5 - cx_ratio) * float(cfg.panorama_hfov_deg)
                detections.append(
                    {"label": label, "conf": conf, "bearing_deg": bearing_deg,
                     "bottom_ratio": bottom_ratio, "box_w": box_w}
                )
                # Blocking gate: in the forward corridor, close (low in frame),
                # and big enough to be real.
                if (
                    abs(bearing_deg) <= float(cfg.corridor_half_width_deg)
                    and bottom_ratio >= float(cfg.min_bottom_row_ratio)
                    and box_w >= float(cfg.min_box_width_ratio)
                ):
                    if best is None or conf > best["conf"]:
                        best = {"conf": conf, "bearing_deg": bearing_deg, "label": label}

        # CONSECUTIVE-streak counters. Field 2026-07-20: the old code
        # incremented _hit_frames cumulatively and only reset it inside the
        # non-blocking else branch — which became unreachable the instant
        # _hit_frames crossed trip_frames (the first `if` always won). So after
        # 3 hits EVER (not even consecutive) the block LATCHED ON forever, and
        # the robot froze on `semantic_edge_stop_front()` (empty label = no
        # current detection, purely the stuck latch) for the whole run. A clean
        # frame now breaks the hit streak and a hit breaks the clean streak, so
        # both counters mean "consecutive" and the latch cannot form.
        if best is not None:
            self._hit_frames += 1
            self._clean_frames = 0
        else:
            self._clean_frames += 1
            self._hit_frames = 0

        with self._lock:
            was_blocking = self._result.blocking
            if was_blocking:
                # Release only after enough CONSECUTIVE clean frames; a single
                # fresh detection re-holds it (clean streak was reset above).
                blocking = self._clean_frames < max(1, int(cfg.clear_frames))
            else:
                # Trip only after enough CONSECUTIVE hit frames.
                blocking = self._hit_frames >= max(1, int(cfg.trip_frames))
            self._result = SemanticEdgeResult(
                blocking=blocking,
                bearing_deg=best["bearing_deg"] if best else None,
                label=best["label"] if best else "",
                confidence=best["conf"] if best else 0.0,
                monotonic=time.monotonic(),
                detections=detections,
            )
