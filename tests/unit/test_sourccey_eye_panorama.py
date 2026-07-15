"""Regression tests for the two-eye panorama warp/calibration math."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import sourccey_eye_panorama as pano


def test_intrinsics_center_and_edge_rays() -> None:
    k = pano.intrinsics(320, 240, 82.0)
    k_inv = np.linalg.inv(k)
    center = k_inv @ np.array([159.5, 119.5, 1.0])
    assert abs(center[0]) < 1e-9 and abs(center[1]) < 1e-9
    # Leftmost column looks half the hfov off-axis.
    edge = k_inv @ np.array([0.0, 119.5, 1.0])
    assert abs(math.degrees(math.atan2(-edge[0], edge[2])) - 41.0) < 0.5


def test_yaw_left_points_forward_ray_left() -> None:
    rot = pano.rotation_cam_to_ref(15.0, 0.0, 0.0)
    forward_in_ref = rot @ np.array([0.0, 0.0, 1.0])
    assert forward_in_ref[0] < -0.2  # x is right; looking left = negative x
    assert abs(forward_in_ref[1]) < 1e-9


def test_warp_grid_is_identity_for_matching_cameras() -> None:
    virt = pano.VirtualCamera(width=320, height=240, hfov_deg=82.0, pitch_down_deg=13.9)
    eye_k = pano.intrinsics(320, 240, 82.0)
    eye_rot = pano.rotation_cam_to_ref(0.0, 13.9, 0.0)
    map_x, map_y, mask = pano.warp_grid(virt, eye_k, eye_rot, (240, 320))
    assert mask.all()
    u, v = np.meshgrid(np.arange(320, dtype=np.float32), np.arange(240, dtype=np.float32))
    assert float(np.abs(map_x - u).max()) < 1e-3
    assert float(np.abs(map_y - v).max()) < 1e-3


def _render_synthetic_pair(cal: pano.EyeCalibration) -> tuple[np.ndarray, np.ndarray]:
    """Render both eyes viewing one distant textured wall: sample each eye
    from a wide virtual texture through the same rotation model the stitcher
    uses (H = K_e R_rel^T K_v^-1), so a solver that inverts it exactly
    recovers the injected calibration."""
    rng = np.random.default_rng(7)
    virt = pano.default_virtual_camera(320, 240, cal.hfov_deg)
    # Structured scene (edges/rectangles), not blurred noise: the score works
    # on gradient statistics, and statistically-uniform noise has none — a
    # real room has walls, furniture edges, posters.
    texture = np.full((virt.height, virt.width, 3), 90, dtype=np.uint8)
    for _ in range(60):
        x0, y0 = rng.integers(0, virt.width - 20), rng.integers(0, virt.height - 20)
        w, h = int(rng.integers(10, 80)), int(rng.integers(10, 60))
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.rectangle(texture, (int(x0), int(y0)), (int(x0) + w, int(y0) + h), color, -1)
    texture = cv2.GaussianBlur(texture, (3, 3), 0.8)
    eye_k = pano.intrinsics(320, 240, cal.hfov_deg)
    rot_l, rot_r = pano._eye_rotations(cal)
    frames = []
    for rot in (rot_l, rot_r):
        # eye pixel -> ray -> reference -> virtual pixel: a dst->src map,
        # which is exactly cv2's WARP_INVERSE_MAP convention.
        h_matrix = virt.k() @ virt.rotation().T @ rot @ np.linalg.inv(eye_k)
        frames.append(
            cv2.warpPerspective(
                texture,
                h_matrix.astype(np.float64),
                (320, 240),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            )
        )
    return frames[0], frames[1]


def test_overlap_score_peaks_at_true_roll() -> None:
    truth = pano.EyeCalibration(roll_left_deg=2.0, roll_right_deg=-1.0)
    left, right = _render_synthetic_pair(truth)
    at_truth = pano.score_calibration([(left, right)], truth)
    off = pano.EyeCalibration(**{**truth.__dict__})
    off.roll_left_deg = truth.roll_left_deg + 2.5
    at_off = pano.score_calibration([(left, right)], off)
    assert at_truth > 0.6
    assert at_truth > at_off + 0.05


def test_blend_covers_full_panorama() -> None:
    cal = pano.EyeCalibration()
    left, right = _render_synthetic_pair(cal)
    lw, rw, lm, rm, _ = pano.warp_pair(left, right, cal)
    fused = pano.blend_panorama(lw, rw, lm, rm)
    coverage = float(np.mean((lm | rm)))
    assert fused.shape[:2] == lm.shape
    assert coverage > 0.6  # both wings plus the shared cone are filled


def test_refine_overlap_reduces_center_ghost() -> None:
    """Local flow refinement must improve overlap registration when the two
    warped eyes disagree by a residual shift the global rotation can't fix
    (parallax / lens distortion). Model that as a small local warp of the
    right view and check the overlap NCC rises."""
    cal = pano.EyeCalibration()
    left, right = _render_synthetic_pair(cal)
    lw, rw, lm, rm, _ = pano.warp_pair(left, right, cal)
    # Inject a residual local shift into the right warp inside the overlap.
    h, w = rw.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    shifted = cv2.remap(rw, xs - 5.0, ys - 3.0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    before = pano.overlap_score(lw, shifted, lm, rm)
    fixed, fixed_mask = pano.refine_overlap(lw, shifted, lm, rm)
    after = pano.overlap_score(lw, fixed, lm, fixed_mask)
    assert after > before + 0.05


def test_refine_overlap_corrects_residual_rotation() -> None:
    """A residual ROTATION between the two warped halves (the table-ledge
    'kink' the user reported) must be removed by the global stage, not just
    the local flow — check the overlap NCC recovers after rotating the right
    warp about the frame center."""
    cal = pano.EyeCalibration()
    left, right = _render_synthetic_pair(cal)
    lw, rw, lm, rm, _ = pano.warp_pair(left, right, cal)
    h, w = rw.shape[:2]
    rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 4.0, 1.0)  # 4 deg residual
    rw_rot = cv2.warpAffine(rw, rot, (w, h))
    rm_rot = cv2.warpAffine(rm.astype(np.uint8), rot, (w, h)) > 0
    before = pano.overlap_score(lw, rw_rot, lm, rm_rot)
    fixed, fixed_mask = pano.refine_overlap(lw, rw_rot, lm, rm_rot)
    after = pano.overlap_score(lw, fixed, lm, fixed_mask)
    assert after > before + 0.1


def test_overlap_refiner_is_temporally_stable_under_jitter() -> None:
    """The live refiner must NOT let per-frame estimation noise wobble the
    output: feeding the same scene with tiny per-frame jitter, the EMA-
    smoothed refiner's output should vary far less frame-to-frame than the
    stateless one-shot (which re-estimates from scratch every call)."""
    import math

    cal = pano.EyeCalibration()
    left, right = _render_synthetic_pair(cal)
    lw, rw, lm, rm, _ = pano.warp_pair(left, right, cal)
    h, w = rw.shape[:2]
    # Fixed residual misalignment the refiner's global stage must correct every
    # frame — this is the transform that moves the solid wings.
    rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 3.0, 1.0)
    rw_mis = cv2.warpAffine(rw, rot, (w, h))
    both = lm & rm

    def angle_of(params):
        return math.degrees(math.atan2(params[1], params[0]))

    raw_angles, ema_angles = [], []
    refiner = pano.OverlapRefiner()
    for seed in range(12):
        rng = np.random.default_rng(seed)
        noisy = np.clip(
            rw_mis.astype(np.float32) + rng.normal(0, 6.0, rw_mis.shape).astype(np.float32), 0, 255
        ).astype(np.uint8)
        m_raw = pano._residual_similarity(lw, noisy, both)  # stateless per-frame estimate
        if m_raw is not None:
            raw_angles.append(angle_of([m_raw[0, 0], m_raw[1, 0]]))
        refiner.refine(lw, noisy, lm, rm)  # updates the EMA transform
        if refiner._m_ema is not None:
            ema_angles.append(angle_of(refiner._m_ema))

    # Over the settled tail, the EMA transform's angle jitters far less than the
    # raw per-frame estimate (which is what makes the wings squiggle live).
    raw_std = float(np.std(raw_angles[-6:]))
    ema_std = float(np.std(ema_angles[-6:]))
    assert ema_std < raw_std


def test_overlap_refiner_freezes_and_locks_the_wings() -> None:
    """After warmup the global transform must FREEZE — subsequent frames,
    however noisy, cannot change it, so the solid wings are truly locked."""
    cal = pano.EyeCalibration()
    left, right = _render_synthetic_pair(cal)
    lw, rw, lm, rm, _ = pano.warp_pair(left, right, cal)
    h, w = rw.shape[:2]
    rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 3.0, 1.0)
    rw_mis = cv2.warpAffine(rw, rot, (w, h))

    refiner = pano.OverlapRefiner(freeze_after=5)
    for seed in range(6):  # warm up past the freeze threshold
        rng = np.random.default_rng(seed)
        noisy = np.clip(
            rw_mis.astype(np.float32) + rng.normal(0, 6.0, rw_mis.shape).astype(np.float32), 0, 255
        ).astype(np.uint8)
        refiner.refine(lw, noisy, lm, rm)
    assert refiner.locked
    frozen = refiner._m_frozen.copy()
    # Feed very different/noisy frames; the frozen transform must not move.
    for seed in range(100, 105):
        rng = np.random.default_rng(seed)
        noisy = np.clip(
            rw_mis.astype(np.float32) + rng.normal(0, 25.0, rw_mis.shape).astype(np.float32), 0, 255
        ).astype(np.uint8)
        refiner.refine(lw, noisy, lm, rm)
    assert np.allclose(refiner._m_frozen, frozen)


def test_feature_calibration_recovers_injected_geometry() -> None:
    """The feature-based solver must recover a known injected calibration
    from a synthetic overlapping pair (this is the regression guard for the
    2026-07-15 rewrite away from the photometric search that could not find
    the true ~22deg yaw / ~20deg relative roll)."""
    truth = pano.EyeCalibration(
        hfov_deg=78.0,
        yaw_left_deg=20.0,
        yaw_right_deg=-20.0,
        pitch_left_deg=10.0,
        pitch_right_deg=10.0,
        roll_left_deg=-8.0,
        roll_right_deg=8.0,
    )
    # Render several textured pairs through the truth calibration.
    pairs = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        virt = pano.default_virtual_camera(
            320, 240, truth.hfov_deg, yaw_out_deg=20.0, pitch_down_deg=10.0
        )
        texture = np.full((virt.height, virt.width, 3), 80, dtype=np.uint8)
        for _ in range(90):
            x0, y0 = rng.integers(0, virt.width - 15), rng.integers(0, virt.height - 15)
            w, h = int(rng.integers(8, 60)), int(rng.integers(8, 50))
            color = tuple(int(c) for c in rng.integers(30, 255, size=3))
            cv2.rectangle(texture, (int(x0), int(y0)), (int(x0) + w, int(y0) + h), color, -1)
        eye_k = pano.intrinsics(320, 240, truth.hfov_deg)
        rot_l, rot_r = pano._eye_rotations(truth)
        frames = []
        for rot in (rot_l, rot_r):
            h_matrix = virt.k() @ virt.rotation().T @ rot @ np.linalg.inv(eye_k)
            frames.append(
                cv2.warpPerspective(
                    texture,
                    h_matrix.astype(np.float64),
                    (320, 240),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                )
            )
        pairs.append((frames[0], frames[1]))

    solved = pano.solve_panorama_calibration(pairs, verbose=False)
    # Relative geometry is what matters (common terms are gauge).
    assert abs((solved.yaw_left_deg - solved.yaw_right_deg) - 40.0) < 4.0
    assert abs((solved.roll_left_deg - solved.roll_right_deg) - (-16.0)) < 5.0
    assert solved.meta.get("median_residual_deg", 99) < 1.0


def test_perception_mosaic_geometry_and_hard_cut() -> None:
    """The perception mosaic must place each eye's content on its own side of
    a centered seam, expose a wide-hfov camera model, and report coverage."""
    cal = pano.EyeCalibration()
    mosaic = pano.PerceptionMosaic(cal)
    left = np.full((240, 320, 3), (255, 0, 0), dtype=np.uint8)  # blue eye
    right = np.full((240, 320, 3), (0, 0, 255), dtype=np.uint8)  # red eye
    out = mosaic.compose(left, right)
    assert out.shape[:2] == (mosaic.virt.height, mosaic.virt.width)
    cov = mosaic.coverage()
    mid_row = mosaic.virt.height // 2
    left_cols = np.where(cov[mid_row, : mosaic.seam_col])[0]
    right_cols = np.where(cov[mid_row, mosaic.seam_col :])[0] + mosaic.seam_col
    assert len(left_cols) and len(right_cols)
    assert out[mid_row, left_cols[len(left_cols) // 2]][0] == 255  # blue left
    assert out[mid_row, right_cols[len(right_cols) // 2]][2] == 255  # red right
    assert mosaic.model.hfov_deg > 90.0  # wide central view
    c0, c1 = mosaic.seam_cols
    assert c0 < mosaic.seam_col < c1
    # Metric attitude is the field-calibrated truth, not the stitch gauge.
    assert abs(mosaic.model.pitch_down_deg - 13.9) < 1e-6
    assert abs(mosaic.model.vfov_deg - 66.0) < 1e-6


def test_panorama_streamer_exposes_jpeg_and_resize(monkeypatch) -> None:
    """The external streaming API must expose fused frames as JPEG bytes and
    honor output_size (the VR right-eye drop-in path)."""
    import time
    import sourccey_elevated_safety as ses

    class _FakeSub:
        def __init__(self, **k):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def latest(self, cam):
            return (np.random.rand(240, 320, 3) * 255).astype(np.uint8), 0.02

    monkeypatch.setattr(ses, "SlamCameraSubscriber", _FakeSub)
    monkeypatch.setattr(ses, "endpoint_from_remote_ip", lambda ip, port=5560: "tcp://x")

    s = pano.PanoramaStreamer("127.0.0.1", output_size=(320, 240), refine=False).start()
    try:
        deadline = time.monotonic() + 3.0
        jpeg = None
        while time.monotonic() < deadline and jpeg is None:
            jpeg = s.latest_jpeg(quality=80)
            time.sleep(0.02)
        assert jpeg is not None and jpeg[:2] == b"\xff\xd8"  # valid JPEG SOI
        bgr = s.latest_bgr()
        assert bgr is not None and bgr.shape == (240, 320, 3)  # resized drop-in
        assert s.frame_age_s is not None and s.frame_age_s < 2.0
    finally:
        s.stop()
