"""Calibrate the collision cameras' FOV/pitch against a lidar-known wall.

Setup: park the robot facing a bare wall section, 1-2.5 m away, with the
floor visible in front of it. No motion is commanded. The script:

  1. Measures the wall distance straight ahead from the lidar (median of
     returns within +-10 deg of forward).
  2. Finds the wall-floor boundary line in each eye camera and in the bottom
     camera (strongest long horizontal line below mid-frame).
  3. Solves each eye's vertical FOV from that one known floor point, and the
     bottom camera's effective pitch (mounting error from level).
  4. Prints the config values to use and saves annotated frames.

Run several times at different wall distances; consistent numbers mean the
calibration is good. Results feed CameraModel fields in
scripts/sourccey_camera_geometry.py (or the matching CLI flags).

  uv run python scripts/sourccey_camera_calibration.py \
    --remote-ip 192.168.1.237 --lidar-host 192.168.1.237
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
from ldlidar_defaults import DEFAULT_LIDAR_FORWARD_ANGLE_DEG
from ldlidar_direct_snapshot_client import DirectLidarFeed, _scan_to_local_points
from sourccey_bottom_camera import default_bottom_camera_model
from sourccey_camera_geometry import default_eye_left, default_eye_right
from sourccey_elevated_safety import SlamCameraSubscriber, endpoint_from_remote_ip


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--remote-ip", type=str, default="192.168.1.237")
    parser.add_argument("--slam-input-endpoint", type=str, default="")
    parser.add_argument("--lidar-host", type=str, default="192.168.1.237")
    parser.add_argument("--lidar-port", type=int, default=8765)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--forward-cone-deg", type=float, default=10.0)
    parser.add_argument(
        "--start-delay-s",
        type=float,
        default=5.0,
        help="Countdown before measuring, so you can step out of the cameras' view.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="artifacts/camera_calibration"
    )
    # For a PARTIAL wall (e.g. a table on its side covering only the center
    # of the robot's line of sight): analyze only the middle band of columns
    # and accept shorter lines. The eyes are yawed 15deg outward, so the
    # central wall sits in the INNER half of each eye's image.
    parser.add_argument(
        "--center-band-ratio",
        type=float,
        default=1.0,
        help="Fraction of image width (centered) to search for the wall-floor line. "
        "Use ~0.6 for a partial wall.",
    )
    parser.add_argument(
        "--min-line-width-ratio",
        type=float,
        default=0.45,
        help="Minimum line length as a fraction of the SEARCHED band width.",
    )
    parser.add_argument(
        "--min-row-ratio",
        type=float,
        default=0.35,
        help="Ignore lines above this row (walls farther away / elevated edges).",
    )
    return parser


def _forward_wall_distance_m(feed: DirectLidarFeed, cone_deg: float, samples: int) -> float | None:
    distances: list[float] = []
    last_frame_id = -1
    deadline = time.monotonic() + 8.0
    while len(distances) < samples and time.monotonic() < deadline:
        frame_id, frame = feed.latest()
        if frame is None or int(frame_id) == last_frame_id:
            time.sleep(0.05)
            continue
        last_frame_id = int(frame_id)
        points_xy = _scan_to_local_points(
            points=frame.points,
            forward_angle_deg=float(DEFAULT_LIDAR_FORWARD_ANGLE_DEG),
            valid_angle_half_width_deg=180.0,
            invert_lateral_axis=True,
            max_distance_m=8.0,
            min_confidence=0,
            min_range_m=0.30,
        )
        if len(points_xy) == 0:
            continue
        bearings = np.degrees(np.arctan2(points_xy[:, 1], points_xy[:, 0]))
        ranges = np.hypot(points_xy[:, 0], points_xy[:, 1])
        mask = np.abs(bearings) <= float(cone_deg)
        if not np.any(mask):
            continue
        distances.append(float(np.median(ranges[mask])))
    if not distances:
        return None
    distances.sort()
    return distances[len(distances) // 2]


def _find_floor_line_row(
    frame_bgr: np.ndarray,
    *,
    center_band_ratio: float = 1.0,
    min_line_width_ratio: float = 0.45,
    min_row_ratio: float = 0.35,
) -> float | None:
    """Row ratio of the strongest long horizontal line in the searched band,
    below min_row_ratio — expected to be the wall-floor boundary."""
    height, width = frame_bgr.shape[:2]
    band = float(np.clip(center_band_ratio, 0.2, 1.0))
    x0 = int(width * (0.5 - band / 2.0))
    x1 = int(width * (0.5 + band / 2.0))
    gray = cv2.cvtColor(frame_bgr[:, x0:x1], cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # White wall on light floor is a FAINT gradient: equalize first and use
    # low Canny thresholds, or the junction produces no edges at all.
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 25, 90)
    band_width = x1 - x0
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(int(band_width * 0.10), 16),
        minLineLength=int(band_width * float(min_line_width_ratio)),
        maxLineGap=int(band_width * 0.12),
    )
    if lines is None:
        return None
    # The wall-floor junction is by definition the LOWEST long horizontal
    # structure — preferring the lowest line also rejects the baseboard TOP
    # in favor of the baseboard-floor contact.
    best_row = None
    for lx1, ly1, lx2, ly2 in np.asarray(lines).reshape(-1, 4):
        dx = abs(float(lx2 - lx1))
        dy = abs(float(ly2 - ly1))
        if dx < 1.0 or dy / dx > 0.25:
            continue
        row = 0.5 * (ly1 + ly2) / max(height - 1, 1)
        if row < float(min_row_ratio) or row > 0.97:
            continue
        if best_row is None or row > best_row:
            best_row = float(row)
    return best_row


def _bottom_junction_row(
    frame_bgr: np.ndarray,
    *,
    expected_row: float,
    window: float = 0.17,
    min_line_width_ratio: float = 0.35,
) -> float | None:
    """Wall-floor junction row for the LEVEL bottom camera.

    Carpet generates full-width Hough lines at every row, so neither longest
    nor lowest works. The junction is the candidate with a SMOOTH band above
    (wall/baseboard) and a TEXTURED band below (carpet) — carpet decoys have
    texture on both sides. Field-validated on real frames 2026-07-09."""
    height, width = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 25, 90)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(int(width * 0.10), 16),
        minLineLength=int(width * float(min_line_width_ratio)),
        maxLineGap=int(width * 0.12),
    )
    if lines is None:
        return None
    row_min = float(expected_row) - float(window)
    row_max = float(expected_row) + float(window)
    density = (edges > 0).mean(axis=1)

    def band_density(r0: float, r1: float) -> float:
        a = int(max(0.0, r0) * (height - 1))
        b = int(min(1.0, r1) * (height - 1))
        return float(density[a:b].mean()) if b > a else 0.0

    best_row = None
    best_score = 0.0
    for lx1, ly1, lx2, ly2 in np.asarray(lines).reshape(-1, 4):
        dx = abs(float(lx2 - lx1))
        dy = abs(float(ly2 - ly1))
        if dx < 1.0 or dy / dx > 0.25:
            continue
        row = 0.5 * (ly1 + ly2) / max(height - 1, 1)
        if row < row_min or row > row_max:
            continue
        score = band_density(row + 0.03, row + 0.13) - band_density(row - 0.13, row - 0.03)
        if score > best_score:
            best_score = score
            best_row = float(row)
    return best_row


def _candidate_line_rows(
    frame_bgr: np.ndarray,
    *,
    center_band_ratio: float = 1.0,
    min_line_width_ratio: float = 0.45,
    min_row_ratio: float = 0.35,
) -> list[float]:
    """All qualifying near-horizontal line rows (deduplicated). The caller
    picks the true junction by CROSS-EYE FOV agreement — a low-contrast
    floor produces many noise lines, but only the real junction implies the
    same field of view in both (identical) eye cameras."""
    height, width = frame_bgr.shape[:2]
    band = float(np.clip(center_band_ratio, 0.2, 1.0))
    x0 = int(width * (0.5 - band / 2.0))
    x1 = int(width * (0.5 + band / 2.0))
    gray = cv2.cvtColor(frame_bgr[:, x0:x1], cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 25, 90)
    band_width = x1 - x0
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(int(band_width * 0.10), 16),
        minLineLength=int(band_width * float(min_line_width_ratio)),
        maxLineGap=int(band_width * 0.12),
    )
    if lines is None:
        return []
    rows: list[float] = []
    for lx1, ly1, lx2, ly2 in np.asarray(lines).reshape(-1, 4):
        dx = abs(float(lx2 - lx1))
        dy = abs(float(ly2 - ly1))
        if dx < 1.0 or dy / dx > 0.25:
            continue
        row = 0.5 * (ly1 + ly2) / max(height - 1, 1)
        if row < float(min_row_ratio) or row > 0.97:
            continue
        if all(abs(row - existing) > 0.01 for existing in rows):
            rows.append(float(row))
    return sorted(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    endpoint = str(args.slam_input_endpoint).strip() or endpoint_from_remote_ip(args.remote_ip)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    delay_s = max(0.0, float(args.start_delay_s))
    if delay_s > 0:
        for remaining in range(int(math.ceil(delay_s)), 0, -1):
            print(f"[calibrate] starting in {remaining}s — clear the cameras' view of the wall")
            time.sleep(1.0)

    print(f"[calibrate] lidar: measuring forward wall distance ({args.lidar_host})")
    feed = DirectLidarFeed(str(args.lidar_host), int(args.lidar_port))
    feed.start()
    wall_from_lidar_m = _forward_wall_distance_m(
        feed, float(args.forward_cone_deg), int(args.samples)
    )
    feed.stop()
    if wall_from_lidar_m is None:
        print("[calibrate] ERROR: no lidar returns straight ahead — is the stream running?")
        return 2
    print(f"[calibrate] wall distance from lidar: {wall_from_lidar_m:.3f}m")

    print(f"[calibrate] subscribing to cameras ({endpoint})")
    subscriber = SlamCameraSubscriber(
        endpoint=endpoint, camera_keys=("front_left", "front_right", "bottom")
    )
    subscriber.start()
    if not subscriber.wait_for_frames(timeout_s=6.0, required=("front_left", "front_right")):
        print("[calibrate] ERROR: no eye camera frames from the slam_input stream")
        subscriber.stop()
        return 2

    models = {
        "front_left": default_eye_left(),
        "front_right": default_eye_right(),
        "bottom": default_bottom_camera_model(),
    }
    # The lidar sits ~0.229m ahead of robot center; each camera has its own
    # forward offset — correct the wall distance into each camera's frame.
    lidar_forward_offset_m = 0.229

    results: dict[str, dict] = {}
    frames: dict[str, np.ndarray] = {}
    eye_rows: dict[str, list[float]] = {}
    for cam_name in models:
        frame, age = subscriber.latest(cam_name)
        if frame is None:
            print(f"[calibrate] {cam_name}: no frames (skipping — enable it on the host?)")
            continue
        if float(np.mean(frame)) < 6.0:
            cv2.imwrite(str(output_dir / f"calibration_{cam_name}.jpg"), frame)
            print(
                f"[calibrate] {cam_name}: frame is BLACK (mean={float(np.mean(frame)):.1f}) — "
                "the camera is not delivering an image. Check the device/lens/exposure "
                "on the Pi host; calibration for this camera is skipped."
            )
            continue
        frames[cam_name] = frame
        if cam_name != "bottom":
            eye_rows[cam_name] = _candidate_line_rows(
                frame,
                center_band_ratio=float(args.center_band_ratio),
                min_line_width_ratio=float(args.min_line_width_ratio),
                min_row_ratio=float(args.min_row_ratio),
            )

    # ---- eyes: pick the junction by CROSS-EYE FOV AGREEMENT ------------------
    # A low-contrast floor yields many candidate lines per eye, but the two
    # eyes are identical cameras: only the true wall-floor junction implies
    # the SAME vertical FOV in both. Noise lines scatter.
    left_model = models["front_left"]
    right_model = models["front_right"]
    wall_left_m = wall_from_lidar_m + lidar_forward_offset_m - float(left_model.forward_offset_m)
    wall_right_m = wall_from_lidar_m + lidar_forward_offset_m - float(right_model.forward_offset_m)
    expected_junction_row = left_model.row_for_depression_deg(
        math.degrees(math.atan2(left_model.height_m, wall_left_m))
    )
    print(
        f"[calibrate] current model predicts the junction at cy={expected_junction_row:.3f} "
        "— the picked rows should land near this if the model is already right"
    )
    agreeing_pairs = []  # (vfov_diff, left_row, right_row, vfov_left, vfov_right)
    for left_row in eye_rows.get("front_left", []):
        vfov_left = left_model.vfov_from_floor_observation(left_row, wall_left_m)
        if vfov_left is None or not (28.0 <= vfov_left <= 85.0):
            continue
        for right_row in eye_rows.get("front_right", []):
            vfov_right = right_model.vfov_from_floor_observation(right_row, wall_right_m)
            if vfov_right is None or not (28.0 <= vfov_right <= 85.0):
                continue
            diff = abs(vfov_left - vfov_right)
            agreeing_pairs.append((diff, left_row, right_row, vfov_left, vfov_right))
    best_pair = None
    if agreeing_pairs:
        # Wall seams/outlets also agree across eyes; the floor junction is the
        # LOWEST agreeing structure. Among pairs close to the best agreement,
        # prefer the lowest one.
        min_diff = min(pair[0] for pair in agreeing_pairs)
        contenders = [pair for pair in agreeing_pairs if pair[0] <= min_diff + 0.75]
        best_pair = max(contenders, key=lambda pair: pair[1] + pair[2])

    for cam_name in ("front_left", "front_right"):
        if cam_name not in frames:
            continue
        frame = frames[cam_name]
        annotated = frame.copy()
        for row in eye_rows.get(cam_name, []):
            y_px = int(row * (frame.shape[0] - 1))
            cv2.line(annotated, (0, y_px), (frame.shape[1], y_px), (0, 180, 255), 1)
        if best_pair is not None:
            picked = best_pair[1] if cam_name == "front_left" else best_pair[2]
            y_px = int(picked * (frame.shape[0] - 1))
            cv2.line(annotated, (0, y_px), (frame.shape[1], y_px), (0, 0, 255), 2)
        cv2.imwrite(str(output_dir / f"calibration_{cam_name}.jpg"), annotated)

    if best_pair is None:
        print(
            "[calibrate] eyes: no junction line found in one or both eyes "
            f"(candidates: left={len(eye_rows.get('front_left', []))}, "
            f"right={len(eye_rows.get('front_right', []))}) — bare wall needed"
        )
    elif best_pair[0] > 2.5:
        print(
            f"[calibrate] eyes: best cross-eye match disagrees by {best_pair[0]:.1f}deg "
            f"(left cy={best_pair[1]:.3f}->{best_pair[3]:.1f}deg, "
            f"right cy={best_pair[2]:.3f}->{best_pair[4]:.1f}deg) — not trustworthy; "
            "check the annotated frames and retry"
        )
    else:
        vfov_agreed = 0.5 * (best_pair[3] + best_pair[4])
        results["eyes"] = {"suggested_vfov_deg": vfov_agreed}
        print(
            f"[calibrate] eyes AGREE: junction left cy={best_pair[1]:.3f} / right "
            f"cy={best_pair[2]:.3f} -> vfov_deg={vfov_agreed:.2f} "
            f"(spread {best_pair[0]:.2f}deg; modeled {left_model.vfov_deg:.1f}) — "
            "verify the RED line sits on the wall-floor junction in both frames"
        )

    # ---- bottom camera: single lowest line + assumed vfov -> pitch -----------
    if "bottom" in frames:
        bottom_model = models["bottom"]
        frame = frames["bottom"]
        wall_bottom_probe_m = wall_from_lidar_m + lidar_forward_offset_m - float(
            bottom_model.forward_offset_m
        )
        expected_row = bottom_model.row_for_depression_deg(
            math.degrees(math.atan2(bottom_model.height_m, wall_bottom_probe_m))
        )
        row = _bottom_junction_row(frame, expected_row=float(expected_row))
        annotated = frame.copy()
        if row is not None:
            y_px = int(row * (frame.shape[0] - 1))
            cv2.line(annotated, (0, y_px), (frame.shape[1], y_px), (0, 0, 255), 2)
        cv2.imwrite(str(output_dir / "calibration_bottom.jpg"), annotated)
        if row is None:
            print("[calibrate] bottom: no clear wall-floor line found")
        else:
            wall_bottom_m = wall_from_lidar_m + lidar_forward_offset_m - float(
                bottom_model.forward_offset_m
            )
            depression_deg = math.degrees(math.atan2(bottom_model.height_m, wall_bottom_m))
            pitch_deg = depression_deg - (row - 0.5) * bottom_model.vfov_deg
            results["bottom"] = {"row": row, "suggested_pitch_down_deg": pitch_deg}
            print(
                f"[calibrate] bottom: floor line at cy={row:.3f} -> "
                f"suggested pitch_down_deg={pitch_deg:.2f} (vs modeled "
                f"{bottom_model.pitch_down_deg:.2f}; assumes vfov={bottom_model.vfov_deg:.1f})"
            )

    subscriber.stop()
    print(f"[calibrate] annotated frames saved to {output_dir}")
    if results:
        print(
            "[calibrate] apply the suggested values to the CameraModel defaults in "
            "scripts/sourccey_camera_geometry.py, then re-run at a different wall "
            "distance to confirm they stay consistent"
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
