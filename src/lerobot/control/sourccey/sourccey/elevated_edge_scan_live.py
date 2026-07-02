from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClient, SourcceyClientConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.control.sourccey.sourccey.survey_rotation_protocol import (
    _apply_min_effective_magnitude,
    _build_action as _survey_build_action,
    _run_action_for_duration,
)


@dataclass
class ElevatedEdgeScanConfig:
    id: str = "sourccey"
    remote_ip: str = "127.0.0.1"
    fps: int = 15
    left_key: str = "front_left"
    right_key: str = "front_right"
    turn_gain: float = 1.55
    min_turn_speed_rad_s: float = 0.72
    max_turn_speed_rad_s: float = 0.98
    center_deadband_ratio: float = 0.14
    min_line_width_ratio: float = 0.36
    min_line_score: float = 860.0
    single_eye_min_line_score: float = 1040.0
    min_shadow_contrast: float = 16.0
    min_vertical_edge_strength: float = 18.0
    min_upper_band_brightness: float = 126.0
    min_top_surface_support_ratio: float = 0.68
    min_lower_shadow_support_ratio: float = 0.60
    min_center_y_ratio: float = 0.46
    max_center_y_ratio: float = 0.82
    min_abs_slope: float = 0.0
    max_abs_slope: float = 1.25
    pair_center_y_tolerance_ratio: float = 0.10
    pair_distance_tolerance_m: float = 0.40
    near_distance_m: float = 0.9144
    far_distance_m: float = 2.7432
    camera_half_width_m: float = 0.07
    camera_yaw_deg: float = 15.0
    z_hold_pos: float | None = None
    search_preview_height_px: int = 220
    status_log_interval_s: float = 1.0
    search_turn_speed_rad_s: float = 0.42
    search_turn_direction: str = "right"
    stable_hits_required: int = 2
    command_hold_s: float = 0.18
    hold_confirm_frames: int = 3
    hold_dwell_s: float = 0.9
    lock_release_missing_frames: int = 6
    lock_max_single_eye_distance_m: float = 1.8
    lock_prefer_paired: bool = True
    viewer_dir: str | None = None
    viewer_write_interval_s: float = 0.4


@dataclass
class EdgeObservation:
    detected: bool
    score: float
    bbox: tuple[int, int, int, int] | None
    line_xy: tuple[tuple[int, int], tuple[int, int]] | None
    center_x_ratio: float | None
    center_y_ratio: float | None
    estimated_distance_m: float | None
    shadow_contrast: float
    vertical_edge_strength: float
    reason: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Native Sourccey elevated-edge scan with direct base turning."
    )
    parser.add_argument("--id", type=str, default="sourccey")
    parser.add_argument("--remote_ip", type=str, default="127.0.0.1")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--left_key", type=str, default="front_left")
    parser.add_argument("--right_key", type=str, default="front_right")
    parser.add_argument("--turn_gain", type=float, default=1.55)
    parser.add_argument("--min_turn_speed_rad_s", type=float, default=0.72)
    parser.add_argument("--max_turn_speed_rad_s", type=float, default=0.98)
    parser.add_argument("--center_deadband_ratio", type=float, default=0.14)
    parser.add_argument("--min_line_width_ratio", type=float, default=0.36)
    parser.add_argument("--min_line_score", type=float, default=860.0)
    parser.add_argument("--single_eye_min_line_score", type=float, default=1040.0)
    parser.add_argument("--min_shadow_contrast", type=float, default=22.0)
    parser.add_argument("--min_vertical_edge_strength", type=float, default=26.0)
    parser.add_argument("--min_upper_band_brightness", type=float, default=126.0)
    parser.add_argument("--min_top_surface_support_ratio", type=float, default=0.68)
    parser.add_argument("--min_lower_shadow_support_ratio", type=float, default=0.60)
    parser.add_argument("--min_center_y_ratio", type=float, default=0.46)
    parser.add_argument("--max_center_y_ratio", type=float, default=0.82)
    parser.add_argument("--min_abs_slope", type=float, default=0.0)
    parser.add_argument("--max_abs_slope", type=float, default=1.25)
    parser.add_argument("--pair_center_y_tolerance_ratio", type=float, default=0.10)
    parser.add_argument("--pair_distance_tolerance_m", type=float, default=0.40)
    parser.add_argument("--near_distance_m", type=float, default=0.9144)
    parser.add_argument("--far_distance_m", type=float, default=2.7432)
    parser.add_argument("--camera_half_width_m", type=float, default=0.07)
    parser.add_argument("--camera_yaw_deg", type=float, default=15.0)
    parser.add_argument("--z_hold_pos", type=float, default=None)
    parser.add_argument("--status_log_interval_s", type=float, default=1.0)
    parser.add_argument("--search_turn_speed_rad_s", type=float, default=0.42)
    parser.add_argument("--search_turn_direction", type=str, choices=("left", "right"), default="right")
    parser.add_argument("--stable_hits_required", type=int, default=2)
    parser.add_argument("--command_hold_s", type=float, default=0.18)
    parser.add_argument("--hold_confirm_frames", type=int, default=3)
    parser.add_argument("--hold_dwell_s", type=float, default=0.9)
    parser.add_argument("--lock_release_missing_frames", type=int, default=6)
    parser.add_argument("--lock_max_single_eye_distance_m", type=float, default=1.8)
    parser.add_argument("--lock_prefer_paired", type=lambda v: str(v).lower() not in ("false", "0", "no"), default=True)
    parser.add_argument("--viewer_dir", type=str, default=None)
    parser.add_argument("--viewer_write_interval_s", type=float, default=0.4)
    return parser


def _connect_with_retry(robot: SourcceyClient, delay_s: float = 0.25) -> None:
    attempt = 0
    while True:
        attempt += 1
        try:
            robot.connect()
            print(f"Elevated edge scan connected after {attempt} attempt(s).")
            return
        except Exception as exc:
            print(f"Elevated edge scan connect attempt {attempt} failed: {exc}")
            time.sleep(delay_s)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _determine_z_hold_pos(
    cfg: ElevatedEdgeScanConfig,
    observation: dict[str, object] | None,
    robot: SourcceyClient,
) -> float:
    if cfg.z_hold_pos is not None:
        return float(cfg.z_hold_pos)
    if isinstance(observation, dict) and "z.pos" in observation:
        return _safe_float(observation.get("z.pos"), 0.0)
    return float(getattr(robot, "_z_pos_cmd", 0.0))


def _build_action(*, theta_vel_rad_s: float, z_hold_pos: float) -> dict[str, float | bool]:
    return _survey_build_action(
        x_vel_m_s=0.0,
        theta_vel_rad_s=float(theta_vel_rad_s),
        z_hold_pos=float(z_hold_pos),
    )


def _send_motion_command(
    *,
    robot: SourcceyClient,
    theta_vel_rad_s: float,
    z_hold_pos: float,
    fps: int,
    hold_s: float,
    minimum_abs_turn_speed_rad_s: float,
) -> None:
    theta_cmd = float(theta_vel_rad_s)
    if abs(theta_cmd) > 1e-6:
        theta_cmd = _apply_min_effective_magnitude(
            theta_cmd,
            minimum_abs=minimum_abs_turn_speed_rad_s,
        )
    action = _build_action(theta_vel_rad_s=theta_cmd, z_hold_pos=z_hold_pos)
    _run_action_for_duration(
        robot=robot,
        action=action,
        duration_s=max(float(hold_s), 0.0),
        fps=max(int(fps), 1),
    )


class ElevatedEdgeDetector:
    def __init__(self, config: ElevatedEdgeScanConfig) -> None:
        self.config = config

    def detect(self, frame_bgr: np.ndarray) -> EdgeObservation:
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[0] < 4 or frame.shape[1] < 4:
            return EdgeObservation(False, 0.0, None, None, None, None, None, 0.0, 0.0, "invalid_frame")

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.equalizeHist(gray)

        y0 = int(height * self.config.min_center_y_ratio)
        y1 = int(height * self.config.max_center_y_ratio)
        if y1 <= y0 + 8:
            return EdgeObservation(False, 0.0, None, None, None, None, None, 0.0, 0.0, "bad_roi")

        roi = gray[y0:y1, :]
        edges = cv2.Canny(roi, 60, 140, apertureSize=3, L2gradient=True)
        min_len = max(int(width * self.config.min_line_width_ratio), 32)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=max(int(width * 0.10), 16),
            minLineLength=min_len,
            maxLineGap=max(int(width * 0.08), 24),
        )
        if lines is None or len(lines) == 0:
            return EdgeObservation(False, 0.0, None, None, None, None, None, 0.0, 0.0, "no_line")

        sobel_y = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        abs_sobel_y = np.abs(sobel_y)
        best: EdgeObservation | None = None

        for line in lines[:, 0, :]:
            x1, y1_local, x2, y2_local = [int(v) for v in line]
            if x2 < x1:
                x1, x2 = x2, x1
                y1_local, y2_local = y2_local, y1_local
            dx = float(x2 - x1)
            if dx < float(min_len):
                continue

            dy = float(y2_local - y1_local)
            slope = dy / max(dx, 1.0)
            abs_slope = abs(slope)
            if abs_slope < self.config.min_abs_slope or abs_slope > self.config.max_abs_slope:
                continue

            center_x = 0.5 * (x1 + x2)
            center_y_local = 0.5 * (y1_local + y2_local)
            center_y_ratio = float((y0 + center_y_local) / max(height - 1, 1))
            if center_y_ratio < self.config.min_center_y_ratio or center_y_ratio > self.config.max_center_y_ratio:
                continue

            x_margin = max(int(dx * 0.08), 8)
            xs0 = int(np.clip(x1 + x_margin, 0, width - 1))
            xs1 = int(np.clip(x2 - x_margin, 0, width - 1))
            if xs1 <= xs0 + 8:
                continue

            band_h = max(int(height * 0.018), 4)
            line_y_mid = int(round(center_y_local))
            upper0 = int(np.clip(line_y_mid - (2 * band_h), 0, roi.shape[0] - 1))
            upper1 = int(np.clip(line_y_mid - band_h, 0, roi.shape[0]))
            lower0 = int(np.clip(line_y_mid + band_h, 0, roi.shape[0] - 1))
            lower1 = int(np.clip(line_y_mid + (2 * band_h), 0, roi.shape[0]))
            if upper1 <= upper0 or lower1 <= lower0:
                continue

            upper_band = roi[upper0:upper1, xs0:xs1]
            lower_band = roi[lower0:lower1, xs0:xs1]
            if upper_band.size == 0 or lower_band.size == 0:
                continue

            upper_mean = float(np.mean(upper_band))
            lower_mean = float(np.mean(lower_band))
            shadow_contrast = upper_mean - lower_mean
            if shadow_contrast < self.config.min_shadow_contrast:
                continue
            if upper_mean < self.config.min_upper_band_brightness:
                continue

            top_surface_support_ratio = float(np.mean(upper_band >= (upper_mean - 8.0)))
            lower_shadow_support_ratio = float(np.mean(lower_band <= (lower_mean + 8.0)))
            if top_surface_support_ratio < self.config.min_top_surface_support_ratio:
                continue
            if lower_shadow_support_ratio < self.config.min_lower_shadow_support_ratio:
                continue

            edge_y0 = int(np.clip(line_y_mid - band_h, 0, roi.shape[0] - 1))
            edge_y1 = int(np.clip(line_y_mid + band_h + 1, 0, roi.shape[0]))
            edge_window = abs_sobel_y[edge_y0:edge_y1, xs0:xs1]
            if edge_window.size == 0:
                continue
            vertical_edge_strength = float(np.mean(edge_window))
            if vertical_edge_strength < self.config.min_vertical_edge_strength:
                continue

            width_score = dx * 1.35
            contrast_score = shadow_contrast * 18.0
            edge_score = vertical_edge_strength * 9.0
            support_score = (top_surface_support_ratio * 160.0) + (lower_shadow_support_ratio * 120.0)
            lower_bias = 180.0 * max(0.0, center_y_ratio - self.config.min_center_y_ratio)
            center_pref = 140.0 * (1.0 - abs(center_y_ratio - 0.68))
            score = width_score + contrast_score + edge_score + support_score + lower_bias + center_pref

            bbox_y0 = int(np.clip(y0 + line_y_mid - (3 * band_h), 0, height - 1))
            bbox_y1 = int(np.clip(y0 + line_y_mid + (3 * band_h), 0, height))
            bbox = (xs0, bbox_y0, xs1, bbox_y1)
            line_xy = ((x1, y0 + y1_local), (x2, y0 + y2_local))
            distance_ratio = float(np.clip((center_y_ratio - self.config.min_center_y_ratio) / max(self.config.max_center_y_ratio - self.config.min_center_y_ratio, 1e-6), 0.0, 1.0))
            estimated_distance_m = float(
                self.config.far_distance_m
                - distance_ratio * (self.config.far_distance_m - self.config.near_distance_m)
            )

            candidate = EdgeObservation(
                detected=score >= self.config.min_line_score,
                score=score,
                bbox=bbox,
                line_xy=line_xy,
                center_x_ratio=float(center_x / max(width - 1, 1)),
                center_y_ratio=center_y_ratio,
                estimated_distance_m=estimated_distance_m,
                shadow_contrast=shadow_contrast,
                vertical_edge_strength=vertical_edge_strength,
                reason="ok" if score >= self.config.min_line_score else "score_low",
            )
            if best is None or candidate.score > best.score:
                best = candidate

        if best is None:
            return EdgeObservation(False, 0.0, None, None, None, None, None, 0.0, 0.0, "filtered_out")
        return best


def _update_stable_hits(observation: EdgeObservation, previous_hits: int) -> int:
    if observation.detected:
        return previous_hits + 1
    return 0


def _merge_paired_observations(left: EdgeObservation, right: EdgeObservation) -> EdgeObservation:
    bbox = None
    if left.bbox is not None and right.bbox is not None:
        bbox = (
            min(left.bbox[0], right.bbox[0]),
            min(left.bbox[1], right.bbox[1]),
            max(left.bbox[2], right.bbox[2]),
            max(left.bbox[3], right.bbox[3]),
        )
    line_xy = left.line_xy if (left.score >= right.score) else right.line_xy
    center_x_ratio = None
    if left.center_x_ratio is not None and right.center_x_ratio is not None:
        center_x_ratio = 0.5 * (left.center_x_ratio + right.center_x_ratio)
    elif left.center_x_ratio is not None:
        center_x_ratio = left.center_x_ratio
    else:
        center_x_ratio = right.center_x_ratio
    center_y_ratio = None
    if left.center_y_ratio is not None and right.center_y_ratio is not None:
        center_y_ratio = 0.5 * (left.center_y_ratio + right.center_y_ratio)
    elif left.center_y_ratio is not None:
        center_y_ratio = left.center_y_ratio
    else:
        center_y_ratio = right.center_y_ratio
    estimated_distance_m = None
    if left.estimated_distance_m is not None and right.estimated_distance_m is not None:
        estimated_distance_m = 0.5 * (left.estimated_distance_m + right.estimated_distance_m)
    elif left.estimated_distance_m is not None:
        estimated_distance_m = left.estimated_distance_m
    else:
        estimated_distance_m = right.estimated_distance_m
    return EdgeObservation(
        detected=True,
        score=max(left.score, right.score),
        bbox=bbox,
        line_xy=line_xy,
        center_x_ratio=center_x_ratio,
        center_y_ratio=center_y_ratio,
        estimated_distance_m=estimated_distance_m,
        shadow_contrast=max(left.shadow_contrast, right.shadow_contrast),
        vertical_edge_strength=max(left.vertical_edge_strength, right.vertical_edge_strength),
        reason="paired",
    )


def _choose_target(
    left: EdgeObservation,
    right: EdgeObservation,
    cfg: ElevatedEdgeScanConfig,
    *,
    left_stable_hits: int,
    right_stable_hits: int,
) -> tuple[str, EdgeObservation | None]:
    required = max(int(cfg.stable_hits_required), 1)
    if left.detected and right.detected:
        cy_ok = (
            left.center_y_ratio is not None
            and right.center_y_ratio is not None
            and abs(left.center_y_ratio - right.center_y_ratio) <= cfg.pair_center_y_tolerance_ratio
        )
        dist_ok = (
            left.estimated_distance_m is not None
            and right.estimated_distance_m is not None
            and abs(left.estimated_distance_m - right.estimated_distance_m) <= cfg.pair_distance_tolerance_m
        )
        if cy_ok and dist_ok and left_stable_hits >= required and right_stable_hits >= required:
            return ("paired", _merge_paired_observations(left, right))
        if left.score >= cfg.single_eye_min_line_score and left.score >= right.score and left_stable_hits >= required:
            return ("left_strong", left)
        if right.score >= cfg.single_eye_min_line_score and right_stable_hits >= required:
            return ("right_strong", right)
        return ("reject_pair", None)
    if left.detected and left.score >= cfg.single_eye_min_line_score and left_stable_hits >= required:
        return ("left_single", left)
    if right.detected and right.score >= cfg.single_eye_min_line_score and right_stable_hits >= required:
        return ("right_single", right)
    return ("none", None)


def _target_is_lockable(choice: str, target: EdgeObservation | None, cfg: ElevatedEdgeScanConfig) -> bool:
    if target is None:
        return False
    if choice == "paired":
        return True
    if bool(cfg.lock_prefer_paired) and choice not in {"left_strong", "right_strong"}:
        return False
    if target.estimated_distance_m is not None and target.estimated_distance_m > float(cfg.lock_max_single_eye_distance_m):
        return False
    return choice in {"left_strong", "right_strong", "left_single", "right_single"}


def _target_turn_command(
    target: EdgeObservation | None,
    cfg: ElevatedEdgeScanConfig,
) -> tuple[str, float]:
    if target is None or target.center_x_ratio is None:
        direction = -1.0 if str(cfg.search_turn_direction).lower() == "left" else 1.0
        return ("search", direction * float(cfg.search_turn_speed_rad_s))
    offset = float(target.center_x_ratio - 0.5)
    if abs(offset) <= cfg.center_deadband_ratio:
        return ("hold", 0.0)
    desired = float(np.clip(offset * cfg.turn_gain, -cfg.max_turn_speed_rad_s, cfg.max_turn_speed_rad_s))
    if abs(desired) < cfg.min_turn_speed_rad_s:
        desired = math.copysign(cfg.min_turn_speed_rad_s, desired)
    direction = "right" if desired > 0.0 else "left"
    return (f"turn_{direction}", desired)


def _accumulate_points(
    points: list[tuple[float, float]],
    observation: EdgeObservation,
    eye_name: str,
    heading_rad: float,
    cfg: ElevatedEdgeScanConfig,
) -> None:
    if not observation.detected or observation.line_xy is None or observation.estimated_distance_m is None:
        return
    (x1, _), (x2, _) = observation.line_xy
    width_px = max(x2 - x1, 1)
    sample_count = max(int(width_px / 14), 4)
    offsets = np.linspace(-0.18, 0.18, num=sample_count)
    eye_yaw_rad = math.radians(-cfg.camera_yaw_deg if eye_name == cfg.left_key else cfg.camera_yaw_deg)
    eye_x = -cfg.camera_half_width_m if eye_name == cfg.left_key else cfg.camera_half_width_m
    distance = float(observation.estimated_distance_m)
    for lateral in offsets:
        local_x = eye_x + lateral
        local_z = distance
        yaw = heading_rad + eye_yaw_rad
        world_x = (local_x * math.cos(yaw)) + (local_z * math.sin(yaw))
        world_z = (local_z * math.cos(yaw)) - (local_x * math.sin(yaw))
        points.append((world_x, world_z))


def _draw_observation(
    frame_bgr: np.ndarray,
    eye_name: str,
    observation: EdgeObservation,
    active: bool,
    *,
    rejected: bool,
    choice: str,
    target_center_x_ratio: float | None,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    if active:
        color = (0, 255, 0)
    elif rejected and observation.detected:
        color = (0, 140, 255)
    elif observation.detected:
        color = (80, 220, 255)
    else:
        color = (0, 180, 255)
    if observation.bbox is not None:
        x0, y0, x1, y1 = observation.bbox
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
    if observation.line_xy is not None:
        cv2.line(canvas, observation.line_xy[0], observation.line_xy[1], (0, 220, 220), 3)
    if target_center_x_ratio is not None:
        marker_x = int(round(target_center_x_ratio * max(canvas.shape[1] - 1, 1)))
        cv2.line(canvas, (marker_x, 0), (marker_x, canvas.shape[0] - 1), (255, 0, 255), 2)
        cv2.circle(canvas, (marker_x, canvas.shape[0] // 2), 6, (255, 0, 255), -1)
    label = "detected" if observation.detected else observation.reason
    active_label = "active" if active else ("rejected" if rejected and observation.detected else "idle")
    lines = [
        f"{eye_name} status={label} {active_label}",
        f"choice={choice}",
        f"score={observation.score:.1f}",
        f"contrast={observation.shadow_contrast:.1f} edge_y={observation.vertical_edge_strength:.1f}",
        f"dist={observation.estimated_distance_m if observation.estimated_distance_m is not None else float('nan'):.2f}m",
    ]
    y = 26
    for text in lines:
        cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
        y += 30
    return canvas



def _ensure_viewer_dir(viewer_dir: str | None) -> Path | None:
    if viewer_dir is None or not str(viewer_dir).strip():
        return None
    path = Path(viewer_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_viewer_files(
    *,
    viewer_dir: Path | None,
    left_canvas: np.ndarray,
    right_canvas: np.ndarray,
    map_canvas: np.ndarray,
    state: str,
    choice: str,
    theta_cmd: float,
    observed_theta: float,
    target_dist: float | None,
    lock_active: bool,
    hold_frames: int,
) -> None:
    if viewer_dir is None:
        return
    left_path = viewer_dir / 'left_latest.jpg'
    right_path = viewer_dir / 'right_latest.jpg'
    map_path = viewer_dir / 'map_latest.jpg'
    html_path = viewer_dir / 'index.html'
    status_path = viewer_dir / 'status.txt'
    cv2.imwrite(str(left_path), left_canvas)
    cv2.imwrite(str(right_path), right_canvas)
    cv2.imwrite(str(map_path), map_canvas)
    status_text = (
        f'state={state}\n'
        f'choice={choice}\n'
        f'theta_cmd={theta_cmd:+.2f}\n'
        f'observed_theta={observed_theta:+.2f}\n'
        f'target_dist_m={target_dist}\n'
        f'lock_active={lock_active}\n'
        f'hold_frames={hold_frames}\n'
    )
    status_path.write_text(status_text, encoding='utf-8')
    timestamp = int(time.time())
    html = f'''<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="1">
  <title>Elevated Edge Scan Viewer</title>
  <style>
    body {{ background:#111; color:#eee; font-family:Arial,sans-serif; margin:16px; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
    img {{ width:100%; border:1px solid #444; background:#000; }}
    pre {{ background:#1a1a1a; padding:12px; border:1px solid #444; }}
  </style>
</head>
<body>
  <h2>Elevated Edge Scan Viewer</h2>
  <pre>{status_text}</pre>
  <div class="grid">
    <div><h3>Left Eye</h3><img src="left_latest.jpg?ts={timestamp}" /></div>
    <div><h3>Right Eye</h3><img src="right_latest.jpg?ts={timestamp}" /></div>
    <div style="grid-column:1 / span 2;"><h3>Map</h3><img src="map_latest.jpg?ts={timestamp}" /></div>
  </div>
</body>
</html>'''
    html_path.write_text(html, encoding='utf-8')


def _draw_map(
    points: list[tuple[float, float]],
    heading_rad: float,
    state: str,
    theta_cmd: float,
    *,
    choice: str,
    lock_active: bool,
    lockable_target: bool,
) -> np.ndarray:
    width, height = 960, 340
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(canvas, "elevated_edge_scan_map", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"state={state} theta={theta_cmd:+.2f} heading_deg={math.degrees(heading_rad):.1f}", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"choice={choice} lock_active={lock_active} lockable_target={lockable_target}", (16, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    origin = (width // 2, height - 44)
    cv2.circle(canvas, origin, 8, (0, 140, 255), -1)
    cv2.putText(canvas, "robot", (origin[0] + 10, origin[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2, cv2.LINE_AA)
    scale = 135.0
    if points:
        for px, pz in points[-1800:]:
            sx = int(round(origin[0] + (px * scale)))
            sy = int(round(origin[1] - (pz * scale)))
            if 0 <= sx < width and 0 <= sy < height:
                canvas[sy, sx] = (255, 255, 255)
    head_x = int(round(origin[0] + math.sin(heading_rad) * 40.0))
    head_y = int(round(origin[1] - math.cos(heading_rad) * 40.0))
    cv2.line(canvas, origin, (head_x, head_y), (0, 220, 220), 3)
    return canvas


def elevated_edge_scan_live(cfg: ElevatedEdgeScanConfig) -> int:
    robot = SourcceyClient(SourcceyClientConfig(remote_ip=cfg.remote_ip, id=cfg.id))
    _connect_with_retry(robot)
    robot.untorque_left_active = True
    robot.untorque_right_active = True
    detector = ElevatedEdgeDetector(cfg)
    observation_cache: dict[str, object] = {}
    map_points: list[tuple[float, float]] = []
    heading_rad = 0.0
    last_loop_time = time.perf_counter()
    preview_available = True
    preview_warning_printed = False
    last_status_log_s = 0.0
    last_viewer_write_s = 0.0
    left_stable_hits = 0
    right_stable_hits = 0
    hold_stable_frames = 0
    hold_until_s = 0.0
    lock_active = False
    lock_missing_frames = 0
    viewer_dir = _ensure_viewer_dir(cfg.viewer_dir)

    try:
        print(
            "Elevated edge scan started "
            f"left_key={cfg.left_key} right_key={cfg.right_key} "
            f"min_line_score={cfg.min_line_score:.1f} "
            f"single_eye_min_line_score={cfg.single_eye_min_line_score:.1f}"
        )
        while True:
            loop_started = time.perf_counter()
            dt = max(loop_started - last_loop_time, 1e-3)
            last_loop_time = loop_started
            observation_cache = robot.get_observation()

            left_frame = observation_cache.get(cfg.left_key)
            right_frame = observation_cache.get(cfg.right_key)
            if not isinstance(left_frame, np.ndarray) or not isinstance(right_frame, np.ndarray):
                precise_sleep(1.0 / max(cfg.fps, 1))
                continue

            left_obs = detector.detect(left_frame)
            right_obs = detector.detect(right_frame)
            left_stable_hits = _update_stable_hits(left_obs, left_stable_hits)
            right_stable_hits = _update_stable_hits(right_obs, right_stable_hits)
            choice, target = _choose_target(
                left_obs,
                right_obs,
                cfg,
                left_stable_hits=left_stable_hits,
                right_stable_hits=right_stable_hits,
            )
            state, theta_cmd = _target_turn_command(target, cfg)

            now_s = time.perf_counter()
            lockable_target = _target_is_lockable(choice, target, cfg)
            if lock_active:
                if not lockable_target:
                    lock_missing_frames += 1
                else:
                    lock_missing_frames = 0
                if lock_missing_frames >= max(int(cfg.lock_release_missing_frames), 1):
                    lock_active = False
                    lock_missing_frames = 0
                    hold_stable_frames = 0
                else:
                    state = "locked_hold"
                    theta_cmd = 0.0
                    hold_until_s = max(hold_until_s, now_s)
            if not lock_active:
                if now_s < hold_until_s:
                    state = "locked_hold"
                    theta_cmd = 0.0
                elif state == "hold" and lockable_target:
                    hold_stable_frames += 1
                    if hold_stable_frames >= max(int(cfg.hold_confirm_frames), 1):
                        hold_until_s = now_s + max(float(cfg.hold_dwell_s), 0.0)
                        lock_active = True
                        lock_missing_frames = 0
                        state = "locked_hold"
                        theta_cmd = 0.0
                else:
                    hold_stable_frames = 0

            observed_theta = _safe_float(observation_cache.get("theta.vel"), 0.0)
            if abs(observed_theta) > 1e-3:
                heading_rad += observed_theta * dt

            z_hold = _determine_z_hold_pos(cfg, observation_cache, robot)
            _send_motion_command(
                robot=robot,
                theta_vel_rad_s=theta_cmd,
                z_hold_pos=z_hold,
                fps=cfg.fps,
                hold_s=min(float(cfg.command_hold_s), 1.0 / max(cfg.fps, 1)),
                minimum_abs_turn_speed_rad_s=float(cfg.min_turn_speed_rad_s),
            )

            status_now_s = time.perf_counter()
            if cfg.status_log_interval_s > 0.0 and (status_now_s - last_status_log_s) >= cfg.status_log_interval_s:
                target_dist = None if target is None else target.estimated_distance_m
                hold_remaining_s = max(0.0, hold_until_s - status_now_s)
                print(
                    "Elevated edge scan status "
                    f"state={state} choice={choice} theta_cmd={theta_cmd:+.2f} observed_theta={observed_theta:+.2f} "
                    f"lock_active={lock_active} lock_missing_frames={lock_missing_frames} lockable_target={lockable_target} "
                    f"hold_frames={hold_stable_frames} hold_remaining_s={hold_remaining_s:.2f} "
                    f"left(score={left_obs.score:.1f}, detected={left_obs.detected}, hits={left_stable_hits}, reason={left_obs.reason}) "
                    f"right(score={right_obs.score:.1f}, detected={right_obs.detected}, hits={right_stable_hits}, reason={right_obs.reason}) "
                    f"target_dist_m={None if target_dist is None else round(float(target_dist), 2)}"
                )
                last_status_log_s = status_now_s

            if target is left_obs:
                _accumulate_points(map_points, left_obs, cfg.left_key, heading_rad, cfg)
            elif target is right_obs:
                _accumulate_points(map_points, right_obs, cfg.right_key, heading_rad, cfg)

            target_center_x_ratio = None if target is None else target.center_x_ratio
            left_canvas = _draw_observation(
                left_frame,
                cfg.left_key,
                left_obs,
                target is left_obs or choice == "paired",
                rejected=(left_obs.detected and not lockable_target and target is None),
                choice=choice,
                target_center_x_ratio=target_center_x_ratio,
            )
            right_canvas = _draw_observation(
                right_frame,
                cfg.right_key,
                right_obs,
                target is right_obs or choice == "paired",
                rejected=(right_obs.detected and not lockable_target and target is None),
                choice=choice,
                target_center_x_ratio=target_center_x_ratio,
            )
            map_canvas = _draw_map(
                map_points,
                heading_rad,
                f"{state}:{choice}",
                theta_cmd,
                choice=choice,
                lock_active=lock_active,
                lockable_target=lockable_target,
            )

            top = np.hstack([left_canvas, right_canvas])
            if top.shape[1] != map_canvas.shape[1]:
                map_canvas = cv2.resize(map_canvas, (top.shape[1], map_canvas.shape[0]), interpolation=cv2.INTER_LINEAR)
            preview = np.vstack([top, map_canvas])
            viewer_now_s = time.perf_counter()
            if viewer_dir is not None and (viewer_now_s - last_viewer_write_s) >= max(float(cfg.viewer_write_interval_s), 0.05):
                target_dist = None if target is None else target.estimated_distance_m
                _write_viewer_files(
                    viewer_dir=viewer_dir,
                    left_canvas=left_canvas,
                    right_canvas=right_canvas,
                    map_canvas=map_canvas,
                    state=state,
                    choice=choice,
                    theta_cmd=theta_cmd,
                    observed_theta=observed_theta,
                    target_dist=target_dist,
                    lock_active=lock_active,
                    hold_frames=hold_stable_frames,
                )
                last_viewer_write_s = viewer_now_s

            if preview_available:
                try:
                    cv2.imshow("Elevated Edge Scan", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                except cv2.error as exc:
                    preview_available = False
                    if not preview_warning_printed:
                        print(
                            "Preview disabled because OpenCV highgui is unavailable in this environment. "
                            f"Continuing headless. ({exc})"
                        )
                        if viewer_dir is not None:
                            print(f"Viewer files will be written to: {viewer_dir}")
                        preview_warning_printed = True

            precise_sleep(max((1.0 / max(cfg.fps, 1)) - (time.perf_counter() - loop_started), 0.0))
    except KeyboardInterrupt:
        print("Elevated edge scan interrupted. Sending final stop.")
    finally:
        try:
            z_hold = _determine_z_hold_pos(cfg, observation_cache, robot)
            _send_motion_command(
                robot=robot,
                theta_vel_rad_s=0.0,
                z_hold_pos=z_hold,
                fps=cfg.fps,
                hold_s=max(float(cfg.command_hold_s), 0.12),
                minimum_abs_turn_speed_rad_s=float(cfg.min_turn_speed_rad_s),
            )
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        if preview_available:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = ElevatedEdgeScanConfig(
        id=args.id,
        remote_ip=args.remote_ip,
        fps=args.fps,
        left_key=args.left_key,
        right_key=args.right_key,
        turn_gain=args.turn_gain,
        min_turn_speed_rad_s=args.min_turn_speed_rad_s,
        max_turn_speed_rad_s=args.max_turn_speed_rad_s,
        center_deadband_ratio=args.center_deadband_ratio,
        min_line_width_ratio=args.min_line_width_ratio,
        min_line_score=args.min_line_score,
        single_eye_min_line_score=args.single_eye_min_line_score,
        min_shadow_contrast=args.min_shadow_contrast,
        min_vertical_edge_strength=args.min_vertical_edge_strength,
        min_upper_band_brightness=args.min_upper_band_brightness,
        min_top_surface_support_ratio=args.min_top_surface_support_ratio,
        min_lower_shadow_support_ratio=args.min_lower_shadow_support_ratio,
        min_center_y_ratio=args.min_center_y_ratio,
        max_center_y_ratio=args.max_center_y_ratio,
        min_abs_slope=args.min_abs_slope,
        max_abs_slope=args.max_abs_slope,
        pair_center_y_tolerance_ratio=args.pair_center_y_tolerance_ratio,
        pair_distance_tolerance_m=args.pair_distance_tolerance_m,
        near_distance_m=args.near_distance_m,
        far_distance_m=args.far_distance_m,
        camera_half_width_m=args.camera_half_width_m,
        camera_yaw_deg=args.camera_yaw_deg,
        z_hold_pos=args.z_hold_pos,
        status_log_interval_s=args.status_log_interval_s,
        search_turn_speed_rad_s=args.search_turn_speed_rad_s,
        search_turn_direction=args.search_turn_direction,
        stable_hits_required=args.stable_hits_required,
        command_hold_s=args.command_hold_s,
        hold_confirm_frames=args.hold_confirm_frames,
        hold_dwell_s=args.hold_dwell_s,
        lock_release_missing_frames=args.lock_release_missing_frames,
        lock_max_single_eye_distance_m=args.lock_max_single_eye_distance_m,
        lock_prefer_paired=args.lock_prefer_paired,
        viewer_dir=args.viewer_dir,
        viewer_write_interval_s=args.viewer_write_interval_s,
    )
    return elevated_edge_scan_live(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
