"""Offline bake-off: monocular metric depth vs Sourccey's edge detector.

Runs Depth-Anything-V2 (metric, indoor) over saved eye snapshots
(scripts/sourccey_eye_snapshot.py output) and, using the field-calibrated
eye-camera geometry, converts the depth map into per-pixel 3D:

    height above floor  ->  "floating surface" = anything 0.30..1.10m up
    forward distance    ->  would the 0.55m forward gate have fired?

For every input frame it writes a composite panel (original | depth |
elevated-surface overlay + verdict) so a human can judge whether the model
found the real table edges and ignored the phantoms.

Scale sanity: monocular metric depth can be globally off. The bottom-center
of each frame is assumed to be floor; the median ratio between the
geometry-predicted floor distance and the model's is reported and applied,
so verdicts are floor-anchored (on the robot, the lidar plays this role).

Run (heavy deps pulled ad hoc, cached by uv):
    uv run --no-project --with torch,transformers,pillow,opencv-python-headless,numpy \
        python scripts/sourccey_depth_eval.py --shots artifacts/eye_snapshots
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

# Field-calibrated eye geometry (two-distance wall protocol, 2026-07-09) —
# keep in sync with sourccey_camera_geometry.CameraModel defaults.
EYE_HEIGHT_M = 0.914
EYE_PITCH_DOWN_DEG = 13.9
EYE_VFOV_DEG = 66.0
EYE_HFOV_DEG = 82.0

ELEVATED_MIN_M = 0.30  # above the lidar plane (0.28m) with margin
ELEVATED_MAX_M = 1.10  # below this = furniture the base can hit
FLOOR_BAND_M = 0.12
GATE_FORWARD_M = 0.55
GATE_RANGE_REPORT_M = 2.50
CORRIDOR_HALF_WIDTH_M = 0.40


def pixel_grid_3d(depth_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth map (meters along the optical axis) -> level-frame 3D per pixel:
    (forward_m, lateral_m, height_above_floor_m)."""
    h, w = depth_m.shape
    fx = (w / 2.0) / math.tan(math.radians(EYE_HFOV_DEG / 2.0))
    fy = (h / 2.0) / math.tan(math.radians(EYE_VFOV_DEG / 2.0))
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    u = np.arange(w, dtype=np.float32)[None, :]
    v = np.arange(h, dtype=np.float32)[:, None]
    x_opt = (u - cx) / fx * depth_m  # right
    y_opt = (v - cy) / fy * depth_m  # down (image convention)
    z_opt = depth_m  # forward along the (pitched) optical axis
    p = math.radians(EYE_PITCH_DOWN_DEG)
    forward = z_opt * math.cos(p) - y_opt * math.sin(p)
    up = -z_opt * math.sin(p) - y_opt * math.cos(p)
    height = EYE_HEIGHT_M + up
    return forward, x_opt, height


def geometric_floor_distance(y_ratio: np.ndarray) -> np.ndarray:
    """Calibrated row->floor distance (what the row WOULD mean if floor)."""
    depression = EYE_PITCH_DOWN_DEG + (y_ratio - 0.5) * EYE_VFOV_DEG
    depression = np.clip(depression, 1.0, 89.0)
    return EYE_HEIGHT_M / np.tan(np.radians(depression))


def floor_scale_factor(depth_m: np.ndarray) -> float:
    """Median (geometry / model) over the assumed-floor patch (bottom-center).
    1.0 = the model's metric scale already matches the calibration."""
    h, w = depth_m.shape
    rows = slice(int(h * 0.86), h)
    cols = slice(int(w * 0.30), int(w * 0.70))
    v = np.arange(h, dtype=np.float32)[:, None] / max(h - 1, 1)
    y_ratio = np.broadcast_to(v, (h, w))[rows, cols]
    geo_floor = geometric_floor_distance(y_ratio)
    p = math.radians(EYE_PITCH_DOWN_DEG)
    # geometric slant range along the ray ~= floor distance / cos(depression);
    # compare against the model's optical-axis depth via forward component.
    model_depth = depth_m[rows, cols]
    model_forward = model_depth * math.cos(p)
    ratio = geo_floor / np.maximum(model_forward, 1e-3)
    return float(np.median(ratio))


def evaluate_frame(image_bgr: np.ndarray, depth_m: np.ndarray) -> tuple[np.ndarray, dict]:
    scale = floor_scale_factor(depth_m)
    depth_scaled = depth_m * scale
    forward, lateral, height = pixel_grid_3d(depth_scaled)

    elevated = (height >= ELEVATED_MIN_M) & (height <= ELEVATED_MAX_M) & (
        forward <= GATE_RANGE_REPORT_M
    ) & (forward > 0.05)
    floorish = np.abs(height) <= FLOOR_BAND_M

    in_corridor = elevated & (np.abs(lateral) <= CORRIDOR_HALF_WIDTH_M)
    nearest_m = float(np.min(forward[in_corridor])) if np.any(in_corridor) else None
    would_stop = nearest_m is not None and nearest_m <= GATE_FORWARD_M

    # Overlay: elevated surfaces red, floor greenish; the verdict on top.
    overlay = image_bgr.copy()
    overlay[elevated] = (0.35 * overlay[elevated] + 0.65 * np.array([0, 0, 220])).astype(np.uint8)
    overlay[floorish] = (0.7 * overlay[floorish] + 0.3 * np.array([0, 160, 0])).astype(np.uint8)
    verdict = (
        f"nearest elevated {nearest_m:.2f}m" if nearest_m is not None else "no elevated surface"
    )
    if would_stop:
        verdict += "  -> GATE STOP"
    cv2.putText(overlay, verdict, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(
        overlay,
        f"floor-scale x{scale:.2f}",
        (6, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 255, 200),
        1,
    )

    depth_vis = cv2.applyColorMap(
        cv2.normalize(depth_scaled, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    panel = np.concatenate([image_bgr, depth_vis, overlay], axis=1)
    meta = {
        "scale": scale,
        "nearest_elevated_m": nearest_m,
        "would_stop": would_stop,
        "elevated_pixel_ratio": float(np.mean(elevated)),
    }
    return panel, meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Depth-Anything-V2 bake-off over eye snapshots.")
    parser.add_argument("--shots", type=str, default="artifacts/eye_snapshots")
    parser.add_argument("--out", type=str, default="artifacts/depth_eval")
    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    )
    args = parser.parse_args()

    shots_dir = Path(args.shots)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(p for p in shots_dir.glob("*front_*.png"))
    if not frames:
        print(f"no front_* snapshots found in {shots_dir}")
        return 1

    print(f"loading {args.model} (first run downloads the weights) ...")
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    depth_pipe = pipeline("depth-estimation", model=str(args.model), device=device)
    print(f"model ready on {'cuda' if device == 0 else 'cpu'}; evaluating {len(frames)} frames")

    from PIL import Image

    for frame_path in frames:
        image_bgr = cv2.imread(str(frame_path))
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        result = depth_pipe(pil)
        depth = result["predicted_depth"]
        depth_np = depth.squeeze().float().cpu().numpy()
        if depth_np.shape != image_bgr.shape[:2]:
            depth_np = cv2.resize(
                depth_np, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        panel, meta = evaluate_frame(image_bgr, depth_np)
        out_path = out_dir / f"{frame_path.stem}_eval.png"
        cv2.imwrite(str(out_path), panel)
        nearest = meta["nearest_elevated_m"]
        print(
            f"{frame_path.name}: scale=x{meta['scale']:.2f} "
            f"nearest_elevated={'-' if nearest is None else f'{nearest:.2f}m'} "
            f"stop={'YES' if meta['would_stop'] else 'no'} "
            f"elevated_px={meta['elevated_pixel_ratio']:.1%}"
        )
    print(f"panels in {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
