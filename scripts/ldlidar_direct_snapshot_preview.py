from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Direct LiDAR Snapshot Preview</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      background: #0f1720;
      color: #e5eef8;
      font-family: Arial, sans-serif;
    }}
    h1 {{
      margin: 0 0 18px 0;
      font-size: 30px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: #141f2c;
      border: 1px solid #243346;
      border-radius: 14px;
      padding: 14px;
    }}
    .title {{
      font-size: 20px;
      margin-bottom: 10px;
    }}
    svg {{
      width: 100%;
      height: auto;
      background: #06090d;
      border-radius: 10px;
      border: 1px solid #203041;
    }}
    .meta {{
      margin-top: 8px;
      font-size: 13px;
      color: #a7bbd0;
      line-height: 1.5;
      white-space: pre-wrap;
    }}
  </style>
</head>
<body>
  <h1>Direct LiDAR Snapshot Preview</h1>
  <div class="grid">
    {cards}
  </div>
</body>
</html>
"""


def _load_snapshot_stems(snapshot_dir: Path) -> list[Path]:
    stems = sorted(
        {
            path.with_name(path.name.replace("_local.npy", ""))
            for path in snapshot_dir.glob("snapshot_*_local.npy")
        }
    )
    return stems


def _svg_points(points_xy: np.ndarray, *, width: int = 520, height: int = 520) -> str:
    if len(points_xy) == 0:
        return f'<svg viewBox="0 0 {width} {height}"></svg>'

    mins = np.min(points_xy, axis=0)
    maxs = np.max(points_xy, axis=0)
    center = (mins + maxs) / 2.0
    span = np.maximum(maxs - mins, 1e-6)
    scale = 0.86 * min(width / span[0], height / span[1])

    def project(point: np.ndarray) -> tuple[float, float]:
        x = (point[0] - center[0]) * scale + width / 2.0
        y = height / 2.0 - (point[1] - center[1]) * scale
        return float(x), float(y)

    grid_lines: list[str] = []
    for frac in np.linspace(0.1, 0.9, 9):
        x = frac * width
        y = frac * height
        grid_lines.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{height}" stroke="#1a2635" stroke-width="1"/>')
        grid_lines.append(f'<line x1="0" y1="{y:.1f}" x2="{width}" y2="{y:.1f}" stroke="#1a2635" stroke-width="1"/>')

    circles: list[str] = []
    for row in points_xy:
        x, y = project(row)
        circles.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.7" fill="#6ab4ff"/>')

    robot_x, robot_y = project(np.array([0.0, 0.0], dtype=np.float32))
    robot = f'<circle cx="{robot_x:.2f}" cy="{robot_y:.2f}" r="5" fill="#ffd166" stroke="#fff3" stroke-width="2"/>'

    return (
        f'<svg viewBox="0 0 {width} {height}">'
        + "".join(grid_lines)
        + "".join(circles)
        + robot
        + "</svg>"
    )


def _build_card(stem: Path) -> str:
    metadata = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    points_xy = np.load(stem.with_name(stem.name + "_local.npy"))
    svg = _svg_points(points_xy)
    meta_text = (
        f"frame_id={metadata.get('frame_id')}\\n"
        f"host_rev={metadata.get('host_revolution_index')}\\n"
        f"scan_ts={metadata.get('scan_ts')}\\n"
        f"host_digest={metadata.get('host_point_digest')}\\n"
        f"local_digest={metadata.get('local_signature', {}).get('digest')}\\n"
        f"points={metadata.get('local_point_count')}"
    )
    return (
        '<div class="card">'
        f'<div class="title">{stem.name}</div>'
        f"{svg}"
        f'<div class="meta">{meta_text}</div>'
        "</div>"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HTML preview for direct LiDAR snapshots.")
    parser.add_argument(
        "--snapshot-dir",
        default="artifacts/direct_lidar_snapshots",
        help="Directory containing snapshot_XXX.json and snapshot_XXX_local.npy files.",
    )
    parser.add_argument(
        "--output-html",
        default="artifacts/direct_lidar_snapshots/latest_preview.html",
        help="Output HTML preview path.",
    )
    args = parser.parse_args()

    snapshot_dir = Path(args.snapshot_dir)
    stems = _load_snapshot_stems(snapshot_dir)
    if not stems:
        raise SystemExit(f"No snapshots found in {snapshot_dir}")

    cards = "\n".join(_build_card(stem) for stem in stems)
    html = HTML_TEMPLATE.format(cards=cards)

    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    print(f"Saved preview to {output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
