# Sourccey Pathfinding

This example drives the Sourccey base in open loop on a 2D grid using Dijkstra pathfinding.

## 1) Start the robot host (Raspberry Pi)

```bash
uv run -m lerobot.robots.sourccey.sourccey.sourccey.sourccey_host
```

## 2) Configure obstacles + tuning (your computer)

Edit `examples/sourccey/pathfinding/map.py`:

- Edit `obstacles` if needed.
- Tune `move_time_s` and `turn_time_s` for your cell size and robot speed.

## 3) Run the pathfinding command (your computer)

```bash
python examples/sourccey/pathfinding/map.py \
  --remote_ip 192.168.1.210 \
  --start_pos 1,1 \
  --start_dir N \
  --goal_pos 4,10
```

## Turn calibration (quick fix for under-rotation)

Use this to tune `turn_time_s` / `theta.vel`:

```bash
python examples/sourccey/pathfinding/calibrate_turn.py \
  --remote_ip 192.168.1.210 \
  --duration_s 0.6 \
  --theta 0.7
```

Adjust `--duration_s` until the robot turns ~90°.

## Camera calibration (stereo)

This captures stereo pairs from `front_left` and `front_right` and produces
intrinsics + stereo extrinsics (baseline, rotation).

1. Print a chessboard (default 9x6 inner corners).
2. Hold it at different angles/distances in front of both cameras.
3. Run:

```bash
python examples/sourccey/pathfinding/calibrate_cameras.py \
  --remote_ip 192.168.1.210 \
  --board_cols 9 \
  --board_rows 6 \
  --square_size_m 0.024 \
  --num_frames 25
```

The output is saved to `examples/sourccey/pathfinding/camera_calibration.json`.

## Rerun visualization

This example logs observations and actions to Rerun (including camera frames).
Install and run the Rerun viewer, then start the script:

```bash
rerun
```

If you want to disable logging, set `log_rerun=False` in `run_pathfinding()` inside `pathfinding.py`.

## Notes

- Movement is open loop: you must tune `move_time_s` and `turn_time_s` so one `cell` matches your real-world spacing.
- Directions use `N`, `E`, `S`, `W`. `x+` is East and `y+` is North.
