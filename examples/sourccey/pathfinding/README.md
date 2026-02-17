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
python examples/sourccey/pathfinding/map.py   --remote_ip 192.168.1.210   --start_pos 1,1   --start_dir N   --goal_pos 4,10
```

## Notes

- Movement is open loop: you must tune `move_time_s` and `turn_time_s` so one ?cell? matches your real-world spacing.
- Directions use `N`, `E`, `S`, `W`. `x+` is East and `y+` is North.
