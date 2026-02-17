import argparse
from pathfinding import PathfindingConfig, run_pathfinding

# Keep this file map-only. Edit obstacles or defaults here, but pass
# start/goal/dir/IP as CLI args.

# Obstacles are fixed in this file.
obstacles = {
    (2, 5),
    (3, 5),
    (2, 6),
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Sourccey open-loop pathfinding")
    parser.add_argument("--remote_ip", required=True, help="Raspberry Pi IP running sourccey_host")
    parser.add_argument("--start_pos", required=True, help="Start position as 'x,y' (e.g. 1,1)")
    parser.add_argument("--start_dir", required=True, choices=["N", "E", "S", "W"], help="Start direction")
    parser.add_argument("--goal_pos", required=True, help="Goal position as 'x,y' (e.g. 4,10)")
    return parser.parse_args()


def _parse_pos(text: str):
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid position '{text}'. Expected 'x,y'.")
    return (int(parts[0].strip()), int(parts[1].strip()))


if __name__ == "__main__":
    args = _parse_args()

    start_pos = _parse_pos(args.start_pos)
    goal_pos = _parse_pos(args.goal_pos)
    start_dir = args.start_dir

    # Map + motion configuration (edit defaults here)
    cfg = PathfindingConfig(
        remote_ip=args.remote_ip,
        robot_id="sourccey",
        reverse=False,
        fps=30,
        min_x=1,
        max_x=4,
        min_y=1,
        max_y=10,
        move_time_s=0.9,
        turn_time_s=0.6,
        move_speed=0.7,
        turn_speed=0.7,
    )

    run_pathfinding(
        start=start_pos,
        start_dir=start_dir,
        goal=goal_pos,
        obstacles=obstacles,
        cfg=cfg,
    )
