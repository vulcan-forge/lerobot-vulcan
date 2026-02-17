import time
import json
import heapq
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClientConfig, SourcceyClient
from lerobot.utils.robot_utils import precise_sleep

GridPos = Tuple[int, int]  # (x, y)
Direction = str  # "N", "E", "S", "W"


def _load_default_arm_actions() -> dict[str, float]:
    """Load default arm actions from teleop JSONs (arms at side)."""
    try:
        from lerobot.teleoperators.sourccey.sourccey.sourccey_leader import sourccey_leader

        base_dir = Path(sourccey_leader.__file__).parent / "defaults"
        left_path = base_dir / "left_arm_default_action.json"
        right_path = base_dir / "right_arm_default_action.json"

        left = json.loads(left_path.read_text(encoding="utf-8")) if left_path.exists() else {}
        right = json.loads(right_path.read_text(encoding="utf-8")) if right_path.exists() else {}

        action = {f"left_{k}": float(v) for k, v in left.items()}
        action.update({f"right_{k}": float(v) for k, v in right.items()})
        return action
    except Exception:
        return {}


@dataclass
class PathfindingConfig:
    remote_ip: str = "192.168.1.237"
    robot_id: str = "sourccey"
    reverse: bool = False
    fps: int = 30

    # Room dimensions
    min_x: int = 1
    max_x: int = 4
    min_y: int = 1
    max_y: int = 10

    # Open-loop motion tuning (seconds per cell / per 90-deg)
    move_time_s: float = 0.9
    turn_time_s: float = 0.6

    # Base command magnitudes (normalized -1..1)
    move_speed: float = 0.7
    turn_speed: float = 0.7


def dijkstra_path(
    start: GridPos,
    goal: GridPos,
    obstacles: Optional[set[GridPos]],
    cfg: PathfindingConfig,
) -> List[GridPos]:
    if obstacles is None:
        obstacles = set()

    def neighbors(p: GridPos) -> List[GridPos]:
        x, y = p
        candidates = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]
        valid = []
        for nx, ny in candidates:
            if nx < cfg.min_x or nx > cfg.max_x or ny < cfg.min_y or ny > cfg.max_y:
                continue
            if (nx, ny) in obstacles:
                continue
            valid.append((nx, ny))
        return valid

    frontier: List[Tuple[int, GridPos]] = []
    heapq.heappush(frontier, (0, start))
    came_from: Dict[GridPos, Optional[GridPos]] = {start: None}
    cost_so_far: Dict[GridPos, int] = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break

        for nxt in neighbors(current):
            new_cost = cost_so_far[current] + 1
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current

    if goal not in came_from:
        return []

    # Reconstruct path
    path = [goal]
    cur = goal
    while came_from[cur] is not None:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


def direction_from_delta(dx: int, dy: int) -> Direction:
    if dx == 1 and dy == 0:
        return "E"
    if dx == -1 and dy == 0:
        return "W"
    if dx == 0 and dy == 1:
        return "N"
    if dx == 0 and dy == -1:
        return "S"
    raise ValueError(f"Invalid delta: ({dx}, {dy})")


def turn_sequence(current: Direction, target: Direction) -> List[str]:
    order = ["N", "E", "S", "W"]
    ci = order.index(current)
    ti = order.index(target)
    right_steps = (ti - ci) % 4
    left_steps = (ci - ti) % 4
    if right_steps <= left_steps:
        return ["R"] * right_steps
    return ["L"] * left_steps


def send_action_for_duration(
    robot: SourcceyClient,
    action: Dict[str, float],
    duration_s: float,
    fps: int,
    arm_action: Optional[Dict[str, float]] = None,
):
    t0 = time.perf_counter()
    while True:
        merged = {**(arm_action or {}), **action}
        robot.send_action(merged)
        precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))
        if time.perf_counter() - t0 >= duration_s:
            break


def stop_base(robot: SourcceyClient, fps: int, arm_action: Optional[Dict[str, float]] = None):
    merged = {**(arm_action or {}), **{"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}}
    robot.send_action(merged)
    precise_sleep(1.0 / fps)


def execute_path_open_loop(
    robot: SourcceyClient,
    path: List[GridPos],
    start_dir: Direction,
    cfg: PathfindingConfig,
    arm_action: Optional[Dict[str, float]] = None,
):
    if not path:
        print("No path found.")
        return

    cur_dir = start_dir
    for idx in range(1, len(path)):
        x0, y0 = path[idx - 1]
        x1, y1 = path[idx]
        dx, dy = x1 - x0, y1 - y0
        step_dir = direction_from_delta(dx, dy)

        # Turn to face movement direction (theta.vel > 0 rotates left)
        for turn in turn_sequence(cur_dir, step_dir):
            theta = cfg.turn_speed if turn == "L" else -cfg.turn_speed
            send_action_for_duration(
                robot,
                {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": theta},
                cfg.turn_time_s,
                cfg.fps,
                arm_action=arm_action,
            )
            stop_base(robot, cfg.fps, arm_action=arm_action)

        cur_dir = step_dir

        # Move one cell forward (x.vel > 0 is forward)
        send_action_for_duration(
            robot,
            {"x.vel": cfg.move_speed, "y.vel": 0.0, "theta.vel": 0.0},
            cfg.move_time_s,
            cfg.fps,
            arm_action=arm_action,
        )
        stop_base(robot, cfg.fps, arm_action=arm_action)


def run_pathfinding(
    start: GridPos,
    start_dir: Direction,
    goal: GridPos,
    obstacles: Optional[set[GridPos]] = None,
    cfg: Optional[PathfindingConfig] = None,
):
    if cfg is None:
        cfg = PathfindingConfig()

    path = dijkstra_path(start, goal, obstacles, cfg)
    print(f"Path: {path}")

    arm_action = _load_default_arm_actions()

    robot_config = SourcceyClientConfig(remote_ip=cfg.remote_ip, id=cfg.robot_id, reverse=cfg.reverse)
    robot = SourcceyClient(robot_config)
    robot.connect()

    try:
        execute_path_open_loop(robot, path, start_dir, cfg, arm_action=arm_action)
    finally:
        stop_base(robot, cfg.fps, arm_action=arm_action)
        robot.disconnect()
