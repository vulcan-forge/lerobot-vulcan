from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass

import zmq

from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClient, SourcceyClientConfig
from lerobot.utils.robot_utils import precise_sleep


@dataclass
class SurveyRotationProtocolConfig:
    id: str = "sourccey"
    remote_ip: str = "127.0.0.1"
    fps: int = 15
    startup_hold_s: float = 1.0
    total_sweep_deg: float = 360.0
    turn_step_deg: float = 12.0
    turn_speed_rad_s: float = 0.35
    min_effective_turn_speed_rad_s: float = 0.65
    slam_input_endpoint: str = "tcp://192.168.1.237:5560"
    imu_turn_completion_ratio: float = 0.92
    imu_poll_timeout_ms: int = 50
    imu_turn_timeout_scale: float = 3.0
    imu_turn_timeout_min_s: float = 1.25
    imu_stale_after_s: float = 0.40
    settle_gyro_threshold_rad_s: float = 0.08
    settle_stable_time_s: float = 0.50
    settle_timeout_s: float = 4.0
    settle_hold_s: float = 1.0
    capture_hold_s: float = 1.5
    direction: str = "left"
    segment_sequence: str | None = None
    step_forward_drive_distance_m: float = 0.0
    step_forward_drive_speed_m_s: float = 0.10
    min_effective_step_forward_speed_m_s: float = 0.18
    step_forward_min_duration_s: float = 0.45
    step_forward_settle_hold_s: float = 0.75
    step_forward_capture_hold_s: float = 0.60
    step_return_after_capture: bool = False
    step_return_settle_hold_s: float = 0.60
    forward_drive_distance_m: float = 0.0
    forward_drive_speed_m_s: float = 0.10
    min_effective_forward_speed_m_s: float = 0.18
    forward_drive_min_duration_s: float = 0.45
    forward_settle_hold_s: float = 0.75
    z_hold_pos: float | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automated turn-settle-map survey sweep for Sourccey."
    )
    parser.add_argument("--id", type=str, default="sourccey", help="Robot id for the Sourccey client.")
    parser.add_argument("--remote_ip", type=str, default="127.0.0.1", help="IP address of the Sourccey host.")
    parser.add_argument("--fps", type=int, default=15, help="Command update rate while the protocol runs.")
    parser.add_argument("--startup_hold_s", type=float, default=1.0, help="Initial stationary hold before the first turn.")
    parser.add_argument("--total_sweep_deg", type=float, default=360.0, help="Total degrees to sweep across the survey.")
    parser.add_argument("--turn_step_deg", type=float, default=12.0, help="Degrees to rotate during each survey step.")
    parser.add_argument("--turn_speed_rad_s", type=float, default=0.35, help="Turn rate command applied during each step.")
    parser.add_argument(
        "--min_effective_turn_speed_rad_s",
        type=float,
        default=0.65,
        help="Minimum absolute turn command to send once a turn step is active.",
    )
    parser.add_argument(
        "--slam_input_endpoint",
        type=str,
        default="tcp://192.168.1.237:5560",
        help="SLAM input endpoint to read IMU samples from for closed-loop turn completion.",
    )
    parser.add_argument(
        "--imu_turn_completion_ratio",
        type=float,
        default=0.92,
        help="Fraction of the requested step angle that must be measured by IMU before ending a turn.",
    )
    parser.add_argument(
        "--imu_poll_timeout_ms",
        type=int,
        default=50,
        help="ZMQ poll timeout while waiting for IMU-backed slam_input packets.",
    )
    parser.add_argument(
        "--imu_turn_timeout_scale",
        type=float,
        default=3.0,
        help="Maximum multiple of the nominal turn duration to allow before falling back to timeout completion.",
    )
    parser.add_argument(
        "--imu_turn_timeout_min_s",
        type=float,
        default=1.25,
        help="Minimum time to allow a turn step before timing out, regardless of nominal duration.",
    )
    parser.add_argument(
        "--imu_stale_after_s",
        type=float,
        default=0.40,
        help="Treat IMU state as stale if no fresh slam_input IMU sample arrives within this many seconds.",
    )
    parser.add_argument(
        "--settle_gyro_threshold_rad_s",
        type=float,
        default=0.08,
        help="Absolute gyro-z threshold used to decide that the robot has settled after a turn.",
    )
    parser.add_argument(
        "--settle_stable_time_s",
        type=float,
        default=0.50,
        help="How long the IMU must stay below the settle gyro threshold before capture begins.",
    )
    parser.add_argument(
        "--settle_timeout_s",
        type=float,
        default=4.0,
        help="Maximum time to wait for IMU-based settle detection before falling back to the fixed settle hold.",
    )
    parser.add_argument("--settle_hold_s", type=float, default=1.0, help="Stationary settle time after each turn step.")
    parser.add_argument("--capture_hold_s", type=float, default=1.5, help="Extra stationary dwell time for mapping after settling.")
    parser.add_argument("--direction", type=str, choices=("left", "right"), default="left", help="Survey sweep turn direction.")
    parser.add_argument(
        "--segment_sequence",
        type=str,
        default=None,
        help="Optional explicit sweep sequence like 'left:30,right:60,left:30'. Overrides total_sweep_deg/direction.",
    )
    parser.add_argument(
        "--step-forward-drive-distance-m",
        type=float,
        default=0.0,
        help="Optional micro-parallax forward distance to drive after every turn/capture step.",
    )
    parser.add_argument(
        "--step-forward-drive-speed-m-s",
        type=float,
        default=0.10,
        help="Forward speed command used for each per-step micro-parallax move.",
    )
    parser.add_argument(
        "--min-effective-step-forward-speed-m-s",
        type=float,
        default=0.18,
        help="Minimum absolute forward speed command to send once a per-step micro-parallax move is active.",
    )
    parser.add_argument(
        "--step-forward-min-duration-s",
        type=float,
        default=0.45,
        help="Minimum time to hold each per-step forward micro-parallax move even if distance/speed math suggests less.",
    )
    parser.add_argument(
        "--step-forward-settle-hold-s",
        type=float,
        default=0.75,
        help="Stationary settle hold applied after each per-step forward micro-parallax move.",
    )
    parser.add_argument(
        "--step-forward-capture-hold-s",
        type=float,
        default=0.60,
        help="Extra stationary mapping dwell applied after each per-step forward micro-parallax move.",
    )
    parser.add_argument(
        "--step-return-after-capture",
        action="store_true",
        help="Drive backward after each per-step forward micro-parallax capture to return near the original spot.",
    )
    parser.add_argument(
        "--step-return-settle-hold-s",
        type=float,
        default=0.60,
        help="Stationary settle hold applied after each per-step return move.",
    )
    parser.add_argument(
        "--forward-drive-distance-m",
        type=float,
        default=0.0,
        help="Optional forward distance to drive after each completed survey segment except the last.",
    )
    parser.add_argument(
        "--forward-drive-speed-m-s",
        type=float,
        default=0.10,
        help="Forward speed command used for each inter-segment drive move.",
    )
    parser.add_argument(
        "--min-effective-forward-speed-m-s",
        type=float,
        default=0.18,
        help="Minimum absolute forward speed command to send once an inter-segment drive move is active.",
    )
    parser.add_argument(
        "--forward-drive-min-duration-s",
        type=float,
        default=0.45,
        help="Minimum time to hold each inter-segment forward move even if distance/speed math suggests less.",
    )
    parser.add_argument(
        "--forward-settle-hold-s",
        type=float,
        default=0.75,
        help="Extra stationary settle hold applied after each inter-segment forward move.",
    )
    parser.add_argument("--z_hold_pos", type=float, default=None, help="Optional fixed z-axis hold position during the sweep.")
    return parser


def _connect_with_retry(robot: SourcceyClient, delay_s: float = 0.25) -> None:
    attempt = 0
    while True:
        attempt += 1
        try:
            robot.connect()
            print(f"Survey rotation protocol connected after {attempt} attempt(s).")
            return
        except Exception as exc:
            print(f"Survey rotation connect attempt {attempt} failed: {exc}")
            time.sleep(delay_s)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _determine_z_hold_pos(
    cfg: SurveyRotationProtocolConfig,
    observation: dict[str, object] | None,
    robot: SourcceyClient,
) -> float:
    if cfg.z_hold_pos is not None:
        return float(cfg.z_hold_pos)
    if isinstance(observation, dict) and "z.pos" in observation:
        return _safe_float(observation.get("z.pos"), 0.0)
    return float(getattr(robot, "_z_pos_cmd", 0.0))


def _build_action(
    *,
    x_vel_m_s: float = 0.0,
    theta_vel_rad_s: float,
    z_hold_pos: float,
) -> dict[str, float | bool]:
    return {
        "x.vel": float(x_vel_m_s),
        "y.vel": 0.0,
        "theta.vel": float(theta_vel_rad_s),
        "z.pos": float(z_hold_pos),
        "untorque_left": True,
        "untorque_right": True,
    }


def _run_action_for_duration(
    *,
    robot: SourcceyClient,
    action: dict[str, float | bool],
    duration_s: float,
    fps: int,
) -> None:
    started = time.perf_counter()
    period_s = 1.0 / max(int(fps), 1)
    while (time.perf_counter() - started) < max(float(duration_s), 0.0):
        loop_started = time.perf_counter()
        robot.send_action(action)
        precise_sleep(max(period_s - (time.perf_counter() - loop_started), 0.0))


def _apply_min_effective_magnitude(value: float, *, minimum_abs: float) -> float:
    minimum_abs = max(float(minimum_abs), 0.0)
    if abs(value) <= 1e-9 or minimum_abs <= 1e-9:
        return float(value)
    if abs(value) < minimum_abs:
        return float(minimum_abs if value > 0.0 else -minimum_abs)
    return float(value)


def _normalize_direction(direction: str) -> str:
    value = str(direction).strip().lower()
    if value not in {"left", "right"}:
        raise ValueError(f"Unsupported direction '{direction}'. Expected 'left' or 'right'.")
    return value


def _parse_segment_sequence(segment_sequence: str | None) -> list[tuple[str, float]]:
    if segment_sequence is None or not str(segment_sequence).strip():
        return []
    segments: list[tuple[str, float]] = []
    for raw_segment in str(segment_sequence).split(","):
        token = raw_segment.strip()
        if not token:
            continue
        parts = token.split(":", 1)
        if len(parts) != 2:
            raise ValueError(
                "Invalid segment token "
                f"'{token}'. Expected format like 'left:30' or 'right:60'."
            )
        direction = _normalize_direction(parts[0])
        try:
            degrees = float(parts[1])
        except Exception as exc:
            raise ValueError(f"Invalid degree value in segment '{token}'.") from exc
        if degrees <= 0.0:
            raise ValueError(f"Segment degrees must be positive in '{token}'.")
        segments.append((direction, degrees))
    if not segments:
        raise ValueError("Segment sequence was provided but no valid segments were parsed.")
    return segments


def _build_segments(cfg: SurveyRotationProtocolConfig) -> list[tuple[str, float]]:
    parsed = _parse_segment_sequence(cfg.segment_sequence)
    if parsed:
        return parsed
    return [(_normalize_direction(cfg.direction), max(float(cfg.total_sweep_deg), 0.0))]


class _SlamImuSubscriber:
    def __init__(self, endpoint: str, *, poll_timeout_ms: int, stale_after_s: float) -> None:
        self.endpoint = str(endpoint).strip()
        self.poll_timeout_ms = max(int(poll_timeout_ms), 1)
        self.stale_after_s = max(float(stale_after_s), 0.05)
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.connect(self.endpoint)
        self._last_sample_ts_ns: int | None = None
        self._last_abs_gz_rad_s: float | None = None
        self._last_fresh_sample_monotonic_s: float | None = None

    def close(self) -> None:
        try:
            self._socket.close(0)
        except Exception:
            pass
        try:
            self._context.term()
        except Exception:
            pass

    def poll_rotation_increment_rad(self) -> tuple[float, bool]:
        try:
            if not self._socket.poll(self.poll_timeout_ms):
                self._refresh_stale_state()
                return 0.0, False
            payload = self._socket.recv()
        except Exception:
            self._refresh_stale_state()
            return 0.0, False

        try:
            data = json.loads(payload.decode("utf-8"))
        except Exception:
            self._refresh_stale_state()
            return 0.0, False

        imu_samples = data.get("imu_samples")
        if not isinstance(imu_samples, list):
            self._refresh_stale_state()
            return 0.0, False

        total_delta_rad = 0.0
        fresh_sample_seen = False
        for sample in imu_samples:
            if not isinstance(sample, dict):
                continue
            capture_ns = sample.get("capture_monotonic_ns")
            gz = sample.get("gz")
            if capture_ns is None or gz is None:
                continue
            try:
                ts_ns = int(capture_ns)
                gz_rad_s = float(gz)
            except Exception:
                continue
            self._last_abs_gz_rad_s = abs(gz_rad_s)
            if self._last_sample_ts_ns is None:
                self._last_sample_ts_ns = ts_ns
                self._last_fresh_sample_monotonic_s = time.monotonic()
                fresh_sample_seen = True
                continue
            if ts_ns <= self._last_sample_ts_ns:
                continue
            dt_s = (ts_ns - self._last_sample_ts_ns) / 1_000_000_000.0
            self._last_sample_ts_ns = ts_ns
            self._last_fresh_sample_monotonic_s = time.monotonic()
            fresh_sample_seen = True
            total_delta_rad += abs(gz_rad_s) * max(dt_s, 0.0)
        self._refresh_stale_state()
        return float(total_delta_rad), fresh_sample_seen

    def latest_abs_gz_rad_s(self) -> float | None:
        self.poll_rotation_increment_rad()
        return self._last_abs_gz_rad_s

    def _refresh_stale_state(self) -> None:
        if self._last_fresh_sample_monotonic_s is None:
            self._last_abs_gz_rad_s = None
            return
        if (time.monotonic() - self._last_fresh_sample_monotonic_s) > self.stale_after_s:
            self._last_abs_gz_rad_s = None


def _run_turn_step_closed_loop(
    *,
    robot: SourcceyClient,
    action: dict[str, float | bool],
    target_turn_rad: float,
    timeout_s: float,
    fps: int,
    imu_subscriber: _SlamImuSubscriber | None,
) -> tuple[float, bool]:
    if imu_subscriber is None:
        _run_action_for_duration(robot=robot, action=action, duration_s=timeout_s, fps=fps)
        return 0.0, False

    accumulated_turn_rad = 0.0
    started = time.perf_counter()
    period_s = 1.0 / max(int(fps), 1)
    stale_poll_count = 0
    while accumulated_turn_rad < max(float(target_turn_rad), 0.0):
        if (time.perf_counter() - started) >= max(float(timeout_s), 0.0):
            return accumulated_turn_rad, False
        loop_started = time.perf_counter()
        robot.send_action(action)
        delta_rad, fresh_sample_seen = imu_subscriber.poll_rotation_increment_rad()
        accumulated_turn_rad += delta_rad
        if fresh_sample_seen:
            stale_poll_count = 0
        else:
            stale_poll_count += 1
            if stale_poll_count >= max(int(fps), 1):
                return accumulated_turn_rad, False
        precise_sleep(max(period_s - (time.perf_counter() - loop_started), 0.0))
    return accumulated_turn_rad, True


def _wait_for_settle(
    *,
    robot: SourcceyClient,
    action: dict[str, float | bool],
    settle_hold_s: float,
    stable_time_s: float,
    settle_timeout_s: float,
    gyro_threshold_rad_s: float,
    fps: int,
    imu_subscriber: _SlamImuSubscriber | None,
) -> tuple[bool, float | None]:
    if imu_subscriber is None:
        _run_action_for_duration(robot=robot, action=action, duration_s=settle_hold_s, fps=fps)
        return False, None

    started = time.perf_counter()
    stable_started: float | None = None
    latest_abs_gz: float | None = None
    period_s = 1.0 / max(int(fps), 1)
    while (time.perf_counter() - started) < max(float(settle_timeout_s), 0.0):
        loop_started = time.perf_counter()
        robot.send_action(action)
        latest_abs_gz = imu_subscriber.latest_abs_gz_rad_s()
        if latest_abs_gz is not None and latest_abs_gz <= float(gyro_threshold_rad_s):
            if stable_started is None:
                stable_started = time.perf_counter()
            elif (time.perf_counter() - stable_started) >= max(float(stable_time_s), 0.0):
                remaining_hold_s = max(float(settle_hold_s), 0.0)
                if remaining_hold_s > 0.0:
                    _run_action_for_duration(
                        robot=robot,
                        action=action,
                        duration_s=remaining_hold_s,
                        fps=fps,
                    )
                return True, latest_abs_gz
        else:
            stable_started = None
        precise_sleep(max(period_s - (time.perf_counter() - loop_started), 0.0))

    if settle_hold_s > 0.0:
        _run_action_for_duration(robot=robot, action=action, duration_s=settle_hold_s, fps=fps)
    return False, latest_abs_gz


def _run_forward_micro_parallax_step(
    *,
    robot: SourcceyClient,
    stop_action: dict[str, float | bool],
    z_hold_pos: float,
    drive_distance_m: float,
    drive_speed_m_s: float,
    min_duration_s: float,
    settle_hold_s: float,
    capture_hold_s: float,
    return_after_capture: bool,
    return_settle_hold_s: float,
    fps: int,
    imu_subscriber: _SlamImuSubscriber | None,
    cfg: SurveyRotationProtocolConfig,
) -> None:
    if drive_distance_m <= 1e-6 or drive_speed_m_s <= 1e-6:
        return
    forward_duration_s = max(
        drive_distance_m / max(drive_speed_m_s, 1e-3),
        max(float(min_duration_s), 0.0),
    )
    forward_action = _build_action(
        x_vel_m_s=drive_speed_m_s,
        theta_vel_rad_s=0.0,
        z_hold_pos=z_hold_pos,
    )
    _run_action_for_duration(
        robot=robot,
        action=forward_action,
        duration_s=forward_duration_s,
        fps=fps,
    )
    _wait_for_settle(
        robot=robot,
        action=stop_action,
        settle_hold_s=settle_hold_s,
        stable_time_s=cfg.settle_stable_time_s,
        settle_timeout_s=cfg.settle_timeout_s,
        gyro_threshold_rad_s=cfg.settle_gyro_threshold_rad_s,
        fps=fps,
        imu_subscriber=imu_subscriber,
    )
    _run_action_for_duration(
        robot=robot,
        action=stop_action,
        duration_s=capture_hold_s,
        fps=fps,
    )
    if not return_after_capture:
        return
    reverse_action = _build_action(
        x_vel_m_s=-drive_speed_m_s,
        theta_vel_rad_s=0.0,
        z_hold_pos=z_hold_pos,
    )
    _run_action_for_duration(
        robot=robot,
        action=reverse_action,
        duration_s=forward_duration_s,
        fps=fps,
    )
    _wait_for_settle(
        robot=robot,
        action=stop_action,
        settle_hold_s=return_settle_hold_s,
        stable_time_s=cfg.settle_stable_time_s,
        settle_timeout_s=cfg.settle_timeout_s,
        gyro_threshold_rad_s=cfg.settle_gyro_threshold_rad_s,
        fps=fps,
        imu_subscriber=imu_subscriber,
    )


def survey_rotation_protocol(cfg: SurveyRotationProtocolConfig) -> int:
    robot_config = SourcceyClientConfig(remote_ip=cfg.remote_ip, id=cfg.id)
    robot = SourcceyClient(robot_config)
    _connect_with_retry(robot)
    robot.untorque_left_active = True
    robot.untorque_right_active = True
    imu_subscriber = _SlamImuSubscriber(
        cfg.slam_input_endpoint,
        poll_timeout_ms=cfg.imu_poll_timeout_ms,
        stale_after_s=cfg.imu_stale_after_s,
    )

    observation: dict[str, object] = {}
    try:
        try:
            current_observation = robot.get_observation()
            if isinstance(current_observation, dict) and current_observation:
                observation = current_observation
        except Exception:
            observation = {}

        z_hold_pos = _determine_z_hold_pos(cfg, observation, robot)
        turn_step_deg = max(float(cfg.turn_step_deg), 0.1)
        turn_speed_rad_s = max(float(cfg.turn_speed_rad_s), 1e-3)
        turn_speed_rad_s = abs(
            _apply_min_effective_magnitude(
                turn_speed_rad_s,
                minimum_abs=float(cfg.min_effective_turn_speed_rad_s),
            )
        )
        startup_hold_s = max(float(cfg.startup_hold_s), 0.0)
        settle_hold_s = max(float(cfg.settle_hold_s), 0.0)
        capture_hold_s = max(float(cfg.capture_hold_s), 0.0)
        step_forward_drive_distance_m = max(float(cfg.step_forward_drive_distance_m), 0.0)
        step_forward_drive_speed_m_s = abs(
            _apply_min_effective_magnitude(
                float(cfg.step_forward_drive_speed_m_s),
                minimum_abs=float(cfg.min_effective_step_forward_speed_m_s),
            )
        )
        step_forward_min_duration_s = max(float(cfg.step_forward_min_duration_s), 0.0)
        step_forward_settle_hold_s = max(float(cfg.step_forward_settle_hold_s), 0.0)
        step_forward_capture_hold_s = max(float(cfg.step_forward_capture_hold_s), 0.0)
        step_return_after_capture = bool(cfg.step_return_after_capture)
        step_return_settle_hold_s = max(float(cfg.step_return_settle_hold_s), 0.0)
        forward_drive_distance_m = max(float(cfg.forward_drive_distance_m), 0.0)
        forward_drive_speed_m_s = abs(
            _apply_min_effective_magnitude(
                float(cfg.forward_drive_speed_m_s),
                minimum_abs=float(cfg.min_effective_forward_speed_m_s),
            )
        )
        forward_drive_min_duration_s = max(float(cfg.forward_drive_min_duration_s), 0.0)
        forward_settle_hold_s = max(float(cfg.forward_settle_hold_s), 0.0)
        turn_completion_ratio = float(max(min(cfg.imu_turn_completion_ratio, 1.0), 0.1))
        turn_timeout_scale = max(float(cfg.imu_turn_timeout_scale), 1.0)
        turn_timeout_min_s = max(float(cfg.imu_turn_timeout_min_s), 0.1)
        segments = [(direction, degrees) for direction, degrees in _build_segments(cfg) if degrees > 0.0]
        if not segments:
            raise ValueError("No positive survey segments were configured.")
        total_planned_steps = sum(max(int(math.ceil(degrees / turn_step_deg)), 1) for _, degrees in segments)
        if cfg.segment_sequence:
            sequence_label = ", ".join(f"{direction}:{degrees:.1f}" for direction, degrees in segments)
        else:
            sequence_label = f"{segments[0][0]}:{segments[0][1]:.1f}"

        print(
            "Survey rotation protocol started "
            f"sequence=[{sequence_label}] "
            f"planned_steps={total_planned_steps} turn_step_deg={turn_step_deg:.1f} "
            f"turn_speed_rad_s={turn_speed_rad_s:.2f} "
            f"settle_hold_s={settle_hold_s:.2f} capture_hold_s={capture_hold_s:.2f}"
            f" step_forward_drive_distance_m={step_forward_drive_distance_m:.2f}"
            f" forward_drive_distance_m={forward_drive_distance_m:.2f}"
        )

        stop_action = _build_action(theta_vel_rad_s=0.0, z_hold_pos=z_hold_pos)
        if startup_hold_s > 0.0:
            print(f"Startup hold for {startup_hold_s:.2f}s.")
            _run_action_for_duration(
                robot=robot,
                action=stop_action,
                duration_s=startup_hold_s,
                fps=cfg.fps,
            )

        cumulative_measured_turn_deg = 0.0
        global_step_index = 0
        for segment_index, (segment_direction, segment_total_deg) in enumerate(segments, start=1):
            direction_sign = 1.0 if segment_direction == "left" else -1.0
            planned_segment_steps = max(int(math.ceil(segment_total_deg / turn_step_deg)), 1)
            segment_measured_turn_deg = 0.0
            commanded_progress_deg = 0.0
            print(
                f"Segment {segment_index}/{len(segments)}: {segment_direction} {segment_total_deg:.1f}deg "
                f"in up to {planned_segment_steps} step(s)."
            )
            for segment_step_index in range(planned_segment_steps):
                if segment_measured_turn_deg >= segment_total_deg:
                    print(
                        f"  segment target reached by IMU ({segment_measured_turn_deg:.1f}/{segment_total_deg:.1f} deg)."
                    )
                    break
                remaining_command_deg = max(0.0, segment_total_deg - commanded_progress_deg)
                if remaining_command_deg <= 1e-6:
                    break
                step_deg = min(turn_step_deg, remaining_command_deg)
                step_deg = max(step_deg, 0.1)
                nominal_turn_duration_s = math.radians(step_deg) / turn_speed_rad_s
                target_turn_rad = math.radians(step_deg) * turn_completion_ratio
                turn_timeout_s = max(nominal_turn_duration_s * turn_timeout_scale, turn_timeout_min_s)
                turn_action = _build_action(
                    theta_vel_rad_s=direction_sign * turn_speed_rad_s,
                    z_hold_pos=z_hold_pos,
                )
                global_step_index += 1
                print(
                    f"Step {global_step_index}/{total_planned_steps} "
                    f"(segment {segment_index} step {segment_step_index + 1}/{planned_segment_steps}): "
                    f"turn {segment_direction} {step_deg:.1f}deg "
                    f"target~{math.degrees(target_turn_rad):.1f}deg via IMU, "
                    f"timeout {turn_timeout_s:.2f}s, settle {settle_hold_s:.2f}s, map {capture_hold_s:.2f}s."
                )
                measured_turn_rad, completed_by_imu = _run_turn_step_closed_loop(
                    robot=robot,
                    action=turn_action,
                    target_turn_rad=target_turn_rad,
                    timeout_s=turn_timeout_s,
                    fps=cfg.fps,
                    imu_subscriber=imu_subscriber,
                )
                settled_by_imu, latest_abs_gz = _wait_for_settle(
                    robot=robot,
                    action=stop_action,
                    settle_hold_s=settle_hold_s,
                    stable_time_s=cfg.settle_stable_time_s,
                    settle_timeout_s=cfg.settle_timeout_s,
                    gyro_threshold_rad_s=cfg.settle_gyro_threshold_rad_s,
                    fps=cfg.fps,
                    imu_subscriber=imu_subscriber,
                )
                _run_action_for_duration(
                    robot=robot,
                    action=stop_action,
                    duration_s=capture_hold_s,
                    fps=cfg.fps,
                )
                measured_turn_deg = math.degrees(measured_turn_rad)
                commanded_progress_deg += step_deg
                segment_measured_turn_deg += measured_turn_deg
                cumulative_measured_turn_deg += measured_turn_deg
                print(
                    f"  completed_by_imu={completed_by_imu} "
                    f"measured_turn_deg~={measured_turn_deg:.1f} "
                    f"segment_turn_deg~={segment_measured_turn_deg:.1f}/{segment_total_deg:.1f} "
                    f"cumulative_turn_deg~={cumulative_measured_turn_deg:.1f} "
                    f"settled_by_imu={settled_by_imu} "
                    f"latest_abs_gz={None if latest_abs_gz is None else round(latest_abs_gz, 3)}"
                )
                if step_forward_drive_distance_m > 1e-6:
                    print(
                        f"  step_micro_parallax: drive forward {step_forward_drive_distance_m:.2f}m "
                        f"for mapping, settle {step_forward_settle_hold_s:.2f}s, "
                        f"map {step_forward_capture_hold_s:.2f}s, "
                        f"return={'yes' if step_return_after_capture else 'no'}."
                    )
                    _run_forward_micro_parallax_step(
                        robot=robot,
                        stop_action=stop_action,
                        z_hold_pos=z_hold_pos,
                        drive_distance_m=step_forward_drive_distance_m,
                        drive_speed_m_s=step_forward_drive_speed_m_s,
                        min_duration_s=step_forward_min_duration_s,
                        settle_hold_s=step_forward_settle_hold_s,
                        capture_hold_s=step_forward_capture_hold_s,
                        return_after_capture=step_return_after_capture,
                        return_settle_hold_s=step_return_settle_hold_s,
                        fps=cfg.fps,
                        imu_subscriber=imu_subscriber,
                        cfg=cfg,
                    )
            if forward_drive_distance_m > 1e-6 and segment_index < len(segments):
                forward_duration_s = max(
                    forward_drive_distance_m / max(forward_drive_speed_m_s, 1e-3),
                    forward_drive_min_duration_s,
                )
                print(
                    f"Segment {segment_index}/{len(segments)} complete: drive forward "
                    f"{forward_drive_distance_m:.2f}m for ~{forward_duration_s:.2f}s, "
                    f"settle {forward_settle_hold_s:.2f}s, map {capture_hold_s:.2f}s."
                )
                forward_action = _build_action(
                    x_vel_m_s=forward_drive_speed_m_s,
                    theta_vel_rad_s=0.0,
                    z_hold_pos=z_hold_pos,
                )
                _run_action_for_duration(
                    robot=robot,
                    action=forward_action,
                    duration_s=forward_duration_s,
                    fps=cfg.fps,
                )
                settled_by_imu, latest_abs_gz = _wait_for_settle(
                    robot=robot,
                    action=stop_action,
                    settle_hold_s=forward_settle_hold_s,
                    stable_time_s=cfg.settle_stable_time_s,
                    settle_timeout_s=cfg.settle_timeout_s,
                    gyro_threshold_rad_s=cfg.settle_gyro_threshold_rad_s,
                    fps=cfg.fps,
                    imu_subscriber=imu_subscriber,
                )
                _run_action_for_duration(
                    robot=robot,
                    action=stop_action,
                    duration_s=capture_hold_s,
                    fps=cfg.fps,
                )
                print(
                    "  forward_move_complete "
                    f"settled_by_imu={settled_by_imu} "
                    f"latest_abs_gz={None if latest_abs_gz is None else round(latest_abs_gz, 3)}"
                )

        print("Survey rotation protocol complete. Sending final stop.")
    except KeyboardInterrupt:
        print("Survey rotation protocol interrupted. Sending final stop.")
    finally:
        try:
            z_hold_pos = _determine_z_hold_pos(cfg, observation, robot)
            robot.send_action(_build_action(theta_vel_rad_s=0.0, z_hold_pos=z_hold_pos))
        except Exception:
            pass
        try:
            robot.disconnect()
        except KeyboardInterrupt:
            pass
        except Exception:
            pass
        imu_subscriber.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = SurveyRotationProtocolConfig(
        id=args.id,
        remote_ip=args.remote_ip,
        fps=args.fps,
        startup_hold_s=args.startup_hold_s,
        total_sweep_deg=args.total_sweep_deg,
        turn_step_deg=args.turn_step_deg,
        turn_speed_rad_s=args.turn_speed_rad_s,
        min_effective_turn_speed_rad_s=args.min_effective_turn_speed_rad_s,
        slam_input_endpoint=args.slam_input_endpoint,
        imu_turn_completion_ratio=args.imu_turn_completion_ratio,
        imu_poll_timeout_ms=args.imu_poll_timeout_ms,
        imu_turn_timeout_scale=args.imu_turn_timeout_scale,
        imu_turn_timeout_min_s=args.imu_turn_timeout_min_s,
        imu_stale_after_s=args.imu_stale_after_s,
        settle_gyro_threshold_rad_s=args.settle_gyro_threshold_rad_s,
        settle_stable_time_s=args.settle_stable_time_s,
        settle_timeout_s=args.settle_timeout_s,
        settle_hold_s=args.settle_hold_s,
        capture_hold_s=args.capture_hold_s,
        direction=args.direction,
        segment_sequence=args.segment_sequence,
        step_forward_drive_distance_m=args.step_forward_drive_distance_m,
        step_forward_drive_speed_m_s=args.step_forward_drive_speed_m_s,
        min_effective_step_forward_speed_m_s=args.min_effective_step_forward_speed_m_s,
        step_forward_min_duration_s=args.step_forward_min_duration_s,
        step_forward_settle_hold_s=args.step_forward_settle_hold_s,
        step_forward_capture_hold_s=args.step_forward_capture_hold_s,
        step_return_after_capture=args.step_return_after_capture,
        step_return_settle_hold_s=args.step_return_settle_hold_s,
        forward_drive_distance_m=args.forward_drive_distance_m,
        forward_drive_speed_m_s=args.forward_drive_speed_m_s,
        min_effective_forward_speed_m_s=args.min_effective_forward_speed_m_s,
        forward_drive_min_duration_s=args.forward_drive_min_duration_s,
        forward_settle_hold_s=args.forward_settle_hold_s,
        z_hold_pos=args.z_hold_pos,
    )
    return survey_rotation_protocol(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
