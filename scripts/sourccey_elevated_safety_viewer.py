"""Standalone viewer and controlled approach test for the elevated safety gate.

This is the brief's "first useful test" — run it BEFORE wiring the gate into
autonomous wandering.

Modes:
  watch    (default) No motion at all. Shows both eye streams with ROI/hazard
           overlays in Rerun and logs every decision transition. Teleop or
           push the robot toward a table edge and confirm the decision flips
           to elevated_stop_* before contact.
  approach Creeps the base forward slowly until the gate denies forward
           motion, stops, then demonstrates that backing away is still
           allowed by reversing briefly. Ends with a PASS/FAIL summary.
           Supervise the robot: if the gate never trips, the script stops on
           its own after --max-approach-s seconds and reports FAIL.

Usage (client/WSL, host + camera stream already running on the Pi):
  uv run --with rerun-sdk python scripts/sourccey_elevated_safety_viewer.py \
    --remote-ip 192.168.1.237 --mode watch

The arms are never commanded by this script.
"""

from __future__ import annotations

import argparse
import time

from ldlidar_auto_snapshot_stitch import _init_rerun, _send_stop
from sourccey_elevated_safety import (
    ElevatedHazardMonitor,
    ElevatedSafetyConfig,
    SlamCameraSubscriber,
    endpoint_from_remote_ip,
    format_hazard_log,
    gate_forward_allowed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--remote-ip", type=str, default="192.168.1.237")
    parser.add_argument("--robot-id", type=str, default="sourccey")
    parser.add_argument(
        "--slam-input-endpoint",
        type=str,
        default="",
        help="Override the slam_input.v1 endpoint (default tcp://<remote-ip>:5560)",
    )
    parser.add_argument("--mode", choices=("watch", "approach"), default="watch")
    parser.add_argument("--rerun-mode", choices=("web", "local", "off"), default="web")
    parser.add_argument("--rerun-grpc-port", type=int, default=9877)
    parser.add_argument("--rerun-web-port", type=int, default=9878)
    parser.add_argument("--log-interval-s", type=float, default=1.0)
    parser.add_argument("--rerun-image-interval-s", type=float, default=0.4)
    # Approach-mode motion parameters. The base needs enough magnitude to
    # overcome wheel stiction (see brief), hence the floor.
    parser.add_argument("--approach-speed", type=float, default=0.78)
    parser.add_argument("--min-effective-speed", type=float, default=0.75)
    parser.add_argument("--max-approach-s", type=float, default=20.0)
    parser.add_argument("--reverse-s", type=float, default=1.2)
    # Safety tunables (kept as flags so ROI calibration is easy in the field).
    parser.add_argument("--forward-block-distance-m", type=float, default=0.55)
    parser.add_argument("--near-freeze-distance-m", type=float, default=0.35)
    parser.add_argument("--trip-frames", type=int, default=3)
    parser.add_argument("--clear-frames", type=int, default=5)
    parser.add_argument("--stale-timeout-s", type=float, default=1.5)
    parser.add_argument("--roi-min-y-ratio", type=float, default=0.46)
    parser.add_argument("--roi-max-y-ratio", type=float, default=0.95)
    parser.add_argument("--min-line-score", type=float, default=860.0)
    parser.add_argument("--near-distance-m", type=float, default=0.20)
    parser.add_argument("--far-distance-m", type=float, default=1.55)
    return parser


def _safety_config_from_args(args: argparse.Namespace) -> ElevatedSafetyConfig:
    endpoint = str(args.slam_input_endpoint).strip() or endpoint_from_remote_ip(args.remote_ip)
    return ElevatedSafetyConfig(
        slam_input_endpoint=endpoint,
        forward_block_distance_m=float(args.forward_block_distance_m),
        near_freeze_distance_m=float(args.near_freeze_distance_m),
        trip_frames=int(args.trip_frames),
        clear_frames=int(args.clear_frames),
        stale_timeout_s=float(args.stale_timeout_s),
        roi_min_y_ratio=float(args.roi_min_y_ratio),
        roi_max_y_ratio=float(args.roi_max_y_ratio),
        min_line_score=float(args.min_line_score),
        near_distance_m=float(args.near_distance_m),
        far_distance_m=float(args.far_distance_m),
    )


def _log_rerun_frames(rr, monitor: ElevatedHazardMonitor, config: ElevatedSafetyConfig) -> None:
    for cam in (config.left_key, config.right_key, config.bottom_key):
        annotated = monitor.annotated(cam)
        if annotated is not None:
            rr.log(f"cameras/{cam}", rr.Image(annotated[:, :, ::-1]))  # BGR -> RGB


def _log_rerun_decision(rr, label: str, detail: str) -> None:
    rr.log("safety/decision", rr.TextLog(f"{label} | {detail}"))


def _send_forward(robot, speed: float) -> None:
    robot.send_action(
        {
            "x.vel": float(speed),
            "y.vel": 0.0,
            "theta.vel": 0.0,
            "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
            "untorque_left": True,
            "untorque_right": True,
        }
    )


def run_watch(args: argparse.Namespace, monitor: ElevatedHazardMonitor, config: ElevatedSafetyConfig, rr) -> int:
    print("[viewer] watch mode: no motion will be commanded; Ctrl+C to exit")
    last_label = None
    last_status_s = 0.0
    last_image_s = 0.0
    try:
        while True:
            state = monitor.state()
            label = state.decision_label()
            now = time.monotonic()
            if label != last_label:
                print(f"[safety] {format_hazard_log(state)}")
                if rr is not None:
                    _log_rerun_decision(rr, label, format_hazard_log(state))
                last_label = label
            elif now - last_status_s >= max(float(args.log_interval_s), 0.2):
                print(f"[viewer] {format_hazard_log(state)}")
                last_status_s = now
            if rr is not None and now - last_image_s >= max(float(args.rerun_image_interval_s), 0.1):
                _log_rerun_frames(rr, monitor, config)
                last_image_s = now
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("[viewer] watch mode ended")
    return 0


def run_approach(args: argparse.Namespace, monitor: ElevatedHazardMonitor, config: ElevatedSafetyConfig, rr) -> int:
    from lerobot.control.sourccey.sourccey.survey_rotation_protocol import (
        _apply_min_effective_magnitude,
    )
    from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
    from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient

    print(
        "[approach] controlled approach test: the robot will creep forward until the "
        "elevated gate denies forward motion, then back away. SUPERVISE THE ROBOT."
    )
    robot = SourcceyClient(SourcceyClientConfig(id=args.robot_id, remote_ip=args.remote_ip))
    robot.connect()
    _send_stop(robot)

    speed = _apply_min_effective_magnitude(
        float(args.approach_speed), minimum_abs=float(args.min_effective_speed)
    )
    tripped = False
    trip_state = None
    deadline = time.monotonic() + max(float(args.max_approach_s), 1.0)
    last_image_s = 0.0
    try:
        while time.monotonic() < deadline:
            state = monitor.state()
            allowed, reason = gate_forward_allowed(state)
            now = time.monotonic()
            if rr is not None and now - last_image_s >= max(float(args.rerun_image_interval_s), 0.1):
                _log_rerun_frames(rr, monitor, config)
                last_image_s = now
            if not allowed:
                _send_stop(robot)
                tripped = True
                trip_state = state
                print(f"[safety] FORWARD DENIED: {format_hazard_log(state)}")
                if rr is not None:
                    _log_rerun_decision(rr, reason, format_hazard_log(state))
                break
            _send_forward(robot, speed)
            time.sleep(0.07)
        _send_stop(robot)

        if tripped:
            print(f"[approach] backing away for {float(args.reverse_s):.1f}s to prove reverse stays allowed")
            reverse_deadline = time.monotonic() + max(float(args.reverse_s), 0.2)
            while time.monotonic() < reverse_deadline:
                _send_forward(robot, -abs(speed))
                time.sleep(0.07)
            _send_stop(robot)
    finally:
        try:
            _send_stop(robot)
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass

    if tripped and trip_state is not None:
        print(
            "[approach] PASS: gate denied forward motion "
            f"(decision={trip_state.decision_label()}, side={trip_state.side}, "
            f"confidence={trip_state.confidence:.2f}, dist="
            f"{'-' if trip_state.est_distance_m is None else f'{trip_state.est_distance_m:.2f}m'}) "
            "and reverse was allowed afterwards"
        )
        return 0
    print(
        "[approach] FAIL: the gate never denied forward motion within "
        f"{float(args.max_approach_s):.0f}s — do not enable --elevated-safety in wander yet"
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _safety_config_from_args(args)

    rr = None
    if args.rerun_mode != "off":
        rr, viewer_url = _init_rerun(
            session_name="sourccey_elevated_safety",
            mode=str(args.rerun_mode),
            grpc_port=int(args.rerun_grpc_port),
            web_port=int(args.rerun_web_port),
        )
        if viewer_url:
            print(f"[viewer] rerun viewer: {viewer_url}")

    print(f"[viewer] subscribing to {config.slam_input_endpoint}")
    subscriber = SlamCameraSubscriber(
        endpoint=config.slam_input_endpoint,
        camera_keys=(config.left_key, config.right_key, config.bottom_key),
    )
    subscriber.start()
    if not subscriber.wait_for_frames(
        timeout_s=6.0, required=(config.left_key, config.right_key)
    ):
        print(
            "[viewer] ERROR: no eye camera frames arrived from the slam_input stream. "
            "Is the Sourccey host running with slam_input_enabled? "
            f"(endpoint={config.slam_input_endpoint}, packets={subscriber.packets_received})"
        )
        subscriber.stop()
        return 2
    if subscriber.latest(config.bottom_key)[0] is None:
        print(
            "[viewer] WARNING: bottom camera not in the stream — low floor objects cannot "
            "be ruled out or ground-gated (host flag: --bottom_camera_enabled=true)"
        )
    print("[viewer] camera frames flowing; starting hazard monitor")
    monitor = ElevatedHazardMonitor(config, subscriber)
    monitor.start()

    try:
        if args.mode == "approach":
            return run_approach(args, monitor, config, rr)
        return run_watch(args, monitor, config, rr)
    finally:
        monitor.stop()
        subscriber.stop()


if __name__ == "__main__":
    raise SystemExit(main())
