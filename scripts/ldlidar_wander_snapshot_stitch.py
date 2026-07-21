from __future__ import annotations

import argparse
import dataclasses
import json
import math
import socket
import time
import traceback
from pathlib import Path

import numpy as np

from ldlidar_auto_snapshot_stitch import (
    _ensure_clean_directory,
    _execute_turn_burst,
    _init_rerun,
    _log_rerun_state,
    _send_stop,
    _wait_for_initial_frame,
)
from ldlidar_defaults import (
    DEFAULT_LIDAR_FORWARD_ANGLE_DEG,
    DEFAULT_LIDAR_STOP_BOX_DISTANCE_M,
    DEFAULT_LIDAR_STOP_BOX_HALF_WIDTH_M,
    DEFAULT_LIDAR_STOP_BOX_MIN_DISTANCE_M,
    DEFAULT_LIDAR_STOP_BOX_THICKNESS_M,
)
from ldlidar_direct_snapshot_client import _scan_to_local_points
from ldlidar_direct_snapshot_stitch import (
    MotionHint,
    Pose2D,
    _advance_pose,
    _build_exploration_grid,
    _plan_frontier_path,
    _prior_weights_for_hint,
    _search_pose,
    _solve_turn_arc_pose,
    _transform_points,
)
from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyClientConfig
from lerobot.robots.sourccey.sourccey.sourccey.sourccey_client import SourcceyClient

# --- Wander subsystem, split by concern into the sourccey_wander/ package.
# main() below is the orchestrator; each import group is one readable module.
# See sourccey_wander/ARCHITECTURE.md for how they connect. ---
from sourccey_wander.imu_heading import (
    IMU_DEAD_RECK_SLACK_DEG,
    ImuYawClient,
    _imu_anchor,
    _imu_resolved_theta,
)
from sourccey_wander.lidar_feed import SelfMaskedLidarFeed
from sourccey_wander.stop_zone import (
    StopZoneConfig,
    _blocked_points_for_frame,
)
from sourccey_wander.wander_types import (
    ExploreTarget,
    FrontierChoice,
    _compose_motion_hints,
    _heading_bin_index,
    _normalize_angle_deg,
    _turn_lever_arm_local_delta,
)
from sourccey_wander.frontier import (
    _compute_drive_steer_theta_vel,
    _doorway_mouth_and_axis,
    _seed_explore_target_from_frontier,
    _select_frontier_choice,
    _select_map_frontier_choice,
    _select_survey_target,
    _target_choice_from_world_point,
    _target_distance_m,
)
from sourccey_wander.boxed_in import _escape_boxed_in
from sourccey_wander.mapping import (
    LidarBoxedInError,
    _append_stitch,
    _capture_snapshot,
)
from sourccey_wander.localization import (
    _accept_relocalized_pose,
    _estimate_pose_against_stitched_map,
    _log_live_pose_state,
    _pose_delta_metrics,
)
from sourccey_wander.calibration import MotionCalibration
from sourccey_wander.driving import (
    _drive_with_tracking,
    _reverse_escape,
    _turn_with_arc_tracking,
)

# Cumulative MEASURED (frame-to-frame, map-free) forward motion, driven while
# the map pose is LOST at the exit threshold, that is treated as "through the
# door". The doorway is the least-localizable pose in the room (a wall behind,
# the unmapped next room ahead), so the pose freezes and the positional exit
# completion can never resolve — this odometry fallback (the same honest
# measured motion the bump detector trusts) lets the run finish the crossing.
EXIT_THROUGH_ODOMETRY_M = 1.1

# EXIT RUN geometry keyed to the CENTROID of the mapped room (the mean of the
# latch vantages), NOT the distance to the nearest single vantage. Field run 24:
# with 11 vantages scattered through the room, "distance to the nearest vantage"
# is ~0.05m everywhere near the door, so completion (>=0.80m from ALL vantages)
# was nearly unreachable, the no-regression invariant switched itself off (its
# >=0.30m activation floor), and the real doorway got vetoed because a vantage
# happened to sit near it. Distance-from-centroid grows monotonically as the
# robot heads out ANY door and cannot be fooled by a nearby vantage.
EXIT_BEYOND_ROOM_M = 0.70  # completion: this far past the mapped footprint radius
EXIT_END_MARGIN_M = 0.30   # veto an opening whose far end lands inside room_radius+this
EXIT_RATCHET_SLACK_M = 0.20  # hard ratchet: inward drift beyond this is forbidden

# NEW-SPACE completion gate. The distance tests above can be FOOLED by driving to
# a far wall inside the same room (field run 2026-07-20: mapped a room by spinning
# in place -> tiny footprint -> "moved ~1m" counted as "exited" while the robot
# drove to a wall and turned back; the relocalization score stayed 15+ the whole
# time, i.e. it was ALWAYS looking at the mapped room). So completion ALSO requires
# the live scan to STOP matching the room the robot latched in: freeze that room as
# a cell-set, and complete only when at most this fraction of the current scan
# still lands on it (the rest is genuinely NEW space beyond a doorway).
EXIT_ROOM_CELL_M = 0.15          # cell size for the frozen room-overlap grid
EXIT_NEW_SPACE_MAX_OVERLAP = 0.50  # complete only when <= this fraction still matches the old room


def _port_is_free(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("0.0.0.0", int(port)))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def _pick_free_rerun_ports(grpc_port: int, web_port: int) -> tuple[int, int]:
    """Step past ports held by zombie rerun servers (crashed runs can leak an
    orphaned listener; rerun then 'starts' but every log message vanishes and
    the viewer sits on the welcome screen forever)."""
    for bump in range(0, 20, 2):
        candidate_grpc = int(grpc_port) + bump
        candidate_web = int(web_port) + bump
        if _port_is_free(candidate_grpc) and _port_is_free(candidate_web):
            if bump:
                print(
                    f"[rerun] ports {grpc_port}/{web_port} are in use (stale server from a "
                    f"previous run?); using {candidate_grpc}/{candidate_web} instead"
                )
            return candidate_grpc, candidate_web
    print(
        f"[rerun] WARNING: no free port pair found near {grpc_port}/{web_port}; "
        "trying the defaults anyway (the viewer may not receive data)"
    )
    return int(grpc_port), int(web_port)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatic wandering LiDAR capture + offline-style stitcher with live Rerun output."
    )
    parser.add_argument("--remote-ip", default="192.168.1.237", help="Sourccey host IP.")
    parser.add_argument("--robot-id", default="sourccey")
    parser.add_argument("--lidar-host", default="192.168.1.237", help="Pi LiDAR stream host.")
    parser.add_argument("--lidar-port", type=int, default=8765)
    parser.add_argument("--output-dir", default="artifacts/wander_snapshot_stitch")
    parser.add_argument(
        "--max-captures",
        type=int,
        default=0,
        help="Maximum snapshots to collect including the initial one. Use 0 for no limit.",
    )
    parser.add_argument("--forward-angle-deg", type=float, default=DEFAULT_LIDAR_FORWARD_ANGLE_DEG)
    parser.add_argument("--valid-angle-half-width-deg", type=float, default=180.0)
    parser.add_argument("--invert-lateral-axis", action="store_true", default=True)
    parser.add_argument("--no-invert-lateral-axis", dest="invert_lateral_axis", action="store_false")
    parser.add_argument("--max-distance-m", type=float, default=8.0)
    parser.add_argument(
        "--min-range-m",
        type=float,
        default=0.30,
        help="Returns closer than this are dropped from captures/tracking/frontiers. Must exceed "
        "the radius at which the lidar sees the robot's own shell (~0.25m), or every capture "
        "stitches phantom self-hit points into the map at the robot's location.",
    )
    parser.add_argument("--min-confidence", type=int, default=0)
    parser.add_argument("--fresh-frame-timeout-s", type=float, default=3.0)
    parser.add_argument("--fresh-frame-advances", type=int, default=1)
    parser.add_argument("--capture-settle-s", type=float, default=0.90)
    parser.add_argument("--tripwire-distance-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_DISTANCE_M)
    parser.add_argument("--tripwire-half-width-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_HALF_WIDTH_M)
    parser.add_argument("--tripwire-thickness-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_THICKNESS_M)
    parser.add_argument(
        "--arm-clearance-m",
        type=float,
        default=0.22,
        help="How far the ARM sticks out ahead of the lidar, plus stopping margin. "
        "Extends the lidar stop box FORWARD by this much so the (fast, 10Hz) box "
        "halts the base while the arm is still clear of the obstacle. Field "
        "2026-07-20: the box only reached 0.14m ahead of the lidar, the arm reaches "
        "further, so the base stopped ON the table and the arm rammed the edge. "
        "MEASURE the arm's forward reach beyond the lidar and set this to that + "
        "~0.12m stopping margin; lower it if the robot now stops too far from things.",
    )
    parser.add_argument("--min-distance-m", type=float, default=DEFAULT_LIDAR_STOP_BOX_MIN_DISTANCE_M)
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument("--move-speed", type=float, default=0.85)
    parser.add_argument(
        "--min-effective-move-speed",
        type=float,
        default=0.75,
        help="Minimum absolute x.vel command to use once a forward burst is requested so wheel stiction is overcome.",
    )
    parser.add_argument("--move-burst-s", type=float, default=1.40)
    parser.add_argument(
        "--drive-bursts-per-capture",
        type=int,
        default=4,
        help="How many forward bursts to chain together before stopping for the next stitched capture.",
    )
    parser.add_argument(
        "--max-drive-bursts-per-capture",
        type=int,
        default=6,
        help="Upper cap on chained forward bursts when the frontier ahead is especially open.",
    )
    parser.add_argument(
        "--inter-burst-pause-s",
        type=float,
        default=0.02,
        help="Short pause between chained forward bursts.",
    )
    parser.add_argument("--move-settle-s", type=float, default=0.70)
    parser.add_argument(
        "--frontier-drive-steer-gain",
        type=float,
        default=0.0035,
        help="Converts small residual frontier angle error in degrees into forward-drive yaw rate after heading alignment.",
    )
    parser.add_argument(
        "--max-drive-steer-theta-vel",
        type=float,
        default=0.10,
        help="Maximum residual yaw rate applied while driving toward a frontier.",
    )
    parser.add_argument(
        "--drive-yaw-hold",
        choices=["on", "off"],
        default="on",
        help="Hold heading straight during forward drives using the IMU gyro (corrects mecanum drift "
        "that curves the robot into furniture). Uses RELATIVE yaw only; falls back to lidar steer if "
        "the IMU is absent/stale/sign-untrustworthy.",
    )
    parser.add_argument(
        "--drive-yaw-hold-gain",
        type=float,
        default=0.010,
        help="Yaw-hold proportional gain: theta.vel (rad/s) commanded per degree the heading has "
        "drifted from the leg's start heading.",
    )
    parser.add_argument(
        "--drive-yaw-hold-max",
        type=float,
        default=0.35,
        help="Maximum corrective yaw rate (rad/s) the IMU yaw-hold may command during a forward drive.",
    )
    parser.add_argument(
        "--drive-steer-deadband-deg",
        type=float,
        default=10.0,
        help="If the chosen frontier is within this heading error, drive straight with zero yaw correction.",
    )
    parser.add_argument("--drive-hint-mps", type=float, default=0.75, help="Used only as the initial translation guess for stitching.")
    parser.add_argument("--drive-search-xy-m", type=float, default=0.35)
    parser.add_argument("--drive-theta-window-deg", type=float, default=12.0)
    parser.add_argument("--turn-direction", choices=("ccw", "cw"), default="ccw")
    parser.add_argument("--turn-deg", type=float, default=90.0)
    parser.add_argument("--turn-speed", type=float, default=0.82)
    parser.add_argument("--turn-burst-s", type=float, default=0.24)
    parser.add_argument("--turn-settle-s", type=float, default=0.18)
    parser.add_argument("--rotation-signature-min-distance-m", type=float, default=0.10)
    parser.add_argument("--rotation-signature-bin-deg", type=float, default=3.0)
    parser.add_argument("--stop-tolerance-deg", type=float, default=12.0)
    parser.add_argument("--min-turn-overlap-ratio", type=float, default=0.18)
    parser.add_argument("--max-turn-bursts", type=int, default=32)
    parser.add_argument("--turn-search-xy-m", type=float, default=0.30)
    parser.add_argument("--turn-theta-window-deg", type=float, default=54.0)
    parser.add_argument("--frontier-min-distance-m", type=float, default=1.20)
    parser.add_argument("--frontier-bin-deg", type=float, default=6.0)
    parser.add_argument(
        "--frontier-goal-step-m",
        type=float,
        default=1.35,
        help="World-space distance to step toward a stitched-map frontier before reseeding another exploration goal.",
    )
    parser.add_argument(
        "--frontier-goal-reached-m",
        type=float,
        default=0.45,
        help="Distance threshold for considering a stitched-map exploration target reached.",
    )
    parser.add_argument(
        "--frontier-align-threshold-deg",
        type=float,
        default=45.0,
        help="Only used for blocked/periodic scan turns; smart mode now prefers driving forward when the stop box is clear.",
    )
    parser.add_argument(
        "--wander-mode",
        choices=("smart", "scan_turn_every_capture", "turn_only"),
        default="smart",
        help=(
            "smart: drive when clear and rotate when blocked, with occasional scan turns; "
            "scan_turn_every_capture: drive burst then rotate before every capture; "
            "turn_only: never drive, just rotate/capture."
        ),
    )
    parser.add_argument(
        "--turn-every-capture",
        action="store_true",
        default=None,
        help="Legacy alias for scan_turn_every_capture behavior.",
    )
    parser.add_argument(
        "--no-turn-every-capture",
        dest="turn_every_capture",
        action="store_false",
        default=None,
        help="Legacy alias for smart behavior.",
    )
    parser.add_argument(
        "--scan-turn-interval",
        type=int,
        default=6,
        help="In smart mode, insert a scan turn after this many clear forward captures.",
    )
    parser.add_argument(
        "--bootstrap-turn-captures",
        type=int,
        default=4,
        help="In smart mode, spend the first N captures rotating in place to build an initial stitched room outline before driving.",
    )
    parser.add_argument("--stitch-resolution-m", type=float, default=0.03)
    parser.add_argument(
        "--robot-radius-m",
        type=float,
        default=0.29,
        help="Planning collision radius: 18in / 0.229m body half-width plus a 0.06m "
        "clearance margin. Map frontiers behind narrower gaps are unreachable.",
    )
    parser.add_argument(
        "--squeeze-radius-m",
        type=float,
        default=0.22,
        help="Reduced planning radius used ONLY on the squeeze/doorway retry paths "
        "(OBSERVE->SQUEEZE and the SQUEEZE replan). The full 0.29m radius seals a real "
        "doorway: field 2026-07-20 the 25in (0.635m) hallway to the exit left only ~1 "
        "free cell after 0.27m inflation at 0.05m grid, so a single noisy doorframe point "
        "made the planner declare it impassable and the robot turned around. The true "
        "body half-width is 0.229m (17in); 0.22m rounds to a 4-cell inflation that leaves "
        "a ~4-cell channel the robot physically fits. Real collision safety is owned by "
        "the live measured-gap squeeze stop-box, NOT this planning radius.",
    )
    parser.add_argument(
        "--lidar-offset-forward-m",
        type=float,
        default=0.2286,
        help="How far the LiDAR sits forward of the robot's rotation center (9in default). "
        "Used to predict the sensor translation caused by in-place turns.",
    )
    parser.add_argument(
        "--min-append-score",
        type=float,
        default=5.5,
        help="Minimum stitch-solver score required to append a capture to the map. "
        "Captures scoring below this are discarded and retried instead of corrupting the map.",
    )
    parser.add_argument(
        "--on-complete",
        choices=("idle", "stop"),
        default="idle",
        help="When the map is complete: 'idle' holds position and watches the live scan, "
        "resuming exploration if new reachable space appears (e.g. a door opens); "
        "'stop' exits immediately.",
    )
    parser.add_argument(
        "--max-consecutive-append-discards",
        type=int,
        default=2,
        help="After this many consecutive discarded captures, the best available pose is "
        "accepted anyway (with a warning) so the run cannot stall forever.",
    )
    parser.add_argument("--rerun-mode", choices=("web", "local"), default="web")
    parser.add_argument("--rerun-grpc-port", type=int, default=9877)
    parser.add_argument("--rerun-web-port", type=int, default=9878)
    parser.add_argument(
        "--camera-feedback-hz",
        type=float,
        default=5.0,
        help="Live Rerun camera-panel update rate (0 disables). Shows the annotated front eyes "
        "and bottom camera WITH the depth model's elevated-hazard overlay (red/orange on "
        "table/counter edges) — the panel to watch to see whether the model catches an edge "
        "lip before the base reaches it. Pass --camera-feedback-hz 0 to hide the panels and "
        "show just the map. Does NOT affect the safety gate; this only controls the Rerun panels.",
    )
    parser.add_argument(
        "--record-eye-frames-hz",
        type=float,
        default=2.0,
        help="Save the annotated eye frames (with the depth elevated-hazard overlay) to "
        "artifacts/wander_snapshot_stitch/eye_record/ throughout the run, at this rate "
        "(0 disables). Numbered f00001.., so the LAST frames are the end of the run — scrub "
        "to the moment it reached a table edge to see exactly what the depth model painted "
        "(the ram leaves no stop bundle because the depth never fires). Recorded during "
        "drives too. Filename carries the gate decision + classification.",
    )
    parser.add_argument(
        "--elevated-safety",
        choices=("off", "on"),
        default="on",
        help="Camera-based elevated obstacle gate (table/counter/shelf edges the "
        "lidar cannot see). ON by default: without it, nothing can stop the base "
        "from driving under a table edge. 'off' only for bench runs with no "
        "elevated hazards.",
    )
    parser.add_argument(
        "--elevated-block-distance-m",
        type=float,
        default=0.75,
        help="Distance at which a camera-detected elevated edge (table lip, counter) "
        "STOPS forward motion. Raised from the old 0.55m to 0.75m (field 2026-07-20: "
        "the robot rammed a white table edge because the base could not stop in the "
        "0.55m the depth model's ~0.55s latency left it). Blocking from further out "
        "gives the base room to halt before the arm/body reaches the edge; lower it if "
        "the robot now stops too far from things it should approach.",
    )
    parser.add_argument(
        "--semantic-edge-detector",
        choices=("off", "on"),
        default="off",
        help="SECOND perception leg (field 2026-07-20): an open-vocab detector "
        "(YOLO-World) that recognizes table/counter/shelf edges SEMANTICALLY where "
        "the monocular DEPTH model fails on white/textureless surfaces. Its stop is "
        "OR-ed with the depth gate. OFF by default while it is tuned; turn ON to "
        "test. REQUIRES ultralytics in the run env: add `--with ultralytics` to the "
        "`uv run` command. Fail-soft: if it can't load, the depth gate + lidar box "
        "still protect and the run continues.",
    )
    parser.add_argument(
        "--slam-input-endpoint",
        type=str,
        default="",
        help="slam_input.v1 camera stream endpoint (default tcp://<remote-ip>:5560).",
    )
    parser.add_argument(
        "--eye-perception",
        choices=("depth", "hough"),
        default="depth",
        help="Eye-camera hazard perception: 'depth' = Depth-Anything-V2 metric "
        "depth (3D obstacle test, lidar-anchored scale; falls back to 'hough' "
        "until the model is ready or if it fails to load); 'hough' = the "
        "classic line detector + parallax pipeline.",
    )
    parser.add_argument(
        "--eye-fusion",
        choices=("panorama", "per-eye"),
        default="panorama",
        help="How depth perception sees: 'panorama' (default) fuses both eyes "
        "into ONE calibrated central forward view (requires the eye panorama "
        "calibration — run scripts/sourccey_eye_panorama.py --mode capture "
        "then --mode calibrate; aborts loudly if missing); 'per-eye' uses the "
        "legacy per-eye camera models (yaw/roll known to be off).",
    )
    parser.add_argument(
        "--edge-detection",
        choices=("on", "off"),
        default="on",
        help="Camera-based ELEVATED edge detection (the eye/depth pipeline for "
        "tabletop/counter/shelf lips the lidar cannot see). 'off' disables it "
        "entirely — no depth model is loaded, and the eyes produce no elevated "
        "stops, holds, or edge-map cells; the bottom-camera FLOOR gate and the "
        "lidar stop box still protect. Use when the elevated edges are at lidar "
        "height (e.g. draped solid) so the lidar itself maps them.",
    )
    parser.add_argument(
        "--use-imu-heading",
        choices=("on", "off"),
        default="on",
        help="Use the host's integrated-gyro yaw as a heading PRIOR to disambiguate "
        "room-symmetric scan matches. The host must be publishing yaw (imu_yaw_pub_enabled, "
        "default on). Only yaw DELTAS between captures are used, so gyro drift is irrelevant. "
        "When 'off' or the yaw feed is silent, the loop falls back to lidar-only heading "
        "(exactly the prior behavior). This is what keeps the robot from getting permanently "
        "lost after a lock-lost turn into an open doorway.",
    )
    parser.add_argument(
        "--imu-host",
        default=None,
        help="Host serving the integrated-yaw ZMQ PUB socket. Defaults to --remote-ip.",
    )
    parser.add_argument(
        "--imu-yaw-port",
        type=int,
        default=8770,
        help="Port for the host's integrated-yaw PUB socket (host imu_yaw_pub_endpoint).",
    )
    parser.add_argument(
        "--imu-yaw-sign",
        type=float,
        default=1.0,
        help="Sign applied to the received yaw so +yaw matches the robot's CCW (SLAM theta) "
        "convention. If the run warns that the IMU sign looks inverted, pass -1.",
    )
    parser.add_argument(
        "--motion-calibration",
        default="artifacts/sourccey_motion_calibration.json",
        help="Path to the motion-calibration file (from scripts/sourccey_wander_calibration.py). "
        "Its commanded->real scale factors dead-reckon motion the LiDAR could not measure "
        "(lock-lost turns, blind maneuvers). Missing file = uncalibrated 1.0x (still runs).",
    )
    parser.add_argument(
        "--edge-mapping",
        choices=("on", "off"),
        default="on",
        help="When an elevated edge stops the wanderer (requires --elevated-safety on): "
        "back off, re-approach slowly to MEASURE it by parallax, back off again, and "
        "sweep the heading to map its full extent — then plan around the mapped "
        "boundary for the rest of the run. 'off' keeps only the plain safety stops.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    snapshot_dir = output_dir / "snapshots"
    stitch_dir = output_dir / "offline_stitch"
    _ensure_clean_directory(output_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stitch_dir.mkdir(parents=True, exist_ok=True)

    rerun_grpc_port, rerun_web_port = _pick_free_rerun_ports(
        int(args.rerun_grpc_port), int(args.rerun_web_port)
    )
    rr, viewer_url = _init_rerun(
        session_name="ldlidar_wander_snapshot_stitch",
        mode=args.rerun_mode,
        grpc_port=rerun_grpc_port,
        web_port=rerun_web_port,
    )

    # ARM-CLEARANCE stop box: extend the box's forward reach by the arm's
    # forward stick-out (+ stopping margin) so the fast lidar halts the BASE
    # before the ARM — which reaches ahead of the lidar — contacts an obstacle
    # (field 2026-07-20: 0.14m box + longer arm = base stops on the table, arm
    # rams the edge). The lateral squeeze band is unchanged (doorways still
    # thread); this only pushes the FAR edge outward.
    _arm_extended_tripwire_m = float(args.tripwire_distance_m) + max(0.0, float(args.arm_clearance_m))
    zone_cfg = StopZoneConfig(
        forward_angle_deg=float(args.forward_angle_deg),
        min_distance_m=float(args.min_distance_m),
        tripwire_distance_m=_arm_extended_tripwire_m,
        tripwire_half_width_m=float(args.tripwire_half_width_m),
        tripwire_thickness_m=float(args.tripwire_thickness_m),
        min_points_to_trigger=max(int(args.min_points), 1),
        # Narrow squeeze band: 4cm inside the full box, floored just past the
        # body's half-width so anything it counts is a genuine collision course.
        squeeze_half_width_m=max(0.26, float(args.tripwire_half_width_m) - 0.02),
    )

    feed = SelfMaskedLidarFeed(args.lidar_host, int(args.lidar_port))
    feed.start()

    robot = SourcceyClient(SourcceyClientConfig(id=args.robot_id, remote_ip=args.remote_ip))
    robot.connect()
    _send_stop(robot)

    # Motion calibration: commanded->real scale factors (from
    # scripts/sourccey_wander_calibration.py). Used ONLY to dead-reckon motion
    # the LiDAR could not measure (e.g. a lock-lost turn), never to override a
    # real measurement. Missing file degrades to uncalibrated 1.0x.
    motion_calibration = MotionCalibration.load(args.motion_calibration)

    # Integrated-gyro yaw heading prior. Decoupled from the lidar feed and the
    # camera/observation stream: a dedicated PUB socket on the host. Optional and
    # non-fatal — if it never delivers, the loop stays lidar-only.
    imu_yaw: ImuYawClient | None = None
    if str(args.use_imu_heading) == "on":
        imu_host = args.imu_host or args.remote_ip
        imu_yaw = ImuYawClient(
            f"tcp://{imu_host}:{int(args.imu_yaw_port)}",
            sign=float(args.imu_yaw_sign),
        )
        imu_yaw.start()
    else:
        print("[wander] IMU heading prior OFF (--use-imu-heading off); heading is lidar-only")

    def _drive_hold_yaw_deg() -> float | None:
        """Current gyro yaw (deg) for the forward-drive heading hold, or None when
        the IMU is off/absent/stale or its sign has proven untrustworthy against
        lidar turns — in which case the drive falls back to the lidar-derived
        steer. The hold consumes only the CHANGE in this value since a leg began
        (never an absolute frame), so it needs no shared zero with the lidar pose;
        the calibration measured the gyro tracks relative rotation to ~0.5deg."""
        if imu_yaw is None or not imu_yaw.sign_trustworthy():
            return None
        return imu_yaw.deg()

    _drive_yaw_hold_gain = (
        float(args.drive_yaw_hold_gain) if str(args.drive_yaw_hold) == "on" else 0.0
    )
    if str(args.drive_yaw_hold) == "on":
        print(
            f"[wander] forward-drive IMU yaw-hold ENABLED (gain={float(args.drive_yaw_hold_gain):.3f} "
            f"rad/s per deg, max={float(args.drive_yaw_hold_max):.2f} rad/s) — holds each drive leg's "
            "start heading to counter mecanum curve; lidar still owns position"
        )

    # Optional camera-based elevated obstacle gate. The lidar map stays the
    # map owner; the eye cameras only VETO unsafe motion (forward on any
    # active hazard, rotation too in the near tier; reverse always allowed).
    hazard_monitor = None
    hazard_subscriber = None
    semantic_worker = None
    if str(args.elevated_safety) == "on":
        from sourccey_elevated_safety import (
            ElevatedHazardMonitor,
            ElevatedSafetyConfig,
            SlamCameraSubscriber,
            endpoint_from_remote_ip,
        )

        safety_endpoint = str(args.slam_input_endpoint).strip() or endpoint_from_remote_ip(
            args.remote_ip
        )
        edge_detection_on = str(args.edge_detection) == "on"
        safety_config = ElevatedSafetyConfig(
            slam_input_endpoint=safety_endpoint,
            elevated_eye_enabled=edge_detection_on,
            forward_block_distance_m=float(args.elevated_block_distance_m),
        )
        print(
            "[safety] elevated edge forward-block distance = "
            f"{float(args.elevated_block_distance_m):.2f}m (stops the base this far before a "
            "camera-detected table/counter edge)"
        )
        if not edge_detection_on:
            print(
                "[safety] elevated EDGE DETECTION DISABLED (--edge-detection off): no depth "
                "model, no elevated eye stops/holds/edge cells. Floor gate + lidar stop box "
                "still active. Elevated obstacles are protected ONLY if the lidar sees them."
            )
        hazard_subscriber = SlamCameraSubscriber(
            endpoint=safety_endpoint,
            camera_keys=(
                safety_config.left_key,
                safety_config.right_key,
                safety_config.bottom_key,
            ),
        )
        hazard_subscriber.start()
        if hazard_subscriber.wait_for_frames(
            timeout_s=5.0, required=(safety_config.left_key, safety_config.right_key)
        ):
            # The lidar referees eye candidates: a candidate whose if-on-floor
            # position matches a lidar return is a floor-standing object the
            # stop box already owns, not an elevated hazard.
            def _safety_lidar_ranges():
                _, safety_frame = feed.latest()
                if safety_frame is None:
                    return None
                safety_points = _scan_to_local_points(
                    points=safety_frame.points,
                    forward_angle_deg=float(args.forward_angle_deg),
                    valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                    invert_lateral_axis=bool(args.invert_lateral_axis),
                    max_distance_m=float(args.max_distance_m),
                    min_confidence=int(args.min_confidence),
                    min_range_m=float(args.min_range_m),
                )
                if len(safety_points) == 0:
                    return None
                return (
                    np.degrees(np.arctan2(safety_points[:, 1], safety_points[:, 0])),
                    np.hypot(safety_points[:, 0], safety_points[:, 1]),
                )

            depth_worker = None
            if edge_detection_on and str(args.eye_perception) == "depth":
                # NO FALLBACK: depth requested = depth required. If it cannot
                # load, the run aborts with the reason — never a silent switch
                # to the old detector (use --eye-perception hough explicitly
                # for that pipeline).
                from sourccey_depth_perception import DepthWorker

                eye_mosaic = None
                if str(args.eye_fusion) == "panorama":
                    # Fused central vision: both eyes hard-cut onto the
                    # CALIBRATED virtual forward camera; one inference on one
                    # straight-ahead view (the legacy per-eye models carried
                    # yaw ~5deg off and unmodeled ~6-10deg roll — every
                    # bearing and edge placement inherited that error).
                    # NO FALLBACK: panorama requested = calibration required.
                    from sourccey_eye_panorama import load_perception_mosaic

                    eye_mosaic = load_perception_mosaic()
                    print(
                        "[safety] eye fusion: calibrated panorama "
                        f"({eye_mosaic.virt.width}x{eye_mosaic.virt.height}, "
                        f"hfov={eye_mosaic.model.hfov_deg:.1f}deg, seam band "
                        f"cols {eye_mosaic.seam_cols[0]}..{eye_mosaic.seam_cols[1]} forgiven)"
                    )
                depth_worker = DepthWorker(
                    hazard_subscriber,
                    {
                        safety_config.left_key: safety_config.eye_left_model,
                        safety_config.right_key: safety_config.eye_right_model,
                    },
                    lidar_ranges_fn=_safety_lidar_ranges,
                    edge_detector_config=safety_config.detector_config(),
                    mosaic=eye_mosaic,
                )
                depth_worker.start()
                print(
                    "[safety] loading depth perception (Depth-Anything-V2 metric indoor; "
                    "first run downloads the weights) ..."
                )
                load_started = time.monotonic()
                while not depth_worker.ready and depth_worker.failure is None:
                    if time.monotonic() - load_started > 300.0:
                        raise SystemExit(
                            "[safety] FATAL: depth perception did not load within 300s"
                        )
                    time.sleep(0.5)
                if depth_worker.failure is not None:
                    raise SystemExit(
                        f"[safety] FATAL: depth perception failed to load "
                        f"({depth_worker.failure}). Fix the environment (is "
                        f"'--with transformers' in the run command?) or run with "
                        f"--eye-perception hough explicitly."
                    )
                # Model loaded — now require a first result on every key the
                # worker serves (the fused panorama, or both eyes) so the
                # mission starts with live depth, not a stale-stop.
                first_result_deadline = time.monotonic() + 60.0
                while time.monotonic() < first_result_deadline:
                    if all(
                        depth_worker.latest(eye, max_age_s=10.0) is not None
                        for eye in depth_worker.eye_keys
                    ):
                        break
                    if depth_worker.failure is not None:
                        raise SystemExit(
                            f"[safety] FATAL: depth inference failed on the first "
                            f"frames ({depth_worker.failure})"
                        )
                    time.sleep(0.25)
                else:
                    raise SystemExit(
                        "[safety] FATAL: depth perception produced no results "
                        "within 60s of loading"
                    )
                print(
                    f"[safety] depth perception live on {'/'.join(depth_worker.eye_keys)} "
                    f"(inference ~{max(depth_worker.inference_s, 0.01):.2f}s/frame)"
                )

            # SECOND perception leg (opt-in): the semantic edge detector. Only
            # meaningful with the fused panorama (it needs one calibrated forward
            # view). Fail-soft — a load failure leaves it non-blocking.
            semantic_worker = None
            if str(args.semantic_edge_detector) == "on" and eye_mosaic is not None:
                from sourccey_semantic_edge import SemanticEdgeConfig, SemanticEdgeWorker

                semantic_worker = SemanticEdgeWorker(
                    hazard_subscriber,
                    eye_mosaic,
                    SemanticEdgeConfig(
                        enabled=True,
                        panorama_hfov_deg=float(eye_mosaic.model.hfov_deg),
                    ),
                )
                semantic_worker.start()
                print(
                    "[safety] semantic edge detector ENABLED (YOLO-World) — second "
                    "stop leg for white table/counter edges the depth model misses"
                )
            elif str(args.semantic_edge_detector) == "on":
                print(
                    "[safety] --semantic-edge-detector on but the fused panorama is "
                    "unavailable; the semantic leg is skipped this run"
                )
            hazard_monitor = ElevatedHazardMonitor(
                safety_config,
                hazard_subscriber,
                lidar_ranges_fn=_safety_lidar_ranges,
                depth_worker=depth_worker,
                semantic_worker=semantic_worker,
            )
            hazard_monitor.start()
            bottom_present = hazard_subscriber.latest(safety_config.bottom_key)[0] is not None
            bottom_label = "present" if bottom_present else "ABSENT (low floor objects unprotected)"
            if not edge_detection_on:
                eye_perception_label = "DISABLED (--edge-detection off)"
            elif depth_worker is not None:
                eye_perception_label = "depth" + (
                    "+panorama" if "panorama" in depth_worker.eye_keys else ""
                )
            else:
                eye_perception_label = "hough"
            print(
                f"[safety] anti-collision gate ARMED (cameras via {safety_endpoint}, "
                f"bottom_camera={bottom_label}, lidar_referee=on, "
                f"eye_perception={eye_perception_label})"
            )
        else:
            hazard_subscriber.stop()
            hazard_subscriber = None
            print(
                "[safety] WARNING: no camera frames from the slam_input stream within 5s; "
                f"running WITHOUT the elevated obstacle gate (endpoint={safety_endpoint})"
            )
    else:
        print(
            "[safety] WARNING: elevated safety is OFF — table/counter/shelf edges "
            "above the lidar plane are INVISIBLE and the base WILL drive into them"
        )

    # Camera-confirmed elevated geometry, stamped into the PLANNING map
    # (world frame) so the wanderer never routes into a table the lidar
    # cannot see. Session-scoped: the map frame is rebuilt each run.
    # Depth mode stamps floor-projected FOOTPRINT POINTS (the region's true
    # shape); hough mode stamps fitted segments.
    elevated_segments_world: list[dict] = []
    # Rendered red points keyed by their 4cm map cell so floor (free-space)
    # evidence can remove them along with the planning cell.
    # Rendered red points keyed by their 4cm map cell so floor (free-space)
    # evidence can remove them along with the planning cell.
    elevated_points_world: dict[tuple[int, int], tuple[float, float, float]] = {}
    # Two-frame persistence gate for FAR footprint cells: beyond 0.9m a cell
    # becomes a planning obstacle only after being observed elevated in two
    # frames separated by >= 0.75s (a different frame, not this moment's
    # other eye). Monocular depth's worst failures are SINGLE-FRAME —
    # textureless-wall bulges and scale wobble (field 2026-07-13:
    # x1.08..x1.77 across frames) splatted one-off cells over the exit
    # corridor, and the add-only map kept them forever. Real furniture
    # re-observes on the same cells; noise does not. cell -> (mono, hits)
    elevated_pending_world: dict[tuple[int, int], tuple[float, int]] = {}
    # Decay metadata for DRIVE-BY eye cells: cell -> [stamp_bearing_deg,
    # miss_count]. A cell the fused view re-inspects (similar bearing, good
    # range) WITHOUT re-detecting counts a miss; 3 misses remove it.
    # Cells measured by an investigation (_stamp_world_segment) have no
    # entry — approach-verified geometry is permanent. Detection needs two
    # agreeing frames to add a cell; absence symmetrically takes it back
    # (field 2026-07-17: edge-bleed cells at a doorway were permanent
    # unfalsifiable walls that sealed a passable corridor).
    elevated_cell_meta: dict[tuple[int, int], list] = {}
    elevated_occupied_world: set[tuple[int, int]] = set()
    # PERMANENT no-go cells (0.04m grid, NEVER decay): where a camera-confirmed
    # elevated obstacle actually STOPPED the robot from driving. Field 2026-07-20
    # (user): the lidar sees a "doorway" THROUGH a desk, the robot approaches, the
    # cameras refuse to ram it, and it re-commits to the SAME phantom doorway
    # forever because the lidar still sees the gap. Unlike elevated_occupied_world
    # (which decays as the lidar "sees through" furniture), these are stamped from
    # an actual forward-block event and stay put, so the opening picker permanently
    # routes AROUND the furniture and re-plans toward a genuinely clear opening.
    camera_blocked_world: set[tuple[int, int]] = set()

    def _stamp_camera_block(pose) -> None:
        """Stamp an elevated obstacle's footprint ~0.3-0.7m ahead of ``pose`` (a
        small patch across the body width) into the PERMANENT camera-blocked set,
        so the opening picker routes around it and re-plans. Called whenever a
        CAMERA gate actually stops forward motion (the robot is nose-to-furniture
        the lidar sees under)."""
        before = len(camera_blocked_world)
        th = math.radians(float(pose.theta_deg))
        cth, sth = math.cos(th), math.sin(th)
        for fwd in (0.30, 0.42, 0.54, 0.66):
            for lat in (-0.16, 0.0, 0.16):
                wx = float(pose.x) + cth * fwd - sth * lat
                wy = float(pose.y) + sth * fwd + cth * lat
                camera_blocked_world.add((int(round(wx / 0.04)), int(round(wy / 0.04))))
        if len(camera_blocked_world) > before:
            print(
                "[wander] camera-confirmed furniture marked as a PERMANENT no-go "
                f"ahead ({len(camera_blocked_world)} cells total) — the opening picker "
                "will route around it and re-plan, not re-commit to this bearing"
            )

    def _denial_is_camera_obstacle(reason: str) -> bool:
        """True when a forward-denial reason is a CAMERA-confirmed elevated edge
        (not the stop box, not a stale frame) — the case to stamp and re-plan."""
        return any(
            k in str(reason)
            for k in ("elevated", "semantic_edge_stop", "depth_elevated",
                      "vanished_near", "line_edge")
        )
    # Single-vantage promotion budget: a wedged/held robot re-observes its
    # own systematic depth artifacts from the SAME pose every cycle, so the
    # two-frame persistence gate confirms them forever (field 2026-07-17:
    # ~100 red cells piled up around one stuck pose and boxed the planner
    # in). Each ~0.25m/20deg pose bucket may promote at most 60 cells;
    # moving to a new vantage earns a fresh budget, so normal exploration
    # mapping is unaffected.
    elevated_vantage_promotions: dict[tuple[int, int, int], int] = {}
    investigated_spots_world: list[tuple[float, float]] = []
    sidestep_spots_world: list[tuple[float, float]] = []

    # Frontier goals that repeatedly failed (path blocked / hazard-held on
    # approach). Two strikes blacklist the frontier for a while so the
    # planner moves on to the rest of the room instead of grinding the same
    # unreachable scrap; entries expire so a transient block cannot
    # permanently hide real space.
    frontier_strike_counts: dict[tuple[int, int], int] = {}
    # Repeated LOCAL elevated-hazard stops on the approach to the SAME frontier.
    # A one-off furniture stop just maps+replans, but if the robot keeps getting
    # blocked by furniture heading for one goal, re-approaching it head-on is a
    # ram risk (field 2026-07-20: it turned back toward a table it had just been
    # stopped at and drove into it). After a few repeats the goal is blacklisted
    # so the planner routes elsewhere instead of re-approaching.
    local_hazard_stop_counts: dict[tuple[int, int], int] = {}
    frontier_blacklist: list[dict[str, float]] = []
    # Anti-fixation: if the SAME frontier is targeted across many consecutive
    # planning cycles while the robot stays pose-lost (never localizing a capture
    # there to strike it the normal way), abandon it and force exploration
    # elsewhere. Without this the robot re-drives the same phantom corner forever
    # — the rest of the room stays "already mapped" in its frozen belief.
    stuck_frontier_face: tuple[float, float] | None = None
    stuck_frontier_lost_cycles = 0
    # Goal commitment: the frontier face chosen last cycle keeps priority in
    # the planner until consumed or blacklisted, so the target cannot flip
    # sides every capture (turn-thrash).
    committed_frontier_face: tuple[float, float] | None = None

    direction_sign = 1.0 if args.turn_direction == "ccw" else -1.0
    signed_turn_deg = float(args.turn_deg) * float(direction_sign)
    wander_mode = str(args.wander_mode)
    if args.turn_every_capture is True:
        wander_mode = "scan_turn_every_capture"
    elif args.turn_every_capture is False and str(args.wander_mode) == "scan_turn_every_capture":
        wander_mode = "smart"

    print(
        "[wander] startup "
        f"mode={wander_mode} "
        f"move_speed={float(args.move_speed):.2f} "
        f"min_effective_move_speed={float(args.min_effective_move_speed):.2f} "
        f"move_burst_s={float(args.move_burst_s):.2f} "
        f"drive_bursts_per_capture={int(args.drive_bursts_per_capture)} "
        f"drive_steer_gain={float(args.frontier_drive_steer_gain):.3f} "
        f"drive_steer_max={float(args.max_drive_steer_theta_vel):.2f} "
        f"turn_deg={float(args.turn_deg):.1f} "
        f"turn_direction={args.turn_direction} "
        f"bootstrap_turn_captures={int(args.bootstrap_turn_captures)} "
        f"scan_turn_interval={int(args.scan_turn_interval)} "
        f"lidar_offset_forward={float(args.lidar_offset_forward_m):.3f}m "
        f"min_append_score={float(args.min_append_score):.2f} "
        f"frontier=(min_distance={float(args.frontier_min_distance_m):.2f}m, "
        f"bin={float(args.frontier_bin_deg):.1f}deg, "
        f"align_threshold={float(args.frontier_align_threshold_deg):.1f}deg, "
        f"goal_step={float(args.frontier_goal_step_m):.2f}m, "
        f"goal_reached={float(args.frontier_goal_reached_m):.2f}m) "
        f"stop_box=(forward={float(args.forward_angle_deg):.1f}deg, "
        f"min={float(args.min_distance_m):.2f}m, "
        f"depth={float(zone_cfg.tripwire_distance_m):.2f}m [base "
        f"{float(args.tripwire_distance_m):.2f}+arm {float(args.arm_clearance_m):.2f}], "
        f"far_edge={float(zone_cfg.tripwire_distance_m) + float(args.tripwire_thickness_m) / 2.0:.2f}m, "
        f"half_width={float(args.tripwire_half_width_m):.2f}m, "
        f"thickness={float(args.tripwire_thickness_m):.2f}m, "
        f"min_points={int(args.min_points)})"
    )
    motion_hints: list[MotionHint] = [
        MotionHint(
            kind="start",
            expected_dx_local_m=0.0,
            expected_dy_local_m=0.0,
            expected_dtheta_deg=0.0,
            search_xy_m=float(args.drive_search_xy_m),
            search_theta_window_deg=float(args.turn_theta_window_deg),
            label="initial_capture",
        )
    ]
    drive_checkpoints_since_scan = 0
    pending_turn_reason: str | None = None
    pending_turn_direction_sign = float(direction_sign)
    pending_turn_deg = float(args.turn_deg)
    force_drive_after_turn = False
    # One committed world-frame probe bearing during no_frontier episodes.
    # Field 2026-07-16: re-picking the widest live gap after every align
    # turn oscillated between openings on opposite sides (54 -> -42 -> 60
    # -> ...) and the robot pirouetted ~15 cycles without a single probe
    # drive. Commit once, finish turning to THAT bearing, then drive.
    probe_commit_world_deg: float | None = None
    probe_commit_align_turns = 0
    rotation_coverage_bin_count = 4
    rotation_coverage_bins_seen: set[int] = set()
    rotation_coverage_complete = False
    consecutive_turn_captures = 0
    active_explore_target: ExploreTarget | None = None
    active_explore_target_stall_count = 0
    active_explore_target_last_distance_m: float | None = None
    active_explore_target_blocked_count = 0
    force_live_frontier_cycles = 0
    # Countdown set whenever an elevated-edge HOLD denied forward motion. While
    # it is > 0 the robot's recent lack of progress is attributable to furniture
    # beside the path, NOT to an unresolvable frontier — so the redundant-capture
    # pirouette breaker must not blacklist the frontier it was approaching (field
    # 2026-07-20: a couch against the wall left of a hallway kept the exit
    # frontier's forward drive held; the breaker read the repeated same-vantage
    # captures as "doorway unresolvable" and blacklisted the real exit, and the
    # robot turned around and left the room).
    elevated_block_recent = 0
    # Deliberate get-unstuck (field 2026-07-20 user): when caught by furniture,
    # back off to a clear spot FIRST (once per episode), then reorient to a clear
    # opening and retry — instead of poking the furniture from new angles.
    caught_retreated = False
    pending_motion_hint: MotionHint | None = None
    # A failed turn capture is recovered by continuing around the room, not by
    # reversing into the same weak-reference view. Two 85deg sectors create
    # fresh overlap before the planner retries the deferred frontier.
    orbit_recovery_turns_remaining = 0
    orbit_recovery_direction_sign: float | None = None
    # At most one recovery turn may follow an unlocalized turn. If it also
    # loses tracking, further rotations are unobservable and only amplify the
    # heading error; force a fresh capture/drive cycle instead of spinning.
    orbit_recovery_turn_attempts = 0
    # Consecutive stationary "scan the new area" captures from one spot: a
    # second identical scan cannot add information, so >= 1 forces a turn.
    stationary_scan_streak = 0
    consecutive_append_discards = 0
    consecutive_blocked_cycles = 0
    failed_reverse_escapes = 0
    # ---- pose-integrity latch -------------------------------------------
    # THE map-corruption invariant (field 2026-07-17, twice): every ghost
    # stitch happened while the system ALREADY had loud evidence it was
    # lost — lock-lost turns, discarded captures, live relocalization
    # rejected at 2.7-3.2 — and appended anyway on a marginal solve
    # (wrong 90deg symmetry modes score up to ~11.9; healthy appends run
    # 13-16.5). When ANY lostness signal fires, the map becomes
    # READ-ONLY: every append path demands an absolute >=12.5 match and
    # elevated-cell stamping stops. The latch clears only when a
    # relocalization RE-PROVES the pose at >=12.0 — including a
    # wide-theta whole-map recovery search that ignores the (meaningless)
    # pose expectation, which is the exit this failure mode lacked.
    pose_lost = False
    POSE_LOST_APPEND_GATE = 12.5
    POSE_RECOVERY_MIN_SCORE = 12.0
    # The first appends after recovery re-anchor everything that follows —
    # keep them strict too (+2.0 on the gate for the next two appends).
    post_recovery_strict_appends = 0
    # Dead-reckoned heading bound for recovery (field 2026-07-17 run 9): in
    # a sparse near-symmetric room the WRONG 90/180deg mode scores 12-15 —
    # as high as truth — so score alone cannot pick the recovery basin. But
    # physics can: a turn that TRACKED cleanly bounds the true heading to
    # ~±30deg, and a recovery candidate 178deg away is impossible no matter
    # its score (observed: trusted at 31deg, tracked turn to ~-46deg, then
    # a 14.17-score recovery "restored" 131.5deg and later appends went in
    # under two different basins). The anchor theta advances by each turn's
    # TRACKED delta; slack grows by each turn's untracked remainder, so
    # after truly-blind rotation chains (run 7) the bound widens to
    # useless and recovery falls back to score-only — exactly right.
    dead_reck_theta_deg: float | None = None
    dead_reck_slack_deg = 15.0
    # Gyro anchor for the dead-reckoned heading: (imu_yaw_at_set, theta_at_set).
    # Re-derived from the gyro at the recovery tiebreaker so the heading stays
    # accurate through ANY rotation path (align turns, recovery spins, blind
    # bursts) — not just the main arc-tracked turn — which is what lets the
    # robot re-anchor instead of staying lost after a lock-lost turn.
    imu_dead_reck_anchor: tuple[float, float] | None = None
    # Consecutive redundant-vantage capture skips (see the redundant-vantage
    # gate at the checkpoint): capped so endless skipping inside covered
    # space cannot starve the planner of fresh captures forever.
    redundant_skip_streak = 0
    # EDGE-SURVEY bookkeeping: per-cluster (0.3m bucket) attempt counts and
    # last-adoption times. Each yellow cluster gets up to 3 deliberate
    # second-angle observation transits before it is left to the
    # squeeze/verification ladder.
    edge_survey_attempts: dict[tuple[int, int], int] = {}
    edge_survey_last_adopt: dict[tuple[int, int], float] = {}
    # Consecutive failed lost-recoveries. Against a ONE-snapshot map,
    # recovery can be mathematically unable to reach the 12.0 bar (field
    # 2026-07-17 run 11: first bootstrap turn lost lock, recoveries capped
    # at 8.6-10.3, robot deadlocked until Ctrl+C). A founding snapshot
    # holds no investment — after 2 failures the map is RE-FOUNDED from
    # the current position instead of waiting forever.
    lost_recovery_failures = 0
    # RELOCALIZE recovery budget (bounded "spin to find yourself", then move,
    # then HALT — it can NEVER doom-loop). `relocalize_spin_deg` is how far the
    # robot has deliberately spun in the current lost episode; `relocalize_cycles`
    # is how many full spin+translate cycles it has spent. Both reset the instant
    # the pose re-locks. After 2 cycles with no lock it stops moving and sits
    # (a visible halt beats endless thrashing or a blind ram). See the ladder below.
    relocalize_spin_deg = 0.0
    relocalize_cycles = 0
    relocalize_halted = False
    # Rotate-away attempts that lost tracking lock in the current stuck
    # episode. A point-starved corner (wedged start) makes rotation
    # untrackable: each attempt gets discarded and re-anchored BACK — an
    # oscillation. Allow one lock-lost rotate-away per episode; after that
    # the ladder escalates straight to the slow blind reverse.
    rotate_away_lock_losses = 0
    # Keep recovery turns in one direction until a real forward drive
    # succeeds. Per-frame left/right hazard flips must not make the robot
    # rotate back and forth in place.
    recovery_turn_sign: float | None = None
    wedged_events = 0
    no_frontier_cycles = 0
    survey_targets_used = 0
    active_survey_xy: tuple[float, float] | None = None
    unreachable_targets_world: list[tuple[float, float]] = []
    # EXIT RUN: once the room interior is mapped, tiny frontier slivers must not
    # hold the robot hostage (field 2026-07-18: a 9-cell phantom beyond the left
    # wall was re-targeted for the entire back half of a run — endless 85-150deg
    # align spins at a room it had already finished, while the real doorway sat
    # visible in the live scan). After enough consecutive sliver/no-frontier
    # plans the room is declared effectively mapped and the planner is forced
    # onto the live-opening probe path — commit to the deepest opening the lidar
    # actually sees and DRIVE THROUGH IT. Crossing into the next room makes real
    # frontiers appear, which clears the mode and resumes normal mapping there.
    sliver_frontier_cycles = 0
    # Explore-stall breaker. When the robot keeps chasing frontiers it can never
    # resolve (a narrow squeeze / an edge cluster), it MILLS in one small patch
    # while those >10-cell frontiers reset the room-mapped counter below — trapping
    # it short of the exit forever (field 2026-07-20: livelocked re-planning a
    # squeeze to (-0.64,-0.50) + an edge survey, never declared ROOM MAPPED, never
    # exited). The robust signal is POSITION, not which frontier it chases: if it
    # stays within EXIT_STALL_RADIUS_M of a reference for EXIT_STALL_CYCLE_LIMIT
    # planning cycles without reaching new ground, the reachable room is mapped and
    # remaining frontiers are treated as EXHAUSTED so the exit run can trigger.
    exit_stall_pos: tuple[float, float] | None = None
    exit_stall_cycles = 0
    EXIT_STALL_RADIUS_M = 0.7
    EXIT_STALL_CYCLE_LIMIT = 10
    # HARD anti-spin terminal (field 2026-07-20 user: "the endless rotate loop you
    # can't seem to stop it from ever doing"). Beyond the soft stall limit above,
    # if the robot sits inside EXIT_STALL_RADIUS_M for even LONGER it is spinning in
    # a dead-end pocket with no way onward. FIRST force a reverse-out the way it
    # came (it beelined IN, so behind it is open); if that still never frees it,
    # HALT to idle rather than rotate forever. These take priority over every other
    # motion decision.
    EXIT_STALL_ESCAPE_LIMIT = 10
    EXIT_STALL_STOP_LIMIT = 30
    # When STUCK and boxed (can't drive forward from THIS heading, can't reverse),
    # SCAN AROUND for a heading the body can drive before ever declaring defeat —
    # up to one full revolution. Field 2026-07-20: a "hold 4 cycles then halt" gave
    # up while facing a desk and NEVER turned to see the wide-open area right beside
    # it (the whole upper-left of the room went unexplored). Only halt after a full
    # circle finds nothing drivable = genuinely walled in on all sides. The watchdog
    # also DEFERS entirely to the planner whenever forward is drivable, so the
    # instant a scan-turn faces an opening the robot drives it instead of spinning.
    stuck_scan_rotations = 0
    STUCK_SCAN_ROTATIONS_HALT = 7   # ~7 x 55deg ~= a full 360 look-around
    # Cap on consecutive reverse-outs while stalled. Field 2026-07-20: at a doorway
    # with a desk in the mouth the robot oscillated approach->desk-blocks->reverse
    # forever, because a reverse ALWAYS succeeds there (room behind) so the watchdog
    # never escalated. After this many reverses without escaping the stall radius,
    # stop reversing (it isn't helping) and fall through to scan-around / halt.
    stuck_reverse_count = 0
    STUCK_REVERSE_CAP = 3
    exit_mode = False
    exit_scan_turns = 0
    # Consecutive exit-run probes that were BLOCKED (barely advanced). Unlike
    # exit_scan_turns (reset every cycle an opening is visible), this survives so a
    # doorway that is visible-but-blocked by furniture cannot be re-committed to
    # forever — the "rotating in the corner" livelock (field 2026-07-20: an
    # elevated table edge across the exit; probe blocked; robot rotated endlessly).
    # After each block it rotates ~55deg to hunt a DIFFERENT opening; after a full
    # revolution of blocked openings it HALTS instead of spinning in place.
    exit_probe_blocked_streak = 0
    EXIT_PROBE_BLOCKED_LIMIT = 7
    # When an exit opening is blocked by an ELEVATED edge (a desk the lidar sees
    # UNDER as a gap but the cameras correctly veto), go MEASURE that edge from a
    # fresh look instead of blindly spinning away — learn its true extent and
    # whether it actually spans the doorway (user 2026-07-20). Bounded per exit
    # episode; reset by a probe that actually advances.
    exit_edge_investigations = 0
    EXIT_EDGE_INVESTIGATION_LIMIT = 2
    # Consecutive captures that came back BOXED IN (nose against a wall). A
    # professional explorer must NEVER crash facing a wall — it backs out / rotates
    # and retries. Reset the moment a healthy capture lands.
    boxed_in_recoveries = 0
    # Vantage positions snapshotted when the exit run latches: completion is
    # POSITIONAL (robot must physically leave this coverage), never inferred
    # from frontier cells alone — cells open by merely seeing through the door.
    exit_latch_poses: list[tuple[float, float]] = []
    # The room the exit run latched in, frozen as a cell-set (see
    # EXIT_NEW_SPACE_MAX_OVERLAP). Completion requires the live scan to stop
    # matching this — proof the robot is looking at NEW space, not the same room.
    exit_latch_map_cells: set[tuple[int, int]] = set()
    # Measured door-crossing odometry (see EXIT_THROUGH_ODOMETRY_M): summed only
    # while pose-lost and advancing on the committed opening, i.e. across the
    # threshold, so it cannot be confused with a healthy in-room approach.
    exit_through_odometry_m = 0.0
    exit_opening_seen = False
    # Room-center geometry (see EXIT_BEYOND_ROOM_M). Set at latch; the outward-max
    # is the monotonic ratchet the hard-stop enforces so the robot can never
    # drive back toward the room center once it has committed to leaving.
    exit_centroid_xy: tuple[float, float] | None = None
    exit_room_radius_m = 0.0
    exit_outward_max_m = 0.0
    # True while the adopted plan is a squeeze-width traversal; switches the
    # stop box to its narrow band (StopZoneConfig.squeeze_active, see there).
    squeeze_traversal_active = False

    def _near_unreachable_target(x_m: float, y_m: float, radius_m: float = 0.60) -> bool:
        return any(
            math.hypot(float(x_m) - ux, float(y_m) - uy) <= float(radius_m)
            for ux, uy in unreachable_targets_world
        )

    def _blacklist_target(target: ExploreTarget, reason: str) -> None:
        unreachable_targets_world.append((float(target.world_x_m), float(target.world_y_m)))
        del unreachable_targets_world[:-12]
        print(
            "[wander] marking exploration target unreachable "
            f"(world=({float(target.world_x_m):.3f}, {float(target.world_y_m):.3f}), "
            f"reason={reason}, blacklist_size={len(unreachable_targets_world)})"
        )

    # Cycles remaining during which reverse escapes are refused because the
    # last one BUMPED (frame-to-frame motion ~zero against a commanded
    # reverse). Repeating the reverse would grind the body against the same
    # unseen obstacle — recovery must rotate/advance instead.
    reverse_bump_cooldown = 0

    def _attempt_reverse_escape_hint(start_pose: Pose2D, allow_blind: bool) -> MotionHint | None:
        nonlocal reverse_bump_cooldown
        if reverse_bump_cooldown > 0:
            reverse_bump_cooldown -= 1
            print(
                "[wander] reverse escape DISABLED after bump "
                f"({reverse_bump_cooldown + 1} cooldown cycles left) — an unseen obstacle "
                "is against the rear; recovery must rotate or advance instead"
            )
            return None
        reverse_result = _reverse_escape(
            robot=robot,
            feed=feed,
            transformed_sets=stitch_state["transformed_sets"],
            start_pose=start_pose,
            resolution_m=float(args.stitch_resolution_m),
            robot_radius_m=float(args.robot_radius_m),
            lidar_offset_forward_m=float(args.lidar_offset_forward_m),
            reverse_speed=max(float(args.min_effective_move_speed), float(args.move_speed) * 0.9),
            burst_s=min(1.0, float(args.move_burst_s)),
            bursts=2,
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            invert_lateral_axis=bool(args.invert_lateral_axis),
            max_distance_m=float(args.max_distance_m),
            min_range_m=float(args.min_range_m),
            min_confidence=int(args.min_confidence),
            allow_blind=bool(allow_blind),
        )
        if reverse_result is None:
            return None
        reverse_pose, reverse_meta = reverse_result
        if bool(reverse_meta.get("bumped")):
            reverse_bump_cooldown = 6
            # Physical contact IS a measurement — the strongest one there is.
            # Stamp the contact point behind the rear bumper as a CONFIRMED
            # obstacle cell so the planner/guards finally know the desk the
            # lidar cannot see is there (user 2026-07-18: "it clearly should
            # have seen that area as an edge"). Only with a trusted pose — a
            # lost pose would stamp the contact somewhere fictional.
            if not pose_lost:
                contact_local_x = -(
                    float(args.lidar_offset_forward_m) + float(args.robot_radius_m) + 0.03
                )
                heading_rad = math.radians(float(reverse_pose.theta_deg))
                contact_x = float(reverse_pose.x) + math.cos(heading_rad) * contact_local_x
                contact_y = float(reverse_pose.y) + math.sin(heading_rad) * contact_local_x
                contact_cell = (
                    int(round(contact_x / 0.04)),
                    int(round(contact_y / 0.04)),
                )
                elevated_occupied_world.add(contact_cell)
                elevated_cell_meta[contact_cell] = [
                    math.degrees(math.atan2(contact_y - reverse_pose.y, contact_x - reverse_pose.x)),
                    0,
                    True,
                ]
                print(
                    "[edge-map] BUMP contact stamped as a CONFIRMED obstacle cell at "
                    f"({contact_x:.2f}, {contact_y:.2f}) — measured by physical contact"
                )
        ddx_world = float(reverse_pose.x) - float(start_pose.x)
        ddy_world = float(reverse_pose.y) - float(start_pose.y)
        theta0_rad = math.radians(float(start_pose.theta_deg))
        cos0 = math.cos(theta0_rad)
        sin0 = math.sin(theta0_rad)
        return MotionHint(
            kind="drive",
            expected_dx_local_m=cos0 * ddx_world + sin0 * ddy_world,
            expected_dy_local_m=-sin0 * ddx_world + cos0 * ddy_world,
            expected_dtheta_deg=_normalize_angle_deg(
                float(reverse_pose.theta_deg) - float(start_pose.theta_deg)
            ),
            search_xy_m=0.45 if bool(reverse_meta.get("locked")) else 0.90,
            search_theta_window_deg=18.0 if bool(reverse_meta.get("locked")) else 30.0,
            label=f"reverse_escape_{capture_index:02d}",
        )

    def _active_frontier_blacklist() -> list[tuple[float, float]]:
        return [
            (float(entry["x"]), float(entry["y"]))
            for entry in frontier_blacklist
            if float(capture_index) - float(entry["added_at"]) <= 30.0
        ]

    def _strike_frontier(face_xy, reason: str, force: bool = False) -> None:
        """A planned frontier failed (path blocked / hazard-held). Two strikes
        blacklist it for ~30 captures: the planner explores the REST of the
        room instead of re-planning the same doomed goal every cycle.
        force=True blacklists immediately (provably-failing actions, e.g. a
        lock-lost face-turn, must never be retried at all)."""
        nonlocal committed_frontier_face
        strike_key = (
            int(round(float(face_xy[0]) / 0.4)),
            int(round(float(face_xy[1]) / 0.4)),
        )
        if force:
            frontier_strike_counts[strike_key] = max(
                2, frontier_strike_counts.get(strike_key, 0) + 1
            )
        else:
            frontier_strike_counts[strike_key] = frontier_strike_counts.get(strike_key, 0) + 1
        if committed_frontier_face is not None and (
            math.hypot(
                float(face_xy[0]) - committed_frontier_face[0],
                float(face_xy[1]) - committed_frontier_face[1],
            )
            <= 0.6
        ):
            # A failed attempt breaks the commitment so the planner is free
            # to pick a different frontier next cycle.
            committed_frontier_face = None
        already_listed = any(
            math.hypot(float(face_xy[0]) - ax, float(face_xy[1]) - ay) <= 0.4
            for ax, ay in _active_frontier_blacklist()
        )
        if frontier_strike_counts[strike_key] >= 2 and not already_listed:
            frontier_blacklist.append(
                {
                    "x": float(face_xy[0]),
                    "y": float(face_xy[1]),
                    "added_at": float(capture_index),
                }
            )
            del frontier_blacklist[:-24]
            print(
                f"[wander] frontier at ({float(face_xy[0]):.2f}, {float(face_xy[1]):.2f}) "
                f"blacklisted after repeated failures ({reason}); exploring elsewhere first"
            )

    def _strike_frontier_if_reached(face_xy, stop_pose: Pose2D, reason: str) -> None:
        """Strike a frontier only when its own face was actually reached.

        A camera stop well before a distant frontier says that a *local*
        obstacle needs mapping and a path replan.  Counting it against the
        distant target blacklisted open exits after two shelf/table stops.
        """
        face_distance_m = math.hypot(
            float(face_xy[0]) - float(stop_pose.x),
            float(face_xy[1]) - float(stop_pose.y),
        )
        if face_distance_m > 0.50:
            # Local obstacle before a distant frontier. Normally: just map it and
            # replan around it (don't blacklist an open exit for one shelf stop).
            # BUT if the SAME frontier keeps getting furniture-blocked on the
            # approach, head-on re-approach is how the robot rammed a table and
            # fell over (2026-07-20). After a few repeats, blacklist this goal so
            # the planner routes to a different frontier instead of driving back
            # at the furniture.
            hz_key = (int(round(float(face_xy[0]) / 0.30)), int(round(float(face_xy[1]) / 0.30)))
            local_hazard_stop_counts[hz_key] = local_hazard_stop_counts.get(hz_key, 0) + 1
            hz_n = local_hazard_stop_counts[hz_key]
            print(
                "[wander] local hazard stop "
                f"{face_distance_m:.2f}m before frontier face (furniture on the approach, "
                f"block {hz_n}/3); mapping/replanning without striking the distant frontier"
            )
            if hz_n >= 3:
                print(
                    f"[wander] frontier at ({float(face_xy[0]):.2f}, {float(face_xy[1]):.2f}) "
                    "is repeatedly blocked by furniture on every approach — blacklisting it "
                    "and routing elsewhere instead of re-approaching (ram guard)"
                )
                _strike_frontier(
                    face_xy, "repeatedly furniture-blocked on approach", force=True
                )
                local_hazard_stop_counts.pop(hz_key, None)
            return
        _strike_frontier(face_xy, reason)

    def _plan_drives_into_blocked_front() -> bool:
        """True when the current plan's FIRST move is forward through the
        occupied stop box — the only case where front occupancy vetoes the
        plan. A waypoint off to the side means the transit starts with an
        in-place align turn, which the stop box does not forbid; the drive
        bursts re-check the box continuously once actually moving."""
        if plan_status not in ("ok", "survey", "observe"):
            return True  # no plan that could redeem the blockage
        wp_dx = float(frontier_plan["waypoint_xy"][0]) - float(current_live_pose.x)
        wp_dy = float(frontier_plan["waypoint_xy"][1]) - float(current_live_pose.y)
        if math.hypot(wp_dx, wp_dy) <= 0.05:
            return True
        wp_delta_deg = _normalize_angle_deg(
            math.degrees(math.atan2(wp_dy, wp_dx)) - float(current_live_pose.theta_deg)
        )
        return abs(wp_delta_deg) <= 40.0

    def _rotate_away_hint(
        start_pose: Pose2D, hazard_side: str, preferred_delta_deg: float | None = None
    ) -> MotionHint | None:
        """Wedged-pocket escape of last resort (field deadlock 2026-07-11):
        forward was held, reverse had no map clearance, and the hold could
        only be released by reversing — the wanderer looped forever. Rotate
        the nose well off the hazard instead: the tracked rotation releases
        reverse-only holds in the monitor, and forward motion afterwards
        moves AWAY from the hazard under the live gates."""
        if hazard_monitor is not None:
            turn_ok, turn_denial = hazard_monitor.turn_allowed()
            if not turn_ok:
                print(f"[safety] cannot rotate away: rotation frozen ({turn_denial})")
                return None
        nonlocal recovery_turn_sign
        if recovery_turn_sign is not None:
            rotate_sign = float(recovery_turn_sign)
        elif preferred_delta_deg is not None and abs(float(preferred_delta_deg)) >= 20.0:
            # Favor the live lidar's open direction over an arbitrary
            # side-based recovery turn.
            rotate_sign = 1.0 if float(preferred_delta_deg) >= 0.0 else -1.0
        else:
            rotate_sign = -1.0 if str(hazard_side) == "left" else 1.0
        recovery_turn_sign = float(rotate_sign)
        print(
            "[safety] rotating away from the hazard "
            f"(side={hazard_side}, target=85.0deg {'cw' if rotate_sign < 0 else 'ccw'})"
        )
        rotate_meta = _turn_with_arc_tracking(
            robot=robot,
            feed=feed,
            transformed_sets=stitch_state["transformed_sets"],
            start_pose=start_pose,
            lidar_offset_forward_m=float(args.lidar_offset_forward_m),
            resolution_m=float(args.stitch_resolution_m),
            target_turn_deg=85.0,
            direction_sign=float(rotate_sign),
            turn_speed=float(args.turn_speed),
            turn_burst_s=float(args.turn_burst_s),
            turn_settle_s=float(args.turn_settle_s),
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            invert_lateral_axis=bool(args.invert_lateral_axis),
            max_distance_m=float(args.max_distance_m),
            min_range_m=float(args.min_range_m),
            min_confidence=int(args.min_confidence),
            stop_tolerance_deg=min(8.0, float(args.stop_tolerance_deg)),
            max_bursts=int(args.max_turn_bursts),
            hazard_monitor=hazard_monitor,
        )
        nonlocal rotate_away_lock_losses
        rotated_deg = float(rotate_meta["turned_deg"])
        rotate_lock_lost = bool(rotate_meta.get("lock_lost"))
        if rotate_lock_lost:
            rotate_away_lock_losses += 1
        if abs(rotated_deg) < 5.0 and not rotate_lock_lost:
            print("[safety] rotate-away made no progress; holding")
            return None
        if rotate_lock_lost:
            # The wheels turned but tracking could not follow: the robot
            # rotated an UNKNOWN amount beyond the tracked degrees. Never
            # swallow that motion (field 2026-07-11: ~70deg untracked at
            # capture 1 wrecked the map) — hand the capture solve a wide
            # theta window so it can find the true heading.
            print(
                "[safety] rotate-away lost tracking "
                f"(tracked={rotated_deg:.1f}deg of an unknown physical rotation); "
                "widening the capture search window"
            )
        lever_dx_m, lever_dy_m = _turn_lever_arm_local_delta(
            rotated_deg, lidar_offset_forward_m=float(args.lidar_offset_forward_m)
        )
        return MotionHint(
            kind="turn",
            expected_dx_local_m=float(lever_dx_m),
            expected_dy_local_m=float(lever_dy_m),
            expected_dtheta_deg=rotated_deg,
            search_xy_m=0.60 if rotate_lock_lost else float(args.turn_search_xy_m),
            search_theta_window_deg=(
                85.0 if rotate_lock_lost else float(args.turn_theta_window_deg)
            ),
            label=f"hazard_rotate_away_{capture_index:02d}",
        )

    def _sidestep_waypoint_hint(
        start_pose: Pose2D,
        hazard_side: str,
        preferred_delta_deg: float | None = None,
    ) -> MotionHint | None:
        """Repair a locally bad waypoint with a guarded lateral offset.

        The base can strafe, but the established lidar stop box and eye gate
        guard forward motion.  Use them unchanged: turn 90 degrees away from
        the close side edge, advance a short amount, then turn back before
        replanning the same frontier from the offset pose.
        """
        if hazard_monitor is not None and not hazard_monitor.turn_allowed()[0]:
            return None
        # A left-side edge means translate right (clockwise turn first), and
        # vice versa. A front edge is the normal table-at-exit case: take the
        # side that points toward the current frontier, then restore heading.
        # This is the bounded 90deg -> forward -> -90deg bypass rather than
        # repeatedly turning in place at the same blocked waypoint.
        if str(hazard_side) == "left":
            turn_sign = -1.0
        elif str(hazard_side) == "right":
            turn_sign = 1.0
        elif preferred_delta_deg is not None and abs(float(preferred_delta_deg)) >= 10.0:
            turn_sign = 1.0 if float(preferred_delta_deg) >= 0.0 else -1.0
        else:
            return None
        print(
            "[wander] repairing blocked waypoint with guarded sidestep "
            f"(edge={hazard_side}, turn=90deg {'cw' if turn_sign < 0 else 'ccw'})"
        )

        first = _turn_with_arc_tracking(
            robot=robot, feed=feed, transformed_sets=stitch_state["transformed_sets"],
            start_pose=start_pose, lidar_offset_forward_m=float(args.lidar_offset_forward_m),
            resolution_m=float(args.stitch_resolution_m), target_turn_deg=90.0,
            direction_sign=turn_sign, turn_speed=float(args.turn_speed),
            turn_burst_s=float(args.turn_burst_s), turn_settle_s=float(args.turn_settle_s),
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            invert_lateral_axis=bool(args.invert_lateral_axis), max_distance_m=float(args.max_distance_m),
            min_range_m=float(args.min_range_m), min_confidence=int(args.min_confidence),
            stop_tolerance_deg=min(8.0, float(args.stop_tolerance_deg)),
            max_bursts=int(args.max_turn_bursts), hazard_monitor=hazard_monitor,
        )
        if bool(first.get("lock_lost")) or abs(float(first["turned_deg"])) < 70.0:
            print("[wander] sidestep cancelled: could not complete the first safe quarter-turn")
            return None
        d1x, d1y = _turn_lever_arm_local_delta(
            float(first["turned_deg"]), lidar_offset_forward_m=float(args.lidar_offset_forward_m)
        )
        t0 = math.radians(float(start_pose.theta_deg))
        side_pose = Pose2D(
            x=float(start_pose.x) + math.cos(t0) * d1x - math.sin(t0) * d1y,
            y=float(start_pose.y) + math.sin(t0) * d1x + math.cos(t0) * d1y,
            theta_deg=float(first["final_theta_deg"]),
        )
        side_pose, drive_meta = _drive_with_tracking(
            robot=robot, feed=feed, zone_cfg=zone_cfg,
            transformed_sets=stitch_state["transformed_sets"], start_pose=side_pose,
            resolution_m=float(args.stitch_resolution_m), forward_speed=float(args.move_speed),
            min_effective_move_speed=float(args.min_effective_move_speed), burst_s=0.25,
            burst_count=1, inter_burst_pause_s=0.0, steer_theta_vel=0.0,
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            invert_lateral_axis=bool(args.invert_lateral_axis), max_distance_m=float(args.max_distance_m),
            min_range_m=float(args.min_range_m), min_confidence=int(args.min_confidence),
            hazard_monitor=hazard_monitor,
            map_side_guard=_map_side_guard,
            pose_trusted=not pose_lost,
        )
        shifted_m = math.hypot(float(side_pose.x) - float(start_pose.x), float(side_pose.y) - float(start_pose.y))
        if (
            bool(drive_meta.get("stopped_by_hazard"))
            or bool(drive_meta.get("stopped_by_block"))
            or bool(drive_meta.get("lock_lost"))
            or shifted_m < 0.08
        ):
            print("[wander] sidestep stopped before a useful offset; keeping the turned safety pose")
            final_pose = side_pose
        else:
            second = _turn_with_arc_tracking(
                robot=robot, feed=feed, transformed_sets=stitch_state["transformed_sets"],
                start_pose=side_pose, lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                resolution_m=float(args.stitch_resolution_m), target_turn_deg=90.0,
                direction_sign=-turn_sign, turn_speed=float(args.turn_speed),
                turn_burst_s=float(args.turn_burst_s), turn_settle_s=float(args.turn_settle_s),
                forward_angle_deg=float(args.forward_angle_deg),
                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                invert_lateral_axis=bool(args.invert_lateral_axis), max_distance_m=float(args.max_distance_m),
                min_range_m=float(args.min_range_m), min_confidence=int(args.min_confidence),
                stop_tolerance_deg=min(8.0, float(args.stop_tolerance_deg)),
                max_bursts=int(args.max_turn_bursts), hazard_monitor=hazard_monitor,
            )
            d2x, d2y = _turn_lever_arm_local_delta(
                float(second["turned_deg"]), lidar_offset_forward_m=float(args.lidar_offset_forward_m)
            )
            t1 = math.radians(float(side_pose.theta_deg))
            final_pose = Pose2D(
                x=float(side_pose.x) + math.cos(t1) * d2x - math.sin(t1) * d2y,
                y=float(side_pose.y) + math.sin(t1) * d2x + math.cos(t1) * d2y,
                theta_deg=float(second["final_theta_deg"]),
            )
        t_start = math.radians(float(start_pose.theta_deg))
        dx = float(final_pose.x) - float(start_pose.x)
        dy = float(final_pose.y) - float(start_pose.y)
        print(f"[wander] sidestep offset={math.hypot(dx, dy):.2f}m; replanning the same frontier")
        return MotionHint(
            kind="mixed",
            expected_dx_local_m=math.cos(t_start) * dx + math.sin(t_start) * dy,
            expected_dy_local_m=-math.sin(t_start) * dx + math.cos(t_start) * dy,
            expected_dtheta_deg=_normalize_angle_deg(float(final_pose.theta_deg) - float(start_pose.theta_deg)),
            search_xy_m=0.55,
            search_theta_window_deg=30.0,
            label=f"waypoint_sidestep_{capture_index:02d}",
        )

    def _live_frontier_probe_hint(start_pose: Pose2D, choice: FrontierChoice) -> MotionHint:
        """Cautiously extend a map that ends before a live lidar opening.

        The stitched grid can have no reachable unknown cells immediately
        after a doorway transit, while the current lidar still sees open
        space.  One normal safety-gated forward burst creates the next map
        evidence; completion is never the right action in that state.
        """
        # Scale the probe to the measured opening (leaving a 0.8m stop
        # margin) — a single 0.15m nudge costs a full 5-12s stitch cycle
        # and barely adds map evidence; every burst is still safety-gated.
        probe_bursts = max(1, min(3, int((float(choice.mean_distance_m) - 0.8) / 0.25)))
        print(
            "[wander] map frontier exhausted but live lidar is open; probing forward "
            f"(delta={choice.delta_deg:.1f}deg, range={choice.mean_distance_m:.2f}m, "
            f"bursts={probe_bursts})"
        )
        end_pose, probe_meta = _drive_with_tracking(
            robot=robot, feed=feed, zone_cfg=zone_cfg,
            transformed_sets=stitch_state["transformed_sets"], start_pose=start_pose,
            resolution_m=float(args.stitch_resolution_m), forward_speed=float(args.move_speed),
            min_effective_move_speed=float(args.min_effective_move_speed),
            burst_s=min(0.80, float(args.move_burst_s)), burst_count=probe_bursts,
            inter_burst_pause_s=0.0, steer_theta_vel=0.0,
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            invert_lateral_axis=bool(args.invert_lateral_axis), max_distance_m=float(args.max_distance_m),
            min_range_m=float(args.min_range_m), min_confidence=int(args.min_confidence),
            hazard_monitor=hazard_monitor,
            map_side_guard=_map_side_guard,
            forward_guard=_exit_ratchet_guard,
            blind_forward_scale=float(motion_calibration.translation_scale),
            pose_trusted=not pose_lost,
        )
        t0 = math.radians(float(start_pose.theta_deg))
        dx = float(end_pose.x) - float(start_pose.x)
        dy = float(end_pose.y) - float(start_pose.y)
        # Forward motion the lidar TRACKED, plus a CALIBRATED estimate of forward
        # motion it could not track (blind bursts) — so a blind creep through the
        # doorway still counts toward the exit-run crossing odometry AND seeds the
        # next relocalization at the right place instead of "didn't move".
        measured_forward_local = math.cos(t0) * dx + math.sin(t0) * dy
        blind_forward_local = float(probe_meta.get("blind_forward_m", 0.0) or 0.0)
        if blind_forward_local > 1e-3:
            print(
                f"[wander] probe: lidar tracked {measured_forward_local:+.2f}m forward + "
                f"~{blind_forward_local:.2f}m estimated through blind bursts (calibrated dead-reckon)"
            )
        if bool(probe_meta.get("stopped_by_hazard")):
            print("[wander] live-frontier probe stopped by a safety gate; replanning from the tracked stop pose")
        return MotionHint(
            kind="drive",
            expected_dx_local_m=measured_forward_local + blind_forward_local,
            expected_dy_local_m=-math.sin(t0) * dx + math.cos(t0) * dy,
            expected_dtheta_deg=_normalize_angle_deg(float(end_pose.theta_deg) - float(start_pose.theta_deg)),
            search_xy_m=0.45,
            search_theta_window_deg=18.0,
            label=f"live_frontier_probe_{capture_index:02d}",
        )

    def _exit_doorway_approach_hint(start_pose: Pose2D, choice: FrontierChoice) -> "MotionHint | None":
        """DOORWAY ENTRY MANEUVER for the exit run (field 2026-07-20, user
        spec): the blind bearing-probe aims at the opening's CURRENT bearing, so
        when the robot sits off to one side of the hallway mouth it drives in
        DIAGONALLY, clips a shoulder / stalls, and never enters. Instead: read
        the two doorframe jambs from the live scan, and drive a CENTRED, SQUARE
        approach —
          (1) turn/drive to a standoff point on the hall centre-line, just
              outside the mouth  (centres the robot on the opening),
          (2) square up to the hall axis (perpendicular to the mouth),
          (3) probe forward THROUGH the middle of the mouth.
        Returns a MotionHint summarising the net motion (to seed the next
        relocalization), or None to fall back to the blind probe when the gap is
        not doorframe-like (no clean jambs / implausible width).
        Phase-1 centring is exempt from the outward ratchet (a legitimate
        reposition, not a retreat); the hazard gate + stop box still own safety.
        """
        _, live_frame_now = feed.latest()
        if live_frame_now is None:
            return None
        geom = _doorway_mouth_and_axis(
            live_frame_now,
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            max_distance_m=float(args.max_distance_m),
            min_range_m=float(args.min_range_m),
            min_confidence=int(args.min_confidence),
            opening_delta_deg=float(choice.delta_deg),
            opening_width_deg=float(choice.width_deg),
            opening_depth_m=float(choice.mean_distance_m),
        )
        if geom is None:
            return None
        gap_w = float(geom["gap_width_m"])
        if gap_w < 0.45 or gap_w > 2.5:
            # Not a doorway-like gap (too tight to fit, or a wide-open wall the
            # jamb finder mismeasured) — the blind probe is the safer default.
            return None

        axis_world_deg = _normalize_angle_deg(
            float(start_pose.theta_deg) + float(geom["axis_delta_deg"])
        )
        mouth_bear = math.radians(
            float(start_pose.theta_deg) + float(geom["mouth_delta_deg"])
        )
        mouth_wx = float(start_pose.x) + float(geom["mouth_range_m"]) * math.cos(mouth_bear)
        mouth_wy = float(start_pose.y) + float(geom["mouth_range_m"]) * math.sin(mouth_bear)
        axis_r = math.radians(axis_world_deg)
        axis_ux, axis_uy = math.cos(axis_r), math.sin(axis_r)
        STANDOFF_M = 0.45
        standoff_x = mouth_wx - axis_ux * STANDOFF_M
        standoff_y = mouth_wy - axis_uy * STANDOFF_M

        print(
            "[wander] EXIT DOORWAY: mouth centre "
            f"({mouth_wx:.2f}, {mouth_wy:.2f}), hall axis {axis_world_deg:.0f}deg, "
            f"gap {gap_w:.2f}m — centring and squaring up before entering"
        )

        pose = start_pose

        def _turn_to(world_heading_deg: float, reason: str) -> None:
            nonlocal pose
            delta = _normalize_angle_deg(world_heading_deg - float(pose.theta_deg))
            if abs(delta) < 8.0:
                return
            meta = _turn_with_arc_tracking(
                robot=robot, feed=feed,
                transformed_sets=stitch_state["transformed_sets"],
                start_pose=pose,
                lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                resolution_m=float(args.stitch_resolution_m),
                target_turn_deg=abs(delta),
                direction_sign=1.0 if delta >= 0.0 else -1.0,
                turn_speed=float(args.turn_speed), turn_burst_s=float(args.turn_burst_s),
                turn_settle_s=float(args.turn_settle_s),
                forward_angle_deg=float(args.forward_angle_deg),
                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                invert_lateral_axis=bool(args.invert_lateral_axis),
                max_distance_m=float(args.max_distance_m), min_range_m=float(args.min_range_m),
                min_confidence=int(args.min_confidence),
                stop_tolerance_deg=min(8.0, float(args.stop_tolerance_deg)),
                max_bursts=int(args.max_turn_bursts), hazard_monitor=hazard_monitor,
            )
            turned = float(meta["turned_deg"])
            pose = dataclasses.replace(pose, theta_deg=_normalize_angle_deg(float(pose.theta_deg) + turned))
            print(f"[wander] EXIT DOORWAY: {reason} (turned {turned:+.0f}deg)")

        def _drive_forward(distance_m: float, use_ratchet: bool, reason: str) -> None:
            nonlocal pose
            bursts = max(1, min(4, int(math.ceil(float(distance_m) / 0.25))))
            end_pose, _meta = _drive_with_tracking(
                robot=robot, feed=feed, zone_cfg=zone_cfg,
                transformed_sets=stitch_state["transformed_sets"], start_pose=pose,
                resolution_m=float(args.stitch_resolution_m), forward_speed=float(args.move_speed),
                min_effective_move_speed=float(args.min_effective_move_speed),
                burst_s=min(0.80, float(args.move_burst_s)), burst_count=bursts,
                inter_burst_pause_s=0.0, steer_theta_vel=0.0,
                forward_angle_deg=float(args.forward_angle_deg),
                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                invert_lateral_axis=bool(args.invert_lateral_axis), max_distance_m=float(args.max_distance_m),
                min_range_m=float(args.min_range_m), min_confidence=int(args.min_confidence),
                hazard_monitor=hazard_monitor, map_side_guard=_map_side_guard,
                forward_guard=_exit_ratchet_guard if use_ratchet else None,
                blind_forward_scale=float(motion_calibration.translation_scale),
                pose_trusted=not pose_lost,
            )
            pose = end_pose
            print(f"[wander] EXIT DOORWAY: {reason}")

        # PHASE 1 — get onto the hall centre-line, just outside the mouth.
        dist_standoff = math.hypot(standoff_x - float(pose.x), standoff_y - float(pose.y))
        if dist_standoff > 0.15:
            bear_standoff = math.degrees(
                math.atan2(standoff_y - float(pose.y), standoff_x - float(pose.x))
            )
            _turn_to(bear_standoff, "facing the centred standoff")
            _drive_forward(min(1.2, dist_standoff), use_ratchet=False, reason="drove to the centred standoff")
        # PHASE 2 — square up to the hall axis (face straight down the middle).
        _turn_to(axis_world_deg, "squared up to the hall axis")
        # PHASE 3 — probe forward through the middle of the mouth.
        through_m = max(0.30, float(geom["mouth_range_m"]) - STANDOFF_M + 0.35)
        _drive_forward(through_m, use_ratchet=True, reason="probing through the mouth")

        t0 = math.radians(float(start_pose.theta_deg))
        dx = float(pose.x) - float(start_pose.x)
        dy = float(pose.y) - float(start_pose.y)
        return MotionHint(
            kind="drive",
            expected_dx_local_m=math.cos(t0) * dx + math.sin(t0) * dy,
            expected_dy_local_m=-math.sin(t0) * dx + math.cos(t0) * dy,
            expected_dtheta_deg=_normalize_angle_deg(float(pose.theta_deg) - float(start_pose.theta_deg)),
            search_xy_m=0.55,
            search_theta_window_deg=25.0,
            label=f"exit_doorway_{capture_index:02d}",
        )

    def _elevated_hazard_engaged() -> bool:
        """True when an elevated-edge stop is in effect — including a
        post-stop HOLD left behind by one. Field run 2026-07-09: the hold
        (reverse-only, vanished_near) denied every drive burst while the
        state read neither active nor blind, so the investigation trigger
        never fired and the wanderer looped plan->denied->capture forever."""
        if hazard_monitor is None:
            return False
        hs = hazard_monitor.state()
        if hs.active or hs.blind_zone:
            return True
        return bool(
            hs.hold
            and str(hs.hold_reason)
            in ("vanished_near", "line_edge", "elevated_parallax", "depth_elevated", "hazard")
        )

    def _jammed_on_hazard() -> bool:
        """The eyes say the robot is RIGHT ON an obstacle — it must NOT turn or
        creep, only REVERSE. Field 2026-07-20 (user): "if it's basically on top
        of an edge ... it should not go forward OR turn ... back up if it can't
        see ANY ground." Turning while jammed catches an arm/caster on the
        furniture and tips the robot (it fell over). True when:
          - an elevated hazard is in the NEAR-FREEZE tier (<0.35m ahead) and it
            is NOT a passable squeeze-through, OR
          - the bottom-camera GROUND gate sees no safe floor, OR
          - an elevated hazard is active with essentially no clear floor ahead
            (clear_width below the body width) — i.e. no ground to move onto.
        The signal is what the eyes report THIS instant; the reflex reverses to
        regain clearance and refuses to rotate until it can see room again."""
        if hazard_monitor is None:
            return False
        hs = hazard_monitor.state()
        if hs.frames_stale:
            # A stale read can't certify "jammed" — but it also can't clear it.
            # Leave that to the stale-stop gate; do not force a reverse on stale.
            return False
        if bool(hs.ground_active):
            return True
        if bool(hs.near_freeze) and not bool(hs.squeeze):
            return True
        if bool(hs.active) and not bool(hs.squeeze):
            cw = hs.clear_width_m
            if cw is None or float(cw) < 2.0 * float(args.robot_radius_m):
                return True
        return False

    def _stamp_world_segment(w1: tuple[float, float], w2: tuple[float, float], height_m: float) -> int:
        """Rasterize one world-frame segment into the planning-map layer and
        the deep-red rerun overlay. Returns how many NEW cells it added."""
        if pose_lost:
            # World coordinates computed from a lost pose are fiction; this
            # run's fallback-pose stamps helped seal the planning grid shut.
            return 0
        elevated_segments_world.append({"p1": w1, "p2": w2, "height_m": float(height_m)})
        seg_len = math.hypot(w2[0] - w1[0], w2[1] - w1[1])
        steps = max(int(seg_len / 0.04), 1)
        segment_new_cells = 0
        for step in range(steps + 1):
            t = step / steps
            cell = (
                int(round((w1[0] + t * (w2[0] - w1[0])) / 0.04)),
                int(round((w1[1] + t * (w2[1] - w1[1])) / 0.04)),
            )
            if cell not in elevated_occupied_world and len(elevated_occupied_world) < 20000:
                elevated_occupied_world.add(cell)
                segment_new_cells += 1
            # Investigation-measured geometry is approach-verified: permanent
            # (no decay metadata).
            elevated_cell_meta.pop(cell, None)
        if segment_new_cells:
            # static=True: the edge layer is session-cumulative and must be
            # visible at EVERY timeline position — logged temporally, the red
            # lines disappear whenever the viewer's capture_index playhead
            # sits before the tick the edge was stamped at.
            rr.log(
                "world/elevated_edges",
                rr.LineStrips3D(
                    [
                        [
                            [seg["p1"][0], seg["p1"][1], seg["height_m"]],
                            [seg["p2"][0], seg["p2"][1], seg["height_m"]],
                        ]
                        for seg in elevated_segments_world
                    ],
                    colors=[[200, 0, 0]] * len(elevated_segments_world),
                    radii=0.015,
                ),
                static=True,
            )
        return segment_new_cells

    stop_debug_dir = output_dir / "stop_debug"
    stop_debug_dir.mkdir(parents=True, exist_ok=True)
    stop_debug_counter = {"n": 0}
    # Rate limiter for planning-denial bundles (same reason at most every 20s).
    planning_denial_last: dict[str, object] = {"reason": "", "mono": -1e9}

    # CONTINUOUS EYE RECORDER (field 2026-07-20, user idea): save the annotated
    # eye frames — with the depth model's elevated-hazard overlay — throughout the
    # run, so after a run the exact frames the robot saw approaching a table edge
    # can be reviewed (the ram leaves NO stop bundle because the depth never
    # fires). Fresh dir each run; frames numbered so the LAST ones are the moment
    # of interest; a rolling cap keeps disk bounded.
    eye_record_dir = output_dir / "eye_record"
    eye_record_counter = {"n": 0}
    last_eye_record_monotonic = {"t": 0.0}

    def _record_eye_frames() -> None:
        if hazard_monitor is None or float(args.record_eye_frames_hz) <= 0.0:
            return
        now_rec = time.monotonic()
        if now_rec - last_eye_record_monotonic["t"] < 1.0 / float(args.record_eye_frames_hz):
            return
        last_eye_record_monotonic["t"] = now_rec
        import cv2 as _cv2

        if eye_record_counter["n"] == 0:
            eye_record_dir.mkdir(parents=True, exist_ok=True)
        eye_record_counter["n"] += 1
        idx = eye_record_counter["n"]
        hs = hazard_monitor.state()
        # Stamp the classification/decision onto the panorama filename so the
        # frame's meaning is scannable without opening a sidecar.
        tag = f"{hs.decision_label()}_{hs.classification}".replace("/", "-").replace(" ", "")
        for cam in ("panorama", "front_left", "front_right"):
            annotated = hazard_monitor.annotated(cam)
            if annotated is not None:
                _cv2.imwrite(
                    str(eye_record_dir / f"f{idx:05d}_{cam}_{tag}.png"), annotated
                )
        # Rolling cap: keep the most recent ~900 frame-sets (~7.5 min at 2Hz).
        cap = 900 * 3
        if idx % 60 == 0:
            files = sorted(eye_record_dir.glob("f*.png"))
            if len(files) > cap:
                for stale in files[: len(files) - cap]:
                    try:
                        stale.unlink()
                    except OSError:
                        pass

    def _dump_stop_debug(stop_pose: Pose2D, note: str) -> None:
        """Field-diagnosis bundle per hazard stop: what the robot SAW (eye
        overlays) and BELIEVED (pose, gate state) at the moment of the stop,
        so misplaced map geometry can be traced to mask vs projection vs
        pose instead of inferred from logs."""
        if hazard_monitor is None:
            return
        import cv2 as _cv2

        stop_debug_counter["n"] += 1
        bundle_index = stop_debug_counter["n"]
        hs = hazard_monitor.state()
        for cam in ("front_left", "front_right", "panorama", "bottom"):
            annotated = hazard_monitor.annotated(cam)
            if annotated is not None:
                _cv2.imwrite(
                    str(stop_debug_dir / f"stop{bundle_index:03d}_{cam}.png"), annotated
                )
        meta = {
            "note": str(note),
            "pose_xy_theta": [
                round(float(stop_pose.x), 4),
                round(float(stop_pose.y), 4),
                round(float(stop_pose.theta_deg), 2),
            ],
            "reason": hs.reason,
            "side": hs.side,
            "est_distance_m": hs.est_distance_m,
            "classification": hs.classification,
            "detail": hs.detail,
            # Ground-gate + hold state: planning-time denials are mostly
            # ground_stop_* / post_stop_hold and were undiagnosable without
            # these (field 2026-07-13 night: ~8 ground denials, zero bundles).
            "ground_active": bool(getattr(hs, "ground_active", False)),
            "ground_distance_m": getattr(hs, "ground_distance_m", None),
            "ground_side": getattr(hs, "ground_side", "none"),
            "hold": bool(getattr(hs, "hold", False)),
            "hold_reason": getattr(hs, "hold_reason", ""),
            "map_cells_total": len(elevated_occupied_world),
        }
        (stop_debug_dir / f"stop{bundle_index:03d}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"[debug] stop bundle #{bundle_index} saved -> {stop_debug_dir}")

    def _plan_edge_survey(grid, robot_pose: Pose2D) -> dict | None:
        """EDGE-SURVEY phase: pick the nearest POTENTIAL (yellow) cluster
        with attempts remaining and plan a transit to a vantage whose
        observation bearing differs >=30deg from the cluster's first
        sighting. Bleed lies along the original viewing ray, so the second
        angle either re-detects the cells at the same world position
        (promote to red — real edge, dimensions solid) or cleanly misses
        (decay — phantom, corridor opens). Runs when normal frontier
        exploration is dry, BEFORE the squeeze/verification ladder: solidify
        the edges first, then push through what remains."""
        if grid is None or not elevated_cell_meta:
            return None
        if exit_mode:
            # EXIT RUN owns the robot: survey vantages are INTERIOR points, and
            # driving back to them is exactly the going-back the exit run
            # forbids (field 2026-07-18: "attempt 1/3" edge-survey plans
            # interleaved with exit probes and dragged the robot back into the
            # room every other cycle — mission stalled at the door). Yellow
            # clusters keep decaying/promoting passively from whatever the eyes
            # see during the traversal; deliberate second-angle trips resume
            # after the robot is through.
            return None
        unconfirmed_cells = [
            c
            for c, m in elevated_cell_meta.items()
            if len(m) >= 3 and not m[2] and c in elevated_occupied_world
        ]
        if not unconfirmed_cells:
            return None
        rx, ry = float(robot_pose.x), float(robot_pose.y)
        unconfirmed_cells.sort(
            key=lambda c: (c[0] * 0.04 - rx) ** 2 + (c[1] * 0.04 - ry) ** 2
        )
        now_mono = time.monotonic()
        tried_buckets: set[tuple[int, int]] = set()
        for seed_cell in unconfirmed_cells[:24]:
            seed_x, seed_y = seed_cell[0] * 0.04, seed_cell[1] * 0.04
            cluster = [
                c
                for c in unconfirmed_cells
                if abs(c[0] * 0.04 - seed_x) <= 0.5 and abs(c[1] * 0.04 - seed_y) <= 0.5
            ]
            centroid_x = sum(c[0] for c in cluster) / len(cluster) * 0.04
            centroid_y = sum(c[1] for c in cluster) / len(cluster) * 0.04
            bucket = (int(round(centroid_x / 0.3)), int(round(centroid_y / 0.3)))
            if bucket in tried_buckets:
                continue
            tried_buckets.add(bucket)
            if edge_survey_attempts.get(bucket, 0) >= 3:
                continue
            first_bearings = [float(elevated_cell_meta[c][0]) for c in cluster]
            base_bearing = math.degrees(
                math.atan2(
                    sum(math.sin(math.radians(b)) for b in first_bearings),
                    sum(math.cos(math.radians(b)) for b in first_bearings),
                )
            )
            # Candidate vantages: side-offset bearings (kept <=95deg — past
            # ~100deg the object may self-occlude its own near boundary).
            for bearing_offset in (55.0, -55.0, 75.0, -75.0, 40.0, -40.0, 90.0, -90.0):
                observe_bearing = base_bearing + bearing_offset
                for standoff_m in (1.0, 1.3, 0.8):
                    vantage_x = centroid_x - standoff_m * math.cos(
                        math.radians(observe_bearing)
                    )
                    vantage_y = centroid_y - standoff_m * math.sin(
                        math.radians(observe_bearing)
                    )
                    candidate = _plan_frontier_path(
                        grid=grid,
                        robot_xy=(rx, ry),
                        robot_radius_m=float(args.robot_radius_m),
                        target_xy=(vantage_x, vantage_y),
                        robot_theta_deg=float(robot_pose.theta_deg),
                    )
                    if str(candidate.get("status")) != "ok":
                        continue
                    goal_x, goal_y = candidate["goal_xy"]
                    if math.hypot(goal_x - vantage_x, goal_y - vantage_y) > 0.35:
                        continue
                    achieved_bearing = math.degrees(
                        math.atan2(centroid_y - goal_y, centroid_x - goal_x)
                    )
                    if abs(_normalize_angle_deg(achieved_bearing - base_bearing)) < 30.0:
                        continue
                    if (
                        math.hypot(goal_x - rx, goal_y - ry) < 0.35
                        and abs(
                            _normalize_angle_deg(
                                math.degrees(
                                    math.atan2(centroid_y - ry, centroid_x - rx)
                                )
                                - float(robot_pose.theta_deg)
                            )
                        )
                        < 35.0
                    ):
                        # Already standing at this vantage FACING the
                        # cluster: the look is happening — re-adopting the
                        # same spot just spins in place (field run 21:
                        # "attempt 1/3" re-adopted forever). Try the next
                        # bearing offset for a genuinely different angle.
                        continue
                    # Attempt accounting: the adoption clock only advances
                    # when an attempt is charged — every-adoption timestamp
                    # updates made the 30s debounce unreachable and the
                    # counter stuck at 1 forever (run 21).
                    if now_mono - edge_survey_last_adopt.get(bucket, -1e9) > 30.0:
                        edge_survey_attempts[bucket] = (
                            edge_survey_attempts.get(bucket, 0) + 1
                        )
                        edge_survey_last_adopt[bucket] = now_mono
                    candidate["face_xy"] = (centroid_x, centroid_y)
                    print(
                        "[wander] EDGE SURVEY: observing yellow cluster at "
                        f"({centroid_x:.2f}, {centroid_y:.2f}) from a new angle "
                        f"(first sighting {base_bearing:.0f}deg -> new "
                        f"{achieved_bearing:.0f}deg, vantage=({goal_x:.2f}, {goal_y:.2f}), "
                        f"attempt {edge_survey_attempts.get(bucket, 0)}/3, "
                        f"path={float(candidate.get('path_length_m', 0.0) or 0.0):.2f}m)"
                    )
                    return candidate
        return None

    def _nearest_mapped_elevated_m(pose: Pose2D) -> float | None:
        """Distance from the body center to the nearest MAPPED elevated cell
        within 1.0m, or None. The map is the only sensor that covers the
        robot's SIDES at tabletop height (panorama ±52° forward, lidar under
        tabletops, bottom camera on the floor) — field 2026-07-17: the robot
        ground its shoulder along a mapped table edge while turning beside
        it ("no rotation progress")."""
        if not elevated_occupied_world:
            return None
        px, py = float(pose.x), float(pose.y)
        best: float | None = None
        for cell_x, cell_y in elevated_occupied_world:
            dist = math.hypot(cell_x * 0.04 - px, cell_y * 0.04 - py)
            if dist <= 1.0 and (best is None or dist < best):
                best = dist
        return best

    map_side_guard_state = {"pose": None, "streak": 0}

    def _exit_ratchet_guard(pose: Pose2D) -> str | None:
        """INDEPENDENT exit-run hard-stop (the user's demanded belt-and-braces
        against "got to the door then turned and left"). Once the exit run has
        committed — latched AND the robot has actually seen new area through the
        opening (``exit_opening_seen``) — the robot's distance from the room
        center may never fall more than ``EXIT_RATCHET_SLACK_M`` below the
        farthest-out it has already reached. This is a pure geometric ratchet on
        the LIVE tracked pose, wholly separate from the planner and its vetoes,
        so a retreat is physically refused no matter what upstream logic ordered
        the drive. Rotation in place is untouched (it does not move the robot),
        so the robot can still spin to find another way out; only DRIVING back
        toward the center is forbidden. If every outward path is blocked it holds
        at the door — the user's explicit preference over ever going back in."""
        if not (exit_mode and exit_opening_seen) or exit_centroid_xy is None:
            return None
        dist_center = math.hypot(
            float(pose.x) - exit_centroid_xy[0], float(pose.y) - exit_centroid_xy[1]
        )
        if dist_center >= exit_outward_max_m - EXIT_RATCHET_SLACK_M:
            return None
        return (
            f"EXIT RATCHET: {dist_center:.2f}m from room center has fallen below the "
            f"committed outward-max {exit_outward_max_m:.2f}m - {EXIT_RATCHET_SLACK_M:.2f}m — "
            "refusing to drive back toward the room; holding at the door"
        )

    def _map_side_guard(pose: Pose2D) -> str | None:
        """Drive-burst veto for the virtual side bumper: halt FORWARD motion
        only for mapped cells in the forward collision corridor — ahead of
        the body center and within body width. Cells beside/behind cannot
        be struck by driving forward; the original direction-blind radius
        check deadlocked the planner for 130+ zero-motion cycles (field
        2026-07-17 run 18: a cell 0.26m from center, beside/behind the
        robot after a reverse, vetoed every forward burst forever).
        Threshold radius + 2cm laterally so legitimate squeeze passes
        (planned at radius - 5cm inflation) are not blocked."""
        if not elevated_occupied_world:
            return None
        guard_r = float(args.robot_radius_m) + 0.02
        fwd_margin = 0.10
        # During squeeze/verification/exit traversals the plan DELIBERATELY
        # approaches eye-mapped cells ("approaching under full gates") — this
        # guard must not zero-halt the very approach the planner ordered
        # (field 2026-07-18: six straight bursts=0 halts at ~0.45m, pinned →
        # back away → replan the identical approach, an infinite loop at the
        # exit). For those passes the guard tightens to the squeeze envelope:
        # a cell truly in the body's path still halts, flanking furniture
        # does not, and the LIVE gates (eye stops, measured-gap creep, stop
        # box) own the traversal — they fired correctly all run.
        if exit_mode or squeeze_traversal_active:
            guard_r = max(0.25, float(args.robot_radius_m) - 0.02)
            fwd_margin = 0.04
        px, py = float(pose.x), float(pose.y)
        theta_rad = math.radians(float(pose.theta_deg))
        cos_g, sin_g = math.cos(theta_rad), math.sin(theta_rad)
        for cell_x, cell_y in elevated_occupied_world:
            cell_meta = elevated_cell_meta.get((cell_x, cell_y))
            if cell_meta is None or len(cell_meta) < 3 or not cell_meta[2]:
                # Unconfirmed yellow claims never zero-halt drives (same
                # contract as the corridor veto): if the furniture is real,
                # the live eye gate stops the robot and confirmation follows.
                continue
            dxc = cell_x * 0.04 - px
            dyc = cell_y * 0.04 - py
            if dxc * dxc + dyc * dyc > 1.0:
                continue
            fwd = cos_g * dxc + sin_g * dyc
            lat = -sin_g * dxc + cos_g * dyc
            if -0.02 <= fwd <= guard_r + fwd_margin and abs(lat) <= guard_r:
                prev_pose = map_side_guard_state["pose"]
                if (
                    prev_pose is not None
                    and math.hypot(px - prev_pose[0], py - prev_pose[1]) < 0.05
                ):
                    map_side_guard_state["streak"] += 1
                else:
                    map_side_guard_state["streak"] = 1
                map_side_guard_state["pose"] = (px, py)
                return (
                    f"mapped elevated cell {math.hypot(dxc, dyc):.2f}m ahead in the "
                    "body corridor (side-collision guard)"
                )
        return None

    # Synthetic blockers REMOVED (user directive 2026-07-17, final): a stop
    # with no measurable geometry stamps NOTHING. Field run: a fabricated
    # 0.6m "blocker" vastly overestimated a desk's length and walled off
    # the room entrance. Stops still teach through frontier strikes and
    # blacklists (non-fabricating); only MEASURED edges reach the map.

    def _stamp_confirmed_edges(stamp_pose: Pose2D, newest_only: bool = False) -> int:
        """Transform freshly measured (robot-frame) edge segments into world
        coordinates at the given tracked pose and stamp them into the
        planning map. Only fresh measurements are used — the pose must match
        the measurement moment. newest_only: the robot was MOVING until the
        stop, so only the final confirmation tick (the one that tripped the
        gate, within ~0.3s of it) matches the stop pose; earlier ones were
        measured metres back and would stamp the edge beyond its true spot."""
        if hazard_monitor is None:
            return 0
        if pose_lost:
            # A lost pose puts every stamped cell somewhere fictional. Drain
            # and DROP the pending measurements (they are stamped against
            # the wrong pose forever) — the map stays clean and the eyes
            # keep measuring fresh ones after recovery.
            hazard_monitor.drain_confirmed_edges()
            if hasattr(hazard_monitor, "drain_floor_evidence"):
                hazard_monitor.drain_floor_evidence()
            if hasattr(hazard_monitor, "drain_view_reports"):
                hazard_monitor.drain_view_reports()
            return 0
        drained_edges = hazard_monitor.drain_confirmed_edges()
        if newest_only and drained_edges:
            newest_mono = max(float(edge.monotonic) for edge in drained_edges)
            drained_edges = [
                edge for edge in drained_edges if newest_mono - float(edge.monotonic) <= 0.3
            ]
        now_mono = time.monotonic()
        theta_rad = math.radians(float(stamp_pose.theta_deg))
        cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
        new_cells = 0
        new_points = 0
        segment_added = False
        # FREE-SPACE pass first: depth-observed floor CLEARS any elevated
        # cell it lands on. The map is otherwise add-only, so one overshot
        # footprint would wall off a doorway for the rest of the run even
        # while the eyes stare straight down its traversable floor (field
        # 2026-07-12). Clear-then-stamp: this drain's own footprints re-add
        # anything genuinely occupied. 3x3 neighborhood: floor evidence is
        # 6cm-decimated while cells are 4cm — exact-cell clearing would
        # leave stripes.
        cleared_cells = 0
        decayed_cells_total = 0
        promoted_cells = 0
        floor_batches = (
            hazard_monitor.drain_floor_evidence()
            if hasattr(hazard_monitor, "drain_floor_evidence")
            else []
        )
        if newest_only and floor_batches:
            newest_floor = max(float(ev.monotonic) for ev in floor_batches)
            floor_batches = [
                ev for ev in floor_batches if newest_floor - float(ev.monotonic) <= 0.3
            ]
        newest_floor_mono: dict[str, float] = {}
        for ev in floor_batches:
            if float(ev.monotonic) > newest_floor_mono.get(str(ev.eye), -1.0):
                newest_floor_mono[str(ev.eye)] = float(ev.monotonic)
        for ev in floor_batches:
            if now_mono - float(ev.monotonic) > 1.0:
                continue
            if float(ev.monotonic) < newest_floor_mono.get(str(ev.eye), -1.0):
                continue
            for pf, pl in ev.points_robot_xy:
                wx = float(stamp_pose.x) + float(pf) * cos_t - float(pl) * sin_t
                wy = float(stamp_pose.y) + float(pf) * sin_t + float(pl) * cos_t
                cx, cy = int(round(wx / 0.04)), int(round(wy / 0.04))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        cell = (cx + dx, cy + dy)
                        elevated_pending_world.pop(cell, None)
                        if cell in elevated_occupied_world:
                            elevated_occupied_world.discard(cell)
                            elevated_points_world.pop(cell, None)
                            elevated_cell_meta.pop(cell, None)
                            cleared_cells += 1
        # Footprints: the NEWEST frame per eye is the stamping frame; the
        # frame before it (>=0.4s older, within a 3.0s window — the robot is
        # settled/travel-compensated over that span) is CONFIRMATION-ONLY.
        # Far cells promote on the INTERSECTION of the two frames: an AND
        # across frames is noise-resistant where stamping the union of the
        # whole queue smeared an 800-point blob (field 2026-07-12). A
        # newest-only + cross-checkpoint persistence design mapped ZERO
        # cells in the field (2026-07-13 evening run): checkpoints are taken
        # at different poses/headings, so far cells never re-observed and
        # pended forever.
        footprint_monos: dict[str, list[float]] = {}
        for edge in drained_edges:
            if getattr(edge, "points_robot_xy", ()) or ():
                eye_key = str(edge.eye)
                mono = float(edge.monotonic)
                if mono not in footprint_monos.setdefault(eye_key, []):
                    footprint_monos[eye_key].append(mono)
        stamp_frame_mono: dict[str, float] = {}
        confirm_frame_mono: dict[str, float] = {}
        for eye_key, monos in footprint_monos.items():
            monos.sort(reverse=True)
            fresh = [m for m in monos if now_mono - m <= 3.0]
            if not fresh:
                continue
            stamp_frame_mono[eye_key] = fresh[0]
            for older in fresh[1:]:
                if fresh[0] - older >= 0.4:
                    confirm_frame_mono[eye_key] = older
                    break
        for edge in drained_edges:
            footprint = getattr(edge, "points_robot_xy", ()) or ()
            if footprint:
                eye_key = str(edge.eye)
                obs_mono = float(edge.monotonic)
                if obs_mono == confirm_frame_mono.get(eye_key):
                    # Confirmation-only frame: its points arm pending cells;
                    # they never stamp directly (weaker pose match).
                    for pf, pl in footprint:
                        wx = float(stamp_pose.x) + float(pf) * cos_t - float(pl) * sin_t
                        wy = float(stamp_pose.y) + float(pf) * sin_t + float(pl) * cos_t
                        cell = (int(round(wx / 0.04)), int(round(wy / 0.04)))
                        if (
                            cell not in elevated_occupied_world
                            and cell not in elevated_pending_world
                            and len(elevated_pending_world) < 40000
                        ):
                            elevated_pending_world[cell] = (obs_mono, 1)
                    continue
                if obs_mono != stamp_frame_mono.get(eye_key):
                    continue
                # Stamping frame. DISTANCE-SCALED TRUST: points within 0.9m
                # stamp immediately (close range is monocular depth's good
                # regime, and a stop must teach the planner NOW — directive:
                # every stop teaches something); farther points only become
                # obstacles when a pending observation >= 0.4s older agrees
                # (this drain's confirmation frame or a previous drain). The
                # exit-corridor splats were all 1.3..1.9m one-offs.
                # Promotions are BUDGETED per vantage (see
                # elevated_vantage_promotions) so a wedged robot cannot
                # confirm its own artifacts indefinitely from one pose.
                vantage_bucket = (
                    int(round(float(stamp_pose.x) / 0.25)),
                    int(round(float(stamp_pose.y) / 0.25)),
                    int(round((float(stamp_pose.theta_deg) % 360.0) / 20.0)),
                )
                vantage_used = elevated_vantage_promotions.get(vantage_bucket, 0)
                vantage_capped = False
                for pf, pl in footprint:
                    wx = float(stamp_pose.x) + float(pf) * cos_t - float(pl) * sin_t
                    wy = float(stamp_pose.y) + float(pf) * sin_t + float(pl) * cos_t
                    cell = (int(round(wx / 0.04)), int(round(wy / 0.04)))
                    if cell in elevated_occupied_world:
                        cell_meta = elevated_cell_meta.get(cell)
                        if cell_meta is not None:
                            # Re-detected: DECREMENT misses, don't reset.
                            # Systematic artifacts (edge-bleed re-detected
                            # from the same vantage every pass) must still
                            # decay when clean looks outnumber dirty ones.
                            cell_meta[1] = max(0, int(cell_meta[1]) - 1)
                            # CROSS-ANGLE CONFIRMATION: bleed lies along the
                            # viewing ray, so a re-detection at the SAME
                            # world position from a bearing >=28deg away is
                            # near-proof of a real edge — promote POTENTIAL
                            # (yellow) to CONFIRMED (red).
                            if len(cell_meta) >= 3 and not cell_meta[2]:
                                obs_bearing_deg = math.degrees(
                                    math.atan2(
                                        wy - float(stamp_pose.y),
                                        wx - float(stamp_pose.x),
                                    )
                                )
                                if (
                                    abs(
                                        _normalize_angle_deg(
                                            obs_bearing_deg - float(cell_meta[0])
                                        )
                                    )
                                    >= 28.0
                                ):
                                    cell_meta[2] = 1
                                    promoted_cells += 1
                        continue
                    if float(pf) > 0.9:
                        pending = elevated_pending_world.get(cell)
                        if pending is None:
                            if len(elevated_pending_world) < 40000:
                                elevated_pending_world[cell] = (obs_mono, 1)
                            continue
                        if obs_mono - pending[0] < 0.4:
                            continue
                    if vantage_used >= 60:
                        vantage_capped = True
                        continue
                    if len(elevated_occupied_world) < 20000:
                        elevated_pending_world.pop(cell, None)
                        elevated_occupied_world.add(cell)
                        # [first_bearing_deg, misses, confirmed] — new cells
                        # start POTENTIAL (yellow) until re-detected from a
                        # bearing >=28deg away (cross-angle confirmation).
                        elevated_cell_meta[cell] = [
                            math.degrees(
                                math.atan2(wy - float(stamp_pose.y), wx - float(stamp_pose.x))
                            ),
                            0,
                            0,
                        ]
                        vantage_used += 1
                        new_cells += 1
                        if len(elevated_points_world) < 6000:
                            elevated_points_world[cell] = (wx, wy, float(edge.height_m))
                            new_points += 1
                elevated_vantage_promotions[vantage_bucket] = vantage_used
                if vantage_capped:
                    print(
                        "[edge-map] vantage promotion budget reached at this pose; "
                        "further cells need a new viewpoint"
                    )
                # Pending entries that never re-confirm are noise; sweep the
                # stale ones so the dict cannot grow without bound.
                if len(elevated_pending_world) > 20000:
                    stale_before = now_mono - 30.0
                    for stale_cell in [
                        c for c, (m, _) in elevated_pending_world.items() if m < stale_before
                    ]:
                        del elevated_pending_world[stale_cell]
                continue
            if now_mono - float(edge.monotonic) > 1.0:
                continue
            w1 = (
                float(stamp_pose.x) + edge.p1_robot_xy[0] * cos_t - edge.p1_robot_xy[1] * sin_t,
                float(stamp_pose.y) + edge.p1_robot_xy[0] * sin_t + edge.p1_robot_xy[1] * cos_t,
            )
            w2 = (
                float(stamp_pose.x) + edge.p2_robot_xy[0] * cos_t - edge.p2_robot_xy[1] * sin_t,
                float(stamp_pose.y) + edge.p2_robot_xy[0] * sin_t + edge.p2_robot_xy[1] * cos_t,
            )
            seg_cells = _stamp_world_segment(w1, w2, float(edge.height_m))
            new_cells += seg_cells
            segment_added = segment_added or seg_cells > 0
        # NEGATIVE-EVIDENCE DECAY: fused frames the monitor analyzed report
        # their (possibly empty) footprint. A decayable cell the view
        # re-inspected — good range, in-view, from a bearing similar to the
        # one it was stamped from — without re-detecting counts a miss;
        # 3 misses remove it. Pose association uses the same freshness
        # windows as edge stamping (the robot is settled at drains).
        view_reports = (
            hazard_monitor.drain_view_reports()
            if hasattr(hazard_monitor, "drain_view_reports")
            else []
        )
        if view_reports and elevated_cell_meta:
            newest_report_mono = max(float(r.monotonic) for r in view_reports)
            if now_mono - newest_report_mono <= 3.0:
                report_window_s = 0.3 if newest_only else 1.5
                selected_reports = [
                    r
                    for r in view_reports
                    if newest_report_mono - float(r.monotonic) <= report_window_s
                ][-2:]
                decayed_cells = 0
                for report in selected_reports:
                    report_pts_world = [
                        (
                            float(stamp_pose.x) + float(pf) * cos_t - float(pl) * sin_t,
                            float(stamp_pose.y) + float(pf) * sin_t + float(pl) * cos_t,
                        )
                        for pf, pl in (report.points_robot_xy or ())
                    ]
                    for cell in list(elevated_cell_meta.keys()):
                        if cell not in elevated_occupied_world:
                            elevated_cell_meta.pop(cell, None)
                            continue
                        cell_meta = elevated_cell_meta[cell]
                        cell_wx, cell_wy = cell[0] * 0.04, cell[1] * 0.04
                        dxc = cell_wx - float(stamp_pose.x)
                        dyc = cell_wy - float(stamp_pose.y)
                        fwd = cos_t * dxc + sin_t * dyc
                        lat = -sin_t * dxc + cos_t * dyc
                        rng = math.hypot(dxc, dyc)
                        if not (
                            0.35 <= rng <= 1.55
                            and fwd > 0.0
                            and abs(math.degrees(math.atan2(lat, fwd))) <= 42.0
                        ):
                            continue
                        world_bearing = math.degrees(math.atan2(dyc, dxc))
                        bearing_diff = abs(
                            _normalize_angle_deg(world_bearing - float(cell_meta[0]))
                        )
                        cell_confirmed = len(cell_meta) >= 3 and bool(cell_meta[2])
                        # Verdict windows: CONFIRMED (red) cells only accept
                        # verdicts from near the original side (<=55deg — a
                        # far-side view legitimately sees a different leading
                        # boundary). POTENTIAL (yellow) cells accept up to
                        # 100deg: a clean look from a genuinely different
                        # angle is exactly the cross-check that disproves
                        # bleed; beyond ~100deg the object may self-occlude
                        # its own near boundary, so no verdict there either.
                        if bearing_diff > (55.0 if cell_confirmed else 100.0):
                            continue
                        near_hit = any(
                            (px - cell_wx) ** 2 + (py - cell_wy) ** 2 <= 0.15 * 0.15
                            for px, py in report_pts_world
                        )
                        if near_hit:
                            cell_meta[1] = max(0, int(cell_meta[1]) - 1)
                            if not cell_confirmed and bearing_diff >= 28.0:
                                cell_meta[2] = 1
                                promoted_cells += 1
                        else:
                            cell_meta[1] = int(cell_meta[1]) + 1
                            if cell_meta[1] >= 3:
                                elevated_occupied_world.discard(cell)
                                elevated_pending_world.pop(cell, None)
                                elevated_points_world.pop(cell, None)
                                elevated_cell_meta.pop(cell, None)
                                decayed_cells += 1
                if decayed_cells:
                    # Counted separately from floor clearing — merging the
                    # counts made the "floor evidence cleared N" line lie
                    # (run 19 logs: identical N on both lines).
                    decayed_cells_total += decayed_cells
                    print(
                        f"[edge-map] negative evidence decayed {decayed_cells} unconfirmed cells "
                        f"(map_cells={len(elevated_occupied_world)})"
                    )
        if new_points or cleared_cells or decayed_cells_total or promoted_cells:
            confirmed_pts: list[list[float]] = []
            potential_pts: list[list[float]] = []
            for point_cell, (px, py, ph) in elevated_points_world.items():
                point_meta = elevated_cell_meta.get(point_cell)
                if point_meta is None or (len(point_meta) >= 3 and point_meta[2]):
                    confirmed_pts.append([px, py, ph])
                else:
                    potential_pts.append([px, py, ph])
            rr.log(
                "world/elevated_regions",
                rr.Points3D(
                    confirmed_pts,
                    colors=[[200, 0, 0]] * len(confirmed_pts),
                    radii=0.02,
                ),
                static=True,
            )
            rr.log(
                "world/elevated_potential",
                rr.Points3D(
                    potential_pts,
                    colors=[[240, 205, 30]] * len(potential_pts),
                    radii=0.02,
                ),
                static=True,
            )
        if promoted_cells:
            print(
                f"[edge-map] cross-angle confirmed {promoted_cells} cells (yellow -> red, "
                f"map_cells={len(elevated_occupied_world)})"
            )
        if cleared_cells:
            print(
                "[edge-map] floor evidence cleared "
                f"{cleared_cells} stale cells (map_cells={len(elevated_occupied_world)})"
            )
        if new_points:
            print(
                "[edge-map] elevated footprint mapped "
                f"(+{new_points} points, map_cells={len(elevated_occupied_world)}, "
                f"pending={len(elevated_pending_world)})"
            )
        if segment_added and elevated_segments_world:
            latest = elevated_segments_world[-1]
            latest_width_m = math.hypot(
                latest["p2"][0] - latest["p1"][0], latest["p2"][1] - latest["p1"][1]
            )
            print(
                "[edge-map] elevated edge mapped "
                f"(height={latest['height_m']:.2f}m, width={latest_width_m:.2f}m, "
                f"segments={len(elevated_segments_world)}, map_cells={len(elevated_occupied_world)})"
            )
        return new_cells

    def _settle_and_stamp(stamp_pose: Pose2D) -> None:
        """Flush measurements taken mid-motion (their pose is unknowable),
        let the monitor re-measure at rest, then stamp with the settled pose."""
        if hazard_monitor is None:
            return
        hazard_monitor.drain_confirmed_edges()
        if hasattr(hazard_monitor, "drain_floor_evidence"):
            hazard_monitor.drain_floor_evidence()
        # Two fresh frames per eye (~0.56s cadence each) must accumulate so
        # the map's two-frame persistence gate can confirm far cells from
        # this settled pose, not just stamp the near band.
        time.sleep(1.3)
        _stamp_confirmed_edges(stamp_pose)

    def _investigate_edge_hint(start_pose: Pose2D) -> MotionHint | None:
        """Active edge survey (--edge-mapping on): back off for a parallax
        runway, re-approach slowly so the edge gets MEASURED, back off to a
        stand-off, then sweep the heading so the edge's full extent crosses
        both eyes. Segments are stamped at each settled tracked pose; the
        next planning cycle routes around the mapped boundary."""
        print("[edge-map] investigating elevated edge (back off -> measure -> sweep)")
        pose = start_pose
        cells_at_start = len(elevated_occupied_world)
        escape_hint = _attempt_reverse_escape_hint(pose, False)
        if escape_hint is None:
            print("[edge-map] no clearance to back off; skipping investigation")
            return None
        pose = _advance_pose(pose, escape_hint)
        _settle_and_stamp(pose)

        # Measurement pass: the slow re-approach IS the parallax baseline;
        # the safety gate stops it before contact.
        pose, approach_meta = _drive_with_tracking(
            robot=robot,
            feed=feed,
            zone_cfg=zone_cfg,
            transformed_sets=stitch_state["transformed_sets"],
            start_pose=pose,
            resolution_m=float(args.stitch_resolution_m),
            forward_speed=float(args.move_speed),
            min_effective_move_speed=float(args.min_effective_move_speed),
            burst_s=min(0.9, float(args.move_burst_s)),
            burst_count=2,
            inter_burst_pause_s=0.15,
            steer_theta_vel=0.0,
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            invert_lateral_axis=bool(args.invert_lateral_axis),
            max_distance_m=float(args.max_distance_m),
            min_range_m=float(args.min_range_m),
            min_confidence=int(args.min_confidence),
            hazard_monitor=hazard_monitor,
            pose_trusted=not pose_lost,
        )
        _settle_and_stamp(pose)
        if len(elevated_occupied_world) == cells_at_start:
            if bool(approach_meta.get("stopped_by_hazard")):
                # The re-approach got gate-stopped again but produced no
                # publishable geometry. NO fabricated scar (synthetic
                # blockers removed) — the frontier strike below is the
                # teaching mechanism.
                print(
                    "[edge-map] hazard stop with no measurable geometry; nothing mapped "
                    "(strikes handle repeat offenders)"
                )
            else:
                # The re-approach drove its full budget WITHOUT a stop: the
                # earlier detection did not reproduce. Treat it as a false
                # positive and map nothing (field 2026-07-11: stamping here
                # anyway planted a phantom wall in the middle of the room).
                print(
                    "[edge-map] re-approach passed cleanly; earlier stop looks like a "
                    "false positive - nothing mapped"
                )

        # Back to a safe survey stand-off (also releases any post-stop hold).
        escape_hint = _attempt_reverse_escape_hint(pose, False)
        if escape_hint is not None:
            pose = _advance_pose(pose, escape_hint)
            _settle_and_stamp(pose)

        # The old 35deg -> 70deg -> 35deg eye sweep ended at the original
        # heading but spent three turns oscillating in place. The leading
        # boundary already suffices for the 2-D planner; preserve this
        # stand-off heading and let the next cycle attempt a forward route.
        print("[edge-map] boundary measured; skipping rotational sweep and replanning forward")

        # One composite motion hint for the checkpoint capture: net tracked
        # motion from where the investigation began to where it ended.
        net_dx_world = float(pose.x) - float(start_pose.x)
        net_dy_world = float(pose.y) - float(start_pose.y)
        theta0_rad = math.radians(float(start_pose.theta_deg))
        cos0, sin0 = math.cos(theta0_rad), math.sin(theta0_rad)
        print(
            "[edge-map] investigation complete; capturing checkpoint and replanning "
            f"around {len(elevated_occupied_world)} mapped edge cells"
        )
        return MotionHint(
            kind="mixed",
            expected_dx_local_m=cos0 * net_dx_world + sin0 * net_dy_world,
            expected_dy_local_m=-sin0 * net_dx_world + cos0 * net_dy_world,
            expected_dtheta_deg=_normalize_angle_deg(
                float(pose.theta_deg) - float(start_pose.theta_deg)
            ),
            search_xy_m=0.50,
            search_theta_window_deg=28.0,
            label=f"edge_investigation_{capture_index:02d}",
        )

    try:
        last_frame_id, latest_frame = _wait_for_initial_frame(feed, timeout_s=5.0)
        print(
            "[wander] LiDAR feed ready "
            f"(frame_id={last_frame_id}, host_rev={latest_frame.revolution_index}, viewer={viewer_url or 'local'})"
        )

        # STARTUP WALL ESCAPE (field 2026-07-20: user started the robot nose-to-a-
        # wall). A large arc of CLOSE returns means the robot is cornered; if we
        # calibrate the self-mask now it swallows the wall (>150deg masked) and
        # starves the baseline — which used to CRASH the whole program. The
        # boxed-in usable-count check can't see this (a wall at 0.4m still counts
        # as "usable"), so we go by how much of the ring is close and DRIVE toward
        # open space first, so the mask learns only the arms.
        def _startup_close_coverage_deg() -> float:
            _wid, _wf = feed.latest()
            if _wf is None:
                return 0.0
            _bins = set()
            for _a, _d, _c in _wf.points:
                if int(_c) >= int(args.min_confidence) and 0.05 < float(_d) < 0.55:
                    _bins.add(int((float(_a) % 360.0) / 4.0))
            return len(_bins) * 4.0

        for _wall_attempt in range(4):
            _cov_deg = _startup_close_coverage_deg()
            if _cov_deg <= 130.0:  # arms (~66deg) + slack -> in the clear
                break
            print(
                f"[wander] STARTUP: ~{_cov_deg:.0f}deg of bearings have close (<0.55m) returns — the "
                "robot is against a wall / cornered (that would blind the self-mask). Driving to "
                f"open space first (attempt {_wall_attempt + 1}/4)..."
            )
            _open_id, _open_frame = feed.latest()
            _open_choice = None if _open_frame is None else _select_frontier_choice(
                _open_frame,
                forward_angle_deg=float(args.forward_angle_deg),
                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                max_distance_m=float(args.max_distance_m),
                min_range_m=float(args.min_range_m),
                min_confidence=int(args.min_confidence),
                frontier_min_distance_m=0.7,
                frontier_bin_deg=float(args.frontier_bin_deg),
            )
            _deg_per_burst = max(4.0, math.degrees(float(args.turn_speed) * float(args.turn_burst_s)))
            if _open_choice is None or float(_open_choice.mean_distance_m) < 0.7:
                # No clear opening from here — rotate ~90deg (can't collide) and re-look.
                for _ in range(max(1, int(round(90.0 / _deg_per_burst)))):
                    _execute_turn_burst(robot=robot, direction_sign=1.0, turn_speed=float(args.turn_speed),
                                        turn_burst_s=float(args.turn_burst_s), turn_settle_s=float(args.turn_settle_s))
                _send_stop(robot)
                continue
            # Face the open direction, then drive a bounded burst toward it (it is
            # clear >=0.7m by construction, so forward is safe).
            _open_sign = 1.0 if float(_open_choice.delta_deg) >= 0.0 else -1.0
            for _ in range(min(12, int(round(abs(float(_open_choice.delta_deg)) / _deg_per_burst)))):
                _execute_turn_burst(robot=robot, direction_sign=_open_sign, turn_speed=float(args.turn_speed),
                                    turn_burst_s=float(args.turn_burst_s), turn_settle_s=float(args.turn_settle_s))
            _send_stop(robot)
            _fwd_deadline = time.monotonic() + 1.0
            while time.monotonic() < _fwd_deadline:
                robot.send_action({"x.vel": max(float(args.min_effective_move_speed), 0.75), "y.vel": 0.0,
                                   "theta.vel": 0.0, "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
                                   "untorque_left": True, "untorque_right": True})
                time.sleep(0.05)
            _send_stop(robot)
            time.sleep(0.4)
            _new_id, _ = feed.latest()
            if _new_id is not None:
                last_frame_id = int(_new_id)
        else:
            print("[wander] STARTUP: still cornered after 4 escape attempts — proceeding anyway; the "
                  "self-mask may be wide and collision detection degraded on those bearings.")

        # SELF-MASK calibration: while still stationary, learn which bearings
        # the lidar sees the robot's OWN arms/shell in and drop those returns
        # from every future frame (user 2026-07-18: "the arms are a bit in the
        # way of the lidar"). Must run BEFORE the stop-box baseline so the
        # baseline measures a clean view and keeps its full trigger headroom.
        mask_sample_frames = []
        mask_frame_id = int(last_frame_id)
        for _mask_index in range(20):
            mask_frame_id_new, mask_frame = feed.wait_for_frame_after(
                after_frame_id=mask_frame_id,
                timeout_s=1.0,
                min_frame_advances=1,
            )
            if mask_frame is None:
                break
            mask_frame_id = int(mask_frame_id_new)
            if len(mask_frame.points) < 150:
                continue
            mask_sample_frames.append(mask_frame)
            if len(mask_sample_frames) >= 8:
                break
        if len(mask_sample_frames) >= 4:
            masked_bins, masked_deg = feed.calibrate_self_mask(mask_sample_frames)
            last_frame_id = mask_frame_id
            if masked_bins > 0:
                print(
                    f"[wander] lidar self-mask ARMED: {masked_deg:.0f}deg of bearings show "
                    "persistent close returns (robot arms/shell) — those returns are now "
                    "filtered from mapping, stop box, and frontier detection"
                )
                # The mask zeroes the stop-box baseline, but arm JITTER during
                # motion still leaks a handful of points past it (run 11: 6-14
                # per frame — with trigger at baseline+6=6 that blocked every
                # single drive burst at 0.05m). Floor the trigger so residual
                # self-jitter cannot block; a thin real obstacle (~12+ pts in
                # the box) and any wall (30-60) still stop the robot.
                if zone_cfg.min_points_to_trigger < 12:
                    zone_cfg.min_points_to_trigger = 12
                    print(
                        "[wander] stop-box trigger floored at 12 points while the self-mask "
                        "is armed (residual arm jitter must not veto drives)"
                    )
            else:
                print("[wander] lidar self-mask: no persistent self-returns found (clean view)")
            if masked_deg > 150.0:
                print(
                    f"[wander] WARNING: self-mask covers {masked_deg:.0f}deg — if the robot is "
                    "parked close to a wall, that wall just got masked and collision "
                    "detection is blinded on those bearings. Reposition with clear space "
                    "and restart if this is not the arms."
                )
        else:
            print(
                "[wander] lidar self-mask skipped (not enough healthy stationary revolutions); "
                "arm returns will pollute the stop box and map"
            )

        # Measure how many points the stationary lidar ALWAYS reports inside the
        # stop box (robot shell / mounts). Blocking then triggers only on points
        # above this baseline; otherwise the robot believes it is permanently
        # blocked and spends the whole run rotating in place.
        baseline_samples: list[int] = []
        squeeze_baseline_samples: list[int] = []
        baseline_frame_id = int(last_frame_id)
        for _sample_index in range(24):
            sample_frame_id, sample_frame = feed.wait_for_frame_after(
                after_frame_id=baseline_frame_id,
                timeout_s=1.0,
                min_frame_advances=1,
            )
            if sample_frame is None:
                break
            baseline_frame_id = int(sample_frame_id)
            if len(sample_frame.points) < 150:
                # Degraded revolution (spin-up / USB stutter): a baseline
                # measured from 7-11-point frames (field 2026-07-17:
                # samples=[0,19,...] baseline=9 vs the true ~40) makes the
                # stop box simultaneously hair-triggered and blind.
                print(
                    f"[wander] skipping degraded revolution in baseline sampling "
                    f"(points={len(sample_frame.points)})"
                )
                continue
            baseline_samples.append(_blocked_points_for_frame(sample_frame, zone_cfg))
            # Same frame, narrow squeeze band: its own self-hit baseline.
            zone_cfg.squeeze_active = True
            squeeze_baseline_samples.append(_blocked_points_for_frame(sample_frame, zone_cfg))
            zone_cfg.squeeze_active = False
            if len(baseline_samples) >= 10:
                break
        if len(baseline_samples) >= 5:
            zone_cfg.baseline_points = int(np.median(np.asarray(baseline_samples, dtype=np.int32)))
            zone_cfg.squeeze_baseline_points = int(
                np.median(np.asarray(squeeze_baseline_samples, dtype=np.int32))
            )
            last_frame_id = baseline_frame_id
        else:
            # Never CRASH here (field 2026-07-20). If we still can't get a clean
            # baseline (feed genuinely degraded, or the robot is wedged despite the
            # wall-escape above), proceed with baseline=0 — the 12-point trigger
            # floor still guards against collisions — instead of killing the run.
            print(
                f"[wander] WARNING: could not collect a clean stop-box baseline (got "
                f"{len(baseline_samples)}) even after the wall-escape — proceeding with baseline=0 "
                "(the 12-point trigger floor still protects against collisions). If the LiDAR feed "
                "is genuinely degraded, check the stream host on the Pi."
            )
            zone_cfg.baseline_points = 0
            zone_cfg.squeeze_baseline_points = 0
        print(
            "[wander] stop box self-hit baseline "
            f"(samples={baseline_samples}, baseline={zone_cfg.baseline_points}, "
            f"trigger_at={zone_cfg.blocked_trigger_count()} points; squeeze band "
            f"{zone_cfg.squeeze_half_width_m:.2f}m baseline={zone_cfg.squeeze_baseline_points})"
        )
        if zone_cfg.baseline_points >= 60:
            print(
                f"[wander] WARNING: the stop box is nearly saturated at rest "
                f"(baseline={zone_cfg.baseline_points} points in a 0.14m box) — the robot is "
                "almost certainly WEDGED against an obstacle. Blocked detection is degraded and "
                "escape maneuvers will struggle; reposition the robot with clear space ahead "
                "before mapping."
            )
        elif zone_cfg.baseline_points > 0:
            print(
                "[wander] note: the lidar permanently sees part of the robot inside the stop box; "
                "consider recalibrating the stop box geometry"
            )

        # BOXED-IN ESCAPE (user directive 2026-07-18): if the founding scan is
        # surrounded by near returns, translate toward open space and keep
        # scanning until healthy — instead of failing. Rotating in place would
        # not help (a 360deg scan is heading-invariant).
        _probe_id, _probe_frame = feed.latest()
        if _probe_frame is not None:
            _probe_usable = _scan_to_local_points(
                points=_probe_frame.points,
                forward_angle_deg=float(args.forward_angle_deg),
                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                invert_lateral_axis=bool(args.invert_lateral_axis),
                max_distance_m=float(args.max_distance_m),
                min_confidence=int(args.min_confidence),
                min_range_m=float(args.min_range_m),
            )
            if len(_probe_usable) < 60:
                print(
                    f"[wander] founding scan is boxed in (usable={len(_probe_usable)} < 60); "
                    "translating toward open space before founding the map"
                )
                last_frame_id = _escape_boxed_in(
                    robot=robot,
                    feed=feed,
                    args=args,
                    after_frame_id=last_frame_id,
                    min_usable=60,
                )

        frame_id, captured_frame, _, initial_snapshot = _capture_snapshot(
            feed=feed,
            output_dir=snapshot_dir,
            request_index=1,
            after_frame_id=last_frame_id,
            forward_angle_deg=float(args.forward_angle_deg),
            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
            invert_lateral_axis=bool(args.invert_lateral_axis),
            max_distance_m=float(args.max_distance_m),
            min_range_m=float(args.min_range_m),
            min_confidence=int(args.min_confidence),
            fresh_frame_timeout_s=float(args.fresh_frame_timeout_s),
            fresh_frame_advances=int(args.fresh_frame_advances),
            capture_config_extra={"motion_hint": motion_hints[-1].kind},
        )
        last_frame_id = int(frame_id)
        last_captured_frame = captured_frame

        stitch_state = _append_stitch(
            stitch_dir=stitch_dir,
            snapshot_dir=snapshot_dir,
            stitch_state={"snapshots": [], "poses": [], "transformed_sets": [], "solve_log": []},
            motion_hints=motion_hints,
            new_snapshot=initial_snapshot,
            resolution_m=float(args.stitch_resolution_m),
        )
        _log_rerun_state(
            rr,
            capture_index=1,
            transformed_sets=stitch_state["transformed_sets"],
            poses=stitch_state["poses"],
            solve_log=stitch_state["solve_log"],
            cone_half_width_deg=float(args.valid_angle_half_width_deg),
            body_radius_m=float(args.robot_radius_m),
        )
        initial_pose = stitch_state["poses"][-1]
        initial_heading_bin = _heading_bin_index(
            float(initial_pose.theta_deg),
            bin_count=rotation_coverage_bin_count,
        )
        rotation_coverage_bins_seen.add(initial_heading_bin)
        print(
            "[wander] rotation coverage update "
            f"(capture=1, heading={float(initial_pose.theta_deg):.1f}deg, "
            f"bins={sorted(rotation_coverage_bins_seen)}/{rotation_coverage_bin_count})"
        )
        print(f"[wander] initial stitched map ready: {stitch_state['html_path']}")

        capture_index = 2
        last_camera_feedback_monotonic = 0.0
        # Panorama is a bounded phase, not a condition that can repeat after
        # discarded captures. Count commanded turns independently of appended
        # snapshots so a weak scan cannot trap the robot spinning in place.
        bootstrap_turn_commands = 0
        # Feed circuit breaker: if the lidar host stops delivering fresh
        # revolutions, ALL motion must stop. Field 2026-07-11: the host died
        # mid-run and the loop kept executing turns and reverses BLIND on a
        # frozen frame, appending the identical snapshot 13 times while the
        # robot physically moved with zero sensing.
        feed_watch_frame_id = -1
        feed_watch_advance_monotonic = time.monotonic()
        feed_stall_announced = False
        while True:
            if int(args.max_captures) > 0 and capture_index > int(args.max_captures):
                break
            if elevated_block_recent > 0:
                # Ages out the elevated-hold marker one iteration at a time so the
                # redundant-capture breaker only spares a frontier while furniture
                # is actually the reason for the stall (not indefinitely).
                elevated_block_recent -= 1
                if elevated_block_recent == 0:
                    # The furniture episode is over (moved on) — allow a fresh
                    # deliberate retreat if it gets caught again somewhere new.
                    caught_retreated = False
            live_frame_id, live_frame = feed.latest()
            if live_frame is None:
                raise RuntimeError("LiDAR feed disappeared during wander loop.")
            if int(live_frame_id) != feed_watch_frame_id:
                feed_watch_frame_id = int(live_frame_id)
                feed_watch_advance_monotonic = time.monotonic()
                if feed_stall_announced:
                    print("[wander] lidar feed recovered; resuming exploration")
                    feed_stall_announced = False
            elif time.monotonic() - feed_watch_advance_monotonic > 5.0:
                _send_stop(robot)
                if not feed_stall_announced:
                    print(
                        "[wander] WARNING: lidar feed STALLED (no new revolution for >5s; "
                        "host down or network lost) — ALL motion halted; waiting for the "
                        "feed to recover. Restart the lidar host on the Pi if this persists."
                    )
                    feed_stall_announced = True
                time.sleep(2.0)
                continue
            current_solved_pose = stitch_state["poses"][-1]
            # The stitched pose LAGS any motion whose capture was skipped as
            # redundant or discarded — the robot has physically moved but the
            # map anchor hasn't. Judge the live scan against the pending-motion-
            # advanced expectation, or a correctly-executed maneuver reads as
            # pose error (field 2026-07-18: a clean -36deg turn before a
            # skipped-as-redundant capture scored theta_error=36deg>24 and
            # latched pose LOST on a CORRECT 12.6-score match — one cycle after
            # the robot had finally driven out of the room).
            live_expected_pose = (
                _advance_pose(current_solved_pose, pending_motion_hint)
                if pending_motion_hint is not None
                else current_solved_pose
            )
            live_points_xy = _scan_to_local_points(
                points=live_frame.points,
                forward_angle_deg=float(args.forward_angle_deg),
                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                invert_lateral_axis=bool(args.invert_lateral_axis),
                max_distance_m=float(args.max_distance_m),
                min_confidence=int(args.min_confidence),
                min_range_m=float(args.min_range_m),
            )
            # IMU HEADING ANCHOR (field 2026-07-20, user: "we have an IMU and a
            # lidar ... why can't it know where it is on the map it just built").
            # The gyro tracks heading to ~0.5deg (calibration-proven) and keeps
            # tracking THROUGH a lock-lost turn, so it is the trustworthy heading
            # exactly when the lidar dead-reckon has DRIFTED. Anchor BOTH the search
            # centre AND the accept-gate reference to the IMU heading: the lidar then
            # only has to solve POSITION (a robust, unambiguous 2-D search), a
            # correct fix is no longer thrown out for disagreeing with a drifted
            # lidar guess, and the room's 90/180deg rotational aliases fall outside
            # the tightened heading window by construction. Falls back to the old
            # lidar-only behaviour when the IMU is absent/stale/sign-untrustworthy.
            live_solve_pose = live_expected_pose
            imu_theta_live = (
                _imu_resolved_theta(imu_yaw, imu_dead_reck_anchor)
                if (imu_yaw is not None and imu_yaw.sign_trustworthy())
                else None
            )
            if imu_theta_live is not None:
                live_solve_pose = dataclasses.replace(
                    live_expected_pose, theta_deg=float(imu_theta_live)
                )
            live_pose_result = _estimate_pose_against_stitched_map(
                points_xy=live_points_xy,
                transformed_sets=stitch_state["transformed_sets"],
                initial_pose=live_solve_pose,
                resolution_m=float(args.stitch_resolution_m),
                search_xy_m=max(float(args.drive_search_xy_m), 0.55),
                theta_window_deg=(
                    24.0
                    if imu_theta_live is not None
                    else max(float(args.drive_theta_window_deg), 36.0)
                ),
                max_translation_from_initial_m=max(0.45, float(args.drive_search_xy_m) + 0.20),
                prior_translation_weight=0.20,
                prior_theta_weight=0.25 if imu_theta_live is not None else 0.08,
            )
            live_pose_accepted = False
            if live_pose_result is not None:
                candidate_live_pose, live_pose_meta = live_pose_result
                if _accept_relocalized_pose(
                    label=f"live_frame_{live_frame_id}",
                    candidate_pose=candidate_live_pose,
                    score_meta=live_pose_meta,
                    expected_pose=live_solve_pose,
                    motion_hint=None,
                    max_translation_error_m=max(0.45, float(args.drive_search_xy_m) + 0.20),
                    max_theta_error_deg=max(24.0, float(args.drive_theta_window_deg) * 0.85),
                    min_score=7.0,
                ):
                    current_live_pose = candidate_live_pose
                    live_pose_accepted = True
                    dead_reck_theta_deg = float(candidate_live_pose.theta_deg)
                    dead_reck_slack_deg = 15.0
                    imu_dead_reck_anchor = _imu_anchor(imu_yaw, dead_reck_theta_deg)
                    if pose_lost and float(live_pose_meta.get("score") or 0.0) >= POSE_RECOVERY_MIN_SCORE:
                        pose_lost = False
                        post_recovery_strict_appends = 2
                        lost_recovery_failures = 0
                        relocalize_spin_deg, relocalize_cycles, relocalize_halted = 0.0, 0, False
                        print(
                            "[wander] pose integrity RESTORED "
                            f"(live score={float(live_pose_meta.get('score') or 0.0):.2f}); "
                            "map writes re-enabled (strict gates for the next 2 appends)"
                        )
                else:
                    current_live_pose = current_solved_pose
                    live_pose_meta = {
                        **live_pose_meta,
                        "source": "rejected_live_fallback",
                    }
                    if not pose_lost:
                        pose_lost = True
                        print(
                            "[wander] pose integrity LOST (live relocalization rejected); "
                            "map is READ-ONLY until a strong relocalization re-proves the pose"
                        )
                    print(
                        "[wander] live relocalization fallback "
                        f"(frame_id={live_frame_id}, using last stitched pose=({float(current_live_pose.x):.3f}, "
                        f"{float(current_live_pose.y):.3f}, {float(current_live_pose.theta_deg):.1f}deg))"
                    )
                    # Re-derive the dead-reckoned heading from the gyro BEFORE
                    # searching. The gyro integrated through every blind align/
                    # recovery turn since the anchor was set, so this stays
                    # accurate where the lidar-only anchor had gone stale.
                    imu_theta = _imu_resolved_theta(imu_yaw, imu_dead_reck_anchor)
                    if imu_theta is not None:
                        dead_reck_theta_deg = imu_theta
                        dead_reck_slack_deg = IMU_DEAD_RECK_SLACK_DEG
                    # Recovery search: when lost, the pose EXPECTATION is
                    # meaningless, so the normal accept gates (which compare
                    # against it — this failure rejected the correct
                    # whole-map candidate for a 58deg "theta error") cannot
                    # ever re-anchor a badly rotated robot.
                    #
                    # SYMMETRY-BREAK via the IMU heading (2026-07-19, explicit
                    # user directive). A wide +-180 search in a rectangular /
                    # symmetric room returns a MIRROR-mode pose that scores as
                    # high as the true one; the dead-reckoning veto below then
                    # rejects it and the robot "stays lost" FOREVER (field run
                    # 23 looped ~2700 frames at a doorway doing exactly this).
                    # The gyro is noisy in magnitude but UNAMBIGUOUS about which
                    # 90/180deg mode the robot is in, so when it is live and its
                    # sign is trusted we CENTER the search on it and clamp the
                    # window below the symmetry period — the alias modes fall
                    # outside the search by construction, rather than being
                    # (unsuccessfully) vetoed after the fact. An earlier
                    # gyro-centered variant ghosted the map by force-accepting
                    # marginal candidates; those guards remain in force — the
                    # veto below, and post_recovery_strict_appends keeping map
                    # WRITES gated until the lidar independently re-confirms — so
                    # a wrong lock re-anchors the pose (recoverable) without
                    # corrupting the map. The user accepted the IMU's
                    # unreliability here over looping forever.
                    recovery_seed_pose = live_expected_pose
                    recovery_theta_window_deg = 180.0
                    recovery_imu_break = (
                        imu_theta is not None
                        and imu_yaw is not None
                        and imu_yaw.sign_trustworthy()
                    )
                    if recovery_imu_break:
                        recovery_seed_pose = dataclasses.replace(
                            live_expected_pose, theta_deg=float(imu_theta)
                        )
                        # 40deg local window; the helper clamps the whole-map
                        # window to 2x (->min 90deg) centered on the gyro
                        # heading, so the +-90 and 180deg aliases sit outside it.
                        recovery_theta_window_deg = 40.0
                    recovery_result = _estimate_pose_against_stitched_map(
                        points_xy=live_points_xy,
                        transformed_sets=stitch_state["transformed_sets"],
                        initial_pose=recovery_seed_pose,
                        resolution_m=float(args.stitch_resolution_m),
                        search_xy_m=1.10,
                        theta_window_deg=recovery_theta_window_deg,
                        max_translation_from_initial_m=1.10,
                        prior_translation_weight=0.03,
                        prior_theta_weight=0.06 if recovery_imu_break else 0.0,
                    )
                    if recovery_imu_break:
                        print(
                            "[wander] wide-theta recovery seeded on the IMU heading "
                            f"({float(imu_theta):.1f}deg, +-{recovery_theta_window_deg:.0f}deg) to "
                            "break the room symmetry"
                        )
                    if recovery_result is not None:
                        recovery_pose, recovery_meta = recovery_result
                        recovery_score = float(recovery_meta.get("score") or -1e9)
                        dead_reck_err_deg: float | None = None
                        if dead_reck_theta_deg is not None:
                            dead_reck_err_deg = abs(
                                _normalize_angle_deg(
                                    float(recovery_pose.theta_deg) - float(dead_reck_theta_deg)
                                )
                            )
                        if recovery_score >= POSE_RECOVERY_MIN_SCORE and (
                            dead_reck_err_deg is None or dead_reck_err_deg <= dead_reck_slack_deg
                        ):
                            current_live_pose = recovery_pose
                            live_pose_meta = {**recovery_meta, "source": "lost_recovery"}
                            live_pose_accepted = True
                            pose_lost = False
                            post_recovery_strict_appends = 2
                            lost_recovery_failures = 0
                            relocalize_spin_deg, relocalize_cycles, relocalize_halted = 0.0, 0, False
                            dead_reck_theta_deg = float(recovery_pose.theta_deg)
                            dead_reck_slack_deg = 15.0
                            imu_dead_reck_anchor = _imu_anchor(imu_yaw, dead_reck_theta_deg)
                            print(
                                "[wander] pose integrity RESTORED via wide-theta recovery "
                                f"(pose=({recovery_pose.x:.3f}, {recovery_pose.y:.3f}, "
                                f"{recovery_pose.theta_deg:.1f}deg), score={recovery_score:.2f})"
                            )
                        elif recovery_score >= POSE_RECOVERY_MIN_SCORE:
                            lost_recovery_failures += 1
                            print(
                                "[wander] wide-theta recovery REJECTED by dead reckoning "
                                f"(candidate theta={recovery_pose.theta_deg:.1f}deg is "
                                f"{dead_reck_err_deg:.0f}deg from the tracked heading, bound=±"
                                f"{dead_reck_slack_deg:.0f}deg, score={recovery_score:.2f}); a "
                                "symmetric wrong mode scores this high too — staying lost"
                            )
                        else:
                            lost_recovery_failures += 1
                            print(
                                "[wander] wide-theta recovery inconclusive "
                                f"(best score={recovery_score:.2f} < {POSE_RECOVERY_MIN_SCORE:.1f}); staying lost"
                            )
                    if (
                        pose_lost
                        and len(stitch_state["snapshots"]) > 1
                        and lost_recovery_failures >= 2
                        and lost_recovery_failures % 2 == 0
                    ):
                        # ---- BOUNDED RELOCALIZE RECOVERY LADDER (never doom-loops) ----
                        # HARD CAP: two full spin+translate cycles and still lost ->
                        # STOP and sit. A visible halt the operator can fix beats
                        # thrashing back and forth or a blind ram (field run: a
                        # reverse-escape on a score-1.17 pose drove into a table).
                        if relocalize_cycles >= 2:
                            _send_stop(robot)
                            if not relocalize_halted:
                                relocalize_halted = True
                                print(
                                    "[wander] CANNOT RELOCALIZE after 2 full spin+translate recovery "
                                    "cycles — HALTING in place (no blind motion). The map is intact; "
                                    "reposition the robot or Ctrl+C."
                                )
                            continue
                        # STEP 1 — DELIBERATE SPIN: physically rotate ~85deg to bring
                        # DIFFERENT mapped features into view, then re-attempt recovery
                        # next iteration. Open-loop bursts (we only need the view to
                        # change, not to track the turn) so a lost/stale pose can't
                        # corrupt anything; in-place rotation cannot collide. This is
                        # the deliberate "look around to figure out where I am" — the
                        # symmetric-doorway case where the old room's landmarks are
                        # simply behind the robot and a turn brings them back.
                        if relocalize_spin_deg < 355.0:
                            _send_stop(robot)
                            deg_per_burst = max(4.0, math.degrees(float(args.turn_speed) * float(args.turn_burst_s)))
                            n_spin_bursts = max(1, int(round(85.0 / deg_per_burst)))
                            for _ in range(n_spin_bursts):
                                _execute_turn_burst(
                                    robot=robot,
                                    direction_sign=1.0,
                                    turn_speed=float(args.turn_speed),
                                    turn_burst_s=float(args.turn_burst_s),
                                    turn_settle_s=float(args.turn_settle_s),
                                )
                            _send_stop(robot)
                            relocalize_spin_deg += float(n_spin_bursts) * deg_per_burst
                            print(
                                "[wander] RELOCALIZE SPIN: rotated ~85deg to change the view "
                                f"(spun {relocalize_spin_deg:.0f}/360deg, cycle {relocalize_cycles + 1}/2) "
                                "— re-attempting to find myself"
                            )
                            continue
                        # STEP 2 — a FULL spin didn't re-lock, so the spot is
                        # genuinely degenerate/symmetric: only MOVING changes that.
                        # Back out along the way in (or advance toward a live
                        # opening), then reset the spin budget and start another
                        # cycle. The reverse helper is live-rear-checked
                        # (robot-relative, immune to the pose error we carry while
                        # lost).
                        if len(stitch_state["snapshots"]) > 1:
                            # ONE 360 SPIN, THEN STOP (field 2026-07-20 user: "if it
                            # is lost, do one giant 360 spin and anchor its position
                            # based on that, no more meandering"). A full revolution
                            # re-attempted relocalization every ~85deg and still did
                            # not re-lock — do NOT translate/change vantage and
                            # wander around (that WAS the meandering). HALT in place;
                            # the map is intact, reposition or Ctrl+C.
                            _send_stop(robot)
                            if not relocalize_halted:
                                relocalize_halted = True
                                print(
                                    "[wander] LOST: a full 360deg spin did not re-anchor the "
                                    "pose — HALTING in place (no meandering). Reposition the "
                                    "robot near a mapped landmark, or Ctrl+C."
                                )
                            continue
                        # Only the founding snapshot exists — nothing worth halting
                        # for; fall through to the re-found path below.
                        relocalize_spin_deg = 0.0
                        relocalize_cycles += 1
                        print(
                            "[wander] full relocalize spin did not re-lock; changing VANTAGE "
                            f"(recovery cycle {relocalize_cycles}/2) — a spin can't fix a "
                            "symmetric/degenerate spot, but moving can"
                        )
                        retreat_hint = _attempt_reverse_escape_hint(
                            current_live_pose, lost_recovery_failures >= 6
                        )
                        if retreat_hint is not None:
                            pending_motion_hint = retreat_hint
                            continue
                        # Rear blocked (map+live agree): translate toward the
                        # deepest LIVE opening instead — robot-relative, so
                        # immune to the pose error we necessarily carry while
                        # lost, and TRANSLATION (not rotation) is what
                        # un-degenerates a view (same physics as the boxed-in
                        # escape). Field 2026-07-18 run 12: lost at the new
                        # room's sparse edge with 0.00m behind, the
                        # rotation-only fallback thrashed lock-lost turns
                        # until Ctrl+C. Only when the opening is roughly
                        # ahead — a big blind turn while lost IS the thrash
                        # this replaces.
                        retreat_choice = _select_frontier_choice(
                            live_frame,
                            forward_angle_deg=float(args.forward_angle_deg),
                            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                            max_distance_m=float(args.max_distance_m),
                            min_range_m=float(args.min_range_m),
                            min_confidence=int(args.min_confidence),
                            frontier_min_distance_m=1.0,
                            frontier_bin_deg=float(args.frontier_bin_deg),
                            min_gap_width_m=2.0 * (float(args.robot_radius_m) + 0.05),
                            corridor_veto=None,
                        )
                        if (
                            retreat_choice is not None
                            and abs(float(retreat_choice.delta_deg)) <= 50.0
                            and float(retreat_choice.mean_distance_m) >= 1.0
                        ):
                            print(
                                "[wander] rear blocked while lost — advancing toward the live "
                                f"opening (delta={retreat_choice.delta_deg:.1f}deg, "
                                f"range={retreat_choice.mean_distance_m:.2f}m) to change the view"
                            )
                            pending_motion_hint = _live_frontier_probe_hint(
                                current_live_pose, retreat_choice
                            )
                            if exit_mode:
                                # Pose-lost + advancing on the committed opening
                                # == crossing the exit threshold. Count the
                                # MEASURED forward motion so the run can finish
                                # the crossing the frozen map pose can't certify.
                                exit_through_odometry_m += max(
                                    0.0, float(pending_motion_hint.expected_dx_local_m)
                                )
                                exit_opening_seen = True
                            continue
                    if (
                        pose_lost
                        and len(stitch_state["snapshots"]) <= 1
                        and lost_recovery_failures >= 2
                    ):
                        # RE-FOUND: the map is only its founding snapshot — it
                        # holds no investment worth a deadlock. Wipe it and
                        # start mapping fresh from wherever the robot stands.
                        print(
                            "[wander] map has only its founding snapshot and the pose cannot be "
                            "recovered — RE-FOUNDING the map from the current position"
                        )
                        _send_stop(robot)
                        motion_hints[:] = [
                            MotionHint(
                                kind="start",
                                expected_dx_local_m=0.0,
                                expected_dy_local_m=0.0,
                                expected_dtheta_deg=0.0,
                                search_xy_m=float(args.drive_search_xy_m),
                                search_theta_window_deg=float(args.turn_theta_window_deg),
                                label=f"refound_capture_{capture_index:02d}",
                            )
                        ]
                        refound_frame_id, refound_frame, _, refound_snapshot = _capture_snapshot(
                            feed=feed,
                            output_dir=snapshot_dir,
                            request_index=capture_index,
                            after_frame_id=int(live_frame_id),
                            forward_angle_deg=float(args.forward_angle_deg),
                            valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                            invert_lateral_axis=bool(args.invert_lateral_axis),
                            max_distance_m=float(args.max_distance_m),
                            min_range_m=float(args.min_range_m),
                            min_confidence=int(args.min_confidence),
                            fresh_frame_timeout_s=float(args.fresh_frame_timeout_s),
                            fresh_frame_advances=int(args.fresh_frame_advances),
                            capture_config_extra={"motion_hint": "start"},
                        )
                        last_frame_id = int(refound_frame_id)
                        last_captured_frame = refound_frame
                        stitch_state = _append_stitch(
                            stitch_dir=stitch_dir,
                            snapshot_dir=snapshot_dir,
                            stitch_state={
                                "snapshots": [],
                                "poses": [],
                                "transformed_sets": [],
                                "solve_log": [],
                            },
                            motion_hints=motion_hints,
                            new_snapshot=refound_snapshot,
                            resolution_m=float(args.stitch_resolution_m),
                        )
                        capture_index += 1
                        # Everything expressed in the OLD world frame is now
                        # fiction — wipe it all.
                        elevated_occupied_world.clear()
                        elevated_pending_world.clear()
                        elevated_vantage_promotions.clear()
                        elevated_segments_world.clear()
                        frontier_strike_counts.clear()
                        frontier_blacklist.clear()
                        unreachable_targets_world.clear()
                        active_explore_target = None
                        active_survey_xy = None
                        pending_motion_hint = None
                        consecutive_append_discards = 0
                        pose_lost = False
                        post_recovery_strict_appends = 0
                        lost_recovery_failures = 0
                        relocalize_spin_deg, relocalize_cycles, relocalize_halted = 0.0, 0, False
                        dead_reck_theta_deg = float(stitch_state["poses"][-1].theta_deg)
                        dead_reck_slack_deg = 15.0
                        imu_dead_reck_anchor = _imu_anchor(imu_yaw, dead_reck_theta_deg)
                        rotation_coverage_bins_seen.clear()
                        rotation_coverage_bins_seen.add(
                            _heading_bin_index(
                                float(stitch_state["poses"][-1].theta_deg),
                                bin_count=rotation_coverage_bin_count,
                            )
                        )
                        rotation_coverage_complete = False
                        bootstrap_turn_commands = 0
                        consecutive_turn_captures = 0
                        print(
                            "[wander] map re-founded "
                            f"(new founding snapshot has {len(refound_snapshot.points_xy)} points); "
                            "bootstrap panorama restarts"
                        )
                        continue
            if live_pose_result is None:
                current_live_pose = current_solved_pose
                print(
                    "[wander] live relocalization unavailable; using last stitched pose "
                    f"(frame_id={live_frame_id}, pose=({float(current_live_pose.x):.3f}, "
                    f"{float(current_live_pose.y):.3f}, {float(current_live_pose.theta_deg):.1f}deg))"
                )
            _log_live_pose_state(
                rr,
                capture_index=max(1, capture_index - 1),
                pose=current_live_pose,
                points_xy=live_points_xy,
                cone_half_width_deg=float(args.valid_angle_half_width_deg),
                body_radius_m=float(args.robot_radius_m),
            )
            # BODY-OVERLAP EVICTION: a decayable eye cell strictly inside the
            # robot's own body radius at a TRUSTED pose is physically
            # disproven — the robot occupies that space and no gate reports
            # contact. Such cells sit below the decay pass's minimum
            # inspection range (0.35m) and would otherwise be unfalsifiable
            # at point-blank (field 2026-07-17 run 18).
            if live_pose_accepted and not pose_lost and elevated_cell_meta:
                overlap_r_sq = (float(args.robot_radius_m) - 0.03) ** 2
                overlap_cells = [
                    c
                    for c in elevated_cell_meta
                    if (c[0] * 0.04 - float(current_live_pose.x)) ** 2
                    + (c[1] * 0.04 - float(current_live_pose.y)) ** 2
                    < overlap_r_sq
                ]
                for cell in overlap_cells:
                    elevated_occupied_world.discard(cell)
                    elevated_pending_world.pop(cell, None)
                    elevated_points_world.pop(cell, None)
                    elevated_cell_meta.pop(cell, None)
                if overlap_cells:
                    print(
                        f"[edge-map] evicted {len(overlap_cells)} cells inside the robot's own "
                        "body footprint (physically disproven)"
                    )
            # PINNED-BY-GUARD ESCAPE: the side guard halting every forward
            # burst at the same pose is a livelock, not protection (run 18:
            # 130+ identical replan/halt cycles). Back away from the cell.
            if map_side_guard_state["streak"] >= 3:
                print(
                    "[wander] side-collision guard has pinned the robot "
                    f"({map_side_guard_state['streak']} zero-motion halts); backing away "
                    "from the mapped cell"
                )
                map_side_guard_state["streak"] = 0
                map_side_guard_state["pose"] = None
                pinned_retreat = _attempt_reverse_escape_hint(current_live_pose, True)
                if pinned_retreat is not None:
                    pending_motion_hint = pinned_retreat
                    continue
            # Keep the Rerun camera panels live between captures. These are
            # the exact annotated frames used by the safety monitor, not a
            # second video path; viewing them cannot affect control.
            _record_eye_frames()
            feedback_hz = max(0.0, float(args.camera_feedback_hz))
            if (
                hazard_monitor is not None
                and feedback_hz > 0.0
                and time.monotonic() - last_camera_feedback_monotonic >= 1.0 / feedback_hz
            ):
                last_camera_feedback_monotonic = time.monotonic()
                live_state = hazard_monitor.state()
                for safety_cam in ("front_left", "front_right", "panorama", "bottom"):
                    safety_frame = hazard_monitor.annotated(safety_cam)
                    if safety_frame is not None:
                        rr.log(f"cameras/live_{safety_cam}", rr.Image(safety_frame[:, :, ::-1]))
                rr.log(
                    "safety/live_decision",
                    rr.TextLog(
                        f"{live_state.decision_label()} side={live_state.side} "
                        f"distance={live_state.est_distance_m} detail={live_state.detail}"
                    ),
                )

            blocked_points = _blocked_points_for_frame(live_frame, zone_cfg)
            blocked = blocked_points >= zone_cfg.blocked_trigger_count()
            drive_hint_m = 0.0
            should_turn = False
            turn_reason = "none"
            chosen_direction_sign = float(direction_sign)
            chosen_turn_deg = float(args.turn_deg)
            motion_hint: MotionHint | None = None
            # Set when a hazard stop happened after so little motion that a
            # checkpoint capture would re-anchor nothing: skip the capture
            # and carry the tracked motion into the next one instead.
            checkpoint_skippable = False
            settle_s = float(args.move_settle_s)
            turn_capture_theta_window_deg = 80.0
            bootstrap_scan_active = (
                wander_mode == "smart"
                and not rotation_coverage_complete
                and bootstrap_turn_commands < max(1, int(args.bootstrap_turn_captures))
            )
            # Legacy heuristic, superseded by the frontier planner in smart mode.
            forced_drive_due_to_turn_streak = (
                wander_mode == "smart-legacy"
                and not blocked
                and consecutive_turn_captures >= max(2, min(4, int(args.bootstrap_turn_captures)))
            )
            if forced_drive_due_to_turn_streak:
                force_drive_after_turn = True
                print(
                    "[wander] forcing forward exploration after repeated turn captures "
                    f"(turn_streak={consecutive_turn_captures}, coverage_complete={rotation_coverage_complete})"
                )

            if pending_turn_reason is not None:
                turn_reason = str(pending_turn_reason)
                chosen_direction_sign = float(pending_turn_direction_sign)
                chosen_turn_deg = float(pending_turn_deg)
                print(
                    "[wander] executing queued turn "
                    f"(reason={turn_reason}, target={chosen_turn_deg:.1f}deg "
                    f"{'ccw' if chosen_direction_sign >= 0.0 else 'cw'})"
                )
                should_turn = True
                pending_turn_reason = None
                pending_turn_direction_sign = float(direction_sign)
                pending_turn_deg = float(args.turn_deg)

            def _elevated_corridor_blocked(delta_deg: float, distance_m: float) -> bool:
                """True when the straight corridor toward a live-lidar opening
                crosses CONFIRMED (red) eye-mapped elevated cells. The scan sees
                under furniture; the eyes mapped that furniture — believe the
                eyes when choosing where to drive.

                Confirmed cells ONLY: yellow cells are by the two-tier design
                UNVERIFIED claims, and field 2026-07-18 (edge detection back on)
                showed a yellow phantom cluster parked in the exit corridor
                deleting the doorway from the candidate list entirely — "no live
                frontier candidate found" while the robot stared at the door,
                mission stalled. A claim awaiting verification must not veto the
                mission; the live eye stops and the measured-gap squeeze creep
                still physically protect the traversal if the furniture is real."""
                if not elevated_occupied_world:
                    return False
                bearing = math.radians(
                    float(current_live_pose.theta_deg) + float(delta_deg)
                )
                cos_b, sin_b = math.cos(bearing), math.sin(bearing)
                corridor_len = min(float(distance_m), 1.5)
                # Two ways to veto the corridor:
                #   red_hits   >= 2  — CONFIRMED (cross-angle) furniture, trusted fast.
                #   total_hits >= 6  — a DENSE cluster of mapped edge cells (red OR
                #                      yellow). Field 2026-07-20: the robot fixated on
                #                      a live-lidar "opening" (delta=-36deg, 2m) that
                #                      was really a DESK — the scan saw under it — and
                #                      drove into it every cycle, wedged, and could not
                #                      leave. The desk WAS mapped (dozens of cells) but
                #                      stayed YELLOW (no cross-angle view from the pin),
                #                      so the red-only veto never fired. A real object
                #                      packs many cells into the corridor; a stray
                #                      phantom (the reason yellow is normally distrusted)
                #                      is 1-2 cells and never reaches 6 — so this
                #                      believes a dense obstacle without letting a
                #                      phantom delete a real doorway.
                red_hits = 0
                total_hits = 0
                for cx, cy in elevated_occupied_world:
                    dxc = cx * 0.04 - float(current_live_pose.x)
                    dyc = cy * 0.04 - float(current_live_pose.y)
                    along = dxc * cos_b + dyc * sin_b
                    if not (0.10 <= along <= corridor_len):
                        continue
                    if abs(-dxc * sin_b + dyc * cos_b) <= 0.33:
                        total_hits += 1
                        cell_meta = elevated_cell_meta.get((cx, cy))
                        if cell_meta is not None and len(cell_meta) >= 3 and cell_meta[2]:
                            red_hits += 1
                        if red_hits >= 2 or total_hits >= 6:
                            return True
                # PERMANENT camera-confirmed blocks (never decay). The cameras
                # already STOPPED the robot driving this way once (a desk it saw
                # under). Two such cells in the corridor veto the opening for good,
                # so the picker re-plans toward a different opening instead of
                # re-committing to the phantom doorway (user 2026-07-20).
                cam_hits = 0
                for cx, cy in camera_blocked_world:
                    dxc = cx * 0.04 - float(current_live_pose.x)
                    dyc = cy * 0.04 - float(current_live_pose.y)
                    along = dxc * cos_b + dyc * sin_b
                    if not (0.10 <= along <= corridor_len):
                        continue
                    if abs(-dxc * sin_b + dyc * cos_b) <= 0.33:
                        cam_hits += 1
                        if cam_hits >= 2:
                            return True
                return False

            live_frontier_choice = _select_frontier_choice(
                live_frame,
                forward_angle_deg=float(args.forward_angle_deg),
                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                max_distance_m=float(args.max_distance_m),
                min_range_m=float(args.min_range_m),
                min_confidence=int(args.min_confidence),
                frontier_min_distance_m=float(args.frontier_min_distance_m),
                frontier_bin_deg=float(args.frontier_bin_deg),
                min_gap_width_m=2.0 * (float(args.robot_radius_m) + 0.05),
                corridor_veto=_elevated_corridor_blocked,
            )
            # EXIT-RUN HARD VETO: an already-mapped open room is ALSO a "deep
            # opening" — from the doorway, looking back across the room clears
            # every depth/width bar, so the per-cycle deepest-opening pick
            # ping-ponged between the exit and the room behind it and the robot
            # kept driving AWAY from the door it had just reached (field
            # 2026-07-18 run 15; user: "HARDSTOP IT FROM EVER GOING BACK").
            # While exit_mode is latched, an opening is acceptable ONLY if its
            # far end lands OUTSIDE the pre-exit coverage — the same 0.80m
            # test as exit completion. Openings leading back into coverage are
            # dead, unconditionally.
            if exit_mode and exit_latch_poses and live_frontier_choice is not None:
                _vetoed_deltas: list[float] = []
                for _exit_veto_round in range(24):
                    _end_rad = math.radians(
                        float(current_live_pose.theta_deg)
                        + float(live_frontier_choice.delta_deg)
                    )
                    _end_reach = 0.85 * float(live_frontier_choice.mean_distance_m)
                    _end_x = float(current_live_pose.x) + _end_reach * math.cos(_end_rad)
                    _end_y = float(current_live_pose.y) + _end_reach * math.sin(_end_rad)
                    # Accept the opening iff its far end lands OUTSIDE the mapped
                    # room footprint (distance from the room center exceeds the
                    # footprint radius + margin). This catches BOTH failure modes
                    # the nearest-vantage test could not: the look-back-across-
                    # the-room opening (far end near the opposite wall, still
                    # within room_radius) AND — crucially — it no longer vetoes
                    # the REAL doorway just because a vantage happened to sit
                    # near it (field run 24: the true exit "far end 0.65m from a
                    # vantage" got killed). Fall back to the old test only until
                    # the centroid is ready.
                    if exit_centroid_xy is not None:
                        _end_from_center = math.hypot(
                            _end_x - exit_centroid_xy[0], _end_y - exit_centroid_xy[1]
                        )
                        _leads_out = _end_from_center >= exit_room_radius_m + EXIT_END_MARGIN_M
                    else:
                        _leads_out = (
                            min(
                                math.hypot(_end_x - lx, _end_y - ly)
                                for lx, ly in exit_latch_poses
                            )
                            >= 0.80
                        )
                    if _leads_out:
                        break
                    print(
                        "[wander] EXIT RUN: opening at delta="
                        f"{live_frontier_choice.delta_deg:.1f}deg leads BACK into the mapped "
                        "room footprint (far end inside room_radius) — vetoed; going back is "
                        "forbidden"
                    )
                    _vetoed_deltas.append(float(live_frontier_choice.delta_deg))
                    live_frontier_choice = _select_frontier_choice(
                        live_frame,
                        forward_angle_deg=float(args.forward_angle_deg),
                        valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                        max_distance_m=float(args.max_distance_m),
                        min_range_m=float(args.min_range_m),
                        min_confidence=int(args.min_confidence),
                        frontier_min_distance_m=float(args.frontier_min_distance_m),
                        frontier_bin_deg=float(args.frontier_bin_deg),
                        min_gap_width_m=2.0 * (float(args.robot_radius_m) + 0.05),
                        corridor_veto=(
                            lambda seg_delta, seg_mean, _blocked=_elevated_corridor_blocked, _bad=tuple(_vetoed_deltas): (
                                _blocked(seg_delta, seg_mean)
                                or any(
                                    abs(_normalize_angle_deg(seg_delta - b)) < 10.0
                                    for b in _bad
                                )
                            )
                        ),
                    )
                    if live_frontier_choice is None:
                        break
            if live_frontier_choice is not None:
                print(
                    "[wander] live frontier candidate "
                    f"(frame_id={live_frame_id}, delta={live_frontier_choice.delta_deg:.1f}deg, "
                    f"distance={live_frontier_choice.mean_distance_m:.2f}m, "
                    f"width={live_frontier_choice.width_deg:.1f}deg, score={live_frontier_choice.score:.2f})"
                )
            else:
                print(f"[wander] no live frontier candidate found (frame_id={live_frame_id})")

            map_frontier_choice: FrontierChoice | None = None
            # Legacy ray-based map frontier, superseded by the exploration-grid
            # planner in smart mode.
            if wander_mode != "smart" and len(stitch_state["snapshots"]) >= 2:
                map_frontier_choice = _select_map_frontier_choice(
                    transformed_sets=stitch_state["transformed_sets"],
                    current_pose=current_live_pose,
                    max_distance_m=float(args.max_distance_m),
                    frontier_min_distance_m=float(args.frontier_min_distance_m),
                    frontier_bin_deg=float(args.frontier_bin_deg),
                    resolution_m=float(args.stitch_resolution_m),
                    robot_radius_m=float(args.robot_radius_m),
                )
            if map_frontier_choice is not None:
                print(
                    "[wander] map frontier candidate "
                    f"(delta={map_frontier_choice.delta_deg:.1f}deg, "
                    f"distance={map_frontier_choice.mean_distance_m:.2f}m, "
                    f"width={map_frontier_choice.width_deg:.1f}deg, score={map_frontier_choice.score:.2f}, "
                    f"pose=({float(current_live_pose.x):.3f}, {float(current_live_pose.y):.3f}, {float(current_live_pose.theta_deg):.1f}deg))"
                )
            elif wander_mode != "smart":
                print(
                    "[wander] no stitched-map frontier candidate "
                    f"(captures={len(stitch_state['snapshots'])}, pose=({float(current_live_pose.x):.3f}, "
                    f"{float(current_live_pose.y):.3f}, {float(current_live_pose.theta_deg):.1f}deg))"
                )
            if map_frontier_choice is not None and float(map_frontier_choice.score) < 2.0:
                # Real frontiers score well above this (clearance + exit
                # bonuses). A weak/negative score is just the least-bad ray —
                # chasing one is how the robot walked into a dead-end pocket.
                print(
                    "[wander] ignoring weak map frontier "
                    f"(score={float(map_frontier_choice.score):.2f} < 2.0)"
                )
                map_frontier_choice = None

            # Nothing worth mapping from HERE: no reachable map frontier and no
            # active target. Before concluding the whole job is done, deliberately
            # relocate to a vantage the robot has not captured from yet (e.g. the
            # opposite corner) — milling around a finished corner wastes captures.
            # Legacy idle/survey/completion, superseded by the frontier
            # planner's own completion check in smart mode.
            idle_here = (
                wander_mode != "smart"
                and rotation_coverage_complete
                and len(stitch_state["snapshots"]) >= 8
                and map_frontier_choice is None
                and active_explore_target is None
            )
            if idle_here and survey_targets_used < 3:
                survey_target_xy = _select_survey_target(
                    transformed_sets=stitch_state["transformed_sets"],
                    current_pose=current_live_pose,
                    capture_poses=list(stitch_state["poses"]),
                    resolution_m=float(args.stitch_resolution_m),
                    robot_radius_m=float(args.robot_radius_m),
                    max_distance_m=float(args.max_distance_m),
                )
                if survey_target_xy is not None and not _near_unreachable_target(*survey_target_xy):
                    survey_targets_used += 1
                    active_explore_target = ExploreTarget(
                        world_x_m=float(survey_target_xy[0]),
                        world_y_m=float(survey_target_xy[1]),
                        source="survey",
                        seeded_capture_index=int(capture_index),
                    )
                    active_explore_target_stall_count = 0
                    active_explore_target_last_distance_m = None
                    active_explore_target_blocked_count = 0
                    force_live_frontier_cycles = 0
                    idle_here = False
                    print(
                        "[wander] nothing left to map from here; relocating to survey vantage "
                        f"({survey_target_xy[0]:.3f}, {survey_target_xy[1]:.3f}) "
                        f"(survey {survey_targets_used}/3)"
                    )
            # Mission completion (legacy modes only): in smart mode the frontier
            # planner owns no_frontier_cycles — resetting it here every cycle
            # made the smart completion threshold unreachable (infinite
            # verification scans).
            if wander_mode != "smart":
                if idle_here:
                    no_frontier_cycles += 1
                else:
                    no_frontier_cycles = 0
            if wander_mode != "smart" and no_frontier_cycles >= 6:
                print(
                    "[wander] mapping complete: rotation coverage done and no reachable "
                    f"unmapped frontiers remain after {no_frontier_cycles} consecutive checks; stopping"
                )
                break

            target_frontier_choice: FrontierChoice | None = None
            if active_explore_target is not None:
                current_target_distance_m = _target_distance_m(active_explore_target, current_live_pose)
                if current_target_distance_m <= float(args.frontier_goal_reached_m):
                    print(
                        "[wander] stitched-map exploration target reached "
                        f"(distance={current_target_distance_m:.2f}m, source={active_explore_target.source}, "
                        f"seed_capture={active_explore_target.seeded_capture_index})"
                    )
                    active_explore_target = None
                    active_explore_target_stall_count = 0
                    active_explore_target_last_distance_m = None
                    active_explore_target_blocked_count = 0
                    force_live_frontier_cycles = 0
                else:
                    target_frontier_choice = _target_choice_from_world_point(
                        target_x_m=float(active_explore_target.world_x_m),
                        target_y_m=float(active_explore_target.world_y_m),
                        current_pose=current_live_pose,
                        source="target",
                    )
                    if target_frontier_choice is not None:
                        print(
                            "[wander] active stitched-map target "
                            f"(delta={target_frontier_choice.delta_deg:.1f}deg, "
                            f"distance={target_frontier_choice.mean_distance_m:.2f}m, "
                            f"world=({float(active_explore_target.world_x_m):.3f}, "
                            f"{float(active_explore_target.world_y_m):.3f}))"
                        )

            if (
                active_explore_target is None
                and wander_mode == "smart"
                and rotation_coverage_complete
                and map_frontier_choice is not None
            ):
                candidate_explore_target = _seed_explore_target_from_frontier(
                    frontier_choice=map_frontier_choice,
                    current_pose=current_live_pose,
                    capture_index=capture_index,
                    step_distance_m=float(args.frontier_goal_step_m),
                )
                if _near_unreachable_target(
                    candidate_explore_target.world_x_m, candidate_explore_target.world_y_m
                ):
                    print(
                        "[wander] skipping frontier near a known-unreachable target "
                        f"(world=({candidate_explore_target.world_x_m:.3f}, "
                        f"{candidate_explore_target.world_y_m:.3f}))"
                    )
                    candidate_explore_target = None
                active_explore_target = candidate_explore_target
                explore_target_just_seeded = candidate_explore_target is not None
            else:
                explore_target_just_seeded = False
            if explore_target_just_seeded:
                active_explore_target_stall_count = 0
                active_explore_target_last_distance_m = None
                active_explore_target_blocked_count = 0
                force_live_frontier_cycles = 0
                target_frontier_choice = _target_choice_from_world_point(
                    target_x_m=float(active_explore_target.world_x_m),
                    target_y_m=float(active_explore_target.world_y_m),
                    current_pose=current_live_pose,
                    source="target",
                )
                print(
                    "[wander] seeded stitched-map exploration target "
                    f"(source={map_frontier_choice.source}, delta={map_frontier_choice.delta_deg:.1f}deg, "
                    f"distance={map_frontier_choice.mean_distance_m:.2f}m, target_world=("
                    f"{float(active_explore_target.world_x_m):.3f}, {float(active_explore_target.world_y_m):.3f}))"
                )

            # DELIBERATE STEERING (field 2026-07-20): the facing/seek turns steer
            # toward planning_frontier_choice, which in smart mode fell back to the
            # reactive live_frontier_choice — the WIDEST live opening, which flips
            # as the robot's heading changes, so it turned back and forth chasing a
            # moving target (the meandering). When the robot is already committed
            # to a deliberate BFS frontier goal (committed_frontier_face), steer
            # toward THAT stable point instead. The goal is recomputed by the
            # planner when reached/blocked; between those it does not move, so the
            # robot drives to it directly instead of oscillating.
            if (
                wander_mode == "smart"
                and target_frontier_choice is None
                and committed_frontier_face is not None
                and force_live_frontier_cycles <= 0
            ):
                target_frontier_choice = _target_choice_from_world_point(
                    target_x_m=float(committed_frontier_face[0]),
                    target_y_m=float(committed_frontier_face[1]),
                    current_pose=current_live_pose,
                    source="committed",
                )
            escape_frontier_active = (
                wander_mode == "smart"
                and force_live_frontier_cycles > 0
                and live_frontier_choice is not None
            )
            if escape_frontier_active:
                planning_frontier_choice = live_frontier_choice or map_frontier_choice or target_frontier_choice
                planning_frontier_source = "live_escape"
                print(
                    "[wander] escape frontier override active "
                    f"(cycles_left={force_live_frontier_cycles}, "
                    f"live_delta={live_frontier_choice.delta_deg:.1f}deg, "
                    f"live_distance={live_frontier_choice.mean_distance_m:.2f}m)"
                )
            else:
                planning_frontier_choice = target_frontier_choice or map_frontier_choice or live_frontier_choice
                planning_frontier_source = (
                    planning_frontier_choice.source if planning_frontier_choice is not None else "none"
                )
            if planning_frontier_choice is not None:
                print(
                    "[wander] planning frontier "
                    f"(source={planning_frontier_source}, delta={planning_frontier_choice.delta_deg:.1f}deg, "
                    f"distance={planning_frontier_choice.mean_distance_m:.2f}m, score={planning_frontier_choice.score:.2f})"
                )

            if hazard_monitor is not None:
                _stamp_confirmed_edges(current_live_pose)

            if (
                wander_mode == "smart"
                and pending_motion_hint is not None
                and not live_pose_accepted
                and abs(float(pending_motion_hint.expected_dtheta_deg)) > 20.0
            ):
                # The last capture was DISCARDED after a turn: the robot has
                # physically rotated away from the map, but the map never
                # absorbed that rotation. Planning a fresh frontier turn from
                # the stale pose rotates even FURTHER from the only overlap we
                # have, scores collapse toward zero, and a garbage pose ends up
                # force-accepted (observed: score 0.053 poisoned a whole run).
                # Recover by turning BACK toward the last stitched heading,
                # where overlap — and therefore a confident solve — is
                # guaranteed, then capture to re-anchor before exploring on.
                if orbit_recovery_turns_remaining <= 0:
                    if orbit_recovery_turn_attempts >= 1:
                        # The prior recovery turn also produced no trusted
                        # pose. Do not issue another blind 85deg command: it
                        # is the source of the endless cw/ccw-looking loop in
                        # the field run. Leave the base stopped, discard the
                        # untrusted turn hint, and let the next cycle capture
                        # a fresh view before the normal safety-gated drive.
                        print(
                            "[wander] unlocalized-turn recovery budget exhausted; "
                            "stopping rotation and forcing a new-view drive cycle"
                        )
                        pending_motion_hint = None
                        orbit_recovery_direction_sign = None
                        force_drive_after_turn = True
                        continue
                    orbit_recovery_direction_sign = (
                        1.0 if float(pending_motion_hint.expected_dtheta_deg) >= 0.0 else -1.0
                    )
                    orbit_recovery_turns_remaining = 1
                    orbit_recovery_turn_attempts += 1
                orbit_recovery_turns_remaining -= 1
                should_turn = True
                pending_turn_reason = None
                bootstrap_turn_commands = max(
                    bootstrap_turn_commands, max(1, int(args.bootstrap_turn_captures))
                )
                rotation_coverage_complete = True
                bootstrap_scan_active = False
                chosen_direction_sign = float(orbit_recovery_direction_sign)
                chosen_turn_deg = 85.0
                turn_reason = "orbit_recovery"
                print(
                    "[wander] turn capture could not be localized; continuing around the room "
                    f"({orbit_recovery_turns_remaining + 1}/1, 85deg "
                    f"{'ccw' if chosen_direction_sign >= 0.0 else 'cw'}) before retrying that frontier"
                )
                if orbit_recovery_turns_remaining == 0:
                    force_drive_after_turn = True
                if False and committed_frontier_face is not None:
                    # A face-turn toward this frontier just lost tracking and
                    # got discarded: facing it from here PROVABLY fails. Never
                    # retry the identical turn — blacklist the frontier
                    # immediately (this also breaks the goal commitment that
                    # would otherwise re-pick it) and explore elsewhere.
                    _strike_frontier(committed_frontier_face, "turn_lock_lost", force=True)
            elif wander_mode == "smart":
                # ================= frontier exploration =================
                # One rule, no special cases: model what has been observed
                # (free / occupied / unknown), find the nearest reachable
                # boundary to unknown space, travel the traversable path to
                # it, face it, scan. Bootstrap spin, corner-leaving, and
                # completion all emerge from this same computation.
                should_turn = False
                pending_turn_reason = None
                turn_reason = "none"
                elevated_extra_xy = (
                    np.asarray(
                        [[c[0] * 0.04, c[1] * 0.04] for c in elevated_occupied_world],
                        dtype=np.float32,
                    )
                    if elevated_occupied_world
                    else None
                )
                # SQUEEZE variant of the elevated obstacles: the squeeze/doorway
                # plans dilate occupancy by only the tight squeeze radius (0.22m)
                # so the body fits a real LiDAR doorframe. But camera-detected
                # TABLE/FURNITURE edges must NOT get that tight margin — the true
                # body (0.229m) plus wider shoulders/arms clip them, and the
                # depth gate is imperfect on thin corners (field 2026-07-20: "it
                # keeps getting caught on table edges"). Pre-pad each elevated
                # cell by ~0.08m so that AFTER the squeeze plan's 0.22m dilation
                # the effective clearance around furniture is ~0.30m (full body +
                # shoulder margin), while LiDAR walls keep the tight radius.
                _elev_pad_cells = 2  # cells at 0.04m -> ~0.08m extra
                _elev_pad_offsets = [
                    (dx, dy)
                    for dx in range(-_elev_pad_cells, _elev_pad_cells + 1)
                    for dy in range(-_elev_pad_cells, _elev_pad_cells + 1)
                    if dx * dx + dy * dy <= _elev_pad_cells * _elev_pad_cells
                ]
                elevated_extra_xy_squeeze = None
                if elevated_occupied_world:
                    _padded_cells = {
                        (cx + dx, cy + dy)
                        for (cx, cy) in elevated_occupied_world
                        for dx, dy in _elev_pad_offsets
                    }
                    elevated_extra_xy_squeeze = np.asarray(
                        [[c[0] * 0.04, c[1] * 0.04] for c in _padded_cells],
                        dtype=np.float32,
                    )
                exploration_grid = _build_exploration_grid(
                    poses=list(stitch_state["poses"]),
                    transformed_sets=list(stitch_state["transformed_sets"]),
                    resolution_m=max(0.05, float(args.stitch_resolution_m)),
                    robot_clear_radius_m=float(args.robot_radius_m) + 0.06,
                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                    extra_occupied_xy=elevated_extra_xy,
                )
                frontier_plan: dict[str, object] = {"status": "no_frontier"}
                if exploration_grid is not None:
                    frontier_plan = _plan_frontier_path(
                        grid=exploration_grid,
                        robot_xy=(float(current_live_pose.x), float(current_live_pose.y)),
                        robot_radius_m=float(args.robot_radius_m),
                        target_xy=active_survey_xy,
                        observed_from_xy=[
                            (float(p.x), float(p.y)) for p in stitch_state["poses"]
                        ],
                        robot_theta_deg=float(current_live_pose.theta_deg),
                        avoid_face_xy=_active_frontier_blacklist(),
                        prefer_face_xy=committed_frontier_face,
                    )
                if (
                    str(frontier_plan.get("status")) == "no_frontier"
                    and active_survey_xy is None
                    and not pose_lost
                ):
                    # EDGE-SURVEY phase (user strategy 2026-07-18): the
                    # low-hanging exploration is done — before squeezing or
                    # verification-driving through YELLOW (single-view)
                    # cells, deliberately arc to a second angle on each
                    # yellow cluster. Real edges re-detect at the same
                    # world position and turn red with solid dimensions;
                    # bleed fails to reappear and decays. Only after the
                    # yellows are resolved does the ladder push into new
                    # areas through what remains.
                    edge_survey_plan = _plan_edge_survey(exploration_grid, current_live_pose)
                    if edge_survey_plan is not None:
                        frontier_plan = edge_survey_plan
                if (
                    str(frontier_plan.get("status")) == "observe"
                    and int(frontier_plan.get("frontier_cells", 0) or 0) >= 12
                    and active_survey_xy is None
                    and not pose_lost
                    and exploration_grid is not None
                ):
                    # OBSERVE -> SQUEEZE-THROUGH: the best plan is to peer at a
                    # SUBSTANTIAL unreachable region from afar (the classic
                    # doorway pinched shut at comfortable margins). The user's
                    # read is correct — the robot "thinks it's too fat" at the
                    # 0.29m planning radius (0.229m body + 0.06m margin) and
                    # sits re-observing the exit instead of going through it.
                    # Before settling for a look-from-afar, try to actually
                    # REACH the region at squeeze margins (radius - 5cm ~=
                    # 0.24m, ~1cm/side clearance over the true body). Only
                    # drive through if it becomes REACHABLE (status ok); if it
                    # is still merely observable, it is a real wall — keep the
                    # observe. The live measured-gap gate owns the traversal.
                    squeeze_grid = _build_exploration_grid(
                        poses=list(stitch_state["poses"]),
                        transformed_sets=list(stitch_state["transformed_sets"]),
                        resolution_m=max(0.05, float(args.stitch_resolution_m)),
                        robot_clear_radius_m=float(args.robot_radius_m),
                        lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                        extra_occupied_xy=elevated_extra_xy_squeeze,
                    )
                    if squeeze_grid is not None:
                        squeeze_plan = _plan_frontier_path(
                            grid=squeeze_grid,
                            robot_xy=(float(current_live_pose.x), float(current_live_pose.y)),
                            robot_radius_m=float(args.squeeze_radius_m),
                            observed_from_xy=[
                                (float(p.x), float(p.y)) for p in stitch_state["poses"]
                            ],
                            robot_theta_deg=float(current_live_pose.theta_deg),
                            avoid_face_xy=_active_frontier_blacklist(),
                            prefer_face_xy=committed_frontier_face,
                        )
                        if str(squeeze_plan.get("status")) == "ok":
                            print(
                                "[wander] OBSERVE->SQUEEZE: the unmapped region is only observable "
                                "at comfortable margins but REACHABLE at squeeze width — driving "
                                "through the gap instead of staring "
                                f"(path={float(squeeze_plan.get('path_length_m', 0.0) or 0.0):.2f}m); "
                                "the measured-gap gate owns the traversal"
                            )
                            squeeze_plan["squeeze"] = True
                            frontier_plan = squeeze_plan
                if (
                    str(frontier_plan.get("status")) == "no_frontier"
                    and int(frontier_plan.get("unreachable_frontier_cells", 0) or 0) >= 24
                    and active_survey_xy is None
                ):
                    # SQUEEZE REPLAN: unknown space exists but every path to
                    # it is pinched shut at COMFORTABLE margins (radius +
                    # 6cm grid carve + radius inflation). The live safety
                    # stack is better informed than this grid — the
                    # measured-gap squeeze creeps through anything the body
                    # actually fits (field run 15: a physically passable
                    # doorway corridor, robot photographed fitting with
                    # margin, was sealed by a table edge-bleed blob +
                    # inflation and the mission stalled at "no_frontier").
                    # Retry the plan at squeeze margins; the gates own the
                    # actual traversal.
                    tight_grid = _build_exploration_grid(
                        poses=list(stitch_state["poses"]),
                        transformed_sets=list(stitch_state["transformed_sets"]),
                        resolution_m=max(0.05, float(args.stitch_resolution_m)),
                        robot_clear_radius_m=float(args.robot_radius_m),
                        lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                        extra_occupied_xy=elevated_extra_xy_squeeze,
                    )
                    if tight_grid is not None:
                        tight_plan = _plan_frontier_path(
                            grid=tight_grid,
                            robot_xy=(float(current_live_pose.x), float(current_live_pose.y)),
                            robot_radius_m=float(args.squeeze_radius_m),
                            observed_from_xy=[
                                (float(p.x), float(p.y)) for p in stitch_state["poses"]
                            ],
                            robot_theta_deg=float(current_live_pose.theta_deg),
                            avoid_face_xy=_active_frontier_blacklist(),
                            prefer_face_xy=committed_frontier_face,
                        )
                        if str(tight_plan.get("status")) in ("ok", "observe"):
                            print(
                                "[wander] SQUEEZE replan: frontier unreachable at comfortable "
                                "margins but passable at squeeze width "
                                f"(status={tight_plan.get('status')}, "
                                f"path={float(tight_plan.get('path_length_m', 0.0) or 0.0):.2f}m); "
                                "proceeding — the measured-gap gate owns the traversal"
                            )
                            tight_plan["squeeze"] = True
                            frontier_plan = tight_plan
                        elif elevated_occupied_world:
                            # VERIFICATION replan: even squeeze margins can't
                            # thread the pinch — eye cells sit IN the
                            # corridor. Eye cells are claims, not measured
                            # lidar geometry; plan WITHOUT them (real walls
                            # still enforced) and approach under full gate
                            # protection. Real furniture stops the robot and
                            # re-confirms the cells; phantom cells get
                            # re-inspected, miss, and decay away. Either way
                            # the ambiguity resolves instead of declaring
                            # completion behind a possibly-false wall.
                            verify_grid = _build_exploration_grid(
                                poses=list(stitch_state["poses"]),
                                transformed_sets=list(stitch_state["transformed_sets"]),
                                resolution_m=max(0.05, float(args.stitch_resolution_m)),
                                robot_clear_radius_m=float(args.robot_radius_m) + 0.06,
                                lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                                extra_occupied_xy=None,
                            )
                            if verify_grid is not None:
                                verify_plan = _plan_frontier_path(
                                    grid=verify_grid,
                                    robot_xy=(
                                        float(current_live_pose.x),
                                        float(current_live_pose.y),
                                    ),
                                    robot_radius_m=float(args.robot_radius_m),
                                    observed_from_xy=[
                                        (float(p.x), float(p.y)) for p in stitch_state["poses"]
                                    ],
                                    robot_theta_deg=float(current_live_pose.theta_deg),
                                    avoid_face_xy=_active_frontier_blacklist(),
                                    prefer_face_xy=committed_frontier_face,
                                )
                                if str(verify_plan.get("status")) in ("ok", "observe"):
                                    print(
                                        "[wander] VERIFICATION replan: the only route to unmapped "
                                        "space crosses EYE-mapped cells (lidar geometry allows it); "
                                        "approaching under full gates to confirm or decay them "
                                        f"(status={verify_plan.get('status')}, "
                                        f"path={float(verify_plan.get('path_length_m', 0.0) or 0.0):.2f}m)"
                                    )
                                    # Verification approaches are squeeze-class
                                    # traversals: the narrow stop box and the
                                    # squeeze-envelope side guard apply, or the
                                    # guard zero-halts the ordered approach.
                                    verify_plan["squeeze"] = True
                                    frontier_plan = verify_plan
                if (
                    active_survey_xy is None
                    and not exit_mode
                    and str(frontier_plan.get("status")) == "no_frontier"
                    and survey_targets_used < 2
                    and exploration_grid is not None
                ):
                    # not exit_mode: survey vantages are INTERIOR points by
                    # construction. Field 2026-07-18 run 19: the doorway
                    # candidate flickered out for ONE cycle (occluded 82-point
                    # view) and this branch sent the robot on a 174deg
                    # about-face back to a vantage by the table — while it was
                    # FACING the exit. During the exit run the fallback for a
                    # missing opening is the exit_scan rotation (re-acquire the
                    # doorway), never an interior trip.
                    # No information left to gain from HERE. Before declaring
                    # the map finished, relocate to an unvisited vantage and
                    # re-check — captures from across the room fill occlusion
                    # shadows that are unreachable-unknown from this spot. The
                    # vantage comes from the SAME BFS grid used for navigation,
                    # so it is pathable by construction.
                    survey_probe = _plan_frontier_path(
                        grid=exploration_grid,
                        robot_xy=(float(current_live_pose.x), float(current_live_pose.y)),
                        robot_radius_m=float(args.robot_radius_m),
                        survey_from_xy=[
                            (float(p.x), float(p.y)) for p in stitch_state["poses"]
                        ],
                        robot_theta_deg=float(current_live_pose.theta_deg),
                    )
                    if str(survey_probe.get("status")) == "survey":
                        active_survey_xy = (
                            float(survey_probe["goal_xy"][0]),
                            float(survey_probe["goal_xy"][1]),
                        )
                        survey_targets_used += 1
                        print(
                            "[wander] no frontier from this vantage; relocating to survey vantage "
                            f"({active_survey_xy[0]:.2f}, {active_survey_xy[1]:.2f}) "
                            f"(spacing={float(survey_probe.get('survey_spacing_m', 0.0)):.2f}m from "
                            f"previous captures, survey {survey_targets_used}/2)"
                        )
                        frontier_plan = survey_probe
                    else:
                        print(
                            "[wander] no unvisited survey vantage either "
                            f"(best spacing {float(survey_probe.get('best_survey_spacing_m', 0.0)):.2f}m "
                            "< 0.80m from previous captures)"
                        )
                if str(frontier_plan.get("status")) == "target_unreachable":
                    print("[wander] survey vantage unreachable; abandoning it")
                    active_survey_xy = None
                    frontier_plan = {"status": "no_frontier"}
                plan_status = str(frontier_plan.get("status"))
                squeeze_traversal_active = bool(frontier_plan.get("squeeze", False))
                # ---- EXIT RUN ---------------------------------------------
                # Sliver fatigue: count consecutive plans that offer no REAL new
                # area (tiny cell counts are grid noise/phantoms, not rooms).
                plan_cells = int(frontier_plan.get("frontier_cells", 0) or 0)
                # Explore-stall detector (robust to WHICH frontier it chases): is
                # the robot still reaching new ground, or milling in one patch?
                _robot_xy = (float(current_live_pose.x), float(current_live_pose.y))
                if exit_stall_pos is not None and math.hypot(
                    _robot_xy[0] - exit_stall_pos[0], _robot_xy[1] - exit_stall_pos[1]
                ) < EXIT_STALL_RADIUS_M:
                    exit_stall_cycles += 1
                else:
                    exit_stall_pos = _robot_xy   # moved to fresh ground -> reset
                    exit_stall_cycles = 0
                    stuck_scan_rotations = 0
                    stuck_reverse_count = 0
                _explore_stalled = exit_stall_cycles >= EXIT_STALL_CYCLE_LIMIT
                if not bootstrap_scan_active and len(stitch_state["poses"]) >= 5:
                    # A wide, DEEP opening the robot has not driven to yet is real
                    # unexplored space — field 2026-07-20: the robot spun near its
                    # start, the BFS returned only a handful of frontier cells (the
                    # MAP was still tiny, not the ROOM explored), it hit plan_cells<=10,
                    # counted 4 "sliver" cycles and declared ROOM MAPPED after barely
                    # translating, then rotated forever in the exit run. A low map
                    # frontier count with a big live opening present means "keep
                    # exploring," NOT "done." Do not count completion while such an
                    # opening exists — unless the robot has genuinely milled in place
                    # unable to reach it (the explore-stall backstop still ends it).
                    strong_live_opening = (
                        live_frontier_choice is not None
                        and float(live_frontier_choice.mean_distance_m) >= 1.5
                        and float(live_frontier_choice.width_deg) >= 48.0
                    )
                    if plan_status == "no_frontier" or (
                        plan_status in ("ok", "observe", "survey")
                        and (plan_cells <= 10 or _explore_stalled)
                    ):
                        if strong_live_opening and not _explore_stalled:
                            if sliver_frontier_cycles != 0:
                                sliver_frontier_cycles = 0
                            print(
                                "[wander] NOT declaring the room mapped: a wide/deep live opening "
                                f"(delta={float(live_frontier_choice.delta_deg):+.0f}deg, "
                                f"{float(live_frontier_choice.mean_distance_m):.1f}m, "
                                f"{float(live_frontier_choice.width_deg):.0f}deg wide) is still "
                                "unexplored — keep exploring toward it, not quitting"
                            )
                        else:
                            if _explore_stalled and plan_cells > 10:
                                print(
                                    f"[wander] explore STALL: milled within {EXIT_STALL_RADIUS_M:.1f}m for "
                                    f"{exit_stall_cycles} cycles without reaching new ground — the "
                                    "reachable room is mapped; treating this frontier as EXHAUSTED so "
                                    "the exit run can trigger"
                                )
                            sliver_frontier_cycles += 1
                    else:
                        sliver_frontier_cycles = 0
                        if exit_mode and plan_cells >= 30:
                            # New area is visible through the committed opening:
                            # the robot is AT the doorway looking out. Arm the
                            # crossing flag (positional completion is POSITIONAL,
                            # not visual — cells open the instant it merely LOOKS
                            # through; field run 13 declared "complete" while
                            # standing inside). Actual completion is the centroid
                            # test that runs every cycle below.
                            exit_opening_seen = True
                            print(
                                "[wander] EXIT RUN: new area visible through the opening "
                                f"({plan_cells} cells) — at the doorway, driving OUT (through "
                                f"when >={exit_room_radius_m + EXIT_BEYOND_ROOM_M:.2f}m from room "
                                "center)"
                            )
                # MEASURED-ODOMETRY completion (checked every cycle, independent
                # of the frontier-cell path above): the robot has driven
                # EXIT_THROUGH_ODOMETRY_M of real forward motion through the
                # committed opening while its map pose was lost at the threshold
                # — physically out of the room even though the map can't fix the
                # pose there. Field run 23 stalled here forever because the only
                # completion test read the frozen (lost) pose and saw "0.00m".
                if (
                    exit_mode
                    and exit_opening_seen
                    and exit_through_odometry_m >= EXIT_THROUGH_ODOMETRY_M
                ):
                    exit_mode = False
                    print(
                        "[wander] EXIT RUN complete: drove "
                        f"{exit_through_odometry_m:.2f}m of MEASURED forward motion through the "
                        "committed opening while the map pose was lost at the threshold (expected "
                        "when leaving the mapped room — measured odometry, not a map fix) — "
                        "THROUGH the exit; resuming mapping in the new space"
                    )
                    exit_latch_poses = []
                    exit_latch_map_cells = set()
                    exit_through_odometry_m = 0.0
                    exit_opening_seen = False
                    exit_centroid_xy = None
                    exit_room_radius_m = 0.0
                    exit_outward_max_m = 0.0
                    sliver_frontier_cycles = 0
                # CENTROID completion + outward ratchet update (runs every exit
                # cycle, healthy-pose path). Distance from the room center grows
                # as the robot heads out any door; when it clears the mapped
                # footprint by EXIT_BEYOND_ROOM_M the robot is physically out.
                if exit_mode and exit_centroid_xy is not None:
                    _dist_centroid = math.hypot(
                        float(current_live_pose.x) - exit_centroid_xy[0],
                        float(current_live_pose.y) - exit_centroid_xy[1],
                    )
                    if _dist_centroid > exit_outward_max_m:
                        exit_outward_max_m = _dist_centroid
                    if _dist_centroid >= exit_room_radius_m + EXIT_BEYOND_ROOM_M:
                        # NEW-SPACE gate — the distance alone is not enough (driving
                        # to a far wall in the same room fakes it). Measure how much
                        # of the CURRENT scan still lands on the frozen room map: if
                        # most of it does, the robot is still looking at the same
                        # room (at a wall), NOT out a door.
                        _room_overlap = 1.0
                        if exit_latch_map_cells and len(live_points_xy):
                            _world = _transform_points(live_points_xy, current_live_pose)
                            _scan_cells = np.round(
                                np.asarray(_world, dtype=np.float64) / EXIT_ROOM_CELL_M
                            ).astype(np.int64)
                            _hits = 0
                            for _sx, _sy in _scan_cells.tolist():
                                _matched = False
                                for _ddx in (-1, 0, 1):
                                    for _ddy in (-1, 0, 1):
                                        if (int(_sx) + _ddx, int(_sy) + _ddy) in exit_latch_map_cells:
                                            _matched = True
                                            break
                                    if _matched:
                                        break
                                if _matched:
                                    _hits += 1
                            _room_overlap = _hits / len(_scan_cells)
                        if _room_overlap <= EXIT_NEW_SPACE_MAX_OVERLAP:
                            exit_mode = False
                            print(
                                "[wander] EXIT RUN complete: robot is "
                                f"{_dist_centroid:.2f}m from room center AND only "
                                f"{_room_overlap * 100:.0f}% of the scan still matches the mapped "
                                "room — genuinely in NEW space, THROUGH the exit; resuming mapping"
                            )
                            exit_latch_poses = []
                            exit_latch_map_cells = set()
                            exit_through_odometry_m = 0.0
                            exit_opening_seen = False
                            exit_centroid_xy = None
                            exit_room_radius_m = 0.0
                            exit_outward_max_m = 0.0
                            sliver_frontier_cycles = 0
                        else:
                            print(
                                "[wander] EXIT RUN: "
                                f"{_dist_centroid:.2f}m from room center but "
                                f"{_room_overlap * 100:.0f}% of the scan STILL matches the mapped "
                                "room — driving to a WALL inside it, not out a door; NOT through, "
                                "keep looking for the real opening"
                            )
                if not exit_mode and sliver_frontier_cycles >= 4:
                    exit_mode = True
                    exit_latch_poses = [
                        (float(p.x), float(p.y)) for p in stitch_state["poses"]
                    ]
                    # Interior-seeking subsystems stand down for the exit run:
                    # any standing survey relocation target is an interior
                    # point and must not outlive the latch.
                    active_survey_xy = None
                    # Fresh crossing odometry for this exit attempt.
                    exit_through_odometry_m = 0.0
                    exit_opening_seen = False
                    # Room center + footprint radius from the latch vantages, and
                    # the outward ratchet seeded at the robot's current distance
                    # from that center.
                    _cx = sum(lx for lx, _ in exit_latch_poses) / len(exit_latch_poses)
                    _cy = sum(ly for _, ly in exit_latch_poses) / len(exit_latch_poses)
                    exit_centroid_xy = (_cx, _cy)
                    exit_room_radius_m = max(
                        math.hypot(lx - _cx, ly - _cy) for lx, ly in exit_latch_poses
                    )
                    exit_outward_max_m = math.hypot(
                        float(current_live_pose.x) - _cx,
                        float(current_live_pose.y) - _cy,
                    )
                    # Freeze the room-so-far as a cell-set. Completion later
                    # requires the live scan to stop matching THIS (new space),
                    # which driving to a far wall inside the same room cannot fake.
                    exit_latch_map_cells = set()
                    for _room_pts in stitch_state["transformed_sets"]:
                        if len(_room_pts):
                            _room_cells = np.round(
                                np.asarray(_room_pts, dtype=np.float64) / EXIT_ROOM_CELL_M
                            ).astype(np.int64)
                            exit_latch_map_cells.update(map(tuple, _room_cells.tolist()))
                    print(
                        "[wander] ROOM MAPPED (4 straight cycles with only sliver/no "
                        "frontiers) — EXIT RUN: interior goals are DONE; committing to the "
                        "deepest opening the live lidar sees and driving THROUGH it "
                        f"(completion = reach {exit_room_radius_m + EXIT_BEYOND_ROOM_M:.2f}m from "
                        f"room center ({_cx:.2f}, {_cy:.2f}); outward ratchet armed)"
                    )
                if exit_mode and live_frontier_choice is not None:
                    pass
                # LEAVE ALREADY-MAPPED AREAS ALONE (field 2026-07-20 user: "if it
                # already mapped an area, leave it the fuck alone"). Once the room
                # is mapped (EXIT RUN latched), an interior plan — a look-from-afar
                # OBSERVE/SURVEY at an already-seen region, or a tiny reachable
                # sliver — is pure meandering: the robot turns to re-scan ground it
                # already has. Suppress ALL of them so the exit run only ever (a)
                # probes the live opening, or (b) does the BOUNDED exit-scan
                # rotation to find the door, then completes. exit_scan_turns is NO
                # LONGER reset on a merely-visible opening (that reset let a
                # flickering opening refill the budget forever); it now counts
                # monotonically toward the 7-turn cap → guaranteed termination, and
                # a successful forward probe refills it (real progress earns more).
                if (
                    exit_mode
                    and plan_status in ("ok", "observe", "survey")
                    and (
                        plan_status in ("observe", "survey")
                        or plan_cells <= 10
                        or _explore_stalled
                    )
                ):
                    if live_frontier_choice is not None:
                        print(
                            f"[wander] EXIT RUN: ignoring the interior {plan_status} "
                            "frontier; heading for the live exit opening "
                            f"(delta={float(live_frontier_choice.delta_deg):.1f}deg, "
                            f"distance={float(live_frontier_choice.mean_distance_m):.2f}m)"
                        )
                    else:
                        print(
                            "[wander] EXIT RUN: not re-scanning already-mapped ground; "
                            "rotating to search for the exit instead"
                        )
                    committed_frontier_face = None
                    frontier_plan = {"status": "no_frontier"}
                    plan_status = "no_frontier"
                # NO-REGRESSION INVARIANT (centroid ratchet): while the exit run
                # is latched, no adopted plan may target a point CLOSER to the
                # room center than the outward-max the robot has already reached.
                # Field run 21: a verification replan marched the robot back to
                # interior goals 4cm from completion. Field run 24: the OLD
                # nearest-vantage version of this test SILENTLY DISABLED itself
                # at the door (its >=0.30m clearance floor — with 11 dense
                # vantages the robot is ~0.05m from one everywhere near the
                # doorway), so nothing stopped the retreat. The centroid metric
                # has no such floor and cannot be fooled by a nearby vantage.
                if (
                    exit_mode
                    and exit_centroid_xy is not None
                    and plan_status in ("ok", "observe", "survey")
                ):
                    _goal_xy = frontier_plan.get("goal_xy")
                    if _goal_xy is not None:
                        _goal_from_center = math.hypot(
                            float(_goal_xy[0]) - exit_centroid_xy[0],
                            float(_goal_xy[1]) - exit_centroid_xy[1],
                        )
                        if _goal_from_center < exit_outward_max_m - EXIT_RATCHET_SLACK_M:
                            print(
                                "[wander] EXIT RUN: plan goal "
                                f"({float(_goal_xy[0]):.2f}, {float(_goal_xy[1]):.2f}) would "
                                f"RETREAT toward the room center (goal {_goal_from_center:.2f}m "
                                f"from center < outward-max {exit_outward_max_m:.2f}m - "
                                f"{EXIT_RATCHET_SLACK_M:.2f}m) — vetoed; holding outward pressure "
                                "at the opening"
                            )
                            committed_frontier_face = None
                            frontier_plan = {"status": "no_frontier"}
                            plan_status = "no_frontier"
                # -----------------------------------------------------------
                # Squeeze/exit traversals get the NARROW stop-box band for the
                # drives executed this cycle (doorway frame posts the body
                # clears must not veto the pass); everything else runs the full
                # box. The value persists on zone_cfg into the next cycle's
                # early blocked check, which is correct for multi-cycle passes.
                zone_cfg.squeeze_active = bool(exit_mode or squeeze_traversal_active)
                if plan_status != "no_frontier":
                    # A real plan supersedes any half-finished probe episode.
                    probe_commit_world_deg = None
                    probe_commit_align_turns = 0
                if plan_status in ("ok", "survey", "observe"):
                    committed_frontier_face = (
                        float(frontier_plan["face_xy"][0]),
                        float(frontier_plan["face_xy"][1]),
                    )
                    # Anti-fixation: count consecutive pose-lost cycles that keep
                    # re-targeting this same face. A healthy cycle (localized)
                    # resets it; a run of lost cycles on one phantom frontier
                    # blacklists it so the planner is forced to the rest of the room.
                    if pose_lost:
                        if stuck_frontier_face is not None and math.hypot(
                            committed_frontier_face[0] - stuck_frontier_face[0],
                            committed_frontier_face[1] - stuck_frontier_face[1],
                        ) <= 0.4:
                            stuck_frontier_lost_cycles += 1
                        else:
                            stuck_frontier_face = committed_frontier_face
                            stuck_frontier_lost_cycles = 1
                    else:
                        stuck_frontier_face = None
                        stuck_frontier_lost_cycles = 0
                    print(
                        "[wander] frontier plan "
                        f"(status={plan_status}, "
                        f"goal=({frontier_plan['goal_xy'][0]:.2f}, {frontier_plan['goal_xy'][1]:.2f}), "
                        f"waypoint=({frontier_plan['waypoint_xy'][0]:.2f}, {frontier_plan['waypoint_xy'][1]:.2f}), "
                        f"face=({frontier_plan['face_xy'][0]:.2f}, {frontier_plan['face_xy'][1]:.2f}), "
                        f"path={float(frontier_plan['path_length_m']):.2f}m, "
                        f"frontier_cells={int(frontier_plan['frontier_cells'])})"
                    )
                    if stuck_frontier_lost_cycles >= 3:
                        print(
                            "[wander] ABANDONING frontier "
                            f"({committed_frontier_face[0]:.2f}, {committed_frontier_face[1]:.2f}): "
                            f"targeted it for {stuck_frontier_lost_cycles} straight cycles while "
                            "pose-lost without ever localizing there — blacklisting it and "
                            "exploring the rest of the room instead"
                        )
                        _strike_frontier(
                            committed_frontier_face,
                            "repeated pose-loss targeting this frontier",
                            force=True,
                        )
                        stuck_frontier_face = None
                        stuck_frontier_lost_cycles = 0
                else:
                    committed_frontier_face = None
                    if plan_status == "no_frontier" and "unknown_cells" in frontier_plan:
                        unreachable_centroid = frontier_plan.get("unreachable_centroid_xy")
                        centroid_txt = (
                            f", unreachable_centroid=({unreachable_centroid[0]:.2f}, "
                            f"{unreachable_centroid[1]:.2f})"
                            if unreachable_centroid
                            else ""
                        )
                        print(
                            "[wander] frontier plan (status=no_frontier, "
                            f"unknown={int(frontier_plan['unknown_cells'])}, "
                            f"reachable_frontier={int(frontier_plan['reachable_frontier_cells'])}, "
                            f"unreachable_frontier={int(frontier_plan['unreachable_frontier_cells'])}"
                            f"{centroid_txt})"
                        )
                    else:
                        print(f"[wander] frontier plan (status={plan_status})")

                # Survey arrival is a STATIONARY action (scan in place), so it
                # must be decided before blocked handling — a vantage near
                # clutter can trip the stop box while the robot stands exactly
                # on the goal, and the blocked branch would spin it forever
                # without ever clearing the survey target.
                survey_arrived = False
                if active_survey_xy is not None and plan_status in ("ok", "survey"):
                    survey_goal_distance_m = math.hypot(
                        float(frontier_plan["goal_xy"][0]) - float(current_live_pose.x),
                        float(frontier_plan["goal_xy"][1]) - float(current_live_pose.y),
                    )
                    survey_arrived = survey_goal_distance_m <= max(
                        0.45, float(args.frontier_goal_reached_m)
                    )
                # Is forward genuinely undrivable from THIS heading right now (stop
                # box occupied OR the camera gate denying)? Only then does the
                # anti-stuck watchdog take over — the instant a scan-turn faces a
                # drivable opening this is False and control falls through to the
                # normal planner, which drives it (so the watchdog can never trap a
                # robot that has somewhere to go).
                _fwd_ok_now, _fwd_denial_now = (
                    hazard_monitor.forward_allowed()
                    if hazard_monitor is not None
                    else (True, "")
                )
                _watchdog_forward_blocked = bool(blocked) or not bool(_fwd_ok_now)
                # STAMP camera-confirmed furniture the instant it blocks forward, in
                # ANY branch/cycle (the stuck-watchdog reverse-escape pre-empts the
                # planning-denial recovery, so stamping only there would miss the
                # desk-in-the-doorway case). Permanent no-go -> the opening picker
                # re-plans around it next cycle instead of re-committing.
                if (
                    not bool(_fwd_ok_now)
                    and hazard_monitor is not None
                    and not bool(hazard_monitor.state().frames_stale)
                    and _denial_is_camera_obstacle(_fwd_denial_now)
                ):
                    _stamp_camera_block(current_live_pose)
                if (
                    not bootstrap_scan_active
                    and exit_stall_cycles >= EXIT_STALL_ESCAPE_LIMIT
                    and _watchdog_forward_blocked
                ):
                    # STUCK and cannot drive forward from this heading. Escape ladder:
                    # (1) reverse OUT the way it came (it drove IN, so back is open);
                    # (2) if it can't reverse, SCAN AROUND — turn to look for a heading
                    # the body CAN drive, up to a full revolution — before giving up;
                    # (3) only halt after a full circle finds nothing drivable and no
                    # room to reverse = genuinely walled in. This pre-empts the other
                    # rotate paths so it can't spin forever, but never quits while a
                    # drivable direction it hasn't faced yet might exist.
                    stuck_reverse = _attempt_reverse_escape_hint(current_live_pose, False)
                    if stuck_reverse is not None and stuck_reverse_count < STUCK_REVERSE_CAP:
                        stuck_reverse_count += 1
                        print(
                            f"[wander] STUCK ({exit_stall_cycles} cycles pinned): reversing OUT "
                            f"the way it came ({stuck_reverse_count}/{STUCK_REVERSE_CAP}) instead of "
                            "rotating again"
                        )
                        motion_hint = stuck_reverse
                        settle_s = float(args.move_settle_s)
                    elif (
                        stuck_scan_rotations < STUCK_SCAN_ROTATIONS_HALT
                        and exit_stall_cycles < EXIT_STALL_STOP_LIMIT
                    ):
                        # Can't forward, can't reverse — but a DIFFERENT heading may be
                        # open. Turn to look (the robot must never halt facing a desk
                        # while the rest of the room is wide open behind it).
                        stuck_scan_rotations += 1
                        chosen_direction_sign = float(direction_sign)
                        chosen_turn_deg = 55.0
                        should_turn = True
                        turn_reason = "stuck_scan"
                        print(
                            f"[wander] STUCK ({exit_stall_cycles} cycles) — can't go forward or "
                            "reverse from this heading; TURNING to look for a way out "
                            f"({stuck_scan_rotations}/{STUCK_SCAN_ROTATIONS_HALT} before giving up)"
                        )
                    else:
                        # Turned a full circle (or hit the hard stall cap) and no
                        # heading is drivable, no room to reverse = genuinely BOXED IN.
                        print(
                            "[wander] BOXED IN: turned a full circle and no heading the body fits is "
                            f"drivable, and no room to reverse (stalled {exit_stall_cycles}) — there is "
                            "no way onward from this spot. HALTING to idle (reposition the robot or "
                            "clear the exit)."
                        )
                        _send_stop(robot)
                        break
                elif bootstrap_scan_active:
                    # Panorama is deliberately finite and monotonic: four
                    # same-direction views, then translation. Do not chase a
                    # missing heading bin forever when turn tracking is weak.
                    bootstrap_turn_commands += 1
                    chosen_direction_sign = float(direction_sign)
                    chosen_turn_deg = min(85.0, float(args.turn_deg))
                    should_turn = True
                    turn_reason = "bootstrap_scan"
                    print(
                        "[wander] bootstrap panorama turn "
                        f"({bootstrap_turn_commands}/{max(1, int(args.bootstrap_turn_captures))}, "
                        f"turn={chosen_turn_deg:.1f}deg {'ccw' if chosen_direction_sign >= 0.0 else 'cw'})"
                    )
                    if bootstrap_turn_commands >= max(1, int(args.bootstrap_turn_captures)):
                        rotation_coverage_complete = True
                        force_drive_after_turn = True
                        print(
                            "[wander] bootstrap panorama budget reached; next cycle must translate "
                            "to a new viewpoint"
                        )
                elif survey_arrived:
                    print(
                        "[wander] survey vantage reached; scanning from the new viewpoint "
                        f"({active_survey_xy[0]:.2f}, {active_survey_xy[1]:.2f})"
                    )
                    active_survey_xy = None
                    no_frontier_cycles = 0
                    motion_hint = MotionHint(
                        kind="drive",
                        expected_dx_local_m=0.0,
                        expected_dy_local_m=0.0,
                        expected_dtheta_deg=0.0,
                        search_xy_m=0.30,
                        search_theta_window_deg=12.0,
                        label=f"survey_scan_{capture_index:02d}",
                    )
                    settle_s = float(args.capture_settle_s)
                elif (
                    hazard_monitor is not None
                    and not hazard_monitor.forward_allowed()[0]
                    and not bootstrap_scan_active
                ):
                    # A camera gate (or the hold it left behind) denies forward
                    # motion. The hazard is invisible to the lidar map, so
                    # replanning alone can never route around it — recover
                    # actively. Ladder: (1) investigate the edge (measure +
                    # map it), (2) back away, (3) rotate the nose off it so
                    # the reverse-only hold releases and forward becomes
                    # retreat, (4) hold and rescan. Field deadlock 2026-07-11:
                    # without (3), a robot wedged against furniture behind
                    # looped plan->denied->capture forever.
                    # NOT during bootstrap: a 1-capture map cannot track
                    # recovery maneuvers (field 2026-07-11: a lock-lost
                    # rotate-away at capture 1 left ~70deg of untracked
                    # rotation and wrecked the map) — bootstrap scans are
                    # turns anyway, and turns release holds by rotation.
                    hazard_state_now = hazard_monitor.state()
                    _forward_ok, forward_denial = hazard_monitor.forward_allowed()
                    no_frontier_cycles = 0
                    # The recovery below rotates/reverses; any committed probe
                    # bearing is stale (and may point INTO the hazard) after it.
                    probe_commit_world_deg = None
                    probe_commit_align_turns = 0
                    print(
                        f"[safety] forward denied while planning ({forward_denial}, "
                        f"side={hazard_state_now.side}); recovering"
                    )
                    # RE-PLAN AROUND CAMERA-CONFIRMED FURNITURE (user 2026-07-20:
                    # "re-plan the path when it encounters this"). When a CAMERA gate
                    # (depth/semantic elevated edge) — not the stop box, not a stale
                    # frame — refuses forward motion, the robot is nose-to-furniture
                    # the lidar sees UNDER (the desk-in-the-doorway case). Stamp that
                    # obstacle's footprint as a PERMANENT no-go so the opening picker
                    # vetoes this bearing and re-plans toward a genuinely clear
                    # opening, instead of re-committing to the phantom doorway forever.
                    if not hazard_state_now.frames_stale and _denial_is_camera_obstacle(
                        forward_denial
                    ):
                        _stamp_camera_block(current_live_pose)
                    if hazard_state_now.frames_stale:
                        # FRESHNESS, not a physical obstacle: the camera/depth
                        # frame is stale (a slow or hiccuping sensor), so the eyes
                        # are momentarily blind — there is nothing to escape. Field
                        # 2026-07-20: with depth forced onto CPU (~3s/frame) this
                        # fired EVERY cycle; the furniture-recovery ladder below
                        # then burned its one rotate-away and tried to REVERSE into
                        # the wall behind (0.06m clear), wedging forever. The
                        # elevated-safety contract (sourccey_elevated_safety: stale
                        # must not arm a hold/retreat) says the correct response is
                        # to HOLD IN PLACE and let driving resume the instant frames
                        # freshen — never maneuver while sensor-blind. Hold here and
                        # skip the whole obstacle-escape ladder.
                        _send_stop(robot)
                        print(
                            "[safety] denial is STALE frames (slow/hiccuping camera "
                            "or depth), not an obstacle — holding still until frames "
                            "freshen; NOT reversing or turning while sensor-blind"
                        )
                        motion_hint = MotionHint(
                            kind="drive",
                            expected_dx_local_m=0.0,
                            expected_dy_local_m=0.0,
                            expected_dtheta_deg=0.0,
                            search_xy_m=0.30,
                            search_theta_window_deg=12.0,
                            label=f"stale_hold_{capture_index:02d}",
                        )
                        settle_s = float(args.capture_settle_s)
                    if _elevated_hazard_engaged():
                        # Mark that the current lack of forward progress is due to
                        # an elevated-edge hold (furniture beside the path), so the
                        # redundant-capture breaker below does not mistake it for an
                        # unresolvable frontier and blacklist the doorway.
                        elevated_block_recent = 8
                    # JAMMED REFLEX (field 2026-07-20, user): if the eyes say the
                    # robot is RIGHT ON an obstacle / can't see ground, the ONLY
                    # safe move is to REVERSE — turning here catches an arm and
                    # tips it (it fell over). Take this BEFORE any sidestep/rotate/
                    # investigate so those never run while jammed. If there is no
                    # clearance behind either, HOLD still (never pirouette on the
                    # obstacle).
                    if motion_hint is None and _jammed_on_hazard():
                        reflex_reverse = _attempt_reverse_escape_hint(
                            current_live_pose, failed_reverse_escapes >= 1
                        )
                        if reflex_reverse is not None:
                            print(
                                "[safety] JAMMED on an obstacle (eyes: on top of an edge / "
                                "no ground) — reversing to regain clearance; NOT turning"
                            )
                            failed_reverse_escapes = 0
                            motion_hint = reflex_reverse
                            settle_s = float(args.move_settle_s)
                        else:
                            print(
                                "[safety] JAMMED on an obstacle with NO clearance behind "
                                "either — holding still (refusing to turn while jammed; "
                                "reposition the robot by hand if this persists)"
                            )
                            _send_stop(robot)
                            motion_hint = MotionHint(
                                kind="drive",
                                expected_dx_local_m=0.0,
                                expected_dy_local_m=0.0,
                                expected_dtheta_deg=0.0,
                                search_xy_m=0.30,
                                search_theta_window_deg=12.0,
                                label=f"jammed_hold_{capture_index:02d}",
                            )
                            settle_s = float(args.capture_settle_s)
                    # Bundle these denials too: they dominate runs (ground
                    # gate / holds) and were undiagnosable from logs alone.
                    # Rate-limited so a hold that denies every cycle does not
                    # flood the directory with identical bundles.
                    now_denial_mono = time.monotonic()
                    if forward_denial != planning_denial_last["reason"] or (
                        now_denial_mono - planning_denial_last["mono"] >= 20.0
                    ):
                        planning_denial_last["reason"] = forward_denial
                        planning_denial_last["mono"] = now_denial_mono
                        _dump_stop_debug(current_live_pose, f"planning_denial:{forward_denial}")
                    # DELIBERATE RETREAT-FIRST (field 2026-07-20 user: "move back
                    # to where it was before, then adjust/turn so it can go
                    # forward WITHOUT getting caught, then try again"). Caught by
                    # furniture: back straight off to a clear spot ONCE per stuck
                    # episode BEFORE any sidestep/edge-investigation (those keep
                    # re-facing the furniture). The reorient (rotate-toward-opening
                    # below, next cycle) then aims at a clear wide LiDAR opening,
                    # and the retry drive is camera-gated. caught_retreated resets
                    # when the furniture episode ages out (elevated_block_recent→0).
                    if (
                        motion_hint is None
                        and _elevated_hazard_engaged()
                        and not caught_retreated
                    ):
                        retreat_hint = _attempt_reverse_escape_hint(current_live_pose, False)
                        if retreat_hint is not None:
                            print(
                                "[safety] caught by furniture — backing off to a clear spot "
                                "first, THEN reorienting to an opening the body fits and retrying"
                            )
                            caught_retreated = True
                            failed_reverse_escapes = 0
                            motion_hint = retreat_hint
                            settle_s = float(args.move_settle_s)
                    if (
                        motion_hint is None
                        and _elevated_hazard_engaged()
                        and not hazard_state_now.frames_stale
                        and not hazard_state_now.blind_zone
                        and not hazard_state_now.ground_active
                        and str(hazard_state_now.side) in ("left", "right", "front")
                        and not any(
                            math.hypot(
                                float(current_live_pose.x) - sx,
                                float(current_live_pose.y) - sy,
                            )
                            <= 0.45
                            for sx, sy in sidestep_spots_world
                        )
                    ):
                        sidestep_hint = _sidestep_waypoint_hint(
                            current_live_pose,
                            str(hazard_state_now.side),
                            (
                                float(planning_frontier_choice.delta_deg)
                                if planning_frontier_choice is not None
                                else None
                            ),
                        )
                        if sidestep_hint is not None:
                            sidestep_spots_world.append(
                                (float(current_live_pose.x), float(current_live_pose.y))
                            )
                            del sidestep_spots_world[:-16]
                            motion_hint = sidestep_hint
                            settle_s = float(args.move_settle_s)
                    # ROTATE TOWARD THE LIVE LIDAR OPENING FIRST (field 2026-07-20).
                    # When an elevated hold denies forward motion but the LiDAR
                    # shows a clear opening well off to one side (a hallway to the
                    # right while a couch sits against the wall on the left), the
                    # human-obvious move is to turn toward the opening — not to keep
                    # investigating/reversing off the furniture, which re-faces the
                    # couch and piles up redundant captures until the exit frontier
                    # gets blacklisted. The depth `side` field is usually 'none'
                    # once the stop decays into a post-stop HOLD, so this steers by
                    # the LiDAR opening, not the depth side. Fires at most once per
                    # stuck episode (recovery_turn_sign guard); force_live_frontier
                    # then drives INTO the opening. Promoted ahead of investigate/
                    # reverse only when a genuinely off-axis opening exists.
                    if (
                        motion_hint is None
                        and _elevated_hazard_engaged()
                        and recovery_turn_sign is None
                        and rotate_away_lock_losses == 0
                        and live_frontier_choice is not None
                        and abs(float(live_frontier_choice.delta_deg)) >= 25.0
                    ):
                        open_rotate_hint = _rotate_away_hint(
                            current_live_pose,
                            hazard_state_now.side,
                            float(live_frontier_choice.delta_deg),
                        )
                        if open_rotate_hint is not None:
                            print(
                                "[safety] elevated hold with a clear lidar opening "
                                f"{float(live_frontier_choice.delta_deg):+.0f}deg to the side "
                                "— turned toward the opening instead of investigating "
                                "the furniture"
                            )
                            failed_reverse_escapes = 0
                            motion_hint = open_rotate_hint
                            settle_s = float(args.capture_settle_s)
                            force_live_frontier_cycles = max(force_live_frontier_cycles, 2)
                    if (
                        motion_hint is None
                        and str(args.edge_mapping) == "on"
                        and _elevated_hazard_engaged()
                        and not hazard_state_now.frames_stale
                        and not any(
                            math.hypot(
                                float(current_live_pose.x) - ix, float(current_live_pose.y) - iy
                            )
                            <= 0.7
                            for ix, iy in investigated_spots_world
                        )
                    ):
                        investigation_hint = _investigate_edge_hint(current_live_pose)
                        if investigation_hint is not None:
                            # Cooldown only on a COMPLETED survey — a skipped
                            # one (no runway) must not suppress later attempts
                            # near here.
                            investigated_spots_world.append(
                                (float(current_live_pose.x), float(current_live_pose.y))
                            )
                            del investigated_spots_world[:-16]
                            motion_hint = investigation_hint
                            settle_s = float(args.move_settle_s)
                    if motion_hint is None:
                        # A prior rotate-away losing tracking lock counts as a
                        # failed escape: in a point-starved corner rotation is
                        # untrackable (discard -> re-anchor -> oscillation), so
                        # escalate to the slow blind reverse instead.
                        escape_hint = _attempt_reverse_escape_hint(
                            current_live_pose,
                            failed_reverse_escapes >= 1 or rotate_away_lock_losses >= 1,
                        )
                        if escape_hint is not None:
                            failed_reverse_escapes = 0
                            rotate_away_lock_losses = 0
                            motion_hint = escape_hint
                            settle_s = float(args.move_settle_s)
                    # One recovery turn is followed by a forward-planning
                    # attempt. Do not spin again just because the next depth
                    # frame calls the hazard the other side.
                    if (
                        motion_hint is None
                        and rotate_away_lock_losses == 0
                        and recovery_turn_sign is None
                    ):
                        rotate_hint = _rotate_away_hint(
                            current_live_pose,
                            hazard_state_now.side,
                            (
                                float(live_frontier_choice.delta_deg)
                                if live_frontier_choice is not None
                                else None
                            ),
                        )
                        if rotate_hint is not None:
                            if rotate_away_lock_losses == 0:
                                failed_reverse_escapes = 0
                            motion_hint = rotate_hint
                            settle_s = float(args.capture_settle_s)
                            force_live_frontier_cycles = max(force_live_frontier_cycles, 2)
                    if motion_hint is None:
                        failed_reverse_escapes += 1
                        motion_hint = MotionHint(
                            kind="drive",
                            expected_dx_local_m=0.0,
                            expected_dy_local_m=0.0,
                            expected_dtheta_deg=0.0,
                            search_xy_m=0.30,
                            search_theta_window_deg=12.0,
                            label=f"hazard_hold_{capture_index:02d}",
                        )
                        settle_s = float(args.capture_settle_s)
                elif (blocked and _plan_drives_into_blocked_front()) or plan_status == "stuck":
                    # The stop box vetoes a plan ONLY when the plan actually
                    # wants to drive through the occupied front. A robot parked
                    # nose-to-wall with a valid path pointing 60deg away must
                    # simply run the transit (align turn, then gated drive) —
                    # field 2026-07-11: this branch used to fire on ANY front
                    # occupancy and spun toward alternating live-frontier
                    # bearings, pirouetting cw/ccw against the wall forever.
                    no_frontier_cycles = 0
                    consecutive_blocked_cycles += 1
                    print(
                        "[wander] path blocked "
                        f"(blocked_points={blocked_points}, status={plan_status}); the next capture adds "
                        "the blocking geometry to the map, then the planner routes around it"
                    )
                    if plan_status in ("ok", "survey", "observe"):
                        _strike_frontier(frontier_plan["face_xy"], "path_blocked")
                    if consecutive_blocked_cycles >= 3:
                        consecutive_blocked_cycles = 0
                        escape_hint = _attempt_reverse_escape_hint(
                            current_live_pose, failed_reverse_escapes >= 1
                        )
                        if escape_hint is not None:
                            failed_reverse_escapes = 0
                            motion_hint = escape_hint
                            settle_s = float(args.move_settle_s)
                        else:
                            failed_reverse_escapes += 1
                    if motion_hint is None:
                        escape_to_live_opening = False
                        if live_frontier_choice is not None:
                            escape_delta_deg = float(live_frontier_choice.delta_deg)
                            escape_to_live_opening = True
                        elif plan_status in ("ok", "survey", "observe"):
                            escape_delta_deg = _normalize_angle_deg(
                                math.degrees(
                                    math.atan2(
                                        float(frontier_plan["waypoint_xy"][1]) - float(current_live_pose.y),
                                        float(frontier_plan["waypoint_xy"][0]) - float(current_live_pose.x),
                                    )
                                )
                                - float(current_live_pose.theta_deg)
                            )
                        else:
                            escape_delta_deg = 55.0 * float(direction_sign)
                        chosen_direction_sign = 1.0 if escape_delta_deg >= 0.0 else -1.0
                        chosen_turn_deg = max(15.0, min(85.0, abs(float(escape_delta_deg))))
                        should_turn = True
                        turn_reason = "blocked_replan"
                        if escape_to_live_opening:
                            # STICKY ESCAPE (field 2026-07-20 user: it turned RIGHT
                            # to escape the block CORRECTLY, then the very next cycle
                            # re-faced the same blocked map goal and turned LEFT
                            # straight back into the furniture). After turning toward
                            # the CLEAR live opening, COMMIT to driving it for a few
                            # cycles (escape_frontier_active) and drive right after
                            # the turn — so the goal-seeker cannot immediately undo
                            # the escape and re-approach the obstacle.
                            force_live_frontier_cycles = max(force_live_frontier_cycles, 3)
                            force_drive_after_turn = True
                            print(
                                "[wander] block-escape: turning toward the clear opening "
                                f"(delta={escape_delta_deg:+.0f}deg) and COMMITTING to drive it "
                                "— not re-facing the blocked goal"
                            )
                elif plan_status == "no_frontier":
                    # A live lidar opening is newer than the stitched grid and
                    # may be beyond its current boundary (especially just
                    # after crossing a doorway). Do not declare completion
                    # while it exists: align to it, then take one guarded
                    # probe drive to extend the map.
                    if live_frontier_choice is not None:
                        no_frontier_cycles = 0
                        # Commit to ONE bearing for the whole probe episode.
                        # The residual is measured against the committed
                        # world bearing, never the per-cycle widest gap —
                        # the widest gap changes with heading, which is what
                        # caused the pirouette loop.
                        if probe_commit_world_deg is None:
                            probe_commit_world_deg = _normalize_angle_deg(
                                float(current_live_pose.theta_deg)
                                + float(live_frontier_choice.delta_deg)
                            )
                            probe_commit_align_turns = 0
                        probe_residual_deg = _normalize_angle_deg(
                            probe_commit_world_deg - float(current_live_pose.theta_deg)
                        )
                        if abs(probe_residual_deg) > 20.0 and probe_commit_align_turns < 2:
                            probe_commit_align_turns += 1
                            chosen_direction_sign = 1.0 if probe_residual_deg >= 0.0 else -1.0
                            chosen_turn_deg = max(15.0, min(85.0, abs(probe_residual_deg)))
                            should_turn = True
                            turn_reason = "live_frontier_probe_align"
                            force_drive_after_turn = True
                            print(
                                "[wander] map has no frontier but live lidar is open; aligning "
                                f"{chosen_turn_deg:.1f}deg toward the committed probe bearing "
                                f"({probe_commit_align_turns}/2)"
                            )
                        else:
                            # Aligned (or align budget spent — the drive is
                            # safety-gated either way): take the probe now
                            # and release the commitment.
                            probe_commit_world_deg = None
                            probe_commit_align_turns = 0
                            # In the exit run, try the CENTRED doorway maneuver
                            # first (square up + centre on the mouth + drive
                            # through the middle); fall back to the blind bearing
                            # probe when the gap is not doorframe-like.
                            motion_hint = None
                            if exit_mode:
                                motion_hint = _exit_doorway_approach_hint(
                                    current_live_pose, live_frontier_choice
                                )
                            if motion_hint is None:
                                motion_hint = _live_frontier_probe_hint(
                                    current_live_pose, live_frontier_choice
                                )
                            _probe_fwd_m = float(motion_hint.expected_dx_local_m)
                            if exit_mode and pose_lost:
                                # Only while LOST: a healthy pose here resolves
                                # via the positional completion. Lost + probing
                                # the committed opening is the threshold cross.
                                exit_through_odometry_m += max(0.0, _probe_fwd_m)
                                exit_opening_seen = True
                            if exit_mode and _probe_fwd_m < 0.06:
                                # BLOCKED PROBE (the acute "rotating in the corner"
                                # bug): an elevated obstacle / stop box is across
                                # THIS opening, so the drive advanced ~nothing. Do
                                # NOT keep re-committing to the same blocked bearing.
                                # Rotate ~55deg to hunt a DIFFERENT opening; after a
                                # full revolution of blocked openings, HALT instead
                                # of spinning forever. (A successful probe below
                                # resets the counter.)
                                exit_probe_blocked_streak += 1
                                motion_hint = None
                                if exit_probe_blocked_streak >= EXIT_PROBE_BLOCKED_LIMIT:
                                    print(
                                        "[wander] EXIT RUN: every opening around the robot is BLOCKED "
                                        "by furniture/obstacles after a full search — the exit is "
                                        "physically obstructed. HALTING (clear the doorway or "
                                        "reposition the robot); NOT rotating in place any further."
                                    )
                                    _send_stop(robot)
                                    break
                                # MEASURE BEFORE HUNTING (user 2026-07-20): when an
                                # ELEVATED edge blocked this opening, don't just spin
                                # away — go look at it properly. Back off and
                                # re-approach so the edge-map captures the edge's true
                                # extent (and a CLEAN re-approach proves the stop was a
                                # false positive and the opening is really open). This
                                # is the desk-at-the-doorway case: lidar reads a gap
                                # UNDER the desk, the cameras correctly veto, and only
                                # a measured look tells us whether the edge actually
                                # spans the mouth or is a corner to round. The
                                # exit-scan rotations between attempts supply the
                                # differing view angles the edge-map cross-confirms.
                                # Bounded per exit episode so it cannot loop.
                                if (
                                    str(args.edge_mapping) == "on"
                                    and _elevated_hazard_engaged()
                                    and exit_edge_investigations < EXIT_EDGE_INVESTIGATION_LIMIT
                                ):
                                    investigation_hint = _investigate_edge_hint(
                                        current_live_pose
                                    )
                                    if investigation_hint is not None:
                                        exit_edge_investigations += 1
                                        motion_hint = investigation_hint
                                        settle_s = float(args.move_settle_s)
                                        print(
                                            "[wander] EXIT RUN: opening blocked by an elevated edge "
                                            "— measuring its true extent (back off + re-approach) to "
                                            "learn whether it really spans the doorway before hunting "
                                            f"elsewhere ({exit_edge_investigations}/"
                                            f"{EXIT_EDGE_INVESTIGATION_LIMIT})"
                                        )
                                if motion_hint is None:
                                    chosen_direction_sign = float(direction_sign)
                                    chosen_turn_deg = 55.0
                                    should_turn = True
                                    turn_reason = "exit_scan"
                                    print(
                                        f"[wander] EXIT RUN: this opening is BLOCKED (probe advanced only "
                                        f"{_probe_fwd_m:.2f}m); rotating ~55deg to hunt a different opening "
                                        f"({exit_probe_blocked_streak}/{EXIT_PROBE_BLOCKED_LIMIT})"
                                    )
                            else:
                                exit_probe_blocked_streak = 0
                                exit_edge_investigations = 0
                                # A probe that actually advanced is real progress —
                                # refill the bounded exit-scan budget so the robot
                                # can look around again from the new spot.
                                exit_scan_turns = 0
                                settle_s = float(args.move_settle_s)
                    else:
                        # The opening vanished from the live scan — the
                        # commitment no longer refers to anything real.
                        probe_commit_world_deg = None
                        probe_commit_align_turns = 0
                        no_frontier_cycles += 1
                    if live_frontier_choice is None:
                        if motion_hint is None and no_frontier_cycles >= 2 and pose_lost:
                            # A lost robot cannot certify anything about the
                            # map — least of all that it is finished (field
                            # 2026-07-17: "mapping complete" declared inside
                            # a ghost-sealed grid with 14833 unknown cells).
                            print(
                                "[wander] map frontier exhausted but the pose is LOST — refusing to "
                                "declare completion; continuing recovery"
                            )
                            no_frontier_cycles = 0
                        elif motion_hint is None and no_frontier_cycles >= 2 and exit_mode and exit_scan_turns < 7:
                            # EXIT RUN with no opening visible from this heading:
                            # do NOT declare completion — rotate and look for the
                            # doorway. Bounded at 7 turns (~a full revolution);
                            # only then may the normal completion path conclude.
                            exit_scan_turns += 1
                            no_frontier_cycles = 0
                            chosen_direction_sign = float(direction_sign)
                            chosen_turn_deg = 55.0
                            should_turn = True
                            turn_reason = "exit_scan"
                            print(
                                "[wander] EXIT RUN: no opening visible from this heading; "
                                f"rotating to search for the doorway ({exit_scan_turns}/7)"
                            )
                        elif motion_hint is None and no_frontier_cycles >= 2:
                            unreachable_now = int(frontier_plan.get("unreachable_frontier_cells", 0) or 0)
                            if unreachable_now >= 24:
                                print(
                                    "[wander] WARNING: declaring completion with "
                                    f"{unreachable_now} UNREACHABLE frontier cells "
                                    f"(unknown={int(frontier_plan.get('unknown_cells', 0) or 0)}) — "
                                    "if that area should be reachable, the map is suspect "
                                    "(ghost walls or clutter seal); inspect the overlay before trusting this map"
                                )
                            print(
                                "[wander] mapping complete: no reachable unmapped space remains and all "
                                "survey vantages have been visited "
                                f"(captures={len(stitch_state['snapshots'])}, "
                                f"html={stitch_state['html_path']})"
                            )
                            if str(args.on_complete) == "stop":
                                break
                            print(
                                "[wander] entering idle watch: holding position; exploration resumes "
                                "automatically if new reachable space appears (Ctrl+C to exit)"
                            )
                            _send_stop(robot)
                            idle_resume = False
                            while not idle_resume:
                                time.sleep(3.0)
                                # Heartbeat: a zero-velocity action every tick keeps
                            # the radio link busy — an idle link lets the Pi's
                                # WiFi power-save kick in and the lidar stream then
                                # stalls into reconnect loops.
                                try:
                                    robot.send_action(
                                        {
                                            "x.vel": 0.0,
                                            "y.vel": 0.0,
                                            "theta.vel": 0.0,
                                            "z.pos": getattr(robot, "_z_pos_cmd", 100.0),
                                            "untorque_left": True,
                                            "untorque_right": True,
                                        }
                                    )
                                except Exception:
                                    pass
                                _idle_frame_id, idle_frame = feed.latest()
                                if idle_frame is None:
                                    continue
                                idle_points_xy = _scan_to_local_points(
                                    points=idle_frame.points,
                                    forward_angle_deg=float(args.forward_angle_deg),
                                    valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                                    invert_lateral_axis=bool(args.invert_lateral_axis),
                                    max_distance_m=float(args.max_distance_m),
                                    min_confidence=int(args.min_confidence),
                                    min_range_m=float(args.min_range_m),
                                )
                                if len(idle_points_xy) < 12:
                                    continue
                                # Treat the live scan as a virtual capture: if the
                                # world changed (a door opened in view), the grid
                                # gains new free space and a frontier appears.
                                idle_world_xy = _transform_points(idle_points_xy, current_live_pose)
                                idle_grid = _build_exploration_grid(
                                    poses=[*list(stitch_state["poses"]), current_live_pose],
                                    transformed_sets=[*list(stitch_state["transformed_sets"]), idle_world_xy],
                                    resolution_m=max(0.05, float(args.stitch_resolution_m)),
                                    robot_clear_radius_m=float(args.robot_radius_m) + 0.06,
                                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                                    extra_occupied_xy=(
                                        np.asarray(
                                            [[c[0] * 0.04, c[1] * 0.04] for c in elevated_occupied_world],
                                            dtype=np.float32,
                                        )
                                        if elevated_occupied_world
                                        else None
                                    ),
                                )
                                if idle_grid is None:
                                    continue
                                idle_plan = _plan_frontier_path(
                                    grid=idle_grid,
                                    robot_xy=(float(current_live_pose.x), float(current_live_pose.y)),
                                    robot_radius_m=float(args.robot_radius_m),
                                    robot_theta_deg=float(current_live_pose.theta_deg),
                                    avoid_face_xy=_active_frontier_blacklist(),
                                )
                                if str(idle_plan.get("status")) == "ok":
                                    print(
                                        "[wander] new reachable space detected during idle watch; "
                                        "resuming exploration"
                                    )
                                    no_frontier_cycles = 0
                                    idle_resume = True
                            continue
                        should_turn = True
                        chosen_direction_sign = float(direction_sign)
                        chosen_turn_deg = 85.0
                        turn_reason = "completion_verification"
                        print("[wander] no frontier and no unvisited vantage; one verification scan before stopping")
                else:
                    no_frontier_cycles = 0
                    goal_dx_m = float(frontier_plan["goal_xy"][0]) - float(current_live_pose.x)
                    goal_dy_m = float(frontier_plan["goal_xy"][1]) - float(current_live_pose.y)
                    goal_distance_m = math.hypot(goal_dx_m, goal_dy_m)
                    if goal_distance_m <= max(0.40, float(args.frontier_goal_reached_m)):
                        face_dx_m = float(frontier_plan["face_xy"][0]) - float(current_live_pose.x)
                        face_dy_m = float(frontier_plan["face_xy"][1]) - float(current_live_pose.y)
                        face_distance_m = math.hypot(face_dx_m, face_dy_m)
                        if face_distance_m < 0.30:
                            # A face point at (or under) the robot has no
                            # meaningful bearing: atan2 over centimeters made
                            # "facing it" trivially true and the robot scanned
                            # in place for 4 identical captures (field
                            # 2026-07-13: "It didn't move"). The live frontier
                            # candidate knows the real direction — use it.
                            if live_frontier_choice is not None:
                                face_delta_deg = float(live_frontier_choice.delta_deg)
                            else:
                                face_delta_deg = 85.0
                            print(
                                "[wander] frontier face point degenerate "
                                f"({face_distance_m:.2f}m away); using live candidate "
                                f"bearing (delta={face_delta_deg:.1f}deg)"
                            )
                        else:
                            face_bearing_deg = math.degrees(math.atan2(face_dy_m, face_dx_m))
                            face_delta_deg = _normalize_angle_deg(
                                face_bearing_deg - float(current_live_pose.theta_deg)
                            )
                        if abs(face_delta_deg) <= 30.0 and stationary_scan_streak >= 1:
                            # Already scanned from this exact pose and the plan
                            # came back unchanged: a second identical scan can
                            # teach nothing. Rotate instead — motion is the
                            # only way out of a stale plan.
                            face_delta_deg = (
                                float(live_frontier_choice.delta_deg)
                                if live_frontier_choice is not None
                                and abs(float(live_frontier_choice.delta_deg)) > 30.0
                                else 85.0
                            )
                            print(
                                "[wander] stationary scan already taken here; forcing a "
                                f"turn (delta={face_delta_deg:.1f}deg) instead of re-scanning"
                            )
                        if abs(face_delta_deg) > 30.0:
                            stationary_scan_streak = 0
                            chosen_direction_sign = 1.0 if face_delta_deg >= 0.0 else -1.0
                            chosen_turn_deg = max(15.0, min(85.0, abs(face_delta_deg)))
                            should_turn = True
                            turn_reason = "face_frontier"
                            print(
                                "[wander] at the frontier; turning to face the unmapped area "
                                f"(delta={face_delta_deg:.1f}deg)"
                            )
                        else:
                            stationary_scan_streak += 1
                            motion_hint = MotionHint(
                                kind="drive",
                                expected_dx_local_m=0.0,
                                expected_dy_local_m=0.0,
                                expected_dtheta_deg=0.0,
                                search_xy_m=0.30,
                                search_theta_window_deg=12.0,
                                label=f"frontier_scan_{capture_index:02d}",
                            )
                            settle_s = float(args.capture_settle_s)
                            print("[wander] at the frontier and facing it; scanning the new area")
                    else:
                        # Multi-leg transit: traveling through already-mapped
                        # space, chain several lidar-tracked turn+drive legs
                        # and commit ONE capture at the end — captures en route
                        # add no map information and slow the journey down.
                        stationary_scan_streak = 0
                        transit_pose = current_live_pose
                        transit_plan: dict[str, object] = frontier_plan
                        transit_hint: MotionHint | None = None
                        transit_traveled_m = 0.0
                        transit_legs = 0
                        transit_stop_reason = "leg_limit"
                        while transit_legs < 3:
                            waypoint_dx_m = float(transit_plan["waypoint_xy"][0]) - float(transit_pose.x)
                            waypoint_dy_m = float(transit_plan["waypoint_xy"][1]) - float(transit_pose.y)
                            waypoint_distance_m = math.hypot(waypoint_dx_m, waypoint_dy_m)
                            waypoint_delta_deg = _normalize_angle_deg(
                                math.degrees(math.atan2(waypoint_dy_m, waypoint_dx_m))
                                - float(transit_pose.theta_deg)
                            )
                            leg_pre_hint: MotionHint | None = None
                            if abs(waypoint_delta_deg) > 32.0:
                                align_turn_deg = max(15.0, min(85.0, abs(waypoint_delta_deg)))
                                align_sign = 1.0 if waypoint_delta_deg >= 0.0 else -1.0
                                print(
                                    "[wander] turning to face the path toward the goal "
                                    f"(delta={waypoint_delta_deg:.1f}deg, turning {align_turn_deg:.1f}deg "
                                    f"{'ccw' if align_sign >= 0.0 else 'cw'})"
                                )
                                align_meta = _turn_with_arc_tracking(
                                    robot=robot,
                                    feed=feed,
                                    transformed_sets=stitch_state["transformed_sets"],
                                    start_pose=transit_pose,
                                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                                    resolution_m=float(args.stitch_resolution_m),
                                    target_turn_deg=float(align_turn_deg),
                                    direction_sign=float(align_sign),
                                    turn_speed=float(args.turn_speed),
                                    turn_burst_s=float(args.turn_burst_s),
                                    turn_settle_s=float(args.turn_settle_s),
                                    forward_angle_deg=float(args.forward_angle_deg),
                                    valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                                    invert_lateral_axis=bool(args.invert_lateral_axis),
                                    max_distance_m=float(args.max_distance_m),
                                    min_range_m=float(args.min_range_m),
                                    min_confidence=int(args.min_confidence),
                                    stop_tolerance_deg=min(8.0, float(args.stop_tolerance_deg)),
                                    max_bursts=int(args.max_turn_bursts),
                    hazard_monitor=hazard_monitor,
                                )
                                align_turned_deg = float(align_meta["turned_deg"])
                                align_lever_dx, align_lever_dy = _turn_lever_arm_local_delta(
                                    align_turned_deg,
                                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                                )
                                leg_pre_hint = MotionHint(
                                    kind="turn",
                                    expected_dx_local_m=float(align_lever_dx),
                                    expected_dy_local_m=float(align_lever_dy),
                                    expected_dtheta_deg=float(align_turned_deg),
                                    search_xy_m=float(args.turn_search_xy_m),
                                    search_theta_window_deg=min(
                                        80.0, 25.0 + 8.0 * float(align_meta["missed_updates"])
                                    ),
                                    label=f"align_turn_{capture_index:02d}",
                                )
                                transit_pose = _advance_pose(transit_pose, leg_pre_hint)
                                transit_hint = (
                                    _compose_motion_hints(transit_hint, leg_pre_hint)
                                    if transit_hint is not None
                                    else leg_pre_hint
                                )
                                waypoint_dx_m = float(transit_plan["waypoint_xy"][0]) - float(transit_pose.x)
                                waypoint_dy_m = float(transit_plan["waypoint_xy"][1]) - float(transit_pose.y)
                                waypoint_distance_m = math.hypot(waypoint_dx_m, waypoint_dy_m)
                                waypoint_delta_deg = _normalize_angle_deg(
                                    math.degrees(math.atan2(waypoint_dy_m, waypoint_dx_m))
                                    - float(transit_pose.theta_deg)
                                )
                                if (
                                    bool(align_meta.get("lock_lost"))
                                    or abs(waypoint_delta_deg) > 40.0
                                    or waypoint_distance_m < 0.25
                                ):
                                    transit_stop_reason = "align_residual"
                                    break
                            planned_bursts = max(1, min(4, int(math.ceil(waypoint_distance_m / 0.30))))
                            steer_theta_vel = max(
                                -float(args.max_drive_steer_theta_vel),
                                min(
                                    float(args.max_drive_steer_theta_vel),
                                    float(waypoint_delta_deg) * float(args.frontier_drive_steer_gain),
                                ),
                            )
                            print(
                                "[wander] driving the planned path "
                                f"(leg={transit_legs + 1}, waypoint_distance={waypoint_distance_m:.2f}m, "
                                f"bursts={planned_bursts}, steer={steer_theta_vel:.3f})"
                            )
                            drive_tracked_pose, drive_track_meta = _drive_with_tracking(
                                robot=robot,
                                feed=feed,
                                zone_cfg=zone_cfg,
                                transformed_sets=stitch_state["transformed_sets"],
                                start_pose=transit_pose,
                                resolution_m=float(args.stitch_resolution_m),
                                forward_speed=float(args.move_speed),
                                min_effective_move_speed=float(args.min_effective_move_speed),
                                burst_s=float(args.move_burst_s),
                                burst_count=int(planned_bursts),
                                inter_burst_pause_s=float(args.inter_burst_pause_s),
                                steer_theta_vel=float(steer_theta_vel),
                                forward_angle_deg=float(args.forward_angle_deg),
                                valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                                invert_lateral_axis=bool(args.invert_lateral_axis),
                                max_distance_m=float(args.max_distance_m),
                                min_range_m=float(args.min_range_m),
                                min_confidence=int(args.min_confidence),
                                hazard_monitor=hazard_monitor,
                                map_side_guard=_map_side_guard,
                                forward_guard=_exit_ratchet_guard,
                                pose_trusted=not pose_lost,
                                frame_recorder=_record_eye_frames,
                                imu_yaw_fn=_drive_hold_yaw_deg,
                                yaw_hold_gain=_drive_yaw_hold_gain,
                                yaw_hold_max=float(args.drive_yaw_hold_max),
                            )
                            drive_stopped_by_block = (
                                bool(drive_track_meta["stopped_by_block"])
                                or bool(drive_track_meta["lock_lost"])
                                or bool(drive_track_meta.get("stopped_by_hazard"))
                            )
                            drive_ddx_world = float(drive_tracked_pose.x) - float(transit_pose.x)
                            drive_ddy_world = float(drive_tracked_pose.y) - float(transit_pose.y)
                            drive_theta0_rad = math.radians(float(transit_pose.theta_deg))
                            drive_cos = math.cos(drive_theta0_rad)
                            drive_sin = math.sin(drive_theta0_rad)
                            drive_hint_dx_local_m = drive_cos * drive_ddx_world + drive_sin * drive_ddy_world
                            drive_hint_dy_local_m = -drive_sin * drive_ddx_world + drive_cos * drive_ddy_world
                            drive_hint_dtheta_deg = _normalize_angle_deg(
                                float(drive_tracked_pose.theta_deg) - float(transit_pose.theta_deg)
                            )
                            if drive_track_meta["lock_lost"] or int(drive_track_meta["missed_updates"]) > 0:
                                drive_search_xy_m = 0.80
                                drive_theta_window_deg = 30.0
                            else:
                                drive_search_xy_m = 0.40
                                drive_theta_window_deg = 16.0
                            print(
                                "[wander] path leg complete "
                                f"(bursts={int(drive_track_meta['bursts_completed'])}, "
                                f"stopped_by_block={bool(drive_track_meta['stopped_by_block'])}, "
                                f"lock_lost={bool(drive_track_meta['lock_lost'])}, "
                                f"tracked_move=({drive_hint_dx_local_m:.3f}m, {drive_hint_dy_local_m:.3f}m, "
                                f"{drive_hint_dtheta_deg:.1f}deg))"
                            )
                            if not drive_stopped_by_block:
                                consecutive_blocked_cycles = 0
                                failed_reverse_escapes = 0
                                rotate_away_lock_losses = 0
                                recovery_turn_sign = None
                                wedged_events = 0
                            drive_leg_hint = MotionHint(
                                kind="drive",
                                expected_dx_local_m=float(drive_hint_dx_local_m),
                                expected_dy_local_m=float(drive_hint_dy_local_m),
                                expected_dtheta_deg=float(drive_hint_dtheta_deg),
                                search_xy_m=float(drive_search_xy_m),
                                search_theta_window_deg=float(drive_theta_window_deg),
                                label=f"drive_capture_{capture_index:02d}",
                            )
                            transit_hint = (
                                _compose_motion_hints(transit_hint, drive_leg_hint)
                                if transit_hint is not None
                                else drive_leg_hint
                            )
                            transit_traveled_m += math.hypot(drive_ddx_world, drive_ddy_world)
                            transit_pose = drive_tracked_pose
                            transit_legs += 1
                            if bool(drive_track_meta.get("stopped_by_hazard")):
                                # A confirmed edge stopped this leg. Its measured
                                # geometry is seconds-fresh and the tracked stop
                                # pose is where it was measured — stamp it into
                                # the planning map NOW. Without this, only full
                                # investigations mapped edges, and stops near a
                                # cooled-down spot taught the planner nothing:
                                # it re-planned the same goal and oscillated
                                # approach/stop/reverse at the same table.
                                _dump_stop_debug(transit_pose, "transit_hazard_stop")
                                _stamp_confirmed_edges(transit_pose, newest_only=True)
                                # A real approach toward this frontier failed on
                                # a camera hazard: strike it (2 strikes
                                # blacklist). Only genuine attempts count — a
                                # hold lingering across planning cycles is one
                                # event, not repeated evidence.
                                _strike_frontier_if_reached(
                                    transit_plan["face_xy"], transit_pose, "hazard_stop"
                                )
                            if drive_stopped_by_block:
                                transit_stop_reason = "blocked_or_lock"
                                break
                            if transit_traveled_m >= 2.6:
                                transit_stop_reason = "distance_budget"
                                break
                            goal_distance_now_m = math.hypot(
                                float(transit_plan["goal_xy"][0]) - float(transit_pose.x),
                                float(transit_plan["goal_xy"][1]) - float(transit_pose.y),
                            )
                            if goal_distance_now_m <= max(0.40, float(args.frontier_goal_reached_m)):
                                transit_stop_reason = "arrived"
                                break
                            # Replan the next leg from the new pose against the
                            # SAME grid (no new captures yet): the waypoint
                            # advances along the path.
                            transit_plan = _plan_frontier_path(
                                grid=exploration_grid,
                                robot_xy=(float(transit_pose.x), float(transit_pose.y)),
                                robot_radius_m=float(args.robot_radius_m),
                                target_xy=active_survey_xy,
                                observed_from_xy=[
                                    (float(p.x), float(p.y)) for p in stitch_state["poses"]
                                ],
                                robot_theta_deg=float(transit_pose.theta_deg),
                                avoid_face_xy=_active_frontier_blacklist(),
                                prefer_face_xy=committed_frontier_face,
                            )
                            if str(transit_plan.get("status")) not in ("ok", "survey", "observe"):
                                transit_stop_reason = "replan_" + str(transit_plan.get("status"))
                                break
                        if transit_stop_reason == "arrived":
                            # Fold the face-the-target turn into the arrival so
                            # the checkpoint capture is already aimed at the
                            # unknown area — no separate face-then-capture cycle.
                            face_dx_m = float(transit_plan["face_xy"][0]) - float(transit_pose.x)
                            face_dy_m = float(transit_plan["face_xy"][1]) - float(transit_pose.y)
                            face_delta_deg = _normalize_angle_deg(
                                math.degrees(math.atan2(face_dy_m, face_dx_m))
                                - float(transit_pose.theta_deg)
                            )
                            if math.hypot(face_dx_m, face_dy_m) > 0.35 and abs(face_delta_deg) > 35.0:
                                face_turn_deg = max(15.0, min(85.0, abs(face_delta_deg)))
                                face_sign = 1.0 if face_delta_deg >= 0.0 else -1.0
                                print(
                                    "[wander] arrived; facing the target area before the capture "
                                    f"(delta={face_delta_deg:.1f}deg)"
                                )
                                face_meta = _turn_with_arc_tracking(
                                    robot=robot,
                                    feed=feed,
                                    transformed_sets=stitch_state["transformed_sets"],
                                    start_pose=transit_pose,
                                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                                    resolution_m=float(args.stitch_resolution_m),
                                    target_turn_deg=float(face_turn_deg),
                                    direction_sign=float(face_sign),
                                    turn_speed=float(args.turn_speed),
                                    turn_burst_s=float(args.turn_burst_s),
                                    turn_settle_s=float(args.turn_settle_s),
                                    forward_angle_deg=float(args.forward_angle_deg),
                                    valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                                    invert_lateral_axis=bool(args.invert_lateral_axis),
                                    max_distance_m=float(args.max_distance_m),
                                    min_range_m=float(args.min_range_m),
                                    min_confidence=int(args.min_confidence),
                                    stop_tolerance_deg=min(8.0, float(args.stop_tolerance_deg)),
                                    max_bursts=int(args.max_turn_bursts),
                    hazard_monitor=hazard_monitor,
                                )
                                face_turned_deg = float(face_meta["turned_deg"])
                                face_lever_dx, face_lever_dy = _turn_lever_arm_local_delta(
                                    face_turned_deg,
                                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                                )
                                face_turn_hint = MotionHint(
                                    kind="turn",
                                    expected_dx_local_m=float(face_lever_dx),
                                    expected_dy_local_m=float(face_lever_dy),
                                    expected_dtheta_deg=float(face_turned_deg),
                                    search_xy_m=float(args.turn_search_xy_m),
                                    search_theta_window_deg=min(
                                        80.0, 25.0 + 8.0 * float(face_meta["missed_updates"])
                                    ),
                                    label=f"face_turn_{capture_index:02d}",
                                )
                                transit_pose = _advance_pose(transit_pose, face_turn_hint)
                                transit_hint = (
                                    _compose_motion_hints(transit_hint, face_turn_hint)
                                    if transit_hint is not None
                                    else face_turn_hint
                                )
                        print(
                            "[wander] transit complete "
                            f"(legs={transit_legs}, traveled={transit_traveled_m:.2f}m, "
                            f"reason={transit_stop_reason}); capturing checkpoint"
                        )
                        motion_hint = transit_hint
                        if (
                            transit_stop_reason == "blocked_or_lock"
                            and transit_hint is not None
                            and transit_traveled_m < 0.15
                            and math.hypot(
                                float(transit_hint.expected_dx_local_m),
                                float(transit_hint.expected_dy_local_m),
                            )
                            < 0.15
                            and abs(float(transit_hint.expected_dtheta_deg)) < 10.0
                        ):
                            # A hazard stop after barely any motion: a full
                            # checkpoint capture would re-anchor nothing and
                            # just stacks rings at the same spot.
                            checkpoint_skippable = True
                        settle_s = float(args.move_settle_s)
                        if motion_hint is None:
                            # Degenerate: nothing moved (immediate align abort);
                            # commit a scan-in-place so the cycle stays sound.
                            motion_hint = MotionHint(
                                kind="drive",
                                expected_dx_local_m=0.0,
                                expected_dy_local_m=0.0,
                                expected_dtheta_deg=0.0,
                                search_xy_m=0.30,
                                search_theta_window_deg=12.0,
                                label=f"transit_hold_{capture_index:02d}",
                            )
            elif should_turn:
                pass
            elif bootstrap_scan_active:
                # Aim each bootstrap turn at the CLOSEST heading bin the map has
                # not covered yet, instead of blindly stepping 90deg the same
                # way — stiction makes actual turn sizes erratic, so blind steps
                # revisit the same headings while leaving one bin unseen.
                unseen_bins = [
                    b for b in range(rotation_coverage_bin_count) if b not in rotation_coverage_bins_seen
                ]
                if unseen_bins:
                    bin_width_deg = 360.0 / float(rotation_coverage_bin_count)
                    current_heading_deg = float(current_live_pose.theta_deg)
                    bin_deltas = [
                        (_normalize_angle_deg((b + 0.5) * bin_width_deg - current_heading_deg), b)
                        for b in unseen_bins
                    ]
                    target_delta_deg, target_bin = min(bin_deltas, key=lambda item: abs(item[0]))
                    chosen_direction_sign = 1.0 if target_delta_deg >= 0.0 else -1.0
                    chosen_turn_deg = max(25.0, min(55.0, abs(float(target_delta_deg))))
                    print(
                        "[wander] bootstrap scan capture targeting unseen heading bin "
                        f"(capture={capture_index}, bin={target_bin}, heading={current_heading_deg:.1f}deg, "
                        f"turn={chosen_turn_deg:.1f}deg {'ccw' if chosen_direction_sign >= 0.0 else 'cw'}, "
                        f"unseen={sorted(unseen_bins)})"
                    )
                else:
                    print(
                        "[wander] bootstrap scan capture "
                        f"(capture={capture_index}, target={float(args.turn_deg):.1f}deg {args.turn_direction})"
                    )
                should_turn = True
                turn_reason = "bootstrap_scan"
            elif wander_mode == "turn_only":
                print(
                    "[wander] turn-only mode active; rotating for capture "
                    f"(frame_id={live_frame_id}, blocked_points={blocked_points})"
                )
                should_turn = True
                turn_reason = "turn_only"
            elif blocked:
                print(
                    "[wander] stop box occupied; rotating in place "
                    f"(frame_id={live_frame_id}, blocked_points={blocked_points})"
                )
                consecutive_blocked_cycles += 1
                reverse_escape_done = False
                if consecutive_blocked_cycles >= 3:
                    # Blocked at every heading we try: the robot is wedged in a
                    # pocket. Rotating cannot free it — back out along the path
                    # it came in on, and stop chasing whatever lured it here.
                    wedged_events += 1
                    if active_explore_target is not None and active_explore_target.source != "retreat":
                        _blacklist_target(active_explore_target, "wedged_pocket")
                        active_explore_target = None
                        active_explore_target_stall_count = 0
                        active_explore_target_last_distance_m = None
                        active_explore_target_blocked_count = 0
                    force_live_frontier_cycles = max(force_live_frontier_cycles, 3)
                    consecutive_blocked_cycles = 0
                    reverse_result = _reverse_escape(
                        robot=robot,
                        feed=feed,
                        transformed_sets=stitch_state["transformed_sets"],
                        start_pose=current_live_pose,
                        resolution_m=float(args.stitch_resolution_m),
                        robot_radius_m=float(args.robot_radius_m),
                        lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                        reverse_speed=max(float(args.min_effective_move_speed), float(args.move_speed) * 0.9),
                        burst_s=min(1.0, float(args.move_burst_s)),
                        bursts=2,
                        forward_angle_deg=float(args.forward_angle_deg),
                        valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                        invert_lateral_axis=bool(args.invert_lateral_axis),
                        max_distance_m=float(args.max_distance_m),
                        min_range_m=float(args.min_range_m),
                        min_confidence=int(args.min_confidence),
                        allow_blind=failed_reverse_escapes >= 1,
                    )
                    if reverse_result is None:
                        failed_reverse_escapes += 1
                    else:
                        failed_reverse_escapes = 0
                    if reverse_result is not None:
                        reverse_pose, reverse_meta = reverse_result
                        rev_ddx_world = float(reverse_pose.x) - float(current_live_pose.x)
                        rev_ddy_world = float(reverse_pose.y) - float(current_live_pose.y)
                        rev_theta0_rad = math.radians(float(current_live_pose.theta_deg))
                        rev_cos = math.cos(rev_theta0_rad)
                        rev_sin = math.sin(rev_theta0_rad)
                        motion_hint = MotionHint(
                            kind="drive",
                            expected_dx_local_m=rev_cos * rev_ddx_world + rev_sin * rev_ddy_world,
                            expected_dy_local_m=-rev_sin * rev_ddx_world + rev_cos * rev_ddy_world,
                            expected_dtheta_deg=_normalize_angle_deg(
                                float(reverse_pose.theta_deg) - float(current_live_pose.theta_deg)
                            ),
                            search_xy_m=0.45 if bool(reverse_meta.get("locked")) else 0.90,
                            search_theta_window_deg=18.0 if bool(reverse_meta.get("locked")) else 30.0,
                            label=f"reverse_escape_{capture_index:02d}",
                        )
                        should_turn = False
                        reverse_escape_done = True
                        settle_s = float(args.move_settle_s)
                    else:
                        print(
                            "[wander] no room to reverse; escaping toward the widest live opening "
                            f"(force_live_frontier_cycles={force_live_frontier_cycles})"
                        )
                    if wedged_events >= 2:
                        # Local greedy escapes aren't working: this whole area
                        # is a clutter pocket. Retreat along the robot's own
                        # pose trail — that path was driven once, so it is
                        # traversable by construction — and resume exploring
                        # from open space instead of thrashing here.
                        retreat_pose = next(
                            (
                                pose
                                for pose in reversed(list(stitch_state["poses"])[:-1])
                                if math.hypot(
                                    float(pose.x) - float(current_live_pose.x),
                                    float(pose.y) - float(current_live_pose.y),
                                )
                                >= 0.90
                            ),
                            None,
                        )
                        if retreat_pose is not None:
                            active_explore_target = ExploreTarget(
                                world_x_m=float(retreat_pose.x),
                                world_y_m=float(retreat_pose.y),
                                source="retreat",
                                seeded_capture_index=int(capture_index),
                            )
                            active_explore_target_stall_count = 0
                            active_explore_target_last_distance_m = None
                            active_explore_target_blocked_count = 0
                            force_live_frontier_cycles = 0
                            wedged_events = 0
                            print(
                                "[wander] wedged repeatedly; retreating along own trail to "
                                f"({retreat_pose.x:.3f}, {retreat_pose.y:.3f})"
                            )
                if not reverse_escape_done:
                    should_turn = True
                    blocked_frontier = live_frontier_choice or planning_frontier_choice
                    if blocked_frontier is not None:
                        chosen_direction_sign = 1.0 if blocked_frontier.delta_deg >= 0.0 else -1.0
                        chosen_turn_deg = max(12.0, min(float(args.turn_deg), abs(float(blocked_frontier.delta_deg))))
                        turn_reason = "blocked_frontier"
                    else:
                        turn_reason = "blocked"
            elif (
                wander_mode == "smart"
                and rotation_coverage_complete
                and not force_drive_after_turn
                and (
                    active_explore_target is None
                    or int(active_explore_target.coarse_turns_used) <= 0
                )
                and planning_frontier_choice is not None
                and abs(float(planning_frontier_choice.delta_deg)) >= 115.0
            ):
                chosen_direction_sign = 1.0 if planning_frontier_choice.delta_deg >= 0.0 else -1.0
                chosen_turn_deg = max(22.0, min(55.0, abs(float(planning_frontier_choice.delta_deg)) - 32.0))
                should_turn = True
                turn_reason = "frontier_seek"
                print(
                    "[wander] rotation coverage is complete; performing one coarse frontier seek turn "
                    f"(frame_id={live_frame_id}, source={planning_frontier_source}, "
                    f"delta={planning_frontier_choice.delta_deg:.1f}deg, "
                    f"distance={planning_frontier_choice.mean_distance_m:.2f}m, "
                    f"width={planning_frontier_choice.width_deg:.1f}deg)"
                )
            elif (
                wander_mode == "smart"
                and not force_drive_after_turn
                and not rotation_coverage_complete
                and planning_frontier_choice is not None
                and abs(float(planning_frontier_choice.delta_deg)) >= float(args.frontier_align_threshold_deg)
            ):
                chosen_direction_sign = 1.0 if planning_frontier_choice.delta_deg >= 0.0 else -1.0
                chosen_turn_deg = max(15.0, min(float(args.turn_deg), abs(float(planning_frontier_choice.delta_deg))))
                should_turn = True
                turn_reason = "frontier_align"
                print(
                    "[wander] frontier is far off heading; aligning before drive "
                    f"(frame_id={live_frame_id}, source={planning_frontier_source}, "
                    f"delta={planning_frontier_choice.delta_deg:.1f}deg, "
                    f"distance={planning_frontier_choice.mean_distance_m:.2f}m, "
                    f"width={planning_frontier_choice.width_deg:.1f}deg)"
                )
            else:
                print(
                    "[wander] forward path is clear; driving sequence "
                    f"(frame_id={live_frame_id}, blocked_points={blocked_points})"
                )
                steer_theta_vel = _compute_drive_steer_theta_vel(
                    planning_frontier_choice,
                    steer_gain=float(args.frontier_drive_steer_gain),
                    steer_max=float(args.max_drive_steer_theta_vel),
                    steer_deadband_deg=float(args.drive_steer_deadband_deg),
                )
                planned_bursts = max(1, int(args.drive_bursts_per_capture))
                if planning_frontier_choice is not None:
                    if planning_frontier_choice.mean_distance_m >= 2.4:
                        planned_bursts += 1
                    if planning_frontier_choice.mean_distance_m >= 3.4 and planning_frontier_choice.width_deg >= 24.0:
                        planned_bursts += 1
                planned_bursts = min(planned_bursts, max(1, int(args.max_drive_bursts_per_capture)))
                print(
                    "[wander] drive plan "
                    f"(planned_bursts={planned_bursts}, frontier_delta="
                    f"{None if planning_frontier_choice is None else round(float(planning_frontier_choice.delta_deg), 1)}, "
                    f"frontier_distance={None if planning_frontier_choice is None else round(float(planning_frontier_choice.mean_distance_m), 2)}, "
                    f"frontier_source={planning_frontier_source}, "
                    f"drive_mode={'frontier_follow' if planning_frontier_choice is not None else 'straight_burst'}, "
                    f"steer_theta_vel={steer_theta_vel:.3f})"
                )
                drive_start_pose = current_live_pose
                drive_tracked_pose, drive_track_meta = _drive_with_tracking(
                    robot=robot,
                    feed=feed,
                    zone_cfg=zone_cfg,
                    transformed_sets=stitch_state["transformed_sets"],
                    start_pose=drive_start_pose,
                    resolution_m=float(args.stitch_resolution_m),
                    forward_speed=float(args.move_speed),
                    min_effective_move_speed=float(args.min_effective_move_speed),
                    burst_s=float(args.move_burst_s),
                    burst_count=int(planned_bursts),
                    inter_burst_pause_s=float(args.inter_burst_pause_s),
                    steer_theta_vel=float(steer_theta_vel),
                    forward_angle_deg=float(args.forward_angle_deg),
                    valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                    invert_lateral_axis=bool(args.invert_lateral_axis),
                    max_distance_m=float(args.max_distance_m),
                    min_range_m=float(args.min_range_m),
                    min_confidence=int(args.min_confidence),
                    hazard_monitor=hazard_monitor,
                    map_side_guard=_map_side_guard,
                    forward_guard=_exit_ratchet_guard,
                    pose_trusted=not pose_lost,
                    frame_recorder=_record_eye_frames,
                    imu_yaw_fn=_drive_hold_yaw_deg,
                    yaw_hold_gain=_drive_yaw_hold_gain,
                    yaw_hold_max=float(args.drive_yaw_hold_max),
                )
                drive_stopped_by_block = (
                    bool(drive_track_meta["stopped_by_block"])
                    or bool(drive_track_meta["lock_lost"])
                    or bool(drive_track_meta.get("stopped_by_hazard"))
                )
                if bool(drive_track_meta.get("stopped_by_hazard")):
                    # Stamp the freshly measured geometry at the tracked
                    # stop pose (see transit path): every hazard stop teaches
                    # the planner the boundary, not just investigations.
                    _dump_stop_debug(drive_tracked_pose, "forward_hazard_stop")
                    _stamp_confirmed_edges(drive_tracked_pose, newest_only=True)
                    if plan_status in ("ok", "survey", "observe"):
                        _strike_frontier_if_reached(
                            frontier_plan["face_xy"], drive_tracked_pose, "hazard_stop"
                        )
                # Motion hint straight from the tracked poses — no commanded-
                # speed or wheel-feedback guessing.
                drive_ddx_world = float(drive_tracked_pose.x) - float(drive_start_pose.x)
                drive_ddy_world = float(drive_tracked_pose.y) - float(drive_start_pose.y)
                drive_theta0_rad = math.radians(float(drive_start_pose.theta_deg))
                drive_cos = math.cos(drive_theta0_rad)
                drive_sin = math.sin(drive_theta0_rad)
                drive_hint_dx_local_m = drive_cos * drive_ddx_world + drive_sin * drive_ddy_world
                drive_hint_dy_local_m = -drive_sin * drive_ddx_world + drive_cos * drive_ddy_world
                drive_hint_dtheta_deg = _normalize_angle_deg(
                    float(drive_tracked_pose.theta_deg) - float(drive_start_pose.theta_deg)
                )
                # LIDAR-ONLY: the tracked heading delta is the hint even after
                # lock loss (the wider search window below owns that case). The
                # gyro never writes into motion hints — advisory only.
                drive_lock_lost = bool(drive_track_meta["lock_lost"]) or int(
                    drive_track_meta["missed_updates"]
                ) > 0
                if drive_lock_lost:
                    drive_search_xy_m = 0.80
                    drive_theta_window_deg = 30.0
                else:
                    drive_search_xy_m = 0.40
                    drive_theta_window_deg = 16.0
                print(
                    "[wander] forward sequence complete "
                    f"(elapsed={float(drive_track_meta['elapsed_s']):.2f}s "
                    f"bursts={int(drive_track_meta['bursts_completed'])} "
                    f"stopped_by_block={bool(drive_track_meta['stopped_by_block'])} "
                    f"lock_lost={bool(drive_track_meta['lock_lost'])} "
                    f"blocked_points={int(drive_track_meta['blocked_points'])} "
                    f"tracked_move=({drive_hint_dx_local_m:.3f}m, {drive_hint_dy_local_m:.3f}m, "
                    f"{drive_hint_dtheta_deg:.1f}deg))"
                )
                force_drive_after_turn = False
                if dead_reck_theta_deg is not None:
                    dead_reck_theta_deg += float(drive_hint_dtheta_deg)
                    dead_reck_slack_deg += 5.0
                    if dead_reck_slack_deg > 100.0:
                        dead_reck_theta_deg = None
                drive_checkpoints_since_scan += 1
                if (
                    bool(drive_track_meta.get("stopped_by_hazard"))
                    and math.hypot(drive_hint_dx_local_m, drive_hint_dy_local_m) < 0.15
                    and abs(drive_hint_dtheta_deg) < 10.0
                ):
                    checkpoint_skippable = True
                if not drive_stopped_by_block:
                    # Only a drive that actually got somewhere disarms the
                    # wedge detection. A one-burst drive straight into the stop
                    # box must keep counting toward "we are stuck here", or the
                    # blocked-turn-drive dance resets the counter forever and
                    # the reverse/retreat escapes never fire.
                    consecutive_blocked_cycles = 0
                    failed_reverse_escapes = 0
                    rotate_away_lock_losses = 0
                    wedged_events = 0
                if wander_mode == "scan_turn_every_capture":
                    pending_turn_reason = "post_drive_scan"
                    pending_turn_direction_sign = float(direction_sign)
                    pending_turn_deg = float(args.turn_deg)
                    should_turn = False
                    turn_reason = "drive_only"
                elif drive_stopped_by_block:
                    should_turn = False
                    if active_explore_target is not None:
                        active_explore_target_blocked_count += 1
                        print(
                            "[wander] stitched-map target blocked during drive "
                            f"(source={active_explore_target.source}, "
                            f"seed_capture={active_explore_target.seeded_capture_index}, "
                            f"blocked_count={active_explore_target_blocked_count})"
                        )
                        force_live_frontier_cycles = max(force_live_frontier_cycles, 2)
                        if active_explore_target_blocked_count >= 3:
                            print(
                                "[wander] abandoning stitched-map target after repeated blocked drives "
                                f"(source={active_explore_target.source}, "
                                f"seed_capture={active_explore_target.seeded_capture_index})"
                            )
                            if active_explore_target.source != "retreat":
                                _blacklist_target(active_explore_target, "blocked_drives")
                            active_explore_target = None
                            active_explore_target_stall_count = 0
                            active_explore_target_last_distance_m = None
                            active_explore_target_blocked_count = 0
                    recovery_frontier = live_frontier_choice or planning_frontier_choice
                    if recovery_frontier is not None:
                        pending_turn_direction_sign = 1.0 if recovery_frontier.delta_deg >= 0.0 else -1.0
                        pending_turn_deg = max(18.0, min(float(args.turn_deg), abs(float(recovery_frontier.delta_deg))))
                        pending_turn_reason = "drive_blocked_frontier"
                    else:
                        pending_turn_direction_sign = float(direction_sign)
                        pending_turn_deg = float(args.turn_deg)
                        pending_turn_reason = "drive_blocked"
                    turn_reason = "drive_only"
                    drive_checkpoints_since_scan = 0
                elif (
                    wander_mode == "smart"
                    and not rotation_coverage_complete
                    and int(args.scan_turn_interval) > 0
                    and drive_checkpoints_since_scan >= int(args.scan_turn_interval)
                ):
                    should_turn = False
                    if planning_frontier_choice is not None and abs(planning_frontier_choice.delta_deg) >= float(args.frontier_align_threshold_deg):
                        pending_turn_direction_sign = 1.0 if planning_frontier_choice.delta_deg >= 0.0 else -1.0
                        pending_turn_deg = max(15.0, min(float(args.turn_deg), abs(float(planning_frontier_choice.delta_deg))))
                        pending_turn_reason = "periodic_frontier_scan"
                    else:
                        pending_turn_direction_sign = float(direction_sign)
                        pending_turn_deg = float(args.turn_deg)
                        pending_turn_reason = "periodic_scan"
                    turn_reason = "drive_only"
                    drive_checkpoints_since_scan = 0
                else:
                    should_turn = False
                    turn_reason = "drive_only"
                    if force_live_frontier_cycles > 0:
                        force_live_frontier_cycles -= 1
                motion_hint = MotionHint(
                    kind="drive",
                    expected_dx_local_m=float(drive_hint_dx_local_m),
                    expected_dy_local_m=float(drive_hint_dy_local_m),
                    expected_dtheta_deg=float(drive_hint_dtheta_deg),
                    search_xy_m=float(drive_search_xy_m),
                    search_theta_window_deg=float(drive_theta_window_deg),
                    label=f"drive_capture_{capture_index:02d}",
                )
                print(
                    "[wander] drive capture hint "
                    f"(pose_source=lidar_tracked, search_xy_m={drive_search_xy_m:.3f}, "
                    f"expected_dx_local_m={drive_hint_dx_local_m:.3f}, "
                    f"expected_dy_local_m={drive_hint_dy_local_m:.3f}, "
                    f"expected_dtheta_deg={drive_hint_dtheta_deg:.1f})"
                )
                settle_s = float(args.move_settle_s)

            if should_turn:
                # SIDE-COLLISION GUARD for rotation: the collision that
                # motivated this (field 2026-07-17) was a commanded turn
                # BESIDE a mapped table edge — "no rotation progress" while
                # the shoulder ground along the tabletop. No live sensor
                # covers the sides at tabletop height; the map does. Too
                # close to a mapped cell -> don't rotate here, back off
                # first and let the planner re-approach with clearance.
                turn_guard_prox = _nearest_mapped_elevated_m(current_live_pose)
                if (
                    turn_guard_prox is not None
                    and turn_guard_prox < float(args.robot_radius_m) + 0.04
                ):
                    print(
                        f"[wander] mapped elevated cell {turn_guard_prox:.2f}m from the body — "
                        "skipping rotation beside it and backing off first (side-collision guard)"
                    )
                    should_turn = False
                    turn_guard_retreat = _attempt_reverse_escape_hint(current_live_pose, True)
                    if turn_guard_retreat is not None:
                        motion_hint = turn_guard_retreat
                        settle_s = float(args.move_settle_s)
            if should_turn:
                turn_cap_deg = 85.0
                if (
                    len(stitch_state["poses"]) < 4
                    or len(rotation_coverage_bins_seen) < rotation_coverage_bin_count
                ) and turn_reason != "reanchor_turn_back":
                    # Until the map covers the full rotation (all 4 coverage
                    # BINS — not the rotation_coverage_complete flag, which
                    # the bootstrap budget force-sets with bins at [0,1]/4;
                    # field run 12: that lie disarmed this cap and every
                    # 85deg face_frontier turn into unmapped bearings lost
                    # lock), an 85deg turn can rotate the view past the
                    # map's angular edge and overlap collapses. 55deg keeps
                    # every solve anchored in mapped bearings; larger goals
                    # just take an extra capture.
                    turn_cap_deg = 55.0
                    if len(stitch_state["poses"]) < 2:
                        # Against the founding snapshot alone, even 55deg
                        # loses lock in sparse spots (run 11: lock lost at
                        # 46.5deg on the very first turn). 40deg keeps
                        # ~140deg of the founding view in frame.
                        turn_cap_deg = 40.0
                if float(chosen_turn_deg) > turn_cap_deg:
                    # A capture must land before the view rotates into mostly
                    # unmapped territory: with a ~180deg FOV, 55deg per capture
                    # keeps >=125deg of the previous view in frame, which keeps
                    # solve scores strong. Larger goals just take two captures.
                    print(
                        f"[wander] capping turn at {turn_cap_deg:.0f}deg per capture to keep map overlap strong "
                        f"(requested={float(chosen_turn_deg):.1f}deg)"
                    )
                    chosen_turn_deg = turn_cap_deg
                print(
                    "[wander] rotating before stitched capture "
                    f"(reason={turn_reason}, target={float(chosen_turn_deg):.1f}deg "
                    f"{'ccw' if chosen_direction_sign >= 0.0 else 'cw'})"
                )
                # Anchor tracking at the dead-reckoned pose INCLUDING any
                # pending (discarded-capture) motion — after a lock-lost
                # discard the robot is physically far from the last stitched
                # pose, and anchoring there makes every retry solve miss.
                # (If live relocalization already re-anchored this cycle, the
                # pending motion is baked into current_live_pose and will be
                # dropped at composition time — don't double-count it here.)
                turn_start_pose = (
                    _advance_pose(current_live_pose, pending_motion_hint)
                    if (pending_motion_hint is not None and not live_pose_accepted)
                    else current_live_pose
                )
                imu_yaw_before_turn = imu_yaw.deg_fresh() if imu_yaw is not None else None
                turn_meta = _turn_with_arc_tracking(
                    robot=robot,
                    feed=feed,
                    transformed_sets=stitch_state["transformed_sets"],
                    start_pose=turn_start_pose,
                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                    resolution_m=float(args.stitch_resolution_m),
                    target_turn_deg=float(chosen_turn_deg),
                    direction_sign=float(chosen_direction_sign),
                    turn_speed=float(args.turn_speed),
                    turn_burst_s=float(args.turn_burst_s),
                    turn_settle_s=float(args.turn_settle_s),
                    forward_angle_deg=float(args.forward_angle_deg),
                    valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                    invert_lateral_axis=bool(args.invert_lateral_axis),
                    max_distance_m=float(args.max_distance_m),
                    min_range_m=float(args.min_range_m),
                    min_confidence=int(args.min_confidence),
                    stop_tolerance_deg=min(8.0, float(args.stop_tolerance_deg)),
                    max_bursts=int(args.max_turn_bursts),
                    hazard_monitor=hazard_monitor,
                )
                tracked_turn_deg = float(turn_meta["turned_deg"])
                turn_lock_lost = int(turn_meta["completed"]) != 1
                # Gyro-measured rotation across the whole turn, bracketed by FRESH
                # samples (a mid-turn stale sample under-reports — it once read
                # -8deg for a real -55deg turn). Unlike the lidar arc tracker, the
                # IMU still measures the rotation when scan-match lock is lost
                # mid-turn (the exact case that used to strand the dead-reckoned
                # heading and let a room-symmetric wrong mode win).
                imu_turn_deg: float | None = None
                if imu_yaw is not None and imu_yaw_before_turn is not None:
                    imu_after = imu_yaw.deg_fresh()
                    if imu_after is not None:
                        # Raw difference: the published yaw is continuous, so this
                        # is exact even for rotations beyond +-180deg.
                        imu_turn_deg = float(imu_after) - float(imu_yaw_before_turn)
                    if not turn_lock_lost:
                        imu_yaw.note_tracked_turn(tracked_turn_deg, imu_turn_deg)
                # LIDAR-ONLY mapping (field 2026-07-18, run 4): the IMU is
                # ADVISORY, never an author. An earlier version substituted the
                # gyro delta into the motion hint on lock-lost turns; those
                # values got composed into append chains and (via marginal
                # guided restores) ghosted the map. The lidar's tracked value —
                # even when it under-counts after lock loss — is what the append
                # gates were tuned around; the gyro is only shown in the log and
                # keeps the recovery tiebreaker's general-idea heading fresh.
                measured_signed_turn_deg = tracked_turn_deg
                # CALIBRATED BLIND-TURN DEAD-RECKON: when the lidar LOST LOCK
                # mid-turn, the robot kept rotating during the untracked bursts,
                # so `tracked_turn_deg` under-counts (this is the exact case the
                # block below marks pose-lost for). The motion calibration gives
                # an honest estimate of that untracked rotation (untracked bursts
                # x commanded-per-burst x the measured rotation scale), so the
                # RECOVERY relocalization gets seeded at the right heading instead
                # of a stale under-count. Safe by construction: a lock-lost turn
                # already freezes the map READ-ONLY, so this touches only the
                # recovery seed and cannot ghost the map (unlike the old IMU
                # substitution). Never reduces the tracked value; never exceeds
                # what was commanded.
                if int(turn_meta.get("lock_lost", 0)) == 1 and abs(motion_calibration.rotation_scale - 1.0) > 1e-6:
                    deg_per_burst = math.degrees(float(args.turn_speed) * float(args.turn_burst_s))
                    untracked_deg = (
                        float(turn_meta["missed_updates"]) * deg_per_burst * float(motion_calibration.rotation_scale)
                    )
                    dr_estimate = tracked_turn_deg + float(chosen_direction_sign) * untracked_deg
                    commanded_signed = float(chosen_direction_sign) * float(chosen_turn_deg)
                    if abs(dr_estimate) > abs(commanded_signed):
                        dr_estimate = commanded_signed   # cannot exceed the commanded turn
                    if abs(dr_estimate) > abs(tracked_turn_deg) + 1.0:
                        print(
                            "[wander] blind-turn dead-reckon: lidar tracked "
                            f"{tracked_turn_deg:+.1f}deg then LOST LOCK; calibration estimates "
                            f"~{dr_estimate:+.1f}deg actually turned "
                            f"(rotation x{motion_calibration.rotation_scale:.2f}) — seeding recovery "
                            "with that instead of the under-count"
                        )
                        measured_signed_turn_deg = dr_estimate
                print(
                    "[wander] turn complete "
                    f"(tracked={tracked_turn_deg:+.1f}deg, "
                    f"imu={'n/a' if imu_turn_deg is None else f'{imu_turn_deg:+.1f}deg'}, "
                    f"final_theta={turn_meta['final_theta_deg']:.1f}deg, "
                    f"missed_updates={int(turn_meta['missed_updates'])}, completed={int(turn_meta['completed'])})"
                )
                if dead_reck_theta_deg is not None:
                    # Advance the general-idea anchor by the TRACKED turn; the
                    # untracked remainder (plus a per-turn margin) widens the
                    # slack. A chain of blind turns widens it past usefulness and
                    # recovery falls back to score-only. (When the gyro feed is
                    # live, the recovery judge re-derives this heading from the
                    # gyro anchor anyway — this path is the lidar-only fallback.)
                    dead_reck_theta_deg += measured_signed_turn_deg
                    dead_reck_slack_deg += 6.0 + max(
                        0.0, float(chosen_turn_deg) - abs(measured_signed_turn_deg)
                    )
                    if dead_reck_slack_deg > 100.0:
                        dead_reck_theta_deg = None
                if int(turn_meta["completed"]) != 1:
                    # Lock-lost turn: the robot physically rotated an UNKNOWN
                    # amount (bursts fire whether or not the solve tracks
                    # them). The measured hint under-counts, so every solve
                    # anchored on it is suspect — latch lost.
                    if not pose_lost:
                        pose_lost = True
                        print(
                            "[wander] pose integrity LOST (turn ended with tracking lock lost); "
                            "map is READ-ONLY until a strong relocalization re-proves the pose"
                        )
                # The capture solve only needs to confirm/refine the tracked
                # heading; widen its window a bit for each burst the tracker
                # could not update on.
                turn_capture_theta_window_deg = min(
                    80.0, 25.0 + 8.0 * float(turn_meta["missed_updates"])
                )
                print(
                    "[wander] turn motion hint "
                    f"(requested_dtheta_deg={float(chosen_turn_deg) * float(chosen_direction_sign):.1f}, "
                    f"measured_dtheta_deg={measured_signed_turn_deg:.1f})"
                )
                if turn_reason in {
                    "blocked",
                    "blocked_frontier",
                    "drive_blocked",
                    "drive_blocked_frontier",
                    "bootstrap_scan",
                    "turn_only",
                    "frontier_seek",
                    "frontier_align",
                    "periodic_frontier_scan",
                    "periodic_scan",
                }:
                    drive_checkpoints_since_scan = 0
                if turn_reason in {"frontier_align", "periodic_frontier_scan", "frontier_seek"}:
                    force_drive_after_turn = True
                    if active_explore_target is not None and turn_reason == "frontier_seek":
                        active_explore_target.coarse_turns_used += 1
                lever_dx_local_m, lever_dy_local_m = _turn_lever_arm_local_delta(
                    float(measured_signed_turn_deg),
                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                )
                motion_hint = MotionHint(
                    kind="turn",
                    expected_dx_local_m=float(lever_dx_local_m),
                    expected_dy_local_m=float(lever_dy_local_m),
                    expected_dtheta_deg=float(measured_signed_turn_deg),
                    search_xy_m=float(args.turn_search_xy_m),
                    search_theta_window_deg=float(args.turn_theta_window_deg),
                    label=f"turn_capture_{capture_index:02d}",
                )
                print(
                    "[wander] turn lever-arm hint "
                    f"(expected_dx_local={lever_dx_local_m:.3f}m, expected_dy_local={lever_dy_local_m:.3f}m, "
                    f"offset={float(args.lidar_offset_forward_m):.3f}m)"
                )
                settle_s = float(args.capture_settle_s)

            if (
                pending_motion_hint is None
                and str(motion_hint.label or "").startswith(("hazard_hold", "edge_hold"))
                and abs(float(motion_hint.expected_dx_local_m)) < 0.02
                and abs(float(motion_hint.expected_dy_local_m)) < 0.02
                and abs(float(motion_hint.expected_dtheta_deg)) < 2.0
            ):
                # Nothing moved since the last capture (a hold denied every
                # recovery action): an identical capture from an identical
                # pose teaches the map NOTHING. Wait briefly and replan
                # instead of stacking duplicate captures at one spot.
                print(
                    "[wander] nothing moved since the last capture (hold cycle); "
                    "skipping the duplicate capture and replanning"
                )
                time.sleep(max(0.5, float(settle_s)))
                continue
            if checkpoint_skippable and pending_motion_hint is None:
                # Hazard stop after <0.15m of motion: skip the checkpoint
                # capture (it would re-anchor nothing) and carry the tracked
                # motion into the NEXT capture's expectation instead.
                print(
                    "[wander] short hazard stop; skipping the checkpoint capture "
                    "and carrying the tracked motion forward"
                )
                pending_motion_hint = motion_hint
                time.sleep(0.3)
                continue

            # REDUNDANT-VANTAGE GATE: once the map is mature, a checkpoint
            # capture is only worth taking from a genuinely NEW viewpoint.
            # Field 2026-07-18 run 20: 22 captures piled up inside a ~0.5m
            # patch — every micro-transit (0.05m stop-box interruptions) and
            # every 55deg face-turn step stopped to append another snapshot
            # of a viewpoint the map already owned. Skip the capture, chain
            # the tracked motion, and keep moving toward the target vantage:
            # drive there, shoot once, plan the next vantage. Appends resume
            # automatically the moment the robot reaches unclaimed ground
            # (or whenever pose trust degrades — recovery needs anchors).
            if (
                not pose_lost
                and rotation_coverage_complete
                and consecutive_append_discards == 0
                and len(stitch_state["poses"]) >= 6
                and redundant_skip_streak < 6
            ):
                redundancy_hint = (
                    motion_hint
                    if pending_motion_hint is None
                    else _compose_motion_hints(pending_motion_hint, motion_hint)
                )
                chain_translation_m = math.hypot(
                    float(redundancy_hint.expected_dx_local_m),
                    float(redundancy_hint.expected_dy_local_m),
                )
                if chain_translation_m <= 1.2 and abs(
                    float(redundancy_hint.expected_dtheta_deg)
                ) <= 120.0:
                    redundancy_pose = _advance_pose(
                        stitch_state["poses"][-1], redundancy_hint
                    )
                    covered_by = None
                    for prior_index, prior_pose in enumerate(stitch_state["poses"]):
                        if (
                            math.hypot(
                                float(prior_pose.x) - float(redundancy_pose.x),
                                float(prior_pose.y) - float(redundancy_pose.y),
                            )
                            < 0.40
                            and abs(
                                _normalize_angle_deg(
                                    float(prior_pose.theta_deg)
                                    - float(redundancy_pose.theta_deg)
                                )
                            )
                            < 60.0
                        ):
                            covered_by = prior_index
                            break
                    if covered_by is not None:
                        redundant_skip_streak += 1
                        print(
                            "[wander] vantage already covered by capture "
                            f"#{covered_by + 1} (within 0.40m/60deg); skipping the "
                            "redundant capture and continuing toward a new viewpoint "
                            f"({redundant_skip_streak}/6 before a forced refresh)"
                        )
                        if redundant_skip_streak == 3 and committed_frontier_face is not None:
                            # PIROUETTE BREAKER (field 2026-07-18: "face the
                            # unmapped area" and "forcing a turn instead of
                            # re-scanning" alternated +-85deg for six straight
                            # skipped captures at one frontier). Three redundant
                            # skips while committed to one face means this
                            # frontier CANNOT be resolved from here — observing
                            # it again teaches nothing. Strike it so the planner
                            # moves on instead of turning in place forever.
                            if elevated_block_recent > 0:
                                # ...UNLESS an elevated-edge hold caused the
                                # non-progress (field 2026-07-20: a couch against
                                # the wall left of a hallway held every forward
                                # drive, so every capture came from the same
                                # vantage — the frontier IS resolvable, the robot
                                # just needs to route around the furniture). Do
                                # NOT blacklist the doorway; steer toward the live
                                # lidar opening and let the recovery ladder rotate
                                # the nose off the furniture next cycle.
                                print(
                                    "[wander] frontier at "
                                    f"({committed_frontier_face[0]:.2f}, {committed_frontier_face[1]:.2f}) "
                                    "produced 3 redundant captures, but they were caused by an "
                                    "elevated-edge hold (furniture beside the path), not an "
                                    "unresolvable frontier — steering toward the live opening "
                                    "instead of blacklisting the doorway"
                                )
                                force_live_frontier_cycles = max(
                                    force_live_frontier_cycles, 3
                                )
                                committed_frontier_face = None
                                redundant_skip_streak = 0
                            else:
                                print(
                                    "[wander] frontier at "
                                    f"({committed_frontier_face[0]:.2f}, {committed_frontier_face[1]:.2f}) "
                                    "produced 3 straight redundant captures from this vantage — "
                                    "striking it and moving on (pirouette breaker)"
                                )
                                _strike_frontier(
                                    committed_frontier_face,
                                    "unresolvable from this vantage (redundant captures)",
                                    force=True,
                                )
                        pending_motion_hint = redundancy_hint
                        continue
            redundant_skip_streak = 0

            print(f"[wander] settling for {settle_s:.2f}s before capture")
            time.sleep(settle_s)

            try:
                frame_id, captured_frame, local_points_xy, captured_snapshot = _capture_snapshot(
                    feed=feed,
                    output_dir=snapshot_dir,
                    request_index=capture_index,
                    after_frame_id=last_frame_id,
                    forward_angle_deg=float(args.forward_angle_deg),
                    valid_angle_half_width_deg=float(args.valid_angle_half_width_deg),
                    invert_lateral_axis=bool(args.invert_lateral_axis),
                    max_distance_m=float(args.max_distance_m),
                    min_range_m=float(args.min_range_m),
                    min_confidence=int(args.min_confidence),
                    fresh_frame_timeout_s=float(args.fresh_frame_timeout_s),
                    fresh_frame_advances=int(args.fresh_frame_advances),
                    capture_config_extra={
                        "motion_hint": motion_hint.kind,
                        "motion_hint_label": motion_hint.label,
                        "expected_dx_local_m": motion_hint.expected_dx_local_m,
                        "expected_dtheta_deg": motion_hint.expected_dtheta_deg,
                    },
                )
            except LidarBoxedInError as exc:
                # BOXED IN (nose against a wall / wedged). NEVER crash — back out
                # and retry. This is the same physical situation the boxed-in start
                # handles; mid-run we have a pose, so a rear-checked reverse (immune
                # to pose error) is the cleanest escape; if the rear is blocked too,
                # rotate in place to bring open space into view.
                boxed_in_recoveries += 1
                print(
                    f"[wander] BOXED IN mid-run ({exc}) — recovering (not crashing), "
                    f"attempt {boxed_in_recoveries}/10"
                )
                _send_stop(robot)
                if boxed_in_recoveries > 10:
                    print(
                        "[wander] still boxed in after 10 recoveries — physically wedged with no "
                        "clear space around the lidar. HALTING; reposition the robot and rerun."
                    )
                    break
                _boxed_hint = _attempt_reverse_escape_hint(
                    current_live_pose, boxed_in_recoveries >= 3
                )
                if _boxed_hint is not None:
                    pending_motion_hint = _boxed_hint
                    continue
                # Rear blocked too: rotate ~90deg in place (open-loop — rotation
                # cannot collide) to bring open space into the drive cone, then retry.
                _bx_deg_per_burst = max(4.0, math.degrees(float(args.turn_speed) * float(args.turn_burst_s)))
                for _ in range(max(1, int(round(90.0 / _bx_deg_per_burst)))):
                    _execute_turn_burst(
                        robot=robot,
                        direction_sign=1.0,
                        turn_speed=float(args.turn_speed),
                        turn_burst_s=float(args.turn_burst_s),
                        turn_settle_s=float(args.turn_settle_s),
                    )
                _send_stop(robot)
                print("[wander] BOXED IN with the rear blocked too — rotated 90deg to find open space; retrying")
                continue
            boxed_in_recoveries = 0  # healthy capture landed -> clear the streak
            if len(local_points_xy) == 0:
                print(f"[wander] warning: snapshot {capture_index} contains zero local points after filtering")
            if int(frame_id) == int(last_frame_id):
                # The feed produced NO new revolution: this "capture" is the
                # previous frame again. Appending it would stitch a duplicate
                # of a frozen world (field 2026-07-11: a dead host got the
                # identical frame appended 13 times). Let the feed breaker
                # at the loop top hold the robot until data flows again.
                print(
                    f"[wander] capture returned the SAME frame (frame_id={frame_id}); "
                    "feed is stalled — discarding it and holding position"
                )
                _send_stop(robot)
                pending_motion_hint = (
                    motion_hint
                    if pending_motion_hint is None
                    else _compose_motion_hints(pending_motion_hint, motion_hint)
                )
                continue

            if pending_motion_hint is not None:
                if live_pose_accepted:
                    print(
                        "[wander] dropping pending discarded-capture motion; "
                        "live relocalization already re-anchored the pose"
                    )
                else:
                    motion_hint = _compose_motion_hints(pending_motion_hint, motion_hint)
                    print(
                        "[wander] composed pending motion from discarded capture into current hint "
                        f"(expected_dx={motion_hint.expected_dx_local_m:.3f}m, "
                        f"expected_dy={motion_hint.expected_dy_local_m:.3f}m, "
                        f"expected_dtheta={motion_hint.expected_dtheta_deg:.1f}deg)"
                    )
                pending_motion_hint = None

            motion_hints.append(motion_hint)
            rebuild_started = time.monotonic()
            print("[wander] appending stitched capture " f"(capture={capture_index}, snapshots={len(motion_hints)})")
            capture_expected_pose = _advance_pose(current_live_pose, motion_hint)
            capture_live_pose = None
            capture_live_meta = None
            append_solver_pose = None
            append_solver_meta = None
            if motion_hint.kind == "turn":
                # In-place turn: the robot center is pinned, so solve on the
                # lever-arm arc. The measured turn angle only centers a wide
                # search window — it is too unreliable to act as a hard prior.
                arc_theta_half_window_deg = min(
                    110.0,
                    float(turn_capture_theta_window_deg)
                    + (30.0 if "+" in str(motion_hint.label or "") else 0.0),
                )
                arc_result = _solve_turn_arc_pose(
                    snapshot_points_xy=captured_snapshot.points_xy,
                    transformed_sets=stitch_state["transformed_sets"],
                    previous_pose=current_live_pose,
                    lidar_offset_forward_m=float(args.lidar_offset_forward_m),
                    resolution_m=float(args.stitch_resolution_m),
                    expected_theta_deg=float(capture_expected_pose.theta_deg),
                    theta_half_window_deg=float(arc_theta_half_window_deg),
                )
                if arc_result is not None:
                    arc_solved_pose, arc_meta = arc_result
                    append_solver_pose = arc_solved_pose
                    append_solver_meta = arc_meta
                    arc_score = float(arc_meta.get("score") or -1e9)
                    solved_turn_deg = _normalize_angle_deg(
                        float(arc_solved_pose.theta_deg) - float(current_live_pose.theta_deg)
                    )
                    print(
                        "[wander] turn arc solve "
                        f"(capture={capture_index}, pose=({arc_solved_pose.x:.3f}, {arc_solved_pose.y:.3f}, "
                        f"{arc_solved_pose.theta_deg:.1f}deg), turned={solved_turn_deg:.1f}deg, "
                        f"hinted={float(motion_hint.expected_dtheta_deg):.1f}deg, score={arc_score:.3f}, "
                        f"window=+-{arc_theta_half_window_deg:.0f}deg)"
                    )
                    hint_discrepancy_deg = abs(
                        _normalize_angle_deg(solved_turn_deg - float(motion_hint.expected_dtheta_deg))
                    )
                    arc_accept_gate = float(args.min_append_score)
                    if consecutive_append_discards > 0:
                        # Previous capture was discarded: the hint anchoring
                        # this solve is suspect — demand a strong match (see
                        # the append-gate escalation below).
                        arc_accept_gate += 2.0
                    if post_recovery_strict_appends > 0:
                        arc_accept_gate += 2.0
                    if pose_lost:
                        # Map is read-only while lost: only an absolute-trust
                        # match may append (field 2026-07-17: an 11.86 arc
                        # solve with a USELESS hint — hinted=0 after lock
                        # loss — passed the raised 9.0 gate at a wrong 90deg
                        # symmetry mode and ghosted the whole map).
                        arc_accept_gate = max(arc_accept_gate, POSE_LOST_APPEND_GATE)
                        print(
                            f"[wander] pose lost: arc append requires score>={arc_accept_gate:.1f}"
                        )
                    if hint_discrepancy_deg > 30.0:
                        # The solver is overriding the tracked turn by a lot. In a
                        # square room the wrong 90-degree rotation mode scores
                        # respectably, so a big override needs strong evidence,
                        # not the standard gate. Discarding here is safe: the
                        # motion hint carries forward and the next capture solves
                        # with a wider window and more map context.
                        arc_accept_gate += 1.5
                        print(
                            "[wander] warning: arc solve disagrees strongly with the measured turn "
                            f"(discrepancy={hint_discrepancy_deg:.1f}deg); possible room-symmetry mode, "
                            f"raising gate to {arc_accept_gate:.2f}"
                        )
                    if arc_score >= arc_accept_gate:
                        capture_live_pose = arc_solved_pose
                        capture_live_meta = arc_meta
                    elif hint_discrepancy_deg <= 8.0:
                        # Rotation is mutually confirmed by geometry and tracking,
                        # so a poor absolute fit usually means the robot CENTER has
                        # drifted (skid accumulates over consecutive in-place turns
                        # and the arc constraint only allows +-10cm). Refine xy
                        # around the arc pose and re-gate at full strength instead
                        # of painting a laterally-offset scan into the map.
                        refine_result = _estimate_pose_against_stitched_map(
                            points_xy=captured_snapshot.points_xy,
                            transformed_sets=stitch_state["transformed_sets"],
                            initial_pose=arc_solved_pose,
                            resolution_m=float(args.stitch_resolution_m),
                            search_xy_m=0.35,
                            theta_window_deg=8.0,
                            max_translation_from_initial_m=0.35,
                            prior_translation_weight=1.2,
                            prior_theta_weight=0.10,
                        )
                        refined_ok = False
                        if refine_result is not None:
                            refined_pose, refined_meta = refine_result
                            refined_score = float(refined_meta.get("score") or -1e9)
                            print(
                                "[wander] turn arc xy-refine "
                                f"(capture={capture_index}, pose=({refined_pose.x:.3f}, {refined_pose.y:.3f}, "
                                f"{refined_pose.theta_deg:.1f}deg), score={refined_score:.3f}, "
                                f"arc_score={arc_score:.3f})"
                            )
                            refine_gate = float(args.min_append_score) + (
                                2.0 if consecutive_append_discards > 0 else 0.0
                            )
                            if post_recovery_strict_appends > 0:
                                refine_gate += 2.0
                            if pose_lost:
                                refine_gate = max(refine_gate, POSE_LOST_APPEND_GATE)
                            if refined_score >= refine_gate:
                                capture_live_pose = refined_pose
                                capture_live_meta = {**refined_meta, "source": "turn_arc_refined"}
                                refined_ok = True
                        if not refined_ok:
                            print(
                                "[wander] turn arc solve below gate even after xy-refine "
                                f"(capture={capture_index}, arc_score={arc_score:.3f}, "
                                f"min={float(args.min_append_score):.2f})"
                            )
                    else:
                        print(
                            "[wander] turn arc solve below gate "
                            f"(capture={capture_index}, score={arc_score:.3f}, min={arc_accept_gate:.2f})"
                        )
                else:
                    print(f"[wander] turn arc solve unavailable (capture={capture_index})")
            else:
                capture_pose_result = _estimate_pose_against_stitched_map(
                    points_xy=captured_snapshot.points_xy,
                    transformed_sets=stitch_state["transformed_sets"],
                    initial_pose=capture_expected_pose,
                    resolution_m=float(args.stitch_resolution_m),
                    search_xy_m=max(float(motion_hint.search_xy_m), 0.45),
                    theta_window_deg=max(float(motion_hint.search_theta_window_deg), 24.0),
                    # The pose is burst-tracked to ~5cm: the solve REFINES the
                    # tracked expectation, it does not search for it. A loose
                    # bound plus a token prior let captures slide ~0.5m along
                    # featureless straight walls at full score (phantom
                    # duplicate-wall lines in the map).
                    max_translation_from_initial_m=max(0.45, float(motion_hint.search_xy_m) + 0.10),
                    prior_translation_weight=2.5,
                    prior_theta_weight=0.12,
                )
                if capture_pose_result is not None:
                    candidate_capture_pose, candidate_capture_meta = capture_pose_result
                    if _accept_relocalized_pose(
                        label=f"capture_{capture_index}",
                        candidate_pose=candidate_capture_pose,
                        score_meta=candidate_capture_meta,
                        expected_pose=capture_expected_pose,
                        motion_hint=motion_hint,
                        max_translation_error_m=max(0.30, 0.65 * float(motion_hint.search_xy_m)),
                        max_theta_error_deg=max(18.0, float(motion_hint.search_theta_window_deg) * 1.10),
                        min_score=POSE_LOST_APPEND_GATE if pose_lost else 7.5,
                    ):
                        capture_live_pose = candidate_capture_pose
                        capture_live_meta = candidate_capture_meta
                    else:
                        print(
                            "[wander] capture relocalization fallback "
                            f"(capture={capture_index}, using strict append solver around prior stitched map)"
                        )
                if capture_pose_result is None:
                    print(f"[wander] capture relocalization unavailable (capture={capture_index})")

            if capture_live_pose is None and motion_hint.kind != "turn":
                # Run the strict append solver here (instead of inside _append_stitch)
                # so its score can be gated before the capture is committed to the map.
                is_turn_like_hint = motion_hint.kind in {"turn", "bootstrap_turn"}
                append_global_points_xy = np.concatenate(
                    [points for points in stitch_state["transformed_sets"] if len(points)], axis=0
                )
                append_prior_pose, append_prior_tw, append_prior_thw = _prior_weights_for_hint(
                    capture_expected_pose, motion_hint
                )
                append_solver_pose, append_solver_meta = _search_pose(
                    snapshot_points_xy=captured_snapshot.points_xy,
                    global_points_xy=append_global_points_xy,
                    initial_pose=capture_expected_pose,
                    resolution_m=float(args.stitch_resolution_m),
                    search_xy_m=float(motion_hint.search_xy_m),
                    coarse_angle_step_deg=4.0,
                    fine_angle_step_deg=0.5,
                    theta_window_deg=float(motion_hint.search_theta_window_deg),
                    whole_map_theta_center_deg=float(capture_expected_pose.theta_deg),
                    whole_map_theta_window_deg=(
                        float(max(motion_hint.search_theta_window_deg, 36.0))
                        if is_turn_like_hint
                        else float(max(motion_hint.search_theta_window_deg, 45.0))
                    ),
                    # Motion is lidar-tracked burst-by-burst now, so even the
                    # fallback search stays bounded — an unbounded whole-map
                    # drive search is how phantom room-sized jumps got in.
                    max_translation_from_initial_m=(
                        float(max(motion_hint.search_xy_m + 0.20, 0.55))
                        if is_turn_like_hint
                        else float(max(motion_hint.search_xy_m + 0.35, 0.90))
                    ),
                    prior_pose=append_prior_pose,
                    prior_translation_weight=float(append_prior_tw),
                    prior_theta_weight=float(append_prior_thw),
                )
                append_solver_meta = {
                    **append_solver_meta,
                    "source": f"append_{append_solver_meta.get('source', 'unknown')}",
                }
                append_solver_score = float(append_solver_meta.get("score") or -1e9)
                append_translation_err_m, append_theta_err_deg = _pose_delta_metrics(
                    capture_expected_pose, append_solver_pose
                )
                append_gate = float(args.min_append_score)
                if consecutive_append_discards > 0:
                    # The PREVIOUS capture was discarded: the pose expectation
                    # this solve is anchored to is already suspect, which is
                    # exactly when a barely-above-gate score is most likely a
                    # wrong mode (square-room symmetry). Field 2026-07-11: a
                    # 6.08-score append after a discard chain stitched a
                    # visibly rotated scan into the map. Demand a STRONG match
                    # or discard again and re-anchor.
                    append_gate = max(append_gate, float(args.min_append_score) + 2.0)
                elif append_theta_err_deg > 20.0:
                    # Solver heading disagrees hard with burst-by-burst lidar
                    # tracking, which rarely errs by 20deg+. A barely-above-
                    # gate score at a rotated mode is the ghost-room
                    # signature (field 2026-07-17: 6.92 at 30deg-off stitched
                    # a rotated copy of the room INSIDE the map and walled
                    # off both the table and the exit). Strong match or
                    # discard.
                    append_gate = max(append_gate, float(args.min_append_score) + 2.0)
                elif append_translation_err_m <= 0.30 and append_theta_err_deg <= 10.0:
                    # Solver landed where burst-by-burst tracking said we are;
                    # mutual confirmation earns a relaxed absolute gate. (Not
                    # after a discard — a shaky expectation confirms nothing.)
                    append_gate = min(append_gate, 3.5)
                if post_recovery_strict_appends > 0:
                    append_gate = max(append_gate, float(args.min_append_score) + 2.0)
                if pose_lost:
                    # Map is read-only while lost: absolute-trust matches only.
                    append_gate = max(append_gate, POSE_LOST_APPEND_GATE)
                    print(
                        f"[wander] pose lost: append requires score>={append_gate:.1f}"
                    )
                # The fallback used to gate on SCORE alone. Near a featureless
                # wall a slid pose scores as well as the true one, so a capture
                # 0.64m from its fully-locked tracked expectation got in and
                # painted a ghost wall. Motion is lidar-tracked: bound the
                # fallback by the expectation just like the strict solve.
                append_translation_gate_m = max(0.40, 0.75 * float(motion_hint.search_xy_m))
                append_theta_gate_deg = max(18.0, float(motion_hint.search_theta_window_deg) * 1.10)
                append_pose_ok = (
                    append_translation_err_m <= append_translation_gate_m
                    and append_theta_err_deg <= append_theta_gate_deg
                )
                if append_solver_score >= append_gate and append_pose_ok:
                    capture_live_pose = append_solver_pose
                    capture_live_meta = append_solver_meta
                else:
                    print(
                        "[wander] append solver rejected "
                        f"(capture={capture_index}, score={append_solver_score:.3f}, min={append_gate:.2f}, "
                        f"translation_error={append_translation_err_m:.3f}m<= {append_translation_gate_m:.2f}m, "
                        f"theta_error={append_theta_err_deg:.1f}deg<= {append_theta_gate_deg:.0f}deg); "
                        "attempting wide-theta rescue relocalization"
                    )
                    # Rescue exists for one failure mode: the HEADING went blind
                    # (lock-lost turn bursts), so theta may be far off while the
                    # position stays bounded by tracked driving — lock loss stops
                    # all translation. Search wide in theta, tight in xy. The old
                    # flat 1.40m/85deg gates predate tracked motion and accepted
                    # a pose 0.97m/43deg out, skewing the whole map.
                    rescue_result = _estimate_pose_against_stitched_map(
                        points_xy=captured_snapshot.points_xy,
                        transformed_sets=stitch_state["transformed_sets"],
                        initial_pose=capture_expected_pose,
                        resolution_m=float(args.stitch_resolution_m),
                        search_xy_m=append_translation_gate_m + 0.10,
                        theta_window_deg=80.0,
                        max_translation_from_initial_m=append_translation_gate_m,
                        prior_translation_weight=0.05,
                        prior_theta_weight=0.02,
                    )
                    if rescue_result is not None:
                        rescue_pose, rescue_meta = rescue_result
                        if _accept_relocalized_pose(
                            label=f"capture_{capture_index}_rescue",
                            candidate_pose=rescue_pose,
                            score_meta=rescue_meta,
                            expected_pose=capture_expected_pose,
                            motion_hint=motion_hint,
                            max_translation_error_m=append_translation_gate_m,
                            max_theta_error_deg=max(
                                35.0, float(motion_hint.search_theta_window_deg) + 10.0
                            ),
                            min_score=POSE_LOST_APPEND_GATE if pose_lost else 7.5,
                        ):
                            capture_live_pose = rescue_pose
                            capture_live_meta = {**rescue_meta, "source": f"rescue_{rescue_meta.get('source', 'unknown')}"}

            if capture_live_pose is None:
                consecutive_append_discards += 1
                if not pose_lost:
                    pose_lost = True
                    print(
                        "[wander] pose integrity LOST (capture solve discarded); "
                        "map is READ-ONLY until a strong relocalization re-proves the pose"
                    )
                if consecutive_append_discards <= max(0, int(args.max_consecutive_append_discards)):
                    print(
                        "[wander] discarding capture to protect the map "
                        f"(capture={capture_index}, best_score="
                        f"{float((append_solver_meta or {}).get('score') or -1e9):.3f}, "
                        f"consecutive_discards={consecutive_append_discards}); "
                        "its expected motion will carry into the next capture"
                    )
                    motion_hints.pop()
                    pending_motion_hint = motion_hint
                    last_frame_id = int(frame_id)
                    last_captured_frame = captured_frame
                    continue
                # Discard limit reached. NEVER force-accept into the map —
                # a below-gate pose stitches the room's own walls back in
                # rotated (field 2026-07-17: a force-accepted 6.63 painted a
                # ghost room INSIDE the room, walling off the table and the
                # exit). The MAP is sacred; the POSE is recoverable: drop
                # this capture AND the accumulated motion chain (the chain
                # is precisely what is untrustworthy after repeated solve
                # failures) and let live relocalization — which keeps
                # scoring 12-16 against the trusted map during these
                # episodes — re-anchor on the next planning cycle.
                force_accept_score = float((append_solver_meta or {}).get("score") or -1e9)
                print(
                    "[wander] discard limit reached; dropping the capture AND its "
                    f"motion chain instead of force-accepting (capture={capture_index}, "
                    f"score={force_accept_score:.3f}); live relocalization re-anchors"
                )
                consecutive_append_discards = 0
                motion_hints.pop()
                pending_motion_hint = None
                last_frame_id = int(frame_id)
                last_captured_frame = captured_frame
                continue
            consecutive_append_discards = 0
            orbit_recovery_turn_attempts = 0
            orbit_recovery_turns_remaining = 0
            orbit_recovery_direction_sign = None
            dead_reck_theta_deg = float(capture_live_pose.theta_deg)
            dead_reck_slack_deg = 15.0
            imu_dead_reck_anchor = _imu_anchor(imu_yaw, dead_reck_theta_deg)
            if pose_lost:
                # This append cleared the absolute-trust lost-gate (>=12.5):
                # the pose is re-proven by construction.
                pose_lost = False
                post_recovery_strict_appends = 2
                lost_recovery_failures = 0
                relocalize_spin_deg, relocalize_cycles, relocalize_halted = 0.0, 0, False
                print(
                    "[wander] pose integrity RESTORED (append cleared the absolute-trust gate); "
                    "map writes re-enabled (strict gates for the next 2 appends)"
                )
            elif post_recovery_strict_appends > 0:
                post_recovery_strict_appends -= 1

            stitch_state = _append_stitch(
                stitch_dir=stitch_dir,
                snapshot_dir=snapshot_dir,
                stitch_state=stitch_state,
                motion_hints=motion_hints,
                new_snapshot=captured_snapshot,
                resolution_m=float(args.stitch_resolution_m),
                solved_pose_override=capture_live_pose,
                solve_meta_override=capture_live_meta,
            )
            rebuild_elapsed = time.monotonic() - rebuild_started
            _log_rerun_state(
                rr,
                capture_index=capture_index,
                transformed_sets=stitch_state["transformed_sets"],
                poses=stitch_state["poses"],
                solve_log=stitch_state["solve_log"],
                cone_half_width_deg=float(args.valid_angle_half_width_deg),
                body_radius_m=float(args.robot_radius_m),
            )
            if hazard_monitor is not None:
                # Eye-camera panels + safety decision alongside the map, so
                # the user sees what the robot sees while it wanders. The image
                # panels honor --camera-feedback-hz (0 = off, the default) so the
                # viewer stays a clean map; the text decision always logs.
                hazard_state_now = hazard_monitor.state()
                if float(args.camera_feedback_hz) > 0.0:
                    for safety_cam in ("front_left", "front_right", "panorama", "bottom"):
                        safety_frame = hazard_monitor.annotated(safety_cam)
                        if safety_frame is not None:
                            rr.log(f"cameras/{safety_cam}", rr.Image(safety_frame[:, :, ::-1]))
                rr.log(
                    "safety/decision",
                    rr.TextLog(
                        f"{hazard_state_now.decision_label()} side={hazard_state_now.side} "
                        f"confidence={hazard_state_now.confidence:.2f}"
                    ),
                )
            capture_index += 1
            _log_live_pose_state(
                rr,
                capture_index=capture_index,
                pose=stitch_state["poses"][-1],
                points_xy=captured_snapshot.points_xy,
                cone_half_width_deg=float(args.valid_angle_half_width_deg),
                body_radius_m=float(args.robot_radius_m),
            )
            last_frame_id = int(frame_id)
            last_captured_frame = captured_frame

            final_pose = stitch_state["poses"][-1]
            previous_pose = stitch_state["poses"][-2] if len(stitch_state["poses"]) >= 2 else None
            solved_dx = 0.0 if previous_pose is None else float(final_pose.x - previous_pose.x)
            solved_dy = 0.0 if previous_pose is None else float(final_pose.y - previous_pose.y)
            solved_dtheta = 0.0 if previous_pose is None else _normalize_angle_deg(float(final_pose.theta_deg - previous_pose.theta_deg))
            solve_meta = stitch_state["solve_log"][-1] if stitch_state["solve_log"] else {}
            search_timing = solve_meta.get("timing_s") if isinstance(solve_meta, dict) else None
            append_timing = stitch_state.get("timing", {})
            print(
                "[wander] stitched capture complete "
                f"(capture={capture_index}, rebuild={rebuild_elapsed:.2f}s, pose=({final_pose.x:.3f}, {final_pose.y:.3f}, {final_pose.theta_deg:.1f}deg), "
                f"html={stitch_state['html_path']})"
            )
            if previous_pose is not None:
                print(
                    "[wander] solved motion delta "
                    f"(capture={capture_index}, dx={solved_dx:.3f}m, dy={solved_dy:.3f}m, dtheta={solved_dtheta:.1f}deg, "
                    f"source={solve_meta.get('solve_source')}, score={solve_meta.get('score')})"
                )
            if isinstance(search_timing, dict):
                print(
                    "[wander] stitch timing "
                    f"(capture={capture_index}, build={float(search_timing.get('build_occupancy', 0.0)):.2f}s, "
                    f"local={float(search_timing.get('local_search', 0.0)):.2f}s, "
                    f"whole_map={float(search_timing.get('whole_map_search', 0.0)):.2f}s, "
                    f"search_total={float(search_timing.get('total_search', 0.0)):.2f}s, "
                    f"write={float(append_timing.get('write_phase_s', 0.0)):.2f}s)"
                )
            if active_explore_target is not None:
                target_distance_after_capture_m = _target_distance_m(active_explore_target, final_pose)
                previous_target_distance_m = active_explore_target_last_distance_m
                if previous_target_distance_m is not None:
                    target_progress_m = float(previous_target_distance_m) - float(target_distance_after_capture_m)
                    if target_progress_m < 0.08:
                        active_explore_target_stall_count += 1
                        print(
                            "[wander] stitched-map target progress stalled "
                            f"(capture={capture_index}, progress={target_progress_m:.3f}m, "
                            f"distance={target_distance_after_capture_m:.3f}m, "
                            f"stall_count={active_explore_target_stall_count})"
                        )
                    else:
                        active_explore_target_stall_count = 0
                        active_explore_target_blocked_count = 0
                        force_live_frontier_cycles = 0
                        print(
                            "[wander] stitched-map target progress improved "
                            f"(capture={capture_index}, progress={target_progress_m:.3f}m, "
                            f"distance={target_distance_after_capture_m:.3f}m)"
                        )
                active_explore_target_last_distance_m = float(target_distance_after_capture_m)
                if active_explore_target_stall_count >= 2:
                    print(
                        "[wander] dropping stitched-map target after repeated low-progress captures "
                        f"(distance={target_distance_after_capture_m:.3f}m, "
                        f"source={active_explore_target.source}, "
                        f"seed_capture={active_explore_target.seeded_capture_index})"
                    )
                    if active_explore_target.source != "retreat":
                        _blacklist_target(active_explore_target, "stalled_progress")
                    active_explore_target = None
                    active_explore_target_stall_count = 0
                    active_explore_target_last_distance_m = None
                    active_explore_target_blocked_count = 0
                    force_live_frontier_cycles = max(force_live_frontier_cycles, 2)
            if motion_hint.kind == "turn":
                consecutive_turn_captures += 1
                solved_heading_bin = _heading_bin_index(
                    float(final_pose.theta_deg),
                    bin_count=rotation_coverage_bin_count,
                )
                previous_bin_count = len(rotation_coverage_bins_seen)
                rotation_coverage_bins_seen.add(solved_heading_bin)
                if len(rotation_coverage_bins_seen) != previous_bin_count or capture_index <= 4:
                    print(
                        "[wander] rotation coverage update "
                        f"(capture={capture_index}, heading={float(final_pose.theta_deg):.1f}deg, "
                        f"bins={sorted(rotation_coverage_bins_seen)}/{rotation_coverage_bin_count})"
                    )
                if not rotation_coverage_complete and len(rotation_coverage_bins_seen) >= rotation_coverage_bin_count:
                    rotation_coverage_complete = True
                    force_drive_after_turn = True
                    drive_checkpoints_since_scan = 0
                    print(
                        "[wander] rotational coverage complete; forcing forward exploration "
                        f"(capture={capture_index}, bins={sorted(rotation_coverage_bins_seen)})"
                    )
            else:
                consecutive_turn_captures = 0

        completed_captures = capture_index - 1 if "capture_index" in locals() else 1
        print(f"[wander] complete. captures={completed_captures} viewer={viewer_url or 'local rerun'}")
        print(f"[wander] stitched html: {stitch_state['html_path']}")
        print(f"[wander] stitched report: {stitch_state['report_path']}")
        return 0
    except KeyboardInterrupt:
        raise
    except Exception:
        # Never let a crash vanish behind a truncated terminal paste again
        # (field 2026-07-20: three separate hardware crashes whose exception
        # line got cut off by the console buffer). Write the FULL traceback to
        # a file where nothing can truncate it, then re-raise so behavior is
        # otherwise unchanged. The robot is stopped in the finally block below.
        try:
            crash_path = Path("artifacts/wander_snapshot_stitch/crash_traceback.txt")
            crash_path.parent.mkdir(parents=True, exist_ok=True)
            crash_path.write_text(traceback.format_exc(), encoding="utf-8")
            print(
                "[wander] FATAL: unhandled exception — full traceback written to "
                f"{crash_path} (paste THAT file, not the terminal, for diagnosis)"
            )
        except Exception:
            pass
        raise
    finally:
        try:
            _send_stop(robot)
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        feed.stop()
        if imu_yaw is not None:
            imu_yaw.stop()
        if hazard_monitor is not None:
            hazard_monitor.stop()
        if semantic_worker is not None:
            semantic_worker.stop()
        if hazard_subscriber is not None:
            hazard_subscriber.stop()


if __name__ == "__main__":
    raise SystemExit(main())
