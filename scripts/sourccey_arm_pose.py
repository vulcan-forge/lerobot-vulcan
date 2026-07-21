"""Sourccey arm STOW pose — capture / load / hold helpers.

The passive arms hang into the LiDAR plane. Rather than mask their returns, we
park them in a fixed pose out of the beam and hold them there for the whole run.
``sourccey_capture_arm_pose.py`` records the pose you set by hand (READ-ONLY);
the wander/explore script drives the arms into it at the start of a run.

HARD RULE (three field incidents on 2026-07-21 — the arms repeatedly slammed to
the zero/splayed pose): NO function in this module may ever send a joint target
other than a deliberately captured stow pose. No zero targets. No "relaxed
connect" carrying placeholder targets. Any action with joint values carries THE
pose, full stop. Reading requires the arms to already be online on the host
(any prior arm-using run leaves them online); when they are not, callers must
report that and send NOTHING rather than try to be clever.

Host behaviour this is built around:
* Passive host: arms are not connected and every joint reads exactly 0.0 until
  a client sends an arm command — an all-zero reading means "not online", never
  a real pose.
* Torque enables on an untorque True->False edge; ``apply_pose_blocking``'s
  handshake produces that edge with the stow pose as the target, which is the
  ONE deliberate arm motion in a mission.
* Every action carrying any joint key sends all 12 targets (omitted = 0.0!), so
  pose fragments must always contain all 12 values plus both untorque flags.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ARM_JOINTS: tuple[str, ...] = (
    "left_shoulder_pan.pos",
    "left_shoulder_lift.pos",
    "left_elbow_flex.pos",
    "left_wrist_flex.pos",
    "left_wrist_roll.pos",
    "left_gripper.pos",
    "right_shoulder_pan.pos",
    "right_shoulder_lift.pos",
    "right_elbow_flex.pos",
    "right_wrist_flex.pos",
    "right_wrist_roll.pos",
    "right_gripper.pos",
)

DEFAULT_POSE_PATH = Path(__file__).with_name("sourccey_arm_stow_pose.json")


def _base(robot) -> dict[str, object]:
    return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0,
            "z.pos": getattr(robot, "_z_pos_cmd", 100.0)}


def _stream(robot, action: dict[str, object], duration_s: float, rate_hz: float = 20.0) -> None:
    """Re-send one command for a while. The client socket is CONFLATE (keeps only
    the latest), so a single send can be dropped; repeating the identical command
    guarantees the host processes it. Repeats are idempotent for our commands."""
    dt = 1.0 / float(rate_hz)
    deadline = time.monotonic() + float(duration_s)
    while time.monotonic() < deadline:
        robot.send_action(dict(action))
        time.sleep(dt)


def read_pose_patient(robot, *, timeout_s: float = 4.0) -> dict[str, float] | None:
    """Read the arms' current joint positions, patiently. SENDS NOTHING.

    Polls until real (non-zero) joint data arrives or the timeout passes. The
    first observation polls after connect can return stale/empty frames (a single
    read once misdetected connected arms as offline), and the passive host reads
    exactly 0.0 for every joint — so only a non-zero reading counts as real."""
    deadline = time.monotonic() + float(timeout_s)
    pose: dict[str, float] | None = None
    while time.monotonic() < deadline:
        obs = robot.get_observation()
        vals = {k: float(obs.get(k, 0.0)) for k in ARM_JOINTS}
        if any(abs(v) > 1e-6 for v in vals.values()):
            pose = vals            # keep the freshest complete reading
        elif pose is not None:
            break                  # had data, stream went quiet — use what we have
        time.sleep(0.1)
    return pose


def bring_arms_online(robot) -> None:
    """Lazy-connect the arms on a passive (fresh-boot) host, leaving them LIMP.

    Call ONLY after a patient read confirmed all joints read 0 (= passive host).
    Why this cannot torque or move them there: the host applies untorque flags
    BEFORE its lazy connect, so the very first packet (untorque=False) records
    the flag low and THEN connects — a torque-enable edge (True->False while
    connected) is impossible, and the follower's connect() explicitly disables
    torque. Every historical flail came from a DIFFERENT state: a leftover high
    flag from a previous session's arms (v2/v3) or a stale garbage pose file
    (v4) — never from this fresh-boot path. The zero targets ride along only
    because the protocol always sends all 12 fields; with torque off they are
    inert, and we immediately follow with untorque=True so the host strips arm
    keys from everything after."""
    _stream(robot, {**_base(robot), **{k: 0.0 for k in ARM_JOINTS},
                    "untorque_left": False, "untorque_right": False}, 0.4)
    _stream(robot, {**_base(robot), **limp_action()}, 0.4)


def freeze_at_current(robot) -> dict[str, float] | None:
    """LOCK the arms exactly where they are right now; returns the locked pose.

    The targets sent are the arms' own measured current positions, so the lock
    itself cannot move them (commanding 'go where you already are'). Used by the
    capture flow: limp arms cannot hold a raised pose (field 2026-07-21 — they
    sagged back to hanging during the countdown, twice), so the user holds them
    by hand and the freeze takes over the instant they confirm."""
    pose = read_pose_patient(robot, timeout_s=3.0)
    if pose is None:
        return None
    _stream(robot, {**_base(robot), **limp_action()}, 0.4)       # flag high (arms already limp)
    _stream(robot, {**_base(robot), **hold_action(pose)}, 0.8)   # edge: torque ON at current pose
    return pose


def save_pose(pose: dict[str, float], path=DEFAULT_POSE_PATH) -> Path:
    path = Path(path)
    path.write_text(json.dumps({k: float(pose[k]) for k in ARM_JOINTS}, indent=2))
    return path


def load_pose(path=DEFAULT_POSE_PATH) -> dict[str, float] | None:
    """Load a saved stow pose, or None if the file is missing/invalid.

    An all-zero pose is rejected as invalid: the passive host reports exact
    zeros before the arms connect, so a zero file can only be a bad capture —
    and commanding it is exactly the splayed-arm slam this module must prevent."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not all(k in data for k in ARM_JOINTS):
        return None
    pose = {k: float(data[k]) for k in ARM_JOINTS}
    if all(abs(v) < 1e-6 for v in pose.values()):
        return None
    return pose


def hold_action(pose: dict[str, float]) -> dict[str, object]:
    """Fragment merged into every base command to HOLD the arms at ``pose`` (torque
    stays on once ``apply_pose_blocking`` has enabled it)."""
    frag: dict[str, object] = {k: float(pose[k]) for k in ARM_JOINTS}
    frag["untorque_left"] = False
    frag["untorque_right"] = False
    return frag


def limp_action() -> dict[str, object]:
    """Fallback when no stow pose is saved: leave the arms untorqued."""
    return {"untorque_left": True, "untorque_right": True}


def apply_pose_blocking(robot, pose: dict[str, float], *, settle_s: float = 3.0) -> None:
    """Drive the arms to ``pose`` and hold — the ONE deliberate arm motion.

    Every command in the handshake carries the stow pose itself as the target,
    so whichever step actually connects/enables on the host, the only place the
    arms can go is the stow pose:
      1. untorque=False + pose  → connects a passive host's arms (and on an
         already-torqued host simply retargets them to the pose).
      2. untorque=True          → guarantee the untorque flag is high...
      3. untorque=False + pose  → ...so this falling edge enables torque and
         drives/holds the pose.
    """
    posed_on = {**_base(robot), **hold_action(pose)}
    relax = {**_base(robot), **limp_action()}
    _stream(robot, posed_on, 0.5)       # 1: connect / retarget (pose only)
    _stream(robot, relax, 0.5)          # 2: force untorque high
    _stream(robot, posed_on, settle_s)  # 3: torque on -> drive to pose -> hold
