"""Sourccey arm STOW pose — capture / load / hold helpers.

The passive arms hang into the LiDAR plane. Rather than mask their returns, we
park them in a fixed pose out of the beam and hold them there for the whole run.
``sourccey_capture_arm_pose.py`` records the pose you set by hand; the
wander/explore script loads it and holds it before spinning.

This is written around how the Sourccey HOST actually behaves (learned the hard
way — capture first read all-zeros):

* The host runs in "passive-arm mode": the follower arms are NOT connected and
  report position 0 until a client sends an arm joint command. (get_observation
  returns 0s until then — that was the all-zeros capture.)
* A follower's ``connect()`` runs ``configure()`` which DISABLES torque and
  nothing re-enables it. So the FIRST arm command connects the arms RELAXED and
  does not move them (with torque off, the goal-position write is ignored). That
  is our safe hook to read the hand-set pose.
* Torque only turns on for an arm on an untorque True->False EDGE. So holding a
  pose is a handshake (connect relaxed -> untorque True -> untorque False+pose),
  not just streaming untorque=False.
* Every action sends all 12 joint targets, defaulting any omitted joint to 0.0,
  so a hold command must always carry all 12 values.

Because reading requires connecting, run capture against a freshly-started host
(arms still in passive mode) and hold the arms in place — they are read RELAXED.
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


def _joint_targets(pose: dict[str, float] | None) -> dict[str, object]:
    # 0.0 for every joint when no pose given (harmless while torque is off).
    return {k: float(pose[k]) if pose else 0.0 for k in ARM_JOINTS}


def _stream(robot, action: dict[str, object], duration_s: float, rate_hz: float = 20.0) -> None:
    """Re-send one command for a while. The client socket is CONFLATE (keeps only
    the latest), so a single send can be dropped; repeating the identical command
    guarantees the host processes it. Repeats are idempotent for our commands."""
    dt = 1.0 / float(rate_hz)
    deadline = time.monotonic() + float(duration_s)
    while time.monotonic() < deadline:
        robot.send_action(dict(action))
        time.sleep(dt)


def _read_joints(robot) -> dict[str, float]:
    obs = robot.get_observation()
    return {k: float(obs.get(k, 0.0)) for k in ARM_JOINTS}


def _arms_reporting(robot) -> bool:
    """True if the arms are connected on the host (they report non-zero positions;
    the passive host reports exactly 0 for every joint until connected)."""
    return any(abs(v) > 1e-6 for v in _read_joints(robot).values())


def relax_arms(robot, *, settle_s: float = 1.0) -> bool:
    """Make the arms LIMP so they can be positioned by hand — WITHOUT ever driving
    them. Returns whether they are reporting afterwards.

    Two cases, chosen by reading first (this is what prevents the flail):
    * Already connected -> send untorque=True. This only disables torque; it can
      never enable it, so the arms cannot be driven. (Sending untorque=False here
      would hit the torque-enable edge and slam the arms to zero — the flail.)
    * Not connected -> send untorque=False + joints. Because the arms are not yet
      connected, the host takes its 'not connected' branch (no torque enable) and
      then connect() leaves torque disabled, so they come online relaxed and do
      not move."""
    if _arms_reporting(robot):
        _stream(robot, {**_base(robot), "untorque_left": True, "untorque_right": True}, settle_s)
    else:
        _stream(robot, {**_base(robot), **_joint_targets(None),
                        "untorque_left": False, "untorque_right": False}, settle_s)
    return _arms_reporting(robot)


def capture_pose(robot, *, reads: int = 12) -> dict[str, float] | None:
    """Read the current (relaxed) arm joint positions. Call ``relax_arms`` first so
    the arms are limp and connected. Returns None if they never report."""
    pose: dict[str, float] | None = None
    for _ in range(int(reads)):
        vals = _read_joints(robot)
        if any(abs(v) > 1e-6 for v in vals.values()):
            pose = vals
        time.sleep(0.05)
    return pose


def save_pose(pose: dict[str, float], path=DEFAULT_POSE_PATH) -> Path:
    path = Path(path)
    path.write_text(json.dumps({k: float(pose[k]) for k in ARM_JOINTS}, indent=2))
    return path


def load_pose(path=DEFAULT_POSE_PATH) -> dict[str, float] | None:
    """Load a saved stow pose, or None if the file is missing/invalid."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not all(k in data for k in ARM_JOINTS):
        return None
    return {k: float(data[k]) for k in ARM_JOINTS}


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
    """Torque the arms and drive them to ``pose``, then hold long enough to arrive.

    Handshake (robust to the arms being connected or not, torqued or not):
      1. untorque=False + pose  → connect the arms (relaxed; they do not move yet).
      2. untorque=True          → guarantee the untorque flag is high...
      3. untorque=False + pose  → ...so THIS falling edge enables torque and drives
         to the pose, and continued sends hold it there.
    """
    posed_on = {**_base(robot), **_joint_targets(pose),
                "untorque_left": False, "untorque_right": False}
    relax = {**_base(robot), "untorque_left": True, "untorque_right": True}
    _stream(robot, posed_on, 0.5)     # 1: connect (relaxed)
    _stream(robot, relax, 0.5)        # 2: force untorque high
    _stream(robot, posed_on, settle_s)  # 3: torque on -> drive to pose -> hold
