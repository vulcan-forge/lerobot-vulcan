# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 Vulcan Robotics, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import signal
import time
from typing import Any

import zmq

from .arm_debug_capture import ARM_POSITION_KEYS, ArmDebugCapture, extract_arm_positions
from .config_sourccey import SourcceyConfig, SourcceyHostConfig
from .sourccey import Sourccey

# Import protobuf modules
from ..protobuf.generated import sourccey_pb2


class SourcceyHost:
    def __init__(self, config: SourcceyHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _new_startup_supervisor_config() -> dict[str, Any]:
    return {
        "enabled": _env_flag("SOURCCEY_STARTUP_SUPERVISOR_ENABLED", True),
        "align_step": max(_env_float("SOURCCEY_ALIGN_STEP", 6.0), 0.0),
        "align_step_shoulder_lift": max(_env_float("SOURCCEY_ALIGN_STEP_SHOULDER_LIFT", 3.0), 0.0),
        "align_tolerance": max(_env_float("SOURCCEY_ALIGN_TOLERANCE", 6.0), 0.0),
        "align_stable_frames": max(_env_int("SOURCCEY_ALIGN_STABLE_FRAMES", 8), 1),
        "align_timeout_s": max(_env_float("SOURCCEY_ALIGN_TIMEOUT_S", 5.0), 0.1),
        "fault_hold_min_s": max(_env_float("SOURCCEY_FAULT_HOLD_MIN_S", 0.5), 0.0),
        "fault_relax_after_s": max(_env_float("SOURCCEY_FAULT_RELAX_AFTER_S", 5.0), 0.5),
        "fault_resume_clean_frames": max(_env_int("SOURCCEY_FAULT_RESUME_CLEAN_FRAMES", 8), 1),
        "stall_error_threshold": max(_env_float("SOURCCEY_STALL_ERROR_THRESHOLD", 25.0), 0.0),
        "stall_progress_threshold": max(_env_float("SOURCCEY_STALL_PROGRESS_THRESHOLD", 0.8), 0.0),
        "stall_consecutive_frames": max(_env_int("SOURCCEY_STALL_CONSECUTIVE_FRAMES", 8), 1),
        "target_latch_window_frames": max(_env_int("SOURCCEY_TARGET_LATCH_WINDOW_FRAMES", 5), 1),
        "target_latch_shoulder_spread_max": max(_env_float("SOURCCEY_TARGET_LATCH_SHOULDER_SPREAD_MAX", 30.0), 0.0),
        "target_latch_shoulder_observed_delta_max": max(
            _env_float("SOURCCEY_TARGET_LATCH_SHOULDER_OBSERVED_DELTA_MAX", 45.0), 0.0
        ),
    }


def _new_startup_supervisor_state() -> dict[str, Any]:
    return {
        "mode": "BOOT",
        "mode_enter_t": time.monotonic(),
        "align_target_arm": {},
        "align_stable_frames": 0,
        "target_latch_buffer": [],
        "target_latch_reject_reason": None,
        "fault_hold_action": {},
        "fault_reason": None,
        "fault_clean_frames": 0,
        "stall_counts": {"left": 0, "right": 0},
        "relax_sent": False,
    }


def _serialize_supervisor_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": str(state.get("mode", "BOOT")),
        "mode_enter_t": state.get("mode_enter_t"),
        "align_target_count": len(state.get("align_target_arm") or {}),
        "align_stable_frames": int(state.get("align_stable_frames", 0)),
        "target_latch_buffer_len": len(state.get("target_latch_buffer") or []),
        "target_latch_reject_reason": state.get("target_latch_reject_reason"),
        "fault_reason": state.get("fault_reason"),
        "fault_clean_frames": int(state.get("fault_clean_frames", 0)),
        "stall_counts": {
            "left": int((state.get("stall_counts") or {}).get("left", 0)),
            "right": int((state.get("stall_counts") or {}).get("right", 0)),
        },
        "relax_sent": bool(state.get("relax_sent", False)),
    }


def _arm_key_to_side(key: str) -> str | None:
    if key.startswith("left_"):
        return "left"
    if key.startswith("right_"):
        return "right"
    return None


def _set_mode(
    state: dict[str, Any],
    *,
    mode: str,
    reason: str,
    host_arm_debug: ArmDebugCapture,
    extra: dict[str, Any] | None = None,
) -> None:
    previous_mode = state.get("mode")
    state["mode"] = mode
    state["mode_enter_t"] = time.monotonic()
    if mode != "ALIGN":
        state["align_stable_frames"] = 0
    if mode != "FAULT_HOLD":
        state["fault_clean_frames"] = 0
    payload: dict[str, Any] = {
        "previous_mode": previous_mode,
        "next_mode": mode,
        "reason": reason,
        "state": _serialize_supervisor_state(state),
    }
    if extra:
        payload["extra"] = extra
    host_arm_debug.record_session("host_mode_transition", payload)


def _extract_side_arm_targets(arm_data: dict[str, float], side: str) -> dict[str, float]:
    prefix = f"{side}_"
    return {k: float(v) for k, v in arm_data.items() if k.startswith(prefix)}


def _median(values: list[float]) -> float:
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _append_target_latch_sample(
    state: dict[str, Any],
    arm_targets: dict[str, float],
    *,
    supervisor_cfg: dict[str, Any],
) -> None:
    buffer = state.setdefault("target_latch_buffer", [])
    buffer.append({k: float(v) for k, v in arm_targets.items()})
    max_len = int(supervisor_cfg["target_latch_window_frames"])
    if len(buffer) > max_len:
        del buffer[:-max_len]


def _build_median_target_from_samples(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        return {}
    keys = sorted({key for sample in samples for key in sample})
    target: dict[str, float] = {}
    for key in keys:
        values = [float(sample[key]) for sample in samples if key in sample]
        if values:
            target[key] = _median(values)
    return target


def _evaluate_startup_target_candidate(
    *,
    samples: list[dict[str, float]],
    candidate_target: dict[str, float],
    observed_arm: dict[str, float],
    supervisor_cfg: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    required_frames = int(supervisor_cfg["target_latch_window_frames"])
    if len(samples) < required_frames:
        return False, {
            "reason": "insufficient_samples",
            "required_frames": required_frames,
            "sample_count": len(samples),
        }

    shoulder_keys = ("left_shoulder_lift.pos", "right_shoulder_lift.pos")
    spread_limit = float(supervisor_cfg["target_latch_shoulder_spread_max"])
    observed_delta_limit = float(supervisor_cfg["target_latch_shoulder_observed_delta_max"])
    shoulder_spreads: dict[str, float] = {}
    shoulder_observed_deltas: dict[str, float] = {}

    for key in shoulder_keys:
        values = [float(sample[key]) for sample in samples if key in sample]
        if len(values) < required_frames:
            return False, {
                "reason": "missing_shoulder_samples",
                "joint": key,
                "required_frames": required_frames,
                "sample_count": len(values),
            }

        spread = max(values) - min(values)
        shoulder_spreads[key] = spread
        if spread > spread_limit:
            return False, {
                "reason": "shoulder_lift_unstable",
                "joint": key,
                "spread": spread,
                "spread_limit": spread_limit,
                "shoulder_spreads": shoulder_spreads,
            }

        observed = observed_arm.get(key)
        candidate = candidate_target.get(key)
        if observed is None or candidate is None:
            return False, {
                "reason": "missing_shoulder_data",
                "joint": key,
                "has_observed": observed is not None,
                "has_candidate": candidate is not None,
            }

        observed_delta = abs(float(candidate) - float(observed))
        shoulder_observed_deltas[key] = observed_delta
        if observed_delta > observed_delta_limit:
            return False, {
                "reason": "shoulder_lift_far_from_observed",
                "joint": key,
                "observed_delta": observed_delta,
                "observed_delta_limit": observed_delta_limit,
                "shoulder_observed_deltas": shoulder_observed_deltas,
            }

    return True, {
        "reason": "ok",
        "required_frames": required_frames,
        "sample_count": len(samples),
        "shoulder_spreads": shoulder_spreads,
        "shoulder_observed_deltas": shoulder_observed_deltas,
    }


def _try_latch_startup_align_target(
    *,
    state: dict[str, Any],
    latest_action: dict[str, Any] | None,
    observed_arm: dict[str, float],
    supervisor_cfg: dict[str, Any],
    host_arm_debug: ArmDebugCapture,
) -> bool:
    if latest_action is None:
        return False

    arm_targets = extract_arm_positions(latest_action)
    if not arm_targets:
        return False

    _append_target_latch_sample(state, arm_targets, supervisor_cfg=supervisor_cfg)
    samples = state.get("target_latch_buffer") or []
    candidate_target = _build_median_target_from_samples(samples)

    accepted, check_info = _evaluate_startup_target_candidate(
        samples=samples,
        candidate_target=candidate_target,
        observed_arm=observed_arm,
        supervisor_cfg=supervisor_cfg,
    )

    check_payload: dict[str, Any] = {
        "accepted": accepted,
        "check_info": check_info,
        "candidate_shoulder_targets": {
            "left_shoulder_lift.pos": candidate_target.get("left_shoulder_lift.pos"),
            "right_shoulder_lift.pos": candidate_target.get("right_shoulder_lift.pos"),
        },
        "observed_shoulder_positions": {
            "left_shoulder_lift.pos": observed_arm.get("left_shoulder_lift.pos"),
            "right_shoulder_lift.pos": observed_arm.get("right_shoulder_lift.pos"),
        },
        "supervisor_state": _serialize_supervisor_state(state),
    }

    if accepted:
        state["align_target_arm"] = candidate_target
        state["align_stable_frames"] = 0
        state["target_latch_buffer"] = []
        state["target_latch_reject_reason"] = None
        host_arm_debug.record("host_startup_target_latch", check_payload)
        return True

    reject_reason = str(check_info.get("reason", "unknown"))
    if state.get("target_latch_reject_reason") != reject_reason:
        host_arm_debug.record("host_startup_target_latch", check_payload)
    state["target_latch_reject_reason"] = reject_reason
    return False


def _build_hold_action(
    observation: dict[str, Any] | None,
    template_action: dict[str, Any] | None,
) -> dict[str, Any]:
    obs_arm = extract_arm_positions(observation)
    action: dict[str, Any] = {k: float(v) for k, v in obs_arm.items()}

    action["x.vel"] = 0.0
    action["y.vel"] = 0.0
    action["theta.vel"] = 0.0

    if isinstance(template_action, dict) and "z.pos" in template_action:
        action["z.pos"] = float(template_action["z.pos"])
    elif isinstance(observation, dict) and "z.pos" in observation:
        action["z.pos"] = float(observation["z.pos"])
    else:
        action["z.pos"] = 100.0

    return action


def _step_toward(
    current: float,
    target: float,
    *,
    step: float,
) -> float:
    if step <= 0.0:
        return target
    delta = target - current
    if abs(delta) <= step:
        return target
    return current + (step if delta > 0 else -step)


def _build_align_action(
    *,
    observation: dict[str, Any] | None,
    align_target_arm: dict[str, float],
    template_action: dict[str, Any] | None,
    supervisor_cfg: dict[str, Any],
) -> dict[str, Any]:
    action = _build_hold_action(observation, template_action)
    observed_arm = extract_arm_positions(observation)

    for key, target in align_target_arm.items():
        current = observed_arm.get(key)
        if current is None:
            continue
        step = (
            float(supervisor_cfg["align_step_shoulder_lift"])
            if key.endswith("shoulder_lift.pos")
            else float(supervisor_cfg["align_step"])
        )
        action[key] = float(_step_toward(float(current), float(target), step=step))

    return action


def _max_abs_arm_delta(target: dict[str, float], observed: dict[str, float]) -> float | None:
    common = [k for k in target if k in observed]
    if not common:
        return None
    return max(abs(float(target[k]) - float(observed[k])) for k in common)


def _extract_status_packet_errors_from_sent_action(sent_action: dict[str, Any]) -> dict[str, str | None]:
    left_error = sent_action.get("left_status_packet_error")
    right_error = sent_action.get("right_status_packet_error")
    return {
        "left": str(left_error) if left_error else None,
        "right": str(right_error) if right_error else None,
    }


def _compute_arm_target_adjustments(
    received_arm_target: dict[str, float],
    sent_arm_target: dict[str, float],
) -> dict[str, Any]:
    common_keys = [k for k in received_arm_target if k in sent_arm_target]
    adjustments = {k: float(sent_arm_target[k] - received_arm_target[k]) for k in common_keys}
    abs_adjustments = {k: abs(v) for k, v in adjustments.items()}
    adjusted = {k: v for k, v in adjustments.items() if abs(v) > 1e-6}
    return {
        "joint_adjustments": adjusted,
        "max_abs_adjustment": max(abs_adjustments.values()) if abs_adjustments else None,
        "adjusted_joint_count": len(adjusted),
    }


def _detect_startup_stall(
    *,
    sent_arm_target: dict[str, float],
    observed_arm_position: dict[str, float],
    previous_observed_arm_position: dict[str, float],
    stall_counts: dict[str, int],
    supervisor_cfg: dict[str, Any],
) -> dict[str, Any]:
    per_side = {
        "left": "left_shoulder_lift.pos",
        "right": "right_shoulder_lift.pos",
    }

    info: dict[str, Any] = {"sides": {}, "triggered_sides": []}
    for side, key in per_side.items():
        sent = sent_arm_target.get(key)
        obs = observed_arm_position.get(key)
        prev_obs = previous_observed_arm_position.get(key)

        side_payload: dict[str, Any] = {
            "sent_target": float(sent) if sent is not None else None,
            "observed_position": float(obs) if obs is not None else None,
            "previous_observed_position": float(prev_obs) if prev_obs is not None else None,
            "target_minus_observed": None,
            "observed_step": None,
            "consecutive_hits": int(stall_counts.get(side, 0)),
            "triggered": False,
        }

        if sent is None or obs is None or prev_obs is None:
            stall_counts[side] = 0
            side_payload["consecutive_hits"] = 0
            info["sides"][side] = side_payload
            continue

        target_error = abs(float(sent - obs))
        observed_step = abs(float(obs - prev_obs))

        side_payload["target_minus_observed"] = float(sent - obs)
        side_payload["observed_step"] = observed_step

        if (
            target_error >= float(supervisor_cfg["stall_error_threshold"])
            and observed_step <= float(supervisor_cfg["stall_progress_threshold"])
        ):
            stall_counts[side] = int(stall_counts.get(side, 0)) + 1
        else:
            stall_counts[side] = 0

        side_payload["consecutive_hits"] = int(stall_counts[side])
        if stall_counts[side] >= int(supervisor_cfg["stall_consecutive_frames"]):
            side_payload["triggered"] = True
            info["triggered_sides"].append(side)

        info["sides"][side] = side_payload

    return info


def main():
    def _handle_termination_signal(signum, _frame):
        logging.info(f"Received signal {signum}. Shutting down Sourccey Host.")
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_termination_signal)

    _silence_camera_warnings_for_host()

    logging.info("Configuring Sourccey")
    robot_config = SourcceyConfig(id="sourccey")
    robot = Sourccey(robot_config)

    logging.info("Connecting Sourccey")
    robot.connect()

    logging.info("Starting Host")
    host_config = SourcceyHostConfig()
    host = SourcceyHost(host_config)

    supervisor_cfg = _new_startup_supervisor_config()
    supervisor_state = _new_startup_supervisor_state()

    host_arm_debug = ArmDebugCapture(
        enabled=_env_flag("SOURCCEY_HOST_ARM_DEBUG_CAPTURE", True),
        duration_s=_env_float("SOURCCEY_HOST_ARM_DEBUG_DURATION_S", 5.0),
        motion_threshold=_env_float("SOURCCEY_HOST_ARM_DEBUG_MOTION_THRESHOLD", 1.0),
        label="host",
        capture_path=os.getenv("SOURCCEY_HOST_ARM_DEBUG_PATH"),
    )
    if host_arm_debug.path:
        logging.warning("Host arm debug capture enabled. Writing to %s", host_arm_debug.path)

    host_arm_debug.record_session(
        "host_startup_supervisor_config",
        {
            "supervisor_config": supervisor_cfg,
            "supervisor_state": _serialize_supervisor_state(supervisor_state),
            "host_debug_path": host_arm_debug.path,
        },
    )

    print("Waiting for commands...")

    last_cmd_time = time.time()
    watchdog_active = False

    latest_action: dict[str, Any] | None = None
    latest_action_received_t: float | None = None
    has_received_action = False

    try:
        try:
            observation = robot.get_observation()
        except Exception:
            observation = {}
        previous_observation = observation

        last_sent_action: dict[str, Any] = {}
        last_sent_arm_target: dict[str, float] = {}
        status_packet_errors: dict[str, str | None] = {"left": None, "right": None}
        stall_info: dict[str, Any] = {"sides": {}, "triggered_sides": []}

        start = time.perf_counter()
        duration = 0.0
        while duration < host.connection_time_s:
            loop_start_time = time.time()
            received_this_loop = False

            try:
                msg_bytes = host.zmq_cmd_socket.recv(zmq.NOBLOCK)
                robot_action = sourccey_pb2.SourcceyRobotAction()
                robot_action.ParseFromString(msg_bytes)
                latest_action = robot.protobuf_converter.protobuf_to_action(robot_action)
                latest_action_received_t = time.monotonic()
                received_this_loop = True
                has_received_action = True
                last_cmd_time = time.time()
                watchdog_active = False

                host_arm_debug.maybe_start(action=latest_action, observation=previous_observation)

                host_arm_debug.record(
                    "host_received_action",
                    {
                        "received_arm_target": extract_arm_positions(latest_action),
                        "received_base_target": {
                            "x.vel": float(latest_action.get("x.vel", 0.0)),
                            "y.vel": float(latest_action.get("y.vel", 0.0)),
                            "theta.vel": float(latest_action.get("theta.vel", 0.0)),
                            "z.pos": float(latest_action.get("z.pos", 0.0)),
                        },
                        "supervisor_state": _serialize_supervisor_state(supervisor_state),
                    },
                )
            except zmq.Again:
                pass
            except Exception as e:
                logging.error("Message fetching failed: %s", e)

            previous_obs_arm = extract_arm_positions(previous_observation)
            mode = str(supervisor_state.get("mode", "BOOT"))

            if not supervisor_cfg.get("enabled", True):
                mode = "RUN"
                supervisor_state["mode"] = "RUN"

            # Allow automatic recovery from FAULT_RELAX when fresh arm commands resume.
            if mode == "FAULT_RELAX" and received_this_loop and latest_action is not None:
                arm_targets = extract_arm_positions(latest_action)
                if arm_targets:
                    supervisor_state["align_target_arm"] = arm_targets
                    supervisor_state["align_stable_frames"] = 0
                    supervisor_state["fault_reason"] = None
                    supervisor_state["fault_clean_frames"] = 0
                    supervisor_state["stall_counts"] = {"left": 0, "right": 0}
                    supervisor_state["relax_sent"] = False
                    _set_mode(
                        supervisor_state,
                        mode="ALIGN",
                        reason="command_received_exit_fault_relax",
                        host_arm_debug=host_arm_debug,
                    )
                    mode = "ALIGN"

            # BOOT: wait for first action, hold current robot pose.
            if mode == "BOOT":
                action_to_send = {
                    "x.vel": 0.0,
                    "y.vel": 0.0,
                    "theta.vel": 0.0,
                    "z.pos": float((previous_observation or {}).get("z.pos", 100.0)),
                }
                if received_this_loop and _try_latch_startup_align_target(
                    state=supervisor_state,
                    latest_action=latest_action,
                    observed_arm=previous_obs_arm,
                    supervisor_cfg=supervisor_cfg,
                    host_arm_debug=host_arm_debug,
                ):
                    _set_mode(
                        supervisor_state,
                        mode="ALIGN",
                        reason="startup_target_latched",
                        host_arm_debug=host_arm_debug,
                    )
                    mode = "ALIGN"

            # ALIGN: ramp from measured current pose to latched startup target.
            if mode == "ALIGN":
                align_target_arm = supervisor_state.get("align_target_arm") or {}
                action_to_send = _build_align_action(
                    observation=previous_observation,
                    align_target_arm=align_target_arm,
                    template_action=latest_action,
                    supervisor_cfg=supervisor_cfg,
                )
                max_align_delta = _max_abs_arm_delta(align_target_arm, previous_obs_arm)

                if max_align_delta is not None and max_align_delta <= float(supervisor_cfg["align_tolerance"]):
                    supervisor_state["align_stable_frames"] = int(supervisor_state.get("align_stable_frames", 0)) + 1
                else:
                    supervisor_state["align_stable_frames"] = 0

                if int(supervisor_state["align_stable_frames"]) >= int(supervisor_cfg["align_stable_frames"]):
                    _set_mode(
                        supervisor_state,
                        mode="RUN",
                        reason="align_complete",
                        host_arm_debug=host_arm_debug,
                        extra={"max_align_delta": max_align_delta},
                    )
                    mode = "RUN"
                elif (time.monotonic() - float(supervisor_state.get("mode_enter_t", time.monotonic()))) >= float(
                    supervisor_cfg["align_timeout_s"]
                ):
                    supervisor_state["fault_hold_action"] = _build_hold_action(previous_observation, latest_action)
                    supervisor_state["fault_reason"] = "align_timeout"
                    _set_mode(
                        supervisor_state,
                        mode="FAULT_HOLD",
                        reason="align_timeout",
                        host_arm_debug=host_arm_debug,
                        extra={"max_align_delta": max_align_delta},
                    )
                    mode = "FAULT_HOLD"

            # RUN: pass policy actions directly.
            if mode == "RUN":
                if latest_action is not None:
                    action_to_send = dict(latest_action)
                else:
                    action_to_send = _build_hold_action(previous_observation, latest_action)

            # FAULT_HOLD: hold current pose, wait for recovery, then re-align.
            if mode == "FAULT_HOLD":
                hold_action = supervisor_state.get("fault_hold_action") or {}
                if not hold_action:
                    hold_action = _build_hold_action(previous_observation, latest_action)
                    supervisor_state["fault_hold_action"] = hold_action
                action_to_send = dict(hold_action)

            # FAULT_RELAX: untorque and wait for operator intervention.
            if mode == "FAULT_RELAX":
                action_to_send = {
                    "x.vel": 0.0,
                    "y.vel": 0.0,
                    "theta.vel": 0.0,
                    "z.pos": float((previous_observation or {}).get("z.pos", 100.0)),
                    "untorque_left": True,
                    "untorque_right": True,
                }

            sent_action = robot.send_action(action_to_send)
            last_sent_action = sent_action
            status_packet_errors = _extract_status_packet_errors_from_sent_action(sent_action)
            sent_arm_target = extract_arm_positions(sent_action)
            last_sent_arm_target = sent_arm_target

            robot.update()

            now = time.time()
            if has_received_action and (now - last_cmd_time > host.watchdog_timeout_ms / 1000.0) and not watchdog_active:
                logging.debug(
                    "Command not received for more than %d milliseconds. Stopping base and releasing arm torque.",
                    host.watchdog_timeout_ms,
                )
                watchdog_active = True
                robot.watchdog_stop_and_relax()
                current_mode = str(supervisor_state.get("mode", "BOOT"))
                if current_mode in {"ALIGN", "RUN", "FAULT_HOLD"}:
                    supervisor_state["fault_reason"] = "watchdog_timeout"
                    _set_mode(
                        supervisor_state,
                        mode="FAULT_RELAX",
                        reason="watchdog_timeout",
                        host_arm_debug=host_arm_debug,
                    )

            if observation is not None and observation != {}:
                previous_observation = observation
            observation = robot.get_observation()

            observed_arm = extract_arm_positions(observation)
            stall_info = _detect_startup_stall(
                sent_arm_target=last_sent_arm_target,
                observed_arm_position=observed_arm,
                previous_observed_arm_position=previous_obs_arm,
                stall_counts=supervisor_state.get("stall_counts", {"left": 0, "right": 0}),
                supervisor_cfg=supervisor_cfg,
            )

            mode = str(supervisor_state.get("mode", "BOOT"))
            has_status_error = any(v is not None for v in status_packet_errors.values())
            has_stall = bool(stall_info.get("triggered_sides"))

            if mode in {"ALIGN", "RUN"} and (has_status_error or has_stall):
                supervisor_state["fault_hold_action"] = _build_hold_action(observation, latest_action)
                supervisor_state["fault_reason"] = "status_packet_error" if has_status_error else "stall_detected"
                _set_mode(
                    supervisor_state,
                    mode="FAULT_HOLD",
                    reason=str(supervisor_state["fault_reason"]),
                    host_arm_debug=host_arm_debug,
                    extra={
                        "status_packet_errors": status_packet_errors,
                        "stall_triggered_sides": stall_info.get("triggered_sides", []),
                    },
                )

            mode = str(supervisor_state.get("mode", "BOOT"))
            if mode == "FAULT_HOLD":
                if not has_status_error and not has_stall:
                    supervisor_state["fault_clean_frames"] = int(supervisor_state.get("fault_clean_frames", 0)) + 1
                else:
                    supervisor_state["fault_clean_frames"] = 0

                mode_elapsed = time.monotonic() - float(supervisor_state.get("mode_enter_t", time.monotonic()))
                if mode_elapsed >= float(supervisor_cfg["fault_relax_after_s"]):
                    _set_mode(
                        supervisor_state,
                        mode="FAULT_RELAX",
                        reason="fault_hold_timeout",
                        host_arm_debug=host_arm_debug,
                        extra={"mode_elapsed_s": mode_elapsed},
                    )
                elif (
                    mode_elapsed >= float(supervisor_cfg["fault_hold_min_s"])
                    and int(supervisor_state.get("fault_clean_frames", 0))
                    >= int(supervisor_cfg["fault_resume_clean_frames"])
                    and latest_action is not None
                    and extract_arm_positions(latest_action)
                ):
                    supervisor_state["align_target_arm"] = extract_arm_positions(latest_action)
                    supervisor_state["align_stable_frames"] = 0
                    _set_mode(
                        supervisor_state,
                        mode="ALIGN",
                        reason="fault_recovered_realign",
                        host_arm_debug=host_arm_debug,
                    )

            mode = str(supervisor_state.get("mode", "BOOT"))
            if mode == "FAULT_RELAX" and not bool(supervisor_state.get("relax_sent", False)):
                robot.watchdog_stop_and_relax()
                supervisor_state["relax_sent"] = True

            received_arm_target = extract_arm_positions(latest_action if latest_action is not None else {})
            target_adjustments = _compute_arm_target_adjustments(received_arm_target, sent_arm_target)

            host_arm_debug.record(
                "host_sent_action",
                {
                    "mode": mode,
                    "received_this_loop": received_this_loop,
                    "latest_action_received_t": latest_action_received_t,
                    "sent_arm_target": sent_arm_target,
                    "target_adjustments": target_adjustments,
                    "sent_base_target": {
                        "x.vel": float(sent_action.get("x.vel", 0.0)),
                        "y.vel": float(sent_action.get("y.vel", 0.0)),
                        "theta.vel": float(sent_action.get("theta.vel", 0.0)),
                        "z.pos": float(sent_action.get("z.pos", 0.0)),
                    },
                    "status_packet_errors": status_packet_errors,
                    "supervisor_state": _serialize_supervisor_state(supervisor_state),
                },
            )

            host_arm_debug.record(
                "host_observation",
                {
                    "mode": mode,
                    "observed_arm_position": observed_arm,
                    "stall_info": stall_info,
                    "status_packet_errors": status_packet_errors,
                    "supervisor_state": _serialize_supervisor_state(supervisor_state),
                    "watchdog_active": watchdog_active,
                },
            )

            # Send the observation to the remote agent
            try:
                if observation is None or observation == {}:
                    observation = previous_observation
                    logging.warning("No observation received. Sending previous observation.")

                if observation is not None and observation != {}:
                    robot_state = robot.protobuf_converter.observation_to_protobuf(observation)
                    host.zmq_observation_socket.send(robot_state.SerializeToString(), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.info("Dropping observation, no client connected")
            except Exception as e:
                logging.error("Failed to send observation: %s", e)

            elapsed = time.time() - loop_start_time
            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
            duration = time.perf_counter() - start

        print("Cycle time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down Sourccey Host.")
        robot.disconnect()
        host.disconnect()
        host_arm_debug.close()

    logging.info("Finished Sourccey cleanly")


def _silence_camera_warnings_for_host() -> None:
    """
    Host-mode ergonomics: camera disconnects are expected sometimes; don't spam WARNING logs.
    """
    # Silence our OpenCV camera wrapper warnings
    logging.getLogger("lerobot.cameras.opencv.camera_opencv").setLevel(logging.ERROR)
    # Silence Sourccey camera fallback warnings (black frame fallback)
    logging.getLogger("lerobot.robots.sourccey.sourccey.sourccey.sourccey").setLevel(logging.ERROR)

    # Best-effort: silence OpenCV's own internal logging if available
    try:
        import cv2  # type: ignore

        # OpenCV 4.x often exposes cv2.utils.logging.setLogLevel
        if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging") and hasattr(cv2.utils.logging, "setLogLevel"):
            level = getattr(cv2.utils.logging, "LOG_LEVEL_ERROR", None)
            if level is None:
                level = getattr(cv2.utils.logging, "LOG_LEVEL_SILENT", None)
            if level is not None:
                cv2.utils.logging.setLogLevel(level)
            return

        # Some builds expose cv2.setLogLevel
        if hasattr(cv2, "setLogLevel"):
            level = getattr(cv2, "LOG_LEVEL_ERROR", None)
            if level is None:
                level = getattr(cv2, "LOG_LEVEL_SILENT", None)
            if level is not None:
                cv2.setLogLevel(level)
    except Exception:
        # Don't fail startup just because OpenCV logging APIs differ
        pass


if __name__ == "__main__":
    main()
