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
from lerobot.motors.motors_bus import get_address

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


def _calibration_to_dict(calibration: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {
        motor: {
            "id": int(cal.id),
            "drive_mode": int(cal.drive_mode),
            "homing_offset": int(cal.homing_offset),
            "range_min": int(cal.range_min),
            "range_max": int(cal.range_max),
        }
        for motor, cal in calibration.items()
    }


def _safe_bus_read_with_error(bus: Any, data_name: str, motor: str) -> tuple[float | None, str | None]:
    try:
        return float(bus.read(data_name, motor, normalize=False, num_retry=0)), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _bus_supports_register(bus: Any, motor: str, data_name: str) -> bool:
    try:
        model = bus.motors[motor].model
        get_address(bus.model_ctrl_table, model, data_name)
        return True
    except Exception:
        return False


SHOULDER_LIFT_DYNAMIC_REGISTERS = (
    ("goal_position_raw", "Goal_Position"),
    ("present_position_raw", "Present_Position"),
    ("present_current_raw", "Present_Current"),
    ("present_voltage_raw", "Present_Voltage"),
    ("present_temperature_raw", "Present_Temperature"),
    ("present_load_raw", "Present_Load"),
    ("torque_enable_raw", "Torque_Enable"),
    ("moving_raw", "Moving"),
    ("status_raw", "Status"),
    ("torque_limit_raw", "Torque_Limit"),
)


SHOULDER_LIFT_CONFIG_REGISTERS = (
    ("operating_mode_raw", "Operating_Mode"),
    ("p_coefficient_raw", "P_Coefficient"),
    ("i_coefficient_raw", "I_Coefficient"),
    ("d_coefficient_raw", "D_Coefficient"),
    ("max_torque_limit_raw", "Max_Torque_Limit"),
    ("protection_current_raw", "Protection_Current"),
    ("protective_torque_raw", "Protective_Torque"),
    ("overload_torque_raw", "Overload_Torque"),
    ("over_current_protection_time_raw", "Over_Current_Protection_Time"),
    ("minimum_startup_force_raw", "Minimum_Startup_Force"),
    ("acceleration_raw", "Acceleration"),
    ("min_voltage_limit_raw", "Min_Voltage_Limit"),
    ("max_voltage_limit_raw", "Max_Voltage_Limit"),
    ("unloading_condition_raw", "Unloading_Condition"),
    ("led_alarm_condition_raw", "LED_Alarm_Condition"),
)


ARM_SIDE_KEYS = {
    "left": tuple(key for key in ARM_POSITION_KEYS if key.startswith("left_")),
    "right": tuple(key for key in ARM_POSITION_KEYS if key.startswith("right_")),
}


def _new_arm_command_governor_config() -> dict[str, Any]:
    return {
        "enabled": _env_flag("SOURCCEY_ARM_GOVERNOR_ENABLED", True),
        "base_step": max(_env_float("SOURCCEY_ARM_GOVERNOR_STEP", 10.0), 0.0),
        "base_step_shoulder_lift": max(_env_float("SOURCCEY_ARM_GOVERNOR_SHOULDER_LIFT_STEP", 6.0), 0.0),
        "settle_step": max(_env_float("SOURCCEY_ARM_GOVERNOR_SETTLE_STEP", 6.0), 0.0),
        "settle_step_shoulder_lift": max(
            _env_float("SOURCCEY_ARM_GOVERNOR_SETTLE_SHOULDER_LIFT_STEP", 3.0), 0.0
        ),
        "settle_engage_delta": max(_env_float("SOURCCEY_ARM_GOVERNOR_SETTLE_ENGAGE_DELTA", 35.0), 0.0),
        "settle_complete_delta": max(_env_float("SOURCCEY_ARM_GOVERNOR_SETTLE_COMPLETE_DELTA", 10.0), 0.0),
        "settle_max_duration_s": max(_env_float("SOURCCEY_ARM_GOVERNOR_SETTLE_MAX_DURATION_S", 2.5), 0.0),
        "rearm_gap_s": max(_env_float("SOURCCEY_ARM_GOVERNOR_REARM_GAP_S", 1.0), 0.0),
    }


def _new_arm_command_governor_state() -> dict[str, Any]:
    return {
        "startup_settle_active": False,
        "startup_settle_started_t": None,
        "last_arm_command_t": None,
    }


def _serialize_arm_command_governor_state(governor_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "startup_settle_active": bool(governor_state.get("startup_settle_active", False)),
        "startup_settle_started_t": governor_state.get("startup_settle_started_t"),
        "last_arm_command_t": governor_state.get("last_arm_command_t"),
    }


def _arm_governor_step_for_joint(
    key: str,
    *,
    startup_settle_active: bool,
    governor_config: dict[str, Any],
) -> float:
    is_shoulder_lift = key.endswith("shoulder_lift.pos")
    if startup_settle_active:
        return (
            float(governor_config["settle_step_shoulder_lift"])
            if is_shoulder_lift
            else float(governor_config["settle_step"])
        )
    return (
        float(governor_config["base_step_shoulder_lift"])
        if is_shoulder_lift
        else float(governor_config["base_step"])
    )


def _apply_arm_command_governor(
    *,
    action: dict[str, Any],
    previous_observed_arm_position: dict[str, float],
    last_sent_arm_target: dict[str, float],
    governor_state: dict[str, Any],
    governor_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not governor_config.get("enabled", False):
        return action, {
            "enabled": False,
            "applied": False,
            "startup_settle_active": False,
            "clamped_joint_count": 0,
        }

    received_arm_target = extract_arm_positions(action)
    if not received_arm_target:
        return action, {
            "enabled": True,
            "applied": False,
            "startup_settle_active": bool(governor_state.get("startup_settle_active", False)),
            "clamped_joint_count": 0,
            "reason": "no_arm_targets",
        }

    now = time.monotonic()
    last_arm_command_t = governor_state.get("last_arm_command_t")
    if last_arm_command_t is None or (now - float(last_arm_command_t)) >= float(governor_config["rearm_gap_s"]):
        governor_state["startup_settle_active"] = False
        governor_state["startup_settle_started_t"] = None

    common_keys = [k for k in received_arm_target if k in previous_observed_arm_position]
    max_abs_delta_target_vs_prev_obs = (
        max(abs(received_arm_target[k] - previous_observed_arm_position[k]) for k in common_keys)
        if common_keys
        else None
    )

    if (
        not governor_state.get("startup_settle_active", False)
        and max_abs_delta_target_vs_prev_obs is not None
        and max_abs_delta_target_vs_prev_obs >= float(governor_config["settle_engage_delta"])
    ):
        governor_state["startup_settle_active"] = True
        governor_state["startup_settle_started_t"] = now

    startup_settle_active = bool(governor_state.get("startup_settle_active", False))
    if startup_settle_active and governor_state.get("startup_settle_started_t") is None:
        governor_state["startup_settle_started_t"] = now

    adjusted_action = dict(action)
    clamped_joints: dict[str, float] = {}
    for key, target in received_arm_target.items():
        baseline = last_sent_arm_target.get(key)
        if baseline is None:
            baseline = previous_observed_arm_position.get(key, target)

        step = _arm_governor_step_for_joint(
            key,
            startup_settle_active=startup_settle_active,
            governor_config=governor_config,
        )
        if step <= 0.0:
            continue

        delta = float(target - baseline)
        if abs(delta) <= step:
            continue
        adjusted = float(baseline + (step if delta > 0 else -step))
        adjusted_action[key] = adjusted
        clamped_joints[key] = adjusted

    adjusted_arm_target = extract_arm_positions(adjusted_action)
    common_adjusted_keys = [k for k in adjusted_arm_target if k in previous_observed_arm_position]
    max_abs_delta_sent_vs_prev_obs = (
        max(abs(adjusted_arm_target[k] - previous_observed_arm_position[k]) for k in common_adjusted_keys)
        if common_adjusted_keys
        else None
    )

    elapsed_s = None
    if governor_state.get("startup_settle_active", False):
        started = governor_state.get("startup_settle_started_t")
        if started is not None:
            elapsed_s = float(now - float(started))
            if (
                max_abs_delta_target_vs_prev_obs is not None
                and max_abs_delta_target_vs_prev_obs <= float(governor_config["settle_complete_delta"])
            ):
                governor_state["startup_settle_active"] = False
                governor_state["startup_settle_started_t"] = None

    governor_state["last_arm_command_t"] = now
    return adjusted_action, {
        "enabled": True,
        "applied": len(clamped_joints) > 0,
        "startup_settle_active": bool(governor_state.get("startup_settle_active", False)),
        "elapsed_s": elapsed_s,
        "max_abs_delta_target_vs_prev_obs": max_abs_delta_target_vs_prev_obs,
        "max_abs_delta_sent_vs_prev_obs": max_abs_delta_sent_vs_prev_obs,
        "clamped_joint_count": len(clamped_joints),
        "clamped_joints": clamped_joints,
    }


def _decode_bitfield(value: float | None) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        intval = int(value)
    except (TypeError, ValueError):
        return None
    bits_set = [bit for bit in range(8) if intval & (1 << bit)]
    return {"value": intval, "binary": f"0b{intval:08b}", "bits_set": bits_set}


def _read_shoulder_lift_register_set(
    arm: Any,
    register_pairs: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    bus = arm.bus
    values: dict[str, float | None] = {}
    read_errors: dict[str, str] = {}
    unsupported_registers: list[str] = []

    for out_key, register_name in register_pairs:
        if not _bus_supports_register(bus, "shoulder_lift", register_name):
            values[out_key] = None
            unsupported_registers.append(register_name)
            continue

        value, error = _safe_bus_read_with_error(bus, register_name, "shoulder_lift")
        values[out_key] = value
        if error is not None:
            read_errors[register_name] = error

    decoded_flags = {
        "status": _decode_bitfield(values.get("status_raw")),
        "unloading_condition": _decode_bitfield(values.get("unloading_condition_raw")),
        "led_alarm_condition": _decode_bitfield(values.get("led_alarm_condition_raw")),
    }

    payload: dict[str, Any] = {
        **values,
        "unsupported_registers": unsupported_registers,
        "read_errors": read_errors,
        "decoded_flags": decoded_flags,
    }
    return payload


def _read_shoulder_lift_diagnostics(robot: Sourccey) -> dict[str, dict[str, Any]]:
    def _read_arm(arm: Any) -> dict[str, Any]:
        return _read_shoulder_lift_register_set(arm, SHOULDER_LIFT_DYNAMIC_REGISTERS)

    return {
        "left_shoulder_lift": _read_arm(robot.left_arm),
        "right_shoulder_lift": _read_arm(robot.right_arm),
    }


def _read_shoulder_lift_config_snapshot(robot: Sourccey) -> dict[str, dict[str, Any]]:
    return {
        "left_shoulder_lift": _read_shoulder_lift_register_set(robot.left_arm, SHOULDER_LIFT_CONFIG_REGISTERS),
        "right_shoulder_lift": _read_shoulder_lift_register_set(robot.right_arm, SHOULDER_LIFT_CONFIG_REGISTERS),
    }


def _new_arm_freeze_state() -> dict[str, dict[str, Any]]:
    return {
        "left": {
            "active": False,
            "good_status_packets": 0,
            "hold_targets": {},
            "last_error": None,
        },
        "right": {
            "active": False,
            "good_status_packets": 0,
            "hold_targets": {},
            "last_error": None,
        },
    }


def _extract_side_arm_targets(arm_data: dict[str, float], side: str) -> dict[str, float]:
    return {key: float(arm_data[key]) for key in ARM_SIDE_KEYS[side] if key in arm_data}


def _serialize_arm_freeze_state(arm_freeze_state: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        side: {
            "active": bool(state.get("active")),
            "good_status_packets": int(state.get("good_status_packets", 0)),
            "hold_target_count": len(state.get("hold_targets") or {}),
            "last_error": state.get("last_error"),
        }
        for side, state in arm_freeze_state.items()
    }


def _apply_arm_freeze_to_action(
    action: dict[str, Any],
    arm_freeze_state: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    adjusted = dict(action)
    applied_freeze: dict[str, dict[str, float]] = {}
    for side, state in arm_freeze_state.items():
        if not state.get("active"):
            continue
        hold_targets = state.get("hold_targets") or {}
        if not hold_targets:
            continue
        for key, value in hold_targets.items():
            adjusted[key] = float(value)
        applied_freeze[side] = {key: float(value) for key, value in hold_targets.items()}
    return adjusted, applied_freeze


def _extract_status_packet_errors_from_sent_action(sent_action: dict[str, Any]) -> dict[str, str | None]:
    left_error = sent_action.get("left_status_packet_error")
    right_error = sent_action.get("right_status_packet_error")
    return {
        "left": str(left_error) if left_error else None,
        "right": str(right_error) if right_error else None,
    }


def _extract_overload_errors_from_shoulder_lift_status(robot: Sourccey) -> dict[str, str | None]:
    def _read_overload(arm: Any, side: str) -> str | None:
        bus = arm.bus
        if not _bus_supports_register(bus, "shoulder_lift", "Status"):
            return None

        _value, error = _safe_bus_read_with_error(bus, "Status", "shoulder_lift")
        if error is None:
            return None
        if "Overload error" in error:
            return f"{side}_shoulder_lift_status_read: {error}"
        return None

    return {
        "left": _read_overload(robot.left_arm, "left"),
        "right": _read_overload(robot.right_arm, "right"),
    }


def _merge_arm_errors(
    primary: dict[str, str | None],
    secondary: dict[str, str | None],
) -> dict[str, str | None]:
    merged: dict[str, str | None] = {}
    for side in ("left", "right"):
        merged[side] = primary.get(side) or secondary.get(side)
    return merged


def _update_arm_freeze_state(
    *,
    arm_freeze_state: dict[str, dict[str, Any]],
    status_packet_errors: dict[str, str | None],
    observed_arm_position: dict[str, float],
    last_sent_arm_target: dict[str, float],
    resume_good_status_packets: int,
    host_arm_debug: ArmDebugCapture,
) -> None:
    for side, error in status_packet_errors.items():
        side_state = arm_freeze_state[side]
        if error is not None:
            side_state["good_status_packets"] = 0
            side_state["last_error"] = error
            if side_state["active"]:
                continue

            hold_targets = _extract_side_arm_targets(observed_arm_position, side)
            if not hold_targets:
                hold_targets = _extract_side_arm_targets(last_sent_arm_target, side)
            side_state["active"] = True
            side_state["hold_targets"] = hold_targets

            transition = {
                "transition": "freeze_activated",
                "side": side,
                "error": error,
                "hold_targets": hold_targets,
            }
            logging.error(
                "Freezing %s arm due to shoulder_lift status read error. error=%s hold_target_count=%d",
                side,
                error,
                len(hold_targets),
            )
            host_arm_debug.record_session("host_arm_freeze_transition", transition)
            continue

        if not side_state["active"]:
            continue

        side_state["good_status_packets"] += 1
        if side_state["good_status_packets"] < resume_good_status_packets:
            continue

        transition = {
            "transition": "freeze_released",
            "side": side,
            "good_status_packets": side_state["good_status_packets"],
            "resume_threshold": resume_good_status_packets,
        }
        logging.warning(
            "Resuming %s arm after %d clean shoulder_lift status packets.",
            side,
            side_state["good_status_packets"],
        )
        host_arm_debug.record_session("host_arm_freeze_transition", transition)
        side_state["active"] = False
        side_state["good_status_packets"] = 0
        side_state["hold_targets"] = {}
        side_state["last_error"] = None


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


def _build_shoulder_lift_tracking(
    sent_arm_target: dict[str, float],
    observed_arm_position: dict[str, float],
    shoulder_lift_diag: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    tracking: dict[str, Any] = {}
    key_to_side = {
        "left_shoulder_lift.pos": "left_shoulder_lift",
        "right_shoulder_lift.pos": "right_shoulder_lift",
    }
    errors: list[float] = []

    for pos_key, side_key in key_to_side.items():
        sent = sent_arm_target.get(pos_key)
        obs = observed_arm_position.get(pos_key)
        if sent is None or obs is None:
            continue

        target_minus_observed = float(sent - obs)
        errors.append(abs(target_minus_observed))

        side_diag = (shoulder_lift_diag or {}).get(side_key, {})
        goal_raw = side_diag.get("goal_position_raw")
        present_raw = side_diag.get("present_position_raw")
        present_current = side_diag.get("present_current_raw")

        raw_gap = None
        if goal_raw is not None and present_raw is not None:
            raw_gap = float(goal_raw - present_raw)

        stall_like = bool(
            raw_gap is not None
            and abs(raw_gap) >= 500
            and present_current is not None
            and abs(float(present_current)) >= 600
        )

        tracking[side_key] = {
            "sent_target": float(sent),
            "observed_position": float(obs),
            "target_minus_observed": target_minus_observed,
            "raw_goal_minus_present": raw_gap,
            "stall_like_signature": stall_like,
        }

    tracking["max_abs_target_minus_observed"] = max(errors) if errors else None
    tracking["stall_like_signature_arms"] = [
        side
        for side, payload in tracking.items()
        if isinstance(payload, dict) and payload.get("stall_like_signature")
    ]
    return tracking


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

    host_arm_debug = ArmDebugCapture(
        enabled=_env_flag("SOURCCEY_HOST_ARM_DEBUG_CAPTURE", True),
        duration_s=_env_float("SOURCCEY_HOST_ARM_DEBUG_DURATION_S", 5.0),
        motion_threshold=_env_float("SOURCCEY_HOST_ARM_DEBUG_MOTION_THRESHOLD", 1.0),
        label="host",
        capture_path=os.getenv("SOURCCEY_HOST_ARM_DEBUG_PATH"),
    )
    if host_arm_debug.path:
        logging.warning("Host arm debug capture enabled. Writing to %s", host_arm_debug.path)
    try:
        host_arm_debug.record_session(
            "host_calibration_snapshot",
            {
                "robot_id": robot.id,
                "left_calibration_file": str(robot.left_arm.calibration_fpath),
                "right_calibration_file": str(robot.right_arm.calibration_fpath),
                "left_file_calibration": _calibration_to_dict(robot.left_arm.calibration),
                "right_file_calibration": _calibration_to_dict(robot.right_arm.calibration),
                "left_motor_calibration": _calibration_to_dict(robot.left_arm.bus.read_calibration()),
                "right_motor_calibration": _calibration_to_dict(robot.right_arm.bus.read_calibration()),
            },
        )
    except Exception as e:
        host_arm_debug.record_session(
            "host_calibration_snapshot_error",
            {"error": str(e)},
        )
    try:
        host_arm_debug.record_session(
            "host_shoulder_lift_config_snapshot",
            {
                "left_arm_port": str(robot.left_arm.bus.port),
                "right_arm_port": str(robot.right_arm.bus.port),
                "register_snapshot": _read_shoulder_lift_config_snapshot(robot),
            },
        )
    except Exception as e:
        host_arm_debug.record_session(
            "host_shoulder_lift_config_snapshot_error",
            {"error": str(e)},
        )

    print("Waiting for commands...")

    last_cmd_time = time.time()
    watchdog_active = False
    resume_good_status_packets = max(_env_int("SOURCCEY_ARM_FREEZE_RESUME_STATUS_PACKETS", 1), 1)
    arm_freeze_state = _new_arm_freeze_state()
    arm_governor_config = _new_arm_command_governor_config()
    arm_governor_state = _new_arm_command_governor_state()
    host_arm_debug.record_session(
        "host_arm_freeze_config",
        {
            "resume_good_status_packets": resume_good_status_packets,
        },
    )
    host_arm_debug.record_session(
        "host_arm_command_governor_config",
        {
            "config": arm_governor_config,
            "state": _serialize_arm_command_governor_state(arm_governor_state),
        },
    )

    try:
        # Business logic
        start = time.perf_counter()
        duration = 0

        try:
            observation = robot.get_observation()
        except Exception:
            observation = {}
        previous_observation = observation
        last_sent_arm_target: dict[str, float] = {}
        status_packet_errors: dict[str, str | None] = {"left": None, "right": None}
        shoulder_lift_overload_errors: dict[str, str | None] = {"left": None, "right": None}
        freeze_trigger_errors: dict[str, str | None] = {"left": None, "right": None}
        while duration < host.connection_time_s:
            loop_start_time = time.time()
            try:
                # Receive protobuf message instead of JSON
                msg_bytes = host.zmq_cmd_socket.recv(zmq.NOBLOCK)

                # Convert protobuf to action dictionary using existing method
                robot_action = sourccey_pb2.SourcceyRobotAction()
                robot_action.ParseFromString(msg_bytes)

                data = robot.protobuf_converter.protobuf_to_action(robot_action)

                host_arm_debug.maybe_start(action=data, observation=previous_observation)
                received_arm_target = extract_arm_positions(data)
                previous_obs_arm = extract_arm_positions(previous_observation)
                governed_action, governor_info = _apply_arm_command_governor(
                    action=data,
                    previous_observed_arm_position=previous_obs_arm,
                    last_sent_arm_target=last_sent_arm_target,
                    governor_state=arm_governor_state,
                    governor_config=arm_governor_config,
                )
                governed_arm_target = extract_arm_positions(governed_action)
                action_to_send, applied_freeze = _apply_arm_freeze_to_action(governed_action, arm_freeze_state)
                freeze_input_arm_target = extract_arm_positions(action_to_send)

                common_keys = [k for k in received_arm_target if k in previous_obs_arm]
                max_abs_delta_target_vs_prev_obs = (
                    max(abs(received_arm_target[k] - previous_obs_arm[k]) for k in common_keys)
                    if common_keys
                    else None
                )
                host_arm_debug.record(
                    "host_received_action",
                    {
                        "received_arm_target": received_arm_target,
                        "previous_observed_arm_position": previous_obs_arm,
                        "max_abs_delta_target_vs_prev_obs": max_abs_delta_target_vs_prev_obs,
                        "received_base_target": {
                            "x.vel": float(data.get("x.vel", 0.0)),
                            "y.vel": float(data.get("y.vel", 0.0)),
                            "theta.vel": float(data.get("theta.vel", 0.0)),
                            "z.pos": float(data.get("z.pos", 0.0)),
                        },
                        "governed_arm_target": governed_arm_target,
                        "governor_info": governor_info,
                        "arm_freeze_state": _serialize_arm_freeze_state(arm_freeze_state),
                        "arm_freeze_applied": applied_freeze,
                        "arm_governor_state": _serialize_arm_command_governor_state(arm_governor_state),
                    },
                )

                # Send action to robot
                _action_sent = robot.send_action(action_to_send)
                status_packet_errors = _extract_status_packet_errors_from_sent_action(_action_sent)
                shoulder_lift_overload_errors = _extract_overload_errors_from_shoulder_lift_status(robot)
                freeze_trigger_errors = _merge_arm_errors(status_packet_errors, shoulder_lift_overload_errors)
                _update_arm_freeze_state(
                    arm_freeze_state=arm_freeze_state,
                    status_packet_errors=freeze_trigger_errors,
                    observed_arm_position=previous_obs_arm,
                    last_sent_arm_target=last_sent_arm_target,
                    resume_good_status_packets=resume_good_status_packets,
                    host_arm_debug=host_arm_debug,
                )
                sent_arm_target = extract_arm_positions(_action_sent)
                target_adjustments = _compute_arm_target_adjustments(received_arm_target, sent_arm_target)
                governor_adjustments = _compute_arm_target_adjustments(received_arm_target, governed_arm_target)
                freeze_adjustments = _compute_arm_target_adjustments(governed_arm_target, freeze_input_arm_target)
                host_arm_debug.record(
                    "host_sent_action",
                    {
                        "sent_arm_target": sent_arm_target,
                        "target_adjustments": target_adjustments,
                        "governor_adjustments": governor_adjustments,
                        "freeze_adjustments": freeze_adjustments,
                        "sent_base_target": {
                            "x.vel": float(_action_sent.get("x.vel", 0.0)),
                            "y.vel": float(_action_sent.get("y.vel", 0.0)),
                            "theta.vel": float(_action_sent.get("theta.vel", 0.0)),
                            "z.pos": float(_action_sent.get("z.pos", 0.0)),
                        },
                        "governor_info": governor_info,
                        "arm_governor_state": _serialize_arm_command_governor_state(arm_governor_state),
                        "arm_freeze_state": _serialize_arm_freeze_state(arm_freeze_state),
                        "arm_freeze_applied": applied_freeze,
                        "status_packet_errors": status_packet_errors,
                        "shoulder_lift_overload_errors": shoulder_lift_overload_errors,
                        "freeze_trigger_errors": freeze_trigger_errors,
                    },
                )
                last_sent_arm_target = sent_arm_target

                # Update the robot
                robot.update()

                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                if not watchdog_active:
                    # logging.warning("No command available")
                    pass
            except Exception as e:
                logging.error("Message fetching failed: %s", e)

            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.debug(
                    f"Command not received for more than {host.watchdog_timeout_ms} milliseconds. "
                    "Stopping base and releasing arm torque."
                )
                watchdog_active = True
                robot.watchdog_stop_and_relax()

            if observation is not None and observation != {}:
                previous_observation = observation
            observation = robot.get_observation()
            observed_arm_position = extract_arm_positions(observation)
            shoulder_lift_diag = (
                _read_shoulder_lift_diagnostics(robot) if host_arm_debug.is_active else None
            )
            shoulder_lift_tracking = (
                _build_shoulder_lift_tracking(last_sent_arm_target, observed_arm_position, shoulder_lift_diag)
                if host_arm_debug.is_active
                else None
            )
            host_arm_debug.record(
                "host_observation",
                {
                    "observed_arm_position": observed_arm_position,
                    "shoulder_lift_diagnostics": shoulder_lift_diag,
                    "shoulder_lift_tracking": shoulder_lift_tracking,
                    "status_packet_errors": status_packet_errors,
                    "shoulder_lift_overload_errors": shoulder_lift_overload_errors,
                    "freeze_trigger_errors": freeze_trigger_errors,
                    "arm_governor_state": _serialize_arm_command_governor_state(arm_governor_state),
                    "arm_freeze_state": _serialize_arm_freeze_state(arm_freeze_state),
                    "watchdog_active": watchdog_active,
                },
            )

            # Send the observation to the remote agent
            try:
                # Don't send an empty observation
                if observation is None or observation == {}:
                    observation = previous_observation
                    logging.warning("No observation received. Sending previous observation.")

                if observation is not None and observation != {}:
                    # Convert observation to protobuf using existing method
                    robot_state = robot.protobuf_converter.observation_to_protobuf(observation)

                    # Send protobuf message instead of JSON
                    host.zmq_observation_socket.send(robot_state.SerializeToString(), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.info("Dropping observation, no client connected")
            except Exception as e:
                logging.error(f"Failed to send observation: {e}")

            # Ensure a short sleep to avoid overloading the CPU.
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
