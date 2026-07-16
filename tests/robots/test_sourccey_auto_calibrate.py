#!/usr/bin/env python

import json

import pytest

import lerobot.robots.sourccey.sourccey.sourccey_z_actuator.sourccey_z_calibrator as z_calibrator_module
import lerobot.robots.sourccey.sourccey.sourccey.sourccey as sourccey_module
import lerobot.robots.sourccey.sourccey.sourccey_z_actuator.sourccey_z_actuator as z_actuator_module
import lerobot.scripts.sourccey.calibration.auto_calibrate as auto_calibrate_script
from lerobot.robots.sourccey.sourccey.sourccey_follower.sourccey_follower_calibrator import (
    SourcceyFollowerCalibrator,
)
from lerobot.robots.sourccey.sourccey.sourccey_z_actuator.sourccey_z_calibrator import (
    CalibrationPhaseError,
    SourcceyZCalibrator,
    ZCalibrationResult,
)
from lerobot.robots.sourccey.sourccey.sourccey.sourccey import Sourccey
from lerobot.robots.sourccey.sourccey.sourccey_z_actuator.sourccey_z_actuator import (
    SourcceyZActuator,
    ZSensor,
)
from lerobot.teleoperators.sourccey.sourccey.bi_sourccey_leader.bi_sourccey_leader import (
    BiSourcceyLeader,
)


class _DummyArm:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[dict] = []
        self.is_connected = False

    def auto_calibrate(self, **kwargs) -> None:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error

    def connect(self, calibrate: bool = True) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False


class _DummyZCalibrator:
    def __init__(self, result: ZCalibrationResult | None = None):
        self.calls: list[bool] = []
        self.result = result or ZCalibrationResult(
            raw_bottom=900,
            raw_top=120,
            raw_min=120,
            raw_max=900,
            invert=True,
        )

    def auto_calibrate(self, *, full_reset: bool = False) -> ZCalibrationResult | None:
        self.calls.append(full_reset)
        return self.result


class _DummyZActuator:
    def __init__(self):
        self.calibrator = _DummyZCalibrator()
        self.use_z_actuator = True


class _FailingDCMotorController:
    def __init__(self):
        self.is_connected = False

    def connect(self) -> None:
        raise RuntimeError("gpiozero hardware not available")

    def disconnect(self) -> None:
        self.is_connected = False

    def set_velocities(self, _motors) -> None:
        return None


class _DummyCamera:
    def __init__(self):
        self.is_connected = False

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False


class _ObservationArm:
    def __init__(self, observation: dict[str, float]) -> None:
        self._observation = observation

    def get_observation(self) -> dict[str, float]:
        return dict(self._observation)


class _ObservationBase:
    def get_velocities(self) -> dict[str, float]:
        return {
            "front_left": 0.0,
            "front_right": 0.0,
            "rear_left": 0.0,
            "rear_right": 0.0,
        }


class _DummyDriver:
    def __init__(self) -> None:
        self.velocity_calls: list[tuple[object, float, bool, bool]] = []

    def set_velocity(self, motor, velocity, normalize=True, instant=True) -> None:
        self.velocity_calls.append((motor, float(velocity), bool(normalize), bool(instant)))
        return None


class _FollowerCalibrationBus:
    def __init__(self) -> None:
        self.motors = {"shoulder_pan": type("Motor", (), {"id": 1})()}
        self._positions = {"shoulder_pan": 1000}
        self._current_reads: dict[str, list[object]] = {"shoulder_pan": []}
        self._position_reads: dict[str, list[object]] = {"shoulder_pan": []}
        self._write_failures: dict[str, list[object]] = {"shoulder_pan": []}
        self.disable_torque_calls = 0
        self.enable_torque_calls = 0

    def disable_torque(self) -> None:
        self.disable_torque_calls += 1
        return None

    def enable_torque(self) -> None:
        self.enable_torque_calls += 1
        return None

    def write_calibration(self, _calibration) -> None:
        return None

    def read(self, register: str, motor_name: str, normalize: bool = False):
        if register == "Present_Current":
            if self._current_reads[motor_name]:
                value = self._current_reads[motor_name].pop(0)
                if isinstance(value, BaseException):
                    raise value
                return value
            return 0

        if register == "Present_Position":
            if self._position_reads[motor_name]:
                value = self._position_reads[motor_name].pop(0)
                if isinstance(value, BaseException):
                    raise value
                return value
            return self._positions[motor_name]

        raise KeyError(register)

    def write(self, register: str, motor_name: str, value, normalize: bool = False) -> None:
        if register != "Goal_Position":
            raise KeyError(register)

        if self._write_failures[motor_name]:
            failure = self._write_failures[motor_name].pop(0)
            if isinstance(failure, BaseException):
                raise failure

        self._positions[motor_name] = int(value)


class _FollowerCalibrationRobot:
    def __init__(self, bus: _FollowerCalibrationBus) -> None:
        self.bus = bus
        self.config = type(
            "Config",
            (),
            {
                "orientation": "left",
                "max_current_calibration_threshold": 75,
            },
        )()
        self.calibration = {}
        self.id = "left_arm"
        self.calibration_fpath = "left_arm.json"

    @property
    def is_calibrated(self) -> bool:
        return True

    def _save_calibration(self) -> None:
        return None


class _CalibrationTestActuator:
    def __init__(self, *, invert: bool = True, motor_invert: bool = True) -> None:
        self.sensor = ZSensor(invert=invert)
        self.invert = invert
        self.motor_invert = motor_invert
        self.driver = _DummyDriver()
        self.motor = "linear_actuator"
        self.saved = False
        self.move_targets: list[float] = []

    @property
    def is_connected(self) -> bool:
        return True

    def stop_position_controller(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def _save_calibration(self) -> None:
        self.saved = True

    def move_to_position_blocking(self, target_pos_m100_100: float) -> float:
        self.move_targets.append(target_pos_m100_100)
        return target_pos_m100_100


def test_sourccey_auto_calibrate_raises_when_arm_thread_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(sourccey_module.time, "sleep", lambda _seconds: None)
    caplog.set_level("INFO", logger=sourccey_module.__name__)

    robot = Sourccey.__new__(Sourccey)
    robot.left_arm = _DummyArm(RuntimeError("left arm failed"))
    robot.right_arm = _DummyArm()
    robot.z_actuator = _DummyZActuator()
    robot._z_hardware_available = True

    with pytest.raises(
        RuntimeError,
        match=r"left arm.*Z calibration was saved successfully",
    ):
        robot.auto_calibrate(full_reset=True)

    assert robot.z_actuator.calibrator.calls == [True]
    assert robot.right_arm.calls == [{"reverse": True, "full_reset": True}]
    log_messages = [record.getMessage() for record in caplog.records]
    z_saved_index = next(
        index for index, message in enumerate(log_messages)
        if "Z auto-calibration saved successfully" in message
    )
    arm_failure_index = next(
        index for index, message in enumerate(log_messages)
        if "Auto-calibration failed for left arm" in message
    )
    assert z_saved_index < arm_failure_index


def test_sourccey_auto_calibrate_aborts_before_arms_when_z_calibration_fails() -> None:
    robot = Sourccey.__new__(Sourccey)
    robot.left_arm = _DummyArm()
    robot.right_arm = _DummyArm()
    robot._z_hardware_available = True

    class _FailingCalibrator:
        def auto_calibrate(self, *, full_reset: bool = False) -> None:
            raise RuntimeError("z failed to return to top")

    robot.z_actuator = type("Z", (), {"calibrator": _FailingCalibrator()})()

    with pytest.raises(RuntimeError, match="z failed to return to top"):
        robot.auto_calibrate(full_reset=True)

    assert robot.left_arm.calls == []
    assert robot.right_arm.calls == []


def test_bi_sourccey_leader_auto_calibrate_raises_when_arm_thread_fails() -> None:
    teleop = BiSourcceyLeader.__new__(BiSourcceyLeader)
    teleop.left_arm = _DummyArm()
    teleop.right_arm = _DummyArm(RuntimeError("right leader failed"))

    with pytest.raises(RuntimeError, match="right"):
        teleop.auto_calibrate()

    assert teleop.left_arm.calls == [{"reverse": False}]


def test_sourccey_auto_calibrate_skips_z_when_gpio_hardware_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sourccey_module.time, "sleep", lambda _seconds: None)

    robot = Sourccey.__new__(Sourccey)
    robot.left_arm = _DummyArm()
    robot.right_arm = _DummyArm()
    robot.z_actuator = _DummyZActuator()
    robot._z_hardware_available = False

    robot.auto_calibrate(full_reset=True)

    assert robot.z_actuator.calibrator.calls == []
    assert robot.left_arm.calls == [{"reverse": False, "full_reset": True}]
    assert robot.right_arm.calls == [{"reverse": True, "full_reset": True}]


def test_sourccey_connect_tolerates_missing_gpio_hardware() -> None:
    robot = Sourccey.__new__(Sourccey)
    robot.left_arm = _DummyArm()
    robot.right_arm = _DummyArm()
    robot.dc_motors_controller = _FailingDCMotorController()
    robot.z_actuator = _DummyZActuator()
    robot.z_actuator.connect = lambda: None
    robot.cameras = {"front": _DummyCamera()}
    robot._connected_cameras = set()

    robot.connect(calibrate=False)

    assert robot.left_arm.is_connected is True
    assert robot.right_arm.is_connected is True
    assert robot._z_hardware_available is False
    assert robot.z_actuator.use_z_actuator is False
    assert robot._connected_cameras == {"front"}


def test_sourccey_z_actuator_ignores_invalid_calibration_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    calibration_dir = tmp_path / "robots" / "sourccey_z_actuator"
    calibration_dir.mkdir(parents=True)
    calibration_path = calibration_dir / "sourccey_z_actuator.json"
    calibration_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(z_actuator_module, "HF_LEROBOT_CALIBRATION", tmp_path)

    actuator = SourcceyZActuator(sensor=ZSensor())

    assert actuator.calibration_fpath == calibration_path
    assert actuator.sensor.calibration_min == 0
    assert actuator.sensor.calibration_max == 1023
    assert actuator.sensor.invert is True


def test_sourccey_z_actuator_loads_valid_calibration_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    calibration_dir = tmp_path / "robots" / "sourccey_z_actuator"
    calibration_dir.mkdir(parents=True)
    calibration_path = calibration_dir / "sourccey_z_actuator.json"
    calibration_path.write_text(
        json.dumps(
            {
                "z_actuator": {
                    "raw_min": 111,
                    "raw_max": 876,
                    "invert": False,
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(z_actuator_module, "HF_LEROBOT_CALIBRATION", tmp_path)

    actuator = SourcceyZActuator(sensor=ZSensor())

    assert actuator.calibration_fpath == calibration_path
    assert actuator.sensor.calibration_min == 111
    assert actuator.sensor.calibration_max == 876
    assert actuator.sensor.invert is False
    assert actuator.invert is False


@pytest.mark.parametrize(
    ("raw_bottom", "raw_top", "expected_invert"),
    [
        (900, 120, True),
        (120, 900, False),
    ],
)
def test_sourccey_z_full_calibration_guarantees_bottom_and_top_mapping(
    monkeypatch: pytest.MonkeyPatch,
    raw_bottom: int,
    raw_top: int,
    expected_invert: bool,
) -> None:
    monkeypatch.setattr(sourccey_module.time, "sleep", lambda _seconds: None)

    actuator = _CalibrationTestActuator(invert=not expected_invert, motor_invert=True)
    calibrator = SourcceyZCalibrator(actuator)
    measured_raws = iter([raw_bottom, raw_top])

    monkeypatch.setattr(
        calibrator,
        "_wait_until_stable",
        lambda _cmd, **_kwargs: next(measured_raws),
    )
    monkeypatch.setattr(calibrator, "_read_raw", lambda: raw_top)

    result = calibrator.auto_calibrate(full_reset=True)

    assert result is not None
    assert result.invert is expected_invert
    assert actuator.saved is True
    assert actuator.move_targets == []
    assert actuator.sensor.invert is expected_invert
    assert actuator.invert is expected_invert
    assert actuator.sensor.raw_to_pos_m100_100(raw_bottom) == pytest.approx(-100.0)
    assert actuator.sensor.raw_to_pos_m100_100(raw_top) == pytest.approx(100.0)
    assert [call[1] for call in actuator.driver.velocity_calls] == [1.0, -1.0]


def test_sourccey_z_drive_uses_motor_invert_not_sensor_mapping() -> None:
    actuator = _CalibrationTestActuator(invert=False, motor_invert=True)
    calibrator = SourcceyZCalibrator(actuator)

    calibrator._drive(0.5)
    actuator.sensor.invert = True
    actuator.invert = True
    calibrator._drive(0.5)

    assert [call[1] for call in actuator.driver.velocity_calls] == [-0.5, -0.5]


def test_sourccey_z_calibrator_default_endpoint_timing() -> None:
    actuator = _CalibrationTestActuator(invert=True)
    calibrator = SourcceyZCalibrator(actuator)

    assert calibrator.stable_s == 10.0
    assert calibrator.max_phase_s == 60.0


def test_sourccey_z_full_calibration_raises_if_return_to_top_verification_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sourccey_module.time, "sleep", lambda _seconds: None)

    actuator = _CalibrationTestActuator(invert=True)
    calibrator = SourcceyZCalibrator(actuator)
    measured_raws = iter([900, 120])

    monkeypatch.setattr(
        calibrator,
        "_wait_until_stable",
        lambda _cmd, **_kwargs: next(measured_raws),
    )
    monkeypatch.setattr(calibrator, "_read_raw", lambda: 150)

    with pytest.raises(CalibrationPhaseError, match="z:return_top verification failed"):
        calibrator.auto_calibrate(full_reset=True)

    assert actuator.saved is False


def test_sourccey_z_full_calibration_logs_all_phases_in_order(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(sourccey_module.time, "sleep", lambda _seconds: None)

    actuator = _CalibrationTestActuator(invert=True)
    calibrator = SourcceyZCalibrator(actuator)
    measured_raws = iter([900, 120])

    monkeypatch.setattr(
        calibrator,
        "_wait_until_stable",
        lambda _cmd, **_kwargs: next(measured_raws),
    )
    monkeypatch.setattr(calibrator, "_read_raw", lambda: 120)
    caplog.set_level("INFO", logger=z_calibrator_module.__name__)

    calibrator.auto_calibrate(full_reset=True)

    seen_phases: list[str] = []
    for record in caplog.records:
        message = record.getMessage()
        for phase in ("seek_bottom", "return_top", "verify_top"):
            if f"z_phase={phase}" in message and (not seen_phases or seen_phases[-1] != phase):
                seen_phases.append(phase)

    assert seen_phases == ["seek_bottom", "return_top", "verify_top"]


def test_sourccey_z_return_to_top_allows_stable_endpoint_without_min_travel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sourccey_module.time, "sleep", lambda _seconds: None)

    actuator = _CalibrationTestActuator(invert=True)
    calibrator = SourcceyZCalibrator(actuator)
    wait_calls: list[dict[str, object]] = []

    def _wait(_cmd: float, **kwargs) -> int:
        wait_calls.append(kwargs)
        return 120

    monkeypatch.setattr(calibrator, "_wait_until_stable", _wait)
    monkeypatch.setattr(calibrator, "_read_raw", lambda: 120)

    calibrator._return_to_top_and_verify()

    assert wait_calls == [
        {
            "phase": "return_top",
            "min_elapsed_s": calibrator.RETURN_TOP_MIN_DRIVE_S,
            "min_travel_raw": calibrator.RETURN_TOP_MIN_TRAVEL_RAW,
        }
    ]


def test_sourccey_z_seek_bottom_allows_starting_at_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sourccey_module.time, "sleep", lambda _seconds: None)

    actuator = _CalibrationTestActuator(invert=True)
    calibrator = SourcceyZCalibrator(actuator)
    wait_calls: list[dict[str, object]] = []

    def _wait(_cmd: float, **kwargs) -> int:
        wait_calls.append(kwargs)
        return 900

    monkeypatch.setattr(calibrator, "_wait_until_stable", _wait)
    monkeypatch.setattr(calibrator, "_return_to_top_and_verify", lambda: 120)

    calibrator.auto_calibrate(full_reset=True)

    assert wait_calls == [
        {
            "phase": "seek_bottom",
            "min_elapsed_s": calibrator.SEEK_BOTTOM_MIN_DRIVE_S,
            "min_travel_raw": calibrator.SEEK_BOTTOM_MIN_TRAVEL_RAW,
        }
    ]


def test_sourccey_z_full_calibration_reverses_from_immediate_stable_bottom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sourccey_module.time, "sleep", lambda _seconds: None)

    actuator = _CalibrationTestActuator(invert=True)
    calibrator = SourcceyZCalibrator(actuator, stable_s=0.0, sample_hz=30.0, max_phase_s=0.7)
    calibrator.SEEK_BOTTOM_MIN_DRIVE_S = 0.3
    calibrator.SEEK_BOTTOM_MIN_TRAVEL_RAW = 0

    monotonic_time = {"value": 0.0}

    def _monotonic() -> float:
        monotonic_time["value"] += 0.11
        return monotonic_time["value"]

    monkeypatch.setattr(z_calibrator_module.time, "monotonic", _monotonic)
    monkeypatch.setattr(calibrator, "_read_raw", lambda: 500)
    monkeypatch.setattr(calibrator, "_return_to_top_and_verify", lambda: 100)

    result = calibrator.auto_calibrate(full_reset=True)

    assert result.raw_bottom == 500
    assert result.raw_top == 100


def test_sourccey_z_stability_window_rejects_slow_continuous_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actuator = _CalibrationTestActuator(invert=True)
    calibrator = SourcceyZCalibrator(actuator, stable_s=0.3, sample_hz=30.0, max_phase_s=0.8)
    monotonic_time = {"value": 0.0}
    raw_value = {"value": 100}

    def _monotonic() -> float:
        monotonic_time["value"] += 0.1
        return monotonic_time["value"]

    def _read_raw() -> int:
        raw_value["value"] += 1
        return raw_value["value"]

    monkeypatch.setattr(z_calibrator_module.time, "monotonic", _monotonic)
    monkeypatch.setattr(z_calibrator_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(calibrator, "_read_raw", _read_raw)

    with pytest.raises(TimeoutError, match="timed out waiting for stability"):
        calibrator._wait_until_stable(calibrator.up_cmd, phase="return_top")


def test_auto_calibrate_script_forwards_arm_to_device(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeRobotConfig:
        def __init__(self) -> None:
            self.id = "sourccey"

    class _DummyDevice:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def connect(self, calibrate: bool = True) -> None:
            return None

        def auto_calibrate(self, *, full_reset: bool = False, arm: str | None = None) -> None:
            self.calls.append({"full_reset": full_reset, "arm": arm})

        def disconnect(self) -> None:
            return None

    device = _DummyDevice()

    monkeypatch.setattr(auto_calibrate_script, "RobotConfig", _FakeRobotConfig)
    monkeypatch.setattr(auto_calibrate_script, "make_robot_from_config", lambda _cfg: device)
    monkeypatch.setattr(auto_calibrate_script, "init_logging", lambda: None)

    cfg = auto_calibrate_script.AutoCalibrateConfig(robot=_FakeRobotConfig(), full_reset=True, arm="right")
    auto_calibrate_fn = getattr(auto_calibrate_script.auto_calibrate, "__wrapped__", auto_calibrate_script.auto_calibrate)

    auto_calibrate_fn(cfg)

    assert device.calls == [{"full_reset": True, "arm": "right"}]


def test_sourccey_get_observation_reuses_last_good_z_on_read_failure() -> None:
    robot = Sourccey.__new__(Sourccey)
    robot.id = "sourccey"
    robot.left_arm = _ObservationArm({"shoulder_pan.pos": 1.0})
    robot.right_arm = _ObservationArm({"shoulder_pan.pos": -1.0})
    robot.dc_motors_controller = _ObservationBase()
    robot._wheel_normalized_to_body = lambda _wheel_vel: {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
    robot.cameras = {}
    robot.config = type("Config", (), {"cameras": {}})()
    robot.z_actuator = type("Z", (), {"is_connected": True, "use_z_actuator": True})()
    robot._last_known_z_pos = -37.5

    def _raise() -> float:
        raise RuntimeError("spi glitch")

    robot.z_actuator.read_position = _raise

    observation = robot.get_observation()

    assert observation["z.pos"] == pytest.approx(-37.5)


def test_sourccey_get_observation_updates_last_good_z_on_success() -> None:
    robot = Sourccey.__new__(Sourccey)
    robot.id = "sourccey"
    robot.left_arm = _ObservationArm({"shoulder_pan.pos": 1.0})
    robot.right_arm = _ObservationArm({"shoulder_pan.pos": -1.0})
    robot.dc_motors_controller = _ObservationBase()
    robot._wheel_normalized_to_body = lambda _wheel_vel: {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
    robot.cameras = {}
    robot.config = type("Config", (), {"cameras": {}})()
    robot.z_actuator = type("Z", (), {"is_connected": True, "use_z_actuator": True})()
    robot._last_known_z_pos = 100.0
    robot.z_actuator.read_position = lambda: -12.25

    observation = robot.get_observation()

    assert observation["z.pos"] == pytest.approx(-12.25)
    assert robot._last_known_z_pos == pytest.approx(-12.25)
