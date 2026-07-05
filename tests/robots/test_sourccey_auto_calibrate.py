#!/usr/bin/env python

import json

import pytest

import lerobot.robots.sourccey.sourccey.sourccey.sourccey as sourccey_module
import lerobot.robots.sourccey.sourccey.sourccey_z_actuator.sourccey_z_actuator as z_actuator_module
from lerobot.robots.sourccey.sourccey.sourccey_follower.sourccey_follower_calibrator import (
    SourcceyFollowerCalibrator,
)
from lerobot.robots.sourccey.sourccey.sourccey_z_actuator.sourccey_z_calibrator import (
    SourcceyZCalibrator,
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
    def __init__(self):
        self.calls: list[bool] = []

    def auto_calibrate(self, *, full_reset: bool = False) -> None:
        self.calls.append(full_reset)


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
    def set_velocity(self, motor, velocity, normalize=True, instant=True) -> None:
        return None


class _FollowerCalibrationBus:
    def __init__(self) -> None:
        self.motors = {"shoulder_pan": type("Motor", (), {"id": 1})()}
        self._positions = {"shoulder_pan": 1000}
        self._current_reads: dict[str, list[object]] = {"shoulder_pan": []}
        self._position_reads: dict[str, list[object]] = {"shoulder_pan": []}
        self._write_failures: dict[str, list[object]] = {"shoulder_pan": []}

    def disable_torque(self) -> None:
        return None

    def enable_torque(self) -> None:
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
    def __init__(self, *, invert: bool = True) -> None:
        self.sensor = ZSensor(invert=invert)
        self.invert = invert
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


def test_sourccey_auto_calibrate_raises_when_arm_thread_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sourccey_module.time, "sleep", lambda _seconds: None)

    robot = Sourccey.__new__(Sourccey)
    robot.left_arm = _DummyArm(RuntimeError("left arm failed"))
    robot.right_arm = _DummyArm()
    robot.z_actuator = _DummyZActuator()
    robot._z_hardware_available = True

    with pytest.raises(RuntimeError, match="left arm"):
        robot.auto_calibrate(full_reset=True)

    assert robot.z_actuator.calibrator.calls == [True]
    assert robot.right_arm.calls == []


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
    ("raw_top", "raw_bottom", "expected_invert"),
    [
        (120, 900, True),
        (900, 120, False),
    ],
)
def test_sourccey_z_full_calibration_guarantees_bottom_and_top_mapping(
    monkeypatch: pytest.MonkeyPatch,
    raw_top: int,
    raw_bottom: int,
    expected_invert: bool,
) -> None:
    monkeypatch.setattr(sourccey_module.time, "sleep", lambda _seconds: None)

    actuator = _CalibrationTestActuator(invert=not expected_invert)
    calibrator = SourcceyZCalibrator(actuator)
    measured_raws = iter([raw_top, raw_bottom, raw_top])

    monkeypatch.setattr(calibrator, "_drive", lambda _cmd: None)
    monkeypatch.setattr(calibrator, "_wait_until_stable", lambda _cmd: next(measured_raws))

    result = calibrator.auto_calibrate(full_reset=True)

    assert result is not None
    assert result.invert is expected_invert
    assert actuator.saved is True
    assert actuator.move_targets == []
    assert actuator.sensor.invert is expected_invert
    assert actuator.invert is expected_invert
    assert actuator.sensor.raw_to_pos_m100_100(raw_bottom) == pytest.approx(-100.0)
    assert actuator.sensor.raw_to_pos_m100_100(raw_top) == pytest.approx(100.0)


def test_sourccey_z_full_calibration_raises_if_return_to_top_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sourccey_module.time, "sleep", lambda _seconds: None)

    actuator = _CalibrationTestActuator(invert=True)
    calibrator = SourcceyZCalibrator(actuator)
    measured_raws = iter([120, 900])

    monkeypatch.setattr(calibrator, "_drive", lambda _cmd: None)

    def _wait(_cmd: float) -> int:
        try:
            return next(measured_raws)
        except StopIteration as exc:
            raise TimeoutError("top return timed out") from exc

    monkeypatch.setattr(calibrator, "_wait_until_stable", _wait)

    with pytest.raises(RuntimeError, match="failed to return the actuator to the top endpoint"):
        calibrator.auto_calibrate(full_reset=True)

    assert actuator.saved is True


def test_sourccey_follower_calibration_current_read_recovers_after_transient_failures() -> None:
    bus = _FollowerCalibrationBus()
    bus._current_reads["shoulder_pan"] = [
        RuntimeError("temporary read failure"),
        RuntimeError("temporary read failure"),
        42,
    ]
    calibrator = SourcceyFollowerCalibrator(_FollowerCalibrationRobot(bus))

    current, limit_reached = calibrator._read_calibration_current("shoulder_pan", max_retries=3, base_delay=0.0)

    assert current == 42
    assert limit_reached is False


def test_sourccey_follower_calibration_current_read_returns_none_when_recovery_fails() -> None:
    bus = _FollowerCalibrationBus()
    bus._current_reads["shoulder_pan"] = [
        RuntimeError("temporary read failure"),
        RuntimeError("temporary read failure"),
        RuntimeError("temporary read failure"),
        RuntimeError("temporary read failure"),
    ]
    calibrator = SourcceyFollowerCalibrator(_FollowerCalibrationRobot(bus))

    current, limit_reached = calibrator._read_calibration_current("shoulder_pan", max_retries=3, base_delay=0.0)

    assert current is None
    assert limit_reached is False


def test_sourccey_follower_calibration_slow_move_recovers_from_transient_write_failure() -> None:
    bus = _FollowerCalibrationBus()
    bus._write_failures["shoulder_pan"] = [RuntimeError("temporary write failure")]
    calibrator = SourcceyFollowerCalibrator(_FollowerCalibrationRobot(bus))

    moved = calibrator._move_calibration_slow(
        "shoulder_pan",
        1010,
        duration=0.2,
        steps_per_second=5.0,
        max_retries=1,
    )

    assert moved is True
    assert bus._positions["shoulder_pan"] == 1010


def test_sourccey_get_observation_reuses_last_good_z_on_read_failure() -> None:
    robot = Sourccey.__new__(Sourccey)
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
