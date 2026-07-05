#!/usr/bin/env python

import json

import pytest

import lerobot.robots.sourccey.sourccey.sourccey.sourccey as sourccey_module
import lerobot.robots.sourccey.sourccey.sourccey_z_actuator.sourccey_z_actuator as z_actuator_module
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


class _DummyDriver:
    def set_velocity(self, motor, velocity, normalize=True, instant=True) -> None:
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
    assert robot.right_arm.calls == [{"reverse": True, "full_reset": True}]


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
    measured_raws = iter([raw_top, raw_bottom])

    monkeypatch.setattr(calibrator, "_drive", lambda _cmd: None)
    monkeypatch.setattr(calibrator, "_wait_until_stable", lambda _cmd: next(measured_raws))

    result = calibrator.auto_calibrate(full_reset=True)

    assert result is not None
    assert result.invert is expected_invert
    assert actuator.saved is True
    assert actuator.move_targets == [100.0]
    assert actuator.sensor.invert is expected_invert
    assert actuator.invert is expected_invert
    assert actuator.sensor.raw_to_pos_m100_100(raw_bottom) == pytest.approx(-100.0)
    assert actuator.sensor.raw_to_pos_m100_100(raw_top) == pytest.approx(100.0)
