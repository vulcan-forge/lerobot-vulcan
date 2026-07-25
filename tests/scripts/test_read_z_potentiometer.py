import json

import pytest

from lerobot.scripts.sourccey.z.read_z_potentiometer import classify_reading, load_calibration


def test_load_calibration(tmp_path) -> None:
    calibration_path = tmp_path / "z.json"
    calibration_path.write_text(
        json.dumps({"z_actuator": {"raw_min": 123, "raw_max": 876, "invert": False}}),
        encoding="utf-8",
    )

    assert load_calibration(calibration_path) == (123, 876, False)


def test_load_calibration_rejects_zero_range(tmp_path) -> None:
    calibration_path = tmp_path / "z.json"
    calibration_path.write_text(
        json.dumps({"z_actuator": {"raw_min": 500, "raw_max": 500, "invert": True}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw_min equals raw_max"):
        load_calibration(calibration_path)


@pytest.mark.parametrize(
    ("raw", "position", "previous", "expected"),
    [
        (512, 0.0, -1.0, "OK"),
        (0, -100.0, 100.0, "ADC_RAIL,LARGE_JUMP"),
        (1023, 100.0, None, "ADC_RAIL"),
        (500, -80.0, 80.0, "LARGE_JUMP"),
    ],
)
def test_classify_reading(raw: int, position: float, previous: float | None, expected: str) -> None:
    assert classify_reading(raw, position, previous) == expected
