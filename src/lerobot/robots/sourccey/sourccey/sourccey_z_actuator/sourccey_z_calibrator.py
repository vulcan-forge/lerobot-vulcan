import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ZCalibrationResult:
    raw_bottom: int
    raw_top: int
    raw_min: int
    raw_max: int
    invert: bool


class SourcceyZCalibrator:
    """
    Autocalibration by stall detection:

    - Drive DOWN at constant command until the sensor position stops changing for `stable_s`
    - Drive UP at constant command until the sensor position stops changing for `stable_s`

    Then write calibration to the ZSensor.

    Notes:
    - This assumes your motor driver / mechanics can safely hit end stops (current limiting!).
    - 'Stable' is defined as abs(pos - last_pos) <= stable_eps_pos continuously for stable_s.
    """

    def __init__(
        self,
        actuator,  # ZActuator
        *,
        stable_s: float = 0.1,
        sample_hz: float = 30.0,
        stable_eps_pos: float = 0.25,
        max_phase_s: float = 30.0,
        down_cmd: float = -1.0,
        up_cmd: float = 1.0,
    ) -> None:
        self.actuator = actuator
        self.stable_s = float(stable_s)
        self.sample_hz = float(sample_hz)
        self.stable_eps_pos = float(stable_eps_pos)
        self.max_phase_s = float(max_phase_s)
        self.down_cmd = float(down_cmd)
        self.up_cmd = float(up_cmd)

    def _drive(self, cmd: float) -> None:

        cmd = -cmd if self.actuator.invert else cmd
        if self.actuator.driver is None:
            raise RuntimeError("ZActuator has no driver; cannot drive motor.")
        # Important: cmd sign here is MOTOR sign. If this moves opposite of expected, swap down_cmd/up_cmd.
        self.actuator.driver.set_velocity(self.actuator.motor, float(cmd), normalize=True, instant=True)

    def _read_pos(self) -> float:
        return float(self.actuator.sensor.read_position_m100_100())

    def _read_raw(self) -> int:
        """Read native MCP3008 units (0..1023)."""
        return int(self.actuator.sensor.read_raw().raw)

    def _wait_until_stable(self) -> int:
        """
        Wait until position change stays within epsilon for stable_s.
        Returns the raw value (0..1023) at the moment stability is achieved.
        """
        period = 1.0 / max(1.0, self.sample_hz)
        t_deadline = time.monotonic() + self.max_phase_s

        last_pos: Optional[float] = None
        stable_start: Optional[float] = None

        while True:
            now = time.monotonic()
            if now >= t_deadline:
                raise TimeoutError("Z calibrator timed out waiting for stability (end stop not detected).")

            pos = self._read_pos()

            if last_pos is None:
                last_pos = pos
                stable_start = None
            else:
                if abs(pos - last_pos) <= self.stable_eps_pos:
                    if stable_start is None:
                        stable_start = now
                    elif (now - stable_start) >= self.stable_s:
                        return self._read_raw()
                else:
                    stable_start = None

                last_pos = pos

            time.sleep(period)

    def auto_calibrate(self) -> ZCalibrationResult:
        """
        Returns calibration and also writes it to ZSensor.
        """
        # Phase 1: DOWN -> bottom
        self._drive(self.down_cmd)
        raw_bottom = self._wait_until_stable()
        print(f"raw_bottom: {raw_bottom}")
        self.actuator.stop()
        time.sleep(0.25)

        # Phase 2: UP -> top
        self._drive(self.up_cmd)
        raw_top = self._wait_until_stable()
        print(f"raw_top: {raw_top}")
        self.actuator.stop()
        time.sleep(0.25)

        # Decide mapping
        raw_min = int(min(raw_bottom, raw_top))
        raw_max = int(max(raw_bottom, raw_top))
        # If bottom reads higher than top, invert so that bottom maps to -100 and top maps to +100.
        invert = bool(raw_bottom > raw_top)

        self.actuator.sensor.set_calibration(raw_min=raw_min, raw_max=raw_max, invert=invert)

        return ZCalibrationResult(
            raw_bottom=int(raw_bottom),
            raw_top=int(raw_top),
            raw_min=int(raw_min),
            raw_max=int(raw_max),
            invert=invert,
        )
