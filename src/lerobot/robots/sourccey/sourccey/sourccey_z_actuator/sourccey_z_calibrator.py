import time
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


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

    - Drive DOWN at constant command until sensor raw ADC counts stop changing for `stable_s`
    - Drive UP at constant command until sensor raw ADC counts stop changing for `stable_s`

    Then write calibration to the ZSensor.

    Notes:
    - This assumes your motor driver / mechanics can safely hit end stops (current limiting!).
    - 'Stable' is defined as abs(raw - last_raw) <= stable_eps_raw continuously for stable_s.
    """

    def __init__(
        self,
        actuator,  # SourcceyZActuator
        *,
        stable_s: float = 2.0,
        sample_hz: float = 30.0,
        stable_eps_pos: float = 1.0,
        stable_eps_raw: int = 2,
        max_phase_s: float = 30.0,
        down_cmd: float = -1.0,
        up_cmd: float = 1.0,
    ) -> None:
        self.actuator = actuator
        self.stable_s = float(stable_s)
        self.sample_hz = float(sample_hz)
        # Backward-compat field kept for callers passing this argument.
        # Stall detection now uses native raw units via `stable_eps_raw`.
        self.stable_eps_pos = float(stable_eps_pos)
        self.stable_eps_raw = int(stable_eps_raw)
        self.max_phase_s = float(max_phase_s)
        self.down_cmd = float(down_cmd)
        self.up_cmd = float(up_cmd)

    def _drive(self, cmd: float) -> None:
        requested_cmd = float(cmd)
        cmd = -cmd if self.actuator.invert else cmd
        if self.actuator.driver is None:
            raise RuntimeError("SourcceyZActuator has no driver; cannot drive motor.")
        logger.info(
            "Z calibrator drive command: motor=%s requested_cmd=%s applied_cmd=%s invert=%s driver=%s",
            getattr(self.actuator, "motor", None),
            requested_cmd,
            cmd,
            getattr(self.actuator, "invert", None),
            type(self.actuator.driver).__name__,
        )
        # Important: cmd sign here is MOTOR sign. If this moves opposite of expected, swap down_cmd/up_cmd.
        self.actuator.driver.set_velocity(self.actuator.motor, float(cmd), normalize=True, instant=True)

    def _read_pos(self) -> float:
        return float(self.actuator.sensor.read_position_m100_100())

    def _read_raw(self) -> int:
        """Read native MCP3008 units (0..1023)."""
        return int(self.actuator.sensor.read_raw().raw)

    def _wait_until_stable(self, cmd: float) -> int:
        period = 1.0 / max(1.0, self.sample_hz)
        t_deadline = time.monotonic() + self.max_phase_s
        logger.info(
            "Z calibrator wait-until-stable start: motor=%s cmd=%s sample_hz=%s stable_s=%s stable_eps_raw=%s max_phase_s=%s",
            getattr(self.actuator, "motor", None),
            cmd,
            self.sample_hz,
            self.stable_s,
            self.stable_eps_raw,
            self.max_phase_s,
        )

        # Detect stall in raw ADC counts so calibration sensitivity is independent
        # of any prior saved mapping.
        last_raw = None
        stable_start = None
        samples = 0

        while True:
            now = time.monotonic()
            if now >= t_deadline:
                logger.error(
                    "Z calibrator wait-until-stable timeout: motor=%s cmd=%s last_raw=%s stable_start=%s samples=%s",
                    getattr(self.actuator, "motor", None),
                    cmd,
                    last_raw,
                    stable_start,
                    samples,
                )
                raise TimeoutError("Z calibrator timed out waiting for stability (end stop not detected).")

            # KEEP MOTOR ALIVE (important for watchdog-style drivers)
            self._drive(cmd)

            raw = self._read_raw()
            samples += 1
            if samples == 1 or samples % 10 == 0:
                logger.info(
                    "Z calibrator sample: motor=%s cmd=%s sample=%s raw=%s last_raw=%s stable_window_active=%s",
                    getattr(self.actuator, "motor", None),
                    cmd,
                    samples,
                    raw,
                    last_raw,
                    stable_start is not None,
                )

            if last_raw is None:
                last_raw = raw
                stable_start = None
            else:
                if abs(raw - last_raw) <= self.stable_eps_raw:
                    if stable_start is None:
                        stable_start = now
                        logger.info(
                            "Z calibrator stability window started: motor=%s cmd=%s raw=%s sample=%s",
                            getattr(self.actuator, "motor", None),
                            cmd,
                            raw,
                            samples,
                        )
                    elif (now - stable_start) >= self.stable_s:
                        logger.info(
                            "Z calibrator stable endpoint detected: motor=%s cmd=%s raw=%s samples=%s duration_s=%.3f",
                            getattr(self.actuator, "motor", None),
                            cmd,
                            raw,
                            samples,
                            now - stable_start,
                        )
                        return raw

                else:
                    if stable_start is not None:
                        logger.info(
                            "Z calibrator stability window reset: motor=%s cmd=%s raw=%s previous_raw=%s delta=%s",
                            getattr(self.actuator, "motor", None),
                            cmd,
                            raw,
                            last_raw,
                            abs(raw - last_raw),
                        )
                    stable_start = None
                
                last_raw = raw

            time.sleep(period)

    def _wait_for_seconds(self, cmd: float, seconds: float) -> int:
        """
        Drive at `cmd` for a fixed time (no end-stop detection).
        Returns the last raw value (0..1023) sampled during the window.

        Useful for "soft" calibration where you *don't* want to hit the mechanical edge.
        """
        period = 1.0 / max(1.0, self.sample_hz)
        t_end = time.monotonic() + float(seconds)

        last_raw = self._read_raw()
        while time.monotonic() < t_end:
            # Keep refreshing the motor command while we wait
            self._drive(cmd)
            last_raw = self._read_raw()
            time.sleep(period)

        return int(last_raw)

    def default_calibrate(self) -> ZCalibrationResult:
        """
        Soft calibration path.

        This path intentionally does not move the Z actuator. It reuses the
        currently loaded sensor calibration values and persists them, which is
        the expected behavior for non-`full_reset` robot auto-calibration.
        """
        try:
            if not self.actuator.is_connected:
                logger.warning("Z default calibration aborted: actuator is not connected")
                return None
        except Exception as e:
            logger.exception("Z default calibration failed while checking actuator connection: %s", e)
            print(f"Error: actuator is not connected: {e}")
            return None

        try:
            self.actuator.stop_position_controller()
        except Exception:
            pass

        logger.info(
            "Z default calibration reusing existing values: raw_min=%s raw_max=%s invert=%s",
            self.actuator.sensor.calibration_min,
            self.actuator.sensor.calibration_max,
            self.actuator.sensor.invert,
        )

        raw_min = int(self.actuator.sensor.calibration_min)
        raw_max = int(self.actuator.sensor.calibration_max)
        invert = bool(self.actuator.sensor.invert)

        self.actuator.sensor.set_calibration(raw_min=raw_min, raw_max=raw_max, invert=invert)
        self.actuator._save_calibration()
        logger.info(
            "Z default calibration completed without movement: raw_min=%s raw_max=%s invert=%s",
            raw_min,
            raw_max,
            invert,
        )

        return ZCalibrationResult(
            raw_bottom=int(raw_min),
            raw_top=int(raw_max),
            raw_min=raw_min,
            raw_max=raw_max,
            invert=invert,
        )

    def auto_calibrate(self, full_reset: bool = False) -> ZCalibrationResult:
        """
        Returns calibration and also writes it to ZSensor.
        """
        logger.info(
            "Z auto_calibrate called: full_reset=%s motor=%s sensor_connected=%s calibration=(min=%s max=%s invert=%s)",
            full_reset,
            getattr(self.actuator, "motor", None),
            getattr(self.actuator, "is_connected", None),
            getattr(self.actuator.sensor, "calibration_min", None),
            getattr(self.actuator.sensor, "calibration_max", None),
            getattr(self.actuator.sensor, "invert", None),
        )
        if not full_reset:
            logger.info("Z auto_calibrate taking default_calibrate path because full_reset is false")
            return self.default_calibrate()

        try:
            if (not self.actuator.is_connected):
                logger.warning("Z full calibration aborted: actuator is not connected")
                return None
        except Exception as e:
            logger.exception("Z full calibration failed while checking actuator connection: %s", e)
            print(f"Error: actuator is not connected: {e}")
            return None

        # Ensure any background position controller isn't fighting direct motor commands.
        try:
            self.actuator.stop_position_controller()
        except Exception:
            pass

        logger.info(
            "Z full calibration starting movement-based limit detection: motor=%s up_cmd=%s down_cmd=%s invert=%s",
            getattr(self.actuator, "motor", None),
            self.up_cmd,
            self.down_cmd,
            self.actuator.invert,
        )

        # If bottom reads higher than top, invert so that bottom maps to -100 and top maps to +100.
        invert = self.actuator.invert

        # Phase 1: UP -> top
        logger.info("Z full calibration phase 1 start: driving toward top endpoint")
        self._drive(self.up_cmd)
        raw_top = self._wait_until_stable(self.up_cmd)
        self.actuator.stop()
        time.sleep(0.25)
        logger.info("Z full calibration phase 1 complete: raw_top=%s", raw_top)

        # Phase 2: DOWN -> bottom
        # The linear actuator can get stuck at the bottom without a hardware block,
        # So we wait for 5 seconds until we have a hardware stop
        logger.info("Z full calibration phase 2 start: driving toward bottom endpoint")
        self._drive(self.down_cmd)
        # raw_bottom = self._wait_for_seconds(self.down_cmd, 5.0)
        raw_bottom = self._wait_until_stable(self.down_cmd)
        self.actuator.stop()
        time.sleep(0.25)
        logger.info("Z full calibration phase 2 complete: raw_bottom=%s", raw_bottom)

        print(f"raw_bottom: {raw_bottom}")
        print(f"raw_top: {raw_top}")

        # Decide mapping
        raw_min = int(min(raw_bottom, raw_top))
        raw_max = int(max(raw_bottom, raw_top))
        logger.info(
            "Z full calibration computed bounds: raw_bottom=%s raw_top=%s raw_min=%s raw_max=%s invert=%s",
            raw_bottom,
            raw_top,
            raw_min,
            raw_max,
            invert,
        )

        self.actuator.sensor.set_calibration(raw_min=raw_min, raw_max=raw_max, invert=invert)
        self.actuator._save_calibration()
        logger.info(
            "Z full calibration saved calibration file: path=%s raw_min=%s raw_max=%s invert=%s",
            getattr(self.actuator, "calibration_fpath", None),
            raw_min,
            raw_max,
            invert,
        )

        # Repositioning after calibration is best-effort only. A failure here
        # should not invalidate the newly detected calibration bounds.
        try:
            logger.info("Z full calibration reposition start: moving to 100.0")
            self.actuator.move_to_position_blocking(100.0)
            logger.info("Z full calibration reposition complete: moved to 100.0")
        except TimeoutError as exc:
            logger.warning(
                "Z calibration saved, but reposition to 100.0 timed out: %s",
                exc,
            )

        return ZCalibrationResult(
            raw_bottom=int(raw_bottom),
            raw_top=int(raw_top),
            raw_min=int(raw_min),
            raw_max=int(raw_max),
            invert=invert,
        )
