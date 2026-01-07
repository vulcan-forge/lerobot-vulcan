import time

from lerobot.motors.dc_pwm.dc_pwm import PWMDCMotorsController
from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import (
    sourccey_dc_motors,
    sourccey_dc_motors_config,
)
from lerobot.robots.sourccey.sourccey.sourccey_z_actuator.sourccey_z_actuator import ZSensor


def set_z_position_m100_100(
    target_pos_m100_100: int = 25,
    *,
    raw_min: int = 0,
    raw_max: int = 4096,
    timeout_s: float = 8.0,
    hz: float = 100.0,
    kp: float = 0.02,
    ki: float = 0.0,          # start at 0.0; increase slightly if it stalls under load
    deadband: float = 1.0,
    max_cmd: float = 0.6,
    i_limit: float = 0.5,
) -> float:
    """
    Closed-loop "servo" move: drive Z until it reaches target_pos_m100_100 (in [-100, 100]).
    Returns the final measured position.
    """
    target = float(target_pos_m100_100)

    dc = PWMDCMotorsController(
        config=sourccey_dc_motors_config(),
        motors=sourccey_dc_motors(),
    )
    sensor = ZSensor(adc_channel=1, vref=3.30, average_samples=50)
    sensor.set_calibration(raw_min=raw_min, raw_max=raw_max)

    i_term = 0.0
    period = 1.0 / float(hz)
    t_end = time.monotonic() + float(timeout_s)
    last_t = time.monotonic()

    dc.connect()
    sensor.connect()
    try:
        while True:
            now = time.monotonic()
            if now >= t_end:
                dc.set_velocity("linear_actuator", 0.0, normalize=True, instant=True)
                raise TimeoutError(f"Timed out driving Z to {target_pos_m100_100} (last pos={pos:.2f})")  # type: ignore

            dt = max(1e-3, now - last_t)
            last_t = now

            pos = float(sensor.read_position_m100_100())
            err = target - pos

            if abs(err) <= float(deadband):
                dc.set_velocity("linear_actuator", 0.0, normalize=True, instant=True)
                return pos

            # PI control
            i_term += err * dt
            i_term = max(-float(i_limit), min(float(i_limit), i_term))

            cmd = float(kp) * err + float(ki) * i_term
            cmd = max(-float(max_cmd), min(float(max_cmd), cmd))

            dc.set_velocity("linear_actuator", cmd, normalize=True, instant=True)
            time.sleep(period)
    finally:
        try:
            dc.set_velocity("linear_actuator", 0.0, normalize=True, instant=True)
        except Exception:
            pass
        try:
            sensor.disconnect()
        except Exception:
            pass
        try:
            dc.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    final_pos = set_z_position_m100_100(25)
    print(f"Reached z≈{final_pos:.2f}")
