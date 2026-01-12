import sys
import time

from lerobot.motors.dc_pwm.dc_pwm import PWMDCMotorsController
from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyConfig


def _get_key_nonblocking():
    """
    Returns a single lowercase character if available, else None.
    Works on Windows (msvcrt) and POSIX (select/termios raw mode).
    """
    if sys.platform.startswith("win"):
        import msvcrt

        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            return ch.lower()
        return None

    # POSIX: stdin must be in raw mode; see _raw_terminal context manager below.
    import select

    r, _, _ = select.select([sys.stdin], [], [], 0)
    if not r:
        return None
    ch = sys.stdin.read(1)
    return ch.lower()


class _raw_terminal:
    """POSIX-only: put terminal into raw mode so we can read single chars."""
    def __enter__(self):
        if sys.platform.startswith("win"):
            return self

        import termios
        import tty

        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setraw(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if sys.platform.startswith("win"):
            return False

        import termios

        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        return False


def jog_linear_actuator_raw(
    *,
    speed: float = 1.0,          # 0..1 (normalized duty). Start low.
    loop_hz: float = 50.0,
    hold_timeout_s: float = 0.25, # if no key seen recently, stop motor
) -> None:
    """
    Raw jog control (no sensor, no ZActuator, no position controller thread).

    Controls:
    - hold / tap 'e' to drive one direction
    - hold / tap 'q' to drive the other direction
    - 's' to stop
    - 'x' to exit
    """
    speed = float(max(0.0, min(1.0, speed)))
    period = 1.0 / max(1.0, float(loop_hz))

    cfg = SourcceyConfig()
    dc = PWMDCMotorsController(motors=cfg.dc_motors, config=cfg.dc_motors_config)

    last_key_t = 0.0
    cmd = 0.0

    print("Raw linear actuator jog (NO position tracker).")
    print("Keys: 'e' / 'q' move, 's' stop, 'x' exit.")
    print(f"Using motor='linear_actuator' at speed={speed:.2f} (normalized).")

    try:
        dc.connect()

        # Ensure we start stopped
        dc.set_velocity("linear_actuator", 0.0, normalize=True, instant=True)

        with _raw_terminal():
            while True:
                t0 = time.monotonic()

                k = _get_key_nonblocking()
                if k is not None:
                    last_key_t = t0

                    if k == "x":
                        break
                    if k == "s":
                        cmd = 0.0
                    elif k == "e":
                        cmd = +speed
                    elif k == "q":
                        cmd = -speed

                # "Hold" behavior: if you stop pressing keys, stop the motor.
                if (t0 - last_key_t) > float(hold_timeout_s):
                    cmd = 0.0

                dc.set_velocity("linear_actuator", cmd, normalize=True, instant=True)

                dt = time.monotonic() - t0
                time.sleep(max(0.0, period - dt))

    finally:
        try:
            dc.set_velocity("linear_actuator", 0.0, normalize=True, instant=True)
        except Exception:
            pass
        try:
            dc.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    jog_linear_actuator_raw(speed=1.0)
