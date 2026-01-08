import time
from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyConfig
from lerobot.robots.sourccey.sourccey.sourccey.sourccey import Sourccey


def servo_z_interactive(
    *,
    hz: float = 5.0,           # background controller rate
    instant: bool = True,
    status_hz: float = 2.0,    # how often we print status while running
) -> None:
    robot = Sourccey(SourcceyConfig())
    robot.connect(calibrate=False)

    target = 0.0
    last_status_t = 0.0

    print("Servo mode. Type target z in [-100, 100]. Type 'q' to quit.")
    print("Note: move_to_position() is non-blocking; it runs a background controller thread.")

    try:
        while True:
            # periodic status
            now = time.monotonic()
            if status_hz > 0 and (now - last_status_t) >= (1.0 / status_hz):
                last_status_t = now
                r = robot.z_sensor.read_raw()
                pos = float(robot.z_actuator.read_position())
                print(
                    {
                        "raw": r.raw,
                        "voltage": round(r.voltage, 4),
                        "pos_m100_100": round(pos, 2),
                        "target_m100_100": round(float(target), 2),
                    }
                )

            # non-blocking input: keep the servo loop running while waiting for a command
            try:
                s = input("target_z> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if s in ("q", "quit", "exit"):
                break

            try:
                target = float(s)
            except ValueError:
                print("Please enter a number or 'q' to quit.")
                continue

            # non-blocking: updates target, background thread moves toward it
            robot.z_actuator.move_to_position(target, hz=hz, instant=instant)

    finally:
        # stop background controller and motor output cleanly
        try:
            robot.z_actuator.stop_position_controller()
        finally:
            robot.disconnect()


if __name__ == "__main__":
    servo_z_interactive(hz=5.0, status_hz=2.0)
