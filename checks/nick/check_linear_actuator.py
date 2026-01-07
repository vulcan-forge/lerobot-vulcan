import time
from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyConfig
from lerobot.robots.sourccey.sourccey.sourccey.sourccey import Sourccey


def set_z_position_m100_100(
    target: int = 25,
    *,
    timeout_s: float = 8.0,
    hz: float = 100.0,
    status_cb=None,
) -> float:
    robot = Sourccey(SourcceyConfig())
    robot.connect(calibrate=False)

    try:
        robot.z_actuator.write_position(float(target))  # or set_target_position_m100_100(...)

        period = 1.0 / float(hz)
        t_end = time.monotonic() + float(timeout_s)
        last_t = time.monotonic()

        while True:
            now = time.monotonic()
            if now >= t_end:
                robot.z_actuator.stop()
                raise TimeoutError("Timed out reaching Z target")

            dt = now - last_t
            last_t = now

            robot.z_actuator.update(dt)

            # Print / callback every loop
            if status_cb is not None:
                r = robot.z_sensor.read_raw()
                pos = float(robot.z_actuator.read_position())
                status_cb(r, pos, float(target))

            pos = float(robot.z_actuator.read_position())
            if abs(pos - float(target)) <= float(robot.z_actuator.deadband):
                robot.z_actuator.stop()
                return pos

            time.sleep(period)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    def status_printer(r, pos, target):
        print(
            {
                "raw_10bit": r.raw_10bit,
                "raw_scaled": r.raw,
                "voltage": round(r.voltage, 4),
                "pos_m100_100": round(float(pos), 2),
                "target_m100_100": round(float(target), 2),
            }
        )

    print("Type a target z position in [-100, 100]. Type 'q' to quit.")
    while True:
        s = input("target_z> ").strip().lower()
        if s in ("q", "quit", "exit"):
            break
        try:
            target = int(float(s))
        except ValueError:
            print("Please enter a number or 'q' to quit.")
            continue

        try:
            final_pos = set_z_position_m100_100(target, hz=30.0, status_cb=status_printer)
            print(f"Reached z≈{final_pos:.2f}")
        except Exception as e:
            print(f"Error: {e}")
