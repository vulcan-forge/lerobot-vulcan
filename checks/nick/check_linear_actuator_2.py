import time
import threading  # NEW
from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyConfig
from lerobot.robots.sourccey.sourccey.sourccey.sourccey import Sourccey


def set_z_position_m100_100(
    target: int = 25,
    *,
    timeout_s: float = 8.0,
    hz: float = 30.0,
    instant: bool = True,
    status_cb=None,
    status_hz: float | None = None,  # NEW: how often to call status_cb while moving
) -> float:
    robot = Sourccey(SourcceyConfig())
    robot.connect(calibrate=False)

    try:
        result: dict[str, object] = {"pos": None, "err": None}

        def _worker():
            try:
                result["pos"] = robot.z_actuator.move_to_position_blocking(
                    float(target),
                    timeout_s=float(timeout_s),
                    hz=float(hz),
                    instant=bool(instant),
                )
            except Exception as e:
                result["err"] = e

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        # Optional live status printing while move_to_position() runs
        if status_cb is not None:
            period = 1.0 / float(status_hz or hz or 30.0)
            while t.is_alive():
                r = robot.z_sensor.read_raw()
                pos = float(robot.z_actuator.read_position())
                status_cb(r, pos, float(target))
                time.sleep(period)

        t.join()

        if result["err"] is not None:
            raise result["err"]  # type: ignore[misc]

        return float(result["pos"])  # type: ignore[arg-type]
    finally:
        # extra safety on exit
        try:
            robot.z_actuator.stop()
        finally:
            robot.disconnect()


if __name__ == "__main__":
    def status_printer(r, pos, target):
        print(
            {
                "raw": r.raw,
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
