# Copyright 2026 Vulcan Robotics, Inc. All rights reserved.

from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import SourcceyConfig
from lerobot.robots.sourccey.sourccey.sourccey_follower.config_sourccey_follower import SourcceyFollowerConfig
from lerobot.robots.sourccey.sourccey.sourccey_follower.sourccey_follower import SourcceyFollower


def main() -> int:
    robot_cfg = SourcceyConfig(id="sourccey")
    right_cfg = SourcceyFollowerConfig(
        id=f"{robot_cfg.id}_right" if robot_cfg.id else None,
        calibration_dir=robot_cfg.calibration_dir,
        motor_models=robot_cfg.right_arm_motor_models,
        port=robot_cfg.right_arm_port,
        orientation="right",
        disable_torque_on_disconnect=robot_cfg.right_arm_disable_torque_on_disconnect,
        max_relative_target=robot_cfg.right_arm_max_relative_target,
        use_degrees=robot_cfg.right_arm_use_degrees,
        cameras={},
    )

    arm = SourcceyFollower(right_cfg)

    try:
        arm.bus.connect()
        raw_positions = arm.bus.sync_read("Present_Position", normalize=False)
        calibration = arm.bus.calibration

        out = []
        print("RIGHT ARM BOUNDS CHECK")
        for motor in arm.bus.motors:
            raw = int(round(float(raw_positions[motor])))
            cal = calibration.get(motor)
            if cal is None:
                print(f"- {motor}: raw={raw} range=[missing calibration] status=OUT_OF_BOUNDS")
                out.append(motor)
                continue

            ok = cal.range_min <= raw <= cal.range_max
            status = "OK" if ok else "OUT_OF_BOUNDS"
            print(f"- {motor}: raw={raw} range=[{cal.range_min}, {cal.range_max}] status={status}")
            if not ok:
                out.append(motor)

        if out:
            print(f"RESULT: FAIL ({len(out)} out of bounds): {', '.join(out)}")
            return 2

        print("RESULT: PASS (all right-arm motors in bounds)")
        return 0
    finally:
        try:
            arm.bus.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
