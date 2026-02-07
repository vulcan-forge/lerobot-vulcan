import argparse

from .config_sourccey import SourcceyConfig
from .sourccey import Sourccey


def main() -> None:
    parser = argparse.ArgumentParser(description="Toggle Sourccey arm torque.")
    parser.add_argument("--left", action="store_true", help="Only affect the left arm.")
    parser.add_argument("--right", action="store_true", help="Only affect the right arm.")
    parser.add_argument("--enable", action="store_true", help="Enable torque instead of disabling it.")
    parser.add_argument("--left-arm-port", default=None, help="Override left arm serial port.")
    parser.add_argument("--right-arm-port", default=None, help="Override right arm serial port.")
    args = parser.parse_args()

    use_left = args.left or not args.right
    use_right = args.right or not args.left

    config = SourcceyConfig(id="sourccey")
    if args.left_arm_port:
        config.left_arm_port = args.left_arm_port
    if args.right_arm_port:
        config.right_arm_port = args.right_arm_port

    robot = Sourccey(config)
    robot.connect()
    try:
        if args.enable:
            if use_left:
                robot.left_arm.bus.enable_torque()
            if use_right:
                robot.right_arm.bus.enable_torque()
            print("Enabled torque.")
        else:
            if use_left:
                robot.left_arm.bus.disable_torque()
            if use_right:
                robot.right_arm.bus.disable_torque()
            print("Disabled torque.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
