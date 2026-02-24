#!/usr/bin/env python3
"""
Drive the Sourccey linear actuator upward for a fixed duration (open-loop).

This ignores any potentiometer/ADC feedback and simply applies motor power
for a specified number of seconds.
"""

import argparse
import time

from lerobot.motors.dc_pwm.dc_pwm import PWMDCMotorsController
from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import (
    sourccey_dc_motors,
    sourccey_dc_motors_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Raise linear actuator open-loop.")
    parser.add_argument(
        "--seconds",
        type=float,
        default=3.0,
        help="How long to drive the actuator (seconds).",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=1.0,
        help="Normalized motor power (0.0..1.0).",
    )
    parser.add_argument(
        "--direction",
        choices=["up", "down"],
        default="up",
        help="Motor direction. If the actuator moves the wrong way, use 'down'.",
    )
    args = parser.parse_args()

    duration_s = max(0.0, float(args.seconds))
    power = max(0.0, min(1.0, float(args.power)))
    direction = 1.0 if args.direction == "up" else -1.0

    controller = PWMDCMotorsController(
        config=sourccey_dc_motors_config(),
        motors=sourccey_dc_motors(),
        protocol="pwm",
    )

    controller.connect()
    try:
        controller.set_velocity("linear_actuator", direction * power, normalize=True, instant=True)
        time.sleep(duration_s)
    finally:
        controller.set_velocity("linear_actuator", 0.0, normalize=True, instant=True)
        controller.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
