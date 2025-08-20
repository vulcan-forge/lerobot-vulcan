#!/usr/bin/env python

"""
Phone teleoperation for Sourccey V3 Beta follower arm (Sourccey-specific teleop).

This script uses PhoneTeleoperatorSourccey, which emits left/right-prefixed
action keys expected by the Sourccey V3 Beta bimanual architecture. Here we
control only the left follower arm locally over serial.

Requirements:
- pip install pyroki viser yourdfpy
- Ensure the phone gRPC server is accessible (started by the teleop)
- URDF/meshes available (uses Sourccey V2 Beta model for IK compatibility)
"""

import time
from pathlib import Path

from lerobot.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from lerobot.robots.sourccey.sourccey_v3beta.sourccey_v3beta_follower import (
    SourcceyV3BetaFollower,
    SourcceyV3BetaFollowerConfig,
)
from lerobot.teleoperators.phone_teleoperator import (
    PhoneTeleoperatorSourccey,
    PhoneTeleoperatorSourcceyConfig,
)


def find_existing_calibration_id(robot_name: str) -> str | None:
    """Find existing calibration file ID for the robot."""
    calib_dir = HF_LEROBOT_CALIBRATION / ROBOTS / robot_name

    if not calib_dir.exists():
        return None

    calib_files = list(calib_dir.glob("*.json"))
    if not calib_files:
        return None

    return calib_files[0].stem


def find_sourccey_model_paths() -> tuple[str, str]:
    """Resolve URDF and mesh paths for the Sourccey model used by IK."""
    current_file = Path(__file__)
    model_dir = (
        current_file.parent.parent.parent
        / "src"
        / "lerobot"
        / "robots"
        / "sourccey"
        / "sourccey_v2beta"
        / "model"
    )

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Could not find Sourccey V2 Beta model directory at {model_dir}. "
            "Ensure files exist in src/lerobot/robots/sourccey/sourccey_v2beta/model/"
        )

    urdf_path = str(model_dir / "Arm.urdf")
    mesh_path = str(model_dir / "meshes")
    print(f"Using URDF: {urdf_path}")
    print(f"Using meshes: {mesh_path}")
    return urdf_path, mesh_path


def main():
    # Resolve model assets for IK
    urdf_path, mesh_path = find_sourccey_model_paths()

    # Calibration id (persisted in HF cache directory)
    existing_id = find_existing_calibration_id("sourccey_v3beta_follower")
    robot_id = existing_id or "sourccey_v3beta_follower_main"
    if existing_id:
        print(f"Found existing calibration for ID: {robot_id}")
    else:
        print(f"No existing calibration found. Using default ID: {robot_id}")
        print("Note: Robot will need to be calibrated on first connection.")

    # Configure follower (left arm by default)
    robot_config = SourcceyV3BetaFollowerConfig(
        id=robot_id,
        port="COM11",  # Adjust per setup (e.g., /dev/ttyUSB0)
        orientation="left",
        use_degrees=True,
        max_relative_target=30.0,
    )

    # Configure Sourccey-specific phone teleoperator (emits left_ keys)
    phone_config = PhoneTeleoperatorSourcceyConfig(
        id="phone_teleop_sourccey_v3beta",
        urdf_path=urdf_path,
        mesh_path=mesh_path,
        target_link_name="Feetech-Servo-Motor-v1-5",
        sensitivity_normal=0.5,
        sensitivity_precision=0.2,
        rotation_sensitivity=1.0,
        initial_position=(0.0, -0.17, 0.237),
        initial_wxyz=(0, 0, 1, 0),
        # Conservative middle pose (radians) for Sourccey arm IK
        rest_pose=(-0.843128, 1.552000, 0.736491, 0.591494, 0.020714, 0.009441),
        enable_visualization=True,
        viser_port=8080,
        gripper_min_pos=0.0,
        gripper_max_pos=50.0,
    )

    robot = SourcceyV3BetaFollower(robot_config)
    phone = PhoneTeleoperatorSourccey(phone_config)

    try:
        print("Connecting to robot...")
        robot.connect()

        print("Connecting to phone teleoperator...")
        phone.connect()

        print("Phone teleoperation ready (Sourccey V3 Beta follower, left arm)!")
        print("- Start the phone app and connect to the gRPC server")
        print("- Use your phone to control the robot")
        print("- After starting teleop, motor positions will be read and displayed after 5 seconds")
        print("- The motor positions will be shown in rest_pose format for easy copying to config")
        print("- Press Ctrl+C to exit")

        while True:
            try:
                observation = robot.get_observation()
                # PhoneTeleoperatorSourccey returns left_* keys; strip prefix for single-arm follower
                arm_action_prefixed = phone.get_action(observation)
                action = {k.replace("left_", ""): v for k, v in arm_action_prefixed.items()}
                robot.send_action(action)
                time.sleep(max(0, 1 / 30))  # ~30 Hz
            except KeyboardInterrupt:
                print("\nStopping teleoperation...")
                break
            except Exception as e:
                print(f"Error in control loop: {e}")
                time.sleep(0.1)
    finally:
        print("Disconnecting devices...")
        try:
            phone.disconnect()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        print("Done!")


if __name__ == "__main__":
    main()


