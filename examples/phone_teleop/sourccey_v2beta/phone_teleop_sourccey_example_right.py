#!/usr/bin/env python

"""
Example usage of PhoneTeleoperator with Sourccey V2 Beta robot.

This script demonstrates how to integrate phone teleoperation into the new lerobot architecture.
It uses the PhoneTeleoperator to receive commands from a mobile phone via gRPC and control a Sourccey V2 Beta robot.

Requirements:
- Install additional dependencies: pip install pyroki viser yourdfpy
- Ensure the daxie package is installed and the gRPC server is accessible
- Have the robot URDF and mesh files available
- Connect your Sourccey V2 Beta robot
"""

import time
from pathlib import Path

from lerobot.robots.sourccey.sourccey_v2beta import SourcceyV2BetaFollower, SourcceyV2BetaFollowerConfig
from lerobot.teleoperators.phone_teleoperator import PhoneTeleoperatorSourccey, PhoneTeleoperatorSourcceyConfig
from lerobot.constants import HF_LEROBOT_CALIBRATION, ROBOTS


def find_existing_calibration_id(robot_name: str) -> str | None:
    """Find existing calibration file ID for the robot."""
    calib_dir = HF_LEROBOT_CALIBRATION / ROBOTS / robot_name
    
    if not calib_dir.exists():
        return None
    
    # Look for .json files in the calibration directory
    calib_files = list(calib_dir.glob("*.json"))
    
    if not calib_files:
        return None
    
    # Return the ID (filename without .json extension) of the first calibration file found
    return calib_files[0].stem


def main():
    # Get URDF and mesh paths from lerobot package
    from pathlib import Path
    
    # Get the path to the Sourccey V2 Beta model directory
    current_file = Path(__file__)
    # Use the Sourccey V2 Beta model located in the model directory
    sourccey_model_path = current_file.parent.parent.parent.parent / "src" / "lerobot" / "robots" / "sourccey" / "sourccey_v3beta" / "model"
    
    if sourccey_model_path.exists():
        urdf_path = str(sourccey_model_path / "Arm.urdf")
        # The URDF references STL files inside the `assets` folder
        mesh_path = str(sourccey_model_path / "meshes")
        print(f"Using URDF: {urdf_path}")
        print(f"Using meshes: {mesh_path}")
    else:
        print(f"ERROR: Could not find Sourccey V3 Beta model directory at {sourccey_model_path}")
        print("Make sure the Sourccey V3 Beta model files are available in src/lerobot/robots/sourccey/sourccey_v3beta/model/")
        return
    
    # Find existing calibration or use default ID
    existing_id = find_existing_calibration_id("sourccey_v3beta_follower")
    
    if existing_id:
        robot_id = existing_id
        print(f"Found existing calibration for ID: {robot_id}")
    else:
        robot_id = "sourccey_v3beta_follower_main"
        print(f"No existing calibration found. Using default ID: {robot_id}")
        print("Note: Robot will need to be calibrated on first connection.")
    
    # Configuration for the Sourccey V2 Beta follower robot
    robot_config = SourcceyV2BetaFollowerConfig(
        id=robot_id,
        port="COM27",  # Adjust based on your setup - could be /dev/ttyUSB1, COM3, etc.
        use_degrees=True,
        max_relative_target=30.0,  # Safety limit in degrees
    )
    
    # Configuration for the phone teleoperator
    phone_config = PhoneTeleoperatorSourcceyConfig(
        id="phone_teleop_main",
        urdf_path=urdf_path,
        mesh_path=mesh_path,
        # In the URDF the end-effector link is named "gripper"
        target_link_name="Feetech-Servo-Motor-v1-5",
        sensitivity_normal=0.5,
        sensitivity_precision=0.2,
        rotation_sensitivity=1.0,
        initial_position=(0.0, -0.17, 0.237),
        initial_wxyz = (0.0, 0.5, 0.866025404, 0.0),  # ~30° right yaw
        # Set rest_pose (radians) to match latest measured left-arm actions
        # Teleop does NOT flip rest pose; values here are the desired joint angles (deg):
        # [-49.506903, 100.0, -97.716150, 5.381376, 0.854701, 99.603960]
        rest_pose=(
            0.640044,    # right_shoulder_pan  (36.671576 deg)
            -1.689644,   # right_shoulder_lift (-96.809571 deg)
            1.732986,    # right_elbow_flex    (99.292517 deg)
            -0.235352,   # right_wrist_flex    (-13.484163 deg)
            0.020884,    # right_wrist_roll    (1.196581 deg)
            0.004511,    # right_gripper       (0.258398 deg)
        ),
        joint_offsets_deg={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
        },
        enable_visualization=True,
        viser_port=8080,
        # Sourccey V2 Beta gripper configuration - matches SourcceyV2BetaFollowerConfig.max_gripper_pos = 50
        gripper_min_pos=0.0,    # Gripper closed (0% on phone slider)
        gripper_max_pos=50.0,   # Gripper open (100% on phone slider) - matches Sourccey V2 Beta max
        # Disable built-in mirroring; we'll do explicit right-arm mirroring below
        # mirror_enabled=False,
        # mirror_neutral_deg=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    
    # Initialize robot and teleoperator
    robot = SourcceyV2BetaFollower(robot_config)
    phone_teleop = PhoneTeleoperatorSourccey(phone_config)
    
    try:
        # Connect devices
        print("Connecting to robot...")
        robot.connect()
        
        print("Connecting to phone teleoperator...")
        phone_teleop.connect()
        
        print("Phone teleoperation ready!")
        print("- Start the phone app and connect to the gRPC server")
        print("- Use your phone to control the robot")
        print("- After starting teleop, motor positions will be read and displayed after 5 seconds")
        print("- The motor positions will be shown in rest_pose format for easy copying to config")
        print("- Press Ctrl+C to exit")
        
        # Main control loop
        while True:
            start_time = time.perf_counter()
            
            try:
                # Get current observation first
                observation = robot.get_observation()
                
                # Get action from phone (emits left_* keys)
                left_action = phone_teleop.get_action(observation)

                # Mirror around neutral and apply axis sign for a right-side mirrored arm
                # v' = r[i] * (2*neutral[i] - v)
                neutral = (0.0, 0.0, 0.0, 0.0, 0.0)
                r = (-1.0, +1.0, +1.0, +1.0, -1.0)  # [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]

                def mir(v, n, s):
                    return s * (2.0 * n - float(v))

                action = {
                    "shoulder_pan.pos":  mir(left_action.get("left_shoulder_pan.pos", 0.0),  neutral[0], r[0]),
                    "shoulder_lift.pos": mir(left_action.get("left_shoulder_lift.pos", 0.0), neutral[1], r[1]),
                    "elbow_flex.pos":    mir(left_action.get("left_elbow_flex.pos", 0.0),    neutral[2], r[2]),
                    "wrist_flex.pos":    mir(left_action.get("left_wrist_flex.pos", 0.0),    neutral[3], r[3]),
                    "wrist_roll.pos":    mir(left_action.get("left_wrist_roll.pos", 0.0),    neutral[4], r[4]),
                    # Gripper typically not mirrored
                    "gripper.pos":       float(left_action.get("left_gripper.pos", 0.0)),
                }


                # Send action to robot
                actual_action = robot.send_action(action)
                                
                # Control frequency (adjust as needed)
                time.sleep(max(0, 1/30))  # Target ~30 Hz
                
            except KeyboardInterrupt:
                print("\nStopping teleoperation...")
                break
            except Exception as e:
                print(f"Error in control loop: {e}")
                # Continue running but with a small delay
                time.sleep(0.1)
    
    finally:
        # Cleanup
        print("Disconnecting devices...")
        try:
            phone_teleop.disconnect()
        except:
            pass
        try:
            robot.disconnect()
        except:
            pass
        print("Done!")


if __name__ == "__main__":
    main() 