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
    sourccey_model_path = current_file.parent.parent.parent / "src" / "lerobot" / "robots" / "sourccey" / "sourccey_v2beta" / "model"
    
    if sourccey_model_path.exists():
        urdf_path = str(sourccey_model_path / "Arm.urdf")
        # The URDF references STL files inside the `assets` folder
        mesh_path = str(sourccey_model_path / "meshes")
        print(f"Using URDF: {urdf_path}")
        print(f"Using meshes: {mesh_path}")
    else:
        print(f"ERROR: Could not find Sourccey V2 Beta model directory at {sourccey_model_path}")
        print("Make sure the Sourccey V2 Beta model files are available in src/lerobot/robots/sourccey/sourccey_v2beta/model/")
        return
    
    # Find existing calibration or use default ID
    existing_id = find_existing_calibration_id("sourccey_v2beta_follower")
    
    if existing_id:
        robot_id = existing_id
        print(f"Found existing calibration for ID: {robot_id}")
    else:
        robot_id = "sourccey_v2beta_follower_main"
        print(f"No existing calibration found. Using default ID: {robot_id}")
        print("Note: Robot will need to be calibrated on first connection.")
    
    # Configuration for the Sourccey V2 Beta follower robot
    robot_config = SourcceyV2BetaFollowerConfig(
        id=robot_id,
        port="COM11",  # Adjust based on your setup - could be /dev/ttyUSB1, COM3, etc.
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
        initial_wxyz=(0, 0, 1, 0),  # wxyz quaternion
        # rest_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # radians - conservative middle position
        enable_visualization=True,
        viser_port=8080,
        # Sourccey V2 Beta gripper configuration - matches SourcceyV2BetaFollowerConfig.max_gripper_pos = 50
        gripper_min_pos=0.0,    # Gripper closed (0% on phone slider)
        gripper_max_pos=50.0,   # Gripper open (100% on phone slider) - matches Sourccey V2 Beta max
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
                
                # Get action from phone (pass observation for current robot state)
                action = phone_teleop.get_action(observation)
                
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