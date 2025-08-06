#!/usr/bin/env python

"""
Phone Teleoperation for Sourccey V3 Beta Robot (Right Arm Only)

This script demonstrates how to integrate phone teleoperation with the Sourccey V3 Beta robot,
controlling only the right arm. It works like the regular teleoperate.py script but uses
phone-based control instead of a physical leader arm.

Requirements:
- Install additional dependencies: pip install pyroki viser yourdfpy
- Ensure the phone teleop gRPC server is accessible
- Have the robot URDF and mesh files available (uses V2 Beta model for compatibility)
- Connect your Sourccey V3 Beta robot via network (client connection)

Usage:
1. Start the Sourccey V3 Beta host on the Raspberry Pi:
   uv run -m lerobot.robots.sourccey.sourccey_v3beta.sourccey_v3beta.sourccey_v3beta_host

2. Run this script on your computer:
   python examples/sourccey/sourccey_v3beta/phone_teleoperate.py

3. Connect your phone app to the gRPC server (port 8765) to start teleoperating
"""

import time
from pathlib import Path

from lerobot.robots.sourccey.sourccey_v3beta.sourccey_v3beta import SourcceyV3Beta, SourcceyV3BetaClientConfig, SourcceyV3BetaClient
from lerobot.teleoperators.phone_teleoperator import PhoneTeleoperatorSourccey, PhoneTeleoperatorSourcceyConfig
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

FPS = 30


def find_sourccey_model_path() -> tuple[str, str]:
    """Find the Sourccey model URDF and mesh paths."""
    # Get the path to the Sourccey V2 Beta model directory (used for V3 Beta compatibility)
    current_file = Path(__file__)
    # Navigate to src/lerobot/robots/sourccey/sourccey_v2beta/model
    sourccey_model_path = current_file.parent.parent.parent.parent / "src" / "lerobot" / "robots" / "sourccey" / "sourccey_v2beta" / "model"
    
    if sourccey_model_path.exists():
        urdf_path = str(sourccey_model_path / "Arm.urdf")
        mesh_path = str(sourccey_model_path / "meshes")
        print(f"Using URDF: {urdf_path}")
        print(f"Using meshes: {mesh_path}")
        return urdf_path, mesh_path
    else:
        print(f"ERROR: Could not find Sourccey V2 Beta model directory at {sourccey_model_path}")
        print("Make sure the Sourccey V2 Beta model files are available in src/lerobot/robots/sourccey/sourccey_v2beta/model/")
        raise FileNotFoundError(f"Sourccey model not found at {sourccey_model_path}")


def main():
    # Get URDF and mesh paths
    urdf_path, mesh_path = find_sourccey_model_path()
    
    # Create robot configuration (client connection to remote host)
    robot_config = SourcceyV3BetaClientConfig(
        remote_ip="192.168.1.219", 
        id="sourccey_v3beta"
    )
    
    # Create phone teleoperator configuration (right arm only)
    phone_config = PhoneTeleoperatorSourcceyConfig(
        id="phone_teleop_sourccey_v3beta",
        urdf_path=urdf_path,
        mesh_path=mesh_path,
        # In the URDF the end-effector link is named "gripper"
        target_link_name="Feetech-Servo-Motor-v1-5",
        sensitivity_normal=0.5,
        sensitivity_precision=0.2,
        rotation_sensitivity=1.0,
        initial_position=(0.0, -0.17, 0.237),
        initial_wxyz=(0, 0, 1, 0),  # wxyz quaternion
        # Rest pose for Sourccey V3 Beta right arm (conservative middle position in radians)
        rest_pose=(-0.843128, 1.552000, 0.736491, 0.591494, 0.020714, 0.009441),
        enable_visualization=True,
        viser_port=8080,
        # Sourccey V3 Beta gripper configuration
        gripper_min_pos=0.0,    # Gripper closed (0% on phone slider)
        gripper_max_pos=50.0,   # Gripper open (100% on phone slider)
        grpc_port=8765,         # Port for phone gRPC communication
        grpc_timeout=100.0      # Timeout for gRPC operations
    )
    
    # Create keyboard configuration for base movement
    keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")
    
    # Initialize devices
    robot = SourcceyV3BetaClient(robot_config)
    phone_teleop = PhoneTeleoperatorSourccey(phone_config)
    keyboard = KeyboardTeleop(keyboard_config)
    
    # Connect devices
    robot.connect()
    phone_teleop.connect()
    keyboard.connect()
    
    # Initialize rerun visualization
    _init_rerun(session_name="sourccey_v3beta_phone_teleop")
    
    if not robot.is_connected or not phone_teleop.is_connected or not keyboard.is_connected:
        raise ValueError("Robot, phone teleoperator, or keyboard is not connected!")
    
    print("Phone teleoperation ready for Sourccey V3 Beta (Right Arm Only)!")
    print("- Start the phone app and connect to the gRPC server (port 8765)")
    print("- Use your phone to control the right robot arm")
    print("- Use keyboard for base movement (WASD keys)")
    print("- Phone app controls:")
    print("  * Move phone to control right arm position")
    print("  * Rotate phone to control right arm orientation")
    print("  * Use slider for right gripper control")
    print("  * Toggle precision mode for fine control")
    print("  * Reset mapping to recalibrate phone-to-robot mapping")
    print("- Press Ctrl+C to exit")
    
    while True:
        t0 = time.perf_counter()
        
        observation = robot.get_observation()
        
        # Get action from phone teleoperator (right arm only)
        arm_action = phone_teleop.get_action(observation)
        
        # Get keyboard input for base movement
        keyboard_keys = keyboard.get_action()
        base_action = robot._from_keyboard_to_base_action(keyboard_keys)
        
        # Log data for visualization
        log_rerun_data(observation, {**arm_action, **base_action})
        
        # Combine arm and base actions
        action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
        
        # Send action to robot
        robot.send_action(action)
        
        # Maintain control frequency
        busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()