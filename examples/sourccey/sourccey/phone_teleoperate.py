#!/usr/bin/env python

"""
Example usage of PhoneTeleoperator with Sourccey robot (left or right arm) sending to remote host.

This script demonstrates how to integrate phone teleoperation with remote robot control.
It uses the PhoneTeleoperator to receive commands from a mobile phone via gRPC and sends them
to a remote Sourccey robot host.

Requirements:
- Install additional dependencies: pip install pyroki viser yourdfpy
- Ensure the daxie package is installed and the gRPC server is accessible
- Have the robot URDF and mesh files available
- Have a remote Sourccey robot host running

Usage:
    python phone_teleop_sourccey_example_remote_host.py [left|right] [remote_ip]
    Default is left arm if no argument provided.
    Default remote IP is 192.168.1.227 if not provided.
"""

import math
import time
import argparse
from pathlib import Path

from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClient, SourcceyClientConfig
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Phone teleoperation for Sourccey robot')
    parser.add_argument('arm_side', nargs='?', default='left', choices=['left', 'right'],
                       help='Which arm to control (default: left)')
    parser.add_argument('remote_ip', nargs='?', default='192.168.1.227',
                       help='Remote host IP address (default: 192.168.1.227)')
    args = parser.parse_args()
    arm_side = args.arm_side
    
    print(f"Controlling {arm_side} arm")
    print(f"Remote host IP: {args.remote_ip}")
    
    # Get URDF and mesh paths from lerobot package
    from pathlib import Path
    
    # Get the path to the Sourccey V2 Beta model directory
    current_file = Path(__file__)
    # Use the Sourccey V2 Beta model located in the model directory
    sourccey_model_path = current_file.parent.parent.parent.parent / "src" / "lerobot" / "robots" / "sourccey" / "sourccey" / "sourccey" / "model"
        
    if sourccey_model_path.exists():
        urdf_path = str(sourccey_model_path / "Arm.urdf")
        mesh_path = str(sourccey_model_path / "meshes")
        print(f"Using URDF: {urdf_path}")
        print(f"Using meshes: {mesh_path}")
    else:
        print(f"ERROR: Could not find Sourccey model directory at {sourccey_model_path}")
        return
    
    # Robot configuration for remote host (connect to sourccey_host)
    robot_config = SourcceyClientConfig(remote_ip=args.remote_ip, id="sourccey")
    robot = SourcceyClient(robot_config)

    # Define initial positions and poses
    initial_position_left = (0.0, -0.17, 0.237)
    initial_wxyz_left = (0.0, 0.0, 1.0, 0.0)  # ~30° right yaw
    rest_pose_left = (
            -0.864068,   # shoulder_pan  (-49.506903 deg)
            2.095329,    # shoulder_lift (100.0 deg)
            -2.205474,   # elbow_flex    (-97.716150 deg)
            0.093922,    # wrist_flex    (5.381376 deg)
            0.014914,    # wrist_roll    (0.854701 deg)
            1.738416,    # gripper       (99.603960 -> used as 0-100)
        )
    
    initial_position_right = (0.09376381640512954, -0.17794639170766768, 0.2820500723608793)
    initial_wxyz_right = (0.0, 0.0, 1.0, 0.0)  # ~30° right yaw
    rest_pose_right = (
            0.640044,    # right_shoulder_pan  (36.671576 deg)
            -2.474699,   # right_shoulder_lift (-141.809571 deg, moved clockwise 45°)
            2.518931,    # right_elbow_flex    (144.292517 deg, moved counterclockwise 45°)
            -0.235352,   # right_wrist_flex    (-13.484163 deg)
            0.020884,    # right_wrist_roll    (1.196581 deg)
            0.004511,    # right_gripper       (0.258398 deg)
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
        initial_position=initial_position_right if arm_side == "right" else initial_position_left,
        initial_wxyz = initial_wxyz_right if arm_side == "right" else initial_wxyz_left,
        # Set rest_pose (radians) to match latest measured left-arm actions
        # Teleop does NOT flip rest pose; values here are the desired joint angles (deg):
        # [-49.506903, 100.0, -97.716150, 5.381376, 0.854701, 99.603960]
        rest_pose=rest_pose_right if arm_side == "right" else rest_pose_left,
        joint_offsets_deg={
            "shoulder_pan": 30.0 if arm_side == "right" else -30.0,
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
    )
    # Initialize teleoperator
    phone_teleop = PhoneTeleoperatorSourccey(phone_config)
    
    # Track last right arm position for proper reset functionality
    last_right_arm_position = None
    
    try:
        # Connect to remote robot host (sourccey_host should already be running)
        print(f"Connecting to remote robot host at {args.remote_ip}...")
        robot.connect()
        
        # Connect phone teleoperator (local - no physical robot needed)
        print("Connecting to phone teleoperator...")
        phone_teleop.connect()
        
        if not robot.is_connected or not phone_teleop.is_connected:
            raise ValueError("Remote robot host or phone teleoperator is not connected!")
        
        print(f"Phone teleoperation ready for {arm_side} arm!")
        print(f"- Connected to remote host at {args.remote_ip}")
        print("- Start the phone app and connect to the gRPC server")
        print("- Use your phone to control the robot")
        print("- Press Ctrl+C to exit")
        
        # Main control loop
        while True:
            try:
                # Get current observation from remote robot (if available)
                try:
                    observation = robot.get_observation()
                except Exception:
                    observation = {}

                # Feed the teleop the SAME schema the local follower uses: unprefixed joint keys
                if observation:
                    obs_for_teleop = {
                        k.replace("left_", ""): v
                        for k, v in observation.items()
                        if k.startswith("left_")
                    }
                else:
                    obs_for_teleop = {}

                if arm_side == "left":
                    # Get action from phone (emits left_* keys)
                    left_action = phone_teleop.get_action(obs_for_teleop)

                    # Keep left_* keys and convert numpy types to Python floats
                    action_left = {k: float(v) for k, v in left_action.items()}

                    # Right arm stays at rest position
                    action_right = {
                        "right_shoulder_pan.pos": float(math.degrees(0)),    # 36.671576 deg
                        "right_shoulder_lift.pos": float(math.degrees(-1.67)),  # -141.809571 deg
                        "right_elbow_flex.pos": float(math.degrees(0)),     # 144.292517 deg
                        "right_wrist_flex.pos": float(math.degrees(0)),    # -13.484163 deg
                        "right_wrist_roll.pos": float(math.degrees(0)),     # 1.196581 deg
                        "right_gripper.pos": float(math.degrees(0)),        # 0.258398 deg
                    }

                    # Combine left and right actions
                    action = {**action_left, **action_right}
                    
                    # Send action to robot
                    try:
                        robot.send_action(action)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                
                elif arm_side == "right":
                    # Get action from phone (emits left_* keys)
                    arm_action = phone_teleop.get_action(obs_for_teleop)

                    # Conditional mirroring: if teleop is at rest (left equals teleop rest), pass through;
                    # if teleop is in reset mode, pass through without mirroring;
                    # otherwise, mirror left_* to right-handed mapping.

                    # Get current reset state from phone teleop
                    current_is_resetting = getattr(phone_teleop, 'prev_is_resetting', False)

                    # Extract left_* vector in declared order
                    arm_vec = (
                        float(arm_action.get("left_shoulder_pan.pos", 0.0)),
                        float(arm_action.get("left_shoulder_lift.pos", 0.0)),
                        float(arm_action.get("left_elbow_flex.pos", 0.0)),
                        float(arm_action.get("left_wrist_flex.pos", 0.0)),
                        float(arm_action.get("left_wrist_roll.pos", 0.0)),
                    )

                    # Compute teleop rest in degrees (teleop may flip shoulder_lift sign only)
                    rest_deg = [
                        math.degrees(0.640044),   # pan
                        -math.degrees(-2.474699), # shoulder_lift flip in teleop rest path
                        math.degrees(2.518931),   # elbow
                        math.degrees(-0.235352),  # wrist_flex
                        math.degrees(0.020884),   # wrist_roll
                    ]

                    def is_close(a, b, eps=1e-3):
                        return abs(a - b) < eps

                    at_rest = all(is_close(lv, rv) for lv, rv in zip(arm_vec, rest_deg))

                    if current_is_resetting and last_right_arm_position is not None:
                        # During reset, use the last right arm position we sent
                        action = last_right_arm_position.copy()
                    elif at_rest:
                        # Pass through without mirroring when at rest
                        action = dict(arm_action)
                    else:
                        # Mirror around neutral (all zeros) with per-axis signs for right arm
                        neutral = (0.0, 0.0, 0.0, 0.0, 0.0)
                        r = (-1.0, +1.0, +1.0, +1.0, -1.0)
                        def mir(v, n, s):
                            return s * (2.0 * n - float(v))
                        action = {
                            "left_shoulder_pan.pos":  mir(arm_action.get("left_shoulder_pan.pos", 0.0),  neutral[0], r[0]),
                            "left_shoulder_lift.pos": mir(arm_action.get("left_shoulder_lift.pos", 0.0), neutral[1], r[1]),
                            "left_elbow_flex.pos":    mir(arm_action.get("left_elbow_flex.pos", 0.0),    neutral[2], r[2]),
                            "left_wrist_flex.pos":    mir(arm_action.get("left_wrist_flex.pos", 0.0),    neutral[3], r[3]),
                            "left_wrist_roll.pos":    mir(arm_action.get("left_wrist_roll.pos", 0.0),    neutral[4], r[4]),
                            # Gripper passthrough
                            "left_gripper.pos":       float(arm_action.get("left_gripper.pos", 0.0)),
                        }

                    # Rename action keys from left to right
                    right_actions = {k.replace("left_", "right_"): v for k, v in action.items()}

                    # Store the last right arm position for reset functionality
                    last_right_arm_position = right_actions.copy()

                    # Now we just have the left arms action be it's rest pose
                    # Set left arm to rest pose

                    left_actions = {
                        "left_shoulder_pan.pos": math.degrees(0),
                        "left_shoulder_lift.pos": math.degrees(1.69),
                        "left_elbow_flex.pos": math.degrees(0),
                        "left_wrist_flex.pos": math.degrees(0),
                        "left_wrist_roll.pos": math.degrees(0),
                        "left_gripper.pos": math.degrees(0),
                    }

                    # Combine left and right actions
                    action = {**left_actions, **right_actions}

                    robot.send_action(action)
                
                else:
                    print(f"Warning: Unknown arm_side '{arm_side}', no action sent")
                    action = {}
                

                
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