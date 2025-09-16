#!/usr/bin/env python

"""
Simple wrapper for PhoneTeleoperator with Sourccey robot.

This script demonstrates how to use the phone teleoperation system with a remote Sourccey robot.
All complex logic (arm selection, mirroring, rest poses) is handled by the teleoperator system.

Requirements:
- Install dependencies: pip install -e .[sourccey,teleop-phone]
- Have a remote Sourccey robot host running

Usage:
    python phone_teleoperate.py [left|right] [remote_ip]
    Default is left arm if no argument provided.
    Default remote IP is 192.168.1.227 if not provided.
"""

import time
import argparse
import os
from pathlib import Path

from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClient, SourcceyClientConfig
from lerobot.teleoperators.phone_teleoperator import PhoneTeleoperatorSourccey, PhoneTeleoperatorSourcceyConfig

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("Warning: pynput not available. Q key exit won't work.")


class QKeyHandler:
    """Handles Q key press for immediate exit."""
    
    def __init__(self):
        self.should_exit = False
        self.listener = None
        
    def start(self):
        """Start listening for Q key press."""
        if not PYNPUT_AVAILABLE:
            return
            
        def on_press(key):
            try:
                if key.char and key.char.lower() == 'q':
                    print("\nQ key pressed - exiting immediately...")
                    self.should_exit = True
                    os._exit(0)  # Force immediate exit
            except AttributeError:
                # Special keys (ctrl, alt, etc.) don't have char attribute
                pass
        
        self.listener = keyboard.Listener(on_press=on_press, suppress=False)
        self.listener.start()
    
    def stop(self):
        """Stop the key listener."""
        if self.listener:
            self.listener.stop()


def find_sourccey_model_path():
    """Find the Sourccey URDF path."""
    current_file = Path(__file__)
    sourccey_model_path = current_file.parent.parent.parent.parent / "src" / "lerobot" / "robots" / "sourccey" / "sourccey" / "sourccey" / "model"
    
    if not sourccey_model_path.exists():
        raise FileNotFoundError(f"Could not find Sourccey model directory at {sourccey_model_path}")
    
    urdf_path = str(sourccey_model_path / "Arm.urdf")
    print(f"Using URDF: {urdf_path}")
    return urdf_path


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Phone teleoperation for Sourccey robot')
    parser.add_argument('arm_side', nargs='?', default='left', choices=['left', 'right'],
                       help='Which arm to control (default: left)')
    parser.add_argument('remote_ip', nargs='?', default='192.168.1.237',
                       help='Remote host IP address (default: 192.168.1.237)')
    args = parser.parse_args()
    
    print(f"Controlling {args.arm_side} arm")
    print(f"Remote host IP: {args.remote_ip}")
    
    # Get URDF path
    urdf_path = find_sourccey_model_path()
    
    # Robot configuration for remote host
    robot_config = SourcceyClientConfig(remote_ip=args.remote_ip, id="sourccey")
    robot = SourcceyClient(robot_config)

    # Configuration for the phone teleoperator
    phone_config = PhoneTeleoperatorSourcceyConfig(
        id="phone_teleop_main",
        urdf_path=urdf_path,
        arm_side=args.arm_side,
        target_link_name="Feetech-Servo-Motor-v1-5",
        sensitivity_normal=0.5,
        sensitivity_precision=0.2,
        rotation_sensitivity=1.0,
        enable_visualization=True,
        viser_port=8080,
        gripper_min_pos=0.0,
        gripper_max_pos=50.0,
    )
    
    # Initialize teleoperator
    phone_teleop = PhoneTeleoperatorSourccey(phone_config)
    
    # Initialize Q key handler for immediate exit
    q_handler = QKeyHandler()
    
    try:
        # Connect to remote robot host
        print(f"Connecting to remote robot host at {args.remote_ip}...")
        robot.connect()
        
        # Connect phone teleoperator
        print("Connecting to phone teleoperator...")
        phone_teleop.connect()
        
        if not robot.is_connected or not phone_teleop.is_connected:
            raise ValueError("Remote robot host or phone teleoperator is not connected!")
        
        # Start Q key handler
        q_handler.start()
        
        print(f"Phone teleoperation ready for {args.arm_side} arm!")
        print(f"- Connected to remote host at {args.remote_ip}")
        print("- Start the phone app and connect to the gRPC server")
        print("- Use your phone to control the robot")
        print("- Press Q to exit")
        
        # Main control loop
        while True:
            try:
                # Get current observation from remote robot
                try:
                    observation = robot.get_observation()
                except Exception:
                    observation = {}

                # Get action from phone teleoperator (handles all logic internally)
                action = phone_teleop.get_action(observation)
                
                # Send action to robot
                try:
                    robot.send_action(action)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                
                # Control frequency
                time.sleep(max(0, 1/30))  # Target ~30 Hz
                
            except KeyboardInterrupt:
                print("\nStopping teleoperation...")
                break
            except Exception as e:
                print(f"Error in control loop: {e}")
                time.sleep(0.1)
    
    finally:
        # Cleanup
        print("Disconnecting devices...")
        try:
            q_handler.stop()
        except:
            pass
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