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
import threading
import os
from pathlib import Path

from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClient, SourcceyClientConfig
from lerobot.teleoperators.phone_teleoperator import PhoneTeleoperatorSourccey, PhoneTeleoperatorSourcceyConfig
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("WARNING: pynput not available - Q key exit disabled")



class QKeyHandler:
    """Handles Q key for immediate exit using pynput"""
    
    def __init__(self):
        self.should_exit = False
        self.listener = None
        
    def start(self):
        """Start the Q key listener"""
        if not PYNPUT_AVAILABLE:
            return
            
        def on_press(key):
            try:
                if key.char == 'q' or key.char == 'Q':
                    print("\nQ key pressed - exiting immediately...")
                    self.should_exit = True
                    os._exit(0)  # Force immediate exit
            except AttributeError:
                pass
                
        self.listener = keyboard.Listener(on_press=on_press, suppress=False)
        self.listener.start()
        
    def stop(self):
        """Stop the Q key listener"""
        if self.listener:
            self.listener.stop()


class ThreadedKeyboardHandler:
    """Handles keyboard input in a separate thread to avoid blocking"""
    
    def __init__(self, robot):
        self.robot = robot
        self.keyboard = None
        self.keyboard_config = KeyboardTeleopConfig(id="keyboard")
        self.current_base_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
        
    def start(self):
        """Start the keyboard handler in a separate thread"""
        self.keyboard = KeyboardTeleop(self.keyboard_config)
        self.keyboard.connect()
        
        if not self.keyboard.is_connected:
            print("WARNING: Keyboard not connected - wheel control disabled")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self.thread.start()
        print("Threaded keyboard handler started")
        
    def stop(self):
        """Stop the keyboard handler"""
        self.running = False
        if self.thread:
            self.thread.join()
        if self.keyboard:
            try:
                self.keyboard.disconnect()
            except:
                pass
                
    def _keyboard_loop(self):
        """Main keyboard loop running in separate thread"""
        while self.running:
            try:
                # Get keyboard input
                keyboard_keys = self.keyboard.get_action()
                
                # Convert to base action
                base_action = self.robot._from_keyboard_to_base_action(keyboard_keys)
                
                # Update current action thread-safely
                with self.lock:
                    self.current_base_action = base_action
                    
                time.sleep(1/60)  # 60 Hz keyboard polling
                
            except Exception as e:
                print(f"Keyboard thread error: {e}")
                time.sleep(0.1)
                
    def get_base_action(self):
        """Get the current base action thread-safely"""
        with self.lock:
            return self.current_base_action.copy()


def find_sourccey_model_path():
    """Find the Sourccey URDF path."""
    current_file = Path(__file__)
    sourccey_model_path = current_file.parent.parent.parent / "src" / "lerobot" / "robots" / "sourccey" / "sourccey" / "sourccey" / "model"
    
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
    
    # Initialize teleoperator, threaded keyboard handler, and Q key handler
    phone_teleop = PhoneTeleoperatorSourccey(phone_config)
    
    # Toggle this to bypass all modifications and use raw IK solver
    phone_teleop.tune["bypass_all_mods"] = False  # Set to True for raw IK behavior
    
    threaded_keyboard = ThreadedKeyboardHandler(robot)
    q_key_handler = QKeyHandler()
    
    try:
        # Connect phone teleoperator first (this starts the gRPC server)
        print("Connecting to phone teleoperator...")
        phone_teleop.connect()
        
        # Connect to remote robot host
        print(f"Connecting to remote robot host at {args.remote_ip}...")
        try:
            robot.connect()
        except Exception as e:
            print(f"WARNING: Could not connect to remote robot: {e}")
            print("Continuing with phone teleop only (gRPC server will still work)...")
        
        # Start threaded keyboard handler
        print("Starting threaded keyboard handler...")
        threaded_keyboard.start()
        
        # Start Q key handler for immediate exit
        print("Starting Q key handler...")
        q_key_handler.start()
        
        # Initialize rerun viewer for visualization
        print("Initializing rerun viewer...")
        _init_rerun(session_name=f"phone_sourccey_{args.arm_side}_teleop")
        
        if not phone_teleop.is_connected:
            raise ValueError("Phone teleoperator is not connected!")
        
        if not robot.is_connected:
            print("WARNING: Robot not connected - phone teleop will work but no robot control")
        
        print(f"Phone teleoperation ready for {args.arm_side} arm!")
        print(f"- Connected to remote host at {args.remote_ip}")
        print("- Phone controls robot arm")
        print("- Keyboard controls wheels (W/A/S/D/Z/X/R/F)")
        print("- Start the phone app and connect to the gRPC server")
        print("- Use your phone to control the robot")
        print("- Press Q to exit immediately")
        print("- Rerun viewer is running for real-time visualization")
        
        # Main control loop
        while True:
            try:
                # Get current observation from remote robot
                try:
                    observation = robot.get_observation()
                except Exception:
                    observation = {}

                # Get arm action from phone teleoperator (handles all logic internally)
                arm_action = phone_teleop.get_action(observation)
                
                # Get base action from threaded keyboard handler
                base_action = threaded_keyboard.get_base_action()
                

                # Combine arm and wheel actions
                # If phone provided base velocities, map them via client's analog path to match keyboard scaling
                phone_x = arm_action.get("x.vel", 0.0)
                phone_y = arm_action.get("y.vel", 0.0)
                phone_theta = arm_action.get("theta.vel", 0.0)

                if any(abs(v) > 0.0 for v in (phone_x, phone_y, phone_theta)):
                    analog_base = robot._from_analog_to_base_action(phone_x, phone_y, phone_theta)
                else:
                    analog_base = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

                # Keyboard overrides phone when non-zero
                kb_active = any(abs(base_action.get(k, 0.0)) > 0.0 for k in ("x.vel","y.vel","theta.vel"))
                effective_base = base_action if kb_active else analog_base
                action = {**arm_action, **effective_base}
                
                # Send combined action to robot (if connected)
                if robot.is_connected:
                    # Debug: show base command being sent and source
                    if any(abs(effective_base.get(k, 0.0)) > 0.0 for k in ("x.vel","y.vel","theta.vel")):
                        src = "keyboard" if kb_active else "phone"
                        print(f"CLIENT WHEELS: Sending base ({src}) x={effective_base.get('x.vel',0.0):.3f}, y={effective_base.get('y.vel',0.0):.3f}, theta={effective_base.get('theta.vel',0.0):.3f}")
                    try:
                        robot.send_action(action)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                else:
                    # Just print the action for debugging when robot not connected
                    if any(abs(v) > 0.0 for v in [action.get("x.vel", 0), action.get("y.vel", 0), action.get("theta.vel", 0)]):
                        print(f"DEBUG: Would send base action: {action}")
                
                # Visualize with rerun
                log_rerun_data(observation=observation, action=action)
                
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
            q_key_handler.stop()
        except:
            pass
        try:
            phone_teleop.disconnect()
        except:
            pass
        try:
            threaded_keyboard.stop()
        except:
            pass
        try:
            robot.disconnect()
        except:
            pass
        print("Done!")


if __name__ == "__main__":
    main() 