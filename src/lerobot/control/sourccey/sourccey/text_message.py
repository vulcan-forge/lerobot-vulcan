#!/usr/bin/env python3
"""
Text Messaging Script for Sourccey Host

This script connects to a Sourccey host and allows sending/receiving text messages
via the ZMQ text communication pipeline.

Usage:
    python text_message.py --remote_ip <IP_ADDRESS>
    python text_message.py --remote_ip 192.168.1.243
"""

import argparse
import sys
import time
from dataclasses import dataclass

from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClientConfig, SourcceyClient


@dataclass
class TextMessageConfig:
    remote_ip: str = "192.168.1.243"
    id: str = "sourccey"


def text_message_client(cfg: TextMessageConfig):
    """Connect to host and send/receive text messages."""
    print(f"Connecting to Sourccey host at {cfg.remote_ip}...")
    
    # Create client configuration
    robot_config = SourcceyClientConfig(remote_ip=cfg.remote_ip, id=cfg.id)
    robot = SourcceyClient(robot_config)
    
    try:
        # Connect to host
        robot.connect()
        print("✓ Connected to host!")
        print(f"  - Command port: {cfg.remote_ip}:{robot.port_zmq_cmd}")
        print(f"  - Observation port: {cfg.remote_ip}:{robot.port_zmq_observations}")
        print(f"  - Text out port: {cfg.remote_ip}:{robot.port_zmq_text_out}")
        print(f"  - Text in port: {cfg.remote_ip}:{robot.port_zmq_text_in}")
        print()
        
        # Set up callback for incoming messages
        def handle_incoming_text(msg: str):
            print(f"[RECEIVED] {msg}")
        
        robot.set_text_message_callback(handle_incoming_text)
        
        print("Text messaging ready!")
        print("Type messages and press Enter to send. Type 'quit' or 'exit' to disconnect.")
        print("-" * 60)
        
        # Main loop: send messages and poll for incoming
        while True:
            try:
                # Poll for incoming messages (non-blocking)
                robot.poll_text_message()
                
                # Check for user input
                if sys.stdin.isatty():
                    # If running in interactive terminal, use input()
                    # Note: input() is blocking, but that's okay for this simple script
                    # The callback will handle incoming messages when we're not waiting for input
                    try:
                        user_input = input()
                        if not user_input.strip():
                            continue
                        
                        # Check for quit commands
                        if user_input.lower().strip() in ['quit', 'exit', 'q']:
                            print("Disconnecting...")
                            break
                        
                        # Send the message
                        success = robot.send_text(user_input)
                        if success:
                            print(f"[SENT] {user_input}")
                        else:
                            print(f"[FAILED] Could not send message: {user_input}")
                    except (EOFError, KeyboardInterrupt):
                        print("\nDisconnecting...")
                        break
                else:
                    # If not in interactive terminal, just poll for messages
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                print("\nDisconnecting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(0.1)
    
    except Exception as e:
        print(f"Failed to connect or communicate: {e}")
        sys.exit(1)
    finally:
        try:
            robot.disconnect()
            print("Disconnected.")
        except:
            pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Text messaging client for Sourccey host',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python text_message.py --remote_ip 192.168.1.243
  python text_message.py --remote_ip 192.168.1.228 --id my_robot
        """
    )
    parser.add_argument(
        '--remote_ip',
        type=str,
        default='192.168.1.243',
        help='IP address of the Sourccey host (default: 192.168.1.243)'
    )
    parser.add_argument(
        '--id',
        type=str,
        default='sourccey',
        help='Robot ID (default: sourccey)'
    )
    
    args = parser.parse_args()
    
    config = TextMessageConfig(remote_ip=args.remote_ip, id=args.id)
    text_message_client(config)


if __name__ == "__main__":
    main()

