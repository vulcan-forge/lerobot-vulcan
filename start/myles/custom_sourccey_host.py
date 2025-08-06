#!/usr/bin/env python3

import base64
import json
import logging
import time

import cv2
import zmq

from start.myles.custom_sourccey_config import CustomSourcceyV2BetaConfig, CustomSourcceyV2BetaHostConfig
from lerobot.common.robots.sourccey_v2beta.sourccey_v2beta import SourcceyV2Beta

class CustomSourcceyV2BetaHost:
    def __init__(self, config: CustomSourcceyV2BetaHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.robot_config = CustomSourcceyV2BetaConfig()
        self.robot = SourcceyV2Beta(self.robot_config)
        
        self.config = config
        self.logger = logging.getLogger(__name__)

    def run(self):
        """Main loop of the host"""
        self.logger.info("Starting Sourccey V2Beta Host...")
        
        try:
            # Connect to the robot
            self.robot.connect()
            self.logger.info("Robot connected successfully!")
            
            # Initialize cameras
            self.robot.cameras.connect()
            self.logger.info("Cameras connected successfully!")
            
            start_time = time.time()
            
            while time.time() - start_time < self.config.connection_timeout_s:
                try:
                    # Check for commands (non-blocking)
                    try:
                        cmd_data = self.zmq_cmd_socket.recv_string(flags=zmq.NOBLOCK)
                        cmd = json.loads(cmd_data)
                        self.logger.info(f"Received command: {cmd}")
                        
                        # Process the command
                        self._process_command(cmd)
                        
                    except zmq.Again:
                        # No command received, continue
                        pass
                    
                    # Get observations
                    obs = self._get_observations()
                    
                    # Send observations
                    obs_data = json.dumps(obs)
                    self.zmq_observation_socket.send_string(obs_data)
                    
                    # Control loop frequency
                    time.sleep(1.0 / self.config.max_loop_freq_hz)
                    
                except KeyboardInterrupt:
                    self.logger.info("Received keyboard interrupt, shutting down...")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    break
                    
        except Exception as e:
            self.logger.error(f"Failed to start host: {e}")
        finally:
            self.cleanup()

    def _process_command(self, cmd):
        """Process incoming commands"""
        if "action" in cmd:
            action = cmd["action"]
            self.robot.apply_action(action)
            self.logger.info(f"Applied action: {action}")

    def _get_observations(self):
        """Get current observations from the robot"""
        obs = {}
        
        # Get robot state
        try:
            obs["robot_state"] = self.robot.get_robot_state()
        except Exception as e:
            self.logger.warning(f"Failed to get robot state: {e}")
            obs["robot_state"] = {}
        
        # Get camera images
        try:
            images = self.robot.cameras.get_images()
            obs["images"] = {}
            for camera_name, image in images.items():
                if image is not None:
                    # Convert image to base64 for JSON serialization
                    _, buffer = cv2.imencode('.jpg', image)
                    obs["images"][camera_name] = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            self.logger.warning(f"Failed to get camera images: {e}")
            obs["images"] = {}
        
        return obs

    def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up...")
        try:
            self.robot.disconnect()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

def main():
    logging.basicConfig(level=logging.INFO)
    config = CustomSourcceyV2BetaHostConfig()
    host = CustomSourcceyV2BetaHost(config)
    host.run()

if __name__ == "__main__":
    main() 