import logging
import time
from typing import Any

import cv2
import numpy as np

from .generated import sourccey_robot_pb2
from .generated import sourccey_common_pb2
from .generated import sourccey_follower_pb2

logger = logging.getLogger(__name__)


class SourcceyProtobuf:
    """Handles protobuf conversion for Sourccey robot actions and observations."""

    def __init__(self, robot_id: str = "sourccey"):
        self.robot_id = robot_id

    def action_to_protobuf(self, action: dict[str, Any]) -> sourccey_robot_pb2.SourcceyRobotAction:
        """Convert action dictionary to protobuf SourcceyRobotAction message."""
        try:
            robot_action = sourccey_robot_pb2.SourcceyRobotAction()
            robot_action.robot_id = self.robot_id
            robot_action.timestamp = time.time()

            # Process left arm action
            left_arm_action = sourccey_follower_pb2.SourcceyFollowerAction()
            left_arm_action.arm_id = "left"
            left_arm_action.timestamp = time.time()

            left_target_positions = sourccey_common_pb2.MotorJoint()
            left_target_positions.shoulder_pan = float(action.get("left_shoulder_pan.pos", 0.0))
            left_target_positions.shoulder_lift = float(action.get("left_shoulder_lift.pos", 0.0))
            left_target_positions.elbow_flex = float(action.get("left_elbow_flex.pos", 0.0))
            left_target_positions.wrist_flex = float(action.get("left_wrist_flex.pos", 0.0))
            left_target_positions.wrist_roll = float(action.get("left_wrist_roll.pos", 0.0))
            left_target_positions.gripper = float(action.get("left_gripper.pos", 0.0))
            left_arm_action.target_positions.CopyFrom(left_target_positions)

            robot_action.left_arm_action.CopyFrom(left_arm_action)

            # Process right arm action
            right_arm_action = sourccey_follower_pb2.SourcceyFollowerAction()
            right_arm_action.arm_id = "right"
            right_arm_action.timestamp = time.time()

            right_target_positions = sourccey_common_pb2.MotorJoint()
            right_target_positions.shoulder_pan = float(action.get("right_shoulder_pan.pos", 0.0))
            right_target_positions.shoulder_lift = float(action.get("right_shoulder_lift.pos", 0.0))
            right_target_positions.elbow_flex = float(action.get("right_elbow_flex.pos", 0.0))
            right_target_positions.wrist_flex = float(action.get("right_wrist_flex.pos", 0.0))
            right_target_positions.wrist_roll = float(action.get("right_wrist_roll.pos", 0.0))
            right_target_positions.gripper = float(action.get("right_gripper.pos", 0.0))
            right_arm_action.target_positions.CopyFrom(right_target_positions)

            robot_action.right_arm_action.CopyFrom(right_arm_action)

            # Process base action
            base_action = sourccey_common_pb2.BaseVelocity()
            base_action.x_vel = float(action.get("x.vel", 0.0))
            base_action.y_vel = float(action.get("y.vel", 0.0))
            base_action.theta_vel = float(action.get("theta.vel", 0.0))
            base_action.z_vel = float(action.get("z.vel", 0.0))
            robot_action.base_action.CopyFrom(base_action)

            return robot_action

        except ImportError as e:
            logger.error(f"Failed to import protobuf modules: {e}")
            logger.error("Run the protobuf setup script first: bash src/lerobot/robots/sourccey/sourccey/protobuf/compile.py")
            raise
        except Exception as e:
            logger.error(f"Failed to convert action to protobuf: {e}")
            raise

    def observation_to_protobuf(self, observation: dict[str, Any], left_arm_connected: bool = True,
                               right_arm_connected: bool = True, left_arm_calibrated: bool = True,
                               right_arm_calibrated: bool = True, robot_connected: bool = True,
                               both_arms_calibrated: bool = True) -> sourccey_robot_pb2.SourcceyRobotState:
        """Convert observation dictionary to protobuf SourcceyRobotState message."""
        try:
            msg = sourccey_robot_pb2.SourcceyRobotState()
            msg.robot_id = self.robot_id
            msg.robot_timestamp = time.time()
            msg.robot_connected = robot_connected
            msg.both_arms_calibrated = both_arms_calibrated

            # Set left arm state
            left_arm_msg = msg.left_arm
            left_arm_msg.arm_id = "left"
            left_arm_msg.orientation = "left"
            left_arm_msg.is_connected = left_arm_connected
            left_arm_msg.is_calibrated = left_arm_calibrated
            left_arm_msg.observation_timestamp = time.time()

            # Set left arm motor positions
            left_motor_pos = left_arm_msg.motor_positions
            left_motor_pos.shoulder_pan = observation.get("left_shoulder_pan.pos", 0.0)
            left_motor_pos.shoulder_lift = observation.get("left_shoulder_lift.pos", 0.0)
            left_motor_pos.elbow_flex = observation.get("left_elbow_flex.pos", 0.0)
            left_motor_pos.wrist_flex = observation.get("left_wrist_flex.pos", 0.0)
            left_motor_pos.wrist_roll = observation.get("left_wrist_roll.pos", 0.0)
            left_motor_pos.gripper = observation.get("left_gripper.pos", 0.0)

            # Set right arm state
            right_arm_msg = msg.right_arm
            right_arm_msg.arm_id = "right"
            right_arm_msg.orientation = "right"
            right_arm_msg.is_connected = right_arm_connected
            right_arm_msg.is_calibrated = right_arm_calibrated
            right_arm_msg.observation_timestamp = time.time()

            # Set right arm motor positions
            right_motor_pos = right_arm_msg.motor_positions
            right_motor_pos.shoulder_pan = observation.get("right_shoulder_pan.pos", 0.0)
            right_motor_pos.shoulder_lift = observation.get("right_shoulder_lift.pos", 0.0)
            right_motor_pos.elbow_flex = observation.get("right_elbow_flex.pos", 0.0)
            right_motor_pos.wrist_flex = observation.get("right_wrist_flex.pos", 0.0)
            right_motor_pos.wrist_roll = observation.get("right_wrist_roll.pos", 0.0)
            right_motor_pos.gripper = observation.get("right_gripper.pos", 0.0)

            # Set base velocity
            base_vel = msg.base_velocity
            base_vel.x_vel = observation.get("x.vel", 0.0)
            base_vel.y_vel = observation.get("y.vel", 0.0)
            base_vel.theta_vel = observation.get("theta.vel", 0.0)
            base_vel.z_vel = observation.get("z.vel", 0.0)

            # Set robot-level cameras
            for cam_key, cam_data in observation.items():
                if isinstance(cam_data, np.ndarray):
                    camera = msg.robot_cameras.add()
                    camera.name = cam_key
                    camera.jpeg_data = cv2.imencode('.jpg', cam_data)[1].tobytes()
                    camera.width = cam_data.shape[1]
                    camera.height = cam_data.shape[0]
                    camera.quality = 90
                    camera.timestamp = time.time()

            return msg

        except ImportError as e:
            logger.error(f"Failed to import protobuf modules: {e}")
            logger.error("Run the protobuf setup script first: bash src/lerobot/robots/sourccey/sourccey/protobuf/compile.py")
            raise
        except Exception as e:
            logger.error(f"Failed to convert observation to protobuf: {e}")
            raise

    def protobuf_to_action(self, action_msg: sourccey_robot_pb2.SourcceyRobotAction) -> dict[str, Any]:
        """Convert protobuf action to internal format."""
        try:
            action = {}

            # Convert left arm action
            if action_msg.HasField('left_arm_action'):
                left_motor_pos = action_msg.left_arm_action.target_positions
                action.update({
                    "left_shoulder_pan.pos": left_motor_pos.shoulder_pan,
                    "left_shoulder_lift.pos": left_motor_pos.shoulder_lift,
                    "left_elbow_flex.pos": left_motor_pos.elbow_flex,
                    "left_wrist_flex.pos": left_motor_pos.wrist_flex,
                    "left_wrist_roll.pos": left_motor_pos.wrist_roll,
                    "left_gripper.pos": left_motor_pos.gripper,
                })

            # Convert right arm action
            if action_msg.HasField('right_arm_action'):
                right_motor_pos = action_msg.right_arm_action.target_positions
                action.update({
                    "right_shoulder_pan.pos": right_motor_pos.shoulder_pan,
                    "right_shoulder_lift.pos": right_motor_pos.shoulder_lift,
                    "right_elbow_flex.pos": right_motor_pos.elbow_flex,
                    "right_wrist_flex.pos": right_motor_pos.wrist_flex,
                    "right_wrist_roll.pos": right_motor_pos.wrist_roll,
                    "right_gripper.pos": right_motor_pos.gripper,
                })

            # Convert base action
            if action_msg.HasField('base_action'):
                base_vel = action_msg.base_action
                action.update({
                    "x.vel": base_vel.x_vel,
                    "y.vel": base_vel.y_vel,
                    "theta.vel": base_vel.theta_vel,
                    "z.vel": base_vel.z_vel,
                })

            return action

        except ImportError as e:
            logger.error(f"Failed to import protobuf modules: {e}")
            logger.error("Run the protobuf setup script first: bash src/lerobot/robots/sourccey/sourccey/protobuf/compile.py")
            raise
        except Exception as e:
            logger.error(f"Failed to convert protobuf to action: {e}")
            raise

    def protobuf_to_observation(self, robot_state: sourccey_robot_pb2.SourcceyRobotState) -> dict[str, Any]:
        """Convert protobuf SourcceyRobotState message to observation dictionary."""
        try:
            observation = {}

            # Process left arm state
            if robot_state.HasField("left_arm"):
                left_arm = robot_state.left_arm
                if left_arm.HasField("motor_positions"):
                    motor_pos = left_arm.motor_positions
                    observation["left_shoulder_pan.pos"] = motor_pos.shoulder_pan
                    observation["left_shoulder_lift.pos"] = motor_pos.shoulder_lift
                    observation["left_elbow_flex.pos"] = motor_pos.elbow_flex
                    observation["left_wrist_flex.pos"] = motor_pos.wrist_flex
                    observation["left_wrist_roll.pos"] = motor_pos.wrist_roll
                    observation["left_gripper.pos"] = motor_pos.gripper

                # Process left arm cameras
                for camera in left_arm.cameras:
                    if camera.jpeg_data:
                        try:
                            # Decode JPEG data to numpy array
                            nparr = np.frombuffer(camera.jpeg_data, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if frame is not None:
                                observation[camera.name] = frame
                        except Exception as e:
                            logger.warning(f"Failed to decode camera image {camera.name}: {e}")

            # Process right arm state
            if robot_state.HasField("right_arm"):
                right_arm = robot_state.right_arm
                if right_arm.HasField("motor_positions"):
                    motor_pos = right_arm.motor_positions
                    observation["right_shoulder_pan.pos"] = motor_pos.shoulder_pan
                    observation["right_shoulder_lift.pos"] = motor_pos.shoulder_lift
                    observation["right_elbow_flex.pos"] = motor_pos.elbow_flex
                    observation["right_wrist_flex.pos"] = motor_pos.wrist_flex
                    observation["right_wrist_roll.pos"] = motor_pos.wrist_roll
                    observation["right_gripper.pos"] = motor_pos.gripper

                # Process right arm cameras
                for camera in right_arm.cameras:
                    if camera.jpeg_data:
                        try:
                            # Decode JPEG data to numpy array
                            nparr = np.frombuffer(camera.jpeg_data, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if frame is not None:
                                observation[camera.name] = frame
                        except Exception as e:
                            logger.warning(f"Failed to decode camera image {camera.name}: {e}")

            # Process base velocity
            if robot_state.HasField("base_velocity"):
                base_vel = robot_state.base_velocity
                observation["x.vel"] = base_vel.x_vel
                observation["y.vel"] = base_vel.y_vel
                observation["theta.vel"] = base_vel.theta_vel
                observation["z.vel"] = base_vel.z_vel

            # Process robot-level cameras
            for camera in robot_state.robot_cameras:
                if camera.jpeg_data:
                    try:
                        # Decode JPEG data to numpy array
                        nparr = np.frombuffer(camera.jpeg_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            observation[camera.name] = frame
                    except Exception as e:
                        logger.warning(f"Failed to decode camera image {camera.name}: {e}")

            return observation

        except ImportError as e:
            logger.error(f"Failed to import protobuf modules: {e}")
            logger.error("Run the protobuf setup script first: bash src/lerobot/robots/sourccey/sourccey/protobuf/compile.py")
            raise
        except Exception as e:
            logger.error(f"Failed to convert protobuf to observation: {e}")
            raise
