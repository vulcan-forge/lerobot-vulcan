#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Optional

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("sourccey_teleop")
@dataclass
class PhoneTeleoperatorSourcceyConfig(TeleoperatorConfig):
    """Configuration for Sourccey phone teleoperation."""
    
    # gRPC server settings
    grpc_port: int = 8765  # Default port to match phone app
    grpc_timeout: float = 100.0
    
    # Robot model paths - same as SO100
    urdf_path: str = "lerobot/robots/sourccey/sourccey_v2beta/model/Arm.urdf"
    mesh_path: str = "lerobot/robots/sourccey/sourccey_v2beta/model/meshes"
    
    # IK solver settings - same as SO100
    target_link_name: str = "Feetech-Servo-Motor-v1-5"
    rest_pose: tuple[float, ...] = (-0.843128, 1.552000, 0.736491, 0.591494, 0.020714, 0.009441)   # Always in radians - initial robot positions for IK solver

    # Phone mapping settings
    rotation_sensitivity: float = 1.0
    sensitivity_normal: float = 0.5
    sensitivity_precision: float = 0.2
    
    # Initial robot pose (when connecting phone) - same as SO100
    initial_position: tuple[float, ...] = (0.0, -0.17, 0.237)  # meters
    initial_wxyz: tuple[float, ...] = (0, 0, 1, 0)  # quaternion (w,x,y,z)
    
    # Visualization settings
    enable_visualization: bool = True
    viser_port: int = 8080
    
    # Gripper settings - same as SO100
    gripper_min_pos: float = 0.0    # Gripper closed position (0% slider)
    gripper_max_pos: float = 50.0   # Gripper open position (100% slider) 