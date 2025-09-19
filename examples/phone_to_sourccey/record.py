#!/usr/bin/env python

"""
Record dataset while teleoperating Sourccey with the phone teleoperator.

This mirrors the SO100 example's record flow but keeps the Sourccey phone
teleoperation stack (PhoneTeleoperatorSourccey + SourcceyClient).

Usage:
  python examples/phone_to_sourccey/record.py [left|right] [remote_ip]
  Defaults: left arm, IP 192.168.1.237
"""

import argparse
import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.model.kinematics import RobotKinematics  # Not used here but kept for parity
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.record import record_loop
from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClient, SourcceyClientConfig
from lerobot.teleoperators.phone_teleoperator import (
    PhoneTeleoperatorSourccey,
    PhoneTeleoperatorSourcceyConfig,
)
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun


# Recording parameters
NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 20
TASK_DESCRIPTION = "Phone teleop with Sourccey"
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"


def main():
    parser = argparse.ArgumentParser(description="Record dataset using phone teleop -> Sourccey")
    parser.add_argument("arm_side", nargs="?", default="left", choices=["left", "right"], help="Arm to control")
    parser.add_argument("remote_ip", nargs="?", default="192.168.1.237", help="Remote host IP")
    args = parser.parse_args()

    # Robot client (receives actions over ZMQ)
    robot_config = SourcceyClientConfig(remote_ip=args.remote_ip, id="sourccey")
    robot = SourcceyClient(robot_config)

    # Phone teleoperator (does mapping + IK internally)
    teleop_config = PhoneTeleoperatorSourcceyConfig(
        id="phone_teleop_record",
        urdf_path=None,  # already configured in teleop config if needed
        arm_side=args.arm_side,
        target_link_name="Feetech-Servo-Motor-v1-5",
        sensitivity_normal=0.5,
        sensitivity_precision=0.2,
        rotation_sensitivity=1.0,
        enable_visualization=False,
        viser_port=8080,
        gripper_min_pos=0.0,
        gripper_max_pos=50.0,
    )
    teleop = PhoneTeleoperatorSourccey(teleop_config)

    # Identity processors: teleop action is already in robot space; observation is passed through
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # Build dataset feature contract from the identity pipelines
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(action=robot.action_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=True,
            ),
        ),
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Connect
    robot.connect()
    teleop.connect()

    # Input + viz
    listener, events = init_keyboard_listener()
    _init_rerun(session_name=f"phone_sourccey_record_{args.arm_side}")

    # Record episodes
    print("Starting record loop. Move your phone to teleoperate the robot...")
    recorded = 0
    try:
        while recorded < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Recording episode {recorded + 1} of {NUM_EPISODES}")
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop=teleop,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )

            # Reset segment (no recording), to let user reposition if needed
            if not events["stop_recording"] and recorded < NUM_EPISODES - 1:
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop=teleop,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    control_time_s=RESET_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=True,
                )

            dataset.save_episode()
            recorded += 1
    finally:
        # Cleanup
        print("Stop recording")
        robot.disconnect()
        teleop.disconnect()
        listener.stop()

    # Optionally push to hub (edit HF_REPO_ID first)
    # dataset.push_to_hub()


if __name__ == "__main__":
    main()


