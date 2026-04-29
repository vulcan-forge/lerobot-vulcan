import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from lerobot.common.control_utils import init_keyboard_listener, predict_action
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClient, SourcceyClientConfig
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class DatasetEvaluateConfig:
    repo_id: str = "sourccey-003/eval__act__sourccey-003__myles__large-towel-fold-a-001-010"
    num_episodes: int = 1
    episode_time_s: int = 30
    reset_time_s: int = 1
    task: str = "Fold the large towel in half"
    fps: int = 30
    push_to_hub: bool = False
    private: bool = False


@dataclass
class SourcceyEvaluateConfig:
    id: str = "sourccey"
    remote_ip: str = "192.168.1.243"
    model_path: str = (
        "outputs/train/act__sourccey-003__myles__large-towel-fold-a-001-010/checkpoints/200000/pretrained_model"
    )
    dataset: DatasetEvaluateConfig = field(default_factory=DatasetEvaluateConfig)


@parser.wrap()
def evaluate(cfg: SourcceyEvaluateConfig):
    robot_config = SourcceyClientConfig(remote_ip=cfg.remote_ip, id=cfg.id)
    robot = SourcceyClient(robot_config)

    policy_config = PreTrainedConfig.from_pretrained(cfg.model_path)
    policy_cls = get_policy_class(policy_config.type)
    policy = policy_cls.from_pretrained(cfg.model_path)
    inference_device = torch.device(str(policy.config.device))

    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    dataset_root = Path(HF_LEROBOT_HOME) / cfg.dataset.repo_id
    if dataset_root.exists():
        if dataset_root.is_dir():
            shutil.rmtree(dataset_root)
        else:
            dataset_root.unlink()
        print(f"Existing dataset directory removed: {dataset_root}")

    dataset = LeRobotDataset.create(
        repo_id=cfg.dataset.repo_id,
        fps=cfg.dataset.fps,
        features=dataset_features,
        root=dataset_root,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=cfg.model_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(inference_device)}},
    )

    robot.connect()
    listener, events = init_keyboard_listener()
    init_rerun(session_name="sourccey_evaluate")
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    print("Starting evaluate loop...")
    control_interval = 1 / cfg.dataset.fps
    recorded_episodes = 0
    try:
        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            log_say(
                f"Running inference, recording eval episode {recorded_episodes} of {cfg.dataset.num_episodes}"
            )
            start_episode_t = time.perf_counter()
            timestamp = 0.0

            while timestamp < cfg.dataset.episode_time_s and not events["stop_recording"]:
                start_loop_t = time.perf_counter()
                if events["exit_early"]:
                    events["exit_early"] = False
                    break

                obs = robot.get_observation()
                obs_processed = robot_observation_processor(obs)
                observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

                action_tensor = predict_action(
                    observation=observation_frame,
                    policy=policy,
                    device=inference_device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=inference_device.type == "cuda" and getattr(policy.config, "use_amp", True),
                    task=cfg.dataset.task,
                    robot_type=robot.name,
                )
                action_values = make_robot_action(action_tensor, dataset.features)
                robot_action_to_send = robot_action_processor((action_values, obs))
                robot.send_action(robot_action_to_send)

                action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
                frame = {**observation_frame, **action_frame, "task": cfg.dataset.task}
                dataset.add_frame(frame)

                log_rerun_data(observation=obs_processed, action=action_values)

                dt_s = time.perf_counter() - start_loop_t
                sleep_time_s = control_interval - dt_s
                if sleep_time_s < 0:
                    logging.warning(
                        "Evaluate loop is running slower (%.1f Hz) than target FPS (%d Hz).",
                        1 / dt_s if dt_s > 0 else float("inf"),
                        cfg.dataset.fps,
                    )
                precise_sleep(max(sleep_time_s, 0.0))
                timestamp = time.perf_counter() - start_episode_t

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded_episodes += 1

            if recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say("Reset the environment")
                precise_sleep(float(cfg.dataset.reset_time_s))
    finally:
        log_say("Stop recording")
        robot.disconnect()
        if listener is not None:
            listener.stop()


def main():
    evaluate()


if __name__ == "__main__":
    main()
