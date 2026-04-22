from dataclasses import dataclass
import time
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClientConfig, SourcceyClient
from lerobot.common.control_utils import init_keyboard_listener, predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from pathlib import Path
import logging
from lerobot.utils.constants import OBS_STR, HF_LEROBOT_HOME
from lerobot.configs import parser


AI_MODELS_ROOT = HF_LEROBOT_HOME / "ai_models"
OUTPUTS_ROOT = Path("outputs")


def _resolve_pretrained_path(candidate: Path) -> Path:
    if not candidate.exists():
        return candidate
    if candidate.is_file():
        return candidate

    config_path = candidate / "config.json"
    if config_path.exists():
        return candidate

    direct_pretrained = candidate / "pretrained_model"
    if direct_pretrained.is_dir():
        return direct_pretrained

    checkpoints_dir = candidate / "checkpoints"
    if not checkpoints_dir.is_dir():
        return candidate

    checkpoint_steps = []
    for entry in checkpoints_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            step = int(entry.name)
        except ValueError:
            continue
        checkpoint_steps.append((step, entry))

    for _, step_dir in sorted(checkpoint_steps, key=lambda item: item[0], reverse=True):
        pretrained_dir = step_dir / "pretrained_model"
        if pretrained_dir.is_dir():
            return pretrained_dir

    return candidate


def resolve_model_path(model_path: str) -> str:
    candidate = Path(model_path)
    if candidate.exists():
        return str(_resolve_pretrained_path(candidate))

    search_roots = [OUTPUTS_ROOT, AI_MODELS_ROOT]
    for root in search_roots:
        if not root.exists():
            continue

        joined = root / model_path
        if joined.exists():
            return str(_resolve_pretrained_path(joined))

        matches = list(root.rglob(model_path))
        if len(matches) == 1:
            return str(_resolve_pretrained_path(matches[0]))
        if len(matches) > 1:
            match_list = "\n".join(str(match) for match in matches[:10])
            raise ValueError(
                "Multiple model paths matched. Please be more specific.\n"
                f"Matches (first 10):\n{match_list}"
            )

    logging.warning(
        "Model path not found in outputs or ai_models cache. "
        "Proceeding with the original path in case it is a hub repo id."
    )
    return model_path


@dataclass
class SourcceyInferenceConfig:
    id: str = "sourccey"
    remote_ip: str = "192.168.1.243"
    model_path: str = "outputs/train/act__sourccey-003__myles__large-towel-fold-a-001-010/checkpoints/200000/pretrained_model"
    episode_time_s: int | float | None = None
    single_task: str = "Run inference"
    fps: int = 30
    # Display all cameras on screen
    display_data: bool = True
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to display compressed images in Rerun
    display_compressed_images: bool = False


def inference_loop(
    *,
    robot: SourcceyClient,
    robot_action_processor,
    robot_observation_processor,
    policy,
    preprocessor,
    postprocessor,
    features: dict,
    fps: int,
    control_time_s: int | float | None,
    single_task: str,
    events: dict,
    display_data: bool,
    display_compressed_images: bool,
):
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    timestamp = 0.0
    start_episode_t = time.perf_counter()
    while control_time_s is None or timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        observation_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
        action_tensor = predict_action(
            observation=observation_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=single_task,
            robot_type=robot.robot_type,
        )

        action_values = make_robot_action(action_tensor, features)
        robot_action_to_send = robot_action_processor((action_values, obs))
        robot.send_action(robot_action_to_send)

        if display_data:
            log_rerun_data(
                observation=obs_processed, action=action_values, compress_images=display_compressed_images
            )

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(1 / fps - dt_s, 0.0))
        timestamp = time.perf_counter() - start_episode_t

@parser.wrap()
def inference(cfg: SourcceyInferenceConfig):

    # Create the robot and teleoperator configurations
    robot_config = SourcceyClientConfig(remote_ip=cfg.remote_ip, id=cfg.id)
    robot = SourcceyClient(robot_config)

    resolved_model_path = resolve_model_path(cfg.model_path)

    # Load config to determine policy type
    policy_config = PreTrainedConfig.from_pretrained(resolved_model_path)
    # Get the correct policy class based on config type
    policy_cls = get_policy_class(policy_config.type)
    # Create policy
    policy = policy_cls.from_pretrained(resolved_model_path)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
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
    )

    # Build Policy Processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=resolved_model_path,
        dataset_stats=None,
        # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    # Connect to the robot
    robot.connect()

    listener, events = init_keyboard_listener()
    if cfg.display_data:
        init_rerun(session_name="inference", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    log_say("Starting inference", blocking=False)
    try:
        inference_loop(
            robot=robot,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            features=dataset_features,
            fps=cfg.fps,
            control_time_s=cfg.episode_time_s,
            single_task=cfg.single_task,
            events=events,
            display_data=cfg.display_data,
            display_compressed_images=display_compressed_images,
        )
    finally:
        log_say("Stop inference", blocking=True)
        if robot.is_connected:
            robot.disconnect()
        if listener:
            listener.stop()

def main():
    inference()

if __name__ == "__main__":
    main()
