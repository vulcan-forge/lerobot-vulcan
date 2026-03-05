from dataclasses import dataclass, field
import time
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClientConfig, SourcceyClient
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.configs import parser

SHOULDER_KEYS = ("left_shoulder_lift.pos", "right_shoulder_lift.pos")


def _build_hold_action(obs: dict, action_keys: tuple[str, ...]) -> dict[str, float]:
    action: dict[str, float] = {}
    for key in action_keys:
        if key.endswith(".vel"):
            action[key] = 0.0
            continue
        if key in obs:
            action[key] = float(obs[key])
            continue
        if key == "z.pos":
            action[key] = 100.0
            continue
        action[key] = 0.0
    return action


def _await_stable_start_pose(
    robot: SourcceyClient,
    *,
    fps: int,
    timeout_s: float,
    min_stable_frames: int,
    max_shoulder_delta: float,
) -> None:
    action_keys = tuple(robot.action_features.keys())
    stable_count = 0
    previous_shoulder_obs: dict[str, float] | None = None
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        obs = robot.get_observation()
        robot.send_action(_build_hold_action(obs, action_keys))

        shoulder_obs = {
            key: float(obs[key])
            for key in SHOULDER_KEYS
            if key in obs and obs[key] is not None
        }

        if len(shoulder_obs) == len(SHOULDER_KEYS) and previous_shoulder_obs is not None:
            max_delta = max(abs(shoulder_obs[k] - previous_shoulder_obs[k]) for k in SHOULDER_KEYS)
            stable_count = stable_count + 1 if max_delta <= max_shoulder_delta else 0
            if stable_count >= min_stable_frames:
                return
        else:
            stable_count = 0

        previous_shoulder_obs = shoulder_obs if len(shoulder_obs) == len(SHOULDER_KEYS) else None
        time.sleep(max(1.0 / max(fps, 1), 0.01))

    raise RuntimeError(
        "Startup pose gate timed out: shoulder readings did not stabilize. "
        "Check arm state/calibration before starting eval."
    )


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
    model_path: str = "outputs/train/act__sourccey-003__myles__large-towel-fold-a-001-010/checkpoints/200000/pretrained_model"
    dataset: DatasetEvaluateConfig = field(default_factory=DatasetEvaluateConfig)
    # Client-side arm debug capture (first few seconds after motion starts).
    debug_capture_enabled: bool = True
    debug_capture_duration_s: float = 5.0
    debug_capture_motion_threshold: float = 1.0
    debug_capture_path: str | None = None
    # Eval-only startup gate: hold until shoulder readings are stable before policy starts.
    startup_pose_gate_enabled: bool = True
    startup_pose_gate_timeout_s: float = 2.0
    startup_pose_gate_min_stable_frames: int = 10
    startup_pose_gate_max_shoulder_delta: float = 8.0

@parser.wrap()
def evaluate(cfg: SourcceyEvaluateConfig):

    # Create the robot and teleoperator configurations
    robot_config = SourcceyClientConfig(
        remote_ip=cfg.remote_ip,
        id=cfg.id,
        debug_capture_enabled=cfg.debug_capture_enabled,
        debug_capture_duration_s=cfg.debug_capture_duration_s,
        debug_capture_motion_threshold=cfg.debug_capture_motion_threshold,
        debug_capture_path=cfg.debug_capture_path,
    )
    robot = SourcceyClient(robot_config)

    # Load config to determine policy type
    policy_config = PreTrainedConfig.from_pretrained(cfg.model_path)
    # Get the correct policy class based on config type
    policy_cls = get_policy_class(policy_config.type)
    # Create policy
    policy = policy_cls.from_pretrained(cfg.model_path)

    # Configure the dataset features
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # Create the dataset
    dataset = LeRobotDataset.create(
        repo_id=cfg.dataset.repo_id,
        fps=cfg.dataset.fps,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Build Policy Processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=cfg.model_path,
        dataset_stats=dataset.meta.stats,
        # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    # Connect to the robot
    robot.connect()

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    listener, events = init_keyboard_listener()
    init_rerun(session_name="recording")

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    print("Starting evaluate loop...")
    recorded_episodes = 0
    while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
        log_say(f"Running inference, recording eval episode {recorded_episodes} of {cfg.dataset.num_episodes}")
        if cfg.startup_pose_gate_enabled:
            _await_stable_start_pose(
                robot,
                fps=cfg.dataset.fps,
                timeout_s=cfg.startup_pose_gate_timeout_s,
                min_stable_frames=cfg.startup_pose_gate_min_stable_frames,
                max_shoulder_delta=cfg.startup_pose_gate_max_shoulder_delta,
            )

        # Run the policy inference loop
        record_loop(
            robot=robot,
            events=events,
            fps=cfg.dataset.fps,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=dataset,
            control_time_s=cfg.dataset.episode_time_s,
            single_task=cfg.dataset.task,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        # Logic for reset env
        if not events["stop_recording"] and (
            (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment")
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                control_time_s=cfg.dataset.reset_time_s,
                single_task=cfg.dataset.task,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

        if events["rerecord_episode"]:
            log_say("Re-record episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        recorded_episodes += 1

    # Upload to hub and clean up
    # dataset.push_to_hub()
    log_say("Stop recording")
    robot.disconnect()
    listener.stop()

def main():
    evaluate()

if __name__ == "__main__":
    main()
