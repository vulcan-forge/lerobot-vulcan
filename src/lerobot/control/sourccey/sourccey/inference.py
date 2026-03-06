from dataclasses import dataclass
import time
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClientConfig, SourcceyClient
from lerobot.utils.control_utils import init_keyboard_listener, predict_action
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
    # Startup preflight to move shoulder_lift out of seam-risk startup poses.
    startup_guard_enabled: bool = False
    startup_guard_shoulder_abs_limit: float = 95.0
    startup_guard_poll_interval_s: float = 0.1
    startup_guard_timeout_s: float = 30.0
    startup_guard_auto_align_enabled: bool = True
    startup_guard_left_target: float = 70.0
    startup_guard_right_target: float = -70.0
    startup_guard_max_step_per_s: float = 20.0
    startup_guard_min_step: float = 0.5
    # Fail-open behavior: after a bounded preflight window, start policy even if
    # shoulder feedback remains seam-aliased/stuck.
    startup_guard_fail_open_enabled: bool = True
    startup_guard_fail_open_after_s: float = 5.0
    startup_guard_fail_open_min_steps: int = 10
    # Startup seam sign canonicalization in client observation.
    startup_shoulder_seam_filter_enabled: bool = True
    startup_shoulder_seam_filter_duration_s: float = 8.0
    startup_shoulder_seam_abs_threshold: float = 90.0
    startup_shoulder_seam_required_plausible_frames: int = 10
    startup_shoulder_seam_max_delta_per_s: float = 250.0
    startup_shoulder_seam_delta_margin: float = 8.0


_ARM_KEYS = (
    "left_shoulder_pan.pos",
    "left_shoulder_lift.pos",
    "left_elbow_flex.pos",
    "left_wrist_flex.pos",
    "left_wrist_roll.pos",
    "left_gripper.pos",
    "right_shoulder_pan.pos",
    "right_shoulder_lift.pos",
    "right_elbow_flex.pos",
    "right_wrist_flex.pos",
    "right_wrist_roll.pos",
    "right_gripper.pos",
)


def _make_hold_action_from_observation(observation: dict[str, float], *, untorque: bool) -> dict[str, float | bool]:
    action: dict[str, float | bool] = {}
    for key in _ARM_KEYS:
        action[key] = float(observation.get(key, 0.0))

    action["x.vel"] = 0.0
    action["y.vel"] = 0.0
    action["theta.vel"] = 0.0
    action["z.pos"] = float(observation.get("z.pos", 0.0))
    action["untorque_left"] = untorque
    action["untorque_right"] = untorque
    return action


def _move_toward(current: float, target: float, max_step: float, min_step: float) -> float:
    delta = target - current
    if abs(delta) <= max_step:
        return float(target)
    step = max(max_step, min_step)
    return float(current + step) if delta > 0 else float(current - step)


def _ensure_startup_align_state(
    current_state: dict[str, float] | None,
    *,
    left: float,
    right: float,
) -> dict[str, float]:
    """Seed/refresh startup controller state from canonicalized shoulder values."""
    if current_state is None:
        return {
            "left_shoulder_lift.pos": float(left),
            "right_shoulder_lift.pos": float(right),
        }
    return current_state


def _is_bad_start_pose(left_shoulder_lift: float, right_shoulder_lift: float, abs_limit: float) -> bool:
    # Startup seam-risk: either shoulder near +/- boundary.
    return (abs(left_shoulder_lift) >= abs_limit) or (abs(right_shoulder_lift) >= abs_limit)


def _canonicalize_startup_shoulders(
    *, left_shoulder_lift: float, right_shoulder_lift: float, abs_limit: float
) -> tuple[float, float, list[str]]:
    """
    Canonicalize seam-aliased startup shoulder signs before making control decisions.
    Near the seam, left should be positive and right should be negative.
    """
    changed: list[str] = []
    left = float(left_shoulder_lift)
    right = float(right_shoulder_lift)

    if abs(left) >= abs_limit and left < 0.0:
        left = abs(left)
        changed.append("left_shoulder_lift.pos")
    if abs(right) >= abs_limit and right > 0.0:
        right = -abs(right)
        changed.append("right_shoulder_lift.pos")

    return left, right, changed


def _startup_shoulder_guard(robot: SourcceyClient, events: dict[str, bool], cfg: SourcceyInferenceConfig) -> bool:
    if not cfg.startup_guard_enabled:
        return True

    deadline_t = time.perf_counter() + max(0.0, float(cfg.startup_guard_timeout_s))
    poll_dt = max(0.02, float(cfg.startup_guard_poll_interval_s))
    limit = float(cfg.startup_guard_shoulder_abs_limit)
    warned_once = False
    last_log_t = 0.0
    align_state: dict[str, float] | None = None
    startup_t0 = time.perf_counter()
    startup_step_count = 0

    while True:
        if events.get("exit_early", False):
            events["exit_early"] = False
            return False

        obs = robot.get_observation()
        raw_left = float(obs.get("left_shoulder_lift.pos", 0.0))
        raw_right = float(obs.get("right_shoulder_lift.pos", 0.0))
        left, right, seam_flipped = _canonicalize_startup_shoulders(
            left_shoulder_lift=raw_left,
            right_shoulder_lift=raw_right,
            abs_limit=limit,
        )
        bad_pose = _is_bad_start_pose(left, right, limit)

        if not bad_pose:
            align_state = None
            robot.send_action(_make_hold_action_from_observation(obs, untorque=False))
            if warned_once:
                logging.info(
                    "Startup shoulder guard released. raw(left=%.2f,right=%.2f) control(left=%.2f,right=%.2f)",
                    raw_left,
                    raw_right,
                    left,
                    right,
                )
            return True

        action = _make_hold_action_from_observation(obs, untorque=False)
        if cfg.startup_guard_auto_align_enabled:
            max_step = max(0.01, float(cfg.startup_guard_max_step_per_s) * poll_dt)
            min_step = max(0.0, float(cfg.startup_guard_min_step))
            align_state = _ensure_startup_align_state(
                align_state,
                left=left,
                right=right,
            )

            if abs(left) >= limit:
                left_target = float(cfg.startup_guard_left_target)
                next_left = _move_toward(
                    float(align_state["left_shoulder_lift.pos"]),
                    left_target,
                    max_step=max_step,
                    min_step=min_step,
                )
                align_state["left_shoulder_lift.pos"] = next_left
                action["left_shoulder_lift.pos"] = next_left
            else:
                align_state["left_shoulder_lift.pos"] = float(left)
                action["left_shoulder_lift.pos"] = float(left)

            if abs(right) >= limit:
                right_target = float(cfg.startup_guard_right_target)
                next_right = _move_toward(
                    float(align_state["right_shoulder_lift.pos"]),
                    right_target,
                    max_step=max_step,
                    min_step=min_step,
                )
                align_state["right_shoulder_lift.pos"] = next_right
                action["right_shoulder_lift.pos"] = next_right
            else:
                align_state["right_shoulder_lift.pos"] = float(right)
                action["right_shoulder_lift.pos"] = float(right)
        else:
            action["untorque_left"] = True
            action["untorque_right"] = True

        robot.send_action(action)
        startup_step_count += 1

        now = time.perf_counter()
        if (not warned_once) or (now - last_log_t >= 2.0):
            if cfg.startup_guard_auto_align_enabled:
                logging.warning(
                    "Startup shoulder auto-align active. raw(left=%.2f,right=%.2f) control(left=%.2f,right=%.2f) "
                    "(trigger limit=%.1f, targets: left=%.1f right=%.1f).",
                    raw_left,
                    raw_right,
                    left,
                    right,
                    limit,
                    float(cfg.startup_guard_left_target),
                    float(cfg.startup_guard_right_target),
                )
                if seam_flipped:
                    logging.warning(
                        "Startup shoulder seam canonicalization applied to: %s",
                        ", ".join(seam_flipped),
                    )
            else:
                logging.warning(
                    "Startup shoulder guard active (manual mode). raw(left=%.2f,right=%.2f) control(left=%.2f,right=%.2f) "
                    "(trigger limit=%.1f). Move shoulders slightly forward; auto-resume when safe.",
                    raw_left,
                    raw_right,
                    left,
                    right,
                    limit,
                )
            warned_once = True
            last_log_t = now

        fail_open_after_s = float(cfg.startup_guard_fail_open_after_s)
        fail_open_min_steps = max(1, int(cfg.startup_guard_fail_open_min_steps))
        if (
            bool(cfg.startup_guard_fail_open_enabled)
            and fail_open_after_s > 0.0
            and (now - startup_t0) >= fail_open_after_s
            and startup_step_count >= fail_open_min_steps
        ):
            logging.warning(
                "Startup guard fail-open release after %.2fs (%d steps). "
                "Proceeding to policy with raw(left=%.2f,right=%.2f), control(left=%.2f,right=%.2f).",
                now - startup_t0,
                startup_step_count,
                raw_left,
                raw_right,
                left,
                right,
            )
            return True

        if cfg.startup_guard_timeout_s > 0 and now >= deadline_t:
            raise RuntimeError(
                "Startup guard timeout: shoulder_lift remained near seam boundary. "
                "Move shoulders slightly forward and rerun inference."
            )

        time.sleep(poll_dt)


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
    robot_config = SourcceyClientConfig(
        remote_ip=cfg.remote_ip,
        id=cfg.id,
        startup_shoulder_seam_filter_enabled=cfg.startup_shoulder_seam_filter_enabled,
        startup_shoulder_seam_filter_duration_s=cfg.startup_shoulder_seam_filter_duration_s,
        startup_shoulder_seam_abs_threshold=cfg.startup_shoulder_seam_abs_threshold,
        startup_shoulder_seam_required_plausible_frames=cfg.startup_shoulder_seam_required_plausible_frames,
        startup_shoulder_seam_max_delta_per_s=cfg.startup_shoulder_seam_max_delta_per_s,
        startup_shoulder_seam_delta_margin=cfg.startup_shoulder_seam_delta_margin,
    )
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
        if not _startup_shoulder_guard(robot, events, cfg):
            return

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
