from dataclasses import dataclass, field
import logging
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
    # Eval-only startup guard to avoid shoulder-lift seam startup flails.
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
    # Startup seam sign canonicalization in client observation/action.
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


def _startup_shoulder_guard(robot: SourcceyClient, events: dict[str, bool], cfg: SourcceyEvaluateConfig) -> bool:
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
        if events.get("stop_recording", False):
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
            # Legacy behavior: untorque and wait for manual reposition.
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
                "Move shoulders slightly forward and rerun evaluate."
            )

        time.sleep(poll_dt)

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
        startup_shoulder_seam_filter_enabled=cfg.startup_shoulder_seam_filter_enabled,
        startup_shoulder_seam_filter_duration_s=cfg.startup_shoulder_seam_filter_duration_s,
        startup_shoulder_seam_abs_threshold=cfg.startup_shoulder_seam_abs_threshold,
        startup_shoulder_seam_required_plausible_frames=cfg.startup_shoulder_seam_required_plausible_frames,
        startup_shoulder_seam_max_delta_per_s=cfg.startup_shoulder_seam_max_delta_per_s,
        startup_shoulder_seam_delta_margin=cfg.startup_shoulder_seam_delta_margin,
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
        # Use policy checkpoint stats. A freshly created eval dataset does not
        # provide representative normalization stats for inference.
        dataset_stats=None,
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

        # Eval-only preflight: block policy until shoulder_lift is out of seam-risk startup range.
        if not _startup_shoulder_guard(robot, events, cfg):
            break

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
