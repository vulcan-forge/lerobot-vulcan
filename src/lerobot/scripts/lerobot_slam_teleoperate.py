#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Teleoperation entrypoint with SLAM enabled through a dedicated boundary module."""

import logging
from numbers import Real
from dataclasses import asdict, dataclass, field
from pprint import pformat

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import make_default_processors
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.scripts.lerobot_teleoperate import teleop_loop
from lerobot.slam.integrations.rollout import (
    RolloutSlamConfig,
    SlamAwareRobotProxy,
    build_rollout_slam_session,
)
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, shutdown_rerun


class _HybridTeleoperator:
    """Combines arm-leader teleop with Sourccey keyboard base commands."""

    def __init__(self, primary: Teleoperator, keyboard_input: Teleoperator | None = None, robot: Robot | None = None) -> None:
        self.primary = primary
        self.keyboard_input = keyboard_input
        self.robot = robot
        self._prev_pressed_keys: set[str] = set()

    def connect(self) -> None:
        self.primary.connect()
        if self.keyboard_input is not None:
            try:
                self.keyboard_input.connect()
            except Exception as exc:
                logging.warning("Keyboard overlay connect failed, disabling overlay: %s", exc)
                self.keyboard_input = None

    def disconnect(self) -> None:
        if self.keyboard_input is not None:
            try:
                self.keyboard_input.disconnect()
            except Exception:
                pass
        self.primary.disconnect()

    def get_action(self) -> dict[str, float]:
        action = dict(self.primary.get_action())
        if self.keyboard_input is not None:
            try:
                pressed_dict = self.keyboard_input.get_action()
            except Exception as exc:
                logging.warning("Keyboard overlay read failed, disabling overlay: %s", exc)
                self.keyboard_input = None
                pressed_dict = {}

            pressed_keys = set(pressed_dict.keys())

            # Edge-triggered callbacks (e.g., speed level up/down) on first key-down.
            if self.robot is not None and hasattr(self.robot, "on_key_down"):
                for key in pressed_keys - self._prev_pressed_keys:
                    try:
                        self.robot.on_key_down(key)
                    except Exception:
                        pass
            self._prev_pressed_keys = pressed_keys

            if self.robot is not None and hasattr(self.robot, "_from_keyboard_to_base_action"):
                try:
                    base_action = self.robot._from_keyboard_to_base_action(list(pressed_keys))
                    action.update(base_action)
                except Exception as exc:
                    logging.warning("Keyboard-to-base mapping failed: %s", exc)
        return action

    def send_feedback(self, feedback: dict) -> None:
        self.primary.send_feedback(feedback)

    @property
    def action_features(self) -> dict:
        return self.primary.action_features

    @property
    def name(self) -> str:
        if self.keyboard_input is None:
            return getattr(self.primary, "name", "teleop")
        return f"{getattr(self.primary, 'name', 'teleop')}+keyboard"


class _RestingActionTeleoperator:
    """Synthetic teleoperator that emits a constant safe hold action."""

    def __init__(self, action: dict[str, float]) -> None:
        self._action = dict(action)

    def connect(self) -> None:
        return

    def disconnect(self) -> None:
        return

    def get_action(self) -> dict[str, float]:
        return dict(self._action)

    def send_feedback(self, _obs: dict) -> None:
        return


@dataclass
class TeleoperateSlamConfig:
    """SLAM-only config for the dedicated SLAM teleoperate entrypoint."""

    enabled: bool = False
    backend: str = "orbslam3"
    source_mode: str = "client_observation"
    stereo_left_key: str = "front_left"
    stereo_right_key: str = "front_right"
    target_hz: float = 15.0
    healthy_timeout_s: float = 0.75
    stale_timeout_s: float = 2.0
    map_save_enabled: bool = False

    def __post_init__(self):
        if self.source_mode not in ("client_observation",):
            raise ValueError(
                f"Unsupported slam.source_mode '{self.source_mode}'. Expected 'client_observation'."
            )
        if self.target_hz <= 0:
            raise ValueError("--slam.target_hz must be > 0")
        if self.healthy_timeout_s <= 0:
            raise ValueError("--slam.healthy_timeout_s must be > 0")
        if self.stale_timeout_s <= 0:
            raise ValueError("--slam.stale_timeout_s must be > 0")
        if self.stale_timeout_s < self.healthy_timeout_s:
            raise ValueError("--slam.stale_timeout_s must be >= --slam.healthy_timeout_s")


@dataclass
class SlamTeleoperateConfig:
    """Teleoperate config plus optional SLAM block."""

    teleop: TeleoperatorConfig
    robot: RobotConfig
    fps: int = 60
    teleop_time_s: float | None = None
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    slam: TeleoperateSlamConfig = field(default_factory=TeleoperateSlamConfig)


def _to_scalar(value: object) -> float | None:
    if isinstance(value, Real):
        return float(value)
    if hasattr(value, "item"):
        try:
            item = value.item()
        except Exception:
            return None
        if isinstance(item, Real):
            return float(item)
    return None


def _build_resting_action(robot: Robot, observation: dict[str, object]) -> dict[str, float]:
    """Build a conservative hold action from current observation/action keys."""
    action: dict[str, float] = {}
    for key in robot.action_features:
        observed = _to_scalar(observation.get(key))
        if observed is not None:
            action[key] = observed
        elif key.endswith(".vel"):
            action[key] = 0.0
        else:
            action[key] = 0.0
    return action


@parser.wrap()
def teleoperate(cfg: SlamTeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)
    teleop_type = getattr(cfg.teleop, "type", "")
    keyboard_overlay = None
    if teleop_type in ("bi_sourccey_leader", "sourccey_leader"):
        try:
            keyboard_overlay = make_teleoperator_from_config(KeyboardTeleopConfig())
            logging.info("Sourccey keyboard overlay enabled (W/S/A/D, Z/X, Q/E, R/F, N/M).")
        except Exception as exc:
            logging.warning("Sourccey keyboard overlay unavailable: %s", exc)
            keyboard_overlay = None
    if keyboard_overlay is not None:
        teleop = _HybridTeleoperator(teleop, keyboard_overlay, robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    slam_session = None
    if cfg.slam.enabled:
        slam_cfg = RolloutSlamConfig(
            enabled=True,
            backend=cfg.slam.backend,
            source_mode=cfg.slam.source_mode,
            stereo_left_key=cfg.slam.stereo_left_key,
            stereo_right_key=cfg.slam.stereo_right_key,
            target_hz=cfg.slam.target_hz,
            healthy_timeout_s=cfg.slam.healthy_timeout_s,
            stale_timeout_s=cfg.slam.stale_timeout_s,
            map_save_enabled=cfg.slam.map_save_enabled,
        )
        slam_session = build_rollout_slam_session(slam_cfg)
        robot = SlamAwareRobotProxy(robot, slam_session)
        logging.info(
            "Teleop SLAM enabled (backend=%s, stereo=(%s,%s), target_hz=%.1f)",
            cfg.slam.backend,
            cfg.slam.stereo_left_key,
            cfg.slam.stereo_right_key,
            cfg.slam.target_hz,
        )

    robot.connect()
    using_rest_fallback = False
    try:
        teleop.connect()
    except Exception as exc:
        obs_for_rest = robot.get_observation()
        rest_action = _build_resting_action(robot, obs_for_rest)
        teleop = _HybridTeleoperator(_RestingActionTeleoperator(rest_action), keyboard_overlay, robot)
        using_rest_fallback = True
        logging.warning(
            "Teleoperator unavailable (%s). Falling back to resting arm action mode.",
            exc,
        )
        teleop.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
        )
        if using_rest_fallback:
            logging.info("Resting-action fallback mode active for this session.")
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            shutdown_rerun()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
