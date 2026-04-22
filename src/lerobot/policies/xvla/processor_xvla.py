# ------------------------------------------------------------------------------
# Copyright 2025 The HuggingFace Inc. team and 2toINF (https://github.com/2toINF)
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
# ------------------------------------------------------------------------------

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    to_absolute_actions,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    IMAGENET_STATS,
    OBS_IMAGES,
    OBS_PREFIX,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_xvla import XVLAConfig
from .utils import rotate6d_to_axis_angle

_RANGE_CLAMP_WARNED_KEYS: set[str] = set()

def _get_pretrained_tokenizer_max_length(pretrained_path: str | Path | None) -> int | None:
    """Best-effort extraction of tokenizer max_length from a pretrained XVLA preprocessor config."""
    if pretrained_path is None:
        return None

    config_filename = "policy_preprocessor.json"
    try:
        path = Path(pretrained_path)
        if path.exists():
            config_path = path / config_filename if path.is_dir() else path
        else:
            config_path = Path(
                hf_hub_download(
                    repo_id=str(pretrained_path),
                    filename=config_filename,
                )
            )
    except Exception as exc:
        logging.debug("Failed to resolve pretrained processor config for '%s': %s", pretrained_path, exc)
        return None

    try:
        with config_path.open() as f:
            config = json.load(f)
    except Exception as exc:
        logging.debug("Failed to read pretrained processor config '%s': %s", config_path, exc)
        return None

    for step in config.get("steps", []):
        if step.get("registry_name") != "tokenizer_processor":
            continue
        max_length = step.get("config", {}).get("max_length")
        if isinstance(max_length, int) and max_length > 0:
            return max_length
    return None


def _warn_clamp_once(
    *,
    step_name: str,
    key: str,
    min_val: float,
    max_val: float,
    soft_eps: float,
    hard_eps: float,
) -> None:
    """Log one clamp warning per step/key per process to avoid spam during training."""
    warning_key = f"{step_name}:{key}"
    if warning_key in _RANGE_CLAMP_WARNED_KEYS:
        return

    _RANGE_CLAMP_WARNED_KEYS.add(warning_key)
    logging.warning(
        (
            "[%s] Clamping image '%s' to [0, 1] due to out-of-range values "
            "(min=%.4f, max=%.4f, soft_eps=%.4f, hard_eps=%.4f)."
        ),
        step_name,
        key,
        min_val,
        max_val,
        soft_eps,
        hard_eps,
    )


def _clamp_unit_range_or_raise(
    *,
    tensor: torch.Tensor,
    key: str,
    step_name: str,
    soft_eps: float,
    hard_eps: float,
) -> torch.Tensor:
    """Clamp minor/moderate range drift into [0, 1], fail fast for major violations."""
    min_val = tensor.min().item()
    max_val = tensor.max().item()

    if min_val >= 0.0 and max_val <= 1.0:
        return tensor

    if min_val < -hard_eps or max_val > 1.0 + hard_eps:
        raise ValueError(
            f"Image '{key}' has values outside hard clamp range around [0, 1]: "
            f"min={min_val:.4f}, max={max_val:.4f}, hard_eps={hard_eps:.4f}. "
            "This indicates materially invalid image values."
        )

    _warn_clamp_once(
        step_name=step_name,
        key=key,
        min_val=min_val,
        max_val=max_val,
        soft_eps=soft_eps,
        hard_eps=hard_eps,
    )
    return tensor.clamp(0.0, 1.0)


@ProcessorStepRegistry.register("xvla_delta_actions_processor")
@dataclass
class XVLARelativeActionsProcessorStep(RelativeActionsProcessorStep):
    """XVLA-specific relative action step carrying chunk-anchor state queue."""

    _absolute_anchor_states: deque[torch.Tensor] = field(default_factory=deque, init=False, repr=False)

    def prime_absolute_anchor_states(self, count: int) -> None:
        if count <= 0 or self._last_state is None:
            return
        for _ in range(count):
            self._absolute_anchor_states.append(self._last_state.detach().clone())

    def pop_absolute_anchor_state(self) -> torch.Tensor | None:
        if self._absolute_anchor_states:
            return self._absolute_anchor_states.popleft()
        return self._last_state

    def reset(self) -> None:
        self._last_state = None
        self._absolute_anchor_states.clear()


@ProcessorStepRegistry.register("xvla_absolute_actions_processor")
@dataclass
class XVLAAbsoluteActionsProcessorStep(AbsoluteActionsProcessorStep):
    """XVLA-specific absolute step that consumes chunk-anchor state queue."""

    relative_step: XVLARelativeActionsProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        if self.relative_step is None:
            raise RuntimeError(
                "XVLAAbsoluteActionsProcessorStep requires a paired XVLARelativeActionsProcessorStep."
            )

        state_for_absolute = self.relative_step.pop_absolute_anchor_state()
        if state_for_absolute is None:
            raise RuntimeError(
                "XVLAAbsoluteActionsProcessorStep requires cached state but none is available. "
                "Ensure preprocessor runs before postprocessor."
            )

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        mask = self.relative_step._build_mask(action.shape[-1])
        new_transition[TransitionKey.ACTION] = to_absolute_actions(action, state_for_absolute, mask)
        return new_transition


def prime_xvla_relative_anchor_states(
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    generated_count: int,
) -> bool:
    """Prime chunk-anchor states for XVLA postprocessing."""
    if generated_count <= 0:
        return False

    relative_step = next(
        (s for s in preprocessor.steps if isinstance(s, XVLARelativeActionsProcessorStep)), None
    )
    absolute_step = next(
        (s for s in postprocessor.steps if isinstance(s, XVLAAbsoluteActionsProcessorStep)), None
    )
    if relative_step is None or absolute_step is None:
        return False
    if not relative_step.enabled or not absolute_step.enabled:
        return False
    if absolute_step.relative_step is not relative_step:
        absolute_step.relative_step = relative_step

    relative_step.prime_absolute_anchor_states(generated_count)
    return True


def upgrade_xvla_relative_processors(
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
) -> None:
    """Upgrade deserialized generic relative/absolute steps to XVLA-specialized steps."""
    rel_idx, rel_step = next(
        (
            (i, s)
            for i, s in enumerate(preprocessor.steps)
            if isinstance(s, RelativeActionsProcessorStep)
        ),
        (None, None),
    )
    abs_idx, abs_step = next(
        (
            (i, s)
            for i, s in enumerate(postprocessor.steps)
            if isinstance(s, AbsoluteActionsProcessorStep)
        ),
        (None, None),
    )
    if rel_idx is None or rel_step is None or abs_idx is None or abs_step is None:
        return

    if isinstance(rel_step, XVLARelativeActionsProcessorStep):
        xvla_rel = rel_step
    else:
        xvla_rel = XVLARelativeActionsProcessorStep(
            enabled=rel_step.enabled,
            exclude_joints=list(rel_step.exclude_joints),
            action_names=list(rel_step.action_names) if rel_step.action_names is not None else None,
        )
        xvla_rel._last_state = rel_step._last_state
        if hasattr(rel_step, "_absolute_anchor_states"):
            for state in rel_step._absolute_anchor_states:
                xvla_rel._absolute_anchor_states.append(state)
        preprocessor.steps[rel_idx] = xvla_rel

    if isinstance(abs_step, XVLAAbsoluteActionsProcessorStep):
        abs_step.relative_step = xvla_rel
    else:
        abs_enabled = getattr(abs_step, "enabled", True)
        postprocessor.steps[abs_idx] = XVLAAbsoluteActionsProcessorStep(
            enabled=abs_enabled,
            relative_step=xvla_rel,
        )


def make_xvla_pre_post_processors(
    config: XVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Build the LeRobot processor pipelines for XVLA.
    """
    tokenizer_max_length = config.tokenizer_max_length
    pretrained_tokenizer_max_length = _get_pretrained_tokenizer_max_length(getattr(config, "pretrained_path", None))
    if (
        config.use_relative_actions
        and pretrained_tokenizer_max_length is not None
        and tokenizer_max_length > pretrained_tokenizer_max_length
    ):
        logging.warning(
            "XVLA relative-actions preprocessor uses tokenizer_max_length=%s from pretrained processors "
            "(requested=%s) to preserve sequence budget.",
            pretrained_tokenizer_max_length,
            tokenizer_max_length,
        )
        tokenizer_max_length = pretrained_tokenizer_max_length

    features = {**config.input_features, **config.output_features}
    relative_step = XVLARelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    # OpenPI-style relative-actions flow for actions:
    # raw -> relative -> normalize -> model -> unnormalize -> absolute
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_name,
            max_length=tokenizer_max_length,
            padding=config.pad_language_to,
            padding_side=config.tokenizer_padding_side,
        ),
        XVLAImageToFloatProcessorStep(),
        XVLAImageNetNormalizeProcessorStep(),
        XVLAAddDomainIdProcessorStep(),
        DeviceProcessorStep(device=config.device),
        relative_step,
        NormalizerProcessorStep(
            features=features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
    ]
    output_steps = [
        # XVLA postprocessing handles action-space transforms only; images are not postprocessed here.
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        XVLAAbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


# Custom XVLA processor steps
@dataclass
class LiberoProcessorStep(ObservationProcessorStep):
    """
    Processes LIBERO observations into the LeRobot format.

    This step handles the specific observation structure from LIBERO environments,
    which includes nested robot_state dictionaries and image observations.

    **State Processing:**
    -   Processes the `robot_state` dictionary which contains nested end-effector,
        gripper, and joint information.
    -   Extracts and concatenates:
        - End-effector position (3D)
        - End-effector quaternion converted to axis-angle (3D)
        - Gripper joint positions (2D)
    -   Maps the concatenated state to `"observation.state"`.

    **Image Processing:**
    -   Rotates images by 180 degrees by flipping both height and width dimensions.
    -   This accounts for the HuggingFaceVLA/libero camera orientation convention.
    """

    def _process_observation(self, observation):
        """
        Processes both image and robot_state observations from LIBERO.
        """
        processed_obs = observation.copy()
        for key in list(processed_obs.keys()):
            if key.startswith(f"{OBS_IMAGES}."):
                img = processed_obs[key]

                if key == f"{OBS_IMAGES}.image":
                    # Flip both H and W
                    img = torch.flip(img, dims=[2, 3])

                processed_obs[key] = img
        # Process robot_state into a flat state vector
        robot_state_str = OBS_PREFIX + "robot_state"
        if robot_state_str in processed_obs:
            robot_state = processed_obs.pop(robot_state_str)

            # Extract components
            eef_pos = robot_state["eef"]["pos"]  # (B, 3,)
            eef_mat = robot_state["eef"]["mat"]  # (B, 3, 3)
            eef_rot6d = self._mat_to_rotate6d(eef_mat)  # (B, 6)

            extra = torch.zeros((eef_pos.shape[0], 1), dtype=torch.float32, device=eef_pos.device)

            proprio_state = torch.cat((eef_pos, eef_rot6d, extra), dim=-1)  # (B, 10)
            state = torch.cat((proprio_state, torch.zeros_like(proprio_state)), dim=-1)  # (B, 20)
            # ensure float32
            state = state.float()
            if state.dim() == 1:
                state = state.unsqueeze(0)

            processed_obs[OBS_STATE] = state
        return processed_obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Transforms feature keys from the LIBERO format to the LeRobot standard.
        """
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {}

        # copy over non-STATE features
        for ft, feats in features.items():
            if ft != PipelineFeatureType.STATE:
                new_features[ft] = feats.copy()

        # rebuild STATE features
        state_feats = {}

        # add our new flattened state
        state_feats[OBS_STATE] = PolicyFeature(
            key=OBS_STATE,
            shape=(20,),
            dtype="float32",
        )

        new_features[PipelineFeatureType.STATE] = state_feats

        return new_features

    def _mat_to_rotate6d(self, rot_mats: torch.Tensor) -> torch.Tensor:
        """
        Convert batched rotation matrices (B, 3, 3) into 6D rotation representation (B, 6).

        Args:
            rot_mats (Tensor): Rotation matrices of shape (B, 3, 3)

        Returns:
            Tensor: 6D rotation representation, shape (B, 6)

        Raises:
            TypeError: if input is not a torch tensor
            ValueError: if shape is not (B, 3, 3)
        """

        if not isinstance(rot_mats, torch.Tensor):
            raise TypeError(f"mat_to_rot6d expects a torch.Tensor, got {type(rot_mats)}")

        if rot_mats.ndim != 3 or rot_mats.shape[1:] != (3, 3):
            raise ValueError(f"mat_to_rot6d expects shape (B, 3, 3), got {tuple(rot_mats.shape)}")

        rot_mats = rot_mats.to(torch.float32)

        col1 = rot_mats[:, :3, 0]  # (B, 3)
        col2 = rot_mats[:, :3, 1]  # (B, 3)

        rot6d = torch.cat([col1, col2], dim=-1)  # (B, 6)

        return rot6d

    def observation(self, observation):
        return self._process_observation(observation)


@dataclass
@ProcessorStepRegistry.register(name="xvla_image_scale")
class XVLAImageScaleProcessorStep(ProcessorStep):
    """Scale image observations by 255 to convert from [0, 1] to [0, 255] range.

    This processor step multiplies all image observations by 255, which is required
    for XVLA models that expect images in uint8-like range.

    Args:
        image_keys: List of observation keys that contain images to scale.
                   If None, will automatically detect keys starting with "observation.images."
    """

    image_keys: list[str] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Scale image observations by 255."""
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if obs is None:
            return new_transition

        # Make a copy of observations to avoid modifying the original
        obs = obs.copy()

        # Determine which keys to scale
        keys_to_scale = self.image_keys
        if keys_to_scale is None:
            # Auto-detect image keys
            keys_to_scale = [k for k in obs if k.startswith(OBS_IMAGES)]

        # Scale each image
        for key in keys_to_scale:
            if key in obs and isinstance(obs[key], torch.Tensor):
                obs[key] = obs[key] * 255

        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(self, features):
        """Image scaling doesn't change feature structure."""
        return features

    def get_config(self) -> dict[str, Any]:
        """Return serializable configuration."""
        return {
            "image_keys": self.image_keys,
        }


@dataclass
@ProcessorStepRegistry.register(name="xvla_image_to_float")
class XVLAImageToFloatProcessorStep(ProcessorStep):
    """Convert image observations from [0, 255] to [0, 1] range.

    This processor step divides image observations by 255 to convert from uint8-like
    range [0, 255] to float range [0, 1]. This is typically used when loading images
    that are stored as uint8 values.

    Args:
        image_keys: List of observation keys that contain images to convert.
                   If None, will automatically detect keys starting with "observation.images."
        validate_range: If True, validates that input values are in [0, 255] range (default: True)

    Raises:
        ValueError: If validate_range is True and image values are not in [0, 255] range.
    """

    image_keys: list[str] | None = None
    validate_range: bool = True
    soft_eps: float = 0.002
    hard_eps: float = 0.05

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Convert image observations from [0, 255] to [0, 1]."""
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if obs is None:
            return new_transition

        # Make a copy of observations to avoid modifying the original
        obs = obs.copy()

        # Determine which keys to convert
        keys_to_convert = self.image_keys
        if keys_to_convert is None:
            # Auto-detect image keys
            keys_to_convert = [k for k in obs if k.startswith(OBS_IMAGES)]

        # Convert each image
        for key in keys_to_convert:
            if key in obs and isinstance(obs[key], torch.Tensor):
                tensor = obs[key]

                min_val = tensor.min().item()
                max_val = tensor.max().item()
                uses_uint8_scale = (not tensor.is_floating_point()) or (tensor.is_floating_point() and max_val > 2.0)

                if uses_uint8_scale:
                    # Validate that values are in [0, 255] range if requested
                    if self.validate_range and (min_val < 0.0 or max_val > 255.0):
                        raise ValueError(
                            f"Image '{key}' has values outside [0, 255] range: "
                            f"min={min_val:.4f}, max={max_val:.4f}. "
                            f"Cannot convert to [0, 1] range."
                        )
                    converted = tensor.float() / 255.0
                else:
                    # Float tensors with max<=2 are treated as already in unit range.
                    converted = tensor.float()

                obs[key] = _clamp_unit_range_or_raise(
                    tensor=converted,
                    key=key,
                    step_name="xvla_image_to_float",
                    soft_eps=self.soft_eps,
                    hard_eps=self.hard_eps,
                )

        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(self, features):
        """Image conversion doesn't change feature structure."""
        return features

    def get_config(self) -> dict[str, Any]:
        """Return serializable configuration."""
        return {
            "image_keys": self.image_keys,
            "validate_range": self.validate_range,
            "soft_eps": self.soft_eps,
            "hard_eps": self.hard_eps,
        }


@dataclass
@ProcessorStepRegistry.register(name="xvla_imagenet_normalize")
class XVLAImageNetNormalizeProcessorStep(ProcessorStep):
    """Normalize image observations using ImageNet statistics.

    This processor step applies ImageNet normalization (mean and std) to image observations.
    It validates that input values are in the [0, 1] range before normalizing.

    The normalization formula is: (image - mean) / std

    Args:
        image_keys: List of observation keys that contain images to normalize.
                   If None, will automatically detect keys starting with "observation.images."

    Raises:
        ValueError: If image values are not in the [0, 1] range.
    """

    image_keys: list[str] | None = None
    soft_eps: float = 0.002
    hard_eps: float = 0.05

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Normalize image observations using ImageNet statistics."""
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if obs is None:
            return new_transition

        # Make a copy of observations to avoid modifying the original
        obs = obs.copy()

        # Determine which keys to normalize
        keys_to_normalize = self.image_keys
        if keys_to_normalize is None:
            # Auto-detect image keys
            keys_to_normalize = [k for k in obs if k.startswith(OBS_IMAGES)]

        # Normalize each image
        for key in keys_to_normalize:
            if key in obs and isinstance(obs[key], torch.Tensor):
                tensor = obs[key]
                tensor = _clamp_unit_range_or_raise(
                    tensor=tensor,
                    key=key,
                    step_name="xvla_imagenet_normalize",
                    soft_eps=self.soft_eps,
                    hard_eps=self.hard_eps,
                )

                # Apply ImageNet normalization
                mean = torch.tensor(IMAGENET_STATS["mean"], device=tensor.device, dtype=tensor.dtype)
                std = torch.tensor(IMAGENET_STATS["std"], device=tensor.device, dtype=tensor.dtype)

                # Expand mean/std to match tensor dims (e.g., BCHW or BNCHW)
                while mean.dim() < tensor.dim():
                    mean = mean.unsqueeze(0)
                    std = std.unsqueeze(0)

                # Normalize: (image - mean) / std
                obs[key] = (tensor - mean) / std

        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(self, features):
        """ImageNet normalization doesn't change feature structure."""
        return features

    def get_config(self) -> dict[str, Any]:
        """Return serializable configuration."""
        return {
            "image_keys": self.image_keys,
            "soft_eps": self.soft_eps,
            "hard_eps": self.hard_eps,
        }


@dataclass
@ProcessorStepRegistry.register(name="xvla_add_domain_id")
class XVLAAddDomainIdProcessorStep(ProcessorStep):
    """Add domain_id to complementary data.

    This processor step adds a domain_id tensor to the complementary data,
    which is used by XVLA to identify different robot embodiments or task domains.

    Args:
        domain_id: The domain ID to add (default: 3)
    """

    domain_id: int = 0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Add domain_id to complementary data."""
        new_transition = transition.copy()
        comp = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        comp = {} if comp is None else comp.copy()

        # Infer batch size from observation tensors
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        batch_size = 1
        if obs:
            for v in obs.values():
                if isinstance(v, torch.Tensor):
                    batch_size = v.shape[0]
                    break

        # Add domain_id tensor
        comp["domain_id"] = torch.tensor([int(self.domain_id)] * batch_size, dtype=torch.long)

        new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return new_transition

    def transform_features(self, features):
        """Domain ID addition doesn't change feature structure."""
        return features

    def get_config(self) -> dict[str, Any]:
        """Return serializable configuration."""
        return {
            "domain_id": self.domain_id,
        }


@dataclass
@ProcessorStepRegistry.register(name="xvla_rotation_6d_to_axis_angle")
class XVLARotation6DToAxisAngleProcessorStep(ProcessorStep):
    """Convert 6D rotation representation to axis-angle and reorganize action dimensions.

    This processor step takes actions with 6D rotation representation and converts them to
    axis-angle representation, reorganizing the action dimensions as:
    - action[:, :3] -> target_eef (end-effector position)
    - action[:, 3:9] -> 6D rotation (converted to axis-angle, 3D)
    - action[:, 9:10] -> gripper action

    Final output: [target_eef (3), axis_angle (3), gripper (1)] = 7D action

    Args:
        expected_action_dim: Expected input action dimension (default: 10, supports 6D rotation + extras)
    """

    expected_action_dim: int = 10

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Convert 6D rotation to axis-angle in action."""
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)

        if action is None or not isinstance(action, torch.Tensor):
            return new_transition

        # Convert to numpy for processing
        device = action.device
        dtype = action.dtype
        action_np = action.cpu().numpy()

        # Extract components
        # action shape: (B, D) where D >= 10
        target_eef = action_np[:, :3]  # (B, 3)
        rotation_6d = action_np[:, 3:9]  # (B, 6)
        target_act = action_np[:, 9:10]  # (B, 1)

        # Convert 6D rotation to axis-angle
        target_axis = rotate6d_to_axis_angle(rotation_6d)  # (B, 3)

        # Concatenate: [eef (3), axis_angle (3), gripper (1)] = 7D
        action_np = np.concatenate([target_eef, target_axis, target_act], axis=-1)

        # Convert gripper action to -1 or 1
        action_np[:, -1] = np.where(action_np[:, -1] > 0.5, 1.0, -1.0)

        # Convert back to tensor
        action = torch.from_numpy(action_np).to(device=device, dtype=dtype)

        new_transition[TransitionKey.ACTION] = action
        return new_transition

    def transform_features(self, features):
        """Rotation conversion changes action dimension from 10 to 7."""
        # Note: This is a simplified version. In practice, you might want to
        # update the action feature shape in the features dict.
        return features

    def get_config(self) -> dict[str, Any]:
        """Return serializable configuration."""
        return {
            "expected_action_dim": self.expected_action_dim,
        }


def make_xvla_libero_pre_post_processors() -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Build the LeRobot processor pipelines for XVLA with LIBERO environment.
    """
    pre_processor_steps: list[ProcessorStep] = []
    post_processor_steps: list[ProcessorStep] = []
    pre_processor_steps.extend(
        [LiberoProcessorStep(), XVLAImageNetNormalizeProcessorStep(), XVLAAddDomainIdProcessorStep()]
    )
    post_processor_steps.extend([XVLARotation6DToAxisAngleProcessorStep()])
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=pre_processor_steps,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=post_processor_steps,
        ),
    )
