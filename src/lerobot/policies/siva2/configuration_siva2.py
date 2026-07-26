# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Configuration for SIVA2, the memory-aware and steerable SIVA successor.

SIVA2 deliberately has its own registered policy type.  SIVA checkpoints and
defaults remain untouched, which makes SIVA versus SIVA2 a clean experiment.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("siva2")
@dataclass
class SIVA2Config(XVLAConfig):
    """Typed-context, temporal-memory, advantage-conditioned SIVA policy."""

    action_mode: str = "auto"
    n_obs_steps: int = 4

    # A fixed, typed context replaces one undifferentiated bottleneck.  The
    # default has 46 tokens, still far shorter than the Florence sequence.
    hidden_size: int = 512
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 3.0
    scene_slots: int = 16
    memory_slots: int = 12
    goal_slots: int = 6
    subgoal_slots: int = 6
    slot_depth: int = 2
    fusion_depth: int = 2
    memory_depth: int = 2
    memory_spatial_queries: int = 2
    max_history_frames: int = 8
    vision_token_dropout: float = 0.1
    condition_dropout: float = 0.15
    subgoal_dropout: float = 0.75

    # Prefix for optional visual goal features.  These images are encoded as
    # desired future states and are never mixed into the current camera list.
    subgoal_feature_prefix: str = "observation.subgoal"

    # Factored structured source: geometry and execution strategy are separate
    # decisions so mixed-quality data need not corrupt a single motion mode.
    num_motion_primitives: int = 8
    num_strategies: int = 4
    num_control_points: int = 6
    domain_adapter_rank: int = 4
    prior_init_scale: float = 0.02
    prior_noise_scale: float = 0.15
    noise_temporal_correlation: float = 0.8
    routing_temperature: float = 1.0
    num_denoising_steps: int = 3
    deterministic_inference: bool = True

    # Batch keys form the explicit data contract for heterogeneous experience.
    quality_key: str = "siva2.quality"
    speed_key: str = "siva2.speed"
    mistake_key: str = "siva2.mistake"
    advantage_key: str = "siva2.advantage"
    progress_key: str = "siva2.progress"
    desired_progress_key: str = "siva2.desired_progress"
    control_mode_key: str = "siva2.control_mode"
    data_source_key: str = "siva2.data_source"
    success_key: str = "siva2.success"
    time_to_completion_key: str = "siva2.time_to_completion"
    intervention_key: str = "siva2.intervention"
    num_control_modes: int = 3  # unknown, joint, end-effector
    num_data_sources: int = 8
    speed_normalizer: float = 2_000.0
    time_normalizer: float = 2_000.0

    # Existing demonstration-only datasets remain trainable.  Autonomous data
    # should always provide explicit values instead of relying on these defaults.
    assume_demonstration_if_unlabeled: bool = True
    default_quality: float = 5.0
    default_advantage: float = 1.0
    inference_quality: float = 5.0
    inference_advantage: float = 1.0

    flow_loss_weight: float = 1.0
    endpoint_loss_weight: float = 0.1
    prior_loss_weight: float = 0.25
    primitive_router_loss_weight: float = 0.1
    strategy_router_loss_weight: float = 0.1
    router_balance_loss_weight: float = 0.01
    smoothness_loss_weight: float = 0.001
    value_success_loss_weight: float = 0.1
    value_progress_loss_weight: float = 0.1
    value_time_loss_weight: float = 0.05
    value_intervention_loss_weight: float = 0.05

    freeze_vlm: bool = False
    # Optional training-only cache for all frozen Florence outputs consumed by
    # the typed scene, temporal-memory, goal, and subgoal branches.
    cache_florence_features: bool = False
    florence_cache_path: str | None = None
    xvla_init_source: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("`hidden_size` must be divisible by `num_heads`.")
        slot_counts = (self.scene_slots, self.memory_slots, self.goal_slots, self.subgoal_slots)
        if any(count <= 0 for count in slot_counts):
            raise ValueError("All SIVA2 typed slot counts must be positive.")
        if min(self.depth, self.slot_depth, self.fusion_depth, self.memory_depth) <= 0:
            raise ValueError("SIVA2 transformer depths must be positive.")
        if self.memory_spatial_queries <= 0 or self.max_history_frames <= 0:
            raise ValueError("SIVA2 memory sizes must be positive.")
        if self.num_motion_primitives < 2 or self.num_strategies < 2:
            raise ValueError("SIVA2 requires at least two motion primitives and strategies.")
        if not 2 <= self.num_control_points <= self.chunk_size:
            raise ValueError("`num_control_points` must be in [2, chunk_size].")
        for name, value in (
            ("vision_token_dropout", self.vision_token_dropout),
            ("condition_dropout", self.condition_dropout),
            ("subgoal_dropout", self.subgoal_dropout),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"`{name}` must be in [0, 1).")
        if not 0.0 <= self.noise_temporal_correlation <= 1.0:
            raise ValueError("`noise_temporal_correlation` must be in [0, 1].")
        if self.domain_adapter_rank <= 0 or self.prior_noise_scale <= 0.0:
            raise ValueError("SIVA2 adapter rank and prior noise scale must be positive.")
        if self.routing_temperature <= 0.0:
            raise ValueError("`routing_temperature` must be positive.")
        if self.num_control_modes <= 0 or self.num_data_sources <= 0:
            raise ValueError("Condition category counts must be positive.")
        if self.speed_normalizer <= 0.0 or self.time_normalizer <= 0.0:
            raise ValueError("Condition normalizers must be positive.")
        if self.cache_florence_features:
            if not self.florence_cache_path:
                raise ValueError(
                    "`florence_cache_path` is required when `cache_florence_features=True`."
                )
            if not self.freeze_vlm:
                raise ValueError("SIVA2 Florence caching requires `freeze_vlm=True`.")

    def validate_features(self) -> None:
        """Validate current cameras separately from optional subgoal images."""
        scene_features = {
            key: feature
            for key, feature in self.image_features.items()
            if not key.startswith(self.subgoal_feature_prefix)
        }
        if not scene_features:
            raise ValueError("SIVA2 requires at least one current-scene visual feature.")
        if self.use_proprio and self.robot_state_feature is None:
            raise ValueError("`use_proprio=True` requires a proprioceptive state feature.")
        if self.num_image_views is None:
            self.num_image_views = len(scene_features) + self.empty_cameras
        else:
            self.num_image_views = max(self.num_image_views, len(scene_features) + self.empty_cameras)
        if self.empty_cameras > 0:
            height, width = self.resize_imgs_with_padding or (480, 640)
            for index in range(self.empty_cameras):
                key = f"{OBS_IMAGES}.empty_camera_{index}"
                if key not in self.input_features:
                    self.input_features[key] = PolicyFeature(
                        type=FeatureType.VISUAL, shape=(3, height, width)
                    )

    @property
    def observation_delta_indices(self) -> list[int]:
        # The current frame is last; earlier entries feed the temporal memory.
        return list(range(1 - self.n_obs_steps, 1))
