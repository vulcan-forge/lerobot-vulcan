from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("vulcan0")
@dataclass
class Vulcan0Config(PreTrainedConfig):
    """Configuration for the Vulcan 0 policy."""

    # Model dimensions
    model_dimension: int = 512

    # How to normalize inputs/outputs (state + action only for this policy)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Training preset hyperparameters
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        super().__post_init__()

    # ---- Required abstract methods ----

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self):
        # No scheduler for now
        return None

    def validate_features(self) -> None:
        # Model uses robot state -> action only
        if self.robot_state_feature is None:
            raise ValueError(
                "Vulcan0 requires a robot state feature with key 'observation.state'."
            )
        if self.action_feature is None:
            raise ValueError(
                "Vulcan0 requires an action feature with key 'action'."
            )

    @property
    def observation_delta_indices(self) -> list | None:
        # No temporal deltas for now
        return None

    @property
    def action_delta_indices(self) -> list | None:
        # Single-step action prediction
        return [0]

    @property
    def reward_delta_indices(self) -> list | None:
        # No reward prediction
        return None
