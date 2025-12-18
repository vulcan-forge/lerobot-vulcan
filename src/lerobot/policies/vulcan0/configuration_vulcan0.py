

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig

@PreTrainedConfig.register_subclass("vulcan0")
@dataclass
class Vulcan0Config(PreTrainedConfig):
    """Configuration for the Vulcan 0 policy."""

    # Input / output structure.
    num_obs_steps: int = 1
    chunk_size: int = 100
    num_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Model dimensions
    model_dimension: int = 512
    num_heads: int = 8
    feed_forward_dim: int = 3200
    dropout: float = 0.1

    # VAE
    latent_dim: int = 32
    num_vae_encoder_layers: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 1

    # Optimizer training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False

