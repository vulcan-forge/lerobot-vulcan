

from dataclasses import dataclass
from lerobot.configs.policies import PreTrainedConfig

@PreTrainedConfig.register_subclass("vulcan0")
@dataclass
class Vulcan0Config(PreTrainedConfig):

    model_dimension: int = 512
    num_heads: int = 8
    feedforward_dim: int = 3200
    dropout: float = 0.1
