import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.vulcan0.configuration_vulcan0 import Vulcan0Config


class Vulcan0Policy(PreTrainedPolicy):
    config_class = Vulcan0Config
    name = "vulcan0"

    def __init__(self, config: Vulcan0Config):
        super().__init__(config)
        self.config = config

        self.model = Vulcan0Model(config)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        return self.model(batch)


class Vulcan0Model(nn.Module):
    def __init__(self, config: Vulcan0Config):
        super().__init__()
        self.config = config

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        pass
