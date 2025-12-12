import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.vulcan0.configuration_vulcan0 import Vulcan0Config
from lerobot.utils.constants import OBS_STATE


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

        # Input Layers
        if self.config.robot_state_feature:
            self.layer_state = nn.Linear(self.config.robot_state_feature.shape[0], self.config.model_dimension)

        # Output Layers
        self.layer_output = nn.Linear(self.config.model_dimension, self.config.action_feature.shape[0])

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:

        x = self.layer_state(batch[OBS_STATE])
        x = F.relu(x)
        x = self.layer_output(x)

        action = x
        return action, {}
