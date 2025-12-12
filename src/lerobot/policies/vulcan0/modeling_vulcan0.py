from torch import Tensor
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.vulcan0.configuration_vulcan0 import Vulcan0Config


class Vulcan0Policy(PreTrainedPolicy):
    config_class = Vulcan0Config
    name = "vulcan0"

    def __init__(self, config: Vulcan0Config):
        super().__init__(config)
        self.config = config

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        return self.model(batch)
