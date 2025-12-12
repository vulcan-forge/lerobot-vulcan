import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.vulcan0.configuration_vulcan0 import Vulcan0Config
from lerobot.utils.constants import ACTION, OBS_STATE


class Vulcan0Policy(PreTrainedPolicy):
    config_class = Vulcan0Config
    name = "vulcan0"

    def __init__(self, config: Vulcan0Config):
        super().__init__(config)
        self.config = config

        self.model = Vulcan0Model(config)

        # No internal recurrent state for now, but keep reset for API.
        self.reset()

    # ---- Required PreTrainedPolicy API ----
    def get_optim_params(self) -> dict:
        # Simple case: single param group, use config’s optimizer settings.
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]

    def reset(self):
        # No stateful caches yet; this is a no-op placeholder.
        pass

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """
        For compatibility with chunk-based policies, we return a
        (batch, 1, action_dim) tensor: a "chunk" of length 1.
        """
        self.eval()
        actions, _ = self.model(batch)  # (B, action_dim)
        return actions.unsqueeze(1)     # (B, 1, action_dim)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Select a single action for env rollout.
        """
        self.eval()
        actions, _ = self.model(batch)  # (B, action_dim)
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        # Predicted actions from the model: (B, action_dim)
        actions_hat, _ = self.model(batch)

        # Ground-truth actions from dataset: (B, 1, action_dim) -> (B, action_dim)
        target = batch[ACTION]
        if target.ndim == 3 and target.shape[1] == 1:
            target = target.squeeze(1)

        # Scalar loss
        loss = F.mse_loss(actions_hat, target)

        loss_dict = {"mse_loss": loss.item()}
        return loss, loss_dict


class Vulcan0Model(nn.Module):
    def __init__(self, config: Vulcan0Config):
        super().__init__()
        self.config = config

        # Input Layers
        if self.config.robot_state_feature:
            self.layer_state = nn.Linear(
                self.config.robot_state_feature.shape[0],
                self.config.model_dimension,
            )

        # Output Layers
        self.layer_output = nn.Linear(
            self.config.model_dimension,
            self.config.action_feature.shape[0],
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        x = self.layer_state(batch[OBS_STATE])
        x = F.relu(x)
        x = self.layer_output(x)

        action = x  # (B, action_dim)
        return action, {}
