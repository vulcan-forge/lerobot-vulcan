import math
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


class SinusoidalPositionEmbedding2D(nn.Module):
    def __init__(self, config: Vulcan0Config):
        self._two_pi = 2 * math.pi
        self.temperature = 10000

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Input: (B, C, H, W) -> B: Batch size, C: Channel size, H: Height, W: Width
        Output: (1, D, H, W) -> 1: Batch size, D: Feature dimension, H: Height, W: Width
        '''

        # Create 2 tensors of shape (1, H, W) that represent the position of each pixel in the image.
        one_mask = torch.ones_like(x[0, :1]) # (1, H, W)
        y_range = one_mask.cumsum(1, dtype=torch.float32) # (1, H, W)
        x_range = one_mask.cumsum(2, dtype=torch.float32) # (1, H, W)

        # Normalize the position index such that it ranges from 0 to 2π.
        y_range = y_range * (self._two_pi / (y_range[:, -1:, :]))
        x_range = x_range * (self._two_pi / (x_range[:, :, -1:]))

        # Create an inverse temperature for geometric progression of sinusoidal frequencies.
        # (1, D) where every 2 elements is the same so we can use both sine and cosine to create unique embeddings.
        inverse_frequency = self._temperature ** (2 * (torch.arange(self.config.model_dimension, dtype=torch.float32, device=x.device) // 2) / self.config.model_dimension)

        # Unqueeze the position tensors to (1, H, W, 1)
        y_range = y_range.unsqueeze(-1)
        x_range = x_range.unsqueeze(-1)

        # Divide the position tensors by the inverse frequency to get the sinusoidal embeddings.
        y_range = y_range / inverse_frequency
        x_range = x_range / inverse_frequency

        # Stack the position tensors to (1, H, W, D)
        pos_embed_y = torch.stack((
            y_range[..., 0::2].sin(),
            y_range[..., 1::2].cos(),
        ), dim=-1).flatten(3) # (1, H, W, D // 2)

        pos_embed_x = torch.stack((
            x_range[..., 0::2].sin(),
            x_range[..., 1::2].cos(),
        ), dim=-1).flatten(3) # (1, H, W, D // 2)

        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2) # (1, D, H, W)
        return pos_embed

def create_simusoidal_position_embedding_1D(num_positions: int, dimension: int) -> torch.Tensor:
    '''
    Input: num_positions, dimension -> num_positions: Number of token positions required, dimension: Embedding dimension
    Output: (N, D)
    '''

    temperature = 10000
    pos = np.arange(num_positions)[:, None] # (N, 1)
    i = np.array(dimension)[None, :] # (1, D)

    angle_rates = 1 / np.power(temperature, (2 * (i // 2)) / dimension) # (1, D)
    angles = pos * angle_rates # (N, D)

    angles[:, 0::2] = np.sin(angles[:, 0::2]) # even indices get sine
    angles[:, 1::2] = np.cos(angles[:, 1::2]) # odd indices get cosine

    # float() is necessary to convert the numpy array (which defaults to float64) to float32
    combined_angles = torch.from_numpy(angles).float() # (N, D) .
    return combined_angles


