import math
import numpy as np

import torch
import torch.nn as nn

from lerobot.policies.vulcan0.configuration_vulcan0 import Vulcan0Config

class Vulcan0Model(nn.Module):
    def __init__(self, config: Vulcan0Config):
        pass

class Vulcan0Encoder(nn.Module):
    def __init__(self, config: Vulcan0Config):
        super().__init__()

        self.layers = nn.ModuleList([Vulcan0EncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.norm = nn.LayerNorm(config.model_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class Vulcan0EncoderLayer(nn.Module):
    def __init__(self, config: Vulcan0Config):
        super().__init__()

        self.multi_head_self_attention = nn.MultiheadAttention(embd_dim=config.model_dimension, num_heads=config.num_heads)
        self.multi_layer_perceptron = nn.Sequential(
            nn.Linear(config.model_dimension, config.feedforward_dim),
            nn.GELU(),
            nn.Linear(config.feedforward_dim, config.model_dimension)
        )

        self.layer_norm1 = nn.LayerNorm(config.model_dimension)
        self.layer_norm2 = nn.LayerNorm(config.model_dimension)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Input: (S, B, D) -> S: Sequence length, B: Batch size, D: Feature dimension
        Output: (S, B, D) -> S: Sequence length, B: Batch size, D: Feature dimension
        '''

        # 1. Multi-head Self-Attention
        x_norm1 = self.layer_norm1(x)
        attention_output, _ = self.multi_head_self_attention(x_norm1, x_norm1, x_norm1, need_weights=False)
        x = x + nn.Dropout(attention_output)

        # 2. Feed-Forward Layer
        x_norm2 = self.layer_norm2(x)
        mlp_output = self.multi_layer_perceptron(x_norm2)
        x = x + nn.Dropout(mlp_output)

        return x


class Vulcan0Decoder(nn.Module):
    def __init__(self, config: Vulcan0Config):
        super().__init__()

        self.layers = nn.ModuleList([Vulcan0DecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.norm = nn.LayerNorm(config.model_dimension)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_out)
        x = self.norm(x)
        return x

class Vulcan0DecoderLayer(nn.Module):
    def __init__(self, config: Vulcan0Config):
        super().__init__()

        self.multi_head_self_attention = nn.MultiheadAttention(embd_dim=config.model_dimension, num_heads=config.num_heads)
        self.cross_attention = nn.MultiheadAttention(embd_dim=config.model_dimension, num_heads=config.num_heads)
        self.multi_layer_perceptron = nn.Sequential(
            nn.Linear(config.model_dimension, config.feedforward_dim),
            nn.GELU(),
            nn.Linear(config.feedforward_dim, config.model_dimension)
        )
        self.layer_norm1 = nn.LayerNorm(config.model_dimension)
        self.layer_norm2 = nn.LayerNorm(config.model_dimension)
        self.layer_norm3 = nn.LayerNorm(self.model_dimension)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        '''
        Input: (S, B, D) -> S: Sequence length, B: Batch size, D: Feature dimension
        Encoder Output: (S, B, D) -> S: Sequence length, B: Batch size, D: Feature dimension
        Output: (S, B, D) -> S: Sequence length, B: Batch size, D: Action dimension
        '''

        # 1. Multi-head Self-Attention
        x_norm1 = self.layer_norm1(x)
        attention_output, _ = self.multi_head_self_attention(x_norm1, x_norm1, x_norm1, need_weights=False)
        x = x + nn.Dropout(attention_output)

        # 2. Cross-Attention
        x_norm2 = self.layer_norm2(x)
        cross_attention_output, _ = self.cross_attention(x_norm2, encoder_out, encoder_out, need_weights=False)
        x = x + nn.Dropout(cross_attention_output)

        # 3. Feed-Forward Layer
        x_norm3 = self.layer_norm3(x)
        mlp_output = self.multi_layer_perceptron(x_norm3)
        x = x + nn.Dropout(mlp_output)

        return x


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

- float() is necessary to convert the numpy array to a torch tensor
    combined_angles = torch.from_numpy(angles).float() # (N, D) .
    return combined_angles


