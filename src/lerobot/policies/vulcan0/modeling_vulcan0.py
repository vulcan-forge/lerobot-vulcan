import torch
import torch.nn as nn

from lerobot.policies.vulcan0.configuration_vulcan0 import Vulcan0Config

class Vulcan0Model(nn.Module):
    def __init__(self, config: Vulcan0Config):
        pass

class Vulcan0Encoder(nn.Module):
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
