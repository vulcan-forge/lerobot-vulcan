import torch
import torch.nn as nn

from lerobot.policies.vulcan0.configuration_vulcan0 import Vulcan0Config

class Vulcan0Model(nn.Module):
    def __init__(self, config: Vulcan0Config):
        super().__init__()
        self.config = config

class Vulcan0Encoder(nn.Module):
    def __init__(self, config: Vulcan0Config):
        super().__init__()

        self.model_dimension = 512
        self.num_heads = 8
        self.dim_feedforward = 3200
        self.dropout = 0.1

        self.multi_head_self_attention = nn.MultiheadAttention(embed_dim=self.model_dimension, num_heads=self.num_heads, dropout=self.dropout)
        self.multi_layer_perceptron = nn.Sequential(
            nn.Linear(self.model_dimension, self.dim_feedforward),
            nn.GELU(),
            nn.Linear(self.dim_feedforward, self.model_dimension),
            nn.Dropout(self.dropout)
        )

        self.layer_norm1 = nn.LayerNorm(self.model_dimension)
        self.layer_norm2 = nn.LayerNorm(self.model_dimension)
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        '''
        Input: (S, B, D): S = Sequence Length, B = Batch Size, D = Model Dimension
        Output: (S, B, D): S = Sequence Length, B = Batch Size, D = Model Dimension
        '''

        # Pre norm so we normalize (mean = 0, std = 1) before the attention layer
        x_norm1 = self.layer_norm1(x)
        attention_output, _ = self.multi_head_self_attention(x_norm1, x_norm1, x_norm1, need_weights=False) # q, k, v: all the same
        x = x + self.dropout1(attention_output)

        # After the attention layer we normalize again, then apply the MLP
        x_norm2 = self.layer_norm2(x)
        mlp_output = self.multi_layer_perceptron(x_norm2)
        x = x + self.dropout2(mlp_output)

        return x

class Vulcan0Decoder(nn.Module):
    def __init__(self, config: Vulcan0Config):
        super().__init__()

        self.model_dimension = 512
        self.num_heads = 8
        self.dim_feedforward = 3200
        self.dropout = 0.1

        self.multi_head_self_attention = nn.MultiheadAttention(embed_dim=self.model_dimension, num_heads=self.num_heads, dropout=self.dropout)
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.model_dimension, num_heads=self.num_heads, dropout=self.dropout)
        self.multi_layer_perceptron = nn.Sequential(
            nn.Linear(self.model_dimension, self.dim_feedforward),
            nn.GELU(),
            nn.Linear(self.dim_feedforward, self.model_dimension),
            nn.Dropout(self.dropout)
        )

        self.layer_norm1 = nn.LayerNorm(self.model_dimension)
        self.layer_norm2 = nn.LayerNorm(self.model_dimension)
        self.layer_norm3 = nn.LayerNorm(self.model_dimension)
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        self.dropout3 = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        '''
        Input: (S, B, D): S = Sequence Length, B = Batch Size, D = Model Dimension
        Encoder Output: (S, B, D): S = Sequence Length, B = Batch Size, D = Model Dimension
        Output: (S, B, D): S = Sequence Length, B = Batch Size, D = Model Dimension
        '''

        # Pre norm so we normalize (mean = 0, std = 1) before the attention layer
        x_norm1 = self.layer_norm1(x)
        attention_output, _ = self.multi_head_self_attention(x_norm1, x_norm1, x_norm1, need_weights=False) # q, k, v: all the same
        x = x + self.dropout1(attention_output)

        # After the attention layer we normalize again, then apply the cross attention
        x_norm2 = self.layer_norm2(x)
        cross_attention_output, _ = self.cross_attention(x_norm2, encoder_out, encoder_out, need_weights=False) # q: decoder, k, v: encoder
        x = x + self.dropout2(cross_attention_output)

        # After the cross attention layer we normalize again, then apply the MLP
        x_norm3 = self.layer_norm3(x)
        mlp_output = self.multi_layer_perceptron(x_norm3)
        x = x + self.dropout3(mlp_output)

        return x
