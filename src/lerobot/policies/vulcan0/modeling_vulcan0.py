import torch
import torch.nn as nn

from lerobot.policies.vulcan0.configuration_vulcan0 import Vulcan0Config
from lerobot.policies.pretrained import PreTrainedPolicy

class Vulcan0Policy(PreTrainedPolicy):
    config_class = Vulcan0Config
    name = "vulcan0"

    def __init__(self, config: Vulcan0Config):
        super().__init__(config)
        self.config = config
        self.model = Vulcan0Model(config)

class Vulcan0Model(nn.Module):
    def __init__(self, config: Vulcan0Config):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class Vulcan0Encoder(nn.Module):
    def __init__(self, config: Vulcan0Config):
        super().__init__()

        model_dimension = 512
        num_heads = 8
        dim_feedforward = 3200
        dropout = 0.1

        self.multi_head_self_attention = nn.MultiheadAttention(embed_dim=model_dimension, num_heads=num_heads, dropout=dropout)
        self.multi_layer_perceptron = nn.Sequential(
            nn.Linear(model_dimension, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, model_dimension),
            nn.Dropout(dropout),
        )

        self.layer_norm1 = nn.LayerNorm(model_dimension)
        self.layer_norm2 = nn.LayerNorm(model_dimension)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Input: (S, B, D) S: sequence length, B: batch size, D: input dimension
        Output: (S, B, D) S: sequence length, B: batch size, D: output dimension
        '''

        x_norm1 = self.layer_norm1(x)
        attention_output = self.multi_head_self_attention(x_norm1, x_norm1, x_norm1)
        attention_output = attention_output[0] # Selects just the output, not the attention weights
        x = x + self.dropout1(attention_output)

        x_norm2 = self.layer_norm2(x)
        mlp_output = self.multi_layer_perceptron(x_norm2)
        x = x + self.dropout2(mlp_output)

        return x

class Vulcan0Decoder(nn.Module):
    def __init__(self, config: Vulcan0Config):
        super().__init__()

        model_dimension = 512
        num_heads = 8
        dim_feedforward = 3200
        dropout = 0.1

        self.multi_head_self_attention = nn.MultiheadAttention(embed_dim=model_dimension, num_heads=num_heads, dropout=dropout)

        # Cross attention used by the decoder to query key / value memory from the encoder
        self.cross_attention = nn.MultiheadAttention(embed_dim=model_dimension, num_heads=num_heads, dropout=dropout)
        self.multi_layer_perceptron = nn.Sequential(
            nn.Linear(model_dimension, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, model_dimension),
            nn.Dropout(dropout),
        )

        self.layer_norm1 = nn.LayerNorm(model_dimension)
        self.layer_norm2 = nn.LayerNorm(model_dimension)
        self.layer_norm3 = nn.LayerNorm(model_dimension)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        '''
        Input: (S, B, D) S: sequence length, B: batch size, D: input dimension
        Input: (S, B, D) S: sequence length, B: batch size, D: encoder output dimension
        Output: (S, B, D) S: sequence length, B: batch size, D: output dimension
        '''

        x_norm1 = self.layer_norm1(x)
        attention_output = self.multi_head_self_attention(x_norm1, x_norm1, x_norm1) # q, k, v: all the same
        attention_output = attention_output[0] # Selects just the output, not the attention weights
        x = x + self.dropout1(attention_output)

        x_norm2 = self.layer_norm2(x)
        cross_attention_output = self.cross_attention(x_norm2, encoder_out, encoder_out) # q, k, v: queries from decoder, keys and values from encoder
        cross_attention_output = cross_attention_output[0] # Selects just the output, not the attention weights
        x = x + self.dropout2(cross_attention_output)

        x_norm3 = self.layer_norm3(x)
        mlp_output = self.multi_layer_perceptron(x_norm3)
        x = x + self.dropout3(mlp_output)

        return x
