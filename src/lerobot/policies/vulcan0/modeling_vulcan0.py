from itertools import chain
import math
import torch
import torch.nn as nn
import numpy as np
import torchvision

from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FrozenBatchNorm2d

from lerobot.policies.vulcan0.configuration_vulcan0 import Vulcan0Config


class Vulcan0Model(nn.Module):
    def __init__(self, config: Vulcan0Config):
        super().__init__()

        self.config = config

        #############################################################
        # Vae Encoder Layers
        #############################################################
        self.vae_encoder = Vulcan0Encoder(config)
        self.class_token = nn.Embedding(1, config.model_dimension)

        self.state_to_dimension = nn.Linear(config.robot_state_feature.shape[0], config.model_dimension)
        self.action_to_dimension = nn.Linear(config.action_feature.shape[0], config.model_dimension)

        self.class_to_latent = nn.Linear(config.model_dimension, config.latent_dim * 2)

        num_vae_tokens = 2 + config.chunk_size

        # Fixed sinusoidal positional embedding for the input to the VAE encoder.
        # This parameter is not trained
        self.register_buffer(
            "vae_pos_embed",
            create_sinusoidal_position_embedding_1D(num_vae_tokens, config.model_dimension)
            .unsqueeze(1),  # (S,1,D)
            persistent=False,
        )

        #############################################################
        # State and Action Layers
        #############################################################

        # Observation images and state
        self.obs_image = nn.Linear(config.image_features.shape[0], config.model_dimension)
        self.obs_state = nn.Linear(config.robot_state_feature.shape[0], config.model_dimension)

        # Action
        self.action_input_proj = nn.Linear(
            config.action_feature.shape[0],
            config.model_dimension
        )
        self.action_output_proj = nn.Linear(
            config.model_dimension,
            config.latent_dim * 2
        )

        # Backbone for image feature extraction,
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Encoder and Decoder
        self.vae_encoder = Vulcan0Encoder(config)
        self.encoder = Vulcan0Encoder(config)
        self.decoder = Vulcan0Decoder(config)

        # Position Embeddings
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.model_dimension)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = SinusoidalPositionEmbedding2D(config.model_dimension // 2)

        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.model_dimension)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.model_dimension, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        '''
        Input: batch -> batch: {OBS_STATE: (S, B, C)} -> S: Sequence length, B: Batch size, C: Channel size
        Output: (S, B, C)
        '''

        # Prepare transformer encoder inputs.



        # We need to input the encoder output into the decoder


        return x, {}


#################################################################
# Encoder
#################################################################
class Vulcan0Encoder(nn.Module):
    def __init__(self, config: Vulcan0Config):
        super().__init__()
        self.layers = nn.ModuleList([Vulcan0EncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.norm = nn.LayerNorm(config.model_dimension)

    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor | None = None) -> torch.Tensor:
        '''
        Input: x, pos_embed -> x: (S, B, C), pos_embed: (S, 1, C) -> S: Sequence length, B: Batch size, C: Channel size
        Output: (S, B, C)
        '''

        for layer in self.layers:
            x = layer(x, pos_embed)
        x = self.norm(x)
        return x

class Vulcan0EncoderLayer(nn.Module):
    def __init__(self, config: Vulcan0Config):
        super().__init__()

        self.multi_head_self_attention = nn.MultiheadAttention(config.model_dimension, config.num_heads, config.dropout)
        self.multi_layer_perceptron = nn.Sequential(
            nn.Linear(config.model_dimension, config.feed_forward_dim),
            nn.GELU(),
            nn.Linear(config.feed_forward_dim, config.model_dimension),
            nn.Dropout(config.dropout),
        )

        self.layer_norm1 = nn.LayerNorm(config.model_dimension)
        self.layer_norm2 = nn.LayerNorm(config.model_dimension)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor | None = None) -> torch.Tensor:
        '''
        Input: x, pos_embed -> x: (S, B, C), pos_embed: (S, 1, C) -> S: Sequence length, B: Batch size, C: Channel size
        Output: (S, B, C)
        '''

        # Multi-head self-attention
        x_norm1 = self.layer_norm1(x)
        if pos_embed is not None:
            q = x_norm1 + pos_embed
            k = x_norm1 + pos_embed
            v = x_norm1
        else:
            q = x_norm1
            k = x_norm1
            v = x_norm1

        attention_output, _ = self.multi_head_self_attention(q, k, v, need_weights=False)
        x = x + self.dropout1(attention_output)

        # Feed forward layers
        x_norm2 = self.layer_norm2(x)
        mlp_output = self.multi_layer_perceptron(x_norm2)
        x = x + self.dropout2(mlp_output)

        return x

#################################################################
# Decoder
#################################################################
class Vulcan0Decoder(nn.Module):
    def __init__(self, config: Vulcan0Config):
        super().__init__()
        self.layers = nn.ModuleList([Vulcan0DecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.norm = nn.LayerNorm(config.model_dimension)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, enc_pos_embed: torch.Tensor | None = None, dec_pos_embed: torch.Tensor | None = None) -> torch.Tensor:
        '''
        Input: x, encoder_out, enc_pos_embed, dec_pos_embed -> x: (S, B, C), encoder_out: (S, B, C), enc_pos_embed: (S, 1, C), dec_pos_embed: (S, 1, C) -> S: Sequence length, B: Batch size, C: Channel size
        Output: (S, B, C)
        '''

        for layer in self.layers:
            x = layer(x, encoder_out, enc_pos_embed, dec_pos_embed)
        x = self.norm(x)
        return x

class Vulcan0DecoderLayer(nn.Module):
    def __init__(self, config: Vulcan0Config):
        super().__init__()

        self.multi_head_self_attention = nn.MultiheadAttention(config.model_dimension, config.num_heads, config.dropout)
        self.multi_head_cross_attention = nn.MultiheadAttention(config.model_dimension, config.num_heads, config.dropout)
        self.multi_layer_perceptron = nn.Sequential(
            nn.Linear(config.model_dimension, config.feed_forward_dim),
            nn.GELU(),
            nn.Linear(config.feed_forward_dim, config.model_dimension),
            nn.Dropout(config.dropout),
        )
        self.layer_norm1 = nn.LayerNorm(config.model_dimension)
        self.layer_norm2 = nn.LayerNorm(config.model_dimension)
        self.layer_norm3 = nn.LayerNorm(config.model_dimension)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, enc_pos_embed: torch.Tensor | None = None, dec_pos_embed: torch.Tensor | None = None) -> torch.Tensor:
        '''
        Input: x, encoder_out, enc_pos_embed, dec_pos_embed -> x: (S, B, C), encoder_out: (S, B, C), enc_pos_embed: (S, 1, C), dec_pos_embed: (S, 1, C) -> S: Sequence length, B: Batch size, C: Channel size
        Output: (S, B, C)
        '''

        # Multi-head self-attention
        x_norm1 = self.layer_norm1(x)
        if dec_pos_embed is not None:
            q = x_norm1 + dec_pos_embed
            k = x_norm1 + dec_pos_embed
            v = x_norm1
        else:
            q = x_norm1
            k = x_norm1
            v = x_norm1

        attention_output, _ = self.multi_head_self_attention(q, k, v, need_weights=False)
        x = x + self.dropout1(attention_output)

        # Multi-head cross-attention
        x_norm2 = self.layer_norm2(x)
        if dec_pos_embed is not None:
            q = x_norm2 + dec_pos_embed
        else:
            q = x_norm2

        if enc_pos_embed is not None:
            k = encoder_out + enc_pos_embed
            v = encoder_out
        else:
            k = encoder_out
            v = encoder_out

        attention_output, _ = self.multi_head_cross_attention(q, k, v, need_weights=False)
        x = x + self.dropout2(attention_output)

        # Feed forward layers
        x_norm3 = self.layer_norm3(x)
        mlp_output = self.multi_layer_perceptron(x_norm3)
        x = x + self.dropout3(mlp_output)

        return x

#################################################################
# Sinuoidal Position Embedding
#################################################################
class SinusoidalPositionEmbedding2D(nn.Module):
    def __init__(self, config: Vulcan0Config):
        super().__init__()
        self.config = config

        self._two_pi = 2 * math.pi
        self._temperature = 10000

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Input: x -> x: (B, C, H, W) -> B: Batch size, C: Channel size, H: Height, W: Width
        Output: (1, D, H, W) -> 1: Batch size, D: Feature dimension, H: Height, W: Width
        '''

        # Create 2 tensors of shape (1, H, W) that represent the position of each pixel in the image.
        ones_mask = torch.ones_like(x[0, :1]) # (1, H, W)
        y_range = ones_mask.cumsum(1, dtype=torch.float32)
        x_range = ones_mask.cumsum(2, dtype=torch.float32)

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
        ), dim=-1).flatten(3)
        pos_embed_x = torch.stack((
            x_range[..., 0::2].sin(),
            x_range[..., 1::2].cos(),
        ), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2) # (1, D, H, W)
        return pos_embed

def create_sinusoidal_position_embedding_1D(num_positions: int, dimension: int) -> torch.Tensor:
    '''
    Input: num_positions, dimension -> num_positions: Number of token positions required, dimension: Embedding dimension
    Output: (N, D)
    '''

    temperature = 10000
    pos = np.arange(num_positions)[:, None] # (N, 1)
    i = np.arrange(dimension)[None, :] # (1, D)

    angle_rate = 1 / np.power(temperature, (2 * (i // 2)) / dimension) # (1, D)
    angles = pos * angle_rate # (N, D)

    angles[:, 0::2] = np.sin(angles[:, 0::2]) # even indices get sine
    angles[:, 1::2] = np.cos(angles[:, 1::2]) # odd indices get cosine

    # float() is necessary to convert the numpy array (which defaults to float64) to float32
    combined_angles = torch.from_numpy(angles).float() # (N, D)
    return combined_angles
