from collections import deque
from itertools import chain
import math
import torch
import torch.nn.functional as F  # noqa: N812
import torch.nn as nn
import numpy as np
import torchvision

from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FrozenBatchNorm2d

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.vulcan0.configuration_vulcan0 import Vulcan0Config
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

class Vulcan0Policy(PreTrainedPolicy):
    config_class = Vulcan0Config
    name = "vulcan0"

    def __init__(self, config: Vulcan0Config):
        super().__init__(config)
        self.config = config

        self.model = Vulcan0Model(config)

        # No internal recurrent state for now, but keep reset for API.
        self.reset()

    # ----------------------------
    # Optim params (same pattern as ACTPolicy)
    # ----------------------------
    def get_optim_params(self):
        return [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    # ----------------------------
    # Reset (queue for chunked actions)
    # ----------------------------
    def reset(self):
        self._action_queue = deque([], maxlen=self.config.num_action_steps)

    # ----------------------------
    # Inference API (env step-by-step)
    # ----------------------------
    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Returns a single action: (B, action_dim) for execution.
        Uses a queue so we only run the model every n_action_steps.
        """
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.num_action_steps]  # (B, n_steps, A)

            # queue expects (n_steps, B, A)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()  # (B, A)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Returns (B, chunk_size, action_dim)
        """
        self.eval()

        # If LeRobot gives OBS_IMAGES as list-of-camera tensors, keep it.
        # If your model expects a single tensor, ensure your environment collator matches that.
        actions_hat, _stats = self.model(batch)
        return actions_hat

    # ----------------------------
    # Training forward (loss)
    # ----------------------------
    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            loss: scalar tensor
            loss_dict: python dict of scalars for logging
        """
        actions_hat, stats = self.model(batch)  # actions_hat: (B, S, A)

        # L1 behavior cloning loss
        l1_loss = F.l1_loss(batch[ACTION], actions_hat, reduction="mean")

        # KL loss (always VAE)
        mu = stats["mu"]
        log_sigma_x2 = stats["log_sigma_x2"]

        kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()

        loss = l1_loss + self.config.kl_weight * kld

        loss_dict = {
            "loss": float(loss.detach().cpu()),
            "l1_loss": float(l1_loss.detach().cpu()),
            "kld_loss": float(kld.detach().cpu()),
        }
        return loss, loss_dict

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
        self.latent_to_dimension = nn.Linear(config.latent_dim, config.model_dimension)

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
        # Image Extractor
        #############################################################
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
            weights=config.pretrained_backbone_weights,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        self.image_to_model = nn.Conv2d(
            backbone_model.fc.in_features,
            config.model_dimension,
            kernel_size=1
        )

        self.image_pos_embed = SinusoidalPositionEmbedding2D(config)

        #############################################################
        # Main Transformer (Encoder + Decoder)
        #############################################################
        self.encoder = Vulcan0Encoder(config)
        self.decoder = Vulcan0Decoder(config)

         # Learned 1D positional embeddings for encoder tokens: [latent, state]
        self.encoder_1d_pos_embed = nn.Embedding(2, config.model_dimension)  # (2, D)

        # Learned decoder "query" positional embeddings (one per action step)
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.model_dimension)  # (S_dec, D)

        # Final action regression head (decoder output -> action_dim)
        self.action_head = nn.Linear(config.model_dimension, config.action_feature.shape[0])

        self._reset_parameters()

    def build_encoder_tokens(
        self,
        z: torch.Tensor,                 # (B, L)       latent sample from VAE
        obs_state: torch.Tensor,          # (B, Sdim)    robot state / proprioception
        obs_images: torch.Tensor,         # (B, Cin, H, W) raw image
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            tokens: (S_total, B, D)
            pos:    (S_total, B, D)
        Where:
            S_total = 2 + (H' * W')   # [latent, state] + image feature tokens
        """

        # ------------------------------------------------------------
        # 1) Build 1D tokens: latent token + state token
        # ------------------------------------------------------------

        # latent: (B, L) -> (B, D)
        latent_tok = self.latent_to_dimension(z)                       # (B, D)

        # state: (B, Sdim) -> (B, D)
        state_tok = self.state_to_dimension(obs_state)                 # (B, D)

        # stack into sequence length 2:
        # (B, D) and (B, D) -> (2, B, D)
        tokens_1d = torch.stack([latent_tok, state_tok], dim=0)    # (2, B, D)

        # learned pos embeddings for [latent, state]
        # encoder_1d_pos_embed.weight: (2, D)
        # -> add batch dimension: (2, 1, D)
        # -> expand across batch: (2, B, D)
        pos_1d = self.encoder_1d_pos_embed.weight[:, None, :].expand(2, z.shape[0], -1)  # (2, B, D)

        # ------------------------------------------------------------
        # 2) Build image tokens from CNN feature map
        # ------------------------------------------------------------

        # backbone output feature map (example):
        # obs_images: (B, Cin, H, W) -> (B, C_backbone, H', W')
        feat = self.backbone(obs_images)["feature_map"]         # (B, C_backbone, H', W')

        # project channels to model dimension D
        feat = self.image_to_model(feat)                         # (B, D, H', W')

        # 2D sinusoidal position embedding for the feature map
        # returns: (1, D, H', W')  (broadcastable across batch)
        img_pos = self.image_pos_embed(feat)                       # (1, D, H', W')

        # flatten feature map to a token sequence:
        # (B, D, H', W') -> (B, D, H'*W') -> (H'*W', B, D)
        feat_tokens = feat.flatten(2).permute(2, 0, 1)             # (H'*W', B, D)

        # flatten pos embedding the same way:
        # (1, D, H', W') -> (1, D, H'*W') -> (H'*W', 1, D)
        # then expand across batch -> (H'*W', B, D)
        img_pos_tokens = img_pos.flatten(2).permute(2, 0, 1)       # (H'*W', 1, D)
        img_pos_tokens = img_pos_tokens.expand(-1, z.shape[0], -1) # (H'*W', B, D)

        # ------------------------------------------------------------
        # 3) Concatenate 1D tokens + image tokens into encoder input
        # ------------------------------------------------------------

        # tokens: (2, B, D) + (H'*W', B, D) -> (2 + H'*W', B, D)
        tokens = torch.cat([tokens_1d, feat_tokens], dim=0)        # (S_total, B, D)

        # pos: same shape
        pos = torch.cat([pos_1d, img_pos_tokens], dim=0)           # (S_total, B, D)

        return tokens, pos

    def encode_latent(
        self,
        obs_state: torch.Tensor,   # (B, Sdim)
        actions: torch.Tensor,     # (B, S, Adim)
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            z: (B, L)
            stats: {"mu": (B, L), "log_sigma_x2": (B, L)}
        """
        B = obs_state.shape[0]
        D = self.config.model_dimension
        device = obs_state.device

        # CLS token: (1, D) -> (B, 1, D)
        cls = self.class_token.weight.unsqueeze(0).expand(B, 1, D)          # (B, 1, D)

        # State token: (B, Sdim) -> (B, 1, D)
        state_tok = self.state_to_dimension(obs_state).unsqueeze(1)         # (B, 1, D)

        # Action tokens: (B, S, Adim) -> (B, S, D)
        action_tok = self.action_to_dimension(actions)                       # (B, S, D)

        # VAE input: [CLS, state, actions]
        x = torch.cat([cls, state_tok, action_tok], dim=1)                   # (B, S+2, D)
        x = x.permute(1, 0, 2)                                               # (S+2, B, D)

        # Fixed sinusoidal positions: (S+2, 1, D)
        pos = self.vae_pos_embed[: x.shape[0]].to(device)                    # (S+2, 1, D)

        out = self.vae_encoder(x, pos_embed=pos)                             # (S+2, B, D)
        cls_out = out[0]                                                     # (B, D)

        params = self.class_to_latent(cls_out)                               # (B, 2L)
        mu, log_sigma_x2 = params.chunk(2, dim=-1)                           # (B, L), (B, L)

        z = mu + (0.5 * log_sigma_x2).exp() * torch.randn_like(mu)           # (B, L)
        return z, {"mu": mu, "log_sigma_x2": log_sigma_x2}

    def encode_memory(
        self,
        z: torch.Tensor,            # (B, L)
        obs_state: torch.Tensor,    # (B, Sdim)
        obs_images: torch.Tensor,   # (B, Cin, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            memory: (S_total, B, D)
            enc_pos: (S_total, B, D)
        """
        enc_tokens, enc_pos = self.build_encoder_tokens(z, obs_state, obs_images)  # (S_total,B,D)
        memory = self.encoder(enc_tokens, pos_embed=enc_pos)                       # (S_total,B,D)
        return memory, enc_pos

    def decode_actions(
        self,
        memory: torch.Tensor,       # (S_total, B, D)
        enc_pos: torch.Tensor,      # (S_total, B, D)
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns:
            actions: (B, S_dec, Adim)
        """
        S_dec = self.config.chunk_size
        D = self.config.model_dimension

        dec_in = torch.zeros((S_dec, batch_size, D), device=device)                 # (S_dec,B,D)
        dec_pos = self.decoder_pos_embed.weight.unsqueeze(1)                        # (S_dec,1,D)

        dec_out = self.decoder(
            dec_in,
            memory,
            enc_pos_embed=enc_pos,
            dec_pos_embed=dec_pos,
        )                                                                          # (S_dec,B,D)

        actions = self.action_head(dec_out.transpose(0, 1))                         # (B,S_dec,Adim)
        return actions

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """
        batch[OBS_STATE]:  (B, Sdim)
        batch[OBS_IMAGES]: (B, Cin, H, W)
        batch[ACTION]:     (B, S, Adim)
        """
        obs_state = batch[OBS_STATE]
        obs_images = batch[OBS_IMAGES]
        actions_gt = batch[ACTION]

        B = obs_state.shape[0]
        device = obs_state.device

        z, stats = self.encode_latent(obs_state, actions_gt)                    # (B,L)
        memory, enc_pos = self.encode_memory(z, obs_state, obs_images)          # (S_total,B,D)
        actions = self.decode_actions(memory, enc_pos, B, device)               # (B,S,Adim)

        return actions, stats


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
        self.dimension = config.model_dimension // 2

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
        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

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
    i = np.arange(dimension)[None, :] # (1, D)

    angle_rate = 1 / np.power(temperature, (2 * (i // 2)) / dimension) # (1, D)
    angles = pos * angle_rate # (N, D)

    angles[:, 0::2] = np.sin(angles[:, 0::2]) # even indices get sine
    angles[:, 1::2] = np.cos(angles[:, 1::2]) # odd indices get cosine

    # float() is necessary to convert the numpy array (which defaults to float64) to float32
    combined_angles = torch.from_numpy(angles).float() # (N, D)
    return combined_angles
