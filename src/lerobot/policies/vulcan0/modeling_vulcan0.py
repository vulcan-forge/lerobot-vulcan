from collections import deque
from itertools import chain
import math
import einops
import numpy as np

import torch
from torch.compiler import F
import torch.nn as nn
import torchvision
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models._utils import IntermediateLayerGetter

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.vulcan0.configuration_vulcan0 import Vulcan0Config
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

class Vulcan0Policy(PreTrainedPolicy):
    def __init__(self, config: Vulcan0Config):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = Vulcan0Model(config)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select a single action given environment observations."""
        self.eval()

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        mean_kld = (
            (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
        )
        loss_dict["kld_loss"] = mean_kld.item()
        loss = l1_loss + mean_kld * self.config.kl_weight
        return loss, loss_dict

class Vulcan0Model(nn.Module):
    def __init__(self, config: Vulcan0Config):
        super().__init__()
        self.config = config

        '''
        Vae Encoder initialization
        '''

        self.vae_encoder = Vulcan0Encoder(config)
        self.vae_encoder_cls_embed = nn.Embedding(1, config.model_dimension)

        # Projection layer for joint-space configuration to hidden dimension
        self.vae_encoder_robot_state_input_proj = nn.Linear(
            self.config.robot_state_feature.shape[0], config.dim_model
        )

        # Projection layer for action (joint-space target) to hidden dimension
        self.vae_encoder_action_input_proj = nn.Linear(
            self.config.action_feature.shape[0],
            config.model_dimension,
        )

        # Projection layer from the VAE encoder's output to the latent distribution's parameter space
        self.vae_encoder_latent_output_proj = nn.Linear(
            config.model_dimension,
            config.latent_dim * 2,
        )

        # Image feature extraction initialization
        backbone_model = getattr(torchvision.models, config.vision_backbone) (
            replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
            weights=config.pretrained_backbone_weights,
            norm_layer=FrozenBatchNorm2d,
        )

        # Note: The forward method of this returns a dict: {"feature_map": output}.
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer Encoder and Decoder
        self.encoder = Vulcan0Encoder(config)
        self.decoder = Vulcan0Decoder(config)

        # Encoder Projections
        self.encoder_robot_state_input_proj = nn.Linear(
            self.config.robot_state_feature.shape[0],
            config.model_dimension,
        )

        self.encoder_env_state_input_proj = nn.Linear(
            self.config.env_state_feature.shape[0],
            config.model_dimension,
        )

        self.encoder_latent_input_proj = nn.Linear(
            config.latent_dim,
            config.model_dimension,
        )

        self.encoder_img_feat_input_proj = nn.Conv2d(
            backbone_model.fc.in_features,
            config.model_dimension,
            kernel_size=1,
        )

        # Transformer encoder positional embeddings.
        n_1d_tokens = 1 # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        if self.config.image_features:
            n_1d_tokens += 1

        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.model_dimension)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = SinusoidalPositionEmbeddings2D(config)

        # Transformer decoder
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.model_dimension)

        self.action_head = nn.Linear(config.model_dimension, self.config.action_feature.shape[0])
        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | tuple[None, None]]:
        assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        # Prepare the latent for input to the transformer encoder.
        if ACTION in batch and self.training:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            pos_embed = self.vae_encoder_pos_enc.clone().detach() # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )

            # Forward pass through the vae encoder
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]

            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )

         # Prepare transformer encoder inputs.
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # Robot state token.
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        # Environment state token.
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            # For a list of images, the H and W may vary but H*W is constant.
            # NOTE: If modifying this section, verify on MPS devices that
            # gradients remain stable (no explosions or NaNs).
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # Rearrange features to (sequence, batch, dim).
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                # Extend immediately instead of accumulating and concatenating
                # Convert to list to extend properly
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)

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


class SinusoidalPositionEmbeddings2D(nn.Module):
    def __init__(self, config: Vulcan0Config):
        super().__init__()

        self._two_pi = 2 * math.pi
        self._temperature = 10000

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Input: (B, C, H, W) -> B: Batch size, C: Channel size, H: Height, W: Width
        Output: (1, D, H, W) -> 1: Batch size, D: feature dimension, H: Height, W: Width
        '''

        # Create 2 tensors of shape (1, H, W) that represent the position of each pixel
        one_mask = torch.ones_like(x[0, :1]) # (1, H, W)
        y_range = one_mask.cumsum(1, dtype=torch.float32) # (1, H, W)
        x_range = one_mask.cumsum(2, dtype=torch.float32) # (1, H, W)

        # Inverse temperature
        inverse_frequency = self._temperature ** (2 * (torch.arange(self.config.model_dimension, dtype=torch.float32, device=x.device) // 2) / self.config.model_dimension)

        # Unqueeze the position tensors to (1, H, W, 1)
        y_range = y_range.unsqueeze(-1)
        x_range = x_range.unsqueeze(-1)

        # Divide the position tensors by the inverse frequency
        y_range = y_range / inverse_frequency
        x_range = x_range / inverse_frequency

        # Stack the position of the tensors and flatten
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


def create_sinusoidal_position_embedding_1D(num_positions: int, dimension: int) -> torch.Tensor:
    '''
    Input: num_positions, dimension -> num_positions: Number of token positions required, dimension: Embedding dimension
    Output: (N, D) -> N: Number of token positions, D: Embedding dimension
    '''

    temperature = 10000
    pos = np.arange(num_positions)[:, None] # (N, 1)
    i = np.array(dimension)[None, :] # (1, D)

    angle_rates = 1 / np.power(temperature, (2 * (i // 2) / dimension)) # (1, D)
    angles = pos * angle_rates # (N, D)

    angles[:, 0::2] = np.sin(angles[:, 0::2]) # even indices get sine
    angles[:, 1::2] = np.cos(angles[:, 1::2]) # odd indices get cosine

    # float() is necessary to convert the numpy array (which defaults to float64) to float32
    combined_angles = torch.from_numpy(angles).float() # (N, D)
    return combined_angles

