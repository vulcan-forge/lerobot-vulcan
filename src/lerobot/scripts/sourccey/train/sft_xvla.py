"""SFT-only XVLA behavior that masks future actions beyond episode boundaries."""

from types import MethodType
from typing import Any

import torch

from lerobot.policies.xvla.modeling_xvla import XVLAPolicy


def _prepare_padding_mask(policy: XVLAPolicy, batch: dict[str, Any], targets: torch.Tensor) -> torch.Tensor:
    if "action_is_pad" not in batch:
        raise ValueError("SFT padded-action masking requires action_is_pad in the batch.")
    mask = batch["action_is_pad"].to(device=targets.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[1] > policy.config.chunk_size:
        mask = mask[:, : policy.config.chunk_size]
    elif mask.shape[1] < policy.config.chunk_size:
        mask = torch.cat(
            [
                mask,
                torch.ones(
                    mask.shape[0],
                    policy.config.chunk_size - mask.shape[1],
                    dtype=torch.bool,
                    device=mask.device,
                ),
            ],
            dim=1,
        )
    if mask.shape != targets.shape[:2]:
        raise ValueError(
            f"action_is_pad shape {tuple(mask.shape)} must match action horizon {tuple(targets.shape[:2])}."
        )
    return mask


def _masked_xvla_forward(self: XVLAPolicy, batch: dict[str, Any]) -> tuple[torch.Tensor, dict]:
    """Mirror XVLA's training forward while masking invalid SFT horizon positions."""
    inputs = self._build_model_inputs(batch)
    targets = self._prepare_action_targets(batch)
    action_is_pad = _prepare_padding_mask(self, batch, targets)
    loss_dims = getattr(self.model.action_space, "real_dim", None)
    if loss_dims is None or loss_dims > targets.shape[-1]:
        raise ValueError(f"Invalid XVLA real action dimension {loss_dims} for targets {tuple(targets.shape)}.")

    extreme = targets.detach().abs() > 1_000_000
    if extreme.any():
        locations = extreme.nonzero()[:16].cpu().tolist()
        padded_at_locations = [bool(action_is_pad[b, t]) for b, t, _ in locations]
        raise RuntimeError(
            "Extreme normalized SFT action target detected before masking: "
            f"max={targets.detach().abs().max().item():.6g}, "
            f"locations={locations}, padded={padded_at_locations}, "
            f"episode_index={batch.get('episode_index')}, frame_index={batch.get('frame_index')}"
        )

    model = self.model
    target_dtype = model._get_target_dtype()
    image_input = inputs["image_input"].to(dtype=target_dtype)
    proprio = inputs["proprio"].to(dtype=target_dtype)
    targets = targets.to(dtype=target_dtype)

    # Sanitize before flow-noise interpolation. Masking only after squaring can
    # turn extreme normalized boundary values into inf and then nan via inf * 0.
    targets = targets.masked_fill(action_is_pad.unsqueeze(-1), 0)
    enc = model.forward_vlm(inputs["input_ids"], image_input, inputs["image_mask"])

    batch_size = inputs["input_ids"].shape[0]
    t = (
        torch.rand(1, device=inputs["input_ids"].device, dtype=target_dtype)
        + torch.arange(batch_size, device=inputs["input_ids"].device, dtype=target_dtype) / batch_size
    ) % (1 - 1e-5)
    action_noisy = (
        torch.randn_like(targets) * t.view(-1, 1, 1) + targets * (1 - t).view(-1, 1, 1)
    )
    proprio_m, action_noisy_m = model.action_space.preprocess(proprio, action_noisy)
    pred_action = model.transformer(
        domain_id=inputs["domain_id"],
        action_with_noise=action_noisy_m,
        t=t,
        proprio=proprio_m,
        **enc,
    )

    squared_error = (pred_action[:, :, :loss_dims] - targets[:, :, :loss_dims]).square()
    valid = (~action_is_pad).unsqueeze(-1).expand_as(squared_error)
    joints_loss = squared_error.masked_select(valid).sum() / valid.sum().clamp_min(1)
    losses = {"joints_loss": joints_loss}
    total_loss = sum(losses.values())
    log_dict = {name: value.detach().item() for name, value in losses.items()}
    log_dict["loss"] = total_loss.detach().item()
    return total_loss, log_dict


def enable_sft_xvla_padding_mask(policy: Any, cfg: Any) -> Any:
    """Install the SFT-only forward on one policy instance when explicitly enabled."""
    if not cfg.dataset.mask_padded_actions:
        return policy
    if not isinstance(policy, XVLAPolicy):
        raise ValueError("mask_padded_actions is currently supported only for XVLA SFT.")
    if policy.config.action_mode.lower() != "auto":
        raise ValueError("mask_padded_actions currently requires XVLA action_mode='auto'.")
    policy.forward = MethodType(_masked_xvla_forward, policy)
    return policy
