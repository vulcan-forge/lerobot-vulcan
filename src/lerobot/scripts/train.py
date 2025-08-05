#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.scripts.eval import eval_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    # Set up logging first
    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Set up distributed training if enabled
    if cfg.distributed_training:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA")

        # Initialize the process group
        torch.distributed.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Set device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        # Only log on main process
        if local_rank == 0:
            logging.info(f"Initialized distributed training with {world_size} GPUs")
    else:
        device = get_safe_torch_device(cfg.policy.device, log=True)
        local_rank = 0
        world_size = 1

    # Set CUDA configurations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    # Create optimizer, scheduler, and grad scaler BEFORE DDP wrapping
    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    # Wrap model with DDP if using distributed training
    if cfg.distributed_training:
        policy = torch.nn.parallel.DistributedDataParallel(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=cfg.ddp_find_unused_parameters
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    # Create distributed sampler if using distributed training
    if cfg.distributed_training:
        # Ensure batch size is divisible by number of GPUs
        if cfg.batch_size % world_size != 0:
            raise ValueError(f"Batch size ({cfg.batch_size}) must be divisible by number of GPUs ({world_size})")
        per_gpu_batch_size = cfg.batch_size // world_size

        if sampler is not None:
            # If we have a custom sampler, we need to wrap it
            sampler = torch.utils.data.distributed.DistributedSampler(
                sampler,
                num_replicas=world_size,
                rank=local_rank
            )
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=local_rank
            )
        shuffle = False  # DistributedSampler handles shuffling
    else:
        per_gpu_batch_size = cfg.batch_size

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=per_gpu_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0 and cfg.dataloader_persistent_workers),
        prefetch_factor=cfg.dataloader_prefetch_factor,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    # Only log on main process
    if local_rank == 0:
        logging.info("Start offline training on a fixed dataset")

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        # Only log, save, and eval on the main process
        if local_rank == 0:
            if is_log_step:
                logging.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()

            if cfg.save_checkpoint and is_saving_step:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)

                # Get the underlying model if using DDP
                policy_to_save = policy.module if isinstance(policy, torch.nn.parallel.DistributedDataParallel) else policy

                save_checkpoint(checkpoint_dir, step, cfg, policy_to_save, optimizer, lr_scheduler)
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            if cfg.env and is_eval_step:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
                ):
                    eval_info = eval_policy(
                        eval_env,
                        policy,
                        cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                    )

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
                )
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                logging.info(eval_tracker)
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env:
        eval_env.close()

    # Clean up distributed training
    if cfg.distributed_training:
        torch.distributed.destroy_process_group()

    logging.info("End of training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


def main():
    init_logging()
    train()


if __name__ == "__main__":
    main()
