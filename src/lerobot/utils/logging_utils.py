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
from typing import Any, Dict
import warnings
from collections.abc import Callable

from lerobot.utils.utils import format_big_number


class AverageMeter:
    """
    Computes and stores the average and current value
    Adapted from https://github.com/pytorch/examples/blob/main/imagenet/main.py
    """

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}:{avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class MetricsTracker:
    """
    A helper class to track and log metrics over time.

    Usage pattern:

    ```python
    # initialize, potentially with non-zero initial step (e.g. if resuming run)
    metrics = {"loss": AverageMeter("loss", ":.3f")}
    train_metrics = MetricsTracker(cfg, dataset, metrics, initial_step=step)

    # update metrics derived from step (samples, episodes, epochs) at each training step
    train_metrics.step()

    # update various metrics
    loss = policy.forward(batch)
    train_metrics.loss = loss

    # display current metrics
    logging.info(train_metrics)

    # export for wandb
    wandb.log(train_metrics.to_dict())

    # reset averages after logging
    train_metrics.reset_averages()
    ```
    """

    __keys__ = [
        "_batch_size",
        "_num_frames",
        "_avg_samples_per_ep",
        "metrics",
        "steps",
        "samples",
        "episodes",
        "epochs",
        "accelerator",
    ]

    def __init__(
        self,
        batch_size: int,
        num_frames: int,
        num_episodes: int,
        metrics: dict[str, AverageMeter],
        initial_step: int = 0,
        accelerator: Callable | None = None,
    ):
        self.__dict__.update(dict.fromkeys(self.__keys__))
        self._batch_size = batch_size
        self._num_frames = num_frames
        self._avg_samples_per_ep = num_frames / num_episodes
        self.metrics = metrics

        self.steps = initial_step
        # A sample is an (observation,action) pair, where observation and action
        # can be on multiple timestamps. In a batch, we have `batch_size` number of samples.
        self.samples = self.steps * self._batch_size
        self.episodes = self.samples / self._avg_samples_per_ep
        self.epochs = self.samples / self._num_frames
        self.accelerator = accelerator

    def __getattr__(self, name: str) -> int | dict[str, AverageMeter] | AverageMeter | Any:
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self.metrics:
            return self.metrics[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dict__:
            super().__setattr__(name, value)
        elif name in self.metrics:
            self.metrics[name].update(value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def step(self) -> None:
        """
        Updates metrics that depend on 'step' for one step.
        """
        self.steps += 1
        self.samples += self._batch_size * (self.accelerator.num_processes if self.accelerator else 1)
        self.episodes = self.samples / self._avg_samples_per_ep
        self.epochs = self.samples / self._num_frames

    def __str__(self) -> str:
        display_list = [
            f"step:{format_big_number(self.steps)}",
            # number of samples seen during training
            f"smpl:{format_big_number(self.samples)}",
            # number of episodes seen during training
            f"ep:{format_big_number(self.episodes)}",
            # number of time all unique samples are seen
            f"epch:{self.epochs:.2f}",
            *[str(m) for m in self.metrics.values()],
        ]
        return " ".join(display_list)

    def to_dict(self, use_avg: bool = True) -> dict[str, int | float]:
        """
        Returns the current metric values (or averages if `use_avg=True`) as a dict.
        """
        return {
            "steps": self.steps,
            "samples": self.samples,
            "episodes": self.episodes,
            "epochs": self.epochs,
            **{k: m.avg if use_avg else m.val for k, m in self.metrics.items()},
        }

    def reset_averages(self) -> None:
        """Resets average meters."""
        for m in self.metrics.values():
            m.reset()

class OncePerMinuteWarningFilter(logging.Filter):
    """Filter that only allows each unique warning message to be logged once per minute."""

    def __init__(self):
        super().__init__()
        self.seen_messages: Dict[str, float] = {}
        self.timeout = 60.0  # 60 seconds

    def filter(self, record):
        # Create a unique key for this message
        message_key = f"{record.levelname}:{record.getMessage()}"
        current_time = time.time()

        # Check if we've seen this message recently
        if message_key in self.seen_messages:
            last_seen = self.seen_messages[message_key]
            if current_time - last_seen < self.timeout:
                return False  # Don't log this message again yet

        # Update the timestamp for this message
        self.seen_messages[message_key] = current_time

        # Clean up old entries to prevent memory leaks
        self._cleanup_old_entries(current_time)

        return True  # Log this message

    def _cleanup_old_entries(self, current_time: float):
        """Remove entries older than timeout to prevent memory leaks."""
        old_keys = [
            key for key, timestamp in self.seen_messages.items()
            if current_time - timestamp > self.timeout
        ]
        for key in old_keys:
            del self.seen_messages[key]

class OncePerMinuteWarningHandler:
    """Handler for Python warnings that only shows each warning once per minute."""

    def __init__(self):
        self.seen_warnings: Dict[str, float] = {}
        self.timeout = 60.0  # 60 seconds
        # Store the original showwarning function
        self._original_showwarning = warnings.showwarning

    def __call__(self, message, category, filename, lineno, file=None, line=None):
        # Create a unique key for this warning
        warning_key = f"{category.__name__}:{message}"
        current_time = time.time()

        # Check if we've seen this warning recently
        if warning_key in self.seen_warnings:
            last_seen = self.seen_warnings[warning_key]
            if current_time - last_seen < self.timeout:
                return  # Don't show this warning again yet

        # Update the timestamp for this warning
        self.seen_warnings[warning_key] = current_time

        # Clean up old entries
        self._cleanup_old_entries(current_time)

        # Show the warning using the original handler
        self._original_showwarning(message, category, filename, lineno, file, line)

    def _cleanup_old_entries(self, current_time: float):
        """Remove entries older than timeout to prevent memory leaks."""
        old_keys = [
            key for key, timestamp in self.seen_warnings.items()
            if current_time - timestamp > self.timeout
        ]
        for key in old_keys:
            del self.seen_warnings[key]

def setup_once_per_minute_logging():
    """Setup logging and warnings to only show each message once per minute."""

    # Setup logging filter
    warning_filter = OncePerMinuteWarningFilter()

    # Apply to root logger
    root_logger = logging.getLogger()
    root_logger.addFilter(warning_filter)

    # Setup warnings handler
    warning_handler = OncePerMinuteWarningHandler()
    warnings.showwarning = warning_handler

    # Set warnings to show once (this works with our custom handler)
    warnings.simplefilter("once", UserWarning)
