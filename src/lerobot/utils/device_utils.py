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

import torch


def auto_select_torch_device() -> torch.device:
    """Tries to select automatically a torch device."""
    if torch.cuda.is_available():
        logging.info("Cuda backend detected, using cuda.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logging.info("Metal backend detected, using mps.")
        return torch.device("mps")
    elif torch.xpu.is_available():
        logging.info("Intel XPU backend detected, using xpu.")
        return torch.device("xpu")
    else:
        logging.warning("No accelerated backend detected. Using default cpu, this will be slow.")
        return torch.device("cpu")


# TODO(Steven): Remove log. log shouldn't be an argument, this should be handled by the logger level
def get_safe_torch_device(try_device: str, log: bool = False) -> torch.device:
    """Given a string, return a torch.device with checks on whether the device is available."""
    try_device = str(try_device)
    if try_device.startswith("cuda"):
        assert torch.cuda.is_available()
        device = torch.device(try_device)
    elif try_device == "mps":
        assert torch.backends.mps.is_available()
        device = torch.device("mps")
    elif try_device == "xpu":
        assert torch.xpu.is_available()
        device = torch.device("xpu")
    elif try_device == "cpu":
        device = torch.device("cpu")
        if log:
            logging.warning("Using CPU, this will be slow.")
    else:
        device = torch.device(try_device)
        if log:
            logging.warning(f"Using custom {try_device} device.")
    return device


def get_safe_dtype(dtype: torch.dtype, device: str | torch.device):
    """
    mps is currently not compatible with float64
    """
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    if device == "xpu" and dtype == torch.float64:
        if hasattr(torch.xpu, "get_device_capability"):
            device_capability = torch.xpu.get_device_capability()
            # NOTE: Some Intel XPU devices do not support double precision (FP64).
            # The `has_fp64` flag is returned by `torch.xpu.get_device_capability()`
            # when available; if False, we fall back to float32 for compatibility.
            if not device_capability.get("has_fp64", False):
                logging.warning(f"Device {device} does not support float64, using float32 instead.")
                return torch.float32
        else:
            logging.warning(
                f"Device {device} capability check failed. Assuming no support for float64, using float32 instead."
            )
            return torch.float32
        return dtype
    else:
        return dtype


def is_native_bfloat16_supported(device: str | torch.device) -> bool:
    """Return whether a CUDA device has native BF16 tensor-core support.

    ``torch.cuda.is_bf16_supported`` may return true when BF16 operations are
    emulated on older GPUs. Policies need native support here because vision
    backbones rely on cuDNN kernels that are not consistently available through
    that emulation path. Native CUDA BF16 starts with compute capability 8.0.
    """
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False

    try:
        major, _minor = torch.cuda.get_device_capability(device)
        runtime_supports_bfloat16 = torch.cuda.is_bf16_supported()
    except (AssertionError, RuntimeError):
        return False
    return major >= 8 and runtime_supports_bfloat16


def get_compatible_policy_dtype(requested_dtype: str, device: str | torch.device) -> str:
    """Keep the requested policy dtype unless BF16 needs an FP32 fallback."""
    if requested_dtype != "bfloat16":
        return requested_dtype
    return "bfloat16" if is_native_bfloat16_supported(device) else "float32"


def is_torch_device_available(try_device: str) -> bool:
    try_device = str(try_device)  # Ensure try_device is a string
    if try_device.startswith("cuda"):
        return torch.cuda.is_available()
    elif try_device == "mps":
        return torch.backends.mps.is_available()
    elif try_device == "xpu":
        return torch.xpu.is_available()
    elif try_device == "cpu":
        return True
    else:
        raise ValueError(f"Unknown device {try_device}. Supported devices are: cuda, mps, xpu or cpu.")


def is_amp_available(device: str):
    if device in ["cuda", "xpu", "cpu"]:
        return True
    elif device == "mps":
        return False
    else:
        raise ValueError(f"Unknown device '{device}.")
