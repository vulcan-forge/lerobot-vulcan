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

import numbers
import os
from typing import Any

import numpy as np
import rerun as rr


def _init_rerun(session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop."""
    print(f"DEBUG: Initializing Rerun with session: {session_name}")

    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    print("DEBUG: rr.init() completed")

    # Get the current recording and ensure it's connected
    recording = rr.get_global_data_recording()
    print(f"DEBUG: Global recording: {recording}")
    if recording is None:
        print("ERROR: No global data recording found!")
        return

    # Always spawn a new viewer
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    print(f"DEBUG: Spawning with memory limit: {memory_limit}")
    try:
        recording.spawn(memory_limit=memory_limit)
        print("DEBUG: Spawn completed successfully")
    except Exception as e:
        print(f"DEBUG: Spawn failed: {e}")


def _is_scalar(x):
    return (
        isinstance(x, float)
        or isinstance(x, numbers.Real)
        or isinstance(x, (np.integer, np.floating))
        or (isinstance(x, np.ndarray) and x.ndim == 0)
    )


def log_rerun_data(
    observation: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
) -> None:
    """Logs observation and action data to Rerun for real-time visualization."""
    print(f"DEBUG: log_rerun_data called with obs={bool(observation)}, action={bool(action)}")

    # Get the current recording
    recording = rr.get_global_data_recording()
    print(f"DEBUG: Recording in log_rerun_data: {recording}")
    if recording is None:
        print("ERROR: No recording available for logging!")
        return

    if observation:
        print(f"DEBUG: Logging {len(observation)} observation keys")
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith("observation.") else f"observation.{k}"
            print(f"DEBUG: Logging observation key: {key}")

            if _is_scalar(v):
                rr.log(key, rr.Scalar(float(v)), recording=recording)
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalar(float(vi)), recording=recording)
                else:
                    rr.log(key, rr.Image(arr), static=True, recording=recording)

    if action:
        print(f"DEBUG: Logging {len(action)} action keys")
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith("action.") else f"action.{k}"
            print(f"DEBUG: Logging action key: {key}")

            if _is_scalar(v):
                rr.log(key, rr.Scalar(float(v)), recording=recording)
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalar(float(vi)), recording=recording)
                else:
                    # Fall back to flattening higher-dimensional arrays
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalar(float(vi)), recording=recording)
