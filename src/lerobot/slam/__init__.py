#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""SLAM boundary package for transport, adapter orchestration, and typed contracts."""

from .adapter import SlamAdapter
from .orchestrator import SlamOrchestrator
from .orbslam3_adapter import ORBSLAM3Adapter
from .runtime import InProcessSlamRuntime
from .types import CameraFrame, SlamHealth, SlamInput, SlamOutput, SlamPose, SlamStatus

__all__ = [
    "CameraFrame",
    "InProcessSlamRuntime",
    "ORBSLAM3Adapter",
    "SlamAdapter",
    "SlamHealth",
    "SlamInput",
    "SlamOrchestrator",
    "SlamOutput",
    "SlamPose",
    "SlamStatus",
]
