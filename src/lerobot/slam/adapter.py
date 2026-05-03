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

"""Stable SLAM adapter contract used by runtime orchestrators."""

from __future__ import annotations

from typing import Protocol

from .types import SlamInput, SlamOutput


class SlamAdapter(Protocol):
    """Backend-agnostic adapter contract."""

    def start(self) -> None:
        """Initialize backend resources."""

    def submit(self, slam_input: SlamInput) -> None:
        """Consume one stereo+odom input sample."""

    def get_latest(self) -> SlamOutput | None:
        """Return latest SLAM output, if available."""

    def save_map(self) -> str | None:
        """Persist map artifact and return path when supported."""

    def stop(self) -> None:
        """Release backend resources."""

