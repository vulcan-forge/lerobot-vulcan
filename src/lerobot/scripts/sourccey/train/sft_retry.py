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

"""Retry Sourccey SFT and resume from its latest safe checkpoint."""

from .retry import retry_main


def main() -> int:
    return retry_main(
        training_module="lerobot.scripts.sourccey.train.sft",
        training_name="sourccey/train/sft.py",
    )


if __name__ == "__main__":
    raise SystemExit(main())
