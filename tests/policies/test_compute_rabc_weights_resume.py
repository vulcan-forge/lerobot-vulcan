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

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.policies.sarm.compute_rabc_weights import (
    _episode_part_path,
    _get_parts_dir,
    _list_completed_episodes,
    _merge_parts_to_output,
    _write_parts_manifest,
)


def _make_episode_table(episode_idx: int, start_index: int) -> pa.Table:
    return pa.table(
        {
            "index": np.array([start_index, start_index + 1], dtype=np.int64),
            "episode_index": np.array([episode_idx, episode_idx], dtype=np.int64),
            "frame_index": np.array([0, 1], dtype=np.int64),
            "progress_sparse": np.array([0.1, 0.2], dtype=np.float32),
        }
    )


def test_parts_manifest_mismatch_raises(tmp_path: Path):
    output_path = tmp_path / "sarm_progress.parquet"
    parts_dir = _get_parts_dir(output_path)

    _write_parts_manifest(
        parts_dir,
        dataset_repo_id="dummy/repo",
        reward_model_path="model-A",
        head_mode="sparse",
        stride=1,
        compute_sparse=True,
        compute_dense=False,
        num_episodes=2,
        reset_parts=False,
    )

    with pytest.raises(ValueError, match="different settings"):
        _write_parts_manifest(
            parts_dir,
            dataset_repo_id="dummy/repo",
            reward_model_path="model-A",
            head_mode="dense",
            stride=1,
            compute_sparse=True,
            compute_dense=False,
            num_episodes=2,
            reset_parts=False,
        )


def test_merge_parts_to_output(tmp_path: Path):
    output_path = tmp_path / "sarm_progress.parquet"
    parts_dir = _get_parts_dir(output_path)
    parts_dir.mkdir(parents=True, exist_ok=True)

    table0 = _make_episode_table(episode_idx=0, start_index=0)
    table1 = _make_episode_table(episode_idx=1, start_index=2)
    pq.write_table(table0, _episode_part_path(parts_dir, 0))
    pq.write_table(table1, _episode_part_path(parts_dir, 1))

    merged = _merge_parts_to_output(
        parts_dir=parts_dir,
        output_path=output_path,
        reward_model_path="model-A",
        expected_num_episodes=2,
    )

    assert output_path.exists()
    assert len(merged) == 4
    assert _list_completed_episodes(parts_dir) == {0, 1}
    assert merged.schema.metadata is not None
    assert merged.schema.metadata[b"reward_model_path"] == b"model-A"


def test_merge_parts_raises_when_missing_episode(tmp_path: Path):
    output_path = tmp_path / "sarm_progress.parquet"
    parts_dir = _get_parts_dir(output_path)
    parts_dir.mkdir(parents=True, exist_ok=True)

    table0 = _make_episode_table(episode_idx=0, start_index=0)
    pq.write_table(table0, _episode_part_path(parts_dir, 0))

    with pytest.raises(RuntimeError, match="Missing 1 episode part files"):
        _merge_parts_to_output(
            parts_dir=parts_dir,
            output_path=output_path,
            reward_model_path="model-A",
            expected_num_episodes=2,
        )
