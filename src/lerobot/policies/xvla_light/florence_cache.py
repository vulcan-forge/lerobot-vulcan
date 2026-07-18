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

"""Disk-backed cache for deterministic, frozen Florence features."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import torch
from safetensors.torch import load as load_safetensors, save as save_safetensors

_SCHEMA_VERSION = "1"


class FlorenceFeatureCache:
    """Store per-dataset-index Florence outputs in a single SQLite database.

    The connection is opened lazily so constructing or serializing a policy does not
    touch the filesystem. SQLite WAL mode supports concurrent readers and independent
    training processes; cache rows are immutable for a given signature.
    """

    def __init__(self, path: str | Path, signature: str) -> None:
        self.path = Path(path).expanduser()
        self.signature = signature
        self._connection: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection

        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=60.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS features (sample_index INTEGER PRIMARY KEY, payload BLOB NOT NULL)"
        )

        expected = {"schema_version": _SCHEMA_VERSION, "signature": self.signature}
        connection.executemany(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)",
            expected.items(),
        )
        connection.commit()
        metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
        for key, value in expected.items():
            if metadata.get(key) != value:
                connection.close()
                raise ValueError(
                    f"Florence cache {self.path} has incompatible {key}: "
                    f"expected {value!r}, found {metadata.get(key)!r}. Use a new cache path."
                )

        self._connection = connection
        return connection

    @staticmethod
    def _serialize(vlm_features: torch.Tensor, aux_visual_inputs: torch.Tensor) -> bytes:
        return save_safetensors(
            {
                "vlm_features": vlm_features.detach().to(device="cpu").contiguous(),
                "aux_visual_inputs": aux_visual_inputs.detach().to(device="cpu").contiguous(),
            }
        )

    @staticmethod
    def _deserialize(payload: bytes) -> dict[str, torch.Tensor]:
        return load_safetensors(payload)

    def get_many(self, sample_indices: list[int]) -> dict[int, dict[str, torch.Tensor]]:
        """Return cached features for the requested indices; cache misses are omitted."""
        if not sample_indices:
            return {}
        unique_indices = list(dict.fromkeys(sample_indices))
        placeholders = ",".join("?" for _ in unique_indices)
        rows = self._connect().execute(
            f"SELECT sample_index, payload FROM features WHERE sample_index IN ({placeholders})",  # noqa: S608
            unique_indices,
        )
        return {int(index): self._deserialize(payload) for index, payload in rows}

    def put_many(
        self,
        features: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Insert newly computed features without replacing existing cache rows."""
        if not features:
            return
        rows = [
            (index, self._serialize(vlm_features, aux_visual_inputs))
            for index, (vlm_features, aux_visual_inputs) in features.items()
        ]
        connection = self._connect()
        connection.executemany(
            "INSERT OR IGNORE INTO features(sample_index, payload) VALUES (?, ?)",
            rows,
        )
        connection.commit()

    def count(self) -> int:
        return int(self._connect().execute("SELECT COUNT(*) FROM features").fetchone()[0])

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __del__(self) -> None:
        self.close()
