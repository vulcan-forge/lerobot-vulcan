#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""SIVA2-owned disk cache for its frozen Florence feature boundary."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import torch
from safetensors.torch import load as load_safetensors, save as save_safetensors

_SCHEMA_VERSION = "1"


class SIVA2FlorenceFeatureCache:
    """Store complete per-sample SIVA2 Florence outputs in SQLite."""

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
            "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)", expected.items()
        )
        connection.commit()
        metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
        for key, value in expected.items():
            if metadata.get(key) != value:
                connection.close()
                raise ValueError(
                    f"SIVA2 Florence cache {self.path} has incompatible {key}: "
                    f"expected {value!r}, found {metadata.get(key)!r}. Use a new cache path."
                )
        self._connection = connection
        return connection

    @staticmethod
    def _serialize(features: dict[str, torch.Tensor]) -> bytes:
        return save_safetensors(
            {
                name: tensor.detach().to(device="cpu").contiguous()
                for name, tensor in features.items()
            }
        )

    def get_many(self, sample_indices: list[int]) -> dict[int, dict[str, torch.Tensor]]:
        if not sample_indices:
            return {}
        unique_indices = list(dict.fromkeys(sample_indices))
        placeholders = ",".join("?" for _ in unique_indices)
        rows = self._connect().execute(
            f"SELECT sample_index, payload FROM features WHERE sample_index IN ({placeholders})",  # noqa: S608
            unique_indices,
        )
        return {int(index): load_safetensors(payload) for index, payload in rows}

    def put_many(self, features: dict[int, dict[str, torch.Tensor]]) -> None:
        if not features:
            return
        connection = self._connect()
        connection.executemany(
            "INSERT OR IGNORE INTO features(sample_index, payload) VALUES (?, ?)",
            [(index, self._serialize(sample)) for index, sample in features.items()],
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
