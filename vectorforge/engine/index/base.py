"""The interface every index implements: flat, HNSW and IVF-PQ.

Indexes work in *slots* (row numbers in the vector file), never in user IDs.
The collection owns the slot <-> ID mapping, so an index is a pure search
structure that can be thrown away and rebuilt from the vector file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class Index(Protocol):
    FILE: str            # where save() writes, inside the collection folder
    trained: bool        # False only for IVF-PQ before training

    def add(self, slots: np.ndarray) -> None:
        """New vectors were written at these slots (slots are append-only)."""

    def remove(self, slots: np.ndarray) -> None:
        """Tombstone these slots: never returned, may still be traversed."""

    def search(self, queries: np.ndarray, k: int, allowed: np.ndarray | None,
               **params) -> tuple[np.ndarray, np.ndarray]:
        """Return (slots, similarities), each (n_queries, k), best first.
        `allowed` is a boolean mask over slots. Padding: slot -1, -inf."""

    @property
    def tombstone_ratio(self) -> float: ...

    def rebuild(self, live_slots: np.ndarray) -> None:
        """Drop tombstones by rebuilding from the live slots."""

    def in_index(self, n: int) -> np.ndarray:
        """Mask over the first n slots: which ones the index holds."""

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> bool:
        """False if the file doesn't match these settings (caller rebuilds)."""

    def stats(self) -> dict: ...
