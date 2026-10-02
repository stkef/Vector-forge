"""Vectors live in one memory-mapped float32 file: row `slot` = one vector.

Contiguous float32 rows keep brute-force scans cache-friendly and let the
OS page cache do the work. The file grows by doubling.

Durability comes from the write-ahead log, not from this file: writes land
in the page cache and are forced to disk only at a checkpoint (sync()).
Rows are append-only, so a crash can only lose rows past the last
checkpoint, and the log replays exactly those.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, path: Path, dim: int, initial_capacity: int = 1024):
        self.path = Path(path)
        self.dim = dim
        self._row_bytes = dim * 4
        if self.path.exists() and self.path.stat().st_size > 0:
            size = self.path.stat().st_size
            if size % self._row_bytes:
                raise ValueError(f"{self.path} size {size} is not a multiple of the row size")
            self._capacity = size // self._row_bytes
        else:
            self._capacity = max(1, initial_capacity)
            with open(self.path, "wb") as f:
                f.truncate(self._capacity * self._row_bytes)
        self._mm: np.memmap | None = self._map()

    def _map(self) -> np.memmap:
        return np.memmap(self.path, dtype=np.float32, mode="r+",
                         shape=(self._capacity, self.dim))

    @property
    def capacity(self) -> int:
        return self._capacity

    def ensure_capacity(self, n: int) -> None:
        if n <= self._capacity:
            return
        new_cap = self._capacity
        while new_cap < n:
            new_cap *= 2
        # The mapping must be released before resizing (required on Windows).
        self._mm.flush()
        self._mm = None
        with open(self.path, "r+b") as f:
            f.truncate(new_cap * self._row_bytes)
        self._capacity = new_cap
        self._mm = self._map()

    def write(self, slots: np.ndarray, vectors: np.ndarray) -> None:
        if len(slots) == 0:
            return
        self.ensure_capacity(int(slots.max()) + 1)
        self._mm[slots] = vectors

    def view(self, n: int) -> np.ndarray:
        """Read-only view of the first n rows. Don't hold it across writes."""
        return self._mm[:n]

    def flush(self) -> None:
        self._mm.flush()

    def sync(self) -> None:
        """Force rows to disk. memmap.flush is msync on Linux but only
        FlushViewOfFile on Windows, so fsync the file handle as well."""
        self._mm.flush()
        with open(self.path, "r+b") as f:
            os.fsync(f.fileno())

    def close(self) -> None:
        if self._mm is not None:
            self._mm.flush()
            self._mm = None
