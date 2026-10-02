"""Exact brute-force search: the ground truth for every recall number."""
from __future__ import annotations

from typing import Callable

import numpy as np

from vectorforge.engine.metrics import Metric, similarity


class FlatIndex:
    """Scans every stored vector in fixed-size chunks.

    Chunking bounds memory: a (n_queries x chunk) similarity block at a time,
    merged into a running top-k, instead of one (n_queries x N) matrix.
    """

    def __init__(self, get_vectors: Callable[[], np.ndarray], metric: Metric,
                 chunk_size: int = 65_536):
        self._get_vectors = get_vectors
        self.metric = metric
        self.chunk_size = chunk_size

    # The flat index reads the vector store directly, so it keeps no state.
    def add(self, slots: np.ndarray) -> None:
        pass

    def remove(self, slots: np.ndarray) -> None:
        pass

    def search_subset(self, queries: np.ndarray, k: int, slots: np.ndarray):
        """Exact search over just these slots (the planner's path for narrow filters)."""
        vectors = self._get_vectors()
        nq = len(queries)
        out_i = np.full((nq, k), -1, np.int64)
        out_s = np.full((nq, k), -np.inf, np.float32)
        if len(slots) == 0:
            return out_i, out_s
        for start in range(0, len(slots), self.chunk_size):
            part = slots[start:start + self.chunk_size]
            sims = similarity(queries, vectors[part], self.metric).astype(np.float32)
            cand_s = np.concatenate([out_s, sims], axis=1)
            cand_i = np.concatenate([out_i, np.broadcast_to(part, sims.shape)], axis=1)
            keep = np.argpartition(-cand_s, k - 1, axis=1)[:, :k]
            out_s = np.take_along_axis(cand_s, keep, 1)
            out_i = np.take_along_axis(cand_i, keep, 1)
        order = np.argsort(-out_s, axis=1, kind="stable")
        out_s = np.take_along_axis(out_s, order, 1)
        out_i = np.take_along_axis(out_i, order, 1)
        out_i[~np.isfinite(out_s)] = -1
        return out_i, out_s

    def search(self, queries: np.ndarray, k: int, allowed: np.ndarray | None = None, **_):
        vectors = self._get_vectors()
        nq, n = queries.shape[0], vectors.shape[0]
        best_s = np.full((nq, k), -np.inf, dtype=np.float32)
        best_i = np.full((nq, k), -1, dtype=np.int64)

        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            sims = similarity(queries, vectors[start:end], self.metric).astype(np.float32)
            if allowed is not None:
                sims[:, ~allowed[start:end]] = -np.inf

            # Top-k of this chunk (unsorted), then merge with the running best.
            kk = min(k, end - start)
            part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
            cand_s = np.concatenate([best_s, np.take_along_axis(sims, part, 1)], axis=1)
            cand_i = np.concatenate([best_i, part + start], axis=1)
            keep = np.argpartition(-cand_s, k - 1, axis=1)[:, :k]
            best_s = np.take_along_axis(cand_s, keep, 1)
            best_i = np.take_along_axis(cand_i, keep, 1)

        order = np.argsort(-best_s, axis=1, kind="stable")
        best_s = np.take_along_axis(best_s, order, 1)
        best_i = np.take_along_axis(best_i, order, 1)
        best_i[~np.isfinite(best_s)] = -1  # masked or missing results
        return best_i, best_s
