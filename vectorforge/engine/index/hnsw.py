"""HNSW index: a multi-layer proximity graph searched greedily from the top.

Parameters (same names and meaning as hnswlib):
    M                links per node on layers >= 1 (2*M on layer 0)
    ef_construction  beam width while inserting (graph quality vs build time)
    ef_search        beam width while querying  (recall vs latency)

Deletes are tombstones: the node stays in the graph so search can still
route through it, but it is never returned. Past a tombstone ratio the
collection rebuilds the graph from the live vectors.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable

import numpy as np

from vectorforge.engine.index import _hnsw_kernels as K
from vectorforge.engine.metrics import Metric

MAX_LEVELS = 16
FORMAT_VERSION = 1


class HNSWIndex:
    FILE = "hnsw.npz"
    trained = True  # a graph needs no training step

    def __init__(self, get_vectors: Callable[[], np.ndarray], metric: Metric,
                 M: int = 16, ef_construction: int = 200, ef_search: int = 64,
                 seed: int = 0):
        if M < 2:
            raise ValueError("M must be at least 2")
        self._get_vectors = get_vectors
        self.metric = metric
        self.M = M
        self.ef_construction = max(ef_construction, M)
        self.ef_search = ef_search
        self._kmetric = K.METRIC_L2 if metric is Metric.L2 else K.METRIC_IP
        self._mL = 1.0 / math.log(M)
        self._rng = np.random.default_rng(seed)
        self._reset()

    def _reset(self) -> None:
        self._cap = 0
        self.level0 = np.empty((0, 2 * self.M), np.int32)
        self.upper = np.empty((0, MAX_LEVELS, self.M), np.int32)
        self.upper_row = np.empty(0, np.int32)
        self.node_level = np.empty(0, np.int8)
        self.deleted = np.empty(0, np.bool_)
        self.visited = np.empty(0, np.uint32)
        self.state = np.array([-1, -1, 0, 0], np.int64)
        self.n_nodes = 0
        self.n_deleted = 0

    # ------------------------------------------------------------ growth
    def _ensure(self, n: int) -> None:
        if n <= self._cap:
            return
        cap = max(n, 2 * self._cap, 1024)

        def grow(a, fill, shape_tail=()):
            out = np.full((cap, *shape_tail), fill, a.dtype)
            out[: len(a)] = a
            return out

        self.level0 = grow(self.level0, -1, (2 * self.M,))
        self.upper_row = grow(self.upper_row, -1)
        self.node_level = grow(self.node_level, -1)
        self.deleted = grow(self.deleted, False)
        self.visited = grow(self.visited, 0)
        self._cap = cap

    def _ensure_upper(self, rows: int) -> None:
        if rows <= len(self.upper):
            return
        out = np.full((max(rows, 2 * len(self.upper), 64), MAX_LEVELS, self.M), -1, np.int32)
        out[: len(self.upper)] = self.upper
        self.upper = out

    def _vectors(self) -> np.ndarray:
        return np.asarray(self._get_vectors())  # plain ndarray view for Numba

    # ---------------------------------------------------------- Index API
    def add(self, slots: np.ndarray) -> None:
        slots = np.asarray(slots, np.int32)
        if len(slots) == 0:
            return
        self._ensure(int(slots.max()) + 1)
        if (self.node_level[slots] >= 0).any():
            raise ValueError("slot already in the graph; vectors are append-only")
        u = 1.0 - self._rng.random(len(slots))
        levels = np.minimum(np.floor(-np.log(u) * self._mL), MAX_LEVELS - 1).astype(np.int8)
        self._ensure_upper(int(self.state[K.N_UPPER]) + int((levels > 0).sum()))
        K.insert_many(self._vectors(), slots, levels, self.state, self.level0, self.upper,
                      self.upper_row, self.node_level, self.visited, self.M,
                      self.ef_construction, self._kmetric)
        self.n_nodes += len(slots)

    def remove(self, slots: np.ndarray) -> None:
        slots = np.asarray(slots, np.int64)
        slots = slots[(slots < self._cap)]
        slots = slots[(self.node_level[slots] >= 0) & ~self.deleted[slots]]
        self.deleted[slots] = True
        self.n_deleted += len(slots)

    def search(self, queries: np.ndarray, k: int, allowed: np.ndarray | None = None,
               ef: int | None = None, **_):
        nq = len(queries)
        if self.n_nodes == 0:
            return np.full((nq, k), -1, np.int64), np.full((nq, k), -np.inf, np.float32)
        mask = ~self.deleted
        if allowed is not None:
            m = min(len(mask), len(allowed))
            mask[:m] &= allowed[:m]
            mask[m:] = False
        use_mask = bool(self.n_deleted) or allowed is not None
        ids, dists = K.search_many(np.ascontiguousarray(queries, np.float32), k,
                                   ef or self.ef_search, self._vectors(), self.level0,
                                   self.upper, self.upper_row, self.visited, self.state,
                                   self._kmetric, mask, use_mask)
        sims = np.where(ids >= 0, -dists, -np.inf).astype(np.float32)
        return ids, sims

    # ----------------------------------------------------------- upkeep
    @property
    def tombstone_ratio(self) -> float:
        return self.n_deleted / self.n_nodes if self.n_nodes else 0.0

    def rebuild(self, live_slots: np.ndarray) -> None:
        """Throw the graph away and insert the live slots again."""
        self._reset()
        self.add(np.sort(np.asarray(live_slots, np.int32)))

    def stats(self) -> dict:
        live = self.node_level[: self._cap] >= 0
        deg0 = (self.level0[: self._cap][live] >= 0).sum(axis=1) if live.any() else np.zeros(1)
        return {
            "type": "hnsw",
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "nodes": self.n_nodes,
            "tombstones": self.n_deleted,
            "max_level": int(self.state[K.MAX_LEVEL]),
            "avg_degree_layer0": round(float(deg0.mean()), 2),
            "memory_bytes": int(self.level0.nbytes + self.upper.nbytes + self.upper_row.nbytes
                                + self.node_level.nbytes + self.deleted.nbytes
                                + self.visited.nbytes),
        }

    # -------------------------------------------------------- persistence
    def params(self) -> dict:
        return {"M": self.M, "ef_construction": self.ef_construction,
                "metric": self.metric.value, "version": FORMAT_VERSION}

    def save(self, path: Path) -> None:
        """Atomic write: a crash mid-save leaves the previous file intact."""
        path = Path(path)
        nodes = np.nonzero(self.node_level[: self._cap] >= 0)[0]
        n = int(nodes[-1]) + 1 if len(nodes) else 0
        rows = int(self.state[K.N_UPPER])
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, level0=self.level0[:n], upper=self.upper[:rows],
                 upper_row=self.upper_row[:n], node_level=self.node_level[:n],
                 deleted=self.deleted[:n], state=self.state,
                 counts=np.array([self.n_nodes, self.n_deleted], np.int64),
                 params=np.array([self.M, self.ef_construction, FORMAT_VERSION], np.int64),
                 metric=np.array(self.metric.value))
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def load(self, path: Path) -> bool:
        """Load a saved graph. Returns False (graph left empty) if incompatible."""
        with np.load(path) as z:
            M, efc, version = z["params"].tolist()
            if (M, version) != (self.M, FORMAT_VERSION) or str(z["metric"]) != self.metric.value:
                return False
            self._reset()
            n = len(z["node_level"])
            self._ensure(max(n, 1))
            self.level0[:n] = z["level0"]
            self.upper_row[:n] = z["upper_row"]
            self.node_level[:n] = z["node_level"]
            self.deleted[:n] = z["deleted"]
            self._ensure_upper(len(z["upper"]))
            self.upper[: len(z["upper"])] = z["upper"]
            self.state[:] = z["state"]
            self.state[K.TAG] = 0
            self.n_nodes, self.n_deleted = z["counts"].tolist()
        return True

    def in_index(self, n: int) -> np.ndarray:
        """Boolean mask over the first n slots: which ones the graph holds."""
        out = np.zeros(n, bool)
        m = min(n, self._cap)
        out[:m] = self.node_level[:m] >= 0
        return out
