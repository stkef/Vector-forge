"""IVF-PQ: an inverted file of clusters, each vector compressed to m bytes.

Train:
  1. k-means on a sample -> `nlist` coarse centroids. Every vector goes into
     the list of its nearest centroid.
  2. Each vector is stored as its residual r = x - centroid, split into m
     sub-vectors. Each sub-vector is replaced by the id (0..255) of its
     nearest codeword in a 256-entry codebook learned for that subspace.
     A 128-d float32 vector (512 bytes) becomes 32 bytes at the default m = 32.

Search:
  1. Score the query against all centroids; open the best `nprobe` lists.
  2. Asymmetric distance (ADC): the query stays exact. Precompute a lookup
     table LUT[j][c] = distance between query sub-vector j and codeword c,
     then each stored vector's distance is m table lookups and adds.
  3. Optional re-rank: take the best `rerank` candidates and recompute exact
     distances from the raw vectors on disk. Compression loses ranking
     detail; re-ranking buys most of it back.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
from numba import njit

from vectorforge.engine.index.kmeans import assign, kmeans
from vectorforge.engine.metrics import Metric, similarity

FORMAT_VERSION = 1
KSUB = 256  # 8-bit codes


@njit(cache=True, fastmath=True)
def _adc(codes, lut):
    n, m = codes.shape
    out = np.empty(n, np.float32)
    for i in range(n):
        s = np.float32(0.0)
        for j in range(m):
            s += lut[j, codes[i, j]]
        out[i] = s
    return out


@njit(cache=True, fastmath=True)
def _lut_l2(r, books):
    """LUT[j, c] = ||r_j - codeword_jc||^2 for one residual r of shape (m, dsub)."""
    m, ksub, dsub = books.shape
    out = np.empty((m, ksub), np.float32)
    for j in range(m):
        for c in range(ksub):
            s = np.float32(0.0)
            for d in range(dsub):
                t = r[j, d] - books[j, c, d]
                s += t * t
            out[j, c] = s
    return out


def default_m(dim: int) -> int:
    """Largest divisor of dim that is <= dim/4, so codes take <= 1/16 of the
    float32 size (4-d sub-vectors). At dim/8 (1/32) recall@10 with re-rank 100
    dropped from ~0.99 to ~0.88 in tests/test_ivfpq.py; see docs/ivfpq.md."""
    for m in range(max(1, dim // 4), 0, -1):
        if dim % m == 0:
            return m
    return 1


class _List:
    __slots__ = ("ids", "codes", "n")

    def __init__(self, m: int):
        self.ids = np.empty(16, np.int32)
        self.codes = np.empty((16, m), np.uint8)
        self.n = 0

    def extend(self, ids: np.ndarray, codes: np.ndarray) -> None:
        need = self.n + len(ids)
        if need > len(self.ids):
            cap = max(need, 2 * len(self.ids))
            self.ids = np.resize(self.ids, cap)
            grown = np.empty((cap, self.codes.shape[1]), np.uint8)
            grown[: self.n] = self.codes[: self.n]
            self.codes = grown
        self.ids[self.n:need] = ids
        self.codes[self.n:need] = codes
        self.n = need


class IVFPQIndex:
    FILE = "ivfpq.npz"

    def __init__(self, get_vectors: Callable[[], np.ndarray], metric: Metric, dim: int,
                 nlist: int | None = None, m: int | None = None, nprobe: int = 16,
                 rerank: int = 100, train_size: int = 100_000, train_after: int = 10_000,
                 seed: int = 0):
        self._get_vectors = get_vectors
        self.metric = metric
        self.dim = dim
        self.m = m or default_m(dim)
        if dim % self.m:
            raise ValueError(f"m={self.m} must divide the dimension {dim}")
        self.dsub = dim // self.m
        self.nlist_setting = nlist
        self.nprobe = nprobe
        self.rerank = rerank
        self.train_size = train_size
        self.train_after = train_after
        self.seed = seed
        self.centroids: np.ndarray | None = None
        self.codebooks: np.ndarray | None = None   # (m, 256, dsub)
        self._clear_lists()

    def _clear_lists(self) -> None:
        nlist = 0 if self.centroids is None else len(self.centroids)
        self.lists = [_List(self.m) for _ in range(nlist)]
        self.present = np.zeros(0, bool)
        self.deleted = np.zeros(0, bool)
        self.n_nodes = 0
        self.n_deleted = 0

    @property
    def trained(self) -> bool:
        return self.centroids is not None

    @property
    def min_train(self) -> int:
        """Points needed before the collection trains: train_after, but never
        fewer than one PQ codebook (256 codewords) needs."""
        return max(self.train_after, KSUB)

    @property
    def nlist(self) -> int:
        return 0 if self.centroids is None else len(self.centroids)

    # ------------------------------------------------------------- train
    def train(self, sample: np.ndarray) -> None:
        sample = np.ascontiguousarray(sample, np.float32)
        n = len(sample)
        nlist = self.nlist_setting or int(np.clip(4 * math.sqrt(n), 16, 65_536))
        nlist = min(nlist, max(1, n // 39))            # ~39 points per centroid minimum
        if n < KSUB:
            raise ValueError(f"IVF-PQ needs at least {KSUB} vectors to train, got {n}")
        self.centroids = kmeans(sample, nlist, iters=20, seed=self.seed)
        residuals = sample - self.centroids[assign(sample, self.centroids)]
        books = np.empty((self.m, KSUB, self.dsub), np.float32)
        for j in range(self.m):
            sub = residuals[:, j * self.dsub:(j + 1) * self.dsub]
            books[j] = kmeans(sub, KSUB, iters=15, seed=self.seed + j + 1, max_points=65_536)
        self.codebooks = books
        self._clear_lists()

    def _encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lists = assign(x, self.centroids)
        r = x - self.centroids[lists]
        codes = np.empty((len(x), self.m), np.uint8)
        for j in range(self.m):
            codes[:, j] = assign(np.ascontiguousarray(r[:, j * self.dsub:(j + 1) * self.dsub]),
                                 self.codebooks[j])
        return lists, codes

    # --------------------------------------------------------- Index API
    def _ensure(self, n: int) -> None:
        if n > len(self.present):
            cap = max(n, 2 * len(self.present), 1024)
            for name in ("present", "deleted"):
                grown = np.zeros(cap, bool)
                old = getattr(self, name)
                grown[: len(old)] = old
                setattr(self, name, grown)

    def add(self, slots: np.ndarray) -> None:
        """No-op until trained: the collection searches exactly until then."""
        slots = np.asarray(slots, np.int64)
        if not self.trained or len(slots) == 0:
            return
        self._ensure(int(slots.max()) + 1)
        vectors = np.asarray(self._get_vectors())
        for s in range(0, len(slots), 65_536):
            chunk = slots[s:s + 65_536]
            lists, codes = self._encode(vectors[chunk])
            order = np.argsort(lists, kind="stable")
            lists, codes, chunk = lists[order], codes[order], chunk[order]
            bounds = np.nonzero(np.diff(lists))[0] + 1
            for part in np.split(np.arange(len(chunk)), bounds):
                self.lists[lists[part[0]]].extend(chunk[part].astype(np.int32), codes[part])
        self.present[slots] = True
        self.n_nodes += len(slots)

    def remove(self, slots: np.ndarray) -> None:
        slots = np.asarray(slots, np.int64)
        slots = slots[slots < len(self.present)]
        slots = slots[self.present[slots] & ~self.deleted[slots]]
        self.deleted[slots] = True
        self.n_deleted += len(slots)

    def search(self, queries: np.ndarray, k: int, allowed: np.ndarray | None = None,
               nprobe: int | None = None, rerank: int | None = None, **_):
        nq = len(queries)
        out_i = np.full((nq, k), -1, np.int64)
        out_s = np.full((nq, k), -np.inf, np.float32)
        if not self.trained or self.n_nodes == 0:
            return out_i, out_s
        nprobe = min(nprobe or self.nprobe, self.nlist)
        rerank = self.rerank if rerank is None else rerank
        mask = ~self.deleted
        if allowed is not None:
            m = min(len(mask), len(allowed))
            mask[:m] &= allowed[:m]
            mask[m:] = False
        vectors = np.asarray(self._get_vectors())
        l2 = self.metric is Metric.L2

        coarse = similarity(queries, self.centroids, self.metric)
        probes = np.argpartition(-coarse, nprobe - 1, axis=1)[:, :nprobe]
        for qi in range(nq):
            q = queries[qi]
            qsub = q.reshape(self.m, self.dsub)
            if not l2:  # inner product: one table per query, bias q.c per list
                lut = np.einsum("jcd,jd->jc", self.codebooks, qsub).astype(np.float32)
            got_s, got_i = [], []
            for lst in probes[qi]:
                L = self.lists[lst]
                if L.n == 0:
                    continue
                ids = L.ids[: L.n]
                if l2:
                    r = (q - self.centroids[lst]).reshape(self.m, self.dsub)
                    s = -_adc(L.codes[: L.n], _lut_l2(r, self.codebooks))
                else:
                    s = _adc(L.codes[: L.n], lut) + coarse[qi, lst]
                keep = mask[ids]
                got_s.append(s[keep])
                got_i.append(ids[keep])
            if not got_s:
                continue
            S = np.concatenate(got_s)
            I = np.concatenate(got_i).astype(np.int64)
            depth = min(len(S), max(k, rerank or 0))
            top = np.argpartition(-S, depth - 1)[:depth]
            S, I = S[top], I[top]
            if rerank:
                S = similarity(q[None, :], vectors[I], self.metric)[0].astype(np.float32)
            take = min(k, len(S))
            best = np.argsort(-S, kind="stable")[:take]
            out_i[qi, :take] = I[best]
            out_s[qi, :take] = S[best]
        return out_i, out_s

    # ------------------------------------------------------------ upkeep
    @property
    def tombstone_ratio(self) -> float:
        return self.n_deleted / self.n_nodes if self.n_nodes else 0.0

    def rebuild(self, live_slots: np.ndarray) -> None:
        """Drop tombstones by re-adding live slots with the same codebooks."""
        self._clear_lists()
        self.add(np.sort(np.asarray(live_slots, np.int64)))

    def in_index(self, n: int) -> np.ndarray:
        out = np.zeros(n, bool)
        m = min(n, len(self.present))
        out[:m] = self.present[:m]
        return out

    def stats(self) -> dict:
        codes = sum(L.n for L in self.lists) * self.m
        ids = sum(L.n for L in self.lists) * 4
        model = 0 if not self.trained else self.centroids.nbytes + self.codebooks.nbytes
        sizes = np.array([L.n for L in self.lists]) if self.lists else np.zeros(1)
        return {
            "type": "ivfpq", "trained": self.trained, "nlist": self.nlist, "m": self.m,
            "nprobe": self.nprobe, "rerank": self.rerank,
            "nodes": self.n_nodes, "tombstones": self.n_deleted,
            "bytes_per_vector": self.m + 4,
            "memory_bytes": int(codes + ids + model),
            "list_size_max": int(sizes.max()), "list_size_mean": round(float(sizes.mean()), 1),
        }

    # -------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        if not self.trained:
            return
        path = Path(path)
        sizes = np.array([L.n for L in self.lists], np.int64)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, centroids=self.centroids, codebooks=self.codebooks, sizes=sizes,
                 ids=np.concatenate([L.ids[: L.n] for L in self.lists]),
                 codes=np.concatenate([L.codes[: L.n] for L in self.lists]),
                 present=self.present, deleted=self.deleted,
                 counts=np.array([self.n_nodes, self.n_deleted], np.int64),
                 params=np.array([self.m, FORMAT_VERSION], np.int64),
                 metric=np.array(self.metric.value))
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def load(self, path: Path) -> bool:
        with np.load(path) as z:
            m, version = z["params"].tolist()
            if (m, version) != (self.m, FORMAT_VERSION) or str(z["metric"]) != self.metric.value:
                return False
            self.centroids = z["centroids"]
            self.codebooks = z["codebooks"]
            self._clear_lists()
            ids, codes, start = z["ids"], z["codes"], 0
            for L, n in zip(self.lists, z["sizes"].tolist()):
                if n:
                    L.extend(ids[start:start + n], codes[start:start + n])
                start += n
            self.present = z["present"].copy()
            self.deleted = z["deleted"].copy()
            self.n_nodes, self.n_deleted = z["counts"].tolist()
        return True
