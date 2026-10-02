"""A collection: fixed dimension and metric, points = (id, vector, payload).

Files (one folder per collection):

    collection.json   name, dimension, metric, index settings
    wal.log           write-ahead log: every write since the last checkpoint
    checkpoint.json   LSN of the last checkpoint (written atomically)
    vectors.f32       memory-mapped float32 rows, one per slot (append-only)
    payload.db        SQLite: id -> slot, payload JSON
    hnsw.npz / ivfpq.npz   saved index (a cache: rebuilt if missing)

Write path: validate -> append to the WAL and fsync (the acknowledgement)
-> apply to the vector file, SQLite and indexes in memory/page cache.

Checkpoint (the "snapshot"): fsync the vector file, checkpoint SQLite,
save the index, record the LSN, empty the WAL. Runs when the WAL passes
`wal_max_bytes`, on close, and after recovery.

Recovery on open: replay every WAL record after the checkpoint LSN, then
checkpoint. Records are blind writes, so replaying one that was already
applied changes nothing.

Vectors are append-only: an upsert of an existing ID writes a new slot and
the old one becomes a hole. A slot's contents never change, which is what
lets a saved index be trusted after a crash; it can only be behind.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from vectorforge.engine import wal as W
from vectorforge.engine.filters import PayloadIndex, validate_filter
from vectorforge.engine.index import FlatIndex, HNSWIndex, IVFPQIndex
from vectorforge.engine.metrics import Metric, as_matrix, prepare, to_user_score
from vectorforge.engine.payload_store import PayloadStore
from vectorforge.engine.planner import Plan, choose_plan
from vectorforge.engine.vector_store import VectorStore

META_FILE = "collection.json"
CHECKPOINT_FILE = "checkpoint.json"
INDEX_DEFAULTS = {
    "flat": {},
    "hnsw": {"M": 16, "ef_construction": 200, "ef_search": 64},
    "ivfpq": {"nlist": None, "m": None, "nprobe": 16, "rerank": 100,
              "train_size": 100_000, "train_after": 10_000},
}
STORAGE_DEFAULTS = {"fsync": True, "wal_max_bytes": 64 * 1024 * 1024}
REBUILD_TOMBSTONE_RATIO = 0.25
MAX_DIM = 65_536
MAX_ID = 2**63 - 1


@dataclass
class Point:
    id: int
    payload: dict[str, Any] = field(default_factory=dict)
    vector: list[float] | None = None


@dataclass
class ScoredPoint:
    id: int
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


def _check_ids(ids: Sequence[int]) -> list[int]:
    out = []
    for pid in ids:
        if isinstance(pid, bool) or not isinstance(pid, (int, np.integer)):
            raise TypeError(f"point IDs must be integers, got {pid!r}")
        pid = int(pid)
        if not 0 <= pid <= MAX_ID:
            raise ValueError(f"point ID {pid} is outside 0..2^63-1")
        out.append(pid)
    return out


def _write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Collection:
    # ---------------------------------------------------------------- create
    @classmethod
    def create(cls, path: Path, name: str, dim: int, metric: Metric | str,
               index: str = "hnsw", params: dict | None = None,
               storage: dict | None = None) -> "Collection":
        if not isinstance(dim, int) or isinstance(dim, bool) or not 1 <= dim <= MAX_DIM:
            raise ValueError(f"dimension must be an integer in 1..{MAX_DIM}")
        metric = Metric(metric)
        if index not in INDEX_DEFAULTS:
            raise ValueError(f"index must be one of {sorted(INDEX_DEFAULTS)}")
        cfg = {**INDEX_DEFAULTS[index], **(params or {})}
        unknown = set(cfg) - set(INDEX_DEFAULTS[index])
        if unknown:
            raise ValueError(f"unknown {index} settings: {sorted(unknown)}")
        if index == "ivfpq":
            IVFPQIndex(lambda: None, metric, dim, **cfg)        # validates m vs dim
        st = {**STORAGE_DEFAULTS, **(storage or {})}
        if set(st) - set(STORAGE_DEFAULTS):
            raise ValueError(f"unknown storage settings: {sorted(set(st) - set(STORAGE_DEFAULTS))}")
        path = Path(path)
        path.mkdir(parents=True)
        _write_json_atomic(path / META_FILE, {
            "name": name, "dim": dim, "metric": metric.value,
            "index": {"type": index, **cfg}, "storage": st,
        })
        return cls(path)

    # ------------------------------------------------------------------ open
    def __init__(self, path: Path):
        self.path = Path(path)
        meta = json.loads((self.path / META_FILE).read_text())
        self.name: str = meta["name"]
        self.dim: int = meta["dim"]
        self.metric = Metric(meta["metric"])
        self.index_config: dict = meta.get("index", {"type": "flat"})
        self.storage: dict = {**STORAGE_DEFAULTS, **meta.get("storage", {})}
        self._lock = threading.RLock()

        self._vectors = VectorStore(self.path / "vectors.f32", self.dim)
        self._payloads = PayloadStore(self.path / "payload.db")
        self._wal = W.WriteAheadLog(self.path / "wal.log", self.dim, fsync=self.storage["fsync"])
        ckpt_path = self.path / CHECKPOINT_FILE
        self._checkpoint_lsn = json.loads(ckpt_path.read_text())["lsn"] if ckpt_path.exists() else 0
        self._wal.last_lsn = max(self._wal.last_lsn, self._checkpoint_lsn)
        self.recovered_records = self._replay()

        self._next_slot = int(self._payloads.get_meta("next_slot", "0"))
        self._slot_ids = np.full(max(self._vectors.capacity, self._next_slot), -1, np.int64)
        rows = self._payloads.all_slots()
        for pid, slot in rows:
            self._slot_ids[slot] = pid
        self._count = len(rows)
        self._filters = PayloadIndex()
        for slot, payload in self._payloads.iter_payloads():
            self._filters.add(slot, payload)

        view = lambda: self._vectors.view(self._next_slot)  # noqa: E731
        self._flat = FlatIndex(view, self.metric)
        kind = self.index_config["type"]
        cfg = {k: v for k, v in self.index_config.items() if k != "type"}
        self._index = None
        if kind == "hnsw":
            self._index = HNSWIndex(view, self.metric, **cfg)
        elif kind == "ivfpq":
            self._index = IVFPQIndex(view, self.metric, self.dim, **cfg)
        if self._index is not None:
            self._open_index()
        if self.recovered_records:
            self.checkpoint()

    def _replay(self) -> int:
        """Re-apply WAL records newer than the checkpoint to the vector file
        and SQLite. In-memory state is built afterwards from SQLite."""
        todo = [r for r in self._wal.records if r.lsn > self._checkpoint_lsn]
        for rec in todo:
            if isinstance(rec, W.UpsertRecord):
                self._vectors.write(rec.slots, rec.vectors)
                self._payloads.upsert(zip(rec.ids.tolist(), rec.slots.tolist(), rec.payloads),
                                      int(rec.slots.max()) + 1)
            else:
                self._payloads.delete(rec.ids.tolist())
        return len(todo)

    def _live_slots(self) -> np.ndarray:
        return np.nonzero(self._slot_ids[: self._next_slot] >= 0)[0]

    def _open_index(self) -> None:
        """Load the saved index and catch up with writes made after it was saved."""
        idx = self._index
        path = self.path / idx.FILE
        if not (path.exists() and idx.load(path)):
            self._build_index()
            return
        n = self._next_slot
        present = idx.in_index(n)
        live = self._slot_ids[:n] >= 0
        idx.remove(np.nonzero(present & ~live)[0])
        idx.add(np.nonzero(live & ~present)[0])
        self._maybe_rebuild()

    def _build_index(self) -> None:
        idx = self._index
        if isinstance(idx, IVFPQIndex):
            if self._count >= idx.min_train:
                self.train_index()
            return
        idx.rebuild(self._live_slots())

    # ------------------------------------------------------------------ info
    def __len__(self) -> int:
        return self._count

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name, "dim": self.dim, "metric": self.metric.value,
                "points": self._count, "slots_used": self._next_slot,
                "holes": self._next_slot - self._count,
                "index": self._index.stats() if self._index else {"type": "flat"},
                "wal_bytes": self._wal.size, "last_lsn": self._wal.last_lsn,
                "checkpoint_lsn": self._checkpoint_lsn,
                "recovered_records": self.recovered_records,
                "payload_fields": self._filters.fields(),
            }

    # ----------------------------------------------------------------- write
    def upsert(self, ids: Sequence[int], vectors, payloads: Sequence[dict] | None = None) -> int:
        """Insert or replace points. Within one batch the last copy of an ID wins.
        Returns once the batch is in the fsynced write-ahead log."""
        ids = _check_ids(ids)
        vecs = prepare(as_matrix(vectors, self.dim), self.metric)
        if len(ids) != len(vecs):
            raise ValueError(f"{len(ids)} IDs but {len(vecs)} vectors")
        payloads = list(payloads) if payloads is not None else [{}] * len(ids)
        if len(payloads) != len(ids):
            raise ValueError(f"{len(ids)} IDs but {len(payloads)} payloads")
        if any(not isinstance(p, dict) for p in payloads):
            raise TypeError("each payload must be a dict")
        if not ids:
            return 0

        last = {pid: i for i, pid in enumerate(ids)}            # last occurrence wins
        pos = list(last.values())
        ids_arr = np.array([ids[i] for i in pos], np.int64)
        vecs = vecs[pos]
        payloads = [payloads[i] for i in pos]
        blobs = [json.dumps(p, allow_nan=False, ensure_ascii=False) for p in payloads]

        with self._lock:
            slots = np.arange(self._next_slot, self._next_slot + len(ids_arr), dtype=np.int64)
            self._wal.append(W.OP_UPSERT, W.encode_upsert(ids_arr, slots, vecs, blobs))
            # ---- acknowledged; everything below is replayable ----
            old = self._payloads.slots_for(ids_arr.tolist())
            old_slots = np.array(list(old.values()), np.int64)
            self._vectors.write(slots, vecs)
            self._payloads.upsert(zip(ids_arr.tolist(), slots.tolist(), blobs),
                                  int(slots[-1]) + 1)
            next_slot = int(slots[-1]) + 1
            if next_slot > len(self._slot_ids):
                grown = np.full(max(next_slot, 2 * len(self._slot_ids)), -1, np.int64)
                grown[: len(self._slot_ids)] = self._slot_ids
                self._slot_ids = grown
            self._slot_ids[old_slots] = -1
            self._slot_ids[slots] = ids_arr
            self._count += len(ids_arr) - len(old_slots)
            self._next_slot = next_slot
            for slot, p in zip(slots.tolist(), payloads):
                self._filters.add(slot, p)
            if self._index is not None:
                self._index.remove(old_slots)
                self._index.add(slots)
                if isinstance(self._index, IVFPQIndex) and not self._index.trained \
                        and self._count >= self._index.min_train:
                    self.train_index()
                self._maybe_rebuild()
            self._maybe_checkpoint()
        return len(ids_arr)

    def delete(self, ids: Sequence[int]) -> int:
        ids = list(dict.fromkeys(_check_ids(ids)))
        if not ids:
            return 0
        with self._lock:
            if not self._payloads.slots_for(ids):
                return 0
            self._wal.append(W.OP_DELETE, W.encode_delete(np.array(ids, np.int64)))
            removed = self._payloads.delete(ids)
            slots = np.array([s for _, s in removed], np.int64)
            self._slot_ids[slots] = -1
            self._count -= len(removed)
            if self._index is not None:
                self._index.remove(slots)
                self._maybe_rebuild()
            self._maybe_checkpoint()
            return len(removed)

    # --------------------------------------------------------------- upkeep
    def _maybe_rebuild(self) -> None:
        if self._index is not None and self._index.tombstone_ratio > REBUILD_TOMBSTONE_RATIO:
            self._index.rebuild(self._live_slots())

    def _maybe_checkpoint(self) -> None:
        if self._wal.size >= self.storage["wal_max_bytes"]:
            self.checkpoint()

    def train_index(self) -> None:
        """(Re)train IVF-PQ on the current live vectors and re-add them all.
        Best called once after a bulk load; HNSW needs no training."""
        with self._lock:
            if not isinstance(self._index, IVFPQIndex):
                return
            live = self._live_slots()
            rng = np.random.default_rng(0)
            if len(live) > self._index.train_size:
                sample = np.sort(rng.choice(live, self._index.train_size, replace=False))
            else:
                sample = live
            self._index.train(np.asarray(self._vectors.view(self._next_slot))[sample])
            self._index.add(live)

    def checkpoint(self) -> None:
        """Make the vector file, SQLite and the index durable, then empty the WAL."""
        with self._lock:
            self._vectors.sync()
            self._payloads.checkpoint()
            if self._index is not None:
                self._index.save(self.path / self._index.FILE)
            lsn = self._wal.last_lsn
            _write_json_atomic(self.path / CHECKPOINT_FILE, {"lsn": lsn})
            self._checkpoint_lsn = lsn
            self._wal.reset(lsn)

    # ------------------------------------------------------------------ read
    def get(self, ids: Sequence[int], with_vectors: bool = False) -> list[Point | None]:
        """Points in the order asked, None where an ID doesn't exist.
        Cosine collections return the stored (normalised) vector."""
        ids = _check_ids(ids)
        with self._lock:
            slots = self._payloads.slots_for(ids)
            payloads = self._payloads.payloads(list(slots))
            out: list[Point | None] = []
            for pid in ids:
                if pid not in slots:
                    out.append(None)
                    continue
                vec = self._vectors.view(self._next_slot)[slots[pid]].tolist() if with_vectors else None
                out.append(Point(pid, payloads[pid], vec))
            return out

    def scroll(self, after: int | None = None, limit: int = 50,
               with_vectors: bool = False) -> tuple[list[Point], int | None]:
        """Browse points in ID order. Returns (points, next `after` or None)."""
        limit = max(1, min(int(limit), 1_000))
        with self._lock:
            rows = self._payloads.scroll(after, limit)
            view = self._vectors.view(self._next_slot)
            pts = [Point(pid, payload, view[slot].tolist() if with_vectors else None)
                   for pid, slot, payload in rows]
        return pts, (rows[-1][0] if len(rows) == limit else None)

    def sample_ids(self, n: int, seed: int | None = None) -> list[int]:
        """Up to n random live point IDs."""
        with self._lock:
            live = self._live_slots()
            if len(live) == 0:
                return []
            pick = np.random.default_rng(seed).choice(live, min(n, len(live)), replace=False)
            return self._slot_ids[pick].tolist()

    def vector_of(self, pid: int) -> np.ndarray:
        with self._lock:
            slot = self._payloads.slots_for([pid]).get(pid)
            if slot is None:
                raise KeyError(f"point {pid} does not exist")
            return np.array(self._vectors.view(self._next_slot)[slot])

    def _plan(self, allowed: np.ndarray, filtered: bool, k: int, exact: bool,
              ef: int | None, nprobe: int | None) -> Plan:
        idx = self._index
        kind = self.index_config["type"]
        return choose_plan(
            index_type=kind, index_ready=idx is not None and idx.trained,
            candidates=int(allowed.sum()), live=self._count, filtered=filtered, exact=exact,
            ef=ef or self.index_config.get("ef_search", 64),
            nprobe=nprobe or self.index_config.get("nprobe", 16), k=k,
            graph_degree=2 * self.index_config.get("M", 16))

    def _allowed(self, flt: dict | None) -> np.ndarray:
        allowed = self._slot_ids[: self._next_slot] >= 0
        if flt is not None:
            allowed &= self._filters.mask(flt, self._next_slot)
        return allowed

    def _prep(self, vectors) -> np.ndarray:
        return prepare(as_matrix(vectors, self.dim), self.metric)

    def explain(self, k: int = 10, *, filter: dict | None = None, exact: bool = False,
                ef: int | None = None, nprobe: int | None = None) -> dict:
        """The plan a search with these arguments would use, without running it."""
        if filter is not None:
            validate_filter(filter)
        with self._lock:
            return self._plan(self._allowed(filter), filter is not None, k, exact,
                              ef, nprobe).to_dict()

    def search(self, vector, k: int = 10, **kwargs) -> list[ScoredPoint]:
        return self.search_batch([vector], k, **kwargs)[0]

    def search_batch(self, vectors, k: int = 10, *, filter: dict | None = None,
                     exact: bool = False, ef: int | None = None, nprobe: int | None = None,
                     rerank: int | None = None) -> list[list[ScoredPoint]]:
        """Top-k per query.

        filter  payload conditions, e.g. {"lang": "te", "year": {"gt": 2024}}
        exact   brute force (the recall reference)
        ef      HNSW beam width; nprobe / rerank: IVF-PQ settings (per call)
        """
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        if filter is not None:
            validate_filter(filter)
        queries = prepare(as_matrix(vectors, self.dim), self.metric)
        with self._lock:
            allowed = self._allowed(filter)
            plan = self._plan(allowed, filter is not None, k, exact, ef, nprobe)
            if plan.strategy == "empty":
                return [[] for _ in range(len(queries))]
            if plan.strategy == "flat_scan":
                slots, sims = self._flat.search(queries, k, allowed)
            elif plan.strategy == "subset_scan":
                slots, sims = self._flat.search_subset(queries, k, np.nonzero(allowed)[0])
            elif plan.strategy == "hnsw":
                slots, sims = self._index.search(queries, k, allowed, ef=plan.params["ef"])
            else:
                slots, sims = self._index.search(queries, k, allowed,
                                                 nprobe=plan.params["nprobe"], rerank=rerank)
            scores = to_user_score(sims, self.metric)
            hit_ids = np.where(slots >= 0, self._slot_ids[np.maximum(slots, 0)], -1)
            payloads = self._payloads.payloads(np.unique(hit_ids[hit_ids >= 0]).tolist())

        return [
            [ScoredPoint(int(pid), float(s), payloads[int(pid)])
             for pid, s in zip(row_ids, row_scores) if pid >= 0]
            for row_ids, row_scores in zip(hit_ids, scores)
        ]

    # -------------------------------------------------------------- lifecycle
    def close(self, checkpoint: bool = True) -> None:
        with self._lock:
            if checkpoint:
                self.checkpoint()
            self._wal.close()
            self._vectors.close()
            self._payloads.close()
