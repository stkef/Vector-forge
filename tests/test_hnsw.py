"""HNSW correctness: recall against the flat index, graph invariants,
tombstones, filters, persistence and crash catch-up."""
from collections import deque

import numpy as np
import pytest

from vectorforge import VectorForge
from vectorforge.engine.index import FlatIndex, HNSWIndex
from vectorforge.engine.metrics import Metric, prepare


from helpers import clustered, recall


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    return clustered(rng, 4_000, 32), clustered(rng, 100, 32)


@pytest.mark.parametrize("metric", list(Metric))
def test_recall_vs_flat(data, metric):
    x, q = (prepare(a, metric) for a in data)
    h = HNSWIndex(lambda: x, metric, M=12, ef_construction=100)
    h.add(np.arange(len(x)))
    truth, _ = FlatIndex(lambda: x, metric).search(q, 10, None)
    # Raw inner product is not a metric, so the graph's "closer to me than to
    # my neighbours" logic is weaker. hnswlib measures the same on this data
    # (0.85 at ef=64, 0.94 at ef=256). See docs/hnsw.md.
    ef, target = (256, 0.92) if metric is Metric.DOT else (64, 0.95)
    found, sims = h.search(q, 10, ef=ef)
    assert recall(found, truth, 10) >= target
    assert np.all(np.diff(sims, axis=1) <= 1e-6), "sorted best first"


def test_graph_invariants(data):
    x, _ = data
    M = 8
    h = HNSWIndex(lambda: x, Metric.L2, M=M, ef_construction=64)
    for chunk in np.array_split(np.arange(len(x)), 7):     # incremental inserts
        h.add(chunk)
    n = len(x)
    l0 = h.level0[:n]
    for node in range(n):
        nb = l0[node][l0[node] >= 0]
        assert len(nb) <= 2 * M
        assert node not in nb, "no self-loops"
        assert len(set(nb.tolist())) == len(nb), "no duplicate links"
        assert (l0[node][len(nb):] == -1).all(), "links are packed, then -1"
        for lvl in range(1, h.node_level[node] + 1):
            up = h.upper[h.upper_row[node], lvl - 1]
            up = up[up >= 0]
            assert len(up) <= M and all(h.node_level[e] >= lvl for e in up)

    # Layer 0 must be reachable from the entry point (the graph is navigable).
    seen, todo = {int(h.state[0])}, deque([int(h.state[0])])
    while todo:
        for e in l0[todo.popleft()]:
            if e >= 0 and e not in seen:
                seen.add(int(e))
                todo.append(int(e))
    assert len(seen) / n >= 0.999


def test_tombstones_and_filters(data):
    x, q = data
    h = HNSWIndex(lambda: x, Metric.L2, M=12, ef_construction=100)
    h.add(np.arange(len(x)))
    dead = np.arange(0, len(x), 3)
    h.remove(dead)
    found, _ = h.search(q, 10)
    assert not set(found.ravel().tolist()) & set(dead.tolist())

    alive = np.ones(len(x), bool)
    alive[dead] = False
    truth, _ = FlatIndex(lambda: x, Metric.L2).search(q, 10, alive)
    assert recall(found, truth, 10) >= 0.95

    allowed = np.zeros(len(x), bool)
    allowed[1::2] = True                       # a 50% filter on top of the deletes
    found, _ = h.search(q, 10, allowed)
    hits = found[found >= 0]
    assert (hits % 2 == 1).all() and not set(hits.tolist()) & set(dead.tolist())


def test_save_load_roundtrip(tmp_path, data):
    x, q = data
    a = HNSWIndex(lambda: x, Metric.COSINE, M=8)
    a.add(np.arange(len(x)))
    a.remove(np.array([5, 6]))
    a.save(tmp_path / "g.npz")
    b = HNSWIndex(lambda: x, Metric.COSINE, M=8)
    assert b.load(tmp_path / "g.npz")
    np.testing.assert_array_equal(a.search(q, 10)[0], b.search(q, 10)[0])
    assert HNSWIndex(lambda: x, Metric.COSINE, M=16).load(tmp_path / "g.npz") is False


def test_empty_and_tiny():
    x = np.eye(3, dtype=np.float32)
    h = HNSWIndex(lambda: x, Metric.DOT)
    ids, sims = h.search(x, 5)
    assert (ids == -1).all()
    h.add(np.arange(3))
    ids, _ = h.search(x, 5)
    assert ids[:, 0].tolist() == [0, 1, 2] and (ids[:, 3:] == -1).all()


# --------------------------------------------------------- collection level
def test_collection_hnsw_end_to_end(tmp_path, data):
    x, q = data
    with VectorForge(tmp_path) as db:
        col = db.create_collection("h", 32, "l2", params={"M": 12, "ef_construction": 100})
        col.upsert(list(range(len(x))), x, [{"i": i} for i in range(len(x))])
        approx = [[h.id for h in r] for r in col.search_batch(q, 10)]
        exact = [[h.id for h in r] for r in col.search_batch(q, 10, exact=True)]
        assert recall(approx, exact, 10) >= 0.95
        assert col.info()["index"]["nodes"] == len(x)
    assert (tmp_path / "collections" / "h" / "hnsw.npz").exists()

    with VectorForge(tmp_path) as db:               # reopen: graph loaded, not rebuilt
        col = db.get_collection("h")
        assert [[h.id for h in r] for r in col.search_batch(q, 10)] == approx


def test_crash_catch_up(tmp_path, data):
    """Graph saved, then more writes and deletes, then no clean close."""
    x, q = data
    db = VectorForge(tmp_path)
    col = db.create_collection("h", 32, "l2", params={"M": 12, "ef_construction": 100})
    col.upsert(list(range(3_000)), x[:3_000])
    col.checkpoint()
    col.upsert(list(range(3_000, 4_000)), x[3_000:])   # not in the saved graph
    col.delete(list(range(0, 300)))                    # deleted after the save
    col._vectors.flush()                               # simulate a crash: no close()

    with VectorForge(tmp_path) as db2:
        col2 = db2.get_collection("h")
        assert len(col2) == 3_700
        assert col2.info()["index"]["nodes"] == 4_000
        assert col2.info()["index"]["tombstones"] == 300
        approx = [[h.id for h in r] for r in col2.search_batch(q, 10)]
        exact = [[h.id for h in r] for r in col2.search_batch(q, 10, exact=True)]
        assert recall(approx, exact, 10) >= 0.95
        assert not {i for r in approx for i in r} & set(range(300))


def test_rebuild_after_many_deletes(db, data):
    x, q = data
    col = db.create_collection("h", 32, "l2", params={"M": 8})
    col.upsert(list(range(1_000)), x[:1_000])
    col.delete(list(range(400)))                       # 40% > threshold
    stats = col.info()["index"]
    assert stats["tombstones"] == 0 and stats["nodes"] == 600
    assert all(h.id >= 400 for h in col.search(q[0], 10))


def test_bad_hnsw_settings(db):
    with pytest.raises(ValueError):
        db.create_collection("a", 4, index="annoy")
    with pytest.raises(ValueError):
        db.create_collection("b", 4, params={"m": 16})
