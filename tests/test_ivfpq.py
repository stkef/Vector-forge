"""IVF-PQ: k-means, compression, recall with re-rank, training lifecycle."""
import numpy as np
import pytest
from helpers import clustered, ids, recall

from vectorforge import VectorForge
from vectorforge.engine.index import FlatIndex, IVFPQIndex
from vectorforge.engine.index.ivfpq import default_m
from vectorforge.engine.index.kmeans import assign, kmeans
from vectorforge.engine.metrics import Metric, prepare


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    return clustered(rng, 6_000, 32), clustered(rng, 100, 32)


def test_kmeans_finds_separated_clusters():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], np.float32)
    x = (centers[rng.integers(0, 4, 2_000)] + rng.standard_normal((2_000, 2))).astype(np.float32)
    c = kmeans(x, 4, seed=1)
    found = sorted(map(tuple, np.round(c / 100).astype(int).tolist()))
    assert found == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert len(np.unique(assign(x, c))) == 4


def test_kmeans_no_empty_clusters_with_duplicates():
    x = np.repeat(np.eye(3, dtype=np.float32), 100, axis=0)     # only 3 distinct points
    c = kmeans(x, 8, iters=5)
    assert np.isfinite(c).all() and c.shape == (8, 3)


def test_default_m():
    assert default_m(128) == 32 and default_m(768) == 192 and default_m(100) == 25
    assert default_m(7) == 1


@pytest.mark.parametrize("metric", [Metric.L2, Metric.COSINE])
def test_recall_with_rerank(data, metric):
    x, q = (prepare(a, metric) for a in data)
    idx = IVFPQIndex(lambda: x, metric, 32, nlist=64)
    idx.train(x)
    idx.add(np.arange(len(x)))
    truth, _ = FlatIndex(lambda: x, metric).search(q, 10, None)
    found, sims = idx.search(q, 10, nprobe=8, rerank=100)
    assert recall(found, truth, 10) >= 0.95
    assert np.all(np.diff(sims, axis=1) <= 1e-6)
    coarse, _ = idx.search(q, 10, nprobe=8, rerank=0)          # compressed scores only
    assert recall(coarse, truth, 10) < recall(found, truth, 10)


def test_memory_is_a_fraction_of_raw(data):
    x, _ = data
    idx = IVFPQIndex(lambda: x, Metric.L2, 32)
    idx.train(x)
    idx.add(np.arange(len(x)))
    st = idx.stats()
    raw = x.nbytes
    codes_and_ids = st["bytes_per_vector"] * len(x)
    assert codes_and_ids <= raw / 8               # 8 + 4 bytes/vector vs 128 raw here
    assert st["nodes"] == len(x)


def test_tombstones_filters_save_load(tmp_path, data):
    x, q = data
    idx = IVFPQIndex(lambda: x, Metric.L2, 32, nlist=64)
    idx.train(x)
    idx.add(np.arange(len(x)))
    idx.remove(np.arange(0, len(x), 2))
    found, _ = idx.search(q, 10, nprobe=16)
    assert (found[found >= 0] % 2 == 1).all()
    allowed = np.zeros(len(x), bool)
    allowed[1::4] = True
    found, _ = idx.search(q, 10, allowed, nprobe=16)
    assert (found[found >= 0] % 4 == 1).all()

    idx.save(tmp_path / "i.npz")
    other = IVFPQIndex(lambda: x, Metric.L2, 32, nlist=64)
    assert other.load(tmp_path / "i.npz")
    np.testing.assert_array_equal(other.search(q, 10)[0], idx.search(q, 10)[0])


def test_collection_lifecycle(tmp_path, data):
    """Exact until trained; trains itself at train_after; survives reopen."""
    x, q = data
    with VectorForge(tmp_path) as db:
        col = db.create_collection("v", 32, "l2", index="ivfpq",
                                   params={"nlist": 32, "train_after": 2_000})
        col.upsert(list(range(1_000)), x[:1_000])
        assert col.explain()["strategy"] == "flat_scan"
        assert not col.info()["index"]["trained"]
        col.upsert(list(range(1_000, 6_000)), x[1_000:])
        assert col.info()["index"]["trained"] and col.info()["index"]["nodes"] == 6_000
        assert col.explain()["strategy"] == "ivfpq"
        approx = ids(col.search_batch(q, 10))
        exact = ids(col.search_batch(q, 10, exact=True))
        assert recall(approx, exact, 10) >= 0.95
        col.train_index()                                        # explicit retrain
        assert col.info()["index"]["nodes"] == 6_000

    with VectorForge(tmp_path) as db:
        col = db.get_collection("v")
        assert col.info()["index"]["trained"]
        assert recall(ids(col.search_batch(q, 10)), exact, 10) >= 0.95


def test_m_must_divide_dim(db):
    with pytest.raises(ValueError):
        db.create_collection("v", 30, index="ivfpq", params={"m": 7})
