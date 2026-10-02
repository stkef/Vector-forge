import math

import numpy as np
import pytest

from vectorforge import Metric, VectorForge


def test_create_list_drop(db):
    db.create_collection("docs", 4, "cosine")
    db.create_collection("imgs", 8, Metric.L2)
    assert db.list_collections() == ["docs", "imgs"]
    with pytest.raises(FileExistsError):
        db.create_collection("docs", 4)
    assert db.drop_collection("docs") is True
    assert db.drop_collection("docs") is False
    assert db.list_collections() == ["imgs"]
    with pytest.raises(KeyError):
        db.get_collection("docs")


@pytest.mark.parametrize("bad", ["", "has space", "a/b", "x" * 65])
def test_bad_names(db, bad):
    with pytest.raises(ValueError):
        db.create_collection(bad, 4)


def test_bad_dimension_and_metric(db):
    with pytest.raises(ValueError):
        db.create_collection("a", 0)
    with pytest.raises(ValueError):
        db.create_collection("b", 4, "hamming")


def test_upsert_get_delete(db):
    col = db.create_collection("c", 3, "dot")
    col.upsert([1, 2, 3], [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
               [{"lang": "te"}, {"lang": "en"}, {}])
    assert len(col) == 3
    p1, missing, p3 = col.get([1, 99, 3], with_vectors=True)
    assert p1.payload == {"lang": "te"} and p1.vector == [1.0, 0.0, 0.0]
    assert missing is None and p3.payload == {}

    assert col.delete([2, 2, 42]) == 1
    assert len(col) == 2 and col.get([2]) == [None]
    assert col.info()["holes"] == 1
    hits = [h.id for h in col.search([0, 1, 0], k=5)]
    assert sorted(hits) == [1, 3]


def test_upsert_replaces_via_new_slot(db):
    """Vectors are append-only: a replaced point moves to a new slot."""
    col = db.create_collection("c", 2, "l2")
    col.upsert([7], [[0, 0]], [{"v": 1}])
    col.upsert([7], [[5, 5]], [{"v": 2}])
    assert len(col) == 1 and col.info()["slots_used"] == 2 and col.info()["holes"] == 1
    assert [h.id for h in col.search([5, 5], k=5)] == [7]
    p = col.get([7], with_vectors=True)[0]
    assert p.vector == [5.0, 5.0] and p.payload == {"v": 2}


def test_duplicate_ids_in_batch_last_wins(db):
    col = db.create_collection("c", 2, "dot")
    col.upsert([1, 1], [[1, 0], [0, 1]], [{"n": "first"}, {"n": "last"}])
    p = col.get([1], with_vectors=True)[0]
    assert len(col) == 1 and p.payload == {"n": "last"} and p.vector == [0.0, 1.0]


def test_validation(db):
    col = db.create_collection("c", 3, "cosine")
    with pytest.raises(ValueError):
        col.upsert([1], [[1, 2]])                 # wrong dimension
    with pytest.raises(ValueError):
        col.upsert([1], [[0, 0, 0]])              # zero vector under cosine
    with pytest.raises(ValueError):
        col.upsert([1], [[float("nan"), 1, 1]])
    with pytest.raises(ValueError):
        col.upsert([1, 2], [[1, 1, 1]])           # count mismatch
    with pytest.raises(TypeError):
        col.upsert(["a"], [[1, 1, 1]])            # non-integer ID
    with pytest.raises(ValueError):
        col.upsert([-1], [[1, 1, 1]])
    with pytest.raises(ValueError):
        col.upsert([1], [[1, 1, 1]], [{"x": float("inf")}])
    with pytest.raises(ValueError):
        col.search([1, 1, 1], k=0)
    assert len(col) == 0


def test_scores_per_metric(db):
    l2 = db.create_collection("l2", 2, "l2")
    l2.upsert([1, 2], [[3, 4], [10, 10]])
    hit = l2.search([0, 0], k=1)[0]
    assert hit.id == 1 and math.isclose(hit.score, 5.0, rel_tol=1e-5)

    cos = db.create_collection("cos", 2, "cosine")
    cos.upsert([1, 2], [[2, 0], [0, 3]])
    hits = cos.search([5, 0], k=2)
    assert [h.id for h in hits] == [1, 2]
    assert math.isclose(hits[0].score, 1.0, rel_tol=1e-5) and abs(hits[1].score) < 1e-6


def test_search_matches_brute_force_at_scale(db, rng):
    """Grows the vector file many times over and checks exact results."""
    col = db.create_collection("big", 32, "cosine", index="flat")
    x = rng.standard_normal((5_000, 32)).astype(np.float32)
    for start in range(0, 5_000, 1_000):
        col.upsert(list(range(start, start + 1_000)), x[start:start + 1_000])
    deleted = set(range(0, 5_000, 7))
    col.delete(sorted(deleted))

    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    for q in rng.standard_normal((20, 32)).astype(np.float32):
        sims = xn @ (q / np.linalg.norm(q))
        sims[list(deleted)] = -np.inf
        expected = np.argsort(-sims)[:10].tolist()
        assert [h.id for h in col.search(q, k=10)] == expected
        assert [h.id for h in col.search(q, k=10, exact=True)] == expected


def test_search_batch_and_empty(db):
    col = db.create_collection("c", 2, "dot")
    assert col.search_batch([[1, 0], [0, 1]], k=3) == [[], []]
    col.upsert([1, 2], [[1, 0], [0, 1]])
    a, b = col.search_batch([[1, 0], [0, 1]], k=1)
    assert a[0].id == 1 and b[0].id == 2


def test_persistence_across_reopen(tmp_path, rng):
    data = tmp_path / "data"
    x = rng.standard_normal((300, 8)).astype(np.float32)
    with VectorForge(data) as db:
        col = db.create_collection("p", 8, "l2")
        col.upsert(list(range(300)), x, [{"i": i} for i in range(300)])
        col.delete([5, 6])
        before = [h.id for h in col.search(x[10], k=5)]

    with VectorForge(data) as db:
        col = db.get_collection("p")
        assert len(col) == 298 and col.info()["holes"] == 2
        assert [h.id for h in col.search(x[10], k=5)] == before
        assert col.get([5]) == [None] and col.get([7])[0].payload == {"i": 7}
        np.testing.assert_allclose(col.get([299], with_vectors=True)[0].vector, x[299])
        col.upsert([1000], [x[0]])                # writes keep working after reopen
        assert col.info()["slots_used"] == 301
