"""Payload filters, the payload index, and the query planner."""
import numpy as np
import pytest
from helpers import clustered, ids, recall

from vectorforge.engine.filters import PayloadIndex, validate_filter


@pytest.mark.parametrize("bad", [
    {}, [], {"x": None}, {"x": {"between": [1, 2]}}, {"x": {"gt": "a"}},
    {"x": {"in": "te"}}, {"x": {"eq": 1, "gt": 0}}, {"x": {"gt": float("nan")}},
    {"x": {"gt": True}}, {"": 1}, {"x": [1, 2]},
])
def test_invalid_filters(bad):
    with pytest.raises(ValueError):
        validate_filter(bad)


def test_payload_index_semantics():
    pi = PayloadIndex()
    rows = [
        {"lang": "te", "year": 2025, "tags": ["craft", "video"], "ok": True},
        {"lang": "en", "year": 2023, "tags": ["craft"], "ok": False},
        {"lang": "te", "year": 2024.5, "n": 1},
        {"lang": "hi", "year": "2026"},          # a string year is not a number
        {},
    ]
    for slot, p in enumerate(rows):
        pi.add(slot, p)
    m = lambda f: np.nonzero(pi.mask(f, len(rows)))[0].tolist()  # noqa: E731
    assert m({"lang": "te"}) == [0, 2]
    assert m({"lang": {"in": ["en", "hi"]}}) == [1, 3]
    assert m({"year": {"gt": 2024}}) == [0, 2]
    assert m({"year": {"gte": 2023, "lt": 2025}}) == [1, 2]
    assert m({"tags": "craft"}) == [0, 1]
    assert m({"tags": "video", "lang": "te"}) == [0]
    assert m({"ok": True}) == [0]
    assert m({"n": 1}) == [2] and m({"n": True}) == []     # 1 and True are different
    assert m({"year": "2026"}) == [3]
    assert m({"missing": 1}) == [] and m({"missing": {"gt": 0}}) == []
    assert pi.mask({"lang": "te"}, 2).tolist() == [True, False]


@pytest.fixture(scope="module")
def filtered_collection(tmp_path_factory):
    from vectorforge import VectorForge
    rng = np.random.default_rng(5)
    x = clustered(rng, 30_000, 32, centers=1_000)
    q = clustered(rng, 50, 32, centers=1_000)
    pay = [{"bucket": i % 100, "lang": ["te", "en", "hi"][i % 3]} for i in range(len(x))]
    db = VectorForge(tmp_path_factory.mktemp("f"))
    out = {}
    for kind in ("hnsw", "ivfpq"):
        col = db.create_collection(kind, 32, "l2", index=kind,
                                   params={"M": 12, "ef_construction": 100} if kind == "hnsw"
                                   else {"nlist": 64, "nprobe": 48, "train_after": 1_000})
        for s in range(0, len(x), 10_000):
            col.upsert(list(range(s, s + 10_000)), x[s:s + 10_000], pay[s:s + 10_000])
        out[kind] = col
    yield out, q
    db.close()


@pytest.mark.parametrize("kind", ["hnsw", "ivfpq"])
@pytest.mark.parametrize("flt,expected_strategy", [
    ({"bucket": 7}, "subset_scan"),                          # 1%  -> exact scan
    ({"bucket": {"lt": 10}}, "subset_scan"),                 # 10% = 3,000 points
    ({"lang": "te"}, "subset_scan"),                         # 33% but only 10,000 points
    ({"bucket": {"lt": 70}}, None),                          # 70% -> the index, mask on
    ({"bucket": {"lt": 90}}, None),                          # 90%
])
def test_filtered_search_recall_and_plan(filtered_collection, kind, flt, expected_strategy):
    cols, q = filtered_collection
    col = cols[kind]
    plan = col.explain(filter=flt)
    assert plan["strategy"] == (expected_strategy or kind)
    got = col.search_batch(q, 10, filter=flt)
    truth = col.search_batch(q, 10, filter=flt, exact=True)
    assert recall(ids(got), ids(truth), 10) >= 0.95
    for row in got:                                            # every hit matches
        for h in row:
            if "bucket" in flt and isinstance(flt["bucket"], int):
                assert h.payload["bucket"] == flt["bucket"]
            if "lang" in flt:
                assert h.payload["lang"] == "te"
            if isinstance(flt.get("bucket"), dict):
                b, c = h.payload["bucket"], flt["bucket"]
                assert b < c.get("lt", 1e9)


def test_planner_edge_cases(filtered_collection):
    cols, q = filtered_collection
    col = cols["hnsw"]
    assert col.explain(filter={"bucket": 1000})["strategy"] == "empty"
    assert col.search(q[0], 5, filter={"bucket": 1000}) == []
    assert col.explain(exact=True)["strategy"] == "flat_scan"
    assert col.explain()["strategy"] == "hnsw"
    p = col.explain(filter={"bucket": {"lt": 80}})
    assert p["strategy"] == "hnsw" and p["params"]["ef"] == 64
    assert p["selectivity"] == 0.8 and p["candidates"] == 24_000
    p = col.explain(filter={"bucket": {"lt": 30}})                   # 9,000 points
    assert p["strategy"] == "subset_scan" and "beats" in p["reason"]
    with pytest.raises(ValueError):
        col.search(q[0], 5, filter={"bucket": {"gt": "x"}})


def test_filter_sees_updates_and_deletes(db):
    col = db.create_collection("u", 2, "l2", index="hnsw")
    col.upsert([1, 2], [[0, 0], [1, 1]], [{"lang": "te"}, {"lang": "te"}])
    col.upsert([1], [[0, 0]], [{"lang": "en"}])               # moved to a new slot
    col.delete([2])
    assert ids([col.search([0, 0], 5, filter={"lang": "te"})]) == [[]]
    assert ids([col.search([0, 0], 5, filter={"lang": "en"})]) == [[1]]
