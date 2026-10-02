"""HTTP API and Python client, run in-process through FastAPI's TestClient."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vectorforge.client import Client, VectorForgeError
from vectorforge.embeddings import HashEmbedder
from vectorforge.server.app import create_app


@pytest.fixture
def api(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "hnsw-demo.json").write_text('{"dataset": "demo", "results": []}')
    (results / "broken.json").write_text("{nope")
    app = create_app(tmp_path / "data", results, embedder=HashEmbedder(dim=64))
    with TestClient(app) as http:
        yield http


@pytest.fixture
def client(api):
    return Client(http=api)


def test_health_and_studio(api):
    assert api.get("/healthz").json()["embedder"] == {"name": "hash", "dim": 64}
    page = api.get("/")
    assert page.status_code == 200 and "VectorForge" in page.text


def test_collection_crud_and_errors(api):
    r = api.post("/collections", json={"name": "a", "dim": 4, "metric": "l2"})
    assert r.status_code == 201 and r.json()["index"]["type"] == "hnsw"
    assert api.post("/collections", json={"name": "a", "dim": 4}).status_code == 409
    assert api.post("/collections", json={"name": "b c", "dim": 4}).status_code == 400
    assert api.post("/collections", json={"name": "b", "dim": 4, "index": "x"}).status_code == 400
    assert api.get("/collections/zzz").status_code == 404
    assert [c["name"] for c in api.get("/collections").json()["collections"]] == ["a"]
    assert api.put("/collections/a/points", json={"points": [
        {"id": 1, "vector": [1, 2, 3]}]}).status_code == 400                 # wrong dimension
    assert api.delete("/collections/a").status_code == 200
    assert api.delete("/collections/a").status_code == 404


def test_points_search_explain_inspect(api):
    api.post("/collections", json={"name": "p", "dim": 8, "metric": "l2",
                                   "params": {"M": 8}})
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 8)).astype(np.float32)
    pts = [{"id": i, "vector": x[i].tolist(), "payload": {"g": i % 3}} for i in range(300)]
    assert api.put("/collections/p/points", json={"points": pts}).json()["upserted"] == 300

    got = api.post("/collections/p/points/get", json={"ids": [5, 999], "with_vectors": True}).json()
    assert got["points"][0]["payload"] == {"g": 2} and got["points"][1] is None
    page = api.get("/collections/p/points", params={"limit": 100}).json()
    assert len(page["points"]) == 100 and page["next"] == 99
    last = api.get("/collections/p/points", params={"after": 199, "limit": 200}).json()
    assert len(last["points"]) == 100 and last["next"] is None

    r = api.post("/collections/p/search", json={"query_id": 7, "k": 5, "compare_exact": True}).json()
    assert r["hits"][0]["id"] == 7 and r["exact"]["recall"] == 1.0 and r["plan"]["strategy"] == "hnsw"
    r = api.post("/collections/p/search", json={"random": True, "k": 5, "filter": {"g": 1}}).json()
    assert all(h["payload"]["g"] == 1 for h in r["hits"]) and r["plan"]["strategy"] == "subset_scan"
    assert api.post("/collections/p/search", json={"k": 5}).status_code == 400
    assert api.post("/collections/p/search", json={"query_id": 12345}).status_code == 404

    b = api.post("/collections/p/search/batch", json={"vectors": x[:3].tolist(), "k": 1}).json()
    assert [r[0]["id"] for r in b["results"]] == [0, 1, 2]
    assert api.post("/collections/p/explain", json={"exact": True}).json()["strategy"] == "flat_scan"

    ins = api.get("/collections/p/inspect").json()
    assert ins["type"] == "hnsw" and ins["nodes_per_layer"][0] == 300
    tr = api.post("/collections/p/inspect/trace", json={"query_id": 7, "sample": 200}).json()
    assert len(tr["sample_xy"]) == 200 and tr["hops"][0]["layer"] == ins["stats"]["max_level"]
    assert tr["found"][0]["id"] == 7 and tr["visited"] > 0

    assert api.post("/collections/p/points/delete", json={"ids": [7, 8]}).json()["deleted"] == 2
    ck = api.post("/collections/p/checkpoint").json()
    assert ck["wal_bytes"] == 0 and ck["points"] == 298


def test_ivfpq_inspect(api):
    api.post("/collections", json={"name": "v", "dim": 8, "metric": "l2", "index": "ivfpq",
                                   "params": {"nlist": 8, "train_after": 300}})
    x = np.random.default_rng(1).standard_normal((400, 8)).astype(np.float32)
    api.put("/collections/v/points", json={"points": [
        {"id": i, "vector": x[i].tolist()} for i in range(400)]})
    ins = api.get("/collections/v/inspect").json()
    assert ins["type"] == "ivfpq" and sum(ins["list_sizes"]) == 400
    tr = api.post("/collections/v/inspect/trace", json={"query_id": 3, "nprobe": 2}).json()
    assert len(tr["probes"]) == 2 and len(tr["centroids_xy"]) == 8
    assert api.post("/collections/v/train").status_code == 200


def test_text_demo_endpoints(api):
    api.post("/collections", json={"name": "t", "dim": 64})
    docs = ["bidri work uses silver inlay on blackened zinc alloy",
            "biryani is cooked with basmati rice and spices",
            "HNSW builds a layered graph for nearest neighbour search"]
    r = api.post("/collections/t/texts", json={"texts": docs, "payloads": [{"src": i} for i in range(3)]})
    assert r.json()["upserted"] == 3
    hits = api.post("/collections/t/search", json={"text": "silver inlay bidri", "k": 1}).json()["hits"]
    assert hits[0]["payload"]["src"] == 0 and hits[0]["payload"]["text"] == docs[0]
    api.post("/collections", json={"name": "wrongdim", "dim": 8})
    assert api.post("/collections/wrongdim/search", json={"text": "x"}).status_code == 400


def test_benchmarks_listing(api):
    files = [b["file"] for b in api.get("/benchmarks").json()["benchmarks"]]
    assert files == ["hnsw-demo.json"]                                       # broken one skipped


def test_python_client(client):
    col = client.create_collection("c", 4, "cosine")
    assert client.list_collections() == ["c"]
    x = np.eye(4, dtype=np.float32)
    assert col.upsert(list(range(4)), x, [{"n": i} for i in range(4)], batch=3) == 4
    assert len(col) == 4
    hits = col.search(x[2], k=2)
    assert hits[0].id == 2 and abs(hits[0].score - 1) < 1e-5 and hits[0].payload == {"n": 2}
    assert col.search(x[2], k=2, filter={"n": 3})[0].id == 3
    assert [r[0].id for r in col.search_batch(x, k=1)] == [0, 1, 2, 3]
    p = col.get([1, 9], with_vectors=True)
    assert p[0].vector == [0, 1, 0, 0] and p[1] is None
    assert col.delete([1]) == 1 and len(col) == 3
    assert col.explain(filter={"n": 0})["strategy"] == "subset_scan"
    with pytest.raises(VectorForgeError) as e:
        client.create_collection("c", 4)
    assert e.value.status == 409
    assert client.drop_collection("c") and not client.drop_collection("c")
