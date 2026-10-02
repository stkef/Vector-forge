"""HTTP API for VectorForge, plus the Studio page at /.

    python -m vectorforge serve --data ./data

Endpoints mirror the Python API. Errors: 400 bad input, 404 unknown
collection or point, 409 collection exists.
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from vectorforge import __version__
from vectorforge.db import VectorForge
from vectorforge.embeddings import Embedder, text_id
from vectorforge.engine import inspect as insp
from vectorforge.engine.collection import Collection
from vectorforge.engine.index import HNSWIndex, IVFPQIndex, warmup

STATIC = Path(__file__).parent / "static"


# ------------------------------------------------------------------ models
class CreateCollection(BaseModel):
    name: str
    dim: int
    metric: str = "cosine"
    index: str = "hnsw"
    params: dict[str, Any] | None = None
    storage: dict[str, Any] | None = None


class PointIn(BaseModel):
    id: int
    vector: list[float]
    payload: dict[str, Any] = Field(default_factory=dict)


class Upsert(BaseModel):
    points: list[PointIn]


class Ids(BaseModel):
    ids: list[int]
    with_vectors: bool = False


class SearchParams(BaseModel):
    k: int = 10
    filter: dict[str, Any] | None = None
    exact: bool = False
    ef: int | None = None
    nprobe: int | None = None
    rerank: int | None = None


class Search(SearchParams):
    vector: list[float] | None = None
    query_id: int | None = None      # use a stored point's vector as the query
    random: bool = False             # use a random stored point
    text: str | None = None          # embed with the server's embedder
    compare_exact: bool = False      # also run brute force and report recall


class SearchBatch(SearchParams):
    vectors: list[list[float]]


class Texts(BaseModel):
    texts: list[str]
    payloads: list[dict[str, Any]] | None = None


class Trace(BaseModel):
    vector: list[float] | None = None
    query_id: int | None = None
    random: bool = False
    text: str | None = None
    ef: int | None = None
    nprobe: int | None = None
    sample: int = 1_500


def _hit(h) -> dict:
    return {"id": h.id, "score": h.score, "payload": h.payload}


def create_app(data_dir: str | Path = "./data", results_dir: str | Path = "./results",
               embedder: Embedder | None = None) -> FastAPI:
    db = VectorForge(data_dir)
    results = Path(results_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        warmup()            # load Numba kernels before the first request
        yield
        db.close()          # clean shutdown = checkpoint every open collection

    app = FastAPI(title="VectorForge", version=__version__, lifespan=lifespan)
    app.state.db = db

    @app.exception_handler(ValueError)
    async def _bad(_: Request, e: ValueError):
        return JSONResponse({"detail": str(e)}, status_code=400)

    @app.exception_handler(TypeError)
    async def _bad_type(_: Request, e: TypeError):
        return JSONResponse({"detail": str(e)}, status_code=400)

    @app.exception_handler(KeyError)
    async def _missing(_: Request, e: KeyError):
        return JSONResponse({"detail": str(e.args[0]) if e.args else "not found"}, status_code=404)

    @app.exception_handler(FileExistsError)
    async def _exists(_: Request, e: FileExistsError):
        return JSONResponse({"detail": str(e)}, status_code=409)

    def col(name: str) -> Collection:
        return db.get_collection(name)

    def query_vector(c: Collection, body) -> tuple[np.ndarray, int | None]:
        if body.vector is not None:
            return np.asarray(body.vector, np.float32), None
        if body.text is not None:
            if embedder is None:
                raise ValueError("this server has no embedder; start it with --embedder gemini")
            if embedder.dim != c.dim:
                raise ValueError(f"embedder makes {embedder.dim}-d vectors, collection is {c.dim}-d")
            return embedder.embed_query(body.text), None
        qid = body.query_id
        if body.random:
            ids = c.sample_ids(1)
            if not ids:
                raise ValueError("collection is empty")
            qid = ids[0]
        if qid is None:
            raise ValueError("give one of: vector, text, query_id, random")
        return c.vector_of(qid), qid

    # ---------------------------------------------------------- service
    @app.get("/healthz")
    def healthz():
        return {"ok": True, "version": __version__,
                "embedder": None if embedder is None else {"name": embedder.name, "dim": embedder.dim}}

    @app.get("/", include_in_schema=False)
    def studio():
        return FileResponse(STATIC / "index.html")

    # ------------------------------------------------------ collections
    @app.get("/collections")
    def list_collections():
        return {"collections": [col(n).info() for n in db.list_collections()]}

    @app.post("/collections", status_code=201)
    def create_collection(body: CreateCollection):
        return db.create_collection(body.name, body.dim, body.metric, index=body.index,
                                    params=body.params, storage=body.storage).info()

    @app.get("/collections/{name}")
    def get_collection(name: str):
        return col(name).info()

    @app.delete("/collections/{name}")
    def drop_collection(name: str):
        if not db.drop_collection(name):
            raise HTTPException(404, f"collection {name!r} does not exist")
        return {"dropped": name}

    @app.post("/collections/{name}/checkpoint")
    def checkpoint(name: str):
        c = col(name)
        t = time.perf_counter()
        c.checkpoint()
        return {"ms": (time.perf_counter() - t) * 1000, **c.info()}

    @app.post("/collections/{name}/train")
    def train(name: str):
        c = col(name)
        t = time.perf_counter()
        c.train_index()
        return {"ms": (time.perf_counter() - t) * 1000, **c.info()}

    # ----------------------------------------------------------- points
    @app.put("/collections/{name}/points")
    def upsert(name: str, body: Upsert):
        c = col(name)
        t = time.perf_counter()
        n = c.upsert([p.id for p in body.points], [p.vector for p in body.points],
                     [p.payload for p in body.points])
        return {"upserted": n, "ms": (time.perf_counter() - t) * 1000}

    @app.post("/collections/{name}/points/get")
    def get_points(name: str, body: Ids):
        pts = col(name).get(body.ids, with_vectors=body.with_vectors)
        return {"points": [None if p is None else p.__dict__ for p in pts]}

    @app.post("/collections/{name}/points/delete")
    def delete_points(name: str, body: Ids):
        return {"deleted": col(name).delete(body.ids)}

    @app.get("/collections/{name}/points")
    def scroll(name: str, after: int | None = None, limit: int = 50, with_vectors: bool = False):
        pts, nxt = col(name).scroll(after, limit, with_vectors)
        return {"points": [p.__dict__ for p in pts], "next": nxt}

    # ----------------------------------------------------------- search
    @app.post("/collections/{name}/search")
    def search(name: str, body: Search):
        c = col(name)
        q, qid = query_vector(c, body)
        opts = dict(filter=body.filter, ef=body.ef, nprobe=body.nprobe, rerank=body.rerank)
        plan = c.explain(body.k, filter=body.filter, exact=body.exact, ef=body.ef, nprobe=body.nprobe)
        t = time.perf_counter()
        hits = c.search(q, body.k, exact=body.exact, **opts)
        ms = (time.perf_counter() - t) * 1000
        out = {"hits": [_hit(h) for h in hits], "plan": plan, "ms": ms, "query_id": qid}
        if body.compare_exact:
            t = time.perf_counter()
            exact = c.search(q, body.k, exact=True, filter=body.filter)
            ex_ms = (time.perf_counter() - t) * 1000
            truth = {h.id for h in exact}
            got = {h.id for h in hits}
            out["exact"] = {"hits": [_hit(h) for h in exact], "ms": ex_ms,
                            "recall": len(truth & got) / len(truth) if truth else 1.0}
        return out

    @app.post("/collections/{name}/search/batch")
    def search_batch(name: str, body: SearchBatch):
        res = col(name).search_batch(body.vectors, body.k, filter=body.filter, exact=body.exact,
                                     ef=body.ef, nprobe=body.nprobe, rerank=body.rerank)
        return {"results": [[_hit(h) for h in r] for r in res]}

    @app.post("/collections/{name}/explain")
    def explain(name: str, body: SearchParams):
        return col(name).explain(body.k, filter=body.filter, exact=body.exact,
                                 ef=body.ef, nprobe=body.nprobe)

    # --------------------------------------------------------- inspect
    @app.get("/collections/{name}/inspect")
    def inspect(name: str):
        c = col(name)
        with c._lock:
            idx = c._index
            if isinstance(idx, HNSWIndex):
                return {"type": "hnsw", **insp.hnsw_structure(idx), "stats": idx.stats()}
            if isinstance(idx, IVFPQIndex):
                return {"type": "ivfpq", **insp.ivf_structure(idx), "stats": idx.stats()}
        return {"type": "flat"}

    @app.post("/collections/{name}/inspect/trace")
    def trace(name: str, body: Trace):
        """A 2-D PCA map of a sample of points, with the query's path drawn on it."""
        c = col(name)
        q, qid = query_vector(c, body)
        q = c._prep(q)[0]
        with c._lock:
            idx = c._index
            live = c._live_slots()
            if len(live) == 0:
                raise ValueError("collection is empty")
            rng = np.random.default_rng(0)
            sample = np.sort(rng.choice(live, min(body.sample, len(live)), replace=False))
            vecs = np.asarray(c._vectors.view(c._next_slot))
            mean, basis = insp.pca_2d(vecs[sample])
            proj = lambda s: ((vecs[s] - mean) @ basis.T).tolist()  # noqa: E731
            out: dict[str, Any] = {
                "type": c.index_config["type"], "query_id": qid,
                "sample": [{"slot": int(s), "id": int(c._slot_ids[s])} for s in sample],
                "sample_xy": proj(sample), "query_xy": ((q - mean) @ basis.T).tolist(),
            }
            if isinstance(idx, HNSWIndex):
                tr = insp.hnsw_trace(idx, q, body.ef or idx.ef_search)
                hop_slots = np.array([h["slot"] for h in tr["hops"]], np.int64)
                out["hops"] = [{**h, "id": int(c._slot_ids[h["slot"]]), "xy": xy}
                               for h, xy in zip(tr["hops"], proj(hop_slots))]
                exp = np.array(tr["expanded"], np.int64)
                out["expanded_xy"] = proj(exp) if len(exp) else []
                found = np.array(tr["found"][:10], np.int64)
                out["found"] = [{"id": int(c._slot_ids[s]), "xy": xy}
                                for s, xy in zip(found, proj(found) if len(found) else [])]
                out["visited"] = tr["visited"]
                out["total"] = idx.n_nodes
            elif isinstance(idx, IVFPQIndex) and idx.trained:
                probes = insp.ivf_probes(idx, q, body.nprobe or idx.nprobe)
                cxy = ((idx.centroids - mean) @ basis.T).tolist()
                out["centroids_xy"] = cxy
                out["probes"] = probes
                out["list_sizes"] = [L.n for L in idx.lists]
        return out

    # ------------------------------------------------------ text (demo)
    @app.post("/collections/{name}/texts")
    def add_texts(name: str, body: Texts):
        if embedder is None:
            raise ValueError("this server has no embedder; start it with --embedder gemini")
        c = col(name)
        payloads = body.payloads or [{} for _ in body.texts]
        if len(payloads) != len(body.texts):
            raise ValueError("texts and payloads differ in length")
        t = time.perf_counter()
        vecs = embedder.embed_documents(body.texts)
        ids = [text_id(t_) for t_ in body.texts]
        c.upsert(ids, vecs, [{**p, "text": t_} for p, t_ in zip(payloads, body.texts)])
        return {"upserted": len(ids), "ids": ids, "ms": (time.perf_counter() - t) * 1000}

    # ------------------------------------------------------- benchmarks
    @app.get("/benchmarks")
    def benchmarks():
        out = []
        if results.exists():
            for f in sorted(results.glob("*.json")):
                try:
                    out.append({"file": f.name, **json.loads(f.read_text())})
                except (json.JSONDecodeError, OSError):
                    continue
        return {"benchmarks": out}

    return app
