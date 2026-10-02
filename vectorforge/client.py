"""Python client for a VectorForge server.

    from vectorforge.client import Client
    vf = Client("http://127.0.0.1:8000")
    docs = vf.create_collection("docs", dim=768, metric="cosine")
    docs.upsert([1, 2], [v1, v2], [{"lang": "te"}, {"lang": "en"}])
    hits = docs.search(q, k=10, filter={"lang": "te"})

Results come back as the same Point / ScoredPoint objects the embedded
engine returns, so code can switch between the two.
"""
from __future__ import annotations

from typing import Any, Sequence

import httpx
import numpy as np

from vectorforge.engine.collection import Point, ScoredPoint


class VectorForgeError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


def _list(v) -> list:
    return np.asarray(v, np.float32).tolist()


class Client:
    def __init__(self, url: str = "http://127.0.0.1:8000", timeout: float = 60.0,
                 http: httpx.Client | None = None):
        self._http = http or httpx.Client(base_url=url.rstrip("/"), timeout=timeout)

    def _call(self, method: str, path: str, **kw) -> Any:
        r = self._http.request(method, path, **kw)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise VectorForgeError(r.status_code, str(detail))
        return r.json()

    def health(self) -> dict:
        return self._call("GET", "/healthz")

    def list_collections(self) -> list[str]:
        return [c["name"] for c in self._call("GET", "/collections")["collections"]]

    def create_collection(self, name: str, dim: int, metric: str = "cosine", index: str = "hnsw",
                          params: dict | None = None, storage: dict | None = None) -> "RemoteCollection":
        self._call("POST", "/collections", json=dict(name=name, dim=dim, metric=metric, index=index,
                                                     params=params, storage=storage))
        return RemoteCollection(self, name)

    def get_collection(self, name: str) -> "RemoteCollection":
        self._call("GET", f"/collections/{name}")
        return RemoteCollection(self, name)

    def drop_collection(self, name: str) -> bool:
        try:
            self._call("DELETE", f"/collections/{name}")
            return True
        except VectorForgeError as e:
            if e.status == 404:
                return False
            raise

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class RemoteCollection:
    def __init__(self, client: Client, name: str):
        self._c = client
        self.name = name
        self._p = f"/collections/{name}"

    def info(self) -> dict:
        return self._c._call("GET", self._p)

    def __len__(self) -> int:
        return self.info()["points"]

    def upsert(self, ids: Sequence[int], vectors, payloads: Sequence[dict] | None = None,
               batch: int = 1_000) -> int:
        vecs = np.asarray(vectors, np.float32)
        payloads = list(payloads) if payloads is not None else [{}] * len(ids)
        total = 0
        for s in range(0, len(ids), batch):
            pts = [{"id": int(i), "vector": v, "payload": p} for i, v, p in
                   zip(ids[s:s + batch], vecs[s:s + batch].tolist(), payloads[s:s + batch])]
            total += self._c._call("PUT", f"{self._p}/points", json={"points": pts})["upserted"]
        return total

    def get(self, ids: Sequence[int], with_vectors: bool = False) -> list[Point | None]:
        out = self._c._call("POST", f"{self._p}/points/get",
                            json={"ids": [int(i) for i in ids], "with_vectors": with_vectors})
        return [None if p is None else Point(**p) for p in out["points"]]

    def delete(self, ids: Sequence[int]) -> int:
        return self._c._call("POST", f"{self._p}/points/delete",
                             json={"ids": [int(i) for i in ids]})["deleted"]

    def search(self, vector, k: int = 10, **opts) -> list[ScoredPoint]:
        out = self._c._call("POST", f"{self._p}/search", json={"vector": _list(vector), "k": k, **opts})
        return [ScoredPoint(**h) for h in out["hits"]]

    def search_batch(self, vectors, k: int = 10, **opts) -> list[list[ScoredPoint]]:
        out = self._c._call("POST", f"{self._p}/search/batch",
                            json={"vectors": _list(vectors), "k": k, **opts})
        return [[ScoredPoint(**h) for h in r] for r in out["results"]]

    def explain(self, k: int = 10, **opts) -> dict:
        return self._c._call("POST", f"{self._p}/explain", json={"k": k, **opts})

    def checkpoint(self) -> dict:
        return self._c._call("POST", f"{self._p}/checkpoint")

    def train_index(self) -> dict:
        return self._c._call("POST", f"{self._p}/train")
