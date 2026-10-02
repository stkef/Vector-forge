"""One benchmark harness for every engine, in the ann-benchmarks style.

Every engine gets the same vectors, the same exact ground truth, the same
k, and comparable settings (same M / ef_construction for graphs, same
nlist / m / re-rank depth for IVF-PQ). Queries are sent one at a time,
single-threaded, and timed individually, so the numbers include each
library's per-call overhead the way an application would see it.

    python scripts/bench.py --synthetic 50000 --dim 128
    python scripts/bench.py --dataset datasets/sift-128-euclidean.hdf5
    python scripts/bench.py --dataset datasets/glove-100-angular.hdf5 --engines vf-hnsw,hnswlib,qdrant

Qdrant runs as a real server (docker compose -f bench/docker-compose.yml up -d).
`--qdrant-url :memory:` uses qdrant-client's local mode, which is a Python
re-implementation, not the Qdrant engine: only for checking the adapter.

Writes results/bench-<dataset>.json and a recall/QPS chart .png.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from vectorforge.engine.index import FlatIndex, HNSWIndex, IVFPQIndex
from vectorforge.engine.index.ivfpq import default_m
from vectorforge.engine.metrics import Metric, prepare

ALL = ["vf-hnsw", "hnswlib", "faiss-hnsw", "qdrant", "vf-ivfpq", "faiss-ivfpq"]


# --------------------------------------------------------------- engines
class Engine:
    name = ""
    sweep_name = ""

    def build(self, x: np.ndarray) -> None: ...
    def sweep(self) -> list[int]: ...
    def set(self, value: int) -> None: ...
    def query(self, q: np.ndarray, k: int) -> np.ndarray: ...
    def memory(self) -> int | None: return None
    def close(self) -> None: pass


class VFHnsw(Engine):
    name, sweep_name = "vf-hnsw", "ef"

    def __init__(self, a, metric):
        self.a, self.metric = a, metric

    def build(self, x):
        warm = HNSWIndex(lambda: x[:300], self.metric, M=self.a.M)   # compile outside the timer
        warm.add(np.arange(300)); warm.search(x[:1], 1)
        self.ix = HNSWIndex(lambda: x, self.metric, M=self.a.M, ef_construction=self.a.ef_construction)
        self.ix.add(np.arange(len(x)))

    def sweep(self): return self.a.ef

    def set(self, v): self.ef = v

    def query(self, q, k): return self.ix.search(q[None], k, ef=self.ef)[0][0]

    def memory(self): return self.ix.stats()["memory_bytes"]


class HnswLib(Engine):
    name, sweep_name = "hnswlib", "ef"

    def __init__(self, a, metric):
        self.a, self.metric = a, metric

    def build(self, x):
        import hnswlib
        self.ix = hnswlib.Index(space="l2" if self.metric is Metric.L2 else "ip", dim=x.shape[1])
        self.ix.init_index(len(x), M=self.a.M, ef_construction=self.a.ef_construction, random_seed=0)
        self.ix.set_num_threads(1)
        self.ix.add_items(x, np.arange(len(x)))

    def sweep(self): return self.a.ef

    def set(self, v): self.ix.set_ef(v)

    def query(self, q, k): return self.ix.knn_query(q, k)[0][0]


class FaissHnsw(Engine):
    name, sweep_name = "faiss-hnsw", "ef"

    def __init__(self, a, metric):
        self.a, self.metric = a, metric

    def build(self, x):
        import faiss
        faiss.omp_set_num_threads(1)
        mt = faiss.METRIC_L2 if self.metric is Metric.L2 else faiss.METRIC_INNER_PRODUCT
        self.ix = faiss.IndexHNSWFlat(x.shape[1], self.a.M, mt)
        self.ix.hnsw.efConstruction = self.a.ef_construction
        self.ix.add(x)

    def sweep(self): return self.a.ef

    def set(self, v): self.ix.hnsw.efSearch = v

    def query(self, q, k): return self.ix.search(q[None], k)[1][0]


class Qdrant(Engine):
    name, sweep_name = "qdrant", "ef"

    def __init__(self, a, metric):
        self.a, self.metric = a, metric
        if a.qdrant_url == ":memory:":
            self.name = "qdrant-local-python"

    def build(self, x):
        from qdrant_client import QdrantClient, models
        self.m = models
        url = self.a.qdrant_url
        self.cl = QdrantClient(location=":memory:") if url == ":memory:" else \
            QdrantClient(url=url, prefer_grpc=self.a.qdrant_grpc, timeout=600)
        self.coll = "vectorforge_bench"
        if self.cl.collection_exists(self.coll):
            self.cl.delete_collection(self.coll)
        dist = models.Distance.EUCLID if self.metric is Metric.L2 else models.Distance.DOT
        self.cl.create_collection(
            self.coll, vectors_config=models.VectorParams(size=x.shape[1], distance=dist),
            hnsw_config=models.HnswConfigDiff(m=self.a.M, ef_construct=self.a.ef_construction))
        self.cl.upload_collection(self.coll, vectors=x, ids=range(len(x)), batch_size=1_000)
        if url != ":memory:":                       # wait until the HNSW is built
            while self.cl.get_collection(self.coll).status != models.CollectionStatus.GREEN:
                time.sleep(1)

    def sweep(self): return self.a.ef

    def set(self, v): self.params = self.m.SearchParams(hnsw_ef=v, exact=False)

    def query(self, q, k):
        r = self.cl.query_points(self.coll, query=q.tolist(), limit=k, search_params=self.params)
        return np.array([p.id for p in r.points])

    def version(self):
        if self.a.qdrant_url == ":memory:":
            return None                             # local mode: no server
        try:
            return self.cl.info().version
        except Exception:  # noqa: BLE001
            return None

    def close(self):
        if self.a.qdrant_url != ":memory:":
            self.cl.delete_collection(self.coll)


class VFIvfpq(Engine):
    name, sweep_name = "vf-ivfpq", "nprobe"

    def __init__(self, a, metric):
        self.a, self.metric = a, metric

    def build(self, x):
        self.ix = IVFPQIndex(lambda: x, self.metric, x.shape[1], nlist=self.a.nlist, m=self.a.m,
                             rerank=self.a.rerank)
        rng = np.random.default_rng(0)
        self.ix.train(x[rng.choice(len(x), min(len(x), self.a.train_size), replace=False)])
        self.ix.add(np.arange(len(x)))
        self.a._nlist = self.ix.nlist           # FAISS uses the same

    def sweep(self): return [p for p in self.a.nprobe if p <= self.ix.nlist]

    def set(self, v): self.nprobe = v

    def query(self, q, k): return self.ix.search(q[None], k, nprobe=self.nprobe)[0][0]

    def memory(self): return self.ix.stats()["memory_bytes"]


class FaissIvfpq(Engine):
    name, sweep_name = "faiss-ivfpq", "nprobe"

    def __init__(self, a, metric):
        self.a, self.metric = a, metric

    def build(self, x):
        import faiss
        faiss.omp_set_num_threads(1)
        d = x.shape[1]
        mt = faiss.METRIC_L2 if self.metric is Metric.L2 else faiss.METRIC_INNER_PRODUCT
        nlist = getattr(self.a, "_nlist", None) or int(np.clip(4 * np.sqrt(len(x)), 16, 65_536))
        m = self.a.m or default_m(d)
        self.base = faiss.IndexIVFPQ(faiss.IndexFlat(d, mt), d, nlist, m, 8, mt)
        rng = np.random.default_rng(0)
        self.base.train(x[rng.choice(len(x), min(len(x), self.a.train_size), replace=False)])
        self.ix = faiss.IndexRefineFlat(self.base)  # exact re-rank, like VectorForge's
        self.ix.add(x)                              # adds to the IVF-PQ and the raw store
        self.nlist = nlist

    def sweep(self): return [p for p in self.a.nprobe if p <= self.nlist]

    def set(self, v):
        import faiss
        self.params = faiss.IndexRefineSearchParameters(
            k_factor=max(1.0, self.a.rerank / self.a.k),
            base_index_params=faiss.SearchParametersIVF(nprobe=v))

    def query(self, q, k): return self.ix.search(q[None], k, params=self.params)[1][0]

    def memory(self):
        # codes + ids; the refine stage keeps raw vectors, like VectorForge's mmap
        return int(self.base.ntotal * (self.base.code_size + 8))


ENGINES = {"vf-hnsw": VFHnsw, "hnswlib": HnswLib, "faiss-hnsw": FaissHnsw, "qdrant": Qdrant,
           "vf-ivfpq": VFIvfpq, "faiss-ivfpq": FaissIvfpq}


# ---------------------------------------------------------------- driver
def load(a):
    if a.synthetic:
        rng = np.random.default_rng(0)
        c = rng.standard_normal((max(100, a.synthetic // 100), a.dim)).astype(np.float32) * 2
        mk = lambda n: (c[rng.integers(0, len(c), n)] + rng.standard_normal((n, a.dim))).astype(np.float32)  # noqa: E731
        return f"synthetic-{a.synthetic}x{a.dim}", mk(a.synthetic), mk(a.queries), None, Metric.L2
    import h5py
    with h5py.File(a.dataset, "r") as f:
        metric = Metric.COSINE if f.attrs.get("distance", "euclidean") == "angular" else Metric.L2
        train = np.asarray(f["train"][: a.limit] if a.limit else f["train"], np.float32)
        test = np.asarray(f["test"][: a.queries], np.float32)
        gt = None if a.limit else np.asarray(f["neighbors"][: a.queries])
    return Path(a.dataset).stem + (f"-{a.limit}" if a.limit else ""), train, test, gt, metric


def run_engine(e: Engine, x, q, gt, k):
    t = time.perf_counter()
    e.build(x)
    build_s = time.perf_counter() - t
    rows = []
    for v in e.sweep():
        e.set(v)
        for qq in q[:5]:
            e.query(qq, k)                          # warm-up
        lat, hits = np.empty(len(q)), 0
        for i, qq in enumerate(q):
            t = time.perf_counter()
            ids = e.query(qq, k)
            lat[i] = time.perf_counter() - t
            hits += len(set(np.asarray(ids)[:k].tolist()) & set(gt[i][:k].tolist()))
        row = dict(engine=e.name, param=e.sweep_name, value=v, recall=hits / (len(q) * k),
                   qps=1.0 / lat.mean(), p50_ms=float(np.percentile(lat, 50) * 1e3),
                   p99_ms=float(np.percentile(lat, 99) * 1e3), build_s=build_s,
                   index_bytes=e.memory())
        rows.append(row)
        print(f"  {e.name:<20}{e.sweep_name}={v:<5} recall={row['recall']:.4f} "
              f"QPS={row['qps']:>8,.0f}  p50={row['p50_ms']:.3f}ms  p99={row['p99_ms']:.3f}ms")
    return rows


def chart(rows, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for eng in dict.fromkeys(r["engine"] for r in rows):
        rs = sorted((r for r in rows if r["engine"] == eng), key=lambda r: r["recall"])
        ax.plot([r["recall"] for r in rs], [r["qps"] for r in rs], marker="o", lw=1.8,
                ls="-" if eng.startswith("vf") else "--", label=eng)
    ax.set_yscale("log")
    ax.set_xlabel("recall@10")
    ax.set_ylabel("queries per second (1 thread, log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--synthetic", type=int)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--queries", type=int, default=1_000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--engines", default=",".join(ALL))
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef", default="10,16,24,32,48,64,96,128,256")
    ap.add_argument("--nlist", type=int)
    ap.add_argument("--m", type=int)
    ap.add_argument("--rerank", type=int, default=100)
    ap.add_argument("--nprobe", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--train-size", type=int, default=100_000)
    ap.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    ap.add_argument("--qdrant-grpc", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    a.ef = [int(v) for v in a.ef.split(",")]
    a.nprobe = [int(v) for v in a.nprobe.split(",")]

    name, x, q, gt, metric = load(a)
    x, q = prepare(x, metric), prepare(q, metric)
    print(f"{name}: {len(x):,} x {x.shape[1]}, {len(q)} queries, {metric.value}, k={a.k}")
    if gt is None:
        gt, _ = FlatIndex(lambda: x, metric).search(q, a.k, None)

    rows, meta = [], {}
    for en in a.engines.split(","):
        e = ENGINES[en](a, metric)
        print(f"\n{e.name}: building ...")
        try:
            rows += run_engine(e, x, q, gt, a.k)
            if isinstance(e, Qdrant):
                meta["qdrant_version"] = e.version()
        except ImportError as err:
            print(f"  skipped: {err}")
        except Exception as err:  # noqa: BLE001  (e.g. Qdrant not running)
            print(f"  failed: {type(err).__name__}: {err}")
        finally:
            try:
                e.close()
            except Exception:  # noqa: BLE001
                pass

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    doc = {"dataset": name, "n": len(x), "dim": int(x.shape[1]), "queries": len(q),
           "metric": metric.value, "k": a.k, "M": a.M, "ef_construction": a.ef_construction,
           "rerank": a.rerank, "mode": "single query, single thread",
           "machine": {"platform": platform.platform(), "cpu": platform.processor(),
                       "cpus": os.cpu_count(), "python": platform.python_version()},
           **meta, "results": rows}
    path = out / f"bench-{name}.json"
    path.write_text(json.dumps(doc, indent=2))
    chart(rows, f"{name}: recall vs QPS (single thread)", out / f"bench-{name}.png")
    print(f"\nwrote {path} and {path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
