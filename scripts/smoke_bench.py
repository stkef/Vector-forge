"""Quick flat-index timing on random data (not the real benchmark; that's M5).

    python scripts/smoke_bench.py --n 100000 --dim 128
"""
import argparse
import tempfile
import time

import numpy as np

from vectorforge import VectorForge

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100_000)
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--queries", type=int, default=200)
args = ap.parse_args()

rng = np.random.default_rng(0)
x = rng.standard_normal((args.n, args.dim)).astype(np.float32)
q = rng.standard_normal((args.queries, args.dim)).astype(np.float32)

with tempfile.TemporaryDirectory() as tmp, VectorForge(tmp) as db:
    col = db.create_collection("bench", args.dim, "l2")
    t = time.perf_counter()
    for s in range(0, args.n, 10_000):
        col.upsert(list(range(s, min(s + 10_000, args.n))), x[s:s + 10_000])
    print(f"ingest  {args.n:,} x {args.dim}: {time.perf_counter() - t:.2f}s")

    t = time.perf_counter()
    for v in q[:50]:
        col.search(v, k=10)
    single = (time.perf_counter() - t) / 50
    print(f"single query: {single * 1000:.1f} ms  ({1 / single:.0f} QPS)")

    t = time.perf_counter()
    col.search_batch(q, k=10)
    batch = time.perf_counter() - t
    print(f"batch of {args.queries}: {batch * 1000:.0f} ms  ({args.queries / batch:.0f} QPS)")
