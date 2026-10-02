"""Filtered search: recall@10 and QPS by filter selectivity, three ways.

  post_filter  search the index for k x 10, then drop non-matching hits
               (what you get by bolting a filter onto an unfiltered index)
  hnsw_mask    filter mask applied inside the graph walk
  subset_scan  exact brute force over just the matching points
  planner      what VectorForge picks automatically

    python scripts/bench_filtered.py --synthetic 100000 --dim 128
    python scripts/bench_filtered.py --dataset datasets/sift-128-euclidean.hdf5 --limit 200000
"""
from __future__ import annotations

import argparse
import csv
import tempfile
import time
from pathlib import Path

import numpy as np

from vectorforge import VectorForge


def load(args):
    if args.synthetic:
        rng = np.random.default_rng(0)
        c = rng.standard_normal((2_000, args.dim)).astype(np.float32) * 2
        mk = lambda n: (c[rng.integers(0, len(c), n)] + rng.standard_normal((n, args.dim))).astype(np.float32)  # noqa: E731
        return f"synthetic-{args.synthetic}", mk(args.synthetic), mk(args.queries)
    import h5py
    with h5py.File(args.dataset, "r") as f:
        return (Path(args.dataset).stem, np.asarray(f["train"][: args.limit], np.float32),
                np.asarray(f["test"][: args.queries], np.float32))


def recall(a, b):
    return float(np.mean([len(set(x) & set(y)) / max(1, len(y)) for x, y in zip(a, b)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--synthetic", type=int)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--limit", type=int, default=200_000)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    name, x, q = load(args)
    k, n = args.k, len(x)
    # bucket = i % 1000 gives filters at any selectivity: bucket < s*1000
    payloads = [{"bucket": i % 1_000} for i in range(n)]

    with tempfile.TemporaryDirectory() as tmp, VectorForge(tmp) as db:
        col = db.create_collection("b", x.shape[1], "l2", storage={"fsync": False})
        t = time.perf_counter()
        for s in range(0, n, 20_000):
            col.upsert(list(range(s, min(s + 20_000, n))), x[s:s + 20_000], payloads[s:s + 20_000])
        print(f"{name}: {n:,} x {x.shape[1]} loaded + HNSW built in {time.perf_counter() - t:.0f}s")
        col.search_batch(q[:2], k, filter={"bucket": {"lt": 500}})

        def run(fn):
            t = time.perf_counter()
            out = fn()
            return out, len(q) / (time.perf_counter() - t)

        rows = []
        print(f"\n{'selectivity':>11} {'points':>8}  {'strategy':<12}{'recall':>8}{'QPS':>9}")
        for sel in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9]:
            flt = {"bucket": {"lt": int(sel * 1_000)}}
            truth = [[h.id for h in r] for r in col.search_batch(q, k, filter=flt, exact=True)]
            idx = col._index
            allowed = col._allowed(flt)
            pts = int(allowed.sum())
            res = {}

            def post_filter():
                slots, _ = idx.search(col._prep(q), k * 10, ef=max(64, k * 10))
                out = []
                for row in slots:
                    keep = [col._slot_ids[s] for s in row if s >= 0 and allowed[s]]
                    out.append(keep[:k])
                return out
            res["post_filter"] = run(post_filter)

            def mask(ef):
                def f():
                    slots, _ = idx.search(col._prep(q), k, allowed, ef=ef)
                    return [[col._slot_ids[s] for s in row if s >= 0] for row in slots]
                return f
            res["hnsw_mask"] = run(mask(64))
            res["subset_scan"] = run(lambda: [[h.id for h in r] for r in
                                              col.search_batch(q, k, filter=flt, exact=True)])
            res["planner"] = run(lambda: [[h.id for h in r] for r in
                                          col.search_batch(q, k, filter=flt)])
            plan = col.explain(k, filter=flt)["strategy"]
            for strat, (found, qps) in res.items():
                r = recall(found, truth)
                label = f"{strat} ({plan})" if strat == "planner" else strat
                print(f"{sel:>11.1%} {pts:>8,}  {label:<12}{r:>8.3f}{qps:>9,.0f}")
                rows.append(dict(selectivity=sel, points=pts, strategy=strat, plan=plan,
                                 recall=r, qps=qps))
            print()

    out = Path("results")
    out.mkdir(exist_ok=True)
    path = out / f"filtered-{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
