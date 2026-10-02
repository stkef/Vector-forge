# Benchmarks: method

`scripts/bench.py` runs every engine on the same vectors and scores them
against the same exact ground truth (the dataset's own neighbours for
ann-benchmarks files, or brute force).

| Engine | What it is | Matched settings |
|---|---|---|
| `vf-hnsw` | VectorForge HNSW | M, ef_construction; sweep ef |
| `hnswlib` | the reference HNSW (C++) | same |
| `faiss-hnsw` | FAISS IndexHNSWFlat | same |
| `qdrant` | Qdrant server (Rust), over HTTP or gRPC | same m, ef_construct; sweep hnsw_ef |
| `vf-ivfpq` | VectorForge IVF-PQ + exact re-rank | nlist, m, 8-bit codes, re-rank 100; sweep nprobe |
| `faiss-ivfpq` | FAISS IndexIVFPQ + IndexRefineFlat | same nlist, m, re-rank depth |

Rules:

- One query at a time, one thread, each query timed separately: QPS is
  1 / mean latency, plus p50 and p99. This includes each library's
  per-call overhead, as an application would see it.
- Library defaults apart from the matched settings. No engine-specific tuning.
- Build time is wall-clock for inserting everything (and training, for IVF-PQ).
- Qdrant runs in Docker (`bench/docker-compose.yml`) and is the only engine
  behind a network call, which adds latency; the results file records its
  version. `--qdrant-url :memory:` is qdrant-client's local mode, a Python
  brute-force stand-in, useful only to check the adapter (it scored 244 QPS
  at recall 1.0 whatever ef was).

```bash
pip install -e ".[bench]"
docker compose -f bench/docker-compose.yml up -d
python scripts/get_datasets.py sift glove
python scripts/bench.py --dataset datasets/sift-128-euclidean.hdf5
python scripts/bench.py --dataset datasets/glove-100-angular.hdf5
python scripts/bench_filtered.py --dataset datasets/sift-128-euclidean.hdf5 --limit 200000
```

Each run writes `results/bench-<dataset>.json` (with machine details) and a
`.png` chart; Studio's Benchmarks page draws the same data.
