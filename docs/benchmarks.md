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

## Results: GloVe 6B-100d (real word embeddings)

- **Data:** 400,000 GloVe 6B 100-d word vectors (gensim `glove-wiki-gigaword-100`), cosine. 1,000 random words are held out as queries and the other 399,000 form the base. Ground truth comes from FAISS `IndexFlatIP` (exact).
- **Run:** [`notebooks/vectorforge-vs-faiss-vs-hnswlib.ipynb`](../notebooks/vectorforge-vs-faiss-vs-hnswlib.ipynb), on 2 cores with one thread per engine. Qdrant is not included because a notebook has no Docker.
- **Results:** [`bench-results/glove-6b-100/bench.json`](bench-results/glove-6b-100/bench.json) (with machine details) and `bench.csv`.

| recall target | vf-hnsw | hnswlib | faiss-hnsw | vf-ivfpq | faiss-ivfpq |
|---|---|---|---|---|---|
| >= 0.80 | 3,808 | 11,705 | 11,192 | 1,877 | 5,853 |
| >= 0.90 | 1,626 | 4,325 | 4,412 | 942 | 2,300 |
| >= 0.95 | 666 | 1,968 | 1,717 | 539 | 1,186 |

Best QPS (single thread) that reaches each recall@10 target. No engine reached 0.99 within ef <= 256 / nprobe <= 128.

| | build (1 thread) | index size | max recall |
|---|---|---|---|
| vf-hnsw | 337 s | 76.8 MB | 0.963 |
| hnswlib | 144 s | | 0.963 |
| faiss-hnsw | 154 s | | 0.960 |
| vf-ivfpq | 70 s | 12.1 MB | 0.957 |
| faiss-ivfpq | 29 s | 12.6 MB | 0.957 |

**What it says:**
- **Recall matches.** VectorForge's recall at each setting matches the references within about 0.01, so the graph and the quantizer behave like theirs.
- **The speed gap grows on real data.** It is 0.33x to 0.45x here, against 0.45x on synthetic data, because GloVe queries need many more hops.
- **Fixed per-query cost dominates at low ef.** vf-hnsw is slightly slower at ef 10 than at ef 16, a sign that the Python call around each query is the floor.

**The same notebook on Kaggle** (CPU notebook, results in [`bench-results/glove-6b-100/bench-kaggle.csv`](bench-results/glove-6b-100/bench-kaggle.csv)):

| recall target | vf-hnsw | hnswlib | faiss-hnsw | vf-ivfpq | faiss-ivfpq |
|---|---|---|---|---|---|
| >= 0.80 | 3,248 | 8,426 | 6,889 | 1,212 | 2,825 |
| >= 0.90 | 1,590 | 3,309 | 2,899 | 463 | 1,083 |
| >= 0.95 | 725 | 1,509 | 1,237 | 263 | 624 |

The recall at every setting matched the first run to three decimals. Speed ratios vary with the CPU: VectorForge ran at 0.39x to 0.59x of the C++ libraries on Kaggle, against 0.33x to 0.45x on the 2-core VM.

