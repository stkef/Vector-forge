# Changelog

## Unreleased

- **Benchmark on real embeddings:** GloVe 6B-100d (399k vectors) against hnswlib and FAISS, as a re-runnable notebook (`notebooks/`) with results in `docs/bench-results/glove-6b-100/`.
  - Same recall as the references.
  - 0.33x to 0.45x their single-thread QPS.
  - IVF-PQ index at 7.9% of raw size.

## 0.1.0 (2026-10-03)

First release, milestones M1 to M6.

- **Storage:** append-only memory-mapped vector file, SQLite payloads,
  collections with cosine / dot / L2.
- **Indexes:** flat (exact), HNSW (Numba, heuristic neighbour selection,
  tombstones, rebuild, saved graph), IVF-PQ (k-means++, residual product
  quantization, ADC, exact re-rank, automatic training).
- **Filtering:** eq / in / range on payload fields, in-memory payload
  index, cost-based planner choosing exact scan or masked index search.
- **Durability:** CRC-checked write-ahead log, checkpoints, idempotent
  replay, torn-tail truncation; SIGKILL test.
- **Server:** FastAPI HTTP API, Python client, CLI (`serve`, `ingest`),
  Studio console.
- **Integrations:** LangChain vector store; Gemini and offline embedders.
- **Benchmarks:** one harness for VectorForge, hnswlib, FAISS and Qdrant;
  filtered-search benchmark.
