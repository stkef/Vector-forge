# Testing

```bash
python -m pytest                 # 90 tests, ~20 s (first run also compiles Numba kernels)
python -m pytest tests/test_durability.py -v
```

| File | Tests | What it proves |
|---|---|---|
| `test_flat_index.py` | 10 | Flat search equals naive NumPy brute force for every metric and chunk size; padding when fewer than k match |
| `test_collection.py` | 14 | Collection CRUD, validation, replacement via new slot, last-wins duplicates, scores per metric, exact search at scale, persistence |
| `test_hnsw.py` | 11 | Recall vs flat per metric, graph invariants (degree caps, no self-loops, packed links, layer-0 reachability), tombstones, filters, save/load, crash catch-up, rebuild |
| `test_ivfpq.py` | 9 | k-means correctness and empty clusters, recall with re-rank, memory fraction, tombstones, filters, save/load, training lifecycle |
| `test_filters.py` | 24 | Filter validation, payload-index semantics, filtered recall and the plan chosen at 1-90% selectivity for HNSW and IVF-PQ, planner edge cases |
| `test_durability.py` | 10 | WAL round trip, torn tail, corrupt byte, checkpoint, auto-checkpoint, replay after losing un-synced state (all index types), idempotent replay, **SIGKILL mid-ingest** |
| `test_server.py` | 7 | API status codes, points, search with plan and recall, inspect and trace, text endpoints, benchmark listing, Python client |
| `test_langchain.py` | 3 | Store from texts, filters, relevance scores, delete, retriever, remote store over HTTP |
| `test_cli.py` | 2 | Chunking and idempotent ingest |

## How recall is tested

Approximate indexes are never compared with hard-coded answers. Each test
computes the exact answer with the flat index on the same data and asserts
recall@10 above a threshold (0.95 for HNSW and filtered search, 0.95 for
IVF-PQ with re-rank). Raw inner product uses 0.92 at ef=256, the level
hnswlib reaches on the same data.

## How durability is tested

- **Process crash:** a child process upserts batches and prints each batch
  number after `upsert` returns. The test kills it (SIGKILL on Linux,
  TerminateProcess on Windows) after 40+ batches and several checkpoints,
  reopens the data, and checks every printed batch byte for byte.
- **Power loss model:** copy the collection at a checkpoint, write more,
  then restore the old vector file and SQLite but keep the newer WAL.
  Everything acknowledged must come back, for flat, HNSW and IVF-PQ.
- **Idempotence:** close without checkpoint after applying, reopen, and
  check the replayed state is identical.

## Continuous integration

`.github/workflows/ci.yml` runs the suite on Ubuntu and Windows with
Python 3.12 on every push and pull request.
