# VectorForge: Product Requirements Document

Oct 2, 2026, Stef. The [Functional Design Document](FDD.md) describes how
each requirement was implemented; its traceability matrix maps every FR
below to code and tests.

## Overview

VectorForge is a small vector database built from first principles. It is
benchmarked honestly against FAISS and Qdrant, and the aim is to show how
approximate nearest-neighbour (ANN) search works, not just how to call it.

**Problem.** Most developers use vector databases such as Qdrant, Pinecone
or pgvector as black boxes. Few can explain how HNSW builds its graph, why
product quantization trades recall for memory, or how a database survives
a crash without losing writes. VectorForge answers those questions with
working code and measured numbers.

**Bet.** A working engine with recall-vs-latency curves on a public
dataset, plus a write-up of the design trade-offs, is stronger evidence
than a second integration project. Success is measured by understanding
and honest numbers, not by beating Qdrant.

## What we're building

A standalone vector database: a server you run, a client library you
install, and a benchmark suite that measures it. Every part is written
from scratch on NumPy, Numba, FastAPI and SQLite.

| Component | What it does | What building it teaches |
|---|---|---|
| Storage engine | Vectors in a memory-mapped float32 file, metadata in SQLite; every write goes to a write-ahead log first, with periodic snapshots | How databases survive crashes; why WAL + snapshot is the standard pattern |
| Flat index | Compares the query with every stored vector; exact but slow | The ground truth every other index is measured against |
| HNSW index | Multi-layer graph where each vector links to its nearest neighbours; search hops down from a sparse top layer | The algorithm inside most production vector databases, and its recall vs speed knobs |
| IVF-PQ index | Groups vectors into clusters, then compresses each vector to a few bytes with product quantization | How to fit 1M vectors in a fraction of the memory, and what recall that costs |
| Filtered search | Returns only points whose metadata matches, applied during the search | Why post-filtering returns too few results, and how real engines fix it |
| Query planner | Chooses graph search or brute force from how many points a filter keeps | Cost-based planning, the same idea SQL databases use |
| HTTP API + Python client | Collections, upsert and delete, search, batch search, stats | API design for a data service |
| Benchmark suite | Same datasets through VectorForge, FAISS, hnswlib and Qdrant; recall, QPS, latency and memory charts | Honest measurement with the ann-benchmarks method |
| Integrations | LangChain / LlamaIndex vector-store adapter and a semantic-search demo app | How frameworks plug into a vector store |

## Goals and non-goals

**Goals**

- Implement three index types from scratch: exact (flat), HNSW and IVF-PQ.
  No FAISS or hnswlib inside the engine.
- Reach >= 0.95 recall@10 on SIFT1M with HNSW at a latency within 3x of
  hnswlib on the same hardware.
- Support metadata filtering, deletes, and crash-safe persistence.
- Expose a small HTTP API and a Python client.
- Publish reproducible benchmarks and a design write-up with trade-offs.

**Non-goals (v1):** distributed clustering, sharding or replication; GPU
search; multi-tenancy, auth or rate limits; beating Qdrant or FAISS on raw
speed; production support or a stability promise.

## Audience

| Reader | What they need | Where they look |
|---|---|---|
| Hiring manager / interviewer | Evidence of systems depth in ten minutes | README, benchmark chart, design decisions, Studio |
| Technical interviewer | Code to probe on HNSW, PQ, WAL | `vectorforge/engine/`, tests, write-ups |
| Stef (builder) | A guided path through ANN internals | Milestones, notes per index |
| Other learners | A readable reference implementation | Docs, small-dataset quick start |

## Functional requirements

| ID | Requirement | Priority | Milestone | Status |
|---|---|---|---|---|
| FR-1 | Collections with fixed dimension and metric (cosine, dot, L2), created and dropped via API | Must | M1 | Done |
| FR-2 | Upsert, get and delete points by ID, each with a vector + JSON payload | Must | M1 | Done |
| FR-3 | Flat (exact) index, used as ground truth for all recall measurements | Must | M1 | Done |
| FR-4 | HNSW with tunable M, efConstruction, efSearch; incremental inserts; tombstone deletes + rebuild | Must | M2 | Done |
| FR-5 | IVF-PQ: k-means coarse quantizer, product quantization, nprobe, optional exact re-rank | Must | M3 | Done |
| FR-6 | Payload filtering (eq, in, range) during search, with brute force for highly selective filters | Must | M4 | Done |
| FR-7 | Durability: WAL, periodic snapshots, recovery with no acknowledged writes lost | Must | M4 | Done |
| FR-8 | HTTP API (FastAPI) + Python client: collections, points, search, batch search, stats | Must | M5 | Done |
| FR-9 | Benchmark harness: recall@k, QPS, p50/p99, build time, memory vs FAISS / hnswlib / Qdrant | Must | M5 | Harness done; SIFT1M/GloVe runs pending |
| FR-10 | Integrations: LangChain / LlamaIndex adapter and a semantic-search demo app | Should | M6 | LangChain + demo done |
| FR-11 | Hybrid search (dense + sparse BM25, RRF fusion) | Could | Later | |
| FR-12 | Rust or SIMD core for distance kernels and HNSW traversal | Could | Later | |

## Non-functional requirements

Measured on SIFT1M (1M x 128-d, L2) on Stef's PC, single process; first
guesses, to revisit after M2. Current status: [FDD section 9](FDD.md#9-non-functional-requirements).

| Area | Target |
|---|---|
| HNSW recall@10 | >= 0.95 at efSearch tuned for best QPS |
| HNSW latency | p50 <= 2 ms, p99 <= 10 ms per query (single thread) |
| HNSW vs hnswlib | QPS within 3x at equal recall |
| IVF-PQ memory | <= 1/8 of raw float32 vectors (<= ~64 MB for SIFT1M) |
| IVF-PQ recall@10 | >= 0.90 with exact re-rank of top 100 |
| Build time | HNSW on 1M vectors <= 15 min |
| Durability | kill -9 during ingest loses zero acknowledged writes |
| Filtered search | recall@10 >= 0.90 at 1% filter selectivity |
| Code quality | >= 85% test coverage on engine/; property tests vs flat index |

## Milestones

Part-time alongside the job search. If M2 misses its gate, everything
after it moves out rather than being cut.

| Milestone | Scope | Gate | Status |
|---|---|---|---|
| M1 | Storage, collections, points, flat index | Exact search matches NumPy brute force | Done |
| M2 | HNSW | Recall >= 0.95 vs flat, within 3x of hnswlib | Done on synthetic data |
| M3 | IVF-PQ | Recall >= 0.90 with re-rank, memory <= 1/8 | Done on synthetic data |
| M4 | Filters, planner, WAL, snapshots | kill -9 test passes; filtered recall >= 0.90 at 1% | Done |
| M5 | HTTP API, client, benchmark harness, Studio | API tests pass; harness runs every engine | Done |
| M6 | LangChain adapter, semantic-search demo | Demo works end to end | Done |

## Risks

| Risk | Impact | Mitigation | Outcome |
|---|---|---|---|
| Pure-Python HNSW too slow for 1M vectors | M2 slips; numbers look bad | NumPy distances; Numba for search and insert; 100k subset for iteration | Queries 2.2x hnswlib; build 2.4x, the main remaining risk |
| HNSW bugs hidden behind decent recall | Wrong conclusions | Property tests vs flat; graph invariant checks | In place |
| Filtered recall collapses at low selectivity | FR-6 missed | Selectivity estimate -> brute force over matches | Planner with fitted cost model |
| Scope creep into a Rust rewrite | Never ships | Rust stays out of v1 (FR-12) | Held |
| Benchmarks unfair to baselines | Credibility | Library defaults + matched settings; publish scripts and raw results | Harness records method and versions |

## Open questions (resolved)

- Numba, Cython or Rust for the hot loop? **Numba**, Rust later.
- Vectors in a separate mmap file or one segment file with the WAL?
  **Separate append-only mmap file**, WAL as its own file.
- Which integration first? **LangChain**.

## Success criteria for v1

- HNSW hits >= 0.95 recall@10 on SIFT1M within 3x of hnswlib QPS.
- The recall-QPS chart and design write-up are in the README.
- The demo app runs semantic search end to end, installed with one pip command.
- Stef can whiteboard HNSW insert/search and PQ encoding without notes.
