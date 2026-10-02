# Design decisions

Each entry: the decision, what else was considered, and what it costs.

## 1. Indexes store slots, not IDs
Every index works on row numbers in the vector file; the collection maps
IDs to slots. **Alternative:** indexes keyed by ID. **Why:** any index can
be thrown away and rebuilt from the vector file, and indexes stay plain
integer arrays that Numba can compile. **Cost:** one extra lookup per
result.

## 2. Vectors are append-only
Replacing a point writes a new slot; the old one becomes a hole.
**Alternative:** overwrite in place. **Why:** a saved index can be behind
after a crash but never wrong; the payload index never deletes postings;
no torn overwrites. **Cost:** the file only grows until compaction exists.

## 3. The write-ahead log is the source of truth
Only the WAL is fsynced per write; the vector file, SQLite and indexes are
derived and made durable at checkpoints. **Alternative:** fsync every file
per write. **Why:** one fsync per acknowledged batch; batching 1,000 points
made the fsync cost ~3%. **Cost:** recovery work on open, bounded by
`wal_max_bytes`.

## 4. WAL records are blind writes
"ID 5 is at slot 812 with this payload", never "change ID 5". **Why:**
replaying records that were already applied converges to the same state,
so recovery doesn't need to know how far the apply got. **Cost:** records
carry full vectors and payloads.

## 5. Payloads in SQLite, filters in memory
SQLite is the durable store; a payload index (postings + numeric columns)
answers filters. **Alternative:** query SQLite per search. **Why:** a mask
over 1M slots in milliseconds instead of a table scan per query.
**Cost:** memory per indexed field, rebuilt from SQLite at open.

## 6. Every metric is "higher is better" internally
Cosine vectors are normalised on write; L2 ranks by negative squared
distance. **Why:** every index finds a top-k of one quantity. **Cost:**
cosine collections return normalised vectors from `get`.

## 7. HNSW as flat NumPy arrays compiled by Numba
Fixed-width neighbour arrays per layer, -1 padded. **Alternative:** Python
objects, Cython, Rust. **Why:** Python objects would be ~100x slower;
Numba keeps the code readable Python. **Cost:** 2.2x hnswlib query speed,
2.4x build time; Rust stays a later option (FR-12).

## 8. Neighbour-selection heuristic, not "M nearest"
**Why:** plain nearest neighbours cluster all links inside dense regions
and the graph splits into islands on clustered data. **Cost:** extra
distance computations at insert.

## 9. Tombstones plus rebuild at 25%
**Alternative:** repair the graph on delete. **Why:** repair is the
hardest part of HNSW to get right; tombstones keep routing intact.
**Cost:** a synchronous rebuild when the threshold is crossed.

## 10. IVF-PQ with residual encoding and exact re-rank
Codes encode `x - centroid`; the top `rerank` candidates are re-scored
from raw vectors. **Why:** residuals quantise far better than raw vectors,
and re-rank recovers ranking lost to compression (recall 0.71 -> 1.00).
**Cost:** raw vectors stay on disk (mmap) for re-rank.

## 11. Default PQ m = dim/4
**Alternative:** dim/8 (more compression). **Why:** at dim/8, recall@10
with re-rank 100 was 0.88 in tests; at dim/4, 0.99, still inside the
<= 1/8 memory target.

## 12. A cost-based planner instead of always using the index
**Why:** post-filtering collapsed to 0.9% recall at 0.1% selectivity, and
the masked graph walk dropped to 18 QPS. Scanning the matching points
exactly was both correct and fastest up to ~20% selectivity at 100k.
**Cost:** a cost constant to recalibrate per hardware.

## 13. No ef boost for filtered HNSW
Tried `ef / sqrt(selectivity)`; the benchmark showed recall was already
>= 0.995 without it, and the boost only slowed things down. Removed.

## 14. Generated IDs are 53-bit
Text and LangChain IDs are hashes truncated to 53 bits. **Why:** found
in Studio testing: 63-bit IDs lose precision in JavaScript.

## 15. Studio is one static page on the public API
**Alternative:** a React app with a build step. **Why:** nothing to build
or install, and it can't do anything the API can't, so it doubles as an
API demo. **Cost:** hand-written DOM code.
