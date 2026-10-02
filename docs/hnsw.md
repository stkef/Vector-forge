# HNSW in VectorForge

Code: `vectorforge/engine/index/hnsw.py` (Python wrapper) and
`_hnsw_kernels.py` (Numba). Paper: Malkov & Yashunin, *Efficient and robust
approximate nearest neighbor search using Hierarchical Navigable Small World
graphs* (2016).

## The idea in one paragraph

Every vector is a node. Each node links to a few near neighbours, chosen so
the links point in different directions. A small random subset of nodes also
lives on higher layers with longer links, like an express train. Search
starts at the top layer, walks greedily towards the query, drops a layer,
repeats, and on layer 0 runs a beam search of width `ef` that returns the
best `k`.

## Data layout (why it is fast enough in Python)

The graph is plain NumPy arrays, never Python objects, so Numba compiles
the hot loops to machine code:

| Array | Shape | Meaning |
|---|---|---|
| `level0` | (capacity, 2M) int32 | layer-0 links, `-1` padded |
| `upper` | (rows, 16, M) int32 | layers 1+, only for nodes that reach them (~1/M of nodes) |
| `upper_row` | (capacity,) int32 | node -> row in `upper` |
| `node_level` | (capacity,) int8 | a node's top layer, `-1` = not in the graph |
| `visited` | (capacity,) uint32 | "visited" stamp per search, so it is never cleared |

Layer-0 memory is `N x 2M x 4` bytes: 128 MB for 1M vectors at M = 16.

## Insert

1. Draw a level `floor(-ln(U) / ln(M))`, so each layer has ~1/M of the one below.
2. Greedy descent (beam width 1) from the entry point down to that level.
3. On each layer from there to 0: beam search with `ef_construction`,
   pick neighbours with the **heuristic**, link both ways.
4. If a neighbour's list overflows, re-run the heuristic over its list plus
   the new node.

**The heuristic:** go through candidates nearest first; keep one only if it
is closer to the new node than to any neighbour already kept. Plain "M
nearest" puts all links inside one dense cluster and the graph falls apart
into islands on clustered data.

## Search, deletes and filters

Deletes are tombstones: the node keeps its links so routes through it still
work, but it never enters the result heap. The same mechanism takes an
`allowed` mask, which is how M4 filtered search will plug in. Past 25%
tombstones the collection rebuilds the graph from live vectors.

Vectors are append-only. Replacing a point writes a new slot, so a slot's
contents never change. That is why a saved graph (`hnsw.npz`) can be
trusted after a crash: on open, VectorForge loads it, tombstones what was
deleted since, and inserts what was added since.

## Measured (single thread, batch queries, same M=16 / efC=200)

Synthetic clustered data, 100k x 128, L2, 1,000 queries, Linux container:

| ef | VectorForge recall@10 | QPS | hnswlib recall@10 | QPS | ratio |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.974 | 14,656 | 0.972 | 38,894 | 2.7x |
| 32 | 0.999 | 10,559 | 0.999 | 23,730 | 2.2x |
| 64 | 1.000 | 6,643 | 1.000 | 15,209 | 2.3x |

Build: 65 s vs 27 s (2.4x). Same recall at the same ef means the graph is
built the same way; the gap is implementation speed. SIFT1M numbers go here
once run on the PC.

## Things learned the hard way

- **Raw dot product is not a metric.** With un-normalised vectors and
  queries unlike the data, recall@10 at ef = 64 was 0.83, and hnswlib got
  0.85 on the same data. The heuristic assumes the triangle inequality. Use
  cosine for embeddings; use dot only when vectors are already normalised.
- Never clear the visited array: stamp it with a counter instead. Clearing
  1M entries per query would cost more than the search.
