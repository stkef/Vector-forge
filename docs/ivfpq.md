# IVF-PQ in VectorForge

Code: `vectorforge/engine/index/ivfpq.py` and `kmeans.py`. Reference:
Jégou, Douze & Schmid, *Product Quantization for Nearest Neighbor Search* (2011).

## Two ideas stacked

**IVF (inverted file).** k-means splits the space into `nlist` cells. Each
vector is stored in the list of its nearest centroid. A query only opens the
`nprobe` lists whose centroids are closest, so it touches ~nprobe/nlist of
the data.

**PQ (product quantization).** Inside a list, a vector is stored as its
residual from the centroid, cut into `m` sub-vectors. Each sub-vector is
replaced by the index (one byte) of its nearest codeword in a 256-word
codebook learned for that subspace. 128 floats (512 bytes) become 32 bytes.

**ADC (asymmetric distance).** The query is never compressed. For each probed
list, build a table `LUT[j][c]` = distance from query sub-vector j to
codeword c (m x 256 numbers). A stored vector's distance is then m table
lookups added up. That loop is the Numba kernel `_adc`.

**Re-rank.** Compressed distances get the neighbourhood right but the order
wrong. Taking the best `rerank` (default 100) candidates and recomputing
exact distances from the raw vectors on disk recovers almost all of it.

## Measured: 100k x 128 synthetic, single thread, nlist 1264, m 32

| | VectorForge | FAISS IndexIVFPQ |
|---|---:|---:|
| recall@10, nprobe 16, no re-rank | 0.707 | 0.683 |
| recall@10, nprobe 16, re-rank 100 | 1.000 | 1.000 |
| QPS, nprobe 16, re-rank 100 | 1,908 | 8,052 |
| train / add | 48.5 s / 2.9 s | 17.0 s / 1.5 s |
| index memory (codes + ids + model) | 4.4 MB = 8.6% of raw | |

Same recall as FAISS at the same settings, so the quantizer is right; 4x
slower per query because the probed lists are walked from Python (one
Numba call per list). Moving the whole per-query loop into one kernel is the
obvious next step.

## Things learned the hard way

- **Random k-means seeds put two centroids in one cluster** and never
  recover (a test with 4 obvious clusters failed). k-means++ seeding fixed
  it: each new seed is picked with probability proportional to its squared
  distance from the seeds so far.
- **Empty clusters** happen with duplicate points; they are fixed by
  splitting the largest cluster (the FAISS trick).
- **m is the real recall knob, not nprobe.** At m = dim/8 (1/32 of float32)
  recall@10 with re-rank 100 was 0.88 in the tests; at m = dim/4 it was 0.99.
  Default is dim/4, which still fits the "<= 1/8 of raw memory" target.
- **IVF needs data before it can exist.** The collection searches exactly
  until `train_after` points (min 256, one codebook's worth), then trains on
  a sample. Centroids learned on early data drift as data grows:
  `col.train_index()` retrains on everything; best called after a bulk load.
