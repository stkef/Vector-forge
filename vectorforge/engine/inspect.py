"""Read-only views into the indexes, for the Studio and for learning.

Nothing here is on the hot path: it re-implements the HNSW descent in plain
Python so every hop can be recorded.
"""
from __future__ import annotations

import heapq

import numpy as np

from vectorforge.engine.index import HNSWIndex, IVFPQIndex
from vectorforge.engine.metrics import Metric, similarity


def _dist(metric: Metric, a: np.ndarray, b: np.ndarray) -> float:
    if metric is Metric.L2:
        d = a - b
        return float(d @ d)
    return float(-(a @ b))


def _links(h: HNSWIndex, node: int, level: int) -> np.ndarray:
    row = h.level0[node] if level == 0 else h.upper[h.upper_row[node], level - 1]
    return row[row >= 0]


def hnsw_trace(h: HNSWIndex, q: np.ndarray, ef: int, max_expansions: int = 400) -> dict:
    """Record the greedy walk down the upper layers and the beam search on layer 0."""
    vecs = np.asarray(h._get_vectors())
    metric = h.metric
    entry = int(h.state[0])
    if entry < 0:
        return {"hops": [], "expanded": [], "visited": 0, "found": []}
    hops = [{"layer": int(h.state[1]), "slot": entry, "dist": _dist(metric, q, vecs[entry])}]
    cur, cur_d = entry, hops[0]["dist"]
    for level in range(int(h.state[1]), 0, -1):
        moved = True
        while moved:
            moved = False
            for nb in _links(h, cur, level):
                d = _dist(metric, q, vecs[nb])
                if d < cur_d:
                    cur, cur_d, moved = int(nb), d, True
            if moved:
                hops.append({"layer": level, "slot": cur, "dist": cur_d})
        hops.append({"layer": level - 1, "slot": cur, "dist": cur_d})

    # layer 0: best-first beam search, recording expansion order
    visited = {cur}
    cand = [(cur_d, cur)]
    best = [(-cur_d, cur)]
    expanded = []
    while cand:
        d, c = heapq.heappop(cand)
        if len(best) >= ef and d > -best[0][0]:
            break
        if len(expanded) < max_expansions:
            expanded.append(int(c))
        for nb in _links(h, c, 0):
            nb = int(nb)
            if nb in visited:
                continue
            visited.add(nb)
            dn = _dist(metric, q, vecs[nb])
            if len(best) < ef or dn < -best[0][0]:
                heapq.heappush(cand, (dn, nb))
                heapq.heappush(best, (-dn, nb))
                if len(best) > ef:
                    heapq.heappop(best)
    found = [s for _, s in sorted(((-d, s) for d, s in best))]
    return {"hops": hops, "expanded": expanded, "visited": len(visited), "found": found}


def hnsw_structure(h: HNSWIndex) -> dict:
    n = h._cap
    levels = h.node_level[:n]
    live = levels >= 0
    per_layer = [int((levels >= lvl).sum()) for lvl in range(int(h.state[1]) + 1)]
    deg = (h.level0[:n][live] >= 0).sum(axis=1) if live.any() else np.zeros(0, int)
    hist = np.bincount(deg, minlength=2 * h.M + 1).tolist()
    return {"nodes_per_layer": per_layer, "degree_histogram_layer0": hist,
            "max_degree_layer0": 2 * h.M, "entry_point": int(h.state[0])}


def ivf_structure(ix: IVFPQIndex) -> dict:
    sizes = [L.n for L in ix.lists]
    return {"list_sizes": sizes, "nlist": ix.nlist, "m": ix.m}


def ivf_probes(ix: IVFPQIndex, q: np.ndarray, nprobe: int) -> list[int]:
    if not ix.trained:
        return []
    s = similarity(q[None, :], ix.centroids, ix.metric)[0]
    return np.argsort(-s)[: min(nprobe, ix.nlist)].tolist()


def pca_2d(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, 2 x d basis) of the top-2 principal components."""
    mean = sample.mean(axis=0)
    _, _, vt = np.linalg.svd(sample - mean, full_matrices=False)
    return mean, vt[:2]
