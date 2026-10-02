"""Numba kernels for HNSW (Malkov & Yashunin, 2016).

Everything here works on plain arrays so Numba can compile it:

    level0     (capacity, 2M) int32   layer-0 neighbours, -1 padded
    upper      (rows, MAX_LEVEL, M) int32   layers 1.. for the few nodes that have them
    upper_row  (capacity,) int32      node -> row in `upper`, -1 if level 0 only
    node_level (capacity,) int8       top layer of each node, -1 if not in the graph
    state      [entry_point, max_level, n_upper_rows, visit_tag]

Distances are "lower is better": squared L2, or the negative dot product
for cosine/dot. The Python wrapper turns them back into similarities.
"""
from __future__ import annotations

import numpy as np
from numba import njit

METRIC_IP = 0
METRIC_L2 = 1

ENTRY, MAX_LEVEL, N_UPPER, TAG = 0, 1, 2, 3


@njit(cache=True, fastmath=True, inline="always")
def _dist(a, b, metric):
    s = np.float32(0.0)
    if metric == METRIC_L2:
        for i in range(a.shape[0]):
            d = a[i] - b[i]
            s += d * d
        return s
    for i in range(a.shape[0]):
        s += a[i] * b[i]
    return -s


# --- a tiny binary min-heap over parallel (distance, id) arrays ------------
@njit(cache=True, inline="always")
def _push(hd, hi, n, d, idx):
    j = n
    hd[j] = d
    hi[j] = idx
    while j > 0:
        p = (j - 1) >> 1
        if hd[p] <= hd[j]:
            break
        hd[p], hd[j] = hd[j], hd[p]
        hi[p], hi[j] = hi[j], hi[p]
        j = p
    return n + 1


@njit(cache=True, inline="always")
def _pop(hd, hi, n):
    """Remove the top. Read hd[0], hi[0] before calling."""
    n -= 1
    hd[0] = hd[n]
    hi[0] = hi[n]
    j = 0
    while True:
        c = 2 * j + 1
        if c >= n:
            break
        if c + 1 < n and hd[c + 1] < hd[c]:
            c += 1
        if hd[j] <= hd[c]:
            break
        hd[c], hd[j] = hd[j], hd[c]
        hi[c], hi[j] = hi[j], hi[c]
        j = c
    return n


@njit(cache=True)
def _grow(hd, hi):
    nd = np.empty(hd.shape[0] * 2, hd.dtype)
    ni = np.empty(hi.shape[0] * 2, hi.dtype)
    nd[: hd.shape[0]] = hd
    ni[: hi.shape[0]] = hi
    return nd, ni


@njit(cache=True, inline="always")
def _links(node, level, level0, upper, upper_row):
    if level == 0:
        return level0[node]
    return upper[upper_row[node], level - 1]


@njit(cache=True)
def _next_tag(state, visited):
    state[TAG] += 1
    if state[TAG] >= 4_000_000_000:  # uint32 wrap: clear and restart
        visited[:] = 0
        state[TAG] = 1
    return np.uint32(state[TAG])


@njit(cache=True)
def _search_layer(q, eps, ef, level, vectors, level0, upper, upper_row,
                  visited, state, metric, allowed, use_mask):
    """Best-first search on one layer.

    Returns (dists, ids, count) sorted nearest first. With use_mask, nodes
    outside `allowed` are still walked through but never enter the results.
    """
    tag = _next_tag(state, visited)
    cd = np.empty(max(64, 4 * ef), np.float32)   # candidates: min-heap
    ci = np.empty(cd.shape[0], np.int32)
    rd = np.empty(ef + 1, np.float32)            # results: max-heap via -dist
    ri = np.empty(ef + 1, np.int32)
    cn = 0
    rn = 0

    for t in range(eps.shape[0]):
        e = eps[t]
        if visited[e] == tag:
            continue
        visited[e] = tag
        d = _dist(q, vectors[e], metric)
        if cn == cd.shape[0]:
            cd, ci = _grow(cd, ci)
        cn = _push(cd, ci, cn, d, e)
        if not use_mask or allowed[e]:
            rn = _push(rd, ri, rn, -d, e)
            if rn > ef:
                rn = _pop(rd, ri, rn)

    while cn > 0:
        d = cd[0]
        c = ci[0]
        cn = _pop(cd, ci, cn)
        if rn == ef and d > -rd[0]:
            break
        nbrs = _links(c, level, level0, upper, upper_row)
        for t in range(nbrs.shape[0]):
            e = nbrs[t]
            if e < 0:
                break
            if visited[e] == tag:
                continue
            visited[e] = tag
            de = _dist(q, vectors[e], metric)
            if rn < ef or de < -rd[0]:
                if cn == cd.shape[0]:
                    cd, ci = _grow(cd, ci)
                cn = _push(cd, ci, cn, de, e)
                if not use_mask or allowed[e]:
                    rn = _push(rd, ri, rn, -de, e)
                    if rn > ef:
                        rn = _pop(rd, ri, rn)

    # Drain the max-heap: worst comes out first, so fill from the back.
    out_d = np.empty(rn, np.float32)
    out_i = np.empty(rn, np.int32)
    n = rn
    for j in range(rn - 1, -1, -1):
        out_d[j] = -rd[0]
        out_i[j] = ri[0]
        n = _pop(rd, ri, n)
    return out_d, out_i, rn


@njit(cache=True)
def _select_heuristic(cand_d, cand_i, count, m_max, vectors, metric, out):
    """Keep a candidate only if it is closer to the base node than to any
    neighbour already kept. Spreads links in different directions, which
    keeps the graph navigable on clustered data. Candidates come sorted."""
    m = 0
    for j in range(count):
        e = cand_i[j]
        good = True
        for r in range(m):
            if _dist(vectors[e], vectors[out[r]], metric) < cand_d[j]:
                good = False
                break
        if good:
            out[m] = e
            m += 1
            if m >= m_max:
                break
    return m


@njit(cache=True)
def _add_link(src, dst, level, m_max, vectors, level0, upper, upper_row, metric):
    """Add dst to src's list; if full, re-run the heuristic over list + dst."""
    nbrs = _links(src, level, level0, upper, upper_row)
    deg = 0
    while deg < m_max and nbrs[deg] >= 0:
        if nbrs[deg] == dst:
            return
        deg += 1
    if deg < m_max:
        nbrs[deg] = dst
        return
    cd = np.empty(m_max + 1, np.float32)
    ci = np.empty(m_max + 1, np.int32)
    for j in range(m_max):
        ci[j] = nbrs[j]
        cd[j] = _dist(vectors[src], vectors[nbrs[j]], metric)
    ci[m_max] = dst
    cd[m_max] = _dist(vectors[src], vectors[dst], metric)
    order = np.argsort(cd)
    cd = cd[order]
    ci = ci[order]
    kept = np.empty(m_max, np.int32)
    m = _select_heuristic(cd, ci, m_max + 1, m_max, vectors, metric, kept)
    for j in range(m_max):
        nbrs[j] = kept[j] if j < m else -1


@njit(cache=True)
def insert_many(vectors, slots, levels, state, level0, upper, upper_row,
                node_level, visited, M, ef_construction, metric):
    dummy = np.zeros(1, np.bool_)
    sel = np.empty(2 * M, np.int32)
    for t in range(slots.shape[0]):
        c = slots[t]
        lvl = levels[t]
        node_level[c] = lvl
        if lvl > 0:
            upper_row[c] = state[N_UPPER]
            state[N_UPPER] += 1
        if state[ENTRY] < 0:
            state[ENTRY] = c
            state[MAX_LEVEL] = lvl
            continue

        q = vectors[c]
        top = state[MAX_LEVEL]
        eps = np.empty(1, np.int32)
        eps[0] = state[ENTRY]
        for lc in range(top, lvl, -1):           # greedy descent above lvl
            _, ri, _ = _search_layer(q, eps, 1, lc, vectors, level0, upper,
                                     upper_row, visited, state, metric, dummy, False)
            eps = ri[:1].copy()

        for lc in range(min(top, lvl), -1, -1):
            rd, ri, rn = _search_layer(q, eps, ef_construction, lc, vectors, level0,
                                       upper, upper_row, visited, state, metric,
                                       dummy, False)
            m_max = 2 * M if lc == 0 else M
            m = _select_heuristic(rd, ri, rn, M, vectors, metric, sel)
            mine = _links(c, lc, level0, upper, upper_row)
            for j in range(mine.shape[0]):
                mine[j] = sel[j] if j < m else -1
            for j in range(m):
                _add_link(sel[j], c, lc, m_max, vectors, level0, upper, upper_row, metric)
            eps = ri[:rn].copy()

        if lvl > top:
            state[ENTRY] = c
            state[MAX_LEVEL] = lvl


@njit(cache=True)
def search_many(queries, k, ef, vectors, level0, upper, upper_row, visited,
                state, metric, allowed, use_mask):
    nq = queries.shape[0]
    out_i = np.full((nq, k), -1, np.int64)
    out_d = np.full((nq, k), np.inf, np.float32)
    if state[ENTRY] < 0:
        return out_i, out_d
    dummy = np.zeros(1, np.bool_)
    ef = max(ef, k)
    for qi in range(nq):
        q = queries[qi]
        eps = np.empty(1, np.int32)
        eps[0] = state[ENTRY]
        for lc in range(state[MAX_LEVEL], 0, -1):
            _, ri, _ = _search_layer(q, eps, 1, lc, vectors, level0, upper,
                                     upper_row, visited, state, metric, dummy, False)
            eps = ri[:1].copy()
        rd, ri, rn = _search_layer(q, eps, ef, 0, vectors, level0, upper, upper_row,
                                   visited, state, metric, allowed, use_mask)
        for j in range(min(k, rn)):
            out_i[qi, j] = ri[j]
            out_d[qi, j] = rd[j]
    return out_i, out_d
