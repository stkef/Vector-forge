"""k-means (Lloyd's algorithm), used for the IVF coarse quantizer and the
product-quantization codebooks. Always clusters by L2."""
from __future__ import annotations

import numpy as np


def assign(x: np.ndarray, centroids: np.ndarray, chunk: int = 32_768) -> np.ndarray:
    """Index of the nearest centroid (L2) for every row of x."""
    c2 = np.einsum("ij,ij->i", centroids, centroids)
    out = np.empty(len(x), np.int64)
    for s in range(0, len(x), chunk):
        block = x[s:s + chunk]
        # argmin ||x - c||^2 = argmin (||c||^2 - 2 x.c); ||x||^2 is constant per row
        out[s:s + chunk] = np.argmin(c2[None, :] - 2.0 * (block @ centroids.T), axis=1)
    return out


def kmeans(x: np.ndarray, k: int, iters: int = 20, seed: int = 0,
           max_points: int | None = None) -> np.ndarray:
    """Return (k, d) float32 centroids.

    k-means++ seeding (on at most 20k points, to bound its k sequential
    passes), Lloyd iterations, and empty clusters fixed by splitting the
    largest cluster in two (the trick FAISS uses).
    """
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(x, np.float32)
    if max_points and len(x) > max_points:
        x = x[rng.choice(len(x), max_points, replace=False)]
    n = len(x)
    if n < k:
        raise ValueError(f"need at least {k} points to train {k} clusters, got {n}")

    c = _kmeans_pp(x[rng.choice(n, min(n, max(20_000, k)), replace=False)], k, rng)
    for _ in range(iters):
        labels = assign(x, c)
        counts = np.bincount(labels, minlength=k)
        order = np.argsort(labels, kind="stable")
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        full = counts > 0
        sums = np.add.reduceat(x[order], starts[full], axis=0)
        c[full] = sums / counts[full, None]

        for j in np.nonzero(~full)[0]:
            big = int(np.argmax(counts))
            noise = rng.standard_normal(c.shape[1]).astype(np.float32) * 1e-4
            c[j] = c[big] + noise
            c[big] = c[big] - noise
            counts[j] = counts[big] // 2
            counts[big] -= counts[j]
    return c.astype(np.float32)


def _kmeans_pp(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Pick each new seed with probability proportional to its squared
    distance from the nearest seed so far: seeds end up spread out."""
    c = np.empty((k, x.shape[1]), np.float32)
    c[0] = x[rng.integers(len(x))]
    d2 = ((x - c[0]) ** 2).sum(1)
    for j in range(1, k):
        total = d2.sum()
        i = rng.choice(len(x), p=d2 / total) if total > 0 else rng.integers(len(x))
        c[j] = x[i]
        d2 = np.minimum(d2, ((x - c[j]) ** 2).sum(1))
    return c
