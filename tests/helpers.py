import numpy as np


def clustered(rng, n, dim, centers=50):
    c = rng.standard_normal((centers, dim)).astype(np.float32) * 3
    return (c[rng.integers(0, centers, n)] + rng.standard_normal((n, dim))).astype(np.float32)


def recall(found, truth, k):
    return float(np.mean([len(set(list(a)[:k]) & set(list(b)[:k])) / k
                          for a, b in zip(found, truth)]))


def ids(results):
    return [[h.id for h in r] for r in results]
