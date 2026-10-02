"""The flat index must match a naive NumPy brute force exactly."""
import numpy as np
import pytest

from vectorforge.engine.index import FlatIndex
from vectorforge.engine.metrics import Metric, prepare


def naive_topk(q, x, k, metric, allowed):
    if metric is Metric.L2:
        sims = -((x - q) ** 2).sum(axis=1)
    else:
        sims = x @ q
    sims = np.where(allowed, sims, -np.inf)
    order = np.argsort(-sims, kind="stable")[:k]
    return [int(i) for i in order if np.isfinite(sims[i])]


@pytest.mark.parametrize("metric", list(Metric))
@pytest.mark.parametrize("chunk", [7, 100, 65_536])  # chunk boundaries must not matter
def test_matches_naive(metric, chunk, rng):
    x = prepare(rng.standard_normal((1_000, 16)).astype(np.float32), metric)
    qs = prepare(rng.standard_normal((25, 16)).astype(np.float32), metric)
    allowed = rng.random(1_000) > 0.3
    idx = FlatIndex(lambda: x, metric, chunk_size=chunk)
    slots, sims = idx.search(qs, 10, allowed)
    for q, row in zip(qs, slots):
        assert row.tolist() == naive_topk(q, x, 10, metric, allowed)
    assert np.all(np.diff(sims, axis=1) <= 0), "results must be sorted best first"


def test_k_larger_than_allowed_pads(rng):
    x = rng.standard_normal((50, 8)).astype(np.float32)
    allowed = np.zeros(50, bool)
    allowed[[3, 17, 40]] = True
    slots, sims = FlatIndex(lambda: x, Metric.DOT, chunk_size=16).search(x[:2], 10, allowed)
    assert sorted(slots[0][:3].tolist()) == [3, 17, 40]
    assert (slots[:, 3:] == -1).all() and np.isneginf(sims[:, 3:]).all()
