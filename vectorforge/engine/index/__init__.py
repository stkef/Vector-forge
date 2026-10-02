from vectorforge.engine.index.base import Index
from vectorforge.engine.index.flat import FlatIndex
from vectorforge.engine.index.hnsw import HNSWIndex
from vectorforge.engine.index.ivfpq import IVFPQIndex

__all__ = ["Index", "FlatIndex", "HNSWIndex", "IVFPQIndex", "warmup"]


def warmup() -> None:
    """Load the compiled Numba kernels (from cache, or compile them) on a
    tiny index, so a server's first real query isn't the one that pays."""
    import numpy as np

    from vectorforge.engine.metrics import Metric
    x = np.random.default_rng(0).standard_normal((300, 8)).astype(np.float32)
    for metric in (Metric.L2, Metric.COSINE):
        h = HNSWIndex(lambda: x, metric, M=4)
        h.add(np.arange(300))
        h.search(x[:2], 3)
        h.search(x[:2], 3, np.ones(300, bool))
    ix = IVFPQIndex(lambda: x, Metric.L2, 8, nlist=4, m=2)
    ix.train(x)
    ix.add(np.arange(300))
    ix.search(x[:2], 3)
    ix_ip = IVFPQIndex(lambda: x, Metric.DOT, 8, nlist=4, m=2)
    ix_ip.train(x)
    ix_ip.add(np.arange(300))
    ix_ip.search(x[:2], 3)
