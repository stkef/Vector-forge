"""Distance metrics.

Internally every metric becomes a *similarity* where higher is better, so the
indexes only ever need to find the top-k largest values:

- cosine: vectors are L2-normalised on the way in, then similarity = dot product
- dot:    similarity = dot product
- l2:     similarity = -||q - v||^2  (converted back to a distance for users)
"""
from __future__ import annotations

from enum import Enum

import numpy as np


class Metric(str, Enum):
    COSINE = "cosine"
    DOT = "dot"
    L2 = "l2"


def as_matrix(vectors, dim: int) -> np.ndarray:
    """Validate and convert input to a contiguous (n, dim) float32 matrix."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"expected vectors of dimension {dim}, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("vectors must not contain NaN or infinity")
    return np.ascontiguousarray(arr)


def prepare(vectors: np.ndarray, metric: Metric) -> np.ndarray:
    """Transform vectors into the form that is stored and searched."""
    if metric is Metric.COSINE:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("the cosine metric cannot use a zero vector")
        return (vectors / norms).astype(np.float32)
    return vectors


def similarity(queries: np.ndarray, vectors: np.ndarray, metric: Metric) -> np.ndarray:
    """(n_queries, n_vectors) similarity matrix. Higher is always better."""
    dots = queries @ vectors.T
    if metric is Metric.L2:
        q2 = np.einsum("ij,ij->i", queries, queries)[:, None]
        v2 = np.einsum("ij,ij->i", vectors, vectors)[None, :]
        return -(q2 + v2 - 2.0 * dots)
    return dots


def to_user_score(sim: np.ndarray, metric: Metric) -> np.ndarray:
    """Convert internal similarity to what the API reports.

    cosine/dot: the similarity itself (higher = closer).
    l2: the Euclidean distance (lower = closer).
    """
    if metric is Metric.L2:
        return np.sqrt(np.maximum(-sim, 0.0))
    return sim
