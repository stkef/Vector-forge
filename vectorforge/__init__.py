"""VectorForge: a small vector database built from first principles."""
from vectorforge.db import VectorForge
from vectorforge.engine.collection import Collection, Point, ScoredPoint
from vectorforge.engine.metrics import Metric

__all__ = ["VectorForge", "Collection", "Point", "ScoredPoint", "Metric"]
__version__ = "0.1.0"
