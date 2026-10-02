"""The database: a data directory holding named collections."""
from __future__ import annotations

import re
import shutil
import threading
from pathlib import Path

from vectorforge.engine.collection import META_FILE, Collection
from vectorforge.engine.metrics import Metric

_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class VectorForge:
    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir) / "collections"
        self.root.mkdir(parents=True, exist_ok=True)
        self._open: dict[str, Collection] = {}
        self._lock = threading.Lock()

    def _path(self, name: str) -> Path:
        if not _NAME.match(name):
            raise ValueError("collection names use letters, digits, _ and - (max 64)")
        return self.root / name

    def create_collection(self, name: str, dim: int, metric: Metric | str = Metric.COSINE,
                          index: str = "hnsw", params: dict | None = None,
                          storage: dict | None = None) -> Collection:
        """index: "hnsw" (default), "ivfpq" or "flat"; params: that index's settings;
        storage: {"fsync": True, "wal_max_bytes": 64 MiB}."""
        path = self._path(name)
        with self._lock:
            if path.exists():
                raise FileExistsError(f"collection {name!r} already exists")
            col = Collection.create(path, name, dim, metric, index=index, params=params,
                                    storage=storage)
            self._open[name] = col
            return col

    def get_collection(self, name: str) -> Collection:
        path = self._path(name)
        with self._lock:
            if name not in self._open:
                if not (path / META_FILE).exists():
                    raise KeyError(f"collection {name!r} does not exist")
                self._open[name] = Collection(path)
            return self._open[name]

    def list_collections(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if (p / META_FILE).exists())

    def drop_collection(self, name: str) -> bool:
        path = self._path(name)
        with self._lock:
            col = self._open.pop(name, None)
            if col is not None:
                col.close(checkpoint=False)
            if not path.exists():
                return False
            shutil.rmtree(path)
            return True

    def close(self) -> None:
        with self._lock:
            for col in self._open.values():
                col.close()
            self._open.clear()

    def __enter__(self) -> "VectorForge":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
