"""Point IDs, their slot in the vector file, and JSON payloads, in SQLite."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

_CHUNK = 900  # stay well under SQLite's bound-parameter limit


def _chunks(items: list, size: int = _CHUNK) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


class PayloadStore:
    def __init__(self, path: Path):
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        # The write-ahead log provides durability; SQLite only needs to stay
        # consistent, and is forced to disk at checkpoints.
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS points (
                id      INTEGER PRIMARY KEY,
                slot    INTEGER NOT NULL UNIQUE,
                payload TEXT NOT NULL
            );
        """)

    @contextmanager
    def _tx(self):
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def slots_for(self, ids: list[int]) -> dict[int, int]:
        out: dict[int, int] = {}
        for chunk in _chunks(ids):
            q = f"SELECT id, slot FROM points WHERE id IN ({','.join('?' * len(chunk))})"
            out.update(self._db.execute(q, chunk).fetchall())
        return out

    def all_slots(self) -> list[tuple[int, int]]:
        return self._db.execute("SELECT id, slot FROM points").fetchall()

    def iter_payloads(self):
        """(slot, payload dict) for every live point; used to build the filter index."""
        for slot, payload in self._db.execute("SELECT slot, payload FROM points"):
            yield slot, json.loads(payload)

    def scroll(self, after: int | None, limit: int) -> list[tuple[int, int, dict]]:
        """(id, slot, payload) in ID order, starting after `after` (keyset paging)."""
        rows = self._db.execute(
            "SELECT id, slot, payload FROM points WHERE id > ? ORDER BY id LIMIT ?",
            (-1 if after is None else after, limit)).fetchall()
        return [(pid, slot, json.loads(p)) for pid, slot, p in rows]

    def checkpoint(self) -> None:
        """Move SQLite's own WAL into the main file and sync it."""
        self._db.execute("PRAGMA wal_checkpoint(FULL)")

    def payloads(self, ids: list[int]) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for chunk in _chunks(ids):
            q = f"SELECT id, payload FROM points WHERE id IN ({','.join('?' * len(chunk))})"
            for pid, payload in self._db.execute(q, chunk):
                out[pid] = json.loads(payload)
        return out

    def upsert(self, rows: Iterable[tuple[int, int, str]], next_slot: int) -> None:
        with self._tx():
            self._db.executemany(
                "INSERT INTO points (id, slot, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET slot = excluded.slot, payload = excluded.payload",
                rows,
            )
            self._db.execute(
                "INSERT INTO meta (key, value) VALUES ('next_slot', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(next_slot),),
            )

    def delete(self, ids: list[int]) -> list[tuple[int, int]]:
        """Delete points; return the (id, slot) pairs that actually existed."""
        removed: list[tuple[int, int]] = []
        with self._tx():
            for chunk in _chunks(ids):
                marks = ",".join("?" * len(chunk))
                removed += self._db.execute(
                    f"SELECT id, slot FROM points WHERE id IN ({marks})", chunk).fetchall()
                self._db.execute(f"DELETE FROM points WHERE id IN ({marks})", chunk)
        return removed

    def close(self) -> None:
        self._db.close()
