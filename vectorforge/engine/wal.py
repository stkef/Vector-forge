"""Write-ahead log: the source of truth for every write.

A write is acknowledged only after its record is appended here and fsynced.
The vector file, SQLite and the indexes are then updated without forcing
anything to disk; if the process or machine dies, recovery replays the log.

Record layout (little-endian):

    u32 body_len | u32 crc32(lsn, op, body) | u64 lsn | u8 op | body

A crash can leave a half-written record at the end. Recovery reads records
until the first one whose length or checksum doesn't add up, then truncates
the file there: that record was never acknowledged, because the fsync that
would have acknowledged it never returned.

Every record is a *blind write* ("point 5 now has this vector, slot and
payload" / "point 5 no longer exists"), never "add one to point 5". That
makes replay idempotent: replaying records that were already applied
before the crash lands in the same final state.
"""
from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<IIQB")
OP_UPSERT = 1
OP_DELETE = 2


@dataclass
class UpsertRecord:
    lsn: int
    ids: np.ndarray       # int64
    slots: np.ndarray     # int64
    vectors: np.ndarray   # float32 (n, dim), already metric-prepared
    payloads: list[str]   # JSON text


@dataclass
class DeleteRecord:
    lsn: int
    ids: np.ndarray       # int64


def encode_upsert(ids: np.ndarray, slots: np.ndarray, vectors: np.ndarray,
                  payloads: list[str]) -> bytes:
    n = len(ids)
    parts = [struct.pack("<I", n),
             np.asarray(ids, "<i8").tobytes(),
             np.asarray(slots, "<i8").tobytes(),
             np.asarray(vectors, "<f4").tobytes()]
    for p in payloads:
        b = p.encode("utf-8")
        parts.append(struct.pack("<I", len(b)))
        parts.append(b)
    return b"".join(parts)


def encode_delete(ids: np.ndarray) -> bytes:
    return struct.pack("<I", len(ids)) + np.asarray(ids, "<i8").tobytes()


def _decode(lsn: int, op: int, body: memoryview, dim: int):
    (n,) = struct.unpack_from("<I", body, 0)
    pos = 4
    ids = np.frombuffer(body, "<i8", n, pos).astype(np.int64)
    pos += 8 * n
    if op == OP_DELETE:
        return DeleteRecord(lsn, ids)
    slots = np.frombuffer(body, "<i8", n, pos).astype(np.int64)
    pos += 8 * n
    vectors = np.frombuffer(body, "<f4", n * dim, pos).reshape(n, dim).astype(np.float32)
    pos += 4 * n * dim
    payloads = []
    for _ in range(n):
        (ln,) = struct.unpack_from("<I", body, pos)
        pos += 4
        payloads.append(bytes(body[pos:pos + ln]).decode("utf-8"))
        pos += ln
    return UpsertRecord(lsn, ids, slots, vectors, payloads)


class WriteAheadLog:
    def __init__(self, path: Path, dim: int, fsync: bool = True):
        self.path = Path(path)
        self.dim = dim
        self.fsync = fsync
        self.path.touch(exist_ok=True)
        self.last_lsn = 0
        self.records = self._scan()            # read once at open, for recovery
        self._f = open(self.path, "ab")

    def _scan(self) -> list:
        data = memoryview(self.path.read_bytes())
        out, pos = [], 0
        while pos + HEADER.size <= len(data):
            body_len, crc, lsn, op = HEADER.unpack_from(data, pos)
            end = pos + HEADER.size + body_len
            if end > len(data) or op not in (OP_UPSERT, OP_DELETE):
                break
            body = data[pos + HEADER.size:end]
            if zlib.crc32(body, zlib.crc32(struct.pack("<QB", lsn, op))) != crc:
                break
            try:
                out.append(_decode(lsn, op, body, self.dim))
            except (struct.error, ValueError):
                break
            self.last_lsn = lsn
            pos = end
        if pos < len(data):                     # torn or corrupt tail: cut it off
            with open(self.path, "r+b") as f:
                f.truncate(pos)
                f.flush()
                os.fsync(f.fileno())
        return out

    @property
    def size(self) -> int:
        return self._f.tell()

    def append(self, op: int, body: bytes) -> int:
        """Append one record; once this returns (with fsync on), it survives a crash."""
        lsn = self.last_lsn + 1
        crc = zlib.crc32(body, zlib.crc32(struct.pack("<QB", lsn, op)))
        self._f.write(HEADER.pack(len(body), crc, lsn, op) + body)
        self._f.flush()
        if self.fsync:
            os.fsync(self._f.fileno())
        self.last_lsn = lsn
        return lsn

    def reset(self, last_lsn: int) -> None:
        """Empty the log after a checkpoint. LSNs keep counting up."""
        self._f.close()
        with open(self.path, "wb") as f:
            f.flush()
            os.fsync(f.fileno())
        self._f = open(self.path, "ab")
        self.last_lsn = last_lsn
        self.records = []

    def close(self) -> None:
        self._f.close()
