"""Write-ahead log, checkpoints and crash recovery.

The headline test kills a writer process with SIGKILL mid-ingest and checks
that every batch it was told had been written is there after restart.
"""
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from vectorforge import VectorForge
from vectorforge.engine import wal as W


def batch(b, dim=16):
    rng = np.random.default_rng(1_000 + b)
    return list(range(b * 50, b * 50 + 50)), rng.standard_normal((50, dim)).astype(np.float32)


def check_batches(col, n_batches):
    for b in range(n_batches):
        bid, vecs = batch(b)
        got = col.get(bid, with_vectors=True)
        assert all(p is not None for p in got), f"batch {b} lost"
        np.testing.assert_array_equal(np.array([p.vector for p in got], np.float32), vecs)
        assert got[0].payload == {"batch": b}


# ------------------------------------------------------------- WAL unit tests
def test_wal_roundtrip_and_torn_tail(tmp_path):
    log = W.WriteAheadLog(tmp_path / "w.log", dim=3)
    log.append(W.OP_UPSERT, W.encode_upsert(np.array([5, 6]), np.array([0, 1]),
                                            np.ones((2, 3), np.float32), ['{"a": 1}', "{}"]))
    log.append(W.OP_DELETE, W.encode_delete(np.array([5])))
    log.close()
    good = (tmp_path / "w.log").stat().st_size
    with open(tmp_path / "w.log", "ab") as f:                 # a half-written record
        f.write(b"\x40\x00\x00\x00garbage")

    log = W.WriteAheadLog(tmp_path / "w.log", dim=3)
    up, de = log.records
    assert up.ids.tolist() == [5, 6] and up.payloads == ['{"a": 1}', "{}"]
    assert de.ids.tolist() == [5] and log.last_lsn == 2
    assert (tmp_path / "w.log").stat().st_size == good        # tail cut off
    assert log.append(W.OP_DELETE, W.encode_delete(np.array([6]))) == 3


def test_wal_corrupt_byte_stops_replay(tmp_path):
    log = W.WriteAheadLog(tmp_path / "w.log", dim=2)
    for i in range(3):
        log.append(W.OP_DELETE, W.encode_delete(np.array([i])))
    log.close()
    raw = bytearray((tmp_path / "w.log").read_bytes())
    raw[-3] ^= 0xFF                                           # flip a bit in the last record
    (tmp_path / "w.log").write_bytes(raw)
    assert [r.lsn for r in W.WriteAheadLog(tmp_path / "w.log", dim=2).records] == [1, 2]


# ------------------------------------------------------- collection recovery
def test_checkpoint_empties_wal(tmp_path):
    with VectorForge(tmp_path) as db:
        col = db.create_collection("c", 16, "l2", index="flat")
        bid, v = batch(0)
        col.upsert(bid, v, [{"batch": 0}] * 50)
        assert col.info()["wal_bytes"] > 0
        col.checkpoint()
        info = col.info()
        assert info["wal_bytes"] == 0 and info["checkpoint_lsn"] == info["last_lsn"] == 1


def test_auto_checkpoint_by_size(tmp_path):
    with VectorForge(tmp_path) as db:
        col = db.create_collection("c", 16, "l2", index="flat",
                                   storage={"wal_max_bytes": 10_000})
        for b in range(5):
            bid, v = batch(b)
            col.upsert(bid, v, [{"batch": b}] * 50)          # ~4 KB per record
        assert col.info()["checkpoint_lsn"] >= 3
        assert col.info()["wal_bytes"] < 10_000


@pytest.mark.parametrize("index", ["flat", "hnsw", "ivfpq"])
def test_replay_after_losing_unsynced_state(tmp_path, index):
    """Power-loss model: the vector file and SQLite fall back to the last
    checkpoint, but the fsynced WAL survives. Everything must come back."""
    params = {"train_after": 100, "nlist": 8} if index == "ivfpq" else None
    db = VectorForge(tmp_path)
    col = db.create_collection("c", 16, "l2", index=index, params=params)
    for b in range(4):
        bid, v = batch(b)
        col.upsert(bid, v, [{"batch": b}] * 50)
    col.checkpoint()
    folder = tmp_path / "collections" / "c"
    saved = tmp_path / "at-checkpoint"
    shutil.copytree(folder, saved)

    for b in range(4, 10):                                    # acknowledged after the checkpoint
        bid, v = batch(b)
        col.upsert(bid, v, [{"batch": b}] * 50)
    col.delete(list(range(0, 50)))                            # batch 0 deleted
    bid, v = batch(1)
    col.upsert(bid, v, [{"batch": 1}] * 50)                   # batch 1 re-written

    wal_bytes = (folder / "wal.log").read_bytes()
    col.close(checkpoint=False)                               # "crash"
    shutil.rmtree(folder)
    shutil.copytree(saved, folder)                            # state rolls back...
    (folder / "wal.log").write_bytes(wal_bytes)               # ...but the WAL survived

    db2 = VectorForge(tmp_path)
    col2 = db2.get_collection("c")
    assert col2.recovered_records == 8
    assert len(col2) == 450 and col2.get(list(range(50))) == [None] * 50
    for b in range(1, 10):
        bid, v = batch(b)
        got = col2.get(bid, with_vectors=True)
        np.testing.assert_array_equal(np.array([p.vector for p in got], np.float32), v)
    hits = col2.search(batch(9)[1][0], 1)
    assert hits[0].id == 450 and hits[0].score < 1e-3
    db2.close()

    with VectorForge(tmp_path) as db3:                        # second open: nothing to replay
        assert db3.get_collection("c").recovered_records == 0


def test_replay_is_idempotent(tmp_path):
    """State already contains the WAL's effects (crash after apply, before
    checkpoint): replaying must not change anything."""
    db = VectorForge(tmp_path)
    col = db.create_collection("c", 16, "l2", index="hnsw")
    for b in range(3):
        bid, v = batch(b)
        col.upsert(bid, v, [{"batch": b}] * 50)
    col.delete([0, 1])
    before = {p.id: p.vector for p in col.get(list(range(150)), with_vectors=True) if p}
    col._vectors.sync()
    col.close(checkpoint=False)
    db2 = VectorForge(tmp_path)
    col2 = db2.get_collection("c")
    assert col2.recovered_records == 4
    after = {p.id: p.vector for p in col2.get(list(range(150)), with_vectors=True) if p}
    assert after == before and len(col2) == 148
    db2.close()


WRITER = textwrap.dedent("""
    import sys
    import numpy as np
    sys.path.insert(0, sys.argv[2])
    from test_durability import batch
    from vectorforge import VectorForge
    db = VectorForge(sys.argv[1])
    col = db.create_collection("c", 16, "l2", index=sys.argv[3],
                               storage={"wal_max_bytes": 40_000})
    b = 0
    while True:
        bid, v = batch(b)
        col.upsert(bid, v, [{"batch": b}] * 50)
        print(b, flush=True)          # acknowledged
        b += 1
""")


@pytest.mark.parametrize("index", ["flat", "hnsw"])
def test_kill_9_loses_no_acknowledged_write(tmp_path, index):
    script = tmp_path / "writer.py"
    script.write_text(WRITER)
    proc = subprocess.Popen([sys.executable, str(script), str(tmp_path / "data"),
                             str(Path(__file__).parent), index],
                            stdout=subprocess.PIPE, text=True)
    acked = -1
    for line in proc.stdout:
        acked = int(line)
        if acked >= 40:                                       # several checkpoints in
            break
    proc.kill()                                               # SIGKILL / TerminateProcess
    proc.wait()
    assert acked >= 40

    with VectorForge(tmp_path / "data") as db:
        col = db.get_collection("c")
        check_batches(col, acked + 1)
        assert len(col) >= (acked + 1) * 50                   # maybe one more, unacknowledged
        if index == "hnsw":
            assert col.info()["index"]["nodes"] == len(col)
