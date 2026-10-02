# Durability: write-ahead log, checkpoints, recovery

Code: `vectorforge/engine/wal.py`, `collection.py`.
Tests: `tests/test_durability.py`.

## The rule

A write is acknowledged when its record is in the write-ahead log and
`fsync` has returned. Everything else (vector file, SQLite, HNSW / IVF-PQ)
is derived state that can be rebuilt from the last checkpoint plus the log.

## Write path

1. Validate (bad input never reaches the log).
2. Assign slots and append one record: IDs, slots, vectors, payloads.
3. `fsync` the log. **Acknowledged.**
4. Apply to the vector file and SQLite without forcing them to disk,
   then to the in-memory filter index and the ANN index.

## Checkpoint (the snapshot)

1. fsync the vector file (memmap flush is not enough on Windows).
2. SQLite `wal_checkpoint(FULL)`.
3. Save the index atomically (`tmp` + fsync + rename).
4. Write `checkpoint.json` with the last LSN, atomically.
5. Empty the log.

Runs when the log passes `wal_max_bytes` (64 MiB), on close, and after
recovery. A crash between any two steps is safe: the log is only emptied
after the LSN that makes its records unnecessary is on disk.

## Recovery

Read the log, stop at the first record whose length or CRC is wrong and cut
the file there (a torn write that was never acknowledged), then replay every
record newer than the checkpoint LSN.

Replay is safe even when the state already contains some of those records,
because every record is a blind write ("ID 5 is now slot 812 with this
payload", "ID 5 is gone"), never read-modify-write. Replaying the whole
suffix in order always lands on the same final state.

The ANN index needs no log of its own. Vectors are append-only, so a saved
index can be behind but never wrong: on open it tombstones what was deleted
since and inserts what was added since.

## What the tests prove

| Test | Failure modelled |
|---|---|
| `test_kill_9_loses_no_acknowledged_write` | writer process SIGKILLed mid-ingest, after several checkpoints; every batch it printed as written is there, byte for byte |
| `test_replay_after_losing_unsynced_state` | power loss: vector file + SQLite roll back to the checkpoint, the fsynced log survives (flat, HNSW, IVF-PQ) |
| `test_replay_is_idempotent` | crash after applying, before checkpointing |
| `test_wal_roundtrip_and_torn_tail` | half-written last record |
| `test_wal_corrupt_byte_stops_replay` | bit flip inside a record |

Not covered: a disk that lies about fsync, and corruption of the vector file
or SQLite *before* the last checkpoint (that needs checksummed pages).

## Cost

Measured in the Linux test container (fast fsync; expect ~1-5 ms per fsync
on a Windows SSD):

| batch size | fsync on | fsync off |
|---:|---:|---:|
| 1 | 4,082 points/s | 10,294 points/s |
| 100 | 65,700 points/s | 105,645 points/s |
| 1,000 | 100,333 points/s | 103,744 points/s |

Batching amortises the fsync: at 1,000 points per call it costs ~3%.
`storage={"fsync": False}` is for bulk loads you can redo; it keeps
consistency but not durability.

Recovery replayed 50 records (50k points) in 0.44 s.
