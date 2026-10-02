# Architecture

## Components

```mermaid
flowchart TB
    subgraph surface[Ways in]
        CLI[cli.py<br/>serve, ingest]
        APP[server/app.py<br/>FastAPI + Studio]
        CL[client.py]
        LCA[integrations/langchain.py]
    end
    DB[db.py<br/>VectorForge: data directory]
    COL[engine/collection.py<br/>write path, read path, recovery]
    subgraph storage[Storage]
        WAL[engine/wal.py]
        VS[engine/vector_store.py<br/>mmap float32]
        PS[engine/payload_store.py<br/>SQLite]
    end
    subgraph query[Query]
        FI[engine/filters.py<br/>payload index]
        PL[engine/planner.py]
        subgraph idx[engine/index]
            FL[flat.py]
            HN[hnsw.py + _hnsw_kernels.py]
            IV[ivfpq.py + kmeans.py]
        end
    end
    CLI --> APP
    CLI --> DB
    CL -->|HTTP| APP
    LCA --> DB
    LCA --> CL
    APP --> DB
    DB --> COL
    COL --> WAL & VS & PS & FI & PL
    COL --> FL & HN & IV
```

Every index implements one interface (`engine/index/base.py`): `add`,
`remove`, `search`, `rebuild`, `save`, `load`, `in_index`, `stats`. The
collection and the planner never depend on which index is behind it.

## Write path

```mermaid
sequenceDiagram
    participant C as Caller
    participant Col as Collection
    participant W as WAL
    participant V as vectors.f32
    participant S as SQLite
    participant I as Payload + ANN index
    C->>Col: upsert(ids, vectors, payloads)
    Col->>Col: validate, dedupe, normalise (cosine)
    Col->>Col: assign new slots at the end
    Col->>W: append record + fsync
    Note over Col,W: acknowledged from here
    Col->>V: write rows (page cache)
    Col->>S: upsert rows, next_slot
    Col->>I: tombstone old slots, add new
    Col-->>C: count
    opt WAL > wal_max_bytes
        Col->>V: fsync
        Col->>S: wal_checkpoint(FULL)
        Col->>I: save index
        Col->>W: write checkpoint.json, empty log
    end
```

## Search path

```mermaid
sequenceDiagram
    participant C as Caller
    participant Col as Collection
    participant F as Payload index
    participant P as Planner
    participant X as Index or exact scan
    participant S as SQLite
    C->>Col: search(q, k, filter)
    Col->>Col: prepare query (normalise for cosine)
    Col->>F: filter -> slot mask
    Col->>Col: mask AND live slots
    Col->>P: candidates, selectivity, index state
    P-->>Col: plan (subset_scan | hnsw | ivfpq | ...)
    Col->>X: search(q, k, mask, params)
    X-->>Col: slots, similarities
    Col->>S: payloads for hit IDs
    Col-->>C: ScoredPoint list
```

## Open and recovery

```mermaid
flowchart LR
    A[open collection] --> B[read wal.log<br/>truncate torn tail]
    B --> C[replay records<br/>LSN > checkpoint]
    C --> D[build slot map and<br/>payload index from SQLite]
    D --> E{saved index<br/>matches settings?}
    E -->|yes| F[load, tombstone dead slots,<br/>insert new slots]
    E -->|no| G[rebuild / train]
    F --> H{replayed anything?}
    G --> H
    H -->|yes| I[checkpoint]
    H -->|no| J[ready]
    I --> J
```

## Data model

```mermaid
erDiagram
    COLLECTION ||--o{ POINT : holds
    COLLECTION ||--|| SETTINGS : "collection.json"
    POINT ||--|| SLOT : "lives at"
    SLOT ||--o| INDEX_ENTRY : "indexed as"
    COLLECTION ||--o{ WAL_RECORD : "logs"
    POINT {
        int id PK
        json payload
    }
    SLOT {
        int slot PK
        float32_array vector
    }
    INDEX_ENTRY {
        int slot
        bool tombstoned
    }
    WAL_RECORD {
        int lsn PK
        int op
        bytes body
    }
```

## Concurrency

One re-entrant lock per collection serialises writes, searches and
checkpoints. The HTTP server runs request handlers in a thread pool, so
different collections proceed in parallel; requests to one collection are
serialised. Numba kernels release no locks of their own. This is the
simplest correct model; readers-writer locking is listed in
[FDD section 13](FDD.md#13-limitations-and-open-items).
