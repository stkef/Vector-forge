"""Command line: run the server, or load text files for the demo.

    python -m vectorforge serve [--data ./data] [--port 8000] [--embedder gemini|hash|none]
    python -m vectorforge ingest docs/ --collection vf-docs [--embedder gemini]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TEXT_SUFFIXES = {".md", ".txt", ".rst"}


def chunk_text(text: str, max_chars: int = 900) -> list[str]:
    """Split on blank lines, then pack paragraphs into chunks of <= max_chars."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > max_chars:                    # very long paragraph: hard split
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:max_chars])
            p = p[max_chars:]
        if cur and len(cur) + len(p) + 2 > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def cmd_serve(a) -> None:
    import uvicorn

    from vectorforge.embeddings import make_embedder
    from vectorforge.server.app import create_app

    app = create_app(a.data, a.results, make_embedder(a.embedder))
    print(f"VectorForge Studio: http://{a.host}:{a.port}/   API docs: http://{a.host}:{a.port}/docs")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


def cmd_ingest(a) -> None:
    from vectorforge.db import VectorForge
    from vectorforge.embeddings import make_embedder, text_id

    emb = make_embedder(a.embedder)
    root = Path(a.path)
    files = [root] if root.is_file() else sorted(
        f for f in root.rglob("*") if f.suffix.lower() in TEXT_SUFFIXES)
    texts, payloads = [], []
    for f in files:
        for i, ch in enumerate(chunk_text(f.read_text(encoding="utf-8", errors="replace"), a.chunk)):
            texts.append(ch)
            payloads.append({"source": f.name, "chunk": i, "text": ch})
    if not texts:
        sys.exit(f"no {', '.join(sorted(TEXT_SUFFIXES))} files under {root}")
    print(f"{len(files)} files -> {len(texts)} chunks; embedding with {emb.name} ({emb.dim}-d)")
    vecs = emb.embed_documents(texts)
    with VectorForge(a.data) as db:
        if a.collection in db.list_collections():
            col = db.get_collection(a.collection)
        else:
            col = db.create_collection(a.collection, emb.dim, "cosine")
        col.upsert([text_id(t) for t in texts], vecs, payloads)
        print(f"collection {a.collection!r} now has {len(col)} points")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vectorforge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the HTTP API and Studio")
    s.add_argument("--data", default="./data")
    s.add_argument("--results", default="./results", help="benchmark results to show in Studio")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--embedder", default="none", help="gemini | hash | none (enables text search)")
    s.set_defaults(fn=cmd_serve)
    g = sub.add_parser("ingest", help="chunk, embed and load .md/.txt files into a collection")
    g.add_argument("path")
    g.add_argument("--collection", default="docs")
    g.add_argument("--data", default="./data")
    g.add_argument("--embedder", default="gemini")
    g.add_argument("--chunk", type=int, default=900)
    g.set_defaults(fn=cmd_ingest)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
