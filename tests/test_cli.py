from vectorforge import VectorForge
from vectorforge.cli import chunk_text, main


def test_chunk_text_packs_paragraphs():
    text = "a" * 300 + "\n\n" + "b" * 300 + "\n\n" + "c" * 500 + "\n\n" + "d" * 2_000
    chunks = chunk_text(text, max_chars=900)
    assert all(len(c) <= 900 for c in chunks)
    assert chunks[0].startswith("a") and "b" * 300 in chunks[0]
    assert "".join(chunks).replace("\n", "").count("d") == 2_000


def test_ingest_with_offline_embedder(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "bidri.md").write_text("Bidri ware: silver inlay on blackened zinc.\n\nMade in Bidar.")
    (docs / "ann.txt").write_text("HNSW is a layered graph for nearest neighbour search.")
    (docs / "skip.bin").write_bytes(b"\x00\x01")
    main(["ingest", str(docs), "--collection", "d", "--data", str(tmp_path / "data"),
          "--embedder", "hash"])
    main(["ingest", str(docs), "--collection", "d", "--data", str(tmp_path / "data"),
          "--embedder", "hash"])                                     # same text -> same IDs
    with VectorForge(tmp_path / "data") as db:
        col = db.get_collection("d")
        assert len(col) == 2 and col.dim == 256
        src = {p.payload["source"] for p in col.scroll()[0]}
        assert src == {"bidri.md", "ann.txt"}
