import numpy as np
import pytest

pytest.importorskip("langchain_core")
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

from vectorforge.embeddings import HashEmbedder  # noqa: E402
from vectorforge.integrations.langchain import VectorForgeStore  # noqa: E402


class HashEmbeddings(Embeddings):
    def __init__(self):
        self.e = HashEmbedder(dim=128)

    def embed_documents(self, texts):
        return self.e.embed_documents(texts).tolist()

    def embed_query(self, text):
        return self.e.embed_query(text).tolist()


TEXTS = ["Bidri ware is made in Bidar with silver inlay on a dark alloy",
         "Hyderabadi biryani layers rice and marinated meat",
         "Product quantization compresses vectors into a few bytes",
         "HNSW search walks down a layered proximity graph"]
METAS = [{"topic": "craft", "lang": "en"}, {"topic": "food", "lang": "en"},
         {"topic": "ann", "lang": "en"}, {"topic": "ann", "lang": "en"}]


def test_from_texts_search_filter_delete(tmp_path):
    store = VectorForgeStore.from_texts(TEXTS, HashEmbeddings(), metadatas=METAS,
                                        ids=["a", "b", "c", "d"], data_dir=str(tmp_path))
    top = store.similarity_search("silver inlay bidri", k=1)[0]
    assert top.page_content == TEXTS[0] and top.metadata == METAS[0] and top.id == "a"

    docs = store.similarity_search("graph vectors", k=4, filter={"topic": "ann"})
    assert {d.id for d in docs} == {"c", "d"}

    scored = store.similarity_search_with_relevance_scores("biryani rice", k=1)
    assert scored[0][0].id == "b" and 0 <= scored[0][1] <= 1

    assert [d.id for d in store.get_by_ids(["b", "zzz"])] == ["b"]
    store.delete(["a"])
    assert "a" not in {d.id for d in store.similarity_search("bidri", k=4)}

    ids = store.add_documents([Document(page_content="Charminar is in Hyderabad",
                                        metadata={"topic": "place"}, id="e")])
    assert ids == ["e"]
    retriever = store.as_retriever(search_kwargs={"k": 1, "filter": {"topic": "place"}})
    assert retriever.invoke("anything")[0].id == "e"


def test_reserved_keys_rejected(tmp_path):
    store = VectorForgeStore.from_texts([], HashEmbeddings(), data_dir=str(tmp_path))
    with pytest.raises(ValueError):
        store.add_texts(["x"], [{"page_content": "y"}])


def test_remote_store(tmp_path):
    from fastapi.testclient import TestClient

    from vectorforge.client import Client
    from vectorforge.server.app import create_app
    with TestClient(create_app(tmp_path / "d", tmp_path / "r")) as http:
        col = Client(http=http).create_collection("lc", 128, "cosine")
        store = VectorForgeStore(col, HashEmbeddings())
        store.add_texts(TEXTS, METAS, ids=["a", "b", "c", "d"])
        assert store.similarity_search("biryani", k=1)[0].id == "b"
        assert store.similarity_search_with_relevance_scores("biryani", k=1)[0][1] > 0.5
