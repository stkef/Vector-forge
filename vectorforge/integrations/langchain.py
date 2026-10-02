"""LangChain vector store backed by VectorForge (embedded or over HTTP).

    from langchain_core.embeddings import ...   # any LangChain Embeddings
    from vectorforge.integrations.langchain import VectorForgeStore

    store = VectorForgeStore.from_texts(texts, embeddings, metadatas=metas,
                                        data_dir="./data", collection="docs")
    store.similarity_search("bidri inlay kaise karte hain", k=4, filter={"lang": "te"})
    retriever = store.as_retriever(search_kwargs={"k": 4})

Point layout: LangChain IDs are strings and VectorForge IDs are integers, so
each document gets a stable 53-bit integer derived from its LangChain ID.
The payload holds the metadata fields at the top level (so VectorForge
filters work on them directly) plus two reserved keys: `_lc_id` and
`page_content`.
"""
from __future__ import annotations

import hashlib
import uuid
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install langchain-core to use the LangChain adapter") from e

RESERVED = ("_lc_id", "page_content")


def lc_int_id(lc_id: str) -> int:
    """53-bit, so IDs stay exact in JavaScript clients too."""
    return int.from_bytes(hashlib.blake2b(lc_id.encode(), digest_size=8).digest(), "little") >> 11


def _to_doc(payload: dict) -> Document:
    meta = {k: v for k, v in payload.items() if k not in RESERVED}
    return Document(page_content=payload.get("page_content", ""), metadata=meta,
                    id=payload.get("_lc_id"))


class VectorForgeStore(VectorStore):
    def __init__(self, collection, embedding: Embeddings):
        """`collection`: a vectorforge Collection or a client RemoteCollection."""
        self.collection = collection
        self._embedding = embedding

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    # ------------------------------------------------------------ write
    def add_texts(self, texts: Iterable[str], metadatas: list[dict] | None = None, *,
                  ids: list[str] | None = None, **kwargs: Any) -> list[str]:
        texts = list(texts)
        if not texts:
            return []
        metadatas = metadatas or [{} for _ in texts]
        ids = list(ids) if ids else [str(uuid.uuid4()) for _ in texts]
        for m in metadatas:
            clash = set(m) & set(RESERVED)
            if clash:
                raise ValueError(f"metadata keys {sorted(clash)} are reserved")
        vecs = np.asarray(self._embedding.embed_documents(texts), np.float32)
        payloads = [{**m, "_lc_id": i, "page_content": t} for m, i, t in zip(metadatas, ids, texts)]
        self.collection.upsert([lc_int_id(i) for i in ids], vecs, payloads)
        return ids

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        ids = kwargs.pop("ids", None) or [d.id or str(uuid.uuid4()) for d in documents]
        return self.add_texts([d.page_content for d in documents],
                              [dict(d.metadata) for d in documents], ids=ids)

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        if not ids:
            return None
        self.collection.delete([lc_int_id(i) for i in ids])
        return True

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        pts = self.collection.get([lc_int_id(i) for i in ids])
        return [_to_doc(p.payload) for p in pts if p is not None]

    # ------------------------------------------------------------- read
    def similarity_search_by_vector_with_score(self, embedding: list[float], k: int = 4,
                                               filter: dict | None = None,
                                               **kwargs: Any) -> list[tuple[Document, float]]:
        hits = self.collection.search(np.asarray(embedding, np.float32), k, filter=filter, **kwargs)
        return [(_to_doc(h.payload), h.score) for h in hits]

    def similarity_search_with_score(self, query: str, k: int = 4, filter: dict | None = None,
                                     **kwargs: Any) -> list[tuple[Document, float]]:
        return self.similarity_search_by_vector_with_score(
            self._embedding.embed_query(query), k, filter=filter, **kwargs)

    def similarity_search_by_vector(self, embedding: list[float], k: int = 4,
                                    filter: dict | None = None, **kwargs: Any) -> list[Document]:
        return [d for d, _ in self.similarity_search_by_vector_with_score(embedding, k, filter, **kwargs)]

    def similarity_search(self, query: str, k: int = 4, filter: dict | None = None,
                          **kwargs: Any) -> list[Document]:
        return [d for d, _ in self.similarity_search_with_score(query, k, filter, **kwargs)]

    def _select_relevance_score_fn(self):
        metric = getattr(self.collection, "metric", None)
        metric = getattr(metric, "value", metric) or self.collection.info()["metric"]
        if metric == "l2":
            return lambda d: 1.0 / (1.0 + d)
        return lambda s: (s + 1.0) / 2.0          # cosine in [-1, 1] -> [0, 1]

    # ---------------------------------------------------------- factory
    @classmethod
    def from_texts(cls, texts: list[str], embedding: Embeddings, metadatas: list[dict] | None = None,
                   *, ids: list[str] | None = None, collection: str = "langchain",
                   data_dir: str | None = None, url: str | None = None, metric: str = "cosine",
                   index: str = "hnsw", **kwargs: Any) -> "VectorForgeStore":
        """Create (or reuse) a collection, embedded (data_dir) or remote (url)."""
        dim = len(embedding.embed_query("dimension probe"))
        if url:
            from vectorforge.client import Client
            client = Client(url)
            col = (client.get_collection(collection) if collection in client.list_collections()
                   else client.create_collection(collection, dim, metric, index=index))
        else:
            from vectorforge.db import VectorForge
            db = VectorForge(data_dir or "./data")
            col = (db.get_collection(collection) if collection in db.list_collections()
                   else db.create_collection(collection, dim, metric, index=index))
        store = cls(col, embedding)
        store.add_texts(texts, metadatas, ids=ids)
        return store
