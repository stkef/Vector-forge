"""Text -> vector, for the semantic-search demo and the LangChain adapter.

The database itself never needs this: it stores whatever vectors you give it.

- GeminiEmbedder: real semantic embeddings (gemini-embedding-001), works for
  romanised Telugu and other languages. Needs GEMINI_API_KEY.
- HashEmbedder: offline and deterministic, for tests and CI. It is *lexical*
  (word and character-trigram hashing), not semantic: it matches shared
  words and spellings, not meaning.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class HashEmbedder:
    name = "hash"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        words = re.findall(r"\w+", text.lower())
        feats = words + [w[i:i + 3] for w in words for i in range(max(1, len(w) - 2))]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) else -1.0
        n = np.linalg.norm(v)
        if n == 0:
            v[0] = 1.0
            return v
        return v / n

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._one(text)


class GeminiEmbedder:
    name = "gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"

    def __init__(self, model: str = "gemini-embedding-001", dim: int = 768,
                 api_key: str | None = None, batch: int = 100):
        self.model = model
        self.dim = dim
        self.batch = batch
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("set GEMINI_API_KEY to use Gemini embeddings (or use --embedder hash)")

    def _call(self, texts: list[str], task: str) -> np.ndarray:
        body = {"requests": [{
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": t}]},
            "taskType": task,
            "outputDimensionality": self.dim,
        } for t in texts]}
        req = urllib.request.Request(
            self.URL.format(model=self.model), data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Gemini embeddings failed ({e.code}): {e.read()[:300]!r}") from e
        return np.array([e["values"] for e in data["embeddings"]], np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = [self._call(texts[i:i + self.batch], "RETRIEVAL_DOCUMENT")
               for i in range(0, len(texts), self.batch)]
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._call([text], "RETRIEVAL_QUERY")[0]


def make_embedder(name: str | None) -> Embedder | None:
    if not name or name == "none":
        return None
    if name == "hash":
        return HashEmbedder()
    if name == "gemini":
        return GeminiEmbedder()
    raise ValueError(f"unknown embedder {name!r}: use gemini, hash or none")


def text_id(text: str) -> int:
    """Stable 53-bit ID for a piece of text (same text -> same point).
    53 bits, not 63, so the ID survives JSON in JavaScript (2^53 limit)."""
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little") >> 11
