# Semantic search demo and LangChain

## Ask questions about a folder of documents

```bash
set GEMINI_API_KEY=...                               # PowerShell: $env:GEMINI_API_KEY="..."
python -m vectorforge ingest docs --collection vf-docs
python -m vectorforge serve --embedder gemini
```

Open http://127.0.0.1:8000, pick `vf-docs`, choose **Text**, and ask
"how does the planner pick a strategy?". `ingest` splits `.md`, `.txt` and
`.rst` files into ~900-character chunks on paragraph boundaries, embeds
them with `gemini-embedding-001` (768-d), and stores the text, file name
and chunk number in the payload. Re-running it is safe: a chunk's ID is a
hash of its text, so unchanged chunks are overwritten, not duplicated.

Any folder works, including romanised Telugu notes: Gemini embeddings are
multilingual, and filters work on the payload (`{"source": "notes.md"}`).

No API key? `--embedder hash` uses an offline hashing embedder. It matches
shared words and spellings, not meaning, so it is fine for trying the
pipeline and for tests, not for judging search quality.

## LangChain

```python
from vectorforge.integrations.langchain import VectorForgeStore

store = VectorForgeStore.from_texts(texts, embeddings, metadatas=metas,
                                    data_dir="./data", collection="kb")
# or against a running server:
store = VectorForgeStore.from_texts(texts, embeddings, url="http://127.0.0.1:8000")

store.similarity_search("bidri inlay process", k=4, filter={"lang": "te"})
retriever = store.as_retriever(search_kwargs={"k": 4})
```

Metadata fields are stored at the top level of the payload, so VectorForge
filters apply to them directly. `page_content` and `_lc_id` are reserved.
`similarity_search_with_relevance_scores` maps cosine similarity to [0, 1]
and L2 distance to 1 / (1 + d).
