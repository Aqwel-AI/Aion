# aion.rag — Package documentation

## 1. Title and overview

**`aion.rag`** is a **reference-tier** RAG toolkit: **text chunking**, pluggable **vector stores** (in-memory and optional **FAISS**), and **`SimpleRAGIndex`** to chunk → embed → store → query. Embeddings default to **`aion.embed`** unless you pass a custom **`embed_fn`**.

This is **not** a full production RAG platform; it is suitable for teaching, prototypes, and small corpora.

---

## 2. Module layout

| Path | Role |
|------|------|
| `chunking.py` | `chunk_text`, `chunk_by_paragraphs`. |
| `types.py` | `VectorStore` protocol, `ScoredChunk`. |
| `stores/memory.py` | `MemoryVectorStore` — NumPy/dot-product style storage. |
| `stores/faiss_store.py` | `FaissVectorStore` — requires FAISS (`pip install faiss-cpu` or `[rag]`). |
| `pipeline.py` | `SimpleRAGIndex` — end-to-end index + search. |

---

## 3. Public API (from `aion.rag`)

| Symbol | Description |
|--------|-------------|
| `chunk_text`, `chunk_by_paragraphs` | Split documents for embedding. |
| `MemoryVectorStore`, `FaissVectorStore` | Vector persistence / search backends. |
| `ScoredChunk`, `VectorStore` | Result type and store interface. |
| `SimpleRAGIndex` | High-level build + top-k query. |

```python
from aion.rag import SimpleRAGIndex, chunk_text

# index = SimpleRAGIndex(embed_fn=..., ...)
```

---

## 4. Examples

Runnable scripts: **[examples/](examples/)** — see [examples/README.md](examples/README.md).

```bash
python -m aion.rag.examples.demo_simple_index
```

---

## 5. Conventions

- **Embeddings:** Dimension must be consistent across all vectors in one store.
- **FAISS:** Optional; import errors should be handled by installing `[rag]` or `faiss-cpu`.

---

## 6. Dependencies

- **NumPy** (core).
- **Optional:** `sentence-transformers`, **FAISS** — via `pip install aqwel-aion[rag]` or extras documented in `pyproject.toml`.

---

## 7. See also

- Embeddings: **`aion.embed`**
- Chunking-only workflows: **`aion.datasets`** for file I/O
- Root README: [../../README.md](../../README.md)
