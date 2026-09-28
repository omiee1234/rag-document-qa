"""Load a persisted index and run top-k retrieval against it."""

import os
import pickle

from .embeddings import Embedder
from .store import InMemoryVectorStore, StoreEntry

_index_cache: dict[str, tuple[float, Embedder, InMemoryVectorStore]] = {}


def load_index(index_path: str):
    """Cached by path + modification time, so repeated queries (the eval
    harness runs 50) don't re-read the pickle, but a rebuilt index is picked
    up."""
    mtime = os.path.getmtime(index_path)
    cached = _index_cache.get(index_path)
    if cached is None or cached[0] != mtime:
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        cached = (mtime, data["embedder"], data["store"])
        _index_cache[index_path] = cached
    return cached[1], cached[2]


def search_index(
    embedder: Embedder,
    store: InMemoryVectorStore,
    query: str,
    top_k: int = 3,
    threshold: float = 0.0,
    rerank: bool = True,
) -> list[tuple[StoreEntry, float]]:
    """Retrieval against an already-loaded (in-memory) embedder/store pair.

    Embedders that implement their own search (hybrid: BM25 + dense fusion +
    re-ranking) are delegated to; the rest use plain cosine similarity.
    `rerank=False` skips the re-ranking stage where an embedder has one.
    """
    if hasattr(embedder, "search"):
        results = embedder.search(store, query, top_k=top_k, rerank=rerank)
        return [(e, s) for e, s in results if s >= threshold]
    query_vector = embedder.embed([query])[0]
    return store.search(query_vector, top_k=top_k, threshold=threshold)


def retrieve(
    query: str, index_path: str, top_k: int = 3, threshold: float = 0.0
) -> list[tuple[StoreEntry, float]]:
    embedder, store = load_index(index_path)
    return search_index(embedder, store, query, top_k=top_k, threshold=threshold)
