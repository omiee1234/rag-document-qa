"""Load a persisted index and run top-k retrieval against it."""

import pickle

from .embeddings import Embedder
from .store import InMemoryVectorStore, StoreEntry


def load_index(index_path: str):
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    return data["embedder"], data["store"]


def search_index(
    embedder: Embedder,
    store: InMemoryVectorStore,
    query: str,
    top_k: int = 3,
    threshold: float = 0.0,
) -> list[tuple[StoreEntry, float]]:
    """Retrieval against an already-loaded (in-memory) embedder/store pair."""
    query_vector = embedder.embed([query])[0]
    return store.search(query_vector, top_k=top_k, threshold=threshold)


def retrieve(
    query: str, index_path: str, top_k: int = 3, threshold: float = 0.0
) -> list[tuple[StoreEntry, float]]:
    embedder, store = load_index(index_path)
    return search_index(embedder, store, query, top_k=top_k, threshold=threshold)
