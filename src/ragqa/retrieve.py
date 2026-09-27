"""Load a persisted index and run top-k retrieval against it."""

import pickle

from .store import InMemoryVectorStore, StoreEntry


def load_index(index_path: str):
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    return data["embedder"], data["store"]


def retrieve(
    query: str, index_path: str, top_k: int = 3, threshold: float = 0.0
) -> list[tuple[StoreEntry, float]]:
    embedder, store = load_index(index_path)
    query_vector = embedder.embed([query])[0]
    return store.search(query_vector, top_k=top_k, threshold=threshold)
