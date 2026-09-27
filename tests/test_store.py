import numpy as np

from ragqa.store import InMemoryVectorStore


def test_search_ranks_by_cosine_similarity():
    store = InMemoryVectorStore()
    store.add("a", "doc", "vector a", np.array([1.0, 0.0]))
    store.add("b", "doc", "vector b", np.array([0.0, 1.0]))
    store.add("c", "doc", "vector c", np.array([0.9, 0.1]))

    results = store.search(np.array([1.0, 0.0]), top_k=2)

    assert [entry.chunk_id for entry, _ in results] == ["a", "c"]


def test_threshold_filters_weak_matches():
    store = InMemoryVectorStore()
    store.add("a", "doc", "vector a", np.array([1.0, 0.0]))
    store.add("b", "doc", "vector b", np.array([0.0, 1.0]))

    results = store.search(np.array([1.0, 0.0]), top_k=2, threshold=0.5)

    assert [entry.chunk_id for entry, _ in results] == ["a"]


def test_empty_store_returns_no_results():
    store = InMemoryVectorStore()
    assert store.search(np.array([1.0, 0.0])) == []
