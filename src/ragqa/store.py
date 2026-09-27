"""In-memory vector store: numpy cosine similarity.

Swap-in path for scale: replace this module's `add`/`search` with calls to
pgvector (Postgres `<=>` operator) or Qdrant's client -- the rest of the
pipeline only depends on this class's interface, not its internals.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class StoreEntry:
    chunk_id: str
    doc_id: str
    text: str
    vector: np.ndarray


class InMemoryVectorStore:
    def __init__(self):
        self.entries: list[StoreEntry] = []

    def add(self, chunk_id: str, doc_id: str, text: str, vector: np.ndarray) -> None:
        self.entries.append(StoreEntry(chunk_id, doc_id, text, vector))

    def __len__(self) -> int:
        return len(self.entries)

    def _matrix(self) -> np.ndarray:
        return np.vstack([e.vector for e in self.entries])

    def search(
        self, query_vector: np.ndarray, top_k: int = 3, threshold: float = 0.0
    ) -> list[tuple[StoreEntry, float]]:
        """Return up to top_k (entry, cosine_similarity) pairs, best first, above threshold."""
        if not self.entries:
            return []

        mat = self._matrix()
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10)
        sims = mat_norm @ query_norm

        ranked = np.argsort(-sims)
        results = []
        for idx in ranked[:top_k]:
            score = float(sims[idx])
            if score < threshold:
                continue
            results.append((self.entries[idx], score))
        return results

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.entries, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self.entries = pickle.load(f)
