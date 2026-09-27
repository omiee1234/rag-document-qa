"""Provider-agnostic embedding layer.

Swapping embedding providers should never touch chunking, storage, retrieval,
generation, or evaluation code -- only `get_embedder(name)` changes.
"""

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        """Learn anything provider-specific from the corpus (vocabulary, etc.)."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dim) float array."""


class TfidfEmbedder(Embedder):
    """Offline baseline: no API key, no network call, deterministic."""

    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called before embed()")
        return self.vectorizer.transform(texts).toarray()


class SentenceTransformerEmbedder(Embedder):
    """Real dense embeddings. No fitting required -- the model is pretrained."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> None:
        pass

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), convert_to_numpy=True))


class OpenAIEmbedder(Embedder):
    """Hosted embeddings via the OpenAI API. Requires OPENAI_API_KEY."""

    name = "openai"

    def __init__(self, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model_name = model_name

    def fit(self, texts: list[str]) -> None:
        pass

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model_name, input=list(texts))
        return np.asarray([d.embedding for d in resp.data])


_EMBEDDERS = {
    "tfidf": TfidfEmbedder,
    "sentence-transformers": SentenceTransformerEmbedder,
    "st": SentenceTransformerEmbedder,
    "openai": OpenAIEmbedder,
}


def get_embedder(name: str) -> Embedder:
    key = name.lower()
    if key not in _EMBEDDERS:
        raise ValueError(f"Unknown embedder '{name}'. Options: {sorted(set(_EMBEDDERS))}")
    return _EMBEDDERS[key]()
