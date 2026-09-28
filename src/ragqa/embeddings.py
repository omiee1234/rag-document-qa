"""Provider-agnostic embedding layer.

Swapping embedding providers should never touch chunking, storage, retrieval,
generation, or evaluation code -- only `get_embedder(name)` changes.
"""

import re
from abc import ABC, abstractmethod

import numpy as np

_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def _normalize_token(token: str) -> str:
    """Crude plural folding so "persons"/"person" and "policies"/"policy"
    match. Not linguistically exact -- it only has to map both sides of a
    query/document pair to the same form, since the same function runs on
    both. Found necessary in testing: "who are the insured persons" missed a
    table whose rows say "Insured Person's Name"."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tfidf_analyzer(text: str) -> list[str]:
    # module-level (not a lambda) so a fitted vectorizer stays picklable
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return [
        _normalize_token(t)
        for t in _TOKEN_RE.findall(text.lower())
        if t not in ENGLISH_STOP_WORDS
    ]


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

        self.vectorizer = TfidfVectorizer(analyzer=tfidf_analyzer)
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
    if key == "hybrid":
        from .hybrid import HybridEmbedder  # imported lazily: hybrid imports this module

        return HybridEmbedder()
    if key not in _EMBEDDERS:
        options = sorted(set(_EMBEDDERS) | {"hybrid"})
        raise ValueError(f"Unknown embedder '{name}'. Options: {options}")
    return _EMBEDDERS[key]()
