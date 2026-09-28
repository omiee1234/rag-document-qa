"""Hybrid retrieval: BM25 keywords + dense embeddings, fused, then re-ranked.

- BM25 catches exact terms: policy numbers, clause ids, names, codes.
- Dense embeddings (bge-small) catch paraphrases: "vacation in London" vs
  "annual leave, UK", which keyword search can't connect.
- Reciprocal Rank Fusion merges the two ranked lists: a chunk ranked high in
  *either* one rises to the top, with no score scaling to tune.
- A cross-encoder re-ranks the fused top candidates by reading the question
  and chunk together -- much better at telling "the chunk that answers" from
  "a chunk that mentions the same words" (e.g. a filled-in "Name of the
  Proposer" field vs. blank signature-block labels).

Models run locally via fastembed (ONNX, no torch), download once, and are
loaded lazily and shared per process -- never pickled with an index.
"""

import numpy as np

from .embeddings import Embedder, tfidf_analyzer
from .store import InMemoryVectorStore, StoreEntry
from .structure import with_context

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RRF_K = 60  # standard RRF constant; dampens the gap between rank 1 and 2
RERANK_CANDIDATES = 20

_models: dict[str, object] = {}


def _dense_model():
    if "dense" not in _models:
        from fastembed import TextEmbedding

        _models["dense"] = TextEmbedding(DENSE_MODEL)
    return _models["dense"]


def _reranker():
    if "rerank" not in _models:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        _models["rerank"] = TextCrossEncoder(RERANK_MODEL)
    return _models["rerank"]


def hybrid_available() -> bool:
    """False if fastembed/onnxruntime can't be imported (not installed, or
    blocked by the OS) -- callers fall back to TF-IDF."""
    try:
        import fastembed  # noqa: F401
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


class BM25:
    """Okapi BM25 over the same tokenizer as TF-IDF (stopwords removed,
    plurals folded), precomputed as a sparse doc x term weight matrix so a
    query is one sparse matrix-vector product."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        from scipy.sparse import csr_matrix
        from sklearn.feature_extraction.text import CountVectorizer

        self.vectorizer = CountVectorizer(analyzer=tfidf_analyzer)
        tf = self.vectorizer.fit_transform(texts).tocoo()
        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avg_len = doc_len.mean() or 1.0
        df = np.bincount(tf.col, minlength=tf.shape[1])
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        denom = tf.data + k1 * (1 - b + b * doc_len[tf.row] / avg_len)
        weights = idf[tf.col] * tf.data * (k1 + 1) / denom
        self.weights = csr_matrix((weights, (tf.row, tf.col)), shape=tf.shape)

    def score(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        q.data[:] = 1.0  # each query term counts once
        return np.asarray((self.weights @ q.T).todense()).ravel()


def _ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank of each item (1 = best)."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def rrf_fuse(*score_lists: np.ndarray, k: int = RRF_K) -> np.ndarray:
    return sum(1.0 / (k + _ranks(s)) for s in score_lists)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class HybridEmbedder(Embedder):
    name = "hybrid"
    # On the re-ranker's sigmoid probability, not TF-IDF cosine -- the two
    # scales aren't comparable. Measured: off-topic questions ("capital of
    # France", "reset my Wi-Fi", "PTO" against an insurance PDF) scored
    # <= 0.001; the lowest correct top answer was 0.023 (an application
    # number on a real PDF), with 0.087 the lowest on the 50-question eval.
    # 0.1 would have flagged both of those correct answers.
    low_confidence_threshold = 0.01

    def __init__(self):
        self.bm25: BM25 | None = None

    def fit(self, texts: list[str]) -> None:
        # texts must be in the same order the entries are added to the store
        self.bm25 = BM25(texts)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(_dense_model().embed(list(texts))))

    def score_passages(self, query: str, passages: list[str]) -> list[float]:
        """Re-ranker probability that each passage answers the query; used to
        pick the key sentences shown as the answer."""
        return list(_sigmoid(np.array(list(_reranker().rerank(query, passages)))))

    def search(
        self,
        store: InMemoryVectorStore,
        query: str,
        top_k: int = 3,
        rerank: bool = True,
    ) -> list[tuple[StoreEntry, float]]:
        if not store.entries or self.bm25 is None:
            return []

        mat = store._matrix()
        q = np.array(list(_dense_model().query_embed(query)))[0]
        dense = (mat @ q) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-10)
        fused = rrf_fuse(dense, self.bm25.score(query))

        n_candidates = max(RERANK_CANDIDATES, top_k) if rerank else top_k
        candidates = np.argsort(-fused, kind="stable")[:n_candidates]
        if not rerank:
            return [(store.entries[i], float(fused[i])) for i in candidates]

        entries = [store.entries[i] for i in candidates]
        passages = [with_context(e.title, e.section, e.text) for e in entries]
        logits = np.array(list(_reranker().rerank(query, passages)))
        order = np.argsort(-logits, kind="stable")[:top_k]
        probs = _sigmoid(logits)
        return [(entries[i], float(probs[i])) for i in order]
