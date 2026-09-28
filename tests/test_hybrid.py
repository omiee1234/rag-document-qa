"""Hybrid retrieval logic, tested with small fake models so CI never
downloads anything. The real models are exercised by the eval harness."""

import numpy as np
import pytest

from ragqa import hybrid
from ragqa.generate import LOW_CONFIDENCE_THRESHOLD, low_confidence_threshold
from ragqa.ingest import build_index_from_documents
from ragqa.retrieve import search_index

# "meaning" axes for the fake dense model: synonyms land on the same axis
_AXES = [
    {"leave", "vacation", "holiday", "holidays"},
    {"travel", "trip", "abroad", "meals"},
    {"password", "passwords", "login"},
]


class FakeDense:
    def _vec(self, text):
        words = {w.strip(".,?").lower() for w in text.split()}
        return np.array([len(words & axis) for axis in _AXES], dtype=float) + 1e-3

    def embed(self, texts):
        return [self._vec(t) for t in texts]

    def query_embed(self, query):
        return [self._vec(query)]


class FakeReranker:
    """Scores a passage high only if it contains the literal answer token."""

    answer = "ANSWER"

    def rerank(self, query, passages):
        return [5.0 if self.answer in p else -5.0 for p in passages]


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(hybrid, "_models", {"dense": FakeDense(), "rerank": FakeReranker()})


DOCS = {
    "uk": "UK employees receive 25 days of annual leave plus bank holidays.",
    "travel": "Meals during business travel are reimbursed up to 60 dollars.",
    "security": "Passwords must be at least 12 characters long.",
}


def test_bm25_ranks_exact_term_match_first():
    bm25 = hybrid.BM25(list(DOCS.values()))
    scores = bm25.score("password length characters")
    assert int(np.argmax(scores)) == 2
    assert scores[0] == 0.0  # no shared terms


def test_rrf_rewards_items_ranked_high_by_either_list():
    keyword = np.array([3.0, 2.0, 1.0])  # item 0 best by keywords
    meaning = np.array([1.0, 2.0, 3.0])  # item 2 best by meaning
    fused = hybrid.rrf_fuse(keyword, meaning)
    # items 0 and 2 are each #1 in one list and #3 in the other: tie, above item 1
    assert fused[0] == pytest.approx(fused[2])
    assert fused[0] > fused[1]


def test_paraphrase_found_by_dense_side_when_keywords_miss(fake_models):
    emb, store = build_index_from_documents(DOCS, embedder_name="hybrid")
    # no keyword overlap with the UK doc ("vacation" vs "leave"), only meaning
    results = search_index(emb, store, "How much vacation do I get in London", top_k=1, rerank=False)
    assert results[0][0].doc_id == "uk"


def test_reranker_decides_final_order(fake_models):
    docs = dict(DOCS)
    docs["form"] = "Name of the Proposer ANSWER Sample Person"
    docs["blank"] = "Name of the Proposer Name of the Proposer Signature of the Proposer"
    emb, store = build_index_from_documents(docs, embedder_name="hybrid")
    results = search_index(emb, store, "What is the name of the proposer", top_k=2)
    assert results[0][0].doc_id == "form"
    assert results[0][1] == pytest.approx(1 / (1 + np.exp(-5.0)))  # sigmoid of the logit


def test_hybrid_index_survives_pickling(fake_models):
    import pickle

    emb, store = build_index_from_documents(DOCS, embedder_name="hybrid")
    restored = pickle.loads(pickle.dumps(emb))
    assert restored.bm25 is not None
    assert "_models" not in vars(restored)  # models are never pickled with the index


def test_each_retriever_declares_its_own_confidence_threshold(fake_models):
    tfidf_emb, _ = build_index_from_documents(DOCS, embedder_name="tfidf")
    hybrid_emb, _ = build_index_from_documents(DOCS, embedder_name="hybrid")
    assert low_confidence_threshold(tfidf_emb) == LOW_CONFIDENCE_THRESHOLD
    assert low_confidence_threshold(hybrid_emb) == hybrid.HybridEmbedder.low_confidence_threshold
