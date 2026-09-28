from ragqa.health import _probe_sentence, retrieval_health
from ragqa.ingest import build_index_from_documents

DISTINCT_DOCS = {
    "leave": "Employees receive eighteen days of paid vacation each calendar year from January.",
    "expenses": "Meals during business travel are reimbursed up to sixty dollars per day with receipts.",
    "security": "Passwords must contain twelve characters and multi factor authentication is mandatory.",
}


def test_distinct_documents_are_fully_findable():
    embedder, store = build_index_from_documents(DISTINCT_DOCS)
    report = retrieval_health(embedder, store, top_k=1)
    assert report.checked == 3
    assert report.score == 1.0


def test_near_duplicate_documents_lower_the_score():
    # Identical text in several documents: at top_k=1 only one copy can win,
    # so the health check flags that chunks compete with each other.
    same = "The free look period lets the insured person return the policy within thirty days."
    docs = {f"copy{i}": same for i in range(4)}
    embedder, store = build_index_from_documents(docs)
    report = retrieval_health(embedder, store, top_k=1)
    assert report.score < 1.0
    assert report.missed_chunk_ids


def test_probe_uses_a_middle_sentence():
    text = (
        "First sentence is about something short here. "
        "Middle sentence talks about the actual coverage details. "
        "Last sentence closes the paragraph with more words."
    )
    assert _probe_sentence(text) == "Middle sentence talks about the actual coverage details."


def test_probe_skips_chunks_too_short_to_test():
    assert _probe_sentence("Yes. No.") is None
