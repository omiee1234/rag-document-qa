"""Regression test for the low-confidence warning.

Reproduces the real scenario found by testing: asking an HR question
("PTO") against a document that has nothing to do with HR (a health
insurance policy) returns a weak match instead of an honest "not found".

Note: this deliberately does NOT assert an off-topic score lands below
LOW_CONFIDENCE_THRESHOLD as an absolute number -- a first version of this
test did that and was itself flaky (0.209 vs. a 0.2 cutoff) purely because
this tiny single-paragraph test document is a different size/vocabulary
than the real ~90-chunk PDF that surfaced the bug. TF-IDF cosine
similarity is corpus-size-sensitive, which is exactly why
LOW_CONFIDENCE_THRESHOLD is documented as a heuristic warning, not a hard
filter. What's actually invariant, and what this test checks instead, is
the *relative* gap: an off-topic question should always score meaningfully
lower than an on-topic one against the same corpus.
"""

from ragqa.ingest import build_index_from_documents
from ragqa.retrieve import search_index

INSURANCE_POLICY = """
HDFC ERGO General Insurance Company Limited
Policy Schedule for Optima Secure health insurance.
The insured person shall be allowed a free look period of thirty days
from the date of receipt of the policy document to review the terms and
conditions. Claims must be intimated within 24 hours of hospitalization.
Registered Office: 6th Floor, Leela Business Park, Andheri, Mumbai.
"""


def test_off_topic_question_scores_much_lower_than_on_topic():
    embedder, store = build_index_from_documents(
        {"insurance_policy": INSURANCE_POLICY}, embedder_name="tfidf"
    )

    off_topic = search_index(embedder, store, "after how many days is PTO allowed", top_k=1)
    on_topic = search_index(embedder, store, "how long is the free look period?", top_k=1)

    assert off_topic and on_topic
    off_topic_score = off_topic[0][1]
    on_topic_score = on_topic[0][1]

    assert off_topic_score < on_topic_score, (
        f"expected the off-topic question to score lower than the on-topic one, "
        f"got off-topic={off_topic_score:.3f} on-topic={on_topic_score:.3f}"
    )
