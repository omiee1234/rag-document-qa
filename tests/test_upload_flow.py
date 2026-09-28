"""Integration test for the upload path used by ui.py: bytes in -> retrieval out.

Exercises load_document_bytes -> build_index_from_documents -> search_index,
the exact chain the Streamlit upload feature runs, without needing a browser.
"""

from ragqa.ingest import build_index_from_documents
from ragqa.loaders import load_document_bytes
from ragqa.retrieve import search_index

BICYCLE_POLICY = b"""Company Bicycle Loan Program

Employees may borrow a company bicycle for commuting purposes for up to
90 days, renewable once with manager approval. A refundable deposit of
$50 is required, returned in full when the bicycle is returned in good
condition. Helmets are provided free of charge and must be worn at all
times while riding a company bicycle. Lost or stolen bicycles must be
reported to Facilities within 24 hours.
"""

UNRELATED_POLICY = b"""Office Supplies Policy

Employees can request standard office supplies (pens, notebooks, sticky
notes) through the facilities portal at no cost. Requests for specialty
equipment over $100 require manager approval.
"""


def test_uploaded_txt_file_is_retrievable():
    text = load_document_bytes("bicycle_policy.txt", BICYCLE_POLICY)
    assert "bicycle" in text.lower()

    documents = {
        "bicycle_policy": text,
        "office_supplies": load_document_bytes("office_supplies.txt", UNRELATED_POLICY),
    }
    embedder, store = build_index_from_documents(documents, embedder_name="tfidf")

    assert len(store) == 2

    results = search_index(embedder, store, "How long can I borrow a company bicycle?", top_k=2)

    assert results[0][0].chunk_id == "bicycle_policy::0"
    assert "90 days" in results[0][0].text


def test_duplicate_filenames_get_unique_doc_ids():
    # Mirrors ui.py's collision handling for two uploads with the same stem.
    documents = {
        "policy": "First version of the policy text about vacation days.",
        "policy_1": "Second version of the policy text about sick leave.",
    }
    embedder, store = build_index_from_documents(documents, embedder_name="tfidf")
    chunk_ids = {e.chunk_id for e in store.entries}
    assert chunk_ids == {"policy::0", "policy_1::0"}
