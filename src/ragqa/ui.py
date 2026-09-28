"""Streamlit demo UI: ask a question, see the cited answer, browse eval results.

Run with: streamlit run src/ragqa/ui.py
Free and local -- no external services, no API key required (uses the
extractive fallback in generate.py unless OPENAI_API_KEY is set).
"""

import hashlib
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragqa.generate import LOW_CONFIDENCE_THRESHOLD, answer_question
from ragqa.health import HealthReport, retrieval_health
from ragqa.ingest import build_index_from_documents
from ragqa.loaders import load_document_bytes
from ragqa.retrieve import retrieve, search_index

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DOCS = ROOT / "data" / "docs"
DEFAULT_QA_SET = ROOT / "data" / "eval" / "qa_set.json"
DEFAULT_INDEX = ROOT / "index.pkl"

MAX_FILES = 10
MAX_TOTAL_BYTES = 10 * 1024 * 1024  # 10 MB

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="centered")
st.title("📄 RAG Document Q&A")
st.caption("Ask a question about the ingested documents. Answers are grounded and cited.")


@st.cache_resource(show_spinner="Building index from data/docs (first run only)...")
def ensure_sample_index() -> None:
    """Build the shared sample-corpus index on first run if it isn't already on disk.

    index.pkl is a build artifact (gitignored, not shipped in the repo) --
    a pickled sklearn/numpy object isn't safe to commit and unpickle across
    different environments (e.g. local Windows dev vs. Streamlit Cloud's
    Linux runtime with possibly different package versions). Rebuilding
    fresh from data/docs on first run is cheap (TF-IDF, ~20 short docs)
    and avoids that whole class of failure. st.cache_resource makes this
    run only once per app instance, not on every rerun.
    """
    if not DEFAULT_INDEX.exists():
        from ragqa.ingest import ingest

        ingest(str(DEFAULT_DOCS), str(DEFAULT_INDEX), embedder_name="tfidf")


def build_uploaded_index(files) -> tuple:
    """Build an in-memory index from uploaded files.

    Never written to disk -- Streamlit Cloud can serve multiple visitors
    from the same app process, so writing uploads to the shared index.pkl
    (or any shared path) would let one user's documents leak into or
    clobber another user's session. Everything here lives only in this
    browser session's st.session_state.
    """
    documents: dict[str, str] = {}
    seen: dict[str, int] = {}
    for f in files:
        text = load_document_bytes(f.name, f.getvalue())
        doc_id = Path(f.name).stem
        if doc_id in seen:
            seen[doc_id] += 1
            doc_id = f"{doc_id}_{seen[doc_id]}"
        else:
            seen[doc_id] = 0
        documents[doc_id] = text
    return build_index_from_documents(documents, embedder_name="tfidf")


def show_health(report: HealthReport | None) -> None:
    if report is None or report.checked == 0:
        return
    msg = (
        f"**Retrieval health: {report.score:.0%}** — {report.found} of "
        f"{report.checked} sections can be found by their own wording."
    )
    if report.score >= 0.9:
        st.sidebar.success(msg + " This document should answer questions well.")
    elif report.score >= 0.75:
        st.sidebar.info(
            msg + " Usually fine; some sections are repetitive or hard to "
            "separate, so check citations on important answers."
        )
    else:
        st.sidebar.warning(
            msg + " Many sections are hard to retrieve — often a scanned PDF, "
            "heavy tables, or a lot of repeated text. Expect weaker answers."
        )


ensure_sample_index()

st.sidebar.header("Document source")
corpus_choice = st.sidebar.radio(
    "Ask questions against:",
    ["Sample HR policies (18 docs)", "My uploaded documents"],
)

active_embedder = None
active_store = None
source_label = ""

if corpus_choice == "Sample HR policies (18 docs)":
    from ragqa.retrieve import load_index

    active_embedder, active_store = load_index(str(DEFAULT_INDEX))
    source_label = "sample HR policy documents"
else:
    st.sidebar.caption(f"Up to {MAX_FILES} files, {MAX_TOTAL_BYTES // (1024*1024)} MB total. PDF, TXT, or MD.")
    st.sidebar.warning(
        "This is a public demo. Don't upload documents containing personal "
        "information (names, addresses, policy/ID numbers, medical details). "
        "Uploads stay in your browser session's memory only and are never "
        "saved to disk, but use sample or public documents to be safe."
    )
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if len(uploaded_files) > MAX_FILES:
            st.sidebar.error(f"Too many files ({len(uploaded_files)}). Max {MAX_FILES}.")
        elif sum(f.size for f in uploaded_files) > MAX_TOTAL_BYTES:
            st.sidebar.error("Total upload size exceeds 10 MB.")
        else:
            fingerprint = hashlib.sha256(
                b"".join(f.name.encode() + f.getvalue() for f in uploaded_files)
            ).hexdigest()

            if st.session_state.get("upload_fingerprint") != fingerprint:
                with st.spinner("Indexing your documents..."):
                    embedder, store = build_uploaded_index(uploaded_files)
                with st.spinner("Checking retrieval health..."):
                    health = retrieval_health(embedder, store)
                st.session_state["upload_fingerprint"] = fingerprint
                st.session_state["upload_embedder"] = embedder
                st.session_state["upload_store"] = store
                st.session_state["upload_names"] = [f.name for f in uploaded_files]
                st.session_state["upload_health"] = health

            active_embedder = st.session_state.get("upload_embedder")
            active_store = st.session_state.get("upload_store")
            source_label = f"{len(st.session_state.get('upload_names', []))} uploaded document(s)"
            st.sidebar.success("Indexed: " + ", ".join(st.session_state.get("upload_names", [])))
            show_health(st.session_state.get("upload_health"))

tab_ask, tab_eval = st.tabs(["Ask a question", "Evaluation results"])

with tab_ask:
    if active_store is None:
        st.info("Upload at least one document in the sidebar to ask questions.")
    else:
        st.caption(f"Querying: {source_label} ({len(active_store)} chunks)")
        top_k = st.slider("How many chunks to retrieve (top-k)", min_value=1, max_value=8, value=5)
        question = st.text_input("Your question", placeholder="How many days of PTO do I get?")

        if st.button("Ask", type="primary") and question.strip():
            with st.spinner("Retrieving and answering..."):
                retrieved = search_index(active_embedder, active_store, question, top_k=top_k)
                answer, citations = answer_question(question, retrieved)

            top_score = retrieved[0][1] if retrieved else 0.0
            if top_score < LOW_CONFIDENCE_THRESHOLD:
                st.warning(
                    f"⚠️ Low-confidence match (top similarity: {top_score:.3f}). "
                    "The selected document(s) may not actually contain a good "
                    "answer to this question -- treat the text below as the "
                    "closest match found, not a reliable answer."
                )

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Cited chunks")
            for entry, score in retrieved:
                with st.expander(f"{entry.chunk_id}  (similarity: {score:.3f})"):
                    st.write(entry.text)

with tab_eval:
    st.write(
        "Scores the retrieval + generation pipeline against a hand-labeled "
        "question set (`data/eval/qa_set.json`) -- recall@k, precision@k, "
        "MRR, and a faithfulness check. This always runs against the "
        "**sample HR policy documents**, since the labeled answer key was "
        "written for that corpus, not whatever you upload."
    )
    if st.button("Run evaluation"):
        with st.spinner("Running evaluation harness..."):
            from ragqa.evaluate import run_evaluation

            results = run_evaluation(str(DEFAULT_QA_SET), str(DEFAULT_INDEX), top_k=3)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Recall@3", f"{results['recall_at_k']:.2f}")
        col2.metric("Precision@3", f"{results['precision_at_k']:.2f}")
        col3.metric("MRR", f"{results['mrr']:.2f}")
        col4.metric("Faithfulness", f"{results['faithfulness']:.2f}")

        st.subheader("Per-question detail")
        st.dataframe(
            [
                {
                    "question": q["question"],
                    "recall": q["recall"],
                    "reciprocal_rank": q["reciprocal_rank"],
                    "faithful": q["faithful"],
                }
                for q in results["per_question"]
            ],
            use_container_width=True,
        )
