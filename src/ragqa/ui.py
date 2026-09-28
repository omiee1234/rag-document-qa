"""Streamlit demo UI: ask a question, see the cited answer, browse eval results.

Run with: streamlit run src/ragqa/ui.py
Free and local -- no external services, no API key required (uses the
extractive fallback in generate.py unless OPENAI_API_KEY is set).
"""

import hashlib
import html
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _ragqa_modules() -> dict:
    return {
        name: module
        for name, module in sys.modules.items()
        if (name == "ragqa" or name.startswith("ragqa.")) and getattr(module, "__file__", None)
    }


def _drop_stale_ragqa_modules() -> None:
    """Streamlit Cloud applies a git push by re-running this script in the
    same Python process, so previously imported ragqa modules stay cached at
    their old version. Seen in production: importing a newly added function
    from an already-loaded module failed with ImportError. If any ragqa
    source file changed since it was loaded, drop *all* ragqa modules --
    dropping only the changed one would leave unchanged modules still
    holding references into its old version."""
    modules = _ragqa_modules()
    for module in modules.values():
        path = module.__file__
        if not os.path.exists(path) or getattr(module, "_loaded_mtime", None) != os.path.getmtime(path):
            for name in modules:
                del sys.modules[name]
            return


_drop_stale_ragqa_modules()

from ragqa.generate import answer_question, low_confidence_threshold
from ragqa.health import HealthReport, retrieval_health
from ragqa.hybrid import hybrid_available
from ragqa.ingest import build_index_from_documents
from ragqa.loaders import load_structured_bytes
from ragqa.present import CONFIDENCE_BADGES, confidence_level, key_sentences, to_markdown
from ragqa.retrieve import load_index, search_index

for _module in _ragqa_modules().values():
    _module._loaded_mtime = os.path.getmtime(_module.__file__)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DOCS = ROOT / "data" / "docs"
DEFAULT_QA_SET = ROOT / "data" / "eval" / "qa_set.json"
SAMPLE_INDEX = {"tfidf": ROOT / "index.pkl", "hybrid": ROOT / "index_hybrid.pkl"}
MODE_LABELS = {
    "hybrid": "Hybrid: keywords + meaning + re-ranker (recommended)",
    "tfidf": "Keyword only: TF-IDF baseline",
}

MAX_FILES = 10
MAX_TOTAL_BYTES = 10 * 1024 * 1024  # 10 MB

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="centered")
st.title("📄 RAG Document Q&A")
st.caption("Ask a question about the ingested documents. Answers are grounded and cited.")


@st.cache_resource(show_spinner="Loading search models (first run downloads ~150 MB, about a minute)...")
def warm_up_hybrid() -> str | None:
    """Load both hybrid models once per process. Returns an error message if
    they can't load (e.g. out of memory on the host) so the app can fall back
    to TF-IDF instead of breaking."""
    try:
        from ragqa.hybrid import _dense_model, _reranker

        list(_dense_model().embed(["warm up"]))
        list(_reranker().rerank("warm up", ["warm up"]))
    except Exception as exc:  # any failure here should degrade, not crash the app
        return f"{type(exc).__name__}: {exc}"
    return None


@st.cache_resource(show_spinner="Building the sample index (first run only)...")
def ensure_sample_index(mode: str) -> None:
    """Build the shared sample-corpus index on first run if it isn't already on disk.

    index.pkl is a build artifact (gitignored, not shipped in the repo) --
    a pickled sklearn/numpy object isn't safe to commit and unpickle across
    different environments (e.g. local Windows dev vs. Streamlit Cloud's
    Linux runtime with possibly different package versions). Rebuilding
    fresh from data/docs on first run is cheap (TF-IDF, ~20 short docs)
    and avoids that whole class of failure. st.cache_resource makes this
    run only once per app instance, not on every rerun.
    """
    import pickle

    from ragqa.ingest import INDEX_FORMAT_VERSION, ingest

    path = SAMPLE_INDEX[mode]
    if path.exists():
        # a redeploy can leave an index built by older code on disk; rebuild
        # it rather than query a different chunking scheme
        with open(path, "rb") as f:
            if pickle.load(f).get("version") == INDEX_FORMAT_VERSION:
                return
    ingest(str(DEFAULT_DOCS), str(path), embedder_name=mode)


def build_uploaded_index(files, mode: str) -> tuple:
    """Build an in-memory index from uploaded files.

    Never written to disk -- Streamlit Cloud can serve multiple visitors
    from the same app process, so writing uploads to the shared index.pkl
    (or any shared path) would let one user's documents leak into or
    clobber another user's session. Everything here lives only in this
    browser session's st.session_state.
    """
    documents = []
    seen: dict[str, int] = {}
    for f in files:
        doc_id = Path(f.name).stem
        if doc_id in seen:
            seen[doc_id] += 1
            doc_id = f"{doc_id}_{seen[doc_id]}"
        else:
            seen[doc_id] = 0
        documents.append(load_structured_bytes(f.name, f.getvalue(), doc_id=doc_id))
    return build_index_from_documents(documents, embedder_name=mode)


def source_label_for(entry) -> str:
    # getattr: a session can still hold entries built by an older version of
    # the app (see _drop_stale_ragqa_modules), which lack these fields
    title, section, page = (getattr(entry, f, None) for f in ("title", "section", "page"))
    parts = [p for p in (title, section) if p]
    if len(parts) == 2 and parts[0] == parts[1]:
        parts = parts[:1]
    where = " › ".join(parts) or entry.doc_id
    return f"{where} · page {page}" if page else where


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


st.sidebar.header("Search mode")
available_modes = ["hybrid", "tfidf"] if hybrid_available() else ["tfidf"]
mode = st.sidebar.radio("Retrieval method:", available_modes, format_func=MODE_LABELS.get)
if not hybrid_available():
    st.sidebar.caption("Hybrid search isn't available here (fastembed couldn't be loaded).")
if mode == "hybrid":
    error = warm_up_hybrid()
    if error:
        st.sidebar.warning(f"Hybrid models failed to load, using keyword search instead. ({error})")
        mode = "tfidf"

ensure_sample_index(mode)

st.sidebar.header("Document source")
corpus_choice = st.sidebar.radio(
    "Ask questions against:",
    ["Sample HR policies (18 docs)", "My uploaded documents"],
)

active_embedder = None
active_store = None
source_label = ""

if corpus_choice == "Sample HR policies (18 docs)":
    active_embedder, active_store = load_index(str(SAMPLE_INDEX[mode]))
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
                mode.encode() + b"".join(f.name.encode() + f.getvalue() for f in uploaded_files)
            ).hexdigest()

            if st.session_state.get("upload_fingerprint") != fingerprint:
                try:
                    with st.spinner("Reading document structure and indexing..."):
                        embedder, store = build_uploaded_index(uploaded_files, mode)
                    with st.spinner("Checking retrieval health..."):
                        health = retrieval_health(embedder, store)
                except ValueError as exc:
                    st.sidebar.error(f"Couldn't index these files: {exc}")
                    st.session_state.pop("upload_fingerprint", None)
                    st.session_state.pop("upload_store", None)
                else:
                    st.session_state["upload_fingerprint"] = fingerprint
                    st.session_state["upload_embedder"] = embedder
                    st.session_state["upload_store"] = store
                    st.session_state["upload_names"] = [f.name for f in uploaded_files]
                    st.session_state["upload_health"] = health

            if st.session_state.get("upload_fingerprint") == fingerprint:
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

            if not retrieved:
                st.info("Nothing in the selected documents matched this question.")
            else:
                top_entry, top_score = retrieved[0]
                icon, label = CONFIDENCE_BADGES[confidence_level(active_embedder, top_score)]

                with st.container(border=True):
                    left, right = st.columns([1, 2])
                    left.markdown(f"{icon} **{label}**")
                    right.markdown(
                        f"<div style='text-align:right; opacity:0.75'>📄 "
                        f"{html.escape(source_label_for(top_entry))}</div>",
                        unsafe_allow_html=True,
                    )
                    if top_score < low_confidence_threshold(active_embedder):
                        st.caption(
                            "The documents may not contain an answer to this — "
                            "this is the closest passage found, not a reliable answer."
                        )
                    if answer == top_entry.text:  # extractive: show the key sentences
                        with st.spinner("Picking the key sentences..."):
                            key = key_sentences(question, top_entry.text, active_embedder)
                    else:  # an LLM-written answer (only if OPENAI_API_KEY is set)
                        key = answer
                    st.markdown(f"> {to_markdown(key, question)}".replace("\n", "\n> "))
                    with st.expander("Show full section"):
                        st.markdown(to_markdown(top_entry.text, question))

                st.markdown("**Sources**")
                for rank, (entry, score) in enumerate(retrieved, 1):
                    badge = CONFIDENCE_BADGES[confidence_level(active_embedder, score)][0]
                    with st.expander(f"{badge} {rank}. {source_label_for(entry)}"):
                        st.markdown(to_markdown(entry.text, question))
                        st.caption(f"{entry.chunk_id} · score {score:.3f}")

with tab_eval:
    st.write(
        "Scores the retrieval + generation pipeline against a hand-labeled "
        "question set (`data/eval/qa_set.json`) -- recall@k, precision@k, "
        "MRR, and a faithfulness check. This always runs against the "
        "**sample HR policy documents**, since the labeled answer key was "
        "written for that corpus, not whatever you upload. Switch the search "
        "mode in the sidebar to compare hybrid against the TF-IDF baseline."
    )
    st.caption(f"Scoring: **{MODE_LABELS[mode]}**")
    cache_key = f"eval::{mode}::{os.path.getmtime(SAMPLE_INDEX[mode])}"
    if st.button("Run evaluation") or cache_key in st.session_state:
        if cache_key not in st.session_state:
            # ~1s per question in hybrid mode (the re-ranker), so show progress
            # and keep the result: the sample corpus doesn't change
            from ragqa.evaluate import run_evaluation

            bar = st.progress(0.0, text="Scoring 50 questions...")
            st.session_state[cache_key] = run_evaluation(
                str(DEFAULT_QA_SET),
                str(SAMPLE_INDEX[mode]),
                top_k=3,
                progress=lambda done, total: bar.progress(done / total, text=f"Question {done + 1} of {total}"),
            )
            bar.empty()
        results = st.session_state[cache_key]

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
