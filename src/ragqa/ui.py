"""Streamlit demo UI: ask a question, see the cited answer, browse eval results.

Run with: streamlit run src/ragqa/ui.py
Free and local -- no external services, no API key required (uses the
extractive fallback in generate.py unless OPENAI_API_KEY is set).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragqa.generate import answer_question
from ragqa.retrieve import retrieve

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DOCS = ROOT / "data" / "docs"
DEFAULT_QA_SET = ROOT / "data" / "eval" / "qa_set.json"
DEFAULT_INDEX = ROOT / "index.pkl"

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="centered")
st.title("📄 RAG Document Q&A")
st.caption("Ask a question about the ingested documents. Answers are grounded and cited.")

if not DEFAULT_INDEX.exists():
    st.warning(
        f"No index found at `{DEFAULT_INDEX}`. Run this first:\n\n"
        f"`ragqa ingest --docs data/docs --index index.pkl --embedder tfidf`"
    )
    st.stop()

tab_ask, tab_eval = st.tabs(["Ask a question", "Evaluation results"])

with tab_ask:
    top_k = st.slider("How many chunks to retrieve (top-k)", min_value=1, max_value=5, value=3)
    question = st.text_input("Your question", placeholder="How many days of PTO do I get?")

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Retrieving and answering..."):
            retrieved = retrieve(question, str(DEFAULT_INDEX), top_k=top_k)
            answer, citations = answer_question(question, retrieved)

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
        "MRR, and a faithfulness check."
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
