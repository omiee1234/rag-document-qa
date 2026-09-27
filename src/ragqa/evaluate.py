"""Evaluation harness: recall@k, precision@k, MRR, and a faithfulness check.

This module is what separates this project from "wire up LangChain and hope":
it scores retrieval and generation against a hand-labeled ground truth set
instead of eyeballing a few example answers.
"""

import json
from pathlib import Path

from .generate import answer_question
from .retrieve import retrieve


def load_qa_set(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recall_at_k(retrieved_ids: list[str], correct_ids: list[str]) -> float:
    """Of all correct chunks, what fraction did we manage to retrieve?"""
    if not correct_ids:
        return 0.0
    hits = len(set(retrieved_ids) & set(correct_ids))
    return hits / len(correct_ids)


def precision_at_k(retrieved_ids: list[str], correct_ids: list[str]) -> float:
    """Of what we retrieved, what fraction was actually correct?"""
    if not retrieved_ids:
        return 0.0
    hits = len(set(retrieved_ids) & set(correct_ids))
    return hits / len(retrieved_ids)


def reciprocal_rank(retrieved_ids: list[str], correct_ids: list[str]) -> float:
    """1/rank of the first correct chunk; 0 if none of the top-k were correct."""
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in correct_ids:
            return 1.0 / rank
    return 0.0


def faithfulness_overlap(answer: str, retrieved_texts: list[str], threshold: float = 0.3) -> bool:
    """Cheap faithfulness proxy: are the answer's content words actually present
    in the retrieved context, or did the model make things up?

    This is intentionally a simple word-overlap heuristic, not an LLM judge --
    it's fast, free, and deterministic, which matters for a CI-friendly eval.
    Swap in an LLM-as-judge call here for a stricter check if needed.
    """
    answer_words = {w.lower().strip(".,;:!?") for w in answer.split() if len(w) > 3}
    if not answer_words:
        return True

    context_words: set[str] = set()
    for text in retrieved_texts:
        context_words.update(w.lower().strip(".,;:!?") for w in text.split())

    overlap = len(answer_words & context_words) / len(answer_words)
    return overlap >= threshold


def run_evaluation(
    qa_set_path: str,
    index_path: str,
    top_k: int = 3,
    use_llm: bool | None = None,
) -> dict:
    qa_set = load_qa_set(qa_set_path)
    if not qa_set:
        raise ValueError(f"No questions found in {qa_set_path}")

    recalls, precisions, rrs, faithful_flags = [], [], [], []
    per_question = []

    for item in qa_set:
        question = item["question"]
        correct_ids = item["correct_chunk_ids"]

        retrieved = retrieve(question, index_path, top_k=top_k)
        retrieved_ids = [e.chunk_id for e, _ in retrieved]

        r = recall_at_k(retrieved_ids, correct_ids)
        p = precision_at_k(retrieved_ids, correct_ids)
        rr = reciprocal_rank(retrieved_ids, correct_ids)
        recalls.append(r)
        precisions.append(p)
        rrs.append(rr)

        answer, citations = answer_question(question, retrieved, use_llm=use_llm)
        retrieved_texts = [e.text for e, _ in retrieved]
        faithful = faithfulness_overlap(answer, retrieved_texts)
        faithful_flags.append(faithful)

        per_question.append(
            {
                "question": question,
                "retrieved_ids": retrieved_ids,
                "correct_ids": correct_ids,
                "recall": r,
                "precision": p,
                "reciprocal_rank": rr,
                "faithful": faithful,
            }
        )

    n = len(qa_set)
    return {
        "n_questions": n,
        "top_k": top_k,
        "recall_at_k": sum(recalls) / n,
        "precision_at_k": sum(precisions) / n,
        "mrr": sum(rrs) / n,
        "faithfulness": sum(faithful_flags) / n,
        "per_question": per_question,
    }
