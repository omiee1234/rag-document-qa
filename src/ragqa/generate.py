"""Answer generation with citations.

Provider-agnostic: uses the OpenAI API if OPENAI_API_KEY is set, otherwise
falls back to a deterministic extractive answer so the whole pipeline
(including the evaluation harness) runs with zero API keys and zero cost.
"""

import os

from .store import StoreEntry

Retrieved = list[tuple[StoreEntry, float]]

# Below this, treat the top match as a low-confidence guess rather than a
# real answer. This is a heuristic, not a calibrated probability: checked
# against the bundled 50-question labeled set, correct top-1 matches range
# from 0.085 to 0.678 -- TF-IDF cosine similarity depends heavily on
# vocabulary overlap, not just relevance, so a hard cutoff would reject some
# genuinely correct (but lexically sparse) answers too. Callers should
# surface this as a warning to the user, not use it to silently withhold
# an answer -- see ui.py and cli.py for how it's used.
LOW_CONFIDENCE_THRESHOLD = 0.2


def low_confidence_threshold(embedder) -> float:
    """Scores aren't comparable across retrievers (TF-IDF cosine vs. the
    hybrid re-ranker's probability), so each can declare its own cutoff."""
    return getattr(embedder, "low_confidence_threshold", LOW_CONFIDENCE_THRESHOLD)


def _extractive_answer(retrieved: Retrieved) -> tuple[str, list[str]]:
    if not retrieved:
        return "I don't have enough information to answer that.", []
    top_entry, _score = retrieved[0]
    citations = [e.chunk_id for e, _ in retrieved]
    return top_entry.text, citations


def _llm_answer(query: str, retrieved: Retrieved, model: str = "gpt-4o-mini") -> tuple[str, list[str]]:
    from openai import OpenAI

    client = OpenAI()
    context = "\n\n".join(f"[{e.chunk_id}] {e.text}" for e, _ in retrieved)
    prompt = (
        "Answer the question using ONLY the context below. "
        "Cite the chunk id(s) you used in square brackets, e.g. [doc::0]. "
        "If the context doesn't contain the answer, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = resp.choices[0].message.content or ""
    citations = [e.chunk_id for e, _ in retrieved]
    return answer, citations


def answer_question(
    query: str, retrieved: Retrieved, use_llm: bool | None = None
) -> tuple[str, list[str]]:
    if use_llm is None:
        use_llm = bool(os.environ.get("OPENAI_API_KEY"))

    if use_llm:
        try:
            return _llm_answer(query, retrieved)
        except Exception as exc:  # pragma: no cover - network/API failure path
            print(f"LLM call failed ({exc}); falling back to extractive answer")

    return _extractive_answer(retrieved)
