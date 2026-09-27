"""Answer generation with citations.

Provider-agnostic: uses the OpenAI API if OPENAI_API_KEY is set, otherwise
falls back to a deterministic extractive answer so the whole pipeline
(including the evaluation harness) runs with zero API keys and zero cost.
"""

import os

from .store import StoreEntry

Retrieved = list[tuple[StoreEntry, float]]


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
