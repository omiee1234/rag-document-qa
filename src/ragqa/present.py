"""Turning a retrieved chunk into a readable answer: the key sentences with
the question's words highlighted, a confidence level, and list formatting.

Everything shown stays verbatim source text -- sentences are *selected*,
never rewritten -- so the grounding guarantee of the extractive answer holds.
"""

import re

from .embeddings import tfidf_analyzer

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])|\n+")
_WORD_RE = re.compile(r"\w+|\W+")
_MARKDOWN_SPECIAL_RE = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~<])")
_MIN_SENTENCE_WORDS = 4

# (low, high) score cut points per retriever. Below `low` is the existing
# low-confidence warning threshold; `high` marks answers the retriever is
# sure of. Hybrid scores are re-ranker probabilities; TF-IDF are cosines.
_BANDS = {"hybrid": (0.01, 0.5), "tfidf": (0.2, 0.4)}

CONFIDENCE_BADGES = {
    "high": ("🟢", "High confidence"),
    "medium": ("🟡", "Medium confidence"),
    "low": ("🔴", "Low confidence"),
}


def confidence_level(embedder, score: float) -> str:
    low, high = _BANDS.get(getattr(embedder, "name", "tfidf"), _BANDS["tfidf"])
    if score >= high:
        return "high"
    if score >= low:
        return "medium"
    return "low"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()]


def _overlap_score(query: str, passages: list[str]) -> list[float]:
    q = set(tfidf_analyzer(query))
    return [len(q & set(tfidf_analyzer(p))) / (len(q) or 1) for p in passages]


def key_sentences(query: str, text: str, embedder=None, max_sentences: int = 2) -> str:
    """The 1-2 sentences of `text` that best answer `query`, in their original
    order. Scored by the retriever's own re-ranker when it has one, otherwise
    by word overlap. Short chunks are returned whole."""
    sentences = split_sentences(text)
    candidates = [s for s in sentences if len(s.split()) >= _MIN_SENTENCE_WORDS]
    if len(candidates) <= max_sentences:
        return text.strip()

    scorer = getattr(embedder, "score_passages", None)
    scores = scorer(query, candidates) if scorer else _overlap_score(query, candidates)
    best = sorted(range(len(candidates)), key=lambda i: -scores[i])[:max_sentences]
    return " ".join(candidates[i] for i in sorted(best))


def escape_markdown(text: str) -> str:
    return _MARKDOWN_SPECIAL_RE.sub(r"\\\1", text)


def highlight(text: str, query: str) -> str:
    """Markdown with the question's words in bold. Matching uses the same
    tokenizer as search (plurals folded, stopwords ignored), so "persons"
    highlights "person". Everything else is escaped verbatim."""
    wanted = set(tfidf_analyzer(query))
    out = []
    for piece in _WORD_RE.findall(text):
        escaped = escape_markdown(piece)
        tokens = tfidf_analyzer(piece)
        out.append(f"**{escaped}**" if tokens and tokens[0] in wanted else escaped)
    return "".join(out)


def to_markdown(text: str, query: str = "") -> str:
    """Render a chunk: a list when it's made of several short lines (table
    rows, bullet points), otherwise paragraphs; line breaks preserved."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    render = (lambda s: highlight(s, query)) if query else escape_markdown
    if len(lines) >= 3 and sum(len(l) for l in lines) / len(lines) < 200:
        return "\n".join(f"- {render(l.lstrip('•·-* ').strip() or l)}" for l in lines)
    return "  \n".join(render(l) for l in lines)
