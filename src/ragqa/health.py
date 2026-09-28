"""Self-retrieval health check for a freshly built index.

For a sample of chunks, take a sentence from the middle of the chunk, use it
as a query, and check the chunk comes back in the top-k. Every chunk *should*
be findable by its own wording; the ones that aren't point at real problems:
garbled extraction, near-duplicate sections competing with each other, or
boilerplate drowning out the content.

This is an upper bound, not an accuracy score: a real user's question won't
share the chunk's exact wording. A low score reliably means trouble; a high
score doesn't guarantee good answers. It exists so a new document gets a
quality signal the moment it's uploaded, instead of only when someone
notices a bad answer by hand.
"""

import re
from dataclasses import dataclass, field

from .embeddings import Embedder
from .retrieve import search_index
from .store import InMemoryVectorStore

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_MIN_PROBE_WORDS = 6


@dataclass
class HealthReport:
    checked: int
    found: int
    missed_chunk_ids: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.found / self.checked if self.checked else 0.0


def _probe_sentence(text: str) -> str | None:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text)]
    sentences = [s for s in sentences if len(s.split()) >= _MIN_PROBE_WORDS]
    if sentences:
        # middle sentence: the start/end of a chunk overlaps its neighbours
        return sentences[len(sentences) // 2]

    words = text.split()
    if len(words) < _MIN_PROBE_WORDS:
        return None
    mid = len(words) // 2
    return " ".join(words[max(0, mid - 6) : mid + 6])


def retrieval_health(
    embedder: Embedder,
    store: InMemoryVectorStore,
    top_k: int = 3,
    max_chunks: int = 150,
) -> HealthReport:
    entries = store.entries
    step = max(1, len(entries) // max_chunks)
    sample = entries[::step][:max_chunks]

    report = HealthReport(checked=0, found=0)
    for entry in sample:
        probe = _probe_sentence(entry.text)
        if probe is None:
            continue
        report.checked += 1
        results = search_index(embedder, store, probe, top_k=top_k)
        if any(e.chunk_id == entry.chunk_id for e, _ in results):
            report.found += 1
        else:
            report.missed_chunk_ids.append(entry.chunk_id)
    return report
