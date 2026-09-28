"""Split document text into overlapping chunks with stable, addressable ids."""

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str  # f"{doc_id}::{index}" -- what citations and eval labels point at
    doc_id: str
    text: str
    start_char: int
    end_char: int


def chunk_text(doc_id: str, text: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Greedy fixed-size chunking with overlap, snapped to whitespace so words aren't split."""
    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            while end < n and not text[end].isspace():
                end += 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{index}",
                    doc_id=doc_id,
                    text=piece,
                    start_char=start,
                    end_char=end,
                )
            )
            index += 1

        if end >= n:
            break
        start = max(end - overlap, start + 1)  # always make forward progress
        # snap forward to the next word boundary so the *next* chunk doesn't
        # start mid-word (end is already snapped to whitespace above, but
        # that only protects the end of THIS chunk, not the start of the
        # next one -- without this, overlap could land inside a word, e.g.
        # a chunk starting with "licable" instead of "applicable")
        while start < n and not text[start - 1].isspace():
            start += 1

    return chunks


def chunk_documents(
    documents: dict[str, str], chunk_size: int = 800, overlap: int = 150
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc_id, text in documents.items():
        all_chunks.extend(chunk_text(doc_id, text, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
