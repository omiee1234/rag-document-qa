"""Split documents into chunks with stable, addressable ids."""

from dataclasses import dataclass

from .structure import HEADING, StructuredDoc, with_context


@dataclass
class Chunk:
    chunk_id: str  # f"{doc_id}::{index}" -- what citations and eval labels point at
    doc_id: str
    text: str
    start_char: int = -1
    end_char: int = -1
    title: str = ""
    section: str = ""
    page: int | None = None

    @property
    def embed_text(self) -> str:
        """What gets embedded: the chunk's text with its title/section in
        front, so it's retrievable by where it sits in the document. `text`
        stays the verbatim source, which is what's shown as the answer."""
        return with_context(self.title, self.section, self.text)


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


def chunk_structured(doc: StructuredDoc, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Pack a document's blocks into chunks of up to chunk_size characters.

    A heading always starts a new chunk and becomes its section, so a chunk
    never mixes two sections. Blocks are never split unless a single block is
    longer than chunk_size, in which case it falls back to character chunking
    (with overlap) within that block.
    """
    chunks: list[Chunk] = []
    section = ""
    buffer: list[str] = []
    buffer_page: int | None = None

    def emit(text: str, page: int | None) -> None:
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{len(chunks)}",
                doc_id=doc.doc_id,
                text=text,
                title=doc.title,
                section=section,
                page=page,
            )
        )

    def flush() -> None:
        if buffer:
            emit("\n".join(buffer), buffer_page)
            buffer.clear()

    for block in doc.blocks:
        text = block.text.strip()
        if not text:
            continue
        if block.kind == HEADING:
            flush()
            section = text
            continue
        if len(text) > chunk_size:
            flush()
            for piece in chunk_text(doc.doc_id, text, chunk_size=chunk_size, overlap=overlap):
                emit(piece.text, block.page)
            continue
        if buffer and sum(len(b) + 1 for b in buffer) + len(text) > chunk_size:
            flush()
        if not buffer:
            buffer_page = block.page
        buffer.append(text)
    flush()

    if not chunks and doc.blocks:
        # headings only (e.g. a one-line document): index them rather than nothing
        emit("\n".join(b.text for b in doc.blocks if b.text.strip()), doc.blocks[0].page)
    return chunks
