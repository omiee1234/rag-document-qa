"""The document schema every source format is parsed into.

Loaders turn a PDF / markdown / text file into a StructuredDoc: a title plus
an ordered list of typed blocks (heading, paragraph, table_row) with the page
each came from. The chunker then packs blocks into chunks without ever
crossing a heading, and every chunk carries its document title, section and
page. That context is what gets embedded, so a chunk is retrievable by
*where it sits* in the document, not only by the words inside it.
"""

import re
from dataclasses import dataclass, field

HEADING = "heading"
PARAGRAPH = "paragraph"
TABLE_ROW = "table_row"


@dataclass
class Block:
    text: str
    kind: str = PARAGRAPH
    page: int | None = None


@dataclass
class StructuredDoc:
    doc_id: str
    title: str = ""
    blocks: list[Block] = field(default_factory=list)

    def plain_text(self) -> str:
        parts = [self.title] if self.title else []
        parts.extend(b.text for b in self.blocks)
        return "\n\n".join(parts)


def context_label(title: str, section: str) -> str:
    parts = [title]
    if section and section != title:
        parts.append(section)
    return " › ".join(p for p in parts if p)


def with_context(title: str, section: str, text: str) -> str:
    """A chunk as the retriever sees it: "Title › Section" on the first line,
    then the verbatim text. Used for embedding and for re-ranking, so both
    judge a chunk with the same context."""
    label = context_label(title, section)
    return f"{label}\n{text}" if label else text


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def parse_markdown(doc_id: str, text: str) -> StructuredDoc:
    """Markdown / plain text: '#' lines are headings, blank lines separate
    paragraphs. The first level-1 heading becomes the document title."""
    doc = StructuredDoc(doc_id=doc_id)
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            doc.blocks.append(Block(" ".join(paragraph), PARAGRAPH))
            paragraph.clear()

    for line in text.splitlines():
        stripped = line.strip()
        match = _MD_HEADING_RE.match(stripped)
        if match:
            flush()
            heading = match.group(2)
            if not doc.title and len(match.group(1)) == 1 and not doc.blocks:
                doc.title = heading
            elif heading:
                doc.blocks.append(Block(heading, HEADING))
        elif not stripped:
            flush()
        else:
            paragraph.append(stripped)
    flush()
    return doc
