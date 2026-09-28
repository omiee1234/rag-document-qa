"""Load source documents (PDF/TXT/MD) into the StructuredDoc schema."""

import io
import math
import re
from collections import Counter
from pathlib import Path

from .structure import HEADING, PARAGRAPH, TABLE_ROW, Block, StructuredDoc, parse_markdown

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}

# Characters that render as a "tofu" box and carry no readable meaning once
# extracted:
#  - \x00-\x1f/\x7f: control characters
#  - �: the Unicode replacement character extractors emit for glyphs
#    they can't decode at all
#  - -: the Private Use Area. Word-generated PDFs (a very common
#    real-world source -- e.g. insurance/legal docs) often encode bullet
#    points in a Symbol/Wingdings font mapped to PUA codepoints (observed:
#    U+F0B7); with no font info attached it comes through as an unreadable box.
_UNDECODABLE_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f�-]")
_EXTRA_SPACES_RE = re.compile(r"[ \t]{2,}")

# Running headers/footers ("Product Name: ... | Page 3 of 8", a company
# registration footer) repeat on every page. Left in, they land in nearly
# every chunk, dilute TF-IDF weights, and surface as boilerplate "answers".
# Only lines near the top/bottom of a page are candidates, so a phrase that
# legitimately repeats in body text is never removed.
#
# Tuned against a real 25-page insurance PDF: it's three bundled documents
# (8 + 11 + 6 pages), each with its own 6-8 line footer block, so a given
# footer appears on only ~30% of pages and sits up to 8 lines from the bottom.
_HEADER_FOOTER_ZONE = 8
_REPEAT_FRACTION = 0.25
_MIN_REPEAT_PAGES = 3
_MIN_LINE_CHARS = 10  # never strip short generic lines ("Yes", "N/A", "Date")
_MIN_PAGES_FOR_STRIPPING = 3
# "Page 3 of 8", "Page | 10", "3 / 8" -- the only header/footer lines that
# legitimately differ page to page. Matched by pattern rather than by
# collapsing digits everywhere, since that would also merge genuinely distinct
# numbered body lines ("Section 4.1 ...", "Section 4.2 ...") into one.
_PAGE_NUMBER_RE = re.compile(r"^\s*(page\s*\|?\s*)?\d+\s*((of|/)\s*\d+)?\s*$", re.IGNORECASE)

# Headings are detected by boldness, not font size: in real multi-part PDFs
# body text is 8pt on one page and 12pt on another, so "bigger than body"
# misfires, while real headings are consistently fully bold.
_HEADING_MIN_BOLD = 0.8
_HEADING_MAX_WORDS = 12
_HEADING_MIN_LETTERS = 3
# A new paragraph starts when the vertical gap to the previous line exceeds
# this fraction of that line's height.
_PARAGRAPH_GAP = 0.8


def _clean_pdf_text(text: str) -> str:
    text = _UNDECODABLE_CHARS_RE.sub("", text)
    return _EXTRA_SPACES_RE.sub(" ", text)


def _normalize_line(line: str) -> str:
    return " ".join(line.split()).lower()


def _header_footer_keep_mask(page_lines: list[list[str]]) -> list[list[bool]]:
    """Per line: False if it's a page number or a repeat of a running
    header/footer line. The first copy of each repeated line is kept -- a
    running header is often the only place a document states its identity
    (product name, policy UIN)."""
    keep = [[True] * len(lines) for lines in page_lines]
    if len(page_lines) < _MIN_PAGES_FOR_STRIPPING:
        return keep

    counts: Counter[str] = Counter()
    longest_run: Counter[str] = Counter()
    current_run: Counter[str] = Counter()
    for lines in page_lines:
        zone = lines[:_HEADER_FOOTER_ZONE] + lines[-_HEADER_FOOTER_ZONE:]
        keys = {_normalize_line(line) for line in zone if len(line.strip()) >= _MIN_LINE_CHARS}
        counts.update(keys)
        current_run = Counter({k: current_run[k] + 1 for k in keys})
        for k, n in current_run.items():
            longest_run[k] = max(longest_run[k], n)
    min_pages = max(_MIN_REPEAT_PAGES, math.ceil(len(page_lines) * _REPEAT_FRACTION))
    # Either common across the whole file, or repeated on consecutive pages:
    # a bundled file's last sub-document had its own header on 6 straight
    # pages -- only 24% of the 25 pages, but unmistakably a running header.
    boilerplate = {
        line
        for line, n in counts.items()
        if n >= min_pages or longest_run[line] >= _MIN_REPEAT_PAGES
    }

    seen: set[str] = set()
    for p, lines in enumerate(page_lines):
        last = len(lines)
        for i, line in enumerate(lines):
            if not (i < _HEADER_FOOTER_ZONE or i >= last - _HEADER_FOOTER_ZONE):
                continue
            if _PAGE_NUMBER_RE.match(line):
                keep[p][i] = False
                continue
            key = _normalize_line(line)
            if key in boilerplate:
                if key in seen:
                    keep[p][i] = False
                else:
                    seen.add(key)
    return keep


def _strip_repeated_headers_footers(pages: list[str]) -> list[str]:
    page_lines = [p.splitlines() for p in pages]
    mask = _header_footer_keep_mask(page_lines)
    return [
        "\n".join(line for line, keep in zip(lines, keep_row) if keep)
        for lines, keep_row in zip(page_lines, mask)
    ]


def _clean_cell(cell: str | None) -> str:
    return _clean_pdf_text(" ".join((cell or "").split()))


def _looks_like_header_row(cells: list[str]) -> bool:
    filled = [c for c in cells if c]
    return (
        len(filled) >= 3
        and not any(ch.isdigit() for c in filled for ch in c)
        and sum(len(c) for c in filled) / len(filled) <= 40
    )


def _merge_continuation_rows(rows: list[list[str]]) -> list[list[str]]:
    """A cell's text wrapped onto the next row shows up as a row whose first
    cell is empty under a row whose first cell isn't ("Type of" /
    "Insurance" / "Product/ Policy") -- glue those back into one row."""
    merged: list[list[str]] = []
    for row in rows:
        prev = merged[-1] if merged else None
        if prev is not None and not row[0] and prev[0] and len(row) == len(prev):
            merged[-1] = [f"{a} {b}".strip() for a, b in zip(prev, row)]
        else:
            merged.append(row)
    return merged


def _table_blocks(rows: list[list[str | None]], page: int) -> list[Block]:
    """Turn one extracted table into blocks.

    A leading single-cell row ("Details of Policyholder") is the table's
    title, so it becomes a heading. If the first multi-cell row is short
    text-only cells, it's a column header, and later rows with the same shape
    are rendered as "Header: value; ..." so each row reads on its own. Only
    the first multi-cell row may be a header: letting any later row qualify
    mislabelled values in testing ("EIA No.: <an intermediary code>").
    Anything else is joined with " | " -- including tables used purely for
    page layout.
    """
    cleaned = [[_clean_cell(c) for c in row] for row in rows]
    cleaned = _merge_continuation_rows([row for row in cleaned if any(row)])
    if not cleaned:
        return []
    if len(cleaned) == 1:
        text = " ".join(c for c in cleaned[0] if c)
        return [Block(text, HEADING if _heading_shaped(text) else PARAGRAPH, page)]

    blocks: list[Block] = []
    header: list[str] | None = None
    header_used = False
    seen_multi_cell_row = False
    for row in cleaned:
        filled = [c for c in row if c]
        if not blocks and header is None and len(filled) == 1 and _heading_shaped(filled[0]):
            blocks.append(Block(filled[0], HEADING, page))
            continue
        if len(filled) >= 2 and not seen_multi_cell_row:
            seen_multi_cell_row = True
            if _looks_like_header_row(row):
                header = row
                continue
        if header is not None and len(row) == len(header):
            # a value under a blank header cell (merged header cells shift
            # columns) is kept bare rather than dropped -- losing a value
            # silently is worse than an unlabelled one
            pairs = [f"{h}: {v}" if h else v for h, v in zip(header, row) if v]
            if pairs:
                blocks.append(Block("; ".join(pairs), TABLE_ROW, page))
                header_used = True
                continue
        blocks.append(Block(" | ".join(filled), TABLE_ROW, page))
    if header is not None and not header_used:
        blocks.append(Block(" | ".join(c for c in header if c), TABLE_ROW, page))
    return blocks


# A bold line ending in one of these is a sentence wrapped onto the next
# line, not a heading ("All exclusions applicable to the base product will").
_DANGLING_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "of", "on", "or", "shall", "the", "to", "was", "which", "will", "with",
}


# A token with this many digits is a value (policy/application number,
# phone, date run), so the line is a bold label+value, not a heading --
# e.g. "Application No HE01862691Q202609" must stay searchable body text.
_VALUE_MIN_DIGITS = 5


# Bold lines that are list items, labels or wrapped fragments rather than
# section titles. Seen in testing: "2. Per Claim Deductible (Applicable for
# each and every claim" became a section and then mislabelled unrelated text
# on the following page as belonging to it -- a wrong section label is worse
# than none, since the page number is always shown anyway.
_LIST_ITEM_RE = re.compile(r"^(\(?(\d{1,2}|[a-z]|[ivx]{1,4})[.)])\s", re.IGNORECASE)


def _heading_shaped(text: str) -> bool:
    words = text.split()
    return (
        1 <= len(words) <= _HEADING_MAX_WORDS
        and sum(ch.isalpha() for ch in text) >= _HEADING_MIN_LETTERS
        and words[-1].lower().strip(",") not in _DANGLING_WORDS
        and not any(sum(ch.isdigit() for ch in w) >= _VALUE_MIN_DIGITS for w in words)
        and not _LIST_ITEM_RE.match(text)
        and not text[0].islower()
        and not text.endswith(("-", ":", ","))
        and "http" not in text.lower()
        and "www." not in text.lower()
    )


def _is_heading(text: str, chars: list[dict]) -> bool:
    if not chars:
        return False
    bold = sum("bold" in c.get("fontname", "").lower() for c in chars) / len(chars)
    return bold >= _HEADING_MIN_BOLD and _heading_shaped(text)


def _inside(line: dict, bbox: tuple) -> bool:
    x0, top, x1, bottom = bbox
    mid = (line["top"] + line["bottom"]) / 2
    return top <= mid <= bottom and line["x0"] < x1 and line["x1"] > x0


# Some PDFs store no space characters at all, so spaces are inferred from the
# gap between glyphs. pdfplumber's default is an absolute 3pt gap, which on
# one real PDF glued whole sentences together ("Createvalueforstakeholders"):
# its letters sat ~0.00 x font size apart and its word gaps >= 0.19 x font
# size, i.e. ~2.3pt at 12pt -- under 3pt. A gap relative to the font size
# sits cleanly between the two. PDFs that do store spaces are unaffected.
_X_TOLERANCE_RATIO = 0.15
# Glyphs from different fonts overlapping by more than this fraction of a
# glyph's width are separate text boxes drawn over each other, not a bold
# word inside a sentence (those sit next to each other).
_OVERLAP_FRACTION = 0.3
_MIN_COLLISIONS = 3


def _text_from_chars(chars: list[dict]) -> str:
    chars = sorted(chars, key=lambda c: c["x0"])
    out = []
    for prev, c in zip([None] + chars, chars):
        if prev is not None and c["x0"] - prev["x1"] > _X_TOLERANCE_RATIO * prev["size"]:
            out.append(" ")
        out.append(c["text"])
    return "".join(out)


def _split_overlapping_fonts(line: dict) -> list[dict]:
    """Seen in a real PDF: a bold "VISION" heading drawn on top of the body
    text beside it, or three differently-fonted phrases on one line, which
    extraction interleaved letter by letter ("VISSTIoOeNnhance ..."). If
    glyphs of different fonts physically overlap, split the line into one
    line per font."""
    chars = sorted(line.get("chars", []), key=lambda c: c["x0"])
    collisions = sum(
        1
        for a, b in zip(chars, chars[1:])
        if a["fontname"] != b["fontname"]
        and a["x1"] - b["x0"] > _OVERLAP_FRACTION * max(a["x1"] - a["x0"], 1e-6)
    )
    if collisions < _MIN_COLLISIONS:
        return [line]

    by_font: dict[str, list[dict]] = {}
    for c in chars:
        by_font.setdefault(c["fontname"], []).append(c)
    parts = []
    for group in by_font.values():
        parts.append(
            {
                "text": _text_from_chars(group),
                "chars": group,
                "x0": min(c["x0"] for c in group),
                "x1": max(c["x1"] for c in group),
                "top": min(c["top"] for c in group),
                "bottom": max(c["bottom"] for c in group),
            }
        )
    return sorted(parts, key=lambda p: p["x0"])


def _parse_pdf(doc_id: str, source) -> StructuredDoc:
    """source: a filesystem path string or a binary file-like object."""
    import pdfplumber

    pages = []
    with pdfplumber.open(source) as pdf:
        for number, page in enumerate(pdf.pages, 1):
            try:
                tables = page.find_tables()
            except Exception:  # malformed table geometry shouldn't sink the whole document
                tables = []
            table_items = [(t.bbox, t.extract(x_tolerance_ratio=_X_TOLERANCE_RATIO)) for t in tables]
            lines = [
                part
                for line in page.extract_text_lines(return_chars=True, x_tolerance_ratio=_X_TOLERANCE_RATIO)
                if not any(_inside(line, bbox) for bbox, _ in table_items)
                for part in _split_overlapping_fonts(line)
            ]
            pages.append((number, lines, table_items))

    mask = _header_footer_keep_mask([[line["text"] for line in lines] for _, lines, _ in pages])

    doc = StructuredDoc(doc_id=doc_id)
    for (number, lines, table_items), keep_row in zip(pages, mask):
        items: list[tuple[float, int, Block]] = []  # (top, order, block) for reading order
        paragraph: list[str] = []
        paragraph_top = 0.0
        prev = None

        def flush() -> None:
            if paragraph:
                items.append((paragraph_top, len(items), Block(" ".join(paragraph), PARAGRAPH, number)))
                paragraph.clear()

        for line, keep in zip(lines, keep_row):
            if not keep:
                continue
            text = _clean_pdf_text(" ".join(line["text"].split()))
            if not text:
                continue
            if _is_heading(text, line.get("chars", [])):
                flush()
                items.append((line["top"], len(items), Block(text, HEADING, number)))
                prev = None
                continue
            if prev is not None:
                gap = line["top"] - prev["bottom"]
                if gap > _PARAGRAPH_GAP * (prev["bottom"] - prev["top"]):
                    flush()
            if not paragraph:
                paragraph_top = line["top"]
            paragraph.append(text)
            prev = line
        flush()

        for bbox, rows in table_items:
            for block in _table_blocks(rows, number):
                items.append((bbox[1], len(items), block))

        items.sort(key=lambda item: (item[0], item[1]))
        doc.blocks.extend(block for _, _, block in items)

    first_heading = next((b for b in doc.blocks if b.kind == HEADING), None)
    if first_heading is not None:
        doc.title = first_heading.text
    return doc


def load_structured(path: Path, doc_id: str | None = None) -> StructuredDoc:
    doc_id = doc_id or path.stem
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(doc_id, str(path))
    if suffix in (".txt", ".md"):
        return parse_markdown(doc_id, path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported file type: {suffix} ({path})")


def load_structured_bytes(filename: str, data: bytes, doc_id: str | None = None) -> StructuredDoc:
    """Same as load_structured, from in-memory bytes (e.g. a browser upload),
    so uploaded content is never written to disk."""
    doc_id = doc_id or Path(filename).stem
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(doc_id, io.BytesIO(data))
    if suffix in (".txt", ".md"):
        return parse_markdown(doc_id, data.decode("utf-8", errors="replace"))
    raise ValueError(f"Unsupported file type: {suffix} ({filename})")


def load_structured_documents(doc_dir: Path) -> list[StructuredDoc]:
    return [
        load_structured(path)
        for path in sorted(Path(doc_dir).iterdir())
        if path.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def load_document(path: Path) -> str:
    """Plain-text view of a document (title + all blocks)."""
    return load_structured(path).plain_text()


def load_document_bytes(filename: str, data: bytes) -> str:
    return load_structured_bytes(filename, data).plain_text()


def load_documents(doc_dir: Path) -> dict[str, str]:
    return {doc.doc_id: doc.plain_text() for doc in load_structured_documents(doc_dir)}
