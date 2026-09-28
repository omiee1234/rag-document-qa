"""Load raw text out of source documents (PDF/TXT/MD)."""

import io
import math
import re
from collections import Counter
from pathlib import Path

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


def _clean_pdf_text(text: str) -> str:
    text = _UNDECODABLE_CHARS_RE.sub("", text)
    return _EXTRA_SPACES_RE.sub(" ", text)


def _normalize_line(line: str) -> str:
    return " ".join(line.split()).lower()


def _strip_repeated_headers_footers(pages: list[str]) -> list[str]:
    if len(pages) < _MIN_PAGES_FOR_STRIPPING:
        return pages

    page_lines = [p.splitlines() for p in pages]
    counts: Counter[str] = Counter()
    for lines in page_lines:
        zone = lines[:_HEADER_FOOTER_ZONE] + lines[-_HEADER_FOOTER_ZONE:]
        counts.update(
            {_normalize_line(line) for line in zone if len(line.strip()) >= _MIN_LINE_CHARS}
        )

    min_pages = max(_MIN_REPEAT_PAGES, math.ceil(len(pages) * _REPEAT_FRACTION))
    boilerplate = {line for line, n in counts.items() if n >= min_pages}
    if not boilerplate:
        return pages

    # The first copy of each repeated line is kept: a running header is often
    # the only place a document states its own identity (product name, policy
    # UIN), so removing every copy would break "what policy is this?".
    seen: set[str] = set()
    cleaned = []
    for lines in page_lines:
        last = len(lines)
        kept = []
        for i, line in enumerate(lines):
            key = _normalize_line(line)
            in_zone = i < _HEADER_FOOTER_ZONE or i >= last - _HEADER_FOOTER_ZONE
            if in_zone and _PAGE_NUMBER_RE.match(line):
                continue
            if in_zone and key in boilerplate:
                if key in seen:
                    continue
                seen.add(key)
            kept.append(line)
        cleaned.append("\n".join(kept))
    return cleaned


def _extract_pdf_text(source) -> str:
    """source: a filesystem path string or a binary file-like object."""
    import pdfplumber

    with pdfplumber.open(source) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    pages = _strip_repeated_headers_footers(pages)
    return _clean_pdf_text("\n".join(pages))


def load_document(path: Path) -> str:
    """Return the raw text contents of a single document."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf_text(str(path))

    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8")

    raise ValueError(f"Unsupported file type: {suffix} ({path})")


def load_documents(doc_dir: Path) -> dict[str, str]:
    """Load every supported document in a directory. Returns {doc_id: text}."""
    doc_dir = Path(doc_dir)
    texts: dict[str, str] = {}
    for path in sorted(doc_dir.iterdir()):
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            texts[path.stem] = load_document(path)
    return texts


def load_document_bytes(filename: str, data: bytes) -> str:
    """Same as load_document, but from in-memory bytes (e.g. a browser upload)
    instead of a filesystem path -- avoids ever writing uploaded content to disk."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf_text(io.BytesIO(data))

    if suffix in (".txt", ".md"):
        return data.decode("utf-8", errors="replace")

    raise ValueError(f"Unsupported file type: {suffix} ({filename})")
