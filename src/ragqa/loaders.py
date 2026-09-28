"""Load raw text out of source documents (PDF/TXT/MD)."""

import re
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}

# Characters that render as a "tofu" box and carry no readable meaning once
# extracted:
#  - \x00-\x1f/\x7f: control characters
#  - �: the Unicode replacement character pypdf emits for glyphs it
#    can't decode at all
#  - -: the Private Use Area. Word-generated PDFs (a very common
#    real-world source -- e.g. insurance/legal docs) frequently encode
#    bullet points using a Symbol/Wingdings font mapped to PUA codepoints
#    (observed: U+F0B7 for a bullet character); pypdf extracts the raw
#    codepoint with no font info, so it comes through as an unreadable box.
_UNDECODABLE_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f�-]")
_EXTRA_SPACES_RE = re.compile(r"[ \t]{2,}")


def _clean_pdf_text(text: str) -> str:
    text = _UNDECODABLE_CHARS_RE.sub("", text)
    return _EXTRA_SPACES_RE.sub(" ", text)


def load_document(path: Path) -> str:
    """Return the raw text contents of a single document."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return _clean_pdf_text(text)

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
        import io

        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return _clean_pdf_text(text)

    if suffix in (".txt", ".md"):
        return data.decode("utf-8", errors="replace")

    raise ValueError(f"Unsupported file type: {suffix} ({filename})")
