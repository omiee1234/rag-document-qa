"""Load raw text out of source documents (PDF/TXT/MD)."""

from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def load_document(path: Path) -> str:
    """Return the raw text contents of a single document."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

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
